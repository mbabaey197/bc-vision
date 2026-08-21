"""Constrained CTC runtime for Hezar's Persian plate CRNN."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .onnx_crnn import CRNN_LABELS
from .plate_rules import format_iran_plate, normalize_plate, plausible_plate
from .next_models import verified_next_manifest


MIN_CAMERA_SESSION_CACHE = 3
DIGIT_POSITIONS = {0, 1, 3, 4, 5, 6, 7}
PRIMARY_ENGINE = "hezar-crnn-fa-v2-onnx"
HEZAR_V2_LABELS = [
    "", "آ", "ا", "ب", "پ", "ت", "ث", "ج", "چ", "ه", "خ",
    "د", "ذ", "ر", "ز", "ژ", "س", "ش", "ص", "ض", "ط", "ظ",
    "ع", "غ", "ف", "ق", "ک", "گ", "ل", "م", "ن", "و", "ه",
    "ی", " ", "۱", "۲", "۳", "۴", "۵", "۶", "۷", "۸", "۹",
    "۰",
]
HEZAR_V2_SPEC = {
    "runtime": PRIMARY_ENGINE,
    "input_height": 32,
    "input_width": 384,
    "channels": 1,
    "mean": 0.6595,
    "std": 0.1501,
    "mirror": True,
    "labels": HEZAR_V2_LABELS,
    "blank_index": 0,
    "reverse_output_digits": True,
    "beam_width": 10,
    "top_k": 5,
    "min_confidence": 0.56,
    "min_position_margin": 0.12,
}


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_verified_primary_cache: tuple | None = None
_invalid_primary_cache: tuple | None = None
_last_status = {
    "engine": PRIMARY_ENGINE,
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "engine_key": "",
    "hypotheses": 0,
    "accepted": False,
    "error": "",
}


def hezar_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_hezar_sessions() -> None:
    global _invalid_primary_cache, _verified_primary_cache
    with _cache_lock:
        _sessions.clear()
        _verified_primary_cache = None
        _invalid_primary_cache = None
        _last_status.update(
            engine=PRIMARY_ENGINE,
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            hypotheses=0,
            accepted=False,
            error="",
        )


def _verified_primary_path() -> Path:
    """Hash Hezar once per file revision instead of once per plate crop."""

    global _invalid_primary_cache, _verified_primary_cache
    from .model_manager import (
        HEZAR_ONNX_SHA256,
        HEZAR_ONNX_SIZE,
        hezar_path,
        verify_file,
    )

    path = hezar_path()
    try:
        stat = path.stat()
    except OSError as exc:
        raise FileNotFoundError(
            f"Verified Hezar CRNN model not found: {path}"
        ) from exc
    cache_key = (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ctime_ns", 0)),
        HEZAR_ONNX_SHA256,
    )
    with _cache_lock:
        if _verified_primary_cache == cache_key:
            return path
        if _invalid_primary_cache == cache_key:
            raise FileNotFoundError(
                f"Verified Hezar CRNN model not found: {path}"
            )
    if not verify_file(path, HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE):
        with _cache_lock:
            _invalid_primary_cache = cache_key
        raise FileNotFoundError(
            f"Verified Hezar CRNN model not found: {path}"
        )
    with _cache_lock:
        _invalid_primary_cache = None
        _verified_primary_cache = cache_key
    return path


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("CTC logits must have shape (time, classes)")
    values -= values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / np.maximum(
        exponent.sum(axis=1, keepdims=True),
        1e-12,
    )


def _allowed_character(position: int, character: str) -> bool:
    if position >= 8:
        return False
    return character.isdigit() == (position in DIGIT_POSITIONS)


def ctc_beam_hypotheses(
    logits: np.ndarray,
    labels=CRNN_LABELS,
    blank_index=None,
    beam_width=8,
    top_k=5,
) -> list[dict]:
    """Return plate-layout-constrained CTC prefixes with relative scores."""

    probabilities = _softmax(np.asarray(logits))
    class_count = probabilities.shape[1]
    blank = len(labels) if blank_index is None else int(blank_index)
    if not 0 <= blank < class_count:
        raise ValueError("Invalid CTC output or blank index")
    # BC Vision's native CRNN stores the blank after all labels.  Hezar v2
    # stores it at class 0 and includes the empty blank label in id2label.
    # Accept either representation so manifest labels map to their real logit
    # indices instead of silently shifting every Persian character.
    if len(labels) == class_count:
        label_items = [
            (index, str(character))
            for index, character in enumerate(labels)
            if index != blank and str(character)
        ]
    elif len(labels) == class_count - 1:
        label_items = list(zip(
            (
                index
                for index in range(class_count)
                if index != blank
            ),
            (str(character) for character in labels),
        ))
    else:
        raise ValueError(
            "CTC label count does not match output classes: "
            f"{len(labels)} labels for {class_count} classes"
        )
    beams = {"": (1.0, 0.0)}
    width = max(2, int(beam_width))

    for timestep in probabilities:
        candidates = defaultdict(lambda: [0.0, 0.0])
        for prefix, (blank_prob, nonblank_prob) in beams.items():
            total = blank_prob + nonblank_prob
            candidates[prefix][0] += total * float(timestep[blank])
            for index, character in label_items:
                probability = float(timestep[index])
                if probability <= 1e-9:
                    continue
                if prefix and prefix[-1] == character:
                    candidates[prefix][1] += nonblank_prob * probability
                    if _allowed_character(len(prefix), character):
                        extended = prefix + character
                        candidates[extended][1] += blank_prob * probability
                elif _allowed_character(len(prefix), character):
                    extended = prefix + character
                    candidates[extended][1] += total * probability
        beams = {
            prefix: tuple(scores)
            for prefix, scores in sorted(
                candidates.items(),
                key=lambda item: sum(item[1]),
                reverse=True,
            )[:width]
        }

    ranked = [
        (prefix, blank_prob + nonblank_prob)
        for prefix, (blank_prob, nonblank_prob) in beams.items()
        if plausible_plate(prefix)
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    ranked = ranked[:max(1, int(top_k))]
    total = sum(score for _, score in ranked)
    timesteps = max(1, probabilities.shape[0])
    return [
        {
            "plate_norm": normalize_plate(prefix),
            "plate": format_iran_plate(prefix),
            "score": float(score),
            # Relative rank alone can turn one weak path into confidence 1.0.
            # Blend it with the geometric path probability so uniformly
            # uncertain logits remain rejectable.
            "confidence": float(
                (
                    score / max(total, 1e-12)
                    * max(score, 1e-300) ** (1.0 / timesteps)
                )
                ** 0.5
            ),
        }
        for prefix, score in ranked
    ]


def hypothesis_position_margins(hypotheses: list[dict]) -> list[dict]:
    details = []
    for position in range(8):
        buckets = defaultdict(float)
        for row in hypotheses:
            plate = normalize_plate(row.get("plate_norm", ""))
            if len(plate) == 8:
                buckets[plate[position]] += max(
                    0.0,
                    float(row.get("confidence", 0.0)),
                )
        ordered = sorted(
            buckets.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        if not ordered:
            details.append({
                "position": position,
                "character": "",
                "probability": 0.0,
                "margin": 0.0,
            })
            continue
        total = sum(value for _, value in ordered)
        first = (ordered[0][0], ordered[0][1] / max(total, 1e-12))
        second = (
            ordered[1][1] / max(total, 1e-12)
            if len(ordered) > 1
            else 0.0
        )
        details.append({
            "position": position,
            "character": first[0],
            "probability": round(first[1], 6),
            "margin": round(first[1] - second, 6),
        })
    return details


def accept_hypotheses(
    hypotheses: list[dict],
    min_confidence=0.56,
    min_position_margin=0.12,
) -> dict:
    if not hypotheses:
        return {
            "accepted": False,
            "plate": "ناخوانا",
            "plate_norm": "",
            "confidence": 0.0,
            "position_details": [],
            "hypotheses": [],
        }
    details = hypothesis_position_margins(hypotheses)
    top = hypotheses[0]
    confidence = float(top.get("confidence", 0.0))
    accepted = bool(
        plausible_plate(top.get("plate_norm", ""))
        and confidence >= float(min_confidence)
        and len(details) == 8
        and min(row["margin"] for row in details)
        >= float(min_position_margin)
    )
    return {
        "accepted": accepted,
        "plate": top["plate"] if accepted else "ناخوانا",
        "plate_norm": top["plate_norm"] if accepted else "",
        "confidence": round(confidence, 6),
        "position_details": details,
        "hypotheses": hypotheses,
    }


def prepare_hezar_input(image, spec: dict) -> np.ndarray | None:
    if image is None or getattr(image, "size", 0) == 0:
        return None
    height = max(24, int(spec.get("input_height", 32)))
    width = max(64, int(spec.get("input_width", 128)))
    channels = int(spec.get("channels", 1))
    if channels == 1:
        source = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image
        )
        if bool(spec.get("mirror", False)):
            source = cv2.flip(source, 1)
        resized = cv2.resize(
            source,
            (width, height),
            interpolation=(
                cv2.INTER_AREA
                if source.shape[1] > width
                else cv2.INTER_LINEAR
            ),
        )
        tensor = resized.astype(np.float32)[None, None] / 255.0
    else:
        source = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        if bool(spec.get("mirror", False)):
            source = cv2.flip(source, 1)
        resized = cv2.resize(
            source,
            (width, height),
            interpolation=(
                cv2.INTER_AREA
                if source.shape[1] > width
                else cv2.INTER_LINEAR
            ),
        )
        tensor = (
            np.transpose(resized, (2, 0, 1))
            .astype(np.float32)[None]
            / 255.0
        )
    raw_mean = spec.get("mean", 0.5)
    raw_std = spec.get("std", 0.5)
    if isinstance(raw_mean, (list, tuple)):
        raw_mean = raw_mean[0] if raw_mean else 0.5
    if isinstance(raw_std, (list, tuple)):
        raw_std = raw_std[0] if raw_std else 0.5
    mean = float(raw_mean)
    std = max(1e-6, float(raw_std))
    return (tensor - mean) / std


def _session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads_per_camera()
    options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _load_path_session(path, engine_key=None):
    path = str(Path(path).resolve())
    camera_key = str(engine_key if engine_key is not None else "default")
    cache_key = (camera_key, path)
    with _cache_lock:
        cached = _sessions.get(cache_key)
        if cached is not None:
            _sessions.move_to_end(cache_key)
            return cached

        import onnxruntime as ort

        session = ort.InferenceSession(
            path,
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        entry = _SessionEntry(
            session=session,
            input_name=session.get_inputs()[0].name,
            run_lock=threading.Lock(),
        )
        _sessions[cache_key] = entry
        while len(_sessions) > max(
            MIN_CAMERA_SESSION_CACHE,
            parallel_camera_limit(),
        ):
            _sessions.popitem(last=False)
        return entry


def _failure_result(exc: Exception) -> dict:
    return {
        "accepted": False,
        "plate": "ناخوانا",
        "plate_norm": "",
        "confidence": 0.0,
        "position_details": [],
        "hypotheses": [],
        "error": f"{type(exc).__name__}: {exc}",
    }


def _read_plate_with_spec(
    image,
    spec: dict,
    engine_key=None,
    runtime=PRIMARY_ENGINE,
) -> dict:
    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        entry = _load_path_session(spec["path"], engine_key)
        tensor = prepare_hezar_input(image, spec)
        if tensor is None:
            raise ValueError("Empty OCR crop")
        with entry.run_lock:
            output = entry.session.run(
                None,
                {entry.input_name: tensor},
            )[0]
        logits = np.asarray(output)
        if logits.ndim == 3:
            logits = logits[0]
        if bool(spec.get("reverse_output_digits", False)):
            # Hezar mirrors the input image and reverses the decoded digit
            # sequence in post-processing.  Reversing the time axis before
            # constrained CTC decoding applies the Iranian layout constraints
            # in the final, human-readable direction.
            logits = logits[::-1]
        labels = list(spec.get("labels") or CRNN_LABELS)
        blank_index = int(spec.get("blank_index", len(labels)))
        hypotheses = ctc_beam_hypotheses(
            logits,
            labels=labels,
            blank_index=blank_index,
            beam_width=int(spec.get("beam_width", 8)),
            top_k=int(spec.get("top_k", 5)),
        )
        result = accept_hypotheses(
            hypotheses,
            min_confidence=float(
                spec.get("min_confidence", 0.56)
            ),
            min_position_margin=float(
                spec.get("min_position_margin", 0.12)
            ),
        )
        with _cache_lock:
            _last_status.update(
                engine=runtime,
                attempted=True,
                model_loaded=True,
                model_path=spec["path"],
                engine_key=camera_key,
                hypotheses=len(hypotheses),
                accepted=result["accepted"],
                error="",
            )
        return result
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                engine=runtime,
                attempted=True,
                model_loaded=False,
                model_path=str(spec.get("path", "")),
                engine_key=camera_key,
                hypotheses=0,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _failure_result(exc)


def read_plate_hezar_primary(image, engine_key=None) -> dict:
    """Read a cropped plate with the fixed, verified Hezar v2 model."""
    from .model_manager import hezar_path

    path = hezar_path()
    try:
        path = _verified_primary_path()
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                engine=PRIMARY_ENGINE,
                attempted=True,
                model_loaded=False,
                model_path=str(path),
                engine_key=str(
                    engine_key if engine_key is not None else "default"
                ),
                hypotheses=0,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _failure_result(exc)
    spec = {**HEZAR_V2_SPEC, "path": str(path)}
    return _read_plate_with_spec(
        image,
        spec,
        engine_key=engine_key,
        runtime=PRIMARY_ENGINE,
    )


def read_plate_hezar(image, engine_key=None) -> dict:
    """Read with a signed candidate Hezar model used by the next engine."""

    try:
        manifest = verified_next_manifest()
        spec = manifest["models"]["ocr"]
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                engine="hezar-ctc-onnx",
                attempted=True,
                model_loaded=False,
                model_path="",
                engine_key=str(
                    engine_key if engine_key is not None else "default"
                ),
                hypotheses=0,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return _failure_result(exc)
    return _read_plate_with_spec(
        image,
        spec,
        engine_key=engine_key,
        runtime="hezar-ctc-onnx",
    )
