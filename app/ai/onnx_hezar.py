"""Constrained multi-hypothesis CTC reader for the RC13 Iranian OCR model."""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .onnx_crnn import CRNN_LABELS
from .plate_rules import format_iran_plate, normalize_plate, plausible_plate
from .next_models import verified_next_manifest


MIN_CAMERA_SESSION_CACHE = 3
DIGIT_POSITIONS = {0, 1, 3, 4, 5, 6, 7}


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_last_status = {
    "engine": "hezar-ctc-onnx",
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
    with _cache_lock:
        _sessions.clear()
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            hypotheses=0,
            accepted=False,
            error="",
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
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
    blank = len(labels) if blank_index is None else int(blank_index)
    if probabilities.ndim != 2 or not 0 <= blank < probabilities.shape[1]:
        raise ValueError("Invalid CTC output or blank index")
    beams = {"": (1.0, 0.0)}
    width = max(2, int(beam_width))

    for timestep in probabilities:
        candidates = defaultdict(lambda: [0.0, 0.0])
        for prefix, (blank_prob, nonblank_prob) in beams.items():
            total = blank_prob + nonblank_prob
            candidates[prefix][0] += total * float(timestep[blank])
            for index, character in enumerate(labels):
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
        first = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
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
        resized = cv2.resize(source, (width, height))
        tensor = resized.astype(np.float32)[None, None] / 255.0
    else:
        source = (
            cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            if image.ndim == 2
            else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        )
        resized = cv2.resize(source, (width, height))
        tensor = (
            np.transpose(resized, (2, 0, 1))
            .astype(np.float32)[None]
            / 255.0
        )
    mean = float(spec.get("mean", 0.5))
    std = max(1e-6, float(spec.get("std", 0.5)))
    return (tensor - mean) / std


def _session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads_per_camera()
    options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return options


def _load_session(engine_key=None):
    manifest = verified_next_manifest()
    path = manifest["models"]["ocr"]["path"]
    camera_key = str(engine_key if engine_key is not None else "default")
    cache_key = (camera_key, path)
    with _cache_lock:
        cached = _sessions.get(cache_key)
        if cached is not None:
            _sessions.move_to_end(cache_key)
            return cached, manifest

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
        return entry, manifest


def read_plate_hezar(image, engine_key=None) -> dict:
    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        entry, manifest = _load_session(engine_key)
        spec = manifest["models"]["ocr"]
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
                attempted=True,
                model_loaded=False,
                engine_key=camera_key,
                hypotheses=0,
                accepted=False,
                error=f"{type(exc).__name__}: {exc}",
            )
        return {
            "accepted": False,
            "plate": "ناخوانا",
            "plate_norm": "",
            "confidence": 0.0,
            "position_details": [],
            "hypotheses": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
