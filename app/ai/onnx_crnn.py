"""Whole-plate Iranian OCR using a CRNN+CTC model in ONNX Runtime.

The reader is intentionally independent from EasyOCR and PyTorch.  A separate
ONNX session is cached for each active camera so the existing per-camera CPU
budget remains enforceable.  Missing or invalid model files fail closed and
leave the legacy OCR path available.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
import os
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


CRNN_HEIGHT = 32
CRNN_WIDTH = 128
MIN_CAMERA_SESSION_CACHE = 3
CRNN_LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "ا", "ب", "ت", "ث", "ج", "ح", "د", "ز", "س", "ش", "ص",
    "ط", "ع", "ق", "ل", "م", "ن", "ه", "و", "پ", "ژ", "ی",
]
DIGIT_POSITIONS = {0, 1, 3, 4, 5, 6, 7}


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_verified_model_cache: tuple[str, int, int] | None = None
_invalid_model_cache: tuple[str, int, int] | None = None
_last_status = {
    "engine": "crnn-onnx",
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "engine_key": "",
    "raw_text": "",
    "confidence": 0.0,
    "decoder": "ctc-constrained-beam",
    "accepted": False,
    "hypotheses": 0,
    "views": 0,
    "minimum_position_margin": 0.0,
    "error": "",
    "threads": 0,
}


def get_crnn_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_crnn_sessions() -> None:
    """Release cached sessions; primarily used after a verified model update."""

    global _invalid_model_cache, _verified_model_cache
    with _cache_lock:
        _sessions.clear()
        _verified_model_cache = None
        _invalid_model_cache = None
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            raw_text="",
            confidence=0.0,
            accepted=False,
            hypotheses=0,
            views=0,
            minimum_position_margin=0.0,
            error="",
            threads=0,
        )


def _softmax(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    exponent = np.exp(values)
    denominator = np.sum(exponent, axis=1, keepdims=True)
    return exponent / np.maximum(denominator, 1e-12)


def ctc_greedy_decode(
    logits: np.ndarray,
    labels: list[str] | tuple[str, ...] = CRNN_LABELS,
) -> tuple[str, float]:
    """Collapse CTC repeats and blanks and return mean emitted confidence."""

    probabilities = _softmax(np.asarray(logits))
    blank = len(labels)
    indices = probabilities.argmax(axis=1)
    characters: list[str] = []
    confidences: list[float] = []
    previous = -1

    for timestep, raw_index in enumerate(indices):
        index = int(raw_index)
        if (
            index != blank
            and index != previous
            and 0 <= index < len(labels)
        ):
            characters.append(labels[index])
            confidences.append(float(probabilities[timestep, index]))
        previous = index

    return (
        "".join(characters),
        float(np.mean(confidences)) if confidences else 0.0,
    )


def _allowed_character(position: int, character: str) -> bool:
    if position >= 8:
        return False
    return character.isdigit() == (position in DIGIT_POSITIONS)


def ctc_beam_hypotheses(
    logits: np.ndarray,
    labels: list[str] | tuple[str, ...] = CRNN_LABELS,
    beam_width: int = 12,
    top_k: int = 5,
) -> list[dict]:
    """Decode CTC while enforcing the real 2+letter+3+2 plate grammar.

    Greedy decoding can discard the correct sequence when one timestep has a
    close runner-up. The small prefix beam keeps those alternatives but never
    permits a digit in the letter slot or a letter in a numeric slot.
    """

    probabilities = _softmax(np.asarray(logits))
    blank = len(labels)
    if probabilities.ndim != 2 or probabilities.shape[1] != blank + 1:
        raise ValueError(
            "Unexpected CRNN output shape: "
            + str(tuple(np.asarray(logits).shape))
        )
    beams: dict[str, tuple[float, float]] = {"": (1.0, 0.0)}
    width = max(4, min(32, int(beam_width)))

    for timestep in probabilities:
        candidates = defaultdict(lambda: [0.0, 0.0])
        for prefix, (blank_probability, text_probability) in beams.items():
            total = blank_probability + text_probability
            candidates[prefix][0] += total * float(timestep[blank])
            for class_index, character in enumerate(labels):
                probability = float(timestep[class_index])
                if probability <= 1e-10:
                    continue
                if prefix and prefix[-1] == character:
                    candidates[prefix][1] += (
                        text_probability * probability
                    )
                    if _allowed_character(len(prefix), character):
                        candidates[prefix + character][1] += (
                            blank_probability * probability
                        )
                elif _allowed_character(len(prefix), character):
                    candidates[prefix + character][1] += total * probability
        beams = {
            prefix: (float(scores[0]), float(scores[1]))
            for prefix, scores in sorted(
                candidates.items(),
                key=lambda item: sum(item[1]),
                reverse=True,
            )[:width]
        }

    ranked = [
        (normalize_plate(prefix), blank_score + text_score)
        for prefix, (blank_score, text_score) in beams.items()
        if plausible_plate(prefix)
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    ranked = ranked[:max(1, int(top_k))]
    total = sum(score for _plate, score in ranked)
    timesteps = max(1, probabilities.shape[0])
    output = []
    for plate, path_score in ranked:
        relative = path_score / max(total, 1e-300)
        geometric = max(path_score, 1e-300) ** (1.0 / timesteps)
        confidence = min(1.0, max(0.0, (relative * geometric) ** 0.5))
        output.append({
            "plate": format_iran_plate(plate),
            "plate_norm": plate,
            "confidence": float(confidence),
            "score": float(confidence),
            "path_score": float(path_score),
            "engine": "crnn-onnx-beam",
        })
    return output


def hypothesis_position_margins(hypotheses: list[dict]) -> list[dict]:
    details = []
    for position in range(8):
        buckets = defaultdict(float)
        for row in hypotheses:
            plate = normalize_plate(
                row.get("plate_norm") or row.get("plate")
            )
            if len(plate) != 8:
                continue
            buckets[plate[position]] += max(
                0.0,
                float(row.get("score", row.get("confidence", 0.0))),
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
        total = sum(value for _character, value in ordered)
        first_probability = ordered[0][1] / max(total, 1e-12)
        second_probability = (
            ordered[1][1] / max(total, 1e-12)
            if len(ordered) > 1
            else 0.0
        )
        details.append({
            "position": position,
            "character": ordered[0][0],
            "probability": round(first_probability, 6),
            "margin": round(
                first_probability - second_probability,
                6,
            ),
        })
    return details


def _adaptive_ocr_variant(image) -> np.ndarray | None:
    if image is None or getattr(image, "size", 0) == 0:
        return None
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    mean = float(np.mean(gray))
    contrast = float(np.std(gray))
    if mean < 82.0 or mean > 188.0:
        gamma = 1.55 if mean < 82.0 else 0.72
        inverse = 1.0 / gamma
        table = np.array(
            [
                ((index / 255.0) ** inverse) * 255.0
                for index in range(256)
            ],
            dtype=np.uint8,
        )
        gray = cv2.LUT(gray, table)
    clip_limit = 3.0 if contrast < 38.0 else 2.0
    enhanced = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(8, 4),
    ).apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 32, 32)
    blurred = cv2.GaussianBlur(denoised, (0, 0), 0.9)
    return cv2.addWeighted(denoised, 1.55, blurred, -0.55, 0)


def _merge_view_hypotheses(view_results: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    for view in view_results:
        view_name = str(view.get("view", "raw"))
        greedy_norm = normalize_plate(view.get("greedy_text", ""))
        greedy_confidence = float(view.get("greedy_confidence", 0.0))
        rows = list(view.get("hypotheses", []))
        if plausible_plate(greedy_norm) and all(
            normalize_plate(row.get("plate_norm")) != greedy_norm
            for row in rows
        ):
            rows.append({
                "plate": format_iran_plate(greedy_norm),
                "plate_norm": greedy_norm,
                "confidence": greedy_confidence,
                "score": greedy_confidence,
                "engine": "crnn-onnx-greedy",
            })
        for row in rows:
            plate = normalize_plate(
                row.get("plate_norm") or row.get("plate")
            )
            if not plausible_plate(plate):
                continue
            confidence = min(
                1.0,
                max(
                    0.0,
                    float(row.get("confidence", row.get("score", 0.0))),
                ),
            )
            target = merged.setdefault(plate, {
                "plate": format_iran_plate(plate),
                "plate_norm": plate,
                "confidence": 0.0,
                "score": 0.0,
                "engine": "crnn-onnx-beam",
                "views": set(),
                "view_confidences": [],
            })
            target["views"].add(view_name)
            target["view_confidences"].append(confidence)
            target["confidence"] = max(target["confidence"], confidence)

    output = []
    for row in merged.values():
        support = len(row["views"])
        confidences = row.pop("view_confidences")
        row["views"] = sorted(row["views"])
        row["view_support"] = support
        row["score"] = min(
            1.0,
            0.78 * max(confidences)
            + 0.22 * (sum(confidences) / len(confidences))
            + 0.055 * max(0, support - 1),
        )
        row["confidence"] = row["score"]
        output.append(row)
    output.sort(
        key=lambda row: (
            float(row.get("score", 0.0)),
            int(row.get("view_support", 0)),
            row["plate_norm"],
        ),
        reverse=True,
    )
    return output[:5]


def accept_crnn_hypotheses(
    hypotheses: list[dict],
    *,
    greedy_text: str = "",
    greedy_confidence: float = 0.0,
    minimum_confidence: float | None = None,
    minimum_margin: float | None = None,
) -> dict:
    """Apply one calibrated acceptance policy in live and Golden paths."""

    confidence_floor = float(
        os.environ.get("BCVISION_CRNN_MIN_CONFIDENCE", "0.50")
        if minimum_confidence is None
        else minimum_confidence
    )
    margin_floor = float(
        os.environ.get("BCVISION_CRNN_MIN_MARGIN", "0.06")
        if minimum_margin is None
        else minimum_margin
    )
    position_details = hypothesis_position_margins(hypotheses)
    top = hypotheses[0] if hypotheses else None
    confidence = float(top.get("confidence", 0.0)) if top else 0.0
    observed_margin = (
        min(row["margin"] for row in position_details)
        if position_details
        else 0.0
    )
    greedy_norm = normalize_plate(greedy_text)
    view_support = int(top.get("view_support", 0)) if top else 0
    accepted = bool(
        top
        and plausible_plate(top.get("plate_norm", ""))
        and confidence >= confidence_floor
        and observed_margin >= margin_floor
        and (
            view_support >= 2
            or (
                top["plate_norm"] == greedy_norm
                and float(greedy_confidence) >= 0.62
            )
            or confidence >= 0.78
        )
    )
    reason = (
        "accepted-multi-view"
        if accepted and view_support >= 2
        else "accepted-decisive"
        if accepted
        else "no-layout-valid-hypothesis"
        if not top
        else "low-sequence-confidence"
        if confidence < confidence_floor
        else "ambiguous-character-margin"
        if observed_margin < margin_floor
        else "insufficient-decoder-agreement"
    )
    normalized = top["plate_norm"] if top else greedy_norm
    return {
        "accepted": accepted,
        "plate": format_iran_plate(normalized) if accepted else "",
        "plate_norm": normalized if accepted else "",
        "raw_guess_norm": normalized,
        "raw_guess_text": (
            format_iran_plate(normalized)
            if plausible_plate(normalized)
            else normalized
        ),
        "confidence": round(confidence, 4),
        "hypotheses": hypotheses,
        "position_details": position_details,
        "view_support": view_support,
        "minimum_position_margin": round(observed_margin, 4),
        "decoder": "ctc-constrained-beam",
        "reason": reason,
    }


def prepare_crnn_input(image) -> np.ndarray | None:
    if image is None or getattr(image, "size", 0) == 0:
        return None
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    resized = cv2.resize(
        gray,
        (CRNN_WIDTH, CRNN_HEIGHT),
        interpolation=(
            cv2.INTER_AREA
            if gray.shape[1] > CRNN_WIDTH
            else cv2.INTER_CUBIC
        ),
    )
    return (
        resized.astype(np.float32).reshape(
            1,
            1,
            CRNN_HEIGHT,
            CRNN_WIDTH,
        )
        / 255.0
    )


def _session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads_per_camera()
    options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if hasattr(ort, "GraphOptimizationLevel"):
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
    add_entry = getattr(options, "add_session_config_entry", None)
    if callable(add_entry):
        add_entry("session.intra_op.allow_spinning", "0")
        add_entry("session.inter_op.allow_spinning", "0")
    return options


def _verified_model_path() -> Path:
    global _invalid_model_cache, _verified_model_cache
    from .model_manager import (
        active_crnn_model,
        verify_file,
    )

    path, expected_sha256, expected_size = active_crnn_model()
    try:
        stat = path.stat()
    except OSError as exc:
        raise FileNotFoundError(
            f"Verified CRNN ONNX model not found: {path}"
        ) from exc
    cache_key = (
        str(path.resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )
    with _cache_lock:
        if _verified_model_cache == cache_key:
            return path
        if _invalid_model_cache == cache_key:
            raise FileNotFoundError(
                f"Verified CRNN ONNX model not found: {path}"
            )
    if not verify_file(path, expected_sha256, expected_size):
        with _cache_lock:
            _invalid_model_cache = cache_key
        raise FileNotFoundError(
            f"Verified CRNN ONNX model not found: {path}"
        )
    with _cache_lock:
        _invalid_model_cache = None
        _verified_model_cache = cache_key
    return path


def _load_session(engine_key=None) -> _SessionEntry:
    path = _verified_model_path()
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    cache_key = (camera_key, str(path.resolve()))

    with _cache_lock:
        cached = _sessions.get(cache_key)
        if cached is not None:
            _sessions.move_to_end(cache_key)
            return cached

        import onnxruntime as ort

        options = _session_options(ort)
        session = ort.InferenceSession(
            str(path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        entry = _SessionEntry(
            session=session,
            input_name=session.get_inputs()[0].name,
            run_lock=threading.Lock(),
        )
        _sessions[cache_key] = entry
        # Concurrent inference remains capped by parallel_camera_limit(), but
        # keep the default three cameras' sessions distinct even on a
        # two-core host where only one inference may run at a time.  Mapping
        # camera IDs through modulo made cameras 1 and 2 share one session on
        # Windows runners and could make live cameras evict/recreate sessions
        # continuously.
        cache_limit = max(
            MIN_CAMERA_SESSION_CACHE,
            parallel_camera_limit(),
        )
        while len(_sessions) > cache_limit:
            _sessions.popitem(last=False)
        return entry


def read_plate_crnn(
    image,
    engine_key=None,
    *,
    alternate_images=None,
    return_details=False,
):
    """Read a crop with constrained beam decoding and gated rescue views."""

    empty_details = {
        "accepted": False,
        "plate": "",
        "plate_norm": "",
        "confidence": 0.0,
        "hypotheses": [],
        "position_details": [],
        "views": 0,
        "decoder": "ctc-constrained-beam",
        "reason": "empty-input",
    }
    tensor = prepare_crnn_input(image)
    if tensor is None:
        return (
            ("", 0.0, empty_details)
            if return_details
            else ("", 0.0)
        )

    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        path = _verified_model_path()
        entry = _load_session(engine_key=engine_key)
        max_views = max(
            1,
            min(
                4,
                int(os.environ.get("BCVISION_CRNN_RESCUE_VIEWS", "3")),
            ),
        )
        candidate_views = [("raw", image)]
        for index, alternate in enumerate(alternate_images or []):
            if (
                alternate is not None
                and getattr(alternate, "size", 0)
                and len(candidate_views) < max_views
            ):
                candidate_views.append((f"geometry-{index + 1}", alternate))
        enhanced = _adaptive_ocr_variant(image)
        if (
            enhanced is not None
            and len(candidate_views) < max_views
            and not np.array_equal(
                enhanced,
                image if getattr(image, "ndim", 0) == 2 else cv2.cvtColor(
                    image,
                    cv2.COLOR_BGR2GRAY,
                ),
            )
        ):
            candidate_views.append(("adaptive-contrast", enhanced))

        view_results = []
        for view_index, (view_name, view_image) in enumerate(candidate_views):
            view_tensor = prepare_crnn_input(view_image)
            if view_tensor is None:
                continue
            with entry.run_lock:
                output = entry.session.run(
                    None,
                    {entry.input_name: view_tensor},
                )[0]
            logits = np.asarray(output)
            if logits.ndim == 3:
                logits = logits[0]
            if (
                logits.ndim != 2
                or logits.shape[1] != len(CRNN_LABELS) + 1
            ):
                raise ValueError(
                    "Unexpected CRNN output shape: "
                    + str(tuple(np.asarray(output).shape))
                )
            greedy_text, greedy_confidence = ctc_greedy_decode(logits)
            hypotheses = ctc_beam_hypotheses(
                logits,
                beam_width=int(
                    os.environ.get("BCVISION_CRNN_BEAM_WIDTH", "12")
                ),
                top_k=5,
            )
            view_results.append({
                "view": view_name,
                "greedy_text": greedy_text,
                "greedy_confidence": float(greedy_confidence),
                "hypotheses": hypotheses,
            })

            # A decisive raw result does not pay for rescue inference. This is
            # the common fast path and retains the two-thread CPU ceiling.
            if view_index == 0 and hypotheses:
                margins = hypothesis_position_margins(hypotheses)
                top = hypotheses[0]
                greedy_norm = normalize_plate(greedy_text)
                if (
                    top["plate_norm"] == greedy_norm
                    and float(greedy_confidence) >= 0.82
                    and float(top.get("confidence", 0.0)) >= 0.72
                    and min(
                        row["margin"] for row in margins
                    ) >= 0.12
                ):
                    break

        hypotheses = _merge_view_hypotheses(view_results)
        first_greedy_text = str(
            view_results[0].get("greedy_text", "")
        ) if view_results else ""
        first_greedy_confidence = float(
            view_results[0].get("greedy_confidence", 0.0)
        ) if view_results else 0.0
        details = accept_crnn_hypotheses(
            hypotheses,
            greedy_text=first_greedy_text,
            greedy_confidence=first_greedy_confidence,
        )
        details["views"] = len(view_results)
        accepted = bool(details["accepted"])
        normalized = str(details["raw_guess_norm"])
        confidence = float(details["confidence"])
        minimum_margin = float(details["minimum_position_margin"])
        text = str(details["plate"])
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=str(path),
                engine_key=camera_key,
                raw_text=normalized,
                confidence=round(confidence, 4),
                accepted=accepted,
                hypotheses=len(hypotheses),
                views=len(view_results),
                minimum_position_margin=round(minimum_margin, 4),
                error="",
                threads=threads_per_camera(),
            )
        if return_details:
            return text, round(confidence, 4), details
        return text, round(confidence, 4)
    except Exception as exc:
        details = {
            **empty_details,
            "reason": "runtime-error",
            "error": f"{type(exc).__name__}: {exc}",
        }
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                engine_key=camera_key,
                raw_text="",
                confidence=0.0,
                accepted=False,
                hypotheses=0,
                views=0,
                minimum_position_margin=0.0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        return (
            ("", 0.0, details)
            if return_details
            else ("", 0.0)
        )
