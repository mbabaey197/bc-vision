"""FastPlateOCR CCT inference for fixed-layout Iranian license plates.

The production runtime intentionally depends only on NumPy, OpenCV and ONNX
Runtime.  Training remains an offline tool.  Every preprocessing and decoding
parameter is carried by the signed next-model manifest so an ONNX file cannot
silently be paired with an incompatible alphabet or image contract.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .next_models import verified_next_manifest
from .plate_rules import (
    ALLOWED_PLATE_LETTERS,
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)


CCT_RUNTIME = "fast-plate-ocr-cct"
CCT_MAX_PLATE_SLOTS = 8
CCT_DEFAULT_ALPHABET = (
    "0123456789" + ALLOWED_PLATE_LETTERS + "_"
)
CCT_DIGIT_POSITIONS = frozenset({0, 1, 3, 4, 5, 6, 7})
CCT_LETTER_POSITIONS = frozenset({2})
MIN_CAMERA_SESSION_CACHE = 3


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_last_status = {
    "engine": CCT_RUNTIME,
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "engine_key": "",
    "raw_text": "",
    "confidence": 0.0,
    "error": "",
    "threads": 0,
}


def cct_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_cct_sessions() -> None:
    with _cache_lock:
        _sessions.clear()
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            raw_text="",
            confidence=0.0,
            error="",
            threads=0,
        )


def _ocr_spec() -> dict:
    manifest = verified_next_manifest()
    spec = dict(manifest["models"]["ocr"])
    runtime = str(spec.get("runtime", "")).strip().lower()
    if runtime != CCT_RUNTIME:
        raise ValueError(
            f"Unexpected OCR runtime for CCT reader: {runtime or 'missing'}"
        )
    return spec


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


def _validate_session_contract(session, spec: dict) -> None:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    alphabet = str(spec.get("alphabet", CCT_DEFAULT_ALPHABET))
    expected_input = [
        1,
        int(spec["input_height"]),
        int(spec["input_width"]),
        3,
    ]
    expected_output = [
        1,
        int(spec["max_plate_slots"]),
        len(alphabet),
    ]
    if (
        len(inputs) != 1
        or len(outputs) != 1
        or list(inputs[0].shape) != expected_input
        or str(inputs[0].type) != "tensor(uint8)"
        or list(outputs[0].shape) != expected_output
        or str(outputs[0].type) not in {
            "tensor(float)",
            "tensor(float16)",
        }
    ):
        raise ValueError(
            "CCT ONNX input/output contract does not match signed metadata"
        )


def _load_session(engine_key=None) -> tuple[_SessionEntry, dict]:
    spec = _ocr_spec()
    path = str(spec["path"])
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    cache_key = (camera_key, path)
    with _cache_lock:
        cached = _sessions.get(cache_key)
        if cached is not None:
            _sessions.move_to_end(cache_key)
            return cached, spec

        import onnxruntime as ort

        session = ort.InferenceSession(
            path,
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        _validate_session_contract(session, spec)
        entry = _SessionEntry(
            session=session,
            input_name=session.get_inputs()[0].name,
            run_lock=threading.Lock(),
        )
        _sessions[cache_key] = entry
        cache_limit = max(
            MIN_CAMERA_SESSION_CACHE,
            parallel_camera_limit(),
        )
        while len(_sessions) > cache_limit:
            _sessions.popitem(last=False)
        return entry, spec


def _interpolation(name: str) -> int:
    return {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "area": cv2.INTER_AREA,
        "lanczos4": cv2.INTER_LANCZOS4,
    }.get(str(name).strip().lower(), cv2.INTER_LINEAR)


def _resize_with_padding(
    image: np.ndarray,
    width: int,
    height: int,
    interpolation: int,
    padding_color,
) -> np.ndarray:
    source_height, source_width = image.shape[:2]
    scale = min(
        height / max(1, source_height),
        width / max(1, source_width),
    )
    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )
    delta_width = width - resized_width
    delta_height = height - resized_height
    left = max(0, delta_width // 2)
    right = max(0, delta_width - left)
    top = max(0, delta_height // 2)
    bottom = max(0, delta_height - top)
    return cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=padding_color,
    )


def prepare_cct_input(image, spec: dict) -> np.ndarray | None:
    """Apply the signed FastPlateOCR image contract to one plate crop."""

    if image is None or getattr(image, "size", 0) == 0:
        return None
    width = int(spec.get("input_width", 128))
    height = int(spec.get("input_height", 64))
    color_mode = str(
        spec.get("image_color_mode", "rgb")
    ).strip().lower()
    layout = str(spec.get("input_layout", "nhwc")).strip().lower()
    dtype = str(spec.get("input_dtype", "uint8")).strip().lower()
    if (
        width < 32
        or height < 16
        or color_mode not in {"rgb", "grayscale"}
        or layout not in {"nhwc", "nchw"}
        or dtype not in {"uint8", "float32"}
    ):
        raise ValueError("Invalid signed CCT input specification")

    if color_mode == "rgb":
        prepared = (
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if image.ndim == 3
            else cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        )
        raw_padding = spec.get("padding_color", [114, 114, 114])
        if isinstance(raw_padding, (int, float)):
            padding_color = (int(raw_padding),) * 3
        else:
            values = [int(value) for value in raw_padding]
            if len(values) != 3:
                raise ValueError("RGB CCT padding_color must have three values")
            padding_color = tuple(values)
    else:
        prepared = (
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3
            else image.copy()
        )
        raw_padding = spec.get("padding_color", 114)
        if isinstance(raw_padding, (list, tuple)):
            raw_padding = raw_padding[0]
        padding_color = int(raw_padding)

    method = _interpolation(spec.get("interpolation", "linear"))
    if bool(spec.get("keep_aspect_ratio", False)):
        resized = _resize_with_padding(
            prepared,
            width,
            height,
            method,
            padding_color,
        )
    else:
        resized = cv2.resize(
            prepared,
            (width, height),
            interpolation=method,
        )
    if color_mode == "grayscale" and resized.ndim == 2:
        resized = resized[:, :, None]
    tensor = resized[None]
    if layout == "nchw":
        tensor = tensor.transpose(0, 3, 1, 2)
    if dtype == "float32":
        tensor = tensor.astype(np.float32)
        scale = float(spec.get("input_scale", 1.0))
        offset = float(spec.get("input_offset", 0.0))
        tensor = tensor * scale + offset
    else:
        tensor = tensor.astype(np.uint8)
    return np.ascontiguousarray(tensor)


def _probabilities(output: np.ndarray) -> np.ndarray:
    values = np.asarray(output, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if values.ndim == 1:
        raise ValueError("CCT output has no plate-position axis")
    if values.ndim != 2:
        raise ValueError(
            f"Unexpected CCT output rank: {tuple(np.asarray(output).shape)}"
        )
    row_sums = values.sum(axis=1)
    already_probabilities = bool(
        np.all(values >= -1e-6)
        and np.all(values <= 1.0 + 1e-6)
        and np.allclose(row_sums, 1.0, atol=2e-3)
    )
    if already_probabilities:
        return np.clip(values, 0.0, 1.0)
    shifted = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(
        exponent.sum(axis=1, keepdims=True),
        1e-12,
    )


def _allowed_indices(position: int, alphabet: str) -> list[int]:
    allowed = (
        set("0123456789")
        if position in CCT_DIGIT_POSITIONS
        else set(ALLOWED_PLATE_LETTERS)
    )
    return [
        index
        for index, character in enumerate(alphabet)
        if character in allowed
    ]


def decode_cct_hypotheses(
    output: np.ndarray,
    alphabet: str = CCT_DEFAULT_ALPHABET,
    max_plate_slots: int = CCT_MAX_PLATE_SLOTS,
    beam_width: int = 16,
    top_k: int = 5,
) -> list[dict]:
    """Decode position-wise probabilities with an Iranian-layout constraint."""

    alphabet = str(alphabet)
    if (
        len(set(alphabet)) != len(alphabet)
        or max_plate_slots != CCT_MAX_PLATE_SLOTS
    ):
        raise ValueError("Invalid CCT alphabet or plate slot count")
    probabilities = _probabilities(output)
    if probabilities.shape != (max_plate_slots, len(alphabet)):
        raise ValueError(
            "Unexpected CCT output shape: "
            + str(tuple(probabilities.shape))
        )

    per_position = []
    layout_conflict = False
    for position in range(max_plate_slots):
        allowed = _allowed_indices(position, alphabet)
        if not allowed:
            raise ValueError(
                f"CCT alphabet has no valid class for position {position}"
            )
        global_best = int(np.argmax(probabilities[position]))
        layout_conflict = layout_conflict or global_best not in allowed
        ranked = sorted(
            (
                (
                    alphabet[index],
                    float(probabilities[position, index]),
                )
                for index in allowed
            ),
            key=lambda row: (row[1], row[0]),
            reverse=True,
        )
        per_position.append(ranked[:3])

    beams = [("", 0.0, [])]
    for candidates in per_position:
        expanded = []
        for prefix, log_score, evidence in beams:
            for character, probability in candidates:
                expanded.append(
                    (
                        prefix + character,
                        log_score + math.log(max(probability, 1e-12)),
                        evidence + [{
                            "character": character,
                            "confidence": probability,
                        }],
                    )
                )
        expanded.sort(
            key=lambda row: (row[1], row[0]),
            reverse=True,
        )
        beams = expanded[:max(2, int(beam_width))]

    hypotheses = []
    for plate, log_score, positions in beams[:max(1, int(top_k))]:
        normalized = normalize_plate(plate)
        if not plausible_plate(normalized):
            continue
        confidence = float(
            math.exp(log_score / max_plate_slots)
        )
        position_margins = []
        for ranked in per_position:
            best = ranked[0][1]
            second = ranked[1][1] if len(ranked) > 1 else 0.0
            position_margins.append(max(0.0, best - second))
        hypotheses.append({
            "plate": format_iran_plate(normalized),
            "plate_norm": normalized,
            # Tracker weights must be non-negative. Keep the log value under
            # an explicit key for diagnostics instead of overloading score.
            "confidence": confidence,
            "score": confidence,
            "log_score": float(log_score),
            "positions": {
                position: evidence
                for position, evidence in enumerate(positions)
            },
            "min_position_confidence": min(
                evidence["confidence"] for evidence in positions
            ),
            "min_position_margin": min(position_margins),
            "layout_conflict": layout_conflict,
        })
    return hypotheses


def accept_cct_hypotheses(
    hypotheses: list[dict],
    min_confidence=0.58,
    min_position_confidence=0.42,
    min_position_margin=0.06,
    min_hypothesis_margin=0.025,
) -> dict:
    if not hypotheses:
        return {
            "plate": "ناخوانا",
            "plate_norm": "",
            "confidence": 0.0,
            "accepted": False,
            "reason": "no-valid-hypothesis",
            "hypotheses": [],
        }
    best = hypotheses[0]
    runner_up = (
        float(hypotheses[1]["confidence"])
        if len(hypotheses) > 1
        else 0.0
    )
    hypothesis_margin = float(best["confidence"]) - runner_up
    reasons = []
    if bool(best.get("layout_conflict")):
        reasons.append("layout-conflict")
    if float(best["confidence"]) < float(min_confidence):
        reasons.append("plate-confidence")
    if (
        float(best["min_position_confidence"])
        < float(min_position_confidence)
    ):
        reasons.append("position-confidence")
    if float(best["min_position_margin"]) < float(min_position_margin):
        reasons.append("position-margin")
    if hypothesis_margin < float(min_hypothesis_margin):
        reasons.append("hypothesis-margin")
    accepted = not reasons and plausible_plate(best["plate_norm"])
    return {
        "plate": best["plate"] if accepted else "ناخوانا",
        "plate_norm": best["plate_norm"] if accepted else "",
        "raw_plate_norm": best["plate_norm"],
        "confidence": round(float(best["confidence"]), 4),
        "accepted": accepted,
        "reason": ",".join(reasons),
        "hypothesis_margin": round(hypothesis_margin, 6),
        "hypotheses": hypotheses,
    }


def read_plate_cct(image, engine_key=None) -> dict:
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    try:
        entry, spec = _load_session(engine_key)
        tensor = prepare_cct_input(image, spec)
        if tensor is None:
            raise ValueError("Empty CCT input")
        with entry.run_lock:
            output = entry.session.run(
                None,
                {entry.input_name: tensor},
            )[0]
        hypotheses = decode_cct_hypotheses(
            output,
            alphabet=str(
                spec.get("alphabet", CCT_DEFAULT_ALPHABET)
            ),
            max_plate_slots=int(
                spec.get("max_plate_slots", CCT_MAX_PLATE_SLOTS)
            ),
            beam_width=int(spec.get("beam_width", 16)),
            top_k=int(spec.get("top_k", 5)),
        )
        result = accept_cct_hypotheses(
            hypotheses,
            min_confidence=float(
                spec.get("min_confidence", 0.58)
            ),
            min_position_confidence=float(
                spec.get("min_position_confidence", 0.42)
            ),
            min_position_margin=float(
                spec.get("min_position_margin", 0.06)
            ),
            min_hypothesis_margin=float(
                spec.get("min_hypothesis_margin", 0.025)
            ),
        )
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=str(spec["path"]),
                engine_key=camera_key,
                raw_text=str(result.get("raw_plate_norm", "")),
                confidence=float(result["confidence"]),
                error="",
                threads=threads_per_camera(),
            )
        return result
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                model_path="",
                engine_key=camera_key,
                raw_text="",
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        return {
            "plate": "ناخوانا",
            "plate_norm": "",
            "raw_plate_norm": "",
            "confidence": 0.0,
            "accepted": False,
            "reason": "runtime-error",
            "hypotheses": [],
        }
