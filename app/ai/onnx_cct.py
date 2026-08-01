"""FastPlateOCR CCT inference for fixed-layout Iranian license plates.

The production runtime intentionally depends only on NumPy, OpenCV and ONNX
Runtime.  Training remains an offline tool.  Every preprocessing and decoding
parameter is carried by the signed next-model manifest so an ONNX file cannot
silently be paired with an incompatible alphabet or image contract.
"""
from __future__ import annotations

from collections import OrderedDict
from contextlib import nullcontext
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
CCT_PREPROCESS_LEGACY = "stretch-v1"
CCT_PREPROCESS_DUAL_VIEW = "stretch-letterbox-geomean-v1"
CCT_FUSION_IDENTITY = "identity-v1"
CCT_FUSION_GEOMETRIC_MEAN = "geometric-mean-v1"
CCT_PREPROCESS_PROFILES = {
    CCT_PREPROCESS_LEGACY,
    CCT_PREPROCESS_DUAL_VIEW,
}


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


@dataclass(frozen=True)
class CCTPreparedView:
    name: str
    tensor: np.ndarray


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


def prepare_cct_views(
    image,
    spec: dict,
) -> tuple[CCTPreparedView, ...]:
    """Prepare the signed single- or dual-view CCT input contract."""

    profile = str(
        spec.get("preprocess_profile", CCT_PREPROCESS_LEGACY)
    ).strip().lower()
    if profile not in CCT_PREPROCESS_PROFILES:
        raise ValueError(f"Unsupported CCT preprocess profile: {profile}")

    stretch_spec = dict(spec)
    stretch_spec["keep_aspect_ratio"] = False
    stretched = prepare_cct_input(image, stretch_spec)
    if stretched is None:
        return ()
    views = [CCTPreparedView("stretch", stretched)]

    if profile == CCT_PREPROCESS_DUAL_VIEW:
        letterbox_spec = dict(spec)
        letterbox_spec["keep_aspect_ratio"] = True
        letterboxed = prepare_cct_input(image, letterbox_spec)
        if letterboxed is None:
            return ()
        views.append(CCTPreparedView("letterbox", letterboxed))
    return tuple(views)


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


def fuse_cct_outputs(
    outputs,
    method=CCT_FUSION_GEOMETRIC_MEAN,
) -> np.ndarray:
    """Fuse fixed-layout CCT outputs in normalized probability space."""

    rows = [
        np.clip(_probabilities(output), 1e-12, 1.0)
        for output in outputs
    ]
    if not rows:
        raise ValueError("CCT fusion requires at least one output")
    shape = rows[0].shape
    if any(row.shape != shape for row in rows):
        raise ValueError("CCT fusion outputs do not share one shape")

    selected = str(method).strip().lower()
    if selected == CCT_FUSION_IDENTITY:
        if len(rows) != 1:
            raise ValueError("Identity CCT fusion requires exactly one output")
        fused = rows[0]
    elif selected == CCT_FUSION_GEOMETRIC_MEAN:
        log_probabilities = np.mean(
            np.log(np.stack(rows, axis=0)),
            axis=0,
        )
        log_probabilities -= np.max(
            log_probabilities,
            axis=1,
            keepdims=True,
        )
        fused = np.exp(log_probabilities)
        fused /= np.maximum(
            fused.sum(axis=1, keepdims=True),
            1e-12,
        )
    else:
        raise ValueError(f"Unsupported CCT fusion method: {selected}")
    return np.ascontiguousarray(fused[None].astype(np.float32))


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


def _view_agreement(plates: list[str]) -> float:
    usable = [
        normalize_plate(plate)
        for plate in plates
        if plausible_plate(plate)
    ]
    if len(usable) <= 1:
        return 1.0
    ratios = []
    for position in range(CCT_MAX_PLATE_SLOTS):
        characters = [plate[position] for plate in usable]
        most_common = max(
            characters.count(value)
            for value in set(characters)
        )
        ratios.append(most_common / len(characters))
    return float(sum(ratios) / CCT_MAX_PLATE_SLOTS)


def infer_cct_session(
    session,
    input_name: str,
    image,
    spec: dict,
    *,
    run_lock=None,
) -> dict:
    """Run one signed CCT inference transaction with shared diagnostics."""

    profile = str(
        spec.get("preprocess_profile", CCT_PREPROCESS_LEGACY)
    ).strip().lower()
    views = prepare_cct_views(image, spec)
    if not views:
        raise ValueError("Empty CCT input")

    unique_tensors: list[np.ndarray] = []
    unique_outputs: list[np.ndarray] = []
    output_indices: list[int] = []
    context = run_lock if run_lock is not None else nullcontext()
    with context:
        for view in views:
            reused_index = next(
                (
                    index
                    for index, tensor in enumerate(unique_tensors)
                    if np.array_equal(tensor, view.tensor)
                ),
                None,
            )
            if reused_index is None:
                unique_tensors.append(view.tensor)
                unique_outputs.append(
                    session.run(
                        None,
                        {input_name: view.tensor},
                    )[0]
                )
                reused_index = len(unique_outputs) - 1
            output_indices.append(reused_index)

    view_outputs = [
        unique_outputs[index]
        for index in output_indices
    ]
    fusion_method = (
        str(
            spec.get(
                "fusion_method",
                CCT_FUSION_GEOMETRIC_MEAN,
            )
        ).strip().lower()
        if profile == CCT_PREPROCESS_DUAL_VIEW
        else CCT_FUSION_IDENTITY
    )
    fused_output = fuse_cct_outputs(
        view_outputs,
        method=fusion_method,
    )
    alphabet = str(spec.get("alphabet", CCT_DEFAULT_ALPHABET))
    slots = int(spec.get("max_plate_slots", CCT_MAX_PLATE_SLOTS))
    beam_width = int(spec.get("beam_width", 16))
    top_k = int(spec.get("top_k", 5))
    hypotheses = decode_cct_hypotheses(
        fused_output,
        alphabet=alphabet,
        max_plate_slots=slots,
        beam_width=beam_width,
        top_k=top_k,
    )
    result = accept_cct_hypotheses(
        hypotheses,
        min_confidence=float(spec.get("min_confidence", 0.58)),
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

    diagnostics = []
    view_plates = []
    for view, output in zip(views, view_outputs, strict=True):
        view_hypotheses = decode_cct_hypotheses(
            output,
            alphabet=alphabet,
            max_plate_slots=slots,
            beam_width=beam_width,
            top_k=top_k,
        )
        best_view = view_hypotheses[0] if view_hypotheses else {}
        plate_norm = normalize_plate(
            best_view.get("plate_norm", "")
        )
        view_plates.append(plate_norm)
        diagnostics.append({
            "name": view.name,
            "plate_norm": plate_norm,
            "confidence": round(
                float(best_view.get("confidence", 0.0)),
                6,
            ),
            "deduplicated": len(unique_outputs) < len(views),
        })

    agreement = _view_agreement(view_plates)
    minimum_view_agreement = float(
        spec.get(
            "min_view_agreement",
            0.0 if profile == CCT_PREPROCESS_LEGACY else 0.75,
        )
    )
    plausible_view_count = sum(
        plausible_plate(plate)
        for plate in view_plates
    )
    if (
        result["accepted"]
        and profile == CCT_PREPROCESS_DUAL_VIEW
        and (
            plausible_view_count != len(view_plates)
            or agreement + 1e-9 < minimum_view_agreement
        )
    ):
        disagreement_reason = (
            "incomplete-view-evidence"
            if plausible_view_count != len(view_plates)
            else "view-disagreement"
        )
        result.update(
            plate="ناخوانا",
            plate_norm="",
            accepted=False,
            reason=",".join(
                filter(
                    None,
                    (
                        str(result.get("reason", "")),
                        disagreement_reason,
                    ),
                )
            ),
        )

    best = hypotheses[0] if hypotheses else {}
    uncalibrated_confidence = float(
        best.get("confidence", result.get("confidence", 0.0))
    )
    if result["accepted"]:
        calibrated_confidence = uncalibrated_confidence
    else:
        # A failed structural or cross-view gate must not be presented as a
        # high-confidence complete read. This is a reliability score.
        calibrated_confidence = min(
            0.69,
            uncalibrated_confidence
            * (0.55 + 0.25 * agreement),
        )
    result["confidence"] = round(calibrated_confidence, 4)
    result["uncalibrated_confidence"] = round(
        uncalibrated_confidence,
        6,
    )
    result["preprocess_profile"] = profile
    result["fusion_method"] = fusion_method
    result["view_agreement"] = round(agreement, 4)
    result["whole_view_agreement"] = bool(
        view_plates
        and all(plate == view_plates[0] for plate in view_plates)
    )
    result["view_diagnostics"] = diagnostics
    result["plausible_view_count"] = plausible_view_count
    result["inference_count"] = len(unique_outputs)
    result["raw_plate_norm"] = normalize_plate(
        best.get("plate_norm", result.get("raw_plate_norm", ""))
    )
    # Rejected hypotheses are review evidence only. Repeating an unaccepted
    # result must never turn it into a confirmed physical event downstream.
    result["temporal_consensus_eligible"] = bool(result["accepted"])
    result["association_plate_norm"] = (
        result["raw_plate_norm"]
        if (
            result["accepted"]
            and result["whole_view_agreement"]
            and uncalibrated_confidence >= 0.72
        )
        else ""
    )
    result["association_plate_strong"] = bool(
        result["association_plate_norm"]
    )
    return result


def read_plate_cct(image, engine_key=None) -> dict:
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    try:
        entry, spec = _load_session(engine_key)
        result = infer_cct_session(
            entry.session,
            entry.input_name,
            image,
            spec,
            run_lock=entry.run_lock,
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
