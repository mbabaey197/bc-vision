"""Whole-plate Iranian OCR using a CRNN+CTC model in ONNX Runtime.

The reader is intentionally independent from EasyOCR and PyTorch.  A separate
ONNX session is cached for each active camera so the existing per-camera CPU
budget remains enforceable.  Missing or invalid model files fail closed and
leave the legacy OCR path available.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
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


def read_plate_crnn(image, engine_key=None) -> tuple[str, float]:
    """Read one complete plate crop, returning raw evidence on invalid layout."""

    tensor = prepare_crnn_input(image)
    if tensor is None:
        return "", 0.0

    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        path = _verified_model_path()
        entry = _load_session(engine_key=engine_key)
        with entry.run_lock:
            output = entry.session.run(
                None,
                {entry.input_name: tensor},
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
        raw_text, confidence = ctc_greedy_decode(logits)
        normalized = normalize_plate(raw_text)
        text = (
            format_iran_plate(normalized)
            if plausible_plate(normalized)
            else normalized
        )
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=str(path),
                engine_key=camera_key,
                raw_text=normalized,
                confidence=round(float(confidence), 4),
                error="",
                threads=threads_per_camera(),
            )
        return text, round(float(confidence), 4)
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                engine_key=camera_key,
                raw_text="",
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        return "", 0.0
