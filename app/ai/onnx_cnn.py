"""Lightweight ONNX character-CNN fallback for Iranian plate crops.

The segmentation and classifier contract are adapted from the MIT-licensed
Platrix reference implementation by AliAkrami1375.  BC Vision runs this only
when the segmentation-free CRNN has no valid full-plate result.  It replaces
the general-purpose EasyOCR/Tesseract fallback without inventing missing
characters.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


CNN_SIZE = 32
NORMAL_HEIGHT = 96
MIN_CAMERA_SESSION_CACHE = 3
CNN_LABELS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "ا", "ب", "ت", "ج", "د", "س", "ص", "ط", "ع", "ق", "ل",
    "م", "ن", "ه", "و", "پ", "ژ", "ی",
]


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_last_status = {
    "engine": "cnn-onnx",
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "engine_key": "",
    "glyphs": 0,
    "raw_text": "",
    "confidence": 0.0,
    "error": "",
    "threads": 0,
}


def get_cnn_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_cnn_sessions() -> None:
    with _cache_lock:
        _sessions.clear()
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            glyphs=0,
            raw_text="",
            confidence=0.0,
            error="",
            threads=0,
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


def _verified_model_path():
    from .model_manager import (
        CNN_SHA256,
        CNN_SIZE as MODEL_SIZE,
        cnn_path,
        verify_file,
    )

    path = cnn_path()
    if not verify_file(path, CNN_SHA256, MODEL_SIZE):
        raise FileNotFoundError(
            f"Verified CNN ONNX model not found: {path}"
        )
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

        session = ort.InferenceSession(
            str(path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
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
        return entry


def warmup_cnn(engine_key=None) -> dict:
    """Load and verify the real CNN session without requiring a plate crop."""
    path = _verified_model_path()
    _load_session(engine_key=engine_key)
    with _cache_lock:
        _last_status.update(
            attempted=True,
            model_loaded=True,
            model_path=str(path),
            engine_key=str(
                engine_key if engine_key is not None else "default"
            ),
            error="",
            threads=threads_per_camera(),
        )
        return dict(_last_status)


def _enhance_gray(image) -> np.ndarray:
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image.copy()
    )
    height, width = gray.shape[:2]
    if height < 80:
        scale = min(4.0, 80.0 / max(height, 1))
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), 80),
            interpolation=cv2.INTER_CUBIC,
        )
    gray = cv2.bilateralFilter(gray, 5, 38, 38)
    gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 4),
    ).apply(gray)
    return gray


def _remove_frame(binary: np.ndarray) -> np.ndarray:
    height, width = binary.shape
    horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(width // 3, 12), 1),
    )
    vertical = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(height // 2, 12)),
    )
    lines = cv2.bitwise_or(
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal),
        cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical),
    )
    cleaned = cv2.subtract(binary, lines)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )
    result = cleaned.copy()
    for index in range(1, count):
        component_width = stats[index, cv2.CC_STAT_WIDTH]
        component_height = stats[index, cv2.CC_STAT_HEIGHT]
        area = stats[index, cv2.CC_STAT_AREA]
        if (
            component_width > 0.72 * width
            or component_height > 0.93 * height
            or area < 0.0006 * height * width
        ):
            result[labels == index] = 0
    return result


def _binarize(image) -> np.ndarray:
    gray = _enhance_gray(image)
    height, width = gray.shape[:2]
    scale = NORMAL_HEIGHT / max(height, 1)
    gray = cv2.resize(
        gray,
        (max(1, int(round(width * scale))), NORMAL_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    if binary.mean() > 127:
        binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
    )
    return _remove_frame(binary)


def _canonical_glyph(glyph: np.ndarray) -> np.ndarray:
    rows = np.where(glyph.sum(axis=1) > 0)[0]
    columns = np.where(glyph.sum(axis=0) > 0)[0]
    if rows.size and columns.size:
        glyph = glyph[
            rows[0]:rows[-1] + 1,
            columns[0]:columns[-1] + 1,
        ]
    height, width = glyph.shape
    side = max(height, width)
    padding = max(2, int(side * 0.18))
    canvas = np.zeros(
        (side + padding * 2, side + padding * 2),
        dtype=np.uint8,
    )
    offset_y = (canvas.shape[0] - height) // 2
    offset_x = (canvas.shape[1] - width) // 2
    canvas[
        offset_y:offset_y + height,
        offset_x:offset_x + width,
    ] = glyph
    return cv2.resize(
        canvas,
        (CNN_SIZE, CNN_SIZE),
        interpolation=cv2.INTER_AREA,
    )


def segment_characters(image, max_chars=10) -> list[np.ndarray]:
    if image is None or getattr(image, "size", 0) == 0:
        return []
    binary = _binarize(image)
    height, width = binary.shape
    active = (
        (binary > 0).sum(axis=0)
        > max(1, int(0.03 * height))
    )
    bands = []
    start = None
    for column in range(width):
        if active[column] and start is None:
            start = column
        elif not active[column] and start is not None:
            bands.append((start, column))
            start = None
    if start is not None:
        bands.append((start, width))

    merged = []
    minimum_gap = max(2, int(0.012 * width))
    for band in bands:
        if merged and band[0] - merged[-1][1] <= minimum_gap:
            merged[-1] = (merged[-1][0], band[1])
        else:
            merged.append(band)

    glyphs = []
    minimum_width = max(2, int(0.008 * width))
    for x1, x2 in merged:
        if x2 - x1 < minimum_width:
            continue
        band = binary[:, x1:x2]
        rows = np.where(band.sum(axis=1) > 0)[0]
        if not rows.size:
            continue
        y1, y2 = int(rows[0]), int(rows[-1]) + 1
        box_height = y2 - y1
        density = (
            int((band > 0).sum())
            / float(max(1, (x2 - x1) * box_height))
        )
        if box_height < 0.18 * height or density < 0.06:
            continue
        glyphs.append(_canonical_glyph(band[y1:y2]))
    return glyphs[:max_chars]


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values - values.max(axis=1, keepdims=True)
    exponent = np.exp(values)
    return exponent / np.maximum(
        exponent.sum(axis=1, keepdims=True),
        1e-12,
    )


def _decode(probs: np.ndarray) -> tuple[str, float]:
    if len(probs) != 8:
        return "", 0.0
    digit_indices = [
        index
        for index, label in enumerate(CNN_LABELS)
        if label.isdigit()
    ]
    letter_indices = [
        index
        for index, label in enumerate(CNN_LABELS)
        if not label.isdigit()
    ]
    characters = []
    confidences = []
    for position, row in enumerate(probs):
        allowed = letter_indices if position == 2 else digit_indices
        selected = max(allowed, key=lambda index: float(row[index]))
        characters.append(CNN_LABELS[selected])
        confidences.append(float(row[selected]))
    text = "".join(characters)
    return text, float(np.mean(confidences))


def read_plate_cnn(image, engine_key=None) -> tuple[str, float]:
    glyphs = segment_characters(image)
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    if len(glyphs) != 8:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                engine_key=camera_key,
                glyphs=len(glyphs),
                raw_text="",
                confidence=0.0,
                error="Expected exactly eight segmented glyphs",
                threads=threads_per_camera(),
            )
        return "", 0.0
    try:
        path = _verified_model_path()
        entry = _load_session(engine_key=engine_key)
        batch = (
            np.stack(glyphs)
            .astype(np.float32)
            .reshape(len(glyphs), 1, CNN_SIZE, CNN_SIZE)
            / 255.0
        )
        with entry.run_lock:
            logits = entry.session.run(
                None,
                {entry.input_name: batch},
            )[0]
        values = np.asarray(logits)
        if values.ndim != 2 or values.shape[1] != len(CNN_LABELS):
            raise ValueError(
                "Unexpected CNN output shape: "
                + str(tuple(values.shape))
            )
        text, confidence = _decode(_softmax(values))
        normalized = normalize_plate(text)
        valid = plausible_plate(normalized)
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=str(path),
                engine_key=camera_key,
                glyphs=len(glyphs),
                raw_text=normalized,
                confidence=round(confidence, 4),
                error="" if valid else "Invalid Iranian plate layout",
                threads=threads_per_camera(),
            )
        if not valid:
            return "", 0.0
        return format_iran_plate(normalized), round(confidence, 4)
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                engine_key=camera_key,
                glyphs=len(glyphs),
                raw_text="",
                confidence=0.0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        return "", 0.0
