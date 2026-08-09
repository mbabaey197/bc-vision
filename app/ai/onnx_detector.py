"""Exclusive YOLO11n/YOLOv8n Iranian plate localization with ONNX Runtime.

The operator-selected graph is hash verified and owns an isolated per-camera
session. Selectable inference never cascades into the other primary or the
retired secondary detector, so field-video counts remain attributable to one
model. Both graphs use the same single-class ``cx,cy,w,h,confidence`` output
contract and the configured CPU ceiling.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera


# Compatibility constants. Runtime input size comes from the selected model
# contract: YOLO11n uses 640 and Platrix YOLOv8n uses 416.
PRIMARY_SIZE = 640
FALLBACK_SIZE = 640
MIN_CAMERA_SESSION_CACHE = 3


@dataclass
class _SessionEntry:
    primary: object
    primary_input: str
    fallback: object | None
    fallback_input: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str, str], _SessionEntry] = (
    OrderedDict()
)
_last_status = {
    "engine": "yolo11n-plate-onnx",
    "selected_variant": "yolo11n",
    "attempted": False,
    "model_loaded": False,
    "primary_path": "",
    "fallback_path": "",
    "fallback_loaded": False,
    "fallback_used": False,
    "engine_key": "",
    "detections": 0,
    "error": "",
    "threads": 0,
}


def detector_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_detector_sessions() -> None:
    with _cache_lock:
        _sessions.clear()
        _last_status.update(
            engine="yolo11n-plate-onnx",
            selected_variant="yolo11n",
            attempted=False,
            model_loaded=False,
            primary_path="",
            fallback_path="",
            fallback_loaded=False,
            fallback_used=False,
            engine_key="",
            detections=0,
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


def _verified_paths(
    detector_variant=None,
) -> tuple[Path, Path | None]:
    from .model_manager import (
        detector_variant_spec,
        verify_file,
    )

    spec = detector_variant_spec(detector_variant)
    primary = Path(spec["path"])
    if not verify_file(primary, spec["sha256"], spec["size"]):
        raise FileNotFoundError(
            "Verified "
            + str(spec["variant"])
            + f" plate detector not found: {primary}"
        )
    # The historical recovery model remains installable for backward
    # compatibility but is deliberately excluded from selectable inference.
    return primary, None


def _load_session(
    engine_key=None,
    detector_variant=None,
) -> _SessionEntry:
    from .model_manager import normalize_detector_variant

    selected_variant = normalize_detector_variant(detector_variant)
    primary_path, fallback_path = _verified_paths(selected_variant)
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    cache_key = (
        camera_key,
        selected_variant + ":" + str(primary_path.resolve()),
        str(fallback_path.resolve()) if fallback_path else "",
    )
    with _cache_lock:
        cached = _sessions.get(cache_key)
        if cached is not None:
            _sessions.move_to_end(cache_key)
            return cached

        import onnxruntime as ort

        options = _session_options(ort)
        primary = ort.InferenceSession(
            str(primary_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        entry = _SessionEntry(
            primary=primary,
            primary_input=primary.get_inputs()[0].name,
            fallback=None,
            fallback_input="",
            run_lock=threading.Lock(),
        )
        _sessions[cache_key] = entry
        # A camera can legitimately own the operator-selected graph and a
        # differently signed Shadow graph. Keep both isolated sessions warm
        # while retaining the existing bounded LRU behavior.
        cache_limit = max(
            MIN_CAMERA_SESSION_CACHE,
            parallel_camera_limit() * 2,
        )
        while len(_sessions) > cache_limit:
            _sessions.popitem(last=False)
        return entry


def _letterbox(
    image: np.ndarray,
    size: int,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    height, width = image.shape[:2]
    ratio = min(size / max(height, 1), size / max(width, 1))
    resized_height = max(1, int(round(height * ratio)))
    resized_width = max(1, int(round(width * ratio)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_AREA
            if ratio < 1.0
            else cv2.INTER_LINEAR
        ),
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x = (size - resized_width) / 2.0
    pad_y = (size - resized_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[
        top:top + resized_height,
        left:left + resized_width,
    ] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = (
        np.transpose(rgb, (2, 0, 1))
        .astype(np.float32)
        / 255.0
    )
    return tensor[None], ratio, (float(left), float(top))


def _normalise_predictions(output) -> np.ndarray:
    values = np.asarray(output)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(
            "Unexpected detector output shape: "
            + str(tuple(np.asarray(output).shape))
        )
    if values.shape[0] == 5 and values.shape[1] != 5:
        values = values.T
    if values.shape[1] < 5:
        raise ValueError(
            "Unexpected detector output width: "
            + str(tuple(values.shape))
        )
    return values.astype(np.float32, copy=False)


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.45,
) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = (
        boxes[:, 0],
        boxes[:, 1],
        boxes[:, 2],
        boxes[:, 3],
    )
    areas = np.maximum(1.0, (x2 - x1) * (y2 - y1))
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(x1[index], x1[remaining])
        yy1 = np.maximum(y1[index], y1[remaining])
        xx2 = np.minimum(x2[index], x2[remaining])
        yy2 = np.minimum(y2[index], y2[remaining])
        intersection = (
            np.maximum(0.0, xx2 - xx1)
            * np.maximum(0.0, yy2 - yy1)
        )
        union = (
            areas[index]
            + areas[remaining]
            - intersection
        )
        overlap = intersection / np.maximum(union, 1e-9)
        order = remaining[overlap <= iou_threshold]
    return keep


def _run(
    frame: np.ndarray,
    session,
    input_name: str,
    size: int,
    confidence: float,
    max_results: int,
    method: str,
) -> list[dict]:
    tensor, ratio, (pad_x, pad_y) = _letterbox(frame, size)
    output = session.run(None, {input_name: tensor})[0]
    predictions = _normalise_predictions(output)
    boxes_xywh = predictions[:, :4]
    scores = predictions[:, 4:].max(axis=1)
    selected = scores >= max(0.05, min(0.99, float(confidence)))
    boxes_xywh = boxes_xywh[selected]
    scores = scores[selected]
    if not len(boxes_xywh):
        return []

    boxes = np.empty_like(boxes_xywh)
    boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
    boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
    boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
    boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / ratio

    height, width = frame.shape[:2]
    detections = []
    for index in _nms(boxes, scores)[:max_results]:
        x1, y1, x2, y2 = boxes[index]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        pad_width = max(2.0, box_width * 0.035)
        pad_height = max(2.0, box_height * 0.10)
        ix1 = max(0, min(width - 1, int(round(x1 - pad_width))))
        iy1 = max(0, min(height - 1, int(round(y1 - pad_height))))
        ix2 = max(ix1 + 1, min(width, int(round(x2 + pad_width))))
        iy2 = max(iy1 + 1, min(height, int(round(y2 + pad_height))))
        if ix2 - ix1 < 24 or iy2 - iy1 < 8:
            continue
        detections.append({
            "crop": frame[iy1:iy2, ix1:ix2].copy(),
            "bbox": (ix1, iy1, ix2, iy2),
            "confidence": float(scores[index]),
            "method": method,
            "crop_geometry": "axis-aligned",
            "direct_ocr_attempted": False,
        })
    return detections


def detect_plates_onnx(
    frame,
    min_confidence=0.25,
    max_results=4,
    engine_key=None,
    detector_variant=None,
    raise_on_error=False,
) -> list[dict]:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    from .model_manager import (
        detector_variant_spec,
        normalize_detector_variant,
    )

    selected_variant = normalize_detector_variant(detector_variant)
    spec = detector_variant_spec(selected_variant)
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    try:
        primary_path, _ = _verified_paths(selected_variant)
        entry = _load_session(
            engine_key=engine_key,
            detector_variant=selected_variant,
        )
        primary_size = max(
            320,
            min(
                640,
                int(os.environ.get(
                    "BCVISION_ONNX_DETECTOR_SIZE",
                    str(spec["input_size"]),
                )),
            ),
        )
        with entry.run_lock:
            detections = _run(
                frame,
                entry.primary,
                entry.primary_input,
                primary_size,
                min_confidence,
                max_results,
                str(spec["method"]),
            )
        with _cache_lock:
            _last_status.update(
                engine=str(spec["method"]),
                selected_variant=selected_variant,
                attempted=True,
                model_loaded=True,
                primary_path=str(primary_path),
                fallback_path="",
                fallback_loaded=False,
                fallback_used=False,
                engine_key=camera_key,
                detections=len(detections),
                error="",
                threads=threads_per_camera(),
            )
        return detections
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                engine=str(spec["method"]),
                selected_variant=selected_variant,
                attempted=True,
                model_loaded=False,
                primary_path=str(spec["path"]),
                fallback_path="",
                fallback_loaded=False,
                fallback_used=False,
                engine_key=camera_key,
                detections=0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        if raise_on_error:
            raise
        return []
