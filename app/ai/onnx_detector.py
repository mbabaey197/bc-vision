"""Lightweight Iranian plate localization with YOLOv8 and ONNX Runtime.

The model and post-processing contract are based on the MIT-licensed Platrix
reference implementation by AliAkrami1375, adapted for BC Vision's verified
model store, per-camera sessions, two-thread CPU ceiling and fail-closed model
loading. The secondary detector and tiled high-resolution rescue run only when
the adaptive confidence policy says the primary evidence is insufficient.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import os
import threading
import time

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera


PRIMARY_SIZE = 416
FALLBACK_SIZE = 640
MIN_CAMERA_SESSION_CACHE = 3
ADAPTIVE_FALLBACK_CONFIDENCE = 0.58
TILE_MIN_WIDTH = 960


@dataclass
class _SessionEntry:
    primary: object
    primary_input: str
    fallback: object | None
    fallback_input: str
    run_lock: threading.Lock
    last_tile_rescue_at: float = 0.0


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str, str], _SessionEntry] = (
    OrderedDict()
)
_last_status = {
    "engine": "yolov8-onnx-light",
    "attempted": False,
    "model_loaded": False,
    "primary_path": "",
    "fallback_path": "",
    "fallback_loaded": False,
    "fallback_used": False,
    "cascade_mode": "adaptive",
    "tile_rescue_used": False,
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
            attempted=False,
            model_loaded=False,
            primary_path="",
            fallback_path="",
            fallback_loaded=False,
            fallback_used=False,
            cascade_mode="adaptive",
            tile_rescue_used=False,
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


def _verified_paths() -> tuple[Path, Path | None]:
    from .model_manager import (
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
        detector_fallback_path,
        detector_path,
        verify_file,
    )

    primary = detector_path()
    if not verify_file(primary, DETECTOR_SHA256, DETECTOR_SIZE):
        raise FileNotFoundError(
            f"Verified lightweight detector not found: {primary}"
        )
    fallback = detector_fallback_path()
    if not verify_file(
        fallback,
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
    ):
        fallback = None
    return primary, fallback


def _load_session(engine_key=None) -> _SessionEntry:
    primary_path, fallback_path = _verified_paths()
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    cache_key = (
        camera_key,
        str(primary_path.resolve()),
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
        fallback = None
        fallback_input = ""
        if fallback_path is not None:
            fallback = ort.InferenceSession(
                str(fallback_path),
                sess_options=_session_options(ort),
                providers=["CPUExecutionProvider"],
            )
            fallback_input = fallback.get_inputs()[0].name
        entry = _SessionEntry(
            primary=primary,
            primary_input=primary.get_inputs()[0].name,
            fallback=fallback,
            fallback_input=fallback_input,
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


def _bbox_iou(left, right) -> float:
    lx1, ly1, lx2, ly2 = (float(value) for value in left)
    rx1, ry1, rx2, ry2 = (float(value) for value in right)
    ix1, iy1 = max(lx1, rx1), max(ly1, ry1)
    ix2, iy2 = min(lx2, rx2), min(ly2, ry2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0.0:
        return 0.0
    left_area = max(1.0, (lx2 - lx1) * (ly2 - ly1))
    right_area = max(1.0, (rx2 - rx1) * (ry2 - ry1))
    return intersection / max(
        1e-9,
        left_area + right_area - intersection,
    )


def _merge_detections(
    *collections,
    max_results: int,
    iou_threshold: float = 0.42,
) -> list[dict]:
    """Merge cascade/tile detections without losing the best crop."""

    merged: list[dict] = []
    candidates = [
        dict(row)
        for rows in collections
        for row in (rows or [])
        if row.get("bbox")
    ]
    candidates.sort(
        key=lambda row: (
            float(row.get("confidence", 0.0)),
            (row["bbox"][2] - row["bbox"][0])
            * (row["bbox"][3] - row["bbox"][1]),
        ),
        reverse=True,
    )
    for candidate in candidates:
        duplicate = next(
            (
                row
                for row in merged
                if _bbox_iou(candidate["bbox"], row["bbox"])
                >= float(iou_threshold)
            ),
            None,
        )
        if duplicate is None:
            merged.append(candidate)
        elif (
            float(candidate.get("confidence", 0.0))
            > float(duplicate.get("confidence", 0.0))
        ):
            duplicate.clear()
            duplicate.update(candidate)
    return merged[:max(1, int(max_results))]


def _order_quad(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _rectified_ocr_variant(crop: np.ndarray):
    """Return a conservative perspective-normalized OCR view when reliable."""

    if crop is None or getattr(crop, "size", 0) == 0:
        return None, None
    height, width = crop.shape[:2]
    if height < 14 or width < 56:
        return None, None
    gray = (
        cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if crop.ndim == 3
        else crop
    )
    enhanced = cv2.createCLAHE(
        clipLimit=2.2,
        tileGridSize=(8, 4),
    ).apply(gray)
    edges = cv2.Canny(enhanced, 45, 150)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)),
        iterations=2,
    )
    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    crop_area = float(height * width)
    best = None
    for contour in sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )[:16]:
        rectangle = cv2.minAreaRect(contour)
        rect_width, rect_height = rectangle[1]
        short = max(1.0, min(rect_width, rect_height))
        long = max(rect_width, rect_height)
        ratio = long / short
        box_area = long * short
        coverage = box_area / max(crop_area, 1.0)
        if not (2.0 <= ratio <= 8.5 and 0.34 <= coverage <= 1.12):
            continue
        score = coverage * min(1.0, ratio / 4.0)
        if best is None or score > best[0]:
            best = (score, cv2.boxPoints(rectangle))
    if best is None:
        return None, None

    box = _order_quad(best[1])
    center = box.mean(axis=0)
    box = center + (box - center) * np.array(
        [1.04, 1.12],
        dtype=np.float32,
    )
    box[:, 0] = np.clip(box[:, 0], 0, width - 1)
    box[:, 1] = np.clip(box[:, 1], 0, height - 1)
    top_left, top_right, bottom_right, bottom_left = box
    output_width = int(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    ))
    output_height = int(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    ))
    if output_height > output_width:
        box = np.array(
            [bottom_left, top_left, top_right, bottom_right],
            dtype=np.float32,
        )
        output_width, output_height = output_height, output_width
    if output_width < 48 or output_height < 12:
        return None, None
    ratio = output_width / max(output_height, 1)
    if not 2.0 <= ratio <= 8.5:
        return None, None
    destination = np.array(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(box, destination)
    rectified = cv2.warpPerspective(
        crop,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if rectified.size == 0:
        return None, None
    return rectified, box


def _attach_ocr_variants(row: dict) -> dict:
    crop = row.get("crop")
    rectified, local_box = _rectified_ocr_variant(crop)
    if rectified is None:
        return row
    output = dict(row)
    output["ocr_crop_variants"] = [rectified]
    output["ocr_variant_geometry"] = "perspective-refined"
    if local_box is not None:
        x1, y1, _x2, _y2 = output["bbox"]
        output["ocr_quadrilateral"] = [
            [
                round(float(point[0]) + float(x1), 3),
                round(float(point[1]) + float(y1), 3),
            ]
            for point in local_box
        ]
    return output


def _tile_regions(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Two overlapping landscape tiles give distant plates more model pixels."""

    height, width = frame.shape[:2]
    if width < TILE_MIN_WIDTH or width / max(height, 1) < 1.25:
        return []
    tile_width = int(round(width * 0.62))
    return [
        (0, 0, tile_width, height),
        (width - tile_width, 0, width, height),
    ]


def _run_tiles(
    frame: np.ndarray,
    session,
    input_name: str,
    confidence: float,
    max_results: int,
) -> list[dict]:
    rows = []
    for tile_index, (x1, y1, x2, y2) in enumerate(
        _tile_regions(frame)
    ):
        tile = frame[y1:y2, x1:x2]
        detections = _run(
            tile,
            session,
            input_name,
            FALLBACK_SIZE,
            confidence,
            max_results,
            "yolov8-onnx-tile-rescue",
        )
        for row in detections:
            translated = dict(row)
            bx1, by1, bx2, by2 = translated["bbox"]
            translated["bbox"] = (
                bx1 + x1,
                by1 + y1,
                bx2 + x1,
                by2 + y1,
            )
            if translated.get("ocr_quadrilateral"):
                translated["ocr_quadrilateral"] = [
                    [
                        float(point[0]) + x1,
                        float(point[1]) + y1,
                    ]
                    for point in translated["ocr_quadrilateral"]
                ]
            translated["tile_index"] = tile_index
            rows.append(translated)
    return _merge_detections(
        rows,
        max_results=max_results,
    )


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
    return [_attach_ocr_variants(row) for row in detections]


def detect_plates_onnx(
    frame,
    min_confidence=0.25,
    max_results=4,
    engine_key=None,
) -> list[dict]:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    camera_key = str(
        engine_key if engine_key is not None else "default"
    )
    try:
        primary_path, fallback_path = _verified_paths()
        entry = _load_session(engine_key=engine_key)
        primary_size = max(
            320,
            min(
                640,
                int(os.environ.get(
                    "BCVISION_ONNX_DETECTOR_SIZE",
                    str(PRIMARY_SIZE),
                )),
            ),
        )
        cascade_mode = os.environ.get(
            "BCVISION_DETECTOR_CASCADE",
            "adaptive",
        ).strip().lower()
        if cascade_mode not in {"off", "adaptive", "accuracy"}:
            cascade_mode = "adaptive"
        with entry.run_lock:
            primary_detections = _run(
                frame,
                entry.primary,
                entry.primary_input,
                primary_size,
                min_confidence,
                max_results,
                "yolov8-onnx-light",
            )
            fallback_used = False
            weakest_primary = min(
                (
                    float(row.get("confidence", 0.0))
                    for row in primary_detections
                ),
                default=0.0,
            )
            fallback_needed = bool(
                entry.fallback is not None
                and cascade_mode != "off"
                and (
                    not primary_detections
                    or cascade_mode == "accuracy"
                    or weakest_primary
                    < ADAPTIVE_FALLBACK_CONFIDENCE
                )
            )
            fallback_detections = []
            if fallback_needed:
                fallback_used = True
                fallback_detections = _run(
                    frame,
                    entry.fallback,
                    entry.fallback_input,
                    FALLBACK_SIZE,
                    min(
                        0.45,
                        max(0.12, float(min_confidence) * 0.78),
                    ),
                    max_results,
                    "yolov8-onnx-light-fallback",
                )
            detections = _merge_detections(
                primary_detections,
                fallback_detections,
                max_results=max_results,
            )
            tile_rescue = []
            tile_interval = max(
                0.0,
                min(
                    3.0,
                    float(os.environ.get(
                        "BCVISION_TILE_RESCUE_INTERVAL",
                        "0.45",
                    )),
                ),
            )
            tile_ready = (
                time.monotonic()
                - float(getattr(entry, "last_tile_rescue_at", 0.0))
                >= tile_interval
            )
            tile_needed = bool(
                entry.fallback is not None
                and cascade_mode != "off"
                and tile_ready
                and _tile_regions(frame)
                and (
                    not detections
                    or (
                        cascade_mode == "accuracy"
                        and len(detections) < max_results
                    )
                    or (
                        len(detections) < 2
                        and max(
                            (
                                float(row.get("confidence", 0.0))
                                for row in detections
                            ),
                            default=0.0,
                        ) < 0.55
                    )
                )
            )
            if tile_needed:
                entry.last_tile_rescue_at = time.monotonic()
                fallback_used = True
                tile_rescue = _run_tiles(
                    frame,
                    entry.fallback,
                    entry.fallback_input,
                    min(
                        0.40,
                        max(0.10, float(min_confidence) * 0.72),
                    ),
                    max_results,
                )
                detections = _merge_detections(
                    detections,
                    tile_rescue,
                    max_results=max_results,
                )
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                primary_path=str(primary_path),
                fallback_path=(
                    str(fallback_path) if fallback_path else ""
                ),
                fallback_loaded=entry.fallback is not None,
                fallback_used=fallback_used,
                cascade_mode=cascade_mode,
                tile_rescue_used=bool(tile_rescue),
                engine_key=camera_key,
                detections=len(detections),
                error="",
                threads=threads_per_camera(),
            )
        return detections
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                fallback_used=False,
                tile_rescue_used=False,
                engine_key=camera_key,
                detections=0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        return []
