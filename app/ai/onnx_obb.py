"""YOLO OBB inference and mandatory perspective correction for RC13."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import threading

import cv2
import numpy as np

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .next_models import verified_next_manifest


MIN_CAMERA_SESSION_CACHE = 3


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_sessions: OrderedDict[tuple[str, str], _SessionEntry] = OrderedDict()
_last_status = {
    "engine": "yolo26-obb-onnx",
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "engine_key": "",
    "detections": 0,
    "error": "",
}


def obb_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_obb_sessions() -> None:
    with _cache_lock:
        _sessions.clear()
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            engine_key="",
            detections=0,
            error="",
        )


def order_quad_points(points) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).reshape(-1)
    ordered[0] = values[np.argmin(sums)]
    ordered[2] = values[np.argmax(sums)]
    ordered[1] = values[np.argmin(differences)]
    ordered[3] = values[np.argmax(differences)]
    return ordered


def rectify_plate(image, corners) -> np.ndarray | None:
    """Warp an OBB quadrilateral to a horizontal plate crop."""

    if image is None or getattr(image, "size", 0) == 0:
        return None
    source = order_quad_points(corners)
    top = float(np.linalg.norm(source[1] - source[0]))
    bottom = float(np.linalg.norm(source[2] - source[3]))
    left = float(np.linalg.norm(source[3] - source[0]))
    right = float(np.linalg.norm(source[2] - source[1]))
    width = max(top, bottom)
    height = max(left, right)
    if width < 12 or height < 5:
        return None
    target_width = max(64, min(512, int(round(width))))
    target_height = max(24, min(192, int(round(height))))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    crop = cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    if crop.shape[0] > crop.shape[1]:
        crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
    return crop


def obb_corners(cx, cy, width, height, angle) -> np.ndarray:
    value = float(angle)
    if abs(value) <= math.pi * 2.0:
        value = math.degrees(value)
    return cv2.boxPoints(
        (
            (float(cx), float(cy)),
            (max(1.0, float(width)), max(1.0, float(height))),
            value,
        )
    ).astype(np.float32)


def _normalise_predictions(output) -> np.ndarray:
    values = np.asarray(output)
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(
            "Unexpected OBB output shape: "
            + str(tuple(np.asarray(output).shape))
        )
    if (
        values.shape[0] in {6, 7}
        and values.shape[1] > 7
    ):
        values = values.T
    if values.shape[1] < 6:
        raise ValueError(
            "Unexpected OBB output width: "
            + str(tuple(values.shape))
        )
    return values.astype(np.float32, copy=False)


def decode_obb_output(
    output,
    min_confidence=0.25,
    max_results=8,
) -> list[dict]:
    """Decode both traditional YOLO OBB and end-to-end ``N x 7`` output."""

    predictions = _normalise_predictions(output)
    rows = []
    threshold = max(0.01, min(0.99, float(min_confidence)))
    for prediction in predictions:
        width = prediction.shape[0]
        if width == 6:
            cx, cy, box_w, box_h, score, angle = prediction
            class_id = 0
        elif width == 7:
            # YOLO26 end-to-end output extends the base
            # [x1, y1, x2, y2, confidence, class_id] row with angle.
            x1, y1, x2, y2, score, class_id, angle = prediction
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            box_w = max(1.0, float(x2 - x1))
            box_h = max(1.0, float(y2 - y1))
        else:
            cx, cy, box_w, box_h = prediction[:4]
            class_scores = prediction[4:-1]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])
            angle = prediction[-1]
        score = float(score)
        if score < threshold:
            continue
        rows.append({
            "corners": obb_corners(
                cx,
                cy,
                box_w,
                box_h,
                angle,
            ),
            "confidence": score,
            "class_id": int(class_id),
            "angle": float(angle),
        })
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    kept = []
    for row in rows:
        if all(
            _rotated_iou(row["corners"], other["corners"]) <= 0.45
            for other in kept
        ):
            kept.append(row)
            if len(kept) >= max(1, int(max_results)):
                break
    return kept


def _rotated_iou(left, right) -> float:
    left_points = np.asarray(left, dtype=np.float32).reshape(4, 2)
    right_points = np.asarray(right, dtype=np.float32).reshape(4, 2)
    left_area = abs(float(cv2.contourArea(left_points)))
    right_area = abs(float(cv2.contourArea(right_points)))
    if left_area <= 0 or right_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(
        left_points,
        right_points,
    )
    union = left_area + right_area - float(intersection)
    return float(intersection) / max(union, 1e-9)


def _letterbox(image: np.ndarray, size: int):
    height, width = image.shape[:2]
    ratio = min(size / max(height, 1), size / max(width, 1))
    target_w = max(1, int(round(width * ratio)))
    target_h = max(1, int(round(height * ratio)))
    resized = cv2.resize(
        image,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left = int(round((size - target_w) / 2.0 - 0.1))
    top = int(round((size - target_h) / 2.0 - 0.1))
    canvas[top:top + target_h, left:left + target_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0
    return tensor[None], ratio, (float(left), float(top))


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
    return options


def _load_session(engine_key=None):
    manifest = verified_next_manifest()
    path = manifest["models"]["detector"]["path"]
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


def detect_plates_obb(
    frame,
    min_confidence=0.25,
    max_results=4,
    engine_key=None,
) -> list[dict]:
    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        entry, manifest = _load_session(engine_key)
        detector_spec = manifest["models"]["detector"]
        input_size = max(
            320,
            min(1280, int(detector_spec.get("input_size", 640))),
        )
        tensor, ratio, (pad_x, pad_y) = _letterbox(frame, input_size)
        with entry.run_lock:
            output = entry.session.run(
                None,
                {entry.input_name: tensor},
            )[0]
        decoded = decode_obb_output(
            output,
            min_confidence=min_confidence,
            max_results=max_results * 2,
        )
        height, width = frame.shape[:2]
        results = []
        for row in decoded:
            corners = row["corners"].copy()
            corners[:, 0] = (corners[:, 0] - pad_x) / ratio
            corners[:, 1] = (corners[:, 1] - pad_y) / ratio
            corners[:, 0] = np.clip(corners[:, 0], 0, width - 1)
            corners[:, 1] = np.clip(corners[:, 1], 0, height - 1)
            crop = rectify_plate(frame, corners)
            if crop is None:
                continue
            x1, y1 = np.floor(corners.min(axis=0)).astype(int)
            x2, y2 = np.ceil(corners.max(axis=0)).astype(int)
            if x2 - x1 < 24 or y2 - y1 < 8:
                continue
            results.append({
                "crop": crop,
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "corners": corners.tolist(),
                "confidence": row["confidence"],
                "angle": row["angle"],
                "method": "yolo26-obb-onnx",
                "direct_ocr_attempted": False,
            })
            if len(results) >= max_results:
                break
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=detector_spec["path"],
                engine_key=camera_key,
                detections=len(results),
                error="",
            )
        return results
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                engine_key=camera_key,
                detections=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        return []
