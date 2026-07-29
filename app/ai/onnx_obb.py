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
    input_names: tuple[str, ...]
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


def decode_ppyoloe_r_outputs(
    pred_bboxes,
    pred_scores,
    min_confidence=0.25,
    nms_threshold=0.1,
    max_results=8,
) -> list[dict]:
    """Decode official PaddleDetection PP-YOLOE-R ONNX outputs.

    The exported graph returns ``B x N x 8`` quadrilaterals and
    ``B x C x N`` class scores. This matches PaddleDetection's official
    ``configs/rotate/tools/onnx_infer.py`` contract.
    """

    boxes = np.asarray(pred_bboxes, dtype=np.float32)
    scores = np.asarray(pred_scores, dtype=np.float32)
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    if scores.ndim == 3 and scores.shape[0] == 1:
        scores = scores[0]
    if boxes.ndim != 2 or boxes.shape[1] != 8:
        raise ValueError(
            "Unexpected PP-YOLOE-R bbox shape: "
            + str(tuple(np.asarray(pred_bboxes).shape))
        )
    if scores.ndim != 2:
        raise ValueError(
            "Unexpected PP-YOLOE-R score shape: "
            + str(tuple(np.asarray(pred_scores).shape))
        )
    if scores.shape[1] != boxes.shape[0]:
        if scores.shape[0] == boxes.shape[0]:
            scores = scores.T
        else:
            raise ValueError(
                "PP-YOLOE-R boxes/scores candidate count differs"
            )

    threshold = max(0.01, min(0.99, float(min_confidence)))
    iou_threshold = max(0.01, min(0.90, float(nms_threshold)))
    rows = []
    for class_id, class_scores in enumerate(scores):
        for index in np.flatnonzero(class_scores >= threshold):
            corners = boxes[int(index)].reshape(4, 2)
            if abs(float(cv2.contourArea(corners))) < 20.0:
                continue
            rows.append({
                "corners": corners,
                "confidence": float(class_scores[int(index)]),
                "class_id": int(class_id),
                "angle": float(cv2.minAreaRect(corners)[2]),
            })
    rows.sort(key=lambda row: row["confidence"], reverse=True)
    kept = []
    for row in rows:
        if all(
            _rotated_iou(row["corners"], other["corners"])
            <= iou_threshold
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


def prepare_ppyoloe_r_input(image: np.ndarray, spec: dict):
    """Apply the signed PaddleDetection test-time preprocessing contract."""

    height, width = image.shape[:2]
    target_h = int(spec["input_height"])
    target_w = int(spec["input_width"])
    ratio = min(
        target_h / max(1, height),
        target_w / max(1, width),
    )
    resized_h = max(1, int(round(height * ratio)))
    resized_w = max(1, int(round(width * ratio)))
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR,
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(
        np.float32,
    ) / 255.0
    mean = np.asarray(spec["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(spec["std"], dtype=np.float32).reshape(1, 1, 3)
    normalized = (rgb - mean) / std
    stride = int(spec.get("pad_to_stride", 32))
    pad_h = int(math.ceil(resized_h / stride) * stride)
    pad_w = int(math.ceil(resized_w / stride) * stride)
    canvas = np.zeros((pad_h, pad_w, 3), dtype=np.float32)
    canvas[:resized_h, :resized_w] = normalized
    tensor = np.transpose(canvas, (2, 0, 1))[None]
    im_shape = np.asarray(
        [[resized_h, resized_w]],
        dtype=np.float32,
    )
    scale_factor = np.asarray(
        [[resized_h / height, resized_w / width]],
        dtype=np.float32,
    )
    return tensor, im_shape, scale_factor


def _ppyoloe_r_feeds(
    input_names,
    tensor,
    im_shape,
    scale_factor,
) -> dict:
    feeds = {}
    for name in input_names:
        key = str(name).lower()
        if key == "image" or key.endswith(":image"):
            feeds[name] = tensor
        elif "im_shape" in key or "image_shape" in key:
            feeds[name] = im_shape
        elif "scale_factor" in key or key.endswith("scale"):
            feeds[name] = scale_factor
        else:
            raise ValueError(
                f"Unknown PP-YOLOE-R ONNX input: {name}"
            )
    required = {id(tensor), id(im_shape), id(scale_factor)}
    supplied = {id(value) for value in feeds.values()}
    if not required <= supplied:
        raise ValueError(
            "PP-YOLOE-R ONNX graph is missing a required input"
        )
    return feeds


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
            input_names=tuple(
                item.name for item in session.get_inputs()
            ),
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
        runtime = str(
            detector_spec.get("runtime", "yolo26-obb-onnx")
        ).strip().lower()
        coordinates_are_original = False
        if runtime == "ppyoloe-r-onnx":
            tensor, im_shape, scale_factor = prepare_ppyoloe_r_input(
                frame,
                detector_spec,
            )
            feeds = _ppyoloe_r_feeds(
                entry.input_names,
                tensor,
                im_shape,
                scale_factor,
            )
            with entry.run_lock:
                outputs = entry.session.run(None, feeds)
            if len(outputs) < 2:
                raise ValueError(
                    "PP-YOLOE-R ONNX graph returned fewer than two outputs"
                )
            decoded = decode_ppyoloe_r_outputs(
                outputs[0],
                outputs[1],
                min_confidence=max(
                    float(min_confidence),
                    float(detector_spec["score_threshold"]),
                ),
                nms_threshold=float(
                    detector_spec["nms_threshold"]
                ),
                max_results=min(
                    int(max_results) * 2,
                    int(detector_spec["max_results"]),
                ),
            )
            coordinates_are_original = True
            ratio, pad_x, pad_y = 1.0, 0.0, 0.0
        else:
            input_size = max(
                320,
                min(
                    1280,
                    int(detector_spec.get("input_size", 640)),
                ),
            )
            tensor, ratio, (pad_x, pad_y) = _letterbox(
                frame,
                input_size,
            )
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
            if not coordinates_are_original:
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
                "method": runtime,
                "direct_ocr_attempted": False,
            })
            if len(results) >= max_results:
                break
        with _cache_lock:
            _last_status.update(
                engine=runtime,
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
