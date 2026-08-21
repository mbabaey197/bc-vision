"""Fail-closed ONNX adapter for a manifest-described custom YOLOX detector."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import threading

import cv2
import numpy as np

from app.cpu_budget import threads_per_camera


@dataclass
class _SessionEntry:
    session: object
    input_name: str
    output_name: str
    run_lock: threading.Lock


_cache_lock = threading.RLock()
_session: _SessionEntry | None = None
_session_identity = ""
_contract_spec: dict | None = None
_contract_fingerprint: tuple | None = None
_last_status = {
    "engine": "yolox-custom-onnx",
    "attempted": False,
    "model_loaded": False,
    "model_path": "",
    "model_revision": "",
    "manifest_path": "",
    "output_format": "",
    "engine_key": "",
    "detections": 0,
    "error": "",
    "threads": 0,
}


def yolox_status() -> dict:
    with _cache_lock:
        return dict(_last_status)


def clear_yolox_session() -> None:
    global _contract_fingerprint, _contract_spec, _session, _session_identity
    with _cache_lock:
        _session = None
        _session_identity = ""
        _contract_spec = None
        _contract_fingerprint = None
        _last_status.update(
            attempted=False,
            model_loaded=False,
            model_path="",
            model_revision="",
            manifest_path="",
            output_format="",
            engine_key="",
            detections=0,
            error="",
            threads=0,
        )


def _stat_fingerprint(path: Path) -> tuple:
    path = Path(path)
    try:
        stat = path.stat()
        return (
            str(path.resolve()),
            int(stat.st_size),
            int(stat.st_mtime_ns),
            int(getattr(stat, "st_ctime_ns", 0)),
            int(getattr(stat, "st_ino", 0)),
        )
    except OSError:
        return (str(path), -1, -1, -1, -1)


def _manifest_context() -> tuple:
    from .model_manager import yolox_manifest_path

    manifest = yolox_manifest_path()
    configured_model = os.environ.get("BCVISION_YOLOX_MODEL", "").strip()
    try:
        manifest_bytes = manifest.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    except OSError:
        manifest_digest = ""
    return (
        _stat_fingerprint(manifest),
        manifest_digest,
        configured_model,
    )


def _spec_fingerprint(spec: dict) -> tuple:
    configured_model = os.environ.get("BCVISION_YOLOX_MODEL", "").strip()
    model = (
        Path(configured_model).expanduser()
        if configured_model
        else Path(spec["path"])
    )
    return (
        _manifest_context(),
        _stat_fingerprint(model),
    )


def _runtime_spec() -> dict:
    """Hash once per installed model/manifest revision, not once per frame."""

    global _contract_fingerprint, _contract_spec
    with _cache_lock:
        if _contract_spec is not None:
            current = _spec_fingerprint(_contract_spec)
            if current == _contract_fingerprint:
                return dict(_contract_spec)

    from .model_manager import yolox_detector_spec

    # Do not pair a spec parsed from manifest A with the stat/digest of a
    # concurrently activated manifest B. That would otherwise pin A in the
    # cache until a later filesystem change.
    for _attempt in range(3):
        before = _manifest_context()
        spec = yolox_detector_spec()
        after = _manifest_context()
        if before != after:
            continue
        spec_manifest_digest = str(spec.get("manifest_digest", ""))
        if spec_manifest_digest and spec_manifest_digest != after[1]:
            continue
        fingerprint = _spec_fingerprint(spec)
        if fingerprint[0] != after:
            continue
        with _cache_lock:
            _contract_spec = dict(spec)
            _contract_fingerprint = fingerprint
        return spec
    raise RuntimeError("YOLOX manifest changed during contract verification")


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


def _load_session(spec: dict) -> _SessionEntry:
    global _session, _session_identity

    path = Path(spec["path"])
    if not spec.get("ready"):
        raise FileNotFoundError(
            spec.get("error") or f"Verified YOLOX model not found: {path}"
        )
    identity = (
        f"{path.resolve()}:{spec['sha256']}:"
        f"{spec.get('model_revision', '')}"
    )
    with _cache_lock:
        if _session is not None and _session_identity == identity:
            return _session

        import onnxruntime as ort

        runtime = ort.InferenceSession(
            str(path),
            sess_options=_session_options(ort),
            providers=["CPUExecutionProvider"],
        )
        inputs = runtime.get_inputs()
        if len(inputs) != 1:
            raise ValueError("YOLOX graph must expose exactly one input")
        input_type = str(getattr(inputs[0], "type", ""))
        if input_type != "tensor(float)":
            raise ValueError(
                f"YOLOX input must be tensor(float); got {input_type}"
            )
        shape = list(getattr(inputs[0], "shape", []) or [])
        expected = [
            1,
            3,
            int(spec["input_height"]),
            int(spec["input_width"]),
        ]
        if len(shape) != 4:
            raise ValueError(f"YOLOX input must be NCHW; got {shape}")
        for actual, wanted in zip(shape, expected):
            if isinstance(actual, int) and actual > 0 and actual != wanted:
                raise ValueError(
                    f"YOLOX input shape {shape} does not match manifest {expected}"
                )
        outputs = runtime.get_outputs()
        output_index = int(spec.get("output_index", 0))
        if not 0 <= output_index < len(outputs):
            raise ValueError("YOLOX output_index is outside graph outputs")
        entry = _SessionEntry(
            session=runtime,
            input_name=inputs[0].name,
            output_name=outputs[output_index].name,
            run_lock=threading.Lock(),
        )
        _session = entry
        _session_identity = identity
        return entry


def _letterbox(frame: np.ndarray, spec: dict):
    input_height = int(spec["input_height"])
    input_width = int(spec["input_width"])
    height, width = frame.shape[:2]
    ratio = min(
        input_width / max(1, width),
        input_height / max(1, height),
    )
    standard_top_left = spec.get("letterbox_mode") == "top-left"
    resized_width = max(
        1,
        int(width * ratio)
        if standard_top_left
        else int(round(width * ratio)),
    )
    resized_height = max(
        1,
        int(height * ratio)
        if standard_top_left
        else int(round(height * ratio)),
    )
    resized = cv2.resize(
        frame,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_LINEAR
            if standard_top_left or ratio >= 1.0
            else cv2.INTER_AREA
        ),
    )
    canvas = np.full(
        (input_height, input_width, 3),
        114,
        dtype=np.uint8,
    )
    if standard_top_left:
        left = top = 0
    else:
        left = int(round((input_width - resized_width) / 2.0 - 0.1))
        top = int(round((input_height - resized_height) / 2.0 - 0.1))
    canvas[top:top + resized_height, left:left + resized_width] = resized
    if spec.get("color", "rgb") == "rgb":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    tensor = np.ascontiguousarray(
        np.transpose(canvas, (2, 0, 1)),
        dtype=np.float32,
    )
    tensor *= float(spec["input_scale"])
    return tensor[None], ratio, (float(left), float(top))


def _rows(output) -> np.ndarray:
    values = np.asarray(output)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError(f"YOLOX output must reduce to rows; got {values.shape}")
    return values.astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _score_rows(rows: np.ndarray, spec: dict):
    class_count = int(spec["class_count"])
    required = 5 + class_count
    if rows.shape[1] != required:
        raise ValueError(
            f"YOLOX output width {rows.shape[1]} does not match manifest {required}"
        )
    objectness = rows[:, 4]
    classes = rows[:, 5:]
    if spec.get("scores_are_logits"):
        objectness = _sigmoid(objectness)
        classes = _sigmoid(classes)
    elif (
        not np.all(np.isfinite(objectness))
        or not np.all(np.isfinite(classes))
        or np.any(objectness < 0.0)
        or np.any(objectness > 1.0)
        or np.any(classes < 0.0)
        or np.any(classes > 1.0)
    ):
        raise ValueError("YOLOX probability scores must be finite within 0..1")
    class_id = int(spec["plate_class_id"])
    scores = objectness * classes[:, class_id]
    if class_count > 1:
        scores = np.where(
            np.argmax(classes, axis=1) == class_id,
            scores,
            0.0,
        )
    return scores


def _grid(input_height: int, input_width: int, strides: list[int]):
    grids = []
    expanded_strides = []
    for stride in strides:
        height = input_height // int(stride)
        width = input_width // int(stride)
        yv, xv = np.meshgrid(
            np.arange(height),
            np.arange(width),
            indexing="ij",
        )
        grids.append(np.stack((xv, yv), axis=2).reshape(-1, 2))
        expanded_strides.append(
            np.full((height * width, 1), int(stride), dtype=np.float32)
        )
    return (
        np.concatenate(grids, axis=0).astype(np.float32),
        np.concatenate(expanded_strides, axis=0),
    )


def decode_output(output, spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Decode one manifest-described output into xyxy boxes and scores."""

    rows = _rows(output)
    if len(rows) > 250_000:
        raise ValueError("YOLOX output exceeds the prediction-row ceiling")
    if not np.all(np.isfinite(rows)):
        raise ValueError("YOLOX output contains NaN or infinity")
    output_format = spec["output_format"]
    expected_space = (
        "grid" if output_format == "raw-grid" else "input-pixels"
    )
    if str(spec.get("coordinate_space", "")) != expected_space:
        raise ValueError(
            f"YOLOX {output_format} coordinate_space must be {expected_space}"
        )
    if output_format == "nms-xyxy":
        if rows.shape[1] not in {5, 6}:
            raise ValueError("nms-xyxy output must contain 5 or 6 columns")
        if rows.shape[1] == 5 and (
            int(spec["class_count"]) != 1
            or int(spec["plate_class_id"]) != 0
        ):
            raise ValueError(
                "multi-class nms-xyxy output must include class_id"
            )
        if spec.get("scores_are_logits"):
            rows = rows.copy()
            rows[:, 4] = _sigmoid(rows[:, 4])
        if (
            not np.all(np.isfinite(rows[:, 4]))
            or np.any(rows[:, 4] < 0.0)
            or np.any(rows[:, 4] > 1.0)
        ):
            raise ValueError("nms-xyxy scores must be finite within 0..1")
        if rows.shape[1] == 6:
            class_values = rows[:, 5]
            rounded = np.rint(class_values)
            if (
                not np.all(np.isfinite(class_values))
                or np.any(np.abs(class_values - rounded) > 1e-5)
                or np.any(rounded < 0)
                or np.any(rounded >= int(spec["class_count"]))
            ):
                raise ValueError("nms-xyxy class ids are invalid")
            rows = rows[
                rounded.astype(np.int64) == int(spec["plate_class_id"])
            ]
        return rows[:, :4].copy(), rows[:, 4].copy()

    scores = _score_rows(rows, spec)
    centers = rows[:, :4].copy()
    if output_format == "raw-grid":
        grid, strides = _grid(
            int(spec["input_height"]),
            int(spec["input_width"]),
            [int(value) for value in spec["strides"]],
        )
        if len(grid) != len(centers):
            raise ValueError(
                f"YOLOX raw-grid rows {len(centers)} do not match grid {len(grid)}"
            )
        centers[:, :2] = (centers[:, :2] + grid) * strides
        centers[:, 2:4] = np.exp(np.clip(centers[:, 2:4], -16.0, 16.0)) * strides
    elif output_format != "decoded-cxcywh":
        raise ValueError(f"Unsupported YOLOX output format: {output_format}")

    boxes = np.empty_like(centers)
    boxes[:, 0] = centers[:, 0] - centers[:, 2] / 2.0
    boxes[:, 1] = centers[:, 1] - centers[:, 3] / 2.0
    boxes[:, 2] = centers[:, 0] + centers[:, 2] / 2.0
    boxes[:, 3] = centers[:, 1] + centers[:, 3] / 2.0
    return boxes, scores


def validate_yolox_model(spec: dict) -> dict:
    """Load and dry-run the exact graph/contract before manifest activation."""

    entry = _load_session(spec)
    tensor = np.zeros(
        (
            1,
            3,
            int(spec["input_height"]),
            int(spec["input_width"]),
        ),
        dtype=np.float32,
    )
    with entry.run_lock:
        output = entry.session.run(
            [entry.output_name],
            {entry.input_name: tensor},
        )[0]
    boxes, scores = decode_output(output, spec)
    if len(boxes) != len(scores) or boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("YOLOX decoder returned an invalid canonical shape")
    return {
        "outputs": 1,
        "prediction_rows": len(scores),
        "output_format": str(spec["output_format"]),
    }


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold=0.45) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        rest = order[1:]
        intersection = (
            np.maximum(0.0, np.minimum(x2[index], x2[rest]) - np.maximum(x1[index], x1[rest]))
            * np.maximum(0.0, np.minimum(y2[index], y2[rest]) - np.maximum(y1[index], y1[rest]))
        )
        union = areas[index] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-9)
        order = rest[iou <= float(threshold)]
    return keep


def _detections(
    frame: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    ratio: float,
    padding: tuple[float, float],
    confidence: float,
    max_results: int,
    spec: dict,
) -> list[dict]:
    selected = np.flatnonzero(
        np.isfinite(scores)
        & np.all(np.isfinite(boxes), axis=1)
        & (scores >= max(0.05, min(0.99, float(confidence))))
    )
    if not len(selected):
        return []
    if len(selected) > 1000:
        ranked = np.argsort(scores[selected])[-1000:]
        selected = selected[ranked]
    boxes = boxes[selected].copy()
    scores = scores[selected]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - padding[0]) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - padding[1]) / ratio
    height, width = frame.shape[:2]
    raw_widths = boxes[:, 2] - boxes[:, 0]
    raw_heights = boxes[:, 3] - boxes[:, 1]
    raw_areas = raw_widths * raw_heights
    raw_aspect = raw_widths / np.maximum(raw_heights, 1e-6)
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(
        clipped[:, [0, 2]],
        0.0,
        float(width),
    )
    clipped[:, [1, 3]] = np.clip(
        clipped[:, [1, 3]],
        0.0,
        float(height),
    )
    visible_widths = np.maximum(0.0, clipped[:, 2] - clipped[:, 0])
    visible_heights = np.maximum(0.0, clipped[:, 3] - clipped[:, 1])
    visible_areas = visible_widths * visible_heights
    visible_fraction = visible_areas / np.maximum(raw_areas, 1e-6)
    preclip_geometry = (
        (raw_widths >= 20.0)
        & (raw_heights >= 6.0)
        & (raw_aspect >= 1.2)
        & (raw_aspect <= 12.0)
        & (raw_widths <= float(width) * 2.0)
        & (raw_heights <= float(height) * 1.5)
        & (raw_areas <= float(width * height) * 0.65)
        & (visible_fraction >= 0.50)
    )
    boxes = clipped[preclip_geometry]
    scores = scores[preclip_geometry]
    if not len(boxes):
        return []
    box_widths = boxes[:, 2] - boxes[:, 0]
    box_heights = boxes[:, 3] - boxes[:, 1]
    aspect = box_widths / np.maximum(box_heights, 1e-6)
    valid_geometry = (
        (box_widths >= 20.0)
        & (box_heights >= 6.0)
        & (aspect >= 1.2)
        & (aspect <= 12.0)
        & (
            box_widths * box_heights
            <= float(width * height) * 0.65
        )
    )
    boxes = boxes[valid_geometry]
    scores = scores[valid_geometry]
    if not len(boxes):
        return []
    rows = []
    for index in _nms(boxes, scores)[:max(1, int(max_results))]:
        x1, y1, x2, y2 = boxes[index]
        if not all(math.isfinite(float(value)) for value in (x1, y1, x2, y2)):
            continue
        box_width = max(0.0, float(x2 - x1))
        box_height = max(0.0, float(y2 - y1))
        pad_width = max(2.0, box_width * 0.035)
        pad_height = max(2.0, box_height * 0.10)
        ix1 = max(0, min(width - 1, int(round(x1 - pad_width))))
        iy1 = max(0, min(height - 1, int(round(y1 - pad_height))))
        ix2 = max(ix1 + 1, min(width, int(round(x2 + pad_width))))
        iy2 = max(iy1 + 1, min(height, int(round(y2 + pad_height))))
        if ix2 - ix1 < 24 or iy2 - iy1 < 8:
            continue
        rows.append({
            "crop": frame[iy1:iy2, ix1:ix2].copy(),
            "bbox": (ix1, iy1, ix2, iy2),
            "confidence": float(scores[index]),
            "method": str(spec["method"]),
            "model_revision": str(spec.get("model_revision", "")),
            "crop_geometry": "axis-aligned",
            "direct_ocr_attempted": False,
        })
    return rows


def detect_plates_yolox(
    frame,
    min_confidence=0.25,
    max_results=4,
    engine_key=None,
    raise_on_error=False,
    expected_model_revision=None,
    runtime_metadata=None,
) -> list[dict]:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    spec = {}
    camera_key = str(engine_key if engine_key is not None else "default")
    try:
        spec = _runtime_spec()
        model_revision = str(spec.get("model_revision", "")).strip()
        if runtime_metadata is not None:
            runtime_metadata.update(
                detector_variant="yolox",
                detector_model_revision=model_revision,
                detector_manifest_path=str(spec.get("manifest_path", "")),
            )
        expected_revision = str(expected_model_revision or "").strip()
        if expected_revision and model_revision != expected_revision:
            raise RuntimeError(
                "YOLOX detector revision changed during pinned inference"
            )
        entry = _load_session(spec)
        tensor, ratio, padding = _letterbox(frame, spec)
        with entry.run_lock:
            output = entry.session.run(
                [entry.output_name],
                {entry.input_name: tensor},
            )[0]
        boxes, scores = decode_output(output, spec)
        rows = _detections(
            frame,
            boxes,
            scores,
            ratio,
            padding,
            min_confidence,
            max_results,
            spec,
        )
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=True,
                model_path=str(spec["path"]),
                model_revision=str(spec.get("model_revision", "")),
                manifest_path=str(spec["manifest_path"]),
                output_format=str(spec["output_format"]),
                engine_key=camera_key,
                detections=len(rows),
                error="",
                threads=threads_per_camera(),
            )
        return rows
    except Exception as exc:
        with _cache_lock:
            _last_status.update(
                attempted=True,
                model_loaded=False,
                model_path=str(spec.get("path", "")),
                model_revision=str(spec.get("model_revision", "")),
                manifest_path=str(spec.get("manifest_path", "")),
                output_format=str(spec.get("output_format", "")),
                engine_key=camera_key,
                detections=0,
                error=f"{type(exc).__name__}: {exc}",
                threads=threads_per_camera(),
            )
        if raise_on_error:
            raise
        return []
