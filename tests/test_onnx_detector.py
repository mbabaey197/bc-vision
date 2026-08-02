import hashlib
import sys
import threading
import types
from pathlib import Path

import numpy as np
import cv2

from app.ai import model_manager, onnx_detector


def test_light_detector_uses_fallback_and_two_thread_sessions(
    tmp_path,
    monkeypatch,
):
    primary_payload = b"primary-detector"
    fallback_payload = b"fallback-detector"
    primary_path = tmp_path / "plate_yolo.onnx"
    fallback_path = tmp_path / "plate_yolo_fallback.onnx"
    primary_path.write_bytes(primary_payload)
    fallback_path.write_bytes(fallback_payload)
    created = []

    class FakeOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            self.path = path
            self.options = sess_options
            self.providers = providers
            created.append(self)

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, _outputs, inputs):
            tensor = inputs["images"]
            assert tensor.shape[0:2] == (1, 3)
            if self.path == str(primary_path):
                return [np.array(
                    [[[100.0], [100.0], [40.0], [20.0], [0.01]]],
                    dtype=np.float32,
                )]
            return [np.array(
                [[[320.0], [320.0], [180.0], [70.0], [0.93]]],
                dtype=np.float32,
            )]

    fake_ort = types.SimpleNamespace(
        SessionOptions=FakeOptions,
        InferenceSession=FakeSession,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SHA256",
        hashlib.sha256(primary_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SIZE",
        len(primary_payload),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SHA256",
        hashlib.sha256(fallback_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SIZE",
        len(fallback_payload),
    )
    monkeypatch.setattr(
        model_manager,
        "detector_path",
        lambda: primary_path,
    )
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: fallback_path,
    )
    monkeypatch.setenv("BCVISION_CPU_THREADS", "9")
    onnx_detector.clear_detector_sessions()

    rows = onnx_detector.detect_plates_onnx(
        np.full((360, 640, 3), 127, dtype=np.uint8),
        engine_key="camera-1",
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "yolov8-onnx-light-fallback"
    assert rows[0]["crop"].size > 0
    assert len(created) == 2
    assert all(item.options.intra_op_num_threads == 2 for item in created)
    assert all(item.options.inter_op_num_threads == 1 for item in created)
    status = onnx_detector.detector_status()
    assert status["model_loaded"] is True
    assert status["fallback_used"] is True
    onnx_detector.clear_detector_sessions()


def test_detector_missing_model_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_manager,
        "detector_path",
        lambda: tmp_path / "missing.onnx",
    )
    onnx_detector.clear_detector_sessions()

    rows = onnx_detector.detect_plates_onnx(
        np.zeros((120, 240, 3), dtype=np.uint8)
    )

    assert rows == []
    assert onnx_detector.detector_status()["model_loaded"] is False


def test_cascade_merge_keeps_stronger_overlapping_detection():
    weak = {
        "bbox": (10, 10, 110, 40),
        "confidence": 0.42,
        "crop": np.zeros((30, 100, 3), dtype=np.uint8),
        "method": "primary",
    }
    strong = {
        "bbox": (12, 10, 112, 40),
        "confidence": 0.91,
        "crop": np.ones((30, 100, 3), dtype=np.uint8),
        "method": "fallback",
    }

    rows = onnx_detector._merge_detections(
        [weak],
        [strong],
        max_results=4,
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "fallback"
    assert rows[0]["confidence"] == 0.91


def test_weak_primary_detection_triggers_adaptive_fallback(monkeypatch):
    primary_session = object()
    fallback_session = object()
    calls = []
    entry = types.SimpleNamespace(
        primary=primary_session,
        primary_input="primary",
        fallback=fallback_session,
        fallback_input="fallback",
        run_lock=threading.Lock(),
        last_tile_rescue_at=0.0,
    )
    monkeypatch.setattr(
        onnx_detector,
        "_verified_paths",
        lambda: (Path("primary.onnx"), Path("fallback.onnx")),
    )
    monkeypatch.setattr(
        onnx_detector,
        "_load_session",
        lambda **_kwargs: entry,
    )

    def fake_run(
        _frame,
        session,
        _input_name,
        _size,
        _confidence,
        _max_results,
        method,
    ):
        calls.append(method)
        confidence = 0.41 if session is primary_session else 0.92
        return [{
            "bbox": (20, 30, 180, 70),
            "confidence": confidence,
            "crop": np.zeros((40, 160, 3), dtype=np.uint8),
            "method": method,
        }]

    monkeypatch.setattr(onnx_detector, "_run", fake_run)
    monkeypatch.setenv("BCVISION_DETECTOR_CASCADE", "adaptive")

    rows = onnx_detector.detect_plates_onnx(
        np.zeros((360, 640, 3), dtype=np.uint8),
    )

    assert calls == [
        "yolov8-onnx-light",
        "yolov8-onnx-light-fallback",
    ]
    assert len(rows) == 1
    assert rows[0]["confidence"] == 0.92
    assert rows[0]["method"] == "yolov8-onnx-light-fallback"


def test_landscape_tile_rescue_translates_boxes_to_full_frame(monkeypatch):
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

    def fake_run(
        _frame,
        _session,
        _input_name,
        _size,
        _confidence,
        _max_results,
        method,
    ):
        return [{
            "bbox": (100, 200, 240, 250),
            "confidence": 0.8,
            "crop": np.zeros((50, 140, 3), dtype=np.uint8),
            "method": method,
        }]

    monkeypatch.setattr(onnx_detector, "_run", fake_run)
    rows = onnx_detector._run_tiles(
        frame,
        object(),
        "input",
        0.2,
        4,
    )

    assert len(rows) == 2
    assert {row["bbox"] for row in rows} == {
        (100, 200, 240, 250),
        (830, 200, 970, 250),
    }
    assert all(row["method"] == "yolov8-onnx-tile-rescue" for row in rows)


def test_rotated_plate_crop_exposes_perspective_ocr_variant():
    crop = np.zeros((90, 260, 3), dtype=np.uint8)
    box = cv2.boxPoints(((130, 45), (220, 52), -8)).astype(np.int32)
    cv2.fillConvexPoly(crop, box, (235, 235, 235))
    for index in range(8):
        x = 35 + index * 25
        cv2.rectangle(crop, (x, 28), (x + 8, 65), (20, 20, 20), 2)

    rectified, quadrilateral = onnx_detector._rectified_ocr_variant(crop)

    assert rectified is not None
    assert rectified.shape[1] > rectified.shape[0] * 2
    assert quadrilateral.shape == (4, 2)
