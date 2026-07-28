import hashlib
import sys
import types

import numpy as np

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
