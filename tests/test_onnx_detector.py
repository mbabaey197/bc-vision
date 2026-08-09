import hashlib
import sys
import types

import numpy as np
import pytest

from app.ai import model_manager, onnx_detector


def test_selected_yolov8n_is_exclusive_and_uses_two_thread_session(
    tmp_path,
    monkeypatch,
):
    primary_payload = b"yolov8n-detector"
    fallback_payload = b"fallback-detector"
    primary_path = tmp_path / "plate_yolov8n.onnx"
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
            assert tensor.shape == (1, 3, 416, 416)
            if self.path == str(primary_path):
                return [np.array(
                    [[[208.0], [208.0], [120.0], [42.0], [0.93]]],
                    dtype=np.float32,
                )]
            raise AssertionError("retired fallback must not be loaded")

    fake_ort = types.SimpleNamespace(
        SessionOptions=FakeOptions,
        InferenceSession=FakeSession,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SHA256",
        hashlib.sha256(primary_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SIZE",
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
        "yolov8n_detector_path",
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
        detector_variant="yolov8n",
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "yolov8n-plate-onnx"
    assert rows[0]["crop"].size > 0
    assert len(created) == 1
    assert all(item.options.intra_op_num_threads == 2 for item in created)
    assert all(item.options.inter_op_num_threads == 1 for item in created)
    status = onnx_detector.detector_status()
    assert status["model_loaded"] is True
    assert status["selected_variant"] == "yolov8n"
    assert status["fallback_loaded"] is False
    assert status["fallback_used"] is False
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


def test_missing_selected_yolov8n_never_loads_available_yolo11n(
    tmp_path,
    monkeypatch,
):
    yolo11 = tmp_path / "plate_yolo11n.onnx"
    yolo11.write_bytes(b"available-yolo11n")
    monkeypatch.setattr(
        model_manager,
        "yolov8n_detector_path",
        lambda: tmp_path / "missing-yolov8n.onnx",
    )
    monkeypatch.setattr(model_manager, "detector_path", lambda: yolo11)
    onnx_detector.clear_detector_sessions()

    rows = onnx_detector.detect_plates_onnx(
        np.zeros((120, 240, 3), dtype=np.uint8),
        detector_variant="yolov8n",
    )

    status = onnx_detector.detector_status()
    assert rows == []
    assert status["selected_variant"] == "yolov8n"
    assert status["model_loaded"] is False
    assert "missing-yolov8n.onnx" in status["error"]


def test_selected_inference_error_can_be_propagated_per_call(
    tmp_path,
    monkeypatch,
):
    selected_path = tmp_path / "plate_yolov8n.onnx"
    entry = types.SimpleNamespace(
        primary=object(),
        primary_input="images",
        run_lock=onnx_detector.threading.Lock(),
    )
    monkeypatch.setattr(
        onnx_detector,
        "_verified_paths",
        lambda _variant=None: (selected_path, None),
    )
    monkeypatch.setattr(
        onnx_detector,
        "_load_session",
        lambda **_kwargs: entry,
    )
    monkeypatch.setattr(
        onnx_detector,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("invalid ONNX output")
        ),
    )
    onnx_detector.clear_detector_sessions()

    with pytest.raises(RuntimeError, match="invalid ONNX output"):
        onnx_detector.detect_plates_onnx(
            np.zeros((120, 240, 3), dtype=np.uint8),
            engine_key="camera-7",
            detector_variant="yolov8n",
            raise_on_error=True,
        )

    status = onnx_detector.detector_status()
    assert status["selected_variant"] == "yolov8n"
    assert status["engine_key"] == "camera-7"
    assert status["model_loaded"] is False
    assert status["error"] == "RuntimeError: invalid ONNX output"


def test_same_camera_cache_is_isolated_by_detector_variant(
    tmp_path,
    monkeypatch,
):
    yolo11_payload = b"cache-yolo11n"
    yolo8_payload = b"cache-yolov8n"
    yolo11 = tmp_path / "plate_yolo11n.onnx"
    yolo8 = tmp_path / "plate_yolov8n.onnx"
    yolo11.write_bytes(yolo11_payload)
    yolo8.write_bytes(yolo8_payload)
    created = []

    class Options:
        def add_session_config_entry(self, *_args):
            pass

    class Session:
        def __init__(self, path, **_kwargs):
            created.append(path)

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, *_args, **_kwargs):
            return [np.zeros((1, 5, 1), dtype=np.float32)]

    monkeypatch.setitem(
        sys.modules,
        "onnxruntime",
        types.SimpleNamespace(
            SessionOptions=Options,
            InferenceSession=Session,
            ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
            GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
        ),
    )
    monkeypatch.setattr(model_manager, "detector_path", lambda: yolo11)
    monkeypatch.setattr(
        model_manager,
        "yolov8n_detector_path",
        lambda: yolo8,
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SHA256",
        hashlib.sha256(yolo11_payload).hexdigest(),
    )
    monkeypatch.setattr(model_manager, "DETECTOR_SIZE", len(yolo11_payload))
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SHA256",
        hashlib.sha256(yolo8_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SIZE",
        len(yolo8_payload),
    )
    monkeypatch.setattr(
        onnx_detector,
        "parallel_camera_limit",
        lambda: 2,
    )
    frame = np.zeros((120, 240, 3), dtype=np.uint8)
    onnx_detector.clear_detector_sessions()

    for camera in ("camera-7", "camera-8"):
        onnx_detector.detect_plates_onnx(
            frame,
            engine_key=camera,
            detector_variant="yolo11n",
        )
        onnx_detector.detect_plates_onnx(
            frame,
            engine_key=camera,
            detector_variant="yolov8n",
        )

    assert created == [str(yolo11), str(yolo8)] * 2
    assert len(onnx_detector._sessions) == 4
    onnx_detector.clear_detector_sessions()
