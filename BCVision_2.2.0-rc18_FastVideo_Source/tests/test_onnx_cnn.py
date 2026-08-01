import hashlib
import sys
import types

import numpy as np

from app.ai import model_manager, onnx_cnn


def test_character_cnn_decodes_iranian_layout(tmp_path, monkeypatch):
    payload = b"verified-cnn"
    model_path = tmp_path / "ocr_cnn.onnx"
    model_path.write_bytes(payload)
    labels = onnx_cnn.CNN_LABELS
    wanted = "12ب34567"
    logits = np.full((8, len(labels)), -8.0, dtype=np.float32)
    for position, character in enumerate(wanted):
        logits[position, labels.index(character)] = 8.0

    class FakeOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

        def add_session_config_entry(self, _key, _value):
            pass

    class FakeSession:
        def __init__(self, _path, sess_options=None, providers=None):
            self.options = sess_options
            self.providers = providers

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _outputs, inputs):
            assert inputs["input"].shape == (8, 1, 32, 32)
            return [logits]

    fake_ort = types.SimpleNamespace(
        SessionOptions=FakeOptions,
        InferenceSession=FakeSession,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL="all"),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        model_manager,
        "CNN_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(model_manager, "CNN_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "cnn_path", lambda: model_path)
    monkeypatch.setattr(
        onnx_cnn,
        "segment_characters",
        lambda _image: [
            np.full((32, 32), 255, dtype=np.uint8)
            for _ in range(8)
        ],
    )
    monkeypatch.setenv("BCVISION_CPU_THREADS", "8")
    onnx_cnn.clear_cnn_sessions()

    text, confidence = onnx_cnn.read_plate_cnn(
        np.zeros((60, 220, 3), dtype=np.uint8),
        engine_key="camera-2",
    )

    assert text == "12-ب-345-67"
    assert confidence > 0.99
    status = onnx_cnn.get_cnn_status()
    assert status["model_loaded"] is True
    assert status["threads"] == 2
    onnx_cnn.clear_cnn_sessions()


def test_cnn_does_not_guess_when_segmentation_is_incomplete(monkeypatch):
    monkeypatch.setattr(
        onnx_cnn,
        "segment_characters",
        lambda _image: [np.zeros((32, 32), dtype=np.uint8)] * 7,
    )
    onnx_cnn.clear_cnn_sessions()

    assert onnx_cnn.read_plate_cnn(
        np.zeros((60, 220, 3), dtype=np.uint8)
    ) == ("", 0.0)
    assert onnx_cnn.get_cnn_status()["glyphs"] == 7
