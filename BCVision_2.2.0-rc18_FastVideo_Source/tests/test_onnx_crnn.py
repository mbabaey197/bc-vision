import hashlib
import sys
import types

import numpy as np

from app.ai import model_manager, onnx_crnn


def _logits_for(text: str) -> np.ndarray:
    blank = len(onnx_crnn.CRNN_LABELS)
    indices = [blank]
    for character in text:
        index = onnx_crnn.CRNN_LABELS.index(character)
        indices.extend((index, index, blank))
    logits = np.full(
        (len(indices), blank + 1),
        -8.0,
        dtype=np.float32,
    )
    for timestep, index in enumerate(indices):
        logits[timestep, index] = 8.0
    return logits


def test_ctc_decoder_collapses_repeats_and_blank():
    text, confidence = onnx_crnn.ctc_greedy_decode(
        _logits_for("12ب34567")
    )

    assert text == "12ب34567"
    assert confidence > 0.99


def test_crnn_preprocessing_has_expected_shape_and_range():
    source = np.full((60, 220, 3), 127, dtype=np.uint8)
    tensor = onnx_crnn.prepare_crnn_input(source)

    assert tensor.shape == (1, 1, 32, 128)
    assert tensor.dtype == np.float32
    assert 0.49 < float(tensor.mean()) < 0.51


def test_per_camera_onnx_sessions_use_two_thread_ceiling(
    tmp_path,
    monkeypatch,
):
    payload = b"verified-crnn-model"
    model_path = tmp_path / "ocr_crnn.onnx"
    model_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    created = []
    logits = _logits_for("31ط55674")

    class FakeOptions:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0
            self.entries = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class FakeSession:
        def __init__(
            self,
            path,
            sess_options=None,
            providers=None,
        ):
            self.path = path
            self.options = sess_options
            self.providers = providers
            created.append(self)

        def get_inputs(self):
            return [types.SimpleNamespace(name="input")]

        def run(self, _outputs, inputs):
            assert inputs["input"].shape == (1, 1, 32, 128)
            return [logits[None, ...]]

    fake_ort = types.SimpleNamespace(
        SessionOptions=FakeOptions,
        InferenceSession=FakeSession,
        ExecutionMode=types.SimpleNamespace(
            ORT_SEQUENTIAL="sequential"
        ),
        GraphOptimizationLevel=types.SimpleNamespace(
            ORT_ENABLE_ALL="all"
        ),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(model_manager, "CRNN_SHA256", digest)
    monkeypatch.setattr(model_manager, "CRNN_SIZE", len(payload))
    monkeypatch.setattr(
        model_manager,
        "crnn_path",
        lambda: model_path,
    )
    # GitHub's Windows runner can expose only two logical CPUs, reducing
    # concurrent inference to one.  Session identity must still be per camera.
    monkeypatch.setattr(
        onnx_crnn,
        "parallel_camera_limit",
        lambda: 1,
    )
    monkeypatch.setenv("BCVISION_CPU_THREADS", "9")
    onnx_crnn.clear_crnn_sessions()

    first = onnx_crnn.read_plate_crnn(
        np.full((40, 160, 3), 150, dtype=np.uint8),
        engine_key=1,
    )
    second = onnx_crnn.read_plate_crnn(
        np.full((40, 160, 3), 150, dtype=np.uint8),
        engine_key=2,
    )
    repeated = onnx_crnn.read_plate_crnn(
        np.full((40, 160, 3), 150, dtype=np.uint8),
        engine_key=1,
    )

    assert first[0] == "31-ط-556-74"
    assert second[0] == "31-ط-556-74"
    assert repeated[0] == "31-ط-556-74"
    assert len(created) == 2
    assert all(
        session.options.intra_op_num_threads == 2
        for session in created
    )
    assert all(
        session.options.inter_op_num_threads == 1
        for session in created
    )
    assert all(
        session.options.entries[
            "session.intra_op.allow_spinning"
        ] == "0"
        for session in created
    )
    assert onnx_crnn.get_crnn_status()["threads"] == 2
    onnx_crnn.clear_crnn_sessions()


def test_missing_model_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(
        model_manager,
        "crnn_path",
        lambda: tmp_path / "missing.onnx",
    )
    onnx_crnn.clear_crnn_sessions()

    assert onnx_crnn.read_plate_crnn(
        np.zeros((32, 128, 3), dtype=np.uint8)
    ) == ("", 0.0)
    assert onnx_crnn.get_crnn_status()["model_loaded"] is False
