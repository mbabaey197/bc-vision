import hashlib
import json
import sys
import types

import numpy as np
import pytest
import cv2

from app.ai import model_manager, onnx_yolox


def _spec(**overrides):
    output_format = str(overrides.get("output_format", "raw-grid"))
    spec = {
        "variant": "yolox",
        "input_height": 32,
        "input_width": 32,
        "output_format": output_format,
        "coordinate_space": (
            "grid" if output_format == "raw-grid" else "input-pixels"
        ),
        "class_count": 1,
        "plate_class_id": 0,
        "strides": [8, 16, 32],
        "scores_are_logits": False,
    }
    spec.update(overrides)
    return spec


def test_raw_grid_decode_uses_objectness_class_and_stride():
    # 32x32 with strides 8/16/32 has 16 + 4 + 1 rows.
    output = np.zeros((1, 21, 6), dtype=np.float32)
    output[0, 20] = [0.5, 0.5, np.log(2.0), np.log(1.0), 0.9, 0.8]

    boxes, scores = onnx_yolox.decode_output(output, _spec())

    assert np.allclose(boxes[20], [-16.0, 0.0, 48.0, 32.0])
    assert float(scores[20]) == pytest.approx(0.72)


def test_standard_preprocess_matches_megvii_top_left_reference():
    frame = np.arange(333 * 777 * 3, dtype=np.uint8).reshape(333, 777, 3)
    spec = _spec(
        input_height=640,
        input_width=640,
        color="bgr",
        input_scale=1.0,
        letterbox_mode="top-left",
    )

    tensor, ratio, padding = onnx_yolox._letterbox(frame, spec)
    resized = cv2.resize(
        frame,
        (int(777 * ratio), int(333 * ratio)),
        interpolation=cv2.INTER_LINEAR,
    )
    expected = np.full((640, 640, 3), 114, dtype=np.uint8)
    expected[:resized.shape[0], :resized.shape[1]] = resized
    expected = np.ascontiguousarray(
        expected.transpose(2, 0, 1),
        dtype=np.float32,
    )[None]

    assert padding == (0.0, 0.0)
    assert tensor.flags.c_contiguous
    assert np.array_equal(tensor, expected)


def test_decoded_contract_does_not_guess_transposed_output():
    output = np.zeros((1, 6, 20), dtype=np.float32)

    with pytest.raises(ValueError, match="output width"):
        onnx_yolox.decode_output(
            output,
            _spec(output_format="decoded-cxcywh"),
        )


def test_contract_rejects_undeclared_extra_output_dimensions():
    output = np.zeros((1, 1, 21, 6), dtype=np.float32)

    with pytest.raises(ValueError, match="reduce to rows"):
        onnx_yolox.decode_output(output, _spec())


def test_decoded_multiclass_contract_requires_plate_class_to_win():
    output = np.array([[
        [10, 10, 20, 8, 0.90, 0.70, 0.99],
        [20, 20, 20, 8, 0.90, 0.80, 0.40],
    ]], dtype=np.float32)

    _boxes, scores = onnx_yolox.decode_output(
        output,
        _spec(
            output_format="decoded-cxcywh",
            class_count=2,
            plate_class_id=0,
        ),
    )

    assert scores.tolist() == pytest.approx([0.0, 0.72])


def test_end_to_end_contract_filters_non_plate_class():
    output = np.array([[
        [1, 2, 20, 10, 0.91, 0],
        [2, 3, 21, 11, 0.99, 1],
    ]], dtype=np.float32)

    boxes, scores = onnx_yolox.decode_output(
        output,
        _spec(
            output_format="nms-xyxy",
            class_count=2,
            plate_class_id=0,
        ),
    )

    assert boxes.tolist() == [[1.0, 2.0, 20.0, 10.0]]
    assert scores.tolist() == pytest.approx([0.91])


def test_multiclass_end_to_end_contract_requires_class_column():
    output = np.array([[[1, 2, 20, 10, 0.91]]], dtype=np.float32)

    with pytest.raises(ValueError, match="must include class_id"):
        onnx_yolox.decode_output(
            output,
            _spec(
                output_format="nms-xyxy",
                class_count=2,
                plate_class_id=0,
            ),
        )


def test_end_to_end_contract_rejects_fractional_class_id():
    output = np.array([[[1, 2, 20, 10, 0.91, 0.5]]], dtype=np.float32)

    with pytest.raises(ValueError, match="class ids are invalid"):
        onnx_yolox.decode_output(
            output,
            _spec(
                output_format="nms-xyxy",
                class_count=2,
                plate_class_id=0,
            ),
        )


def test_geometry_filter_drops_mostly_off_frame_box_before_result_limit():
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    boxes = np.array([
        [-1_000_000, 40, 140, 80],
        [50, 60, 180, 90],
    ], dtype=np.float32)
    scores = np.array([0.99, 0.80], dtype=np.float32)

    rows = onnx_yolox._detections(
        frame,
        boxes,
        scores,
        ratio=1.0,
        padding=(0.0, 0.0),
        confidence=0.25,
        max_results=1,
        spec={"method": "yolox-test", "model_revision": "test"},
    )

    assert len(rows) == 1
    assert rows[0]["confidence"] == pytest.approx(0.80)
    assert rows[0]["bbox"][0] > 0


def test_install_is_versioned_and_invalid_contract_preserves_active_model(
    tmp_path,
    monkeypatch,
):
    data = tmp_path / "data"
    first = tmp_path / "first.onnx"
    first.write_bytes(b"first-yolox-model")
    second = tmp_path / "second.onnx"
    second.write_bytes(b"second-yolox-model")
    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    monkeypatch.setattr(
        onnx_yolox,
        "validate_yolox_model",
        lambda _spec: {"outputs": 1},
    )
    for variable in ("BCVISION_YOLOX_MODEL", "BCVISION_YOLOX_MANIFEST"):
        monkeypatch.delenv(variable, raising=False)

    installed = model_manager.install_yolox_model(
        first,
        input_size=640,
        output_format="raw-grid",
        color="bgr",
        input_scale=1.0,
        letterbox_mode="top-left",
    )
    active_manifest = model_manager.yolox_manifest_path().read_bytes()

    assert installed["ready"] is True
    assert installed["path"].name.startswith("yolox-")
    assert installed["sha256"] == hashlib.sha256(first.read_bytes()).hexdigest().upper()
    with pytest.raises(ValueError, match="output_format"):
        model_manager.install_yolox_model(
            second,
            input_size=640,
            output_format="guessed-layout",
        )
    assert model_manager.yolox_manifest_path().read_bytes() == active_manifest
    assert model_manager.yolox_detector_spec()["sha256"] == installed["sha256"]
    assert model_manager.yolox_detector_spec()["ready"] is True


def test_manifest_hash_mismatch_fails_closed(tmp_path, monkeypatch):
    data = tmp_path / "data"
    root = data / "models" / "plate"
    root.mkdir(parents=True)
    model = root / "custom.onnx"
    model.write_bytes(b"corrupt")
    manifest = root / "yolox-custom.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "filename": model.name,
        "sha256": "A" * 64,
        "size": model.stat().st_size,
        "input_size": 640,
        "output_format": "raw-grid",
        "coordinate_space": "grid",
        "class_count": 1,
        "plate_class_id": 0,
        "strides": [8, 16, 32],
        "color": "bgr",
        "input_scale": 1.0,
        "letterbox_mode": "top-left",
    }), encoding="utf-8")
    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    monkeypatch.delenv("BCVISION_YOLOX_MODEL", raising=False)
    monkeypatch.delenv("BCVISION_YOLOX_MANIFEST", raising=False)

    spec = model_manager.yolox_detector_spec()

    assert spec["ready"] is False
    assert "does not match" in spec["error"]


def test_runtime_contract_retries_concurrent_manifest_activation(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "yolox.json"
    manifest.write_text("A", encoding="utf-8")
    model = tmp_path / "yolox.onnx"
    model.write_bytes(b"model")
    calls = []

    def spec():
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            manifest.write_text("B", encoding="utf-8")
            return {"path": model, "model_revision": "A"}
        return {"path": model, "model_revision": "B"}

    monkeypatch.setattr(model_manager, "yolox_manifest_path", lambda: manifest)
    monkeypatch.setattr(model_manager, "yolox_detector_spec", spec)
    monkeypatch.delenv("BCVISION_YOLOX_MODEL", raising=False)
    onnx_yolox.clear_yolox_session()

    selected = onnx_yolox._runtime_spec()
    cached = onnx_yolox._runtime_spec()

    assert selected["model_revision"] == "B"
    assert cached["model_revision"] == "B"
    assert calls == [1, 2]
    onnx_yolox.clear_yolox_session()


def test_cli_forwards_output_index_and_score_logit_contract(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "custom.onnx"
    captured = {}

    def fake_install(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return {"ready": True}

    monkeypatch.setattr(model_manager, "install_yolox_model", fake_install)
    monkeypatch.setattr(
        model_manager,
        "model_status",
        lambda **_kwargs: {"detector_yolox_ready": True},
    )

    result = model_manager.main([
        "--install-yolox",
        str(source),
        "--yolox-output-index",
        "2",
        "--yolox-scores-are-logits",
    ])

    assert result == 0
    assert captured["path"] == source
    assert captured["kwargs"]["output_index"] == 2
    assert captured["kwargs"]["scores_are_logits"] is True


def test_manifest_replace_failure_preserves_active_install(tmp_path, monkeypatch):
    data = tmp_path / "data"
    first = tmp_path / "first.onnx"
    first.write_bytes(b"first-active-yolox-model")
    second = tmp_path / "second.onnx"
    second.write_bytes(b"second-candidate-yolox-model")
    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    monkeypatch.setattr(
        onnx_yolox,
        "validate_yolox_model",
        lambda _spec: {"outputs": 1},
    )
    for variable in ("BCVISION_YOLOX_MODEL", "BCVISION_YOLOX_MANIFEST"):
        monkeypatch.delenv(variable, raising=False)

    active = model_manager.install_yolox_model(first)
    manifest = model_manager.yolox_manifest_path()
    active_manifest = manifest.read_bytes()
    second_digest = hashlib.sha256(second.read_bytes()).hexdigest().lower()
    second_target = manifest.parent / f"yolox-{second_digest[:12]}.onnx"
    real_replace = model_manager.os.replace

    def fail_second_manifest_replace(source, destination):
        if destination == manifest:
            raise OSError("simulated manifest activation failure")
        return real_replace(source, destination)

    monkeypatch.setattr(
        model_manager.os,
        "replace",
        fail_second_manifest_replace,
    )

    with pytest.raises(OSError, match="manifest activation failure"):
        model_manager.install_yolox_model(second)

    assert manifest.read_bytes() == active_manifest
    preserved = model_manager.yolox_detector_spec()
    assert preserved["ready"] is True
    assert preserved["sha256"] == active["sha256"]
    assert not second_target.exists()
    assert list(manifest.parent.glob("*.part")) == []


def test_configured_fixed_target_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "active.onnx"
    target.write_bytes(b"active-model-bytes")
    candidate = tmp_path / "candidate.onnx"
    candidate.write_bytes(b"candidate-model-bytes")
    manifest = tmp_path / "active.json"
    monkeypatch.setenv("BCVISION_YOLOX_MODEL", str(target))
    monkeypatch.setenv("BCVISION_YOLOX_MANIFEST", str(manifest))

    with pytest.raises(ValueError, match="read-only"):
        model_manager.install_yolox_model(candidate)

    assert target.read_bytes() == b"active-model-bytes"
    assert not manifest.exists()


def test_failed_graph_preflight_never_activates_manifest(tmp_path, monkeypatch):
    data = tmp_path / "data"
    source = tmp_path / "invalid.onnx"
    source.write_bytes(b"not-an-onnx-graph")
    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    monkeypatch.delenv("BCVISION_YOLOX_MODEL", raising=False)
    monkeypatch.delenv("BCVISION_YOLOX_MANIFEST", raising=False)
    monkeypatch.setattr(
        onnx_yolox,
        "validate_yolox_model",
        lambda _spec: (_ for _ in ()).throw(ValueError("invalid graph")),
    )

    with pytest.raises(ValueError, match="invalid graph"):
        model_manager.install_yolox_model(source)

    root = data / "models" / "plate"
    assert not model_manager.yolox_manifest_path().exists()
    assert list(root.glob("*.part")) == []
    assert list(root.glob("yolox-*.onnx")) == []


def test_runtime_uses_one_shared_session_for_multiple_camera_keys(
    tmp_path,
    monkeypatch,
):
    payload = b"verified-yolox"
    path = tmp_path / "yolox.onnx"
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest().upper()
    created = []
    output = np.array([[[320, 320, 180, 50, 0.9, 0.9]]], dtype=np.float32)

    class Options:
        def __init__(self):
            self.intra_op_num_threads = 0
            self.inter_op_num_threads = 0

        def add_session_config_entry(self, *_args):
            return None

    class Session:
        def __init__(self, model_path, sess_options=None, providers=None):
            assert model_path == str(path)
            created.append(self)

        def get_inputs(self):
            return [types.SimpleNamespace(
                name="images",
                shape=[1, 3, 640, 640],
                type="tensor(float)",
            )]

        def get_outputs(self):
            return [types.SimpleNamespace(name="predictions")]

        def run(self, names, values):
            assert names == ["predictions"]
            assert values["images"].shape == (1, 3, 640, 640)
            return [output]

    fake_ort = types.SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=Session,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    spec = {
        "variant": "yolox",
        "path": path,
        "manifest_path": tmp_path / "yolox.json",
        "sha256": digest,
        "size": len(payload),
        "input_size": 640,
        "input_height": 640,
        "input_width": 640,
        "output_format": "decoded-cxcywh",
        "coordinate_space": "input-pixels",
        "output_index": 0,
        "class_count": 1,
        "plate_class_id": 0,
        "strides": [8, 16, 32],
        "color": "bgr",
        "input_scale": 1.0,
        "letterbox_mode": "top-left",
        "scores_are_logits": False,
        "method": "yolox-custom-onnx",
        "model_revision": digest[:12].lower(),
        "ready": True,
        "error": "",
    }
    monkeypatch.setattr(model_manager, "yolox_detector_spec", lambda: spec)
    onnx_yolox.clear_yolox_session()
    frame = np.zeros((640, 640, 3), dtype=np.uint8)

    first = onnx_yolox.detect_plates_yolox(frame, engine_key="camera-1")
    second = onnx_yolox.detect_plates_yolox(frame, engine_key="camera-2")

    assert len(first) == len(second) == 1
    assert len(created) == 1
    assert onnx_yolox.yolox_status()["engine_key"] == "camera-2"
    onnx_yolox.clear_yolox_session()


def test_pinned_inference_rejects_changed_revision_before_session(
    monkeypatch,
):
    monkeypatch.setattr(
        onnx_yolox,
        "_runtime_spec",
        lambda: {
            "model_revision": "revision-b",
            "manifest_path": "manifest-b.json",
        },
    )
    monkeypatch.setattr(
        onnx_yolox,
        "_load_session",
        lambda _spec: (_ for _ in ()).throw(
            AssertionError("changed revision must not load or run")
        ),
    )
    metadata = {}

    with pytest.raises(RuntimeError, match="revision changed"):
        onnx_yolox.detect_plates_yolox(
            np.zeros((32, 64, 3), dtype=np.uint8),
            expected_model_revision="revision-a",
            runtime_metadata=metadata,
            raise_on_error=True,
        )

    assert metadata["detector_model_revision"] == "revision-b"
