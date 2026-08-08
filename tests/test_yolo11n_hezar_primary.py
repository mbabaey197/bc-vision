import hashlib
import sys
import types

import numpy as np

from app.ai import model_manager, ocr, onnx_detector, onnx_hezar
from app.ai.pipeline import process_frame


def test_yolo11n_primary_uses_640_contract(tmp_path, monkeypatch):
    payload = b"yolo11n"
    path = tmp_path / "plate_yolo11n.onnx"
    path.write_bytes(payload)

    class Options:
        def add_session_config_entry(self, *_args):
            pass

    class Session:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="images")]

        def run(self, _outputs, inputs):
            assert inputs["images"].shape == (1, 3, 640, 640)
            # One YOLO11n single-class detection in channels-first form.
            return [np.array(
                [[[320.0], [180.0], [180.0], [54.0], [0.93]]],
                dtype=np.float32,
            )]

    fake_ort = types.SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=Session,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(model_manager, "DETECTOR_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "detector_path", lambda: path)
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: tmp_path / "missing-fallback.onnx",
    )
    onnx_detector.clear_detector_sessions()

    rows = onnx_detector.detect_plates_onnx(
        np.full((360, 640, 3), 127, dtype=np.uint8),
        engine_key="yolo11-test",
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "yolo11n-plate-onnx"
    assert onnx_detector.detector_status()["engine"] == (
        "yolo11n-plate-onnx"
    )
    onnx_detector.clear_detector_sessions()


def test_verified_hezar_v2_is_active_primary(tmp_path, monkeypatch):
    payload = b"hezar-v2"
    path = tmp_path / "crnn_fa_v2.onnx"
    path.write_bytes(payload)
    target = "۳۱ط۵۵۶۷۴"
    decoded = np.full(
        (len(target) * 2, len(onnx_hezar.HEZAR_V2_LABELS)),
        -12.0,
        dtype=np.float32,
    )
    for position, character in enumerate(target):
        decoded[position * 2, onnx_hezar.HEZAR_V2_LABELS.index(
            character
        )] = 12.0
        decoded[position * 2 + 1, 0] = 12.0

    class Options:
        pass

    class Session:
        def __init__(self, *_args, **_kwargs):
            pass

        def get_inputs(self):
            return [types.SimpleNamespace(name="pixel_values")]

        def run(self, _outputs, inputs):
            assert inputs["pixel_values"].shape == (1, 1, 32, 384)
            # The fixed Hezar contract reverses the time axis after inference.
            return [decoded[::-1][None]]

    fake_ort = types.SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=Session,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(
        model_manager,
        "HEZAR_ONNX_SHA256",
        hashlib.sha256(payload).hexdigest(),
    )
    monkeypatch.setattr(model_manager, "HEZAR_ONNX_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "hezar_path", lambda: path)
    onnx_hezar.clear_hezar_sessions()

    result = onnx_hezar.read_plate_hezar_primary(
        np.full((32, 160, 3), 180, dtype=np.uint8),
        engine_key="hezar-test",
    )

    assert result["accepted"] is True
    assert result["plate_norm"] == "31ط55674"
    assert onnx_hezar.hezar_status()["model_loaded"] is True
    onnx_hezar.clear_hezar_sessions()


def test_ocr_prefers_hezar_before_legacy_readers(monkeypatch):
    monkeypatch.delenv("BCVISION_OCR_ENGINE", raising=False)
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "plate_norm": "31ط55674",
            "confidence": 0.91,
            "hypotheses": [{"plate_norm": "31ط55674"}],
        },
    )
    monkeypatch.setattr(
        ocr,
        "read_plate_crnn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("legacy CRNN must not run after accepted Hezar")
        ),
    )

    result = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8)
    )

    assert result == (
        "31-ط-556-74",
        0.91,
        "hezar-crnn-fa-v2-onnx",
    )


def test_hezar_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    payload = b"hezar-seed"
    digest = hashlib.sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    source = seed / "hezar" / "crnn_fa_v2.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    target = tmp_path / "data" / "hezar" / "crnn_fa_v2.onnx"

    monkeypatch.setattr(model_manager, "HEZAR_ONNX_SHA256", digest)
    monkeypatch.setattr(model_manager, "HEZAR_ONNX_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "hezar_path", lambda: target)
    monkeypatch.delenv("BCVISION_HEZAR_SOURCE_DIR", raising=False)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert model_manager.ensure_hezar_model(download=False) == target
    assert target.read_bytes() == payload


def test_stronger_hezar_disagreement_keeps_both_reads(monkeypatch):
    crop = np.full((42, 168, 3), 170, dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 178, 62),
            "confidence": 0.86,
            "method": "yolo11n-plate-onnx",
            "direct_text": "31-ط-556-74",
            "direct_ocr_confidence": 0.64,
            "direct_ocr_attempted": True,
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            "31-ط-558-74",
            0.88,
            "hezar-crnn-fa-v2-onnx",
        ),
    )

    row = process_frame(
        np.full((100, 220, 3), 100, dtype=np.uint8),
        engine_key=9,
    )[0]

    assert row["plate"] == "31-ط-558-74"
    assert row["ocr_engine"] == "hezar-crnn-fa-v2-onnx"
    assert row["ocr_alternative"] == "31-ط-556-74"
    assert row["ocr_disagreement"] is True
    assert row["needs_review"] is True
