from app.ai import model_manager
from app.ai.model_manager import sha256_file, verify_file


def test_hash_verification(tmp_path):
    path = tmp_path / "model.bin"
    payload = b"bc-vision-model"
    path.write_bytes(payload)
    digest = sha256_file(path)
    assert verify_file(path, digest, len(payload))
    assert not verify_file(path, "0" * 64)
    assert not verify_file(path, digest, 1)


def test_detector_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    payload = b"detector-seed"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    (seed / "plate").mkdir(parents=True)
    (seed / "plate" / "plate_yolo.onnx").write_bytes(payload)
    target = tmp_path / "data" / "plate_yolo.onnx"

    monkeypatch.setattr(model_manager, "DETECTOR_SHA256", digest)
    monkeypatch.setattr(model_manager, "DETECTOR_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "detector_path", lambda: target)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert model_manager.ensure_detector_model(download=False) == target
    assert target.read_bytes() == payload


def test_detector_fallback_is_bootstrapped_from_seed(tmp_path, monkeypatch):
    payload = b"detector-fallback-seed"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    source = seed / "plate" / "plate_yolo_fallback.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    target = tmp_path / "data" / "plate_yolo_fallback.onnx"

    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SHA256",
        digest,
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SIZE",
        len(payload),
    )
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: target,
    )
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert (
        model_manager.ensure_detector_fallback_model(download=False)
        == target
    )
    assert target.read_bytes() == payload


def test_crnn_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    payload = b"crnn-onnx-seed"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    source = seed / "crnn" / "ocr_crnn.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    target = tmp_path / "data" / "crnn" / "ocr_crnn.onnx"

    monkeypatch.setattr(model_manager, "CRNN_SHA256", digest)
    monkeypatch.setattr(model_manager, "CRNN_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "crnn_path", lambda: target)
    monkeypatch.delenv("BCVISION_CRNN_SOURCE_DIR", raising=False)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert model_manager.ensure_crnn_model(download=False) == target
    assert target.read_bytes() == payload


def test_cnn_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    payload = b"cnn-onnx-seed"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    source = seed / "cnn" / "ocr_cnn.onnx"
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    target = tmp_path / "data" / "cnn" / "ocr_cnn.onnx"

    monkeypatch.setattr(model_manager, "CNN_SHA256", digest)
    monkeypatch.setattr(model_manager, "CNN_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "cnn_path", lambda: target)
    monkeypatch.delenv("BCVISION_CNN_SOURCE_DIR", raising=False)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert model_manager.ensure_cnn_model(download=False) == target
    assert target.read_bytes() == payload
