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
    (seed / "plate" / "best.pt").write_bytes(payload)
    target = tmp_path / "data" / "best.pt"

    monkeypatch.setattr(model_manager, "DETECTOR_SHA256", digest)
    monkeypatch.setattr(model_manager, "DETECTOR_SIZE", len(payload))
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "detector_path", lambda: target)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)

    assert model_manager.ensure_detector_model(download=False) == target
    assert target.read_bytes() == payload


def test_easyocr_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    seed = tmp_path / "seed"
    hashes = {}
    for name, payload in {
        "arabic.pth": b"arabic-seed",
        "craft_mlt_25k.pth": b"craft-seed",
    }.items():
        path = seed / "easyocr" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        hashes[name] = __import__("hashlib").sha256(payload).hexdigest()
    target = tmp_path / "data" / "easyocr"

    monkeypatch.setattr(model_manager, "EASYOCR_HASHES", hashes)
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: seed)
    monkeypatch.setattr(model_manager, "easyocr_dir", lambda: target)
    monkeypatch.delenv("BCVISION_EASYOCR_SOURCE_DIR", raising=False)

    assert model_manager.ensure_easyocr_models(download=False) == target
    assert all((target / name).is_file() for name in hashes)


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
