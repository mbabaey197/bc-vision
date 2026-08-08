import hashlib
import ssl
import urllib.error

import pytest

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


class _DownloadResponse:
    def __init__(self, payload):
        self._chunks = [payload, b""]

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self, _size):
        return self._chunks.pop(0)


def test_verified_download_retries_transient_ssl_failure(
    tmp_path,
    monkeypatch,
):
    payload = b"verified-model"
    target = tmp_path / "model.onnx"
    calls = 0

    def download(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 3
        if calls == 1:
            raise urllib.error.URLError(
                ssl.SSLEOFError(8, "unexpected EOF")
            )
        return _DownloadResponse(payload)

    monkeypatch.setattr(model_manager.urllib.request, "urlopen", download)

    result = model_manager._download_verified(
        "https://example.invalid/model.onnx",
        target,
        hashlib.sha256(payload).hexdigest(),
        len(payload),
        timeout=3,
        attempts=2,
        retry_delay=0,
    )

    assert result == target
    assert target.read_bytes() == payload
    assert calls == 2
    assert list(tmp_path.glob("*.part")) == []


def test_verified_download_exhausts_network_retries_cleanly(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "model.onnx"
    calls = 0

    def unavailable(_request, timeout):
        nonlocal calls
        calls += 1
        assert timeout == 3
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(
        model_manager.urllib.request,
        "urlopen",
        unavailable,
    )

    with pytest.raises(urllib.error.URLError):
        model_manager._download_verified(
            "https://example.invalid/model.onnx",
            target,
            "0" * 64,
            10,
            timeout=3,
            attempts=3,
            retry_delay=0,
        )

    assert calls == 3
    assert not target.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_detector_is_bootstrapped_from_packaged_seed(tmp_path, monkeypatch):
    payload = b"detector-seed"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    seed = tmp_path / "seed"
    (seed / "plate").mkdir(parents=True)
    (seed / "plate" / "plate_yolo11n.onnx").write_bytes(payload)
    target = tmp_path / "data" / "plate_yolo11n.onnx"

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


def test_promoted_crnn_keeps_verified_training_checkpoint(
    tmp_path,
    monkeypatch,
):
    data = tmp_path / "data"
    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    candidate = tmp_path / "candidate.onnx"
    checkpoint = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate-onnx")
    checkpoint.write_bytes(b"weights-only-state-dict")
    candidate_digest = sha256_file(candidate)
    checkpoint_digest = sha256_file(checkpoint)

    promoted = model_manager.promote_crnn_candidate(
        candidate,
        candidate_digest,
        source_run_id=7,
        training_checkpoint=checkpoint,
        training_checkpoint_sha256=checkpoint_digest,
    )
    active_checkpoint = (
        model_manager.active_crnn_training_checkpoint()
    )

    assert promoted["sha256"] == candidate_digest
    assert active_checkpoint is not None
    assert active_checkpoint[1] == checkpoint_digest
    assert active_checkpoint[0].read_bytes() == (
        b"weights-only-state-dict"
    )
