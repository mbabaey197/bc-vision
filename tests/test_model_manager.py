import hashlib
from pathlib import Path
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


def test_yolov8n_reuses_verified_legacy_model_without_download(
    tmp_path,
    monkeypatch,
):
    payload = b"legacy-platrix-yolov8n"
    digest = hashlib.sha256(payload).hexdigest()
    data = tmp_path / "data"
    legacy = data / "models" / "plate" / "plate_yolo.onnx"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(payload)
    target = data / "models" / "plate" / "plate_yolov8n.onnx"

    monkeypatch.setattr(model_manager, "_data_dir", lambda: data)
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SHA256",
        digest,
    )
    monkeypatch.setattr(
        model_manager,
        "YOLOV8N_DETECTOR_SIZE",
        len(payload),
    )
    monkeypatch.setattr(model_manager, "packaged_seed_dir", lambda: None)
    monkeypatch.delenv("BCVISION_MODEL_SOURCE_DIR", raising=False)
    monkeypatch.delenv("BCVISION_PLATE_YOLOV8N_MODEL", raising=False)
    monkeypatch.delenv("BCVISION_PLATE_YOLO8N_MODEL", raising=False)

    result = model_manager.ensure_yolov8n_detector_model(
        download=False,
    )

    assert result == target
    assert target.read_bytes() == payload
    assert legacy.read_bytes() == payload


def test_model_status_readiness_follows_selected_detector(
    tmp_path,
    monkeypatch,
):
    yolo11_payload = b"ready-yolo11n"
    yolo8_payload = b"ready-yolov8n"
    yolo11 = tmp_path / "plate_yolo11n.onnx"
    yolo8 = tmp_path / "plate_yolov8n.onnx"
    yolo11.write_bytes(yolo11_payload)
    yolo8.write_bytes(yolo8_payload)
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
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SIZE",
        len(yolo11_payload),
    )
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

    selected = model_manager.model_status("yolo8n")
    yolo8.unlink()
    missing = model_manager.model_status("yolov8n")
    default = model_manager.model_status()

    assert selected["selected_detector"] == "yolov8n"
    assert selected["detector_path"] == str(yolo8)
    assert selected["detector_ready"] is True
    assert missing["detector_ready"] is False
    assert default["selected_detector"] == "yolo11n"
    assert default["detector_ready"] is True


def test_model_status_exposes_launcher_preparation_failure(monkeypatch):
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_STATE_ENV,
        "error",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_ERROR_ENV,
        "ValueError: YOLOv8n SHA-256 mismatch",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_ATTEMPT_ENV,
        "2",
    )

    status = model_manager.model_status("yolov8n")

    assert status["preparation_state"] == "error"
    assert (
        status["preparation_error"]
        == "ValueError: YOLOv8n SHA-256 mismatch"
    )
    assert status["preparation_attempt"] == 2


def test_offline_seed_contains_both_selectable_detectors(
    tmp_path,
    monkeypatch,
):
    sources = {}

    def source(name, payload):
        path = tmp_path / "sources" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        sources[name] = path
        return path

    yolo11 = source("plate_yolo11n.onnx", b"seed-yolo11")
    yolo8 = source("plate_yolov8n.onnx", b"seed-yolo8")
    fallback = source("plate_yolo_fallback.onnx", b"seed-fallback")
    crnn = source("ocr_crnn.onnx", b"seed-crnn")
    cnn = source("ocr_cnn.onnx", b"seed-cnn")
    hezar = source("crnn_fa_v2.onnx", b"seed-hezar")
    contracts = (
        ("DETECTOR", yolo11),
        ("YOLOV8N_DETECTOR", yolo8),
        ("DETECTOR_FALLBACK", fallback),
        ("CRNN", crnn),
        ("CNN", cnn),
        ("HEZAR_ONNX", hezar),
    )
    for prefix, path in contracts:
        monkeypatch.setattr(
            model_manager,
            prefix + "_SHA256",
            sha256_file(path),
        )
        monkeypatch.setattr(
            model_manager,
            prefix + "_SIZE",
            path.stat().st_size,
        )
    monkeypatch.setattr(model_manager, "ensure_detector_model", lambda **_: yolo11)
    monkeypatch.setattr(
        model_manager,
        "ensure_yolov8n_detector_model",
        lambda **_: yolo8,
    )
    monkeypatch.setattr(
        model_manager,
        "ensure_detector_fallback_model",
        lambda **_: fallback,
    )
    monkeypatch.setattr(model_manager, "ensure_crnn_model", lambda **_: crnn)
    monkeypatch.setattr(model_manager, "ensure_cnn_model", lambda **_: cnn)
    monkeypatch.setattr(model_manager, "ensure_hezar_model", lambda **_: hezar)

    result = model_manager.prepare_seed(tmp_path / "seed", download=False)

    assert Path(result["detector_yolo11n"]).read_bytes() == yolo11.read_bytes()
    assert Path(result["detector_yolov8n"]).read_bytes() == yolo8.read_bytes()


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
