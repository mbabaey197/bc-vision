import base64
import hashlib
import json

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)

from app.ai import next_models


def _signed_bundle(tmp_path, monkeypatch):
    data = tmp_path / "data"
    root = data / "models" / "next"
    root.mkdir(parents=True)
    detector = root / "detector.onnx"
    ocr = root / "ocr.onnx"
    detector.write_bytes(b"rc13-obb")
    ocr.write_bytes(b"rc13-hezar")
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (root / "model_public_key.pem").write_bytes(public_key)
    payload = {
        "schema": 1,
        "engine": "bcvision-rc13",
        "release_id": "fixture-1",
        "models": {
            "detector": {
                "filename": "detector.onnx",
                "sha256": hashlib.sha256(
                    detector.read_bytes()
                ).hexdigest(),
                "size": detector.stat().st_size,
            },
            "ocr": {
                "filename": "ocr.onnx",
                "sha256": hashlib.sha256(
                    ocr.read_bytes()
                ).hexdigest(),
                "size": ocr.stat().st_size,
            },
        },
    }
    payload["signature"] = base64.b64encode(
        private_key.sign(
            next_models.canonical_manifest_bytes(payload)
        )
    ).decode("ascii")
    (root / "active-models.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    monkeypatch.setattr(next_models, "_data_root", lambda: data)
    monkeypatch.delenv("BCVISION_NEXT_MANIFEST", raising=False)
    monkeypatch.delenv("BCVISION_ANPR_MODEL_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("BCVISION_ANPR_MODE", raising=False)
    return root


def test_signed_next_model_bundle_is_verified(tmp_path, monkeypatch):
    _signed_bundle(tmp_path, monkeypatch)

    status = next_models.next_models_status()

    assert status["ready"] is True
    assert status["release_id"] == "fixture-1"


def test_tampered_next_model_bundle_fails_closed(tmp_path, monkeypatch):
    root = _signed_bundle(tmp_path, monkeypatch)
    (root / "ocr.onnx").write_bytes(b"tampered")

    status = next_models.next_models_status()

    assert status["ready"] is False
    assert "verification failed" in status["error"]


def test_shadow_mode_can_be_rolled_back_atomically(tmp_path, monkeypatch):
    _signed_bundle(tmp_path, monkeypatch)
    next_models.set_engine_mode("shadow", reason="test")
    assert next_models.engine_mode() == "shadow"

    state = next_models.rollback_to_baseline("golden-regression")

    assert state["mode"] == "baseline"
    assert state["rollback_lock"] is True
    monkeypatch.setenv("BCVISION_ANPR_MODE", "next")
    assert next_models.engine_mode() == "baseline"
