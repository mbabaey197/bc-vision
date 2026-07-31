import base64
import json
import subprocess
from datetime import date

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app import license


def _configure_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(license, "LICENSE_PATH", tmp_path / "license.json")
    monkeypatch.setattr(license, "PUBLIC_KEY_PATH", tmp_path / "public.pem")
    monkeypatch.setattr(license, "TRIAL_PATH", tmp_path / ".trial.json")
    monkeypatch.setattr(
        license,
        "_STATE_PATH",
        tmp_path / ".license-state.json",
    )
    monkeypatch.setattr(
        license,
        "_STATE_KEY_PATH",
        tmp_path / ".license-state.key",
    )


def _signed_license(tmp_path, payload):
    private_key = Ed25519PrivateKey.generate()
    (tmp_path / "public.pem").write_bytes(
        private_key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return json.dumps(
        {
            "payload": payload,
            "signature": base64.b64encode(
                private_key.sign(license._canonical(payload))
            ).decode("ascii"),
        },
        ensure_ascii=False,
    )


def test_windows_machine_id_uses_hidden_child_process(monkeypatch):
    calls = []
    monkeypatch.setattr(license.platform, "system", lambda: "Windows")
    monkeypatch.setattr(license.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(license.platform, "node", lambda: "test-host")
    monkeypatch.setattr(license.uuid, "getnode", lambda: 123)
    monkeypatch.setattr(
        subprocess,
        "CREATE_NO_WINDOW",
        0x08000000,
        raising=False,
    )

    def check_output(command, **kwargs):
        calls.append((command, kwargs))
        return "UUID\nABC-123\n"

    monkeypatch.setattr(license.subprocess, "check_output", check_output)
    result = license.machine_id()
    assert len(result) == 32
    assert calls
    assert calls[0][1]["creationflags"] == 0x08000000


def test_machine_id_keeps_legacy_candidate(monkeypatch):
    monkeypatch.setattr(license.platform, "system", lambda: "Windows")
    monkeypatch.setattr(license.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(license.platform, "node", lambda: "old-host")
    monkeypatch.setattr(license.uuid, "getnode", lambda: 456)
    monkeypatch.setattr(
        license,
        "_windows_hardware_uuid",
        lambda: "ABC-123",
    )
    candidates = license.machine_id_candidates()
    expected_legacy = license._digest(
        ["Windows", "AMD64", "old-host", "456", "ABC-123"]
    )
    assert candidates[0] == license._digest(
        ["Windows", "AMD64", "ABC-123"]
    )
    assert expected_legacy in candidates


def test_signed_license_is_accepted_and_tamper_is_rejected(
    monkeypatch,
    tmp_path,
):
    _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(license, "machine_id", lambda: "MACHINE-A")
    monkeypatch.setattr(
        license,
        "machine_id_candidates",
        lambda: ["MACHINE-A"],
    )
    payload = {
        "license_id": "LIC-1",
        "customer": "Test",
        "machine_id": "MACHINE-A",
        "plan": "professional",
        "camera_limit": 8,
        "issued_at": date.today().isoformat(),
        "expires_at": "perpetual",
    }
    raw = _signed_license(tmp_path, payload)
    ok, _ = license.install_license(raw)
    assert ok is True
    assert license.status()["camera_limit"] == 8

    document = json.loads(raw)
    document["payload"]["camera_limit"] = 64
    license.LICENSE_PATH.write_text(json.dumps(document), encoding="utf-8")
    result = license.status()
    assert result["valid"] is False
    assert result["mode"] == "invalid"


def test_trial_state_tampering_fails_closed(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(license, "machine_id", lambda: "MACHINE-A")
    monkeypatch.setattr(
        license,
        "machine_id_candidates",
        lambda: ["MACHINE-A"],
    )
    first = license.status()
    assert first["valid"] is True
    document = json.loads(license.TRIAL_PATH.read_text(encoding="utf-8"))
    document["data"]["started"] = "2099-01-01"
    license.TRIAL_PATH.write_text(json.dumps(document), encoding="utf-8")
    second = license.status()
    assert second["valid"] is False
    assert "دست‌کاری" in second["message"]


def test_camera_capacity_is_backend_enforced(monkeypatch):
    monkeypatch.setattr(
        license,
        "status",
        lambda: {
            "valid": True,
            "plan": "basic",
            "camera_limit": 2,
        },
    )
    assert license.camera_capacity(1) == (True, "")
    allowed, message = license.camera_capacity(2)
    assert allowed is False
    assert "2" in message


def test_online_activation_requires_https():
    ok, message = license.activate_online(
        "http://license.example.test",
        "ABC",
    )
    assert ok is False
    assert "HTTPS" in message


def test_invalid_camera_limit_fails_closed(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(license, "machine_id", lambda: "MACHINE-A")
    monkeypatch.setattr(
        license,
        "machine_id_candidates",
        lambda: ["MACHINE-A"],
    )
    payload = {
        "license_id": "LIC-2",
        "customer": "Test",
        "machine_id": "MACHINE-A",
        "plan": "basic",
        "camera_limit": -1,
        "issued_at": date.today().isoformat(),
        "expires_at": "perpetual",
    }
    license.LICENSE_PATH.write_text(
        _signed_license(tmp_path, payload),
        encoding="utf-8",
    )
    result = license.status()
    assert result["valid"] is False
    assert "دوربین" in result["message"]
