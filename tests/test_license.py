import base64
import json
import subprocess
from datetime import date, timedelta

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from app import license
from app.license_format import encode_document


def _configure_paths(monkeypatch, tmp_path):
    original_write_state = license._write_state
    monkeypatch.setattr(license, "LICENSE_PATH", tmp_path / "license.dat")
    monkeypatch.setattr(
        license,
        "LEGACY_LICENSE_PATH",
        tmp_path / "license.json",
    )
    monkeypatch.setattr(license, "PUBLIC_KEY_PATH", tmp_path / "public.pem")
    monkeypatch.setattr(license, "TRIAL_PATH", tmp_path / ".trial.dat")
    monkeypatch.setattr(
        license,
        "_STATE_PATH",
        tmp_path / ".license-state.dat",
    )
    monkeypatch.setattr(
        license,
        "_STATE_KEY_PATH",
        tmp_path / ".license-state.key",
    )

    def write_state(data, path=None):
        return original_write_state(
            data,
            license._STATE_PATH if path is None else path,
        )

    monkeypatch.setattr(license, "_write_state", write_state)


def _signed_document(tmp_path, payload):
    private_key = Ed25519PrivateKey.generate()
    public_pem = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    (tmp_path / "public.pem").write_bytes(public_pem)
    return (
        {
            "payload": payload,
            "signature": base64.b64encode(
                private_key.sign(license._canonical(payload))
            ).decode("ascii"),
        },
        public_pem,
    )


def _payload(machine="MACHINE-A", camera_limit=8):
    return {
        "format_version": 2,
        "product": "bc-vision",
        "license_id": "LIC-1",
        "customer": "Test",
        "machine_id": machine,
        "machine_ids": [machine],
        "plan": "professional",
        "camera_limit": camera_limit,
        "features": ["anpr", "events", "reports"],
        "issued_at": date.today().isoformat(),
        "expires_at": "perpetual",
    }


def _configure_machine(monkeypatch, machine="MACHINE-A"):
    monkeypatch.setattr(license, "machine_id", lambda: machine)
    monkeypatch.setattr(
        license,
        "machine_id_candidates",
        lambda: [machine],
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
        return "ABC-123\n"

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
        "_hardware_claims",
        lambda: {
            "system": "WINDOWS",
            "machine": "AMD64",
            "uuid": "ABC-123",
        },
    )
    candidates = license.machine_id_candidates()
    expected_legacy = license._digest(
        ["Windows", "AMD64", "old-host", "456", "ABC-123"]
    )
    assert license._digest(["WINDOWS", "AMD64", "ABC-123"]) in candidates
    assert expected_legacy in candidates


def test_license_dat_is_accepted_and_tamper_is_rejected(
    monkeypatch,
    tmp_path,
):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    document, public_pem = _signed_document(tmp_path, _payload())
    token = encode_document(document, ["MACHINE-A"], public_pem)

    ok, message = license.install_license(token)
    assert ok is True, message
    assert license.LICENSE_PATH.name == "license.dat"
    assert license.status()["camera_limit"] == 8

    text = license.LICENSE_PATH.read_text(encoding="ascii")
    index = max(len("BCV2.") + 3, len(text) // 2)
    replacement = "A" if text[index] != "A" else "B"
    license.LICENSE_PATH.write_text(
        text[:index] + replacement + text[index + 1 :],
        encoding="ascii",
    )
    result = license.status()
    assert result["valid"] is False


def test_license_dat_cannot_be_copied_to_another_machine(
    monkeypatch,
    tmp_path,
):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch, "MACHINE-A")
    document, public_pem = _signed_document(tmp_path, _payload())
    token = encode_document(document, ["MACHINE-A"], public_pem)
    ok, _ = license.install_license(token)
    assert ok is True

    _configure_machine(monkeypatch, "MACHINE-B")
    result = license.status()
    assert result["valid"] is False
    assert "دستگاه" in result["message"]


def test_deleting_local_license_state_fails_closed(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    document, public_pem = _signed_document(tmp_path, _payload())
    ok, _ = license.install_license(
        encode_document(document, ["MACHINE-A"], public_pem)
    )
    assert ok is True
    license._STATE_PATH.unlink()
    result = license.status()
    assert result["valid"] is False
    assert "وضعیت" in result["message"]


def test_clock_rollback_fails_closed(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    document, public_pem = _signed_document(tmp_path, _payload())
    ok, _ = license.install_license(
        encode_document(document, ["MACHINE-A"], public_pem)
    )
    assert ok is True
    state = license._read_state(license._STATE_PATH)
    state["last_seen"] = (date.today() + timedelta(days=10)).isoformat()
    license._write_state(state)
    result = license.status()
    assert result["valid"] is False
    assert "عقب" in result["message"]


def test_trial_is_disabled_by_default(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    monkeypatch.delenv("BCVISION_ENABLE_TRIAL", raising=False)
    result = license.status()
    assert result["valid"] is False
    assert result["mode"] == "unlicensed"


def test_legacy_signed_json_is_migrated_to_license_dat(
    monkeypatch,
    tmp_path,
):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    document, _ = _signed_document(tmp_path, _payload())
    license.LEGACY_LICENSE_PATH.write_text(
        json.dumps(document),
        encoding="utf-8",
    )
    result = license.status()
    assert result["valid"] is True
    assert license.LICENSE_PATH.exists()
    assert not license.LEGACY_LICENSE_PATH.exists()


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


def test_online_activation_is_disabled_without_network_access(monkeypatch):
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("network must not be used")

    monkeypatch.setattr(subprocess, "run", forbidden)
    ok, message = license.activate_online("https://example.test", "ABC")
    assert ok is False
    assert called is False
    assert "غیرفعال" in message


def test_invalid_camera_limit_fails_closed(monkeypatch, tmp_path):
    _configure_paths(monkeypatch, tmp_path)
    _configure_machine(monkeypatch)
    document, public_pem = _signed_document(
        tmp_path,
        _payload(camera_limit=-1),
    )
    token = encode_document(document, ["MACHINE-A"], public_pem)
    ok, message = license.install_license(token)
    assert ok is False
    assert "دوربین" in message
