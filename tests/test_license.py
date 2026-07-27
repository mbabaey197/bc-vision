import subprocess

from app import license


def test_windows_machine_id_uses_hidden_child_process(monkeypatch):
    calls = []

    monkeypatch.setattr(license.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        license.platform,
        "machine",
        lambda: "AMD64",
    )
    monkeypatch.setattr(
        license.platform,
        "node",
        lambda: "test-host",
    )
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
