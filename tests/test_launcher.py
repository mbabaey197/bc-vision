import json
from urllib.error import URLError

import launcher


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _Thread:
    targets = []

    def __init__(self, target, **_kwargs):
        self.target = target
        self.targets.append(target)

    def start(self):
        return None


def test_service_ready_requires_bcvision_health_identity(monkeypatch):
    monkeypatch.setattr(
        launcher,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {"service": "bc-vision", "status": "ok"}
        ),
    )
    assert launcher.service_ready() is True

    monkeypatch.setattr(
        launcher,
        "urlopen",
        lambda *_args, **_kwargs: _Response(
            {"service": "another-app", "status": "ok"}
        ),
    )
    assert launcher.service_ready() is False

    def unavailable(*_args, **_kwargs):
        raise URLError("offline")

    monkeypatch.setattr(launcher, "urlopen", unavailable)
    assert launcher.service_ready() is False


def test_main_runs_server_in_foreground_without_keepalive_window(
    monkeypatch,
):
    calls = []
    _Thread.targets = []
    monkeypatch.setattr(launcher.threading, "Thread", _Thread)
    monkeypatch.setattr(launcher, "service_ready", lambda: False)
    monkeypatch.setattr(launcher, "port_open", lambda: False)
    monkeypatch.setattr(
        launcher,
        "run_server",
        lambda: calls.append("server"),
    )

    launcher.main()

    assert calls == ["server"]
    assert launcher.open_panel_when_ready in _Thread.targets


def test_main_reopens_existing_bcvision_without_second_server(
    monkeypatch,
):
    opened = []
    _Thread.targets = []
    monkeypatch.setattr(launcher.threading, "Thread", _Thread)
    monkeypatch.setattr(launcher, "service_ready", lambda: True)
    monkeypatch.setattr(
        launcher.webbrowser,
        "open",
        lambda url: opened.append(url),
    )
    monkeypatch.setattr(
        launcher,
        "run_server",
        lambda: (_ for _ in ()).throw(
            AssertionError("server must not start")
        ),
    )

    launcher.main()

    assert opened == [launcher.PANEL_URL]


def test_main_rejects_foreign_service_on_port_8000(monkeypatch):
    errors = []
    _Thread.targets = []
    monkeypatch.setattr(launcher.threading, "Thread", _Thread)
    monkeypatch.setattr(launcher, "service_ready", lambda: False)
    monkeypatch.setattr(launcher, "port_open", lambda: True)
    monkeypatch.setattr(
        launcher,
        "show_startup_error",
        lambda message: errors.append(message),
    )
    monkeypatch.setattr(
        launcher,
        "run_server",
        lambda: (_ for _ in ()).throw(
            AssertionError("server must not start")
        ),
    )

    launcher.main()

    assert errors == ["پورت 8000 در اختیار برنامه دیگری است."]
