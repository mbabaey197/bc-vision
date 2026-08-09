import json
from urllib.error import URLError

import launcher
from app.ai import model_manager


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


def test_hide_service_console_only_for_packaged_windows(monkeypatch):
    calls = []

    class _Kernel32:
        @staticmethod
        def GetConsoleWindow():
            return 123

    class _User32:
        @staticmethod
        def ShowWindow(window, command):
            calls.append((window, command))

    class _Windll:
        kernel32 = _Kernel32()
        user32 = _User32()

    import ctypes

    monkeypatch.setattr(launcher.sys, "platform", "win32")
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(ctypes, "windll", _Windll(), raising=False)

    assert launcher.hide_service_console() is True
    assert calls == [(123, 0)]


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


def _models_not_ready():
    return {
        "detector_yolo11n_ready": False,
        "detector_yolov8n_ready": False,
        "detector_fallback_ready": False,
        "hezar_ready": False,
        "crnn_ready": False,
        "cnn_ready": False,
    }


def test_model_preparation_retries_once_and_clears_error(monkeypatch):
    calls = []
    monkeypatch.setattr(model_manager, "model_status", _models_not_ready)

    def prepare_models(*, download):
        assert download is True
        calls.append(True)
        if len(calls) == 1:
            raise OSError("temporary download failure")
        return {"ready": True}

    monkeypatch.setattr(model_manager, "prepare_models", prepare_models)
    monkeypatch.setattr(launcher, "log", lambda _message: None)
    for name in (
        launcher.MODEL_PREPARATION_STATE_ENV,
        launcher.MODEL_PREPARATION_ERROR_ENV,
        launcher.MODEL_PREPARATION_ATTEMPT_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    result = launcher.prepare_anpr_models(
        max_attempts=2,
        retry_delay=0,
    )

    assert result is True
    assert len(calls) == 2
    assert (
        launcher.os.environ[launcher.MODEL_PREPARATION_STATE_ENV]
        == "ready"
    )
    assert (
        launcher.MODEL_PREPARATION_ERROR_ENV
        not in launcher.os.environ
    )
    assert (
        launcher.os.environ[launcher.MODEL_PREPARATION_ATTEMPT_ENV]
        == "2"
    )


def test_model_preparation_publishes_terminal_failure_reason(monkeypatch):
    calls = []
    monkeypatch.setattr(model_manager, "model_status", _models_not_ready)

    def prepare_models(*, download):
        assert download is True
        calls.append(True)
        raise ValueError("YOLOv8n SHA-256 mismatch")

    monkeypatch.setattr(model_manager, "prepare_models", prepare_models)
    monkeypatch.setattr(launcher, "log", lambda _message: None)

    result = launcher.prepare_anpr_models(
        max_attempts=2,
        retry_delay=0,
    )

    assert result is False
    assert len(calls) == 2
    assert (
        launcher.os.environ[launcher.MODEL_PREPARATION_STATE_ENV]
        == "error"
    )
    assert (
        launcher.os.environ[launcher.MODEL_PREPARATION_ERROR_ENV]
        == "ValueError: YOLOv8n SHA-256 mismatch"
    )
    assert (
        launcher.os.environ[launcher.MODEL_PREPARATION_ATTEMPT_ENV]
        == "2"
    )
