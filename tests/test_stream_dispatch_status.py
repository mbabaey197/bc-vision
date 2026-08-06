import numpy as np

from app import streams


def test_dispatch_failure_is_visible_and_recovers(monkeypatch):
    camera = streams.CameraStream(7, "demo://", "Gate")
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    def fail(*_args, **_kwargs):
        raise RuntimeError("worker unavailable")

    monkeypatch.setattr(
        "app.ai.live_worker.submit_live_frame",
        fail,
    )
    camera._publish(frame, captured_at=10.0)

    assert camera.state.online is True
    assert camera.state.ai_submitted_frames == 0
    assert camera.state.ai_submit_errors == 1
    assert camera.state.last_ai_error == (
        "RuntimeError: worker unavailable"
    )

    monkeypatch.setattr(
        "app.ai.live_worker.submit_live_frame",
        lambda *_args, **_kwargs: None,
    )
    camera._publish(frame, captured_at=10.2)

    assert camera.state.ai_submitted_frames == 1
    assert camera.state.ai_submit_errors == 1
    assert camera.state.last_ai_error == ""


def test_manager_surfaces_dispatch_and_status_channel_errors(monkeypatch):
    manager = streams.StreamManager()
    camera = streams.CameraStream(9, "demo://", "Gate")
    camera.state.ai_submit_errors = 2
    camera.state.last_ai_error = "RuntimeError: queue closed"
    manager.streams[9] = camera

    monkeypatch.setattr(
        "app.ai.live_worker.live_anpr_status",
        lambda _camera_id: {
            "active": True,
            "last_error": "",
        },
    )
    status = manager.status(9)
    assert status["anpr"]["dispatch_errors"] == 2
    assert status["anpr"]["last_error"] == (
        "frame-dispatch: RuntimeError: queue closed"
    )

    def broken_status(_camera_id):
        raise ValueError("status unavailable")

    monkeypatch.setattr(
        "app.ai.live_worker.live_anpr_status",
        broken_status,
    )
    status = manager.status(9)
    assert status["anpr"]["active"] is False
    assert status["anpr"]["dispatch_errors"] == 2
    assert status["anpr"]["last_error"] == (
        "status-channel: ValueError: status unavailable"
    )
