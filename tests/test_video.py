from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app.ai.video_test as video_test
import app.media_storage as media_storage
import app.streams as streams
from app.streams import CameraStream


def _write_video(path: Path, frames=8):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((180, 320, 3), 30, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (80 + index, 80),
            (240 + index, 120),
            (235, 235, 235),
            -1,
        )
        writer.write(frame)
    writer.release()


def test_video_emits_one_consensus_event(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path)
    calls = {"count": 0}

    def fake_process(frame, threshold):
        calls["count"] += 1
        plate = (
            "12-ب-345-67"
            if calls["count"] != 2
            else "12-ب-345-76"
        )
        confidence = 0.74 if calls["count"] != 2 else 0.55
        return [{
            "plate": plate,
            "plate_norm": plate.replace("-", ""),
            "valid": True,
            "confidence": confidence,
            "detector_confidence": 0.8,
            "ocr_confidence": 0.7,
            "quality_score": 0.8,
            "bbox": (80, 80, 240, 120),
            "crop": None,
            "method": "test",
            "vehicle_type": "سواری",
            "vehicle_color": "سفید",
            "vehicle_brand": "نامشخص",
            "vehicle_confidence": 0.5,
            "vehicle_bbox": (30, 30, 290, 160),
        }]

    monkeypatch.setattr(
        video_test,
        "process_frame",
        fake_process,
    )
    info, events = video_test.process_video(
        video_path,
        tmp_path / "پلاک‌ها",
        tmp_path / "خودروها",
        frame_step=1,
        duplicate_seconds=20,
        min_confidence=0.5,
    )
    assert info["frames"] >= 8
    assert len(events) == 1
    assert events[0]["plate"] == "12-ب-345-67"
    assert events[0]["consensus_votes"] >= 2
    assert events[0]["media_status"] == "complete"
    assert events[0]["media_error"] == ""
    for key in ("plate_path", "image_path"):
        image_path = Path(events[0][key])
        payload = image_path.read_bytes()
        assert len(payload) > 0
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None
        assert decoded.size > 0


def test_video_media_failure_keeps_result_and_reports_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        media_storage.cv2,
        "imencode",
        lambda *_args, **_kwargs: (False, None),
    )
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.9,
        "bbox": (40, 50, 140, 82),
        "crop": frame[50:82, 40:140].copy(),
    }

    event = video_test._save_event(
        result,
        frame,
        frame_no=10,
        fps=10.0,
        plate_dir=tmp_path / "plates",
        snapshot_dir=tmp_path / "snapshots",
        video_path=tmp_path / "source.mp4",
    )

    assert event["plate"] == "12-ب-345-67"
    assert event["plate_path"] == ""
    assert event["image_path"] == ""
    assert event["media_status"] == "error"
    assert "plate: JPEG encoder returned no data" in event["media_error"]
    assert "vehicle: JPEG encoder returned no data" in event["media_error"]


def test_video_shadow_fails_closed_once_when_bundle_is_missing(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path, frames=3)
    monkeypatch.setattr(
        video_test,
        "next_models_status",
        lambda: {
            "ready": False,
            "error": "signed candidate bundle is missing",
        },
    )
    monkeypatch.setattr(
        video_test.engine_router,
        "process",
        lambda *_args, **_kwargs: pytest.fail(
            "router must not run without a verified bundle"
        ),
    )
    monkeypatch.setattr(
        video_test,
        "process_frame",
        lambda *_args, **_kwargs: [],
    )

    info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "snapshots",
        frame_step=1,
        include_candidate_shadow=True,
    )

    assert events == []
    assert info["candidate_shadow_requested"] is True
    assert "signed candidate bundle is missing" in (
        info["candidate_shadow_error"]
    )


def test_video_shadow_complete_consensus_is_auto_confirmed(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path, frames=6)
    monkeypatch.setattr(
        video_test,
        "next_models_status",
        lambda: {"ready": True, "error": ""},
    )

    def candidate(frame):
        return {
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.88,
            "detector_confidence": 0.90,
            "ocr_confidence": 0.86,
            "quality_score": 0.80,
            "bbox": (80, 80, 240, 120),
            "crop": frame[80:120, 80:240].copy(),
            "method": "candidate-test",
            "ocr_engine": "fast-plate-ocr-cct",
        }

    monkeypatch.setattr(
        video_test.engine_router,
        "process",
        lambda frame, **_kwargs: SimpleNamespace(
            primary=[],
            shadow=[candidate(frame)],
            error="",
        ),
    )

    _, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "snapshots",
        frame_step=1,
        include_candidate_shadow=True,
    )

    confirmed = [
        event for event in events
        if event.get("auto_confirmed")
    ]
    assert len(confirmed) == 1
    assert confirmed[0]["plate_norm"] == "31ط55674"
    assert confirmed[0]["read_status"] == "auto-confirmed"
    assert confirmed[0]["needs_review"] is True
    assert confirmed[0]["experimental"] is True


def test_uploaded_video_stream_loops_without_becoming_offline(tmp_path, monkeypatch):
    video_path = tmp_path / "loop.avi"
    _write_video(video_path, frames=3)
    stream = CameraStream(
        91,
        f"video://{video_path}",
        "Uploaded video",
        fps=30,
    )
    published = []
    monkeypatch.setattr(stream, "_publish", lambda frame: published.append(frame))

    stream.start()
    for _ in range(50):
        if len(published) >= 5:
            break
        time.sleep(0.02)
    stream.stop()
    stream.thread.join(timeout=2)

    assert len(published) >= 5
    assert not stream.thread.is_alive()


def test_uploaded_video_uses_ffmpeg_fallback_when_opencv_has_no_frames(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "camera-export.mp4"
    source.write_bytes(b"codec-fixture")
    stream = CameraStream(
        92,
        f"video://{source}",
        "HEVC export",
        fps=30,
    )
    published = []

    class FailedCapture:
        def __init__(self, *_args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    class VideoFrame:
        def to_ndarray(self, format):
            assert format == "bgr24"
            return np.full((24, 32, 3), 90, dtype=np.uint8)

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, video):
            assert video == 0
            yield VideoFrame()

    monkeypatch.setattr(streams.cv2, "VideoCapture", FailedCapture)
    monkeypatch.setattr(
        streams.av,
        "open",
        lambda path: Container(),
    )

    def publish(frame):
        published.append(frame)
        stream.stop_event.set()

    monkeypatch.setattr(stream, "_publish", publish)
    stream._run()

    assert len(published) == 1
    assert published[0].shape == (24, 32, 3)


def test_uploaded_video_stream_produces_real_jpeg(tmp_path):
    video_path = tmp_path / "dashboard.avi"
    _write_video(video_path, frames=3)
    stream = CameraStream(
        93,
        f"video://{video_path}",
        "Dashboard upload",
        fps=30,
    )
    stream.start()
    for _ in range(100):
        if stream.latest:
            break
        time.sleep(0.01)
    stream.stop()
    stream.thread.join(timeout=2)

    assert stream.latest
    assert stream.latest.startswith(b"\xff\xd8")
    assert stream.latest.endswith(b"\xff\xd9")


def test_uploaded_video_can_pause_and_resume(tmp_path, monkeypatch):
    video_path = tmp_path / "playback.avi"
    _write_video(video_path, frames=20)
    stream = CameraStream(
        94,
        f"video://{video_path}",
        "Playback controls",
        fps=30,
    )
    published = []
    monkeypatch.setattr(
        stream,
        "_publish",
        lambda frame: published.append(frame),
    )

    stream.start()
    for _ in range(100):
        if len(published) >= 3:
            break
        time.sleep(0.01)
    assert stream.pause() is True
    paused_count = len(published)
    time.sleep(0.12)
    assert len(published) <= paused_count + 1
    assert stream.state.paused is True

    assert stream.resume() is True
    for _ in range(100):
        if len(published) >= paused_count + 3:
            break
        time.sleep(0.01)
    stream.stop()
    stream.thread.join(timeout=2)

    assert len(published) >= paused_count + 3
    assert stream.state.paused is False
