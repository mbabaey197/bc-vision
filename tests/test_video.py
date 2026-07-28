from pathlib import Path
import time

import cv2
import numpy as np

import app.ai.video_test as video_test
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
            "crop": frame[80:120, 80:240].copy(),
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
        tmp_path / "plates",
        tmp_path / "snapshots",
        frame_step=1,
        duplicate_seconds=20,
        min_confidence=0.5,
    )
    assert info["frames"] >= 8
    assert len(events) == 1
    assert events[0]["plate"] == "12-ب-345-67"
    assert events[0]["consensus_votes"] >= 2
    assert Path(events[0]["plate_path"]).is_file()
    assert Path(events[0]["image_path"]).is_file()


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
