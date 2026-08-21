import sqlite3

import cv2
import numpy as np

import app.ai.live_worker
import app.streams
from app.streams import CameraStream
from app.streams import StreamManager


def test_start_enabled_cameras_uses_persistent_settings(
    tmp_path,
    monkeypatch,
):
    db = tmp_path / "cameras.db"
    with sqlite3.connect(db) as con:
        con.executescript("""
        CREATE TABLE cameras(
            id INTEGER PRIMARY KEY,
            name TEXT,
            rtsp_url TEXT,
            enabled INTEGER,
            sort_order INTEGER,
            video_anpr_started INTEGER NOT NULL DEFAULT 0,
            video_anpr_completed INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO cameras(id,name,rtsp_url,enabled,sort_order) VALUES(
            1,'Gate','rtsp://gate',1,2
        );
        INSERT INTO cameras(id,name,rtsp_url,enabled,sort_order) VALUES(
            2,'Disabled','rtsp://off',0,1
        );
        INSERT INTO cameras(id,name,rtsp_url,enabled,sort_order) VALUES(
            3,'Demo','demo://camera',1,1
        );
        INSERT INTO cameras(
            id,name,rtsp_url,enabled,sort_order,
            video_anpr_started,video_anpr_completed
        ) VALUES(
            4,'Completed upload','video:///tmp/completed.avi',1,3,1,1
        );
        INSERT INTO cameras(
            id,name,rtsp_url,enabled,sort_order,
            video_anpr_started,video_anpr_completed
        ) VALUES(
            5,'Interrupted upload','video:///tmp/interrupted.avi',1,4,1,0
        );
        """)

    import app.database

    def fake_connect():
        connection = sqlite3.connect(db)
        connection.row_factory = sqlite3.Row
        return connection

    monkeypatch.setattr(
        app.database,
        "connect",
        fake_connect,
    )
    monkeypatch.setattr(
        app.database,
        "get_setting",
        lambda key, default="": {
            "stream_width": "960",
            "live_fps": "7",
            "jpeg_quality": "82",
        }.get(key, default),
    )

    manager = StreamManager()
    calls = []
    monkeypatch.setattr(
        manager,
        "get",
        lambda *args: calls.append(args),
    )

    assert manager.start_enabled_cameras() == 4
    assert calls == [
        (3, "demo://camera", "Demo", 960, 7, 82, False, False),
        (1, "rtsp://gate", "Gate", 960, 7, 82, False, False),
        (
            4,
            "video:///tmp/completed.avi",
            "Completed upload",
            960,
            7,
            82,
            True,
            True,
        ),
        (
            5,
            "video:///tmp/interrupted.avi",
            "Interrupted upload",
            960,
            7,
            82,
            True,
            False,
        ),
    ]


def test_stop_all_stops_every_stream():
    manager = StreamManager()

    class FakeStream:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    first = FakeStream()
    second = FakeStream()
    manager.streams = {1: first, 2: second}

    manager.stop_all()

    assert first.stopped
    assert second.stopped
    assert manager.streams == {}


def test_publish_prioritizes_anpr_and_skips_jpeg_without_viewer(
    monkeypatch,
):
    stream = CameraStream(10, "rtsp://gate", "Gate", fps=5)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    calls = []
    monkeypatch.setattr(
        stream,
        "_queue_anpr",
        lambda _frame: calls.append("anpr"),
    )
    monkeypatch.setattr(
        stream,
        "_encode",
        lambda _frame: calls.append("jpeg") or b"jpeg",
    )

    stream._publish(frame)

    assert calls == ["anpr"]
    assert stream.latest is None

    stream._register_viewer()
    try:
        calls.clear()
        stream._publish(frame)
        assert calls == ["anpr", "jpeg"]
        assert stream.latest == b"jpeg"
    finally:
        stream._unregister_viewer()


def test_frames_tracks_viewer_lifetime_and_releases_preview(monkeypatch):
    stream = CameraStream(11, "demo://camera", "Demo", fps=30)
    monkeypatch.setattr(stream, "start", lambda: None)
    with stream.lock:
        stream.latest = b"jpeg"
        stream.latest_frame = np.zeros((2, 2, 3), dtype=np.uint8)

    frames = stream.frames()
    chunk = next(frames)

    assert stream.viewer_count == 1
    assert b"Content-Type: image/jpeg" in chunk
    assert chunk.endswith(b"jpeg\r\n")

    frames.close()

    assert stream.viewer_count == 0
    assert stream.latest is None
    assert stream.latest_frame is None


def test_rtsp_ingest_is_not_throttled_by_dashboard_fps(monkeypatch):
    stream = CameraStream(12, "rtsp://gate", "Gate", fps=5)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    wait_calls = []

    class RecordingEvent:
        def __init__(self):
            self.value = False

        def clear(self):
            self.value = False

        def is_set(self):
            return self.value

        def set(self):
            self.value = True

        def wait(self, timeout=None):
            wait_calls.append(timeout)
            return self.value

    class Capture:
        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def read(self):
            return True, frame

        def release(self):
            pass

    stream.stop_event = RecordingEvent()
    published = []

    def publish(_frame):
        published.append(_frame)
        if len(published) == 3:
            stream.stop_event.set()

    monkeypatch.setattr(
        app.streams.cv2,
        "VideoCapture",
        lambda *_args: Capture(),
    )
    monkeypatch.setattr(stream, "_publish", publish)

    stream._run()

    assert len(published) == 3
    assert wait_calls == []


def test_live_overlay_survives_stream_resize(monkeypatch):
    stream = CameraStream(
        camera_id=7,
        url="demo://camera",
        name="Gate",
        width=160,
        fps=5,
        quality=70,
    )
    monkeypatch.setattr(
        app.ai.live_worker,
        "live_anpr_detections",
        lambda _camera_id: [{
            "bbox": (80, 60, 240, 140),
            "plate": "12-ب-345-67",
            "confidence": 0.92,
            "valid": True,
        }],
    )
    encoded = {}

    def fake_imencode(_suffix, image, _params):
        encoded["image"] = image.copy()
        return True, np.array([1, 2, 3], dtype=np.uint8)

    monkeypatch.setattr(app.streams.cv2, "imencode", fake_imencode)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)

    assert stream._encode(frame) == b"\x01\x02\x03"
    display = encoded["image"]
    assert display.shape[:2] == (100, 160)
    # The green rectangle drawn before resize must still be visible.
    assert int(display[:, :, 1].max()) > 150


def test_live_overlay_tracks_plate_motion_between_inference_frames():
    rng = np.random.default_rng(20260727)
    previous = np.zeros((140, 240, 3), dtype=np.uint8)
    texture = rng.integers(
        25,
        235,
        (50, 80, 3),
        dtype=np.uint8,
    )
    previous[40:90, 60:140] = texture
    current = np.zeros_like(previous)
    current[46:96, 73:153] = texture

    tracked = CameraStream._track_overlay_rows(
        previous,
        current,
        [{
            "bbox": (60, 40, 140, 90),
            "plate": "12-ب-345-67",
            "confidence": 0.9,
            "valid": True,
        }],
    )

    assert len(tracked) == 1
    x1, y1, x2, y2 = tracked[0]["bbox"]
    assert abs(x1 - 73) <= 2
    assert abs(y1 - 46) <= 2
    assert abs(x2 - 153) <= 2
    assert abs(y2 - 96) <= 2


def test_live_overlay_tracks_plate_motion_and_scale_change():
    rng = np.random.default_rng(20260728)
    previous = np.zeros((180, 300, 3), dtype=np.uint8)
    texture = rng.integers(
        20,
        240,
        (40, 100, 3),
        dtype=np.uint8,
    )
    previous[60:100, 80:180] = texture
    current = np.zeros_like(previous)
    enlarged = app.streams.cv2.resize(
        texture,
        (112, 45),
        interpolation=app.streams.cv2.INTER_CUBIC,
    )
    current[70:115, 101:213] = enlarged

    tracked = CameraStream._track_overlay_rows(
        previous,
        current,
        [{
            "bbox": (80, 60, 180, 100),
            "plate": "12-ب-345-67",
            "confidence": 0.9,
            "valid": True,
        }],
    )

    assert len(tracked) == 1
    x1, y1, x2, y2 = tracked[0]["bbox"]
    assert abs(x1 - 101) <= 5
    assert abs(y1 - 70) <= 5
    assert abs(x2 - 213) <= 6
    assert abs(y2 - 115) <= 5
    assert tracked[0]["tracking_confidence"] > 0


def test_empty_detection_revision_clears_old_overlay(monkeypatch):
    stream = CameraStream(
        camera_id=8,
        url="demo://camera",
        name="Gate",
    )
    stream._overlay_rows = [{
        "bbox": (20, 20, 100, 50),
        "plate": "12-ب-345-67",
        "valid": True,
    }]
    stream._overlay_revision = 3
    stream._overlay_updated_at = 1.0
    frame = np.zeros((100, 180, 3), dtype=np.uint8)
    monkeypatch.setattr(
        app.ai.live_worker,
        "live_anpr_detection_snapshot",
        lambda *_args, **_kwargs: {
            "revision": 4,
            "detections": [],
            "frame": frame.copy(),
            "max_age": 4.0,
        },
    )

    assert stream._live_overlays(frame) == []
    assert stream._overlay_revision == 4


def test_failed_optical_match_never_redraws_stale_coordinates(
    monkeypatch,
):
    stream = CameraStream(
        camera_id=9,
        url="demo://camera",
        name="Gate",
    )
    reference = np.zeros((100, 180, 3), dtype=np.uint8)
    current = np.zeros_like(reference)
    monkeypatch.setattr(
        app.ai.live_worker,
        "live_anpr_detection_snapshot",
        lambda *_args, **_kwargs: {
            "revision": 1,
            "detections": [{
                "bbox": (20, 20, 100, 50),
                "plate": "12-ب-345-67",
                "valid": True,
            }],
            "frame": reference,
            "max_age": 4.0,
        },
    )

    assert stream._live_overlays(current) == []
