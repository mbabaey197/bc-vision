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
            sort_order INTEGER
        );
        INSERT INTO cameras VALUES(
            1,'Gate','rtsp://gate',1,2
        );
        INSERT INTO cameras VALUES(
            2,'Disabled','rtsp://off',0,1
        );
        INSERT INTO cameras VALUES(
            3,'Demo','demo://camera',1,1
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

    assert manager.start_enabled_cameras() == 2
    assert calls == [
        (3, "demo://camera", "Demo", 960, 7, 82),
        (1, "rtsp://gate", "Gate", 960, 7, 82),
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
