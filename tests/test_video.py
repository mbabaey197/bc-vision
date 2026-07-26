from pathlib import Path

import cv2
import numpy as np

import app.ai.video_test as video_test


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
