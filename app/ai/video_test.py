"""Video ANPR with ROI support, multi-frame voting and duplicate suppression."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import cv2

from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    process_frame,
)
from .plate_rules import normalize_plate


class VideoTester:
    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError("فایل ویدئو قابل باز شدن نیست.")

    def info(self):
        frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        return {
            "frames": frames,
            "fps": fps,
            "width": int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": frames / max(fps, 1.0),
        }

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield frame

    def close(self):
        self.cap.release()


def _roi_frame(frame, roi):
    if not roi:
        return frame, 0, 0
    height, width = frame.shape[:2]
    rx, ry, rw, rh = roi
    x1 = max(0, min(width - 1, int(width * float(rx) / 100.0)))
    y1 = max(0, min(height - 1, int(height * float(ry) / 100.0)))
    x2 = max(x1 + 1, min(width, int(width * (float(rx) + float(rw)) / 100.0)))
    y2 = max(y1 + 1, min(height, int(height * (float(ry) + float(rh)) / 100.0)))
    return frame[y1:y2, x1:x2], x1, y1


def _translate_result(result, offset_x, offset_y):
    if not (offset_x or offset_y):
        return result
    translated = dict(result)
    x1, y1, x2, y2 = result["bbox"]
    translated["bbox"] = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
    if result.get("vehicle_bbox"):
        vx1, vy1, vx2, vy2 = result["vehicle_bbox"]
        translated["vehicle_bbox"] = (
            vx1 + offset_x, vy1 + offset_y, vx2 + offset_x, vy2 + offset_y,
        )
    return translated


def _save_event(result, frame, frame_no, fps, plate_dir, snapshot_dir, video_path):
    result = add_vehicle_analysis(result, frame)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    plate_file = plate_dir / f"plate-{stamp}.jpg"
    snap_file = snapshot_dir / f"vehicle-{stamp}.jpg"
    crop = result.get("crop")
    if crop is not None and getattr(crop, "size", 0):
        cv2.imwrite(str(plate_file), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    annotated = frame.copy()
    x1, y1, x2, y2 = result["bbox"]
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    vehicle_box = result.get("vehicle_bbox") or result["bbox"]
    vx1, vy1, vx2, vy2 = vehicle_box
    cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), (255, 180, 0), 2)
    cv2.imwrite(str(snap_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    event = {key: value for key, value in result.items() if key != "crop"}
    event.update({
        "plate_path": str(plate_file),
        "image_path": str(snap_file),
        "frame": frame_no,
        "video_second": round(frame_no / fps, 2),
        "video_path": str(video_path),
    })
    return event


def process_video(
    video_path,
    plate_dir,
    snapshot_dir,
    frame_step=5,
    max_events=100,
    min_confidence=0.20,
    duplicate_seconds=2.5,
    roi=None,
):
    tester = VideoTester(video_path)
    info = tester.info()
    fps = max(info["fps"], 1.0)
    plate_dir = Path(plate_dir)
    snapshot_dir = Path(snapshot_dir)
    plate_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    tracker = PlateConsensusTracker(
        min_votes=2,
        max_age_seconds=max(1.2, float(frame_step) * 4.0 / fps),
        emit_cooldown=max(0.0, float(duplicate_seconds)),
    )
    events = []
    seen: dict[str, float] = {}
    saved_track_ids = set()
    frame_no = 0
    last_frame = None

    def accept(rows, frame):
        for result in rows:
            if result["confidence"] < float(min_confidence):
                continue
            key = result.get("plate_norm") or normalize_plate(result.get("plate"))
            now_sec = frame_no / fps
            if key and key in seen and now_sec - seen[key] < max(0.0, float(duplicate_seconds)):
                continue
            if result.get("track_id") in saved_track_ids:
                continue
            if key:
                seen[key] = now_sec
            saved_track_ids.add(result.get("track_id"))
            events.append(_save_event(result, frame, frame_no, fps, plate_dir, snapshot_dir, video_path))
            if len(events) >= int(max_events):
                return True
        return False

    try:
        for frame in tester.frames():
            frame_no += 1
            last_frame = frame
            if frame_no % max(1, int(frame_step)) != 0:
                continue
            source, offset_x, offset_y = _roi_frame(frame, roi)
            results = [
                _translate_result(result, offset_x, offset_y)
                for result in process_frame(source, min_confidence)
            ]
            stable = tracker.update(results, timestamp=frame_no / fps)
            if accept(stable, frame):
                return info, events
        if last_frame is not None:
            accept(tracker.flush(), last_frame)
    finally:
        tester.close()
    return info, events
