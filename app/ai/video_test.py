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
from .next_engine import engine_router
from .next_models import next_models_status
from .review_policy import auto_confirm_guess


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


def _save_event(
    result,
    frame,
    frame_no,
    fps,
    plate_dir,
    snapshot_dir,
    video_path,
    existing=None,
):
    result = add_vehicle_analysis(result, frame)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    plate_file = Path(
        existing.get("plate_path")
        if existing
        else plate_dir / f"plate-{stamp}.jpg"
    )
    snap_file = Path(
        existing.get("image_path")
        if existing
        else snapshot_dir / f"vehicle-{stamp}.jpg"
    )
    crop = result.get("crop")
    if crop is not None and getattr(crop, "size", 0):
        cv2.imwrite(str(plate_file), crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
    vehicle = result.get("vehicle_crop")
    using_vehicle_crop = bool(
        vehicle is not None and getattr(vehicle, "size", 0)
    )
    annotated = vehicle.copy() if using_vehicle_crop else frame.copy()
    x1, y1, x2, y2 = result["bbox"]
    if using_vehicle_crop and result.get("vehicle_bbox"):
        vx1, vy1, _, _ = result["vehicle_bbox"]
        x1, x2 = x1 - vx1, x2 - vx1
        y1, y2 = y1 - vy1, y2 - vy1
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(str(snap_file), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    event = {
        key: value
        for key, value in result.items()
        if key not in {"crop", "vehicle_crop", "capture_frame"}
    }
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
    include_candidate_shadow=False,
):
    tester = VideoTester(video_path)
    info = tester.info()
    candidate_status = (
        next_models_status()
        if include_candidate_shadow
        else {"ready": False, "error": ""}
    )
    shadow_enabled = bool(
        include_candidate_shadow and candidate_status.get("ready")
    )
    info["candidate_shadow_requested"] = bool(
        include_candidate_shadow
    )
    info["candidate_shadow_error"] = (
        ""
        if shadow_enabled
        else str(candidate_status.get("error") or "")
    )
    fps = max(info["fps"], 1.0)
    plate_dir = Path(plate_dir)
    snapshot_dir = Path(snapshot_dir)
    plate_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    trackers = {
        lane: PlateConsensusTracker(
            min_votes=2,
            max_age_seconds=max(
                1.2,
                float(frame_step) * 4.0 / fps,
            ),
            emit_cooldown=max(0.0, float(duplicate_seconds)),
            emit_unreadable=True,
        )
        for lane in (
            ("baseline", "candidate-shadow")
            if shadow_enabled
            else ("baseline",)
        )
    }
    events = []
    events_by_track: dict[tuple[str, int], int] = {}
    seen: dict[tuple[str, str], float] = {}
    frame_no = 0
    last_frame = None
    shadow_error = info["candidate_shadow_error"]

    def accept(rows, frame, lane="baseline"):
        for result in rows:
            result = dict(result)
            result["engine_lane"] = lane
            capture_only = bool(result.get("capture_only"))
            if lane == "candidate-shadow":
                result["experimental"] = True
                result["needs_review"] = True
                if not capture_only:
                    result = auto_confirm_guess(result)
                elif result.get("valid"):
                    result["read_status"] = "experimental-guess"
            if (
                not capture_only
                and result["confidence"] < float(min_confidence)
            ):
                continue
            key = result.get("plate_norm") or normalize_plate(result.get("plate"))
            now_sec = frame_no / fps
            track_id = int(result.get("track_id") or 0)
            track_key = (lane, track_id)
            event_index = events_by_track.get(track_key)
            if (
                event_index is None
                and key
                and (lane, key) in seen
                and now_sec - seen[(lane, key)]
                < max(0.0, float(duplicate_seconds))
            ):
                continue
            if key:
                seen[(lane, key)] = now_sec
            capture_frame = result.pop("capture_frame", None)
            persistence_frame = (
                capture_frame
                if capture_frame is not None
                and getattr(capture_frame, "size", 0)
                else frame
            )
            existing = (
                events[event_index]
                if event_index is not None
                else None
            )
            saved = _save_event(
                result,
                persistence_frame,
                frame_no,
                fps,
                plate_dir,
                snapshot_dir,
                video_path,
                existing=existing,
            )
            if event_index is None:
                events_by_track[track_key] = len(events)
                events.append(saved)
            else:
                events[event_index] = saved
            if (
                len(events) >= int(max_events)
                and not capture_only
            ):
                return True
        return False

    try:
        for frame in tester.frames():
            frame_no += 1
            last_frame = frame
            if frame_no % max(1, int(frame_step)) != 0:
                continue
            source, offset_x, offset_y = _roi_frame(frame, roi)
            if shadow_enabled:
                outcome = engine_router.process(
                    source,
                    baseline=lambda: process_frame(
                        source,
                        min_confidence,
                    ),
                    min_detection_confidence=min_confidence,
                    mode="shadow",
                )
                primary = outcome.primary
                shadow = outcome.shadow
                if outcome.error:
                    shadow_error = outcome.error
            else:
                primary = process_frame(source, min_confidence)
                shadow = []
            primary = [
                _translate_result(result, offset_x, offset_y)
                for result in primary
            ]
            stable = trackers["baseline"].update(
                primary,
                timestamp=frame_no / fps,
                frame=frame,
            )
            if accept(stable, frame, "baseline"):
                return info, events
            if shadow_enabled:
                shadow = [
                    _translate_result(result, offset_x, offset_y)
                    for result in shadow
                ]
                stable_shadow = trackers["candidate-shadow"].update(
                    shadow,
                    timestamp=frame_no / fps,
                    frame=frame,
                )
                if accept(
                    stable_shadow,
                    frame,
                    "candidate-shadow",
                ):
                    return info, events
        if last_frame is not None:
            accept(
                trackers["baseline"].flush(),
                last_frame,
                "baseline",
            )
            if shadow_enabled:
                accept(
                    trackers["candidate-shadow"].flush(),
                    last_frame,
                    "candidate-shadow",
                )
    finally:
        tester.close()
    info["candidate_shadow_requested"] = bool(
        include_candidate_shadow
    )
    info["candidate_shadow_error"] = shadow_error
    return info, events
