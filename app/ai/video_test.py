"""Video ANPR with ROI support, multi-frame voting and duplicate suppression."""
from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import secrets
import cv2
import numpy as np

from app.media_storage import (
    MediaWriteError,
    PendingMediaFile,
    crop_from_bbox,
    encode_jpeg_bytes,
    save_encoded_event_images,
    save_event_images,
    settle_pending_media,
    validate_encoded_jpeg_bytes,
    vehicle_snapshot,
)

from .event_dedup import (
    PlateVisitLedger,
    candidate_plate_key,
)
from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    process_frame,
)


_VIDEO_RESULT_TRANSPORT_TYPE = "bcvision.video-process-result"
_VIDEO_RESULT_TRANSPORT_VERSION = 2
_VIDEO_RESULT_TRANSPORT_KEYS = {
    "transport_type",
    "version",
    "info",
    "events",
}
_TRANSPORT_PLATE_JPEG_KEY = "_transport_plate_jpeg"
_TRANSPORT_VEHICLE_JPEG_KEY = "_transport_vehicle_jpeg"
_TRANSPORT_PLATE_ERROR_KEY = "_transport_plate_error"
_TRANSPORT_VEHICLE_ERROR_KEY = "_transport_vehicle_error"
_TRANSPORT_MEDIA_KEYS = frozenset({
    _TRANSPORT_PLATE_JPEG_KEY,
    _TRANSPORT_VEHICLE_JPEG_KEY,
    _TRANSPORT_PLATE_ERROR_KEY,
    _TRANSPORT_VEHICLE_ERROR_KEY,
})
DEFAULT_VIDEO_RESULT_BYTES = 48 * 1024 * 1024
MAX_VIDEO_RESULT_EVENTS = 10_000


def _encode_transport_value(
    value,
    *,
    active: set[int] | None = None,
    depth: int = 0,
):
    """Convert an engine result to the subprocess runner's data subset."""

    if isinstance(value, PendingMediaFile):
        return value.to_transport_descriptor()
    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if isinstance(value, np.generic):
        return _encode_transport_value(
            value.item(),
            active=active,
            depth=depth,
        )
    if isinstance(value, Path):
        return str(value)
    if depth >= 64 or not isinstance(value, (list, tuple, dict)):
        raise TypeError(
            "video result contains a non-transport value or excessive nesting"
        )
    active = active if active is not None else set()
    identity = id(value)
    if identity in active:
        raise TypeError("video result contains a reference cycle")
    active.add(identity)
    try:
        if isinstance(value, dict):
            encoded = {}
            for key, item in value.items():
                if type(key) is not str:
                    raise TypeError(
                        "video result dictionaries require string keys"
                    )
                encoded[key] = _encode_transport_value(
                    item,
                    active=active,
                    depth=depth + 1,
                )
            return encoded
        encoded_items = tuple(
            _encode_transport_value(
                item,
                active=active,
                depth=depth + 1,
            )
            for item in value
        )
        return encoded_items if isinstance(value, tuple) else list(
            encoded_items
        )
    finally:
        active.remove(identity)


def _decode_transport_value(
    value,
    *,
    allow_pending_media: bool = False,
    depth: int = 0,
):
    """Validate subprocess data and rebuild only pending-media handles."""

    if value is None or type(value) in {bool, int, float, str, bytes}:
        return value
    if depth >= 64 or not isinstance(value, (list, tuple, dict)):
        raise TypeError(
            "video transport contains a non-data value or excessive nesting"
        )
    if isinstance(value, dict):
        if value.get("transport_type") == "bcvision.pending-media":
            if not allow_pending_media:
                raise TypeError(
                    "isolated video results may not contain pending media"
                )
            return PendingMediaFile.from_transport_descriptor(value)
        decoded = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    "video transport dictionaries require string keys"
                )
            decoded[key] = _decode_transport_value(
                item,
                allow_pending_media=allow_pending_media,
                depth=depth + 1,
            )
        return decoded
    decoded_items = tuple(
        _decode_transport_value(
            item,
            allow_pending_media=allow_pending_media,
            depth=depth + 1,
        )
        for item in value
    )
    return decoded_items if isinstance(value, tuple) else list(decoded_items)


def serialize_process_video_result(info, events) -> dict[str, object]:
    """Build a versioned primitive-only subprocess result envelope."""

    if not isinstance(info, dict) or not isinstance(events, list):
        raise TypeError("video process result shape is invalid")
    return {
        "transport_type": _VIDEO_RESULT_TRANSPORT_TYPE,
        "version": _VIDEO_RESULT_TRANSPORT_VERSION,
        "info": _encode_transport_value(info),
        "events": _encode_transport_value(events),
    }


def restore_process_video_result(
    value,
    *,
    allow_pending_media: bool = False,
    max_events: int = MAX_VIDEO_RESULT_EVENTS,
    max_media_bytes: int = DEFAULT_VIDEO_RESULT_BYTES,
) -> tuple[dict, list]:
    """Validate an isolated result completely before any parent-side write."""

    if (
        not isinstance(value, dict)
        or set(value) != _VIDEO_RESULT_TRANSPORT_KEYS
        or value.get("transport_type") != _VIDEO_RESULT_TRANSPORT_TYPE
        or type(value.get("version")) is not int
        or value["version"] != _VIDEO_RESULT_TRANSPORT_VERSION
    ):
        raise TypeError("video process transport envelope is invalid")
    info = _decode_transport_value(
        value["info"],
        allow_pending_media=allow_pending_media,
    )
    events = _decode_transport_value(
        value["events"],
        allow_pending_media=allow_pending_media,
    )
    if not isinstance(info, dict) or not isinstance(events, list):
        raise TypeError("video process transport payload shape is invalid")
    if (
        type(max_events) is not int
        or max_events < 0
        or len(events) > max_events
    ):
        raise TypeError("video process result has too many events")
    if type(max_media_bytes) is not int or max_media_bytes <= 0:
        raise TypeError("video process media bound is invalid")

    def validate_data(item, *, depth=0):
        if item is None or type(item) in {bool, int, str, bytes}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise TypeError("video process result contains a non-finite number")
            return
        if depth >= 64 or not isinstance(item, (list, tuple, dict)):
            raise TypeError("video process result contains an invalid value")
        values = item.values() if isinstance(item, dict) else item
        for child in values:
            validate_data(child, depth=depth + 1)

    validate_data(info)
    total_media_bytes = 0
    for event in events:
        if not isinstance(event, dict):
            raise TypeError("video process event is not a dictionary")
        if not allow_pending_media and "_pending_media" in event:
            raise TypeError("isolated video event contains pending media")
        present_transport_keys = _TRANSPORT_MEDIA_KEYS.intersection(event)
        if not allow_pending_media and (
            present_transport_keys != _TRANSPORT_MEDIA_KEYS
        ):
            raise TypeError("isolated video event media schema is incomplete")
        if allow_pending_media and not present_transport_keys:
            continue
        for payload_key in (
            _TRANSPORT_PLATE_JPEG_KEY,
            _TRANSPORT_VEHICLE_JPEG_KEY,
        ):
            payload = event[payload_key]
            if payload is not None:
                validate_encoded_jpeg_bytes(payload)
                total_media_bytes += len(payload)
                if total_media_bytes > max_media_bytes:
                    raise TypeError("video process media exceeds its result bound")
        for error_key in (
            _TRANSPORT_PLATE_ERROR_KEY,
            _TRANSPORT_VEHICLE_ERROR_KEY,
        ):
            if type(event[error_key]) is not str:
                raise TypeError("isolated video event media error is invalid")
        validate_data(event)
    return info, events


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
    transport_media=False,
):
    result = add_vehicle_analysis(result, frame)
    event = {
        key: value
        for key, value in result.items()
        if key not in {"crop", "vehicle_crop", "capture_frame"}
    }
    if transport_media:
        plate_payload = (
            existing.get(_TRANSPORT_PLATE_JPEG_KEY)
            if existing
            else None
        )
        vehicle_payload = (
            existing.get(_TRANSPORT_VEHICLE_JPEG_KEY)
            if existing
            else None
        )
        plate_error = ""
        vehicle_error = ""
        if plate_payload is None:
            try:
                plate_image = result.get("crop")
                if plate_image is None or not getattr(plate_image, "size", 0):
                    plate_image = crop_from_bbox(frame, result.get("bbox"))
                plate_payload = encode_jpeg_bytes(plate_image, quality=94)
            except Exception as exc:  # noqa: BLE001 - retain textual result
                plate_error = (
                    str(exc)
                    if isinstance(exc, MediaWriteError)
                    else f"{type(exc).__name__}: {exc}"
                )
        if vehicle_payload is None:
            try:
                vehicle_payload = encode_jpeg_bytes(
                    vehicle_snapshot(result, frame),
                    quality=90,
                )
            except Exception as exc:  # noqa: BLE001 - retain textual result
                vehicle_error = (
                    str(exc)
                    if isinstance(exc, MediaWriteError)
                    else f"{type(exc).__name__}: {exc}"
                )
        complete = plate_payload is not None and vehicle_payload is not None
        partial = plate_payload is not None or vehicle_payload is not None
        errors = []
        if plate_error:
            errors.append(f"plate: {plate_error}")
        if vehicle_error:
            errors.append(f"vehicle: {vehicle_error}")
        event.update({
            "plate_path": "",
            "image_path": "",
            "media_status": (
                "complete" if complete else ("partial" if partial else "error")
            ),
            "media_error": "; ".join(errors),
            "frame": frame_no,
            "video_second": round(frame_no / fps, 2),
            "video_path": str(video_path),
            _TRANSPORT_PLATE_JPEG_KEY: plate_payload,
            _TRANSPORT_VEHICLE_JPEG_KEY: vehicle_payload,
            _TRANSPORT_PLATE_ERROR_KEY: plate_error,
            _TRANSPORT_VEHICLE_ERROR_KEY: vehicle_error,
        })
        return event

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    existing_plate_path = (
        str(existing.get("plate_path") or "") if existing else ""
    )
    existing_vehicle_path = (
        str(existing.get("image_path") or "") if existing else ""
    )
    plate_file = Path(
        existing_plate_path
        if existing_plate_path
        else plate_dir / f"plate-{stamp}.jpg"
    )
    snap_file = Path(
        existing_vehicle_path
        if existing_vehicle_path
        else snapshot_dir / f"vehicle-{stamp}.jpg"
    )
    media = save_event_images(
        result,
        frame,
        plate_target=plate_file,
        vehicle_target=snap_file,
        existing_plate_path=existing_plate_path,
        existing_vehicle_path=existing_vehicle_path,
        defer_commit=True,
    )
    pending_media = tuple(
        existing.get("_pending_media") or ()
    ) if existing else ()
    pending_media += tuple(media.pending_writes)
    event.update({
        "plate_path": media.plate_path,
        "image_path": media.image_path,
        "media_status": media.media_status,
        "media_error": media.media_error,
        "frame": frame_no,
        "video_second": round(frame_no / fps, 2),
        "video_path": str(video_path),
        "_pending_media": pending_media,
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
    detector_variant=None,
    *,
    transport_media=False,
    transport_media_bytes=DEFAULT_VIDEO_RESULT_BYTES,
):
    from .model_manager import normalize_detector_variant

    if detector_variant is None:
        try:
            from app.database import get_setting

            detector_variant = get_setting(
                "anpr_detector_model",
                "yolo11n",
            )
        except Exception:
            detector_variant = "yolo11n"
    detector_variant = normalize_detector_variant(detector_variant)
    pinned_detector_revision = ""
    if detector_variant == "yolox":
        from .model_manager import yolox_detector_spec

        pinned_spec = yolox_detector_spec()
        if not pinned_spec.get("ready"):
            raise FileNotFoundError(
                pinned_spec.get("error")
                or "Verified YOLOX detector is not installed"
            )
        pinned_detector_revision = str(
            pinned_spec.get("model_revision", "")
        ).strip()
        if not pinned_detector_revision:
            raise ValueError("YOLOX detector revision is missing")
    tester = VideoTester(video_path)
    info = tester.info()
    info["detector_variant"] = detector_variant
    info["detector_execution_mode"] = "exclusive-baseline"
    info["exclusive_detector"] = True
    if pinned_detector_revision:
        info["detector_model_revision"] = pinned_detector_revision
    info["candidate_shadow_requested"] = bool(
        include_candidate_shadow
    )
    info["candidate_shadow_enabled"] = False
    info["candidate_shadow_error"] = (
        "موتور Shadow برای حفظ اجرای انحصاری مدل انتخاب‌شده غیرفعال است."
        if include_candidate_shadow
        else ""
    )
    fps = max(info["fps"], 1.0)
    plate_dir = Path(plate_dir)
    snapshot_dir = Path(snapshot_dir)
    if transport_media:
        if (
            type(transport_media_bytes) is not int
            or transport_media_bytes <= 0
            or transport_media_bytes > 512 * 1024 * 1024
        ):
            raise ValueError("video result media bound is invalid")
    else:
        plate_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
    def new_tracker():
        return PlateConsensusTracker(
            min_votes=2,
            max_age_seconds=max(
                1.2,
                float(frame_step) * 4.0 / fps,
            ),
            emit_cooldown=max(0.0, float(duplicate_seconds)),
            emit_unreadable=True,
        )
    trackers = {"baseline": new_tracker()}
    visits = {"baseline": PlateVisitLedger()}
    events = []
    events_by_track: dict[tuple[str, int], int] = {}
    frame_no = 0
    last_frame = None
    encoded_media_bytes = 0

    def accept(rows, frame, lane="baseline"):
        nonlocal encoded_media_bytes
        for result in rows:
            result = dict(result)
            result["engine_lane"] = lane
            capture_only = bool(result.get("capture_only"))
            if capture_only and result.get("provisional"):
                # Do not turn every short tracker fragment into a database
                # row.  The tracker retains the best capture until a strict
                # consensus or final unreadable result is available.
                continue
            if (
                not capture_only
                and result["confidence"] < float(min_confidence)
            ):
                continue
            now_sec = frame_no / fps
            track_id = int(result.get("track_id") or 0)
            track_key = (lane, track_id)
            event_index = events_by_track.get(track_key)
            key, visit_event_index = visits[lane].event_ref(
                result,
                now_sec,
                duplicate_seconds,
                allow_candidate=True,
            )
            if visit_event_index is not None:
                event_index = visit_event_index
            existing = (
                events[event_index]
                if event_index is not None
                else None
            )
            existing_identity = (
                candidate_plate_key(existing) if existing else ""
            )
            incoming_identity = candidate_plate_key(result)
            if existing_identity and not incoming_identity:
                # A final unreadable fragment may never downgrade an event
                # that this visit already identified as a complete strict or
                # review-only candidate.
                continue
            if (
                existing_identity
                and incoming_identity
                and incoming_identity != existing_identity
            ):
                event_index = None
                existing = None
            if event_index is None and len(events) >= int(max_events):
                return True
            capture_frame = result.pop("capture_frame", None)
            persistence_frame = (
                capture_frame
                if capture_frame is not None
                and getattr(capture_frame, "size", 0)
                else frame
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
                transport_media=transport_media,
            )
            if transport_media:
                old_size = sum(
                    len(existing.get(key) or b"")
                    for key in (
                        _TRANSPORT_PLATE_JPEG_KEY,
                        _TRANSPORT_VEHICLE_JPEG_KEY,
                    )
                ) if existing else 0
                new_size = sum(
                    len(saved.get(key) or b"")
                    for key in (
                        _TRANSPORT_PLATE_JPEG_KEY,
                        _TRANSPORT_VEHICLE_JPEG_KEY,
                    )
                )
                projected_size = encoded_media_bytes - old_size + new_size
                if projected_size > transport_media_bytes:
                    raise ValueError(
                        "حجم تصاویر نتیجه از حد امن عبور کرد؛ "
                        "ویدئو را به بخش‌های کوتاه‌تر تقسیم کنید."
                    )
                encoded_media_bytes = projected_size
            if event_index is None:
                event_index = len(events)
                events.append(saved)
            else:
                events[event_index] = saved
            events_by_track[track_key] = event_index
            if key:
                visits[lane].register(
                    result,
                    event_index,
                    now_sec,
                    allow_candidate=True,
                )
            if len(events) >= int(max_events):
                return True
        return False

    processing_succeeded = False
    limit_reached = False
    try:
        for frame in tester.frames():
            frame_no += 1
            last_frame = frame
            if frame_no % max(1, int(frame_step)) != 0:
                continue
            source, offset_x, offset_y = _roi_frame(frame, roi)
            inference_metadata = {}
            process_kwargs = {"detector_variant": detector_variant}
            if pinned_detector_revision:
                process_kwargs.update(
                    expected_detector_revision=pinned_detector_revision,
                    runtime_metadata=inference_metadata,
                )
            primary = process_frame(
                source,
                min_confidence,
                **process_kwargs,
            )
            primary = [
                {
                    **result,
                    "engine_lane": "baseline",
                    "detector_variant": detector_variant,
                    "detector_selection_exclusive": True,
                }
                for result in primary
            ]
            primary = [
                _translate_result(result, offset_x, offset_y)
                for result in primary
            ]
            detector_revisions = {
                str(result.get("detector_model_revision", "")).strip()
                for result in primary
                if str(result.get("detector_model_revision", "")).strip()
            }
            call_detector_revision = str(
                inference_metadata.get("detector_model_revision", "")
            ).strip()
            if call_detector_revision:
                detector_revisions.add(call_detector_revision)
            if len(detector_revisions) > 1:
                raise RuntimeError(
                    "one video frame returned mixed detector revisions"
                )
            if pinned_detector_revision:
                if detector_revisions != {pinned_detector_revision}:
                    raise RuntimeError(
                        "YOLOX detector revision changed during video test; "
                        "retry with one pinned manifest"
                    )
            elif detector_revisions:
                info["detector_model_revision"] = next(
                    iter(detector_revisions)
                )
            tracker = trackers["baseline"]
            stable = tracker.update(
                primary,
                timestamp=frame_no / fps,
                frame=frame,
            )
            retired_tracks = visits["baseline"].observe(
                primary,
                tracker.active_track_ids(),
                frame_no / fps,
                duplicate_seconds,
            )
            if retired_tracks:
                tracker.retire_tracks(retired_tracks)
                for track_id in retired_tracks:
                    events_by_track.pop(("baseline", track_id), None)
            events_by_track.update({
                ("baseline", track_id): event_index
                for track_id, event_index in visits[
                    "baseline"
                ].track_event_refs().items()
            })
            if accept(stable, frame, "baseline"):
                limit_reached = True
                break
        if last_frame is not None and not limit_reached:
            accept(
                trackers["baseline"].flush(),
                last_frame,
                "baseline",
            )
        processing_succeeded = True
    finally:
        try:
            tester.close()
        finally:
            if not processing_succeeded:
                settle_pending_media(
                    pending
                    for event in events
                    for pending in tuple(
                        event.get("_pending_media") or ()
                    )
                )
    return info, events


def process_video_transport(
    video_path,
    plate_dir,
    snapshot_dir,
    frame_step=5,
    max_events=100,
    min_confidence=0.20,
    duplicate_seconds=2.5,
    roi=None,
    include_candidate_shadow=False,
    detector_variant=None,
    transport_media_bytes=DEFAULT_VIDEO_RESULT_BYTES,
):
    """Run isolated analysis without any managed-media or acceptance write."""

    if detector_variant is None or not str(detector_variant).strip():
        raise ValueError("isolated video processing requires a pinned detector")

    info, events = process_video(
        video_path,
        plate_dir,
        snapshot_dir,
        frame_step=frame_step,
        max_events=max_events,
        min_confidence=min_confidence,
        duplicate_seconds=duplicate_seconds,
        roi=roi,
        include_candidate_shadow=include_candidate_shadow,
        detector_variant=detector_variant,
        transport_media=True,
        transport_media_bytes=transport_media_bytes,
    )
    return serialize_process_video_result(info, events)


def persist_transport_event_media(events, plate_dir, snapshot_dir) -> list:
    """Create parent-owned pending media from a fully validated child result."""

    plate_dir = Path(plate_dir)
    snapshot_dir = Path(snapshot_dir)
    persisted = []
    pending_media = []
    try:
        for index, raw in enumerate(events, start=1):
            event = dict(raw)
            plate_payload = event.pop(_TRANSPORT_PLATE_JPEG_KEY)
            vehicle_payload = event.pop(_TRANSPORT_VEHICLE_JPEG_KEY)
            plate_error = event.pop(_TRANSPORT_PLATE_ERROR_KEY)
            vehicle_error = event.pop(_TRANSPORT_VEHICLE_ERROR_KEY)
            event.pop("_pending_media", None)
            stamp = (
                datetime.now().strftime("%Y%m%d-%H%M%S-%f")
                + f"-{index:06d}-{secrets.token_hex(4)}"
            )
            initial_errors = []
            if plate_error:
                initial_errors.append(f"plate: {plate_error}")
            if vehicle_error:
                initial_errors.append(f"vehicle: {vehicle_error}")
            media = save_encoded_event_images(
                plate_payload=plate_payload,
                vehicle_payload=vehicle_payload,
                plate_target=plate_dir / f"plate-{stamp}.jpg",
                vehicle_target=snapshot_dir / f"vehicle-{stamp}.jpg",
                initial_errors=initial_errors,
                defer_commit=True,
            )
            event.update({
                "plate_path": media.plate_path,
                "image_path": media.image_path,
                "media_status": media.media_status,
                "media_error": media.media_error,
                "_pending_media": tuple(media.pending_writes),
            })
            pending_media.extend(media.pending_writes)
            persisted.append(event)
        return persisted
    except BaseException:
        settle_pending_media(pending_media)
        raise
