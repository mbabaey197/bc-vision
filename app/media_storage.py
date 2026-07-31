"""Reliable filesystem persistence for ANPR image evidence."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from uuid import uuid4

import cv2


class MediaWriteError(RuntimeError):
    """Raised when an image cannot be encoded or published safely."""


@dataclass(frozen=True)
class MediaSaveResult:
    plate_path: str = ""
    image_path: str = ""
    media_status: str = "disabled"
    media_error: str = ""


def _nonempty_image(image) -> bool:
    return bool(image is not None and getattr(image, "size", 0))


def _verified_path(path_value) -> str:
    if not path_value:
        return ""
    try:
        path = Path(path_value)
        if path.is_file() and path.stat().st_size > 0:
            return str(path)
    except OSError:
        pass
    return ""


def crop_from_bbox(frame, bbox):
    """Return a clipped plate crop when the detector crop was not retained."""
    if not _nonempty_image(frame) or not bbox or len(bbox) != 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if _nonempty_image(crop) else None


def vehicle_snapshot(result: dict, frame):
    """Keep the original full-resolution vehicle frame as event evidence.

    Detector-provided vehicle crops can cut off the bonnet, roof or rear of a
    fast-moving vehicle.  The live frame is therefore the primary evidence;
    the crop remains only a recovery fallback for callers without a frame.
    """
    if _nonempty_image(frame):
        annotated = frame.copy()
        using_vehicle_crop = False
    else:
        vehicle = result.get("vehicle_crop")
        if not _nonempty_image(vehicle):
            return None
        annotated = vehicle.copy()
        using_vehicle_crop = True

    bbox = result.get("bbox")
    if not bbox or len(bbox) != 4:
        return annotated
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    if using_vehicle_crop and result.get("vehicle_bbox"):
        vx1, vy1, _, _ = result["vehicle_bbox"]
        x1, x2 = x1 - int(round(vx1)), x2 - int(round(vx1))
        y1, y2 = y1 - int(round(vy1)), y2 - int(round(vy1))
    height, width = annotated.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated


def write_jpeg_atomic(target, image, quality=90) -> Path:
    """Encode in memory and atomically publish a verified non-empty JPEG.

    Encoding before opening the destination avoids OpenCV's platform-specific
    filename handling and therefore supports Unicode Windows paths.
    """
    if not _nonempty_image(image):
        raise MediaWriteError("image is empty")

    destination = Path(target)
    temporary = destination.with_name(
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            encoded, buffer = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(quality)))],
            )
        except Exception as exc:
            raise MediaWriteError(
                f"JPEG encoder failed: {type(exc).__name__}: {exc}"
            ) from exc
        payload = bytes(buffer) if encoded and buffer is not None else b""
        if not payload:
            raise MediaWriteError("JPEG encoder returned no data")

        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.stat().st_size <= 0:
            raise MediaWriteError("temporary JPEG is empty")
        temporary.replace(destination)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise MediaWriteError("published JPEG is missing or empty")
        return destination
    except MediaWriteError:
        temporary.unlink(missing_ok=True)
        raise
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise MediaWriteError(
            f"filesystem write failed: {type(exc).__name__}: {exc}"
        ) from exc


def save_event_images(
    result: dict,
    frame,
    *,
    plate_target,
    vehicle_target,
    save_plate=True,
    save_vehicle=True,
    existing_plate_path="",
    existing_vehicle_path="",
) -> MediaSaveResult:
    """Persist plate and vehicle evidence independently.

    One failed image never prevents the other image or the textual event from
    being retained. Existing verified evidence also survives a failed refresh.
    """
    plate_path = _verified_path(existing_plate_path)
    image_path = _verified_path(existing_vehicle_path)
    errors = []
    requested = int(bool(save_plate)) + int(bool(save_vehicle))

    if save_plate:
        try:
            plate_image = result.get("crop")
            if not _nonempty_image(plate_image):
                plate_image = crop_from_bbox(frame, result.get("bbox"))
            plate_path = str(
                write_jpeg_atomic(plate_target, plate_image, quality=94)
            )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, MediaWriteError)
                else f"{type(exc).__name__}: {exc}"
            )
            errors.append(f"plate: {message}")

    if save_vehicle:
        try:
            image_path = str(
                write_jpeg_atomic(
                    vehicle_target,
                    vehicle_snapshot(result, frame),
                    quality=90,
                )
            )
        except Exception as exc:
            message = (
                str(exc)
                if isinstance(exc, MediaWriteError)
                else f"{type(exc).__name__}: {exc}"
            )
            errors.append(f"vehicle: {message}")

    plate_complete = (not save_plate) or bool(plate_path)
    vehicle_complete = (not save_vehicle) or bool(image_path)
    if not requested:
        status = "disabled"
    elif plate_complete and vehicle_complete:
        status = "complete"
    elif plate_path or image_path:
        status = "partial"
    else:
        status = "error"
    return MediaSaveResult(
        plate_path=plate_path,
        image_path=image_path,
        media_status=status,
        media_error="; ".join(errors),
    )
