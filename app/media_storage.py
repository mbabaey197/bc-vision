"""Reliable filesystem persistence for ANPR image evidence."""
from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from app.file_identity import descriptor_file_identity, path_file_identity
from app.storage_policy import StorageWriteRejected, begin_media_write


class MediaWriteError(RuntimeError):
    """Raised when an image cannot be encoded or published safely."""


class MediaTransportError(MediaWriteError):
    """Raised when a subprocess media handoff cannot be trusted."""


_PENDING_MEDIA_TRANSPORT_TYPE = "bcvision.pending-media"
_PENDING_MEDIA_TRANSPORT_VERSION = 1
_PENDING_MEDIA_TRANSPORT_KEYS = {
    "transport_type",
    "version",
    "path",
    "acceptance_id",
    "device",
    "inode",
    "size_bytes",
}
MAX_ENCODED_JPEG_BYTES = 64 * 1024 * 1024
MAX_JPEG_DIMENSION = 16_384
MAX_JPEG_PIXELS = 50_000_000
_JPEG_START_OF_FRAME_MARKERS = frozenset({
    0xC0,
    0xC1,
    0xC2,
    0xC3,
    0xC5,
    0xC6,
    0xC7,
    0xC9,
    0xCA,
    0xCB,
    0xCD,
    0xCE,
    0xCF,
})


@dataclass(frozen=True)
class MediaSaveResult:
    plate_path: str = ""
    image_path: str = ""
    media_status: str = "disabled"
    media_error: str = ""
    pending_writes: tuple["PendingMediaFile", ...] = field(
        default_factory=tuple,
        repr=False,
        compare=False,
    )


class PendingMediaFile:
    """A durable file waiting for its SQLite owner transaction."""

    def __init__(
        self,
        path: Path,
        reservation,
        acceptance_id: str,
        identity: tuple[int, int],
        size_bytes: int,
        *,
        recovery_required: bool = False,
    ):
        self.path = Path(path)
        self.acceptance_id = str(acceptance_id)
        self.identity = (int(identity[0]), int(identity[1]))
        self.size_bytes = int(size_bytes)
        self._reservation = reservation
        self._recovery_required = bool(recovery_required)

    def _validate_current_file(self, *, allow_missing: bool = False) -> bool:
        try:
            current = self.path.lstat()
        except FileNotFoundError:
            if allow_missing:
                return False
            raise MediaTransportError(
                "pending media disappeared before subprocess handoff"
            ) from None
        except OSError as exc:
            raise MediaTransportError(
                "pending media could not be inspected"
            ) from exc
        if (
            not stat.S_ISREG(current.st_mode)
            or int(current.st_nlink) != 1
        ):
            raise MediaTransportError(
                "pending media is not a private regular file"
            )
        try:
            identity = path_file_identity(self.path, details=current)
        except OSError as exc:
            raise MediaTransportError(
                "pending media identity could not be inspected"
            ) from exc
        if identity != self.identity or int(current.st_size) != self.size_bytes:
            raise MediaTransportError(
                "pending media identity changed during subprocess handoff"
            )
        return True

    def _validate_intent(self, intent, *, accepted: bool | None = None) -> None:
        if intent is None:
            raise MediaTransportError(
                "pending media acceptance intent is missing"
            )
        try:
            target = Path(str(intent.get("target_path", "")))
            state = str(intent.get("state", ""))
        except (AttributeError, TypeError, ValueError) as exc:
            raise MediaTransportError(
                "pending media acceptance intent is invalid"
            ) from exc
        if target != self.path:
            raise MediaTransportError(
                "pending media acceptance target changed"
            )
        if state not in {"pending", "accepted"}:
            raise MediaTransportError(
                "pending media acceptance state is invalid"
            )
        if accepted is not None and (state == "accepted") != accepted:
            expected = "accepted" if accepted else "pending"
            raise MediaTransportError(
                f"pending media acceptance intent is not {expected}"
            )
        if state == "accepted":
            try:
                intent_identity = (
                    int(intent.get("device")),
                    int(intent.get("inode")),
                )
                intent_size = int(intent.get("size_bytes"))
            except (TypeError, ValueError) as exc:
                raise MediaTransportError(
                    "accepted media identity is invalid"
                ) from exc
            if (
                intent_identity != self.identity
                or intent_size != self.size_bytes
                or not str(intent.get("owner_kind") or "").strip()
                or not str(intent.get("owner_id") or "").strip()
            ):
                raise MediaTransportError(
                    "accepted media does not match subprocess handoff"
                )

    def _recover_journal(self) -> None:
        from app.storage_policy import storage_status

        # Never mutate the path directly here. The durable quota journal owns
        # rollback/commit and revalidates its exact inode before either action.
        storage_status(force=True)

    def to_transport_descriptor(self) -> dict[str, object]:
        """Export a versioned, data-only description for a spawned parent."""

        from app.media_acceptance import load_intent

        if self._recovery_required:
            raise MediaTransportError(
                "recovered media cannot be exported as a new handoff"
            )
        canonical = self.path.resolve(strict=False)
        if not canonical.is_absolute():
            raise MediaTransportError(
                "pending media transport path must be absolute"
            )
        self.path = canonical
        self._validate_current_file()
        self._validate_intent(load_intent(self.acceptance_id), accepted=False)
        return {
            "transport_type": _PENDING_MEDIA_TRANSPORT_TYPE,
            "version": _PENDING_MEDIA_TRANSPORT_VERSION,
            "path": str(self.path),
            "acceptance_id": self.acceptance_id,
            "device": self.identity[0],
            "inode": self.identity[1],
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_transport_descriptor(cls, value) -> "PendingMediaFile":
        """Rebuild a recovery-only handle without trusting pickle objects."""

        if not isinstance(value, dict) or set(value) != (
            _PENDING_MEDIA_TRANSPORT_KEYS
        ):
            raise MediaTransportError(
                "pending media transport descriptor has an invalid schema"
            )
        if (
            value.get("transport_type") != _PENDING_MEDIA_TRANSPORT_TYPE
            or type(value.get("version")) is not int
            or value["version"] != _PENDING_MEDIA_TRANSPORT_VERSION
            or type(value.get("path")) is not str
            or type(value.get("acceptance_id")) is not str
            or type(value.get("device")) is not int
            or type(value.get("inode")) is not int
            or type(value.get("size_bytes")) is not int
        ):
            raise MediaTransportError(
                "pending media transport descriptor is invalid"
            )
        if (
            value["device"] < 0
            or value["inode"] < 0
            or value["size_bytes"] < 0
        ):
            raise MediaTransportError(
                "pending media transport identity is invalid"
            )
        acceptance_id = value["acceptance_id"].strip().lower()
        if (
            acceptance_id != value["acceptance_id"]
            or len(acceptance_id) != 32
            or any(
                character not in "0123456789abcdef"
                for character in acceptance_id
            )
        ):
            raise MediaTransportError(
                "pending media transport acceptance id is invalid"
            )
        path = Path(value["path"])
        try:
            canonical = path.resolve(strict=False)
        except (OSError, RuntimeError, ValueError) as exc:
            raise MediaTransportError(
                "pending media transport path could not be resolved"
            ) from exc
        if not path.is_absolute() or canonical != path:
            raise MediaTransportError(
                "pending media transport path is not canonical"
            )
        pending = cls(
            path,
            None,
            acceptance_id,
            (value["device"], value["inode"]),
            value["size_bytes"],
            recovery_required=True,
        )
        pending._validate_current_file()
        from app.media_acceptance import load_intent

        pending._validate_intent(load_intent(acceptance_id), accepted=False)
        return pending

    def accept(self, connection, *, owner_kind: str, owner_id) -> None:
        from app.media_acceptance import accept_intent

        self._validate_current_file()
        accept_intent(
            connection,
            self.acceptance_id,
            self.path,
            self.identity,
            self.size_bytes,
            owner_kind=owner_kind,
            owner_id=owner_id,
        )

    def finalize(self) -> None:
        reservation = self._reservation
        if reservation is None and not self._recovery_required:
            return
        if self._recovery_required:
            from app.media_acceptance import load_intent

            self._validate_current_file()
            self._validate_intent(
                load_intent(self.acceptance_id),
                accepted=True,
            )
            self._recover_journal()
            if load_intent(self.acceptance_id) is not None:
                raise MediaTransportError(
                    "accepted media journal recovery is incomplete"
                )
            self._validate_current_file()
            self._recovery_required = False
            return
        reservation.close(
            success=True,
            actual_bytes=self.size_bytes,
        )
        self._reservation = None

    def rollback(self) -> None:
        reservation = self._reservation
        if reservation is None and not self._recovery_required:
            return
        if self._recovery_required:
            from app.media_acceptance import (
                discard_pending_intent,
                load_intent,
            )

            self._validate_current_file()
            intent = load_intent(self.acceptance_id)
            self._validate_intent(intent)
            if intent.get("state") == "accepted":
                self.finalize()
                return
            revoked = discard_pending_intent(
                self.acceptance_id,
                self.path,
            )
            if not revoked:
                # A concurrent FULL-synchronous owner commit wins over this
                # compensating rollback. Re-read rather than assuming which
                # transaction acquired SQLite's write lock first.
                intent = load_intent(self.acceptance_id)
                if intent is not None and intent.get("state") == "accepted":
                    self.finalize()
                    return
                if intent is not None:
                    raise MediaTransportError(
                        "pending media rollback lost its intent transition"
                    )
            self._recover_journal()
            if load_intent(self.acceptance_id) is not None:
                raise MediaTransportError(
                    "pending media rollback left its acceptance intent"
                )
            if self._validate_current_file(allow_missing=True):
                raise MediaTransportError(
                    "pending media journal rollback is incomplete"
                )
            self._recovery_required = False
            return
        reservation.close(success=False)
        self._reservation = None

    def settle_after_owner_attempt(self) -> None:
        """Resolve an uncertain DB commit without deleting accepted media."""

        if self._recovery_required:
            self.rollback()
            return
        from app.media_acceptance import load_intent

        intent = load_intent(self.acceptance_id)
        if intent is not None and intent.get("state") == "accepted":
            self.finalize()
        else:
            self.rollback()


def _raise_pending_errors(message: str, errors: list[BaseException]) -> None:
    if not errors:
        return
    if all(isinstance(error, Exception) for error in errors):
        raise ExceptionGroup(message, errors)
    raise BaseExceptionGroup(message, errors)


def settle_pending_media(writes) -> None:
    """Settle every uncertain owner attempt without stranding later writes."""

    errors = []
    for pending in reversed(tuple(writes)):
        try:
            pending.settle_after_owner_attempt()
        except BaseException as exc:
            errors.append(exc)
    _raise_pending_errors("media owner settlement failed", errors)


def finalize_pending_media(writes) -> None:
    """Finalize every DB-accepted write, attempting all even after a failure."""

    errors = []
    for pending in tuple(writes):
        try:
            pending.finalize()
        except BaseException as exc:
            errors.append(exc)
    _raise_pending_errors("media journal finalization failed", errors)


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


def _fsync_directory(directory: Path) -> None:
    """Persist a published directory entry on POSIX filesystems."""

    if os.name == "nt":
        return
    descriptor = os.open(
        str(directory),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _mkdir_durable(directory: Path) -> None:
    """Create a directory chain and persist every new ancestor entry."""

    directory = Path(directory)
    missing = []
    current = directory
    while not current.exists() and current != current.parent:
        missing.append(current)
        current = current.parent
    directory.mkdir(parents=True, exist_ok=True)
    for created in reversed(missing):
        _fsync_directory(created.parent)


def _unlink_owned_file(path: Path, identity: tuple[int, int] | None) -> bool:
    """Remove only the exact private inode created by this write attempt."""

    if identity is None:
        return False
    try:
        current = path.lstat()
    except FileNotFoundError:
        return True
    if (
        not stat.S_ISREG(current.st_mode)
        or int(current.st_nlink) != 1
        or path_file_identity(path, details=current) != identity
    ):
        return False
    path.unlink()
    _fsync_directory(path.parent)
    return True


def crop_from_bbox(frame, bbox):
    """Return a clipped plate crop when the detector crop was not retained."""
    if not _nonempty_image(frame) or not bbox or len(bbox) != 4:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if _nonempty_image(crop) else None


def vehicle_snapshot(result: dict, frame):
    """Build a real vehicle image with the detected plate marked."""
    vehicle = result.get("vehicle_crop")
    using_vehicle_crop = _nonempty_image(vehicle)
    if using_vehicle_crop:
        annotated = vehicle.copy()
    elif _nonempty_image(frame):
        annotated = frame.copy()
    else:
        return None

    bbox = result.get("bbox")
    if not bbox or len(bbox) != 4:
        return annotated
    x1, y1, x2, y2 = (round(float(value)) for value in bbox)
    if using_vehicle_crop and result.get("vehicle_bbox"):
        vx1, vy1, _, _ = result["vehicle_bbox"]
        x1, x2 = x1 - round(vx1), x2 - round(vx1)
        y1, y2 = y1 - round(vy1), y2 - round(vy1)
    height, width = annotated.shape[:2]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return annotated


def validate_encoded_jpeg_bytes(payload: bytes) -> tuple[int, int]:
    """Validate a bounded JPEG header without decompressing hostile pixels."""

    if type(payload) is not bytes:
        raise MediaWriteError("encoded JPEG payload must be bytes")
    if not payload:
        raise MediaWriteError("encoded JPEG payload is empty")
    if len(payload) > MAX_ENCODED_JPEG_BYTES:
        raise MediaWriteError("encoded JPEG exceeds the per-image bound")
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise MediaWriteError("encoded JPEG payload is invalid")

    offset = 2
    payload_size = len(payload)
    while offset + 1 < payload_size:
        if payload[offset] != 0xFF:
            raise MediaWriteError("encoded JPEG marker stream is invalid")
        while offset < payload_size and payload[offset] == 0xFF:
            offset += 1
        if offset >= payload_size:
            break
        marker = payload[offset]
        offset += 1
        if marker == 0xD9:
            break
        if marker == 0xDA:
            break
        if marker == 0x00 or marker == 0xD8 or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > payload_size:
            raise MediaWriteError("encoded JPEG segment is truncated")
        segment_size = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_size < 2 or offset + segment_size > payload_size:
            raise MediaWriteError("encoded JPEG segment size is invalid")
        if marker in _JPEG_START_OF_FRAME_MARKERS:
            if segment_size < 8:
                raise MediaWriteError("encoded JPEG frame header is invalid")
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            if (
                width <= 0
                or height <= 0
                or width > MAX_JPEG_DIMENSION
                or height > MAX_JPEG_DIMENSION
                or width * height > MAX_JPEG_PIXELS
            ):
                raise MediaWriteError("encoded JPEG dimensions are unsafe")
            return width, height
        offset += segment_size
    raise MediaWriteError("encoded JPEG frame header is missing")


def encode_jpeg_bytes(image, quality=90) -> bytes:
    """Encode one bounded JPEG without touching managed storage."""

    if not _nonempty_image(image):
        raise MediaWriteError("image is empty")
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
    if len(payload) > MAX_ENCODED_JPEG_BYTES:
        raise MediaWriteError("encoded JPEG exceeds the per-image bound")
    validate_encoded_jpeg_bytes(payload)
    return payload


def write_jpeg_bytes_atomic(
    target,
    payload: bytes,
    *,
    defer_commit=False,
) -> Path | PendingMediaFile:
    """Publish already-encoded JPEG bytes through the normal quota journal."""

    validate_encoded_jpeg_bytes(payload)

    destination = Path(target)
    reservation = None
    acceptance_id = None
    created_identity = None
    claim_succeeded = False
    try:

        # Reserve the encoded size before creating the destination. Concurrent
        # image/video writes share the same policy reservation ledger.
        try:
            if defer_commit:
                from app.media_acceptance import create_intent

                acceptance_id = create_intent(destination)
            reservation = begin_media_write(
                destination,
                len(payload),
                acceptance_id=acceptance_id,
            )
        except StorageWriteRejected as exc:
            if acceptance_id is not None:
                from app.media_acceptance import discard_intent

                discard_intent(acceptance_id)
            raise MediaWriteError(f"storage policy rejected write: {exc}") from exc

        _mkdir_durable(destination.parent)

        try:
            stream_context = destination.open("xb")
        except FileExistsError as exc:
            raise MediaWriteError(
                "refusing to overwrite existing evidence"
            ) from exc
        with stream_context as stream:
            created = os.fstat(stream.fileno())
            created_identity = descriptor_file_identity(
                stream.fileno(),
                details=created,
            )
            claim_created_path = getattr(
                reservation,
                "claim_created_path",
                None,
            )
            if callable(claim_created_path):
                claim_created_path(destination)
                claim_succeeded = True
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(destination.parent)
        if not destination.is_file() or destination.stat().st_size <= 0:
            raise MediaWriteError("published JPEG is missing or empty")
        size_bytes = int(destination.stat().st_size)
        if defer_commit:
            return PendingMediaFile(
                destination,
                reservation,
                acceptance_id,
                created_identity,
                size_bytes,
            )
        reservation.close(success=True, actual_bytes=size_bytes)
        reservation = None
        return destination
    except Exception as exc:
        cleanup_errors = []
        rollback_completed = False
        if reservation is not None:
            try:
                reservation.close(success=False)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
            else:
                rollback_completed = True
        # The reservation normally removes the claimed inode. This local,
        # identity-checked fallback also covers a failure while durably
        # recording the claim and compatibility reservations used by tests.
        if not claim_succeeded or not rollback_completed:
            try:
                _unlink_owned_file(destination, created_identity)
            except OSError as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if reservation is None and acceptance_id is not None:
            try:
                from app.media_acceptance import discard_intent

                discard_intent(acceptance_id)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            exc.add_note(
                "JPEG cleanup errors: "
                + "; ".join(str(error) for error in cleanup_errors)
            )
        if isinstance(exc, MediaWriteError):
            raise
        raise MediaWriteError(
            f"filesystem write failed: {type(exc).__name__}: {exc}"
        ) from exc


def write_jpeg_atomic(
    target,
    image,
    quality=90,
    *,
    defer_commit=False,
) -> Path | PendingMediaFile:
    """Encode in memory and atomically publish a verified non-empty JPEG.

    Encoding before opening the destination avoids OpenCV's platform-specific
    filename handling and therefore supports Unicode Windows paths.
    """

    return write_jpeg_bytes_atomic(
        target,
        encode_jpeg_bytes(image, quality=quality),
        defer_commit=defer_commit,
    )


def save_encoded_event_images(
    *,
    plate_payload: bytes | None,
    vehicle_payload: bytes | None,
    plate_target,
    vehicle_target,
    initial_errors=(),
    defer_commit=False,
) -> MediaSaveResult:
    """Publish child-encoded evidence only after returning to the parent."""

    plate_path = ""
    image_path = ""
    errors = [str(error) for error in initial_errors if str(error)]
    pending_writes = []
    for label, payload, target in (
        ("plate", plate_payload, plate_target),
        ("vehicle", vehicle_payload, vehicle_target),
    ):
        if payload is None:
            continue
        try:
            written = write_jpeg_bytes_atomic(
                target,
                payload,
                defer_commit=defer_commit,
            )
            if isinstance(written, PendingMediaFile):
                pending_writes.append(written)
                path_value = str(written.path)
            else:
                path_value = str(written)
            if label == "plate":
                plate_path = path_value
            else:
                image_path = path_value
        except Exception as exc:  # noqa: BLE001 - retain textual event
            message = (
                str(exc)
                if isinstance(exc, MediaWriteError)
                else f"{type(exc).__name__}: {exc}"
            )
            errors.append(f"{label}: {message}")

    if plate_path and image_path:
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
        pending_writes=tuple(pending_writes),
    )


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
    reuse_existing_targets=False,
    defer_commit=False,
) -> MediaSaveResult:
    """Persist plate and vehicle evidence independently.

    One failed image never prevents the other image or the textual event from
    being retained. Existing verified evidence also survives a failed refresh.
    """
    plate_path = _verified_path(existing_plate_path)
    image_path = _verified_path(existing_vehicle_path)
    if reuse_existing_targets:
        plate_path = plate_path or _verified_path(plate_target)
        image_path = image_path or _verified_path(vehicle_target)
    errors = []
    pending_writes = []
    requested = int(bool(save_plate)) + int(bool(save_vehicle))

    # Published evidence is immutable. A verified DB path always wins; the
    # retry flag additionally discovers a deterministic file that was
    # published just before a database failure.
    if save_plate and not plate_path:
        try:
            plate_image = result.get("crop")
            if not _nonempty_image(plate_image):
                plate_image = crop_from_bbox(frame, result.get("bbox"))
            written = write_jpeg_atomic(
                plate_target,
                plate_image,
                quality=94,
                defer_commit=defer_commit,
            )
            if isinstance(written, PendingMediaFile):
                pending_writes.append(written)
                plate_path = str(written.path)
            else:
                plate_path = str(written)
        except Exception as exc:  # noqa: BLE001 - preserve textual ANPR event
            message = (
                str(exc)
                if isinstance(exc, MediaWriteError)
                else f"{type(exc).__name__}: {exc}"
            )
            errors.append(f"plate: {message}")

    if save_vehicle and not image_path:
        try:
            written = write_jpeg_atomic(
                vehicle_target,
                vehicle_snapshot(result, frame),
                quality=90,
                defer_commit=defer_commit,
            )
            if isinstance(written, PendingMediaFile):
                pending_writes.append(written)
                image_path = str(written.path)
            else:
                image_path = str(written)
        except Exception as exc:  # noqa: BLE001 - preserve textual ANPR event
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
        pending_writes=tuple(pending_writes),
    )
