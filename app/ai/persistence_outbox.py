"""Durable, process-independent storage for pending ANPR event writes.

The live worker keeps its short-lived scheduling state in memory, but an event
must not depend on that memory surviving a camera removal or a process crash.
This module provides a small SQLite sidecar whose only job is to retain the
immutable event payload and image evidence until the primary database
acknowledges it.

The sidecar deliberately does not use pickle.  Results are reduced to an
explicit allow-list and encoded as canonical JSON, while images are stored as
verified JPEG blobs.  A checksum covers both the immutable metadata and the
payload.  Rows that cannot be verified during recovery are quarantined rather
than silently discarded.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
from uuid import uuid4

import cv2
import numpy as np


LEGACY_OUTBOX_SCHEMA_VERSION = 1
OUTBOX_SCHEMA_VERSION = 2
MAX_RESULT_JSON_BYTES = 1 * 1024 * 1024
MAX_JPEG_BYTES = 32 * 1024 * 1024
MAX_ERROR_LENGTH = 4000

# Only fields required to reproduce a persistence decision are retained.
# Inference-only arrays and diagnostic hypotheses are intentionally excluded.
RESULT_FIELD_ALLOWLIST = frozenset({
    "assisted_candidate",
    "auto_confirmed",
    "bbox",
    "best_effort",
    "capture_only",
    "capture_refresh",
    "city",
    "confidence",
    "confirmation_source",
    "consensus_votes",
    "detector_confidence",
    "detector_method",
    "detector_model_revision",
    "direction",
    "experimental",
    "method",
    "model_revision",
    "needs_review",
    "ocr_alternative",
    "ocr_confidence",
    "ocr_disagreement",
    "ocr_engine",
    "ocr_model_revision",
    "operator_learned",
    "operator_reviewed",
    "plate",
    "plate_norm",
    "provisional",
    "quality_score",
    "quadrilateral",
    "raw_guess_confidence",
    "raw_guess_engine",
    "raw_guess_norm",
    "raw_guess_reason",
    "raw_guess_text",
    "read_status",
    "track_id",
    "tracker_finalized",
    "tracking_bbox",
    "tracking_engine",
    "unreadable_final",
    "valid",
    "vehicle_bbox",
    "vehicle_brand",
    "vehicle_color",
    "vehicle_confidence",
    "vehicle_type",
    "visit_identity_stable",
})


class OutboxError(RuntimeError):
    """Base error raised by durable-outbox operations."""


class OutboxPayloadError(OutboxError):
    """Raised when a payload cannot be safely serialized or decoded."""


class OutboxCorruptionError(OutboxPayloadError):
    """Raised when a stored row fails its integrity checks."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _normalized_utc(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise OutboxPayloadError("observed_at_utc is required")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OutboxPayloadError("observed_at_utc is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise OutboxPayloadError("observed_at_utc must include a timezone")
    return parsed.astimezone(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise OutboxPayloadError(f"{field_name} must be numeric") from exc
    if not math.isfinite(result):
        raise OutboxPayloadError(f"{field_name} must be finite")
    return result


def _safe_json_value(value: Any, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise OutboxPayloadError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, np.generic):
        return _safe_json_value(value.item(), path)
    if isinstance(value, (list, tuple)):
        return [
            _safe_json_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise OutboxPayloadError(
                    f"{path} contains a non-string dictionary key"
                )
            converted[key] = _safe_json_value(item, f"{path}.{key}")
        return converted
    raise OutboxPayloadError(
        f"{path} contains unsupported type {type(value).__name__}"
    )


def sanitize_result(result: dict) -> dict:
    """Return a JSON-safe persistence payload with unknown fields removed."""

    if not isinstance(result, dict):
        raise OutboxPayloadError("result must be a dictionary")
    return {
        key: _safe_json_value(result[key], f"result.{key}")
        for key in sorted(RESULT_FIELD_ALLOWLIST.intersection(result))
    }


def _result_json(result: dict) -> str:
    try:
        encoded = json.dumps(
            sanitize_result(result),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise OutboxPayloadError("result is not valid JSON") from exc
    if len(encoded.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
        raise OutboxPayloadError("result JSON exceeds the size limit")
    return encoded


def encode_jpeg(image, *, quality: int = 95) -> bytes:
    """Encode a non-empty image into a bounded JPEG byte string."""

    if image is None or not getattr(image, "size", 0):
        raise OutboxPayloadError("image is empty")
    try:
        encoded, buffer = cv2.imencode(
            ".jpg",
            image,
            [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, int(quality)))],
        )
    except Exception as exc:
        raise OutboxPayloadError(
            f"JPEG encoding failed: {type(exc).__name__}: {exc}"
        ) from exc
    payload = bytes(buffer) if encoded and buffer is not None else b""
    if not payload:
        raise OutboxPayloadError("JPEG encoder returned no data")
    if len(payload) > MAX_JPEG_BYTES:
        raise OutboxPayloadError("JPEG payload exceeds the size limit")
    return payload


def decode_jpeg(payload: bytes, *, field_name: str = "frame_jpeg"):
    """Decode and validate one JPEG blob from the outbox."""

    raw = bytes(payload or b"")
    if not raw:
        raise OutboxCorruptionError(f"{field_name} is empty")
    if len(raw) > MAX_JPEG_BYTES:
        raise OutboxCorruptionError(f"{field_name} exceeds the size limit")
    try:
        image = cv2.imdecode(
            np.frombuffer(raw, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception as exc:
        raise OutboxCorruptionError(
            f"{field_name} cannot be decoded: {type(exc).__name__}: {exc}"
        ) from exc
    if image is None or not getattr(image, "size", 0):
        raise OutboxCorruptionError(f"{field_name} is not a valid JPEG")
    return image


def _text(value: Any, field_name: str, *, maximum: int = 512) -> str:
    result = str(value or "").strip()
    if not result:
        raise OutboxPayloadError(f"{field_name} is required")
    if len(result) > maximum:
        raise OutboxPayloadError(f"{field_name} exceeds the size limit")
    return result


@dataclass(frozen=True)
class OutboxEntry:
    retry_id: str
    state_scope: str
    camera_id: int
    camera_name: str
    result: dict
    frame_jpeg: bytes
    observed_at_utc: str
    processing_ms: float
    duplicate_seconds: float
    detector_generation: int
    detector_revision: str
    track_id: int
    identity: str
    emission_kind: str
    predecessor_id: str = ""
    plate_root: str = ""
    snapshot_root: str = ""
    save_plate: bool = True
    save_vehicle: bool = True
    crop_jpeg: bytes | None = None
    event_id: int | None = None
    ledger_key: str = ""
    attempts: int = 0
    first_failed_at_utc: str = ""
    next_attempt_at_epoch: float = 0.0
    last_error: str = ""
    seq: int = 0
    payload_sha256: str = ""
    created_at_utc: str = ""
    updated_at_utc: str = ""

    @classmethod
    def from_images(cls, *, frame, crop=None, **values) -> "OutboxEntry":
        """Build an entry from OpenCV images without retaining raw arrays."""

        return cls(
            frame_jpeg=encode_jpeg(frame, quality=95),
            crop_jpeg=(
                encode_jpeg(crop, quality=98)
                if crop is not None and getattr(crop, "size", 0)
                else None
            ),
            **values,
        )

    def decode_frame(self):
        if not self.frame_jpeg:
            return None
        return decode_jpeg(self.frame_jpeg, field_name="frame_jpeg")

    def decode_crop(self):
        if self.crop_jpeg is None:
            return None
        return decode_jpeg(self.crop_jpeg, field_name="crop_jpeg")

    @property
    def retry_key(self) -> tuple:
        return (
            int(self.detector_generation),
            str(self.detector_revision),
            int(self.track_id),
            str(self.identity),
            str(self.emission_kind),
        )


@dataclass(frozen=True)
class QuarantinedEntry:
    seq: int
    retry_id: str
    camera_id: int
    quarantine_reason: str
    quarantined_at_utc: str


@dataclass(frozen=True)
class RecoveryReport:
    entries: tuple[OutboxEntry, ...]
    quarantined: tuple[QuarantinedEntry, ...]


def _integrity_metadata(
    entry: OutboxEntry,
    result_json: str,
    *,
    schema_version: int = OUTBOX_SCHEMA_VERSION,
    include_media_policy: bool = True,
) -> bytes:
    payload = {
        "schema_version": int(schema_version),
        "retry_id": entry.retry_id,
        "state_scope": entry.state_scope,
        "camera_id": int(entry.camera_id),
        "camera_name": entry.camera_name,
        "detector_generation": int(entry.detector_generation),
        "detector_revision": entry.detector_revision,
        "track_id": int(entry.track_id),
        "identity": entry.identity,
        "emission_kind": entry.emission_kind,
        "event_id": entry.event_id,
        "ledger_key": entry.ledger_key,
        "observed_at_utc": entry.observed_at_utc,
        "processing_ms": float(entry.processing_ms),
        "duplicate_seconds": float(entry.duplicate_seconds),
        "result_json": result_json,
    }
    if include_media_policy:
        payload.update({
            "predecessor_id": entry.predecessor_id,
            "plate_root": entry.plate_root,
            "snapshot_root": entry.snapshot_root,
            "save_plate": bool(entry.save_plate),
            "save_vehicle": bool(entry.save_vehicle),
        })
    return json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_digest(
    entry: OutboxEntry,
    result_json: str,
    frame_jpeg: bytes,
    crop_jpeg: bytes | None,
    *,
    schema_version: int = OUTBOX_SCHEMA_VERSION,
    include_media_policy: bool = True,
) -> str:
    digest = hashlib.sha256()
    for chunk in (
        _integrity_metadata(
            entry,
            result_json,
            schema_version=schema_version,
            include_media_policy=include_media_policy,
        ),
        bytes(frame_jpeg),
        bytes(crop_jpeg or b""),
    ):
        digest.update(len(chunk).to_bytes(8, "big"))
        digest.update(chunk)
    return digest.hexdigest().upper()


class PersistenceOutbox:
    """SQLite-backed queue with ordered recovery and explicit quarantine."""

    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(
            self.path,
            timeout=5.0,
            check_same_thread=False,
        )
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=5000")
        con.execute("PRAGMA synchronous=FULL")
        return con

    def _initialize(self) -> None:
        with self._lock, self._connect() as con:
            mode = con.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise OutboxError("retry outbox could not enable WAL mode")
            con.execute("PRAGMA synchronous=FULL")
            con.execute("PRAGMA wal_autocheckpoint=256")
            stored_version = int(
                con.execute("PRAGMA user_version").fetchone()[0]
            )
            if stored_version > OUTBOX_SCHEMA_VERSION:
                raise OutboxError(
                    "retry outbox was created by a newer BC Vision version"
                )
            con.executescript("""
            CREATE TABLE IF NOT EXISTS retry_outbox(
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                retry_id TEXT NOT NULL UNIQUE,
                state_scope TEXT NOT NULL,
                schema_version INTEGER NOT NULL,
                camera_id INTEGER NOT NULL,
                camera_name TEXT NOT NULL,
                detector_generation INTEGER NOT NULL,
                detector_revision TEXT NOT NULL DEFAULT '',
                track_id INTEGER NOT NULL,
                identity TEXT NOT NULL DEFAULT '',
                emission_kind TEXT NOT NULL,
                predecessor_id TEXT NOT NULL DEFAULT '',
                plate_root TEXT NOT NULL DEFAULT '',
                snapshot_root TEXT NOT NULL DEFAULT '',
                save_plate INTEGER NOT NULL DEFAULT 1,
                save_vehicle INTEGER NOT NULL DEFAULT 1,
                result_json TEXT NOT NULL,
                frame_jpeg BLOB NOT NULL,
                crop_jpeg BLOB,
                event_id INTEGER,
                ledger_key TEXT NOT NULL DEFAULT '',
                observed_at_utc TEXT NOT NULL,
                processing_ms REAL NOT NULL,
                duplicate_seconds REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                first_failed_at_utc TEXT NOT NULL DEFAULT '',
                next_attempt_at_epoch REAL NOT NULL DEFAULT 0,
                last_error TEXT NOT NULL DEFAULT '',
                payload_sha256 TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending'
                    CHECK(status IN ('pending','quarantined')),
                quarantine_reason TEXT NOT NULL DEFAULT '',
                quarantined_at_utc TEXT NOT NULL DEFAULT '',
                created_at_utc TEXT NOT NULL,
                updated_at_utc TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_retry_outbox_due
                ON retry_outbox(status,next_attempt_at_epoch,seq);
            CREATE INDEX IF NOT EXISTS idx_retry_outbox_scope
                ON retry_outbox(state_scope,seq);
            CREATE INDEX IF NOT EXISTS idx_retry_outbox_camera_status
                ON retry_outbox(camera_id,status);
            """)
            con.execute("BEGIN IMMEDIATE")
            self._upgrade_schema(con)
            con.execute(f"PRAGMA user_version={OUTBOX_SCHEMA_VERSION}")

    @staticmethod
    def _upgrade_schema(con: sqlite3.Connection) -> None:
        """Upgrade v1 rows only after validating their original checksum."""

        columns = {
            str(row[1])
            for row in con.execute(
                "PRAGMA table_info(retry_outbox)"
            ).fetchall()
        }
        media_columns = {
            "predecessor_id": "TEXT NOT NULL DEFAULT ''",
            "plate_root": "TEXT NOT NULL DEFAULT ''",
            "snapshot_root": "TEXT NOT NULL DEFAULT ''",
            "save_plate": "INTEGER NOT NULL DEFAULT 1",
            "save_vehicle": "INTEGER NOT NULL DEFAULT 1",
        }
        present_media_columns = set(media_columns).intersection(columns)
        if present_media_columns and present_media_columns != set(media_columns):
            raise OutboxError(
                "retry outbox has a partial media-policy schema"
            )
        had_media_policy = present_media_columns == set(media_columns)

        valid_legacy: list[OutboxEntry] = []
        invalid_legacy: list[tuple[str, str]] = []
        rows = con.execute(
            "SELECT * FROM retry_outbox "
            "WHERE status='pending' AND schema_version=? ORDER BY seq",
            (LEGACY_OUTBOX_SCHEMA_VERSION,),
        ).fetchall()
        for row in rows:
            retry_id = str(row["retry_id"])
            try:
                if had_media_policy:
                    try:
                        restored = PersistenceOutbox._entry_from_row(
                            row,
                            expected_schema_version=(
                                LEGACY_OUTBOX_SCHEMA_VERSION
                            ),
                            include_media_policy=True,
                        )
                    except OutboxCorruptionError:
                        # An early development build added the columns before
                        # bumping the schema. Only rows whose new fields are
                        # untouched defaults may still carry the older digest.
                        if not (
                            str(row["predecessor_id"] or "") == ""
                            and str(row["plate_root"] or "") == ""
                            and str(row["snapshot_root"] or "") == ""
                            and bool(row["save_plate"])
                            and bool(row["save_vehicle"])
                        ):
                            raise
                        restored = PersistenceOutbox._entry_from_row(
                            row,
                            expected_schema_version=(
                                LEGACY_OUTBOX_SCHEMA_VERSION
                            ),
                            include_media_policy=False,
                        )
                else:
                    restored = PersistenceOutbox._entry_from_row(
                        row,
                        expected_schema_version=(
                            LEGACY_OUTBOX_SCHEMA_VERSION
                        ),
                        include_media_policy=False,
                    )
                valid_legacy.append(restored)
            except OutboxCorruptionError as exc:
                invalid_legacy.append(
                    (retry_id, f"legacy upgrade rejected: {exc}")
                )

        now = _utc_now()
        for retry_id, reason in invalid_legacy:
            con.execute(
                "UPDATE retry_outbox SET status='quarantined',"
                "quarantine_reason=?,quarantined_at_utc=?,updated_at_utc=? "
                "WHERE retry_id=? AND schema_version=?",
                (
                    reason[:MAX_ERROR_LENGTH],
                    now,
                    now,
                    retry_id,
                    LEGACY_OUTBOX_SCHEMA_VERSION,
                ),
            )

        for name, declaration in media_columns.items():
            if name not in columns:
                con.execute(
                    f"ALTER TABLE retry_outbox ADD COLUMN {name} "
                    f"{declaration}"
                )

        for restored in valid_legacy:
            result_json = _result_json(restored.result)
            upgraded_digest = _payload_digest(
                restored,
                result_json,
                restored.frame_jpeg,
                restored.crop_jpeg,
            )
            con.execute(
                "UPDATE retry_outbox SET schema_version=?,"
                "payload_sha256=?,updated_at_utc=? "
                "WHERE retry_id=? AND status='pending' AND schema_version=?",
                (
                    OUTBOX_SCHEMA_VERSION,
                    upgraded_digest,
                    now,
                    restored.retry_id,
                    LEGACY_OUTBOX_SCHEMA_VERSION,
                ),
            )

    @staticmethod
    def _validated_entry(entry: OutboxEntry) -> tuple[OutboxEntry, str]:
        if not isinstance(entry, OutboxEntry):
            raise TypeError("entry must be an OutboxEntry")
        retry_id = _text(entry.retry_id, "retry_id", maximum=128)
        state_scope = _text(
            entry.state_scope,
            "state_scope",
            maximum=128,
        )
        camera_name = _text(
            entry.camera_name,
            "camera_name",
            maximum=512,
        )
        emission_kind = _text(
            entry.emission_kind,
            "emission_kind",
            maximum=64,
        )
        observed_at_utc = _normalized_utc(entry.observed_at_utc)
        processing_ms = _finite_float(entry.processing_ms, "processing_ms")
        duplicate_seconds = _finite_float(
            entry.duplicate_seconds,
            "duplicate_seconds",
        )
        next_attempt = _finite_float(
            entry.next_attempt_at_epoch,
            "next_attempt_at_epoch",
        )
        if processing_ms < 0 or duplicate_seconds < 0 or next_attempt < 0:
            raise OutboxPayloadError("numeric retry values cannot be negative")
        frame_jpeg = bytes(entry.frame_jpeg or b"")
        crop_jpeg = (
            bytes(entry.crop_jpeg)
            if entry.crop_jpeg is not None
            else None
        )
        if frame_jpeg:
            decode_jpeg(frame_jpeg, field_name="frame_jpeg")
        if crop_jpeg is not None:
            decode_jpeg(crop_jpeg, field_name="crop_jpeg")
        result_json = _result_json(entry.result)
        validated = replace(
            entry,
            retry_id=retry_id,
            state_scope=state_scope,
            camera_id=int(entry.camera_id),
            camera_name=camera_name,
            result=json.loads(result_json),
            frame_jpeg=frame_jpeg,
            crop_jpeg=crop_jpeg,
            event_id=(
                int(entry.event_id) if entry.event_id is not None else None
            ),
            ledger_key=str(entry.ledger_key or "")[:512],
            observed_at_utc=observed_at_utc,
            processing_ms=processing_ms,
            duplicate_seconds=duplicate_seconds,
            detector_generation=int(entry.detector_generation),
            detector_revision=str(entry.detector_revision or "")[:512],
            track_id=int(entry.track_id),
            identity=str(entry.identity or "")[:512],
            emission_kind=emission_kind,
            predecessor_id=str(entry.predecessor_id or "")[:128],
            plate_root=str(entry.plate_root or "")[:4096],
            snapshot_root=str(entry.snapshot_root or "")[:4096],
            save_plate=bool(entry.save_plate),
            save_vehicle=bool(entry.save_vehicle),
            attempts=max(0, int(entry.attempts)),
            first_failed_at_utc=str(entry.first_failed_at_utc or "")[:64],
            next_attempt_at_epoch=next_attempt,
            last_error=str(entry.last_error or "")[:MAX_ERROR_LENGTH],
        )
        return validated, result_json

    def upsert(self, entry: OutboxEntry) -> int:
        """Durably insert or refresh one emission and return its stable seq."""

        validated, result_json = self._validated_entry(entry)
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            previous = con.execute(
                "SELECT seq,event_id,attempts,first_failed_at_utc,"
                "next_attempt_at_epoch,last_error,created_at_utc "
                "FROM retry_outbox WHERE retry_id=?",
                (validated.retry_id,),
            ).fetchone()
            now = _utc_now()
            if previous:
                event_id = (
                    int(previous["event_id"])
                    if previous["event_id"] is not None
                    else validated.event_id
                )
                stored = replace(validated, event_id=event_id)
                digest = _payload_digest(
                    stored,
                    result_json,
                    stored.frame_jpeg,
                    stored.crop_jpeg,
                )
                con.execute(
                    "UPDATE retry_outbox SET state_scope=?,schema_version=?,"
                    "camera_id=?,camera_name=?,detector_generation=?,"
                    "detector_revision=?,track_id=?,identity=?,emission_kind=?,"
                    "predecessor_id=?,plate_root=?,snapshot_root=?,"
                    "save_plate=?,save_vehicle=?,"
                    "result_json=?,frame_jpeg=?,crop_jpeg=?,event_id=?,"
                    "ledger_key=?,observed_at_utc=?,processing_ms=?,"
                    "duplicate_seconds=?,payload_sha256=?,status='pending',"
                    "quarantine_reason='',quarantined_at_utc='',updated_at_utc=? "
                    "WHERE retry_id=?",
                    (
                        stored.state_scope,
                        OUTBOX_SCHEMA_VERSION,
                        stored.camera_id,
                        stored.camera_name,
                        stored.detector_generation,
                        stored.detector_revision,
                        stored.track_id,
                        stored.identity,
                        stored.emission_kind,
                        stored.predecessor_id,
                        stored.plate_root,
                        stored.snapshot_root,
                        int(stored.save_plate),
                        int(stored.save_vehicle),
                        result_json,
                        sqlite3.Binary(stored.frame_jpeg),
                        (
                            sqlite3.Binary(stored.crop_jpeg)
                            if stored.crop_jpeg is not None
                            else None
                        ),
                        stored.event_id,
                        stored.ledger_key,
                        stored.observed_at_utc,
                        stored.processing_ms,
                        stored.duplicate_seconds,
                        digest,
                        now,
                        stored.retry_id,
                    ),
                )
                return int(previous["seq"])

            digest = _payload_digest(
                validated,
                result_json,
                validated.frame_jpeg,
                validated.crop_jpeg,
            )
            cursor = con.execute(
                "INSERT INTO retry_outbox("
                "retry_id,state_scope,schema_version,camera_id,camera_name,"
                "detector_generation,detector_revision,track_id,identity,"
                "emission_kind,predecessor_id,plate_root,snapshot_root,"
                "save_plate,save_vehicle,result_json,frame_jpeg,crop_jpeg,event_id,"
                "ledger_key,observed_at_utc,processing_ms,duplicate_seconds,"
                "attempts,first_failed_at_utc,next_attempt_at_epoch,last_error,"
                "payload_sha256,created_at_utc,updated_at_utc"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    validated.retry_id,
                    validated.state_scope,
                    OUTBOX_SCHEMA_VERSION,
                    validated.camera_id,
                    validated.camera_name,
                    validated.detector_generation,
                    validated.detector_revision,
                    validated.track_id,
                    validated.identity,
                    validated.emission_kind,
                    validated.predecessor_id,
                    validated.plate_root,
                    validated.snapshot_root,
                    int(validated.save_plate),
                    int(validated.save_vehicle),
                    result_json,
                    sqlite3.Binary(validated.frame_jpeg),
                    (
                        sqlite3.Binary(validated.crop_jpeg)
                        if validated.crop_jpeg is not None
                        else None
                    ),
                    validated.event_id,
                    validated.ledger_key,
                    validated.observed_at_utc,
                    validated.processing_ms,
                    validated.duplicate_seconds,
                    validated.attempts,
                    validated.first_failed_at_utc,
                    validated.next_attempt_at_epoch,
                    validated.last_error,
                    digest,
                    now,
                    now,
                ),
            )
            return int(cursor.lastrowid)

    @staticmethod
    def _entry_from_row(
        row: sqlite3.Row,
        *,
        expected_schema_version: int = OUTBOX_SCHEMA_VERSION,
        include_media_policy: bool = True,
    ) -> OutboxEntry:
        try:
            if int(row["schema_version"]) != int(expected_schema_version):
                raise OutboxCorruptionError(
                    "unsupported payload schema version "
                    f"{row['schema_version']}"
                )
            result_json = str(row["result_json"])
            if len(result_json.encode("utf-8")) > MAX_RESULT_JSON_BYTES:
                raise OutboxCorruptionError("result JSON exceeds size limit")
            result = json.loads(result_json)
            if not isinstance(result, dict):
                raise OutboxCorruptionError("result JSON is not an object")
            unknown = set(result).difference(RESULT_FIELD_ALLOWLIST)
            if unknown:
                raise OutboxCorruptionError(
                    "result JSON contains fields outside the allow-list"
                )
            sanitized = sanitize_result(result)
            canonical = json.dumps(
                sanitized,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != result_json:
                raise OutboxCorruptionError("result JSON is not canonical")
            frame_jpeg = bytes(row["frame_jpeg"] or b"")
            crop_jpeg = (
                bytes(row["crop_jpeg"])
                if row["crop_jpeg"] is not None
                else None
            )
            entry = OutboxEntry(
                retry_id=str(row["retry_id"]),
                state_scope=str(row["state_scope"]),
                camera_id=int(row["camera_id"]),
                camera_name=str(row["camera_name"]),
                result=sanitized,
                frame_jpeg=frame_jpeg,
                crop_jpeg=crop_jpeg,
                event_id=(
                    int(row["event_id"])
                    if row["event_id"] is not None
                    else None
                ),
                ledger_key=str(row["ledger_key"]),
                observed_at_utc=str(row["observed_at_utc"]),
                processing_ms=float(row["processing_ms"]),
                duplicate_seconds=float(row["duplicate_seconds"]),
                detector_generation=int(row["detector_generation"]),
                detector_revision=str(row["detector_revision"]),
                track_id=int(row["track_id"]),
                identity=str(row["identity"]),
                emission_kind=str(row["emission_kind"]),
                predecessor_id=(
                    str(row["predecessor_id"])
                    if include_media_policy else ""
                ),
                plate_root=(
                    str(row["plate_root"])
                    if include_media_policy else ""
                ),
                snapshot_root=(
                    str(row["snapshot_root"])
                    if include_media_policy else ""
                ),
                save_plate=(
                    bool(row["save_plate"])
                    if include_media_policy else True
                ),
                save_vehicle=(
                    bool(row["save_vehicle"])
                    if include_media_policy else True
                ),
                attempts=int(row["attempts"]),
                first_failed_at_utc=str(row["first_failed_at_utc"]),
                next_attempt_at_epoch=float(row["next_attempt_at_epoch"]),
                last_error=str(row["last_error"]),
                seq=int(row["seq"]),
                payload_sha256=str(row["payload_sha256"]),
                created_at_utc=str(row["created_at_utc"]),
                updated_at_utc=str(row["updated_at_utc"]),
            )
            expected = _payload_digest(
                entry,
                result_json,
                frame_jpeg,
                crop_jpeg,
                schema_version=expected_schema_version,
                include_media_policy=include_media_policy,
            )
            if expected != entry.payload_sha256.upper():
                raise OutboxCorruptionError("payload checksum mismatch")
            if frame_jpeg:
                decode_jpeg(frame_jpeg, field_name="frame_jpeg")
            if crop_jpeg is not None:
                decode_jpeg(crop_jpeg, field_name="crop_jpeg")
            return entry
        except OutboxCorruptionError:
            raise
        except Exception as exc:
            raise OutboxCorruptionError(
                f"payload validation failed: {type(exc).__name__}: {exc}"
            ) from exc

    def load(
        self,
        *,
        due_at_epoch: float | None = None,
        limit: int | None = None,
        after_seq: int = 0,
    ) -> list[OutboxEntry]:
        """Load verified pending rows in seq order, quarantining bad rows."""

        if limit is not None and int(limit) <= 0:
            return []
        due = (
            _finite_float(due_at_epoch, "due_at_epoch")
            if due_at_epoch is not None
            else None
        )
        entries: list[OutboxEntry] = []
        last_seq = max(0, int(after_seq))
        batch_size = 64
        while limit is None or len(entries) < int(limit):
            clauses = ["status='pending'", "seq>?"]
            parameters: list[Any] = [last_seq]
            if due is not None:
                clauses.append("next_attempt_at_epoch<=?")
                parameters.append(due)
            sql = (
                "SELECT * FROM retry_outbox WHERE "
                + " AND ".join(clauses)
                + " ORDER BY seq LIMIT ?"
            )
            parameters.append(batch_size)
            with self._lock, self._connect() as con:
                rows = con.execute(sql, tuple(parameters)).fetchall()
            if not rows:
                break
            for row in rows:
                last_seq = int(row["seq"])
                try:
                    entries.append(self._entry_from_row(row))
                except OutboxCorruptionError as exc:
                    self.quarantine(
                        str(row["retry_id"]),
                        f"{type(exc).__name__}: {exc}",
                    )
                if limit is not None and len(entries) >= int(limit):
                    break
        return entries

    def recover(
        self,
        *,
        due_at_epoch: float | None = None,
        limit: int | None = None,
    ) -> RecoveryReport:
        """Recover valid rows and report every retained quarantine row."""

        entries = self.load(due_at_epoch=due_at_epoch, limit=limit)
        return RecoveryReport(
            entries=tuple(entries),
            quarantined=tuple(self.quarantined()),
        )

    def pending_stats(self, camera_id: int | None = None) -> tuple[int, int]:
        """Return pending row count and encoded-image bytes without loading BLOBs."""

        clauses = ["status='pending'"]
        parameters: list[Any] = []
        if camera_id is not None:
            clauses.append("camera_id=?")
            parameters.append(int(camera_id))
        with self._lock, self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*),COALESCE(SUM("
                "length(frame_jpeg)+COALESCE(length(crop_jpeg),0)"
                "),0) FROM retry_outbox WHERE " + " AND ".join(clauses),
                tuple(parameters),
            ).fetchone()
        return int(row[0]), int(row[1])

    def delete(self, retry_id: str) -> bool:
        """Delete one acknowledged row; return whether it existed."""

        value = _text(retry_id, "retry_id", maximum=128)
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cursor = con.execute(
                "DELETE FROM retry_outbox WHERE retry_id=?",
                (value,),
            )
            return cursor.rowcount == 1

    def update_failure(
        self,
        retry_id: str,
        error: str,
        *,
        next_attempt_at_epoch: float,
        failed_at_utc: str | None = None,
    ) -> int | None:
        """Record one failed attempt and return the new attempt count."""

        value = _text(retry_id, "retry_id", maximum=128)
        failure = str(error or "unknown persistence error")[:MAX_ERROR_LENGTH]
        failed_at = (
            _normalized_utc(failed_at_utc)
            if failed_at_utc is not None
            else _utc_now()
        )
        next_attempt = _finite_float(
            next_attempt_at_epoch,
            "next_attempt_at_epoch",
        )
        if next_attempt < 0:
            raise OutboxPayloadError("next_attempt_at_epoch cannot be negative")
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cursor = con.execute(
                "UPDATE retry_outbox SET attempts=attempts+1,"
                "first_failed_at_utc=CASE "
                "WHEN first_failed_at_utc='' THEN ? ELSE first_failed_at_utc END,"
                "next_attempt_at_epoch=?,last_error=?,updated_at_utc=? "
                "WHERE retry_id=? AND status='pending'",
                (
                    failed_at,
                    next_attempt,
                    failure,
                    _utc_now(),
                    value,
                ),
            )
            if cursor.rowcount != 1:
                return None
            row = con.execute(
                "SELECT attempts FROM retry_outbox WHERE retry_id=?",
                (value,),
            ).fetchone()
            return int(row["attempts"])

    def quarantine(self, retry_id: str, reason: str) -> bool:
        """Retain but isolate a payload that cannot be safely replayed."""

        value = _text(retry_id, "retry_id", maximum=128)
        explanation = str(reason or "invalid payload")[:MAX_ERROR_LENGTH]
        now = _utc_now()
        with self._lock, self._connect() as con:
            con.execute("BEGIN IMMEDIATE")
            cursor = con.execute(
                "UPDATE retry_outbox SET status='quarantined',"
                "quarantine_reason=?,quarantined_at_utc=?,updated_at_utc=? "
                "WHERE retry_id=?",
                (explanation, now, now, value),
            )
            return cursor.rowcount == 1

    def quarantined(self) -> list[QuarantinedEntry]:
        with self._lock, self._connect() as con:
            rows = con.execute(
                "SELECT seq,retry_id,camera_id,quarantine_reason,"
                "quarantined_at_utc FROM retry_outbox "
                "WHERE status='quarantined' ORDER BY seq"
            ).fetchall()
        return [
            QuarantinedEntry(
                seq=int(row["seq"]),
                retry_id=str(row["retry_id"]),
                camera_id=int(row["camera_id"]),
                quarantine_reason=str(row["quarantine_reason"]),
                quarantined_at_utc=str(row["quarantined_at_utc"]),
            )
            for row in rows
        ]

    def pending_count(self) -> int:
        with self._lock, self._connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM retry_outbox WHERE status='pending'"
            ).fetchone()[0])

    def quarantined_count(self) -> int:
        with self._lock, self._connect() as con:
            return int(con.execute(
                "SELECT COUNT(*) FROM retry_outbox "
                "WHERE status='quarantined'"
            ).fetchone()[0])

    def backup(self, destination) -> Path:
        """Create an atomic SQLite snapshot, including pending WAL rows."""

        target = Path(destination).expanduser().resolve()
        if target == self.path:
            raise ValueError("outbox backup must use a different path")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(
            f".{target.name}.{uuid4().hex}.tmp"
        )
        try:
            with self._lock:
                # sqlite3.Connection's context manager commits/rolls back but
                # does not close the file handle.  Close both databases before
                # os.replace so backup works on Windows, where an open SQLite
                # handle prevents replacing the temporary file.
                with closing(self._connect()) as source, closing(
                    sqlite3.connect(temporary)
                ) as snapshot:
                    source.backup(snapshot)
                    checked = snapshot.execute(
                        "PRAGMA quick_check"
                    ).fetchone()
                    if not checked or checked[0] != "ok":
                        raise OutboxError(
                            "retry outbox backup integrity failed"
                        )
            with temporary.open("rb") as snapshot_file:
                os.fsync(snapshot_file.fileno())
            temporary.replace(target)
            from app.storage_policy import fsync_parent_directory

            fsync_parent_directory(target)
            return target
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
