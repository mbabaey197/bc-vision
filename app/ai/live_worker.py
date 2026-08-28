"""Asynchronous live-camera ANPR worker.

Streaming threads submit the newest frame without blocking. Per-camera workers
apply ROI, multi-frame consensus, duplicate suppression, and persist events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import os
import threading
import time
from uuid import uuid4

import cv2
import numpy as np

from app.config import DATA_DIR, PLATE_DIR, SNAPSHOT_DIR
from app.cpu_budget import parallel_camera_limit, threads_per_camera
from app.media_storage import (
    finalize_pending_media,
    save_event_images,
    settle_pending_media,
)

from .activity import FrameActivityAnalyzer
from .event_dedup import (
    PlateVisitLedger,
    candidate_plate_key,
    fragmented_review_can_migrate,
    review_identity_can_migrate,
    strict_plate_key,
)
from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    bbox_iou,
    process_frame,
)
from .plate_rules import normalize_plate, split_iran_plate
from .persistence_outbox import OutboxEntry, PersistenceOutbox
from .feedback import apply_learned_correction
from .review_policy import (
    auto_confirm_guess,
    tag_assisted_candidate,
)


ENGINE_V3_INFERENCE_KEY = "engine-v3-shared"
PERSISTENCE_RETRY_HIGH_COUNT = 32
PERSISTENCE_RETRY_LOW_COUNT = 16
PERSISTENCE_RETRY_HIGH_BYTES = 64 * 1024 * 1024
PERSISTENCE_RETRY_LOW_BYTES = 32 * 1024 * 1024


def camera_confidence_result(
    result: dict,
    min_confidence: float,
) -> dict | None:
    """Downgrade a gated identity to review evidence instead of losing it."""

    recognized = bool(
        result.get("valid") or result.get("auto_confirmed")
    )
    if (
        float(result.get("confidence", 0.0)) >= float(min_confidence)
        or not recognized
    ):
        return result
    plate_norm = normalize_plate(
        result.get("plate_norm") or result.get("plate")
    )
    if not split_iran_plate(plate_norm):
        return None
    review = dict(result)
    review.update({
        "plate_norm": "",
        "valid": False,
        "best_effort": True,
        "needs_review": True,
        "read_status": "experimental-guess",
        "experimental": True,
        "auto_confirmed": False,
        "confirmation_source": "ai-suggestion",
        "auto_confirmation_blocked": "below-camera-confidence",
        "raw_guess_text": str(result.get("plate") or ""),
        "raw_guess_norm": plate_norm,
        "raw_guess_confidence": float(
            result.get(
                "ocr_confidence",
                result.get("confidence", 0.0),
            )
        ),
        "raw_guess_engine": str(result.get("ocr_engine", "")),
        "raw_guess_reason": "below-camera-confidence",
    })
    return review


def operator_assisted_rows(primary: list, shadow: list) -> list:
    """Prefer a complete Shadow guess over an overlapping unreadable row.

    Strict baseline reads retain priority. Candidate output stays tagged as
    experimental until the tracker has enough temporal evidence to emit one
    automatically confirmed, operator-reviewable event.
    """

    selected = [dict(row) for row in primary]
    for raw_candidate in shadow:
        candidate = tag_assisted_candidate(raw_candidate)
        if candidate is None or not candidate.get("bbox"):
            continue
        overlaps = [
            (bbox_iou(row.get("bbox"), candidate["bbox"]), index)
            for index, row in enumerate(selected)
            if row.get("bbox")
        ]
        overlap, index = max(overlaps, default=(0.0, -1))
        if overlap >= 0.28:
            baseline = selected[index]
            if baseline.get("valid") and not baseline.get("needs_review"):
                continue
            selected[index] = candidate
        else:
            selected.append(candidate)
    return selected


@dataclass(eq=False)
class _PersistenceRetry:
    retry_key: tuple
    persistence_id: str
    state_scope: str
    predecessor_id: str
    camera_id: int
    camera_name: str
    result: dict
    frame: object
    event_id: int | None
    ledger_key: str
    observed_at: float
    observed_at_utc: str
    processing_ms: float
    duplicate_seconds: float
    detector_generation: int
    detector_revision: str = ""
    plate_root: str = ""
    snapshot_root: str = ""
    save_plate: bool = True
    save_vehicle: bool = True
    attempts: int = 0
    first_failed_at: float = 0.0
    first_failed_at_utc: str = ""
    next_attempt_at_epoch: float = 0.0
    last_error: str = ""
    durably_spooled: bool = False
    committed_as_insert: bool = False
    primary_committed: bool = False


@dataclass(eq=False)
class _RetryLineage:
    """A persisted retry awaiting a spatially continuous correction."""

    event_id: int
    result: dict
    observed_at: float
    detector_generation: int
    detector_revision: str = ""


@dataclass
class _CameraState:
    metrics_started_at: float = field(default_factory=time.monotonic)
    frame_counter: int = 0
    busy: bool = False
    retired: bool = False
    pending: tuple | None = None
    config: dict | None = None
    config_loaded_at: float = 0.0
    tracker: PlateConsensusTracker = field(
        default_factory=lambda: PlateConsensusTracker(
            min_votes=2,
            max_age_seconds=2.2,
            emit_cooldown=5.0,
            emit_unreadable=True,
        )
    )
    seen: dict[str, float] = field(default_factory=dict)
    visits: PlateVisitLedger = field(default_factory=PlateVisitLedger)
    track_event_ids: dict[int, int] = field(default_factory=dict)
    persistence_retry: dict[tuple, _PersistenceRetry] = field(
        default_factory=dict
    )
    retry_lineages: list[_RetryLineage] = field(default_factory=list)
    last_error: str = ""
    processing_errors: int = 0
    persistence_errors: int = 0
    persistence_backpressure: bool = False
    persistence_backpressure_frames: int = 0
    last_processing_error: str = ""
    last_event_at: float = 0.0
    processed_frames: int = 0
    inference_calls: int = 0
    inference_seconds: float = 0.0
    last_inference_ms: float = 0.0
    coalesced_frames: int = 0
    detected_candidates: int = 0
    emitted_events: int = 0
    whole_plate_ocr_attempts: int = 0
    ocr_agreements: int = 0
    ocr_disagreements: int = 0
    crnn_selected: int = 0
    character_reader_selected: int = 0
    detector_model_revision: str = ""
    last_processed_at: float = 0.0
    last_processing_ms: float = 0.0
    processing_seconds_ema: float = 0.0
    no_plate_streak: int = 0
    next_inference_at: float = 0.0
    latest_detections: list = field(default_factory=list)
    latest_detections_at: float = 0.0
    latest_detection_frame: object | None = None
    detection_revision: int = 0
    last_submitted_at: float = 0.0
    burst_frames_remaining: int = 0
    plate_visible: bool = False
    shadow_frames: int = 0
    shadow_candidates: int = 0
    shadow_errors: int = 0
    activity: FrameActivityAnalyzer = field(
        default_factory=FrameActivityAnalyzer
    )
    motion_score: float = 0.0
    motion_wakeups: int = 0
    overlay_mask_pixels: int = 0
    static_overlay_hits: dict = field(default_factory=dict)
    static_overlay_blocked_until: dict = field(default_factory=dict)
    model_switch_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )

    def __post_init__(self):
        # Keep the legacy ``seen`` view used by diagnostics/tests backed by
        # the visit ledger's canonical timestamps.
        self.visits.seen = self.seen


class LiveANPRWorker:
    def __init__(
        self,
        max_workers=None,
        *,
        background_retry=False,
        retry_outbox_path=None,
        _defer_persistence_start=False,
    ):
        self._states: dict[int, _CameraState] = {}
        automatic_capacity = parallel_camera_limit()
        self._worker_capacity = (
            automatic_capacity
            if max_workers is None
            else max(1, min(automatic_capacity, int(max_workers)))
        )
        self._executor = ThreadPoolExecutor(
            max_workers=self._worker_capacity,
            thread_name_prefix="bc-anpr",
        )
        self._lock = threading.RLock()
        self._event_commit_locks: dict[int, threading.RLock] = {}
        self._detached_retry_states: list[
            tuple[int, _CameraState]
        ] = []
        self._stopped = False
        self._model_state = {}
        self._model_state_at = 0.0
        self._model_state_variant = ""
        self._detector_generation = 0
        self._shadow_enabled_cache = False
        self._shadow_setting_at = -1e12
        self._state_scope = uuid4().hex
        self._outbox_required = retry_outbox_path is not None
        self._outbox = None
        self._outbox_error = ""
        self._outbox_quarantined = 0
        self._outbox_recovery_cursor = 0
        self._loaded_outbox_ids: set[str] = set()
        self._outbox_state_lock = threading.RLock()
        self._retry_outbox_path = retry_outbox_path
        self._background_retry = bool(background_retry)
        self._persistence_started = False
        self._retry_stop = threading.Event()
        self._retry_wakeup = threading.Event()
        self._retry_thread = None
        if not _defer_persistence_start:
            self._start_persistence_lifecycle()

    def _start_persistence_lifecycle(self) -> None:
        """Open the durable queue and start its pump at most once."""

        if self._persistence_started:
            return
        self._persistence_started = True
        if self._retry_outbox_path is not None:
            try:
                self._outbox = PersistenceOutbox(self._retry_outbox_path)
                self._restore_outbox_entries()
            except Exception as exc:
                self._outbox_error = f"{type(exc).__name__}: {exc}"
                self._outbox = None
        if self._background_retry:
            self._retry_thread = threading.Thread(
                target=self._retry_pump,
                daemon=True,
                name="bc-anpr-persistence-retry",
            )
            self._retry_thread.start()

    @staticmethod
    def _outbox_observation_timestamp(value: str) -> float:
        try:
            parsed = datetime.fromisoformat(
                str(value).replace("Z", "+00:00")
            )
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except (TypeError, ValueError):
            return 0.0

    def _restore_outbox_entries(self) -> None:
        if self._outbox is None:
            return
        self._hydrate_outbox_entries()
        self._outbox_quarantined = self._outbox.quarantined_count()

    @staticmethod
    def _retry_from_outbox(stored: OutboxEntry) -> _PersistenceRetry:
        return _PersistenceRetry(
            retry_key=tuple(stored.retry_key),
            persistence_id=stored.retry_id,
            state_scope=stored.state_scope,
            predecessor_id=stored.predecessor_id,
            camera_id=int(stored.camera_id),
            camera_name=stored.camera_name,
            result=dict(stored.result),
            frame=bytes(stored.frame_jpeg),
            event_id=stored.event_id,
            ledger_key=stored.ledger_key,
            observed_at=LiveANPRWorker._outbox_observation_timestamp(
                stored.observed_at_utc
            ),
            observed_at_utc=stored.observed_at_utc,
            processing_ms=float(stored.processing_ms),
            duplicate_seconds=float(stored.duplicate_seconds),
            detector_generation=int(stored.detector_generation),
            detector_revision=stored.detector_revision,
            plate_root=stored.plate_root,
            snapshot_root=stored.snapshot_root,
            save_plate=bool(stored.save_plate),
            save_vehicle=bool(stored.save_vehicle),
            attempts=int(stored.attempts),
            first_failed_at_utc=stored.first_failed_at_utc,
            next_attempt_at_epoch=float(stored.next_attempt_at_epoch),
            last_error=stored.last_error,
            durably_spooled=True,
        )

    def _hydrate_outbox_entries(self) -> int:
        """Page durable rows into bounded memory without loading every JPEG."""

        if self._outbox is None:
            return 0
        with self._lock:
            all_states = list(self._states.values()) + [
                state for _camera_id, state in self._detached_retry_states
            ]
        unique_states = list({id(state): state for state in all_states}.values())
        loaded_count = sum(
            len(self._retry_entries(state)) for state in unique_states
        )
        loaded_bytes = sum(
            self._retry_memory_bytes(state) for state in unique_states
        )

        # A full page of permanently failing rows must not monopolize recovery
        # forever. Once a detached row has failed at least once, its durable
        # outbox copy is sufficient ownership while it waits for a later turn.
        # Rotate one leaf out of memory, preserving predecessor ordering, so an
        # independent row (including another camera) can make progress while
        # retaining the hard memory bound.
        if (
            loaded_count >= PERSISTENCE_RETRY_LOW_COUNT
            or loaded_bytes >= PERSISTENCE_RETRY_LOW_BYTES
        ):
            with self._lock:
                detached_ids = {
                    id(state)
                    for _camera_id, state in self._detached_retry_states
                }
            evicted = None
            candidates = []
            for state in unique_states:
                if id(state) not in detached_ids and not state.retired:
                    continue
                with state.model_switch_lock:
                    entries = tuple(state.persistence_retry.items())
                    predecessor_ids = {
                        entry.predecessor_id
                        for _retry_key, entry in entries
                        if entry.predecessor_id
                    }
                    for retry_key, entry in entries:
                        if (
                            entry.durably_spooled
                            and entry.attempts > 0
                            and entry.persistence_id not in predecessor_ids
                        ):
                            candidates.append((state, retry_key, entry))
            if candidates:
                state, retry_key, entry = max(
                    candidates,
                    key=lambda item: (
                        int(item[2].attempts),
                        float(item[2].next_attempt_at_epoch),
                        float(item[2].observed_at),
                    ),
                )
                with state.model_switch_lock:
                    if state.persistence_retry.get(retry_key) is entry:
                        state.persistence_retry.pop(retry_key, None)
                        evicted = entry
                if evicted is not None:
                    with self._outbox_state_lock:
                        self._loaded_outbox_ids.discard(
                            evicted.persistence_id
                        )
                    loaded_count -= 1
                    loaded_bytes = max(
                        0,
                        loaded_bytes
                        - self._retry_entry_memory_bytes(evicted),
                    )
            if evicted is None:
                self._outbox_quarantined = (
                    self._outbox.quarantined_count()
                )
                return 0

        existing_groups = {}
        with self._lock:
            detached_snapshot = list(self._detached_retry_states)
        for camera_id, state in detached_snapshot:
            scopes = {
                entry.state_scope for entry in self._retry_entries(state)
            }
            for scope in scopes:
                existing_groups[(scope, int(camera_id))] = state

        added = 0
        new_groups: dict[tuple[str, int], _CameraState] = {}
        wrapped = False
        while (
            loaded_count < PERSISTENCE_RETRY_LOW_COUNT
            and loaded_bytes < PERSISTENCE_RETRY_LOW_BYTES
        ):
            with self._outbox_state_lock:
                rows = self._outbox.load(
                    limit=1,
                    after_seq=self._outbox_recovery_cursor,
                )
                if not rows:
                    if self._outbox_recovery_cursor and not wrapped:
                        self._outbox_recovery_cursor = 0
                        wrapped = True
                        continue
                    break
                stored = rows[0]
                self._outbox_recovery_cursor = max(
                    self._outbox_recovery_cursor,
                    int(stored.seq),
                )
                if stored.retry_id in self._loaded_outbox_ids:
                    continue
                self._loaded_outbox_ids.add(stored.retry_id)
            entry = self._retry_from_outbox(stored)
            group_key = (entry.state_scope, int(entry.camera_id))
            state = existing_groups.get(group_key) or new_groups.get(group_key)
            if state is None:
                state = _CameraState(retired=True)
                new_groups[group_key] = state
            with state.model_switch_lock:
                retry_key = entry.retry_key
                if retry_key in state.persistence_retry:
                    retry_key = retry_key + (entry.persistence_id,)
                    entry.retry_key = retry_key
                state.persistence_retry[retry_key] = entry
            loaded_count += 1
            loaded_bytes += self._retry_entry_memory_bytes(entry)
            added += 1
        if new_groups:
            with self._lock:
                self._detached_retry_states.extend(
                    (camera_id, state)
                    for (_scope, camera_id), state in new_groups.items()
                )
        self._outbox_quarantined = self._outbox.quarantined_count()
        return added

    @staticmethod
    def _outbox_entry(entry: _PersistenceRetry) -> OutboxEntry:
        return OutboxEntry(
            retry_id=entry.persistence_id,
            state_scope=entry.state_scope,
            predecessor_id=entry.predecessor_id,
            camera_id=int(entry.camera_id),
            camera_name=entry.camera_name,
            result=dict(entry.result),
            frame_jpeg=bytes(entry.frame or b""),
            event_id=entry.event_id,
            ledger_key=entry.ledger_key,
            observed_at_utc=entry.observed_at_utc,
            processing_ms=float(entry.processing_ms),
            duplicate_seconds=float(entry.duplicate_seconds),
            detector_generation=int(entry.detector_generation),
            detector_revision=entry.detector_revision,
            track_id=int(entry.result.get("track_id") or 0),
            identity=normalize_plate(
                entry.result.get("plate_norm")
                or entry.result.get("raw_guess_norm")
                or entry.result.get("plate")
            ),
            emission_kind=LiveANPRWorker._persistence_kind(entry.result),
            plate_root=entry.plate_root,
            snapshot_root=entry.snapshot_root,
            save_plate=bool(entry.save_plate),
            save_vehicle=bool(entry.save_vehicle),
            attempts=int(entry.attempts),
            first_failed_at_utc=entry.first_failed_at_utc,
            next_attempt_at_epoch=float(entry.next_attempt_at_epoch),
            last_error=entry.last_error,
        )

    def _spool_retry(self, entry: _PersistenceRetry) -> None:
        if self._outbox is None:
            if self._outbox_required:
                raise RuntimeError(
                    self._outbox_error or "retry outbox is unavailable"
                )
            entry.durably_spooled = True
            return
        self._outbox.upsert(self._outbox_entry(entry))
        with self._outbox_state_lock:
            self._loaded_outbox_ids.add(entry.persistence_id)
        self._outbox_error = ""
        entry.durably_spooled = True

    def _selected_detector_variant(self) -> str:
        from .model_manager import normalize_detector_variant

        with self._lock:
            return normalize_detector_variant(
                self._setting("anpr_detector_model", "yolo11n")
            )

    @staticmethod
    def _truthy_setting(value) -> bool:
        return str(value or "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
            "enabled",
        }

    def _engine_v2_shadow_enabled(self, now=None) -> bool:
        override = os.environ.get("BCVISION_ENGINE_V2_SHADOW")
        if override is not None:
            return self._truthy_setting(override)
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            if timestamp - self._shadow_setting_at >= 2.0:
                self._shadow_enabled_cache = self._truthy_setting(
                    self._setting("anpr_engine_v2_shadow", "0")
                )
                self._shadow_setting_at = timestamp
            return self._shadow_enabled_cache

    def configure_engine_v2_shadow(self, enabled: bool) -> None:
        with self._lock:
            self._shadow_enabled_cache = bool(enabled)
            self._shadow_setting_at = time.monotonic()
            detector_variant = self._selected_detector_variant()
        from app.engine_v2.live_shadow import (
            configure_live_event_callback,
            configure_live_shadow,
        )

        # V2 is observation-only until a measured production promotion.
        # Never attach its event callback to the durable baseline path.
        configure_live_event_callback(None)
        configure_live_shadow(bool(enabled), detector_variant)

    def _submit_engine_v2_shadow(
        self,
        camera_id: int,
        frame,
        timestamp: float,
        roi: tuple[int, int, int, int],
        state: _CameraState,
    ) -> bool:
        if not self._engine_v2_shadow_enabled(timestamp):
            return False
        try:
            from app.engine_v2.live_shadow import (
                live_shadow_status,
                submit_live_shadow_frame,
            )

            accepted = submit_live_shadow_frame(
                camera_id,
                frame,
                ts=timestamp,
                roi=roi,
                detector_variant=self._selected_detector_variant(),
            )
            state.shadow_frames += int(bool(accepted))
            status = live_shadow_status(camera_id)
            return bool(accepted and status.get("ready"))
        except Exception:
            state.shadow_errors += 1
            return False

    def _ingest_engine_v2_event(self, event, frame) -> None:
        """Count a legacy V2 callback without allowing persistence."""
        try:
            camera_id = int(event.camera_id)
        except (TypeError, ValueError):
            return
        with self._lock:
            state = self._states.get(camera_id)
            if state is not None and not state.retired:
                state.shadow_candidates += 1

    def _observe_engine_v2_baseline(
        self,
        camera_id: int,
        rows: list,
        timestamp: float,
        state: _CameraState,
    ) -> None:
        if not self._engine_v2_shadow_enabled(timestamp):
            return
        try:
            from app.engine_v2.live_shadow import observe_live_shadow_baseline

            observe_live_shadow_baseline(
                camera_id,
                rows,
                ts=timestamp,
            )
        except Exception:
            state.shadow_errors += 1

    def _shadow_status(
        self,
        camera_id: int,
        state: _CameraState | None = None,
    ) -> dict:
        if not self._engine_v2_shadow_enabled():
            return {
                "enabled": False,
                "ready": False,
                "side_effects": False,
                "persistence": False,
                "frames": state.shadow_frames if state else 0,
                "candidates": state.shadow_candidates if state else 0,
                "events": state.shadow_candidates if state else 0,
                "errors": state.shadow_errors if state else 0,
            }
        try:
            from app.engine_v2.live_shadow import live_shadow_status

            result = dict(live_shadow_status(camera_id))
            result["candidates"] = int(result.get("events", 0))
            return result
        except Exception as exc:
            if state is not None:
                state.shadow_errors += 1
            return {
                "enabled": True,
                "ready": False,
                "side_effects": False,
                "persistence": False,
                "frames": state.shadow_frames if state else 0,
                "candidates": state.shadow_candidates if state else 0,
                "events": state.shadow_candidates if state else 0,
                "errors": (state.shadow_errors if state else 0) + 1,
                "last_error": f"{type(exc).__name__}: {exc}",
            }

    def _merge_shadow_detections(
        self,
        camera_id: int,
        baseline: list,
    ) -> list:
        rows = [dict(row) for row in baseline]
        if not self._engine_v2_shadow_enabled():
            return rows
        try:
            from app.engine_v2.live_shadow import live_shadow_detections

            rows.extend(live_shadow_detections(camera_id))
        except Exception:
            pass
        return rows

    def _event_commit_lock(self, camera_id: int) -> threading.RLock:
        """Return a lock that survives per-camera state recreation."""

        with self._lock:
            return self._event_commit_locks.setdefault(
                int(camera_id),
                threading.RLock(),
            )

    @staticmethod
    def _persistence_kind(result: dict) -> str:
        if result.get("unreadable_final"):
            return "unreadable"
        if result.get("capture_refresh"):
            return "capture-refresh"
        if result.get("capture_only"):
            return "capture"
        if result.get("valid") or result.get("auto_confirmed"):
            return "confirmed"
        return "review"

    @staticmethod
    def _encode_retry_frame(frame) -> bytes:
        if isinstance(frame, (bytes, bytearray, memoryview)):
            return bytes(frame)
        if frame is None or not getattr(frame, "size", 0):
            return b""
        try:
            encoded, buffer = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
        except Exception:
            return b""
        return bytes(buffer) if encoded and buffer is not None else b""

    @staticmethod
    def _decode_retry_frame(payload):
        if payload is None or not payload:
            return None
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            return payload
        try:
            frame = cv2.imdecode(
                np.frombuffer(bytes(payload), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
        except Exception:
            return None
        return frame if frame is not None and frame.size else None

    @staticmethod
    def _retry_entries(state: _CameraState) -> tuple[_PersistenceRetry, ...]:
        with state.model_switch_lock:
            return tuple(state.persistence_retry.values())

    @staticmethod
    def _retry_entry_memory_bytes(entry: _PersistenceRetry) -> int:
        total = 0
        frame = entry.frame
        if isinstance(frame, (bytes, bytearray, memoryview)):
            total += len(frame)
        elif frame is not None:
            total += int(getattr(frame, "nbytes", 0))
        for value in entry.result.values():
            total += int(getattr(value, "nbytes", 0))
        return total

    @staticmethod
    def _retry_memory_bytes(state: _CameraState) -> int:
        return sum(
            LiveANPRWorker._retry_entry_memory_bytes(entry)
            for entry in LiveANPRWorker._retry_entries(state)
        )

    def _retry_backpressure_active(
        self,
        state: _CameraState,
        camera_id: int | None = None,
    ) -> bool:
        if self._outbox_required and self._outbox is None:
            state.persistence_backpressure = True
            return True
        entries = self._retry_entries(state)
        unspooled = [entry for entry in entries if not entry.durably_spooled]
        if unspooled:
            # A transient sidecar failure must remain fail-closed.  Clearing
            # pressure merely because the in-memory queue is small would let
            # new inference outrun the only copy that survives a restart.
            state.persistence_backpressure = True
            return True
        resolved_camera_id = camera_id
        if resolved_camera_id is None and entries:
            resolved_camera_id = int(entries[0].camera_id)
        if self._outbox is not None:
            try:
                count, memory_bytes = self._outbox.pending_stats(
                    resolved_camera_id
                )
                self._outbox_error = ""
            except Exception as exc:
                self._outbox_error = f"{type(exc).__name__}: {exc}"
                state.persistence_backpressure = True
                return True
        else:
            count = len(entries)
            memory_bytes = self._retry_memory_bytes(state)
        if state.persistence_backpressure:
            state.persistence_backpressure = not (
                count <= PERSISTENCE_RETRY_LOW_COUNT
                and memory_bytes <= PERSISTENCE_RETRY_LOW_BYTES
            )
        else:
            state.persistence_backpressure = bool(
                count >= PERSISTENCE_RETRY_HIGH_COUNT
                or memory_bytes >= PERSISTENCE_RETRY_HIGH_BYTES
            )
        return state.persistence_backpressure

    def _make_persistence_retry(
        self,
        camera_id: int,
        camera_name: str,
        result: dict,
        frame,
        event_id: int | None,
        ledger_key: str,
        observed_at: float,
        processing_ms: float,
        duplicate_seconds: float,
        detector_generation: int,
        detector_revision: str = "",
        observed_at_epoch: float | None = None,
    ) -> _PersistenceRetry:
        stored_result = dict(result)
        stored_frame = self._encode_retry_frame(frame)
        # The full observation is retained once as compressed JPEG. Plate and
        # vehicle crops can be reconstructed from its detector coordinates.
        for image_key in ("crop", "vehicle_crop", "capture_frame"):
            stored_result.pop(image_key, None)
        identity = normalize_plate(
            stored_result.get("plate_norm")
            or stored_result.get("raw_guess_norm")
            or stored_result.get("plate")
        )
        resolved_revision = str(
            detector_revision
            or stored_result.get("detector_model_revision", "")
        ).strip()
        retry_key = (
            int(detector_generation),
            resolved_revision,
            int(stored_result.get("track_id") or 0),
            identity,
            self._persistence_kind(stored_result),
        )
        plate_root = Path(
            self._setting("plate_path", str(PLATE_DIR))
        ).expanduser().resolve()
        snapshot_root = Path(
            self._setting("snapshot_path", str(SNAPSHOT_DIR))
        ).expanduser().resolve()
        observed_epoch = (
            float(observed_at_epoch)
            if observed_at_epoch is not None
            else time.time()
        )
        return _PersistenceRetry(
            retry_key=retry_key,
            persistence_id=uuid4().hex,
            state_scope=(
                f"{self._state_scope}:camera-{int(camera_id)}"
            ),
            predecessor_id="",
            camera_id=int(camera_id),
            camera_name=str(camera_name),
            result=stored_result,
            frame=stored_frame,
            event_id=(int(event_id) if event_id is not None else None),
            ledger_key=str(ledger_key or ""),
            observed_at=float(observed_at),
            observed_at_utc=datetime.fromtimestamp(
                observed_epoch,
                timezone.utc,
            ).isoformat(timespec="microseconds").replace("+00:00", "Z"),
            processing_ms=float(processing_ms),
            duplicate_seconds=float(duplicate_seconds),
            detector_generation=int(detector_generation),
            detector_revision=resolved_revision,
            plate_root=str(plate_root),
            snapshot_root=str(snapshot_root),
            save_plate=self._setting("save_plate_images", "1") == "1",
            save_vehicle=self._setting("save_snapshots", "1") == "1",
        )

    def _select_retry_predecessor(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
    ) -> str:
        candidates = [
            previous
            for previous in state.persistence_retry.values()
            if (
                previous.persistence_id != entry.persistence_id
                and previous.event_id is None
                and self._retry_can_follow(previous, entry)
            )
        ]
        same_scope = [
            previous
            for previous in candidates
            if self._retry_scope(previous) == self._retry_scope(entry)
        ]
        selected = same_scope[-1:] if same_scope else candidates
        return (
            selected[0].persistence_id
            if len(selected) == 1
            else ""
        )

    def _enqueue_persistence_retry(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
    ) -> None:
        """Queue without silent eviction; replace only the same emission."""

        previous = state.persistence_retry.get(entry.retry_key)
        if previous is not None:
            if previous.primary_committed:
                # A receipt freezes the payload owned by its persistence id.
                # If primary commit succeeded but sidecar ACK deletion failed,
                # a later improvement must receive a fresh id instead of being
                # silently treated as replay of the older payload.
                state.persistence_retry.pop(entry.retry_key, None)
                previous_storage_key = entry.retry_key + (
                    previous.persistence_id,
                )
                state.persistence_retry[previous_storage_key] = previous
                entry.predecessor_id = previous.persistence_id
                entry.event_id = previous.event_id
                entry.durably_spooled = False
                state.persistence_retry[entry.retry_key] = entry
                try:
                    self._spool_retry(entry)
                except Exception as exc:
                    entry.last_error = f"{type(exc).__name__}: {exc}"
                    state.persistence_backpressure = True
                    self._record_persistence_error(state, exc)
                self._retry_wakeup.set()
                return
            entry.persistence_id = previous.persistence_id
            entry.state_scope = previous.state_scope
            entry.predecessor_id = previous.predecessor_id
            entry.event_id = (
                previous.event_id
                if previous.event_id is not None
                else entry.event_id
            )
            entry.attempts = previous.attempts
            entry.first_failed_at = previous.first_failed_at
            entry.first_failed_at_utc = previous.first_failed_at_utc
            entry.next_attempt_at_epoch = previous.next_attempt_at_epoch
            entry.last_error = previous.last_error
            entry.committed_as_insert = previous.committed_as_insert
            entry.primary_committed = previous.primary_committed
        elif not entry.predecessor_id:
            entry.predecessor_id = self._select_retry_predecessor(
                state,
                entry,
            )
        entry.durably_spooled = False
        state.persistence_retry[entry.retry_key] = entry
        try:
            self._spool_retry(entry)
        except Exception as exc:
            entry.last_error = f"{type(exc).__name__}: {exc}"
            state.persistence_backpressure = True
            self._record_persistence_error(state, exc)
        self._retry_wakeup.set()

    @staticmethod
    def _record_persistence_error(
        state: _CameraState,
        exc: Exception,
    ) -> str:
        error = f"{type(exc).__name__}: {exc}"
        state.last_error = error
        state.processing_errors += 1
        state.persistence_errors += 1
        state.last_processing_error = error
        return error

    def _mark_retry_failure(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
        exc: Exception,
        *,
        update_outbox=True,
    ) -> str:
        entry.attempts += 1
        if not entry.first_failed_at:
            entry.first_failed_at = time.monotonic()
        if not entry.first_failed_at_utc:
            entry.first_failed_at_utc = datetime.now(
                timezone.utc
            ).isoformat(timespec="microseconds").replace("+00:00", "Z")
        delay = (
            0.0
            if entry.attempts <= 1
            else 0.25 * (2 ** min(5, entry.attempts - 2))
        )
        entry.next_attempt_at_epoch = time.time() + delay
        error = f"{type(exc).__name__}: {exc}"
        entry.last_error = error
        if self._retry_context_is_current(state, entry):
            self._record_persistence_error(state, exc)
        if update_outbox and self._outbox is not None:
            try:
                stored_attempts = self._outbox.update_failure(
                    entry.persistence_id,
                    error,
                    next_attempt_at_epoch=entry.next_attempt_at_epoch,
                    failed_at_utc=entry.first_failed_at_utc,
                )
                if stored_attempts is not None:
                    entry.attempts = int(stored_attempts)
                self._outbox_error = ""
            except Exception as outbox_exc:
                self._outbox_error = (
                    f"{type(outbox_exc).__name__}: {outbox_exc}"
                )
                state.persistence_backpressure = True
        return error

    def _retry_context_is_current(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
    ) -> bool:
        if state.retired:
            return False
        if entry.detector_generation != self._detector_generation:
            return False
        current_revision = str(state.detector_model_revision or "").strip()
        return bool(
            not entry.detector_revision
            or not current_revision
            or entry.detector_revision == current_revision
        )

    @staticmethod
    def _retry_scope(entry) -> tuple[int, str, int]:
        return (
            int(entry.detector_generation),
            str(entry.detector_revision or ""),
            int(entry.result.get("track_id") or 0),
        )

    @staticmethod
    def _retry_can_follow(predecessor, successor) -> bool:
        """Return whether two queued writes belong to one visual visit."""

        if (
            int(predecessor.detector_generation)
            != int(successor.detector_generation)
            or str(predecessor.detector_revision or "")
            != str(successor.detector_revision or "")
        ):
            return False
        gap = float(successor.observed_at) - float(
            predecessor.observed_at
        )
        max_gap = 1.5
        if gap < 0.0 or gap > max_gap:
            return False

        previous_track = int(
            predecessor.result.get("track_id") or 0
        )
        current_track = int(successor.result.get("track_id") or 0)
        old_key = candidate_plate_key(predecessor.result)
        new_key = candidate_plate_key(successor.result)
        if previous_track and previous_track == current_track:
            if not old_key or not new_key or old_key == new_key:
                return True
            if successor.result.get("visit_identity_stable") is False:
                return True
            return bool(
                not strict_plate_key(predecessor.result)
                and (
                    review_identity_can_migrate(old_key, new_key)
                    or (
                        strict_plate_key(successor.result)
                        and PlateVisitLedger._provisional_upgrade_source(
                            predecessor.result
                        )
                    )
                )
            )
        return fragmented_review_can_migrate(
            predecessor.result,
            successor.result,
        )

    def _remember_retry_lineage(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
        saved_id: int,
    ) -> None:
        if (
            not self._retry_context_is_current(state, entry)
            or entry.attempts <= 0
            or not candidate_plate_key(entry.result)
        ):
            return
        track_id = int(entry.result.get("track_id") or 0)
        if track_id in state.tracker.active_track_ids():
            return
        state.retry_lineages.append(
            _RetryLineage(
                event_id=int(saved_id),
                result=dict(entry.result),
                observed_at=float(entry.observed_at),
                detector_generation=int(entry.detector_generation),
                detector_revision=str(entry.detector_revision or ""),
            )
        )
        # This is only a short bridge between a failed provisional write and
        # its next clear observation. Keep it bounded independently from the
        # durable retry queue.
        del state.retry_lineages[:-32]

    def _retry_lineage_event(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
    ) -> tuple[int | None, _RetryLineage | None]:
        if not self._retry_context_is_current(state, entry):
            return None, None
        matches = [
            lineage
            for lineage in state.retry_lineages
            if self._retry_can_follow(lineage, entry)
        ]
        if len(matches) != 1:
            return None, None
        return int(matches[0].event_id), matches[0]

    @staticmethod
    def _persistence_receipt_event_id(persistence_id: str) -> int | None:
        from app.database import connect

        with connect() as con:
            row = con.execute(
                "SELECT event_id FROM anpr_persistence_receipts "
                "WHERE persistence_key=? LIMIT 1",
                (str(persistence_id),),
            ).fetchone()
        return int(row["event_id"]) if row else None

    def _ack_persistence_retry(
        self,
        state: _CameraState,
        entry: _PersistenceRetry,
        saved_id: int,
        effective_event_id: int | None,
    ) -> None:
        """Apply durable-write state only inside the originating context."""

        if not self._retry_context_is_current(state, entry):
            return
        track_id = int(entry.result.get("track_id") or 0)
        track_is_active = track_id in state.tracker.active_track_ids()
        # A fresh expired/final result historically registers its visit once.
        # A delayed retry whose visual track is gone must not revive that old
        # visit or merge it with a later return of the same plate.
        update_visit_state = bool(entry.attempts == 0 or track_is_active)
        if update_visit_state:
            state.track_event_ids[track_id] = int(saved_id)
            if entry.ledger_key:
                state.visits.register(
                    entry.result,
                    int(saved_id),
                    entry.observed_at,
                    allow_candidate=True,
                )
        if effective_event_id is None:
            state.emitted_events += 1
        if effective_event_id is None or not entry.result.get("capture_only"):
            state.last_event_at = time.time()

    def _drain_persistence_retry_locked(
        self,
        state: _CameraState,
        event_commit_lock: threading.RLock,
        *,
        allow_retired=False,
    ) -> str:
        """Try every independent event and remove entries only after ACK."""

        last_error = ""
        failed_predecessors = []
        resolved_predecessors = []
        for retry_key in tuple(state.persistence_retry):
            entry = state.persistence_retry.get(retry_key)
            if entry is None:
                continue
            if entry.next_attempt_at_epoch > time.time():
                failed_predecessors.append(entry)
                continue
            if entry.predecessor_id and any(
                predecessor.persistence_id == entry.predecessor_id
                for predecessor in state.persistence_retry.values()
                if predecessor is not entry
            ):
                continue
            track_scope = self._retry_scope(entry)
            if any(
                self._retry_can_follow(predecessor, entry)
                for predecessor in failed_predecessors
            ):
                continue
            if not entry.durably_spooled:
                try:
                    self._spool_retry(entry)
                except Exception as exc:
                    last_error = self._mark_retry_failure(
                        state,
                        entry,
                        exc,
                        update_outbox=False,
                    )
                    state.persistence_backpressure = True
                    failed_predecessors.append(entry)
                    continue
            effective_event_id = entry.event_id
            used_lineage = None
            if effective_event_id is None:
                same_scope = [
                    (predecessor, saved_id)
                    for predecessor, saved_id in resolved_predecessors
                    if self._retry_scope(predecessor) == track_scope
                ]
                compatible = [
                    (predecessor, saved_id)
                    for predecessor, saved_id in resolved_predecessors
                    if self._retry_can_follow(predecessor, entry)
                ]
                selected = (
                    same_scope[-1:]
                    if same_scope
                    else compatible
                )
                if len(selected) == 1:
                    effective_event_id = int(selected[0][1])
            if (
                effective_event_id is None
                and self._retry_context_is_current(state, entry)
                and state.visits.can_reuse_track_event(
                    track_scope[2],
                    entry.result,
                )
            ):
                effective_event_id = state.track_event_ids.get(
                    track_scope[2]
                )
            if effective_event_id is None:
                effective_event_id, used_lineage = (
                    self._retry_lineage_event(state, entry)
                )
            if effective_event_id is None and entry.predecessor_id:
                try:
                    effective_event_id = (
                        self._persistence_receipt_event_id(
                            entry.predecessor_id
                        )
                    )
                    if effective_event_id is None:
                        raise RuntimeError(
                            "durable retry predecessor has no receipt"
                        )
                except Exception as exc:
                    last_error = self._mark_retry_failure(
                        state,
                        entry,
                        exc,
                    )
                    failed_predecessors.append(entry)
                    continue
            persist_result = dict(entry.result)
            persist_result["_persistence_id"] = entry.persistence_id
            persist_result["_observed_at_utc"] = entry.observed_at_utc
            persist_result["_allow_recent_reuse"] = bool(
                self._retry_context_is_current(state, entry)
            )
            persist_result["_plate_root"] = entry.plate_root
            persist_result["_snapshot_root"] = entry.snapshot_root
            persist_result["_save_plate"] = bool(entry.save_plate)
            persist_result["_save_vehicle"] = bool(entry.save_vehicle)
            # Every durable retry owns deterministic, persistence-id-scoped
            # media targets.  A process may crash after the atomic image
            # replace but before the database write or failure counter update;
            # verified targets must therefore be reusable even at attempt 0.
            persist_result["_reuse_media_targets"] = bool(
                entry.persistence_id
            )
            if effective_event_id is not None:
                # Preserve the resolved insert→update chain even if the
                # process stops between this write and its in-memory ACK.
                if entry.event_id != int(effective_event_id):
                    entry.event_id = int(effective_event_id)
                    try:
                        self._spool_retry(entry)
                    except Exception as exc:
                        last_error = self._mark_retry_failure(
                            state,
                            entry,
                            exc,
                            update_outbox=False,
                        )
                        state.persistence_backpressure = True
                        failed_predecessors.append(entry)
                        continue
            try:
                with event_commit_lock:
                    if state.retired and not allow_retired:
                        return last_error
                    saved_id = self._persist(
                        entry.camera_id,
                        entry.camera_name,
                        self._decode_retry_frame(entry.frame),
                        persist_result,
                        entry.processing_ms,
                        effective_event_id,
                        entry.duplicate_seconds,
                    )
            except Exception as exc:
                last_error = self._mark_retry_failure(
                    state,
                    entry,
                    exc,
                )
                failed_predecessors.append(entry)
                continue

            # The durable write already succeeded. Bookkeeping must not turn
            # a committed event into another persistence attempt.
            if effective_event_id is None:
                entry.committed_as_insert = True
            entry.event_id = int(saved_id)
            entry.primary_committed = True
            if self._outbox is not None:
                try:
                    self._outbox.delete(entry.persistence_id)
                    with self._outbox_state_lock:
                        self._loaded_outbox_ids.discard(
                            entry.persistence_id
                        )
                    self._outbox_error = ""
                except Exception as exc:
                    last_error = self._mark_retry_failure(
                        state,
                        entry,
                        exc,
                    )
                    failed_predecessors.append(entry)
                    continue
            resolved_predecessors.append((entry, int(saved_id)))
            ack_event_id = (
                None
                if entry.committed_as_insert
                else effective_event_id
            )
            try:
                self._ack_persistence_retry(
                    state,
                    entry,
                    int(saved_id),
                    ack_event_id,
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if self._retry_context_is_current(state, entry):
                    self._record_persistence_error(state, exc)
            self._remember_retry_lineage(state, entry, int(saved_id))
            if used_lineage is not None:
                try:
                    state.retry_lineages.remove(used_lineage)
                except ValueError:
                    pass
            if state.persistence_retry.get(retry_key) is entry:
                state.persistence_retry.pop(retry_key, None)
        return last_error

    def _flush_persistence_retry(
        self,
        camera_id: int,
        state: _CameraState,
        *,
        deadline: float,
        allow_retired=False,
    ) -> bool:
        """Retry queued writes without requiring another camera frame."""

        event_commit_lock = self._event_commit_lock(camera_id)
        while self._retry_entries(state):
            before = len(self._retry_entries(state))
            with state.model_switch_lock:
                self._drain_persistence_retry_locked(
                    state,
                    event_commit_lock,
                    allow_retired=allow_retired,
                )
            remaining = self._retry_entries(state)
            if not remaining:
                return True
            if time.monotonic() >= float(deadline):
                return False
            if len(remaining) >= before:
                due_in = min(
                    (
                        max(0.0, entry.next_attempt_at_epoch - time.time())
                        for entry in remaining
                    ),
                    default=0.05,
                )
                time.sleep(max(0.01, min(0.50, due_in or 0.05)))
        return True

    def _retry_pump(self) -> None:
        """Retry active and detached durable writes without camera frames."""

        wait_seconds = 0.50
        while not self._retry_stop.is_set():
            self._retry_wakeup.wait(wait_seconds)
            self._retry_wakeup.clear()
            if self._retry_stop.is_set():
                return
            try:
                self._hydrate_outbox_entries()
            except Exception as exc:
                self._outbox_error = f"{type(exc).__name__}: {exc}"
            if self._retry_stop.is_set():
                return
            with self._lock:
                states = list(self._states.items())
                states.extend(self._detached_retry_states)
            pending = []
            for camera_id, state in states:
                if self._retry_stop.is_set():
                    return
                state_entries = self._retry_entries(state)
                if not state_entries:
                    continue
                event_commit_lock = self._event_commit_lock(camera_id)
                acquired = state.model_switch_lock.acquire(blocking=False)
                if not acquired:
                    pending.extend(state_entries)
                    continue
                try:
                    self._drain_persistence_retry_locked(
                        state,
                        event_commit_lock,
                        allow_retired=state.retired,
                    )
                    self._retry_backpressure_active(state, camera_id)
                except Exception as exc:
                    if not state.retired:
                        self._record_persistence_error(state, exc)
                finally:
                    state.model_switch_lock.release()
                pending.extend(self._retry_entries(state))
            with self._lock:
                self._detached_retry_states = [
                    (camera_id, state)
                    for camera_id, state in self._detached_retry_states
                    if self._retry_entries(state)
                ]
            if pending:
                wait_seconds = max(
                    0.02,
                    min(
                        0.50,
                        min(
                            max(
                                0.0,
                                entry.next_attempt_at_epoch - time.time(),
                            )
                            for entry in pending
                        )
                        or 0.05,
                    ),
                )
            else:
                wait_seconds = 0.50

    def begin_video_pass(self, camera_id: int) -> dict:
        """Capture the detector/error generation owned by one video pass."""

        with self._lock:
            state = self._states.get(int(camera_id))
            return {
                "detector_generation": self._detector_generation,
                "processing_errors": (
                    int(state.processing_errors) if state else 0
                ),
                "persistence_errors": (
                    int(state.persistence_errors) if state else 0
                ),
                "processed_frames": (
                    int(state.processed_frames) if state else 0
                ),
                "persistence_backpressure_frames": (
                    int(state.persistence_backpressure_frames)
                    if state else 0
                ),
            }

    def invalidate_model_cache(
        self,
        detector_variant=None,
        persist_setting=None,
    ) -> None:
        """Atomically isolate the next detector generation.

        Detector sessions are only part of the state involved in an A/B
        switch. Consensus votes, duplicate cooldowns and published overlays
        must also be discarded so the new detector cannot inherit evidence
        or suppression decisions from the old detector.
        """

        from .model_manager import (
            detector_variant_spec,
            normalize_detector_variant,
        )
        from .onnx_detector import clear_detector_sessions

        with self._lock:
            selected = None
            if detector_variant is not None:
                selected = normalize_detector_variant(detector_variant)
                selected_spec = detector_variant_spec(selected)
                if selected == "yolox" and not selected_spec.get("ready"):
                    raise FileNotFoundError(
                        selected_spec.get("error")
                        or "Verified YOLOX detector is not installed"
                    )
                if persist_setting is None:
                    from app.database import set_setting

                    persist_setting = set_setting

            # Acquire every per-camera commit boundary before changing the
            # persisted selection or generation. A transaction that already
            # passed its generation guard may finish first, but it remains on
            # the old setting side of this atomic switch boundary.
            states = list(self._states.values())
            locked_states = []
            try:
                for state in states:
                    state.model_switch_lock.acquire()
                    locked_states.append(state)

                if selected is not None:
                    persist_setting("anpr_detector_model", selected)
                self._detector_generation += 1
                self._model_state = {}
                self._model_state_at = 0.0
                self._model_state_variant = ""
                for state in states:
                    duplicate_seconds = max(
                        0.0,
                        float(
                            (state.config or {}).get(
                                "duplicate_seconds",
                                5.0,
                            )
                        ),
                    )
                    state.tracker = PlateConsensusTracker(
                        min_votes=2,
                        max_age_seconds=2.2,
                        emit_cooldown=duplicate_seconds,
                        emit_unreadable=True,
                    )
                    state.pending = None
                    state.visits.reset_tracker_bindings()
                    state.track_event_ids.clear()
                    state.retry_lineages.clear()
                    state.latest_detections = []
                    state.latest_detections_at = 0.0
                    state.latest_detection_frame = None
                    state.detection_revision += 1
                    state.last_error = ""
                    state.processing_errors = 0
                    state.persistence_errors = 0
                    state.last_processing_error = ""
                    state.last_event_at = 0.0
                    state.processed_frames = 0
                    state.inference_calls = 0
                    state.inference_seconds = 0.0
                    state.last_inference_ms = 0.0
                    state.coalesced_frames = 0
                    state.metrics_started_at = time.monotonic()
                    state.detected_candidates = 0
                    state.emitted_events = 0
                    state.whole_plate_ocr_attempts = 0
                    state.ocr_agreements = 0
                    state.ocr_disagreements = 0
                    state.crnn_selected = 0
                    state.character_reader_selected = 0
                    state.detector_model_revision = ""
                    state.last_processed_at = 0.0
                    state.last_processing_ms = 0.0
                    state.processing_seconds_ema = 0.0
                    state.no_plate_streak = 0
                    state.next_inference_at = 0.0
                    state.last_submitted_at = 0.0
                    state.burst_frames_remaining = 0
                    state.plate_visible = False
                    state.shadow_frames = 0
                    state.shadow_candidates = 0
                    state.shadow_errors = 0
                    state.motion_score = 0.0
                    state.motion_wakeups = 0
                    state.overlay_mask_pixels = 0
                    state.static_overlay_hits.clear()
                    state.static_overlay_blocked_until.clear()
                    state.frame_counter = 0
                clear_detector_sessions()
            finally:
                for state in reversed(locked_states):
                    state.model_switch_lock.release()

    def _exclusive_engine_status(self) -> dict:
        return {
            "mode": "baseline",
            "detector_variant": self._selected_detector_variant(),
            "exclusive_detector": True,
            "candidate_inference": False,
        }

    def _models(self) -> dict:
        now = time.monotonic()
        selected_variant = self._selected_detector_variant()
        cached_preparation_state = str(
            self._model_state.get("preparation_state", "")
        ).strip().lower()
        preparation_state = os.environ.get(
            "BCVISION_MODEL_PREPARATION_STATE",
            "",
        ).strip().lower()
        refresh_seconds = (
            4.0
            if preparation_state in {"preparing", "retrying"}
            or (
                not self._model_state.get("detector_ready")
                and preparation_state != "error"
            )
            else 30.0
        )
        if (
            now - self._model_state_at >= refresh_seconds
            or preparation_state != cached_preparation_state
            or selected_variant != self._model_state_variant
        ):
            try:
                from .model_manager import model_status
                self._model_state = model_status(
                    selected_detector=selected_variant,
                )
            except Exception as exc:
                self._model_state = {
                    "selected_detector": selected_variant,
                    "detector_ready": False,
                    "crnn_ready": False,
                    "cnn_ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._model_state_at = now
            self._model_state_variant = selected_variant
        status = dict(self._model_state)
        status["ocr_primary_ready"] = bool(status.get("hezar_ready"))
        status["ocr_fallback_ready"] = bool(status.get("crnn_ready"))
        status["ocr_degraded"] = bool(
            not status["ocr_primary_ready"]
            and status["ocr_fallback_ready"]
        )
        # CNN/custom CRNN availability cannot make the production stack ready.
        # Hezar v2 is the required primary; fixed Platrix is an optional
        # degraded-path fallback after a Hezar rejection.
        status["ocr_ready"] = status["ocr_primary_ready"]
        status["ready"] = bool(
            status.get("detector_ready")
            and status["ocr_ready"]
        )
        return status

    @staticmethod
    def _load_config(camera_id: int) -> dict | None:
        from app.database import connect
        with connect() as con:
            row = con.execute(
                "SELECT * FROM cameras WHERE id=?",
                (camera_id,),
            ).fetchone()
        return dict(row) if row else None

    def _config(
        self,
        camera_id: int,
        state: _CameraState,
        now: float,
    ) -> dict | None:
        if state.config is None or now - state.config_loaded_at >= 5.0:
            state.config = self._load_config(camera_id)
            state.config_loaded_at = now
            if state.config:
                duplicate_seconds = max(
                    0.0,
                    float(state.config.get("duplicate_seconds", 30)),
                )
                state.tracker.emit_cooldown = duplicate_seconds
        return state.config

    @staticmethod
    def _roi_frame(frame, config):
        height, width = frame.shape[:2]
        rx = float(config.get("roi_x", 0))
        ry = float(config.get("roi_y", 0))
        rw = float(config.get("roi_w", 100))
        rh = float(config.get("roi_h", 100))
        x1 = max(0, min(width - 1, int(width * rx / 100.0)))
        y1 = max(0, min(height - 1, int(height * ry / 100.0)))
        x2 = max(
            x1 + 1,
            min(width, int(width * (rx + rw) / 100.0)),
        )
        y2 = max(
            y1 + 1,
            min(height, int(height * (ry + rh) / 100.0)),
        )
        return frame[y1:y2, x1:x2], x1, y1

    @staticmethod
    def _translate(result, offset_x, offset_y):
        if not (offset_x or offset_y):
            return result
        row = dict(result)
        x1, y1, x2, y2 = row["bbox"]
        row["bbox"] = (
            x1 + offset_x,
            y1 + offset_y,
            x2 + offset_x,
            y2 + offset_y,
        )
        if row.get("vehicle_bbox"):
            vx1, vy1, vx2, vy2 = row["vehicle_bbox"]
            row["vehicle_bbox"] = (
                vx1 + offset_x,
                vy1 + offset_y,
                vx2 + offset_x,
                vy2 + offset_y,
            )
        if row.get("quadrilateral"):
            row["quadrilateral"] = [
                [
                    float(point[0]) + offset_x,
                    float(point[1]) + offset_y,
                ]
                for point in row["quadrilateral"]
            ]
        return row

    @staticmethod
    def _setting(key, default=""):
        try:
            from app.database import get_setting
            return get_setting(key, default)
        except Exception:
            # Inference must retain safe defaults during first-run database
            # creation or a transient settings migration.
            return default

    def _persist(
        self,
        camera_id: int,
        camera_name: str,
        frame,
        result: dict,
        processing_ms: float,
        event_id: int | None = None,
        duplicate_seconds: float = 0.0,
    ):
        from app.database import connect
        from app.config import PLATE_DIR, SNAPSHOT_DIR

        result = dict(result)
        allow_recent_reuse = bool(
            result.pop("_allow_recent_reuse", True)
        )
        persistence_id = str(
            result.pop("_persistence_id", "") or ""
        ).strip()
        observed_at_utc = str(
            result.pop("_observed_at_utc", "") or ""
        ).strip()
        configured_plate_root = str(
            result.pop("_plate_root", "") or ""
        ).strip()
        configured_snapshot_root = str(
            result.pop("_snapshot_root", "") or ""
        ).strip()
        save_plate_override = result.pop("_save_plate", None)
        save_vehicle_override = result.pop("_save_vehicle", None)
        reuse_media_targets = bool(
            result.pop("_reuse_media_targets", False)
        )
        try:
            observed_datetime = datetime.fromisoformat(
                observed_at_utc.replace("Z", "+00:00")
            )
            if observed_datetime.tzinfo is None:
                observed_datetime = observed_datetime.replace(
                    tzinfo=timezone.utc
                )
            observed_datetime = observed_datetime.astimezone(timezone.utc)
        except (TypeError, ValueError):
            observed_datetime = datetime.now(timezone.utc)
        observed_at_utc = observed_datetime.strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        )
        if frame is not None and getattr(frame, "size", 0):
            result = add_vehicle_analysis(result, frame)
        else:
            result.setdefault("vehicle_type", "نامشخص")
            result.setdefault("vehicle_color", "نامشخص")
            result.setdefault("vehicle_brand", "نامشخص")
            result.setdefault("vehicle_confidence", 0.0)
            result.setdefault("vehicle_bbox", None)
        plate_dir = Path(
            configured_plate_root
            or self._setting("plate_path", str(PLATE_DIR))
        )
        snapshot_dir = Path(
            configured_snapshot_root
            or self._setting("snapshot_path", str(SNAPSHOT_DIR))
        )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        media_token = "".join(
            character
            for character in persistence_id
            if character.isalnum() or character in {"-", "_"}
        )[:96] or stamp
        plate_path = ""
        image_path = ""
        receipt_replay = False
        with connect() as con:
            try:
                camera_row = con.execute(
                    "SELECT city,location,rtsp_url "
                    "FROM cameras WHERE id=?",
                    (int(camera_id),),
                ).fetchone()
            except Exception:
                # Compatibility with pre-migration/minimal recovery schemas.
                camera_row = None
            incoming_strict_identity = strict_plate_key(result)
            if persistence_id:
                committed = con.execute(
                    "SELECT event_id FROM anpr_persistence_receipts "
                    "WHERE persistence_key=? LIMIT 1",
                    (persistence_id,),
                ).fetchone()
                if committed:
                    event_id = int(committed["event_id"])
                    receipt_replay = True
            recent_window = max(
                0.0,
                float(duplicate_seconds),
                2.0 if incoming_strict_identity else 0.0,
            )
            if (
                event_id is None
                and incoming_strict_identity
                and allow_recent_reuse
            ):
                cutoff = datetime.fromtimestamp(
                    observed_datetime.timestamp() - recent_window,
                    timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S.%f")
                upper_bound = datetime.fromtimestamp(
                    observed_datetime.timestamp() + 2.0,
                    timezone.utc,
                ).strftime("%Y-%m-%d %H:%M:%S.%f")
                try:
                    recent = con.execute(
                        "SELECT id FROM plate_events "
                        "WHERE camera_id=? AND plate_norm=? "
                        "AND COALESCE(source,'live')='live' "
                        "AND COALESCE(updated_at,created_at)>=? "
                        "AND COALESCE(updated_at,created_at)<=? "
                        "ORDER BY COALESCE(updated_at,created_at) DESC,id DESC "
                        "LIMIT 1",
                        (
                            int(camera_id),
                            incoming_strict_identity,
                            cutoff,
                            upper_bound,
                        ),
                    ).fetchone()
                except Exception:
                    # Minimal recovery/test schemas may not have lifecycle
                    # columns. The visit ledger remains the primary guard.
                    recent = None
                if recent:
                    event_id = int(recent["id"])
            if event_id:
                try:
                    existing = con.execute(
                        "SELECT image_path,plate_image_path,city,plate_norm,"
                        "review_status,operator_reviewed,confirmation_source,"
                        "created_at,updated_at "
                        "FROM plate_events WHERE id=?",
                        (int(event_id),),
                    ).fetchone()
                except Exception:
                    try:
                        existing = con.execute(
                            "SELECT image_path,plate_image_path,plate_norm "
                            "FROM plate_events WHERE id=?",
                            (int(event_id),),
                        ).fetchone()
                    except Exception:
                        existing = con.execute(
                            "SELECT image_path,plate_image_path "
                            "FROM plate_events WHERE id=?",
                            (int(event_id),),
                        ).fetchone()
            else:
                existing = None
        if receipt_replay and not existing:
            # Retention may intentionally delete the event while the receipt
            # remains as an idempotency tombstone. A replay acknowledges that
            # outcome instead of resurrecting an expired event.
            return int(event_id)
        incoming_identity = normalize_plate(
            result.get("plate_norm", "")
        )
        existing_identity = (
            normalize_plate(existing["plate_norm"])
            if (
                existing
                and "plate_norm" in existing.keys()
            )
            else ""
        )
        if existing_identity and incoming_identity != existing_identity:
            if (
                incoming_identity
                and (
                    result.get("valid")
                    or result.get("auto_confirmed")
                )
            ):
                # A different identity must never overwrite the confirmed
                # event. Preserve it as a separate reviewable observation:
                # the stale event_id itself is evidence of an association
                # conflict, so automatic confirmation is unsafe.
                result = {
                    **result,
                    "valid": False,
                    "auto_confirmed": False,
                    "needs_review": True,
                    "read_status": "identity-conflict",
                    "raw_guess_norm": incoming_identity,
                    "raw_guess_text": result.get("plate", ""),
                    "raw_guess_reason": "tracker-identity-conflict",
                }
                existing = None
                event_id = None
            else:
                # A reviewable, unreadable, or capture-only row may never
                # erase/downgrade an already identified event.
                if persistence_id and not receipt_replay:
                    with connect() as con:
                        con.execute("PRAGMA synchronous=FULL")
                        con.execute(
                            "INSERT INTO anpr_persistence_receipts("
                            "persistence_key,event_id) VALUES(?,?)",
                            (persistence_id, int(event_id)),
                        )
                return int(event_id)
        if (
            existing
            and event_id
            and incoming_identity
            and incoming_identity == existing_identity
            and "updated_at" in existing.keys()
            and str(existing["updated_at"] or "") > observed_at_utc
        ):
            # A delayed retry may acknowledge its durable observation, but it
            # must never replace newer confidence, provenance or media.
            if persistence_id and not receipt_replay:
                with connect() as con:
                    con.execute("PRAGMA synchronous=FULL")
                    con.execute(
                        "INSERT INTO anpr_persistence_receipts("
                        "persistence_key,event_id) VALUES(?,?)",
                        (persistence_id, int(event_id)),
                    )
            return int(event_id)
        if existing:
            plate_path = existing["plate_image_path"] or ""
            image_path = existing["image_path"] or ""
        plate_root = plate_dir.expanduser().resolve()
        snapshot_root = snapshot_dir.expanduser().resolve()

        def media_target(existing_path, root, filename):
            if existing_path:
                try:
                    current = Path(existing_path).expanduser().resolve()
                    if current.is_file() and current.is_relative_to(root):
                        return current
                except OSError:
                    pass
            return root / filename

        media = save_event_images(
            result,
            frame,
            plate_target=(
                media_target(
                    plate_path,
                    plate_root,
                    f"plate-live-{media_token}.jpg",
                )
            ),
            vehicle_target=(
                media_target(
                    image_path,
                    snapshot_root,
                    f"vehicle-live-{media_token}.jpg",
                )
            ),
            save_plate=(
                bool(save_plate_override)
                if save_plate_override is not None
                else self._setting("save_plate_images", "1") == "1"
            ),
            save_vehicle=(
                bool(save_vehicle_override)
                if save_vehicle_override is not None
                else self._setting("save_snapshots", "1") == "1"
            ),
            existing_plate_path=plate_path,
            existing_vehicle_path=image_path,
            reuse_existing_targets=reuse_media_targets,
            defer_commit=True,
        )
        pending_media = tuple(media.pending_writes)
        plate_path = media.plate_path
        image_path = media.image_path
        plate_identity_norm = (
            normalize_plate(result.get("plate_norm"))
            or normalize_plate(result.get("raw_guess_norm"))
            or normalize_plate(result.get("plate"))
        )
        plate_parts = split_iran_plate(plate_identity_norm)
        recognized = bool(
            plate_parts
            and (result.get("valid") or result.get("auto_confirmed"))
            and not result.get("unreadable_final")
        )
        plate_norm = plate_identity_norm if recognized else ""
        plate_text = (
            result.get("plate")
            if recognized
            else (
                result.get("raw_guess_text") or result.get("plate")
                if result.get("needs_review")
                else "ناخوانا"
            )
        ) or "ناخوانا"
        review_status = (
            "auto-confirmed"
            if recognized and result.get("auto_confirmed")
            else (
                "confirmed-ai"
                if recognized
                else (
                    "suggested"
                    if result.get("needs_review")
                    else "unreadable"
                )
            )
        )
        camera_city = (
            str(camera_row["city"] or "")
            if camera_row else ""
        )
        event_city = (
            str(existing["city"] or "")
            if existing and "city" in existing.keys()
            else str(result.get("city") or camera_city)
        )
        camera_url = str(camera_row["rtsp_url"] or "") if camera_row else ""
        updated_at_utc = observed_at_utc
        if existing and "updated_at" in existing.keys():
            updated_at_utc = max(
                observed_at_utc,
                str(existing["updated_at"] or ""),
            )

        values = {
            "plate_text": plate_text,
            "plate_norm": plate_norm,
            "plate_region": (
                plate_parts["region"] if plate_parts else ""
            ),
            "confidence": float(result["confidence"]),
            "camera_id": camera_id,
            "camera_name": camera_name,
            "city": event_city,
            "image_path": image_path,
            "plate_image_path": plate_path,
            "media_status": media.media_status,
            "media_error": media.media_error,
            "created_at": observed_at_utc,
            "updated_at": updated_at_utc,
            "video_path": (
                camera_url[len("video://"):]
                if camera_url.startswith("video://")
                else ""
            ),
            "video_second": 0.0,
            "detector_method": result.get("method", "live"),
            "ocr_confidence": float(
                result.get("ocr_confidence", 0.0)
            ),
            "ocr_engine": result.get("ocr_engine", ""),
            "ocr_alternative": result.get(
                "ocr_alternative",
                "",
            ),
            "ocr_disagreement": int(
                bool(result.get("ocr_disagreement"))
            ),
            "vehicle_type": result.get(
                "vehicle_type",
                "نامشخص",
            ),
            "vehicle_color": result.get(
                "vehicle_color",
                "نامشخص",
            ),
            "vehicle_brand": result.get(
                "vehicle_brand",
                "نامشخص",
            ),
            "vehicle_confidence": float(
                result.get("vehicle_confidence", 0.0)
            ),
            "direction": result.get("direction", "stationary"),
            "quality_score": float(
                result.get("quality_score", 0.0)
            ),
            "consensus_votes": int(
                result.get("consensus_votes", 1)
            ),
            "source": "live",
            "processing_ms": float(processing_ms),
            "review_status": review_status,
            "confirmation_source": result.get(
                "confirmation_source",
                (
                    "operator-learned"
                    if result.get("operator_learned")
                    else "ai-strict"
                ),
            ),
            "operator_reviewed": int(bool(
                result.get("operator_reviewed")
            )),
            "raw_guess_text": result.get(
                "raw_guess_text",
                result.get("plate", ""),
            ),
            "raw_guess_norm": normalize_plate(
                result.get("raw_guess_norm")
                or result.get("raw_guess_text")
                or result.get("plate")
            ),
            "raw_guess_confidence": float(
                result.get(
                    "raw_guess_confidence",
                    result.get("ocr_confidence", 0.0),
                )
            ),
            "raw_guess_engine": result.get(
                "raw_guess_engine",
                result.get("ocr_engine", ""),
            ),
            "raw_guess_reason": result.get(
                "raw_guess_reason",
                "",
            ),
            "model_revision": result.get(
                "model_revision",
                result.get("ocr_engine", ""),
            ),
            "experimental": int(bool(
                result.get("experimental")
                or result.get("needs_review")
            )),
        }
        if (
            existing
            and "operator_reviewed" in existing.keys()
            and bool(existing["operator_reviewed"])
        ):
            # Camera refreshes may improve media, but they must never undo a
            # human decision already attached to the canonical event.
            values["operator_reviewed"] = 1
            if "review_status" in existing.keys():
                values["review_status"] = existing["review_status"]
            if "confirmation_source" in existing.keys():
                values["confirmation_source"] = existing[
                    "confirmation_source"
                ]
        duplicate_saved_id = None
        try:
            with connect() as con:
                if persistence_id or pending_media:
                    # The owner+intent commit must be at least as durable as
                    # the FULL-synchronous outbox ACK that follows it.
                    con.execute("PRAGMA synchronous=FULL")
                columns = {
                    row[1]
                    for row in con.execute(
                        "PRAGMA table_info(plate_events)"
                    ).fetchall()
                }
                receipt = None
                if persistence_id:
                    receipt = con.execute(
                        "SELECT event_id "
                        "FROM anpr_persistence_receipts "
                        "WHERE persistence_key=? LIMIT 1",
                        (persistence_id,),
                    ).fetchone()
                    if receipt and not receipt_replay:
                        duplicate_saved_id = int(receipt["event_id"])
                if duplicate_saved_id is None:
                    selected = [key for key in values if key in columns]
                    if existing and event_id:
                        selected = [
                            key for key in selected if key != "created_at"
                        ]
                        if receipt_replay:
                            # Idempotency receipts freeze the observation
                            # payload. Replays may repair media only.
                            selected = [
                                key
                                for key in selected
                                if key in {
                                    "image_path",
                                    "plate_image_path",
                                    "media_status",
                                    "media_error",
                                }
                            ]
                        if selected:
                            assignments = ",".join(
                                f"{key}=?" for key in selected
                            )
                            con.execute(
                                f"UPDATE plate_events SET {assignments} "
                                "WHERE id=?",
                                tuple(values[key] for key in selected)
                                + (int(event_id),),
                            )
                        saved_id = int(event_id)
                    else:
                        placeholders = ",".join("?" for _ in selected)
                        cursor = con.execute(
                            f"INSERT INTO plate_events({','.join(selected)}) "
                            f"VALUES({placeholders})",
                            tuple(values[key] for key in selected),
                        )
                        saved_id = int(cursor.lastrowid)
                    if persistence_id and not receipt:
                        con.execute(
                            "INSERT INTO anpr_persistence_receipts("
                            "persistence_key,event_id) VALUES(?,?)",
                            (persistence_id, saved_id),
                        )
                    for pending in pending_media:
                        pending.accept(
                            con,
                            owner_kind="plate-event",
                            owner_id=saved_id,
                        )
        except BaseException:
            settle_pending_media(pending_media)
            raise
        if duplicate_saved_id is not None:
            settle_pending_media(pending_media)
            return duplicate_saved_id
        finalize_pending_media(pending_media)
        return saved_id

    @staticmethod
    def _selection_score(frame, config) -> float:
        source, _, _ = LiveANPRWorker._roi_frame(frame, config)
        height, width = source.shape[:2]
        if width > 320:
            scale = 320.0 / width
            source = cv2.resize(
                source,
                (320, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        exposure = max(0.0, 1.0 - abs(brightness - 135.0) / 135.0)
        sharpness = min(
            1.0,
            float(cv2.Laplacian(gray, cv2.CV_64F).var()) / 420.0,
        )
        return round(0.72 * sharpness + 0.28 * exposure, 5)

    @staticmethod
    def _post_inference_delay(
        processing_seconds_ema: float,
        no_plate_streak: int,
    ) -> float:
        processing_gap = max(
            0.20,
            min(1.60, float(processing_seconds_ema) * 0.55),
        )
        empty_gap = (
            min(
                3.20,
                0.40 * (2 ** min(3, int(no_plate_streak) - 1)),
            )
            if no_plate_streak
            else 0.0
        )
        return max(processing_gap, empty_gap)

    def submit(self, camera_id: int, camera_name: str, frame):
        if (
            self._stopped
            or frame is None
            or getattr(frame, "size", 0) == 0
        ):
            return
        now = time.monotonic()
        observed_at_epoch = time.time()
        with self._lock:
            state = self._states.setdefault(
                int(camera_id),
                _CameraState(),
            )
            state.frame_counter += 1
            try:
                config = self._config(int(camera_id), state, now)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                state.last_error = error
                state.processing_errors += 1
                state.last_processing_error = error
                return
            if (
                not config
                or not int(config.get("enabled", 0))
                or not int(config.get("lpr_enabled", 0))
            ):
                return
            if self._retry_backpressure_active(state, int(camera_id)):
                state.persistence_backpressure_frames += 1
                state.pending = None
                self._retry_wakeup.set()
                return
            selection_score = self._selection_score(frame, config)
            activity_source, roi_x, roi_y = self._roi_frame(frame, config)
            shadow_roi = (
                roi_x,
                roi_y,
                roi_x + int(activity_source.shape[1]),
                roi_y + int(activity_source.shape[0]),
            )
            self._submit_engine_v2_shadow(
                int(camera_id),
                frame,
                now,
                shadow_roi,
                state,
            )
            activity = state.activity.observe(activity_source)
            state.motion_score = float(activity.motion_score)
            state.overlay_mask_pixels = (
                int(cv2.countNonZero(activity.exclusion_mask))
                if activity.exclusion_mask is not None
                else 0
            )
            if activity.wake_inference:
                state.motion_wakeups += 1
                state.burst_frames_remaining = max(
                    state.burst_frames_remaining,
                    4,
                )
                state.next_inference_at = min(
                    state.next_inference_at,
                    now,
                )
                selection_score += min(
                    0.40,
                    0.18 + float(activity.motion_score),
                )
            payload = (
                int(camera_id),
                str(camera_name),
                frame.copy(),
                now,
                selection_score,
                activity,
                self._detector_generation,
                observed_at_epoch,
            )
            if state.busy:
                pending_score = (
                    float(state.pending[4])
                    if state.pending is not None and len(state.pending) > 4
                    else -1.0
                )
                pending_at = (
                    float(state.pending[3])
                    if state.pending is not None
                    else -1e12
                )
                if (
                    state.pending is None
                    or selection_score >= pending_score
                    or now - pending_at >= 0.12
                ):
                    if state.pending is not None:
                        state.coalesced_frames += 1
                    state.pending = payload
                else:
                    state.coalesced_frames += 1
                return
            # Do not let a slow CPU run ANPR continuously with no breathing
            # room. Keep the newest frame and cap inference frequency
            # adaptively; this reduces load without lowering image quality.
            minimum_interval = max(
                0.0,
                (
                    0.0
                    if activity.wake_inference
                    else state.next_inference_at - now
                ),
                (
                    0.0
                    if state.burst_frames_remaining
                    else (
                        max(
                            0.20,
                            min(
                                1.25,
                                state.processing_seconds_ema * 0.45,
                            ),
                        )
                        if state.processing_seconds_ema
                        else 0.0
                    )
                ),
            )
            if (
                now - state.last_submitted_at < minimum_interval
                or (
                    now < state.next_inference_at
                    and not activity.wake_inference
                )
            ):
                pending_score = (
                    float(state.pending[4])
                    if state.pending is not None and len(state.pending) > 4
                    else -1.0
                )
                if (
                    state.pending is None
                    or selection_score >= pending_score
                ):
                    if state.pending is not None:
                        state.coalesced_frames += 1
                    state.pending = payload
                else:
                    state.coalesced_frames += 1
                return
            if state.pending is not None:
                pending_score = float(state.pending[4])
                pending_at = float(state.pending[3])
                if (
                    pending_score > selection_score
                    and now - pending_at < 0.30
                ):
                    payload = state.pending
                state.coalesced_frames += 1
                state.pending = None
            state.last_submitted_at = now
            state.busy = True
        try:
            self._executor.submit(self._process, state, payload)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            with self._lock:
                state.busy = False
                state.last_error = error
                state.processing_errors += 1
                state.last_processing_error = error
            raise

    def drain_video_pass(
        self,
        camera_id: int,
        pass_token: dict | None = None,
        timeout: float = 60.0,
    ) -> dict:
        """Promote pending work and wait for one video pass to become idle."""

        token = dict(pass_token or {})
        expected_generation = int(
            token.get("detector_generation", self._detector_generation)
        )
        baseline_errors = max(
            0,
            int(token.get("processing_errors", 0)),
        )
        baseline_persistence_errors = max(
            0,
            int(token.get("persistence_errors", 0)),
        )
        baseline_non_persistence_errors = max(
            0,
            baseline_errors - baseline_persistence_errors,
        )
        baseline_processed = max(
            0,
            int(token.get("processed_frames", 0)),
        )
        baseline_backpressure_frames = max(
            0,
            int(token.get("persistence_backpressure_frames", 0)),
        )
        deadline = time.monotonic() + max(
            0.1,
            min(300.0, float(timeout)),
        )
        camera_id = int(camera_id)
        while True:
            retry_state = None
            with self._lock:
                if expected_generation != self._detector_generation:
                    return {
                        "ok": False,
                        "error": (
                            "RuntimeError: detector selection changed "
                            "during uploaded-video processing"
                        ),
                    }
                state = self._states.get(camera_id)
                if state is None:
                    return {
                        "ok": False,
                        "error": (
                            "RuntimeError: uploaded video reached EOF "
                            "without an ANPR worker submission"
                        ),
                    }
                if not state.busy and state.pending is not None:
                    payload = state.pending
                    payload_generation = (
                        int(payload[6])
                        if len(payload) > 6
                        else self._detector_generation
                    )
                    if payload_generation != expected_generation:
                        return {
                            "ok": False,
                            "error": (
                                "RuntimeError: detector selection changed "
                                "during uploaded-video processing"
                            ),
                        }
                    state.pending = None
                    state.last_submitted_at = time.monotonic()
                    state.busy = True
                    try:
                        self._executor.submit(self._process, state, payload)
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        state.busy = False
                        state.last_error = error
                        state.processing_errors += 1
                        state.last_processing_error = error
                elif not state.busy and self._retry_entries(state):
                    retry_state = state
                elif not state.busy:
                    if (
                        state.persistence_backpressure_frames
                        > baseline_backpressure_frames
                    ):
                        return {
                            "ok": False,
                            "error": (
                                "RuntimeError: persistence backlog skipped "
                                "uploaded-video frames"
                            ),
                            "processed_frames": state.processed_frames,
                            "emitted_events": state.emitted_events,
                            "persistence_backpressure_frames": (
                                state.persistence_backpressure_frames
                            ),
                        }
                    non_persistence_errors = max(
                        0,
                        state.processing_errors
                        - state.persistence_errors,
                    )
                    if (
                        non_persistence_errors
                        > baseline_non_persistence_errors
                    ):
                        return {
                            "ok": False,
                            "error": (
                                state.last_processing_error
                                or state.last_error
                                or "RuntimeError: ANPR processing failed"
                            ),
                            "processed_frames": state.processed_frames,
                            "emitted_events": state.emitted_events,
                        }
                    if state.processed_frames <= baseline_processed:
                        return {
                            "ok": False,
                            "error": (
                                "RuntimeError: uploaded video reached EOF "
                                "without a completed ANPR frame"
                            ),
                            "processed_frames": state.processed_frames,
                            "emitted_events": state.emitted_events,
                        }
                    return {
                        "ok": True,
                        "error": "",
                        "processed_frames": state.processed_frames,
                        "emitted_events": state.emitted_events,
                    }
            if retry_state is not None:
                if not self._flush_persistence_retry(
                    camera_id,
                    retry_state,
                    deadline=deadline,
                ):
                    return {
                        "ok": False,
                        "error": (
                            retry_state.last_processing_error
                            or retry_state.last_error
                            or "RuntimeError: event persistence retry failed"
                        ),
                        "processed_frames": retry_state.processed_frames,
                        "emitted_events": retry_state.emitted_events,
                        "pending_retry_count": len(
                            self._retry_entries(retry_state)
                        ),
                    }
                continue
            if time.monotonic() >= deadline:
                return {
                    "ok": False,
                    "error": (
                        "TimeoutError: ANPR worker did not drain before "
                        "uploaded-video completion"
                    ),
                    "pending_retry_count": (
                        len(self._retry_entries(state)) if state else 0
                    ),
                }
            time.sleep(0.01)

    @staticmethod
    def _local_motion_score(previous_frame, current_frame, bbox) -> float:
        if (
            previous_frame is None
            or current_frame is None
            or previous_frame.shape[:2] != current_frame.shape[:2]
        ):
            return 1.0
        height, width = current_frame.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        box_w = max(4, x2 - x1)
        box_h = max(4, y2 - y1)
        x1 = max(0, x1 - box_w // 2)
        x2 = min(width, x2 + box_w // 2)
        y1 = max(0, y1 - box_h)
        y2 = min(height, y2 + box_h)
        if x2 - x1 < 12 or y2 - y1 < 8:
            return 0.0
        before = previous_frame[y1:y2, x1:x2]
        after = current_frame[y1:y2, x1:x2]
        target_w = min(128, max(32, after.shape[1]))
        target_h = min(72, max(18, after.shape[0]))
        before = cv2.resize(before, (target_w, target_h), interpolation=cv2.INTER_AREA)
        after = cv2.resize(after, (target_w, target_h), interpolation=cv2.INTER_AREA)
        before = cv2.GaussianBlur(cv2.cvtColor(before, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        after = cv2.GaussianBlur(cv2.cvtColor(after, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        difference = cv2.absdiff(before, after)
        changed = cv2.threshold(difference, 18, 255, cv2.THRESH_BINARY)[1]
        changed = cv2.morphologyEx(
            changed,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        )
        return float(cv2.countNonZero(changed)) / max(1, changed.size)

    @staticmethod
    def _overlay_region_key(bbox, frame) -> tuple:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (float(value) for value in bbox)
        center_x = (x1 + x2) * 0.5 / max(1.0, width)
        center_y = (y1 + y2) * 0.5 / max(1.0, height)
        box_w = max(1.0, x2 - x1) / max(1.0, width)
        box_h = max(1.0, y2 - y1) / max(1.0, height)
        return (
            int(round(center_x * 24)),
            int(round(center_y * 16)),
            int(round(box_w * 32)),
            int(round(box_h * 32)),
        )

    def _overlay_candidates(
        self,
        state,
        display_rows,
        min_confidence,
        frame,
    ) -> list[dict]:
        # Publish strong, complete reads immediately on the live image while
        # keeping review/experimental guesses hidden. Repeated low-motion
        # detections at one coordinate are remembered as static hard negatives
        # for 25 seconds. Strong OCR can still keep a genuinely parked vehicle
        # visible, including at night.
        selected = []
        now = time.monotonic()
        for key, until in list(state.static_overlay_blocked_until.items()):
            if float(until) <= now:
                state.static_overlay_blocked_until.pop(key, None)
                state.static_overlay_hits.pop(key, None)

        for source in display_rows:
            row = dict(source)
            bbox = row.get("tracking_bbox") or row.get("bbox")
            if not bbox:
                continue
            normalized = normalize_plate(
                row.get("plate_norm")
                or row.get("raw_guess_norm")
                or row.get("plate")
            )
            if (
                len(normalized) != 8
                or not row.get("valid")
                or row.get("needs_review")
                or row.get("experimental")
            ):
                continue

            votes = max(3, int(row.get("consensus_votes", 0)))
            combined_confidence = float(row.get("confidence", 0.0))
            raw_detector_confidence = row.get("detector_confidence")
            detector_confidence = (
                combined_confidence
                if raw_detector_confidence is None
                else float(raw_detector_confidence)
            )
            raw_ocr_confidence = row.get("ocr_confidence")
            if raw_ocr_confidence is None:
                raw_ocr_confidence = row.get("raw_guess_confidence")
            ocr_confidence = (
                combined_confidence
                if raw_ocr_confidence is None
                else float(raw_ocr_confidence)
            )
            method = str(row.get("method", "")).lower()
            if method.startswith("opencv"):
                continue
            if (
                detector_confidence < max(0.32, float(min_confidence) * 0.52)
                or ocr_confidence < 0.40
                or combined_confidence < max(0.38, float(min_confidence) * 0.62)
            ):
                continue

            motion_score = self._local_motion_score(
                state.latest_detection_frame,
                frame,
                bbox,
            )
            region_key = self._overlay_region_key(bbox, frame)
            strong_static_read = bool(
                votes >= 5
                and detector_confidence >= 0.62
                and ocr_confidence >= 0.70
                and combined_confidence >= 0.70
            )
            blocked_until = float(
                state.static_overlay_blocked_until.get(region_key, 0.0)
            )
            if blocked_until > now and motion_score < 0.025 and not strong_static_read:
                continue

            if motion_score < 0.010 and not strong_static_read:
                hits = int(state.static_overlay_hits.get(region_key, 0)) + 1
                state.static_overlay_hits[region_key] = hits
                if hits >= 2:
                    state.static_overlay_blocked_until[region_key] = now + 25.0
                continue
            state.static_overlay_hits[region_key] = max(
                0,
                int(state.static_overlay_hits.get(region_key, 0)) - 2,
            )
            row["bbox"] = tuple(bbox)
            row["experimental"] = False
            row["needs_review"] = False
            row["local_motion_score"] = round(motion_score, 5)
            selected.append(row)
        return selected

    @staticmethod
    def _merge_payload_wake(selected, discarded):
        """Keep wake intent while using masks from the selected frame only."""

        selected_activity = selected[5] if len(selected) > 5 else None
        discarded_activity = discarded[5] if len(discarded) > 5 else None
        discarded_wake = bool(
            discarded_activity is not None
            and discarded_activity.wake_inference
        )
        if not discarded_wake:
            return selected
        if selected_activity is None:
            from .activity import FrameActivity

            merged_activity = FrameActivity(
                motion_score=0.0,
                moving=False,
                scene_change=False,
                wake_inference=True,
                exclusion_mask=None,
            )
        elif selected_activity.wake_inference:
            return selected
        else:
            merged_activity = replace(
                selected_activity,
                wake_inference=True,
            )
        values = list(selected)
        while len(values) <= 5:
            values.append(None)
        values[5] = merged_activity
        return tuple(values)

    def _claim_latest_payload(self, state: _CameraState, scheduled):
        """Replace an executor-queued frame with the newest pending frame.

        ``busy`` remains owned by the already scheduled Future. New submits
        therefore keep replacing one per-camera pending slot instead of
        creating a second Future for the same camera.
        """

        with self._lock:
            pending = state.pending
            if pending is None:
                return scheduled
            state.pending = None
            state.coalesced_frames += 1
            current_generation = self._detector_generation
            scheduled_generation = (
                int(scheduled[6])
                if len(scheduled) > 6
                else current_generation
            )
            pending_generation = (
                int(pending[6])
                if len(pending) > 6
                else current_generation
            )
            if (
                pending_generation == current_generation
                and scheduled_generation != current_generation
            ):
                selected, discarded = pending, scheduled
            elif (
                scheduled_generation == current_generation
                and pending_generation != current_generation
            ):
                selected, discarded = scheduled, pending
            elif (
                scheduled_generation == current_generation
                and pending_generation == current_generation
                and float(pending[3]) >= float(scheduled[3])
            ):
                selected, discarded = pending, scheduled
            else:
                # If both payloads are obsolete, keep the scheduled one so
                # the existing generation guard discards it fail-closed.
                selected, discarded = scheduled, pending
            return self._merge_payload_wake(selected, discarded)

    def _process(self, state: _CameraState, payload):
        payload = self._claim_latest_payload(state, payload)
        camera_id, camera_name, frame, timestamp = payload[:4]
        # Resolve the worker-stable lock before taking model_switch_lock.
        # invalidate_model_cache takes those locks in the opposite outer
        # scope, so looking this up from inside the commit section would
        # create a lock-order inversion.
        event_commit_lock = self._event_commit_lock(camera_id)
        activity = payload[5] if len(payload) > 5 else None
        detector_generation = (
            int(payload[6])
            if len(payload) > 6
            else self._detector_generation
        )
        observed_at_epoch = (
            float(payload[7]) if len(payload) > 7 else time.time()
        )
        model_switch_locked = False
        started = time.perf_counter()
        try:
            config = state.config or {}
            source, offset_x, offset_y = self._roi_frame(
                frame,
                config,
            )
            min_confidence = max(
                0.01,
                min(
                    0.99,
                    float(config.get("lpr_confidence", 60)) / 100.0,
                ),
            )
            exclusion_mask = (
                activity.exclusion_mask
                if activity is not None
                else None
            )

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )
            detector_variant = self._selected_detector_variant()
            inference_metadata = {}

            def baseline_process():
                kwargs = {
                    "engine_key": camera_id,
                    # Tracking and pending-frame state remain per camera, but
                    # ONNX detector/OCR sessions are shared service-wide. This
                    # bounds RAM on installations with many cameras.
                    "inference_key": ENGINE_V3_INFERENCE_KEY,
                    # This call-owned dict carries the detector revision even
                    # when the graph returns no boxes. A global status object
                    # would race across concurrent cameras.
                    "runtime_metadata": inference_metadata,
                }
                if exclusion_mask is not None:
                    kwargs["exclusion_mask"] = exclusion_mask
                # Limit expensive OCR work without changing process_frame's
                # signature. Only the strongest two candidates continue.
                return process_frame(
                    source,
                    live_detection_threshold,
                    max_candidates=2,
                    detector_variant=detector_variant,
                    **kwargs,
                )

            inference_started = time.monotonic()
            try:
                raw_primary_rows = baseline_process()
            finally:
                inference_seconds = time.monotonic() - inference_started
                state.inference_calls += 1
                state.inference_seconds += inference_seconds
                state.last_inference_ms = inference_seconds * 1000.0

            primary_rows = []
            for raw_row in raw_primary_rows:
                translated = self._translate(
                    raw_row,
                    offset_x,
                    offset_y,
                )
                # Corrections learned from an older/custom OCR model are not
                # portable evidence for Hezar v2 or fixed Platrix. Preserve
                # the model read exactly; operator feedback remains available
                # to training and diagnostics without mutating production.
                if translated.get("ocr_engine") in {
                    "hezar-crnn-fa-v2-onnx",
                    "platrix-crnn-onnx",
                }:
                    row = translated
                    row["learned_correction_eligible"] = False
                else:
                    row = apply_learned_correction(translated)
                row["engine_lane"] = "baseline"
                row["detector_variant"] = detector_variant
                row["detector_selection_exclusive"] = True
                primary_rows.append(row)

            # Serialize only the result-commit phase against a detector
            # switch. Inference can remain parallel across cameras. If the
            # setting changed while this frame was running, its old-model
            # observations are discarded before tracker/persistence state.
            state.model_switch_lock.acquire()
            model_switch_locked = True
            if (
                state.retired
                or detector_generation != self._detector_generation
            ):
                return
            rows = primary_rows
            display_rows = rows
            detector_revisions = {
                str(row.get("detector_model_revision", "")).strip()
                for row in rows
                if str(row.get("detector_model_revision", "")).strip()
            }
            call_detector_revision = str(
                inference_metadata.get("detector_model_revision", "")
            ).strip()
            if call_detector_revision:
                detector_revisions.add(call_detector_revision)
            if len(detector_revisions) > 1:
                raise RuntimeError(
                    "one inference returned mixed detector revisions"
                )
            if detector_revisions:
                detector_revision = next(iter(detector_revisions))
                if (
                    state.detector_model_revision
                    and detector_revision != state.detector_model_revision
                ):
                    # A content-addressed YOLOX manifest may be activated while
                    # the service is running. Never let old/new detector crops
                    # contribute to the same temporal identity.
                    duplicate_seconds = max(
                        0.0,
                        float(config.get("duplicate_seconds", 5.0)),
                    )
                    state.tracker = PlateConsensusTracker(
                        min_votes=2,
                        max_age_seconds=2.2,
                        emit_cooldown=duplicate_seconds,
                        emit_unreadable=True,
                    )
                    state.visits.reset_tracker_bindings()
                    state.track_event_ids.clear()
                    state.retry_lineages.clear()
                    state.latest_detections = []
                    state.latest_detection_frame = None
                    state.detection_revision += 1
                    state.static_overlay_hits.clear()
                    state.static_overlay_blocked_until.clear()
                    state.plate_visible = False
                state.detector_model_revision = detector_revision
            processing_seconds = time.perf_counter() - started
            if state.processing_seconds_ema:
                state.processing_seconds_ema = (
                    state.processing_seconds_ema * 0.70
                    + processing_seconds * 0.30
                )
            else:
                state.processing_seconds_ema = processing_seconds
            state.tracker.max_age_seconds = max(
                2.4,
                min(6.0, state.processing_seconds_ema * 2.0 + 1.0),
            )
            state.processed_frames += 1
            state.detected_candidates += len(rows)
            for row in rows:
                state.whole_plate_ocr_attempts += int(
                    bool(row.get("whole_plate_ocr_attempted"))
                )
                state.ocr_agreements += int(
                    str(row.get("ocr_engine", "")).startswith(
                        "multi-engine-agreement"
                    )
                )
                state.ocr_disagreements += int(
                    bool(row.get("ocr_disagreement"))
                )
                state.crnn_selected += int(
                    row.get("ocr_engine") in {
                        "hezar-crnn-fa-v2-onnx",
                        "crnn-onnx",
                        "platrix-crnn-onnx",
                    }
                )
                state.character_reader_selected += int(
                    row.get("ocr_engine") in {
                        "dedicated-character-detector",
                        "cnn-onnx",
                    }
                )
            if rows:
                if not state.plate_visible:
                    state.burst_frames_remaining = 3
                elif state.burst_frames_remaining:
                    state.burst_frames_remaining -= 1
                state.plate_visible = True
                state.no_plate_streak = 0
            else:
                state.plate_visible = False
                if state.burst_frames_remaining:
                    state.burst_frames_remaining -= 1
                if activity is not None and activity.wake_inference:
                    state.no_plate_streak = 0
                else:
                    state.no_plate_streak = min(
                        12,
                        state.no_plate_streak + 1,
                    )
            duplicate_seconds = max(
                0.0,
                float(config.get("duplicate_seconds", 30)),
            )
            stable = state.tracker.update(
                rows,
                timestamp=timestamp,
                frame=frame,
                min_emit_confidence=min_confidence,
            )
            active_tracks = state.tracker.active_track_ids()
            retired_tracks = state.visits.observe(
                rows,
                active_tracks,
                timestamp,
                duplicate_seconds,
            )
            if retired_tracks:
                state.tracker.retire_tracks(retired_tracks)
                for track_id in retired_tracks:
                    state.track_event_ids.pop(track_id, None)
                active_tracks = state.tracker.active_track_ids()
            state.track_event_ids.update(
                state.visits.track_event_refs()
            )
            stable = [
                auto_confirm_guess(row)
                if (
                    row.get("assisted_candidate")
                    and not row.get("capture_only")
                )
                else row
                for row in stable
            ]
            self._observe_engine_v2_baseline(
                camera_id,
                stable,
                timestamp,
                state,
            )
            overlay_rows = self._overlay_candidates(
                state,
                display_rows,
                min_confidence,
                frame,
            )
            state.latest_detections = [
                    {
                        "bbox": tuple(
                            row.get("tracking_bbox")
                            or row["bbox"]
                        ),
                        "plate": row.get("plate", "ناخوانا"),
                        "confidence": float(row.get("confidence", 0.0)),
                        "track_id": int(row.get("track_id") or 0),
                        "tracking_engine": row.get(
                            "tracking_engine",
                            "bytetrack-kalman+optical-flow",
                        ),
                        "valid": bool(row.get("valid")),
                        "best_effort": bool(
                            row.get("best_effort")
                        ),
                        "needs_review": bool(
                            row.get("needs_review")
                        ),
                        "ocr_engine": row.get(
                            "ocr_engine",
                            "",
                        ),
                        "ocr_alternative": row.get(
                            "ocr_alternative",
                            "",
                        ),
                        "ocr_disagreement": bool(
                            row.get("ocr_disagreement")
                        ),
                        "raw_guess_text": row.get(
                            "raw_guess_text",
                            row.get("plate", ""),
                        ),
                        "raw_guess_confidence": float(
                            row.get(
                                "raw_guess_confidence",
                                row.get("ocr_confidence", 0.0),
                            )
                        ),
                        "raw_guess_reason": row.get(
                            "raw_guess_reason",
                            "",
                        ),
                        "model_revision": row.get(
                            "model_revision",
                            row.get("ocr_engine", ""),
                        ),
                        "engine_lane": row.get(
                            "engine_lane",
                            "baseline",
                        ),
                        "experimental": bool(
                            row.get("experimental")
                        ),
                    }
                    for row in overlay_rows
                ]
            state.latest_detection_frame = frame.copy()
            state.latest_detections_at = time.time()
            # Empty inference is also a new display state. Publishing its
            # revision clears an old box immediately instead of leaving it on
            # screen until a wall-clock timeout.
            state.detection_revision += 1
            state.last_processed_at = time.time()
            state.last_processing_ms = processing_seconds * 1000.0
            processing_ms = processing_seconds * 1000.0
            for result in stable:
                if state.retired:
                    return
                result = dict(result)
                track_id = int(result.get("track_id") or 0)
                event_id = state.track_event_ids.get(track_id)
                if (
                    result.get("capture_only")
                    and result.get("provisional")
                ):
                    # A volatile track fragment is not a durable event.  Its
                    # best frame remains in the tracker until consensus or a
                    # final unreadable result can own exactly one row.
                    continue
                if not result.get("capture_only"):
                    result = camera_confidence_result(
                        result,
                        min_confidence,
                    )
                    if result is None:
                        continue
                if (
                    event_id is not None
                    and not state.visits.can_reuse_track_event(
                        track_id,
                        result,
                    )
                ):
                    state.track_event_ids.pop(track_id, None)
                    event_id = None
                key, visit_event_id = state.visits.event_ref(
                    result,
                    timestamp,
                    duplicate_seconds,
                    allow_candidate=True,
                )
                if visit_event_id is not None:
                    event_id = visit_event_id
                if (
                    event_id is not None
                    and result.get("visit_identity_stable") is False
                ):
                    # The raw observation already refreshed the bound visit.
                    # Never let a one-frame OCR flicker downgrade or relabel
                    # an existing durable event, even after a tracker split.
                    continue
                if event_id is not None and not key:
                    # The raw observation already refreshed this active
                    # visit. A finalized OCR-less fragment must not erase a
                    # complete strict or review-only candidate previously
                    # stored for it.
                    continue
                capture_frame = result.pop("capture_frame", None)
                persistence_frame = (
                    capture_frame
                    if capture_frame is not None
                    and getattr(capture_frame, "size", 0)
                    else frame
                )
                if not result.get("capture_only") and not key:
                    continue
                self._enqueue_persistence_retry(
                    state,
                    self._make_persistence_retry(
                        camera_id,
                        camera_name,
                        result,
                        persistence_frame,
                        event_id,
                        key,
                        timestamp,
                        processing_ms,
                        duplicate_seconds,
                        detector_generation,
                        state.detector_model_revision,
                        observed_at_epoch,
                    ),
                )
            persistence_error = self._drain_persistence_retry_locked(
                state,
                event_commit_lock,
            )
            state.track_event_ids = {
                track_id: event_id
                for track_id, event_id in state.track_event_ids.items()
                if track_id in active_tracks
            }
            if not persistence_error and not state.persistence_retry:
                state.last_error = ""
        except Exception as exc:
            if detector_generation == self._detector_generation:
                error = f"{type(exc).__name__}: {exc}"
                state.last_error = error
                state.processing_errors += 1
                state.last_processing_error = error
        finally:
            if model_switch_locked:
                state.model_switch_lock.release()
            with self._lock:
                if (
                    state.retired
                    or detector_generation != self._detector_generation
                ):
                    state.pending = None
                    state.next_inference_at = 0.0
                    state.busy = False
                else:
                    # Always leave real idle time after an expensive
                    # transaction. Previously a queued frame was submitted
                    # immediately here, keeping detector/OCR threads busy even
                    # when every inference returned no plate.
                    state.next_inference_at = time.monotonic() + (
                        0.04
                        if (
                            state.burst_frames_remaining
                            or (
                                activity is not None
                                and activity.wake_inference
                            )
                        )
                        else self._post_inference_delay(
                            state.processing_seconds_ema,
                            state.no_plate_streak,
                        )
                    )
                    state.busy = False

    def status(self, camera_id: int) -> dict:
        with self._lock:
            camera_id = int(camera_id)
            state = self._states.get(camera_id)
            detached = [
                retry_state
                for detached_camera_id, retry_state
                in self._detached_retry_states
                if int(detached_camera_id) == camera_id
            ]
            detached_retry_count = sum(
                len(self._retry_entries(retry_state))
                for retry_state in detached
            )
            detached_retry_bytes = sum(
                self._retry_memory_bytes(retry_state)
                for retry_state in detached
            )
            if not state:
                return {
                    "active": bool(detached_retry_count),
                    "received_frames": 0,
                    "processed_frames": 0,
                    "inference_calls": 0,
                    "inference_seconds": 0.0,
                    "mean_inference_ms": 0.0,
                    "last_inference_ms": 0.0,
                    "inference_fps": 0.0,
                    "coalesced_frames": 0,
                    "worker_frame_drop_rate": 0.0,
                    "detected_candidates": 0,
                    "emitted_events": 0,
                    "pending_retry_count": detached_retry_count,
                    "pending_retry_bytes": detached_retry_bytes,
                    "persistence_backpressure": bool(
                        detached_retry_count or self._outbox_error
                    ),
                    "persistence_backpressure_frames": 0,
                    "retry_outbox_error": self._outbox_error,
                    "retry_outbox_quarantined": (
                        self._outbox_quarantined
                    ),
                    "detector_model_revision": "",
                    "last_error": (
                        next(
                            (
                                entry.last_error
                                for retry_state in detached
                                for entry in self._retry_entries(retry_state)
                                if entry.last_error
                            ),
                            "",
                        )
                        or self._outbox_error
                    ),
                    "anpr_engine": self._exclusive_engine_status(),
                    "shadow": self._shadow_status(
                        camera_id,
                    ),
                    "models": self._models(),
                    "ocr_ab": {
                        "whole_plate_attempts": 0,
                        "agreements": 0,
                        "disagreements": 0,
                        "crnn_selected": 0,
                        "character_reader_selected": 0,
                    },
                    "threads_per_camera": threads_per_camera(),
                    "parallel_camera_limit": self._worker_capacity,
                }
            metrics_elapsed = max(
                0.0,
                time.monotonic() - state.metrics_started_at,
            )
            return {
                "active": bool(state.busy or state.config),
                "received_frames": state.frame_counter,
                "processed_frames": state.processed_frames,
                "inference_calls": state.inference_calls,
                "inference_seconds": round(state.inference_seconds, 6),
                "mean_inference_ms": round(
                    state.inference_seconds * 1000.0
                    / max(1, state.inference_calls),
                    6,
                ),
                "last_inference_ms": round(state.last_inference_ms, 6),
                "inference_fps": round(
                    state.inference_calls / max(metrics_elapsed, 1e-9),
                    6,
                ),
                "coalesced_frames": state.coalesced_frames,
                "worker_frame_drop_rate": round(
                    state.coalesced_frames / max(1, state.frame_counter),
                    6,
                ),
                "detected_candidates": state.detected_candidates,
                "emitted_events": state.emitted_events,
                "pending_retry_count": (
                    len(self._retry_entries(state)) + detached_retry_count
                ),
                "pending_retry_bytes": (
                    self._retry_memory_bytes(state)
                    + detached_retry_bytes
                ),
                "persistence_backpressure": bool(
                    state.persistence_backpressure
                    or detached_retry_count
                    or self._outbox_error
                ),
                "persistence_backpressure_frames": (
                    state.persistence_backpressure_frames
                ),
                "retry_outbox_error": self._outbox_error,
                "retry_outbox_quarantined": self._outbox_quarantined,
                "detector_model_revision": state.detector_model_revision,
                "last_event_at": state.last_event_at,
                "last_processed_at": state.last_processed_at,
                "last_processing_ms": round(
                    state.last_processing_ms,
                    1,
                ),
                "idle_mode": bool(state.no_plate_streak >= 2),
                "no_plate_streak": state.no_plate_streak,
                "next_inference_seconds": round(
                    max(0.0, state.next_inference_at - time.monotonic()),
                    2,
                ),
                "burst_frames_remaining": state.burst_frames_remaining,
                "motion_score": round(state.motion_score, 5),
                "motion_wakeups": state.motion_wakeups,
                "overlay_mask_pixels": state.overlay_mask_pixels,
                "anpr_engine": self._exclusive_engine_status(),
                "shadow": self._shadow_status(
                    camera_id,
                    state,
                ),
                "consensus_window_seconds": round(
                    state.tracker.max_age_seconds,
                    2,
                ),
                "last_error": state.last_error or self._outbox_error,
                "models": self._models(),
                "ocr_ab": {
                    "whole_plate_attempts": (
                        state.whole_plate_ocr_attempts
                    ),
                    "agreements": state.ocr_agreements,
                    "disagreements": state.ocr_disagreements,
                    "crnn_selected": state.crnn_selected,
                    "character_reader_selected": (
                        state.character_reader_selected
                    ),
                },
                "threads_per_camera": threads_per_camera(),
                "parallel_camera_limit": self._worker_capacity,
            }

    def detections(self, camera_id: int, max_age=1.6) -> list:
        with self._lock:
            state = self._states.get(int(camera_id))
            baseline = []
            if (
                state
                and time.time() - state.latest_detections_at
                <= float(max_age)
            ):
                baseline = state.latest_detections
            return self._merge_shadow_detections(camera_id, baseline)

    def detection_snapshot(
        self,
        camera_id: int,
        after_revision=0,
        max_age=3.0,
    ) -> dict:
        with self._lock:
            state = self._states.get(int(camera_id))
            if not state:
                return {"revision": 0, "detections": [], "frame": None}
            if (
                state.detection_revision <= int(after_revision)
                or time.time() - state.latest_detections_at > float(max_age)
            ):
                return {
                    "revision": state.detection_revision,
                    "detections": [],
                    "frame": None,
                }
            return {
                "revision": state.detection_revision,
                "detections": self._merge_shadow_detections(
                    camera_id,
                    state.latest_detections,
                ),
                "frame": (
                    state.latest_detection_frame.copy()
                    if state.latest_detection_frame is not None
                    else None
                ),
                "max_age": max(
                    0.75,
                    min(
                        2.0,
                        state.processing_seconds_ema * 1.20 + 0.45,
                    ),
                ),
            }

    def remove(self, camera_id: int, retry_timeout=1.0) -> bool:
        camera_id = int(camera_id)
        with self._lock:
            state = self._states.pop(camera_id, None)
        try:
            from app.engine_v2.live_shadow import stop_live_shadow_camera

            stop_live_shadow_camera(camera_id)
        except Exception:
            pass
        if state is None:
            return True
        # The commit lock is stable for the lifetime of this detached state.
        # An inference already past its commit boundary may finish first; an
        # inference still running is marked retired and must discard output.
        with state.model_switch_lock:
            state.retired = True
            state.pending = None
        flushed = self._flush_persistence_retry(
            camera_id,
            state,
            deadline=(
                time.monotonic()
                + max(0.0, min(10.0, float(retry_timeout)))
            ),
            allow_retired=True,
        )
        if not flushed:
            # Keep ownership explicit so a later shutdown flush can retry;
            # removing a camera must never silently discard an event.
            with self._lock:
                self._detached_retry_states.append((camera_id, state))
            self._retry_wakeup.set()
        return flushed

    def shutdown(self, retry_timeout=2.0) -> bool:
        self._stopped = True
        if self._shadow_enabled_cache:
            try:
                from app.engine_v2.live_shadow import shutdown_live_shadow

                shutdown_live_shadow()
            except Exception:
                pass
        self._retry_stop.set()
        self._retry_wakeup.set()
        if (
            self._retry_thread is not None
            and self._retry_thread is not threading.current_thread()
        ):
            self._retry_thread.join(
                timeout=max(0.0, min(2.0, float(retry_timeout)))
            )
        self._executor.shutdown(
            wait=True,
            cancel_futures=False,
        )
        with self._lock:
            states = [
                (camera_id, state)
                for camera_id, state in self._states.items()
            ]
            states.extend(self._detached_retry_states)
        queued_states = []
        for camera_id, state in states:
            with state.model_switch_lock:
                state.retired = True
                state.pending = None
            if self._retry_entries(state):
                queued_states.append((camera_id, state))
        if not queued_states:
            with self._lock:
                self._detached_retry_states = []
            return not bool(
                self._retry_thread is not None
                and self._retry_thread.is_alive()
            )
        deadline = (
            time.monotonic()
            + max(0.0, min(30.0, float(retry_timeout)))
        )
        flushed_all = True
        remaining_detached = []
        for camera_id, state in queued_states:
            flushed = self._flush_persistence_retry(
                camera_id,
                state,
                deadline=deadline,
                allow_retired=True,
            )
            flushed_all = bool(flushed_all and flushed)
            if not flushed:
                remaining_detached.append((camera_id, state))
        with self._lock:
            self._detached_retry_states = remaining_detached
        return bool(
            flushed_all
            and not (
                self._retry_thread is not None
                and self._retry_thread.is_alive()
            )
        )

    def backup_retry_outbox(self, destination) -> Path:
        """Snapshot the durable retry queue after inference is quiesced."""

        if not self._stopped:
            raise RuntimeError(
                "live ANPR worker must be stopped before outbox backup"
            )
        if (
            self._retry_thread is not None
            and self._retry_thread.is_alive()
        ):
            raise RuntimeError(
                "live ANPR retry thread must stop before outbox backup"
            )
        if self._outbox is None:
            raise RuntimeError(
                self._outbox_error or "retry outbox is unavailable"
            )
        return self._outbox.backup(destination)


_GLOBAL_OUTBOX_PATH = DATA_DIR / "bcvision-retry.db"
_GLOBAL_WORKER_LOCK = threading.RLock()
worker = LiveANPRWorker(
    background_retry=True,
    retry_outbox_path=_GLOBAL_OUTBOX_PATH,
    _defer_persistence_start=True,
)


def start_live_anpr_worker():
    global worker
    with _GLOBAL_WORKER_LOCK:
        if not getattr(worker, "_persistence_started", True):
            worker._start_persistence_lifecycle()
        if worker._stopped or (
            worker._outbox_required and worker._outbox is None
        ):
            if not worker._stopped:
                worker.shutdown(retry_timeout=0.0)
            worker = LiveANPRWorker(
                background_retry=True,
                retry_outbox_path=_GLOBAL_OUTBOX_PATH,
            )
        return worker


def shutdown_live_anpr_worker(retry_timeout=5.0):
    with _GLOBAL_WORKER_LOCK:
        if not getattr(worker, "_persistence_started", True):
            return True
        if worker._stopped:
            return bool(
                not worker._detached_retry_states
                and not (
                    worker._retry_thread is not None
                    and worker._retry_thread.is_alive()
                )
            )
        return worker.shutdown(retry_timeout=retry_timeout)


def backup_live_anpr_outbox(destination):
    with _GLOBAL_WORKER_LOCK:
        return worker.backup_retry_outbox(destination)


def _running_live_anpr_worker():
    current = worker
    if (
        not getattr(current, "_persistence_started", True)
        or getattr(current, "_stopped", False)
        or (
            getattr(current, "_outbox_required", False)
            and getattr(current, "_outbox", None) is None
        )
    ):
        return start_live_anpr_worker()
    return current


def submit_live_frame(camera_id, camera_name, frame):
    _running_live_anpr_worker().submit(camera_id, camera_name, frame)


def begin_live_video_pass(camera_id):
    return _running_live_anpr_worker().begin_video_pass(camera_id)


def drain_live_video_pass(camera_id, pass_token=None, timeout=60.0):
    return _running_live_anpr_worker().drain_video_pass(
        camera_id,
        pass_token=pass_token,
        timeout=timeout,
    )


def live_anpr_status(camera_id):
    return _running_live_anpr_worker().status(camera_id)


def live_anpr_detections(camera_id):
    return _running_live_anpr_worker().detections(camera_id)


def live_anpr_detection_snapshot(camera_id, after_revision=0):
    return _running_live_anpr_worker().detection_snapshot(
        camera_id,
        after_revision,
    )


def stop_live_camera(camera_id):
    return _running_live_anpr_worker().remove(camera_id)


def configure_live_engine_v2_shadow(enabled):
    _running_live_anpr_worker().configure_engine_v2_shadow(bool(enabled))


def invalidate_live_anpr_model_cache(
    detector_variant=None,
    persist_setting=None,
):
    _running_live_anpr_worker().invalidate_model_cache(
        detector_variant=detector_variant,
        persist_setting=persist_setting,
    )


def switch_live_anpr_detector(
    detector_variant,
    persist_setting=None,
):
    _running_live_anpr_worker().invalidate_model_cache(
        detector_variant=detector_variant,
        persist_setting=persist_setting,
    )
