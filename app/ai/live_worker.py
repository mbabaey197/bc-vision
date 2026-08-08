"""Asynchronous live-camera ANPR worker.

Streaming threads submit the newest frame without blocking. Per-camera workers
apply ROI, multi-frame consensus, duplicate suppression, and persist events.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import math
import threading
import time

import cv2

from app.cpu_budget import parallel_camera_limit, threads_per_camera
from app.media_storage import save_event_images

from .activity import FrameActivityAnalyzer
from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    bbox_iou,
    process_frame,
)
from .plate_rules import normalize_plate, split_iran_plate
from .feedback import apply_learned_correction
from .frame_budget import calculate_frame_budget
from .next_engine import engine_router
from .review_policy import (
    auto_confirm_guess,
    tag_assisted_candidate,
)


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


@dataclass
class _CameraState:
    lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
    )
    frame_counter: int = 0
    busy: bool = False
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
    track_event_ids: dict[int, int] = field(default_factory=dict)
    last_error: str = ""
    last_event_at: float = 0.0
    processed_frames: int = 0
    detected_candidates: int = 0
    emitted_events: int = 0
    whole_plate_ocr_attempts: int = 0
    ocr_agreements: int = 0
    ocr_disagreements: int = 0
    crnn_selected: int = 0
    character_reader_selected: int = 0
    last_processed_at: float = 0.0
    last_processing_ms: float = 0.0
    processing_seconds_ema: float = 0.0
    processing_samples: deque = field(
        default_factory=lambda: deque(maxlen=60)
    )
    positive_processing_samples: deque = field(
        default_factory=lambda: deque(maxlen=60)
    )
    no_plate_streak: int = 0
    next_inference_at: float = 0.0
    latest_detections: list = field(default_factory=list)
    latest_detections_at: float = 0.0
    latest_detection_frame: object | None = None
    detection_revision: int = 0
    last_submitted_at: float = 0.0
    last_received_at: float = 0.0
    source_fps_ema: float = 0.0
    source_max_gap_seconds: float = 0.0
    source_p95_gap_seconds: float = 0.0
    source_timestamps: deque = field(
        default_factory=lambda: deque(maxlen=240)
    )
    pending_replacements: int = 0
    admission_drops: int = 0
    expired_frames: int = 0
    pending_timer: threading.Timer | None = field(
        default=None,
        repr=False,
    )
    target_burst_fps: float = 0.0
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
    config_generation: int = 0
    cancelled: bool = False
    processing_done: threading.Event = field(
        default_factory=threading.Event
    )


class LiveANPRWorker:
    def __init__(self, max_workers=None):
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
        self._stopped = False
        self._model_state = {}
        self._model_state_at = 0.0

    def _models(self) -> dict:
        now = time.monotonic()
        if now - self._model_state_at >= 30.0:
            try:
                from .model_manager import model_status
                self._model_state = model_status()
            except Exception as exc:
                self._model_state = {
                    "detector_ready": False,
                    "crnn_ready": False,
                    "cnn_ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._model_state_at = now
        status = dict(self._model_state)
        status["ocr_ready"] = bool(
            status.get("crnn_ready")
            or status.get("cnn_ready")
        )
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
        rx = max(0.0, min(99.9, float(config.get("roi_x", 0))))
        ry = max(0.0, min(99.9, float(config.get("roi_y", 0))))
        rw = max(0.1, min(100.0 - rx, float(config.get("roi_w", 100))))
        rh = max(0.1, min(100.0 - ry, float(config.get("roi_h", 100))))
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
    def _is_uploaded_video(config) -> bool:
        """Return whether a camera is backed by a user-uploaded video.

        Uploaded files are an offline verification source, not a calibrated
        copy of the camera whose confidence settings were selected at upload
        time.  In particular, that camera's ROI and motion history must never
        make an unrelated file invisible to ANPR.
        """

        return str((config or {}).get("rtsp_url", "")).startswith(
            "video://"
        )

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
    ):
        from app.database import connect
        from app.config import PLATE_DIR, SNAPSHOT_DIR

        result = add_vehicle_analysis(result, frame)
        plate_dir = Path(self._setting("plate_path", str(PLATE_DIR)))
        snapshot_dir = Path(
            self._setting("snapshot_path", str(SNAPSHOT_DIR))
        )

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        plate_path = ""
        image_path = ""
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
            if event_id:
                try:
                    existing = con.execute(
                        "SELECT image_path,plate_image_path,city,plate_norm "
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
                    f"plate-live-{stamp}.jpg",
                )
            ),
            vehicle_target=(
                media_target(
                    image_path,
                    snapshot_root,
                    f"vehicle-live-{stamp}.jpg",
                )
            ),
            save_plate=(
                self._setting("save_plate_images", "1") == "1"
            ),
            save_vehicle=(
                self._setting("save_snapshots", "1") == "1"
            ),
            existing_plate_path=plate_path,
            existing_vehicle_path=image_path,
        )
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
            "updated_at": datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            ),
            # Live ANPR events intentionally retain still evidence only.
            # Uploaded video may continue driving a virtual camera, but its
            # path is never attached to individual archived events.
            "video_path": "",
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
        with connect() as con:
            columns = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(plate_events)"
                ).fetchall()
            }
            selected = [key for key in values if key in columns]
            if existing and event_id:
                assignments = ",".join(f"{key}=?" for key in selected)
                con.execute(
                    f"UPDATE plate_events SET {assignments} WHERE id=?",
                    tuple(values[key] for key in selected) + (int(event_id),),
                )
                return int(event_id)
            placeholders = ",".join("?" for _ in selected)
            cursor = con.execute(
                f"INSERT INTO plate_events({','.join(selected)}) "
                f"VALUES({placeholders})",
                tuple(values[key] for key in selected),
            )
            return int(cursor.lastrowid)

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

    @staticmethod
    def _observe_source_rate(state: _CameraState, now: float) -> None:
        if (
            state.last_received_at > 0.0
            and float(now) - state.last_received_at > 2.0
        ):
            state.source_timestamps.clear()
            state.source_fps_ema = 0.0
            state.source_max_gap_seconds = 0.0
            state.source_p95_gap_seconds = 0.0
        state.source_timestamps.append(float(now))
        while (
            len(state.source_timestamps) > 2
            and float(now) - state.source_timestamps[0] > 3.0
        ):
            state.source_timestamps.popleft()
        if len(state.source_timestamps) >= 5:
            elapsed = (
                state.source_timestamps[-1]
                - state.source_timestamps[0]
            )
            # A count-over-time window is robust to RTSP jitter. Averaging
            # reciprocal frame intervals can overstate a 20 FPS source as
            # more than 50 FPS when short and long intervals alternate.
            if elapsed >= 2.0:
                state.source_fps_ema = (
                    (len(state.source_timestamps) - 1) / elapsed
                )
                gaps = sorted(
                    later - earlier
                    for earlier, later in zip(
                        state.source_timestamps,
                        list(state.source_timestamps)[1:],
                    )
                )
                state.source_max_gap_seconds = gaps[-1]
                state.source_p95_gap_seconds = gaps[
                    min(
                        len(gaps) - 1,
                        math.ceil(len(gaps) * 0.95) - 1,
                    )
                ]
        state.last_received_at = float(now)

    @staticmethod
    def _frame_budget(
        state: _CameraState,
        config: dict,
        now: float | None = None,
    ) -> dict:
        with state.lock:
            samples = sorted(
                float(value) for value in state.processing_samples
            )
            positive_samples = sorted(
                float(value)
                for value in state.positive_processing_samples
            )
            observed_now = (
                time.monotonic()
                if now is None
                else float(now)
            )
            telemetry_fresh = (
                state.last_received_at > 0.0
                and observed_now - state.last_received_at <= 2.0
            )
            source_fps = (
                float(state.source_fps_ema)
                if telemetry_fresh
                else 0.0
            )
            source_max_gap_ms = (
                float(state.source_max_gap_seconds) * 1000.0
                if telemetry_fresh
                else 0.0
            )
            source_p95_gap_ms = (
                float(state.source_p95_gap_seconds) * 1000.0
                if telemetry_fresh
                else 0.0
            )
        if len(samples) >= 20 and positive_samples:
            all_p95 = samples[
                min(
                    len(samples) - 1,
                    math.ceil(len(samples) * 0.95) - 1,
                )
            ]
            positive_p95 = (
                positive_samples[
                    min(
                        len(positive_samples) - 1,
                        math.ceil(len(positive_samples) * 0.95) - 1,
                    )
                ]
                if len(positive_samples) >= 20
                else max(positive_samples)
            )
            p95_seconds = max(all_p95, positive_p95)
        else:
            p95_seconds = 0.0
        budget = calculate_frame_budget(
            max_speed_kmh=config.get("max_vehicle_speed_kmh", 150),
            recognition_zone_m=config.get("recognition_zone_m", 10),
            source_fps=source_fps,
            source_max_gap_ms=source_max_gap_ms,
            source_p95_gap_ms=source_p95_gap_ms,
            processing_p95_ms=p95_seconds * 1000.0,
            geometry_calibrated=bool(
                int(config.get("recognition_zone_calibrated", 0) or 0)
            ),
            telemetry_required=True,
        )
        return budget.as_dict()

    @classmethod
    def _burst_interval(
        cls,
        state: _CameraState,
        config: dict,
        now: float | None = None,
    ) -> float:
        plan = cls._frame_budget(state, config, now=now)
        recommended = float(plan["recommended_capture_fps"])
        source_fps = float(plan["source_fps"])
        target = (
            min(source_fps, recommended)
            if source_fps > 0.0
            else recommended
        )
        state.target_burst_fps = max(1.0, min(120.0, target))
        return 1.0 / state.target_burst_fps

    @staticmethod
    def _max_frame_age(config: dict) -> float:
        budget = calculate_frame_budget(
            max_speed_kmh=config.get("max_vehicle_speed_kmh", 150),
            recognition_zone_m=config.get("recognition_zone_m", 10),
        )
        return max(0.08, min(1.50, float(budget.zone_seconds)))

    @staticmethod
    def _with_dispatch_time(payload, dispatched_at: float):
        values = list(payload)
        values[4] = float(dispatched_at)
        return tuple(values)

    @staticmethod
    def _cancel_pending_timer_locked(state: _CameraState) -> None:
        timer = state.pending_timer
        state.pending_timer = None
        if timer is not None:
            timer.cancel()

    def _schedule_pending_locked(
        self,
        camera_id: int,
        state: _CameraState,
        deadline: float,
    ) -> None:
        self._cancel_pending_timer_locked(state)
        delay = max(0.0, float(deadline) - time.monotonic())
        timer = threading.Timer(
            delay,
            self._dispatch_pending,
            args=(int(camera_id), state),
        )
        timer.daemon = True
        state.pending_timer = timer
        timer.start()

    def _dispatch_pending(
        self,
        camera_id: int,
        state: _CameraState,
    ) -> None:
        payload = None
        with state.lock:
            state.pending_timer = None
            if (
                self._stopped
                or state.cancelled
                or state.busy
                or state.pending is None
            ):
                return
            now = time.monotonic()
            if now + 1e-6 < state.next_inference_at:
                self._schedule_pending_locked(
                    camera_id,
                    state,
                    state.next_inference_at,
                )
                return
            payload = state.pending
            state.pending = None
            if now - float(payload[3]) > self._max_frame_age(
                state.config or {}
            ):
                state.expired_frames += 1
                state.admission_drops += 1
                return
            state.last_submitted_at = now
            state.processing_done.clear()
            state.busy = True
            payload = self._with_dispatch_time(payload, now)
        try:
            self._executor.submit(self._process, state, payload)
        except RuntimeError:
            with state.lock:
                state.busy = False
                state.processing_done.set()

    def submit(self, camera_id: int, camera_name: str, frame):
        if (
            self._stopped
            or frame is None
            or getattr(frame, "size", 0) == 0
        ):
            return
        now = time.monotonic()
        with self._lock:
            state = self._states.setdefault(
                int(camera_id),
                _CameraState(),
            )
        with state.lock:
            if state.cancelled:
                return
            state.frame_counter += 1
            self._observe_source_rate(state, now)
            try:
                config = self._config(int(camera_id), state, now)
            except Exception as exc:
                state.last_error = f"{type(exc).__name__}: {exc}"
                return
            if (
                not config
                or not int(config.get("enabled", 0))
                or not int(config.get("lpr_enabled", 0))
            ):
                return
            config_generation = state.config_generation
        # Activity analysis is per camera and intentionally runs outside the
        # worker-wide mapping lock. Four camera decoder threads must not be
        # serialized through resize/blur/morphology/Canny work.
        uploaded_video = self._is_uploaded_video(config)
        try:
            activity_source = (
                frame
                if uploaded_video
                else self._roi_frame(frame, config)[0]
            )
            activity = state.activity.observe(activity_source)
        except Exception as exc:
            with state.lock:
                state.last_error = f"{type(exc).__name__}: {exc}"
            return

        payload_to_submit = None
        with state.lock:
            if (
                state.cancelled
                or state.config_generation != config_generation
            ):
                return
            state.motion_score = float(activity.motion_score)
            state.overlay_mask_pixels = (
                int(cv2.countNonZero(activity.exclusion_mask))
                if activity.exclusion_mask is not None
                else 0
            )
            if activity.wake_inference or uploaded_video:
                if activity.wake_inference:
                    state.motion_wakeups += 1
                plan = self._frame_budget(state, config, now=now)
                planned_observations = max(
                    5,
                    min(
                        8,
                        int(math.ceil(
                            float(plan["recommended_capture_fps"])
                            * float(plan["zone_seconds"])
                        )),
                    ),
                )
                state.burst_frames_remaining = max(
                    state.burst_frames_remaining,
                    planned_observations,
                )
                state.next_inference_at = min(
                    state.next_inference_at,
                    now,
                )
            burst_active = bool(
                activity.wake_inference
                or state.burst_frames_remaining
                or uploaded_video
            )
            payload = (
                int(camera_id),
                str(camera_name),
                frame,
                now,
                0.0,
                activity,
                config_generation,
                uploaded_video,
            )
            if state.busy:
                if state.pending is not None:
                    state.pending_replacements += 1
                # Exactly one latest-frame slot is retained. Quality ranking
                # happens on per-track crops, never by keeping an old frame in
                # the inference queue. Idle work clears this slot on finish;
                # a motion burst consumes it at its start-to-start deadline.
                state.pending = payload
                return
            # Do not let a slow CPU run ANPR continuously with no breathing
            # room. Keep the newest frame and cap inference frequency
            # adaptively; this reduces load without lowering image quality.
            minimum_interval = (
                self._burst_interval(state, config, now=now)
                if (
                    activity.wake_inference
                    or state.burst_frames_remaining
                    or uploaded_video
                )
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
            )
            due_at = max(
                state.last_submitted_at + minimum_interval,
                state.next_inference_at,
            )
            if now + 1e-9 < due_at:
                state.admission_drops += 1
                if burst_active:
                    if state.pending is not None:
                        state.pending_replacements += 1
                    state.pending = payload
                    self._schedule_pending_locked(
                        int(camera_id),
                        state,
                        due_at,
                    )
                else:
                    state.pending = None
                    self._cancel_pending_timer_locked(state)
                return
            if state.pending is not None:
                state.admission_drops += 1
                state.pending = None
            self._cancel_pending_timer_locked(state)
            state.last_submitted_at = now
            state.processing_done.clear()
            state.busy = True
            payload_to_submit = self._with_dispatch_time(payload, now)
        try:
            self._executor.submit(
                self._process,
                state,
                payload_to_submit,
            )
        except RuntimeError:
            with state.lock:
                state.busy = False
                state.processing_done.set()

    def _process(self, state: _CameraState, payload):
        camera_id, camera_name, frame, timestamp = payload[:4]
        has_dispatch_timestamp = bool(
            len(payload) > 4 and payload[4]
        )
        activity = payload[5] if len(payload) > 5 else None
        config_generation = payload[6] if len(payload) > 6 else 0
        uploaded_video = bool(payload[7]) if len(payload) > 7 else False
        started = time.perf_counter()
        model_seconds = None
        positive_path = False
        try:
            with state.lock:
                if state.cancelled:
                    return
                config = dict(state.config or {})
            frame_age = time.monotonic() - float(timestamp)
            if (
                has_dispatch_timestamp
                and frame_age > self._max_frame_age(config)
            ):
                with state.lock:
                    state.expired_frames += 1
                return
            if uploaded_video or self._is_uploaded_video(config):
                # A file selected for verification is always examined in its
                # own full coordinate space.  It must not inherit a gate ROI
                # or fixed-overlay mask learned from another camera.
                uploaded_video = True
                source, offset_x, offset_y = frame, 0, 0
            else:
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
                if activity is not None and not uploaded_video
                else None
            )

            def baseline_process():
                kwargs = {"engine_key": camera_id}
                if exclusion_mask is not None:
                    kwargs["exclusion_mask"] = exclusion_mask
                return process_frame(
                    source,
                    min_confidence * 0.45,
                    **kwargs,
                )

            outcome = engine_router.process(
                source,
                baseline=baseline_process,
                min_detection_confidence=min_confidence * 0.45,
                engine_key=camera_id,
                exclusion_mask=exclusion_mask,
            )
            primary_rows = [
                apply_learned_correction(
                    self._translate(row, offset_x, offset_y)
                )
                for row in outcome.primary
            ]
            shadow_rows = [
                {
                    **self._translate(row, offset_x, offset_y),
                    "engine_lane": "candidate-shadow",
                    "experimental": True,
                    "needs_review": True,
                }
                for row in outcome.shadow
            ]
            assisted_enabled = (
                self._setting("anpr_auto_confirm_guesses", "1") == "1"
            )
            rows = (
                operator_assisted_rows(primary_rows, shadow_rows)
                if outcome.mode == "shadow"
                and assisted_enabled
                and shadow_rows
                else primary_rows
            )
            display_rows = (
                rows
                if outcome.mode == "shadow" and assisted_enabled
                else rows + shadow_rows
            )
            positive_path = bool(rows or shadow_rows)
            # A dashboard ROI change invalidates work already running against
            # the old area.  Such detections must not reach tracking or disk.
            with state.lock:
                if (
                    state.cancelled
                    or state.config_generation != config_generation
                ):
                    return
            if outcome.mode == "shadow":
                state.shadow_frames += 1
                state.shadow_candidates += len(outcome.shadow)
                state.shadow_errors += int(bool(outcome.error))
            model_seconds = time.perf_counter() - started
            state.tracker.max_age_seconds = max(
                2.4,
                min(
                    6.0,
                    max(
                        state.processing_seconds_ema,
                        model_seconds,
                    ) * 2.0 + 1.0,
                ),
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
                    row.get("ocr_engine") == "crnn-onnx"
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
                if uploaded_video:
                    # Empty frames in a file never enter motion backoff.  The
                    # newest decoded frame remains eligible until EOF.
                    state.no_plate_streak = 0
                elif activity is not None and activity.wake_inference:
                    state.no_plate_streak = 0
                else:
                    state.no_plate_streak = min(
                        12,
                        state.no_plate_streak + 1,
                    )
            stable = state.tracker.update(
                rows,
                timestamp=timestamp,
                frame=frame,
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
                    for row in display_rows
                ]
            state.latest_detection_frame = frame.copy()
            state.latest_detections_at = time.time()
            # Empty inference is also a new display state. Publishing its
            # revision clears an old box immediately instead of leaving it on
            # screen until a wall-clock timeout.
            state.detection_revision += 1
            state.last_processed_at = time.time()
            duplicate_seconds = max(
                0.0,
                float(config.get("duplicate_seconds", 30)),
            )
            processing_ms = model_seconds * 1000.0
            for result in stable:
                if state.cancelled:
                    return
                track_id = int(result.get("track_id") or 0)
                event_id = state.track_event_ids.get(track_id)
                capture_frame = result.pop("capture_frame", None)
                persistence_frame = (
                    capture_frame
                    if capture_frame is not None
                    and getattr(capture_frame, "size", 0)
                    else frame
                )
                if result.get("capture_only"):
                    saved_id = self._persist(
                        camera_id,
                        camera_name,
                        persistence_frame,
                        result,
                        processing_ms,
                        event_id,
                    )
                    state.track_event_ids[track_id] = saved_id
                    if event_id is None:
                        state.emitted_events += 1
                        state.last_event_at = time.time()
                    continue
                # A low-confidence read is still valuable as an explicitly
                # reviewable suggestion.  Camera confidence remains the gate
                # for automatic/confirmed reads, not for operator training
                # samples requested by the user.
                if (
                    result["confidence"] < min_confidence
                    and not result.get("needs_review")
                ):
                    continue
                key = result.get("plate_norm") or normalize_plate(
                    result.get("plate")
                )
                if not key:
                    continue
                previous = state.seen.get(key, -1e12)
                if (
                    event_id is None
                    and timestamp - previous < duplicate_seconds
                ):
                    continue
                state.seen[key] = timestamp
                saved_id = self._persist(
                    camera_id,
                    camera_name,
                    persistence_frame,
                    result,
                    processing_ms,
                    event_id,
                )
                state.track_event_ids[track_id] = saved_id
                if event_id is None:
                    state.emitted_events += 1
                state.last_event_at = time.time()
            active_tracks = state.tracker.active_track_ids()
            state.track_event_ids = {
                track_id: event_id
                for track_id, event_id in state.track_event_ids.items()
                if track_id in active_tracks
            }
            state.last_error = ""
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            remove_state = False
            finished_at = time.monotonic()
            # Capture-to-finish latency includes shared-executor waiting,
            # tracking, image copies and persistence. Capacity warnings must
            # reflect the time a camera is really busy, not model time alone.
            end_to_end_seconds = max(
                0.0,
                (
                    finished_at - float(timestamp)
                    if has_dispatch_timestamp
                    else (
                        model_seconds
                        if model_seconds is not None
                        else time.perf_counter() - started
                    )
                ),
            )
            with state.lock:
                state.processing_samples.append(end_to_end_seconds)
                if positive_path:
                    state.positive_processing_samples.append(
                        end_to_end_seconds
                    )
                if state.processing_seconds_ema:
                    state.processing_seconds_ema = (
                        state.processing_seconds_ema * 0.70
                        + end_to_end_seconds * 0.30
                    )
                else:
                    state.processing_seconds_ema = end_to_end_seconds
                state.last_processing_ms = end_to_end_seconds * 1000.0
                burst_active = bool(
                    state.burst_frames_remaining
                    or uploaded_video
                    or (
                        activity is not None
                        and activity.wake_inference
                    )
                )
                if burst_active:
                    interval = self._burst_interval(
                        state,
                        state.config or {},
                        now=finished_at,
                    )
                    # Deadline is start-to-start. If inference already took
                    # longer than the interval, the latest pending frame may
                    # run immediately instead of paying the interval twice.
                    state.next_inference_at = max(
                        finished_at,
                        state.last_submitted_at + interval,
                    )
                else:
                    state.next_inference_at = (
                        finished_at
                        + self._post_inference_delay(
                            state.processing_seconds_ema,
                            state.no_plate_streak,
                        )
                    )
                state.busy = False
                state.processing_done.set()
                if (
                    not state.cancelled
                    and state.pending is not None
                    and burst_active
                ):
                    self._schedule_pending_locked(
                        int(camera_id),
                        state,
                        state.next_inference_at,
                    )
                else:
                    self._cancel_pending_timer_locked(state)
                    if not burst_active:
                        state.pending = None
                remove_state = state.cancelled
            if remove_state:
                with self._lock:
                    if self._states.get(int(camera_id)) is state:
                        self._states.pop(int(camera_id), None)

    def status(self, camera_id: int) -> dict:
        with self._lock:
            state = self._states.get(int(camera_id))
            if not state:
                return {
                    "active": False,
                    "received_frames": 0,
                    "processed_frames": 0,
                    "detected_candidates": 0,
                    "emitted_events": 0,
                    "last_error": "",
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
                    "sampling": calculate_frame_budget(
                        geometry_calibrated=False,
                        telemetry_required=True,
                    ).as_dict(),
                }
            sampling = self._frame_budget(
                state,
                state.config or {},
            )
            return {
                "active": bool(state.busy or state.config),
                "received_frames": state.frame_counter,
                "processed_frames": state.processed_frames,
                "detected_candidates": state.detected_candidates,
                "emitted_events": state.emitted_events,
                "last_event_at": state.last_event_at,
                "last_processed_at": state.last_processed_at,
                "last_processing_ms": round(
                    state.last_processing_ms,
                    1,
                ),
                "idle_mode": bool(state.no_plate_streak >= 2),
                "uploaded_video_mode": self._is_uploaded_video(
                    state.config or {}
                ),
                "no_plate_streak": state.no_plate_streak,
                "next_inference_seconds": round(
                    max(0.0, state.next_inference_at - time.monotonic()),
                    2,
                ),
                "burst_frames_remaining": state.burst_frames_remaining,
                "source_fps": round(state.source_fps_ema, 2),
                "target_burst_fps": round(
                    state.target_burst_fps,
                    2,
                ),
                "pending_replacements": state.pending_replacements,
                "admission_drops": state.admission_drops,
                "expired_frames": state.expired_frames,
                "queue_depth": int(state.pending is not None),
                "sampling": sampling,
                "motion_score": round(state.motion_score, 5),
                "motion_wakeups": state.motion_wakeups,
                "overlay_mask_pixels": state.overlay_mask_pixels,
                "anpr_engine": engine_router.status(camera_id),
                "shadow": {
                    "frames": state.shadow_frames,
                    "candidates": state.shadow_candidates,
                    "errors": state.shadow_errors,
                },
                "consensus_window_seconds": round(
                    state.tracker.max_age_seconds,
                    2,
                ),
                "last_error": state.last_error,
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

    def detections(self, camera_id: int, max_age=2.5) -> list:
        with self._lock:
            state = self._states.get(int(camera_id))
            if (
                not state
                or time.time() - state.latest_detections_at
                > float(max_age)
            ):
                return []
            return [dict(row) for row in state.latest_detections]

    def detection_snapshot(
        self,
        camera_id: int,
        after_revision=0,
        max_age=8.0,
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
                "detections": [
                    dict(row) for row in state.latest_detections
                ],
                "frame": (
                    state.latest_detection_frame.copy()
                    if state.latest_detection_frame is not None
                    else None
                ),
                "max_age": max(
                    3.0,
                    min(10.0, state.processing_seconds_ema * 2.2 + 2.0),
                ),
            }

    def remove(self, camera_id: int, wait=False, timeout=3.0):
        with self._lock:
            state = self._states.get(int(camera_id))
            if state is not None:
                with state.lock:
                    state.cancelled = True
                    state.config_generation += 1
                    state.pending = None
                    self._cancel_pending_timer_locked(state)
                    state.latest_detections = []
                    state.latest_detection_frame = None
                    state.latest_detections_at = time.time()
                    state.detection_revision += 1
                    busy = bool(state.busy)
                if not busy:
                    self._states.pop(int(camera_id),None)
            else:
                busy = False
        if wait and state is not None and busy:
            stopped = state.processing_done.wait(
                max(0.0,float(timeout))
            )
            if stopped:
                with self._lock:
                    if self._states.get(int(camera_id)) is state:
                        self._states.pop(int(camera_id),None)
            return stopped
        return True

    def invalidate_config(self, camera_id: int):
        """Apply a changed camera ROI immediately and clear stale tracks."""
        with self._lock:
            state = self._states.get(int(camera_id))
            if not state:
                return
            with state.lock:
                state.config_generation += 1
                state.config = None
                state.config_loaded_at = 0.0
                state.pending = None
                self._cancel_pending_timer_locked(state)
                state.tracker = PlateConsensusTracker(
                    min_votes=2,
                    max_age_seconds=2.2,
                    emit_cooldown=5.0,
                    emit_unreadable=True,
                )
                state.seen.clear()
                state.track_event_ids.clear()
                state.latest_detections = []
                state.latest_detection_frame = None
                state.latest_detections_at = time.time()
                state.detection_revision += 1

    def shutdown(self):
        self._stopped = True
        with self._lock:
            states = list(self._states.values())
        for state in states:
            with state.lock:
                state.pending = None
                self._cancel_pending_timer_locked(state)
        self._executor.shutdown(
            wait=False,
            cancel_futures=True,
        )


worker = LiveANPRWorker()


def submit_live_frame(camera_id, camera_name, frame):
    worker.submit(camera_id, camera_name, frame)


def live_anpr_status(camera_id):
    return worker.status(camera_id)


def live_anpr_detections(camera_id):
    return worker.detections(camera_id)


def live_anpr_detection_snapshot(camera_id, after_revision=0):
    return worker.detection_snapshot(camera_id, after_revision)


def stop_live_camera(camera_id, wait=False, timeout=3.0):
    return worker.remove(camera_id,wait=wait,timeout=timeout)


def reload_live_camera_config(camera_id):
    worker.invalidate_config(camera_id)
