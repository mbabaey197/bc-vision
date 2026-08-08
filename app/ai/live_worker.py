"""Asynchronous live-camera ANPR worker.

Streaming threads submit the newest frame without blocking. Per-camera workers
apply ROI, multi-frame consensus, duplicate suppression, and persist events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
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
            status.get("hezar_ready")
            or status.get("crnn_ready")
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
        camera_url = str(camera_row["rtsp_url"] or "") if camera_row else ""

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
                1.10,
                0.30 * (2 ** min(3, int(no_plate_streak) - 1)),
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
        with self._lock:
            state = self._states.setdefault(
                int(camera_id),
                _CameraState(),
            )
            state.frame_counter += 1
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
            selection_score = self._selection_score(frame, config)
            activity_source, _, _ = self._roi_frame(frame, config)
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
                    state.pending = payload
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
                    state.pending = payload
                return
            if state.pending is not None:
                pending_score = float(state.pending[4])
                pending_at = float(state.pending[3])
                if (
                    pending_score > selection_score
                    and now - pending_at < 0.30
                ):
                    payload = state.pending
                state.pending = None
            state.last_submitted_at = now
            state.busy = True
        self._executor.submit(self._process, state, payload)


    @staticmethod
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
        stable_rows,
        min_confidence,
        frame,
    ) -> list[dict]:
        # Publish only multi-frame confirmed plates on the live image.
        # Yellow/raw guesses remain available in review data but never draw
        # a rectangle. Repeated low-motion detections at one camera coordinate
        # are remembered as static hard negatives for 25 seconds. Strong OCR
        # can still keep a genuinely parked vehicle visible, including at night.
        selected = []
        now = time.monotonic()
        for key, until in list(state.static_overlay_blocked_until.items()):
            if float(until) <= now:
                state.static_overlay_blocked_until.pop(key, None)
                state.static_overlay_hits.pop(key, None)

        for source in stable_rows:
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
            detector_confidence = float(row.get("detector_confidence", 0.0))
            ocr_confidence = float(row.get("ocr_confidence", 0.0))
            combined_confidence = float(row.get("confidence", 0.0))
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

    def _process(self, state: _CameraState, payload):
        camera_id, camera_name, frame, timestamp = payload[:4]
        activity = payload[5] if len(payload) > 5 else None
        started = time.perf_counter()
        try:
            config = state.config or {}
            is_uploaded_video = str(
                config.get("rtsp_url", "")
            ).startswith("video://")
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

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )

            live_detection_threshold = max(
                0.22,
                min(0.70, min_confidence * 0.68),
            )

            def baseline_process():
                kwargs = {"engine_key": camera_id}
                if exclusion_mask is not None:
                    kwargs["exclusion_mask"] = exclusion_mask
                # Limit expensive OCR work without changing process_frame's
                # signature. Only the strongest three candidates continue.
                return process_frame(
                    source,
                    live_detection_threshold,
                    max_candidates=2,
                    **kwargs,
                )

            if is_uploaded_video:
                class _BaselineOnlyOutcome:
                    mode = "baseline"
                    shadow = []
                    error = ""
                outcome = _BaselineOnlyOutcome()
                outcome.primary = baseline_process()
            else:
                outcome = engine_router.process(
                    source,
                    baseline=baseline_process,
                    min_detection_confidence=live_detection_threshold,
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
                not is_uploaded_video
                and self._setting("anpr_auto_confirm_guesses", "1") == "1"
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
            if outcome.mode == "shadow":
                state.shadow_frames += 1
                state.shadow_candidates += len(outcome.shadow)
                state.shadow_errors += int(bool(outcome.error))
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
            overlay_rows = self._overlay_candidates(
                state,
                stable,
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
            duplicate_seconds = max(
                0.0,
                float(config.get("duplicate_seconds", 30)),
            )
            processing_ms = processing_seconds * 1000.0
            for result in stable:
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
            with self._lock:
                # Always leave real idle time after an expensive transaction.
                # Previously a queued frame was submitted immediately here,
                # which kept detector/OCR threads continuously busy even when
                # every inference returned no plate.
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
                }
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
                "no_plate_streak": state.no_plate_streak,
                "next_inference_seconds": round(
                    max(0.0, state.next_inference_at - time.monotonic()),
                    2,
                ),
                "burst_frames_remaining": state.burst_frames_remaining,
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

    def detections(self, camera_id: int, max_age=1.6) -> list:
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
                "detections": [
                    dict(row) for row in state.latest_detections
                ],
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

    def remove(self, camera_id: int):
        with self._lock:
            self._states.pop(int(camera_id), None)

    def shutdown(self):
        self._stopped = True
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


def stop_live_camera(camera_id):
    worker.remove(camera_id)
