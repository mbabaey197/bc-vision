"""Asynchronous live-camera ANPR worker.

Streaming threads submit the newest frame without blocking. Per-camera workers
apply ROI, multi-frame consensus, duplicate suppression, and persist events.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import threading
import time

import cv2

from app.cpu_budget import parallel_camera_limit, threads_per_camera

from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    process_frame,
)
from .plate_rules import normalize_plate
from .feedback import apply_learned_correction


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
                    "easyocr_ready": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            self._model_state_at = now
        status = dict(self._model_state)
        status["ocr_ready"] = bool(
            status.get("crnn_ready")
            or status.get("easyocr_ready")
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
        return row

    @staticmethod
    def _setting(key, default=""):
        from app.database import get_setting
        return get_setting(key, default)

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
        plate_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        plate_path = ""
        image_path = ""
        with connect() as con:
            existing = (
                con.execute(
                    "SELECT image_path,plate_image_path "
                    "FROM plate_events WHERE id=?",
                    (int(event_id),),
                ).fetchone()
                if event_id
                else None
            )
        if existing:
            plate_path = existing["plate_image_path"] or ""
            image_path = existing["image_path"] or ""
        crop = result.get("crop")

        if (
            self._setting("save_plate_images", "1") == "1"
            and crop is not None
            and getattr(crop, "size", 0)
        ):
            target = (
                Path(plate_path)
                if plate_path
                else plate_dir / f"plate-live-{stamp}.jpg"
            )
            if cv2.imwrite(
                str(target),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            ):
                plate_path = str(target)

        if self._setting("save_snapshots", "1") == "1":
            vehicle = result.get("vehicle_crop")
            using_vehicle_crop = bool(
                vehicle is not None and getattr(vehicle, "size", 0)
            )
            annotated = (
                vehicle.copy()
                if using_vehicle_crop
                else frame.copy()
            )
            x1, y1, x2, y2 = result["bbox"]
            if result.get("vehicle_bbox") and using_vehicle_crop:
                vx1, vy1, _, _ = result["vehicle_bbox"]
                x1, x2 = x1 - vx1, x2 - vx1
                y1, y2 = y1 - vy1, y2 - vy1
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            target = (
                Path(image_path)
                if image_path
                else snapshot_dir / f"vehicle-live-{stamp}.jpg"
            )
            if cv2.imwrite(
                str(target),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            ):
                image_path = str(target)

        values = {
            "plate_text": result.get("plate") or "ناخوانا",
            "plate_norm": (
                result.get("plate_norm")
                or normalize_plate(result.get("plate"))
                if result.get("valid")
                else ""
            ),
            "confidence": float(result["confidence"]),
            "camera_id": camera_id,
            "camera_name": camera_name,
            "image_path": image_path,
            "plate_image_path": plate_path,
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
            "review_status": (
                "unreadable"
                if result.get("unreadable_final")
                else (
                    "suggested"
                    if result.get("needs_review")
                    else "confirmed-ai"
                )
            ),
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
            payload = (
                int(camera_id),
                str(camera_name),
                frame.copy(),
                now,
                selection_score,
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
                    or now - pending_at >= 0.35
                ):
                    state.pending = payload
                return
            # Do not let a slow CPU run ANPR continuously with no breathing
            # room. Keep the newest frame and cap inference frequency
            # adaptively; this reduces load without lowering image quality.
            minimum_interval = max(
                0.0,
                state.next_inference_at - now,
                (
                max(
                    0.20,
                    min(1.25, state.processing_seconds_ema * 0.45),
                )
                if state.processing_seconds_ema
                else 0.0
                ),
            )
            if (
                now - state.last_submitted_at < minimum_interval
                or now < state.next_inference_at
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
                    and now - pending_at < 0.8
                ):
                    payload = state.pending
                state.pending = None
            state.last_submitted_at = now
            state.busy = True
        self._executor.submit(self._process, state, payload)

    def _process(self, state: _CameraState, payload):
        camera_id, camera_name, frame, timestamp = payload[:4]
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
            rows = [
                apply_learned_correction(
                    self._translate(row, offset_x, offset_y)
                )
                for row in process_frame(
                    source,
                    min_confidence * 0.45,
                    engine_key=camera_id,
                )
            ]
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
                min(45.0, state.processing_seconds_ema * 3.5 + 1.0),
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
                    row.get("ocr_engine")
                    == "dedicated-character-detector"
                )
            if rows:
                state.no_plate_streak = 0
            else:
                state.no_plate_streak = min(
                    12,
                    state.no_plate_streak + 1,
                )
            state.latest_detections = [
                    {
                        "bbox": tuple(row["bbox"]),
                        "plate": row.get("plate", "ناخوانا"),
                        "confidence": float(row.get("confidence", 0.0)),
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
                    }
                    for row in rows
                ]
            state.latest_detection_frame = frame.copy()
            state.latest_detections_at = time.time()
            # Empty inference is also a new display state. Publishing its
            # revision clears an old box immediately instead of leaving it on
            # screen until a wall-clock timeout.
            state.detection_revision += 1
            state.last_processed_at = time.time()
            state.last_processing_ms = processing_seconds * 1000.0
            stable = state.tracker.update(
                rows,
                timestamp=timestamp,
                frame=frame,
            )
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
                    self._post_inference_delay(
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
