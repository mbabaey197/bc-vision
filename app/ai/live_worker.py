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

from .pipeline import (
    PlateConsensusTracker,
    add_vehicle_analysis,
    process_frame,
)
from .plate_rules import normalize_plate


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
        )
    )
    seen: dict[str, float] = field(default_factory=dict)
    last_error: str = ""
    last_event_at: float = 0.0
    processed_frames: int = 0
    emitted_events: int = 0


class LiveANPRWorker:
    def __init__(self, max_workers=2):
        self._states: dict[int, _CameraState] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="bc-anpr",
        )
        self._lock = threading.RLock()
        self._stopped = False

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
        crop = result.get("crop")

        if (
            self._setting("save_plate_images", "1") == "1"
            and crop is not None
            and getattr(crop, "size", 0)
        ):
            target = plate_dir / f"plate-live-{stamp}.jpg"
            if cv2.imwrite(
                str(target),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            ):
                plate_path = str(target)

        if self._setting("save_snapshots", "1") == "1":
            annotated = frame.copy()
            x1, y1, x2, y2 = result["bbox"]
            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )
            target = snapshot_dir / f"vehicle-live-{stamp}.jpg"
            if cv2.imwrite(
                str(target),
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            ):
                image_path = str(target)

        values = {
            "plate_text": result["plate"],
            "plate_norm": result.get("plate_norm")
            or normalize_plate(result["plate"]),
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
        }
        with connect() as con:
            columns = {
                row[1]
                for row in con.execute(
                    "PRAGMA table_info(plate_events)"
                ).fetchall()
            }
            selected = [key for key in values if key in columns]
            placeholders = ",".join("?" for _ in selected)
            con.execute(
                f"INSERT INTO plate_events({','.join(selected)}) "
                f"VALUES({placeholders})",
                tuple(values[key] for key in selected),
            )

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
            frame_step = max(1, int(config.get("frame_step", 5)))
            if state.frame_counter % frame_step:
                return
            payload = (
                int(camera_id),
                str(camera_name),
                frame.copy(),
                now,
            )
            if state.busy:
                state.pending = payload
                return
            state.busy = True
        self._executor.submit(self._process, state, payload)

    def _process(self, state: _CameraState, payload):
        camera_id, camera_name, frame, timestamp = payload
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
                self._translate(row, offset_x, offset_y)
                for row in process_frame(
                    source,
                    min_confidence * 0.45,
                )
            ]
            state.processed_frames += 1
            stable = state.tracker.update(rows, timestamp=timestamp)
            duplicate_seconds = max(
                0.0,
                float(config.get("duplicate_seconds", 30)),
            )
            processing_ms = (
                time.perf_counter() - started
            ) * 1000.0
            for result in stable:
                if result["confidence"] < min_confidence:
                    continue
                key = result.get("plate_norm") or normalize_plate(
                    result.get("plate")
                )
                if not key:
                    continue
                previous = state.seen.get(key, -1e12)
                if timestamp - previous < duplicate_seconds:
                    continue
                state.seen[key] = timestamp
                self._persist(
                    camera_id,
                    camera_name,
                    frame,
                    result,
                    processing_ms,
                )
                state.emitted_events += 1
                state.last_event_at = time.time()
            state.last_error = ""
        except Exception as exc:
            state.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                next_payload = state.pending
                state.pending = None
                if next_payload is None or self._stopped:
                    state.busy = False
                else:
                    self._executor.submit(
                        self._process,
                        state,
                        next_payload,
                    )

    def status(self, camera_id: int) -> dict:
        with self._lock:
            state = self._states.get(int(camera_id))
            if not state:
                return {
                    "active": False,
                    "processed_frames": 0,
                    "emitted_events": 0,
                    "last_error": "",
                }
            return {
                "active": bool(state.busy or state.config),
                "processed_frames": state.processed_frames,
                "emitted_events": state.emitted_events,
                "last_event_at": state.last_event_at,
                "last_error": state.last_error,
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


def stop_live_camera(camera_id):
    worker.remove(camera_id)
