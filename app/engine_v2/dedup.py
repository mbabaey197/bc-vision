from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DuplicateDecision:
    duplicate: bool
    reason: str = "new"
    previous_ts: float | None = None


@dataclass(frozen=True, slots=True)
class DuplicateSuppressorConfig:
    same_camera_window_seconds: float = 20.0
    cross_camera_window_seconds: float = 1.5
    max_entries: int = 10_000

    def __post_init__(self) -> None:
        for name, value in (
            ("same_camera_window_seconds", self.same_camera_window_seconds),
            ("cross_camera_window_seconds", self.cross_camera_window_seconds),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if int(self.max_entries) < 1:
            raise ValueError("max_entries must be positive")


class DuplicateSuppressor:
    """Bounded exact-match event cache.

    Fuzzy plate matching is intentionally excluded: suppressing a different
    vehicle is much more damaging than emitting a reviewable duplicate.
    """

    def __init__(self, config: DuplicateSuppressorConfig | None = None) -> None:
        self.config = config or DuplicateSuppressorConfig()
        self._by_camera: OrderedDict[tuple[str, str], float] = OrderedDict()
        self._global: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    def check_and_record(self, camera_id: str, normalized_plate: str, ts: float) -> DuplicateDecision:
        camera_id = str(camera_id)
        plate = str(normalized_plate).strip()
        timestamp = float(ts)
        if not plate:
            return DuplicateDecision(False, "empty_not_cached")
        if not math.isfinite(timestamp):
            raise ValueError("ts must be finite")

        with self._lock:
            key = (camera_id, plate)
            previous = self._by_camera.get(key)
            if (
                self.config.same_camera_window_seconds > 0
                and previous is not None
                and abs(timestamp - previous) <= self.config.same_camera_window_seconds
            ):
                self._record_seen(camera_id, plate, timestamp)
                self._trim(max(timestamp, previous))
                return DuplicateDecision(True, "same_camera_window", previous)

            global_previous = self._global.get(plate)
            if global_previous is not None:
                previous_camera, previous_ts = global_previous
                if (
                    self.config.cross_camera_window_seconds > 0
                    and previous_camera != camera_id
                    and abs(timestamp - previous_ts) <= self.config.cross_camera_window_seconds
                ):
                    self._record_seen(camera_id, plate, timestamp)
                    self._trim(max(timestamp, previous_ts))
                    return DuplicateDecision(True, "overlapping_camera_window", previous_ts)

            # A delayed event outside the duplicate window may be emitted, but
            # it must never move last-seen state backwards.
            self._record_seen(camera_id, plate, timestamp)
            self._trim(timestamp)
            return DuplicateDecision(False)

    def clear(self) -> None:
        with self._lock:
            self._by_camera.clear()
            self._global.clear()

    def _record_seen(self, camera_id: str, plate: str, timestamp: float) -> None:
        key = (camera_id, plate)
        previous = self._by_camera.get(key)
        self._by_camera[key] = timestamp if previous is None else max(previous, timestamp)
        self._by_camera.move_to_end(key)

        global_previous = self._global.get(plate)
        if global_previous is None or timestamp >= global_previous[1]:
            self._global[plate] = (camera_id, timestamp)
        self._global.move_to_end(plate)

    def _trim(self, now: float) -> None:
        horizon = max(
            self.config.same_camera_window_seconds,
            self.config.cross_camera_window_seconds,
        )
        if len(self._by_camera) > self.config.max_entries:
            expired = [
                key
                for key, ts in self._by_camera.items()
                if now >= ts and now - ts > horizon
            ]
            for key in expired:
                self._by_camera.pop(key, None)
        if len(self._global) > self.config.max_entries:
            expired = [
                plate
                for plate, value in self._global.items()
                if now >= value[1] and now - value[1] > horizon
            ]
            for plate in expired:
                self._global.pop(plate, None)

        limit = int(self.config.max_entries)
        while len(self._by_camera) > limit:
            self._by_camera.popitem(last=False)
        while len(self._global) > limit:
            self._global.popitem(last=False)
