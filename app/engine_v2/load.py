from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import IntEnum


class LoadLevel(IntEnum):
    NORMAL = 0
    ELEVATED = 1
    HIGH = 2
    CRITICAL = 3


@dataclass(frozen=True, slots=True)
class LoadSnapshot:
    timestamp: float
    cpu_percent: float
    detector_latency_ms: float
    ocr_latency_ms: float
    queue_depth: int
    queue_capacity: int
    active_cameras: int
    total_cameras: int
    stale_drop_rate: float = 0.0

    @property
    def queue_ratio(self) -> float:
        return max(0.0, min(1.0, self.queue_depth / max(1, self.queue_capacity)))


@dataclass(frozen=True, slots=True)
class LoadPolicy:
    level: LoadLevel
    detector_stride_multiplier: int
    idle_stride_multiplier: int
    tracking_frames_between_detection: int
    max_ocr_candidates: int
    max_queue_age_ms: float
    active_fps_scale: float
    idle_fps_scale: float


_POLICIES: dict[LoadLevel, LoadPolicy] = {
    LoadLevel.NORMAL: LoadPolicy(LoadLevel.NORMAL, 1, 1, 0, 3, 1_000.0, 1.00, 1.00),
    LoadLevel.ELEVATED: LoadPolicy(LoadLevel.ELEVATED, 2, 2, 1, 3, 750.0, 0.80, 0.50),
    LoadLevel.HIGH: LoadPolicy(LoadLevel.HIGH, 3, 4, 2, 2, 450.0, 0.58, 0.25),
    LoadLevel.CRITICAL: LoadPolicy(LoadLevel.CRITICAL, 5, 8, 4, 1, 250.0, 0.35, 0.10),
}


@dataclass(slots=True)
class AdaptiveLoadConfig:
    target_cpu_percent: float = 72.0
    high_cpu_percent: float = 85.0
    critical_cpu_percent: float = 94.0
    target_detector_latency_ms: float = 80.0
    target_ocr_latency_ms: float = 45.0
    elevated_queue_ratio: float = 0.25
    high_queue_ratio: float = 0.55
    critical_queue_ratio: float = 0.82
    elevated_active_cameras: int = 8
    high_active_cameras: int = 16
    critical_active_cameras: int = 32
    ema_alpha: float = 0.28
    recovery_samples: int = 3

    def __post_init__(self) -> None:
        if not (
            0 < self.target_cpu_percent < self.high_cpu_percent < self.critical_cpu_percent
        ):
            raise ValueError("CPU thresholds must be positive and strictly increasing")
        if self.target_detector_latency_ms <= 0 or self.target_ocr_latency_ms <= 0:
            raise ValueError("latency targets must be positive")
        if not (
            0 < self.elevated_queue_ratio
            < self.high_queue_ratio
            < self.critical_queue_ratio
            <= 1
        ):
            raise ValueError("queue thresholds must be ordered within 0..1")
        if not (
            0 < self.elevated_active_cameras
            < self.high_active_cameras
            < self.critical_active_cameras
        ):
            raise ValueError("active-camera thresholds must be positive and increasing")
        if not 0 < self.ema_alpha <= 1:
            raise ValueError("ema_alpha must be within (0, 1]")
        if self.recovery_samples < 1:
            raise ValueError("recovery_samples must be at least one")


class AdaptiveLoadController:
    """Hysteretic resource controller for detector/OCR scheduling.

    Escalation is immediate to protect real-time latency. Recovery is gradual,
    so temporary CPU dips cannot cause oscillation. Returning to NORMAL restores
    three-candidate OCR and the highest configured detector cadence.
    """

    def __init__(self, config: AdaptiveLoadConfig | None = None) -> None:
        self.config = config or AdaptiveLoadConfig()
        self._level = LoadLevel.NORMAL
        self._pressure_ema = 0.0
        self._recovery_count = 0
        self._last_snapshot: LoadSnapshot | None = None
        self._lock = threading.RLock()

    @property
    def level(self) -> LoadLevel:
        with self._lock:
            return self._level

    @property
    def last_snapshot(self) -> LoadSnapshot | None:
        with self._lock:
            return self._last_snapshot

    @property
    def policy(self) -> LoadPolicy:
        with self._lock:
            return _POLICIES[self._level]

    def observe(self, snapshot: LoadSnapshot) -> LoadPolicy:
        with self._lock:
            self._last_snapshot = snapshot
            instantaneous = self._pressure(snapshot)
            alpha = self.config.ema_alpha
            self._pressure_ema = alpha * instantaneous + (1.0 - alpha) * self._pressure_ema
            requested = max(
                self._level_for_pressure(instantaneous),
                self._level_for_pressure(self._pressure_ema),
            )

            if requested > self._level:
                self._level = requested
                self._recovery_count = 0
            elif requested < self._level:
                self._recovery_count += 1
                if self._recovery_count >= self.config.recovery_samples:
                    self._level = LoadLevel(self._level - 1)
                    self._recovery_count = 0
            else:
                self._recovery_count = 0
            return _POLICIES[self._level]

    def reset(self) -> None:
        with self._lock:
            self._level = LoadLevel.NORMAL
            self._pressure_ema = 0.0
            self._recovery_count = 0
            self._last_snapshot = None

    def _pressure(self, snapshot: LoadSnapshot) -> float:
        cpu = self._severity(
            max(0.0, snapshot.cpu_percent),
            self.config.target_cpu_percent,
            self.config.high_cpu_percent,
            self.config.critical_cpu_percent,
        )
        detector_target = max(1.0, self.config.target_detector_latency_ms)
        detector = self._severity(
            max(0.0, snapshot.detector_latency_ms),
            detector_target,
            detector_target * 1.75,
            detector_target * 2.75,
        )
        ocr_target = max(1.0, self.config.target_ocr_latency_ms)
        ocr = self._severity(
            max(0.0, snapshot.ocr_latency_ms),
            ocr_target,
            ocr_target * 1.75,
            ocr_target * 2.75,
        )
        queue = self._severity(
            snapshot.queue_ratio,
            self.config.elevated_queue_ratio,
            self.config.high_queue_ratio,
            self.config.critical_queue_ratio,
        )
        active = self._severity(
            float(max(0, snapshot.active_cameras)),
            float(max(1, self.config.elevated_active_cameras)),
            float(max(self.config.elevated_active_cameras + 1, self.config.high_active_cameras)),
            float(max(self.config.high_active_cameras + 1, self.config.critical_active_cameras)),
        )
        stale = max(0.0, snapshot.stale_drop_rate) / 0.05
        # Latency/queue are stronger signals than the raw camera count. Idle
        # cameras therefore add virtually no AI pressure by themselves.
        return max(
            cpu,
            detector,
            ocr,
            queue,
            stale,
            0.50 * active,
            0.30 * cpu + 0.22 * detector + 0.18 * ocr + 0.15 * queue + 0.15 * active,
        )

    @staticmethod
    def _severity(value: float, elevated: float, high: float, critical: float) -> float:
        if value >= critical:
            return 2.7 + (value - critical) / max(critical, 1e-6)
        if value >= high:
            fraction = (value - high) / max(critical - high, 1e-6)
            return 1.75 + 0.95 * fraction
        if value >= elevated:
            fraction = (value - elevated) / max(high - elevated, 1e-6)
            return 1.0 + 0.75 * fraction
        return max(0.0, value / max(elevated, 1e-6))

    def _level_for_pressure(self, pressure: float) -> LoadLevel:
        if pressure >= 2.7:
            return LoadLevel.CRITICAL
        if pressure >= 1.75:
            return LoadLevel.HIGH
        if pressure >= 1.0:
            return LoadLevel.ELEVATED
        return LoadLevel.NORMAL


class SystemLoadSampler:
    """Optional psutil adapter kept outside the deterministic controller."""

    def __init__(self) -> None:
        self._psutil = None
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None

    def cpu_percent(self) -> float:
        if self._psutil is None:
            return 0.0
        try:
            return float(self._psutil.cpu_percent(interval=None))
        except Exception:
            return 0.0

    @staticmethod
    def now() -> float:
        return time.monotonic()
