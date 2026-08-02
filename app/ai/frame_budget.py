"""Physical frame-budget planning for high-speed ANPR capture.

Dashboard preview cadence is intentionally absent from the recognition math.
The camera's native stream, the calibrated readable road distance, and the
maximum vehicle speed determine whether enough independent crops can exist.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


STANDARD_CAPTURE_RATES = (8, 10, 12, 15, 20, 25, 30, 50, 60, 90, 120)


def _bounded(value, minimum, maximum) -> float:
    return min(float(maximum), max(float(minimum), float(value)))


def _standard_capture_rate(minimum_fps: float) -> int:
    for rate in STANDARD_CAPTURE_RATES:
        if rate + 1e-9 >= float(minimum_fps):
            return rate
    return int(math.ceil(float(minimum_fps)))


@dataclass(frozen=True)
class FrameBudget:
    max_speed_kmh: float
    recognition_zone_m: float
    source_fps: float
    processing_p95_ms: float
    required_good_crops: int
    usable_frame_ratio: float
    safety_margin: float
    speed_mps: float
    zone_seconds: float
    minimum_native_fps: float
    recommended_capture_fps: int
    expected_raw_frames: float
    expected_usable_frames: float
    detector_capacity_fps: float
    expected_processed_observations: float
    source_sufficient: bool
    processing_sufficient: bool

    @property
    def sufficient(self) -> bool:
        return self.source_sufficient and self.processing_sufficient

    def as_dict(self) -> dict:
        value = asdict(self)
        value["sufficient"] = self.sufficient
        value["warning"] = self.warning()
        return value

    def warning(self) -> str:
        if not self.source_sufficient:
            return (
                "source-fps-insufficient: camera supplies "
                f"{self.source_fps:.1f} FPS but this road geometry needs "
                f"about {self.recommended_capture_fps} FPS"
            )
        if not self.processing_sufficient:
            return (
                "processing-capacity-insufficient: estimated detector "
                f"capacity yields {self.expected_processed_observations:.1f} "
                f"observations; at least {self.required_good_crops} are needed"
            )
        return ""


def calculate_frame_budget(
    *,
    max_speed_kmh=150.0,
    recognition_zone_m=10.0,
    source_fps=0.0,
    processing_p95_ms=0.0,
    required_good_crops=3,
    usable_frame_ratio=0.70,
    safety_margin=1.25,
) -> FrameBudget:
    """Return the capture and processing budget for one calibrated camera.

    ``source_fps=0`` and ``processing_p95_ms=0`` mean that runtime telemetry is
    not available yet. Unknown telemetry is not reported as a failure.
    """

    speed_kmh = _bounded(max_speed_kmh, 5.0, 300.0)
    zone_m = _bounded(recognition_zone_m, 1.0, 100.0)
    native_fps = _bounded(source_fps, 0.0, 240.0)
    p95_ms = _bounded(processing_p95_ms, 0.0, 60_000.0)
    good_crops = int(_bounded(required_good_crops, 3, 8))
    usable_ratio = _bounded(usable_frame_ratio, 0.35, 1.0)
    margin = _bounded(safety_margin, 1.0, 2.0)

    speed_mps = speed_kmh / 3.6
    zone_seconds = zone_m / speed_mps
    minimum_native_fps = (
        margin * good_crops / usable_ratio / zone_seconds
    )
    recommended = _standard_capture_rate(minimum_native_fps)
    expected_raw = native_fps * zone_seconds
    expected_usable = expected_raw * usable_ratio / margin

    detector_capacity = (
        1000.0 / p95_ms
        if p95_ms > 0.0
        else 0.0
    )
    processed = (
        min(native_fps, detector_capacity) * zone_seconds
        if native_fps > 0.0 and detector_capacity > 0.0
        else 0.0
    )
    source_sufficient = (
        True
        if native_fps <= 0.0
        else native_fps + 0.25 >= minimum_native_fps
    )
    processing_sufficient = (
        True
        if p95_ms <= 0.0 or native_fps <= 0.0
        else processed + 1e-9 >= good_crops
    )

    return FrameBudget(
        max_speed_kmh=round(speed_kmh, 3),
        recognition_zone_m=round(zone_m, 3),
        source_fps=round(native_fps, 3),
        processing_p95_ms=round(p95_ms, 3),
        required_good_crops=good_crops,
        usable_frame_ratio=round(usable_ratio, 3),
        safety_margin=round(margin, 3),
        speed_mps=round(speed_mps, 4),
        zone_seconds=round(zone_seconds, 5),
        minimum_native_fps=round(minimum_native_fps, 3),
        recommended_capture_fps=recommended,
        expected_raw_frames=round(expected_raw, 3),
        expected_usable_frames=round(expected_usable, 3),
        detector_capacity_fps=round(detector_capacity, 3),
        expected_processed_observations=round(processed, 3),
        source_sufficient=source_sufficient,
        processing_sufficient=processing_sufficient,
    )
