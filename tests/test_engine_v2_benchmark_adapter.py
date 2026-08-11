from __future__ import annotations

import numpy as np

from app.engine_v2.benchmark import BenchmarkFrameJob, BenchmarkScenario
from app.engine_v2.benchmark_adapters import EngineV2RuntimePerformanceAdapter
from app.engine_v2.runtime import EngineV2Config, EventDrivenANPREngine
from app.engine_v2.types import OCRResult


class EmptyDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        return []


class UnusedOCR:
    def read(self, crop: np.ndarray) -> OCRResult:
        raise AssertionError("OCR must not run without a detection")


def test_runtime_benchmark_adapter_measures_real_engine_but_never_claims_evidence() -> None:
    detector = EmptyDetector()
    engine = EventDrivenANPREngine(
        detector,
        UnusedOCR(),
        EngineV2Config(idle_stride=1, active_stride=1, load_control_enabled=False),
    )
    adapter = EngineV2RuntimePerformanceAdapter(engine)
    scenario = BenchmarkScenario("runtime", camera_count=2, active_cameras=1)
    adapter.prepare_scenario(scenario)

    telemetry = adapter.process(
        BenchmarkFrameJob("camera-01", 1, 1.0, produced_monotonic=0.0)
    )
    idle = adapter.observe_idle("camera-02", 1)

    assert telemetry.detector_inferences == 1
    assert telemetry.ocr_inferences == 0
    assert idle.detector_inferences == 0
    assert adapter.production_evidence is False
    assert telemetry.decode_utilization_percent is None
    assert telemetry.decode_utilization_source.startswith("unavailable:")


def test_runtime_benchmark_adapter_primes_default_idle_cadence() -> None:
    detector = EmptyDetector()
    engine = EventDrivenANPREngine(
        detector,
        UnusedOCR(),
        EngineV2Config(load_control_enabled=False),
    )
    adapter = EngineV2RuntimePerformanceAdapter(engine)
    adapter.prepare_scenario(BenchmarkScenario("runtime", camera_count=1, active_cameras=1))

    first = adapter.process(BenchmarkFrameJob("camera-01", 2, 0.0, produced_monotonic=0.0))
    second = adapter.process(BenchmarkFrameJob("camera-01", 4, 0.1, produced_monotonic=0.0))

    assert first.detector_inferences == 1
    assert second.detector_inferences == 1
    assert detector.calls == 2
