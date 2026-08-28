from __future__ import annotations

import numpy as np

from app.engine_v2.benchmark import BenchmarkFrameJob, BenchmarkScenario
from app.engine_v2.benchmark_adapters import EngineV2RuntimePerformanceAdapter
from app.engine_v2.load import LoadSnapshot
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
    assert telemetry.active_cameras == 1
    assert telemetry.idle_cameras == 0
    assert idle.detector_inferences == 0
    assert idle.active_cameras == 1
    assert idle.idle_cameras == 1
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

    results = [
        adapter.process(
            BenchmarkFrameJob(
                "camera-01",
                sequence,
                sequence / 10.0,
                produced_monotonic=0.0,
            )
        )
        for sequence in (2, 4, 6, 8, 10)
    ]

    assert [item.detector_inferences for item in results] == [1, 1, 1, 1, 1]
    assert detector.calls == 5


def test_runtime_benchmark_adapter_respects_critical_adaptive_cadence() -> None:
    detector = EmptyDetector()
    engine = EventDrivenANPREngine(
        detector,
        UnusedOCR(),
        EngineV2Config(load_control_enabled=False),
    )
    adapter = EngineV2RuntimePerformanceAdapter(engine)
    adapter.prepare_scenario(BenchmarkScenario("runtime", camera_count=1, active_cameras=1))
    first = adapter.process(BenchmarkFrameJob("camera-01", 1, 0.0, produced_monotonic=0.0))
    assert first.detector_inferences == 1

    engine.inject_load_snapshot(
        LoadSnapshot(
            timestamp=1.0,
            cpu_percent=99.0,
            detector_latency_ms=300.0,
            ocr_latency_ms=100.0,
            queue_depth=120,
            queue_capacity=128,
            active_cameras=32,
            total_cameras=32,
            stale_drop_rate=0.5,
        )
    )
    shed = [
        adapter.process(
            BenchmarkFrameJob("camera-01", sequence, timestamp, produced_monotonic=0.0)
        )
        for sequence, timestamp in ((2, 0.1), (3, 0.2), (4, 0.3), (5, 0.4))
    ]

    # The idle->active transition forces one accuracy-protecting detector
    # sample; critical cadence suppresses the following slots.
    assert [item.detector_inferences for item in shed] == [1, 0, 0, 0]
    assert detector.calls == 2
