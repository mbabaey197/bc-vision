from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .benchmark import AdapterTelemetry, BenchmarkFrameJob, BenchmarkScenario
from .runtime import EventDrivenANPREngine
from .streams import AdaptiveFrameAdmissionController
from .types import FramePacket


@dataclass(frozen=True, slots=True)
class RuntimeBenchmarkFrameConfig:
    detector_shape: tuple[int, int] = (360, 640)
    main_shape: tuple[int, int] = (720, 1280)
    idle_value: int = 24


class EngineV2RuntimePerformanceAdapter:
    """Feed the real V2 scheduler/models with deterministic synthetic pixels.

    The adapter measures real runtime/model CPU and RAM, but deliberately marks
    the result non-production because the input is not camera footage and no
    decoder utilization can be inferred. It exists to provide a reproducible
    lower-level performance smoke test without fabricating accuracy evidence.
    """

    adapter_name = "engine-v2-real-runtime-synthetic-frames"
    evidence_kind = "real-runtime-synthetic-input"
    production_evidence = False
    decode_utilization_kind = "unavailable"
    decode_utilization_source = "unavailable:no-decoder-in-runtime-adapter"

    def __init__(
        self,
        engine: EventDrivenANPREngine,
        *,
        frames: RuntimeBenchmarkFrameConfig | None = None,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.engine = engine
        self.frames = frames or RuntimeBenchmarkFrameConfig()
        self.close_callback = close_callback
        self._sequences: dict[str, int] = {}
        self._sequence_offsets: dict[str, int] = {}
        self._active_frame_indexes: dict[str, int] = {}
        self._admission: dict[str, AdaptiveFrameAdmissionController] = {}
        self._initialized: set[str] = set()
        self._source_fps = 10.0
        self._ticks_per_second = 10
        self._idle_main, self._idle_detector = self._make_idle_frames()
        self._active_frames = self._make_active_frames()

    def prepare_scenario(self, scenario: BenchmarkScenario) -> None:
        self.engine.reset_runtime_state()
        self._sequences.clear()
        self._sequence_offsets.clear()
        self._active_frame_indexes.clear()
        self._admission.clear()
        self._initialized.clear()
        self._source_fps = float(scenario.ticks_per_second * scenario.producer_burst)
        self._ticks_per_second = int(scenario.ticks_per_second)

    def process(self, job: BenchmarkFrameJob) -> AdapterTelemetry:
        self._ensure_camera(job.camera_id, job.nominal_timestamp)
        before = self.engine.telemetry()
        # Preserve the outer producer's natural newest-survivor sequence gaps.
        # The runtime, not the adapter, must decide whether this packet is a
        # detector or tracking-only cadence slot under current load.
        offset = self._sequence_offsets[job.camera_id]
        sequence = max(
            self._sequences[job.camera_id] + 1,
            offset + int(job.sequence),
        )
        self._sequences[job.camera_id] = sequence
        frame_index = self._active_frame_indexes.get(job.camera_id, 0)
        self._active_frame_indexes[job.camera_id] = frame_index + 1
        main, detector = self._active_frames[frame_index % len(self._active_frames)]
        admission = self._admission[job.camera_id]
        admission.update(
            self.engine.producer_cadence_policy(job.camera_id, self._source_fps)
        )
        decision = admission.decide(float(job.nominal_timestamp))
        if not decision.admit:
            return self._delta(before, self.engine.telemetry())
        accepted = self.engine.submit_frame(
            FramePacket(
                camera_id=job.camera_id,
                seq=sequence,
                ts=float(job.nominal_timestamp),
                frame=main,
                detector_frame=detector,
                metadata={
                    "benchmark_input": "synthetic-active",
                    "adaptive_admission": True,
                    "detector_due": decision.detector_due,
                },
            )
        )
        if accepted is False and decision.detector_due:
            admission.detector_unaccepted()
        self.engine.process_available(limit=8)
        return self._delta(before, self.engine.telemetry())

    def observe_idle(self, camera_id: str, tick: int) -> AdapterTelemetry:
        self._ensure_camera(camera_id, float(tick))
        before = self.engine.telemetry()
        admission = self._admission[camera_id]
        admission.update(
            self.engine.producer_cadence_policy(camera_id, self._source_fps)
        )
        timestamp = float(tick) / float(max(1, self._ticks_per_second))
        decision = admission.decide(timestamp)
        if not decision.admit:
            return self._delta(before, self.engine.telemetry())
        accepted = self.engine.submit_frame(
            FramePacket(
                camera_id=camera_id,
                seq=self._next_sequence(camera_id),
                ts=timestamp,
                frame=self._idle_main,
                detector_frame=self._idle_detector,
                metadata={
                    "benchmark_input": "synthetic-idle",
                    "adaptive_admission": True,
                    "detector_due": decision.detector_due,
                },
            )
        )
        if accepted is False and decision.detector_due:
            admission.detector_unaccepted()
        self.engine.process_available(limit=2)
        return self._delta(before, self.engine.telemetry())

    def close(self) -> None:
        if self.close_callback is not None:
            callback, self.close_callback = self.close_callback, None
            callback()

    def _ensure_camera(self, camera_id: str, timestamp: float) -> None:
        if camera_id in self._initialized:
            return
        self._initialized.add(camera_id)
        # Prime the motion baseline on an actual idle-gate cadence.  Without
        # this alignment, a default idle stride of eight can leave the first
        # active benchmark jobs permanently asleep (especially when the outer
        # harness uses newest-frame-wins with an even producer burst).
        stride = max(
            1,
            self.engine.config.idle_stride
            * self.engine.policy.idle_stride_multiplier,
        )
        self._sequences[camera_id] = stride - 1
        self._admission[camera_id] = AdaptiveFrameAdmissionController(
            self.engine.producer_cadence_policy(camera_id, self._source_fps)
        )
        self.engine.submit_frame(
            FramePacket(
                camera_id=camera_id,
                seq=self._next_sequence(camera_id),
                ts=timestamp,
                frame=self._idle_main,
                detector_frame=self._idle_detector,
                metadata={
                    "benchmark_input": "synthetic-baseline",
                    "adaptive_admission": True,
                    "detector_due": True,
                },
            )
        )
        self._sequence_offsets[camera_id] = self._sequences[camera_id]

    def _next_sequence(self, camera_id: str) -> int:
        value = self._sequences.get(camera_id, 0) + 1
        self._sequences[camera_id] = value
        return value

    def _make_idle_frames(self) -> tuple[np.ndarray, np.ndarray]:
        main_h, main_w = self.frames.main_shape
        detector_h, detector_w = self.frames.detector_shape
        main = np.full((main_h, main_w, 3), self.frames.idle_value, dtype=np.uint8)
        detector = np.full(
            (detector_h, detector_w, 3),
            self.frames.idle_value,
            dtype=np.uint8,
        )
        return main, detector

    def _make_active_frames(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        variants: list[tuple[np.ndarray, np.ndarray]] = []
        for offset in (70, 125):
            main = self._idle_main.copy()
            detector = self._idle_detector.copy()
            main[::4, :, :] = min(255, self.frames.idle_value + offset)
            main[:, ::7, :] = min(255, self.frames.idle_value + offset // 2)
            detector[::4, :, :] = min(255, self.frames.idle_value + offset)
            detector[:, ::7, :] = min(255, self.frames.idle_value + offset // 2)
            variants.append((main, detector))
        return tuple(variants)

    def _delta(self, before: dict[str, object], after: dict[str, object]) -> AdapterTelemetry:
        return AdapterTelemetry(
            detector_inferences=max(
                0,
                int(after["detector_inferences"]) - int(before["detector_inferences"]),
            ),
            ocr_inferences=max(0, int(after["ocr_inferences"]) - int(before["ocr_inferences"])),
            plate_events=max(0, int(after["events"]) - int(before["events"])),
            decode_utilization_percent=None,
            decode_utilization_kind=self.decode_utilization_kind,
            decode_utilization_source=self.decode_utilization_source,
            active_cameras=int(after["active_cameras"]),
            idle_cameras=int(after["idle_cameras"]),
        )


__all__ = ["EngineV2RuntimePerformanceAdapter", "RuntimeBenchmarkFrameConfig"]
