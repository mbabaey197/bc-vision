from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .dedup import DuplicateSuppressor, DuplicateSuppressorConfig
from .load import AdaptiveLoadController, LoadPolicy, LoadSnapshot, SystemLoadSampler
from .motion import AdaptiveMotionGate
from .ocr import OCRTask, SharedOCRWorker, TemporalOCRVoter
from .quality import BestPlateFrameSelector
from .scheduler import LatestOnlyPriorityQueue
from .streams import ProducerActivity, ProducerCadencePolicy
from .tracking import LightweightMultiObjectTracker, TrackObservation, TrackerConfig
from .types import FramePacket, OCRResult, PlateCandidate, PlateDetector, PlateEvent, PlateOCR, TrackPhase
from .validator import IranianPlateValidator


_RUNTIME_EPOCH_METADATA_KEY = "_engine_v2_runtime_epoch"


@dataclass(slots=True)
class EngineV2Config:
    idle_stride: int = 8
    active_stride: int = 2
    min_detector_confidence: float = 0.30
    min_ocr_confidence: float = 0.55
    min_quality: float = 0.32
    early_ocr_quality: float = 0.82
    selector_capacity: int = 5
    min_candidates_before_ocr: int = 2
    max_collection_frames: int = 8
    max_ocr_candidates: int = 3
    max_ocr_attempts: int = 2
    crop_padding_ratio: float = 0.06
    max_main_stream_skew_ms: float = 350.0
    done_cooldown_frames: int = 25
    max_done_hold_frames: int = 300
    exit_quiet_samples: int = 2
    active_quiet_samples: int = 10
    queue_size: int = 128
    ocr_queue_size: int = 256
    max_queue_age_seconds: float = 1.0
    tracker_max_missed: int = 4
    load_control_enabled: bool = True
    same_camera_duplicate_seconds: float = 20.0
    cross_camera_duplicate_seconds: float = 1.5

    def __post_init__(self) -> None:
        positive_ints = {
            "idle_stride": self.idle_stride,
            "active_stride": self.active_stride,
            "selector_capacity": self.selector_capacity,
            "min_candidates_before_ocr": self.min_candidates_before_ocr,
            "max_collection_frames": self.max_collection_frames,
            "max_ocr_candidates": self.max_ocr_candidates,
            "max_ocr_attempts": self.max_ocr_attempts,
            "done_cooldown_frames": self.done_cooldown_frames,
            "max_done_hold_frames": self.max_done_hold_frames,
            "exit_quiet_samples": self.exit_quiet_samples,
            "active_quiet_samples": self.active_quiet_samples,
            "queue_size": self.queue_size,
            "ocr_queue_size": self.ocr_queue_size,
        }
        for name, value in positive_ints.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.tracker_max_missed < 0:
            raise ValueError("tracker_max_missed cannot be negative")
        for name, value in {
            "min_detector_confidence": self.min_detector_confidence,
            "min_ocr_confidence": self.min_ocr_confidence,
            "min_quality": self.min_quality,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within 0..1")
        # Values above one are a useful, explicit way to disable the early-OCR
        # shortcut while retaining normal candidate-count/timeout triggers.
        if self.early_ocr_quality < 0:
            raise ValueError("early_ocr_quality cannot be negative")
        if self.crop_padding_ratio < 0:
            raise ValueError("crop_padding_ratio cannot be negative")
        if self.max_main_stream_skew_ms < 0:
            raise ValueError("max_main_stream_skew_ms cannot be negative")
        if self.max_queue_age_seconds < 0:
            raise ValueError("max_queue_age_seconds cannot be negative")
        if self.same_camera_duplicate_seconds < 0 or self.cross_camera_duplicate_seconds < 0:
            raise ValueError("duplicate suppression windows cannot be negative")


@dataclass(slots=True)
class TrackEpisode:
    track_id: int
    episode_id: str
    first_seq: int
    last_seq: int
    last_ts: float
    phase: TrackPhase = TrackPhase.TRACKING
    selector: BestPlateFrameSelector = field(default_factory=BestPlateFrameSelector)
    transitions: list[TrackPhase] = field(
        default_factory=lambda: [TrackPhase.IDLE, TrackPhase.ACTIVE, TrackPhase.TRACKING]
    )
    ocr_submitted: bool = False
    ocr_attempts: int = 0
    event_emitted: bool = False
    tracker_removed: bool = False
    last_bbox: tuple[int, int, int, int] | None = None
    observations: list[OCRResult] = field(default_factory=list)

    def advance(self, phase: TrackPhase) -> None:
        if self.phase is phase:
            return
        self.phase = phase
        self.transitions.append(phase)


@dataclass(slots=True)
class CameraState:
    phase: TrackPhase = TrackPhase.IDLE
    last_detection_seq: int = -1
    last_done_seq: int = -10_000
    last_received_seq: int = -1
    last_activity_seq: int = -1
    quiet_samples: int = 0
    episode_number: int = 0
    source_epoch: str | None = None
    runtime_epoch: int = 0
    exit_motion_seen: bool = False
    tracks: dict[int, TrackEpisode] = field(default_factory=dict)
    # First-slice compatibility/observability fields.
    best_quality: float = 0.0
    best_crop: np.ndarray | None = None
    best_bbox: tuple[int, int, int, int] | None = None
    observations: list[OCRResult] = field(default_factory=list)

    def reset(self) -> None:
        self.phase = TrackPhase.IDLE
        self.last_detection_seq = -1
        self.last_activity_seq = -1
        self.quiet_samples = 0
        self.exit_motion_seen = False
        self.tracks.clear()
        self.best_quality = 0.0
        self.best_crop = None
        self.best_bbox = None
        self.observations.clear()


@dataclass(slots=True)
class EngineV2Metrics:
    frames_received: int = 0
    out_of_order_frames: int = 0
    motion_evaluations: int = 0
    motion_wakeups: int = 0
    predicted_track_frames: int = 0
    detector_inferences: int = 0
    detector_seconds: float = 0.0
    detector_latency_ema_seconds: float = 0.0
    events_emitted: int = 0
    duplicates_suppressed: int = 0
    restart_stale_frames: int = 0
    restart_stale_ocr_tasks: int = 0

    @property
    def mean_detector_seconds(self) -> float:
        if self.detector_inferences == 0:
            return 0.0
        return self.detector_seconds / self.detector_inferences


class EventDrivenANPREngine:
    """Independent event-driven ANPR V2 core.

    Cameras are producers only. This object owns exactly one detector reference,
    one shared OCR worker/session, one central latest-only scheduler, and cheap
    per-camera motion/tracking state. It is not connected to the legacy runtime.
    """

    def __init__(
        self,
        detector: PlateDetector,
        ocr: PlateOCR,
        config: EngineV2Config | None = None,
        on_event: Callable[[PlateEvent], None] | None = None,
        *,
        validator: IranianPlateValidator | None = None,
        load_controller: AdaptiveLoadController | None = None,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.config = config or EngineV2Config()
        self.on_event = on_event
        self.validator = validator or IranianPlateValidator()
        self.load_controller = load_controller or AdaptiveLoadController()
        self.load_sampler = SystemLoadSampler()
        self.queue: LatestOnlyPriorityQueue[FramePacket] = LatestOnlyPriorityQueue(self.config.queue_size)
        self.ocr_worker = SharedOCRWorker(
            ocr,
            TemporalOCRVoter(self.validator),
            queue_size=self.config.ocr_queue_size,
        )
        self.deduplicator = DuplicateSuppressor(
            DuplicateSuppressorConfig(
                same_camera_window_seconds=self.config.same_camera_duplicate_seconds,
                cross_camera_window_seconds=self.config.cross_camera_duplicate_seconds,
            )
        )
        self.metrics = EngineV2Metrics()
        self._states: dict[str, CameraState] = {}
        self._gates: dict[str, AdaptiveMotionGate] = {}
        self._trackers: dict[str, LightweightMultiObjectTracker] = {}
        self._rois: dict[str, tuple[int, int, int, int] | None] = {}
        self._state_lock = threading.RLock()
        self._detector_lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._last_load_submitted = 0
        self._last_load_stale = 0

    @property
    def policy(self) -> LoadPolicy:
        return self.load_controller.policy

    def set_roi(self, camera_id: str, roi: tuple[int, int, int, int] | None) -> None:
        with self._state_lock:
            self._rois[camera_id] = roi

    def state_for(self, camera_id: str) -> CameraState:
        with self._state_lock:
            return self._states.setdefault(camera_id, CameraState())

    def target_detector_fps(self, camera_id: str, source_fps: float = 25.0) -> float:
        with self._state_lock:
            phase = self._states.setdefault(camera_id, CameraState()).phase
        policy = self.policy
        return self._target_detector_fps_for(phase, source_fps, policy)

    def _target_detector_fps_for(
        self,
        phase: TrackPhase,
        source_fps: float,
        policy: LoadPolicy,
    ) -> float:
        if phase is TrackPhase.IDLE or phase is TrackPhase.DONE:
            stride = max(1, self.config.idle_stride * policy.idle_stride_multiplier)
            scale = policy.idle_fps_scale
        else:
            stride = max(1, self.config.active_stride * policy.detector_stride_multiplier)
            scale = policy.active_fps_scale
        return max(0.05, float(source_fps) * scale / stride)

    def producer_cadence_policy(
        self,
        camera_id: str,
        source_fps: float = 25.0,
    ) -> ProducerCadencePolicy:
        """Return a thread-safe admission policy for an RTSP producer.

        ``DualStreamRTSPProducer`` may call this from its sub-stream thread.
        Runtime-side modulo strides are bypassed for packets carrying the
        adaptive admission contract, preventing duplicate throttling.
        """

        with self._state_lock:
            phase = self._states.setdefault(camera_id, CameraState()).phase
        policy = self.policy
        idle = phase in (TrackPhase.IDLE, TrackPhase.DONE)
        return ProducerCadencePolicy(
            target_detector_fps=self._target_detector_fps_for(
                phase,
                source_fps,
                policy,
            ),
            activity=ProducerActivity.IDLE if idle else ProducerActivity.ACTIVE,
            tracking_frames_between_detection=(
                0 if idle else policy.tracking_frames_between_detection
            ),
        )

    def submit_frame(self, packet: FramePacket) -> bool:
        """Accept a producer packet and schedule only useful detector work."""

        with self._state_lock:
            self.metrics.frames_received += 1
            state = self._states.setdefault(packet.camera_id, CameraState())
            gate = self._gates.setdefault(packet.camera_id, AdaptiveMotionGate())
            tracker = self._trackers.setdefault(
                packet.camera_id,
                LightweightMultiObjectTracker(TrackerConfig(max_missed=self.config.tracker_max_missed)),
            )

            source_epoch_value = packet.metadata.get("producer_epoch")
            source_epoch = None if source_epoch_value is None else str(source_epoch_value)
            resume_after_restart = False
            if source_epoch is not None:
                if state.source_epoch is None:
                    state.source_epoch = source_epoch
                elif state.source_epoch != source_epoch:
                    resume_after_restart = state.phase not in (TrackPhase.IDLE, TrackPhase.DONE)
                    self._discard_pending_ocr(state)
                    state.runtime_epoch += 1
                    state.reset()
                    state.last_received_seq = -1
                    state.source_epoch = source_epoch
                    tracker.reset()
                    gate.reset()
            if packet.seq <= state.last_received_seq:
                self.metrics.out_of_order_frames += 1
                return False
            state.last_received_seq = int(packet.seq)
            packet.metadata[_RUNTIME_EPOCH_METADATA_KEY] = state.runtime_epoch
            detector_frame = packet.detector_frame if packet.detector_frame is not None else packet.frame
            roi = self._rois.get(packet.camera_id)
            adaptive_admission = packet.metadata.get("adaptive_admission") is True
            detector_due = packet.metadata.get("detector_due")
            adaptive_cadence = adaptive_admission and isinstance(detector_due, bool)

            if resume_after_restart:
                state.phase = TrackPhase.ACTIVE
                state.episode_number += 1
                state.last_activity_seq = packet.seq
                self.metrics.motion_wakeups += 1
                # The prior stream was active, so run one detector frame
                # immediately instead of waiting for a new motion delta after
                # the decoder has reset its background/sequence clock.
                return self.queue.submit(packet.camera_id, packet, priority=5)

            if state.phase is TrackPhase.DONE:
                return self._handle_done_frame(packet, detector_frame, roi, state, gate, tracker)

            if state.phase is TrackPhase.IDLE:
                stride = max(1, self.config.idle_stride * self.policy.idle_stride_multiplier)
                if not adaptive_cadence and packet.seq % stride != 0:
                    return False
                score = gate.score(detector_frame, roi)
                self.metrics.motion_evaluations += 1
                if score < gate.config.changed_ratio_threshold:
                    # An adaptively admitted idle packet was consumed by the
                    # motion gate even though no detector job was necessary.
                    return adaptive_cadence
                state.phase = TrackPhase.ACTIVE
                state.episode_number += 1
                state.last_activity_seq = packet.seq
                state.quiet_samples = 0
                tracker.reset()
                self.metrics.motion_wakeups += 1
                return self.queue.submit(packet.camera_id, packet, priority=10)

            detector_stride = max(1, self.config.active_stride * self.policy.detector_stride_multiplier)
            tracking_only = adaptive_cadence and detector_due is False
            if tracking_only or (not adaptive_cadence and packet.seq % detector_stride != 0):
                # Tracking prediction is deliberately cheaper than a detector
                # call and lets us keep gathering high-resolution crops.
                observations = tracker.predict(packet.seq)
                if observations:
                    self.metrics.predicted_track_frames += 1
                    self._harvest_observations(packet, state, observations)
                return tracking_only

            score = gate.score(detector_frame, roi)
            self.metrics.motion_evaluations += 1
            if score >= gate.config.changed_ratio_threshold:
                state.quiet_samples = 0
                state.last_activity_seq = packet.seq
            else:
                state.quiet_samples += 1
            priority = 5 if any(len(track.selector) for track in state.tracks.values()) else 10
            return self.queue.submit(packet.camera_id, packet, priority=priority)

    def process_next(self) -> PlateEvent | None:
        """Run one central detector job and at most one central OCR job."""

        with self._process_lock:
            max_age = min(
                max(0.0, self.config.max_queue_age_seconds),
                max(0.0, self.policy.max_queue_age_ms / 1_000.0),
            )
            packet = self.queue.pop(max_age_seconds=max_age)
            if packet is not None:
                self._process_detector_packet(packet)
            ocr_tasks_before = self.ocr_worker.stats.task_count
            event = self._process_one_ocr()
            if packet is not None or self.ocr_worker.stats.task_count != ocr_tasks_before:
                self._observe_load()
            return event

    def notify_stream_restart(self, camera_id: str, *, preserve_activity: bool = True) -> None:
        """Reset producer sequence/tracking state after an external decoder restart.

        ``DualStreamRTSPProducer`` supplies a ``producer_epoch`` automatically;
        this hook exists for third-party producers that cannot attach metadata.
        """

        with self._state_lock:
            state = self._states.setdefault(camera_id, CameraState())
            was_active = preserve_activity and state.phase not in (TrackPhase.IDLE, TrackPhase.DONE)
            self._discard_pending_ocr(state)
            state.runtime_epoch += 1
            state.episode_number += 1
            state.reset()
            state.last_received_seq = -1
            # The explicit hook represents a new source lifecycle even when a
            # third-party producer cannot provide its own epoch metadata. If a
            # later packet does provide an epoch, it becomes the new baseline
            # instead of causing a second restart transition.
            state.source_epoch = None
            if was_active:
                state.phase = TrackPhase.ACTIVE
            gate = self._gates.setdefault(camera_id, AdaptiveMotionGate())
            gate.reset()
            tracker = self._trackers.setdefault(
                camera_id,
                LightweightMultiObjectTracker(TrackerConfig(max_missed=self.config.tracker_max_missed)),
            )
            tracker.reset()

    def reset_runtime_state(self) -> None:
        """Reset queues/camera episodes while retaining the two shared models.

        This is intended for isolated benchmark scenarios and controlled test
        runs. Production camera reconfiguration should use per-camera stream
        restart handling instead.
        """

        with self._process_lock, self._state_lock:
            self.queue.clear()
            self.ocr_worker.reset()
            self.deduplicator.clear()
            self._states.clear()
            self._gates.clear()
            self._trackers.clear()
            self._rois.clear()
            self.metrics = EngineV2Metrics()
            self.load_controller.reset()
            self._last_load_submitted = 0
            self._last_load_stale = 0

    def process_available(self, limit: int = 128) -> list[PlateEvent]:
        events: list[PlateEvent] = []
        for _ in range(max(1, int(limit))):
            had_work = len(self.queue) > 0 or len(self.ocr_worker.queue) > 0
            if not had_work:
                break
            event = self.process_next()
            if event is not None:
                events.append(event)
        return events

    def inject_load_snapshot(self, snapshot: LoadSnapshot) -> LoadPolicy:
        """Deterministic benchmark/test hook; no system metrics are fabricated."""

        return self.load_controller.observe(snapshot)

    def telemetry(self) -> dict[str, object]:
        with self._state_lock:
            active = sum(
                state.phase not in (TrackPhase.IDLE, TrackPhase.DONE)
                for state in self._states.values()
            )
            total = len(self._states)
            return {
                "frames_received": self.metrics.frames_received,
                "motion_evaluations": self.metrics.motion_evaluations,
                "motion_wakeups": self.metrics.motion_wakeups,
                "detector_inferences": self.metrics.detector_inferences,
                "detector_mean_ms": self.metrics.mean_detector_seconds * 1_000.0,
                "detector_latency_ema_ms": self.metrics.detector_latency_ema_seconds * 1_000.0,
                "ocr_inferences": self.ocr_worker.stats.inference_count,
                "ocr_mean_ms": self.ocr_worker.stats.mean_inference_seconds * 1_000.0,
                "queue_depth": len(self.queue),
                "ocr_queue_depth": len(self.ocr_worker.queue),
                "dropped_stale_frames": self.queue.stats.stale_dropped,
                "queue_replaced": self.queue.stats.replaced,
                "queue_expired": self.queue.stats.expired,
                "events": self.metrics.events_emitted,
                "duplicates_suppressed": self.metrics.duplicates_suppressed,
                "restart_stale_frames": self.metrics.restart_stale_frames,
                "restart_stale_ocr_tasks": self.metrics.restart_stale_ocr_tasks,
                "active_cameras": active,
                "idle_cameras": total - active,
                "load_level": self.load_controller.level.name.lower(),
                "policy": self.policy,
            }

    def _handle_done_frame(
        self,
        packet: FramePacket,
        detector_frame: np.ndarray,
        roi: tuple[int, int, int, int] | None,
        state: CameraState,
        gate: AdaptiveMotionGate,
        tracker: LightweightMultiObjectTracker,
    ) -> bool:
        adaptive_admission = packet.metadata.get("adaptive_admission") is True
        detector_due = packet.metadata.get("detector_due")
        adaptive_cadence = adaptive_admission and isinstance(detector_due, bool)
        stride = max(1, self.config.idle_stride * self.policy.idle_stride_multiplier)
        if not adaptive_cadence and packet.seq % stride != 0:
            return False
        score = gate.score(detector_frame, roi)
        self.metrics.motion_evaluations += 1
        moving = score >= gate.config.changed_ratio_threshold
        if moving:
            state.exit_motion_seen = True
            state.quiet_samples = 0
        elif state.exit_motion_seen:
            state.quiet_samples += 1

        held = packet.seq - state.last_done_seq
        if (
            held >= self.config.done_cooldown_frames
            and state.exit_motion_seen
            and state.quiet_samples >= self.config.exit_quiet_samples
        ) or held >= self.config.max_done_hold_frames:
            state.reset()
            tracker.reset()
            gate.reset()
            if not moving:
                return adaptive_cadence
            # A new motion transition at the bounded DONE timeout is useful
            # detector work for a new passage, so do not throw this frame away.
            state.phase = TrackPhase.ACTIVE
            state.episode_number += 1
            state.last_activity_seq = packet.seq
        if moving:
            # DONE belongs to completed track episodes, not to the whole
            # camera. Probe every motion-gate/adaptive-admission sample so a fast
            # second vehicle cannot disappear inside a camera-wide cooldown.
            # The idle producer cadence limits cost; existing DONE tracks remain
            # in the tracker and _harvest_observations refuses to OCR them again.
            return self.queue.submit(packet.camera_id, packet, priority=10)
        return adaptive_cadence

    def _discard_pending_ocr(self, state: CameraState) -> None:
        discarded = sum(
            self.ocr_worker.queue.discard(episode.episode_id)
            for episode in state.tracks.values()
        )
        self.metrics.restart_stale_ocr_tasks += discarded

    def _process_detector_packet(self, packet: FramePacket) -> None:
        packet_runtime_epoch = packet.metadata.get(_RUNTIME_EPOCH_METADATA_KEY)
        with self._state_lock:
            state = self._states.setdefault(packet.camera_id, CameraState())
            if packet_runtime_epoch != state.runtime_epoch:
                self.metrics.restart_stale_frames += 1
                return
            was_done = state.phase is TrackPhase.DONE

        detector_frame = packet.detector_frame if packet.detector_frame is not None else packet.frame
        started = time.perf_counter()
        with self._detector_lock:
            detected = list(self.detector.detect(detector_frame))
        elapsed = time.perf_counter() - started
        self.metrics.detector_inferences += 1
        self.metrics.detector_seconds += elapsed
        if self.metrics.detector_inferences == 1:
            self.metrics.detector_latency_ema_seconds = elapsed
        else:
            self.metrics.detector_latency_ema_seconds = (
                0.20 * elapsed + 0.80 * self.metrics.detector_latency_ema_seconds
            )

        candidates = [
            self._map_candidate(candidate, detector_frame.shape[:2], packet.frame.shape[:2])
            for candidate in detected
            if candidate.confidence >= self.config.min_detector_confidence
        ]
        with self._state_lock:
            state = self._states.setdefault(packet.camera_id, CameraState())
            # A restart can happen while the shared detector is running. Check
            # the epoch again before detections are allowed to mutate new
            # tracker/episode state.
            if packet_runtime_epoch != state.runtime_epoch:
                self.metrics.restart_stale_frames += 1
                return
            tracker = self._trackers.setdefault(
                packet.camera_id,
                LightweightMultiObjectTracker(TrackerConfig(max_missed=self.config.tracker_max_missed)),
            )
            state.last_detection_seq = packet.seq
            update = tracker.update(candidates, packet.seq)
            if update.observations:
                state.phase = TrackPhase.TRACKING
                state.last_activity_seq = packet.seq
                state.quiet_samples = 0
                self._harvest_observations(packet, state, update.observations)
            for track_id in update.removed_track_ids:
                episode = state.tracks.get(track_id)
                if episode is not None:
                    episode.tracker_removed = True
                    self._maybe_submit_ocr(packet.camera_id, episode, force=True)

            active_episodes = [
                episode for episode in state.tracks.values()
                if episode.phase is not TrackPhase.DONE
            ]
            if was_done and candidates and not active_episodes:
                # Motion woke the detector, but every observation still belongs
                # to a completed track. Preserve camera DONE so active-cadence
                # detection cannot repeatedly OCR or spin on the same vehicle.
                state.phase = TrackPhase.DONE
            if not candidates and not active_episodes and state.quiet_samples >= self.config.active_quiet_samples:
                state.reset()
                tracker.reset()

    def _harvest_observations(
        self,
        packet: FramePacket,
        state: CameraState,
        observations: list[TrackObservation],
    ) -> None:
        skew = packet.metadata.get("main_detector_skew_ms")
        if skew is None and isinstance(packet.metadata.get("main_age_seconds"), (int, float)):
            skew = float(packet.metadata["main_age_seconds"]) * 1_000.0
        if isinstance(skew, (int, float)) and abs(float(skew)) > self.config.max_main_stream_skew_ms:
            return

        for observation in observations:
            episode = state.tracks.get(observation.track_id)
            if episode is None:
                episode_id = (
                    f"{packet.camera_id}:{state.runtime_epoch}:"
                    f"{state.episode_number}:{observation.track_id}"
                )
                episode = TrackEpisode(
                    track_id=observation.track_id,
                    episode_id=episode_id,
                    first_seq=packet.seq,
                    last_seq=packet.seq,
                    last_ts=packet.ts,
                    selector=BestPlateFrameSelector(self.config.selector_capacity, min_sequence_gap=0),
                )
                state.tracks[observation.track_id] = episode
            if episode.phase in (TrackPhase.DONE, TrackPhase.OCR):
                continue

            episode.last_seq = packet.seq
            episode.last_ts = packet.ts
            episode.last_bbox = observation.bbox
            episode.advance(TrackPhase.PLATE_FOUND)
            padded = self._pad_bbox(observation.bbox, packet.frame.shape[:2])
            crop = self._crop_bbox(packet.frame, padded)
            added = episode.selector.add(
                crop,
                bbox=observation.bbox,
                seq=packet.seq,
                ts=packet.ts,
                detector_confidence=observation.confidence,
                frame_shape=packet.frame.shape[:2],
            )
            if added is None:
                continue
            episode.advance(TrackPhase.COLLECTING)
            state.phase = TrackPhase.COLLECTING
            best = episode.selector.best
            if best is not None and best.quality.score >= state.best_quality:
                state.best_quality = best.quality.score
                state.best_crop = best.crop.copy()
                state.best_bbox = best.bbox
            self._maybe_submit_ocr(packet.camera_id, episode)

    def _maybe_submit_ocr(self, camera_id: str, episode: TrackEpisode, force: bool = False) -> bool:
        if episode.ocr_submitted or episode.phase is TrackPhase.DONE:
            return False
        if episode.ocr_attempts >= self.config.max_ocr_attempts:
            episode.advance(TrackPhase.DONE)
            return False
        selected = episode.selector.selected(
            self.config.max_ocr_candidates,
            min_quality=self.config.min_quality,
        )
        if not selected:
            return False
        age = episode.last_seq - episode.first_seq
        enough = len(selected) >= self.config.min_candidates_before_ocr
        excellent = selected[0].quality.score >= self.config.early_ocr_quality
        timed_out = age >= self.config.max_collection_frames
        if not (force or enough or excellent or timed_out):
            return False

        limit = min(self.config.max_ocr_candidates, self.policy.max_ocr_candidates)
        # During critical load a mediocre single crop is deferred or, when the
        # vehicle is leaving, paired with one more candidate to protect accuracy.
        if limit == 1 and selected[0].quality.score < self.config.early_ocr_quality:
            if not force:
                return False
            limit = min(2, len(selected))
        selected = selected[: max(1, limit)]
        episode.ocr_submitted = True
        episode.ocr_attempts += 1
        episode.advance(TrackPhase.OCR)
        state = self._states[camera_id]
        state.phase = TrackPhase.OCR
        accepted = self.ocr_worker.submit(
            OCRTask(
                key=episode.episode_id,
                crops=[frame.crop for frame in selected],
                qualities=[frame.quality.score for frame in selected],
                sequences=[frame.seq for frame in selected],
                priority=5 if force else 10,
                metadata={
                    "camera_id": camera_id,
                    "track_id": episode.track_id,
                    "episode_id": episode.episode_id,
                    "runtime_epoch": state.runtime_epoch,
                    "bbox": selected[0].bbox,
                    "seq": episode.last_seq,
                    "ts": episode.last_ts,
                    "quality": selected[0].quality.score,
                    "force": force,
                },
            )
        )
        if accepted is False:
            episode.ocr_submitted = False
            episode.ocr_attempts -= 1
            episode.advance(TrackPhase.COLLECTING)
            self._states[camera_id].phase = TrackPhase.COLLECTING
            return False
        return True

    def _process_one_ocr(self) -> PlateEvent | None:
        processed = self.ocr_worker.process_next()
        if processed is None:
            return None
        task, vote = processed
        camera_id = str(task.metadata["camera_id"])
        track_id = int(task.metadata["track_id"])
        with self._state_lock:
            state = self._states.get(camera_id)
            episode = state.tracks.get(track_id) if state is not None else None
            task_runtime_epoch = task.metadata.get("runtime_epoch")
            if (
                state is None
                or task_runtime_epoch != state.runtime_epoch
                or episode is None
                or episode.episode_id != task.key
            ):
                return None
            episode.observations.extend(item.result for item in vote.results)
            state.observations.extend(item.result for item in vote.results)

            if not vote.valid or vote.confidence < self.config.min_ocr_confidence:
                episode.ocr_submitted = False
                terminal = (
                    episode.tracker_removed
                    or episode.ocr_attempts >= self.config.max_ocr_attempts
                )
                if terminal:
                    episode.advance(TrackPhase.DONE)
                    state.last_done_seq = int(task.metadata["seq"])
                    unfinished = any(item.phase is not TrackPhase.DONE for item in state.tracks.values())
                    state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
                else:
                    episode.advance(TrackPhase.COLLECTING)
                    state.phase = TrackPhase.COLLECTING
                return None

            episode.advance(TrackPhase.VALIDATED)
            duplicate = self.deduplicator.check_and_record(camera_id, vote.text, float(task.metadata["ts"]))
            episode.advance(TrackPhase.DONE)
            episode.event_emitted = not duplicate.duplicate
            state.last_done_seq = int(task.metadata["seq"])
            unfinished = any(item.phase is not TrackPhase.DONE for item in state.tracks.values())
            state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
            if duplicate.duplicate:
                self.metrics.duplicates_suppressed += 1
                return None

            event = PlateEvent(
                camera_id=camera_id,
                frame_seq=int(task.metadata["seq"]),
                ts=float(task.metadata["ts"]),
                text=vote.text,
                confidence=float(vote.confidence),
                bbox=tuple(task.metadata["bbox"]),
                quality=float(task.metadata["quality"]),
                track_id=str(track_id),
                episode_id=episode.episode_id,
                observations=vote.observations,
                metadata={
                    "vote_reason": vote.reason,
                    "vote_support": vote.support,
                    "load_level": self.load_controller.level.name.lower(),
                },
            )
            self.metrics.events_emitted += 1

        if self.on_event is not None:
            self.on_event(event)
        return event

    def _observe_load(self) -> None:
        if not self.config.load_control_enabled:
            return
        with self._state_lock:
            active = sum(
                state.phase not in (TrackPhase.IDLE, TrackPhase.DONE)
                for state in self._states.values()
            )
            total = len(self._states)
        submitted_total = self.queue.stats.submitted
        stale_total = self.queue.stats.stale_dropped
        submitted_delta = max(0, submitted_total - self._last_load_submitted)
        stale_delta = max(0, stale_total - self._last_load_stale)
        self._last_load_submitted = submitted_total
        self._last_load_stale = stale_total
        snapshot = LoadSnapshot(
            timestamp=self.load_sampler.now(),
            cpu_percent=self.load_sampler.cpu_percent(),
            detector_latency_ms=self.metrics.detector_latency_ema_seconds * 1_000.0,
            ocr_latency_ms=self.ocr_worker.stats.last_inference_seconds * 1_000.0,
            queue_depth=len(self.queue),
            queue_capacity=self.config.queue_size,
            active_cameras=active,
            total_cameras=total,
            stale_drop_rate=stale_delta / max(1, submitted_delta),
        )
        self.load_controller.observe(snapshot)

    @staticmethod
    def _map_candidate(
        candidate: PlateCandidate,
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> PlateCandidate:
        sh, sw = source_hw
        th, tw = target_hw
        if sw <= 0 or sh <= 0 or (sw == tw and sh == th):
            return candidate
        sx = tw / float(sw)
        sy = th / float(sh)
        x1, y1, x2, y2 = candidate.bbox
        return PlateCandidate(
            (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)),
            candidate.confidence,
            candidate.class_id,
            candidate.track_hint,
        )

    def _pad_bbox(
        self,
        bbox: tuple[int, int, int, int],
        frame_hw: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = bbox
        h, w = frame_hw
        pad_x = round(max(0, x2 - x1) * self.config.crop_padding_ratio)
        pad_y = round(max(0, y2 - y1) * self.config.crop_padding_ratio)
        return (
            max(0, x1 - pad_x),
            max(0, y1 - pad_y),
            min(w, x2 + pad_x),
            min(h, y2 + pad_y),
        )

    @staticmethod
    def _crop_bbox(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1, x2 = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
        y1, y2 = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def _crop(frame: np.ndarray, candidate: PlateCandidate) -> np.ndarray:
        """Compatibility helper retained for first-slice callers/tests."""

        return EventDrivenANPREngine._crop_bbox(frame, candidate.bbox)
