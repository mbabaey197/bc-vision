from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from .dedup import DuplicateSuppressor, DuplicateSuppressorConfig
from .load import AdaptiveLoadController, LoadPolicy, LoadSnapshot, SystemLoadSampler
from .motion import AdaptiveMotionGate
from .ocr import (
    AbandonedOCRTask,
    OCRInFlightHandle,
    OCRTask,
    OCRVote,
    SharedOCRWorker,
    TemporalOCRVoter,
)
from .quality import BestPlateFrameSelector
from .scheduler import LatestOnlyPriorityQueue
from .streams import ProducerActivity, ProducerCadencePolicy
from .tcam import (
    PlateEvidenceAccumulator,
    RecognitionPhase,
    TemporalFusionConfig,
    TrackRecognitionSession,
)
from .tracking import LightweightMultiObjectTracker, TrackerConfig, TrackObservation
from .types import (
    FramePacket,
    OCRResult,
    PlateCandidate,
    PlateDetector,
    PlateEvent,
    PlateOCR,
    TrackPhase,
)
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
    # Opt-in while the new track-centric policy is calibrated against V1.
    # The legacy Engine V2 voting path remains the control in the same branch.
    track_temporal_fusion_enabled: bool = False
    temporal_fusion: TemporalFusionConfig = field(default_factory=TemporalFusionConfig)

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
        if (
            self.same_camera_duplicate_seconds < 0
            or self.cross_camera_duplicate_seconds < 0
        ):
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
        default_factory=lambda: [
            TrackPhase.IDLE,
            TrackPhase.ACTIVE,
            TrackPhase.TRACKING,
        ]
    )
    ocr_submitted: bool = False
    ocr_attempts: int = 0
    event_emitted: bool = False
    emitted_event: PlateEvent | None = None
    pending_ocr_result: tuple[OCRTask, OCRVote] | None = None
    tracker_removed: bool = False
    last_bbox: tuple[int, int, int, int] | None = None
    observations: list[OCRResult] = field(default_factory=list)
    recognition: TrackRecognitionSession | None = None
    ocr_sequences_seen: set[int] = field(default_factory=set)
    last_ocr_schedule_reason: str | None = None

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
    input_finalized: bool = False
    finalization_complete: bool = False
    final_seq: int | None = None
    final_ts: float | None = None
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
        self.queue: LatestOnlyPriorityQueue[FramePacket] = LatestOnlyPriorityQueue(
            self.config.queue_size
        )
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
        self._runtime_generation = 0
        self._last_load_submitted = 0
        self._last_load_stale = 0

    @property
    def policy(self) -> LoadPolicy:
        return self.load_controller.policy

    def _new_recognition_session(self) -> TrackRecognitionSession | None:
        if not self.config.track_temporal_fusion_enabled:
            return None
        policy = replace(
            self.config.temporal_fusion,
            min_ocr_quality=self.config.min_quality,
        )
        return TrackRecognitionSession(PlateEvidenceAccumulator(self.validator, policy))

    def set_roi(self, camera_id: str, roi: tuple[int, int, int, int] | None) -> None:
        with self._state_lock:
            self._rois[camera_id] = roi

    def _state_for_locked(self, camera_id: str) -> CameraState:
        state = self._states.get(camera_id)
        if state is None:
            state = CameraState(runtime_epoch=self._runtime_generation)
            self._states[camera_id] = state
        return state

    def state_for(self, camera_id: str) -> CameraState:
        with self._state_lock:
            return self._state_for_locked(camera_id)

    def target_detector_fps(self, camera_id: str, source_fps: float = 25.0) -> float:
        with self._state_lock:
            phase = self._state_for_locked(camera_id).phase
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
            stride = max(
                1, self.config.active_stride * policy.detector_stride_multiplier
            )
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
            phase = self._state_for_locked(camera_id).phase
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
            state = self._state_for_locked(packet.camera_id)
            gate = self._gates.setdefault(packet.camera_id, AdaptiveMotionGate())
            tracker = self._trackers.setdefault(
                packet.camera_id,
                LightweightMultiObjectTracker(
                    TrackerConfig(max_missed=self.config.tracker_max_missed)
                ),
            )

            source_epoch_value = packet.metadata.get("producer_epoch")
            source_epoch = (
                None if source_epoch_value is None else str(source_epoch_value)
            )
            resume_after_restart = False
            reopen_finalized = (
                state.input_finalized
                and source_epoch is not None
                and source_epoch != state.source_epoch
            )
            if state.input_finalized and not reopen_finalized:
                return False
            if reopen_finalized:
                # A finalized source is immutable.  Only a producer lifecycle
                # with a different explicit epoch can reopen it implicitly;
                # third-party producers use ``notify_stream_restart``.
                self._discard_pending_ocr(state)
                state.runtime_epoch += 1
                state.episode_number += 1
                state.reset()
                state.last_received_seq = -1
                state.source_epoch = source_epoch
                state.input_finalized = False
                state.finalization_complete = False
                state.final_seq = None
                state.final_ts = None
                tracker.reset()
                gate.reset()
            elif source_epoch is not None:
                if state.source_epoch is None:
                    state.source_epoch = source_epoch
                elif state.source_epoch != source_epoch:
                    resume_after_restart = state.phase not in (
                        TrackPhase.IDLE,
                        TrackPhase.DONE,
                    )
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
            detector_frame = (
                packet.detector_frame
                if packet.detector_frame is not None
                else packet.frame
            )
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
                return self._handle_done_frame(
                    packet, detector_frame, roi, state, gate, tracker
                )

            if state.phase is TrackPhase.IDLE:
                stride = max(
                    1, self.config.idle_stride * self.policy.idle_stride_multiplier
                )
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

            detector_stride = max(
                1, self.config.active_stride * self.policy.detector_stride_multiplier
            )
            tracking_only = adaptive_cadence and detector_due is False
            if tracking_only or (
                not adaptive_cadence and packet.seq % detector_stride != 0
            ):
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
            priority = (
                5 if any(len(track.selector) for track in state.tracks.values()) else 10
            )
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
            abandoned_before = self.ocr_worker.stats.abandoned_task_count
            event = self._process_one_ocr()
            if (
                packet is not None
                or self.ocr_worker.stats.task_count != ocr_tasks_before
                or self.ocr_worker.stats.abandoned_task_count != abandoned_before
            ):
                self._observe_load()
            return event

    def finalize_camera(
        self,
        camera_id: str,
        *,
        final_seq: int | None = None,
        final_ts: float | None = None,
    ) -> list[PlateEvent]:
        """Finalize one producer lifecycle and drain its real OCR evidence.

        This method is the V2 end-of-input contract for finite media and
        explicitly stopped streams.  It never invents a frame or detection:
        only candidates already harvested from submitted frames are eligible.
        ``final_seq`` and ``final_ts`` describe the producer boundary for
        observability; emitted events retain the sequence and timestamp of
        their actual best evidence.

        The call is serialized with central processing, rejects later packets
        from the same producer lifecycle, forces each eligible unfinished track
        through the one shared OCR session, and terminally drops tracks below
        the quality floor.  OCR work belonging to other cameras stays queued.
        Repeating the call returns an empty list.
        """

        normalized_camera_id = str(camera_id)
        if not normalized_camera_id:
            raise ValueError("camera_id cannot be empty")
        normalized_final_seq: int | None = None
        if final_seq is not None:
            normalized_final_seq = int(final_seq)
            if isinstance(final_seq, bool) or normalized_final_seq != final_seq:
                raise ValueError("final_seq must be an integer")
        normalized_final_ts: float | None = None
        if final_ts is not None:
            normalized_final_ts = float(final_ts)
            if not math.isfinite(normalized_final_ts):
                raise ValueError("final_ts must be finite")

        tasks: list[OCRTask] = []
        completed_results: list[tuple[OCRTask, OCRVote]] = []
        events: list[PlateEvent] = []
        event_episode_ids: set[str] = set()
        notify_events: list[PlateEvent] = []
        callback_errors: list[Exception] = []
        processing_errors: list[Exception] = []
        failed_episode_ids: set[str] = set()
        with self._process_lock:
            with self._state_lock:
                # EOF is a lifecycle fact even if no frame reached the runtime.
                # Latching an empty state prevents a late producer packet from
                # silently creating work after its end-of-input notification.
                state = self._state_for_locked(normalized_camera_id)
                if state.input_finalized and state.finalization_complete:
                    return []

                if not state.input_finalized:
                    # Latch EOF before releasing the state lock. A racing
                    # producer packet is therefore either admitted before this
                    # boundary or rejected after it; it cannot resurrect a
                    # drained episode.
                    state.input_finalized = True
                    state.finalization_complete = False
                    state.final_seq = (
                        normalized_final_seq
                        if normalized_final_seq is not None
                        else (
                            state.last_received_seq
                            if state.last_received_seq >= 0
                            else None
                        )
                    )
                    state.final_ts = normalized_final_ts

                # A queued detector packet has not produced evidence yet.  EOF
                # drops that job instead of fabricating a replacement frame or
                # consuming scheduler work belonging to another camera.
                self.queue.discard(normalized_camera_id)

                preexisting_events = {
                    episode.episode_id
                    for episode in state.tracks.values()
                    if episode.event_emitted
                }
                pending_episode_ids: list[str] = []
                for episode in state.tracks.values():
                    if episode.phase is TrackPhase.DONE or episode.event_emitted:
                        episode.selector.clear()
                        continue
                    episode.tracker_removed = True
                    if episode.ocr_submitted:
                        pending_episode_ids.append(episode.episode_id)

            # Queue removal and background ownership publication are atomic in
            # SharedOCRWorker.  Wait for the exact in-flight task outside the
            # state lock so its callback can complete; never run OCR twice.
            ownership: dict[str, tuple[bool, OCRInFlightHandle | None]] = {}
            for episode_id in pending_episode_ids:
                discarded, handle = self.ocr_worker.cancel_queued_or_observe_inflight(
                    episode_id
                )
                ownership[episode_id] = (discarded, handle)
            for _, handle in ownership.values():
                if handle is not None:
                    handle.done.wait()
                    if handle.callback_error is not None:
                        callback_errors.append(handle.callback_error)

            with self._state_lock:
                state = self._states[normalized_camera_id]
                for episode in state.tracks.values():
                    if (
                        episode.event_emitted
                        and episode.episode_id not in preexisting_events
                        and episode.emitted_event is not None
                        and episode.episode_id not in event_episode_ids
                    ):
                        events.append(episode.emitted_event)
                        event_episode_ids.add(episode.episode_id)

                    if episode.phase is TrackPhase.DONE or episode.event_emitted:
                        episode.selector.clear()
                        continue

                    if episode.pending_ocr_result is not None:
                        completed_results.append(episode.pending_ocr_result)
                        continue

                    ownership_result = ownership.get(episode.episode_id)
                    if episode.ocr_submitted and ownership_result is not None:
                        discarded, handle = ownership_result
                        if handle is not None and handle.processed is not None:
                            # The background callback did not apply the result
                            # to this engine. Reuse its completed inference.
                            completed_results.append(handle.processed)
                            continue
                        # A queued task was cancelled, or a completed callback
                        # left no applicable result. Rebuild the logical attempt
                        # from the existing real candidates without charging a
                        # retry. No inference has to be repeated in this branch.
                        episode.ocr_submitted = False
                        if discarded:
                            if episode.recognition is None:
                                episode.ocr_attempts = max(0, episode.ocr_attempts - 1)
                            elif episode.recognition.in_flight:
                                episode.recognition.release_ocr(retryable=True)
                                episode.ocr_attempts = episode.recognition.attempts
                                if episode.recognition.last_ocr_seq is not None:
                                    episode.ocr_sequences_seen.discard(
                                        episode.recognition.last_ocr_seq
                                    )

                    task = self._prepare_ocr_task(
                        normalized_camera_id,
                        episode,
                        force=True,
                    )
                    if task is not None:
                        tasks.append(task)
                        continue

                    # No real candidate survives the configured quality floor.
                    # The episode is terminal and its crops can be released.
                    episode.ocr_submitted = False
                    if episode.recognition is not None:
                        episode.recognition.close()
                    episode.advance(TrackPhase.DONE)
                    state.last_done_seq = max(
                        state.last_done_seq,
                        state.final_seq
                        if state.final_seq is not None
                        else episode.last_seq,
                    )
                    episode.selector.clear()

            for processed_task, vote in completed_results:
                try:
                    event = self._apply_ocr_result(
                        processed_task,
                        vote,
                        notify_event=False,
                    )
                except Exception as exc:
                    processing_errors.append(exc)
                    applied_event: PlateEvent | None = None
                    with self._state_lock:
                        state = self._states.get(normalized_camera_id)
                        track_value = processed_task.metadata.get("track_id")
                        try:
                            track_id = int(track_value)
                        except (TypeError, ValueError):
                            track_id = -1
                        episode = (
                            state.tracks.get(track_id) if state is not None else None
                        )
                        if (
                            episode is not None
                            and episode.episode_id == processed_task.key
                        ):
                            if episode.phase is TrackPhase.DONE:
                                # An exception raised after the state commit
                                # must not hide or retry an already-emitted
                                # event. Preserve it for callback/return.
                                applied_event = episode.emitted_event
                                episode.pending_ocr_result = None
                            else:
                                failed_episode_ids.add(processed_task.key)
                                # OCR already completed; retain its exact vote
                                # so a safe retry never repeats inference.
                                episode.pending_ocr_result = (processed_task, vote)
                    if (
                        applied_event is not None
                        and applied_event.episode_id not in event_episode_ids
                    ):
                        events.append(applied_event)
                        event_episode_ids.add(str(applied_event.episode_id))
                        notify_events.append(applied_event)
                    continue
                if event is not None and event.episode_id not in event_episode_ids:
                    events.append(event)
                    event_episode_ids.add(str(event.episode_id))
                    notify_events.append(event)
                with self._state_lock:
                    state = self._states.get(normalized_camera_id)
                    track_value = processed_task.metadata.get("track_id")
                    try:
                        track_id = int(track_value)
                    except (TypeError, ValueError):
                        track_id = -1
                    episode = state.tracks.get(track_id) if state is not None else None
                    if episode is not None and episode.episode_id == processed_task.key:
                        episode.pending_ocr_result = None

            for task in tasks:
                try:
                    processed_task, vote = self.ocr_worker.process_task(task)
                except Exception as exc:
                    processing_errors.append(exc)
                    failed_episode_ids.add(task.key)
                    continue
                try:
                    event = self._apply_ocr_result(
                        processed_task,
                        vote,
                        notify_event=False,
                    )
                except Exception as exc:
                    processing_errors.append(exc)
                    applied_event: PlateEvent | None = None
                    with self._state_lock:
                        state = self._states.get(normalized_camera_id)
                        track_value = processed_task.metadata.get("track_id")
                        try:
                            track_id = int(track_value)
                        except (TypeError, ValueError):
                            track_id = -1
                        episode = (
                            state.tracks.get(track_id) if state is not None else None
                        )
                        if (
                            episode is not None
                            and episode.episode_id == processed_task.key
                        ):
                            if episode.phase is TrackPhase.DONE:
                                applied_event = episode.emitted_event
                                episode.pending_ocr_result = None
                            else:
                                failed_episode_ids.add(processed_task.key)
                                episode.pending_ocr_result = (processed_task, vote)
                    if (
                        applied_event is not None
                        and applied_event.episode_id not in event_episode_ids
                    ):
                        events.append(applied_event)
                        event_episode_ids.add(str(applied_event.episode_id))
                        notify_events.append(applied_event)
                    continue
                if event is not None and event.episode_id not in event_episode_ids:
                    events.append(event)
                    event_episode_ids.add(str(event.episode_id))
                    notify_events.append(event)

            with self._state_lock:
                state = self._states.get(normalized_camera_id)
                if state is not None and state.input_finalized:
                    for episode in state.tracks.values():
                        self.ocr_worker.queue.discard(episode.episode_id)
                        episode.ocr_submitted = False
                        if episode.recognition is not None:
                            episode.recognition.release_ocr()
                        if episode.episode_id in failed_episode_ids:
                            if episode.pending_ocr_result is None:
                                episode.ocr_attempts = max(0, episode.ocr_attempts - 1)
                            if episode.phase is not TrackPhase.DONE:
                                episode.advance(TrackPhase.COLLECTING)
                            continue
                        episode.pending_ocr_result = None
                        if episode.phase is not TrackPhase.DONE:
                            episode.advance(TrackPhase.DONE)
                        episode.selector.clear()
                    state.finalization_complete = not failed_episode_ids
                    if state.finalization_complete:
                        if state.final_seq is not None:
                            state.last_done_seq = max(
                                state.last_done_seq, state.final_seq
                            )
                        state.best_quality = 0.0
                        state.best_crop = None
                        state.best_bbox = None
                        state.phase = (
                            TrackPhase.DONE if state.tracks else TrackPhase.IDLE
                        )
                    else:
                        state.phase = TrackPhase.COLLECTING
                    tracker = self._trackers.get(normalized_camera_id)
                    if tracker is not None:
                        tracker.reset()
                    gate = self._gates.get(normalized_camera_id)
                    if gate is not None:
                        gate.reset()

            # Callback failures are isolated from OCR/state processing. Every
            # event and remaining track is finalized before the first failure
            # is re-raised to preserve the public error signal safely.
            if self.on_event is not None:
                for event in notify_events:
                    try:
                        self.on_event(event)
                    except Exception as exc:
                        callback_errors.append(exc)

            all_errors = processing_errors + callback_errors
            if all_errors:
                first_error = all_errors[0]
                if len(all_errors) > 1:
                    first_error.add_note(
                        f"{len(all_errors) - 1} additional finalize error(s) occurred"
                    )
                raise first_error

        return events

    def notify_stream_restart(
        self, camera_id: str, *, preserve_activity: bool = True
    ) -> None:
        """Reset producer sequence/tracking state after an external decoder restart.

        ``DualStreamRTSPProducer`` supplies a ``producer_epoch`` automatically;
        this hook exists for third-party producers that cannot attach metadata.
        """

        with self._state_lock:
            state = self._state_for_locked(camera_id)
            was_active = preserve_activity and state.phase not in (
                TrackPhase.IDLE,
                TrackPhase.DONE,
            )
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
            state.input_finalized = False
            state.finalization_complete = False
            state.final_seq = None
            state.final_ts = None
            if was_active:
                state.phase = TrackPhase.ACTIVE
            gate = self._gates.setdefault(camera_id, AdaptiveMotionGate())
            gate.reset()
            tracker = self._trackers.setdefault(
                camera_id,
                LightweightMultiObjectTracker(
                    TrackerConfig(max_missed=self.config.tracker_max_missed)
                ),
            )
            tracker.reset()

    def reset_runtime_state(self) -> None:
        """Reset queues/camera episodes while retaining the two shared models.

        This is intended for isolated benchmark scenarios and controlled test
        runs. Production camera reconfiguration should use per-camera stream
        restart handling instead.
        """

        with self._process_lock, self._state_lock:
            # Never reuse an epoch after clearing state.  An OCR callback that
            # was already in flight before reset can then be rejected even if a
            # new camera/track happens to reuse the same visible identifiers.
            self._runtime_generation += 1
            self.queue.clear(reset_stats=True)
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
            recognition_sessions = [
                episode.recognition
                for state in self._states.values()
                for episode in state.tracks.values()
                if episode.recognition is not None
            ]
            recognition_phases = [
                session.decision.phase for session in recognition_sessions
            ]
            return {
                "frames_received": self.metrics.frames_received,
                "motion_evaluations": self.metrics.motion_evaluations,
                "motion_wakeups": self.metrics.motion_wakeups,
                "detector_inferences": self.metrics.detector_inferences,
                "detector_mean_ms": self.metrics.mean_detector_seconds * 1_000.0,
                "detector_latency_ema_ms": self.metrics.detector_latency_ema_seconds
                * 1_000.0,
                "ocr_inferences": self.ocr_worker.stats.inference_count,
                "ocr_mean_ms": self.ocr_worker.stats.mean_inference_seconds * 1_000.0,
                "ocr_abandoned_tasks": self.ocr_worker.stats.abandoned_task_count,
                "ocr_expired_tasks": self.ocr_worker.stats.expired_task_count,
                "ocr_evicted_tasks": self.ocr_worker.stats.evicted_task_count,
                "queue_depth": len(self.queue),
                "ocr_queue_depth": len(self.ocr_worker.queue),
                "dropped_stale_frames": self.queue.stats.stale_dropped,
                "queue_replaced": self.queue.stats.replaced,
                "queue_expired": self.queue.stats.expired,
                "events": self.metrics.events_emitted,
                "duplicates_suppressed": self.metrics.duplicates_suppressed,
                "restart_stale_frames": self.metrics.restart_stale_frames,
                "restart_stale_ocr_tasks": self.metrics.restart_stale_ocr_tasks,
                "track_temporal_fusion_enabled": self.config.track_temporal_fusion_enabled,
                "fusion_tracks": len(recognition_sessions),
                "fusion_provisional_tracks": recognition_phases.count(
                    RecognitionPhase.PROVISIONAL
                ),
                "fusion_locked_tracks": recognition_phases.count(
                    RecognitionPhase.SOFT_LOCKED
                ),
                "fusion_finalized_tracks": recognition_phases.count(
                    RecognitionPhase.FINALIZED
                ),
                "fusion_ocr_attempts": sum(
                    session.attempts for session in recognition_sessions
                ),
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
            state = self._state_for_locked(packet.camera_id)
            if packet_runtime_epoch != state.runtime_epoch:
                self.metrics.restart_stale_frames += 1
                return
            was_done = state.phase is TrackPhase.DONE

        detector_frame = (
            packet.detector_frame if packet.detector_frame is not None else packet.frame
        )
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
            self._map_candidate(
                candidate, detector_frame.shape[:2], packet.frame.shape[:2]
            )
            for candidate in detected
            if candidate.confidence >= self.config.min_detector_confidence
        ]
        with self._state_lock:
            state = self._state_for_locked(packet.camera_id)
            # A restart can happen while the shared detector is running. Check
            # the epoch again before detections are allowed to mutate new
            # tracker/episode state.
            if packet_runtime_epoch != state.runtime_epoch:
                self.metrics.restart_stale_frames += 1
                return
            tracker = self._trackers.setdefault(
                packet.camera_id,
                LightweightMultiObjectTracker(
                    TrackerConfig(max_missed=self.config.tracker_max_missed)
                ),
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
                    submitted = self._maybe_submit_ocr(
                        packet.camera_id, episode, force=True
                    )
                    if (
                        not submitted
                        and not episode.ocr_submitted
                        and episode.phase is not TrackPhase.DONE
                    ):
                        # The track has left and no candidate meets the quality
                        # floor. Avoid OCR on bad evidence without leaving an
                        # episode active when no future crops can arrive.
                        episode.advance(TrackPhase.DONE)
                        state.last_done_seq = max(state.last_done_seq, episode.last_seq)

            active_episodes = [
                episode
                for episode in state.tracks.values()
                if episode.phase is not TrackPhase.DONE
            ]
            if state.tracks and not active_episodes:
                state.phase = TrackPhase.DONE
            if was_done and candidates and not active_episodes:
                # Motion woke the detector, but every observation still belongs
                # to a completed track. Preserve camera DONE so active-cadence
                # detection cannot repeatedly OCR or spin on the same vehicle.
                state.phase = TrackPhase.DONE
            if (
                not candidates
                and not active_episodes
                and state.quiet_samples >= self.config.active_quiet_samples
            ):
                state.reset()
                tracker.reset()

    def _harvest_observations(
        self,
        packet: FramePacket,
        state: CameraState,
        observations: list[TrackObservation],
    ) -> None:
        skew = packet.metadata.get("main_detector_skew_ms")
        if skew is None and isinstance(
            packet.metadata.get("main_age_seconds"), (int, float)
        ):
            skew = float(packet.metadata["main_age_seconds"]) * 1_000.0
        if (
            isinstance(skew, (int, float))
            and abs(float(skew)) > self.config.max_main_stream_skew_ms
        ):
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
                    selector=BestPlateFrameSelector(
                        self.config.selector_capacity, min_sequence_gap=0
                    ),
                    recognition=self._new_recognition_session(),
                )
                state.tracks[observation.track_id] = episode
            if episode.phase is TrackPhase.DONE or (
                episode.phase is TrackPhase.OCR and episode.recognition is None
            ):
                continue

            # A detector may finish after a newer tracking-only packet was
            # harvested. Keep the newest episode clock/evidence pointer; the
            # older detector crop may still be a useful quality candidate.
            if packet.seq >= episode.last_seq:
                episode.last_seq = packet.seq
                episode.last_ts = packet.ts
                episode.last_bbox = observation.bbox
            ocr_in_flight = episode.ocr_submitted
            if not ocr_in_flight:
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
            if not ocr_in_flight:
                episode.advance(TrackPhase.COLLECTING)
                state.phase = TrackPhase.COLLECTING
            best = episode.selector.best
            if best is not None and best.quality.score >= state.best_quality:
                state.best_quality = best.quality.score
                state.best_crop = best.crop.copy()
                state.best_bbox = best.bbox
            if not ocr_in_flight:
                self._maybe_submit_ocr(packet.camera_id, episode)

    def _maybe_submit_ocr(
        self, camera_id: str, episode: TrackEpisode, force: bool = False
    ) -> bool:
        task = self._prepare_ocr_task(camera_id, episode, force=force)
        if task is None:
            return False
        accepted = self.ocr_worker.submit(task)
        if accepted is False:
            episode.ocr_submitted = False
            finalize_only = bool(task.metadata.get("fusion_finalize"))
            if not finalize_only:
                episode.ocr_attempts -= 1
            if episode.recognition is not None and not finalize_only:
                episode.recognition.release_ocr(retryable=True)
            episode.advance(TrackPhase.COLLECTING)
            self._states[camera_id].phase = TrackPhase.COLLECTING
            return False
        episode.ocr_sequences_seen.update(task.sequences)
        return True

    def _prepare_ocr_task(
        self,
        camera_id: str,
        episode: TrackEpisode,
        *,
        force: bool = False,
    ) -> OCRTask | None:
        """Build and reserve one OCR attempt without choosing its transport."""

        if episode.ocr_submitted or episode.phase is TrackPhase.DONE:
            return None
        recognition = episode.recognition
        if recognition is not None and recognition.decision.finalized:
            return None
        attempt_limit = (
            recognition.config.max_ocr_attempts
            if recognition is not None
            else self.config.max_ocr_attempts
        )
        if episode.ocr_attempts >= attempt_limit and (
            recognition is None or not recognition.decision.soft_locked
        ):
            episode.advance(TrackPhase.DONE)
            return None
        selection_limit = (
            self.config.selector_capacity
            if recognition is not None
            else self.config.max_ocr_candidates
        )
        selected = episode.selector.selected(
            selection_limit,
            min_quality=self.config.min_quality,
        )
        if not selected:
            return None
        schedule_reason: str | None = None
        finalize_only = False
        finalization_reason: str | None = None
        if recognition is not None:
            evidence_frame = max(selected, key=lambda frame: frame.seq)
            fresh = [
                frame
                for frame in selected
                if frame.seq not in episode.ocr_sequences_seen
            ]
            current = recognition.decision
            if current.soft_locked:
                schedule = None
                if fresh:
                    audit_frame = max(fresh, key=lambda frame: frame.seq)
                    x1, y1, x2, y2 = audit_frame.bbox
                    schedule = recognition.should_schedule_ocr(
                        seq=audit_frame.seq,
                        ts=audit_frame.ts,
                        quality=audit_frame.quality.score,
                        bbox_area=max(0, x2 - x1) * max(0, y2 - y1),
                        plate_width=max(0, x2 - x1),
                        plate_height=max(0, y2 - y1),
                        near_exit=force,
                    )
                if schedule is not None and schedule.run_ocr:
                    selected = [audit_frame]
                elif recognition.should_finalize(ts=episode.last_ts, near_exit=force):
                    selected = [evidence_frame]
                    finalize_only = True
                    if force:
                        finalization_reason = "track_exit"
                    elif (
                        recognition.config.max_audit_attempts > 0
                        and recognition.audit_attempts
                        >= recognition.config.max_audit_attempts
                    ):
                        finalization_reason = "audit_complete"
                    else:
                        finalization_reason = "soft_lock_hold_elapsed"
                else:
                    return None
            else:
                if not fresh:
                    return None
                selected = [max(fresh, key=lambda frame: frame.seq)]

            if not finalize_only:
                # Read one fresh crop per attempt. Already-read crops live in
                # the accumulator and are never billed to OCR twice.
                frame = selected[0]
                x1, y1, x2, y2 = frame.bbox
                bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
                schedule = recognition.should_schedule_ocr(
                    seq=frame.seq,
                    ts=frame.ts,
                    quality=frame.quality.score,
                    bbox_area=bbox_area,
                    plate_width=max(0, x2 - x1),
                    plate_height=max(0, y2 - y1),
                    near_exit=force,
                )
                if not schedule.run_ocr:
                    return None
                schedule_reason = schedule.reason
                recognition.reserve_ocr(
                    seq=frame.seq,
                    ts=frame.ts,
                    quality=frame.quality.score,
                    bbox_area=bbox_area,
                )
                episode.last_ocr_schedule_reason = schedule_reason
        else:
            age = episode.last_seq - episode.first_seq
            enough = len(selected) >= self.config.min_candidates_before_ocr
            excellent = selected[0].quality.score >= self.config.early_ocr_quality
            timed_out = age >= self.config.max_collection_frames
            if not (force or enough or excellent or timed_out):
                return None

        limit = min(self.config.max_ocr_candidates, self.policy.max_ocr_candidates)
        # During critical load a mediocre single crop is deferred or, when the
        # vehicle is leaving, paired with one more candidate to protect accuracy.
        if (
            episode.recognition is None
            and limit == 1
            and selected[0].quality.score < self.config.early_ocr_quality
        ):
            if not force:
                return None
            limit = min(2, len(selected))
        selected = selected[: max(1, limit)]
        episode.ocr_submitted = True
        if not finalize_only:
            episode.ocr_attempts += 1
        if recognition is not None:
            episode.ocr_attempts = recognition.attempts
        episode.advance(TrackPhase.OCR)
        state = self._states[camera_id]
        state.phase = TrackPhase.OCR
        return OCRTask(
            key=episode.episode_id,
            crops=[] if finalize_only else [frame.crop for frame in selected],
            qualities=[]
            if finalize_only
            else [frame.quality.score for frame in selected],
            sequences=[] if finalize_only else [frame.seq for frame in selected],
            timestamps=[] if finalize_only else [frame.ts for frame in selected],
            bbox_sizes=[]
            if finalize_only
            else [
                (
                    max(0, frame.bbox[2] - frame.bbox[0]),
                    max(0, frame.bbox[3] - frame.bbox[1]),
                )
                for frame in selected
            ],
            priority=0 if finalize_only else (5 if force else 10),
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
                "fusion_finalize": finalize_only,
                "fusion_finalization_reason": finalization_reason,
                "ocr_schedule_reason": schedule_reason,
            },
        )

    def _process_one_ocr(self) -> PlateEvent | None:
        processed, abandoned = self.ocr_worker.process_next_with_abandoned()
        for item in abandoned:
            self._reconcile_abandoned_ocr(item)
        if processed is None:
            return None
        task, vote = processed
        return self._apply_ocr_result(task, vote)

    def _apply_ocr_result(
        self,
        task: OCRTask,
        vote: OCRVote,
        *,
        notify_event: bool = True,
    ) -> PlateEvent | None:
        """Apply one shared-worker result if its episode is still live."""

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
                or episode.phase is TrackPhase.DONE
                or episode.event_emitted
            ):
                return None
            if episode.recognition is not None:
                event = self._apply_temporal_fusion_result(
                    camera_id,
                    track_id,
                    state,
                    episode,
                    task,
                    vote,
                )
            else:
                event = self._apply_compatibility_vote_result(
                    camera_id,
                    track_id,
                    state,
                    episode,
                    task,
                    vote,
                )

        if notify_event and self.on_event is not None and event is not None:
            self.on_event(event)
        return event

    def _apply_temporal_fusion_result(
        self,
        camera_id: str,
        track_id: int,
        state: CameraState,
        episode: TrackEpisode,
        task: OCRTask,
        vote: OCRVote,
    ) -> PlateEvent | None:
        recognition = episode.recognition
        if recognition is None:
            raise RuntimeError("temporal fusion result requires a recognition session")
        episode.observations.extend(item.result for item in vote.results)
        state.observations.extend(item.result for item in vote.results)
        episode.ocr_submitted = False
        decision = recognition.decision
        finalize_only = bool(task.metadata.get("fusion_finalize"))
        if finalize_only:
            decision = recognition.finalize(
                reason=str(
                    task.metadata.get(
                        "fusion_finalization_reason", "soft_lock_hold_elapsed"
                    )
                )
            )
        elif vote.results:
            for item in vote.results:
                width = item.bbox_size[0] if item.bbox_size is not None else None
                height = item.bbox_size[1] if item.bbox_size is not None else None
                decision = recognition.observe(
                    item.result,
                    quality=item.candidate_quality,
                    seq=item.seq,
                    ts=item.ts,
                    plate_width=width,
                    plate_height=height,
                )
        else:
            recognition.release_ocr()
        episode.ocr_attempts = recognition.attempts

        if decision.soft_locked and recognition.should_finalize(
            ts=float(task.metadata["ts"]),
            near_exit=episode.tracker_removed,
        ):
            if episode.tracker_removed:
                reason = "track_exit"
            elif recognition.audit_attempts:
                reason = "audit_complete"
            else:
                reason = "soft_lock_hold_elapsed"
            decision = recognition.finalize(reason=reason)

        if not decision.finalized:
            terminal = (
                episode.tracker_removed
                or episode.ocr_attempts >= recognition.config.max_ocr_attempts
            )
            if terminal and not decision.soft_locked:
                recognition.close()
                episode.advance(TrackPhase.DONE)
                state.last_done_seq = int(task.metadata["seq"])
                unfinished = any(
                    item.phase is not TrackPhase.DONE for item in state.tracks.values()
                )
                state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
            else:
                episode.advance(TrackPhase.COLLECTING)
                state.phase = TrackPhase.COLLECTING
            return None

        if not recognition.claim_event():
            return None
        episode.advance(TrackPhase.VALIDATED)
        duplicate = self.deduplicator.check_and_record(
            camera_id,
            decision.text,
            float(task.metadata["ts"]),
        )
        episode.advance(TrackPhase.DONE)
        episode.event_emitted = not duplicate.duplicate
        state.last_done_seq = int(task.metadata["seq"])
        unfinished = any(
            item.phase is not TrackPhase.DONE for item in state.tracks.values()
        )
        state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
        if duplicate.duplicate:
            self.metrics.duplicates_suppressed += 1
            return None

        event = PlateEvent(
            camera_id=camera_id,
            frame_seq=int(task.metadata["seq"]),
            ts=float(task.metadata["ts"]),
            text=decision.text,
            confidence=float(decision.confidence),
            bbox=tuple(task.metadata["bbox"]),
            quality=float(task.metadata["quality"]),
            track_id=str(track_id),
            episode_id=episode.episode_id,
            observations=decision.observations,
            metadata={
                "recognition_phase": RecognitionPhase.FINALIZED.value,
                "fusion_reason": decision.reason,
                "soft_lock_reason": recognition.soft_lock_reason,
                "finalization_reason": recognition.finalization_reason,
                "audit_attempts": recognition.audit_attempts,
                "independent_observations": decision.independent_observations,
                "full_sequence_support": decision.full_sequence_support,
                "slot_confidences": [slot.confidence for slot in decision.slots],
                "slot_margins": [slot.margin for slot in decision.slots],
                "ocr_schedule_reason": task.metadata.get("ocr_schedule_reason"),
                "ocr_attempts": episode.ocr_attempts,
                "load_level": self.load_controller.level.name.lower(),
            },
        )
        episode.emitted_event = event
        self.metrics.events_emitted += 1
        return event

    def _apply_compatibility_vote_result(
        self,
        camera_id: str,
        track_id: int,
        state: CameraState,
        episode: TrackEpisode,
        task: OCRTask,
        vote: OCRVote,
    ) -> PlateEvent | None:
        episode.observations.extend(item.result for item in vote.results)
        state.observations.extend(item.result for item in vote.results)
        episode.ocr_submitted = False
        if not vote.valid or vote.confidence < self.config.min_ocr_confidence:
            terminal = (
                episode.tracker_removed
                or episode.ocr_attempts >= self.config.max_ocr_attempts
            )
            if terminal:
                episode.advance(TrackPhase.DONE)
                state.last_done_seq = int(task.metadata["seq"])
                unfinished = any(
                    item.phase is not TrackPhase.DONE for item in state.tracks.values()
                )
                state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
            else:
                episode.advance(TrackPhase.COLLECTING)
                state.phase = TrackPhase.COLLECTING
            return None

        episode.advance(TrackPhase.VALIDATED)
        duplicate = self.deduplicator.check_and_record(
            camera_id,
            vote.text,
            float(task.metadata["ts"]),
        )
        episode.advance(TrackPhase.DONE)
        episode.event_emitted = not duplicate.duplicate
        state.last_done_seq = int(task.metadata["seq"])
        unfinished = any(
            item.phase is not TrackPhase.DONE for item in state.tracks.values()
        )
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
        episode.emitted_event = event
        self.metrics.events_emitted += 1
        return event

    def _reconcile_abandoned_ocr(self, abandoned: AbandonedOCRTask) -> None:
        task = abandoned.task
        if task.metadata.get("fusion_finalize"):
            # Finalization tasks contain no crop and consume no inference. They
            # remain safe to apply after queue eviction/expiry and must not
            # strand a soft-locked track that has already left the scene.
            self._apply_ocr_result(task, OCRVote())
            return
        camera_value = task.metadata.get("camera_id")
        track_value = task.metadata.get("track_id")
        if camera_value is None or track_value is None:
            return
        camera_id = str(camera_value)
        try:
            track_id = int(track_value)
        except (TypeError, ValueError):
            return

        with self._state_lock:
            state = self._states.get(camera_id)
            episode = state.tracks.get(track_id) if state is not None else None
            if (
                state is None
                or task.metadata.get("runtime_epoch") != state.runtime_epoch
                or episode is None
                or episode.episode_id != task.key
                or not episode.ocr_submitted
            ):
                return

            episode.ocr_submitted = False
            if episode.recognition is not None:
                episode.recognition.release_ocr(retryable=True)
                episode.ocr_attempts = episode.recognition.attempts
                for seq in task.sequences:
                    episode.ocr_sequences_seen.discard(seq)
            attempt_limit = (
                episode.recognition.config.max_ocr_attempts
                if episode.recognition is not None
                else self.config.max_ocr_attempts
            )
            terminal = episode.tracker_removed or episode.ocr_attempts >= attempt_limit
            if terminal:
                episode.advance(TrackPhase.DONE)
                try:
                    task_seq = int(task.metadata.get("seq", episode.last_seq))
                except (TypeError, ValueError):
                    task_seq = episode.last_seq
                state.last_done_seq = max(state.last_done_seq, task_seq)
                unfinished = any(
                    item.phase is not TrackPhase.DONE for item in state.tracks.values()
                )
                state.phase = TrackPhase.TRACKING if unfinished else TrackPhase.DONE
            else:
                # Wait for genuinely fresh evidence before retrying. Immediate
                # re-submission of the same stale/evicted crops would hot-loop.
                episode.advance(TrackPhase.COLLECTING)
                state.phase = TrackPhase.COLLECTING

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
