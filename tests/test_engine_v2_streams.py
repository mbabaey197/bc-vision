from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable

import numpy as np

from app.engine_v2.streams import (
    AdaptiveFrameAdmissionController,
    AutoDecoderFactory,
    DecodedFrame,
    DualStreamRTSPProducer,
    HardwareDecodePlan,
    LatestMainFrameCache,
    ProducerActivity,
    ProducerCadencePolicy,
    RTSPProducerConfig,
    ReconnectPolicy,
    StreamRole,
    StreamSpec,
    select_hardware_decode,
)
from app.engine_v2.types import FramePacket


class FakeDecoderSession:
    backend_name = "fake"
    hardware_accelerator = "qsv"

    def __init__(
        self,
        frames: list[DecodedFrame],
        *,
        before_first_read: threading.Event | None = None,
        after_wait_delay: float = 0.0,
        after_frame: Callable[[], None] | None = None,
    ) -> None:
        self._frames = deque(frames)
        self._before_first_read = before_first_read
        self._after_wait_delay = after_wait_delay
        self._after_frame = after_frame
        self._first_read = True
        self.closed = threading.Event()

    def read(self) -> DecodedFrame | None:
        if self._first_read and self._before_first_read is not None:
            self._before_first_read.wait(1.0)
            if self._after_wait_delay:
                time.sleep(self._after_wait_delay)
        self._first_read = False
        if self.closed.is_set():
            return None
        if self._frames:
            frame = self._frames.popleft()
            if self._after_frame is not None:
                self._after_frame()
            return frame
        self.closed.wait(2.0)
        return None

    def close(self) -> None:
        self.closed.set()


class FakeDecoderFactory:
    def __init__(
        self,
        main_frame: DecodedFrame,
        sub_frame: DecodedFrame,
        *,
        sub_open_failures: int = 0,
        leak_url_in_error: bool = False,
    ) -> None:
        self.main_frame = main_frame
        self.sub_frame = sub_frame
        self.sub_open_failures = sub_open_failures
        self.leak_url_in_error = leak_url_in_error
        self.main_ready = threading.Event()
        self.attempts: defaultdict[StreamRole, int] = defaultdict(int)
        self.sessions: list[FakeDecoderSession] = []
        self._lock = threading.Lock()

    def open(self, spec: StreamSpec) -> FakeDecoderSession:
        with self._lock:
            self.attempts[spec.role] += 1
            attempt = self.attempts[spec.role]
        if spec.role is StreamRole.SUB and attempt <= self.sub_open_failures:
            suffix = f": {spec.url}" if self.leak_url_in_error else ""
            raise ConnectionError(f"temporary fake connection failure{suffix}")
        if spec.role is StreamRole.MAIN:
            session = FakeDecoderSession([self.main_frame], after_frame=self.main_ready.set)
        else:
            # The event fires when the fake decoder returns its main frame. Give
            # the producer thread a scheduling turn to commit it to the cache.
            session = FakeDecoderSession(
                [self.sub_frame],
                before_first_read=self.main_ready,
                after_wait_delay=0.005,
            )
        with self._lock:
            self.sessions.append(session)
        return session


def _image(height: int, width: int, value: int) -> np.ndarray:
    return np.full((height, width, 3), value, dtype=np.uint8)


def _config(**overrides: object) -> RTSPProducerConfig:
    values: dict[str, object] = {
        "camera_id": "cam-1",
        "main_url": "rtsp://admin:secret@example.test/main",
        "sub_url": "rtsp://admin:secret@example.test/sub",
        "reconnect": ReconnectPolicy(
            initial_delay_seconds=0.01,
            maximum_delay_seconds=0.04,
            multiplier=2,
            jitter_ratio=0,
        ),
    }
    values.update(overrides)
    return RTSPProducerConfig(**values)  # type: ignore[arg-type]


def test_hardware_selection_prefers_intel_qsv_and_falls_back_safely() -> None:
    plan = select_hardware_decode(
        "auto",
        system="Linux",
        available_accelerators=("vaapi", "qsv"),
        device_exists=lambda path: path == "/dev/dri/renderD128",
    )
    assert plan.enabled is True
    assert plan.accelerator == "qsv"
    assert plan.device == "/dev/dri/renderD128"

    unavailable = select_hardware_decode(
        "auto",
        system="Linux",
        available_accelerators=("qsv", "vaapi"),
        device_exists=lambda _path: False,
    )
    assert unavailable.enabled is False
    assert unavailable.accelerator is None

    disabled = select_hardware_decode("off", available_accelerators=("qsv",))
    assert disabled.enabled is False


def test_auto_decoder_falls_back_to_software_when_hardware_cannot_emit_frame() -> None:
    attempts: list[str | None] = []

    class CandidateSession:
        backend_name = "fake"

        def __init__(self, plan: HardwareDecodePlan) -> None:
            self.hardware_accelerator = plan.accelerator
            self.closed = False
            self._frames = deque(
                [] if plan.enabled else [DecodedFrame(_image(12, 16, 7))]
            )

        def read(self) -> DecodedFrame | None:
            return self._frames.popleft() if self._frames else None

        def close(self) -> None:
            self.closed = True

    class ProbeFactory(AutoDecoderFactory):
        def _backend_available(self, backend: str) -> bool:
            return backend == "opencv"

        def _open_backend(
            self,
            backend: str,
            spec: StreamSpec,
            plan: HardwareDecodePlan,
        ) -> CandidateSession:
            del backend, spec
            attempts.append(plan.accelerator)
            return CandidateSession(plan)

    factory = ProbeFactory(
        backend_preference=("opencv",),
        hardware_plan=HardwareDecodePlan("qsv", reason="test"),
    )
    session = factory.open(StreamSpec("cam", StreamRole.SUB, "rtsp://example.test/sub"))
    assert attempts == ["qsv", None]
    assert session.hardware_accelerator is None
    assert session.read() is not None
    session.close()


def test_reconnect_policy_is_bounded_exponential_backoff() -> None:
    policy = ReconnectPolicy(0.25, 2.0, 2.0, 0.0)
    assert [policy.delay_for(attempt) for attempt in range(1, 6)] == [0.25, 0.5, 1.0, 2.0, 2.0]


def test_latest_main_cache_replaces_frame_and_maps_high_resolution_crop() -> None:
    source = _image(240, 320, 10)
    source[60:180, 80:240] = 220
    cache = LatestMainFrameCache("cam-1", copy_frames=True)
    first = cache.put(
        1,
        DecodedFrame(source, captured_at=100.0, monotonic_at=10.0),
        backend_name="fake",
        hardware_accelerator="qsv",
    )

    source.fill(0)
    crop = first.crop_from_detector_bbox((20, 15, 60, 45), (60, 80, 3))
    assert crop is not None
    assert crop.shape == (120, 160, 3)
    assert np.all(crop == 220)

    cache.put(2, DecodedFrame(_image(240, 320, 33), captured_at=101.0, monotonic_at=11.0))
    assert cache.latest().seq == 2  # type: ignore[union-attr]
    assert cache.latest(reference_monotonic=12.1, maximum_age_seconds=1.0) is None


def test_dual_stream_producer_emits_substream_packet_with_latest_main_frame() -> None:
    main = DecodedFrame(
        _image(240, 320, 70),
        captured_at=100.0,
        monotonic_at=10.0,
        source_pts=9.9,
    )
    sub = DecodedFrame(
        _image(60, 80, 20),
        captured_at=100.1,
        monotonic_at=10.1,
        source_pts=10.0,
    )
    factory = FakeDecoderFactory(main, sub)
    received: list[FramePacket] = []
    packet_ready = threading.Event()

    def accept(packet: FramePacket) -> bool:
        received.append(packet)
        packet_ready.set()
        return True

    producer = DualStreamRTSPProducer(_config(), accept, decoder_factory=factory)
    assert producer.start() is True
    assert producer.start() is False
    assert packet_ready.wait(1.0)
    assert producer.stop(1.0) is True

    assert len(received) == 1
    packet = received[0]
    assert packet.camera_id == "cam-1"
    assert packet.seq == 1
    assert packet.ts == 100.1
    assert packet.frame.shape == (240, 320, 3)
    assert packet.detector_frame is not None
    assert packet.detector_frame.shape == (60, 80, 3)
    assert packet.metadata["producer_epoch"] == producer.producer_epoch
    assert packet.metadata["main_seq"] == 1
    assert abs(packet.metadata["main_age_seconds"] - 0.1) < 1e-9
    assert abs(packet.metadata["main_detector_skew_ms"] - 100.0) < 1e-9
    assert packet.metadata["main_hardware_accelerator"] == "qsv"
    assert packet.metadata["sub_hardware_accelerator"] == "qsv"
    assert producer.stats.packets_emitted == 1
    assert producer.stats.main.decoded_frames == 1
    assert producer.stats.sub.decoded_frames == 1
    assert all(session.closed.is_set() for session in factory.sessions)
    assert producer.running is False


def test_stale_main_frame_is_not_sent_to_detector_scheduler() -> None:
    main = DecodedFrame(_image(240, 320, 1), captured_at=10.0, monotonic_at=1.0)
    sub = DecodedFrame(_image(60, 80, 2), captured_at=12.0, monotonic_at=3.0)
    factory = FakeDecoderFactory(main, sub)
    received: list[FramePacket] = []
    producer = DualStreamRTSPProducer(
        _config(maximum_main_frame_age_seconds=0.5),
        received.append,
        decoder_factory=factory,
    )
    producer.start()

    deadline = time.monotonic() + 1.0
    while producer.stats.packets_dropped_stale_main == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    assert producer.stop(1.0) is True
    assert received == []
    assert producer.stats.packets_dropped_stale_main == 1


def test_future_main_frame_outside_sync_window_is_not_emitted() -> None:
    received: list[FramePacket] = []
    producer = DualStreamRTSPProducer(
        _config(maximum_main_frame_age_seconds=0.5),
        received.append,
        decoder_factory=object(),  # This test drives the pairing method directly.
    )
    producer.main_frame_cache.put(
        1,
        DecodedFrame(_image(240, 320, 1), captured_at=12.0, monotonic_at=3.0),
    )

    class Session:
        backend_name = "fake"
        hardware_accelerator = None

    producer._emit_detector_packet(
        1,
        DecodedFrame(_image(60, 80, 2), captured_at=10.0, monotonic_at=1.0),
        Session(),
    )
    assert received == []
    assert producer.stats.packets_dropped_stale_main == 1


def test_producer_reconnects_with_backoff_and_redacts_stream_credentials() -> None:
    main = DecodedFrame(_image(240, 320, 1), captured_at=10.0, monotonic_at=1.0)
    sub = DecodedFrame(_image(60, 80, 2), captured_at=10.1, monotonic_at=1.1)
    factory = FakeDecoderFactory(main, sub, sub_open_failures=2, leak_url_in_error=True)
    delays: list[float] = []
    errors = []
    packet_ready = threading.Event()

    def wait_for_stop(stop_event: threading.Event, delay: float) -> bool:
        delays.append(delay)
        return stop_event.wait(0.001)

    producer = DualStreamRTSPProducer(
        _config(),
        lambda _packet: packet_ready.set(),
        decoder_factory=factory,
        on_error=errors.append,
        wait_for_stop=wait_for_stop,
    )
    producer.start()
    assert packet_ready.wait(1.0)
    assert producer.stop(1.0) is True

    assert factory.attempts[StreamRole.SUB] == 3
    assert producer.stats.sub.reconnects == 2
    assert delays[:2] == [0.01, 0.02]
    assert len(errors) == 2
    assert all("secret" not in error.message for error in errors)
    assert all("<redacted-stream-url>" in error.message for error in errors)


def test_main_stream_can_fall_back_to_substream_only_when_explicitly_allowed() -> None:
    main = DecodedFrame(_image(240, 320, 1), captured_at=10.0, monotonic_at=1.0)
    sub = DecodedFrame(_image(60, 80, 9), captured_at=12.0, monotonic_at=3.0)
    factory = FakeDecoderFactory(main, sub)
    received: list[FramePacket] = []
    packet_ready = threading.Event()

    def accept(packet: FramePacket) -> None:
        received.append(packet)
        packet_ready.set()

    producer = DualStreamRTSPProducer(
        _config(require_main_frame=False, maximum_main_frame_age_seconds=0.5),
        accept,
        decoder_factory=factory,
    )
    producer.start()
    assert packet_ready.wait(1.0)
    assert producer.stop(1.0) is True

    assert received[0].frame.shape == (60, 80, 3)
    assert received[0].metadata["main_fallback_to_sub"] is True
    assert producer.stats.packets_dropped_stale_main == 1


def test_producer_sequence_survives_clean_stop_and_restart() -> None:
    main = DecodedFrame(_image(240, 320, 1), captured_at=10.0, monotonic_at=1.0)
    sub = DecodedFrame(_image(60, 80, 2), captured_at=10.1, monotonic_at=1.1)
    factory = FakeDecoderFactory(main, sub)
    received: list[FramePacket] = []
    packet_ready = threading.Event()

    def accept(packet: FramePacket) -> bool:
        received.append(packet)
        packet_ready.set()
        return True

    producer = DualStreamRTSPProducer(_config(), accept, decoder_factory=factory)
    assert producer.start() is True
    assert packet_ready.wait(1.0)
    assert producer.stop(1.0) is True

    packet_ready.clear()
    factory.main_ready.clear()
    assert producer.start() is True
    assert packet_ready.wait(1.0)
    assert producer.stop(1.0) is True
    assert [packet.seq for packet in received] == [1, 2]
    assert received[0].metadata["producer_epoch"] != received[1].metadata["producer_epoch"]


def test_new_producer_instance_emits_a_distinct_epoch() -> None:
    received: list[FramePacket] = []

    class Session:
        backend_name = "fake"
        hardware_accelerator = None

    def emit_one(producer: DualStreamRTSPProducer) -> None:
        producer.main_frame_cache.put(
            1,
            DecodedFrame(_image(240, 320, 1), captured_at=10.0, monotonic_at=1.0),
        )
        producer._emit_detector_packet(
            1,
            DecodedFrame(_image(60, 80, 2), captured_at=10.1, monotonic_at=1.1),
            Session(),
        )

    first = DualStreamRTSPProducer(_config(), received.append, decoder_factory=object())
    second = DualStreamRTSPProducer(_config(), received.append, decoder_factory=object())
    emit_one(first)
    emit_one(second)

    assert len(received) == 2
    assert received[0].metadata["producer_epoch"] == first.producer_epoch
    assert received[1].metadata["producer_epoch"] == second.producer_epoch
    assert received[0].metadata["producer_epoch"] != received[1].metadata["producer_epoch"]


def test_adaptive_admission_enforces_detector_and_tracking_cadence() -> None:
    policy = ProducerCadencePolicy(
        target_detector_fps=2.0,
        activity=ProducerActivity.ACTIVE,
        tracking_frames_between_detection=2,
    )
    controller = AdaptiveFrameAdmissionController(policy)
    decisions = [controller.decide(value) for value in (0.0, 0.05, 1 / 6, 2 / 6, 0.5)]

    assert [(item.admit, item.detector_due) for item in decisions] == [
        (True, True),
        (False, False),
        (True, False),
        (True, False),
        (True, True),
    ]
    assert policy.effective_detector_fps == 2.0
    assert policy.target_admission_fps == 6.0


def test_idle_floor_and_activity_transition_protect_motion_accuracy() -> None:
    idle = ProducerCadencePolicy(
        target_detector_fps=0.05,
        activity=ProducerActivity.IDLE,
        minimum_idle_admission_fps=2.0,
    )
    controller = AdaptiveFrameAdmissionController(idle)
    assert controller.decide(0.0).admit is True
    assert controller.decide(0.25).admit is False
    assert controller.decide(0.5).admit is True

    controller.update(
        ProducerCadencePolicy(
            target_detector_fps=1.0,
            activity=ProducerActivity.ACTIVE,
            tracking_frames_between_detection=1,
        )
    )
    transition = controller.decide(0.51)
    assert transition.admit is True
    assert transition.detector_due is True

    overloaded_active = ProducerCadencePolicy(
        target_detector_fps=0.05,
        activity=ProducerActivity.ACTIVE,
        tracking_frames_between_detection=4,
    )
    assert overloaded_active.effective_detector_fps == 0.5
    assert overloaded_active.target_admission_fps == 2.5


def test_rejected_detector_job_is_retried_at_next_cadence_slot() -> None:
    controller = AdaptiveFrameAdmissionController(
        ProducerCadencePolicy(2.0, ProducerActivity.ACTIVE, 1)
    )
    first = controller.decide(0.0)
    assert first.admit is True and first.detector_due is True
    controller.detector_unaccepted()

    retry = controller.decide(0.25)
    assert retry.admit is True
    assert retry.detector_due is True


def test_admission_controller_is_thread_safe_for_same_newest_frame() -> None:
    controller = AdaptiveFrameAdmissionController(
        ProducerCadencePolicy(2.0, ProducerActivity.ACTIVE, 1)
    )
    barrier = threading.Barrier(8)
    results = []
    result_lock = threading.Lock()

    def decide() -> None:
        barrier.wait()
        result = controller.decide(10.0)
        with result_lock:
            results.append(result)

    threads = [threading.Thread(target=decide) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1.0)

    assert len(results) == 8
    assert sum(item.admit for item in results) == 1


def test_producer_admits_only_newest_due_frames_without_sleeping_decode_loop() -> None:
    current_policy = [
        ProducerCadencePolicy(
            target_detector_fps=2.0,
            activity=ProducerActivity.ACTIVE,
            tracking_frames_between_detection=1,
        )
    ]
    received: list[FramePacket] = []

    class Session:
        backend_name = "fake"
        hardware_accelerator = None

    producer = DualStreamRTSPProducer(
        _config(),
        received.append,
        decoder_factory=object(),
        cadence_provider=lambda _camera_id: current_policy[0],
    )
    for seq, timestamp in enumerate((0.0, 0.1, 0.25, 0.5), start=1):
        producer.main_frame_cache.put(
            seq,
            DecodedFrame(_image(240, 320, seq), captured_at=timestamp, monotonic_at=timestamp),
        )
        producer._emit_detector_packet(
            seq,
            DecodedFrame(_image(60, 80, seq), captured_at=timestamp, monotonic_at=timestamp),
            Session(),
        )

    # Switching activity forces the current newest frame through immediately;
    # it does not wait or flush buffered old frames.
    current_policy[0] = ProducerCadencePolicy(
        target_detector_fps=0.05,
        activity=ProducerActivity.IDLE,
        minimum_idle_admission_fps=2.0,
    )
    producer.main_frame_cache.put(
        5,
        DecodedFrame(_image(240, 320, 5), captured_at=0.51, monotonic_at=0.51),
    )
    producer._emit_detector_packet(
        5,
        DecodedFrame(_image(60, 80, 5), captured_at=0.51, monotonic_at=0.51),
        Session(),
    )

    assert [packet.seq for packet in received] == [1, 3, 4, 5]
    assert [packet.metadata["detector_due"] for packet in received] == [True, False, True, True]
    assert all(packet.metadata["adaptive_admission"] is True for packet in received)
    assert received[-1].metadata["producer_activity"] == "idle"
    assert received[-1].metadata["producer_effective_detector_fps"] == 2.0
    assert producer.stats.packets_dropped_by_admission == 1
    assert producer.stats.detector_due_packets == 3
    assert producer.stats.tracking_only_packets == 1


def test_bad_or_unbound_cadence_provider_fails_open_for_accuracy() -> None:
    received: list[FramePacket] = []
    errors = []

    class Session:
        backend_name = "fake"
        hardware_accelerator = None

    def broken_provider(_camera_id: str) -> ProducerCadencePolicy:
        raise RuntimeError("synthetic cadence failure")

    producer = DualStreamRTSPProducer(
        _config(),
        received.append,
        decoder_factory=object(),
        cadence_provider=broken_provider,
        on_error=errors.append,
    )
    producer.main_frame_cache.put(
        1,
        DecodedFrame(_image(240, 320, 1), captured_at=1.0, monotonic_at=1.0),
    )
    producer._emit_detector_packet(
        1,
        DecodedFrame(_image(60, 80, 1), captured_at=1.0, monotonic_at=1.0),
        Session(),
    )
    assert len(received) == 1
    assert "adaptive_admission" not in received[0].metadata
    assert producer.stats.cadence_provider_errors == 1
    assert errors[0].stage == "cadence"

    producer.update_adaptive_cadence(
        ProducerCadencePolicy(1.0, ProducerActivity.ACTIVE)
    )
    assert producer.cadence_policy is not None
    producer.set_cadence_provider(None)
    assert producer.cadence_policy is None
