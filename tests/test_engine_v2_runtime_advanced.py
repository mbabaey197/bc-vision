from __future__ import annotations

import numpy as np

from app.engine_v2 import EngineV2Config, EventDrivenANPREngine, FramePacket, OCRResult, PlateCandidate
from app.engine_v2.streams import ProducerActivity
from app.engine_v2.types import TrackPhase


class CentralDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        h, w = frame.shape[:2]
        return [PlateCandidate((w // 4, h // 3, 3 * w // 4, 2 * h // 3), 0.94)]


class SequenceOCR:
    def __init__(self) -> None:
        self.calls = 0
        self.crop_shapes: list[tuple[int, ...]] = []

    def read(self, crop: np.ndarray) -> OCRResult:
        self.crop_shapes.append(crop.shape)
        values = ("12ب34567", "۱۲ب۳۴۵۶۷", "12ب34568")
        value = values[min(self.calls, len(values) - 1)]
        self.calls += 1
        return OCRResult(value, 0.92, True)


class OneShotDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        if self.calls > 1:
            return []
        h, w = frame.shape[:2]
        return [PlateCandidate((w // 4, h // 3, 3 * w // 4, 2 * h // 3), 0.94)]


class InvalidOCR:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop: np.ndarray) -> OCRResult:
        del crop
        self.calls += 1
        return OCRResult("", 0.10, False)


class NewVehicleDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        del frame
        self.calls += 1
        first = PlateCandidate((40, 80, 200, 160), 0.95)
        if self.calls == 1:
            return [first]
        return [first, PlateCandidate((280, 80, 440, 160), 0.96)]


class DistinctPlateOCR:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop: np.ndarray) -> OCRResult:
        del crop
        values = ("12ب34567", "34م56789")
        value = values[min(self.calls, len(values) - 1)]
        self.calls += 1
        return OCRResult(value, 0.96, True)


def _frame(value: int, shape: tuple[int, int] = (240, 480)) -> np.ndarray:
    h, w = shape
    image = np.full((h, w, 3), value, dtype=np.uint8)
    y1, y2 = h // 3, 2 * h // 3
    x1, x2 = w // 4, 3 * w // 4
    image[y1:y2, x1:x2:2] = min(255, value + 120)
    image[y1:y2:2, x1:x2] = max(0, value - 30)
    return image


def test_episode_collects_candidates_votes_and_reaches_done() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=2,
            early_ocr_quality=1.1,
            load_control_enabled=False,
        ),
    )
    assert engine.submit_frame(FramePacket("cam", 1, 1.0, _frame(20))) is False
    assert engine.submit_frame(FramePacket("cam", 2, 2.0, _frame(100))) is True
    assert engine.process_next() is None
    assert engine.submit_frame(FramePacket("cam", 3, 3.0, _frame(105))) is True
    event = engine.process_next()

    assert event is not None
    assert event.text == "12ب34567"
    assert event.observations == 2
    assert detector.calls == 2
    assert ocr.calls == 2
    episode = next(iter(engine.state_for("cam").tracks.values()))
    ordered = [
        TrackPhase.TRACKING,
        TrackPhase.PLATE_FOUND,
        TrackPhase.COLLECTING,
        TrackPhase.OCR,
        TrackPhase.VALIDATED,
        TrackPhase.DONE,
    ]
    cursor = -1
    for phase in ordered:
        cursor = episode.transitions.index(phase, cursor + 1)

    # DONE prevents another OCR for the same passage.
    engine.submit_frame(FramePacket("cam", 4, 4.0, _frame(110)))
    assert engine.process_next() is None
    assert ocr.calls == 2


def test_substream_detection_maps_to_high_resolution_main_crop() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            load_control_enabled=False,
        ),
    )
    main = _frame(30, (720, 1280))
    sub_idle = _frame(10, (180, 320))
    sub_motion = _frame(100, (180, 320))
    engine.submit_frame(FramePacket("dual", 1, 1.0, main, sub_idle))
    assert engine.submit_frame(FramePacket("dual", 2, 2.0, main, sub_motion))
    event = engine.process_next()
    assert event is not None
    assert event.bbox == (320, 240, 960, 480)
    assert ocr.crop_shapes[0][0] > 180 // 3
    assert ocr.crop_shapes[0][1] > 320 // 2


def test_sixteen_idle_cameras_add_zero_detector_or_ocr_work() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(idle_stride=1, load_control_enabled=False),
    )
    idle = np.zeros((90, 160, 3), dtype=np.uint8)
    for camera_index in range(16):
        camera = f"idle-{camera_index}"
        engine.submit_frame(FramePacket(camera, 1, 1.0, idle))
        engine.submit_frame(FramePacket(camera, 2, 2.0, idle))

    assert engine.process_available() == []
    telemetry = engine.telemetry()
    assert detector.calls == 0
    assert ocr.calls == 0
    assert telemetry["active_cameras"] == 0
    assert telemetry["idle_cameras"] == 16


def test_removed_track_with_invalid_forced_ocr_reaches_terminal_done() -> None:
    detector = OneShotDetector()
    ocr = InvalidOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=2,
            early_ocr_quality=1.1,
            tracker_max_missed=0,
            load_control_enabled=False,
        ),
    )
    engine.submit_frame(FramePacket("cam-invalid", 1, 1.0, _frame(10)))
    assert engine.submit_frame(FramePacket("cam-invalid", 2, 2.0, _frame(100)))
    assert engine.process_next() is None

    assert engine.submit_frame(FramePacket("cam-invalid", 3, 3.0, _frame(105)))
    assert engine.process_next() is None
    state = engine.state_for("cam-invalid")
    episode = next(iter(state.tracks.values()))
    assert ocr.calls == 1
    assert episode.tracker_removed is True
    assert episode.phase is TrackPhase.DONE
    assert state.phase is TrackPhase.DONE


def test_producer_epoch_allows_safe_sequence_restart_without_blind_window() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=3,
            early_ocr_quality=1.1,
            load_control_enabled=False,
        ),
    )
    metadata_a = {"producer_epoch": "reader-a:1"}
    engine.submit_frame(FramePacket("cam-restart", 1, 1.0, _frame(10), metadata=metadata_a))
    assert engine.submit_frame(
        FramePacket("cam-restart", 2, 2.0, _frame(100), metadata=metadata_a)
    )
    assert engine.process_next() is None
    assert detector.calls == 1

    metadata_b = {"producer_epoch": "reader-b:1"}
    assert engine.submit_frame(
        FramePacket("cam-restart", 1, 3.0, _frame(105), metadata=metadata_b)
    )
    assert engine.process_next() is None
    assert detector.calls == 2
    assert engine.metrics.out_of_order_frames == 0
    assert engine.state_for("cam-restart").source_epoch == "reader-b:1"


def test_done_episode_allows_motion_probe_for_a_new_track_without_reocr() -> None:
    detector = NewVehicleDetector()
    ocr = DistinctPlateOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            # Keep the production default: a new fast track must not wait for
            # the completed track's 25-frame hold interval.
            done_cooldown_frames=25,
            load_control_enabled=False,
        ),
    )
    engine.submit_frame(FramePacket("two-cars", 1, 1.0, _frame(10)))
    assert engine.submit_frame(FramePacket("two-cars", 2, 2.0, _frame(100)))
    first_event = engine.process_next()
    assert first_event is not None
    assert first_event.text == "12ب34567"
    assert engine.state_for("two-cars").phase is TrackPhase.DONE

    # A second vehicle changes the scene while the first track remains DONE.
    # The camera must wake the detector at idle cadence, associate the old
    # detection with its completed track, and create a fresh second track.
    assert engine.submit_frame(FramePacket("two-cars", 3, 3.0, _frame(210)))
    second_event = engine.process_next()

    assert second_event is not None
    assert second_event.text == "34م56789"
    assert second_event.track_id != first_event.track_id
    assert detector.calls == 2
    assert ocr.calls == 2
    episodes = sorted(engine.state_for("two-cars").tracks.values(), key=lambda item: item.track_id)
    assert len(episodes) == 2
    assert [episode.ocr_attempts for episode in episodes] == [1, 1]
    assert all(episode.phase is TrackPhase.DONE for episode in episodes)


def test_restart_hook_rotates_runtime_epoch_and_rejects_pending_old_ocr() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=2,
            early_ocr_quality=1.1,
            load_control_enabled=False,
        ),
    )
    engine.submit_frame(FramePacket("hook-restart", 1, 1.0, _frame(10)))
    for seq, value in ((2, 100), (3, 105)):
        assert engine.submit_frame(FramePacket("hook-restart", seq, float(seq), _frame(value)))
        packet = engine.queue.pop()
        assert packet is not None
        engine._process_detector_packet(packet)

    old_state = engine.state_for("hook-restart")
    old_epoch = old_state.runtime_epoch
    old_episode_number = old_state.episode_number
    old_episode_id = next(iter(old_state.tracks.values())).episode_id
    assert len(engine.ocr_worker.queue) == 1

    engine.notify_stream_restart("hook-restart", preserve_activity=True)
    restarted = engine.state_for("hook-restart")
    assert restarted.runtime_epoch == old_epoch + 1
    assert restarted.episode_number == old_episode_number + 1
    assert restarted.last_received_seq == -1
    assert len(engine.ocr_worker.queue) == 0
    assert engine.metrics.restart_stale_ocr_tasks == 1
    assert ocr.calls == 0

    # The new producer can safely restart its sequence at one. Its first track
    # must have a different identity even though the tracker also restarts at 1.
    assert engine.submit_frame(FramePacket("hook-restart", 1, 10.0, _frame(150)))
    packet = engine.queue.pop()
    assert packet is not None
    engine._process_detector_packet(packet)
    new_episode = next(iter(engine.state_for("hook-restart").tracks.values()))
    assert new_episode.episode_id != old_episode_id

    # The pending task performs no state transition/event in the new epoch.
    assert engine._process_one_ocr() is None
    assert new_episode.phase is TrackPhase.COLLECTING
    assert new_episode.observations == []

    # A second new-source candidate produces the real new-epoch event.
    assert engine.submit_frame(FramePacket("hook-restart", 2, 11.0, _frame(155)))
    packet = engine.queue.pop()
    assert packet is not None
    engine._process_detector_packet(packet)
    event = engine._process_one_ocr()
    assert event is not None
    assert event.episode_id == new_episode.episode_id
    assert event.frame_seq == 2


def test_restart_hook_rejects_detector_frame_queued_by_previous_source() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            load_control_enabled=False,
        ),
    )
    engine.submit_frame(FramePacket("queued-restart", 1, 1.0, _frame(10)))
    assert engine.submit_frame(FramePacket("queued-restart", 2, 2.0, _frame(100)))
    assert detector.calls == 0

    engine.notify_stream_restart("queued-restart", preserve_activity=True)
    assert engine.process_next() is None
    assert detector.calls == 0
    assert engine.metrics.restart_stale_frames == 1

    # Sequence one from the replacement source is accepted immediately and is
    # the only frame allowed to reach the shared detector.
    assert engine.submit_frame(FramePacket("queued-restart", 1, 10.0, _frame(120)))
    event = engine.process_next()
    assert event is not None
    assert event.frame_seq == 1
    assert detector.calls == 1


def test_adaptive_packets_bypass_modulo_and_route_tracking_only_frames() -> None:
    detector = CentralDetector()
    ocr = SequenceOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=100,
            active_stride=100,
            min_quality=0.0,
            min_candidates_before_ocr=5,
            early_ocr_quality=1.1,
            max_collection_frames=100,
            load_control_enabled=False,
        ),
    )

    def packet(seq: int, value: int, *, detector_due: bool) -> FramePacket:
        return FramePacket(
            "adaptive",
            seq,
            float(seq),
            _frame(value),
            metadata={"adaptive_admission": True, "detector_due": detector_due},
        )

    # Producer admission replaces runtime modulo throttling. The baseline is
    # consumed by the motion gate even though it does not schedule inference.
    assert engine.submit_frame(packet(1, 10, detector_due=True)) is True
    assert engine.submit_frame(packet(2, 100, detector_due=True)) is True
    assert engine.process_next() is None
    assert detector.calls == 1

    # Two admitted tracking frames harvest predicted high-resolution crops and
    # deliberately avoid the detector queue.
    assert engine.submit_frame(packet(3, 105, detector_due=False)) is True
    assert engine.submit_frame(packet(4, 110, detector_due=False)) is True
    assert detector.calls == 1
    assert engine.metrics.predicted_track_frames == 2

    assert engine.submit_frame(packet(5, 115, detector_due=True)) is True
    assert engine.process_next() is None
    assert detector.calls == 2

    cadence = engine.producer_cadence_policy("adaptive", source_fps=25.0)
    assert cadence.activity is ProducerActivity.ACTIVE
    assert cadence.tracking_frames_between_detection == engine.policy.tracking_frames_between_detection
    assert cadence.target_detector_fps == engine.target_detector_fps("adaptive", 25.0)
