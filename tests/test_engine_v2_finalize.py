from __future__ import annotations

import threading

import numpy as np

from app.engine_v2 import EngineV2Config, EventDrivenANPREngine, FramePacket, OCRResult, PlateCandidate
from app.engine_v2.types import TrackPhase


PLATE = "12ب34567"


class _Detector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        height, width = frame.shape[:2]
        return [PlateCandidate((width // 4, height // 3, 3 * width // 4, 2 * height // 3), 0.96)]


class _OCR:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, crop: np.ndarray) -> OCRResult:
        assert crop.size > 0
        self.calls += 1
        return OCRResult(PLATE, 0.97, True)


class _BlockingOCR(_OCR):
    def __init__(self) -> None:
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()

    def read(self, crop: np.ndarray) -> OCRResult:
        self.entered.set()
        assert self.release.wait(1.0)
        return super().read(crop)


class _TwoPlateDetector(_Detector):
    def detect(self, frame: np.ndarray):
        self.calls += 1
        height, width = frame.shape[:2]
        return [
            PlateCandidate((width // 12, height // 3, 5 * width // 12, 2 * height // 3), 0.96),
            PlateCandidate((7 * width // 12, height // 3, 11 * width // 12, 2 * height // 3), 0.97),
        ]


class _TwoPlateOCR(_OCR):
    def read(self, crop: np.ndarray) -> OCRResult:
        assert crop.size > 0
        values = ("12ب34567", "34م56789")
        value = values[min(self.calls, len(values) - 1)]
        self.calls += 1
        return OCRResult(value, 0.97, True)


def _frame(value: int) -> np.ndarray:
    image = np.full((180, 360, 3), value, dtype=np.uint8)
    image[60:120, 90:270:2] = min(255, value + 120)
    image[60:120:2, 90:270] = max(0, value - 30)
    return image


def _engine(ocr: _OCR, **overrides: object) -> EventDrivenANPREngine:
    values: dict[str, object] = {
        "idle_stride": 1,
        "active_stride": 1,
        "min_quality": 0.0,
        "min_candidates_before_ocr": 1,
        "early_ocr_quality": 1.1,
        "same_camera_duplicate_seconds": 20.0,
        "cross_camera_duplicate_seconds": 0.0,
        "load_control_enabled": False,
    }
    values.update(overrides)
    return EventDrivenANPREngine(_Detector(), ocr, EngineV2Config(**values))


def _harvest_one(
    engine: EventDrivenANPREngine,
    camera_id: str,
    *,
    first_seq: int = 1,
    first_ts: float = 1.0,
) -> None:
    assert engine.submit_frame(
        FramePacket(camera_id, first_seq, first_ts, _frame(10))
    ) is False
    assert engine.submit_frame(
        FramePacket(camera_id, first_seq + 1, first_ts + 1.0, _frame(100))
    ) is True
    packet = engine.queue.pop()
    assert packet is not None
    engine._process_detector_packet(packet)


def test_finalize_camera_drains_pending_real_candidate_once() -> None:
    ocr = _OCR()
    callbacks = []
    engine = _engine(ocr)
    engine.on_event = callbacks.append
    _harvest_one(engine, "finite-video")

    assert len(engine.ocr_worker.queue) == 1
    detector_calls = engine.detector.calls
    events = engine.finalize_camera("finite-video", final_seq=99, final_ts=123.5)

    assert len(events) == 1
    assert callbacks == events
    assert engine.detector.calls == detector_calls
    assert ocr.calls == 1
    # EOF metadata is a lifecycle boundary, not a fabricated evidence frame.
    assert events[0].frame_seq == 2
    assert events[0].ts == 2.0
    state = engine.state_for("finite-video")
    assert state.input_finalized is True
    assert state.final_seq == 99
    assert state.final_ts == 123.5
    assert state.phase is TrackPhase.DONE
    episode = next(iter(state.tracks.values()))
    assert episode.phase is TrackPhase.DONE
    assert episode.event_emitted is True
    assert len(episode.selector) == 0
    assert len(engine.ocr_worker.queue) == 0

    assert engine.finalize_camera("finite-video", final_seq=100, final_ts=124.0) == []
    assert engine.submit_frame(FramePacket("finite-video", 100, 124.0, _frame(120))) is False
    assert ocr.calls == 1
    assert callbacks == events


def test_finalize_camera_drops_weak_candidate_without_ocr_or_frame_fabrication() -> None:
    ocr = _OCR()
    engine = _engine(ocr, min_quality=1.0)
    _harvest_one(engine, "weak")

    assert engine.finalize_camera("weak", final_seq=20, final_ts=20.0) == []
    state = engine.state_for("weak")
    episode = next(iter(state.tracks.values()))
    assert ocr.calls == 0
    assert episode.phase is TrackPhase.DONE
    assert episode.ocr_submitted is False
    assert len(episode.selector) == 0
    assert state.phase is TrackPhase.DONE
    assert state.best_crop is None
    assert state.best_bbox is None
    assert engine.telemetry()["events"] == 0


def test_finalize_camera_does_not_consume_other_camera_ocr_work() -> None:
    ocr = _OCR()
    engine = _engine(ocr)
    _harvest_one(engine, "camera-a")
    _harvest_one(engine, "camera-b", first_seq=10, first_ts=10.0)
    assert engine.submit_frame(FramePacket("camera-c", 1, 1.0, _frame(10))) is False
    assert engine.submit_frame(FramePacket("camera-c", 2, 2.0, _frame(100))) is True
    camera_b = engine.state_for("camera-b")
    camera_b_episode = next(iter(camera_b.tracks.values()))

    assert len(engine.ocr_worker.queue) == 2
    assert len(engine.queue) == 1
    events = engine.finalize_camera("camera-a")

    assert [event.camera_id for event in events] == ["camera-a"]
    assert len(engine.ocr_worker.queue) == 1
    assert len(engine.queue) == 1
    assert camera_b.input_finalized is False
    assert camera_b.phase is TrackPhase.OCR
    assert camera_b_episode.ocr_submitted is True
    assert camera_b_episode.observations == []

    camera_b_event = engine.process_next()
    assert camera_b_event is not None
    assert camera_b_event.camera_id == "camera-b"
    assert ocr.calls == 2


def test_finalize_camera_is_thread_safe_and_rejects_racing_target_frames() -> None:
    ocr = _BlockingOCR()
    engine = _engine(ocr)
    _harvest_one(engine, "racing")
    results: list[list[object]] = []

    first = threading.Thread(
        target=lambda: results.append(engine.finalize_camera("racing")),
    )
    first.start()
    assert ocr.entered.wait(1.0)

    second = threading.Thread(
        target=lambda: results.append(engine.finalize_camera("racing")),
    )
    second.start()
    assert engine.submit_frame(FramePacket("racing", 3, 3.0, _frame(130))) is False
    # Another camera remains independently usable while OCR is in flight.
    assert engine.submit_frame(FramePacket("other", 1, 1.0, _frame(10))) is False

    ocr.release.set()
    first.join(1.0)
    second.join(1.0)
    assert not first.is_alive()
    assert not second.is_alive()
    assert sorted(len(items) for items in results) == [0, 1]
    assert ocr.calls == 1


def test_runtime_reset_clears_finalized_state_and_duplicate_history() -> None:
    ocr = _OCR()
    engine = _engine(ocr)
    _harvest_one(engine, "replay")
    first = engine.finalize_camera("replay")
    assert len(first) == 1

    engine.reset_runtime_state()
    assert engine.telemetry()["events"] == 0
    _harvest_one(engine, "replay")
    second = engine.finalize_camera("replay")

    # The same plate and timestamps are valid in a new isolated runtime.
    assert len(second) == 1
    assert second[0].text == first[0].text
    assert ocr.calls == 2


def test_finalize_waits_for_background_owned_task_without_double_ocr() -> None:
    ocr = _OCR()
    engine = _engine(ocr)
    _harvest_one(engine, "background")
    callback_entered = threading.Event()
    callback_release = threading.Event()

    def delayed_apply(task, vote):
        callback_entered.set()
        assert callback_release.wait(1.0)
        return engine._apply_ocr_result(task, vote)

    assert engine.ocr_worker.start(delayed_apply) is True
    assert callback_entered.wait(1.0)
    results: list[list[object]] = []
    errors: list[Exception] = []

    def finalize() -> None:
        try:
            results.append(engine.finalize_camera("background"))
        except Exception as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    thread = threading.Thread(target=finalize)
    thread.start()
    assert thread.is_alive()
    callback_release.set()
    thread.join(1.0)
    assert engine.ocr_worker.stop()

    assert not thread.is_alive()
    assert errors == []
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].camera_id == "background"
    assert ocr.calls == 1
    assert engine.telemetry()["events"] == 1
    assert engine.finalize_camera("background") == []


def test_finalize_reuses_background_result_when_callback_does_not_apply_it() -> None:
    ocr = _BlockingOCR()
    engine = _engine(ocr)
    _harvest_one(engine, "background-unapplied")
    callback_calls = []
    assert engine.ocr_worker.start(lambda task, vote: callback_calls.append((task, vote)))
    assert ocr.entered.wait(1.0)

    results: list[list[object]] = []
    thread = threading.Thread(
        target=lambda: results.append(engine.finalize_camera("background-unapplied"))
    )
    thread.start()
    assert thread.is_alive()
    ocr.release.set()
    thread.join(1.0)
    assert engine.ocr_worker.stop()

    assert not thread.is_alive()
    assert len(callback_calls) == 1
    assert len(results) == 1
    assert len(results[0]) == 1
    assert results[0][0].camera_id == "background-unapplied"
    assert ocr.calls == 1
    assert engine.telemetry()["events"] == 1


def test_callback_failure_is_raised_only_after_all_final_tracks_are_emitted() -> None:
    ocr = _TwoPlateOCR()
    callbacks = []

    def partly_broken_callback(event) -> None:
        callbacks.append(event)
        if len(callbacks) == 1:
            raise RuntimeError("synthetic first callback failure")

    engine = EventDrivenANPREngine(
        _TwoPlateDetector(),
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            early_ocr_quality=1.1,
            same_camera_duplicate_seconds=0.0,
            cross_camera_duplicate_seconds=0.0,
            load_control_enabled=False,
        ),
        on_event=partly_broken_callback,
    )
    _harvest_one(engine, "two-tracks")
    assert len(engine.ocr_worker.queue) == 2

    try:
        engine.finalize_camera("two-tracks")
    except RuntimeError as exc:
        assert str(exc) == "synthetic first callback failure"
    else:  # pragma: no cover - assertion aid
        raise AssertionError("finalize must preserve the callback error signal")

    assert ocr.calls == 2
    assert len(callbacks) == 2
    assert {event.text for event in callbacks} == {"12ب34567", "34م56789"}
    assert engine.telemetry()["events"] == 2
    episodes = list(engine.state_for("two-tracks").tracks.values())
    assert len(episodes) == 2
    assert all(episode.phase is TrackPhase.DONE for episode in episodes)
    assert all(episode.event_emitted for episode in episodes)
    assert engine.finalize_camera("two-tracks") == []


def test_finalize_unseen_camera_latches_eof_until_explicit_new_lifecycle() -> None:
    ocr = _OCR()
    engine = _engine(ocr)

    assert engine.finalize_camera("unseen", final_seq=0, final_ts=5.0) == []
    unseen = engine.state_for("unseen")
    assert unseen.input_finalized is True
    assert unseen.final_seq == 0
    assert unseen.final_ts == 5.0
    assert engine.submit_frame(FramePacket("unseen", 1, 6.0, _frame(100))) is False
    assert unseen.last_received_seq == -1
    assert engine.detector.calls == 0

    engine.notify_stream_restart("unseen", preserve_activity=False)
    assert engine.state_for("unseen").input_finalized is False
    assert engine.submit_frame(FramePacket("unseen", 1, 7.0, _frame(10))) is False
    assert engine.state_for("unseen").last_received_seq == 1

    assert engine.finalize_camera("epoch-reopen") == []
    assert engine.submit_frame(
        FramePacket(
            "epoch-reopen",
            1,
            8.0,
            _frame(10),
            metadata={"producer_epoch": "new-source:1"},
        )
    ) is False
    reopened = engine.state_for("epoch-reopen")
    assert reopened.input_finalized is False
    assert reopened.last_received_seq == 1


def test_reset_epoch_rejects_old_background_result_without_dedup_leakage() -> None:
    ocr = _BlockingOCR()
    engine = _engine(ocr)
    callbacks = []
    new_event = threading.Event()

    def record_event(event) -> None:
        callbacks.append(event)
        new_event.set()

    engine.on_event = record_event
    _harvest_one(engine, "reset-race")
    assert engine.ocr_worker.start(engine._apply_ocr_result)
    assert ocr.entered.wait(1.0)

    old_epoch = engine.state_for("reset-race").runtime_epoch
    engine.reset_runtime_state()
    _harvest_one(engine, "reset-race")
    fresh_state = engine.state_for("reset-race")
    assert fresh_state.runtime_epoch > old_epoch

    ocr.release.set()
    assert new_event.wait(1.0)
    assert engine.ocr_worker.stop()

    # The old callback cannot mutate the new episode or seed duplicate state;
    # only the post-reset task emits even though plate/timestamps are identical.
    assert len(callbacks) == 1
    assert callbacks[0].episode_id == next(iter(fresh_state.tracks.values())).episode_id
    assert engine.telemetry()["events"] == 1
    assert ocr.calls == 2


def test_voter_failure_does_not_drop_later_track_and_failed_track_can_retry() -> None:
    ocr = _OCR()
    callbacks = []
    engine = EventDrivenANPREngine(
        _TwoPlateDetector(),
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            early_ocr_quality=1.1,
            same_camera_duplicate_seconds=0.0,
            cross_camera_duplicate_seconds=0.0,
            load_control_enabled=False,
        ),
        on_event=callbacks.append,
    )
    _harvest_one(engine, "voter-retry")
    real_voter = engine.ocr_worker.voter

    class _FailOnceVoter:
        calls = 0

        def vote(self, observations):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic voter failure")
            return real_voter.vote(observations)

    engine.ocr_worker.voter = _FailOnceVoter()
    try:
        engine.finalize_camera("voter-retry")
    except RuntimeError as exc:
        assert str(exc) == "synthetic voter failure"
    else:  # pragma: no cover - assertion aid
        raise AssertionError("the isolated voter failure must still be reported")

    state = engine.state_for("voter-retry")
    assert len(callbacks) == 1
    assert engine.telemetry()["events"] == 1
    assert state.input_finalized is True
    assert state.finalization_complete is False
    assert sum(episode.phase is TrackPhase.DONE for episode in state.tracks.values()) == 1
    failed = [
        episode for episode in state.tracks.values()
        if episode.phase is not TrackPhase.DONE
    ]
    assert len(failed) == 1
    assert len(failed[0].selector) == 1

    retry_events = engine.finalize_camera("voter-retry")
    assert len(retry_events) == 1
    assert len(callbacks) == 2
    assert engine.telemetry()["events"] == 2
    assert state.finalization_complete is True
    assert all(episode.phase is TrackPhase.DONE for episode in state.tracks.values())
    assert engine.finalize_camera("voter-retry") == []
