from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import app.engine_v2.quality as quality_module
from app.engine_v2.dedup import DuplicateSuppressor, DuplicateSuppressorConfig
from app.engine_v2.ocr import (
    OCRObservation,
    OCRTask,
    SharedOCRWorker,
    TemporalOCRVoter,
)
from app.engine_v2.quality import BestPlateFrameSelector, PlateFrame, QualityBreakdown
from app.engine_v2.tracking import LightweightMultiObjectTracker, TrackerConfig
from app.engine_v2.types import OCRResult, PlateCandidate
from app.engine_v2.validator import IranianPlateValidator, IranianPlateValidatorConfig


PLATE = "12ب34567"


def _crop(value: int = 90) -> np.ndarray:
    return np.full((20, 60, 3), value, dtype=np.uint8)


def _quality(score: float) -> QualityBreakdown:
    return QualityBreakdown(score, score, score, score)


def _task(key: str, *, seq: int = 1) -> OCRTask:
    return OCRTask(key, [_crop()], [0.8], [seq])


def test_duplicate_suppressor_enforces_hard_cap_and_validates_configuration() -> None:
    suppressor = DuplicateSuppressor(
        DuplicateSuppressorConfig(
            same_camera_window_seconds=100.0,
            cross_camera_window_seconds=100.0,
            max_entries=2,
        )
    )

    for index in range(7):
        assert suppressor.check_and_record("cam", f"plate-{index}", 0.0).duplicate is False

    assert len(suppressor._by_camera) == 2
    assert len(suppressor._global) == 2
    with pytest.raises(ValueError, match="max_entries"):
        DuplicateSuppressorConfig(max_entries=0)
    with pytest.raises(ValueError, match="finite non-negative"):
        DuplicateSuppressorConfig(same_camera_window_seconds=float("nan"))
    with pytest.raises(ValueError, match="ts must be finite"):
        suppressor.check_and_record("cam", PLATE, float("inf"))


def test_duplicate_suppressor_preserves_last_seen_for_out_of_order_events() -> None:
    suppressor = DuplicateSuppressor(
        DuplicateSuppressorConfig(
            same_camera_window_seconds=5.0,
            cross_camera_window_seconds=1.5,
        )
    )

    assert suppressor.check_and_record("cam", PLATE, 100.0).duplicate is False
    # This older event is outside the window and can remain reviewable, but it
    # must not replace the newer last-seen timestamp.
    assert suppressor.check_and_record("cam", PLATE, 90.0).duplicate is False
    assert suppressor._by_camera[("cam", PLATE)] == 100.0
    decision = suppressor.check_and_record("cam", PLATE, 101.0)
    assert decision.duplicate is True
    assert decision.previous_ts == 100.0
    assert suppressor._by_camera[("cam", PLATE)] == 101.0


def test_duplicate_suppressor_records_suppressed_cross_camera_sightings() -> None:
    suppressor = DuplicateSuppressor(
        DuplicateSuppressorConfig(
            same_camera_window_seconds=20.0,
            cross_camera_window_seconds=1.5,
        )
    )

    assert suppressor.check_and_record("A", PLATE, 0.0).duplicate is False
    assert suppressor.check_and_record("B", PLATE, 1.0).duplicate is True
    third = suppressor.check_and_record("C", PLATE, 2.0)
    assert third.duplicate is True
    assert third.previous_ts == 1.0


def test_duplicate_suppressor_uses_sliding_last_seen_window() -> None:
    suppressor = DuplicateSuppressor(
        DuplicateSuppressorConfig(
            same_camera_window_seconds=5.0,
            cross_camera_window_seconds=0.0,
        )
    )

    assert suppressor.check_and_record("cam", PLATE, 0.0).duplicate is False
    assert suppressor.check_and_record("cam", PLATE, 4.0).duplicate is True
    decision = suppressor.check_and_record("cam", PLATE, 8.0)
    assert decision.duplicate is True
    assert decision.previous_ts == 4.0


def test_validator_canonicalizes_diplomatic_case_and_rejects_partial_ir_label() -> None:
    validator = IranianPlateValidator()

    upper = validator.validate("12D34567")
    lower = validator.validate("12d34567")
    assert upper.valid is lower.valid is True
    assert upper.normalized == lower.normalized == "12D34567"
    assert validator.validate("12D345IR67").valid is False
    assert validator.validate("12D345IRAN67").normalized == "12D34567"


def test_validator_canonicalizes_alef_word_to_ctc_glyph_contract() -> None:
    validator = IranianPlateValidator()
    short = validator.validate("12ا34567")
    word = validator.validate("12الف34567")
    assert short.valid is True
    assert word.valid is True
    assert short.normalized == word.normalized == "12ا34567"


def test_validator_removes_bidi_controls_and_validates_configuration() -> None:
    validator = IranianPlateValidator()

    result = validator.validate("۱۲ ب ۳۴۵ ایران \u202e۶۷")
    assert result.valid is True
    assert result.normalized == PLATE
    with pytest.raises(ValueError, match="allowed_letters"):
        IranianPlateValidatorConfig(allowed_letters=())
    with pytest.raises(ValueError, match="province range"):
        IranianPlateValidatorConfig(min_province=90, max_province=20)


def test_temporal_voter_requires_distinct_sequences_and_observation_floors() -> None:
    voter = TemporalOCRVoter()
    same_frame = [
        OCRObservation(OCRResult(PLATE, 0.80), 0.70, 7),
        OCRObservation(OCRResult(PLATE, 0.82), 0.72, 7),
    ]
    distinct_frames = [same_frame[0], OCRObservation(OCRResult(PLATE, 0.82), 0.72, 8)]

    same_vote = voter.vote(same_frame)
    assert same_vote.valid is False
    assert same_vote.support == 1
    assert voter.vote(distinct_frames).valid is True

    low_confidence = voter.vote(
        [
            OCRObservation(OCRResult(PLATE, 0.0), 0.9, 1),
            OCRObservation(OCRResult(PLATE, 0.0), 0.9, 2),
        ]
    )
    assert low_confidence.valid is False
    assert low_confidence.reason == "no_observations_above_floor"

    low_quality = voter.vote(
        [
            OCRObservation(OCRResult(PLATE, 0.9), 0.0, 1),
            OCRObservation(OCRResult(PLATE, 0.9), 0.0, 2),
        ]
    )
    assert low_quality.valid is False


def test_temporal_voter_groups_diplomatic_case_canonically() -> None:
    vote = TemporalOCRVoter().vote(
        [
            OCRObservation(OCRResult("12d34567", 0.8), 0.7, 1),
            OCRObservation(OCRResult("12D34567", 0.8), 0.7, 2),
        ]
    )

    assert vote.valid is True
    assert vote.text == "12D34567"
    assert vote.support == 2


class _ConstantOCR:
    def read(self, crop: np.ndarray) -> OCRResult:
        del crop
        return OCRResult(PLATE, 0.95)


class _FailingOCR:
    def read(self, crop: np.ndarray) -> OCRResult:
        del crop
        raise RuntimeError("synthetic OCR failure")


def test_shared_ocr_worker_reports_queue_rejection_and_expires_stale_tasks() -> None:
    worker = SharedOCRWorker(_ConstantOCR(), queue_size=1, max_task_age_seconds=None)
    assert worker.submit(_task("first")) is True
    assert worker.submit(_task("second")) is False
    assert worker.process_next() is not None

    stale_worker = SharedOCRWorker(_ConstantOCR(), max_task_age_seconds=0.0)
    assert stale_worker.submit(_task("stale")) is True
    time.sleep(0.001)
    assert stale_worker.process_next() is None
    assert stale_worker.queue.stats.expired == 1


def test_shared_ocr_worker_converts_inference_error_to_failed_vote() -> None:
    worker = SharedOCRWorker(_FailingOCR(), max_task_age_seconds=None)
    assert worker.submit(_task("failure")) is True

    processed = worker.process_next()
    assert processed is not None
    _, vote = processed
    assert vote.valid is False
    assert vote.reason == "ocr_error"
    assert worker.stats.inference_count == 1
    assert worker.stats.failed_inference_count == 1
    assert worker.stats.failed_task_count == 1
    assert "synthetic OCR failure" in (worker.stats.last_error or "")


def test_shared_ocr_worker_survives_callback_errors() -> None:
    worker = SharedOCRWorker(_ConstantOCR(), max_task_age_seconds=None)
    two_callbacks = threading.Event()
    callback_count = 0

    def callback(task: OCRTask, vote: object) -> None:
        nonlocal callback_count
        del task, vote
        callback_count += 1
        if callback_count >= 2:
            two_callbacks.set()
        raise RuntimeError("synthetic callback failure")

    assert worker.start(callback) is True
    assert worker.submit(_task("one", seq=1)) is True
    assert worker.submit(_task("two", seq=2)) is True
    assert two_callbacks.wait(1.0)
    assert worker.stop(1.0) is True
    assert worker.stats.task_count == 2
    assert worker.stats.callback_error_count == 2


def test_shared_ocr_worker_does_not_restart_while_timed_out_thread_is_alive() -> None:
    entered = threading.Event()
    release = threading.Event()

    class BlockingOCR:
        def read(self, crop: np.ndarray) -> OCRResult:
            del crop
            entered.set()
            release.wait(1.0)
            return OCRResult(PLATE, 0.95)

    worker = SharedOCRWorker(BlockingOCR(), max_task_age_seconds=None)
    assert worker.start(lambda _task, _vote: None) is True
    assert worker.submit(_task("blocking")) is True
    assert entered.wait(1.0)
    assert worker.stop(0.001) is False
    assert worker.start(lambda _task, _vote: None) is False
    release.set()
    assert worker.stop(1.0) is True


def test_tracker_validates_configuration_detection_and_sequence() -> None:
    with pytest.raises(ValueError, match="max_center_distance"):
        TrackerConfig(max_center_distance=0.0)
    with pytest.raises(ValueError, match="min_iou"):
        TrackerConfig(min_iou=1.1)

    tracker = LightweightMultiObjectTracker()
    with pytest.raises(ValueError, match="positive width"):
        tracker.update([PlateCandidate((5, 0, 5, 10), 0.9)], 1)

    tracker.update([PlateCandidate((0, 0, 10, 10), 0.9)], 1)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.update([], 1)


def test_tracker_age_uses_sequence_span_without_double_counting_misses() -> None:
    tracker = LightweightMultiObjectTracker(TrackerConfig(max_missed=10))
    tracker.update([PlateCandidate((0, 0, 10, 10), 0.9)], 1)
    tracker.update([], 5)
    assert tracker.tracks[0].age == 5
    tracker.update([], 6)
    assert tracker.tracks[0].age == 6


def test_best_frame_selector_enforces_gap_against_every_nearby_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = BestPlateFrameSelector(capacity=5, min_sequence_gap=2)
    crop = _crop()
    selector._frames = [
        PlateFrame(crop, (0, 0, 60, 20), 1, 1.0, 0.9, _quality(0.9)),
        PlateFrame(crop, (0, 0, 60, 20), 5, 5.0, 0.1, _quality(0.1)),
    ]
    monkeypatch.setattr(quality_module, "evaluate_plate_quality", lambda *args, **kwargs: _quality(0.5))

    result = selector.add(
        crop,
        bbox=(0, 0, 60, 20),
        seq=3,
        ts=3.0,
        detector_confidence=0.5,
        frame_shape=(20, 60),
    )

    assert result is None
    assert [frame.seq for frame in selector.selected(5)] == [1, 5]
    assert selector.selected(0) == []


def test_best_frame_selector_returns_none_when_capacity_evicts_new_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selector = BestPlateFrameSelector(capacity=1, min_sequence_gap=0)
    crop = _crop()
    selector._frames = [
        PlateFrame(crop, (0, 0, 60, 20), 1, 1.0, 0.9, _quality(0.9)),
    ]
    monkeypatch.setattr(quality_module, "evaluate_plate_quality", lambda *args, **kwargs: _quality(0.1))

    added = selector.add(
        crop,
        bbox=(0, 0, 60, 20),
        seq=10,
        ts=10.0,
        detector_confidence=0.1,
        frame_shape=(20, 60),
    )

    assert added is None
    assert selector.best is not None
    assert selector.best.seq == 1
