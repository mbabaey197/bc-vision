from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.engine_v2.dedup import DuplicateSuppressor, DuplicateSuppressorConfig
from app.engine_v2.load import AdaptiveLoadConfig, AdaptiveLoadController, LoadLevel, LoadSnapshot
from app.engine_v2.motion import AdaptiveMotionGate, MotionGateConfig
from app.engine_v2.ocr import OCRObservation, TemporalOCRVoter
from app.engine_v2.quality import BestPlateFrameSelector, evaluate_plate_quality
from app.engine_v2.scheduler import LatestOnlyPriorityQueue
from app.engine_v2.tracking import LightweightMultiObjectTracker, TrackerConfig
from app.engine_v2.types import OCRResult, PlateCandidate
from app.engine_v2.validator import IranianPlateValidator


def test_iranian_validator_normalizes_digits_and_rejects_bad_structure() -> None:
    validator = IranianPlateValidator()
    result = validator.validate("۱۲ ب ۳۴۵ ایران ۶۷")
    assert result.valid is True
    assert result.normalized == "12ب34567"
    assert result.province == "67"
    assert validator.validate("12X34567").valid is False
    assert validator.validate("00ب00067").reason == "zero_serial"


def test_temporal_voting_combines_equivalent_persian_reads() -> None:
    voter = TemporalOCRVoter(min_support=2)
    vote = voter.vote(
        [
            OCRObservation(OCRResult("12ب34567", 0.86), 0.72, 1),
            OCRObservation(OCRResult("۱۲ب۳۴۵۶۷", 0.91), 0.82, 2),
            OCRObservation(OCRResult("12ب34568", 0.78), 0.65, 3),
        ]
    )
    assert vote.valid is True
    assert vote.text == "12ب34567"
    assert vote.support == 2
    weak_single = voter.vote([OCRObservation(OCRResult("12ب34567", 0.70), 0.95)])
    assert weak_single.valid is False


def test_quality_selector_uses_sharpness_exposure_size_and_confidence() -> None:
    rng = np.random.default_rng(42)
    sharp = rng.integers(40, 220, size=(48, 160, 3), dtype=np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    sharp_score = evaluate_plate_quality(
        sharp,
        detector_confidence=0.92,
        frame_shape=(240, 320),
        bbox=(40, 80, 200, 128),
    )
    blurred_score = evaluate_plate_quality(
        blurred,
        detector_confidence=0.60,
        frame_shape=(240, 320),
        bbox=(40, 80, 120, 104),
    )
    assert sharp_score.score > blurred_score.score
    assert sharp_score.motion_blur > blurred_score.motion_blur

    selector = BestPlateFrameSelector(capacity=2, min_sequence_gap=0)
    selector.add(
        blurred,
        bbox=(40, 80, 120, 104),
        seq=1,
        ts=1.0,
        detector_confidence=0.60,
        frame_shape=(240, 320),
    )
    selector.add(
        sharp,
        bbox=(40, 80, 200, 128),
        seq=2,
        ts=2.0,
        detector_confidence=0.92,
        frame_shape=(240, 320),
    )
    assert selector.best is not None
    assert selector.best.seq == 2


def test_motion_gate_clamps_roi_instead_of_using_negative_numpy_indices() -> None:
    gate = AdaptiveMotionGate(MotionGateConfig(blur_kernel=1))
    baseline = np.zeros((20, 20, 3), dtype=np.uint8)
    changed = baseline.copy()
    changed[:10, :10] = 255
    roi = (-5, -5, 10, 10)
    assert gate.score(baseline, roi) == 0.0
    assert gate.score(changed, roi) == 1.0


def test_motion_gate_rejects_invalid_thresholds_and_channels() -> None:
    with pytest.raises(ValueError, match="pixel_threshold"):
        MotionGateConfig(pixel_threshold=0)
    gate = AdaptiveMotionGate()
    with pytest.raises(ValueError, match="1, 3, or 4 channels"):
        gate.score(np.zeros((10, 10, 2), dtype=np.uint8))


def test_lightweight_tracker_keeps_two_vehicle_identities_and_predicts() -> None:
    tracker = LightweightMultiObjectTracker(TrackerConfig(max_missed=1))
    first = tracker.update(
        [PlateCandidate((10, 10, 50, 25), 0.9), PlateCandidate((150, 20, 195, 36), 0.88)],
        seq=1,
    )
    assert [item.track_id for item in first.observations] == [1, 2]
    second = tracker.update(
        [PlateCandidate((15, 10, 55, 25), 0.91), PlateCandidate((145, 20, 190, 36), 0.9)],
        seq=2,
    )
    assert [item.track_id for item in second.observations] == [1, 2]
    predicted = tracker.predict(3)
    assert len(predicted) == 2
    assert all(item.predicted for item in predicted)
    tracker.update([], seq=3)
    removed = tracker.update([], seq=4)
    assert set(removed.removed_track_ids) == {1, 2}


def test_latest_only_queue_counts_replacement_and_expiration() -> None:
    queue = LatestOnlyPriorityQueue[int](max_items=2)
    assert queue.submit("cam-a", 1)
    assert queue.submit("cam-a", 2)
    assert queue.pop() == 2
    assert queue.stats.replaced == 1
    assert queue.submit("cam-b", 3)
    time.sleep(0.001)
    assert queue.pop(max_age_seconds=0.0) is None
    assert queue.stats.expired == 1


def test_priority_scheduler_does_not_starve_other_cameras() -> None:
    queue = LatestOnlyPriorityQueue[str](max_items=4, fairness_penalty=8)
    queue.submit("cam-a", "a1", priority=10)
    queue.submit("cam-b", "b1", priority=10)
    assert queue.pop() == "a1"
    # cam-a is now near an event (priority 5), but it cannot monopolize every
    # scheduling turn while cam-b already has real-time work waiting.
    queue.submit("cam-a", "a2", priority=5)
    assert queue.pop() == "b1"
    assert queue.pop() == "a2"


def test_latest_only_queue_never_resurrects_old_node_after_key_reuse() -> None:
    queue = LatestOnlyPriorityQueue[str](max_items=2)
    assert queue.submit("cam", "old", priority=50)
    assert queue.submit("cam", "new-now", priority=0)
    assert queue.pop() == "new-now"

    # The old priority-50 heap node is still physically present. Reusing the
    # key must not make that tombstone look live again.
    assert queue.submit("cam", "new-later", priority=50)
    assert queue.pop() == "new-later"


def test_latest_only_queue_never_resurrects_evicted_node() -> None:
    queue = LatestOnlyPriorityQueue[str](max_items=2)
    assert queue.submit("cam-a", "a-old", priority=50)
    assert queue.submit("cam-b", "b", priority=10)
    assert queue.submit("cam-c", "c", priority=0)  # evicts cam-a
    assert queue.pop() == "c"
    assert queue.submit("cam-a", "a-new", priority=50)
    assert queue.pop() == "b"
    assert queue.pop() == "a-new"


def test_duplicate_suppression_is_exact_and_bounded_by_camera_windows() -> None:
    cache = DuplicateSuppressor(
        DuplicateSuppressorConfig(
            same_camera_window_seconds=10,
            cross_camera_window_seconds=2,
        )
    )
    assert cache.check_and_record("a", "12ب34567", 100).duplicate is False
    assert cache.check_and_record("a", "12ب34567", 101).reason == "same_camera_window"
    assert cache.check_and_record("b", "12ب34567", 101).reason == "overlapping_camera_window"
    assert cache.check_and_record("a", "12ب34568", 101).duplicate is False


def _snapshot(cpu: float, queue_depth: int = 0) -> LoadSnapshot:
    return LoadSnapshot(
        timestamp=1.0,
        cpu_percent=cpu,
        detector_latency_ms=10,
        ocr_latency_ms=8,
        queue_depth=queue_depth,
        queue_capacity=100,
        active_cameras=4,
        total_cameras=16,
    )


def test_adaptive_load_sheds_immediately_and_recovers_gradually() -> None:
    controller = AdaptiveLoadController(AdaptiveLoadConfig(recovery_samples=2))
    normal = controller.policy
    critical = controller.observe(_snapshot(99, queue_depth=90))
    assert critical.level is LoadLevel.CRITICAL
    assert critical.detector_stride_multiplier > normal.detector_stride_multiplier
    assert critical.max_ocr_candidates < normal.max_ocr_candidates
    assert critical.idle_fps_scale < normal.idle_fps_scale

    controller.observe(_snapshot(10))
    controller.observe(_snapshot(10))
    assert controller.level is LoadLevel.HIGH
    for _ in range(6):
        controller.observe(_snapshot(10))
    assert controller.level is LoadLevel.NORMAL


def test_adaptive_load_honors_configured_cpu_and_queue_boundaries() -> None:
    controller = AdaptiveLoadController(AdaptiveLoadConfig(ema_alpha=0.01))
    assert controller.observe(_snapshot(72)).level is LoadLevel.ELEVATED
    controller.reset()
    assert controller.observe(_snapshot(85)).level is LoadLevel.HIGH
    controller.reset()
    assert controller.observe(_snapshot(94)).level is LoadLevel.CRITICAL

    controller.reset()
    assert controller.observe(_snapshot(0, queue_depth=25)).level is LoadLevel.ELEVATED
    controller.reset()
    assert controller.observe(_snapshot(0, queue_depth=55)).level is LoadLevel.HIGH
    controller.reset()
    assert controller.observe(_snapshot(0, queue_depth=82)).level is LoadLevel.CRITICAL


def test_adaptive_load_rejects_ambiguous_threshold_configuration() -> None:
    with pytest.raises(ValueError, match="CPU thresholds"):
        AdaptiveLoadConfig(target_cpu_percent=90, high_cpu_percent=80)
    with pytest.raises(ValueError, match="queue thresholds"):
        AdaptiveLoadConfig(elevated_queue_ratio=0.7, high_queue_ratio=0.5)
