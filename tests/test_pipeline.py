import numpy as np

from app.ai.pipeline import (
    PlateConsensusTracker,
    image_quality,
    plate_similarity,
)


def result(
    plate,
    confidence,
    bbox=(100, 100, 300, 150),
    quality=0.7,
):
    return {
        "plate": plate,
        "plate_norm": plate.replace("-", ""),
        "valid": True,
        "confidence": confidence,
        "quality_score": quality,
        "bbox": bbox,
        "crop": np.zeros((30, 120, 3), dtype=np.uint8),
    }


def test_consensus_requires_repeated_observations():
    tracker = PlateConsensusTracker(
        min_votes=2,
        emit_cooldown=10,
    )
    assert tracker.update([
        result("12-ب-345-67", 0.62)
    ], timestamp=0.0) == []
    emitted = tracker.update([
        result(
            "12-ب-345-67",
            0.72,
            bbox=(104, 102, 304, 152),
        )
    ], timestamp=0.2)
    assert len(emitted) == 1
    assert emitted[0]["plate"] == "12-ب-345-67"
    assert emitted[0]["consensus_votes"] == 2
    assert emitted[0]["confidence"] > 0.70


def test_consensus_outvotes_single_bad_read():
    tracker = PlateConsensusTracker(
        min_votes=2,
        emit_cooldown=10,
    )
    tracker.update([
        result("12-ب-345-67", 0.64)
    ], timestamp=0.0)
    tracker.update([
        result("12-ب-345-76", 0.60)
    ], timestamp=0.1)
    emitted = tracker.update([
        result("12-ب-345-67", 0.75)
    ], timestamp=0.2)
    assert emitted
    assert emitted[0]["plate"] == "12-ب-345-67"


def test_no_duplicate_emission_during_cooldown():
    tracker = PlateConsensusTracker(
        min_votes=2,
        emit_cooldown=5,
    )
    tracker.update([
        result("12-ب-345-67", 0.7)
    ], timestamp=0)
    assert tracker.update([
        result("12-ب-345-67", 0.8)
    ], timestamp=0.2)
    assert tracker.update([
        result("12-ب-345-67", 0.9)
    ], timestamp=0.4) == []


def test_quality_extremes_are_bounded():
    for image in [
        np.zeros((50, 200, 3), dtype=np.uint8),
        np.full((50, 200, 3), 255, dtype=np.uint8),
        np.random.default_rng(42).integers(
            0,
            256,
            (50, 200, 3),
            dtype=np.uint8,
        ),
    ]:
        quality = image_quality(image)
        assert 0 <= quality["score"] <= 1


def test_plate_similarity_handles_formatting():
    assert plate_similarity(
        "12-ب-345-67",
        "12ب34567",
    ) == 1.0
    assert plate_similarity("", "12ب34567") == 0.0
