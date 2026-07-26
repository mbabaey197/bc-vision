import numpy as np

from app.ai.pipeline import (
    PlateConsensusTracker,
    image_quality,
    plate_similarity,
    process_frame,
)
from app.ai.plate_rules import normalize_plate


def result(
    plate,
    confidence,
    bbox=(100, 100, 300, 150),
    quality=0.8,
):
    return {
        "plate": plate,
        "plate_norm": normalize_plate(plate),
        "valid": True,
        "confidence": confidence,
        "quality_score": quality,
        "bbox": bbox,
        "crop": np.zeros((30, 120, 3), dtype=np.uint8),
    }


def test_consensus_requires_three_repeated_observations():
    tracker = PlateConsensusTracker(
        min_votes=1,
        emit_cooldown=10,
    )
    assert tracker.min_votes == 3
    assert tracker.update([
        result("31-ط-556-74", 0.96)
    ], timestamp=0.0) == []
    assert tracker.update([
        result("31-ط-556-74", 0.94, bbox=(102, 100, 302, 150))
    ], timestamp=0.1) == []
    emitted = tracker.update([
        result("31-ط-556-74", 0.92, bbox=(104, 100, 304, 150))
    ], timestamp=0.2)
    assert len(emitted) == 1
    assert emitted[0]["plate"] == "31-ط-556-74"
    assert emitted[0]["consensus_votes"] == 3


def test_position_voting_corrects_lam_tah_confusion():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    observations = [
        ("31-ل-556-74", 0.98),
        ("31-ط-556-74", 0.72),
        ("31-ط-556-74", 0.70),
        ("31-ط-556-74", 0.68),
    ]
    emitted = []
    for index, (plate, confidence) in enumerate(observations):
        emitted.extend(tracker.update([
            result(
                plate,
                confidence,
                bbox=(100 + index * 2, 100, 300 + index * 2, 150),
            )
        ], timestamp=index * 0.1))
    assert emitted
    assert emitted[-1]["plate"] == "31-ط-556-74"
    assert emitted[-1]["position_agreement"][2]["character"] == "ط"
    assert emitted[-1]["position_agreement"][2]["votes"] == 3


def test_position_voting_corrects_wrong_middle_digit():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    observations = [
        ("55-ط-629-74", 0.99),
        ("55-ط-639-74", 0.74),
        ("55-ط-639-74", 0.72),
        ("55-ط-639-74", 0.70),
    ]
    emitted = []
    for index, (plate, confidence) in enumerate(observations):
        emitted.extend(tracker.update([
            result(
                plate,
                confidence,
                bbox=(100 + index * 2, 100, 300 + index * 2, 150),
            )
        ], timestamp=index * 0.1))
    assert emitted
    assert emitted[-1]["plate"] == "55-ط-639-74"


def test_reference_plate_with_region_digits_is_preserved():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    emitted = []
    for index, plate in enumerate([
        "84-ب-571-32",
        "84-ب-571-33",
        "84-ب-571-33",
        "84-ب-571-33",
    ]):
        emitted.extend(tracker.update([
            result(
                plate,
                0.75,
                bbox=(100 + index * 2, 100, 300 + index * 2, 150),
            )
        ], timestamp=index * 0.1))
    assert emitted
    assert emitted[-1]["plate"] == "84-ب-571-33"


def test_ambiguous_character_position_is_rejected():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    observations = [
        "31-ط-556-74",
        "31-ل-556-74",
        "31-ط-556-74",
        "31-ل-556-74",
    ]
    emitted = []
    for index, plate in enumerate(observations):
        emitted.extend(tracker.update([
            result(
                plate,
                0.8,
                bbox=(100 + index * 2, 100, 300 + index * 2, 150),
            )
        ], timestamp=index * 0.1))
    assert emitted == []
    assert tracker.flush() == []


def test_single_exceptional_read_is_never_emitted():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    assert tracker.update([
        result("31-ط-556-74", 1.0, quality=1.0)
    ], timestamp=0.0) == []
    assert tracker.flush() == []


def test_no_duplicate_emission_during_cooldown():
    tracker = PlateConsensusTracker(emit_cooldown=5)
    tracker.update([result("31-ط-556-74", 0.8)], timestamp=0.0)
    tracker.update([result("31-ط-556-74", 0.8)], timestamp=0.1)
    assert tracker.update([
        result("31-ط-556-74", 0.8)
    ], timestamp=0.2)
    assert tracker.update([
        result("31-ط-556-74", 0.9)
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
        "31-ط-556-74",
        "31ط55674",
    ) == 1.0
    assert plate_similarity("", "31ط55674") == 0.0


def test_direct_yolo_text_skips_expensive_ocr_and_vehicle_ai(
    monkeypatch,
):
    crop = np.full((40, 160, 3), 180, dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 170, 60),
            "confidence": 0.8,
            "method": "yolo-plate+chars",
            "direct_text": "27-ط-253-74",
            "direct_ocr_confidence": 0.86,
            "direct_ocr_attempted": True,
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("EasyOCR should not run")
        ),
    )
    monkeypatch.setattr(
        "app.ai.pipeline.analyze_vehicle",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("vehicle AI should be deferred")
        ),
    )

    rows = process_frame(
        np.full((100, 200, 3), 100, dtype=np.uint8)
    )

    assert len(rows) == 1
    assert rows[0]["plate"] == "27-ط-253-74"
    assert rows[0]["valid"] is True
    assert rows[0]["vehicle_type"] == "نامشخص"
