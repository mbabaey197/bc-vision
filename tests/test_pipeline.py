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


def test_unreadable_capture_upgrades_on_same_track_with_clearest_frame():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        emit_cooldown=30,
    )
    dark = np.full((100, 200, 3), 25, dtype=np.uint8)
    clear = np.full((100, 200, 3), 210, dtype=np.uint8)
    unreadable = result(
        "ناخوانا",
        0.31,
        bbox=(20, 30, 150, 65),
        quality=0.20,
    )
    unreadable.update({
        "valid": False,
        "plate_norm": "",
        "detector_confidence": 0.78,
    })

    first = tracker.update([unreadable], timestamp=0.0, frame=dark)
    assert len(first) == 1
    assert first[0]["capture_only"] is True
    assert first[0]["plate"] == "در حال بررسی"
    assert first[0]["provisional"] is True
    track_id = first[0]["track_id"]

    recognized = result(
        "31-ط-556-74",
        0.92,
        bbox=(23, 30, 153, 65),
        quality=0.95,
    )
    recognized["detector_confidence"] = 0.94
    tracker.update([recognized], timestamp=0.2, frame=clear)
    tracker.update([recognized], timestamp=0.4, frame=clear)
    final = tracker.update([recognized], timestamp=0.6, frame=clear)

    recognized_rows = [
        row for row in final if not row.get("capture_only")
    ]
    assert len(recognized_rows) == 1
    assert recognized_rows[0]["track_id"] == track_id
    assert recognized_rows[0]["plate_norm"] == "31ط55674"
    assert int(recognized_rows[0]["capture_frame"].mean()) == 210


def test_unreadable_is_delayed_until_repeated_failed_reads():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 200, 3), 120, dtype=np.uint8)
    unreadable = result(
        "ناخوانا",
        0.35,
        bbox=(20, 30, 150, 65),
        quality=0.65,
    )
    unreadable.update({
        "valid": False,
        "plate_norm": "",
        "detector_confidence": 0.82,
        "dedicated_ocr_attempted": True,
        "generic_ocr_attempted": True,
    })

    first = tracker.update([unreadable], timestamp=0.0, frame=frame)
    second = tracker.update([unreadable], timestamp=0.4, frame=frame)
    third = tracker.update([unreadable], timestamp=0.9, frame=frame)

    assert first[0]["plate"] == "در حال بررسی"
    assert all(row["plate"] != "ناخوانا" for row in first + second)
    finalized = [
        row for row in third
        if row.get("unreadable_final")
    ]
    assert len(finalized) == 1
    assert finalized[0]["plate"] == "ناخوانا"


def test_failed_dedicated_reader_uses_generic_ocr_on_good_crop(
    monkeypatch,
):
    rng = np.random.default_rng(7)
    crop = rng.integers(0, 255, (48, 180, 3), dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 190, 68),
            "confidence": 0.88,
            "method": "yolo",
            "direct_text": "",
            "direct_ocr_confidence": 0.0,
            "direct_ocr_attempted": True,
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate",
        lambda _crop: ("55-ط-639-74", 0.81),
    )

    rows = process_frame(
        np.full((100, 220, 3), 100, dtype=np.uint8)
    )

    assert rows[0]["plate_norm"] == "55ط63974"
    assert rows[0]["generic_ocr_attempted"] is True


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
