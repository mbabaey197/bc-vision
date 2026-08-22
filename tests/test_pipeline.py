import numpy as np

from app.ai.pipeline import (
    PlateConsensusTracker,
    _ocr_crop_maturity,
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


def test_equal_ocr_text_never_links_distant_vehicles():
    tracker = PlateConsensusTracker(
        min_votes=3,
        max_age_seconds=3.0,
    )
    tracker.update([
        result(
            "31-ط-556-74",
            0.91,
            bbox=(20, 30, 180, 70),
        )
    ], timestamp=0.0)
    tracker.update([
        result(
            "31-ط-556-74",
            0.92,
            bbox=(900, 500, 1060, 540),
        )
    ], timestamp=0.1)

    assert len(tracker.active_track_ids()) == 2


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


def test_temporal_consensus_recovers_consistent_second_hypothesis():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    correct = "31ط55674"
    emitted = []
    for index, wrong in enumerate((
        "30ط55674",
        "32ط55674",
        "33ط55674",
    )):
        row = result(
            wrong,
            0.82,
            bbox=(100 + index * 2, 100, 300 + index * 2, 150),
        )
        row["plate_hypotheses"] = [
            {
                "plate_norm": wrong,
                "confidence": 0.82,
                "score": 0.82,
            },
            {
                "plate_norm": correct,
                "confidence": 0.78,
                "score": 0.78,
            },
        ]
        emitted.extend(
            tracker.update([row], timestamp=index * 0.1)
        )

    assert emitted == []
    assert tracker.flush() == []


def test_mature_hezar_top_k_can_reach_strict_consensus():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    correct = "31ط55674"
    emitted = []
    for index, wrong in enumerate((
        "31ط55874",
        "31ط55974",
        "31ط55374",
    )):
        row = result(
            wrong,
            0.48,
            bbox=(100 + index * 2, 100, 300 + index * 2, 150),
        )
        row.update({
            "plate_norm": "",
            "valid": False,
            "needs_review": True,
            "best_effort": True,
            "raw_guess_text": row["plate"],
            "raw_guess_norm": normalize_plate(row["plate"]),
            # This is the actual process_frame provenance when Hezar rejects
            # its primary path but exposes eligible Top-K temporal evidence.
            "ocr_engine": "none",
            "raw_guess_engine": "hezar-crnn-fa-v2-onnx",
            "ocr_confidence": 0.48,
            "detector_confidence": 0.90,
            "ocr_crop_mature": True,
            "hypotheses_accepted_for_consensus": True,
            "plate_hypotheses": [
                {
                    "plate_norm": wrong,
                    "confidence": 0.48,
                    "score": 0.48,
                    "engine": "hezar-crnn-fa-v2-onnx",
                    "temporal_evidence": True,
                },
                {
                    "plate_norm": correct,
                    "confidence": 0.44,
                    "score": 0.44,
                    "engine": "hezar-crnn-fa-v2-onnx",
                    "temporal_evidence": True,
                },
            ],
        })
        observed = tracker.update(
            [row],
            timestamp=index * 0.1,
        )
        if index < 2:
            assert observed == []
        emitted.extend(observed)

    assert len(emitted) == 1
    assert emitted[0]["plate_norm"] == correct
    assert emitted[0]["plate"] == "31-ط-556-74"
    assert emitted[0]["valid"] is True
    assert emitted[0]["read_status"] == "confirmed-ai"
    assert emitted[0]["needs_review"] is False
    assert emitted[0]["guess_supporting_frames"] == 3
    assert emitted[0]["ocr_engine"] == "hezar-crnn-fa-v2-onnx"
    assert emitted[0]["raw_guess_engine"] == "hezar-crnn-fa-v2-onnx"


def test_explicitly_ineligible_hypothesis_gets_no_strict_vote():
    row = result("31-ط-556-74", 0.82)
    row["ocr_confidence"] = 0.82
    row["plate_hypotheses"] = [{
        "plate_norm": "31ط55874",
        "confidence": 0.99,
        "score": 0.99,
        "temporal_evidence": False,
    }]

    probabilities = PlateConsensusTracker._position_probabilities(row)

    assert probabilities[5] == {"6": 1.0}


def test_ineligible_hypothesis_cannot_inflate_consensus_or_engine():
    def consensus(include_rejected):
        tracker = PlateConsensusTracker(emit_cooldown=10)
        emitted = []
        for index in range(3):
            row = result(
                "31-ط-556-74",
                0.82,
                bbox=(100 + index * 2, 100, 300 + index * 2, 150),
            )
            row.update({
                "ocr_confidence": 0.82,
                "detector_confidence": 0.90,
                "ocr_engine": "hezar-crnn-fa-v2-onnx",
                "plate_hypotheses": (
                    [{
                        "plate_norm": "31ط55674",
                        "confidence": 0.99,
                        "score": 0.99,
                        "engine": "dedicated-character-detector",
                        "temporal_evidence": False,
                    }]
                    if include_rejected
                    else []
                ),
            })
            emitted.extend(tracker.update(
                [row],
                timestamp=index * 0.1,
            ))
        assert len(emitted) == 1
        return emitted[0]

    baseline = consensus(False)
    injected = consensus(True)

    assert injected["ocr_confidence"] == baseline["ocr_confidence"]
    assert injected["confidence"] == baseline["confidence"]
    assert injected["ocr_engine"] == baseline["ocr_engine"]
    assert injected["ocr_engine"] == "hezar-crnn-fa-v2-onnx"


def test_ocr_waits_for_native_crop_maturity_before_identity(
    monkeypatch,
):
    rng = np.random.default_rng(42)
    heights = iter((8, 12, 20, 32, 40))
    ocr_calls = []

    def detect(*_args, **_kwargs):
        height = next(heights)
        crop = rng.integers(
            0,
            256,
            (height, height * 4, 3),
            dtype=np.uint8,
        )
        return [{
            "crop": crop,
            "bbox": (100, 100, 300, 150),
            "confidence": 0.90,
            "method": "yolo11n",
            "direct_text": "",
            "direct_ocr_confidence": 0.0,
            "direct_ocr_attempted": False,
        }]

    def read(*_args, **_kwargs):
        ocr_calls.append(1)
        return ("31-ط-556-74", 0.91, "hezar-crnn-fa-v2-onnx")

    monkeypatch.setattr("app.ai.pipeline.detect_plates", detect)
    monkeypatch.setattr("app.ai.pipeline.read_plate_candidate", read)
    tracker = PlateConsensusTracker(emit_cooldown=10)
    observations = []
    emitted = []
    frame = np.full((300, 500, 3), 120, dtype=np.uint8)

    for index in range(5):
        row = process_frame(
            frame,
            detector_variant="yolo11n",
        )[0]
        observations.append(row)
        current = tracker.update(
            [row],
            timestamp=index * 0.1,
        )
        if index < 4:
            assert current == []
        emitted.extend(current)

    assert [
        row["ocr_crop_height"] for row in observations
    ] == [8, 12, 20, 32, 40]
    assert [
        row["ocr_crop_mature"] for row in observations
    ] == [False, False, True, True, True]
    assert [
        row["whole_plate_ocr_attempted"] for row in observations
    ] == [False, False, True, True, True]
    assert observations[0]["ocr_skip_reason"] == "crop-height"
    assert observations[1]["ocr_skip_reason"] == "crop-height"
    assert observations[0]["plate_norm"] == ""
    assert observations[1]["plate_norm"] == ""
    assert len(ocr_calls) == 3
    assert len(emitted) == 1
    assert emitted[0]["plate_norm"] == "31ط55674"


def test_rejected_guess_below_temporal_floor_is_not_a_visit_key(
    monkeypatch,
):
    crop = np.random.default_rng(9).integers(
        0,
        256,
        (32, 128, 3),
        dtype=np.uint8,
    )
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 138, 52),
            "confidence": 0.90,
            "method": "yolo11n",
            "direct_text": "",
            "direct_ocr_confidence": 0.0,
            "direct_ocr_attempted": False,
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: {
            "plate": "",
            "plate_norm": "",
            "confidence": 0.0,
            "engine": "none",
            "hypotheses": [{
                "plate_norm": "31ط55874",
                "confidence": 0.20,
                "score": 0.20,
                "engine": "hezar-crnn-fa-v2-onnx",
                "temporal_evidence": False,
            }],
        },
    )

    row = process_frame(
        np.full((100, 220, 3), 100, dtype=np.uint8),
        detector_variant="yolo11n",
    )[0]

    assert row["raw_guess_norm"] == "31ط55874"
    assert row["raw_guess_reason"] == "strict-decoder-rejected"
    assert row["visit_identity_stable"] is False
    assert row["hypotheses_accepted_for_consensus"] is False


def test_equal_character_hypotheses_remain_unreadable():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    for index in range(4):
        row = result(
            "31-ط-556-74",
            0.80,
            bbox=(100 + index * 2, 100, 300 + index * 2, 150),
        )
        row["plate_hypotheses"] = [
            {"plate_norm": "31ط55674", "score": 0.80},
            {"plate_norm": "31ل55674", "score": 0.80},
        ]
        assert tracker.update([row], timestamp=index * 0.1) == []

    assert tracker.flush() == []


def test_incomplete_character_frames_combine_by_position():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    correct = "31ط55674"
    emitted = []
    for index, missing_position in enumerate((0, 3, 6, 7)):
        row = result(
            "ناخوانا",
            0.62,
            bbox=(100 + index * 2, 100, 300 + index * 2, 150),
        )
        row.update({
            "plate_norm": "",
            "valid": False,
            "position_hypotheses": [{
                "positions": {
                    position: {
                        "character": character,
                        "confidence": 0.80,
                    }
                    for position, character in enumerate(correct)
                    if position != missing_position
                },
                "coverage": 7,
                "score": 0.80,
            }],
        })
        emitted.extend(
            tracker.update([row], timestamp=index * 0.1)
        )

    assert emitted == []
    assert tracker.flush() == []


def test_positional_consensus_never_confirms_unseen_hybrid():
    tracker = PlateConsensusTracker(emit_cooldown=10)
    emitted = []
    for index, plate in enumerate((
        "12-ب-103-45",
        "12-ب-103-45",
        "12-ب-013-45",
        "02-ب-113-45",
        "02-ب-113-45",
    )):
        emitted.extend(
            tracker.update(
                [result(plate, 0.90)],
                timestamp=index * 0.1,
            )
        )

    assert emitted == []
    assert tracker.flush() == []


def test_confirmation_span_is_enforced():
    tracker = PlateConsensusTracker(
        min_confirmation_span_seconds=0.12,
    )
    plate = result("31-ط-556-74", 0.94)

    assert tracker.update([plate], timestamp=0.00) == []
    assert tracker.update([plate], timestamp=0.03) == []
    assert tracker.update([plate], timestamp=0.06) == []
    emitted = tracker.update([plate], timestamp=0.13)

    assert len(emitted) == 1
    assert emitted[0]["consensus_span_seconds"] >= 0.12


def test_sequential_same_lane_vehicles_get_distinct_tracks():
    tracker = PlateConsensusTracker(
        max_age_seconds=3.0,
        min_confirmation_span_seconds=0.10,
    )
    first_rows = []
    for timestamp in (0.0, 0.1, 0.2):
        first_rows.extend(
            tracker.update(
                [result("12-ب-345-67", 0.95)],
                timestamp=timestamp,
            )
        )
    second_rows = []
    for timestamp in (1.0, 1.1, 1.2):
        second_rows.extend(
            tracker.update(
                [result("98-م-765-43", 0.95)],
                timestamp=timestamp,
            )
        )

    assert len(first_rows) == 1
    assert len(second_rows) == 1
    assert first_rows[0]["track_id"] != second_rows[0]["track_id"]
    assert first_rows[0]["plate_norm"] == "12ب34567"
    assert second_rows[0]["plate_norm"] == "98م76543"


def test_similar_plate_after_emission_gets_a_new_track():
    tracker = PlateConsensusTracker(
        max_age_seconds=3.0,
        min_confirmation_span_seconds=0.10,
    )
    first_rows = []
    for timestamp in (0.0, 0.1, 0.2):
        first_rows.extend(
            tracker.update(
                [result("12-ب-345-67", 0.95)],
                timestamp=timestamp,
            )
        )
    second_rows = []
    for timestamp in (0.4, 0.5, 0.6):
        second_rows.extend(
            tracker.update(
                [result("12-ب-345-68", 0.95)],
                timestamp=timestamp,
            )
        )

    assert len(first_rows) == 1
    assert len(second_rows) == 1
    assert first_rows[0]["track_id"] != second_rows[0]["track_id"]
    assert second_rows[0]["plate_norm"] == "12ب34568"


def test_confirmed_track_refreshes_clearer_same_identity_capture():
    tracker = PlateConsensusTracker(
        min_confirmation_span_seconds=0.10,
    )
    dim_frame = np.full((120, 220, 3), 40, dtype=np.uint8)
    clear_frame = np.full((120, 220, 3), 220, dtype=np.uint8)
    emitted = []
    for timestamp in (0.0, 0.1, 0.2):
        row = result("12-ب-345-67", 0.95)
        row["quality_score"] = 0.30
        emitted.extend(
            tracker.update(
                [row],
                timestamp=timestamp,
                frame=dim_frame,
            )
        )
    for timestamp in (0.4, 0.5, 0.6):
        row = result("12-ب-345-67", 0.95)
        row["quality_score"] = 0.95
        emitted.extend(
            tracker.update(
                [row],
                timestamp=timestamp,
                frame=clear_frame,
            )
        )

    confirmed = [
        row for row in emitted
        if not row.get("capture_refresh")
    ]
    refreshed = [
        row for row in emitted
        if row.get("capture_refresh")
    ]
    assert len(confirmed) == 1
    assert len(refreshed) == 1
    assert confirmed[0]["track_id"] == refreshed[0]["track_id"]
    assert refreshed[0]["plate_norm"] == "12ب34567"
    assert float(refreshed[0]["capture_frame"].mean()) == 220.0
    assert tracker.flush() == []


def test_flush_does_not_duplicate_already_emitted_consensus():
    tracker = PlateConsensusTracker()
    emitted = []
    for timestamp in (0.0, 0.1, 0.2):
        emitted.extend(
            tracker.update(
                [result("31-ط-556-74", 0.94)],
                timestamp=timestamp,
            )
        )

    assert len(emitted) == 1
    assert tracker.flush() == []


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


def test_complete_low_confidence_hypothesis_is_exposed_for_review(
    monkeypatch,
):
    crop = np.full((44, 170, 3), 145, dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 180, 64),
            "confidence": 0.72,
            "method": "yolo-plate+chars",
            "direct_text": "",
            "direct_ocr_confidence": 0.0,
            "direct_ocr_attempted": True,
            "plate_hypotheses": [{
                "plate_norm": "31ط55674",
                "confidence": 0.38,
                "score": 0.41,
            }],
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: ("", 0.0, "none"),
    )

    rows = process_frame(
        np.full((100, 220, 3), 100, dtype=np.uint8)
    )

    assert rows[0]["plate"] == "31-ط-556-74"
    assert rows[0]["valid"] is False
    assert rows[0]["best_effort"] is True
    assert rows[0]["needs_review"] is True
    assert rows[0]["raw_guess_norm"] == "31ط55674"
    assert rows[0]["read_status"] == "experimental-guess"


def test_position_only_ambiguity_stays_unconfirmed_for_review():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    base = result(
        "ناخوانا",
        0.42,
        bbox=(20, 30, 170, 70),
        quality=0.7,
    )
    base.update({
        "valid": False,
        "plate_norm": "",
        "detector_confidence": 0.84,
        "position_hypotheses": [
            {
                "positions": {
                    index: {"character": character}
                    for index, character in enumerate("31ط55674")
                },
                "score": 0.5,
            },
            {
                "positions": {
                    index: {"character": character}
                    for index, character in enumerate("31ط55874")
                },
                "score": 0.5,
            },
        ],
    })

    emitted = []
    for index in range(5):
        emitted.extend(
            tracker.update(
                [base],
                timestamp=index * 0.25,
                frame=frame,
            )
        )

    final = [
        row for row in emitted
        if row.get("best_effort") and not row.get("capture_only")
    ]
    assert len(final) == 1
    assert final[0]["valid"] is False
    assert final[0]["needs_review"] is True
    assert final[0]["plate"] != "ناخوانا"
    assert final[0]["raw_guess_norm"]
    assert final[0]["plate_norm"] == ""
    assert final[0]["auto_confirmed"] is False
    assert final[0]["read_status"] == "experimental-guess"
    assert final[0]["auto_confirmation_blocked"] == (
        "insufficient-independent-frame-evidence"
    )
    assert final[0]["experimental"] is True


def test_one_complete_guess_plus_unreadable_frames_never_auto_confirms():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    guess = result(
        "31-ط-556-74",
        0.45,
        bbox=(20, 30, 170, 70),
        quality=0.72,
    )
    guess.update({
        "valid": False,
        "plate_norm": "",
        "needs_review": True,
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "plate_hypotheses": [{
            "plate_norm": "31ط55674",
            "score": 0.62,
        }],
        "hypotheses_accepted_for_consensus": False,
    })
    unreadable = {
        **guess,
        "plate": "ناخوانا",
        "raw_guess_text": "",
        "raw_guess_norm": "",
        "plate_hypotheses": [],
    }

    emitted = tracker.update(
        [guess],
        timestamp=0.0,
        frame=frame,
    )
    for index in range(1, 6):
        emitted.extend(
            tracker.update(
                [unreadable],
                timestamp=index * 0.25,
                frame=frame,
            )
        )

    assert not any(row.get("auto_confirmed") for row in emitted)
    suggestion = [
        row
        for row in emitted
        if row.get("raw_guess_norm") == "31ط55674"
        and not row.get("capture_only")
    ]
    assert len(suggestion) == 1
    assert suggestion[0]["valid"] is False
    assert suggestion[0]["guess_supporting_frames"] == 1


def test_rejected_hypotheses_never_become_strict_consensus():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    rejected = result(
        "31-ط-556-74",
        0.42,
        bbox=(20, 30, 170, 70),
        quality=0.7,
    )
    rejected.update({
        "valid": False,
        "needs_review": True,
        "best_effort": True,
        "plate_norm": "",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "hypotheses_accepted_for_consensus": False,
        "plate_hypotheses": [{
            "plate_norm": "31ط55674",
            "confidence": 0.95,
            "score": 0.95,
        }],
    })

    emitted = []
    for index in range(6):
        emitted.extend(
            tracker.update(
                [rejected],
                timestamp=index * 0.25,
                frame=frame,
            )
        )

    assert not any(
        row.get("read_status") == "confirmed-ai"
        for row in emitted
    )
    auto_confirmed = [
        row for row in emitted
        if row.get("read_status") == "auto-confirmed"
    ]
    assert len(auto_confirmed) == 1
    assert auto_confirmed[0]["plate_norm"] == "31ط55674"
    assert auto_confirmed[0]["needs_review"] is True
    assert auto_confirmed[0]["confirmation_source"] == "ai-auto-guess"


def test_explicitly_rejected_hezar_evidence_never_auto_confirms():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    rejected = result(
        "31-ط-556-74",
        0.20,
        bbox=(20, 30, 170, 70),
        quality=0.7,
    )
    rejected.update({
        "valid": False,
        "needs_review": True,
        "best_effort": True,
        "plate_norm": "",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "visit_identity_stable": False,
        "hypotheses_accepted_for_consensus": True,
        "plate_hypotheses": [{
            "plate_norm": "31ط55674",
            "confidence": 0.20,
            "score": 0.20,
            "engine": "hezar-crnn-fa-v2-onnx",
            "temporal_evidence": False,
        }],
    })

    emitted = []
    for index in range(5):
        emitted.extend(tracker.update(
            [rejected],
            timestamp=index * 0.25,
            frame=frame,
            min_emit_confidence=0.85,
        ))

    suggestions = [
        row
        for row in emitted
        if not row.get("capture_only")
    ]
    assert len(suggestions) == 1
    assert suggestions[0]["valid"] is False
    assert suggestions[0]["auto_confirmed"] is False
    assert suggestions[0]["plate_norm"] == ""
    assert suggestions[0]["raw_guess_norm"] == "31ط55674"
    assert suggestions[0]["read_status"] == "experimental-guess"
    assert suggestions[0]["tracker_finalized"] is True


def test_partial_character_evidence_is_kept_instead_of_unreadable():
    tracker = PlateConsensusTracker(
        emit_unreadable=True,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    )
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    base = result(
        "ناخوانا",
        0.38,
        bbox=(20, 30, 170, 70),
        quality=0.7,
    )
    base.update({
        "valid": False,
        "plate_norm": "",
        "detector_confidence": 0.84,
        "position_hypotheses": [{
            "positions": {
                index: {"character": character}
                for index, character in enumerate("31ط556")
            },
            "score": 0.48,
        }],
    })

    emitted = []
    for index in range(5):
        emitted.extend(
            tracker.update(
                [base],
                timestamp=index * 0.25,
                frame=frame,
            )
        )

    partial = [row for row in emitted if row.get("partial_final")]
    assert len(partial) == 1
    assert partial[0]["plate"].startswith("31-ط-556")
    assert "؟" in partial[0]["plate"]
    assert partial[0]["needs_review"] is True


def test_failed_crnn_uses_character_cnn_on_good_crop(
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
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            "55-ط-639-74",
            0.81,
            "cnn-onnx",
        ),
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


def test_below_emit_gate_waits_for_clearer_consensus():
    tracker = PlateConsensusTracker(emit_cooldown=5)
    review_rows = []
    for index in range(3):
        low = result("31-ط-556-74", 0.55)
        low.update({
            "ocr_confidence": 0.45,
            "detector_confidence": 0.55,
        })
        review_rows.extend(tracker.update(
            [low],
            timestamp=index * 0.1,
            min_emit_confidence=0.85,
        ))

    track = next(iter(tracker._tracks.values()))
    assert track.emitted_plate == ""
    assert len(review_rows) == 1
    assert review_rows[0]["below_emit_confidence"] is True

    confirmed = []
    for index in range(3, 6):
        clear = result("31-ط-556-74", 0.98, quality=0.95)
        clear.update({
            "ocr_confidence": 0.96,
            "detector_confidence": 0.98,
        })
        confirmed.extend(
            row
            for row in tracker.update(
                [clear],
                timestamp=index * 0.1,
                min_emit_confidence=0.85,
            )
            if not row.get("below_emit_confidence")
        )

    assert len(confirmed) == 1
    assert confirmed[0]["plate_norm"] == "31ط55674"
    assert confirmed[0]["confidence"] >= 0.85
    assert track.emitted_plate == "31ط55674"


def test_expired_consensus_preserves_below_emit_gate_marker():
    tracker = PlateConsensusTracker(
        max_age_seconds=0.2,
        emit_cooldown=5,
    )
    for index in range(3):
        low = result("31-ط-556-74", 0.55)
        low.update({
            "ocr_confidence": 0.45,
            "detector_confidence": 0.55,
        })
        tracker.update(
            [low],
            timestamp=index * 0.1,
            min_emit_confidence=0.85,
        )

    expired = tracker.update(
        [],
        timestamp=1.0,
        min_emit_confidence=0.85,
    )

    assert len(expired) == 1
    assert expired[0]["valid"] is True
    assert expired[0]["confidence"] < 0.85
    assert expired[0]["below_emit_confidence"] is True


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


def test_ocr_crop_maturity_boundaries_are_closed():
    quality = {"score": 0.50}

    assert _ocr_crop_maturity(
        np.zeros((18, 64, 3), dtype=np.uint8),
        0.25,
        {"score": 0.20},
    )["mature"] is True
    assert _ocr_crop_maturity(
        np.zeros((17, 68, 3), dtype=np.uint8),
        0.90,
        quality,
    )["reason"] == "crop-height"
    assert _ocr_crop_maturity(
        np.zeros((18, 63, 3), dtype=np.uint8),
        0.90,
        quality,
    )["reason"] == "crop-width"
    assert _ocr_crop_maturity(
        np.zeros((100, 179, 3), dtype=np.uint8),
        0.90,
        quality,
    )["reason"] == "crop-aspect"
    assert _ocr_crop_maturity(
        np.zeros((100, 180, 3), dtype=np.uint8),
        0.90,
        quality,
    )["mature"] is True
    assert _ocr_crop_maturity(
        np.zeros((20, 170, 3), dtype=np.uint8),
        0.90,
        quality,
    )["mature"] is True
    assert _ocr_crop_maturity(
        np.zeros((100, 851, 3), dtype=np.uint8),
        0.90,
        quality,
    )["reason"] == "crop-aspect"
    assert _ocr_crop_maturity(
        np.zeros((20, 80, 3), dtype=np.uint8),
        0.249,
        quality,
    )["reason"] == "detector-confidence"
    assert _ocr_crop_maturity(
        np.zeros((20, 80, 3), dtype=np.uint8),
        0.25,
        {"score": 0.199},
    )["reason"] == "crop-quality"


def test_plate_similarity_handles_formatting():
    assert plate_similarity(
        "31-ط-556-74",
        "31ط55674",
    ) == 1.0
    assert plate_similarity("", "31ط55674") == 0.0


def test_direct_yolo_text_is_compared_with_crnn_without_vehicle_ai(
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
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            "27-ط-253-74",
            0.91,
            "crnn-onnx",
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
    assert rows[0]["whole_plate_ocr_attempted"] is True
    assert rows[0]["ocr_engine"].startswith(
        "multi-engine-agreement"
    )
    assert rows[0]["vehicle_type"] == "نامشخص"


def test_stronger_crnn_disagreement_is_reviewable_and_keeps_both_reads(
    monkeypatch,
):
    crop = np.full((42, 168, 3), 170, dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 178, 62),
            "confidence": 0.86,
            "method": "yolo-plate+chars",
            "direct_text": "31-ط-556-74",
            "direct_ocr_confidence": 0.64,
            "direct_ocr_attempted": True,
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            "31-ط-558-74",
            0.88,
            "crnn-onnx",
        ),
    )

    rows = process_frame(
        np.full((100, 220, 3), 100, dtype=np.uint8),
        engine_key=9,
    )

    row = rows[0]
    assert row["plate"] == "31-ط-558-74"
    assert row["ocr_engine"] == "crnn-onnx"
    assert row["ocr_alternative"] == "31-ط-556-74"
    assert row["ocr_disagreement"] is True
    assert row["needs_review"] is True
    assert {
        hypothesis["plate_norm"]
        for hypothesis in row["plate_hypotheses"]
    } == {"31ط55674", "31ط55874"}


def test_consensus_records_support_from_both_dedicated_readers():
    tracker = PlateConsensusTracker(min_votes=3)
    observation = result("31-ط-556-74", 0.90)
    observation.update({
        "ocr_engine": "multi-engine-agreement:crnn-onnx",
        "ocr_disagreement": False,
        "plate_hypotheses": [
            {
                "plate_norm": "31ط55674",
                "confidence": 0.88,
                "score": 0.88,
                "engine": "dedicated-character-detector+crnn-onnx",
            }
        ],
    })

    tracker.update([observation], timestamp=0.0)
    tracker.update([observation], timestamp=0.2)
    emitted = tracker.update([observation], timestamp=0.4)

    assert len(emitted) == 1
    assert emitted[0]["plate_norm"] == "31ط55674"
    assert emitted[0]["ocr_engine"] == "multi-engine-consensus"
    assert emitted[0]["ocr_disagreement"] is False


def test_bytetrack_second_pass_keeps_low_confidence_detection_identity():
    tracker = PlateConsensusTracker(min_votes=3)
    first = {
        "bbox": (20, 20, 120, 55),
        "confidence": 0.82,
        "detector_confidence": 0.82,
        "quality_score": 0.60,
        "valid": False,
        "plate": "",
        "plate_norm": "",
    }
    tracker.update([first], timestamp=1.0)
    original_track = first["track_id"]

    low_confidence = {
        **first,
        "bbox": (27, 21, 127, 56),
        "confidence": 0.24,
        "detector_confidence": 0.24,
    }
    low_confidence.pop("track_id", None)
    low_confidence.pop("tracking_bbox", None)
    low_confidence.pop("tracking_engine", None)
    tracker.update([low_confidence], timestamp=1.1)

    assert low_confidence["track_id"] == original_track
    assert low_confidence["tracking_engine"] == (
        "bytetrack-kalman+optical-flow"
    )
    assert len(low_confidence["tracking_bbox"]) == 4


def test_global_motion_assignment_preserves_two_crossing_vehicles():
    tracker = PlateConsensusTracker(min_votes=3)

    def detection(bbox):
        return {
            "bbox": bbox,
            "confidence": 0.85,
            "detector_confidence": 0.85,
            "quality_score": 0.70,
            "valid": False,
            "plate": "",
            "plate_norm": "",
        }

    first_left = detection((10, 30, 70, 55))
    first_right = detection((170, 48, 230, 73))
    tracker.update(
        [first_left, first_right],
        timestamp=0.0,
    )
    left_track = first_left["track_id"]
    right_track = first_right["track_id"]

    second_left = detection((50, 30, 110, 55))
    second_right = detection((130, 48, 190, 73))
    tracker.update(
        [second_right, second_left],
        timestamp=0.1,
    )

    crossed_left_to_right = detection((95, 30, 155, 55))
    crossed_right_to_left = detection((85, 48, 145, 73))
    tracker.update(
        [crossed_right_to_left, crossed_left_to_right],
        timestamp=0.2,
    )

    assert crossed_left_to_right["track_id"] == left_track
    assert crossed_right_to_left["track_id"] == right_track
