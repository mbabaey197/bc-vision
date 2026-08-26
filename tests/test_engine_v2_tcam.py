from __future__ import annotations

import pytest

from app.engine_v2.tcam import (
    PlateEvidenceAccumulator,
    RecognitionPhase,
    TemporalFusionConfig,
    TrackRecognitionSession,
)
from app.engine_v2.types import OCRResult

PLATE = "12ب34567"


def _result(
    text: str,
    confidence: float = 0.86,
    character_confidences: tuple[float, ...] = (),
) -> OCRResult:
    return OCRResult(text, confidence, True, character_confidences)


def test_raw_75_percent_read_is_provisional_not_final() -> None:
    fusion = PlateEvidenceAccumulator()

    decision = fusion.add(_result(PLATE, 0.75), quality=0.80, seq=1)

    assert decision.phase is RecognitionPhase.PROVISIONAL
    assert decision.locked is False
    assert decision.text == PLATE
    assert decision.reason == "awaiting_independent_confirmation"


def test_strict_first_frame_express_path_can_soft_lock() -> None:
    fusion = PlateEvidenceAccumulator()

    decision = fusion.add(_result(PLATE, 0.96), quality=0.85, seq=1)

    assert decision.phase is RecognitionPhase.SOFT_LOCKED
    assert decision.reason == "express_high_confidence"
    assert decision.independent_observations == 1


def test_independent_multi_frame_votes_lock_and_correct_one_character() -> None:
    fusion = PlateEvidenceAccumulator()
    fusion.add(_result("12ب34568", 0.82), quality=0.70, seq=1)
    fusion.add(_result(PLATE, 0.88), quality=0.78, seq=3)
    decision = fusion.add(_result(PLATE, 0.90), quality=0.82, seq=5)

    assert decision.phase is RecognitionPhase.SOFT_LOCKED
    assert decision.text == PLATE
    assert decision.reason == "independent_temporal_consensus"
    assert decision.slots[-1].character == "7"
    assert decision.slots[-1].support == 2
    assert decision.full_sequence_support == 2


def test_partial_slots_remain_open_and_are_filled_by_later_frames() -> None:
    fusion = PlateEvidenceAccumulator()
    partial = fusion.add(_result("12ب34?67", 0.86), quality=0.75, seq=1)

    assert partial.phase is RecognitionPhase.READING
    assert partial.unresolved_slots == (5,)

    fusion.add(_result(PLATE, 0.88), quality=0.78, seq=3)
    complete = fusion.add(_result(PLATE, 0.90), quality=0.82, seq=5)
    assert complete.phase is RecognitionPhase.SOFT_LOCKED
    assert complete.text == PLATE


def test_correlated_adjacent_frames_do_not_fake_independent_consensus() -> None:
    fusion = PlateEvidenceAccumulator()
    first = fusion.add(_result(PLATE, 0.87), quality=0.80, seq=10)
    second = fusion.add(_result(PLATE, 0.87), quality=0.80, seq=11)

    assert first.phase is RecognitionPhase.PROVISIONAL
    assert second.phase is RecognitionPhase.PROVISIONAL
    assert second.independent_observations == 1
    assert second.full_sequence_support == 1

    third = fusion.add(_result(PLATE, 0.87), quality=0.80, seq=12)
    assert third.phase is RecognitionPhase.SOFT_LOCKED
    assert third.independent_observations == 2


def test_scheduler_runs_first_crop_then_only_on_material_reason() -> None:
    session = TrackRecognitionSession()

    first = session.should_schedule_ocr(seq=1, quality=0.60, bbox_area=1_000)
    assert (first.run_ocr, first.reason) == (True, "first_usable_crop")
    session.reserve_ocr(seq=1, quality=0.60, bbox_area=1_000)
    provisional = session.observe(_result(PLATE, 0.80), quality=0.60, seq=1)
    assert provisional.phase is RecognitionPhase.PROVISIONAL

    unchanged = session.should_schedule_ocr(seq=2, quality=0.62, bbox_area=1_050)
    assert (unchanged.run_ocr, unchanged.reason) == (False, "no_material_gain")
    confirmation = session.should_schedule_ocr(seq=3, quality=0.62, bbox_area=1_050)
    assert (confirmation.run_ocr, confirmation.reason) == (
        True,
        "provisional_confirmation_due",
    )


def test_scheduler_rereads_on_area_growth_then_allows_one_soft_lock_audit() -> None:
    session = TrackRecognitionSession()
    session.reserve_ocr(seq=1, quality=0.70, bbox_area=1_000)
    session.observe(_result(PLATE, 0.88), quality=0.70, seq=1)

    growth = session.should_schedule_ocr(seq=2, quality=0.70, bbox_area=1_200)
    assert (growth.run_ocr, growth.reason) == (True, "plate_area_grew")
    session.reserve_ocr(seq=3, quality=0.75, bbox_area=1_200)
    decision = session.observe(_result(PLATE, 0.90), quality=0.75, seq=3)
    assert decision.locked is True
    audit = session.should_schedule_ocr(seq=10, quality=1.0, bbox_area=4_000)
    assert (audit.run_ocr, audit.reason) == (True, "soft_lock_quality_audit")
    session.reserve_ocr(seq=10, quality=1.0, bbox_area=4_000)
    session.observe(_result(PLATE, 0.96), quality=1.0, seq=10)
    assert (
        session.should_schedule_ocr(seq=11, quality=1.0, bbox_area=5_000).reason
        == "soft_lock_audit_complete"
    )


def test_one_finalized_track_can_claim_only_one_event() -> None:
    session = TrackRecognitionSession()
    session.reserve_ocr(seq=1, quality=0.90, bbox_area=1_000)
    assert session.observe(_result(PLATE, 0.97), quality=0.90, seq=1).locked

    assert session.claim_event() is False
    decision = session.finalize(reason="unit_test")
    assert decision.phase is RecognitionPhase.FINALIZED
    assert session.claim_event() is True
    assert session.claim_event() is False


def test_real_timestamps_not_frame_numbers_define_independent_evidence() -> None:
    fusion = PlateEvidenceAccumulator()
    fusion.add(_result(PLATE, 0.87), quality=0.80, seq=1, ts=10.00)
    correlated = fusion.add(_result(PLATE, 0.87), quality=0.80, seq=100, ts=10.04)
    independent = fusion.add(_result(PLATE, 0.87), quality=0.80, seq=101, ts=10.08)

    assert correlated.phase is RecognitionPhase.PROVISIONAL
    assert correlated.independent_observations == 1
    assert independent.phase is RecognitionPhase.SOFT_LOCKED
    assert independent.independent_observations == 2


def test_small_plate_cannot_enter_ocr_or_express_lock() -> None:
    session = TrackRecognitionSession()
    schedule = session.should_schedule_ocr(
        seq=1,
        ts=1.0,
        quality=0.95,
        bbox_area=1_200,
        plate_width=60,
        plate_height=20,
    )
    assert (schedule.run_ocr, schedule.reason) == (
        False,
        "plate_width_below_floor",
    )

    fusion = PlateEvidenceAccumulator()
    decision = fusion.add(
        _result(PLATE, 0.98),
        quality=0.95,
        seq=1,
        ts=1.0,
        plate_width=90,
        plate_height=22,
    )
    assert decision.phase is RecognitionPhase.PROVISIONAL


def test_audit_can_reopen_a_wrong_express_soft_lock() -> None:
    session = TrackRecognitionSession()
    wrong = "12ب34568"
    session.reserve_ocr(seq=1, ts=1.0, quality=0.90, bbox_area=4_000)
    first = session.observe(_result(wrong, 0.97), quality=0.90, seq=1, ts=1.0)
    assert first.phase is RecognitionPhase.SOFT_LOCKED

    assert session.should_schedule_ocr(
        seq=2,
        ts=1.1,
        quality=0.98,
        bbox_area=5_000,
    ).run_ocr
    session.reserve_ocr(seq=2, ts=1.1, quality=0.98, bbox_area=5_000)
    audited = session.observe(_result(PLATE, 0.99), quality=0.98, seq=2, ts=1.1)

    assert audited.phase is not RecognitionPhase.FINALIZED
    assert session.claim_event() is False


def test_top_k_candidate_can_win_after_later_independent_evidence() -> None:
    wrong = "12ب34568"
    first = OCRResult(
        wrong,
        0.91,
        True,
        (0.95,) * 8,
        metadata={
            "candidates": [
                {
                    "text": wrong,
                    "confidence": 0.91,
                    "weight": 0.55,
                    "character_confidences": (0.95,) * 8,
                },
                {
                    "text": PLATE,
                    "confidence": 0.89,
                    "weight": 0.45,
                    "character_confidences": (0.93,) * 8,
                },
            ]
        },
    )
    fusion = PlateEvidenceAccumulator()
    initial = fusion.add(first, quality=0.85, seq=1, ts=1.0)
    confirmed = fusion.add(
        _result(PLATE, 0.90, (0.92,) * 8),
        quality=0.85,
        seq=2,
        ts=1.1,
    )

    assert initial.text == wrong
    assert confirmed.phase is RecognitionPhase.SOFT_LOCKED
    assert confirmed.text == PLATE


def test_temporal_fusion_configuration_rejects_unsafe_threshold_order() -> None:
    with pytest.raises(ValueError, match="provisional_confidence"):
        TemporalFusionConfig(provisional_confidence=0.90, lock_confidence=0.85)
    with pytest.raises(ValueError, match="max_ocr_attempts"):
        TemporalFusionConfig(max_ocr_attempts=0)
