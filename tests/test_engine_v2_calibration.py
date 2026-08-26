from __future__ import annotations

import pytest

from app.engine_v2.calibration import (
    CalibrationDataset,
    CalibrationObservation,
    CalibrationRequirements,
    CalibrationTrack,
    analyze_static_ocr,
    calibrate,
    evaluate_config,
)
from app.engine_v2.tcam import TemporalFusionConfig

PLATE = "12ب34567"
WRONG = "12ب34568"


def _track(
    identifier: str,
    *,
    split: str,
    profile: str,
    expected: str | None,
    read: str,
    confidence: float,
) -> CalibrationTrack:
    return CalibrationTrack(
        track_id=identifier,
        split=split,
        profile=profile,
        expected_plate=expected,
        observations=(
            CalibrationObservation(
                seq=1,
                ts=0.0,
                text=read,
                confidence=confidence,
                quality=0.90,
                plate_width=120,
                plate_height=30,
                character_confidences=(confidence,) * 8,
            ),
        ),
    )


def test_static_trace_calibrates_express_precision_without_faking_temporal_votes() -> (
    None
):
    tracks = (
        _track(
            "p",
            split="train",
            profile="day",
            expected=PLATE,
            read=PLATE,
            confidence=0.96,
        ),
        _track(
            "n",
            split="train",
            profile="day",
            expected=None,
            read=WRONG,
            confidence=0.90,
        ),
    )

    metrics = evaluate_config(tracks, TemporalFusionConfig())

    assert metrics.exact_matches == 1
    assert metrics.false_accepts == 0
    assert metrics.mean_ocr_calls_per_track == 1.0


def test_calibration_selects_on_train_and_validates_day_and_night_holdouts() -> None:
    tracks = []
    for profile in ("day", "night"):
        tracks.extend(
            [
                _track(
                    f"{profile}-train-p",
                    split="train",
                    profile=profile,
                    expected=PLATE,
                    read=PLATE,
                    confidence=0.96,
                ),
                _track(
                    f"{profile}-train-n",
                    split="train",
                    profile=profile,
                    expected=None,
                    read=WRONG,
                    confidence=0.90,
                ),
                _track(
                    f"{profile}-holdout-p",
                    split="holdout",
                    profile=profile,
                    expected=PLATE,
                    read=PLATE,
                    confidence=0.96,
                ),
                _track(
                    f"{profile}-holdout-n",
                    split="holdout",
                    profile=profile,
                    expected=None,
                    read=WRONG,
                    confidence=0.90,
                ),
            ]
        )
    dataset = CalibrationDataset("synthetic", tuple(tracks), "abc")
    requirements = CalibrationRequirements(
        target_exact_accuracy=1.0,
        minimum_event_recall=1.0,
        minimum_event_precision=1.0,
        maximum_false_accept_rate=0.0,
        maximum_wrong_event_rate=0.0,
        maximum_mean_character_error_rate=0.0,
        minimum_train_tracks=2,
        minimum_holdout_tracks=2,
    )
    grids = {
        profile: {"express_lock_confidence": [0.93, 0.97]}
        for profile in ("day", "night")
    }

    report = calibrate(dataset, grids=grids, requirements=requirements)

    assert report.valid is True
    assert all(profile.config is not None for profile in report.profiles)
    assert all(
        profile.config.express_lock_confidence == 0.93
        for profile in report.profiles
        if profile.config is not None
    )


def test_static_ir_lpr_style_positive_only_data_is_useful_but_never_promotable() -> (
    None
):
    dataset = CalibrationDataset(
        "ir-lpr-static",
        (
            _track(
                "train",
                split="train",
                profile="day",
                expected=PLATE,
                read=PLATE,
                confidence=0.96,
            ),
            _track(
                "holdout",
                split="holdout",
                profile="day",
                expected=PLATE,
                read=PLATE,
                confidence=0.96,
            ),
        ),
        "ir",
    )
    requirements = CalibrationRequirements(
        minimum_train_tracks=1,
        minimum_holdout_tracks=1,
    )

    report = calibrate(dataset, requirements=requirements)

    day = next(profile for profile in report.profiles if profile.profile == "day")
    assert day.train_metrics is not None
    assert day.holdout_metrics is not None
    assert report.valid is False
    assert "day:train_missing_negative_tracks" in report.blockers
    assert any(blocker.startswith("night:") for blocker in report.blockers)


def test_static_ocr_report_measures_confidence_without_promoting_policy() -> None:
    dataset = CalibrationDataset(
        "ir-lpr-static",
        (
            _track(
                "correct",
                split="train",
                profile="day",
                expected=PLATE,
                read=PLATE,
                confidence=0.90,
            ),
            _track(
                "wrong",
                split="holdout",
                profile="day",
                expected=PLATE,
                read=WRONG,
                confidence=0.80,
            ),
        ),
        "ir",
    )

    report = analyze_static_ocr(dataset, confidence_thresholds=(0.75, 0.85))

    assert report.promotion_eligible is False
    assert report.overall.exact_accuracy == 0.5
    assert report.overall.mean_character_error_rate == 0.0625
    assert report.overall.brier_score == pytest.approx(0.325)
    at_75, at_85 = report.overall.thresholds
    assert at_75.coverage == 1.0
    assert at_75.selective_exact_accuracy == 0.5
    assert at_85.coverage == 0.5
    assert at_85.selective_exact_accuracy == 1.0
    assert {slice_.scope for slice_ in report.slices} == {
        "day/train",
        "day/holdout",
    }
