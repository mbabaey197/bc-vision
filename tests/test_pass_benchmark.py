from dataclasses import replace
import hashlib

import pytest

from app.ai.pass_benchmark import (
    REQUIRED_PASS_SLICES,
    VERIFIED_PRODUCTION_PASS,
    evaluate_accuracy_claim,
    score_passages,
    wilson_interval,
)


_CONDITION_SLICES = REQUIRED_PASS_SLICES[2:]


def _plate(index):
    letters = "بپتثجدزژسشصطعفقکلمنوهی"
    return (
        f"{10 + index % 90:02d}"
        f"{letters[index % len(letters)]}"
        f"{100 + index % 900:03d}"
        f"{10 + (index // 90) % 90:02d}"
    )


def _evidence(
    readable=400,
    negatives=800,
    *,
    evidence_kind=VERIFIED_PRODUCTION_PASS,
):
    rows = []
    total = readable + negatives
    for index in range(total):
        expected = _plate(index % 100) if index < readable else ""
        illumination = "day" if index % 2 == 0 else "night"
        rows.append({
            "passage_id": f"passage-{index}",
            "evidence_digest": hashlib.sha256(
                f"immutable-passage-evidence-{index}".encode()
            ).hexdigest(),
            "evidence_kind": evidence_kind,
            "pipeline_revision": "production-revision-a",
            "camera_id": f"camera-{index % 3}",
            "session_id": f"session-{index % 3}",
            "slices": [
                illumination,
                _CONDITION_SLICES[index % len(_CONDITION_SLICES)],
            ],
            "expected_plate": expected,
            "readable": bool(expected),
            "accepted_events": [expected] if expected else [],
        })
    return rows


def _claim(rows):
    return evaluate_accuracy_claim(
        score_passages(rows),
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )


def test_small_perfect_crop_result_cannot_claim_99_percent():
    rows = _evidence(
        readable=40,
        negatives=0,
        evidence_kind="verified-ocr-crop-golden",
    )
    metrics = score_passages(rows)

    decision = evaluate_accuracy_claim(
        metrics,
        evaluation_kind="verified-ocr-crop-golden",
    )

    assert decision["claim_ready"] is False
    assert "production-pass-evidence-required" in decision["reasons"]
    assert "insufficient-readable-passages" in decision["reasons"]
    assert "insufficient-negative-passages" in decision["reasons"]
    assert any(
        reason.startswith("invalid-evidence-kind:")
        for reason in decision["reasons"]
    )


def test_sufficient_perfect_production_pass_evidence_is_claim_ready():
    metrics = score_passages(_evidence())

    decision = evaluate_accuracy_claim(
        metrics,
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )

    assert decision["claim_ready"] is True
    assert decision["reasons"] == []
    assert decision["pipeline_revision"] == "production-revision-a"
    assert decision["exact_accuracy_ci95"][0] >= 0.99
    assert decision["false_accept_ci95"][1] <= 0.005
    assert decision["duplicate_event_ci95"][1] <= 0.005


def test_miss_wrong_false_accept_and_duplicate_use_correct_denominators():
    rows = _evidence()
    rows[0]["accepted_events"] = []
    rows[1]["accepted_events"] = [_plate(2)]
    rows[400]["accepted_events"] = [_plate(3)]
    rows[2]["accepted_events"] = [
        rows[2]["expected_plate"],
        rows[2]["expected_plate"],
    ]

    metrics = score_passages(rows)
    decision = evaluate_accuracy_claim(
        metrics,
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )

    assert metrics.exact_passages == 397
    assert metrics.missed_passages == 1
    assert metrics.wrong_read_passages == 2
    assert metrics.false_accept_passages == 1
    assert metrics.duplicate_event_passages == 1
    assert decision["miss_rate"] == pytest.approx(1 / 400)
    assert decision["wrong_read_rate"] == pytest.approx(2 / 400)
    assert decision["false_accept_rate"] == pytest.approx(1 / 800)
    assert decision["duplicate_event_rate"] == pytest.approx(
        1 / 1200,
        abs=1e-6,
    )
    assert decision["claim_ready"] is False
    assert "exact-accuracy-confidence-bound" in decision["reasons"]
    assert "false-accept-confidence-bound" in decision["reasons"]


def test_duplicate_passage_or_evidence_identity_fails_closed():
    rows = _evidence()
    rows[1]["passage_id"] = rows[0]["passage_id"]
    rows[2]["evidence_digest"] = rows[0]["evidence_digest"]

    decision = _claim(rows)

    assert decision["claim_ready"] is False
    assert any(
        reason.startswith("duplicate-passage-id:")
        for reason in decision["reasons"]
    )
    assert any(
        reason.startswith("duplicate-evidence-digest:")
        for reason in decision["reasons"]
    )


@pytest.mark.parametrize(
    ("field", "value", "reason_prefix"),
    [
        ("slices", "day", "slices-must-be-nonempty-list:"),
        ("slices", ["day", "night"], "invalid-slices:"),
        ("readable", 1, "readable-must-be-boolean:"),
        ("expected_plate", 123, "invalid-expected-label:"),
        ("accepted_events", None, "accepted-events-must-be-list:"),
        ("camera_id", 1, "missing-passage-provenance:"),
        ("pipeline_revision", " ", "missing-pipeline-revision:"),
        ("evidence_digest", "not-a-sha256", "invalid-evidence-digest:"),
    ],
)
def test_malformed_passage_fields_fail_closed(field, value, reason_prefix):
    rows = _evidence()
    rows[0][field] = value

    decision = _claim(rows)

    assert decision["claim_ready"] is False
    assert any(
        reason.startswith(reason_prefix)
        for reason in decision["reasons"]
    )


def test_non_list_dataset_fails_closed_without_raising():
    metrics = score_passages(None)
    decision = evaluate_accuracy_claim(
        metrics,
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )

    assert decision["claim_ready"] is False
    assert "passages-must-be-list" in decision["reasons"]


def test_conflicting_event_fields_and_nonempty_negative_label_are_invalid():
    rows = _evidence()
    rows[0]["accepted_events"] = [{
        "plate_norm": rows[0]["expected_plate"],
        "plate": _plate(1),
    }]
    rows[400]["expected_plate"] = "N/A"

    decision = _claim(rows)

    assert decision["claim_ready"] is False
    assert any(
        reason.startswith("conflicting-accepted-event:")
        for reason in decision["reasons"]
    )
    assert any(
        reason.startswith("negative-must-have-empty-label:")
        for reason in decision["reasons"]
    )


def test_mixed_pipeline_revisions_cannot_be_pooled_for_claim():
    rows = _evidence()
    rows[0]["pipeline_revision"] = "production-revision-b"

    decision = _claim(rows)

    assert decision["claim_ready"] is False
    assert "single-pipeline-revision-required" in decision["reasons"]


def test_slice_and_provenance_coverage_cannot_be_filled_only_by_negatives():
    rows = _evidence()
    for row in rows[:400]:
        row["camera_id"] = "camera-0"
        row["session_id"] = "session-0"
        row["slices"] = ["day", "fast"]

    decision = _claim(rows)

    assert decision["claim_ready"] is False
    assert "insufficient-readable-camera-coverage" in decision["reasons"]
    assert "insufficient-readable-session-coverage" in decision["reasons"]
    assert "insufficient-readable-slice:night" in decision["reasons"]
    assert "insufficient-readable-slice:angle" in decision["reasons"]


def test_one_plate_cannot_dominate_an_otherwise_large_sample():
    rows = _evidence()
    dominant = _plate(0)
    for index, row in enumerate(rows[:301]):
        if index >= 100:
            row["expected_plate"] = dominant
            row["accepted_events"] = [dominant]

    decision = _claim(rows)

    assert decision["unique_plates"] == 100
    assert decision["claim_ready"] is False
    assert "excessive-single-plate-concentration" in decision["reasons"]


def test_tampered_metrics_object_fails_internal_consistency_checks():
    metrics = score_passages(_evidence())
    tampered = replace(metrics, missed_passages=1)

    decision = evaluate_accuracy_claim(
        tampered,
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )

    assert decision["claim_ready"] is False
    assert "inconsistent-readable-outcomes" in decision["reasons"]


@pytest.mark.parametrize(
    ("successes", "trials", "z"),
    [
        (1.5, 2, 1.96),
        (True, 2, 1.96),
        (3, 2, 1.96),
        (0, 0, 1.96),
        (1, 2, float("nan")),
        (1, 2, 0),
    ],
)
def test_wilson_interval_is_maximally_uncertain_for_malformed_input(
    successes,
    trials,
    z,
):
    assert wilson_interval(successes, trials, z=z) == (0.0, 1.0)
