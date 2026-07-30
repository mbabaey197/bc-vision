from app.ai.benchmark import (
    evaluate_promotion,
    score_predictions,
    validate_golden_decision_evidence,
)


def test_candidate_is_promoted_only_after_golden_accuracy_gain():
    baseline = score_predictions([
        {"expected_plate": "31ط55674", "predicted_plate": "31ط55674"},
        {"expected_plate": "84ب57133", "predicted_plate": "84ب57132"},
        {"expected_plate": "12الف34567", "predicted_plate": "12الف34567"},
        {"expected_plate": "22د22222", "predicted_plate": "22د22222"},
        {"expected_plate": "", "predicted_plate": ""},
    ])
    candidate = score_predictions([
        {"expected_plate": "31ط55674", "predicted_plate": "31ط55674"},
        {"expected_plate": "84ب57133", "predicted_plate": "84ب57133"},
        {"expected_plate": "12الف34567", "predicted_plate": "12الف34567"},
        {"expected_plate": "22د22222", "predicted_plate": "22د22222"},
        {"expected_plate": "", "predicted_plate": ""},
    ])

    decision = evaluate_promotion(baseline, candidate)

    assert decision["promote"] is True
    assert decision["candidate_exact_accuracy"] == 1.0


def test_false_accept_regression_blocks_candidate():
    baseline = score_predictions([
        {"expected_plate": "31ط55674", "predicted_plate": "31ط55674"},
        {"expected_plate": "", "predicted_plate": ""},
    ])
    candidate = score_predictions([
        {"expected_plate": "31ط55674", "predicted_plate": "31ط55674"},
        {"expected_plate": "", "predicted_plate": "84ب57133"},
    ])

    decision = evaluate_promotion(
        baseline,
        candidate,
        minimum_exact_gain=0.0,
    )

    assert decision["promote"] is False
    assert "false-accept-regression" in decision["reasons"]


def test_small_relative_gain_cannot_bypass_absolute_accuracy_floor():
    baseline_rows = []
    candidate_rows = []
    for index in range(40):
        expected = f"{10 + index % 20:02d}ب{100 + index:03d}22"
        baseline_rows.append({
            "expected_plate": expected,
            "predicted_plate": "99ی99999",
            "slices": ["day"],
        })
        candidate_rows.append({
            "expected_plate": expected,
            "predicted_plate": (
                expected if index == 0 else "98ی99999"
            ),
            "slices": ["day"],
        })

    decision = evaluate_promotion(
        score_predictions(baseline_rows),
        score_predictions(candidate_rows),
    )

    assert decision["promote"] is False
    assert decision["candidate_exact_accuracy"] == 0.025
    assert "candidate-accuracy-floor" in decision["reasons"]
    assert "candidate-slice-floor:day" in decision["reasons"]


def test_non_finite_stored_golden_metrics_are_rejected():
    reasons = validate_golden_decision_evidence(
        {
            "promote": True,
            "reasons": [],
            "baseline_exact_accuracy": 0.90,
            "candidate_exact_accuracy": float("nan"),
            "baseline_false_accept_rate": 0.0,
            "candidate_false_accept_rate": 0.0,
            "baseline_mean_character_error": 0.10,
            "candidate_mean_character_error": 0.05,
            "evaluation_kind": "verified-ocr-crop-golden",
            "golden_manifest_sha256": "A" * 64,
            "samples": 40,
        },
        {
            "manifest_sha256": "A" * 64,
            "samples": 40,
        },
    )

    assert "golden-metrics-invalid" in reasons
