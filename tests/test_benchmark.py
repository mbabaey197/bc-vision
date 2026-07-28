from app.ai.benchmark import evaluate_promotion, score_predictions


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
