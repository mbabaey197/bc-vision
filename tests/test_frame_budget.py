from app.ai.frame_budget import calculate_frame_budget


def test_150_kmh_capture_rate_is_derived_from_readable_zone():
    assert calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=10,
    ).recommended_capture_fps == 25
    assert calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
    ).recommended_capture_fps == 30
    assert calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=5,
    ).recommended_capture_fps == 50


def test_eight_fps_source_is_not_claimed_sufficient_for_150_kmh():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=10,
        source_fps=8,
    )

    assert budget.zone_seconds == 0.24
    assert budget.expected_raw_frames == 1.92
    assert budget.source_sufficient is False
    assert budget.sufficient is False
    assert "source-fps-insufficient" in budget.warning()


def test_detector_latency_capacity_is_checked_separately_from_source():
    slow = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
        source_fps=30,
        processing_p95_ms=80,
    )
    fast = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
        source_fps=30,
        processing_p95_ms=30,
    )

    assert slow.source_sufficient is True
    assert slow.processing_sufficient is False
    assert slow.expected_processed_raw_frames == 2.4
    assert slow.expected_processed_observations == 1.344
    assert "processing-capacity-insufficient" in slow.warning()
    assert fast.processing_sufficient is True
    assert fast.sufficient is True


def test_unknown_runtime_telemetry_does_not_raise_false_alarm():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=10,
        telemetry_required=True,
    )

    assert budget.source_fps == 0.0
    assert budget.processing_p95_ms == 0.0
    assert budget.source_verified is False
    assert budget.processing_verified is False
    assert budget.sufficient is False
    assert "source-fps-warming-up" in budget.warning()


def test_uncalibrated_geometry_is_never_reported_safe():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=10,
        source_fps=30,
        processing_p95_ms=30,
        geometry_calibrated=False,
        telemetry_required=True,
    )

    assert budget.sufficient is False
    assert "recognition-zone-uncalibrated" in budget.warning()


def test_processing_budget_applies_quality_and_safety_allowance():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
        source_fps=30,
        processing_p95_ms=50,
    )

    assert budget.expected_processed_raw_frames == 3.84
    assert budget.expected_processed_observations == 2.15
    assert budget.processing_sufficient is False


def test_long_source_gap_rejects_a_safe_average_fps():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
        source_fps=30,
        source_max_gap_ms=200,
        source_p95_gap_ms=33,
        processing_p95_ms=30,
        telemetry_required=True,
    )

    assert budget.source_sufficient is True
    assert budget.cadence_sufficient is False
    assert budget.sufficient is False
    assert "source-cadence-insufficient" in budget.warning()


def test_frequent_jitter_gap_must_fit_the_crop_budget():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=8,
        source_fps=30,
        source_max_gap_ms=60,
        source_p95_gap_ms=50,
        processing_p95_ms=30,
        telemetry_required=True,
    )

    assert budget.source_sufficient is True
    assert budget.cadence_sufficient is False
    assert budget.sufficient is False
