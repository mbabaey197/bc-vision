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
        processing_p95_ms=50,
    )

    assert slow.source_sufficient is True
    assert slow.processing_sufficient is False
    assert slow.expected_processed_observations == 2.4
    assert "processing-capacity-insufficient" in slow.warning()
    assert fast.processing_sufficient is True
    assert fast.sufficient is True


def test_unknown_runtime_telemetry_does_not_raise_false_alarm():
    budget = calculate_frame_budget(
        max_speed_kmh=150,
        recognition_zone_m=10,
    )

    assert budget.source_fps == 0.0
    assert budget.processing_p95_ms == 0.0
    assert budget.sufficient is True
    assert budget.warning() == ""
