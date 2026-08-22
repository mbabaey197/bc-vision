import pytest

from app.ai.capacity_baseline import (
    aggregate_capacity_run,
    evaluate_passage_evidence,
)


def _status(*, decoded=100, emitted=2):
    return {
        "error": "",
        "stream_metrics": {
            "decoded_frames": decoded,
            "decode_seconds": 2.0,
            "capture_failures": 0,
            "jpeg_attempts": 50,
            "jpeg_frames": 50,
            "jpeg_seconds": 1.0,
            "jpeg_bytes": 5000,
            "anpr_queue_frames": decoded,
            "anpr_queue_coalesced_frames": 10,
            "anpr_submitted_frames": decoded - 10,
        },
        "anpr": {
            "received_frames": decoded - 10,
            "processed_frames": 20,
            "inference_calls": 20,
            "inference_seconds": 4.0,
            "coalesced_frames": 70,
            "emitted_events": emitted,
            "persistence_backpressure_frames": 0,
            "last_error": "",
            "models": {
                "ready": True,
                "selected_detector": "yolo11n",
                "production_ocr_policy": "hezar-v2-then-fixed-platrix",
            },
        },
    }


def test_capacity_report_has_all_required_cost_and_drop_metrics():
    report = aggregate_capacity_run(
        [_status()],
        camera_count=1,
        wall_seconds=10.0,
        process_cpu_seconds=5.0,
        logical_cpu_count=4,
        source_frames=100,
        source_fps=10.0,
        completed=True,
    )

    assert report["valid"] is True
    assert report["process_cpu_core_percent"] == 50.0
    assert report["process_cpu_host_percent"] == 12.5
    assert report["decode"]["aggregate_fps"] == 10.0
    assert report["inference"]["mean_ms"] == 200.0
    assert report["jpeg"]["fps"] == 5.0
    assert report["frame_drop"]["decode_shortfall_frames"] == 0
    assert report["frame_drop"]["application_coalesced_frames"] == 80
    assert report["events"]["emitted"] == 2


def test_incomplete_or_model_unready_run_is_not_comparable_evidence():
    status = _status(decoded=90)
    status["anpr"]["models"]["ready"] = False

    report = aggregate_capacity_run(
        [status, _status(decoded=90), _status(decoded=90)],
        camera_count=3,
        wall_seconds=10.0,
        process_cpu_seconds=10.0,
        logical_cpu_count=8,
        source_frames=100,
        source_fps=10.0,
        completed=True,
    )

    assert report["valid"] is False
    assert "production-models-not-ready" in report["invalid_reasons"]
    assert "decoded-frame-shortfall" in report["invalid_reasons"]


def test_capacity_contract_requires_exactly_one_status_per_supported_count():
    with pytest.raises(ValueError):
        aggregate_capacity_run(
            [_status()],
            camera_count=6,
            wall_seconds=1.0,
            process_cpu_seconds=1.0,
            logical_cpu_count=1,
            source_frames=1,
            source_fps=1.0,
            completed=True,
        )


def test_no_passage_dataset_never_turns_capacity_into_99_percent_claim():
    decision = evaluate_passage_evidence(None)

    assert decision["claim_ready"] is False
    assert decision["reasons"] == [
        "independent-labelled-passage-dataset-required"
    ]


def test_passage_rows_without_independent_provenance_fail_closed():
    decision = evaluate_passage_evidence({"passages": []})

    assert decision["claim_ready"] is False
    assert "independent-annotation-provenance-required" in decision["reasons"]
