from copy import deepcopy

from tools.compare_capacity_baseline import compare_capacity_reports


def _report():
    runs = []
    for count in (1, 3, 6):
        runs.append({
            "camera_count": count,
            "valid": True,
            "process_cpu_host_percent": 10.0 * count,
            "decode": {"mean_ms": 2.0, "per_camera_fps": 10.0},
            "inference": {"mean_ms": 20.0, "per_camera_fps": 5.0},
            "jpeg": {"attempts": 0, "frames": 0},
            "frame_drop": {"application_coalesced_rate": 0.10},
            "events": {"emitted": 2, "persisted": 2, "count_match": True},
        })
    return {
        "evidence_kind": "production-pipeline-capacity-baseline",
        "comparable": True,
        "source": {"sha256": "a" * 64},
        "host": {
            "platform": "Windows-benchmark-host",
            "machine": "AMD64",
            "logical_cpu_count": 8,
        },
        "settings": {
            "camera_counts": [1, 3, 6],
            "detector": "yolo11n",
            "live_fps": 5,
            "stream_width": 640,
            "jpeg_quality": 70,
            "lpr_confidence": 60,
            "duplicate_seconds": 30.0,
            "viewers_per_camera": 0,
        },
        "runs": runs,
        "passage_accuracy": {"claim_ready": False},
    }


def test_identical_fixed_host_reports_pass():
    baseline = _report()
    decision = compare_capacity_reports(baseline, deepcopy(baseline))

    assert decision["pass"] is True
    assert decision["failures"] == []
    assert len(decision["comparisons"]) == 3


def test_cpu_regression_fails_only_affected_camera_count():
    baseline = _report()
    current = deepcopy(baseline)
    current["runs"][2]["process_cpu_host_percent"] = 70.0

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is False
    assert "6-camera:cpu-regression" in decision["failures"]
    assert "1-camera:cpu-regression" not in decision["failures"]


def test_headless_jpeg_work_is_a_regression_even_when_cpu_is_acceptable():
    baseline = _report()
    current = deepcopy(baseline)
    current["runs"][1]["jpeg"] = {"attempts": 4, "frames": 4}

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is False
    assert "3-camera:headless-jpeg-work-detected" in decision["failures"]


def test_different_source_video_cannot_be_compared():
    baseline = _report()
    current = deepcopy(baseline)
    current["source"]["sha256"] = "b" * 64

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is False
    assert "source-video-sha256-mismatch" in decision["failures"]


def test_fixed_host_contract_rejects_cpu_topology_change():
    baseline = _report()
    current = deepcopy(baseline)
    current["host"]["logical_cpu_count"] = 16

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is False
    assert "host-logical-cpu-count-mismatch" in decision["failures"]


def test_platform_patch_change_warns_without_invalidating_same_architecture_host():
    baseline = _report()
    current = deepcopy(baseline)
    current["host"]["platform"] = "Windows-benchmark-host-patched"

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is True
    assert decision["warnings"] == ["host-platform-string-changed"]


def test_claim_ready_accuracy_must_not_disappear():
    baseline = _report()
    current = deepcopy(baseline)
    baseline["passage_accuracy"] = {
        "claim_ready": True,
        "exact_accuracy_ci95": [0.992, 1.0],
    }
    current["passage_accuracy"] = {
        "claim_ready": False,
        "exact_accuracy_ci95": [0.990, 1.0],
    }

    decision = compare_capacity_reports(baseline, current)

    assert decision["pass"] is False
    assert "passage-accuracy-claim-regressed" in decision["failures"]


def test_accuracy_ci_lower_bound_has_small_configurable_regression_budget():
    baseline = _report()
    current = deepcopy(baseline)
    baseline["passage_accuracy"] = {
        "claim_ready": True,
        "exact_accuracy_ci95": [0.995, 1.0],
    }
    current["passage_accuracy"] = {
        "claim_ready": True,
        "exact_accuracy_ci95": [0.988, 0.999],
    }

    decision = compare_capacity_reports(
        baseline,
        current,
        max_accuracy_ci_drop=0.005,
    )

    assert decision["pass"] is False
    assert "passage-accuracy-ci-regression" in decision["failures"]
