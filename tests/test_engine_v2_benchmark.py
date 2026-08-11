from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from app.engine_v2 import benchmark as benchmark_module
from app.engine_v2.benchmark import (
    REQUIRED_ACCURACY_CATEGORIES,
    BenchmarkScenario,
    CallableAccuracyAdapter,
    CommandAccuracyAdapter,
    SyntheticControlPlaneAdapter,
    all_active_camera_scenarios,
    compare_accuracy_adapters,
    default_camera_scenarios,
    load_accuracy_manifest,
    normalize_plate_text,
    run_performance_scenario,
    run_performance_suite,
    run_standard_performance_matrices,
    write_accuracy_outputs,
    write_performance_outputs,
)


def _verified_manifest(tmp_path: Path) -> Path:
    samples = []
    for index, category in enumerate(REQUIRED_ACCURACY_CATEGORIES, start=1):
        media = tmp_path / f"sample-{index:02d}.bin"
        media.write_bytes(f"fixture-{category}".encode("utf-8"))
        sample = {
            "id": f"sample-{index:02d}",
            "category": category,
            "input": {"path": media.name, "media_type": "video"},
            "label_status": "verified",
        }
        if category == "multiple_vehicles":
            sample["expected_events"] = [
                {"plate": "۱۲ب۳۴۵۶۷", "start_ms": 0, "end_ms": 500},
                {"plate": "۳۴د۷۶۵۴۳", "start_ms": 501, "end_ms": 1000},
            ]
        else:
            sample["expected_plate"] = "۱۲ب۳۴۵۶۷"
        samples.append(sample)
    negative_media = tmp_path / "negative-01.bin"
    negative_media.write_bytes(b"verified-empty-lane-fixture")
    samples.append(
        {
            "id": "negative-01",
            "category": "clear_plate",
            "input": {"path": negative_media.name, "media_type": "video"},
            "expected_plate": None,
            "label_status": "verified",
        }
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema": "bcvision.anpr.accuracy-manifest/v1",
                "dataset_id": "unit-test-verified-fixtures",
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_default_suite_has_required_camera_counts_and_optional_32() -> None:
    scenarios = default_camera_scenarios(include_32=True, active_cameras=1)

    assert [scenario.camera_count for scenario in scenarios] == [1, 4, 8, 16, 32]
    assert all(scenario.active_cameras == 1 for scenario in scenarios)
    assert scenarios[-1].idle_cameras == 31

    busy = all_active_camera_scenarios(include_32=True)
    assert [scenario.camera_count for scenario in busy] == [1, 4, 8, 16, 32]
    assert all(scenario.active_cameras == scenario.camera_count for scenario in busy)


def test_standard_matrix_runs_idle_and_all_active_sweeps() -> None:
    report = run_standard_performance_matrices(
        SyntheticControlPlaneAdapter(),
        nominal_seconds=0.1,
        ticks_per_second=1,
    )

    assert set(report["performance_matrices"]) == {
        "fixed_active_idle_scaling",
        "all_active_busy_scaling",
    }
    assert len(report["performance_matrices"]["fixed_active_idle_scaling"]) == 4
    assert len(report["performance_matrices"]["all_active_busy_scaling"]) == 4
    assert len(report["busy_camera_scaling"]["comparisons"]) == 3


def test_resource_sampler_falls_back_when_psutil_process_is_inconsistent(monkeypatch) -> None:
    class FakeResource:
        RUSAGE_SELF = 0

        @staticmethod
        def getrusage(_who):
            # ``resource`` is absent on the Windows CI runner. Inject a stable
            # cross-platform peak-RSS reference so this test exercises the
            # PID-namespace sanity fallback instead of the host OS API.
            return SimpleNamespace(ru_maxrss=4096)

    class WrongPidProcess:
        def cpu_times(self):
            return SimpleNamespace(user=10.0, system=5.0)

        def memory_info(self):
            return SimpleNamespace(rss=1)

    sampler = benchmark_module._ResourceSampler()
    sampler._process = WrongPidProcess()
    monkeypatch.setattr(benchmark_module, "resource", FakeResource())
    perf_values = iter((100.0, 101.0))
    cpu_values = iter((10.0, 10.2))
    monkeypatch.setattr(benchmark_module.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(benchmark_module.time, "process_time", lambda: next(cpu_values))

    sampler.start()
    measured = sampler.finish()

    assert measured["process_cpu_percent"] == pytest.approx(20.0)
    assert measured["cpu_percent_source"].startswith("time.process_time/")
    assert measured["ram_source"] == "resource.getrusage.maxrss"
    assert measured["ram_mb"] > 1.0
    assert "psutil_cpu_rejected_inconsistent_with_process_time" in measured[
        "resource_sampling_warnings"
    ]
    assert "psutil_rss_rejected_inconsistent_with_resource_maxrss" in measured[
        "resource_sampling_warnings"
    ]


def test_synthetic_scenario_records_newest_frame_and_required_metrics() -> None:
    result = run_performance_scenario(
        BenchmarkScenario(
            name="four-cameras",
            camera_count=4,
            active_cameras=1,
            nominal_seconds=0.2,
            ticks_per_second=10,
            producer_burst=3,
            consumer_budget_per_tick=1,
        ),
        SyntheticControlPlaneAdapter(ocr_every=1, event_every=1),
    )

    assert result["active_cameras"] == 1
    assert result["idle_cameras"] == 3
    assert result["latest_frame_replacements"] == 4
    assert result["dropped_stale_frames"] == 4
    assert result["idle_detector_inferences"] == 0
    assert result["detector_inferences"] == 2
    assert result["ocr_inferences"] == 2
    assert result["plate_events"] == 2
    assert result["decode_utilization_percent"] is None
    assert result["decode_utilization_kind"] == "unavailable"
    assert result["decode_utilization_source"].startswith("unavailable:")
    assert result["production_evidence"] is False
    for metric in (
        "cpu_percent",
        "ram_mb",
        "queue_depth_average",
        "average_latency_ms",
        "p95_latency_ms",
        "plate_events_per_second",
    ):
        assert metric in result


def test_decode_utilization_requires_and_preserves_provenance() -> None:
    class MeasuredAdapter:
        adapter_name = "measured-decoder"
        evidence_kind = "real-callable-adapter"
        production_evidence = True

        def process(self, job):
            del job
            return {
                "detector_inferences": 1,
                "decode_utilization_percent": 42.5,
                "decode_utilization_kind": "measured",
                "decode_utilization_source": "intel-gpu-busy-counter",
            }

    result = run_performance_scenario(
        BenchmarkScenario(
            name="decode-provenance",
            camera_count=1,
            active_cameras=1,
            nominal_seconds=0.1,
            ticks_per_second=1,
        ),
        MeasuredAdapter(),
    )

    assert result["decode_utilization_percent"] == 42.5
    assert result["decode_utilization_kind"] == "measured"
    assert result["decode_utilization_source"] == "intel-gpu-busy-counter"
    assert result["production_evidence"] is False
    assert "evidence-metadata-missing" in result["evidence_validation"]["reasons"]


def test_production_evidence_requires_verified_input_and_model_files(tmp_path: Path) -> None:
    media = tmp_path / "camera.mp4"
    model = tmp_path / "detector.onnx"
    media.write_bytes(b"operator-owned-camera-fixture")
    model.write_bytes(b"verified-model-fixture")

    class VerifiedAdapter:
        adapter_name = "verified-adapter"
        evidence_kind = "real-callable-adapter"
        production_evidence = True
        evidence_metadata = {
            "resource_scope": "current-process",
            "uses_child_processes": False,
            "execution_provider": "CPUExecutionProvider",
            "input_files": [
                {"path": str(media), "sha256": hashlib.sha256(media.read_bytes()).hexdigest()}
            ],
            "model_files": [
                {"path": str(model), "sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
            ],
        }

        def process(self, job):
            del job
            return {"detector_inferences": 1}

    result = run_performance_scenario(
        BenchmarkScenario("verified", 1, 1, nominal_seconds=0.1, ticks_per_second=1),
        VerifiedAdapter(),
    )

    assert result["production_evidence"] is True
    assert result["evidence_validation"]["valid"] is True
    assert len(result["evidence_validation"]["verified_files"]) == 2


def test_constrained_scheduler_rotates_active_cameras_fairly() -> None:
    result = run_performance_scenario(
        BenchmarkScenario(
            name="fairness",
            camera_count=4,
            active_cameras=4,
            nominal_seconds=1.0,
            ticks_per_second=10,
            producer_burst=1,
            consumer_budget_per_tick=2,
        ),
        SyntheticControlPlaneAdapter(),
    )

    processed = [row["processed_jobs"] for row in result["per_camera"].values()]
    assert min(processed) > 0
    assert max(processed) - min(processed) <= 1
    assert result["scheduler_fairness"]["starved_active_cameras"] == 0
    assert result["scheduler_fairness"]["jain_index"] > 0.98


def test_performance_suite_writes_json_and_csv(tmp_path: Path) -> None:
    scenarios = default_camera_scenarios(
        active_cameras=1,
        nominal_seconds=0.1,
        ticks_per_second=1,
    )
    report = run_performance_suite(scenarios, SyntheticControlPlaneAdapter())
    json_path = tmp_path / "performance.json"
    csv_path = tmp_path / "performance.csv"

    write_performance_outputs(report, json_path=json_path, csv_path=csv_path)

    stored = json.loads(json_path.read_text(encoding="utf-8"))
    assert stored["production_decision_allowed"] is False
    assert len(stored["scenarios"]) == 4
    assert len(stored["idle_camera_scaling"]["comparisons"]) == 3
    header = csv_path.read_text(encoding="utf-8").splitlines()[0]
    assert "decode_utilization_source" in header
    assert "dropped_stale_frames" in header


def test_accuracy_compare_uses_same_verified_manifest_and_separate_results(tmp_path: Path) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))
    observed_keys = []

    def v1(sample):
        observed_keys.append(set(sample))
        if sample["id"] == "negative-01":
            return None
        if sample["category"] == "night":
            return {"plate": "12ب34566", "confidence": 0.7, "accepted": True}
        if sample["category"] == "multiple_vehicles":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        return {"plate": "12ب34567", "confidence": 0.9, "accepted": True}

    def v2(sample):
        observed_keys.append(set(sample))
        if sample["id"] == "negative-01":
            return None
        if sample["category"] == "multiple_vehicles":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        return {"plate": "12ب34567", "confidence": 0.95, "accepted": True}

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(v1, name="legacy-test-adapter"),
        CallableAccuracyAdapter(v2, name="v2-test-adapter"),
    )

    assert report["same_manifest_for_both_engines"] is True
    assert report["negative_sample_count"] == 1
    assert report["v1"]["metrics"]["exact_accuracy"] == 0.875
    assert report["v2"]["metrics"]["exact_accuracy"] == 1.0
    assert report["comparison"]["exact_accuracy_delta"] == 0.125
    assert report["v2"]["metrics"]["expected_events"] == 9
    assert report["v2"]["metrics"]["event_recall"] == 1.0
    assert report["production_decision_allowed"] is False
    assert len(report["v1"]["predictions"]) == len(report["v2"]["predictions"]) == 9
    assert all(keys <= {"id", "category", "input", "adapter_input"} for keys in observed_keys)
    assert all("expected_plate" not in keys for keys in observed_keys)

    json_path = tmp_path / "accuracy.json"
    csv_path = tmp_path / "accuracy.csv"
    write_accuracy_outputs(report, json_path=json_path, csv_path=csv_path)
    assert json.loads(json_path.read_text(encoding="utf-8"))["manifest_sha256"] == manifest.sha256
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 19


def test_accuracy_gate_rejects_per_category_regression_even_when_totals_tie(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))

    def prediction(sample, *, wrong_category):
        if sample["id"] == "negative-01":
            return None
        if sample["category"] == "multiple_vehicles":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        if sample["category"] == wrong_category:
            return "99ج99999"
        return "12ب34567"

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(
            lambda sample: prediction(sample, wrong_category="night"),
            name="v1-night-regression",
        ),
        CallableAccuracyAdapter(
            lambda sample: prediction(sample, wrong_category="motion_blur"),
            name="v2-motion-regression",
        ),
    )

    assert report["v1"]["metrics"]["exact_accuracy"] == report["v2"]["metrics"][
        "exact_accuracy"
    ]
    assert report["comparison"]["v2_accuracy_not_worse"] is False
    regressions = report["comparison"]["category_regressions"]
    assert [row["category"] for row in regressions] == ["motion_blur"]
    assert "exact_set_accuracy" in regressions[0]["metrics"]
    assert "event_recall" in regressions[0]["metrics"]
    assert "false_positive_events" in regressions[0]["metrics"]


def test_category_regression_helper_checks_all_required_metrics() -> None:
    baseline = {
        category: {
            "exact_set_accuracy": 1.0,
            "event_recall": 1.0,
            "false_accept_rate": 0.0,
            "false_positive_events": 0,
            "duplicate_events": 0,
        }
        for category in REQUIRED_ACCURACY_CATEGORIES
    }
    candidate = {category: dict(values) for category, values in baseline.items()}
    candidate["night"].update(
        {
            "exact_set_accuracy": 0.5,
            "event_recall": 0.5,
            "false_accept_rate": 0.25,
            "false_positive_events": 1,
            "duplicate_events": 1,
        }
    )

    regressions = benchmark_module._category_accuracy_regressions(
        baseline,
        candidate,
    )

    assert regressions == [
        {
            "category": "night",
            "metrics": [
                "exact_set_accuracy",
                "event_recall",
                "false_accept_rate",
                "false_positive_events",
                "duplicate_events",
            ],
            "deltas": {
                "exact_set_accuracy": -0.5,
                "event_recall": -0.5,
                "false_accept_rate": 0.25,
                "false_positive_events": 1,
                "duplicate_events": 1,
            },
        }
    ]


def test_command_accuracy_adapter_supports_json_contract() -> None:
    script = (
        "import json,sys; request=json.load(sys.stdin); "
        "assert 'expected_plate' not in request['sample']; "
        "print(json.dumps({'plate': '12ب34567', "
        "'confidence': 0.8, 'accepted': True}, ensure_ascii=False))"
    )
    adapter = CommandAccuracyAdapter(
        [sys.executable, "-c", script],
        name="command-v2",
    )

    prediction = adapter.predict(
        {
            "id": "sample-1",
            "input": {"path": "operator-owned.mp4"},
            "category": "clear_plate",
        }
    )

    assert normalize_plate_text(prediction.plate) == "12ب34567"
    assert prediction.accepted is True


def test_windows_command_parser_preserves_backslashes_and_quoted_paths(monkeypatch) -> None:
    monkeypatch.setattr(benchmark_module.os, "name", "nt")

    tokens = benchmark_module._split_command(
        r'"C:\Program Files\Python313\python.exe" "C:\Program Files\adapter.py"'
    )

    assert tokens == [
        r"C:\Program Files\Python313\python.exe",
        r"C:\Program Files\adapter.py",
    ]


def test_multiple_vehicle_scoring_counts_missed_false_positive_and_duplicate_events(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))

    def duplicate_first_vehicle(sample):
        if sample["id"] == "negative-01":
            return None
        if sample["category"] == "multiple_vehicles":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "12ب34567", "timestamp_ms": 200},
                ]
            }
        return "12ب34567"

    def correct(sample):
        if sample["id"] == "negative-01":
            return None
        if sample["category"] == "multiple_vehicles":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        return "12ب34567"

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(duplicate_first_vehicle, name="duplicates"),
        CallableAccuracyAdapter(correct, name="correct"),
    )
    bucket = report["v1"]["metrics"]["categories"]["multiple_vehicles"]

    assert bucket["matched_events"] == 1
    assert bucket["missed_events"] == 1
    assert bucket["false_positive_events"] == 1
    assert bucket["duplicate_events"] == 1
    assert bucket["event_recall"] == 0.5
    assert bucket["event_precision"] == 0.5
    assert bucket["exact_set_accuracy"] == 0.0


def test_template_manifest_is_rejected_as_evidence() -> None:
    template = Path(__file__).parent / "fixtures" / "engine_v2_accuracy_manifest.template.json"

    with pytest.raises(ValueError, match="template manifests"):
        load_accuracy_manifest(template, require_input_files=False)


def test_manifest_requires_every_accuracy_category(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"] = [
        sample for sample in payload["samples"] if sample["category"] != "partial_dirty_plate"
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required categories"):
        load_accuracy_manifest(path)


def test_manifest_rejects_all_null_ground_truth(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    for sample in payload["samples"]:
        sample.pop("expected_events", None)
        sample["expected_plate"] = None
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="no readable event labels"):
        load_accuracy_manifest(path)


def test_manifest_rejects_placeholder_or_malformed_plate_labels(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["expected_plate"] = "REPLACE_WITH_PLATE"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="structurally valid Iranian plate"):
        load_accuracy_manifest(path)


def test_manifest_requires_at_least_one_verified_negative_sample(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"] = [
        sample
        for sample in payload["samples"]
        if sample.get("expected_plate", "positive") is not None
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="at least one verified negative sample"):
        load_accuracy_manifest(path)


def test_manifest_optionally_verifies_media_sha256(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    media = tmp_path / sample["input"]["path"]
    expected_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
    sample["input"]["sha256"] = expected_sha256
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    manifest = load_accuracy_manifest(path)
    assert manifest.verified_media_sha256s == ((sample["id"], expected_sha256),)

    media.write_bytes(b"tampered-media")
    with pytest.raises(ValueError, match="input.sha256 mismatch"):
        load_accuracy_manifest(path)
