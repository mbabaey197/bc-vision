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
    run_accuracy_adapter,
    run_performance_scenario,
    run_performance_suite,
    run_standard_performance_matrices,
    write_accuracy_outputs,
    write_performance_outputs,
)


def _verified_manifest(tmp_path: Path) -> Path:
    media_root = tmp_path / "media"
    media_root.mkdir()
    samples = []
    for index, category in enumerate(REQUIRED_ACCURACY_CATEGORIES, start=1):
        media_bytes = f"fixture-{category}".encode("utf-8")
        digest = hashlib.sha256(media_bytes).hexdigest()
        media = media_root / f"{digest}.bin"
        media.write_bytes(media_bytes)
        sample = {
            "id": f"sample-{index:02d}",
            "category": category,
            "input": {
                "path": media.relative_to(tmp_path).as_posix(),
                "media_type": "video",
                "sha256": digest,
                "size_bytes": media.stat().st_size,
            },
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
    negative_bytes = b"verified-empty-lane-fixture"
    negative_digest = hashlib.sha256(negative_bytes).hexdigest()
    negative_media = media_root / f"{negative_digest}.bin"
    negative_media.write_bytes(negative_bytes)
    samples.append(
        {
            "id": "negative-01",
            "category": "clear_plate",
            "input": {
                "path": negative_media.relative_to(tmp_path).as_posix(),
                "media_type": "video",
                "sha256": negative_digest,
                "size_bytes": negative_media.stat().st_size,
            },
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
                "label_source": "unit-test-operator",
                "training_allowed": False,
                "samples": samples,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def _known_positive_manifest(tmp_path: Path) -> Path:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample["label_scope"] = "known_positives"
    payload["samples"] = [sample]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _valid_v1_reproducibility_metadata(
    *,
    frame_step: int = 1,
    roi=None,
) -> dict:
    return {
        "schema": "bcvision.anpr.accuracy-adapter-metadata/v1",
        "adapter": "legacy-process-video",
        "settings": {"frame_step": frame_step, "roi": roi},
        "model_identity": {
            "files": [
                {"role": "detector-selected", "exists": True, "sha256": "a" * 64},
                {"role": "ocr-crnn-active", "exists": True, "sha256": "b" * 64},
            ],
            "execution_provider_contract": "CPUExecutionProvider",
            "device_contract": "CPU",
        },
    }


def _valid_v2_reproducibility_metadata(
    *,
    frame_step: int = 1,
    max_frames=None,
) -> dict:
    return {
        "schema": "bcvision.anpr.accuracy-adapter-metadata/v1",
        "adapter": "engine-v2-offline-shared-inference",
        "models": [
            {"role": "detector", "sha256": "c" * 64},
            {"role": "ocr", "sha256": "d" * 64},
        ],
        "selected_shared_model_runtime": {
            "detector": {
                "backend": "onnxruntime",
                "device": "CPU",
                "providers": ["CPUExecutionProvider"],
            },
            "ocr": {
                "backend": "onnxruntime",
                "device": "CPU",
                "providers": ["CPUExecutionProvider"],
            },
        },
        "config": {
            "adapter": {
                "frame_step": frame_step,
                "max_frames": max_frames,
            }
        },
    }


def _correct_opaque_prediction(sample):
    if sample["id"] == "sample-000009":
        return None
    if sample["id"] == "sample-000006":
        return {
            "events": [
                {"plate": "12ب34567", "timestamp_ms": 100},
                {"plate": "34د76543", "timestamp_ms": 700},
            ]
        }
    return "12ب34567"


def test_multi_event_matching_uses_maximum_cardinality_for_overlapping_windows() -> None:
    expected = [
        {"plate": "12ب34567", "start_ms": 0, "end_ms": 100},
        {"plate": "12ب34567", "start_ms": 40, "end_ms": 60},
    ]
    predicted = [
        {"plate": "12ب34567", "timestamp_ms": 49},
        {"plate": "12ب34567", "timestamp_ms": 90},
    ]

    result = benchmark_module._match_event_sets(expected, predicted)

    assert result["matched_events"] == 2
    assert result["missed_events"] == 0
    assert result["false_positive_events"] == 0
    assert result["exact_set_match"] is True


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

        def configure_scenario(self, scenario, cameras):
            del scenario, cameras

        def start_scenario(self, scenario):
            del scenario

        def stop_scenario(self, scenario):
            del scenario

        def process(self, job):
            del job
            return {
                "detector_inferences": 1,
                "decode_utilization_percent": 12.5,
                "decode_utilization_kind": "measured",
                "decode_utilization_source": "unit-test-decoder-counter",
            }

    result = run_performance_scenario(
        BenchmarkScenario(
            "verified",
            1,
            1,
            nominal_seconds=0.01,
            ticks_per_second=1,
            realtime_pacing=True,
        ),
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
    observed_requests = []

    def v1(sample):
        observed_requests.append(dict(sample))
        if sample["id"] == "sample-000009":
            return None
        if sample["id"] == "sample-000002":
            return {"plate": "12ب34566", "confidence": 0.7, "accepted": True}
        if sample["id"] == "sample-000006":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        return {"plate": "12ب34567", "confidence": 0.9, "accepted": True}

    def v2(sample):
        observed_requests.append(dict(sample))
        if sample["id"] == "sample-000009":
            return None
        if sample["id"] == "sample-000006":
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
    assert report["same_input_bytes_for_both_engines"] is True
    assert report["dataset_fingerprint"] == manifest.dataset_fingerprint
    assert report["negative_sample_count"] == 1
    assert report["v1"]["metrics"]["exact_accuracy"] == 0.875
    assert report["v2"]["metrics"]["exact_accuracy"] == 1.0
    assert report["comparison"]["exact_accuracy_delta"] == 0.125
    assert report["v2"]["metrics"]["expected_events"] == 9
    assert report["v2"]["metrics"]["event_recall"] == 1.0
    assert report["production_decision_allowed"] is False
    assert len(report["v1"]["predictions"]) == len(report["v2"]["predictions"]) == 9
    assert all(set(request) == {"id", "input"} for request in observed_requests)
    assert [request["id"] for request in observed_requests[:9]] == [
        f"sample-{index:06d}" for index in range(1, 10)
    ]
    assert [request["id"] for request in observed_requests[9:]] == [
        f"sample-{index:06d}" for index in range(1, 10)
    ]
    assert all("category" not in request for request in observed_requests)
    assert all(
        set(request["input"])
        <= {
            "path",
            "media_type",
            "sha256",
            "size_bytes",
            "start_ms",
            "end_ms",
            "resolved_path",
        }
        for request in observed_requests
    )
    assert not ({sample["id"] for sample in manifest.samples} & {
        request["id"] for request in observed_requests
    })

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
        category_ids = {
            "night": "sample-000002",
            "motion_blur": "sample-000004",
        }
        if sample["id"] == "sample-000009":
            return None
        if sample["id"] == "sample-000006":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "34د76543", "timestamp_ms": 700},
                ]
            }
        if sample["id"] == category_ids[wrong_category]:
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
            "event_precision": 1.0,
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
            "event_precision": 0.5,
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
                "event_precision",
                "false_accept_rate",
                "false_positive_events",
                "duplicate_events",
            ],
            "deltas": {
                "exact_set_accuracy": -0.5,
                "event_recall": -0.5,
                "event_precision": -0.5,
                "false_accept_rate": 0.25,
                "false_positive_events": 1,
                "duplicate_events": 1,
            },
        }
    ]


def test_command_accuracy_adapter_supports_json_contract() -> None:
    script = (
        "import json,sys; request=json.load(sys.stdin); "
        "assert set(request['sample']) == {'id', 'input'}; "
        "assert set(request['sample']['input']) <= "
        "{'path','resolved_path','media_type','sha256','size_bytes','start_ms','end_ms'}; "
        "print(json.dumps({'plate': '12ب34567', "
        "'confidence': 0.8, 'accepted': True}, ensure_ascii=False))"
    )
    adapter = CommandAccuracyAdapter(
        [sys.executable, "-c", script],
        name="command-v2",
    )

    prediction = adapter.predict(
        {
            "id": "sample-000001",
            "input": {"path": "operator-owned.mp4"},
            "category": "clear_plate",
            "expected_plate": "99ج99999",
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
        if sample["id"] == "sample-000009":
            return None
        if sample["id"] == "sample-000006":
            return {
                "events": [
                    {"plate": "12ب34567", "timestamp_ms": 100},
                    {"plate": "12ب34567", "timestamp_ms": 200},
                ]
            }
        return "12ب34567"

    def correct(sample):
        if sample["id"] == "sample-000009":
            return None
        if sample["id"] == "sample-000006":
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


def test_manifest_verifies_declared_media_sha256(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    media = tmp_path / sample["input"]["path"]
    expected_sha256 = hashlib.sha256(media.read_bytes()).hexdigest()
    sample["input"]["sha256"] = expected_sha256
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    manifest = load_accuracy_manifest(path)
    assert (sample["id"], expected_sha256) in manifest.verified_media_sha256s

    media.write_bytes(b"x" * sample["input"]["size_bytes"])
    with pytest.raises(ValueError, match="input.sha256 mismatch"):
        load_accuracy_manifest(path)


@pytest.mark.parametrize("missing_field", ["sha256", "size_bytes"])
def test_strict_manifest_requires_hash_and_byte_size(
    tmp_path: Path,
    missing_field: str,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"].pop(missing_field)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=rf"input\.{missing_field} is required"):
        load_accuracy_manifest(path)


def test_permissive_manifest_is_explicit_and_allows_unverifiable_uri(
    tmp_path: Path,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"] = {
        "path": "https://example.invalid/operator-fixture.mp4",
        "media_type": "video",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="URI inputs are not allowed"):
        load_accuracy_manifest(path)

    manifest = load_accuracy_manifest(
        path,
        require_input_files=False,
        strict_evidence=False,
    )
    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(lambda _sample: None, name="permissive-v1"),
        CallableAccuracyAdapter(lambda _sample: None, name="permissive-v2"),
    )

    assert manifest.strict_evidence is False
    assert report["same_input_bytes_for_both_engines"] is False
    comparison = report["comparison"]
    assert comparison["accuracy_gate_evaluable"] is False
    assert "non-strict-evidence" in comparison["accuracy_gate_blockers"]


def test_non_strict_evidence_is_the_only_blocker_for_otherwise_proven_run(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(
        _verified_manifest(tmp_path),
        strict_evidence=False,
    )

    class MetadataCallable:
        def __init__(self, metadata):
            self.reproducibility_metadata = metadata

        def __call__(self, sample):
            return _correct_opaque_prediction(sample)

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(
            MetadataCallable(_valid_v1_reproducibility_metadata()),
            name="proven-v1",
        ),
        CallableAccuracyAdapter(
            MetadataCallable(_valid_v2_reproducibility_metadata()),
            name="proven-v2",
        ),
    )

    assert report["same_input_bytes_for_both_engines"] is True
    assert report["comparison"]["accuracy_gate_blockers"] == [
        "non-strict-evidence"
    ]
    assert report["comparison"]["accuracy_gate_evaluable"] is False
    assert report["comparison"]["v2_accuracy_not_worse"] is False


def test_strict_manifest_rejects_parent_path_escape(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.bin"
    outside.write_bytes(b"outside-dataset-root")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"] = {
        "path": f"../{outside.name}",
        "media_type": "video",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "size_bytes": outside.stat().st_size,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="content-addressed"):
        load_accuracy_manifest(path)


def test_strict_manifest_rejects_symlink_escape(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-symlink-target.bin"
    outside.write_bytes(b"outside-symlink-target")
    digest = hashlib.sha256(outside.read_bytes()).hexdigest()
    link = tmp_path / "media" / f"{digest}.bin"
    try:
        link.symlink_to(outside)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable on this runner")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"] = {
        "path": link.name,
        "media_type": "video",
        "sha256": digest,
        "size_bytes": outside.stat().st_size,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    payload["samples"][0]["input"]["path"] = f"media/{link.name}"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="symbolic-link or junction"):
        load_accuracy_manifest(path)


def test_strict_manifest_rejects_symlinked_media_parent(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    media = tmp_path / "media"
    target = tmp_path / "real-media"
    media.rename(target)
    try:
        media.symlink_to(target, target_is_directory=True)
    except (NotImplementedError, OSError):
        pytest.skip("symlink creation is unavailable on this runner")

    with pytest.raises(ValueError, match="symbolic-link or junction"):
        load_accuracy_manifest(path)


def test_strict_manifest_rejects_non_content_addressed_path(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    original = tmp_path / sample["input"]["path"]
    meaningful = tmp_path / "media" / "clear-plate.bin"
    meaningful.write_bytes(original.read_bytes())
    sample["input"]["path"] = "media/clear-plate.bin"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="content-addressed"):
        load_accuracy_manifest(path)


def test_manifest_rejects_invalid_input_clip_order(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"].update({"start_ms": 200, "end_ms": 100})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="input clip ends before it starts"):
        load_accuracy_manifest(path)


def test_manifest_rejects_mismatched_declared_byte_size(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["input"]["size_bytes"] += 1
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="input.size_bytes mismatch"):
        load_accuracy_manifest(path)


def test_dataset_fingerprint_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first = load_accuracy_manifest(_verified_manifest(first_root))
    second_path = _verified_manifest(second_root)
    second = load_accuracy_manifest(second_path)

    assert first.dataset_fingerprint == second.dataset_fingerprint

    payload = json.loads(second_path.read_text(encoding="utf-8"))
    media = second_root / payload["samples"][0]["input"]["path"]
    media.write_bytes(media.read_bytes() + b"-changed")
    changed_digest = hashlib.sha256(media.read_bytes()).hexdigest()
    changed_media = media.with_name(f"{changed_digest}{media.suffix}")
    media.rename(changed_media)
    payload["samples"][0]["input"]["path"] = changed_media.relative_to(
        second_root
    ).as_posix()
    payload["samples"][0]["input"]["sha256"] = changed_digest
    payload["samples"][0]["input"]["size_bytes"] = changed_media.stat().st_size
    second_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    changed = load_accuracy_manifest(second_path)
    assert changed.dataset_fingerprint != first.dataset_fingerprint


def test_manifest_rejects_a_stale_claimed_dataset_fingerprint(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    loaded = load_accuracy_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_fingerprint_sha256"] = loaded.dataset_fingerprint
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert load_accuracy_manifest(path).dataset_fingerprint == loaded.dataset_fingerprint

    payload["dataset_fingerprint_sha256"] = "0" * 64
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match"):
        load_accuracy_manifest(path)


def test_dataset_fingerprint_binds_adapter_execution_order(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    first = load_accuracy_manifest(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("dataset_fingerprint_sha256", None)
    payload["samples"] = list(reversed(payload["samples"]))
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    reordered = load_accuracy_manifest(path)
    assert reordered.dataset_fingerprint != first.dataset_fingerprint


@pytest.mark.parametrize("strict_evidence", [True, False])
@pytest.mark.parametrize("enabled", [True, False])
def test_manifest_rejects_adapter_input_in_every_mode(
    tmp_path: Path,
    strict_evidence: bool,
    enabled: bool,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["enabled"] = enabled
    payload["samples"][0]["adapter_input"] = {
        "runtime": {"items": [{"expected-plate": "12ب34567"}]}
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_input is not permitted"):
        load_accuracy_manifest(path, strict_evidence=strict_evidence)


@pytest.mark.parametrize("level", ["root", "sample", "input", "event"])
def test_strict_manifest_rejects_unknown_fields_at_every_level(
    tmp_path: Path,
    level: str,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if level == "root":
        payload["dataset_alias"] = "hint"
    elif level == "sample":
        payload["samples"][0]["sample_id"] = "meaningful-alias"
    elif level == "input":
        payload["samples"][0]["input"]["source_path"] = "meaningful-alias.mp4"
    else:
        multiple = next(
            sample
            for sample in payload["samples"]
            if sample["category"] == "multiple_vehicles"
        )
        multiple["expected_events"][0]["answer"] = "12ب34567"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        load_accuracy_manifest(path)


def test_strict_manifest_validates_unknown_fields_on_disabled_samples(
    tmp_path: Path,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0].update({"enabled": False, "sample_alias": "hidden"})
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported field"):
        load_accuracy_manifest(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.pop("training_allowed"), "training_allowed"),
        (lambda payload: payload.pop("label_source"), "label_source"),
        (
            lambda payload: payload["samples"][0]["input"].pop("media_type"),
            "media_type",
        ),
    ],
)
def test_strict_manifest_requires_policy_and_media_provenance(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    path = _verified_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutation(payload)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_accuracy_manifest(path)


def test_known_positive_scope_is_partial_and_never_passes_accuracy_gate(
    tmp_path: Path,
) -> None:
    path = _known_positive_manifest(tmp_path)
    with pytest.raises(ValueError, match="exhaustive labels"):
        load_accuracy_manifest(path)

    manifest = load_accuracy_manifest(
        path,
        require_all_categories=False,
        require_negative_sample=False,
    )
    assert manifest.known_positive_sample_count == 1
    assert manifest.coverage_complete is False

    def predictions(_sample):
        return {
            "events": [
                {"plate": "12ب34567"},
                {"plate": "34د76543"},
            ]
        }

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(predictions, name="known-v1"),
        CallableAccuracyAdapter(predictions, name="known-v2"),
    )

    for engine in ("v1", "v2"):
        metrics = report[engine]["metrics"]
        row = report[engine]["predictions"][0]
        assert metrics["event_recall"] == 1.0
        assert metrics["mean_character_error_rate"] == 0.0
        assert metrics["exact_set_accuracy"] is None
        assert metrics["event_precision"] is None
        assert metrics["false_positive_events"] is None
        assert metrics["duplicate_events"] is None
        assert metrics["precision_unscored_predicted_events"] == 2
        assert metrics["unmatched_unscored_events"] == 1
        assert row["label_scope"] == "known_positives"
        assert row["exact_set_match"] is None
        assert row["false_positive_events"] is None
        assert row["duplicate_events"] is None
        assert row["unmatched_predicted_events"] == 1

    comparison = report["comparison"]
    assert comparison["accuracy_gate_evaluable"] is False
    assert comparison["v2_accuracy_not_worse"] is False
    assert "non-exhaustive-label-scope" in comparison["accuracy_gate_blockers"]
    assert comparison["exact_accuracy_delta"] is None
    assert comparison["false_positive_event_delta"] is None
    assert comparison["duplicate_event_delta"] is None
    assert comparison["event_recall_delta"] == 0.0


def test_known_positive_scope_rejects_null_label_and_scores_misses(tmp_path: Path) -> None:
    path = _known_positive_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["samples"][0]["expected_plate"] = None
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="requires at least one label"):
        load_accuracy_manifest(
            path,
            require_all_categories=False,
            require_negative_sample=False,
        )

    payload["samples"][0]["expected_plate"] = "12ب34567"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = load_accuracy_manifest(
        path,
        require_all_categories=False,
        require_negative_sample=False,
    )
    result = run_accuracy_adapter(
        manifest,
        CallableAccuracyAdapter(lambda _sample: None, name="known-miss"),
        engine_label="v2",
    )
    assert result["metrics"]["event_recall"] == 0.0
    assert result["metrics"]["mean_character_error_rate"] == 1.0


def test_character_error_rate_rejects_correct_text_outside_expected_window(
    tmp_path: Path,
) -> None:
    path = _known_positive_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample.pop("expected_plate")
    sample["expected_events"] = [
        {"plate": "12ب34567", "start_ms": 0, "end_ms": 100}
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = load_accuracy_manifest(
        path,
        require_all_categories=False,
        require_negative_sample=False,
    )

    result = run_accuracy_adapter(
        manifest,
        CallableAccuracyAdapter(
            lambda _sample: {
                "events": [{"plate": "12ب34567", "timestamp_ms": 101}]
            },
            name="out-of-window",
        ),
        engine_label="v2",
    )

    assert result["metrics"]["matched_events"] == 0
    assert result["metrics"]["event_recall"] == 0.0
    assert result["metrics"]["mean_character_error_rate"] == 1.0


def test_character_error_rate_does_not_reuse_one_prediction_for_two_labels(
    tmp_path: Path,
) -> None:
    path = _known_positive_manifest(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    sample = payload["samples"][0]
    sample.pop("expected_plate")
    sample["expected_events"] = [
        {"plate": "12ب34567"},
        {"plate": "12ب34567"},
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    manifest = load_accuracy_manifest(
        path,
        require_all_categories=False,
        require_negative_sample=False,
    )

    result = run_accuracy_adapter(
        manifest,
        CallableAccuracyAdapter(lambda _sample: "12ب34567", name="one-event"),
        engine_label="v2",
    )

    assert result["metrics"]["matched_events"] == 1
    assert result["metrics"]["missed_events"] == 1
    assert result["metrics"]["mean_character_error_rate"] == 0.5


def test_exhaustive_zero_predictions_make_accuracy_gate_non_evaluable(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))
    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(lambda _sample: None, name="zero-v1"),
        CallableAccuracyAdapter(lambda _sample: None, name="zero-v2"),
    )

    assert report["v1"]["metrics"]["event_precision"] is None
    assert report["v2"]["metrics"]["event_precision"] is None
    comparison = report["comparison"]
    assert comparison["accuracy_gate_evaluable"] is False
    assert comparison["v2_accuracy_not_worse"] is False
    assert "event_precision" in comparison["unavailable_gate_metrics"]
    assert "unavailable-gate-metrics" in comparison["accuracy_gate_blockers"]
    assert comparison["event_precision_delta"] is None


def test_accuracy_compare_aborts_if_media_changed_before_v1(tmp_path: Path) -> None:
    path = _verified_manifest(tmp_path)
    manifest = load_accuracy_manifest(path)
    media = Path(manifest.samples[0]["_resolved_input_path"])
    media.write_bytes(b"changed-before-v1")

    with pytest.raises(RuntimeError, match="immediately-before-v1"):
        compare_accuracy_adapters(
            manifest,
            CallableAccuracyAdapter(lambda _sample: None, name="v1"),
            CallableAccuracyAdapter(lambda _sample: None, name="v2"),
        )


def test_accuracy_compare_aborts_if_v1_tampers_with_media(tmp_path: Path) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))
    media = Path(manifest.samples[0]["_resolved_input_path"])
    v2_called = False

    def tampering_v1(_sample):
        media.write_bytes(b"changed-by-v1")
        return None

    def v2(_sample):
        nonlocal v2_called
        v2_called = True
        return None

    with pytest.raises(RuntimeError, match="between-v1-and-v2"):
        compare_accuracy_adapters(
            manifest,
            CallableAccuracyAdapter(tampering_v1, name="tampering-v1"),
            CallableAccuracyAdapter(v2, name="v2-must-not-run"),
        )

    assert v2_called is False


def test_accuracy_compare_aborts_if_v2_tampers_with_media(tmp_path: Path) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))
    media = Path(manifest.samples[0]["_resolved_input_path"])

    def tampering_v2(_sample):
        media.write_bytes(b"changed-by-v2")
        return None

    with pytest.raises(RuntimeError, match="immediately-after-v2"):
        compare_accuracy_adapters(
            manifest,
            CallableAccuracyAdapter(lambda _sample: None, name="v1"),
            CallableAccuracyAdapter(tampering_v2, name="tampering-v2"),
        )


def test_accuracy_adapters_record_reproducibility_and_close(tmp_path: Path) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))

    class ClosableAdapter:
        def __init__(self, revision: str) -> None:
            self.reproducibility_metadata = {"model_revision": revision, "seed": 7}
            self.closed = 0

        def __call__(self, _sample):
            return None

        def close(self) -> None:
            self.closed += 1

    v1_function = ClosableAdapter("v1-test")
    v2_function = ClosableAdapter("v2-test")
    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(v1_function, name="closable-v1"),
        CallableAccuracyAdapter(v2_function, name="closable-v2"),
    )

    assert report["v1"]["adapter_reproducibility"]["declared"] == {
        "model_revision": "v1-test",
        "seed": 7,
    }
    assert report["v2"]["adapter_reproducibility"]["declared"] == {
        "model_revision": "v2-test",
        "seed": 7,
    }
    assert v1_function.closed == v2_function.closed == 1
    assert all(
        result["closed"]
        for result in report["adapter_lifecycle"]["close_results"]
    )


def test_builtin_reproducibility_metadata_shapes_are_gate_valid() -> None:
    for declared in (
        _valid_v1_reproducibility_metadata(),
        _valid_v2_reproducibility_metadata(),
    ):
        result = benchmark_module._validate_accuracy_adapter_reproducibility(
            {"declared": declared}
        )
        assert result["valid"] is True
        assert result["reasons"] == []
        assert len(result["model_sha256s"]) == 2
        assert result["runtime_identity"] is not None


def test_generic_adapters_without_metadata_fail_reproducibility_gate(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))
    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(_correct_opaque_prediction, name="generic-v1"),
        CallableAccuracyAdapter(_correct_opaque_prediction, name="generic-v2"),
    )

    comparison = report["comparison"]
    assert comparison["accuracy_gate_evaluable"] is False
    assert "adapter-reproducibility-not-proven" in comparison["accuracy_gate_blockers"]
    for engine in ("v1", "v2"):
        validation = report["adapter_reproducibility_validation"][engine]
        assert validation["valid"] is False
        assert "declared-metadata-missing" in validation["reasons"]


def test_valid_adapter_metadata_is_invoked_and_can_make_gate_evaluable(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))

    class MetadataCallable:
        def __init__(self, metadata):
            self.metadata = metadata
            self.metadata_calls = 0

        def __call__(self, sample):
            return _correct_opaque_prediction(sample)

        def reproducibility_metadata(self):
            self.metadata_calls += 1
            return self.metadata

    v1 = MetadataCallable(_valid_v1_reproducibility_metadata())
    v2 = MetadataCallable(_valid_v2_reproducibility_metadata())
    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(v1, name="metadata-v1"),
        CallableAccuracyAdapter(v2, name="metadata-v2"),
    )

    assert v1.metadata_calls == v2.metadata_calls == 1
    assert all(
        report["adapter_reproducibility_validation"][engine]["valid"]
        for engine in ("v1", "v2")
    )
    assert report["effective_input_options"]["symmetric"] is True
    assert report["comparison"]["accuracy_gate_blockers"] == []
    assert report["comparison"]["accuracy_gate_evaluable"] is True
    assert report["comparison"]["v2_accuracy_not_worse"] is True


def test_asymmetric_effective_input_options_block_gate_and_are_reported(
    tmp_path: Path,
) -> None:
    manifest = load_accuracy_manifest(_verified_manifest(tmp_path))

    class MetadataCallable:
        def __init__(self, metadata):
            self.reproducibility_metadata = metadata

        def __call__(self, sample):
            return _correct_opaque_prediction(sample)

    report = compare_accuracy_adapters(
        manifest,
        CallableAccuracyAdapter(
            MetadataCallable(_valid_v1_reproducibility_metadata()),
            name="full-v1",
        ),
        CallableAccuracyAdapter(
            MetadataCallable(
                _valid_v2_reproducibility_metadata(max_frames=10)
            ),
            name="truncated-v2",
        ),
    )

    effective = report["effective_input_options"]
    assert effective["symmetric"] is False
    assert effective["differences"] == [
        {"option": "max_frames", "v1": None, "v2": 10}
    ]
    comparison = report["comparison"]
    assert comparison["accuracy_gate_evaluable"] is False
    assert "asymmetric-effective-input-options" in comparison[
        "accuracy_gate_blockers"
    ]
