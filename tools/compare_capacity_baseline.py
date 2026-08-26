"""Compare two BC Vision production capacity reports fail-closed.

The production runner in :mod:`app.ai.capacity_baseline` records the real
1/3/6-camera path.  This utility adds the missing regression gate for a fixed
benchmark workstation.  It intentionally does not turn replay event counts
into an accuracy claim; passage accuracy remains governed by the independent
labelled-passage evidence contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVIDENCE_KIND = "production-pipeline-capacity-baseline"
SUPPORTED_CAMERA_COUNTS = (1, 3, 6)
_SETTINGS_KEYS = (
    "camera_counts",
    "detector",
    "live_fps",
    "stream_width",
    "jpeg_quality",
    "lpr_confidence",
    "duplicate_seconds",
    "viewers_per_camera",
)


def _number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{label} must be finite")
    return number


def _nested(mapping: dict, *keys: str, label: str) -> float:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing {label}")
        value = value[key]
    return _number(value, label=label)


def _runs_by_count(report: dict, *, label: str) -> dict[int, dict]:
    rows = report.get("runs")
    if not isinstance(rows, list):
        raise ValueError(f"{label}.runs must be a list")
    indexed: dict[int, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{label}.runs contains a non-object")
        try:
            count = int(row.get("camera_count"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{label} run has invalid camera_count") from exc
        if count in indexed:
            raise ValueError(f"{label} contains duplicate {count}-camera run")
        indexed[count] = row
    if tuple(sorted(indexed)) != SUPPORTED_CAMERA_COUNTS:
        raise ValueError(f"{label} must contain exactly 1, 3 and 6 camera runs")
    return indexed


def _relative_limit(baseline: float, percent: float) -> float:
    return baseline * (1.0 + percent / 100.0)


def _relative_floor(baseline: float, percent: float) -> float:
    return baseline * max(0.0, 1.0 - percent / 100.0)


def _validate_threshold(value: float, *, label: str, maximum: float = 1000.0) -> float:
    parsed = _number(value, label=label)
    if not 0.0 <= parsed <= maximum:
        raise ValueError(f"{label} must be within 0..{maximum:g}")
    return parsed


def compare_capacity_reports(
    baseline: dict,
    current: dict,
    *,
    max_cpu_regression_percent: float = 10.0,
    max_latency_regression_percent: float = 10.0,
    max_fps_regression_percent: float = 10.0,
    max_coalesced_rate_increase: float = 0.05,
    max_accuracy_ci_drop: float = 0.005,
) -> dict:
    """Return a deterministic regression decision for one fixed host.

    Performance thresholds are relative percentages except for application
    coalescing and the optional accuracy confidence-bound drop, which are
    absolute fractions in the 0..1 range.
    """

    if not isinstance(baseline, dict) or not isinstance(current, dict):
        raise ValueError("baseline and current reports must be JSON objects")
    cpu_limit = _validate_threshold(
        max_cpu_regression_percent,
        label="max_cpu_regression_percent",
    )
    latency_limit = _validate_threshold(
        max_latency_regression_percent,
        label="max_latency_regression_percent",
    )
    fps_limit = _validate_threshold(
        max_fps_regression_percent,
        label="max_fps_regression_percent",
    )
    coalesced_limit = _validate_threshold(
        max_coalesced_rate_increase,
        label="max_coalesced_rate_increase",
        maximum=1.0,
    )
    accuracy_limit = _validate_threshold(
        max_accuracy_ci_drop,
        label="max_accuracy_ci_drop",
        maximum=1.0,
    )

    failures: list[str] = []
    warnings: list[str] = []

    for label, report in (("baseline", baseline), ("current", current)):
        if report.get("evidence_kind") != EVIDENCE_KIND:
            failures.append(f"{label}:wrong-evidence-kind")
        if report.get("comparable") is not True:
            failures.append(f"{label}:report-not-comparable")

    baseline_source = baseline.get("source") or {}
    current_source = current.get("source") or {}
    if baseline_source.get("sha256") != current_source.get("sha256"):
        failures.append("source-video-sha256-mismatch")

    baseline_settings = baseline.get("settings") or {}
    current_settings = current.get("settings") or {}
    for key in _SETTINGS_KEYS:
        if baseline_settings.get(key) != current_settings.get(key):
            failures.append(f"settings-mismatch:{key}")

    baseline_host = baseline.get("host") or {}
    current_host = current.get("host") or {}
    if baseline_host.get("logical_cpu_count") != current_host.get("logical_cpu_count"):
        failures.append("host-logical-cpu-count-mismatch")
    if baseline_host.get("machine") != current_host.get("machine"):
        failures.append("host-machine-architecture-mismatch")
    if baseline_host.get("platform") != current_host.get("platform"):
        warnings.append("host-platform-string-changed")

    baseline_runs = _runs_by_count(baseline, label="baseline")
    current_runs = _runs_by_count(current, label="current")
    comparisons: list[dict] = []

    for count in SUPPORTED_CAMERA_COUNTS:
        old = baseline_runs[count]
        new = current_runs[count]
        prefix = f"{count}-camera"
        if old.get("valid") is not True:
            failures.append(f"baseline:{prefix}:invalid-run")
        if new.get("valid") is not True:
            failures.append(f"current:{prefix}:invalid-run")

        cpu_old = _nested(old, "process_cpu_host_percent", label=f"baseline {prefix} cpu")
        cpu_new = _nested(new, "process_cpu_host_percent", label=f"current {prefix} cpu")
        decode_old = _nested(old, "decode", "mean_ms", label=f"baseline {prefix} decode mean")
        decode_new = _nested(new, "decode", "mean_ms", label=f"current {prefix} decode mean")
        inference_old = _nested(old, "inference", "mean_ms", label=f"baseline {prefix} inference mean")
        inference_new = _nested(new, "inference", "mean_ms", label=f"current {prefix} inference mean")
        fps_old = _nested(old, "inference", "per_camera_fps", label=f"baseline {prefix} inference fps")
        fps_new = _nested(new, "inference", "per_camera_fps", label=f"current {prefix} inference fps")
        coalesced_old = _nested(
            old,
            "frame_drop",
            "application_coalesced_rate",
            label=f"baseline {prefix} coalesced rate",
        )
        coalesced_new = _nested(
            new,
            "frame_drop",
            "application_coalesced_rate",
            label=f"current {prefix} coalesced rate",
        )

        if cpu_new > _relative_limit(cpu_old, cpu_limit) + 1e-9:
            failures.append(f"{prefix}:cpu-regression")
        if decode_new > _relative_limit(decode_old, latency_limit) + 1e-9:
            failures.append(f"{prefix}:decode-latency-regression")
        if inference_new > _relative_limit(inference_old, latency_limit) + 1e-9:
            failures.append(f"{prefix}:inference-latency-regression")
        if fps_old > 0.0 and fps_new + 1e-9 < _relative_floor(fps_old, fps_limit):
            failures.append(f"{prefix}:inference-fps-regression")
        if coalesced_new > coalesced_old + coalesced_limit + 1e-9:
            failures.append(f"{prefix}:application-coalescing-regression")

        events = new.get("events") or {}
        if events.get("count_match") is not True:
            failures.append(f"{prefix}:emitted-persisted-event-count-mismatch")

        # A headless production stream must not spend CPU on preview JPEG.
        if current_settings.get("viewers_per_camera") == 0:
            jpeg = new.get("jpeg") or {}
            if int(jpeg.get("attempts") or 0) != 0 or int(jpeg.get("frames") or 0) != 0:
                failures.append(f"{prefix}:headless-jpeg-work-detected")

        comparisons.append({
            "camera_count": count,
            "baseline": {
                "cpu_host_percent": cpu_old,
                "decode_mean_ms": decode_old,
                "inference_mean_ms": inference_old,
                "inference_fps_per_camera": fps_old,
                "application_coalesced_rate": coalesced_old,
            },
            "current": {
                "cpu_host_percent": cpu_new,
                "decode_mean_ms": decode_new,
                "inference_mean_ms": inference_new,
                "inference_fps_per_camera": fps_new,
                "application_coalesced_rate": coalesced_new,
            },
        })

    baseline_accuracy = baseline.get("passage_accuracy") or {}
    current_accuracy = current.get("passage_accuracy") or {}
    if baseline_accuracy.get("claim_ready") is True:
        if current_accuracy.get("claim_ready") is not True:
            failures.append("passage-accuracy-claim-regressed")
        else:
            old_ci = baseline_accuracy.get("exact_accuracy_ci95")
            new_ci = current_accuracy.get("exact_accuracy_ci95")
            if not (
                isinstance(old_ci, (list, tuple))
                and len(old_ci) == 2
                and isinstance(new_ci, (list, tuple))
                and len(new_ci) == 2
            ):
                failures.append("passage-accuracy-ci-missing")
            else:
                old_lower = _number(old_ci[0], label="baseline exact accuracy CI lower")
                new_lower = _number(new_ci[0], label="current exact accuracy CI lower")
                if new_lower + accuracy_limit + 1e-12 < old_lower:
                    failures.append("passage-accuracy-ci-regression")

    return {
        "schema": 1,
        "pass": not failures,
        "failures": sorted(set(failures)),
        "warnings": sorted(set(warnings)),
        "thresholds": {
            "max_cpu_regression_percent": cpu_limit,
            "max_latency_regression_percent": latency_limit,
            "max_fps_regression_percent": fps_limit,
            "max_coalesced_rate_increase": coalesced_limit,
            "max_accuracy_ci_drop": accuracy_limit,
        },
        "comparisons": comparisons,
    }


def _load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare BC Vision 1/3/6-camera capacity reports")
    parser.add_argument("baseline")
    parser.add_argument("current")
    parser.add_argument("--output")
    parser.add_argument("--max-cpu-regression-percent", type=float, default=10.0)
    parser.add_argument("--max-latency-regression-percent", type=float, default=10.0)
    parser.add_argument("--max-fps-regression-percent", type=float, default=10.0)
    parser.add_argument("--max-coalesced-rate-increase", type=float, default=0.05)
    parser.add_argument("--max-accuracy-ci-drop", type=float, default=0.005)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    decision = compare_capacity_reports(
        _load(Path(args.baseline)),
        _load(Path(args.current)),
        max_cpu_regression_percent=args.max_cpu_regression_percent,
        max_latency_regression_percent=args.max_latency_regression_percent,
        max_fps_regression_percent=args.max_fps_regression_percent,
        max_coalesced_rate_increase=args.max_coalesced_rate_increase,
        max_accuracy_ci_drop=args.max_accuracy_ci_drop,
    )
    rendered = json.dumps(decision, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if decision["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
