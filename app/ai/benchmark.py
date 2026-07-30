"""Golden-dataset metrics and promotion gate for baseline versus RC13."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .evaluation import character_distance
from .plate_rules import normalize_plate, plausible_plate


@dataclass(frozen=True)
class BenchmarkMetrics:
    samples: int
    readable_samples: int
    exact_matches: int
    false_accepts: int
    unreadable_rejections: int
    mean_latency_ms: float
    mean_character_error: float
    slices: dict

    @property
    def exact_accuracy(self) -> float:
        return self.exact_matches / max(1, self.readable_samples)

    @property
    def false_accept_rate(self) -> float:
        return self.false_accepts / max(
            1,
            self.samples - self.readable_samples,
        )


def score_predictions(rows) -> BenchmarkMetrics:
    exact = 0
    false_accepts = 0
    rejected = 0
    readable = 0
    latencies = []
    character_errors = []
    slices = defaultdict(
        lambda: {
            "samples": 0,
            "readable": 0,
            "exact": 0,
            "false_accepts": 0,
        }
    )
    for row in rows:
        expected = normalize_plate(row.get("expected_plate", ""))
        predicted = normalize_plate(row.get("predicted_plate", ""))
        is_readable = plausible_plate(expected)
        accepted = plausible_plate(predicted)
        labels = row.get("slices")
        if not isinstance(labels, (list, tuple, set)) or not labels:
            labels = [str(row.get("slice", "all"))]
        buckets = [slices[str(label)] for label in labels]
        for bucket in buckets:
            bucket["samples"] += 1
        if is_readable:
            readable += 1
            for bucket in buckets:
                bucket["readable"] += 1
            if predicted == expected:
                exact += 1
                for bucket in buckets:
                    bucket["exact"] += 1
            character_errors.append(
                character_distance(predicted, expected)
            )
        elif accepted:
            false_accepts += 1
            for bucket in buckets:
                bucket["false_accepts"] += 1
        else:
            rejected += 1
        latencies.append(max(0.0, float(row.get("latency_ms", 0.0))))
    return BenchmarkMetrics(
        samples=len(rows),
        readable_samples=readable,
        exact_matches=exact,
        false_accepts=false_accepts,
        unreadable_rejections=rejected,
        mean_latency_ms=(
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        mean_character_error=(
            sum(character_errors) / len(character_errors)
            if character_errors
            else 0.0
        ),
        slices=dict(slices),
    )


def evaluate_promotion(
    baseline: BenchmarkMetrics,
    candidate: BenchmarkMetrics,
    minimum_exact_gain=0.01,
    maximum_false_accept_rate=0.005,
    maximum_latency_ratio=1.75,
) -> dict:
    reasons = []
    if candidate.samples != baseline.samples or candidate.samples == 0:
        reasons.append("dataset-mismatch")
    if (
        candidate.exact_accuracy
        < baseline.exact_accuracy + float(minimum_exact_gain)
    ):
        reasons.append("exact-accuracy-gain")
    if (
        candidate.false_accept_rate
        > min(
            baseline.false_accept_rate,
            float(maximum_false_accept_rate),
        )
    ):
        reasons.append("false-accept-regression")
    baseline_latency = max(1.0, baseline.mean_latency_ms)
    if (
        candidate.mean_latency_ms
        > baseline_latency * float(maximum_latency_ratio)
    ):
        reasons.append("latency-regression")
    if (
        candidate.mean_character_error
        > baseline.mean_character_error + 1e-9
    ):
        reasons.append("character-error-regression")
    for label, baseline_slice in baseline.slices.items():
        candidate_slice = candidate.slices.get(label)
        if not candidate_slice:
            reasons.append(f"missing-slice:{label}")
            continue
        baseline_accuracy = baseline_slice["exact"] / max(
            1,
            baseline_slice["readable"],
        )
        candidate_accuracy = candidate_slice["exact"] / max(
            1,
            candidate_slice["readable"],
        )
        if candidate_accuracy + 0.01 < baseline_accuracy:
            reasons.append(f"slice-regression:{label}")
    return {
        "promote": not reasons,
        "reasons": reasons,
        "baseline_exact_accuracy": round(
            baseline.exact_accuracy,
            6,
        ),
        "candidate_exact_accuracy": round(
            candidate.exact_accuracy,
            6,
        ),
        "baseline_false_accept_rate": round(
            baseline.false_accept_rate,
            6,
        ),
        "candidate_false_accept_rate": round(
            candidate.false_accept_rate,
            6,
        ),
        "baseline_mean_character_error": round(
            baseline.mean_character_error,
            6,
        ),
        "candidate_mean_character_error": round(
            candidate.mean_character_error,
            6,
        ),
    }


def assess_training_candidate(result: dict, golden: dict) -> dict:
    """Fail-closed gate for an administrator-trained CRNN candidate."""

    reasons = []
    validation_samples = int(result.get("validation_samples", 0))
    baseline_accuracy = float(result.get("baseline_accuracy", 0.0))
    candidate_accuracy = float(result.get("candidate_accuracy", 0.0))
    baseline_error = float(
        result.get("baseline_mean_character_error", 99.0)
    )
    candidate_error = float(
        result.get("candidate_mean_character_error", 99.0)
    )
    if validation_samples < 12:
        reasons.append("validation-too-small")
    if candidate_accuracy < max(0.70, baseline_accuracy):
        reasons.append("validation-exact-regression")
    if candidate_error > baseline_error + 1e-9:
        reasons.append("validation-character-regression")
    if int(result.get("validation_regressions", 0)) > 0:
        reasons.append("validation-sample-regression")
    if not str(result.get("baseline_sha256", "")).strip():
        reasons.append("baseline-identity-missing")
    if str(result.get("initialization_mode", "")) not in {
        "active-checkpoint",
        "active-model-distillation",
    }:
        reasons.append("untrusted-initialization")
    if not golden.get("ready"):
        reasons.append("golden-not-ready")
    golden_decision = result.get("golden_decision")
    if not isinstance(golden_decision, dict):
        reasons.append("golden-comparison-missing")
    elif not golden_decision.get("promote"):
        reasons.extend(
            f"golden:{reason}"
            for reason in golden_decision.get("reasons", [])
        )
    return {
        "schema": 1,
        "promote": not reasons,
        "reasons": sorted(set(reasons)),
        "baseline_sha256": str(
            result.get("baseline_sha256", "")
        ).upper(),
        "validation_samples": validation_samples,
        "baseline_accuracy": round(baseline_accuracy, 6),
        "candidate_accuracy": round(candidate_accuracy, 6),
        "baseline_mean_character_error": round(
            baseline_error,
            6,
        ),
        "candidate_mean_character_error": round(
            candidate_error,
            6,
        ),
        "validation_regressions": int(
            result.get("validation_regressions", 0)
        ),
        "initialization_mode": str(
            result.get("initialization_mode", "")
        ),
        "golden": {
            "ready": bool(golden.get("ready")),
            "samples": int(golden.get("samples", 0)),
            "unique_plates": int(golden.get("unique_plates", 0)),
            "slice_counts": dict(golden.get("slice_counts", {})),
            "errors": list(golden.get("errors", [])),
        },
        "golden_decision": golden_decision,
    }
