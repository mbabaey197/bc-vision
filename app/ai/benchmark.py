"""Golden-dataset metrics and promotion gate for baseline versus RC13."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .plate_rules import normalize_plate, plausible_plate


@dataclass(frozen=True)
class BenchmarkMetrics:
    samples: int
    readable_samples: int
    exact_matches: int
    false_accepts: int
    unreadable_rejections: int
    mean_latency_ms: float
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
        label = str(row.get("slice", "all"))
        bucket = slices[label]
        bucket["samples"] += 1
        if is_readable:
            readable += 1
            bucket["readable"] += 1
            if predicted == expected:
                exact += 1
                bucket["exact"] += 1
        elif accepted:
            false_accepts += 1
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
    }
