"""Golden-dataset metrics and promotion gate for baseline versus RC13."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
import time

from .evaluation import character_distance
from .plate_rules import normalize_plate, plausible_plate

MIN_GOLDEN_PROMOTION_SAMPLES = 40
MIN_CANDIDATE_EXACT_ACCURACY = 0.90
MIN_READABLE_SLICE_ACCURACY = 0.70
MAXIMUM_FALSE_ACCEPT_RATE = 0.005


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
    minimum_candidate_exact_accuracy=MIN_CANDIDATE_EXACT_ACCURACY,
    minimum_readable_slice_accuracy=MIN_READABLE_SLICE_ACCURACY,
    maximum_false_accept_rate=MAXIMUM_FALSE_ACCEPT_RATE,
    maximum_latency_ratio=1.75,
) -> dict:
    reasons = []
    if candidate.samples != baseline.samples or candidate.samples == 0:
        reasons.append("dataset-mismatch")
    if (
        candidate.exact_accuracy
        < float(minimum_candidate_exact_accuracy)
    ):
        reasons.append("candidate-accuracy-floor")
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
        if (
            int(candidate_slice["readable"]) > 0
            and candidate_accuracy
            < float(minimum_readable_slice_accuracy)
        ):
            reasons.append(f"candidate-slice-floor:{label}")
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


def validate_golden_decision_evidence(
    decision: dict,
    golden: dict,
) -> list[str]:
    """Validate stored comparison evidence independently of its promote bit."""

    if not isinstance(decision, dict):
        return ["golden-comparison-missing"]
    reasons = []
    digest = str(
        decision.get("golden_manifest_sha256", "")
    ).strip().upper()
    current_digest = str(
        golden.get("manifest_sha256", "")
    ).strip().upper()
    if (
        len(digest) != 64
        or any(character not in "0123456789ABCDEF" for character in digest)
        or digest != current_digest
    ):
        reasons.append("golden-identity-mismatch")
    if decision.get("promote") is not True:
        if not decision.get("reasons"):
            reasons.append("golden-decision-not-clean")
        return sorted(set(reasons))
    if (
        decision.get("evaluation_kind")
        != "verified-ocr-crop-golden"
    ):
        reasons.append("golden-evaluation-kind")
    try:
        samples = int(decision.get("samples", 0))
        baseline_accuracy = float(
            decision["baseline_exact_accuracy"]
        )
        candidate_accuracy = float(
            decision["candidate_exact_accuracy"]
        )
        baseline_false_accept = float(
            decision["baseline_false_accept_rate"]
        )
        candidate_false_accept = float(
            decision["candidate_false_accept_rate"]
        )
        baseline_error = float(
            decision["baseline_mean_character_error"]
        )
        candidate_error = float(
            decision["candidate_mean_character_error"]
        )
    except (KeyError, TypeError, ValueError):
        reasons.append("golden-metrics-missing")
        return sorted(set(reasons))
    numeric_metrics = (
        baseline_accuracy,
        candidate_accuracy,
        baseline_false_accept,
        candidate_false_accept,
        baseline_error,
        candidate_error,
    )
    if (
        not all(math.isfinite(value) for value in numeric_metrics)
        or not 0.0 <= baseline_accuracy <= 1.0
        or not 0.0 <= candidate_accuracy <= 1.0
        or not 0.0 <= baseline_false_accept <= 1.0
        or not 0.0 <= candidate_false_accept <= 1.0
        or baseline_error < 0.0
        or candidate_error < 0.0
    ):
        reasons.append("golden-metrics-invalid")
        return sorted(set(reasons))
    if (
        samples < MIN_GOLDEN_PROMOTION_SAMPLES
        or samples != int(golden.get("samples", 0))
    ):
        reasons.append("golden-sample-count")
    if candidate_accuracy < MIN_CANDIDATE_EXACT_ACCURACY:
        reasons.append("candidate-accuracy-floor")
    if candidate_accuracy < baseline_accuracy + 0.01:
        reasons.append("exact-accuracy-gain")
    if candidate_false_accept > min(
        baseline_false_accept,
        MAXIMUM_FALSE_ACCEPT_RATE,
    ):
        reasons.append("false-accept-regression")
    if candidate_error > baseline_error + 1e-9:
        reasons.append("character-error-regression")
    if decision.get("reasons") != []:
        reasons.append("golden-decision-not-clean")
    return sorted(set(reasons))


def _predict_crnn_session(session, image) -> tuple[str, float]:
    import numpy as np

    from .onnx_crnn import (
        CRNN_LABELS,
        ctc_greedy_decode,
        prepare_crnn_input,
    )

    tensor = prepare_crnn_input(image)
    if tensor is None:
        raise ValueError("Unreadable Golden OCR crop")
    input_name = session.get_inputs()[0].name
    started = time.perf_counter()
    output = session.run(None, {input_name: tensor})[0]
    latency_ms = (time.perf_counter() - started) * 1000
    logits = np.asarray(output)
    if logits.ndim == 3:
        logits = logits[0]
    if (
        logits.ndim != 2
        or logits.shape[1] != len(CRNN_LABELS) + 1
    ):
        raise ValueError(
            "Unexpected Golden CRNN output shape: "
            + str(tuple(np.asarray(output).shape))
        )
    raw, _confidence = ctc_greedy_decode(logits)
    return normalize_plate(raw), latency_ms


def compare_crnn_candidate_on_golden(
    baseline_model: Path,
    candidate_model: Path,
    golden: dict,
    *,
    session_factory=None,
) -> dict:
    """Compare active and candidate CRNNs on verified OCR-crop Golden rows."""

    if not golden.get("ready"):
        return {"promote": False, "reasons": ["golden-not-ready"]}
    golden_manifest_sha256 = str(
        golden.get("manifest_sha256", "")
    ).strip().upper()
    if (
        len(golden_manifest_sha256) != 64
        or any(
            character not in "0123456789ABCDEF"
            for character in golden_manifest_sha256
        )
    ):
        return {
            "promote": False,
            "reasons": ["golden-identity-missing"],
        }
    rows = list(golden.get("rows") or [])
    if not rows or any(
        row.get("media_kind") != "ocr-crop"
        for row in rows
    ):
        return {
            "promote": False,
            "reasons": ["ocr-crop-media-required"],
            "golden_manifest_sha256": golden_manifest_sha256,
        }
    if session_factory is None:
        import onnxruntime as ort

        def session_factory(path):
            return ort.InferenceSession(
                str(path),
                providers=["CPUExecutionProvider"],
            )

    baseline_session = session_factory(Path(baseline_model))
    candidate_session = session_factory(Path(candidate_model))
    baseline_rows = []
    candidate_rows = []
    import cv2
    import hashlib
    import numpy as np

    for row in rows:
        media_bytes = Path(row["media_path"]).read_bytes()
        expected_media_sha256 = str(
            row.get("sha256", "")
        ).strip().lower()
        if (
            len(expected_media_sha256) != 64
            or hashlib.sha256(media_bytes).hexdigest()
            != expected_media_sha256
        ):
            raise ValueError("Changed Golden OCR crop")
        image = cv2.imdecode(
            np.frombuffer(media_bytes, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError("Unreadable Golden OCR crop")
        baseline_text, baseline_latency = _predict_crnn_session(
            baseline_session,
            image,
        )
        candidate_text, candidate_latency = _predict_crnn_session(
            candidate_session,
            image,
        )
        common = {
            "expected_plate": row.get("expected_plate", ""),
            "slices": row.get("slices", []),
        }
        baseline_rows.append({
            **common,
            "predicted_plate": baseline_text,
            "latency_ms": baseline_latency,
        })
        candidate_rows.append({
            **common,
            "predicted_plate": candidate_text,
            "latency_ms": candidate_latency,
        })
    decision = evaluate_promotion(
        score_predictions(baseline_rows),
        score_predictions(candidate_rows),
    )
    return {
        **decision,
        "evaluation_kind": "verified-ocr-crop-golden",
        "golden_manifest_sha256": golden_manifest_sha256,
        "samples": len(rows),
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
    validation_metrics = (
        baseline_accuracy,
        candidate_accuracy,
        baseline_error,
        candidate_error,
    )
    if (
        not all(math.isfinite(value) for value in validation_metrics)
        or not 0.0 <= baseline_accuracy <= 1.0
        or not 0.0 <= candidate_accuracy <= 1.0
        or baseline_error < 0.0
        or candidate_error < 0.0
    ):
        reasons.append("validation-metrics-invalid")
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
    if result.get("training_rights_verified") is not True:
        reasons.append("training-rights-unverified")
    if not golden.get("ready"):
        reasons.append("golden-not-ready")
    golden_decision = result.get("golden_decision")
    evidence_reasons = validate_golden_decision_evidence(
        golden_decision,
        golden,
    )
    reasons.extend(evidence_reasons)
    if isinstance(golden_decision, dict) and not golden_decision.get(
        "promote"
    ):
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
        "training_rights_verified": bool(
            result.get("training_rights_verified", False)
        ),
        "golden": {
            "ready": bool(golden.get("ready")),
            "manifest_sha256": str(
                golden.get("manifest_sha256", "")
            ).upper(),
            "samples": int(golden.get("samples", 0)),
            "unique_plates": int(golden.get("unique_plates", 0)),
            "slice_counts": dict(golden.get("slice_counts", {})),
            "errors": list(golden.get("errors", [])),
        },
        "golden_decision": golden_decision,
    }
