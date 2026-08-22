"""Pass-level evidence for an end-to-end ANPR accuracy claim.

Crop-only OCR scores are useful diagnostics, but they cannot measure detector
misses, tracking failures, false accepts, or duplicate persisted events. This
module accepts only independently labelled vehicle passages evaluated through
one pinned production-pipeline revision. Invalid or incomplete evidence is
retained as a blocking reason instead of disappearing from a denominator.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
import math

from .plate_rules import normalize_plate, plausible_plate


VERIFIED_PRODUCTION_PASS = "verified-production-pass"
REQUIRED_PASS_SLICES = (
    "day",
    "night",
    "fast",
    "angle",
    "blur",
    "glare",
    "multi-vehicle",
)
MIN_READABLE_PASSAGES = 400
MIN_NEGATIVE_PASSAGES = 800
MIN_UNIQUE_PLATES = 100
MIN_CAMERAS = 3
MIN_SESSIONS = 3
MIN_PASSAGES_PER_SLICE = 30
MIN_PASSAGES_PER_PROVENANCE_GROUP = 30
TARGET_EXACT_ACCURACY = 0.99
MAX_FALSE_ACCEPT_RATE = 0.005
MAX_DUPLICATE_RATE = 0.005
MAX_SINGLE_PLATE_SHARE = 0.05
_Z_95_TWO_SIDED = 1.959963984540054
_MAX_IDENTIFIER_LENGTH = 256
_MAX_PLATE_VALUE_LENGTH = 64
_MAX_ACCEPTED_EVENTS = 16


def _is_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def wilson_interval(
    successes: int,
    trials: int,
    *,
    z: float = _Z_95_TWO_SIDED,
) -> tuple[float, float]:
    """Return a bounded two-sided Wilson score interval.

    Malformed counts deliberately return the maximally uncertain interval.
    Silently truncating floats here could otherwise turn malformed evidence
    into an apparently strong accuracy result.
    """

    if (
        not _is_count(successes)
        or not _is_count(trials)
        or trials <= 0
        or successes > trials
    ):
        return 0.0, 1.0
    try:
        z_value = float(z)
    except (TypeError, ValueError, OverflowError):
        return 0.0, 1.0
    if not math.isfinite(z_value) or z_value <= 0.0:
        return 0.0, 1.0
    proportion = successes / trials
    z_squared = z_value**2
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z_value
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


@dataclass(frozen=True)
class PassageMetrics:
    passages: int
    readable_passages: int
    negative_passages: int
    exact_passages: int
    missed_passages: int
    wrong_read_passages: int
    false_accept_passages: int
    duplicate_event_passages: int
    unique_plates: int
    cameras: int
    sessions: int
    slice_counts: dict[str, int]
    readable_slice_counts: dict[str, int]
    negative_slice_counts: dict[str, int]
    exact_slice_counts: dict[str, int]
    false_accept_slice_counts: dict[str, int]
    duplicate_slice_counts: dict[str, int]
    plate_counts: dict[str, int]
    readable_camera_counts: dict[str, int]
    negative_camera_counts: dict[str, int]
    readable_session_counts: dict[str, int]
    negative_session_counts: dict[str, int]
    pipeline_revisions: tuple[str, ...]
    invalid_reasons: tuple[str, ...]

    @property
    def exact_accuracy(self) -> float:
        return self.exact_passages / max(1, self.readable_passages)

    @property
    def miss_rate(self) -> float:
        return self.missed_passages / max(1, self.readable_passages)

    @property
    def wrong_read_rate(self) -> float:
        return self.wrong_read_passages / max(1, self.readable_passages)

    @property
    def false_accept_rate(self) -> float:
        return self.false_accept_passages / max(1, self.negative_passages)

    @property
    def duplicate_event_rate(self) -> float:
        return self.duplicate_event_passages / max(1, self.passages)

    @property
    def exact_accuracy_interval(self) -> tuple[float, float]:
        return wilson_interval(self.exact_passages, self.readable_passages)

    @property
    def false_accept_interval(self) -> tuple[float, float]:
        return wilson_interval(
            self.false_accept_passages,
            self.negative_passages,
        )

    @property
    def duplicate_event_interval(self) -> tuple[float, float]:
        return wilson_interval(
            self.duplicate_event_passages,
            self.passages,
        )


def _slice_counter() -> dict[str, int]:
    return {label: 0 for label in REQUIRED_PASS_SLICES}


def _empty_metrics(invalid_reasons: tuple[str, ...]) -> PassageMetrics:
    return PassageMetrics(
        passages=0,
        readable_passages=0,
        negative_passages=0,
        exact_passages=0,
        missed_passages=0,
        wrong_read_passages=0,
        false_accept_passages=0,
        duplicate_event_passages=0,
        unique_plates=0,
        cameras=0,
        sessions=0,
        slice_counts=_slice_counter(),
        readable_slice_counts=_slice_counter(),
        negative_slice_counts=_slice_counter(),
        exact_slice_counts=_slice_counter(),
        false_accept_slice_counts=_slice_counter(),
        duplicate_slice_counts=_slice_counter(),
        plate_counts={},
        readable_camera_counts={},
        negative_camera_counts={},
        readable_session_counts={},
        negative_session_counts={},
        pipeline_revisions=(),
        invalid_reasons=invalid_reasons,
    )


def _identifier(value: object) -> str:
    if not isinstance(value, str):
        return ""
    if not value or value != value.strip() or len(value) > _MAX_IDENTIFIER_LENGTH:
        return ""
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return ""
    return value


def _evidence_digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        return ""
    lowered = value.lower()
    if any(character not in "0123456789abcdef" for character in lowered):
        return ""
    return lowered


def _accepted_plates(
    raw_events: object,
    passage_id: str,
) -> tuple[list[str], list[str]]:
    if not isinstance(raw_events, list):
        return [], [f"accepted-events-must-be-list:{passage_id}"]
    if len(raw_events) > _MAX_ACCEPTED_EVENTS:
        return [], [f"too-many-accepted-events:{passage_id}"]

    accepted: list[str] = []
    errors: list[str] = []
    for index, raw in enumerate(raw_events):
        values: list[str] = []
        type_error = False
        if isinstance(raw, str):
            values = [raw]
        elif isinstance(raw, dict):
            for key in ("plate_norm", "plate"):
                if key not in raw:
                    continue
                value = raw[key]
                if value is None:
                    continue
                if not isinstance(value, str):
                    errors.append(
                        f"invalid-accepted-event-type:{passage_id}:{index}"
                    )
                    type_error = True
                    break
                if value:
                    values.append(value)
        else:
            errors.append(f"invalid-accepted-event-type:{passage_id}:{index}")
            continue

        if type_error:
            continue
        if not values:
            errors.append(f"empty-accepted-event:{passage_id}:{index}")
            continue
        if any(len(value) > _MAX_PLATE_VALUE_LENGTH for value in values):
            errors.append(f"oversized-accepted-event:{passage_id}:{index}")
            continue
        normalized_values = {normalize_plate(value) for value in values}
        if len(normalized_values) != 1:
            errors.append(f"conflicting-accepted-event:{passage_id}:{index}")
            continue
        normalized = normalized_values.pop()
        if not plausible_plate(normalized):
            errors.append(f"invalid-accepted-event:{passage_id}:{index}")
            continue
        accepted.append(normalized)
    return accepted, errors


def _labels(raw_labels: object, passage_id: str) -> tuple[set[str], list[str]]:
    if not isinstance(raw_labels, list) or not raw_labels:
        return set(), [f"slices-must-be-nonempty-list:{passage_id}"]
    if len(raw_labels) > len(REQUIRED_PASS_SLICES):
        return set(), [f"too-many-slices:{passage_id}"]
    if any(not isinstance(value, str) for value in raw_labels):
        return set(), [f"invalid-slices:{passage_id}"]
    normalized = [value.strip().lower() for value in raw_labels]
    labels = set(normalized)
    if (
        any(not value for value in normalized)
        or len(labels) != len(normalized)
        or any(label not in REQUIRED_PASS_SLICES for label in labels)
        or len(labels.intersection({"day", "night"})) != 1
    ):
        return set(), [f"invalid-slices:{passage_id}"]
    return labels, []


def score_passages(rows: object) -> PassageMetrics:
    """Score independent production passages without hiding bad rows.

    ``evidence_digest`` must be a unique SHA-256 digest of the immutable
    evidence artifact for that passage. It catches accidental reuse under a
    new ``passage_id``; it is not a substitute for an independently controlled
    annotation and sampling process.
    """

    if not isinstance(rows, list):
        return _empty_metrics(("passages-must-be-list",))

    seen_passages: set[str] = set()
    seen_evidence: set[str] = set()
    cameras: set[str] = set()
    sessions: set[str] = set()
    pipeline_revisions: set[str] = set()
    slice_counts = _slice_counter()
    readable_slice_counts = _slice_counter()
    negative_slice_counts = _slice_counter()
    exact_slice_counts = _slice_counter()
    false_accept_slice_counts = _slice_counter()
    duplicate_slice_counts = _slice_counter()
    plate_counts: dict[str, int] = defaultdict(int)
    readable_camera_counts: dict[str, int] = defaultdict(int)
    negative_camera_counts: dict[str, int] = defaultdict(int)
    readable_session_counts: dict[str, int] = defaultdict(int)
    negative_session_counts: dict[str, int] = defaultdict(int)
    invalid_reasons: list[str] = []
    readable = 0
    negatives = 0
    exact = 0
    missed = 0
    wrong = 0
    false_accepts = 0
    duplicates = 0
    valid_passages = 0

    for index, raw in enumerate(rows):
        if not isinstance(raw, dict):
            invalid_reasons.append(f"passage-must-be-object:{index}")
            continue

        passage_id = _identifier(raw.get("passage_id"))
        if not passage_id:
            invalid_reasons.append(f"invalid-passage-id:{index}")
            continue
        if passage_id in seen_passages:
            invalid_reasons.append(f"duplicate-passage-id:{passage_id}")
            continue
        seen_passages.add(passage_id)

        evidence_digest = _evidence_digest(raw.get("evidence_digest"))
        if not evidence_digest:
            invalid_reasons.append(f"invalid-evidence-digest:{passage_id}")
            continue
        if evidence_digest in seen_evidence:
            invalid_reasons.append(f"duplicate-evidence-digest:{passage_id}")
            continue
        seen_evidence.add(evidence_digest)

        camera_id = _identifier(raw.get("camera_id"))
        session_id = _identifier(raw.get("session_id"))
        pipeline_revision = _identifier(raw.get("pipeline_revision"))
        row_errors: list[str] = []
        if not camera_id or not session_id:
            row_errors.append(f"missing-passage-provenance:{passage_id}")
        if not pipeline_revision:
            row_errors.append(f"missing-pipeline-revision:{passage_id}")
        evidence_kind = raw.get("evidence_kind")
        if (
            not isinstance(evidence_kind, str)
            or evidence_kind != VERIFIED_PRODUCTION_PASS
        ):
            row_errors.append(f"invalid-evidence-kind:{passage_id}")

        labels, label_errors = _labels(raw.get("slices"), passage_id)
        row_errors.extend(label_errors)
        readable_value = raw.get("readable")
        if not isinstance(readable_value, bool):
            row_errors.append(f"readable-must-be-boolean:{passage_id}")

        expected_raw = raw.get("expected_plate")
        expected = ""
        if (
            not isinstance(expected_raw, str)
            or len(expected_raw) > _MAX_PLATE_VALUE_LENGTH
        ):
            row_errors.append(f"invalid-expected-label:{passage_id}")
        else:
            expected = normalize_plate(expected_raw)
            if readable_value is True and not plausible_plate(expected):
                row_errors.append(f"invalid-expected-label:{passage_id}")
            if readable_value is False and expected_raw.strip():
                row_errors.append(f"negative-must-have-empty-label:{passage_id}")

        accepted, event_errors = _accepted_plates(
            raw.get("accepted_events"),
            passage_id,
        )
        row_errors.extend(event_errors)
        if row_errors:
            invalid_reasons.extend(row_errors)
            continue

        valid_passages += 1
        cameras.add(camera_id)
        sessions.add(session_id)
        pipeline_revisions.add(pipeline_revision)
        duplicate = len(accepted) > 1
        if duplicate:
            duplicates += 1
        for label in labels:
            slice_counts[label] += 1
            if duplicate:
                duplicate_slice_counts[label] += 1

        if readable_value:
            readable += 1
            plate_counts[expected] += 1
            readable_camera_counts[camera_id] += 1
            readable_session_counts[session_id] += 1
            for label in labels:
                readable_slice_counts[label] += 1
            if len(accepted) == 1 and accepted[0] == expected:
                exact += 1
                for label in labels:
                    exact_slice_counts[label] += 1
            elif not accepted:
                missed += 1
            else:
                wrong += 1
        else:
            negatives += 1
            negative_camera_counts[camera_id] += 1
            negative_session_counts[session_id] += 1
            for label in labels:
                negative_slice_counts[label] += 1
            if accepted:
                false_accepts += 1
                for label in labels:
                    false_accept_slice_counts[label] += 1

    return PassageMetrics(
        passages=valid_passages,
        readable_passages=readable,
        negative_passages=negatives,
        exact_passages=exact,
        missed_passages=missed,
        wrong_read_passages=wrong,
        false_accept_passages=false_accepts,
        duplicate_event_passages=duplicates,
        unique_plates=len(plate_counts),
        cameras=len(cameras),
        sessions=len(sessions),
        slice_counts=dict(slice_counts),
        readable_slice_counts=dict(readable_slice_counts),
        negative_slice_counts=dict(negative_slice_counts),
        exact_slice_counts=dict(exact_slice_counts),
        false_accept_slice_counts=dict(false_accept_slice_counts),
        duplicate_slice_counts=dict(duplicate_slice_counts),
        plate_counts=dict(plate_counts),
        readable_camera_counts=dict(readable_camera_counts),
        negative_camera_counts=dict(negative_camera_counts),
        readable_session_counts=dict(readable_session_counts),
        negative_session_counts=dict(negative_session_counts),
        pipeline_revisions=tuple(sorted(pipeline_revisions)),
        invalid_reasons=tuple(sorted(set(invalid_reasons))),
    )


def _valid_count_mapping(value: object) -> bool:
    return isinstance(value, Mapping) and all(
        isinstance(key, str) and bool(key) and _is_count(count)
        for key, count in value.items()
    )


def _metric_integrity_reasons(metrics: PassageMetrics) -> list[str]:
    scalar_names = (
        "passages",
        "readable_passages",
        "negative_passages",
        "exact_passages",
        "missed_passages",
        "wrong_read_passages",
        "false_accept_passages",
        "duplicate_event_passages",
        "unique_plates",
        "cameras",
        "sessions",
    )
    mapping_names = (
        "slice_counts",
        "readable_slice_counts",
        "negative_slice_counts",
        "exact_slice_counts",
        "false_accept_slice_counts",
        "duplicate_slice_counts",
        "plate_counts",
        "readable_camera_counts",
        "negative_camera_counts",
        "readable_session_counts",
        "negative_session_counts",
    )
    if any(not _is_count(getattr(metrics, name, None)) for name in scalar_names):
        return ["invalid-metric-counts"]
    if any(
        not _valid_count_mapping(getattr(metrics, name, None))
        for name in mapping_names
    ):
        return ["invalid-metric-mappings"]
    if (
        not isinstance(metrics.pipeline_revisions, tuple)
        or any(not _identifier(value) for value in metrics.pipeline_revisions)
        or len(set(metrics.pipeline_revisions)) != len(metrics.pipeline_revisions)
        or not isinstance(metrics.invalid_reasons, tuple)
        or any(not isinstance(value, str) for value in metrics.invalid_reasons)
    ):
        return ["invalid-metric-provenance"]

    reasons: list[str] = []
    if metrics.passages != metrics.readable_passages + metrics.negative_passages:
        reasons.append("inconsistent-passage-counts")
    if (
        metrics.exact_passages
        + metrics.missed_passages
        + metrics.wrong_read_passages
        != metrics.readable_passages
    ):
        reasons.append("inconsistent-readable-outcomes")
    if metrics.false_accept_passages > metrics.negative_passages:
        reasons.append("inconsistent-false-accept-count")
    if metrics.duplicate_event_passages > metrics.passages:
        reasons.append("inconsistent-duplicate-count")
    if (
        metrics.unique_plates != len(metrics.plate_counts)
        or sum(metrics.plate_counts.values()) != metrics.readable_passages
    ):
        reasons.append("inconsistent-plate-counts")
    if (
        sum(metrics.readable_camera_counts.values())
        != metrics.readable_passages
        or sum(metrics.negative_camera_counts.values())
        != metrics.negative_passages
        or metrics.cameras
        != len(
            set(metrics.readable_camera_counts)
            | set(metrics.negative_camera_counts)
        )
    ):
        reasons.append("inconsistent-camera-counts")
    if (
        sum(metrics.readable_session_counts.values())
        != metrics.readable_passages
        or sum(metrics.negative_session_counts.values())
        != metrics.negative_passages
        or metrics.sessions
        != len(
            set(metrics.readable_session_counts)
            | set(metrics.negative_session_counts)
        )
    ):
        reasons.append("inconsistent-session-counts")

    slice_mappings = (
        metrics.slice_counts,
        metrics.readable_slice_counts,
        metrics.negative_slice_counts,
        metrics.exact_slice_counts,
        metrics.false_accept_slice_counts,
        metrics.duplicate_slice_counts,
    )
    if any(set(mapping) != set(REQUIRED_PASS_SLICES) for mapping in slice_mappings):
        reasons.append("inconsistent-slice-keys")
    else:
        for label in REQUIRED_PASS_SLICES:
            total = metrics.slice_counts[label]
            readable = metrics.readable_slice_counts[label]
            negative = metrics.negative_slice_counts[label]
            if total != readable + negative or total > metrics.passages:
                reasons.append(f"inconsistent-slice-count:{label}")
            if metrics.exact_slice_counts[label] > min(
                readable,
                metrics.exact_passages,
            ):
                reasons.append(f"inconsistent-exact-slice:{label}")
            if metrics.false_accept_slice_counts[label] > min(
                negative,
                metrics.false_accept_passages,
            ):
                reasons.append(f"inconsistent-false-accept-slice:{label}")
            if metrics.duplicate_slice_counts[label] > min(
                total,
                metrics.duplicate_event_passages,
            ):
                reasons.append(f"inconsistent-duplicate-slice:{label}")
    return reasons


def _qualified_groups(counts: Mapping[str, int]) -> int:
    return sum(
        count >= MIN_PASSAGES_PER_PROVENANCE_GROUP
        for count in counts.values()
    )


def _decision_payload(
    metrics: PassageMetrics,
    *,
    evaluation_kind: object,
    reasons: list[str],
) -> dict:
    exact_lower, exact_upper = metrics.exact_accuracy_interval
    false_lower, false_upper = metrics.false_accept_interval
    duplicate_lower, duplicate_upper = metrics.duplicate_event_interval
    return {
        "claim_ready": not reasons,
        "reasons": sorted(set(reasons)),
        "evaluation_kind": (
            evaluation_kind if isinstance(evaluation_kind, str) else ""
        ),
        "pipeline_revision": (
            metrics.pipeline_revisions[0]
            if len(metrics.pipeline_revisions) == 1
            else ""
        ),
        "passages": metrics.passages,
        "readable_passages": metrics.readable_passages,
        "negative_passages": metrics.negative_passages,
        "unique_plates": metrics.unique_plates,
        "cameras": metrics.cameras,
        "sessions": metrics.sessions,
        "exact_passages": metrics.exact_passages,
        "exact_accuracy": round(metrics.exact_accuracy, 6),
        "exact_accuracy_ci95": [round(exact_lower, 6), round(exact_upper, 6)],
        "missed_passages": metrics.missed_passages,
        "miss_rate": round(metrics.miss_rate, 6),
        "wrong_read_passages": metrics.wrong_read_passages,
        "wrong_read_rate": round(metrics.wrong_read_rate, 6),
        "false_accept_passages": metrics.false_accept_passages,
        "false_accept_rate": round(metrics.false_accept_rate, 6),
        "false_accept_ci95": [round(false_lower, 6), round(false_upper, 6)],
        "duplicate_event_passages": metrics.duplicate_event_passages,
        "duplicate_event_rate": round(metrics.duplicate_event_rate, 6),
        "duplicate_event_ci95": [
            round(duplicate_lower, 6),
            round(duplicate_upper, 6),
        ],
        "slice_counts": dict(metrics.slice_counts),
        "readable_slice_counts": dict(metrics.readable_slice_counts),
        "negative_slice_counts": dict(metrics.negative_slice_counts),
    }


def evaluate_accuracy_claim(
    metrics: PassageMetrics,
    *,
    evaluation_kind: str,
) -> dict:
    """Fail closed unless evidence statistically supports the 99% claim.

    Confidence intervals are marginal two-sided 95% Wilson intervals. Slice
    checks additionally prevent aggregate results from hiding a known weak
    operating condition; they are coverage/observed-rate guards rather than
    separate 99%-confidence claims.
    """

    if not isinstance(metrics, PassageMetrics):
        safe = _empty_metrics(("invalid-metrics-object",))
        return _decision_payload(
            safe,
            evaluation_kind=evaluation_kind,
            reasons=["invalid-metrics-object"],
        )

    integrity_reasons = _metric_integrity_reasons(metrics)
    if integrity_reasons:
        safe_reasons = list(metrics.invalid_reasons) + integrity_reasons
        safe = _empty_metrics(tuple(sorted(set(safe_reasons))))
        return _decision_payload(
            safe,
            evaluation_kind=evaluation_kind,
            reasons=safe_reasons,
        )

    reasons = list(metrics.invalid_reasons)
    if evaluation_kind != VERIFIED_PRODUCTION_PASS:
        reasons.append("production-pass-evidence-required")
    if len(metrics.pipeline_revisions) != 1:
        reasons.append("single-pipeline-revision-required")
    if metrics.readable_passages < MIN_READABLE_PASSAGES:
        reasons.append("insufficient-readable-passages")
    if metrics.negative_passages < MIN_NEGATIVE_PASSAGES:
        reasons.append("insufficient-negative-passages")
    if metrics.unique_plates < MIN_UNIQUE_PLATES:
        reasons.append("insufficient-unique-plates")
    if metrics.cameras < MIN_CAMERAS:
        reasons.append("insufficient-cameras")
    if metrics.sessions < MIN_SESSIONS:
        reasons.append("insufficient-sessions")
    if _qualified_groups(metrics.readable_camera_counts) < MIN_CAMERAS:
        reasons.append("insufficient-readable-camera-coverage")
    if _qualified_groups(metrics.negative_camera_counts) < MIN_CAMERAS:
        reasons.append("insufficient-negative-camera-coverage")
    if _qualified_groups(metrics.readable_session_counts) < MIN_SESSIONS:
        reasons.append("insufficient-readable-session-coverage")
    if _qualified_groups(metrics.negative_session_counts) < MIN_SESSIONS:
        reasons.append("insufficient-negative-session-coverage")

    if metrics.readable_passages:
        largest_plate_share = max(
            metrics.plate_counts.values(),
            default=0,
        ) / metrics.readable_passages
        if largest_plate_share > MAX_SINGLE_PLATE_SHARE:
            reasons.append("excessive-single-plate-concentration")

    for label in REQUIRED_PASS_SLICES:
        readable_count = metrics.readable_slice_counts[label]
        negative_count = metrics.negative_slice_counts[label]
        total_count = metrics.slice_counts[label]
        if readable_count < MIN_PASSAGES_PER_SLICE:
            reasons.append(f"insufficient-readable-slice:{label}")
        if negative_count < MIN_PASSAGES_PER_SLICE:
            reasons.append(f"insufficient-negative-slice:{label}")
        if (
            readable_count
            and metrics.exact_slice_counts[label] / readable_count
            < TARGET_EXACT_ACCURACY
        ):
            reasons.append(f"slice-exact-accuracy-below-target:{label}")
        if (
            negative_count
            and metrics.false_accept_slice_counts[label] / negative_count
            > MAX_FALSE_ACCEPT_RATE
        ):
            reasons.append(f"slice-false-accept-above-target:{label}")
        if (
            total_count
            and metrics.duplicate_slice_counts[label] / total_count
            > MAX_DUPLICATE_RATE
        ):
            reasons.append(f"slice-duplicate-above-target:{label}")

    exact_lower, _ = metrics.exact_accuracy_interval
    _, false_upper = metrics.false_accept_interval
    _, duplicate_upper = metrics.duplicate_event_interval
    if exact_lower < TARGET_EXACT_ACCURACY:
        reasons.append("exact-accuracy-confidence-bound")
    if false_upper > MAX_FALSE_ACCEPT_RATE:
        reasons.append("false-accept-confidence-bound")
    if duplicate_upper > MAX_DUPLICATE_RATE:
        reasons.append("duplicate-confidence-bound")

    return _decision_payload(
        metrics,
        evaluation_kind=evaluation_kind,
        reasons=reasons,
    )
