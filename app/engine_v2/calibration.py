"""Fail-closed calibration for Engine V2 temporal plate recognition.

Calibration consumes dense, precomputed OCR traces. A trace contains every
usable crop for a labelled tracker episode, not only the crops selected by the
current scheduler. This lets candidate policies be replayed without running a
model again and makes OCR calls per track part of the optimization target.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

from .tcam import (
    PlateEvidenceAccumulator,
    TemporalFusionConfig,
    TrackRecognitionSession,
)
from .types import OCRResult
from .validator import IranianPlateValidator

CALIBRATION_SCHEMA = "bcvision.anpr.tcam-calibration/v1"
CALIBRATION_SPLITS = frozenset({"train", "holdout"})
CALIBRATION_PROFILES = frozenset({"day", "night"})
STATIC_CONFIDENCE_THRESHOLDS = (
    0.50,
    0.60,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.93,
    0.95,
    0.97,
    0.99,
    0.995,
    0.997,
    0.998,
    0.999,
)


@dataclass(frozen=True, slots=True)
class CalibrationObservation:
    seq: int
    ts: float
    text: str
    confidence: float
    quality: float
    plate_width: int
    plate_height: int
    valid: bool = True
    character_confidences: tuple[float, ...] = ()
    candidates: tuple[Mapping[str, object], ...] = ()

    @property
    def bbox_area(self) -> int:
        return self.plate_width * self.plate_height

    def result(self) -> OCRResult:
        metadata: dict[str, object] = {}
        if self.candidates:
            metadata["candidates"] = [dict(candidate) for candidate in self.candidates]
        return OCRResult(
            self.text,
            self.confidence,
            self.valid,
            self.character_confidences,
            metadata,
        )


@dataclass(frozen=True, slots=True)
class CalibrationTrack:
    track_id: str
    split: str
    profile: str
    expected_plate: str | None
    observations: tuple[CalibrationObservation, ...]


@dataclass(frozen=True, slots=True)
class CalibrationDataset:
    dataset_id: str
    tracks: tuple[CalibrationTrack, ...]
    fingerprint_sha256: str
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrackReplay:
    track_id: str
    expected_plate: str | None
    emitted_plate: str | None
    ocr_calls: int
    latency_seconds: float | None
    finalization_reason: str | None


@dataclass(frozen=True, slots=True)
class CalibrationMetrics:
    tracks: int
    positive_tracks: int
    negative_tracks: int
    emitted_events: int
    exact_matches: int
    wrong_events: int
    missed_events: int
    false_accepts: int
    exact_accuracy: float
    event_precision: float
    event_recall: float
    false_accept_rate: float
    wrong_event_rate: float
    mean_character_error_rate: float
    mean_ocr_calls_per_track: float
    p95_ocr_calls_per_track: float
    mean_latency_seconds: float | None


@dataclass(frozen=True, slots=True)
class CalibrationRequirements:
    target_exact_accuracy: float = 0.99
    minimum_event_recall: float = 0.99
    minimum_event_precision: float = 0.995
    maximum_false_accept_rate: float = 0.001
    maximum_wrong_event_rate: float = 0.001
    maximum_mean_character_error_rate: float = 0.01
    minimum_train_tracks: int = 50
    minimum_holdout_tracks: int = 50
    maximum_grid_candidates: int = 50_000

    def __post_init__(self) -> None:
        for name in (
            "target_exact_accuracy",
            "minimum_event_recall",
            "minimum_event_precision",
            "maximum_false_accept_rate",
            "maximum_wrong_event_rate",
            "maximum_mean_character_error_rate",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and within 0..1")
        for name in (
            "minimum_train_tracks",
            "minimum_holdout_tracks",
            "maximum_grid_candidates",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ProfileCalibration:
    profile: str
    config: TemporalFusionConfig | None
    train_metrics: CalibrationMetrics | None
    holdout_metrics: CalibrationMetrics | None
    candidates_evaluated: int
    valid: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    schema: str
    dataset_id: str
    dataset_fingerprint_sha256: str
    profiles: tuple[ProfileCalibration, ...]
    valid: bool
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StaticConfidenceBin:
    lower_bound: float
    upper_bound: float
    observations: int
    mean_confidence: float | None
    exact_accuracy: float | None


@dataclass(frozen=True, slots=True)
class StaticThresholdMetrics:
    threshold: float
    accepted: int
    coverage: float
    selective_exact_accuracy: float | None
    wrong_accept_rate: float


@dataclass(frozen=True, slots=True)
class StaticOCRSlice:
    scope: str
    observations: int
    exact_matches: int
    exact_accuracy: float
    mean_character_error_rate: float
    brier_score: float
    expected_calibration_error: float
    confidence_bins: tuple[StaticConfidenceBin, ...]
    thresholds: tuple[StaticThresholdMetrics, ...]


@dataclass(frozen=True, slots=True)
class StaticOCRReport:
    schema: str
    dataset_id: str
    dataset_fingerprint_sha256: str
    overall: StaticOCRSlice
    slices: tuple[StaticOCRSlice, ...]
    promotion_eligible: bool
    limitations: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_OBSERVATION_KEYS = frozenset(
    {
        "seq",
        "ts",
        "text",
        "confidence",
        "quality",
        "plate_width",
        "plate_height",
        "valid",
        "character_confidences",
        "candidates",
    }
)
_TRACK_KEYS = frozenset(
    {"track_id", "split", "profile", "expected_plate", "observations"}
)
_DATASET_KEYS = frozenset({"schema", "dataset_id", "label_scope", "tracks", "metadata"})
_TUNABLE_FIELDS = frozenset(field.name for field in fields(TemporalFusionConfig))


def load_calibration_dataset(path: str | Path) -> CalibrationDataset:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("calibration dataset root must be an object")
    _reject_unknown(payload, _DATASET_KEYS, "dataset")
    if payload.get("schema") != CALIBRATION_SCHEMA:
        raise ValueError(f"calibration schema must be {CALIBRATION_SCHEMA!r}")
    if payload.get("label_scope") != "exhaustive":
        raise ValueError("calibration requires exhaustive labels, including negatives")
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("dataset_id cannot be empty")
    raw_tracks = payload.get("tracks")
    if not isinstance(raw_tracks, list) or not raw_tracks:
        raise ValueError("tracks must be a non-empty array")
    validator = IranianPlateValidator()
    tracks = tuple(_parse_track(item, validator) for item in raw_tracks)
    identifiers = [track.track_id for track in tracks]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("track_id values must be unique")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("calibration dataset metadata must be an object")
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return CalibrationDataset(
        dataset_id,
        tracks,
        hashlib.sha256(canonical).hexdigest(),
        dict(metadata),
    )


def _parse_track(raw: object, validator: IranianPlateValidator) -> CalibrationTrack:
    if not isinstance(raw, dict):
        raise TypeError("each calibration track must be an object")
    _reject_unknown(raw, _TRACK_KEYS, "track")
    track_id = str(raw.get("track_id", "")).strip()
    if not track_id:
        raise ValueError("track_id cannot be empty")
    split = str(raw.get("split", "")).strip().lower()
    if split not in CALIBRATION_SPLITS:
        raise ValueError(f"track {track_id!r} split must be train or holdout")
    profile = str(raw.get("profile", "")).strip().lower()
    if profile not in CALIBRATION_PROFILES:
        raise ValueError(f"track {track_id!r} profile must be day or night")
    raw_expected = raw.get("expected_plate")
    expected: str | None
    if raw_expected is None:
        expected = None
    else:
        validation = validator.validate(str(raw_expected))
        if not validation.valid:
            raise ValueError(f"track {track_id!r} expected_plate is invalid")
        expected = validation.normalized
    raw_observations = raw.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ValueError(f"track {track_id!r} observations must be non-empty")
    observations = tuple(
        _parse_observation(item, track_id) for item in raw_observations
    )
    if any(
        next_.seq <= current.seq for current, next_ in itertools.pairwise(observations)
    ):
        raise ValueError(f"track {track_id!r} observation seq values must increase")
    if any(
        next_.ts < current.ts for current, next_ in itertools.pairwise(observations)
    ):
        raise ValueError(f"track {track_id!r} observation timestamps must not decrease")
    return CalibrationTrack(track_id, split, profile, expected, observations)


def _parse_observation(raw: object, track_id: str) -> CalibrationObservation:
    if not isinstance(raw, dict):
        raise TypeError(f"track {track_id!r} observation must be an object")
    _reject_unknown(raw, _OBSERVATION_KEYS, f"track {track_id!r} observation")
    seq = _integer(raw.get("seq"), "seq")
    ts = _finite(raw.get("ts"), "ts")
    confidence = _probability(raw.get("confidence"), "confidence")
    quality = _probability(raw.get("quality"), "quality")
    width = _positive_integer(raw.get("plate_width"), "plate_width")
    height = _positive_integer(raw.get("plate_height"), "plate_height")
    raw_characters = raw.get("character_confidences", ())
    if not isinstance(raw_characters, (list, tuple)):
        raise TypeError("character_confidences must be an array")
    character_confidences = tuple(
        _probability(value, "character confidence") for value in raw_characters
    )
    raw_candidates = raw.get("candidates", ())
    if not isinstance(raw_candidates, (list, tuple)) or any(
        not isinstance(candidate, dict) for candidate in raw_candidates
    ):
        raise ValueError("candidates must be an array of objects")
    return CalibrationObservation(
        seq=seq,
        ts=ts,
        text=str(raw.get("text", "")),
        confidence=confidence,
        quality=quality,
        plate_width=width,
        plate_height=height,
        valid=bool(raw.get("valid", True)),
        character_confidences=character_confidences,
        candidates=tuple(raw_candidates),
    )


def replay_track(
    track: CalibrationTrack,
    config: TemporalFusionConfig,
    validator: IranianPlateValidator | None = None,
) -> TrackReplay:
    session = TrackRecognitionSession(
        PlateEvidenceAccumulator(validator or IranianPlateValidator(), config),
        profile_name=track.profile,
    )
    read_sequences: set[int] = set()
    emitted: str | None = None
    finalized_ts: float | None = None
    first_ts = track.observations[0].ts

    def read(observation: CalibrationObservation) -> None:
        nonlocal finalized_ts
        session.reserve_ocr(
            seq=observation.seq,
            ts=observation.ts,
            quality=observation.quality,
            bbox_area=observation.bbox_area,
        )
        read_sequences.add(observation.seq)
        session.observe(
            observation.result(),
            quality=observation.quality,
            seq=observation.seq,
            ts=observation.ts,
            plate_width=observation.plate_width,
            plate_height=observation.plate_height,
        )
        if session.should_finalize(ts=observation.ts):
            reason = (
                "audit_complete" if session.audit_attempts else "soft_lock_hold_elapsed"
            )
            session.finalize(reason=reason)
            finalized_ts = observation.ts

    for observation in track.observations:
        if session.decision.finalized:
            break
        schedule = session.should_schedule_ocr(
            seq=observation.seq,
            ts=observation.ts,
            quality=observation.quality,
            bbox_area=observation.bbox_area,
            plate_width=observation.plate_width,
            plate_height=observation.plate_height,
        )
        if schedule.run_ocr:
            read(observation)
            continue
        if session.should_finalize(ts=observation.ts):
            session.finalize(reason="soft_lock_hold_elapsed")
            finalized_ts = observation.ts

    last = track.observations[-1]
    if not session.decision.finalized:
        if last.seq not in read_sequences:
            schedule = session.should_schedule_ocr(
                seq=last.seq,
                ts=last.ts,
                quality=last.quality,
                bbox_area=last.bbox_area,
                plate_width=last.plate_width,
                plate_height=last.plate_height,
                near_exit=True,
            )
            if schedule.run_ocr:
                read(last)
        if session.decision.soft_locked:
            session.finalize(reason="track_exit")
            finalized_ts = last.ts

    if session.decision.finalized and session.claim_event():
        emitted = session.decision.text
    return TrackReplay(
        track_id=track.track_id,
        expected_plate=track.expected_plate,
        emitted_plate=emitted,
        ocr_calls=session.attempts,
        latency_seconds=(
            max(0.0, finalized_ts - first_ts) if finalized_ts is not None else None
        ),
        finalization_reason=session.finalization_reason,
    )


def evaluate_config(
    tracks: Sequence[CalibrationTrack],
    config: TemporalFusionConfig,
) -> CalibrationMetrics:
    if not tracks:
        raise ValueError("at least one calibration track is required")
    validator = IranianPlateValidator()
    replays = [replay_track(track, config, validator) for track in tracks]
    positives = [replay for replay in replays if replay.expected_plate is not None]
    negatives = [replay for replay in replays if replay.expected_plate is None]
    exact = sum(replay.emitted_plate == replay.expected_plate for replay in positives)
    wrong = sum(
        replay.emitted_plate is not None
        and replay.emitted_plate != replay.expected_plate
        for replay in positives
    )
    missed = sum(replay.emitted_plate is None for replay in positives)
    false_accepts = sum(replay.emitted_plate is not None for replay in negatives)
    emitted = exact + wrong + false_accepts
    character_errors = [
        _normalized_edit_distance(
            replay.expected_plate or "", replay.emitted_plate or ""
        )
        for replay in positives
    ]
    calls = sorted(replay.ocr_calls for replay in replays)
    latencies = [
        replay.latency_seconds
        for replay in replays
        if replay.latency_seconds is not None
    ]
    return CalibrationMetrics(
        tracks=len(replays),
        positive_tracks=len(positives),
        negative_tracks=len(negatives),
        emitted_events=emitted,
        exact_matches=exact,
        wrong_events=wrong,
        missed_events=missed,
        false_accepts=false_accepts,
        exact_accuracy=exact / max(1, len(positives)),
        event_precision=exact / max(1, emitted),
        event_recall=exact / max(1, len(positives)),
        false_accept_rate=false_accepts / max(1, len(negatives)),
        wrong_event_rate=wrong / max(1, len(positives)),
        mean_character_error_rate=sum(character_errors) / max(1, len(character_errors)),
        mean_ocr_calls_per_track=sum(calls) / max(1, len(calls)),
        p95_ocr_calls_per_track=float(
            calls[min(len(calls) - 1, math.ceil(0.95 * len(calls)) - 1)]
        ),
        mean_latency_seconds=(sum(latencies) / len(latencies) if latencies else None),
    )


def analyze_static_ocr(
    dataset: CalibrationDataset,
    *,
    confidence_thresholds: Sequence[float] = STATIC_CONFIDENCE_THRESHOLDS,
    confidence_bin_count: int = 10,
) -> StaticOCRReport:
    """Measure OCR confidence calibration on labelled positive observations.

    This report is intentionally non-promotable. Static crops can measure
    exact-match/CER and whether confidence is over- or under-confident, but
    cannot prove tracking behavior or a false-accept rate on camera negatives.
    """

    if int(confidence_bin_count) < 1:
        raise ValueError("confidence_bin_count must be positive")
    thresholds = tuple(
        sorted(
            {
                _probability(value, "static confidence threshold")
                for value in confidence_thresholds
            }
        )
    )
    if not thresholds:
        raise ValueError("at least one static confidence threshold is required")
    positives = [track for track in dataset.tracks if track.expected_plate is not None]
    if not positives:
        raise ValueError("static OCR analysis requires labelled positive tracks")

    grouped: dict[tuple[str, str], list[CalibrationTrack]] = {}
    for track in positives:
        grouped.setdefault((track.profile, track.split), []).append(track)
    overall = _static_ocr_slice(
        "overall",
        positives,
        thresholds=thresholds,
        confidence_bin_count=int(confidence_bin_count),
    )
    slices = tuple(
        _static_ocr_slice(
            f"{profile}/{split}",
            tracks,
            thresholds=thresholds,
            confidence_bin_count=int(confidence_bin_count),
        )
        for (profile, split), tracks in sorted(grouped.items())
    )
    return StaticOCRReport(
        schema="bcvision.anpr.static-ocr-calibration-report/v1",
        dataset_id=dataset.dataset_id,
        dataset_fingerprint_sha256=dataset.fingerprint_sha256,
        overall=overall,
        slices=slices,
        promotion_eligible=False,
        limitations=(
            "no_temporal_tracking_evidence",
            "no_camera_negative_false_accept_evidence",
            "confidence_thresholds_are_diagnostic_not_runtime_policy",
        ),
    )


def _static_ocr_slice(
    scope: str,
    tracks: Sequence[CalibrationTrack],
    *,
    thresholds: tuple[float, ...],
    confidence_bin_count: int,
) -> StaticOCRSlice:
    validator = IranianPlateValidator()
    rows: list[tuple[CalibrationObservation, bool, float]] = []
    for track in tracks:
        expected = track.expected_plate
        if expected is None:
            continue
        for observation in track.observations:
            normalized = validator.validate(observation.text)
            exact = normalized.valid and normalized.normalized == expected
            rows.append(
                (
                    observation,
                    exact,
                    _normalized_edit_distance(
                        expected,
                        normalized.normalized if normalized.normalized else "",
                    ),
                )
            )
    if not rows:
        raise ValueError(f"static OCR slice {scope!r} has no observations")

    bins: list[StaticConfidenceBin] = []
    for index in range(confidence_bin_count):
        lower = index / confidence_bin_count
        upper = (index + 1) / confidence_bin_count
        members = [
            row
            for row in rows
            if lower <= row[0].confidence < upper
            or (index == confidence_bin_count - 1 and row[0].confidence == 1.0)
        ]
        bins.append(
            StaticConfidenceBin(
                lower_bound=lower,
                upper_bound=upper,
                observations=len(members),
                mean_confidence=(
                    sum(row[0].confidence for row in members) / len(members)
                    if members
                    else None
                ),
                exact_accuracy=(
                    sum(row[1] for row in members) / len(members) if members else None
                ),
            )
        )
    threshold_metrics = []
    for threshold in thresholds:
        accepted = [
            row for row in rows if row[0].valid and row[0].confidence >= threshold
        ]
        exact_accepted = sum(row[1] for row in accepted)
        wrong_accepted = len(accepted) - exact_accepted
        threshold_metrics.append(
            StaticThresholdMetrics(
                threshold=threshold,
                accepted=len(accepted),
                coverage=len(accepted) / len(rows),
                selective_exact_accuracy=(
                    exact_accepted / len(accepted) if accepted else None
                ),
                wrong_accept_rate=wrong_accepted / len(rows),
            )
        )
    ece = sum(
        confidence_bin.observations
        / len(rows)
        * abs(
            float(confidence_bin.mean_confidence) - float(confidence_bin.exact_accuracy)
        )
        for confidence_bin in bins
        if confidence_bin.observations
    )
    exact_matches = sum(row[1] for row in rows)
    return StaticOCRSlice(
        scope=scope,
        observations=len(rows),
        exact_matches=exact_matches,
        exact_accuracy=exact_matches / len(rows),
        mean_character_error_rate=sum(row[2] for row in rows) / len(rows),
        brier_score=sum((row[0].confidence - float(row[1])) ** 2 for row in rows)
        / len(rows),
        expected_calibration_error=ece,
        confidence_bins=tuple(bins),
        thresholds=tuple(threshold_metrics),
    )


def calibrate(
    dataset: CalibrationDataset,
    *,
    base_configs: Mapping[str, TemporalFusionConfig] | None = None,
    grids: Mapping[str, Mapping[str, Sequence[object]]] | None = None,
    requirements: CalibrationRequirements | None = None,
) -> CalibrationReport:
    policy = requirements or CalibrationRequirements()
    bases = dict(base_configs or {})
    search_grids = dict(grids or {})
    reports: list[ProfileCalibration] = []
    global_blockers: list[str] = []

    for profile in sorted(CALIBRATION_PROFILES):
        profile_tracks = [track for track in dataset.tracks if track.profile == profile]
        train = [track for track in profile_tracks if track.split == "train"]
        holdout = [track for track in profile_tracks if track.split == "holdout"]
        blockers: list[str] = []
        if len(train) < policy.minimum_train_tracks:
            blockers.append("insufficient_train_tracks")
        if len(holdout) < policy.minimum_holdout_tracks:
            blockers.append("insufficient_holdout_tracks")
        for split_name, split_tracks in (("train", train), ("holdout", holdout)):
            if not any(track.expected_plate is None for track in split_tracks):
                blockers.append(f"{split_name}_missing_negative_tracks")
            if not any(track.expected_plate is not None for track in split_tracks):
                blockers.append(f"{split_name}_missing_positive_tracks")
        cannot_score = (
            not train
            or not holdout
            or not any(track.expected_plate is not None for track in train)
            or not any(track.expected_plate is not None for track in holdout)
        )
        if cannot_score:
            reports.append(
                ProfileCalibration(
                    profile, None, None, None, 0, False, tuple(sorted(set(blockers)))
                )
            )
            global_blockers.extend(f"{profile}:{blocker}" for blocker in blockers)
            continue

        candidates = _candidate_configs(
            bases.get(profile, TemporalFusionConfig()),
            search_grids.get(profile, {}),
            policy.maximum_grid_candidates,
        )
        scored = [
            (candidate, evaluate_config(train, candidate)) for candidate in candidates
        ]
        best_config, train_metrics = max(
            scored,
            key=lambda item: _rank(item[1], policy),
        )
        holdout_metrics = evaluate_config(holdout, best_config)
        if not _meets_requirements(train_metrics, policy):
            blockers.append("train_quality_gate_failed")
        if not _meets_requirements(holdout_metrics, policy):
            blockers.append("holdout_quality_gate_failed")
        valid = not blockers
        reports.append(
            ProfileCalibration(
                profile=profile,
                config=best_config,
                train_metrics=train_metrics,
                holdout_metrics=holdout_metrics,
                candidates_evaluated=len(candidates),
                valid=valid,
                blockers=tuple(blockers),
            )
        )
        global_blockers.extend(f"{profile}:{blocker}" for blocker in blockers)

    return CalibrationReport(
        schema="bcvision.anpr.tcam-calibration-report/v1",
        dataset_id=dataset.dataset_id,
        dataset_fingerprint_sha256=dataset.fingerprint_sha256,
        profiles=tuple(reports),
        valid=not global_blockers and all(report.valid for report in reports),
        blockers=tuple(global_blockers),
    )


def _candidate_configs(
    base: TemporalFusionConfig,
    grid: Mapping[str, Sequence[object]],
    maximum: int,
) -> tuple[TemporalFusionConfig, ...]:
    unknown = set(grid) - _TUNABLE_FIELDS
    if unknown:
        raise ValueError(f"unknown TemporalFusionConfig grid fields: {sorted(unknown)}")
    names = tuple(sorted(grid))
    values = []
    for name in names:
        options = tuple(grid[name])
        if not options:
            raise ValueError(f"calibration grid {name!r} cannot be empty")
        values.append(options)
    combinations = math.prod(len(options) for options in values) if values else 1
    if combinations > maximum:
        raise ValueError(
            f"calibration grid contains {combinations} candidates; maximum is {maximum}"
        )
    output: dict[str, TemporalFusionConfig] = {}
    for selected in itertools.product(*values) if values else [()]:
        changes = dict(zip(names, selected, strict=True))
        try:
            candidate = replace(base, **changes)
        except (TypeError, ValueError):
            continue
        key = json.dumps(asdict(candidate), sort_keys=True, separators=(",", ":"))
        output[key] = candidate
    if not output:
        raise ValueError("calibration grid contains no valid policy candidate")
    return tuple(output.values())


def _meets_requirements(
    metrics: CalibrationMetrics, requirements: CalibrationRequirements
) -> bool:
    return (
        metrics.exact_accuracy >= requirements.target_exact_accuracy
        and metrics.event_recall >= requirements.minimum_event_recall
        and metrics.event_precision >= requirements.minimum_event_precision
        and metrics.false_accept_rate <= requirements.maximum_false_accept_rate
        and metrics.wrong_event_rate <= requirements.maximum_wrong_event_rate
        and metrics.mean_character_error_rate
        <= requirements.maximum_mean_character_error_rate
    )


def _rank(
    metrics: CalibrationMetrics, requirements: CalibrationRequirements
) -> tuple[float, ...]:
    return (
        float(_meets_requirements(metrics, requirements)),
        metrics.exact_accuracy,
        metrics.event_precision,
        metrics.event_recall,
        -metrics.false_accept_rate,
        -metrics.wrong_event_rate,
        -metrics.mean_character_error_rate,
        -metrics.mean_ocr_calls_per_track,
        -(metrics.mean_latency_seconds or 0.0),
    )


def _normalized_edit_distance(expected: str, actual: str) -> float:
    if expected == actual:
        return 0.0
    if not expected:
        return 1.0 if actual else 0.0
    previous = list(range(len(actual) + 1))
    for row, expected_character in enumerate(expected, start=1):
        current = [row]
        for column, actual_character in enumerate(actual, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (expected_character != actual_character),
                )
            )
        previous = current
    return min(1.0, previous[-1] / max(1, len(expected)))


def _reject_unknown(
    raw: Mapping[str, object], allowed: frozenset[str], context: str
) -> None:
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"{context} contains unknown fields: {sorted(unknown)}")


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _probability(value: object, name: str) -> float:
    number = _finite(value, name)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be within 0..1")
    return number


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    number = int(value)
    if number != value:
        raise ValueError(f"{name} must be an integer")
    return number


def _positive_integer(value: object, name: str) -> int:
    number = _integer(value, name)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


__all__ = [
    "CALIBRATION_PROFILES",
    "CALIBRATION_SCHEMA",
    "STATIC_CONFIDENCE_THRESHOLDS",
    "CalibrationDataset",
    "CalibrationMetrics",
    "CalibrationObservation",
    "CalibrationReport",
    "CalibrationRequirements",
    "CalibrationTrack",
    "ProfileCalibration",
    "StaticConfidenceBin",
    "StaticOCRReport",
    "StaticOCRSlice",
    "StaticThresholdMetrics",
    "TrackReplay",
    "analyze_static_ocr",
    "calibrate",
    "evaluate_config",
    "load_calibration_dataset",
    "replay_track",
]
