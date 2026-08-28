from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from typing import Sequence

from .types import PlateCandidate, TrackPhase

BBox = tuple[int, int, int, int]


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def _center(box: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


@dataclass(slots=True)
class TrackerConfig:
    min_iou: float = 0.18
    max_center_distance: float = 2.2
    max_missed: int = 4
    confirm_hits: int = 1
    bbox_smoothing: float = 0.72
    velocity_smoothing: float = 0.55

    def __post_init__(self) -> None:
        if not isfinite(float(self.min_iou)) or not 0.0 <= float(self.min_iou) <= 1.0:
            raise ValueError("min_iou must be finite and within 0..1")
        if not isfinite(float(self.max_center_distance)) or self.max_center_distance <= 0.0:
            raise ValueError("max_center_distance must be finite and positive")
        if int(self.max_missed) < 0:
            raise ValueError("max_missed must be non-negative")
        if int(self.confirm_hits) < 1:
            raise ValueError("confirm_hits must be positive")
        for name, value in (
            ("bbox_smoothing", self.bbox_smoothing),
            ("velocity_smoothing", self.velocity_smoothing),
        ):
            if not isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be finite and within 0..1")


@dataclass(slots=True)
class TrackedObject:
    track_id: int
    bbox: BBox
    confidence: float
    first_seq: int
    last_seq: int
    hits: int = 1
    missed: int = 0
    age: int = 1
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    phase: TrackPhase = TrackPhase.TRACKING
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def confirmed(self) -> bool:
        return bool(self.metadata.get("confirmed", False))

    def predict_bbox(self, seq: int) -> BBox:
        steps = max(0, min(4, int(seq) - int(self.last_seq)))
        if steps == 0:
            return self.bbox
        predicted = tuple(
            round(value + delta * steps)
            for value, delta in zip(self.bbox, self.velocity, strict=True)
        )
        x1, y1, x2, y2 = predicted
        if x2 <= x1 or y2 <= y1:
            return self.bbox
        return predicted


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    bbox: BBox
    confidence: float
    seq: int
    predicted: bool = False
    new_track: bool = False


@dataclass(slots=True)
class TrackerUpdate:
    observations: list[TrackObservation] = field(default_factory=list)
    removed_track_ids: list[int] = field(default_factory=list)


class LightweightMultiObjectTracker:
    """Small CPU-only plate tracker used between shared detector calls.

    It deliberately owns no neural-network state. Association combines IoU
    with a scale-normalized centre distance and keeps a smoothed box velocity,
    allowing the scheduler to skip detector frames under load while still
    harvesting high-resolution candidate crops.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, TrackedObject] = {}
        self._next_id = 1
        self._last_update_seq: int | None = None

    @property
    def tracks(self) -> tuple[TrackedObject, ...]:
        return tuple(self._tracks.values())

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1
        self._last_update_seq = None

    def predict(self, seq: int) -> list[TrackObservation]:
        seq = int(seq)
        if self._last_update_seq is not None and seq < self._last_update_seq:
            raise ValueError("prediction seq must not precede the last tracker update")
        return [
            TrackObservation(
                track_id=track.track_id,
                bbox=track.predict_bbox(seq),
                confidence=max(0.0, track.confidence * (0.92 ** max(0, seq - track.last_seq))),
                seq=seq,
                predicted=True,
            )
            for track in self._tracks.values()
            if track.missed <= self.config.max_missed
        ]

    def update(self, detections: Sequence[PlateCandidate], seq: int) -> TrackerUpdate:
        seq = int(seq)
        if self._last_update_seq is not None and seq <= self._last_update_seq:
            raise ValueError("tracker update seq must be strictly increasing")
        for detection in detections:
            self._validate_detection(detection)
        self._last_update_seq = seq

        predicted = {track_id: track.predict_bbox(seq) for track_id, track in self._tracks.items()}
        possible: list[tuple[float, int, int]] = []
        for track_id, box in predicted.items():
            track = self._tracks[track_id]
            for detection_index, detection in enumerate(detections):
                score = self._association_score(box, detection.bbox)
                if score is not None:
                    # A detector supplied hint is a strong, but not mandatory,
                    # association signal.
                    if detection.track_hint and detection.track_hint == str(track_id):
                        score += 2.0
                    possible.append((score, track_id, detection_index))

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _, track_id, detection_index in sorted(possible, reverse=True):
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            assignments.append((track_id, detection_index))

        observations: list[TrackObservation] = []
        for track_id, detection_index in assignments:
            detection = detections[detection_index]
            track = self._tracks[track_id]
            old_box = track.bbox
            new_box = self._smooth_box(predicted[track_id], detection.bbox)
            elapsed = max(1, seq - track.last_seq)
            measured_velocity = tuple((new - old) / elapsed for old, new in zip(old_box, new_box, strict=True))
            beta = min(1.0, max(0.0, self.config.velocity_smoothing))
            track.velocity = tuple(
                beta * measured + (1.0 - beta) * prior
                for prior, measured in zip(track.velocity, measured_velocity, strict=True)
            )
            track.bbox = new_box
            track.confidence = float(detection.confidence)
            track.last_seq = int(seq)
            track.hits += 1
            track.age = max(1, seq - track.first_seq + 1)
            track.missed = 0
            track.metadata["confirmed"] = track.hits >= self.config.confirm_hits
            observations.append(TrackObservation(track_id, new_box, track.confidence, seq))

        removed: list[int] = []
        for track_id, track in list(self._tracks.items()):
            if track_id in matched_tracks:
                continue
            track.missed += 1
            track.age = max(1, seq - track.first_seq + 1)
            if track.missed > self.config.max_missed:
                removed.append(track_id)
                del self._tracks[track_id]

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track_id = self._next_id
            self._next_id += 1
            track = TrackedObject(
                track_id=track_id,
                bbox=detection.bbox,
                confidence=float(detection.confidence),
                first_seq=int(seq),
                last_seq=int(seq),
                metadata={"confirmed": self.config.confirm_hits <= 1},
            )
            self._tracks[track_id] = track
            observations.append(
                TrackObservation(track_id, track.bbox, track.confidence, seq, new_track=True)
            )

        observations.sort(key=lambda item: item.track_id)
        return TrackerUpdate(observations=observations, removed_track_ids=removed)

    @staticmethod
    def _validate_detection(detection: PlateCandidate) -> None:
        try:
            x1, y1, x2, y2 = detection.bbox
        except (TypeError, ValueError) as exc:
            raise ValueError("detection bbox must contain four coordinates") from exc
        if not all(isfinite(float(value)) for value in (x1, y1, x2, y2)):
            raise ValueError("detection bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("detection bbox must have positive width and height")
        confidence = float(detection.confidence)
        if not isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("detection confidence must be finite and within 0..1")

    def _association_score(self, predicted: BBox, detected: BBox) -> float | None:
        iou = bbox_iou(predicted, detected)
        pcx, pcy = _center(predicted)
        dcx, dcy = _center(detected)
        pw = max(1.0, predicted[2] - predicted[0])
        ph = max(1.0, predicted[3] - predicted[1])
        normalized_distance = hypot(dcx - pcx, dcy - pcy) / hypot(pw, ph)
        if iou < self.config.min_iou and normalized_distance > self.config.max_center_distance:
            return None
        return 2.0 * iou + max(0.0, 1.0 - normalized_distance / self.config.max_center_distance)

    def _smooth_box(self, predicted: BBox, detected: BBox) -> BBox:
        alpha = min(1.0, max(0.0, self.config.bbox_smoothing))
        values = tuple(
            round(alpha * observed + (1.0 - alpha) * expected)
            for expected, observed in zip(predicted, detected, strict=True)
        )
        x1, y1, x2, y2 = values
        return detected if x2 <= x1 or y2 <= y1 else values
