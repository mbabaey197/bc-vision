from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from .types import OCRResult
from .validator import IranianPlateValidator


class RecognitionPhase(str, Enum):
    """Recognition lifecycle owned by one tracker episode."""

    READING = "reading"
    PROVISIONAL = "provisional"
    LOCKED = "locked"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class TemporalFusionConfig:
    """Calibratable policy for track-centric multi-frame recognition.

    ``provisional_confidence`` is intentionally not an event threshold.  A
    reading must satisfy either the strict one-frame express path or the
    independent multi-frame lock path before it can leave the OCR cycle.
    """

    provisional_confidence: float = 0.75
    lock_confidence: float = 0.86
    express_lock_confidence: float = 0.93
    min_slot_confidence: float = 0.78
    express_min_slot_confidence: float = 0.90
    min_slot_margin: float = 0.16
    min_independent_observations: int = 2
    min_winner_support: int = 2
    independent_frame_gap: int = 2
    correlated_evidence_weight: float = 0.25
    min_ocr_quality: float = 0.32
    express_min_quality: float = 0.65
    reread_quality_gain: float = 0.08
    reread_area_gain: float = 0.20
    max_frames_between_reads: int = 3
    max_ocr_attempts: int = 4

    def __post_init__(self) -> None:
        thresholds = {
            "provisional_confidence": self.provisional_confidence,
            "lock_confidence": self.lock_confidence,
            "express_lock_confidence": self.express_lock_confidence,
            "min_slot_confidence": self.min_slot_confidence,
            "express_min_slot_confidence": self.express_min_slot_confidence,
            "min_slot_margin": self.min_slot_margin,
            "correlated_evidence_weight": self.correlated_evidence_weight,
            "min_ocr_quality": self.min_ocr_quality,
            "express_min_quality": self.express_min_quality,
            "reread_quality_gain": self.reread_quality_gain,
            "reread_area_gain": self.reread_area_gain,
        }
        for name, value in thresholds.items():
            number = float(value)
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{name} must be finite and within 0..1")
        positive = {
            "min_independent_observations": self.min_independent_observations,
            "min_winner_support": self.min_winner_support,
            "independent_frame_gap": self.independent_frame_gap,
            "max_frames_between_reads": self.max_frames_between_reads,
            "max_ocr_attempts": self.max_ocr_attempts,
        }
        for name, value in positive.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if self.provisional_confidence > self.lock_confidence:
            raise ValueError("provisional_confidence cannot exceed lock_confidence")
        if self.lock_confidence > self.express_lock_confidence:
            raise ValueError("lock_confidence cannot exceed express_lock_confidence")


@dataclass(frozen=True, slots=True)
class TrackOCRObservation:
    result: OCRResult
    quality: float
    seq: int


@dataclass(frozen=True, slots=True)
class SlotDecision:
    index: int
    kind: str
    character: str | None
    confidence: float
    margin: float
    support: int
    alternatives: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class FusionDecision:
    phase: RecognitionPhase
    text: str
    confidence: float
    reason: str
    slots: tuple[SlotDecision, ...]
    observations: int
    independent_observations: int
    full_sequence_support: int

    @property
    def unresolved_slots(self) -> tuple[int, ...]:
        return tuple(slot.index for slot in self.slots if slot.character is None)

    @property
    def locked(self) -> bool:
        return self.phase is RecognitionPhase.LOCKED


@dataclass(frozen=True, slots=True)
class OCRScheduleDecision:
    run_ocr: bool
    reason: str


@dataclass(frozen=True, slots=True)
class _AlignedObservation:
    observation: TrackOCRObservation
    normalized: str
    aligned: tuple[tuple[str, float] | None, ...]
    independent: bool
    novelty_weight: float


class IranianPlateLayout:
    """Logical slot contract: two digits, one letter, then five digits."""

    slot_kinds: tuple[str, ...] = ("digit", "digit", "letter") + ("digit",) * 5
    placeholders = frozenset({"?", "_", "*", "□", "-"})

    def __init__(self, validator: IranianPlateValidator) -> None:
        self.validator = validator
        self.letters = frozenset(validator.config.allowed_letters) | frozenset(
            {"D", "S"}
        )

    @property
    def size(self) -> int:
        return len(self.slot_kinds)

    def normalize_partial(self, text: str) -> str:
        # Preserve explicit unknown positions through the validator's normalizer,
        # which otherwise removes common separators such as '_' and '-'.
        marker = "¤"
        protected = "".join(
            marker if char in self.placeholders else char for char in str(text or "")
        )
        return self.validator.normalize(protected).replace(marker, "?")

    def accepts(self, index: int, character: str) -> bool:
        if character == "?":
            return True
        if self.slot_kinds[index] == "digit":
            return character.isascii() and character.isdigit()
        return character in self.letters

    def align(
        self,
        text: str,
        confidences: Sequence[float],
        priors: Sequence[dict[str, float]],
        fallback_confidence: float,
    ) -> tuple[str, tuple[tuple[str, float] | None, ...]]:
        normalized = self.normalize_partial(text)
        characters = list(normalized)
        values = [
            _unit(
                confidences[index] if index < len(confidences) else fallback_confidence
            )
            for index in range(len(characters))
        ]

        if len(characters) == self.size and all(
            char == "?" or self.accepts(index, char)
            for index, char in enumerate(characters)
        ):
            direct = tuple(
                None if char == "?" else (char, values[index])
                for index, char in enumerate(characters)
            )
            return normalized, direct

        # Dynamic alignment lets partial OCR reads contribute without guessing
        # that every missing glyph belongs at the end of the plate. Existing
        # slot evidence is only a tie-breaker; it cannot make an invalid type
        # (letter in a digit slot, for example) fit.
        memo: dict[
            tuple[int, int], tuple[float, tuple[tuple[str, float] | None, ...]]
        ] = {}

        def solve(
            slot_index: int, char_index: int
        ) -> tuple[float, tuple[tuple[str, float] | None, ...]]:
            key = (slot_index, char_index)
            if key in memo:
                return memo[key]
            if slot_index == self.size:
                result = (-0.45 * (len(characters) - char_index), ())
                memo[key] = result
                return result

            skip_score, skip_tail = solve(slot_index + 1, char_index)
            best = (skip_score - 0.22, (None,) + skip_tail)
            if char_index < len(characters):
                # Ignore an OCR noise token.
                noise_score, noise_tail = solve(slot_index, char_index + 1)
                noise = (noise_score - 0.45, noise_tail)
                if noise[0] > best[0]:
                    best = noise

                char = characters[char_index]
                if self.accepts(slot_index, char):
                    match_score, match_tail = solve(slot_index + 1, char_index + 1)
                    prior = priors[slot_index].get(char, 0.0) if char != "?" else 0.0
                    matched = (
                        match_score + 1.0 + 0.35 * prior,
                        (None if char == "?" else (char, values[char_index]),)
                        + match_tail,
                    )
                    # Prefer a valid match on equal scores so a short digit run
                    # aligns left-to-right until evidence indicates otherwise.
                    if matched[0] >= best[0]:
                        best = matched
            memo[key] = best
            return best

        return normalized, solve(0, 0)[1]


class PlateEvidenceAccumulator:
    """Fuse character-position and full-string evidence for one Track ID."""

    def __init__(
        self,
        validator: IranianPlateValidator | None = None,
        config: TemporalFusionConfig | None = None,
    ) -> None:
        self.validator = validator or IranianPlateValidator()
        self.config = config or TemporalFusionConfig()
        self.layout = IranianPlateLayout(self.validator)
        self._by_seq: dict[int, TrackOCRObservation] = {}
        self._locked: FusionDecision | None = None

    @property
    def observations(self) -> tuple[TrackOCRObservation, ...]:
        return tuple(self._by_seq[key] for key in sorted(self._by_seq))

    def add(self, result: OCRResult, *, quality: float, seq: int) -> FusionDecision:
        observation = TrackOCRObservation(
            result=result, quality=_unit(quality), seq=int(seq)
        )
        existing = self._by_seq.get(observation.seq)
        if existing is None or self._observation_strength(
            observation
        ) > self._observation_strength(existing):
            self._by_seq[observation.seq] = observation
        return self.decision()

    def decision(self) -> FusionDecision:
        if self._locked is not None:
            return self._locked
        aligned = self._aligned_observations()
        slots = self._slot_decisions(aligned)
        text = (
            ""
            if any(slot.character is None for slot in slots)
            else "".join(slot.character or "" for slot in slots)
        )
        valid = self.validator.validate(text).valid if text else False
        independent = sum(item.independent for item in aligned)
        full_text, full_support, full_confidence = self._full_sequence_vote(aligned)

        if not text:
            return FusionDecision(
                RecognitionPhase.READING,
                "",
                0.0,
                "unresolved_slots",
                slots,
                len(aligned),
                independent,
                full_support,
            )
        slot_confidences = [slot.confidence for slot in slots]
        confidence = 0.65 * (
            sum(slot_confidences) / len(slot_confidences)
        ) + 0.35 * min(slot_confidences)
        if (
            full_text == text
            and full_support >= self.config.min_independent_observations
        ):
            confidence = min(1.0, confidence + 0.03 * min(2, full_support - 1))
            confidence = max(confidence, min(0.98, full_confidence))
        confidence = _unit(confidence)

        min_slot = min(slot_confidences)
        min_margin = min(slot.margin for slot in slots)
        min_support = min(slot.support for slot in slots)
        best_quality = max((item.observation.quality for item in aligned), default=0.0)
        express = (
            len(aligned) == 1
            and valid
            and confidence >= self.config.express_lock_confidence
            and min_slot >= self.config.express_min_slot_confidence
            and best_quality >= self.config.express_min_quality
        )
        consensus = (
            valid
            and independent >= self.config.min_independent_observations
            and confidence >= self.config.lock_confidence
            and min_slot >= self.config.min_slot_confidence
            and min_margin >= self.config.min_slot_margin
            and min_support >= self.config.min_winner_support
        )
        if express or consensus:
            reason = (
                "express_high_confidence"
                if express
                else "independent_temporal_consensus"
            )
            decision = FusionDecision(
                RecognitionPhase.LOCKED,
                text,
                confidence,
                reason,
                slots,
                len(aligned),
                independent,
                full_support,
            )
            self._locked = decision
            return decision
        if valid and confidence >= self.config.provisional_confidence:
            return FusionDecision(
                RecognitionPhase.PROVISIONAL,
                text,
                confidence,
                "awaiting_independent_confirmation",
                slots,
                len(aligned),
                independent,
                full_support,
            )
        return FusionDecision(
            RecognitionPhase.READING,
            text if valid else "",
            confidence,
            "invalid_fused_structure" if text else "unresolved_slots",
            slots,
            len(aligned),
            independent,
            full_support,
        )

    @staticmethod
    def _observation_strength(observation: TrackOCRObservation) -> float:
        return _unit(observation.result.confidence) * (
            0.55 + 0.45 * observation.quality
        )

    def _aligned_observations(self) -> list[_AlignedObservation]:
        priors: list[dict[str, float]] = [{} for _ in range(self.layout.size)]
        output: list[_AlignedObservation] = []
        last_independent_seq: int | None = None
        for observation in self.observations:
            if not observation.result.valid or not observation.result.text.strip():
                continue
            normalized, aligned = self.layout.align(
                observation.result.text,
                observation.result.character_confidences,
                priors,
                observation.result.confidence,
            )
            independent = (
                last_independent_seq is None
                or observation.seq - last_independent_seq
                >= self.config.independent_frame_gap
            )
            novelty_weight = (
                1.0 if independent else self.config.correlated_evidence_weight
            )
            if independent:
                last_independent_seq = observation.seq
            item = _AlignedObservation(
                observation=observation,
                normalized=normalized,
                aligned=aligned,
                independent=independent,
                novelty_weight=novelty_weight,
            )
            output.append(item)
            for index, value in enumerate(aligned):
                if value is None:
                    continue
                char, char_confidence = value
                prior = self._evidence_weight(item, char_confidence)
                priors[index][char] = priors[index].get(char, 0.0) + prior
        return output

    def _evidence_weight(
        self, item: _AlignedObservation, char_confidence: float
    ) -> float:
        sequence_confidence = _unit(item.observation.result.confidence)
        quality_factor = 0.55 + 0.45 * item.observation.quality
        return (
            sequence_confidence
            * _unit(char_confidence)
            * quality_factor
            * item.novelty_weight
        )

    def _slot_decisions(
        self, aligned: Sequence[_AlignedObservation]
    ) -> tuple[SlotDecision, ...]:
        decisions: list[SlotDecision] = []
        for index, kind in enumerate(self.layout.slot_kinds):
            weights: dict[str, float] = {}
            reliability_sum: dict[str, float] = {}
            support: dict[str, int] = {}
            for item in aligned:
                value = item.aligned[index]
                if value is None:
                    continue
                char, char_confidence = value
                weight = self._evidence_weight(item, char_confidence)
                weights[char] = weights.get(char, 0.0) + weight
                reliability = min(
                    _unit(char_confidence), _unit(item.observation.result.confidence)
                )
                reliability_sum[char] = (
                    reliability_sum.get(char, 0.0) + reliability * weight
                )
                if item.independent:
                    support[char] = support.get(char, 0) + 1
            if not weights:
                decisions.append(SlotDecision(index, kind, None, 0.0, 0.0, 0))
                continue
            ranked = sorted(
                weights.items(), key=lambda item: (item[1], item[0]), reverse=True
            )
            char, top_weight = ranked[0]
            total_weight = sum(weights.values())
            runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
            purity = top_weight / max(1e-9, total_weight)
            raw_confidence = reliability_sum[char] / max(1e-9, top_weight)
            winner_support = support.get(char, 0)
            confidence = raw_confidence * (0.70 + 0.30 * purity)
            confidence += 0.04 * min(2, max(0, winner_support - 1))
            alternatives = tuple(
                (candidate, weight / max(1e-9, total_weight))
                for candidate, weight in ranked[:3]
            )
            decisions.append(
                SlotDecision(
                    index=index,
                    kind=kind,
                    character=char,
                    confidence=_unit(confidence),
                    margin=_unit((top_weight - runner_up) / max(1e-9, total_weight)),
                    support=winner_support,
                    alternatives=alternatives,
                )
            )
        return tuple(decisions)

    def _full_sequence_vote(
        self,
        aligned: Sequence[_AlignedObservation],
    ) -> tuple[str, int, float]:
        groups: dict[str, list[_AlignedObservation]] = {}
        for item in aligned:
            validation = self.validator.validate(item.observation.result.text)
            if validation.valid:
                groups.setdefault(validation.normalized, []).append(item)
        if not groups:
            return "", 0, 0.0
        text, members = max(
            groups.items(),
            key=lambda pair: (
                sum(
                    self._observation_strength(item.observation) * item.novelty_weight
                    for item in pair[1]
                ),
                sum(item.independent for item in pair[1]),
            ),
        )
        independent_members = [item for item in members if item.independent]
        weight = sum(
            self._observation_strength(item.observation) * item.novelty_weight
            for item in members
        )
        confidence = sum(
            _unit(item.observation.result.confidence)
            * self._observation_strength(item.observation)
            * item.novelty_weight
            for item in members
        ) / max(1e-9, weight)
        return text, len(independent_members), _unit(confidence)


@dataclass(slots=True)
class TrackRecognitionSession:
    """Per-track owner for fusion, OCR scheduling, and one-shot emission."""

    accumulator: PlateEvidenceAccumulator = field(
        default_factory=PlateEvidenceAccumulator
    )
    attempts: int = 0
    in_flight: bool = False
    event_claimed: bool = False
    closed: bool = False
    last_ocr_seq: int | None = None
    last_ocr_quality: float = 0.0
    last_ocr_area: int = 0

    @property
    def config(self) -> TemporalFusionConfig:
        return self.accumulator.config

    @property
    def decision(self) -> FusionDecision:
        if self.closed:
            current = self.accumulator.decision()
            return FusionDecision(
                RecognitionPhase.CLOSED,
                current.text,
                current.confidence,
                "track_closed",
                current.slots,
                current.observations,
                current.independent_observations,
                current.full_sequence_support,
            )
        return self.accumulator.decision()

    def should_schedule_ocr(
        self,
        *,
        seq: int,
        quality: float,
        bbox_area: int,
        near_exit: bool = False,
    ) -> OCRScheduleDecision:
        if self.closed:
            return OCRScheduleDecision(False, "track_closed")
        if self.decision.locked:
            return OCRScheduleDecision(False, "recognition_locked")
        if self.in_flight:
            return OCRScheduleDecision(False, "ocr_in_flight")
        if self.attempts >= self.config.max_ocr_attempts:
            return OCRScheduleDecision(False, "attempt_budget_exhausted")
        quality = _unit(quality)
        area = max(0, int(bbox_area))
        if quality < self.config.min_ocr_quality:
            return OCRScheduleDecision(False, "quality_below_floor")
        if self.attempts == 0:
            return OCRScheduleDecision(True, "first_usable_crop")
        if near_exit:
            return OCRScheduleDecision(True, "pre_exit_final_read")
        if quality >= self.last_ocr_quality + self.config.reread_quality_gain:
            return OCRScheduleDecision(True, "quality_improved")
        if self.last_ocr_area > 0 and area >= math.ceil(
            self.last_ocr_area * (1.0 + self.config.reread_area_gain)
        ):
            return OCRScheduleDecision(True, "plate_area_grew")
        frame_gap = int(seq) - int(
            self.last_ocr_seq if self.last_ocr_seq is not None else seq
        )
        current = self.decision
        if (
            current.phase is RecognitionPhase.PROVISIONAL
            and frame_gap >= self.config.independent_frame_gap
        ):
            return OCRScheduleDecision(True, "provisional_confirmation_due")
        if current.reason == "invalid_fused_structure" and frame_gap >= 1:
            return OCRScheduleDecision(True, "conflicting_evidence")
        if current.unresolved_slots and frame_gap >= 1:
            return OCRScheduleDecision(True, "unresolved_slots")
        if frame_gap >= self.config.max_frames_between_reads:
            return OCRScheduleDecision(True, "periodic_refresh")
        return OCRScheduleDecision(False, "no_material_gain")

    def reserve_ocr(self, *, seq: int, quality: float, bbox_area: int) -> None:
        if self.in_flight:
            raise RuntimeError("OCR is already in flight for this track")
        self.in_flight = True
        self.attempts += 1
        self.last_ocr_seq = int(seq)
        self.last_ocr_quality = _unit(quality)
        self.last_ocr_area = max(0, int(bbox_area))

    def release_ocr(self, *, retryable: bool = False) -> None:
        self.in_flight = False
        if retryable and self.attempts > 0:
            self.attempts -= 1

    def observe(self, result: OCRResult, *, quality: float, seq: int) -> FusionDecision:
        self.in_flight = False
        return self.accumulator.add(result, quality=quality, seq=seq)

    def claim_event(self) -> bool:
        if not self.decision.locked or self.event_claimed:
            return False
        self.event_claimed = True
        return True

    def close(self) -> None:
        self.in_flight = False
        self.closed = True


def _unit(value: float) -> float:
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))
