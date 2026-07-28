"""BC Vision ANPR pipeline with quality scoring and strict multi-frame consensus."""
from __future__ import annotations

from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
import math
import time

import cv2
import numpy as np

from .detector import detect_plates
from .ocr import read_plate_candidate
from .plate_rules import format_iran_plate, normalize_plate, plausible_plate
from .vehicle_intelligence import analyze_vehicle


def image_quality(image) -> dict:
    if image is None or getattr(image, "size", 0) == 0:
        return {
            "score": 0.0,
            "blur": 0.0,
            "brightness": 0.0,
            "contrast": 0.0,
            "dark": True,
            "glare": False,
        }
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_value = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean(gray >= 248))
    dark_ratio = float(np.mean(gray <= 22))
    brightness_score = max(0.0, 1.0 - abs(brightness - 135.0) / 145.0)
    contrast_score = min(1.0, contrast / 58.0)
    blur_score = min(1.0, math.log1p(max(0.0, blur_value)) / math.log1p(850.0))
    exposure_score = max(0.0, 1.0 - min(1.0, glare_ratio * 2.7 + dark_ratio * 1.8))
    score = (
        0.27 * brightness_score
        + 0.24 * contrast_score
        + 0.34 * blur_score
        + 0.15 * exposure_score
    )
    return {
        "score": round(min(1.0, max(0.0, score)), 4),
        "blur": round(blur_value, 3),
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "dark": brightness < 62 or dark_ratio > 0.45,
        "glare": glare_ratio > 0.16,
    }


def _combined_confidence(
    detector_confidence: float,
    ocr_confidence: float,
    quality_score: float,
    valid: bool,
) -> float:
    detector_confidence = min(1.0, max(0.0, float(detector_confidence)))
    ocr_confidence = min(1.0, max(0.0, float(ocr_confidence)))
    quality_score = min(1.0, max(0.0, float(quality_score)))
    combined = 0.34 * detector_confidence + 0.56 * ocr_confidence + 0.10 * quality_score
    if valid:
        combined += 0.08
    else:
        combined *= 0.62
    return round(min(1.0, max(0.0, combined)), 4)


def _partial_plate_text(positions: dict[int, str]) -> str:
    """Format visible OCR evidence without inventing a missing character."""

    chars = [positions.get(position, "؟") for position in range(8)]
    return (
        f"{chars[0]}{chars[1]}-"
        f"{chars[2]}-"
        f"{chars[3]}{chars[4]}{chars[5]}-"
        f"{chars[6]}{chars[7]}"
    )


def process_frame(
    frame,
    min_detection_confidence=0.25,
    engine_key=None,
):
    results = []
    detector_kwargs = {
        "min_confidence": min_detection_confidence,
    }
    if engine_key is not None:
        detector_kwargs["engine_key"] = engine_key
    for item in detect_plates(frame, **detector_kwargs):
        crop = item["crop"]
        quality = image_quality(crop)
        text = item.get("direct_text", "")
        ocr_confidence = float(
            item.get("direct_ocr_confidence", 0.0)
        )
        direct_valid = plausible_plate(text)
        direct_norm = normalize_plate(text) if direct_valid else ""
        ocr_engine = (
            "dedicated-character-detector"
            if item.get("direct_ocr_attempted")
            else "none"
        )
        whole_plate_ocr_attempted = False
        generic_ocr_attempted = False
        # The dedicated Iranian character detector is the preferred reader,
        # but an incomplete eight-character sequence must not end the read.
        # Use the already bundled generic OCR only for a credible, usable
        # plate crop. This recovers difficult glyphs without running generic
        # OCR on frames where no physical plate was localized.
        generic_fallback_eligible = bool(
            not direct_valid
            and (
                not item.get("direct_ocr_attempted")
                or (
                    float(item.get("confidence", 0.0)) >= 0.38
                    and quality["score"] >= 0.30
                    and crop.shape[0] >= 18
                    and crop.shape[1] >= 64
                )
            )
        )
        crnn_eligible = bool(
            float(item.get("confidence", 0.0)) >= 0.25
            and quality["score"] >= 0.12
            and crop.shape[0] >= 12
            and crop.shape[1] >= 48
        )
        fallback_text = ""
        fallback_confidence = 0.0
        fallback_engine = "none"
        if crnn_eligible or generic_fallback_eligible:
            whole_plate_ocr_attempted = True
            (
                fallback_text,
                fallback_confidence,
                fallback_engine,
            ) = read_plate_candidate(
                crop,
                engine_key=engine_key,
                allow_legacy=generic_fallback_eligible,
            )
            generic_ocr_attempted = bool(
                generic_fallback_eligible
                and fallback_engine in {"easyocr", "tesseract", "none"}
            )
        plate_hypotheses = []
        for hypothesis in item.get("plate_hypotheses", []):
            normalized = normalize_plate(
                hypothesis.get("plate_norm")
                or hypothesis.get("plate")
            )
            if not plausible_plate(normalized):
                continue
            plate_hypotheses.append({
                "plate": format_iran_plate(normalized),
                "plate_norm": normalized,
                "engine": hypothesis.get(
                    "engine",
                    "dedicated-character-detector",
                ),
                "confidence": min(
                    1.0,
                    max(
                        0.0,
                        float(hypothesis.get("confidence", 0.0)),
                    ),
                ),
                "score": min(
                    1.0,
                    max(
                        0.0,
                        float(
                            hypothesis.get(
                                "score",
                                hypothesis.get("confidence", 0.0),
                            )
                        ),
                    ),
                ),
            })
        if direct_valid and all(
            row["plate_norm"] != direct_norm
            for row in plate_hypotheses
        ):
            plate_hypotheses.insert(0, {
                "plate": format_iran_plate(direct_norm),
                "plate_norm": direct_norm,
                "engine": "dedicated-character-detector",
                "confidence": min(
                    1.0,
                    max(0.0, float(ocr_confidence)),
                ),
                "score": min(
                    1.0,
                    max(0.0, float(ocr_confidence)),
                ),
            })
        fallback_valid = plausible_plate(fallback_text)
        fallback_norm = (
            normalize_plate(fallback_text)
            if fallback_valid
            else ""
        )
        if fallback_valid:
            matched = next(
                (
                    row
                    for row in plate_hypotheses
                    if row["plate_norm"] == fallback_norm
                ),
                None,
            )
            if matched is None:
                plate_hypotheses.append({
                    "plate": format_iran_plate(fallback_norm),
                    "plate_norm": fallback_norm,
                    "engine": fallback_engine,
                    "confidence": min(
                        1.0,
                        max(0.0, float(fallback_confidence)),
                    ),
                    "score": min(
                        1.0,
                        max(0.0, float(fallback_confidence)),
                    ),
                })
            else:
                matched["confidence"] = max(
                    matched["confidence"],
                    float(fallback_confidence),
                )
                matched["score"] = max(
                    matched["score"],
                    float(fallback_confidence),
                )
                if fallback_engine not in matched["engine"]:
                    matched["engine"] += "+" + fallback_engine
        position_hypotheses = []
        for hypothesis in item.get("position_hypotheses", []):
            positions = {}
            for raw_position, raw_value in hypothesis.get(
                "positions",
                {},
            ).items():
                try:
                    position = int(raw_position)
                except (TypeError, ValueError):
                    continue
                character = normalize_plate(
                    raw_value.get("character", "")
                )
                if (
                    not 0 <= position < 8
                    or len(character) != 1
                    or (position == 2) == character.isdigit()
                ):
                    continue
                positions[position] = {
                    "character": character,
                    "confidence": min(
                        1.0,
                        max(
                            0.0,
                            float(raw_value.get("confidence", 0.0)),
                        ),
                    ),
                }
            if len(positions) < 5:
                continue
            position_hypotheses.append({
                "positions": positions,
                "coverage": len(positions),
                "score": min(
                    1.0,
                    max(0.0, float(hypothesis.get("score", 0.0))),
                ),
            })
        ocr_disagreement = bool(
            direct_valid
            and fallback_valid
            and direct_norm != fallback_norm
        )
        ocr_alternative = ""
        if fallback_valid and not direct_valid:
            text = fallback_text
            ocr_confidence = fallback_confidence
            ocr_engine = fallback_engine
        elif direct_valid and fallback_valid:
            if direct_norm == fallback_norm:
                ocr_confidence = max(
                    float(ocr_confidence),
                    float(fallback_confidence),
                )
                ocr_engine = (
                    "multi-engine-agreement:"
                    + fallback_engine
                )
            elif (
                fallback_engine == "crnn-onnx"
                and float(fallback_confidence)
                >= float(ocr_confidence) + 0.08
            ):
                ocr_alternative = format_iran_plate(direct_norm)
                text = fallback_text
                ocr_confidence = fallback_confidence
                ocr_engine = fallback_engine
            else:
                ocr_alternative = format_iran_plate(fallback_norm)

        valid = plausible_plate(text)
        best_effort = False
        needs_review = ocr_disagreement
        if ocr_disagreement:
            best_effort = True
        if valid and all(
            row["plate_norm"] != normalize_plate(text)
            for row in plate_hypotheses
        ):
            plate_hypotheses.insert(0, {
                "plate": format_iran_plate(text),
                "plate_norm": normalize_plate(text),
                "engine": ocr_engine,
                "confidence": float(ocr_confidence),
                "score": float(ocr_confidence),
            })
        elif not valid and plate_hypotheses:
            # The character detector often has a complete second hypothesis
            # even when its strict decoder rejects the primary read.  Surface
            # that real hypothesis as a reviewable suggestion instead of
            # throwing it away as "ناخوانا".
            suggestion = max(
                plate_hypotheses,
                key=lambda row: (
                    row["score"],
                    row["confidence"],
                    row["plate_norm"],
                ),
            )
            text = suggestion["plate"]
            ocr_confidence = max(
                float(ocr_confidence),
                float(suggestion["confidence"]),
            )
            valid = True
            best_effort = True
            needs_review = True
        elif not valid and position_hypotheses:
            # Preserve 5–7 observed positions.  Question marks explicitly mean
            # "not seen"; they are not guessed characters and remain easy for
            # the operator to replace in the correction form.
            partial = max(
                position_hypotheses,
                key=lambda row: (
                    row["coverage"],
                    row["score"],
                ),
            )
            visible = {
                position: metadata["character"]
                for position, metadata in partial["positions"].items()
            }
            text = _partial_plate_text(visible)
            ocr_confidence = max(
                float(ocr_confidence),
                float(partial["score"]) * 0.72,
            )
            best_effort = True
            needs_review = True
        combined = _combined_confidence(
            item["confidence"],
            ocr_confidence,
            quality["score"],
            valid,
        )
        plate_hypotheses.sort(
            key=lambda row: (
                row["score"],
                row["confidence"],
                row["plate_norm"],
            ),
            reverse=True,
        )
        results.append({
            "plate": text or "ناخوانا",
            "plate_norm": normalize_plate(text) if valid else "",
            "confidence": combined,
            "detector_confidence": float(item["confidence"]),
            "ocr_confidence": float(ocr_confidence),
            "quality_score": quality["score"],
            "quality": quality,
            "bbox": item["bbox"],
            "crop": crop,
            "method": item["method"],
            "ocr_engine": ocr_engine,
            "ocr_alternative": ocr_alternative,
            "ocr_disagreement": ocr_disagreement,
            "whole_plate_ocr_attempted": whole_plate_ocr_attempted,
            "dedicated_ocr_attempted": bool(
                item.get("direct_ocr_attempted")
            ),
            "generic_ocr_attempted": generic_ocr_attempted,
            "plate_hypotheses": plate_hypotheses[:5],
            "position_hypotheses": position_hypotheses[:5],
            "recovery_attempted": bool(
                item.get("recovery_attempted")
            ),
            "recovery_selected": bool(
                item.get("recovery_selected")
            ),
            "recovery_decision": item.get(
                "recovery_decision",
                "",
            ),
            "recovery_confidence": float(
                item.get("recovery_confidence", 0.0)
            ),
            "valid": valid,
            "best_effort": best_effort,
            "needs_review": needs_review,
            "vehicle_type": "نامشخص",
            "vehicle_color": "نامشخص",
            "vehicle_brand": "نامشخص",
            "vehicle_confidence": 0.0,
            "vehicle_bbox": item["bbox"],
        })
    results.sort(key=lambda row: (row["valid"], row["confidence"]), reverse=True)
    return results


def add_vehicle_analysis(result: dict, frame) -> dict:
    enriched = dict(result)
    vehicle = analyze_vehicle(frame, result["bbox"])
    enriched.update({
        "vehicle_type": vehicle["vehicle_type"],
        "vehicle_color": vehicle["vehicle_color"],
        "vehicle_brand": vehicle["vehicle_brand"],
        "vehicle_confidence": vehicle["vehicle_confidence"],
        "vehicle_bbox": vehicle["vehicle_bbox"],
        "vehicle_crop": vehicle.get("vehicle_crop"),
    })
    return enriched


def bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return intersection / float(area_a + area_b - intersection)


def plate_similarity(a: str, b: str) -> float:
    left, right = normalize_plate(a), normalize_plate(b)
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


@dataclass
class _Track:
    track_id: int
    bbox: tuple
    first_seen: float
    last_seen: float
    observations: deque = field(default_factory=lambda: deque(maxlen=20))
    emitted_plate: str = ""
    emitted_at: float = 0.0
    capture_event_emitted: bool = False
    unreadable_finalized: bool = False
    capture_event_score: float = -1.0
    best_capture_score: float = -1.0
    best_capture_result: dict | None = None
    best_frame: object | None = None


class PlateConsensusTracker:
    """Emit strict consensus first and a clearly marked best effort last."""

    def __init__(
        self,
        min_votes=3,
        max_age_seconds=2.4,
        emit_cooldown=4.0,
        min_position_ratio=0.60,
        min_position_margin=0.18,
        emit_unreadable=False,
        min_unreadable_observations=3,
        min_unreadable_seconds=0.8,
    ):
        # One or two frames are never enough for a definitive CCTV read.
        self.min_votes = max(3, int(min_votes))
        self.max_age_seconds = max(0.2, float(max_age_seconds))
        self.emit_cooldown = max(0.0, float(emit_cooldown))
        self.min_position_ratio = min(0.95, max(0.50, float(min_position_ratio)))
        self.min_position_margin = min(0.90, max(0.05, float(min_position_margin)))
        self.emit_unreadable = bool(emit_unreadable)
        self.min_unreadable_observations = max(
            self.min_votes,
            int(min_unreadable_observations),
        )
        self.min_unreadable_seconds = max(
            0.2,
            float(min_unreadable_seconds),
        )
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1

    @staticmethod
    def _capture_score(result: dict, frame) -> float:
        quality = min(
            1.0,
            max(0.0, float(result.get("quality_score", 0.0))),
        )
        detector = min(
            1.0,
            max(0.0, float(result.get("detector_confidence", 0.0))),
        )
        x1, y1, x2, y2 = result["bbox"]
        frame_area = max(1, int(frame.shape[0]) * int(frame.shape[1]))
        plate_area = max(1, (x2 - x1) * (y2 - y1))
        size_score = min(1.0, plate_area / max(1.0, frame_area * 0.018))
        return round(0.60 * quality + 0.25 * detector + 0.15 * size_score, 5)

    def _consider_capture(self, track: _Track, result: dict, frame) -> bool:
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        score = self._capture_score(result, frame)
        if score <= track.best_capture_score + 0.015:
            return False
        track.best_capture_score = score
        track.best_capture_result = deepcopy(result)
        track.best_frame = frame.copy()
        return True

    @staticmethod
    def _capture_result(
        track: _Track,
        refresh=False,
        final_unreadable=False,
    ) -> dict | None:
        if track.best_capture_result is None or track.best_frame is None:
            return None
        result = deepcopy(track.best_capture_result)
        result.update({
            "plate": (
                "ناخوانا"
                if final_unreadable
                else "در حال بررسی"
            ),
            "plate_norm": "",
            "valid": False,
            "ocr_confidence": 0.0,
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "capture_only": True,
            "capture_refresh": bool(refresh),
            "provisional": not bool(final_unreadable),
            "unreadable_final": bool(final_unreadable),
            "best_effort": False,
            "needs_review": True,
            "read_status": (
                "unreadable"
                if final_unreadable
                else "processing"
            ),
            "capture_frame": track.best_frame.copy(),
            "consensus_votes": 0,
        })
        return result

    def _unreadable_ready(
        self,
        track: _Track,
        timestamp: float,
    ) -> bool:
        if track.unreadable_finalized or track.emitted_plate:
            return False
        observations = list(track.observations)
        if len(observations) < self.min_unreadable_observations:
            return False
        if timestamp - track.first_seen < self.min_unreadable_seconds:
            return False
        valid_count = sum(bool(row.get("valid")) for row in observations)
        hypothesis_count = sum(
            bool(
                row.get("plate_hypotheses")
                or row.get("position_hypotheses")
            )
            for row in observations
        )
        # If partial plausible reads exist, allow two more observations for
        # temporal voting before declaring the track genuinely unreadable.
        required = (
            self.min_unreadable_observations
            if valid_count == 0 and hypothesis_count == 0
            else self.min_unreadable_observations + 2
        )
        return len(observations) >= required

    def _match(self, result: dict, timestamp: float) -> _Track | None:
        best = None
        best_score = 0.0
        for track in self._tracks.values():
            if timestamp - track.last_seen > self.max_age_seconds:
                continue
            overlap = bbox_iou(track.bbox, result["bbox"])
            text_score = 0.0
            if result.get("valid"):
                for observation in track.observations:
                    if observation.get("valid"):
                        text_score = max(
                            text_score,
                            plate_similarity(result["plate"], observation["plate"]),
                        )
            score = max(overlap, text_score * 0.88)
            if score > best_score and (overlap >= 0.18 or text_score >= 0.74):
                best, best_score = track, score
        return best

    def _expire(self, timestamp: float):
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.max_age_seconds * 2.2
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)

    @staticmethod
    def _observation_weight(row: dict) -> float:
        confidence = min(1.0, max(0.0, float(row.get("confidence", 0.0))))
        quality = min(1.0, max(0.0, float(row.get("quality_score", 0.0))))
        return confidence * max(0.25, quality)

    @staticmethod
    def _position_probabilities(row: dict) -> list[dict[str, float]]:
        candidates = {}
        primary = normalize_plate(row.get("plate", ""))
        if row.get("valid") and plausible_plate(primary):
            candidates[primary] = max(
                0.05,
                float(
                    row.get("ocr_confidence")
                    or row.get("confidence", 0.0)
                ),
            )
        for hypothesis in row.get("plate_hypotheses", []):
            normalized = normalize_plate(
                hypothesis.get("plate_norm")
                or hypothesis.get("plate")
            )
            if not plausible_plate(normalized):
                continue
            weight = max(
                0.02,
                float(
                    hypothesis.get(
                        "score",
                        hypothesis.get("confidence", 0.0),
                    )
                ),
            )
            candidates[normalized] = max(
                candidates.get(normalized, 0.0),
                weight,
            )
        hypotheses = [
            {
                "positions": {
                    position: character
                    for position, character in enumerate(plate)
                },
                "score": weight,
            }
            for plate, weight in candidates.items()
        ]
        for hypothesis in row.get("position_hypotheses", []):
            positions = {}
            for raw_position, raw_value in hypothesis.get(
                "positions",
                {},
            ).items():
                try:
                    position = int(raw_position)
                except (TypeError, ValueError):
                    continue
                character = normalize_plate(
                    raw_value.get("character", "")
                )
                if 0 <= position < 8 and len(character) == 1:
                    positions[position] = character
            if len(positions) >= 5:
                hypotheses.append({
                    "positions": positions,
                    "score": max(
                        0.02,
                        float(hypothesis.get("score", 0.0)),
                    ),
                })
        if not hypotheses:
            return []

        distributions = [defaultdict(float) for _ in range(8)]
        for position in range(8):
            available = [
                hypothesis
                for hypothesis in hypotheses
                if position in hypothesis["positions"]
            ]
            # Square scores so a decisive model result remains decisive while
            # close alternatives survive for temporal disambiguation.
            total = sum(
                hypothesis["score"] * hypothesis["score"]
                for hypothesis in available
            )
            for hypothesis in available:
                probability = (
                    hypothesis["score"] * hypothesis["score"]
                    / max(total, 1e-9)
                )
                distributions[position][
                    hypothesis["positions"][position]
                ] += probability
        return [dict(distribution) for distribution in distributions]

    def _consensus(self, track: _Track) -> dict | None:
        evidence = []
        for row in track.observations:
            probabilities = self._position_probabilities(row)
            if probabilities:
                copied = deepcopy(row)
                copied["_position_probabilities"] = probabilities
                evidence.append(copied)

        if len(evidence) < self.min_votes:
            return None

        winner_chars = []
        winner_counts = []
        agreement_ratios = []
        position_margins = []
        position_details = []

        for position in range(8):
            buckets = defaultdict(lambda: {"count": 0, "weight": 0.0})
            total_weight = 0.0
            for row in evidence:
                row_weight = self._observation_weight(row)
                distribution = row["_position_probabilities"][position]
                for character, probability in distribution.items():
                    weighted = row_weight * probability
                    buckets[character]["weight"] += weighted
                    total_weight += weighted
                    if probability >= 0.20:
                        buckets[character]["count"] += 1

            ordered = sorted(
                buckets.items(),
                key=lambda item: (item[1]["count"], item[1]["weight"], item[0]),
                reverse=True,
            )
            if not ordered:
                return None
            winner, metadata = ordered[0]
            runner_weight = ordered[1][1]["weight"] if len(ordered) > 1 else 0.0
            # Compare the winner directly with its strongest rival. With
            # top-k OCR hypotheses, several different weak alternatives can
            # split the remaining probability even when one character is the
            # only temporally consistent answer.
            ratio = metadata["weight"] / max(
                metadata["weight"] + runner_weight,
                1e-9,
            )
            margin = (metadata["weight"] - runner_weight) / max(total_weight, 1e-9)

            # Every one of the eight positions must independently have at least
            # three agreeing observations. Confidence alone cannot override a
            # weak positional majority.
            if (
                metadata["count"] < self.min_votes
                or ratio < self.min_position_ratio
                or margin < self.min_position_margin
            ):
                return None

            winner_chars.append(winner)
            winner_counts.append(metadata["count"])
            agreement_ratios.append(ratio)
            position_margins.append(margin)
            position_details.append({
                "position": position,
                "character": winner,
                "votes": metadata["count"],
                "ratio": round(ratio, 4),
                "margin": round(margin, 4),
            })

        winner_norm = "".join(winner_chars)
        if not plausible_plate(winner_norm):
            return None

        matching_scores = []
        for row in evidence:
            support = sum(
                row["_position_probabilities"][position].get(
                    character,
                    0.0,
                )
                for position, character in enumerate(winner_norm)
            )
            matching_scores.append((support, row))

        best_match_count = max(score for score, _ in matching_scores)
        best = max(
            (row for score, row in matching_scores if score == best_match_count),
            key=lambda row: (row.get("confidence", 0.0), row.get("quality_score", 0.0)),
        )

        total_weight = sum(self._observation_weight(row) for row in evidence)
        weighted_confidence = sum(
            float(row.get("confidence", 0.0)) * self._observation_weight(row)
            for row in evidence
        ) / max(total_weight, 1e-9)
        average_agreement = sum(agreement_ratios) / len(agreement_ratios)
        minimum_margin = min(position_margins)

        result = deepcopy(best)
        result.pop("_position_probabilities", None)
        result["plate"] = format_iran_plate(winner_norm)
        result["plate_norm"] = winner_norm
        result["confidence"] = round(
            min(
                1.0,
                0.72 * weighted_confidence
                + 0.18 * average_agreement
                + 0.10 * min(1.0, minimum_margin / 0.50),
            ),
            4,
        )
        result["consensus_votes"] = min(winner_counts)
        result["consensus_observations"] = len(evidence)
        result["position_agreement"] = position_details
        result["ambiguity_margin"] = round(minimum_margin, 4)
        result["best_effort"] = False
        result["needs_review"] = False
        result["read_status"] = "confirmed-ai"

        engine_support = defaultdict(float)
        alternative_support = defaultdict(float)
        saw_ab_disagreement = False
        for row in evidence:
            row_weight = self._observation_weight(row)
            primary = normalize_plate(row.get("plate", ""))
            primary_engine = str(row.get("ocr_engine", "")).strip()
            if primary == winner_norm and primary_engine:
                engine_support[primary_engine] += row_weight
            for hypothesis in row.get("plate_hypotheses", []):
                candidate = normalize_plate(
                    hypothesis.get("plate_norm")
                    or hypothesis.get("plate")
                )
                if candidate != winner_norm:
                    continue
                engine = str(hypothesis.get("engine", "")).strip()
                if engine:
                    engine_support[engine] += (
                        row_weight
                        * max(
                            0.02,
                            float(
                                hypothesis.get(
                                    "score",
                                    hypothesis.get(
                                        "confidence",
                                        0.0,
                                    ),
                                )
                            ),
                        )
                    )
            alternative = normalize_plate(
                row.get("ocr_alternative", "")
            )
            if (
                plausible_plate(alternative)
                and alternative != winner_norm
            ):
                alternative_support[alternative] += row_weight
            saw_ab_disagreement = (
                saw_ab_disagreement
                or bool(row.get("ocr_disagreement"))
            )

        engine_names = set(engine_support)
        crnn_supported = any(
            "crnn-onnx" in name
            for name in engine_names
        )
        character_supported = any(
            "dedicated-character-detector" in name
            or "multi-engine-agreement" in name
            for name in engine_names
        )
        if crnn_supported and character_supported:
            result["ocr_engine"] = "multi-engine-consensus"
        elif engine_support:
            result["ocr_engine"] = max(
                engine_support,
                key=lambda name: (
                    engine_support[name],
                    name,
                ),
            )
        if alternative_support:
            alternative = max(
                alternative_support,
                key=lambda plate: (
                    alternative_support[plate],
                    plate,
                ),
            )
            result["ocr_alternative"] = format_iran_plate(
                alternative
            )
        result["ocr_disagreement"] = saw_ab_disagreement

        centers = [
            (
                (row["bbox"][0] + row["bbox"][2]) / 2.0,
                (row["bbox"][1] + row["bbox"][3]) / 2.0,
            )
            for row in track.observations
            if row.get("bbox")
        ]
        direction = "stationary"
        if len(centers) >= 2:
            delta_y = centers[-1][1] - centers[0][1]
            average_height = sum(
                max(1.0, row["bbox"][3] - row["bbox"][1])
                for row in track.observations
            ) / len(track.observations)
            if delta_y > max(3.0, average_height * 0.12):
                direction = "down"
            elif delta_y < -max(3.0, average_height * 0.12):
                direction = "up"

        result["track_id"] = track.track_id
        result["first_seen"] = track.first_seen
        result["last_seen"] = track.last_seen
        result["direction"] = direction
        if track.best_frame is not None:
            result["capture_frame"] = track.best_frame.copy()
        if track.best_capture_result is not None:
            result["bbox"] = tuple(track.best_capture_result["bbox"])
        return result

    def _best_effort_result(self, track: _Track) -> dict | None:
        """Return the strongest observed read without inventing evidence.

        This path runs only after the strict consensus window has expired.  A
        complete low-margin plate is surfaced as a reviewable suggestion; an
        incomplete read keeps visible question marks.  With no character
        evidence at all the caller still emits the genuinely unreadable state.
        """

        evidence = []
        for row in track.observations:
            probabilities = self._position_probabilities(row)
            if any(probabilities):
                evidence.append((row, probabilities))
        if not evidence:
            return None

        winners = {}
        vote_counts = {}
        support_ratios = {}
        for position in range(8):
            buckets = defaultdict(lambda: {"count": 0, "weight": 0.0})
            total_weight = 0.0
            for row, probabilities in evidence:
                row_weight = max(0.02, self._observation_weight(row))
                for character, probability in probabilities[position].items():
                    weighted = row_weight * probability
                    buckets[character]["weight"] += weighted
                    total_weight += weighted
                    if probability >= 0.12:
                        buckets[character]["count"] += 1
            if not buckets:
                continue
            winner, metadata = max(
                buckets.items(),
                key=lambda item: (
                    item[1]["count"],
                    item[1]["weight"],
                    item[0],
                ),
            )
            winners[position] = winner
            vote_counts[position] = metadata["count"]
            support_ratios[position] = (
                metadata["weight"] / max(total_weight, 1e-9)
            )

        if len(winners) < 5:
            return None

        basis = deepcopy(
            track.best_capture_result
            or max(
                (row for row, _ in evidence),
                key=lambda row: (
                    row.get("quality_score", 0.0),
                    row.get("confidence", 0.0),
                ),
            )
        )
        normalized = "".join(
            winners.get(position, "")
            for position in range(8)
        )
        complete = (
            len(winners) == 8
            and plausible_plate(normalized)
        )
        average_support = sum(support_ratios.values()) / len(
            support_ratios
        )
        source_confidence = max(
            float(row.get("confidence", 0.0))
            for row, _ in evidence
        )
        confidence = min(
            0.69,
            max(
                0.12,
                source_confidence
                * (0.55 + 0.30 * average_support)
                * (len(winners) / 8.0),
            ),
        )
        basis.update({
            "plate": (
                format_iran_plate(normalized)
                if complete
                else _partial_plate_text(winners)
            ),
            "plate_norm": normalized if complete else "",
            "valid": complete,
            "best_effort": True,
            "needs_review": True,
            "read_status": "probable" if complete else "partial",
            "confidence": round(confidence, 4),
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "capture_only": not complete,
            "capture_refresh": True,
            "provisional": False,
            "unreadable_final": False,
            "partial_final": not complete,
            "consensus_votes": min(vote_counts.values()),
            "consensus_observations": len(evidence),
        })
        if track.best_frame is not None:
            basis["capture_frame"] = track.best_frame.copy()
        if track.best_capture_result is not None:
            basis["bbox"] = tuple(track.best_capture_result["bbox"])
        return basis

    def update(self, results, timestamp=None, frame=None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        self._expire(timestamp)
        emitted = []

        for result in results:
            track = self._match(result, timestamp)
            if track is None:
                track = _Track(
                    track_id=self._next_track_id,
                    bbox=result["bbox"],
                    first_seen=timestamp,
                    last_seen=timestamp,
                )
                self._tracks[track.track_id] = track
                self._next_track_id += 1

            track.bbox = result["bbox"]
            track.last_seen = timestamp
            track.observations.append(deepcopy(result))
            capture_improved = self._consider_capture(track, result, frame)
            consensus = self._consensus(track)
            if consensus is None:
                unreadable_ready = (
                    self.emit_unreadable
                    and self._unreadable_ready(track, timestamp)
                )
                if self.emit_unreadable and not track.capture_event_emitted:
                    capture = self._capture_result(track)
                    if capture is not None:
                        track.capture_event_emitted = True
                        track.capture_event_score = track.best_capture_score
                        emitted.append(capture)
                elif (
                    self.emit_unreadable
                    and not unreadable_ready
                    and capture_improved
                    and track.best_capture_score
                    > track.capture_event_score + 0.04
                ):
                    capture = self._capture_result(track, refresh=True)
                    if capture is not None:
                        track.capture_event_score = track.best_capture_score
                        emitted.append(capture)
                if (
                    unreadable_ready
                ):
                    final_read = (
                        self._best_effort_result(track)
                        or self._capture_result(
                            track,
                            refresh=True,
                            final_unreadable=True,
                        )
                    )
                    if final_read is not None:
                        track.unreadable_finalized = True
                        emitted.append(final_read)
                continue

            changed = consensus["plate_norm"] != track.emitted_plate
            cooled_down = timestamp - track.emitted_at >= self.emit_cooldown
            if changed or cooled_down:
                track.emitted_plate = consensus["plate_norm"]
                track.emitted_at = timestamp
                emitted.append(consensus)

        return emitted

    def active_track_ids(self) -> set[int]:
        return set(self._tracks)

    def flush(self):
        rows = []
        for track in list(self._tracks.values()):
            consensus = self._consensus(track)
            if consensus is not None:
                rows.append(consensus)
            elif (
                self.emit_unreadable
                and self._unreadable_ready(
                    track,
                    max(
                        track.last_seen,
                        track.first_seen + self.min_unreadable_seconds,
                    ),
                )
            ):
                final_read = (
                    self._best_effort_result(track)
                    or self._capture_result(
                        track,
                        refresh=True,
                        final_unreadable=True,
                    )
                )
                if final_read is not None:
                    track.unreadable_finalized = True
                    rows.append(final_read)
        return rows
