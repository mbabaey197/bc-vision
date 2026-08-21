"""BC Vision ANPR pipeline with quality scoring and strict multi-frame consensus."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from functools import lru_cache
import math
import time

import cv2
import numpy as np

from .detector import detect_plates
from .ocr import read_plate_candidate
from .plate_rules import format_iran_plate, normalize_plate, plausible_plate
from .review_policy import auto_confirm_guess
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
    exclusion_mask=None,
    max_results=8,
    max_candidates=None,
    detector_variant=None,
    inference_key=None,
    expected_detector_revision=None,
    runtime_metadata=None,
):
    results = []
    detector_kwargs = {
        "min_confidence": min_detection_confidence,
        "exclusion_mask": exclusion_mask,
        "max_results": max(1, int(max_results)),
    }
    if max_candidates is not None:
        detector_kwargs["max_results"] = max(
            1, int(max_candidates)
        )
    model_key = inference_key if inference_key is not None else engine_key
    if model_key is not None:
        detector_kwargs["engine_key"] = model_key
    if detector_variant is not None:
        detector_kwargs["detector_variant"] = detector_variant
    detector_metadata = (
        runtime_metadata if runtime_metadata is not None else {}
    )
    if runtime_metadata is not None:
        detector_kwargs["runtime_metadata"] = detector_metadata
    if expected_detector_revision:
        detector_kwargs["expected_model_revision"] = (
            expected_detector_revision
        )
    detected_items = detect_plates(frame, **detector_kwargs)
    call_detector_revision = str(
        detector_metadata.get("detector_model_revision", "")
    ).strip()
    for item in detected_items:
        crop = item["crop"]
        quality = image_quality(crop)
        # Engine V3 has one authoritative OCR route. Detector-attached or
        # promoted custom OCR evidence is deliberately excluded from both the
        # selected plate and temporal hypotheses; it can still be inspected by
        # explicit training/diagnostic tools outside this production path.
        production_policy = detector_variant is not None
        text = "" if production_policy else item.get("direct_text", "")
        ocr_confidence = (
            0.0
            if production_policy
            else float(item.get("direct_ocr_confidence", 0.0))
        )
        direct_valid = plausible_plate(text)
        direct_norm = normalize_plate(text) if direct_valid else ""
        ocr_engine = (
            "dedicated-character-detector"
            if item.get("direct_ocr_attempted") and not production_policy
            else "none"
        )
        whole_plate_ocr_attempted = False
        generic_ocr_attempted = False
        # Production OCR is immutable: Hezar v2 first, then the fixed Platrix
        # CRNN only after a Hezar rejection/error. Promoted custom CRNNs,
        # detector-attached text and the character CNN are diagnostic-only.
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
        whole_plate_ocr_eligible = bool(
            float(item.get("confidence", 0.0)) >= 0.25
            and quality["score"] >= 0.12
            and crop.shape[0] >= 12
            and crop.shape[1] >= 48
        )
        fallback_text = ""
        fallback_confidence = 0.0
        fallback_engine = "none"
        if whole_plate_ocr_eligible or generic_fallback_eligible:
            whole_plate_ocr_attempted = True
            (
                fallback_text,
                fallback_confidence,
                fallback_engine,
            ) = read_plate_candidate(
                crop,
                engine_key=model_key,
                allow_legacy=generic_fallback_eligible,
            )
            generic_ocr_attempted = bool(
                generic_fallback_eligible
                and fallback_engine in {"cnn-onnx", "none"}
            )
        plate_hypotheses = []
        for hypothesis in (
            ()
            if production_policy
            else item.get("plate_hypotheses", [])
        ):
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
        for hypothesis in (
            ()
            if production_policy
            else item.get("position_hypotheses", [])
        ):
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
                fallback_engine in {
                    "hezar-crnn-fa-v2-onnx",
                    "crnn-onnx",
                }
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
        raw_guess_text = ""
        raw_guess_norm = ""
        raw_guess_confidence = 0.0
        raw_guess_engine = ""
        raw_guess_reason = ""
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
            # Preserve the actual best hypothesis for operator review, but do
            # not turn a rejected single-frame hypothesis into accepted truth.
            # It is displayed and measured as an experimental guess only.
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
            best_effort = True
            needs_review = True
            raw_guess_text = suggestion["plate"]
            raw_guess_norm = suggestion["plate_norm"]
            raw_guess_confidence = float(suggestion["confidence"])
            raw_guess_engine = str(suggestion["engine"])
            raw_guess_reason = "strict-decoder-rejected"
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
            raw_guess_text = text
            raw_guess_confidence = float(partial["score"])
            raw_guess_engine = ocr_engine
            raw_guess_reason = "partial-character-evidence"
        if valid:
            raw_guess_norm = normalize_plate(text)
            raw_guess_text = format_iran_plate(raw_guess_norm)
            raw_guess_confidence = float(ocr_confidence)
            raw_guess_engine = ocr_engine
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
        detector_model_revision = str(
            item.get("model_revision", "") or call_detector_revision
        ).strip()
        ocr_model_revision = str(
            raw_guess_engine or ocr_engine
        ).strip()
        detector_revision_label = str(
            detector_metadata.get("detector_variant")
            or detector_variant
            or "detector"
        ).strip().lower()
        runtime_model_revision = (
            detector_revision_label
            + ":"
            + detector_model_revision
            + "+ocr:"
            + ocr_model_revision
            if detector_model_revision
            else ocr_model_revision
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
            "quadrilateral": item.get("quadrilateral"),
            "crop_geometry": item.get(
                "crop_geometry",
                "axis-aligned",
            ),
            "static_overlay_overlap": float(
                item.get("static_overlay_overlap", 0.0)
            ),
            "ocr_engine": ocr_engine,
            "ocr_alternative": ocr_alternative,
            "ocr_disagreement": ocr_disagreement,
            "whole_plate_ocr_attempted": whole_plate_ocr_attempted,
            "dedicated_ocr_attempted": bool(
                item.get("direct_ocr_attempted")
            ),
            "dedicated_ocr_ignored": bool(
                production_policy
                and (
                    item.get("direct_ocr_attempted")
                    or item.get("direct_text")
                    or item.get("plate_hypotheses")
                    or item.get("position_hypotheses")
                )
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
            "read_status": (
                "experimental-guess"
                if best_effort and needs_review
                else "confirmed-ai"
                if valid
                else "unreadable"
            ),
            "raw_guess_text": raw_guess_text,
            "raw_guess_norm": raw_guess_norm,
            "raw_guess_confidence": raw_guess_confidence,
            "raw_guess_engine": raw_guess_engine,
            "raw_guess_reason": raw_guess_reason,
            "detector_model_revision": detector_model_revision,
            "ocr_model_revision": ocr_model_revision,
            "model_revision": runtime_model_revision,
            "experimental": bool(best_effort and needs_review),
            "hypotheses_accepted_for_consensus": bool(
                (valid and not needs_review)
                or (
                    not plate_hypotheses
                    and bool(position_hypotheses)
                )
            ),
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


class _BoxKalman:
    """Small constant-velocity Kalman filter for one plate rectangle."""

    def __init__(self, bbox):
        x1, y1, x2, y2 = (float(value) for value in bbox)
        width = max(2.0, x2 - x1)
        height = max(2.0, y2 - y1)
        self.state = np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                width,
                height,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            dtype=np.float64,
        )
        self.covariance = np.diag(
            [16.0, 16.0, 9.0, 9.0, 100.0, 100.0, 36.0, 36.0]
        )

    @staticmethod
    def _bbox(values) -> tuple[float, float, float, float]:
        cx, cy, width, height = (
            float(value) for value in values[:4]
        )
        width = max(2.0, width)
        height = max(2.0, height)
        return (
            cx - width / 2.0,
            cy - height / 2.0,
            cx + width / 2.0,
            cy + height / 2.0,
        )

    def predict(self, elapsed: float) -> tuple:
        elapsed = min(1.5, max(0.01, float(elapsed)))
        transition = np.eye(8, dtype=np.float64)
        transition[0, 4] = elapsed
        transition[1, 5] = elapsed
        transition[2, 6] = elapsed
        transition[3, 7] = elapsed
        process = np.diag(
            [2.0, 2.0, 1.0, 1.0, 12.0, 12.0, 5.0, 5.0]
        ) * elapsed
        self.state = transition @ self.state
        self.covariance = (
            transition @ self.covariance @ transition.T + process
        )
        return self._bbox(self.state)

    def update(self, bbox) -> tuple:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        measurement = np.array(
            [
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                max(2.0, x2 - x1),
                max(2.0, y2 - y1),
            ],
            dtype=np.float64,
        )
        observation = np.zeros((4, 8), dtype=np.float64)
        observation[:4, :4] = np.eye(4, dtype=np.float64)
        noise = np.diag([7.0, 7.0, 5.0, 5.0])
        innovation = measurement - observation @ self.state
        innovation_covariance = (
            observation @ self.covariance @ observation.T + noise
        )
        gain = (
            self.covariance
            @ observation.T
            @ np.linalg.inv(innovation_covariance)
        )
        self.state = self.state + gain @ innovation
        self.covariance = (
            np.eye(8, dtype=np.float64) - gain @ observation
        ) @ self.covariance
        return self._bbox(self.state)


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
    kalman: _BoxKalman | None = None
    predicted_bbox: tuple | None = None
    last_prediction: float = 0.0
    hits: int = 1
    misses: int = 0
    centers: deque = field(default_factory=lambda: deque(maxlen=6))


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
        min_confirmation_span_seconds=0.12,
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
        self.min_confirmation_span_seconds = max(
            0.05,
            float(min_confirmation_span_seconds),
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

    def _capture_result(
        self,
        track: _Track,
        refresh=False,
        final_unreadable=False,
    ) -> dict | None:
        if track.best_capture_result is None or track.best_frame is None:
            return None
        result = deepcopy(track.best_capture_result)
        raw_guess_norm = normalize_plate(
            result.get("raw_guess_norm")
            or result.get("raw_guess_text")
            or (
                result.get("plate")
                if result.get("best_effort")
                else ""
            )
        )
        has_raw_guess = plausible_plate(raw_guess_norm)
        raw_guess_text = (
            format_iran_plate(raw_guess_norm)
            if has_raw_guess
            else str(result.get("raw_guess_text") or "")
        )
        result.update({
            "plate": (
                raw_guess_text
                if raw_guess_text
                else (
                    "ناخوانا"
                    if final_unreadable
                    else "در حال بررسی"
                )
            ),
            "plate_norm": "",
            "valid": False,
            "ocr_confidence": float(
                result.get("raw_guess_confidence")
                or result.get("ocr_confidence", 0.0)
            ),
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "capture_only": True,
            "capture_refresh": bool(refresh),
            "provisional": not bool(final_unreadable),
            "unreadable_final": bool(
                final_unreadable and not raw_guess_text
            ),
            "best_effort": bool(raw_guess_text),
            "needs_review": True,
            "read_status": (
                "experimental-guess"
                if raw_guess_text
                else (
                    "unreadable"
                    if final_unreadable
                    else "processing"
                )
            ),
            "raw_guess_text": raw_guess_text,
            "raw_guess_norm": raw_guess_norm if has_raw_guess else "",
            "experimental": bool(raw_guess_text),
            "capture_frame": track.best_frame.copy(),
            "consensus_votes": 0,
            "guess_supporting_frames": 0,
            "consensus_span_seconds": 0.0,
            "auto_confirm_min_frames": self.min_votes,
            "auto_confirm_min_span_seconds": (
                self.min_confirmation_span_seconds
            ),
        })
        if final_unreadable and has_raw_guess:
            return auto_confirm_guess(result)
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
            score, overlap, proximity, size_ratio, _direction = (
                self._association_score(track, result)
            )
            if score > best_score and (
                overlap >= 0.18
                or (proximity >= 0.62 and size_ratio >= 0.55)
            ):
                best, best_score = track, score
        return best

    @staticmethod
    def _tracking_confidence(result: dict) -> float:
        return min(
            1.0,
            max(
                0.0,
                float(
                    result.get(
                        "detector_confidence",
                        result.get("confidence", 0.0),
                    )
                ),
            ),
        )

    @staticmethod
    def _strong_plate_identity(result: dict) -> str:
        explicit = normalize_plate(
            result.get("association_plate_norm", "")
        )
        if (
            result.get("association_plate_strong")
            and plausible_plate(explicit)
        ):
            return explicit
        accepted = normalize_plate(result.get("plate_norm", ""))
        confidence = float(
            result.get(
                "ocr_confidence",
                result.get("confidence", 0.0),
            )
        )
        if (
            result.get("valid")
            and confidence >= 0.78
            and plausible_plate(accepted)
        ):
            return accepted
        return ""

    @classmethod
    def _track_plate_identity(cls, track: _Track) -> str:
        emitted = normalize_plate(track.emitted_plate)
        if plausible_plate(emitted):
            return emitted
        identities = [
            cls._strong_plate_identity(row)
            for row in track.observations
        ]
        counts = Counter(value for value in identities if value)
        if not counts:
            return ""
        identity, count = counts.most_common(1)[0]
        return identity if count >= 2 else ""

    @classmethod
    def _identity_conflict(
        cls,
        track: _Track,
        result: dict,
    ) -> bool:
        anchored = cls._track_plate_identity(track)
        observed = cls._strong_plate_identity(result)
        if not anchored or not observed or anchored == observed:
            return False
        distance = sum(
            left != right
            for left, right in zip(anchored, observed)
        )
        # Once an event has been emitted, its identity is immutable. A later
        # strong full-plate read with even one changed slot must start a new
        # physical track instead of being swallowed by the one-shot emitter.
        if plausible_plate(track.emitted_plate):
            return True
        # A short detection gap is also a vehicle-boundary signal. It lets two
        # very similar plates use the same lane without treating ordinary
        # one-frame OCR noise as a new vehicle.
        if track.misses >= 2:
            return True
        return bool(
            distance >= 3
            or (anchored[2] != observed[2] and distance >= 2)
        )

    @staticmethod
    def _association_score(track: _Track, result: dict) -> tuple:
        predicted = track.predicted_bbox or track.bbox
        if len(track.centers) >= 2:
            previous_x, previous_y = track.centers[-2]
            current_x, current_y = track.centers[-1]
            velocity_x = current_x - previous_x
            velocity_y = current_y - previous_y
            if math.hypot(velocity_x, velocity_y) >= 2.0:
                width = max(2.0, predicted[2] - predicted[0])
                height = max(2.0, predicted[3] - predicted[1])
                projected_x = current_x + velocity_x
                projected_y = current_y + velocity_y
                predicted = (
                    projected_x - width / 2.0,
                    projected_y - height / 2.0,
                    projected_x + width / 2.0,
                    projected_y + height / 2.0,
                )
        detected = result["bbox"]
        overlap = bbox_iou(predicted, detected)
        px1, py1, px2, py2 = (
            float(value) for value in predicted
        )
        dx1, dy1, dx2, dy2 = (
            float(value) for value in detected
        )
        predicted_width = max(2.0, px2 - px1)
        predicted_height = max(2.0, py2 - py1)
        detected_width = max(2.0, dx2 - dx1)
        detected_height = max(2.0, dy2 - dy1)
        center_distance = math.hypot(
            (px1 + px2 - dx1 - dx2) / 2.0,
            (py1 + py2 - dy1 - dy2) / 2.0,
        )
        scale = max(
            predicted_width,
            predicted_height,
            detected_width,
            detected_height,
        )
        proximity = max(0.0, 1.0 - center_distance / (scale * 1.6))
        predicted_area = predicted_width * predicted_height
        detected_area = detected_width * detected_height
        size_ratio = min(predicted_area, detected_area) / max(
            predicted_area,
            detected_area,
        )
        direction_consistency = 0.5
        if len(track.centers) >= 2:
            previous_x, previous_y = track.centers[-2]
            current_x, current_y = track.centers[-1]
            candidate_x = (dx1 + dx2) / 2.0
            candidate_y = (dy1 + dy2) / 2.0
            velocity_x = current_x - previous_x
            velocity_y = current_y - previous_y
            observed_x = candidate_x - current_x
            observed_y = candidate_y - current_y
            velocity_length = math.hypot(velocity_x, velocity_y)
            observed_length = math.hypot(observed_x, observed_y)
            if velocity_length >= 1.0 and observed_length >= 1.0:
                cosine = (
                    velocity_x * observed_x
                    + velocity_y * observed_y
                ) / (velocity_length * observed_length)
                direction_consistency = min(
                    1.0,
                    max(0.0, (cosine + 1.0) / 2.0),
                )
        score = (
            0.55 * overlap
            + 0.20 * proximity
            + 0.10 * size_ratio
            + 0.15 * direction_consistency
        )
        return (
            score,
            overlap,
            proximity,
            size_ratio,
            direction_consistency,
        )

    def _associate(self, results, timestamp: float) -> dict[int, _Track]:
        """ByteTrack-style two-pass association over high/low detections."""

        active = [
            track
            for track in self._tracks.values()
            if timestamp - track.last_seen <= self.max_age_seconds
        ]
        for track in active:
            if track.kalman is None:
                track.kalman = _BoxKalman(track.bbox)
            elapsed = timestamp - (
                track.last_prediction or track.last_seen
            )
            track.predicted_bbox = track.kalman.predict(elapsed)
            track.last_prediction = timestamp

        high = [
            index
            for index, result in enumerate(results)
            if self._tracking_confidence(result) >= 0.45
        ]
        low = [
            index
            for index, result in enumerate(results)
            if index not in high
        ]
        assigned: dict[int, _Track] = {}
        available = {track.track_id: track for track in active}

        def match(indices, second_pass=False):
            candidates = {}
            for index in indices:
                for track in available.values():
                    (
                        score,
                        overlap,
                        proximity,
                        size_ratio,
                        direction_consistency,
                    ) = (
                        self._association_score(track, results[index])
                    )
                    accepted = (
                        overlap >= (0.08 if second_pass else 0.12)
                        or (
                            proximity
                            >= (0.52 if second_pass else 0.62)
                            and size_ratio >= 0.55
                        )
                    )
                    if (
                        len(track.centers) >= 2
                        and direction_consistency < 0.10
                        and overlap < 0.28
                    ):
                        accepted = False
                    if self._identity_conflict(
                        track,
                        results[index],
                    ):
                        accepted = False
                    if accepted:
                        candidates.setdefault(index, []).append(
                            (track.track_id, float(score))
                        )

            ordered_indices = tuple(
                index for index in indices if index in candidates
            )
            track_ids = tuple(sorted(available))
            track_bits = {
                track_id: 1 << position
                for position, track_id in enumerate(track_ids)
            }

            if len(track_ids) > 12 or len(ordered_indices) > 10:
                ranked = sorted(
                    (
                        (score, index, track_id)
                        for index, options in candidates.items()
                        for track_id, score in options
                    ),
                    reverse=True,
                )
                selected_pairs = []
                used_detections = set()
                used_tracks = set()
                for _score, index, track_id in ranked:
                    if (
                        index in used_detections
                        or track_id in used_tracks
                    ):
                        continue
                    selected_pairs.append((index, track_id))
                    used_detections.add(index)
                    used_tracks.add(track_id)
            else:

                @lru_cache(maxsize=None)
                def solve(position, used_mask):
                    if position >= len(ordered_indices):
                        return 0.0, ()
                    index = ordered_indices[position]
                    best_score, best_pairs = solve(
                        position + 1,
                        used_mask,
                    )
                    for track_id, score in candidates[index]:
                        bit = track_bits[track_id]
                        if used_mask & bit:
                            continue
                        tail_score, tail_pairs = solve(
                            position + 1,
                            used_mask | bit,
                        )
                        candidate_score = score + tail_score
                        candidate_pairs = (
                            (index, track_id),
                            *tail_pairs,
                        )
                        if (
                            candidate_score > best_score + 1e-9
                            or (
                                abs(candidate_score - best_score) <= 1e-9
                                and (
                                    not best_pairs
                                    or candidate_pairs < best_pairs
                                )
                            )
                        ):
                            best_score = candidate_score
                            best_pairs = candidate_pairs
                    return best_score, best_pairs

                _score, selected_pairs = solve(0, 0)
            used_detections = set()
            used_tracks = set()
            for index, track_id in selected_pairs:
                if track_id not in available:
                    continue
                assigned[index] = available[track_id]
                used_detections.add(index)
                used_tracks.add(track_id)
            for track_id in used_tracks:
                available.pop(track_id, None)
            return used_detections

        matched_high = match(high, second_pass=False)
        unmatched_low = [
            index for index in low if index not in matched_high
        ]
        match(unmatched_low, second_pass=True)

        for track in available.values():
            track.misses += 1
            if track.predicted_bbox is not None:
                track.bbox = track.predicted_bbox

        for index, result in enumerate(results):
            if index in assigned:
                continue
            track = _Track(
                track_id=self._next_track_id,
                bbox=tuple(result["bbox"]),
                first_seen=timestamp,
                last_seen=timestamp,
                kalman=_BoxKalman(result["bbox"]),
                last_prediction=timestamp,
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1
            assigned[index] = track
        return assigned

    def _expire(self, timestamp: float):
        stale = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen > self.max_age_seconds * 2.2
        ]
        finalized = []
        for track_id in stale:
            track = self._tracks.pop(track_id, None)
            if track is None:
                continue
            final_read = self._final_track_result(
                track,
                timestamp,
                force=True,
            )
            if final_read is not None:
                track.unreadable_finalized = True
                finalized.append(final_read)
        return finalized

    def _final_track_result(
        self,
        track: _Track,
        timestamp: float,
        force=False,
    ) -> dict | None:
        if track.emitted_plate or track.unreadable_finalized:
            return None
        consensus = self._consensus(track)
        if consensus is not None:
            return consensus
        if not self.emit_unreadable:
            return None
        if not force and not self._unreadable_ready(track, timestamp):
            return None
        return (
            self._best_effort_result(track)
            or self._capture_result(
                track,
                refresh=True,
                final_unreadable=True,
            )
        )

    @staticmethod
    def _observation_weight(row: dict) -> float:
        confidence = min(
            1.0,
            max(
                0.0,
                float(
                    row.get("ocr_confidence")
                    or row.get("raw_guess_confidence")
                    or row.get("confidence", 0.0)
                ),
            ),
        )
        quality = min(1.0, max(0.0, float(row.get("quality_score", 0.0))))
        return confidence * max(0.25, quality)

    @staticmethod
    def _position_probabilities(
        row: dict,
        include_rejected=False,
    ) -> list[dict[str, float]]:
        candidates = {}
        primary = normalize_plate(row.get("plate", ""))
        consensus_allowed = bool(
            row.get(
                "hypotheses_accepted_for_consensus",
                (
                    row.get("valid")
                    and not row.get("needs_review")
                )
                or (
                    bool(row.get("position_hypotheses"))
                    and not bool(row.get("plate_hypotheses"))
                ),
            )
        )
        if (
            row.get("valid")
            and (include_rejected or consensus_allowed)
            and plausible_plate(primary)
        ):
            candidates[primary] = max(
                0.05,
                float(
                    row.get("ocr_confidence")
                    or row.get("confidence", 0.0)
                ),
            )
        for hypothesis in (
            row.get("plate_hypotheses", [])
            if include_rejected or consensus_allowed
            else []
        ):
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
        for hypothesis in (
            row.get("position_hypotheses", [])
            if include_rejected or consensus_allowed
            else []
        ):
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
            probabilities = self._position_probabilities(
                row,
                include_rejected=False,
            )
            if probabilities:
                copied = deepcopy(row)
                copied["_position_probabilities"] = probabilities
                evidence.append(copied)

        if len(evidence) < self.min_votes:
            return None
        # Keep only the five clearest plate crops. Long tracks otherwise let
        # many blurred frames overwhelm the few frames that contain the actual
        # character detail.
        evidence = sorted(
            evidence,
            key=lambda row: (
                self._observation_weight(row),
                float(row.get("quality_score", 0.0)),
                float(row.get("detector_confidence", 0.0)),
            ),
            reverse=True,
        )[:5]

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

        # Positional voting may only resolve characters after the same complete
        # plate has appeared as the top full-plate observation in independent
        # frames. This prevents a synthetic hybrid that never existed.
        whole_plate_support = []
        for row in evidence:
            candidates = {
                normalize_plate(row.get("plate_norm")),
                normalize_plate(row.get("plate")),
                normalize_plate(row.get("raw_guess_norm")),
                normalize_plate(row.get("raw_guess_text")),
            }
            if winner_norm in candidates:
                whole_plate_support.append(row)
        support_times = [
            float(row.get("_observed_at", track.last_seen))
            for row in whole_plate_support
        ]
        support_span = (
            max(support_times) - min(support_times)
            if len(support_times) >= 2
            else 0.0
        )
        if (
            len(whole_plate_support) < self.min_votes
            or support_span + 1e-9
            < self.min_confirmation_span_seconds
        ):
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
        result["experimental"] = False
        result["raw_guess_text"] = result["plate"]
        result["raw_guess_norm"] = winner_norm
        result["raw_guess_confidence"] = float(
            result.get("ocr_confidence", result["confidence"])
        )
        result["raw_guess_engine"] = str(
            result.get("ocr_engine", "")
        )
        result["guess_supporting_frames"] = len(
            whole_plate_support
        )
        result["consensus_span_seconds"] = round(
            support_span,
            4,
        )
        result["auto_confirm_min_frames"] = self.min_votes
        result["auto_confirm_min_span_seconds"] = (
            self.min_confirmation_span_seconds
        )

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
            or "hezar-crnn-fa" in name
            for name in engine_names
        )
        character_supported = any(
            "dedicated-character-detector" in name
            or "cnn-onnx" in name
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
            probabilities = self._position_probabilities(
                row,
                include_rejected=True,
            )
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
            "plate_norm": "",
            "valid": False,
            "best_effort": True,
            "needs_review": True,
            "read_status": (
                "experimental-guess"
                if complete
                else "partial"
            ),
            "confidence": round(confidence, 4),
            "track_id": track.track_id,
            "first_seen": track.first_seen,
            "last_seen": track.last_seen,
            "capture_only": not complete,
            "capture_refresh": True,
            "provisional": False,
            "unreadable_final": False,
            "partial_final": not complete,
            "raw_guess_text": (
                format_iran_plate(normalized)
                if complete
                else _partial_plate_text(winners)
            ),
            "raw_guess_norm": normalized if complete else "",
            "raw_guess_confidence": round(confidence, 4),
            "raw_guess_reason": (
                "multi-frame-rejected-hypotheses"
                if complete
                else "partial-character-evidence"
            ),
            "experimental": True,
            "hypotheses_accepted_for_consensus": False,
            "consensus_votes": min(vote_counts.values()),
            "consensus_observations": len(evidence),
        })
        full_support = []
        if complete:
            for row, _probabilities in evidence:
                candidates = {
                    normalize_plate(row.get("plate")),
                    normalize_plate(row.get("raw_guess_norm")),
                    normalize_plate(row.get("raw_guess_text")),
                }
                candidates.update(
                    normalize_plate(
                        hypothesis.get("plate_norm")
                        or hypothesis.get("plate")
                    )
                    for hypothesis in row.get(
                        "plate_hypotheses",
                        [],
                    )
                )
                if normalized in candidates:
                    full_support.append(row)
        support_times = [
            float(row.get("_observed_at", track.last_seen))
            for row in full_support
        ]
        basis["guess_supporting_frames"] = len(full_support)
        basis["consensus_span_seconds"] = round(
            (
                max(support_times) - min(support_times)
                if len(support_times) >= 2
                else 0.0
            ),
            4,
        )
        basis["auto_confirm_min_frames"] = self.min_votes
        basis["auto_confirm_min_span_seconds"] = (
            self.min_confirmation_span_seconds
        )
        if complete:
            basis = auto_confirm_guess(basis)
        if track.best_frame is not None:
            basis["capture_frame"] = track.best_frame.copy()
        if track.best_capture_result is not None:
            basis["bbox"] = tuple(track.best_capture_result["bbox"])
        return basis

    def update(self, results, timestamp=None, frame=None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        emitted = self._expire(timestamp)
        results = list(results)
        assigned = self._associate(results, timestamp)

        for index, result in enumerate(results):
            track = assigned[index]
            if track.kalman is None:
                track.kalman = _BoxKalman(result["bbox"])
            filtered_bbox = track.kalman.update(result["bbox"])
            track.predicted_bbox = filtered_bbox
            track.bbox = filtered_bbox
            track.last_seen = timestamp
            track.last_prediction = timestamp
            track.hits += 1
            track.misses = 0
            observed_bbox = result["bbox"]
            track.centers.append((
                (observed_bbox[0] + observed_bbox[2]) / 2.0,
                (observed_bbox[1] + observed_bbox[3]) / 2.0,
            ))
            result["track_id"] = track.track_id
            result["tracking_engine"] = (
                "bytetrack-kalman+optical-flow"
            )
            result["tracking_bbox"] = tuple(
                round(float(value), 3)
                for value in filtered_bbox
            )
            observation = deepcopy(result)
            observation["_observed_at"] = timestamp
            track.observations.append(observation)
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

            # One physical track owns one immutable identity. A clearer frame
            # for that same identity may refresh its media, but it must reuse
            # the event_id in LiveANPRWorker rather than create a duplicate.
            if not track.emitted_plate:
                track.emitted_plate = consensus["plate_norm"]
                track.emitted_at = timestamp
                track.capture_event_score = track.best_capture_score
                emitted.append(consensus)
            elif (
                consensus["plate_norm"] == track.emitted_plate
                and capture_improved
                and track.best_capture_score
                > track.capture_event_score + 0.04
            ):
                consensus["capture_refresh"] = True
                track.capture_event_score = track.best_capture_score
                emitted.append(consensus)

        return emitted

    def active_track_ids(self) -> set[int]:
        return set(self._tracks)

    def retire_tracks(self, track_ids) -> None:
        """End tracker fragments after the visit ledger confirms absence."""

        for track_id in track_ids:
            self._tracks.pop(int(track_id), None)

    def flush(self):
        rows = []
        for track in list(self._tracks.values()):
            final_read = self._final_track_result(
                track,
                max(
                    track.last_seen,
                    track.first_seen + self.min_unreadable_seconds,
                ),
                force=True,
            )
            if final_read is not None:
                track.unreadable_finalized = True
                rows.append(final_read)
        return rows
