"""Strict OCR adapter for Iranian license plates.

The adapter creates difficult-condition image variants, reassembles split OCR
tokens in spatial order, and only accepts characters that are valid in the
Iranian 2+letter+3+2 plate layout. Latin look-alike letters are deliberately
not converted to Persian letters because that caused false reads such as
``ط`` being reported as ``ل`` in real CCTV footage.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import cv2
import numpy as np

from .plate_rules import (
    ALLOWED_PLATE_LETTERS,
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)
from .onnx_crnn import (
    get_crnn_status,
    read_plate_crnn,  # compatibility/diagnostic API; not production-routed
    read_plate_platrix,
)
from .onnx_cnn import (
    get_cnn_status,
    read_plate_cnn,  # compatibility diagnostic API
)
from .onnx_hezar import (
    PRIMARY_ENGINE as HEZAR_ENGINE,
    hezar_status,
    read_plate_hezar_primary,
)

_last_status = {
    "engine": "none",
    "policy": "hezar-v2-then-fixed-platrix-then-character-cnn",
    "hezar_error": "",
    "crnn_error": "",
    "cnn_error": "",
    "candidate_count": 0,
}
PLATRIX_MIN_CONFIDENCE = 0.55
HEZAR_TEMPORAL_MIN_CONFIDENCE = 0.35

# These replacements are only used in numeric positions.
DIGIT_CONFUSIONS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "|": "1",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "ه": "5",
    "ع": "6",
    "G": "6",
    "T": "7",
    "B": "8",
    "P": "9",
    "و": "9",
}

@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    x_center: float
    y_center: float = 0.0


def get_ocr_status():
    status = dict(_last_status)
    hezar = hezar_status()
    status.update({
        "hezar_attempted": bool(hezar.get("attempted")),
        "hezar_model_loaded": bool(hezar.get("model_loaded")),
        "hezar_model_path": hezar.get("model_path", ""),
        "hezar_hypotheses": int(hezar.get("hypotheses", 0)),
        "hezar_accepted": bool(hezar.get("accepted")),
        "hezar_error": hezar.get("error", ""),
    })
    crnn = get_crnn_status()
    status.update({
        "crnn_attempted": bool(crnn.get("attempted")),
        "crnn_model_loaded": bool(crnn.get("model_loaded")),
        "crnn_model_path": crnn.get("model_path", ""),
        "crnn_raw_text": crnn.get("raw_text", ""),
        "crnn_confidence": float(crnn.get("confidence", 0.0)),
        "crnn_error": crnn.get("error", ""),
        "crnn_threads": int(crnn.get("threads", 0)),
    })
    cnn = get_cnn_status()
    status.update({
        "cnn_attempted": bool(cnn.get("attempted")),
        "cnn_model_loaded": bool(cnn.get("model_loaded")),
        "cnn_model_path": cnn.get("model_path", ""),
        "cnn_raw_text": cnn.get("raw_text", ""),
        "cnn_confidence": float(cnn.get("confidence", 0.0)),
        "cnn_error": cnn.get("error", ""),
        "cnn_threads": int(cnn.get("threads", 0)),
    })
    return status


def _safe_confidence(value) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return 0.0


def _gamma(gray: np.ndarray, value: float) -> np.ndarray:
    inv = 1.0 / max(value, 0.01)
    table = np.array(
        [((index / 255.0) ** inv) * 255 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(gray, table)


def _variants(image: np.ndarray | None) -> list[np.ndarray]:
    if image is None or getattr(image, "size", 0) == 0:
        return []

    _, width = image.shape[:2]
    scale = min(5.0, max(1.0, 520.0 / max(width, 1)))
    resized = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if resized.ndim == 3
        else resized.copy()
    )
    clahe = cv2.createCLAHE(
        clipLimit=2.7,
        tileGridSize=(8, 8),
    ).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 7, 40, 40)
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.15)
    sharpened = cv2.addWeighted(
        denoised,
        1.85,
        blur,
        -0.85,
        0,
    )

    mean = float(np.mean(gray))
    illumination = _gamma(
        gray,
        1.65 if mean < 85 else (0.72 if mean > 190 else 1.0),
    )
    adaptive = cv2.adaptiveThreshold(
        sharpened,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    otsu = cv2.threshold(
        sharpened,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]

    variants = [
        resized,
        clahe,
        sharpened,
        illumination,
        adaptive,
        otsu,
    ]
    unique: list[np.ndarray] = []
    fingerprints = set()

    for variant in variants:
        thumb = cv2.resize(
            variant,
            (64, 16),
            interpolation=cv2.INTER_AREA,
        )
        key = thumb.tobytes()
        if key not in fingerprints:
            fingerprints.add(key)
            unique.append(variant)

    return unique


def _token_variants(
    text: str,
    role: str = "middle",
) -> list[tuple[str, float]]:
    normalized = normalize_plate(text)
    if not normalized:
        return []

    candidates: dict[str, float] = {
        normalized: 0.0,
    }

    if len(normalized) > 1:
        candidates.setdefault(
            normalized[::-1],
            0.055,
        )

    digits = "".join(
        character
        for character in normalized
        if character.isdigit()
    )
    letters = "".join(
        character
        for character in normalized
        if not character.isdigit()
    )

    if digits and len(letters) == 1 and len(normalized) <= 4:
        digits_penalty = 0.0 if role == "first" else 0.025
        letter_penalty = 0.0 if role == "last" else 0.025
        candidates[digits + letters] = min(
            candidates.get(digits + letters, 99.0),
            digits_penalty,
        )
        candidates[letters + digits] = min(
            candidates.get(letters + digits, 99.0),
            letter_penalty,
        )
        if any(
            "\u0600" <= character <= "\u06ff"
            for character in normalized
        ):
            candidates[normalized] = max(
                candidates.get(normalized, 0.0),
                0.035,
            )

    return sorted(
        candidates.items(),
        key=lambda item: (item[1], item[0]),
    )[:4]


def _position_choices(
    character: str,
    position: int,
) -> list[tuple[str, float]]:
    digit_positions = {0, 1, 3, 4, 5, 6, 7}

    if position in digit_positions:
        if character.isdigit():
            return [(character, 0.0)]
        replacement = DIGIT_CONFUSIONS.get(character)
        return [(replacement, 0.20)] if replacement else []

    # The plate-letter position is strict. No Latin look-alike is converted to
    # a Persian letter. D/S remain valid directly through plate_rules.
    if character in ALLOWED_PLATE_LETTERS:
        return [(character, 0.0)]

    return []


def _align_to_template(
    text: str,
    max_skips: int = 3,
) -> list[tuple[str, float]]:
    source = normalize_plate(text)
    if len(source) < 7 or len(source) > 12:
        return []

    beam = [(0, "", 0.0, 0)]

    for position in range(8):
        next_states = []

        for source_index, output, penalty, skipped in beam:
            max_index = min(
                len(source),
                source_index + (max_skips - skipped) + 1,
            )

            for index in range(source_index, max_index):
                extra_skips = index - source_index

                for replacement, replacement_penalty in _position_choices(
                    source[index],
                    position,
                ):
                    next_states.append((
                        index + 1,
                        output + replacement,
                        penalty
                        + replacement_penalty
                        + 0.16 * extra_skips,
                        skipped + extra_skips,
                    ))

        next_states.sort(
            key=lambda state: (
                state[2],
                -state[0],
                state[1],
            )
        )
        beam = next_states[:48]

        if not beam:
            return []

    repaired: dict[str, float] = {}

    for source_index, output, penalty, skipped in beam:
        trailing = len(source) - source_index
        total_skips = skipped + trailing

        if (
            total_skips > max_skips
            or not plausible_plate(output)
        ):
            continue

        total_penalty = penalty + 0.16 * trailing
        repaired[output] = min(
            repaired.get(output, 99.0),
            total_penalty,
        )

    return sorted(
        repaired.items(),
        key=lambda item: (item[1], item[0]),
    )[:12]


def _weighted_confidence(
    tokens: Sequence[OCRToken],
) -> float:
    total_weight = 0.0
    weighted = 0.0

    for token in tokens:
        length = max(
            1,
            len(normalize_plate(token.text)),
        )
        weighted += (
            _safe_confidence(token.confidence)
            * length
        )
        total_weight += length

    return weighted / total_weight if total_weight else 0.0


def _coerce_tokens(
    detections: Iterable[dict | OCRToken],
) -> list[OCRToken]:
    tokens = []

    for item in detections:
        if isinstance(item, OCRToken):
            token = item
        else:
            token = OCRToken(
                text=str(item.get("text", "")),
                confidence=_safe_confidence(
                    item.get("confidence", 0.0)
                ),
                x_center=float(item.get("x_center", 0.0)),
                y_center=float(item.get("y_center", 0.0)),
            )

        normalized = normalize_plate(token.text)
        if (
            not normalized
            or normalized in {"IR", "IRI", "IRAN"}
        ):
            continue

        tokens.append(
            OCRToken(
                normalized,
                token.confidence,
                token.x_center,
                token.y_center,
            )
        )

    return tokens


def _candidate_token_groups(
    tokens: Sequence[OCRToken],
) -> list[list[OCRToken]]:
    ordered = sorted(
        tokens,
        key=lambda token: (
            token.y_center,
            token.x_center,
        ),
    )

    if len(ordered) > 7:
        ordered = sorted(
            ordered,
            key=lambda token: token.confidence,
            reverse=True,
        )[:7]
        ordered.sort(
            key=lambda token: (
                token.y_center,
                token.x_center,
            )
        )

    groups: list[list[OCRToken]] = []

    for start in range(len(ordered)):
        for end in range(
            start + 1,
            min(len(ordered), start + 6) + 1,
        ):
            group = ordered[start:end]
            character_count = sum(
                len(token.text)
                for token in group
            )
            if 6 <= character_count <= 12:
                groups.append(group)

    if not groups and ordered:
        groups.append(list(ordered))

    return groups[:48]


def _assemble_detections(
    detections: Iterable[dict | OCRToken],
) -> tuple[str, float]:
    tokens = _coerce_tokens(detections)
    if not tokens:
        return "", 0.0

    best_text = ""
    best_score = 0.0

    for group in _candidate_token_groups(tokens):
        base_confidence = _weighted_confidence(group)
        options = [
            _token_variants(
                token.text,
                (
                    "first"
                    if index == 0
                    else (
                        "last"
                        if index == len(group) - 1
                        else "middle"
                    )
                ),
            )
            for index, token in enumerate(group)
        ]

        if any(not option for option in options):
            continue

        orderings = (
            (options, 0.0),
            (list(reversed(options)), 0.09),
        )

        for ordering, order_penalty in orderings:
            for selected in product(*ordering):
                raw = "".join(
                    text
                    for text, _ in selected
                )
                token_penalty = sum(
                    penalty
                    for _, penalty in selected
                )

                for repaired, repair_penalty in _align_to_template(
                    raw
                ):
                    vote_bonus = min(
                        0.12,
                        0.025 * max(0, len(group) - 1),
                    )
                    score = (
                        base_confidence
                        + 0.17
                        + vote_bonus
                        - token_penalty
                        - order_penalty
                        - repair_penalty
                    )
                    score = min(
                        1.0,
                        max(0.0, score),
                    )

                    if score > best_score:
                        best_text = repaired
                        best_score = score

    if not best_text:
        return "", 0.0

    return format_iran_plate(best_text), best_score


def read_plate_candidate(
    image,
    engine_key=None,
    allow_legacy=True,
    include_evidence=False,
) -> tuple[str, float, str] | dict:
    """Read a plate with whole-plate OCR and character-by-character fallback.

    Hezar v2 and the fixed Platrix model remain the preferred readers.  When
    both reject a mature live crop, ``allow_legacy`` enables the bundled
    character CNN so existing cameras keep the proven per-position fallback.
    """

    def result(text, confidence, engine, hypotheses=()):
        payload = {
            "plate": str(text or ""),
            "plate_norm": normalize_plate(text),
            "confidence": _safe_confidence(confidence),
            "engine": str(engine or "none"),
            "hypotheses": list(hypotheses),
        }
        if include_evidence:
            return payload
        return (
            payload["plate"],
            payload["confidence"],
            payload["engine"],
        )

    if image is None or getattr(image, "size", 0) == 0:
        return result("", 0.0, "none")

    hezar = read_plate_hezar_primary(
        image,
        engine_key=engine_key,
    )
    hezar_hypotheses = []
    for hypothesis in hezar.get("hypotheses", []):
        normalized = normalize_plate(
            hypothesis.get("plate_norm")
            or hypothesis.get("plate")
        )
        confidence = _safe_confidence(
            hypothesis.get(
                "confidence",
                hypothesis.get("score", 0.0),
            )
        )
        ctc_path_score = _safe_confidence(
            hypothesis.get("score", confidence)
        )
        if not plausible_plate(normalized):
            continue
        hezar_hypotheses.append({
            "plate": format_iran_plate(normalized),
            "plate_norm": normalized,
            "engine": HEZAR_ENGINE,
            "confidence": confidence,
            # Consensus uses normalized sequence confidence. Raw CTC path
            # probabilities can be vanishingly small, which would collapse
            # distinct candidates to the same downstream weight floor.
            "score": confidence,
            "ctc_path_score": ctc_path_score,
            # A rejected path is never accepted from one frame. Mature crops
            # may retain sufficiently strong alternatives only as temporal
            # evidence for the strict multi-frame consensus gate.
            "temporal_evidence": bool(
                confidence >= HEZAR_TEMPORAL_MIN_CONFIDENCE
            ),
        })
    hezar_hypotheses.sort(
        key=lambda row: (
            row["score"],
            row["confidence"],
            row["plate_norm"],
        ),
        reverse=True,
    )
    hezar_hypotheses = hezar_hypotheses[:5]
    if hezar.get("accepted") and plausible_plate(
        hezar.get("plate_norm", "")
    ):
        _last_status.update(
            engine=HEZAR_ENGINE,
            hezar_error="",
            crnn_error="",
            cnn_error="",
            candidate_count=len(hezar.get("hypotheses", [])),
        )
        return result(
            format_iran_plate(hezar["plate_norm"]),
            float(hezar.get("confidence", 0.0)),
            HEZAR_ENGINE,
            hezar_hypotheses,
        )

    _last_status.update(
        hezar_error=hezar_status().get("error", "") or "rejected",
        candidate_count=len(hezar.get("hypotheses", [])),
        cnn_error="",
    )
    platrix_text, platrix_confidence = read_plate_platrix(
        image,
        engine_key=engine_key,
    )
    if (
        plausible_plate(platrix_text)
        and float(platrix_confidence) >= PLATRIX_MIN_CONFIDENCE
    ):
        _last_status.update(
            engine="platrix-crnn-onnx",
            crnn_error="",
        )
        platrix_norm = normalize_plate(platrix_text)
        hypotheses = list(hezar_hypotheses)
        if all(
            row["plate_norm"] != platrix_norm
            for row in hypotheses
        ):
            hypotheses.append({
                "plate": format_iran_plate(platrix_norm),
                "plate_norm": platrix_norm,
                "engine": "platrix-crnn-onnx",
                "confidence": _safe_confidence(platrix_confidence),
                "score": _safe_confidence(platrix_confidence),
                "temporal_evidence": False,
            })
        return result(
            format_iran_plate(platrix_text),
            float(platrix_confidence),
            "platrix-crnn-onnx",
            hypotheses,
        )
    crnn_error = get_crnn_status().get("error", "")
    if plausible_plate(platrix_text) and not crnn_error:
        crnn_error = "below-production-confidence"

    if allow_legacy:
        cnn_text, cnn_confidence = read_plate_cnn(
            image,
            engine_key=engine_key,
        )
        if plausible_plate(cnn_text):
            cnn_norm = normalize_plate(cnn_text)
            hypotheses = list(hezar_hypotheses)
            if all(
                row["plate_norm"] != cnn_norm
                for row in hypotheses
            ):
                hypotheses.append({
                    "plate": format_iran_plate(cnn_norm),
                    "plate_norm": cnn_norm,
                    "engine": "cnn-onnx",
                    "confidence": _safe_confidence(cnn_confidence),
                    "score": _safe_confidence(cnn_confidence),
                    "temporal_evidence": False,
                })
            _last_status.update(
                engine="cnn-onnx",
                crnn_error=crnn_error or "rejected",
                cnn_error="",
            )
            return result(
                format_iran_plate(cnn_norm),
                float(cnn_confidence),
                "cnn-onnx",
                hypotheses,
            )
        cnn_error = get_cnn_status().get("error", "") or "rejected"
    else:
        cnn_error = "disabled"
    _last_status.update(
        engine="none",
        crnn_error=crnn_error or "rejected",
        cnn_error=cnn_error,
    )

    return result(
        "",
        0.0,
        "none",
        hezar_hypotheses,
    )


def read_plate(
    image,
    engine_key=None,
    allow_legacy=True,
) -> tuple[str, float]:
    text, confidence, _engine = read_plate_candidate(
        image,
        engine_key=engine_key,
        allow_legacy=allow_legacy,
    )
    return text, confidence
