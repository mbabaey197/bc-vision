"""Robust OCR adapter for Iranian plates.

The module keeps optional OCR engines isolated, creates illumination/blur variants,
reassembles split OCR tokens using their spatial order, and repairs only
position-compatible OCR confusions against the Iranian 2+letter+3+2 layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import os
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np

from .plate_rules import (
    ALLOWED_PLATE_LETTERS,
    PERSIAN_PLATE_LETTERS,
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)

_easy_reader = None
_easy_reader_key = None
_last_status = {
    "engine": "none",
    "easyocr_error": "",
    "tesseract_error": "",
    "candidate_count": 0,
}

DIGIT_CONFUSIONS = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1", "|": "1",
    "Z": "2", "E": "3", "A": "4", "S": "5",
    "ه": "5", "ع": "6", "G": "6", "T": "7",
    "B": "8", "P": "9", "و": "9",
}
LETTER_CONFUSIONS = {
    "A": ("ا",), "B": ("ب",), "P": ("پ",),
    "T": ("ت", "ط"), "J": ("ج",), "D": ("د",),
    "Z": ("ز", "ژ"), "S": ("س", "ص"), "E": ("ع",),
    "F": ("ف",), "Q": ("ق",), "K": ("ک",),
    "G": ("گ",), "L": ("ل",), "M": ("م",),
    "N": ("ن",), "V": ("و",), "H": ("ه",), "Y": ("ی",),
    "ث": ("ث",), "ش": ("ش",),
}

EASYOCR_ALLOWLIST = (
    "0123456789"
    "۰۱۲۳۴۵۶۷۸۹"
    "٠١٢٣٤٥٦٧٨٩"
    + PERSIAN_PLATE_LETTERS
    + "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


@dataclass(frozen=True)
class OCRToken:
    text: str
    confidence: float
    x_center: float
    y_center: float = 0.0


def get_ocr_status():
    return dict(_last_status)


def _safe_confidence(value) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return 0.0


def _gamma(gray: np.ndarray, value: float) -> np.ndarray:
    inv = 1.0 / max(value, 0.01)
    table = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def _variants(image: np.ndarray | None) -> list[np.ndarray]:
    if image is None or getattr(image, "size", 0) == 0:
        return []
    height, width = image.shape[:2]
    scale = min(5.0, max(1.0, 520.0 / max(width, 1)))
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized.copy()
    clahe = cv2.createCLAHE(clipLimit=2.7, tileGridSize=(8, 8)).apply(gray)
    denoised = cv2.bilateralFilter(clahe, 7, 40, 40)
    blur = cv2.GaussianBlur(denoised, (0, 0), 1.15)
    sharpened = cv2.addWeighted(denoised, 1.85, blur, -0.85, 0)
    mean = float(np.mean(gray))
    illumination = _gamma(gray, 1.65 if mean < 85 else (0.72 if mean > 190 else 1.0))
    adaptive = cv2.adaptiveThreshold(
        sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7,
    )
    otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    variants = [resized, clahe, sharpened, illumination, adaptive, otsu]
    unique: list[np.ndarray] = []
    fingerprints = set()
    for variant in variants:
        thumb = cv2.resize(variant, (64, 16), interpolation=cv2.INTER_AREA)
        key = thumb.tobytes()
        if key not in fingerprints:
            fingerprints.add(key)
            unique.append(variant)
    return unique


def _easyocr_model_dir() -> Path:
    configured = os.environ.get("BCVISION_EASYOCR_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        from app.config import DATA_DIR
        return Path(DATA_DIR) / "models" / "easyocr"
    except Exception:
        return Path.home() / ".EasyOCR" / "model"


def _get_easyocr_reader():
    global _easy_reader, _easy_reader_key
    model_dir = _easyocr_model_dir()
    key = str(model_dir.resolve())
    if _easy_reader is not None and _easy_reader_key == key:
        return _easy_reader
    import easyocr
    model_dir.mkdir(parents=True, exist_ok=True)
    allow_download = os.environ.get("BCVISION_EASYOCR_DOWNLOAD", "0") == "1"
    _easy_reader = easyocr.Reader(
        ["fa", "en"], gpu=False, verbose=False,
        model_storage_directory=str(model_dir),
        download_enabled=allow_download,
    )
    _easy_reader_key = key
    return _easy_reader


def _token_variants(text: str, role: str = "middle") -> list[tuple[str, float]]:
    normalized = normalize_plate(text)
    if not normalized:
        return []
    candidates: dict[str, float] = {normalized: 0.0}
    if len(normalized) > 1:
        candidates.setdefault(normalized[::-1], 0.055)
    digits = "".join(ch for ch in normalized if ch.isdigit())
    letters = "".join(ch for ch in normalized if not ch.isdigit())
    if digits and len(letters) == 1 and len(normalized) <= 4:
        digits_penalty = 0.0 if role == "first" else 0.025
        letter_penalty = 0.0 if role == "last" else 0.025
        candidates[digits + letters] = min(candidates.get(digits + letters, 99.0), digits_penalty)
        candidates[letters + digits] = min(candidates.get(letters + digits, 99.0), letter_penalty)
        if any("\u0600" <= ch <= "\u06ff" for ch in normalized):
            candidates[normalized] = max(candidates.get(normalized, 0.0), 0.035)
    return sorted(candidates.items(), key=lambda item: (item[1], item[0]))[:4]


def _position_choices(char: str, position: int) -> list[tuple[str, float]]:
    digit_positions = {0, 1, 3, 4, 5, 6, 7}
    if position in digit_positions:
        if char.isdigit():
            return [(char, 0.0)]
        replacement = DIGIT_CONFUSIONS.get(char)
        return [(replacement, 0.20)] if replacement else []
    choices: dict[str, float] = {}
    if char in ALLOWED_PLATE_LETTERS:
        choices[char] = 0.0
    for replacement in LETTER_CONFUSIONS.get(char, ()):
        if replacement in ALLOWED_PLATE_LETTERS:
            choices[replacement] = min(choices.get(replacement, 99.0), 0.18)
    return sorted(choices.items(), key=lambda item: (item[1], item[0]))


def _align_to_template(text: str, max_skips: int = 3) -> list[tuple[str, float]]:
    source = normalize_plate(text)
    if len(source) < 7 or len(source) > 12:
        return []
    beam = [(0, "", 0.0, 0)]
    for position in range(8):
        next_states = []
        for source_index, output, penalty, skipped in beam:
            max_index = min(len(source), source_index + (max_skips - skipped) + 1)
            for index in range(source_index, max_index):
                extra_skips = index - source_index
                for replacement, replacement_penalty in _position_choices(source[index], position):
                    next_states.append((
                        index + 1,
                        output + replacement,
                        penalty + replacement_penalty + (0.16 * extra_skips),
                        skipped + extra_skips,
                    ))
        next_states.sort(key=lambda state: (state[2], -state[0], state[1]))
        beam = next_states[:48]
        if not beam:
            return []
    repaired: dict[str, float] = {}
    for source_index, output, penalty, skipped in beam:
        trailing = len(source) - source_index
        total_skips = skipped + trailing
        if total_skips > max_skips or not plausible_plate(output):
            continue
        total_penalty = penalty + (0.16 * trailing)
        repaired[output] = min(repaired.get(output, 99.0), total_penalty)
    return sorted(repaired.items(), key=lambda item: (item[1], item[0]))[:12]


def _weighted_confidence(tokens: Sequence[OCRToken]) -> float:
    total_weight = 0.0
    weighted = 0.0
    for token in tokens:
        length = max(1, len(normalize_plate(token.text)))
        weighted += _safe_confidence(token.confidence) * length
        total_weight += length
    return weighted / total_weight if total_weight else 0.0


def _coerce_tokens(detections: Iterable[dict | OCRToken]) -> list[OCRToken]:
    tokens = []
    for item in detections:
        if isinstance(item, OCRToken):
            token = item
        else:
            token = OCRToken(
                text=str(item.get("text", "")),
                confidence=_safe_confidence(item.get("confidence", 0.0)),
                x_center=float(item.get("x_center", 0.0)),
                y_center=float(item.get("y_center", 0.0)),
            )
        normalized = normalize_plate(token.text)
        if not normalized or normalized in {"IR", "IRI", "IRAN"}:
            continue
        tokens.append(OCRToken(normalized, token.confidence, token.x_center, token.y_center))
    return tokens


def _candidate_token_groups(tokens: Sequence[OCRToken]) -> list[list[OCRToken]]:
    ordered = sorted(tokens, key=lambda token: (token.y_center, token.x_center))
    if len(ordered) > 7:
        ordered = sorted(ordered, key=lambda token: token.confidence, reverse=True)[:7]
        ordered.sort(key=lambda token: (token.y_center, token.x_center))
    groups: list[list[OCRToken]] = []
    for start in range(len(ordered)):
        for end in range(start + 1, min(len(ordered), start + 6) + 1):
            group = ordered[start:end]
            char_count = sum(len(token.text) for token in group)
            if 6 <= char_count <= 12:
                groups.append(group)
    if not groups and ordered:
        groups.append(list(ordered))
    return groups[:48]


def _assemble_detections(detections: Iterable[dict | OCRToken]) -> tuple[str, float]:
    tokens = _coerce_tokens(detections)
    if not tokens:
        return "", 0.0
    best_text = ""
    best_score = 0.0
    fallback_text = ""
    fallback_score = 0.0
    for group in _candidate_token_groups(tokens):
        base_conf = _weighted_confidence(group)
        options = [
            _token_variants(
                token.text,
                "first" if index == 0 else ("last" if index == len(group) - 1 else "middle"),
            )
            for index, token in enumerate(group)
        ]
        if any(not option for option in options):
            continue
        orderings = ((options, 0.0), (list(reversed(options)), 0.09))
        for ordering, order_penalty in orderings:
            for selected in product(*ordering):
                raw = "".join(text for text, _ in selected)
                token_penalty = sum(penalty for _, penalty in selected)
                raw_score = base_conf - token_penalty - order_penalty
                normalized = normalize_plate(raw)
                current_fallback = min(0.45, max(0.0, raw_score * 0.52))
                if current_fallback > fallback_score:
                    fallback_text, fallback_score = normalized, current_fallback
                for repaired, repair_penalty in _align_to_template(raw):
                    vote_bonus = min(0.12, 0.025 * max(0, len(group) - 1))
                    score = base_conf + 0.17 + vote_bonus - token_penalty - order_penalty - repair_penalty
                    score = min(1.0, max(0.0, score))
                    if score > best_score:
                        best_text, best_score = repaired, score
    if best_text:
        return format_iran_plate(best_text), best_score
    return format_iran_plate(fallback_text), fallback_score


def _easyocr(image) -> tuple[str, float]:
    variants = _variants(image)
    if not variants:
        return "", 0.0
    try:
        reader = _get_easyocr_reader()
        best = ("", 0.0)
        total_candidates = 0
        for variant in variants:
            raw = reader.readtext(
                variant, detail=1, paragraph=False,
                allowlist=EASYOCR_ALLOWLIST,
                decoder="beamsearch", beamWidth=5,
                text_threshold=0.45, low_text=0.25,
                link_threshold=0.25, mag_ratio=1.0,
            )
            detections = []
            for box, text, confidence in raw:
                xs = [float(point[0]) for point in box]
                ys = [float(point[1]) for point in box]
                detections.append({
                    "text": text,
                    "confidence": confidence,
                    "x_center": sum(xs) / len(xs),
                    "y_center": sum(ys) / len(ys),
                })
            total_candidates += len(detections)
            candidate = _assemble_detections(detections)
            if candidate[1] > best[1]:
                best = candidate
        _last_status.update(engine="easyocr", easyocr_error="", candidate_count=total_candidates)
        return best
    except Exception as exc:
        _last_status["easyocr_error"] = f"{type(exc).__name__}: {exc}"
        return "", 0.0


def _tesseract(image) -> tuple[str, float]:
    variants = _variants(image)
    if not variants:
        return "", 0.0
    try:
        import pytesseract
        best = ("", 0.0)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        for variant in variants:
            data = pytesseract.image_to_data(variant, config=config, output_type=pytesseract.Output.DICT)
            detections = []
            for text, confidence, left, width, top, height in zip(
                data.get("text", []), data.get("conf", []),
                data.get("left", []), data.get("width", []),
                data.get("top", []), data.get("height", []),
            ):
                if not normalize_plate(text):
                    continue
                detections.append({
                    "text": text,
                    "confidence": max(0.0, float(confidence)) / 100.0,
                    "x_center": float(left) + float(width) / 2.0,
                    "y_center": float(top) + float(height) / 2.0,
                })
            candidate = _assemble_detections(detections)
            if candidate[1] > best[1]:
                best = candidate
        _last_status.update(engine="tesseract", tesseract_error="")
        return best
    except Exception as exc:
        _last_status["tesseract_error"] = f"{type(exc).__name__}: {exc}"
        return "", 0.0


def read_plate(image) -> tuple[str, float]:
    if image is None or getattr(image, "size", 0) == 0:
        return "", 0.0
    candidates = [_easyocr(image), _tesseract(image)]
    valid = [candidate for candidate in candidates if plausible_plate(candidate[0])]
    text, confidence = max(valid or candidates, key=lambda candidate: candidate[1])
    return format_iran_plate(text), float(confidence)
