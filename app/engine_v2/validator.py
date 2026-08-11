from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


_DIGIT_TRANSLATION = str.maketrans(
    "۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩",
    "01234567890123456789",
)
_CHAR_TRANSLATION = str.maketrans(
    {
        "ي": "ی",
        "ى": "ی",
        "ئ": "ی",
        "ك": "ک",
        "ة": "ه",
        "ۀ": "ه",
        "ؤ": "و",
    }
)
_SEPARATORS = re.compile(r"[\s\-–—_:،,./\\|()\[\]{}]+")
_BIDI_AND_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_LATIN_COUNTRY_LABEL = re.compile(r"(?i)(?<![A-Z])I\.?R\.?A\.?N(?![A-Z])")


@dataclass(frozen=True, slots=True)
class PlateValidation:
    raw: str
    normalized: str
    valid: bool
    reason: str
    pattern: str | None = None
    serial_prefix: str | None = None
    letter: str | None = None
    serial_number: str | None = None
    province: str | None = None


@dataclass(frozen=True, slots=True)
class IranianPlateValidatorConfig:
    # Common private/public/service series. Multi-character tokens must come
    # first when building the regex.
    allowed_letters: tuple[str, ...] = (
        "ا",
        "ب",
        "ج",
        "د",
        "س",
        "ص",
        "ط",
        "ق",
        "ل",
        "م",
        "ن",
        "و",
        "ه",
        "ی",
        "ت",
        "ع",
        "پ",
        "ث",
        "ز",
        "ژ",
        "ش",
        "ف",
        "ک",
        "گ",
    )
    min_province: int = 10
    max_province: int = 99
    allow_diplomatic_latin: bool = True

    def __post_init__(self) -> None:
        if not self.allowed_letters or any(not str(token).strip() for token in self.allowed_letters):
            raise ValueError("allowed_letters must contain non-empty tokens")
        if not 0 <= int(self.min_province) <= int(self.max_province) <= 99:
            raise ValueError("province range must satisfy 0 <= min <= max <= 99")


class IranianPlateValidator:
    """Normalize Persian OCR output and validate Iranian plate structure.

    This is intentionally a structural validator rather than an auto-corrector:
    ambiguous OCR characters are left for temporal voting. Silently replacing
    them here would improve apparent accuracy while increasing false events.
    """

    def __init__(self, config: IranianPlateValidatorConfig | None = None) -> None:
        self.config = config or IranianPlateValidatorConfig()
        tokens = sorted(set(self.config.allowed_letters), key=len, reverse=True)
        letter_pattern = "|".join(re.escape(token) for token in tokens)
        self._standard = re.compile(
            rf"^(?P<prefix>\d{{2}})(?P<letter>{letter_pattern})(?P<serial>\d{{3}})(?P<province>\d{{2}})$"
        )
        self._diplomatic = re.compile(
            r"^(?P<prefix>\d{2})(?P<letter>[DS])(?P<serial>\d{3})(?P<province>\d{2})$",
            re.IGNORECASE,
        )

    @staticmethod
    def normalize(text: str) -> str:
        value = unicodedata.normalize("NFKC", str(text or ""))
        value = value.translate(_DIGIT_TRANSLATION).translate(_CHAR_TRANSLATION)
        value = _BIDI_AND_ZERO_WIDTH.sub("", value)
        # Remove only the complete country label. The previous permissive
        # expression also removed a bare "IR" in the middle of arbitrary OCR
        # output and could turn a malformed plate into a valid diplomatic one.
        value = _LATIN_COUNTRY_LABEL.sub("", value)
        value = value.replace("ایران", "")
        # The shared CTC contract emits the single glyph ``ا``. Some OCR/UI
        # sources spell the same token as ``الف``; normalize both before
        # temporal voting so they cannot become competing readings.
        value = value.replace("الف", "ا")
        value = _SEPARATORS.sub("", value)
        # OCR models sometimes emit the presentation form هـ.
        value = value.replace("هـ", "ه")
        # Diplomatic OCR output is case-insensitive, but the canonical value
        # used by temporal voting must not split `d` and `D` into two groups.
        value = re.sub(r"[ds]", lambda match: match.group(0).upper(), value)
        return value.strip()

    def validate(self, text: str) -> PlateValidation:
        raw = str(text or "")
        normalized = self.normalize(raw)
        if not normalized:
            return PlateValidation(raw, normalized, False, "empty")

        match = self._standard.fullmatch(normalized)
        pattern = "standard"
        if match is None and self.config.allow_diplomatic_latin:
            match = self._diplomatic.fullmatch(normalized)
            pattern = "diplomatic"
        if match is None:
            return PlateValidation(raw, normalized, False, "invalid_structure")

        parts = match.groupdict()
        province = parts["province"]
        province_number = int(province)
        if not self.config.min_province <= province_number <= self.config.max_province:
            return PlateValidation(
                raw,
                normalized,
                False,
                "invalid_province",
                pattern,
                parts["prefix"],
                parts["letter"],
                parts["serial"],
                province,
            )
        if parts["prefix"] == "00" or parts["serial"] == "000":
            return PlateValidation(
                raw,
                normalized,
                False,
                "zero_serial",
                pattern,
                parts["prefix"],
                parts["letter"],
                parts["serial"],
                province,
            )
        return PlateValidation(
            raw=raw,
            normalized=normalized,
            valid=True,
            reason="ok",
            pattern=pattern,
            serial_prefix=parts["prefix"],
            letter=parts["letter"],
            serial_number=parts["serial"],
            province=province,
        )


def validate_iranian_plate(text: str) -> PlateValidation:
    return IranianPlateValidator().validate(text)
