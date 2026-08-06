"""Canonical normalization and validation rules for Iranian license plates."""

import re
import unicodedata


PERSIAN_DIGITS = "\u06f0\u06f1\u06f2\u06f3\u06f4\u06f5\u06f6\u06f7\u06f8\u06f9"
ARABIC_DIGITS = "\u0660\u0661\u0662\u0663\u0664\u0665\u0666\u0667\u0668\u0669"

DIGIT_TRANS = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    "0123456789" * 2,
)
PERSIAN_DIGIT_TRANS = str.maketrans(
    "0123456789" + ARABIC_DIGITS,
    PERSIAN_DIGITS * 2,
)

CHAR_TRANS = str.maketrans({
    "\u064a": "\u06cc",
    "\u0649": "\u06cc",
    "\u0643": "\u06a9",
    "\u0629": "\u0647",
    "\u06c0": "\u0647",
    "\u0624": "\u0648",
    "\u0623": "\u0627",
    "\u0625": "\u0627",
    "\u0622": "\u0627",
})

# Increment this whenever canonical normalization semantics change. Database
# migrations use the value to re-key historical rows instead of only filling
# empty keys once.
PLATE_NORMALIZATION_VERSION = 2

IRAN_WORD = "\u0627\u06cc\u0631\u0627\u0646"
ALEF_WORD = "\u0627\u0644\u0641"

# Personal and special-purpose Iranian vehicle plate letters.
PERSIAN_PLATE_LETTERS = (
    "\u0627"  # Alef / government
    "\u0628"
    "\u067e"
    "\u062a"
    "\u062b"
    "\u062c"
    "\u062f"
    "\u0632"
    "\u0698"
    "\u0633"
    "\u0634"
    "\u0635"
    "\u0637"
    "\u0639"
    "\u0641"
    "\u0642"
    "\u06a9"
    "\u06af"
    "\u0644"
    "\u0645"
    "\u0646"
    "\u0648"
    "\u0647"
    "\u06cc"
)

# Latin letters used by diplomatic and embassy-service plates.
# OCR aliases such as B -> ب are handled by the OCR layer, not validation.
LATIN_PLATE_ALIASES = "DS"

ALLOWED_PLATE_LETTERS = (
    PERSIAN_PLATE_LETTERS
    + LATIN_PLATE_ALIASES
)

IRAN_PLATE_PATTERN = re.compile(
    rf"^(?P<prefix>\d{{2}})"
    rf"(?P<letter>[{re.escape(ALLOWED_PLATE_LETTERS)}])"
    rf"(?P<serial>\d{{3}})"
    rf"(?P<region>\d{{2}})$"
)


def normalize_plate(text):
    """Return a stable key using ASCII digits and a normalized plate letter."""

    # NFKC folds Arabic presentation forms before the explicit Persian/Arabic
    # character mapping.  Applying the mapping before removing ``ایران`` also
    # makes Persian and Arabic yeh spellings of that word equivalent.
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.translate(DIGIT_TRANS).translate(CHAR_TRANS).upper()
    value = "".join(
        char for char in value if not unicodedata.combining(char)
    )

    value = value.replace(ALEF_WORD, "\u0627")
    value = value.replace(IRAN_WORD, "")
    value = value.replace("IRAN", "")
    value = value.replace("IRI", "")
    value = value.replace("IR", "")

    normalized = []
    for char in value:
        # Combining marks, Arabic punctuation, zero-width controls and other
        # symbols must never become part of the database/search identity.
        if unicodedata.combining(char):
            continue
        if "0" <= char <= "9" or "A" <= char <= "Z":
            normalized.append(char)
            continue
        if (
            "\u0600" <= char <= "\u06ff"
            and unicodedata.category(char).startswith("L")
        ):
            normalized.append(char)
    return "".join(normalized)


def split_iran_plate(text):
    """Return the four standard Iranian plate parts or None."""

    match = IRAN_PLATE_PATTERN.fullmatch(normalize_plate(text))

    if match is None:
        return None

    return {
        "prefix": match.group("prefix"),
        "letter": match.group("letter"),
        "serial": match.group("serial"),
        "region": match.group("region"),
    }


def plausible_plate(text):
    """Validate the common 2-letter-3-2 Iranian plate layout."""

    return split_iran_plate(text) is not None


def format_iran_plate(text):
    """Format a valid canonical plate while preserving invalid OCR text."""

    normalized = normalize_plate(text)
    parts = split_iran_plate(normalized)

    if parts is None:
        return normalized

    return (
        f"{parts['prefix']}-"
        f"{parts['letter']}-"
        f"{parts['serial']}-"
        f"{parts['region']}"
    )


def persian_digits(text):
    """Render digits with Persian glyphs without changing the stored key."""

    return str(text or "").translate(PERSIAN_DIGIT_TRANS)


def iran_plate_parts(text):
    """Return display-ready Persian plate parts or None for invalid input."""

    parts = split_iran_plate(text)
    if parts is None:
        return None
    return {
        "prefix": persian_digits(parts["prefix"]),
        "letter": parts["letter"],
        "serial": persian_digits(parts["serial"]),
        "region": persian_digits(parts["region"]),
    }
