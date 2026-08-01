"""Operator-confirmed ANPR corrections.

Every correction is durable labelled training data.  A complete wrong OCR
reading is corrected immediately when it is seen again, while repeated
character-level mistakes gently re-rank the detector's alternative readings.
Generic labels and incomplete reads are never used as global replacements.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import sqlite3
import threading
import time

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


_CACHE_LOCK = threading.RLock()
_CACHE_KEY = ""
_CACHE_AT = 0.0
_CACHE_ROWS = []
_CACHE_SECONDS = 3.0


def validate_correction(text: str) -> tuple[str, str]:
    normalized = normalize_plate(text)
    if not plausible_plate(normalized):
        raise ValueError("قالب پلاک صحیح نیست")
    return format_iran_plate(normalized), normalized


def learned_plate(text: str) -> tuple[str, str] | None:
    """Return an exact operator-confirmed replacement, when one exists."""

    observed = normalize_plate(text)
    # Learning "ناخوانا" -> one particular plate would replace every future
    # unreadable capture with that plate.  Exact reuse is safe only when the
    # original OCR result itself had a complete Iranian plate shape.
    if not plausible_plate(observed):
        return None
    for row in _confirmed_feedback():
        if normalize_plate(row["observed_norm"]) == observed:
            corrected = normalize_plate(row["corrected_norm"])
            if plausible_plate(corrected):
                return format_iran_plate(corrected), corrected
    return None


def _confirmed_feedback():
    """Load compact confirmed pairs without making ANPR depend on migration."""

    import app.database as database

    global _CACHE_AT, _CACHE_KEY, _CACHE_ROWS
    now = time.monotonic()
    cache_key = str(database.DB_PATH)
    with _CACHE_LOCK:
        if (
            cache_key == _CACHE_KEY
            and now - _CACHE_AT < _CACHE_SECONDS
        ):
            return list(_CACHE_ROWS)

    try:
        with database.connect() as con:
            rows = con.execute(
                "SELECT observed_norm,corrected_norm "
                "FROM anpr_feedback WHERE status='confirmed' "
                "AND observed_norm<>'' AND corrected_norm<>'' "
                "ORDER BY id DESC"
            ).fetchall()
    except sqlite3.DatabaseError:
        rows = []
    compact = [
        {
            "observed_norm": row["observed_norm"],
            "corrected_norm": row["corrected_norm"],
        }
        for row in rows
    ]
    with _CACHE_LOCK:
        _CACHE_KEY = cache_key
        _CACHE_AT = now
        _CACHE_ROWS = compact
    return list(compact)


def invalidate_feedback_cache():
    """Make a newly submitted operator correction visible immediately."""

    global _CACHE_AT, _CACHE_KEY, _CACHE_ROWS
    with _CACHE_LOCK:
        _CACHE_KEY = ""
        _CACHE_AT = 0.0
        _CACHE_ROWS = []


def _adaptive_hypothesis(result: dict) -> tuple[str, int] | None:
    """Select an OCR alternative supported by repeated operator feedback.

    One correction teaches the exact wrong string immediately via
    :func:`learned_plate`.  General character substitutions require at least
    two consistent confirmations and are allowed to select only an alternative
    the OCR engine actually supplied; no unseen plate character is invented.
    """

    primary = normalize_plate(result.get("plate", ""))
    if not plausible_plate(primary):
        return None

    candidates = {}
    for hypothesis in result.get("plate_hypotheses", []):
        normalized = normalize_plate(
            hypothesis.get("plate_norm") or hypothesis.get("plate")
        )
        if not plausible_plate(normalized) or normalized == primary:
            continue
        candidates[normalized] = max(
            candidates.get(normalized, 0.0),
            float(
                hypothesis.get(
                    "score",
                    hypothesis.get("confidence", 0.0),
                )
            ),
        )
    if not candidates:
        return None

    transitions = defaultdict(Counter)
    for row in _confirmed_feedback():
        observed = normalize_plate(row["observed_norm"])
        corrected = normalize_plate(row["corrected_norm"])
        if not plausible_plate(observed) or not plausible_plate(corrected):
            continue
        for position, (wrong, right) in enumerate(zip(observed, corrected)):
            if wrong != right:
                transitions[(position, wrong)][right] += 1

    best = None
    for candidate, model_score in candidates.items():
        learned_positions = 0
        evidence = 0
        for position, (wrong, proposed) in enumerate(zip(primary, candidate)):
            if wrong == proposed:
                continue
            counts = transitions.get((position, wrong))
            if not counts:
                continue
            total = sum(counts.values())
            confirmations = counts[proposed]
            if confirmations >= 2 and confirmations / max(total, 1) >= 0.75:
                learned_positions += 1
                evidence += confirmations
        if learned_positions == 0:
            continue
        # Prefer repeated human evidence, then the OCR model's own score.
        key = (learned_positions, evidence, model_score)
        if best is None or key > best[0]:
            best = (key, candidate, evidence)

    if best is None:
        return None
    return best[1], best[2]


def apply_learned_correction(result: dict) -> dict:
    replacement = learned_plate(result.get("plate", ""))
    row = dict(result)
    if replacement is not None:
        row["plate"], row["plate_norm"] = replacement
        row["valid"] = True
        row["needs_review"] = False
        row["operator_learned"] = True
        row["learning_mode"] = "exact"
        return row

    adaptive = _adaptive_hypothesis(row)
    if adaptive is None:
        return result
    normalized, confirmations = adaptive
    row["plate"] = format_iran_plate(normalized)
    row["plate_norm"] = normalized
    row["valid"] = True
    row["needs_review"] = True
    row["operator_learned"] = True
    row["learning_mode"] = "character-confusion"
    row["learning_confirmations"] = int(confirmations)
    return row
