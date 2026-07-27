"""Operator-confirmed ANPR corrections.

Corrections are deliberately conservative: exact prior OCR readings can be
reused locally, while every correction is retained as labelled training data
for a later reviewed model-training run.
"""
from __future__ import annotations

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


def validate_correction(text: str) -> tuple[str, str]:
    normalized = normalize_plate(text)
    if not plausible_plate(normalized):
        raise ValueError("قالب پلاک صحیح نیست")
    return format_iran_plate(normalized), normalized


def learned_plate(text: str) -> tuple[str, str] | None:
    """Return an exact operator-confirmed replacement, when one exists."""

    observed = normalize_plate(text)
    if not observed:
        return None
    from app.database import connect

    with connect() as con:
        row = con.execute(
            "SELECT corrected_text,corrected_norm FROM anpr_feedback "
            "WHERE observed_norm=? AND status='confirmed' "
            "ORDER BY id DESC LIMIT 1",
            (observed,),
        ).fetchone()
    if not row:
        return None
    return row["corrected_text"], row["corrected_norm"]


def apply_learned_correction(result: dict) -> dict:
    replacement = learned_plate(result.get("plate", ""))
    if replacement is None:
        return result
    row = dict(result)
    row["plate"], row["plate_norm"] = replacement
    row["valid"] = True
    row["operator_learned"] = True
    return row
