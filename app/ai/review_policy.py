"""Operator-assisted policy for complete, reviewable ANPR guesses.

An automatically confirmed guess is an operational event, not a training
label. It stays visibly attributable to the AI and requires an explicit
operator confirmation or correction before it can enter the training dataset.
"""
from __future__ import annotations

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


AUTO_CONFIRMED_STATUS = "auto-confirmed"


def complete_guess_norm(result: dict) -> str:
    """Return a complete Iranian plate guess or an empty string."""

    normalized = normalize_plate(
        result.get("raw_guess_norm")
        or result.get("raw_guess_text")
        or result.get("plate_norm")
        or result.get("plate")
    )
    return normalized if plausible_plate(normalized) else ""


def tag_assisted_candidate(result: dict) -> dict | None:
    """Prepare a Shadow result for multi-frame, operator-assisted review."""

    normalized = complete_guess_norm(result)
    if not normalized:
        return None
    row = dict(result)
    row["plate"] = format_iran_plate(normalized)
    row["raw_guess_text"] = row["plate"]
    row["raw_guess_norm"] = normalized
    row["needs_review"] = True
    row["experimental"] = True
    row["assisted_candidate"] = True
    row["engine_lane"] = "candidate-shadow"
    row["read_status"] = "experimental-guess"
    return row


def auto_confirm_guess(result: dict) -> dict:
    """Promote a complete guess to an AI-confirmed, reviewable event."""

    normalized = complete_guess_norm(result)
    if not normalized:
        return result
    row = dict(result)
    row["plate"] = format_iran_plate(normalized)
    row["plate_norm"] = normalized
    row["raw_guess_text"] = row["plate"]
    row["raw_guess_norm"] = normalized
    row["valid"] = True
    row["best_effort"] = True
    row["needs_review"] = True
    row["auto_confirmed"] = True
    row["confirmation_source"] = "ai-auto-guess"
    row["read_status"] = AUTO_CONFIRMED_STATUS
    # Keep this marker until a human confirms/corrects the result. Downstream
    # training code must never mistake the AI guess for a human label.
    row["experimental"] = True
    return row
