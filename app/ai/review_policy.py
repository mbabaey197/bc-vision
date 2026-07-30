"""Operator-assisted policy for complete, reviewable ANPR guesses.

An automatically confirmed guess is an operational event, not a training
label. It stays visibly attributable to the AI and requires an explicit
operator confirmation or correction before it can enter the training dataset.
"""
from __future__ import annotations

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate


AUTO_CONFIRMED_STATUS = "auto-confirmed"
MIN_AUTO_CONFIRM_FRAMES = 3
MIN_AUTO_CONFIRM_SPAN_SECONDS = 0.12


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
    supporting_frames = int(
        row.get(
            "guess_supporting_frames",
            row.get("consensus_votes", 0),
        )
        or 0
    )
    temporal_span = float(
        row.get("consensus_span_seconds", 0.0)
        or 0.0
    )
    minimum_frames = max(
        MIN_AUTO_CONFIRM_FRAMES,
        int(row.get("auto_confirm_min_frames", 0) or 0),
    )
    minimum_span = max(
        MIN_AUTO_CONFIRM_SPAN_SECONDS,
        float(row.get("auto_confirm_min_span_seconds", 0.0) or 0.0),
    )
    if (
        supporting_frames < minimum_frames
        or temporal_span < minimum_span
    ):
        # Keep the useful guess visible, but fail closed. In particular, one
        # good guess followed by unreadable frames cannot become confirmed.
        row["plate_norm"] = ""
        row["valid"] = False
        row["best_effort"] = True
        row["needs_review"] = True
        row["auto_confirmed"] = False
        row["confirmation_source"] = "ai-suggestion"
        row["read_status"] = "experimental-guess"
        row["experimental"] = True
        row["auto_confirmation_blocked"] = (
            "insufficient-independent-frame-evidence"
        )
        return row
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
    row["auto_confirmation_blocked"] = ""
    return row
