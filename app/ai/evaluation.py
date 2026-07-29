"""Operator-labelled ANPR quality measurements.

Only an operator correction supplies truth. Raw guesses are observable, but
they never become training labels or accuracy claims on their own.
"""
from __future__ import annotations

from collections import defaultdict

from .plate_rules import normalize_plate, plausible_plate


def character_distance(observed: str, corrected: str) -> int:
    """Return Levenshtein distance between canonical plate strings."""

    left = normalize_plate(observed)
    right = normalize_plate(corrected)
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1]
                + int(left_character != right_character),
            ))
        previous = current
    return previous[-1]


def feedback_quality_summary(connection) -> dict:
    """Summarize reviewed guesses overall and by immutable model revision."""

    rows = connection.execute(
        "SELECT observed_norm,corrected_norm,observed_engine,"
        "observed_model_revision,character_distance,exact_match "
        "FROM anpr_feedback WHERE status='confirmed' "
        "ORDER BY id"
    ).fetchall()
    groups = defaultdict(list)
    usable = []
    for row in rows:
        observed = normalize_plate(row["observed_norm"])
        corrected = normalize_plate(row["corrected_norm"])
        if not plausible_plate(corrected):
            continue
        distance = (
            character_distance(observed, corrected)
            if observed
            else len(corrected)
        )
        exact = bool(observed and observed == corrected)
        item = {
            "distance": distance,
            "exact": exact,
            "has_guess": plausible_plate(observed),
        }
        usable.append(item)
        revision = str(row["observed_model_revision"] or "").strip()
        engine = str(row["observed_engine"] or "").strip()
        groups[revision or engine or "نامشخص"].append(item)

    def summarize(items):
        guessed = [item for item in items if item["has_guess"]]
        exact = sum(item["exact"] for item in guessed)
        return {
            "reviewed": len(items),
            "guessed": len(guessed),
            "exact": exact,
            "wrong": len(guessed) - exact,
            "exact_accuracy": (
                round(exact / len(guessed), 6)
                if guessed
                else 0.0
            ),
            "mean_character_error": (
                round(
                    sum(item["distance"] for item in guessed)
                    / len(guessed),
                    4,
                )
                if guessed
                else 0.0
            ),
        }

    return {
        **summarize(usable),
        "by_model": [
            {"model_revision": name, **summarize(items)}
            for name, items in sorted(groups.items())
        ],
    }
