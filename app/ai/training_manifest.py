"""Portable integrity helpers for operator-confirmed ANPR snapshots."""
from __future__ import annotations

import hashlib
import json


def operator_dataset_fingerprint(samples: list[dict]) -> str:
    """Hash label/split metadata without binding a snapshot to one host."""

    normalized = [
        {
            "feedback_id": int(row.get("feedback_id", 0)),
            "sha256": str(row.get("sha256", "")).upper(),
            "plate": str(row.get("plate", "")),
            "group_id": str(row.get("group_id", "")),
            "split": str(row.get("split", "")).lower(),
        }
        for row in samples
    ]
    return hashlib.sha256(
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()
