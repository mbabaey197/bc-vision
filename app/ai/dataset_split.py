"""Deterministic group-aware dataset splitting for ANPR training."""
from __future__ import annotations

import hashlib
import random


def stable_split_for_group(group, validation_ratio=0.20) -> str:
    ratio = min(0.50, max(0.05, float(validation_ratio)))
    digest = hashlib.sha256(str(group).encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2 ** 64)
    return "validation" if value < ratio else "train"


def grouped_train_validation_split(
    samples,
    group_key="group_id",
    validation_ratio=0.20,
    seed=20260728,
) -> tuple[list, list]:
    """Split whole tracks/plates so related frames never cross the boundary."""

    groups = {}
    for sample in samples:
        group = sample.get(group_key)
        if group in {None, ""}:
            raise ValueError(f"Missing dataset group key: {group_key}")
        groups.setdefault(str(group), []).append(sample)
    if len(groups) < 2:
        raise ValueError("At least two independent groups are required")

    keys = sorted(groups)
    random.Random(int(seed)).shuffle(keys)
    validation_count = max(
        1,
        min(
            len(keys) - 1,
            int(round(len(keys) * float(validation_ratio))),
        ),
    )
    validation_keys = set(keys[:validation_count])
    train = []
    validation = []
    for key in sorted(groups):
        target = validation if key in validation_keys else train
        target.extend(groups[key])
    return train, validation
