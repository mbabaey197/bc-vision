"""Operator-labelled Golden Dataset contract for ANPR promotion.

Golden media is persistent customer data and is never committed or admitted
to training.  This module validates coverage and provenance before a model
promotion comparison is allowed to claim accuracy.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .plate_rules import normalize_plate, plausible_plate


REQUIRED_GOLDEN_SLICES = (
    "day",
    "night",
    "fast",
    "angle",
    "blur",
    "glare",
    "unreadable",
    "multi-vehicle",
)
MIN_GOLDEN_SAMPLES = 40
MIN_GOLDEN_UNIQUE_PLATES = 20
MIN_SAMPLES_PER_SLICE = 3


def golden_root() -> Path:
    from app.config import DATA_DIR

    return Path(DATA_DIR) / "anpr-golden"


def golden_manifest_path() -> Path:
    return golden_root() / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().lower()


def validate_golden_manifest(
    payload: dict,
    manifest_dir: Path,
    verify_media=True,
) -> dict:
    errors = []
    samples = payload.get("samples", [])
    if int(payload.get("schema", 0)) != 2:
        errors.append("schema-must-be-2")
    if payload.get("training_allowed") is not False:
        errors.append("training-must-be-forbidden")
    if not isinstance(samples, list):
        samples = []
        errors.append("samples-must-be-a-list")

    seen_ids = set()
    seen_media = set()
    normalized_rows = []
    slice_counts = {label: 0 for label in REQUIRED_GOLDEN_SLICES}
    unique_plates = set()
    for index, raw in enumerate(samples):
        sample_id = str(raw.get("id", "")).strip()
        if not sample_id or sample_id in seen_ids:
            errors.append(f"invalid-or-duplicate-id:{index}")
            continue
        seen_ids.add(sample_id)

        expected = normalize_plate(raw.get("expected_plate", ""))
        readable = bool(raw.get("readable", bool(expected)))
        if readable != plausible_plate(expected):
            errors.append(f"invalid-readable-label:{sample_id}")
            continue
        if readable:
            unique_plates.add(expected)

        labels = sorted({
            str(value).strip().lower()
            for value in raw.get("slices", [])
            if str(value).strip()
        })
        unknown = [
            label
            for label in labels
            if label not in REQUIRED_GOLDEN_SLICES
        ]
        if unknown or not labels:
            errors.append(f"invalid-slices:{sample_id}")
            continue
        if not readable and "unreadable" not in labels:
            errors.append(f"unreadable-slice-required:{sample_id}")
            continue

        relative = str(
            raw.get("frame_path")
            or raw.get("crop_path")
            or ""
        ).strip()
        digest = str(raw.get("sha256", "")).strip().lower()
        if (
            not relative
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            errors.append(f"invalid-media-contract:{sample_id}")
            continue
        media = (Path(manifest_dir) / relative).resolve()
        try:
            media.relative_to(Path(manifest_dir).resolve())
        except ValueError:
            errors.append(f"media-outside-golden-root:{sample_id}")
            continue
        media_key = (str(media), int(raw.get("frame_index", -1)))
        if media_key in seen_media:
            errors.append(f"duplicate-media-frame:{sample_id}")
            continue
        seen_media.add(media_key)
        if verify_media and (
            not media.is_file()
            or _sha256(media) != digest
        ):
            errors.append(f"media-hash-mismatch:{sample_id}")
            continue

        for label in labels:
            slice_counts[label] += 1
        normalized_rows.append({
            **raw,
            "id": sample_id,
            "expected_plate": expected,
            "readable": readable,
            "slices": labels,
            "media_path": str(media),
            "sha256": digest,
        })

    if len(normalized_rows) < MIN_GOLDEN_SAMPLES:
        errors.append("insufficient-total-samples")
    if len(unique_plates) < MIN_GOLDEN_UNIQUE_PLATES:
        errors.append("insufficient-unique-plates")
    for label, count in slice_counts.items():
        if count < MIN_SAMPLES_PER_SLICE:
            errors.append(f"insufficient-slice:{label}")

    return {
        "ready": not errors,
        "errors": sorted(set(errors)),
        "samples": len(normalized_rows),
        "unique_plates": len(unique_plates),
        "slice_counts": slice_counts,
        "required_samples": MIN_GOLDEN_SAMPLES,
        "required_unique_plates": MIN_GOLDEN_UNIQUE_PLATES,
        "required_per_slice": MIN_SAMPLES_PER_SLICE,
        "rows": normalized_rows,
    }


def golden_status(path: Path | None = None, verify_media=True) -> dict:
    path = Path(path or golden_manifest_path())
    if not path.is_file():
        return {
            "ready": False,
            "errors": ["manifest-missing"],
            "samples": 0,
            "unique_plates": 0,
            "slice_counts": {
                label: 0 for label in REQUIRED_GOLDEN_SLICES
            },
            "required_samples": MIN_GOLDEN_SAMPLES,
            "required_unique_plates": MIN_GOLDEN_UNIQUE_PLATES,
            "required_per_slice": MIN_SAMPLES_PER_SLICE,
            "rows": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ready": False,
            "errors": [
                f"manifest-invalid:{type(exc).__name__}",
            ],
            "samples": 0,
            "unique_plates": 0,
            "slice_counts": {
                label: 0 for label in REQUIRED_GOLDEN_SLICES
            },
            "required_samples": MIN_GOLDEN_SAMPLES,
            "required_unique_plates": MIN_GOLDEN_UNIQUE_PLATES,
            "required_per_slice": MIN_SAMPLES_PER_SLICE,
            "rows": [],
        }
    return validate_golden_manifest(
        payload,
        path.parent,
        verify_media=verify_media,
    )
