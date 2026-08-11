"""Materialize a content-addressed, inference-only Engine V2 accuracy dataset.

The input is an operator-reviewed draft accuracy manifest.  Every enabled
sample is copied into a new content-addressed dataset directory and bound to
its exact byte size and SHA-256 digest.  The output is deterministic and is
never admitted to training.

This tool deliberately does not infer labels, categories, or event windows
from model output.  Those fields must already be present and verified in the
draft manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine_v2.benchmark import (  # noqa: E402
    ACCURACY_LABEL_SCOPES,
    ACCURACY_MANIFEST_SCHEMA,
    REQUIRED_ACCURACY_CATEGORIES,
    load_accuracy_manifest,
)
from app.engine_v2.validator import IranianPlateValidator  # noqa: E402


_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")
_MEDIA_TYPES = frozenset({"image", "video"})
_VALIDATOR = IranianPlateValidator()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _load_draft(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid draft manifest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("draft manifest root must be an object")
    if payload.get("schema") != ACCURACY_MANIFEST_SCHEMA:
        raise ValueError(f"draft schema must be {ACCURACY_MANIFEST_SCHEMA!r}")
    if payload.get("template") is True:
        raise ValueError("a template cannot be materialized as verified evidence")
    if payload.get("training_allowed") is not False:
        raise ValueError("accuracy datasets must explicitly set training_allowed=false")
    label_source = payload.get("label_source")
    if not isinstance(label_source, str) or not label_source.strip():
        raise ValueError("draft label_source must be an explicit non-empty string")
    payload["label_source"] = label_source.strip()
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("draft dataset_id is required")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError("draft samples must be a non-empty array")
    return payload


def _verified_plate(value: Any, *, context: str) -> str:
    validation = _VALIDATOR.validate(str(value or ""))
    if not validation.valid:
        raise ValueError(f"{context} is not a valid Iranian plate: {validation.reason}")
    return validation.normalized


def _validated_labels(sample: Mapping[str, Any], identifier: str) -> dict[str, Any]:
    has_plate = "expected_plate" in sample
    has_events = "expected_events" in sample
    if has_plate == has_events:
        raise ValueError(
            f"sample {identifier!r} must contain exactly one of expected_plate or expected_events"
        )
    if has_plate:
        value = sample.get("expected_plate")
        return {
            "expected_plate": (
                None
                if value is None
                else _verified_plate(value, context=f"sample {identifier!r} expected_plate")
            )
        }

    raw_events = sample.get("expected_events")
    if not isinstance(raw_events, list) or not raw_events:
        raise ValueError(f"sample {identifier!r} expected_events must be non-empty")
    events: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sample {identifier!r} event {index} must be an object")
        event: dict[str, Any] = {
            "plate": _verified_plate(
                raw.get("plate"),
                context=f"sample {identifier!r} event {index} plate",
            )
        }
        start_ms = raw.get("start_ms")
        end_ms = raw.get("end_ms")
        if start_ms is not None:
            if isinstance(start_ms, bool):
                raise ValueError(f"sample {identifier!r} event {index} has invalid start_ms")
            start_ms = float(start_ms)
            if not math.isfinite(start_ms) or start_ms < 0:
                raise ValueError(f"sample {identifier!r} event {index} has invalid start_ms")
            event["start_ms"] = start_ms
        if end_ms is not None:
            if isinstance(end_ms, bool):
                raise ValueError(f"sample {identifier!r} event {index} has invalid end_ms")
            end_ms = float(end_ms)
            if not math.isfinite(end_ms) or end_ms < 0:
                raise ValueError(f"sample {identifier!r} event {index} has invalid end_ms")
            event["end_ms"] = end_ms
        if start_ms is not None and end_ms is not None and end_ms < start_ms:
            raise ValueError(f"sample {identifier!r} event {index} ends before it starts")
        if "track_id" in raw:
            track_id = raw["track_id"]
            if isinstance(track_id, bool) or not isinstance(track_id, (str, int)):
                raise ValueError(
                    f"sample {identifier!r} event {index} track_id must be a string or integer"
                )
            event["track_id"] = track_id
        events.append(event)
    return {"expected_events": events}


def _source_path(draft_path: Path, raw: Any, identifier: str) -> Path:
    value = str(raw or "").strip()
    if not value or "://" in value:
        raise ValueError(f"sample {identifier!r} must reference one local media file")
    source = Path(value).expanduser()
    if not source.is_absolute():
        source = draft_path.parent / source

    # Reject a link at any component, not only at the leaf. Otherwise a path
    # such as linked-directory/source.mp4 could silently change provenance.
    absolute_source = source.absolute()
    for component in (absolute_source, *absolute_source.parents):
        is_junction = getattr(component, "is_junction", None)
        if component.is_symlink() or (callable(is_junction) and is_junction()):
            raise ValueError(
                f"sample {identifier!r} media path must not contain a symlink or junction"
            )
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def _coverage(samples: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = {category: 0 for category in REQUIRED_ACCURACY_CATEGORIES}
    readable_event_counts = {
        category: 0 for category in REQUIRED_ACCURACY_CATEGORIES
    }
    negatives = 0
    multi_event = 0
    known_positive_samples = 0
    for sample in samples:
        counts[str(sample["category"])] += 1
        category = str(sample["category"])
        if sample.get("label_scope", "exhaustive") == "known_positives":
            known_positive_samples += 1
        if sample.get("expected_plate") is None and "expected_plate" in sample:
            negatives += 1
        events = sample.get("expected_events")
        if "expected_plate" in sample and sample.get("expected_plate") is not None:
            readable_event_counts[category] += 1
        elif isinstance(events, list):
            readable_event_counts[category] += len(events)
        if category == "multiple_vehicles" and isinstance(events, list) and len(events) >= 2:
            multi_event += 1
    missing = [category for category, count in counts.items() if count == 0]
    unreadable = [
        category
        for category, event_count in readable_event_counts.items()
        if event_count == 0
    ]
    return {
        "sample_counts": counts,
        "readable_event_counts": readable_event_counts,
        "negative_samples": negatives,
        "multi_event_samples": multi_event,
        "known_positive_samples": known_positive_samples,
        "missing_categories": missing,
        "unreadable_categories": unreadable,
        "complete": (
            not missing
            and not unreadable
            and negatives > 0
            and multi_event > 0
            and known_positive_samples == 0
        ),
    }


def prepare_accuracy_dataset(
    draft_manifest: str | Path,
    output: str | Path,
    *,
    require_all_categories: bool = True,
    require_negative_sample: bool = True,
) -> dict[str, Any]:
    """Copy and bind verified media into a new immutable dataset directory."""

    draft_path = Path(draft_manifest).resolve()
    output_path = Path(output).resolve()
    if output_path.exists():
        raise FileExistsError(
            f"accuracy dataset output already exists; choose a new directory: {output_path}"
        )
    payload = _load_draft(draft_path)
    identifiers: set[str] = set()
    prepared_samples: list[dict[str, Any]] = []
    source_rows: list[tuple[Path, str, str]] = []

    for index, raw in enumerate(payload["samples"]):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sample {index} must be an object")
        if raw.get("enabled", True) is not True:
            continue
        identifier = str(raw.get("id", "")).strip()
        if not _SAFE_ID.fullmatch(identifier) or identifier in identifiers:
            raise ValueError(f"sample {index} has an unsafe or duplicate id")
        identifiers.add(identifier)
        category = str(raw.get("category", "")).strip()
        if category not in REQUIRED_ACCURACY_CATEGORIES:
            raise ValueError(f"sample {identifier!r} has unsupported category {category!r}")
        if raw.get("label_status") != "verified":
            raise ValueError(f"sample {identifier!r} must have label_status='verified'")
        if "training_allowed" in raw and raw.get("training_allowed") is not False:
            raise ValueError(
                f"sample {identifier!r} must set training_allowed=false when present"
            )
        if "adapter_input" in raw:
            raise ValueError(
                f"sample {identifier!r} adapter_input is not permitted in prepared "
                "accuracy evidence; use identical engine-level CLI configuration"
            )
        raw_label_source = raw.get("label_source", payload["label_source"])
        if not isinstance(raw_label_source, str) or not raw_label_source.strip():
            raise ValueError(
                f"sample {identifier!r} label_source must be a non-empty string"
            )
        label_scope = str(raw.get("label_scope", "exhaustive")).strip().lower()
        if label_scope not in ACCURACY_LABEL_SCOPES:
            raise ValueError(
                f"sample {identifier!r} label_scope must be one of: "
                + ", ".join(ACCURACY_LABEL_SCOPES)
            )
        raw_input = raw.get("input")
        if not isinstance(raw_input, Mapping):
            raise ValueError(f"sample {identifier!r} input must be an object")
        source = _source_path(draft_path, raw_input.get("path"), identifier)
        media_type = str(raw_input.get("media_type", "")).strip().lower()
        if media_type not in _MEDIA_TYPES:
            raise ValueError(f"sample {identifier!r} media_type must be image or video")
        start_ms = raw_input.get("start_ms")
        end_ms = raw_input.get("end_ms")
        normalized_input: dict[str, Any] = {"media_type": media_type}
        if start_ms is not None:
            if isinstance(start_ms, bool):
                raise ValueError(f"sample {identifier!r} input.start_ms is invalid")
            start_ms = float(start_ms)
            if not math.isfinite(start_ms) or start_ms < 0:
                raise ValueError(f"sample {identifier!r} input.start_ms is invalid")
            normalized_input["start_ms"] = start_ms
        if end_ms is not None:
            if isinstance(end_ms, bool):
                raise ValueError(f"sample {identifier!r} input.end_ms is invalid")
            end_ms = float(end_ms)
            if not math.isfinite(end_ms) or end_ms < 0:
                raise ValueError(f"sample {identifier!r} input.end_ms is invalid")
            normalized_input["end_ms"] = end_ms
        if start_ms is not None and end_ms is not None and end_ms <= start_ms:
            raise ValueError(f"sample {identifier!r} input clip must have end_ms > start_ms")

        digest = _sha256(source)
        size_bytes = source.stat().st_size
        reviewed_digest = str(raw_input.get("sha256", "")).strip().lower()
        if reviewed_digest and reviewed_digest != digest:
            raise ValueError(
                f"sample {identifier!r} reviewed SHA-256 does not match the source file"
            )
        reviewed_size = raw_input.get("size_bytes")
        if reviewed_size is not None:
            if isinstance(reviewed_size, bool) or not isinstance(reviewed_size, int):
                raise ValueError(
                    f"sample {identifier!r} reviewed byte size must be an integer"
                )
            if reviewed_size != size_bytes:
                raise ValueError(
                    f"sample {identifier!r} reviewed byte size does not match the source file"
                )
        suffix = source.suffix.lower() or ".bin"
        relative = Path("media") / f"{digest}{suffix}"
        normalized_input.update(
            {
                "path": relative.as_posix(),
                "sha256": digest,
                "size_bytes": size_bytes,
            }
        )
        labels = _validated_labels(raw, identifier)
        if (
            label_scope == "known_positives"
            and "expected_plate" in labels
            and labels["expected_plate"] is None
        ):
            raise ValueError(
                f"sample {identifier!r} known_positives scope requires at least one label"
            )
        prepared: dict[str, Any] = {
            "id": identifier,
            "category": category,
            "input": normalized_input,
            "label_status": "verified",
            "label_source": raw_label_source.strip(),
            "label_scope": label_scope,
            "training_allowed": False,
            **labels,
        }
        if str(raw.get("notes", "")).strip():
            prepared["notes"] = str(raw["notes"]).strip()
        prepared_samples.append(prepared)
        source_rows.append((source, relative.as_posix(), digest))

    if not prepared_samples:
        raise ValueError("draft contains no enabled samples")
    coverage = _coverage(prepared_samples)
    if require_all_categories and coverage["missing_categories"]:
        raise ValueError(
            "dataset is missing required categories: "
            + ", ".join(coverage["missing_categories"])
        )
    if require_all_categories and coverage["unreadable_categories"]:
        raise ValueError(
            "dataset has no readable event labels for categories: "
            + ", ".join(coverage["unreadable_categories"])
        )
    if require_all_categories and coverage["known_positive_samples"]:
        raise ValueError(
            "complete accuracy evidence requires exhaustive labels for every sample"
        )
    if require_negative_sample and coverage["negative_samples"] < 1:
        raise ValueError("dataset requires at least one verified negative sample")
    if require_all_categories and coverage["multi_event_samples"] < 1:
        raise ValueError("dataset requires one sample with at least two expected events")

    manifest_payload = {
        "schema": ACCURACY_MANIFEST_SCHEMA,
        "dataset_id": str(payload["dataset_id"]).strip(),
        "description": str(payload.get("description", "")).strip(),
        "label_source": payload["label_source"],
        "training_allowed": False,
        "coverage": coverage,
        "samples": prepared_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_path.name}.tmp-", dir=output_path.parent)
    )
    try:
        for source, relative, expected_digest in source_rows:
            target = temporary / relative
            if target.exists():
                if _sha256(target) != expected_digest:
                    raise ValueError(f"content-address collision for {relative}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix(target.suffix + ".partial")
            shutil.copyfile(source, partial)
            if _sha256(partial) != expected_digest:
                raise ValueError(f"copied media hash mismatch: {source}")
            os.replace(partial, target)

        manifest_path = temporary / "manifest.json"
        _write_json(manifest_path, manifest_payload)
        # Use the benchmark loader itself as the canonical fingerprint source,
        # so the preparer and the V1/V2 runner cannot silently disagree about
        # which logical fields identify a dataset.
        loaded = load_accuracy_manifest(
            manifest_path,
            require_all_categories=require_all_categories,
            require_negative_sample=require_negative_sample,
            strict_evidence=True,
        )
        dataset_fingerprint = loaded.dataset_fingerprint
        manifest_payload["dataset_fingerprint_sha256"] = dataset_fingerprint
        _write_json(manifest_path, manifest_payload)
        reloaded = load_accuracy_manifest(
            manifest_path,
            require_all_categories=require_all_categories,
            require_negative_sample=require_negative_sample,
            strict_evidence=True,
        )
        if reloaded.dataset_fingerprint != dataset_fingerprint:
            raise RuntimeError("prepared dataset fingerprint is not stable")
        manifest_sha256 = _sha256(manifest_path)
        inventory = [
            {
                "path": sample["input"]["path"],
                "sha256": sample["input"]["sha256"],
                "size_bytes": sample["input"]["size_bytes"],
            }
            for sample in prepared_samples
        ]
        integrity = {
            "schema": "bcvision.anpr.accuracy-dataset-integrity/v1",
            "dataset_id": manifest_payload["dataset_id"],
            "dataset_fingerprint_sha256": dataset_fingerprint,
            "manifest_path": "manifest.json",
            "manifest_sha256": manifest_sha256,
            "media": inventory,
            "training_allowed": False,
            "coverage_complete": coverage["complete"],
        }
        _write_json(temporary / "integrity.json", integrity)
        if output_path.exists():
            raise FileExistsError(
                f"accuracy dataset output appeared during preparation: {output_path}"
            )
        os.replace(temporary, output_path)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return {
        **integrity,
        "dataset_root": str(output_path),
        "manifest": str(output_path / "manifest.json"),
        "integrity_file": str(output_path / "integrity.json"),
        "sample_count": len(prepared_samples),
        "coverage": coverage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy an operator-verified ANPR accuracy manifest into a new "
            "content-addressed, SHA-256-bound, tamper-evident, training-forbidden dataset."
        )
    )
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help="prepare a non-promotional subset even when the eight categories are incomplete",
    )
    parser.add_argument(
        "--allow-no-negative",
        action="store_true",
        help="prepare a non-promotional subset without a verified negative sample",
    )
    args = parser.parse_args(argv)
    result = prepare_accuracy_dataset(
        args.draft_manifest,
        args.output,
        require_all_categories=not args.allow_partial_coverage,
        require_negative_sample=not args.allow_no_negative,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["coverage_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
