"""Prepare company-owned, operator-labelled crops for CCT training."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.dataset_split import grouped_train_validation_split
from app.ai.plate_rules import normalize_plate, plausible_plate
from app.ai.training_manifest import operator_dataset_fingerprint


ALLOWED_SOURCE_LICENSES = {
    "bcvision-company-owned",
    "operator-confirmed-company-owned",
    "operator-confirmed-rights-unverified",
    "cc0-1.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _canonical_sha256(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest().upper()


def _contained_file(root: Path, raw_path: str, *, context: str) -> Path:
    relative = Path(str(raw_path or "").strip())
    if (
        not str(relative)
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise ValueError(f"{context} must be a relative contained path")
    root = root.resolve()
    candidate = root / relative
    for index in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:index]).is_symlink():
            raise ValueError(f"{context} cannot traverse a symlink")
    resolved = candidate.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{context} escapes its source root") from exc
    if not resolved.is_file():
        raise ValueError(f"{context} is missing or is not a file")
    return resolved


def _validate_consistency(rows: list[dict]) -> list[dict]:
    if not rows:
        raise ValueError("Source manifest has no usable rows")
    plates_by_group = {}
    for row in rows:
        plates_by_group.setdefault(
            row["group_id"],
            set(),
        ).add(row["plate_text"])
    inconsistent_groups = [
        group
        for group, plates in plates_by_group.items()
        if len(plates) != 1
    ]
    if inconsistent_groups:
        raise ValueError(
            "A camera track/group cannot contain multiple plate identities"
        )
    splits_by_plate = {}
    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        if split:
            splits_by_plate.setdefault(
                row["plate_text"],
                set(),
            ).add(split)
    if any(len(splits) != 1 for splits in splits_by_plate.values()):
        raise ValueError(
            "One plate identity cannot cross declared dataset splits"
        )
    by_digest = {}
    by_path = set()
    for row in rows:
        canonical_path = str(row["image"].resolve())
        if canonical_path in by_path:
            raise ValueError("The same crop path appears more than once")
        by_path.add(canonical_path)
        previous = by_digest.get(row["sha256"])
        if previous:
            raise ValueError(
                "The same crop digest appears more than once"
            )
        by_digest[row["sha256"]] = row
    licenses = {row["source_license"] for row in rows}
    if len(licenses) != 1:
        raise ValueError(
            "One prepared dataset cannot mix source-license contracts"
        )
    if len({
        bool(row.get("distribution_allowed"))
        for row in rows
    }) != 1:
        raise ValueError(
            "One prepared dataset cannot mix distribution policies"
        )
    return list(rows)


def _load_operator_feedback_json(manifest: Path) -> list[dict]:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if (
        int(payload.get("schema", 0)) != 2
        or payload.get("training_source") != "operator-confirmed-only"
        or payload.get("golden_benchmark_data") is not False
    ):
        raise ValueError(
            "Operator feedback JSON must be a non-Golden schema-2 "
            "operator-confirmed training snapshot"
        )
    source_license = str(
        payload.get("source_license", "")
    ).strip().lower()
    ownership_attested = payload.get("ownership_attested")
    distribution_allowed = payload.get("distribution_allowed")
    license_evidence = str(
        payload.get("license_evidence", "")
    ).strip()
    if source_license == "operator-confirmed-rights-unverified":
        if (
            ownership_attested is not False
            or distribution_allowed is not False
            or license_evidence
        ):
            raise ValueError(
                "Unverified operator snapshot must remain non-distributable"
            )
    elif source_license == "operator-confirmed-company-owned":
        if (
            ownership_attested is not True
            or distribution_allowed is not True
            or not license_evidence
        ):
            raise ValueError(
                "Company-owned operator snapshot requires an explicit "
                "ownership attestation and license evidence"
            )
    else:
        raise ValueError(
            "Operator feedback rights/license attestation is missing"
        )
    samples = payload.get("samples")
    if not isinstance(samples, list):
        raise ValueError("Operator feedback JSON samples must be a list")
    expected_fingerprint = str(
        payload.get("dataset_fingerprint", "")
    ).strip().upper()
    actual_fingerprint = operator_dataset_fingerprint(samples)
    if (
        len(expected_fingerprint) != 64
        or actual_fingerprint != expected_fingerprint
    ):
        raise ValueError("Operator feedback dataset fingerprint mismatch")

    rows = []
    snapshot_root = (
        manifest.parent.parent
        if manifest.parent.name == "manifests"
        else manifest.parent
    ).resolve()
    for number, raw in enumerate(samples, 1):
        image = Path(str(raw.get("image_path", ""))).expanduser()
        if not image.is_absolute():
            image = (manifest.parent / image).resolve()
        else:
            image = image.resolve()
        try:
            image.relative_to(snapshot_root)
        except ValueError as exc:
            raise ValueError(
                f"Operator crop escapes snapshot root at JSON item {number}"
            ) from exc
        plate = normalize_plate(raw.get("plate", ""))
        group = str(raw.get("group_id") or plate).strip()
        split = str(raw.get("split", "")).strip().lower()
        expected_digest = str(raw.get("sha256", "")).strip().upper()
        if (
            not image.is_file()
            or not plausible_plate(plate)
            or not group
            or split not in {"train", "validation"}
            or len(expected_digest) != 64
        ):
            raise ValueError(
                f"Invalid operator feedback sample at JSON item {number}"
            )
        actual_digest = _sha256(image)
        if actual_digest != expected_digest:
            raise ValueError(
                f"Operator feedback sample hash mismatch at JSON item "
                f"{number}"
            )
        decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if decoded is None or min(decoded.shape[:2]) < 12:
            raise ValueError(
                f"Unreadable operator crop at JSON item {number}"
            )
        rows.append({
            "image": image,
            "plate_text": plate,
            "group_id": group,
            "source_license": source_license,
            "ownership_attested": ownership_attested,
            "distribution_allowed": distribution_allowed,
            "license_evidence": license_evidence,
            "sha256": actual_digest,
            "split": split,
        })
    return _validate_consistency(rows)


def _load_rows(manifest: Path) -> list[dict]:
    manifest = Path(manifest).resolve()
    if manifest.suffix.lower() == ".json":
        return _load_operator_feedback_json(manifest)

    rows = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "image_path",
            "plate_text",
            "group_id",
            "source_license",
            "usage",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Source CSV is missing one or more required columns"
            )
        for number, raw in enumerate(reader, 2):
            if None in raw or not any(
                str(value or "").strip() for value in raw.values()
            ):
                raise ValueError(f"Malformed or blank CSV row at line {number}")
            source = str(raw.get("source_license", "")).strip().lower()
            if source not in ALLOWED_SOURCE_LICENSES:
                raise ValueError(
                    f"Unapproved source license at CSV line {number}: "
                    f"{source or 'missing'}"
                )
            usage = str(raw.get("usage", "")).strip().lower()
            if usage != "train":
                raise ValueError(
                    "Golden/benchmark or other non-training usage is "
                    f"forbidden at line {number}"
                )
            image = _contained_file(
                manifest.parent,
                str(raw.get("image_path", "")),
                context=f"Source image at CSV line {number}",
            )
            plate = normalize_plate(raw.get("plate_text", ""))
            group = str(raw.get("group_id", "")).strip()
            if (
                not image.is_file()
                or not plausible_plate(plate)
                or not group
            ):
                raise ValueError(
                    f"Invalid image, plate or group at CSV line {number}"
                )
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None or min(decoded.shape[:2]) < 12:
                raise ValueError(f"Unreadable crop at CSV line {number}")
            rows.append({
                "image": image,
                "plate_text": plate,
                "group_id": group,
                "source_license": source,
                "ownership_attested": source in {
                    "bcvision-company-owned",
                    "operator-confirmed-company-owned",
                },
                "distribution_allowed": True,
                "license_evidence": (
                    "explicit-source-csv-license-declaration"
                ),
                "sha256": _sha256(image),
            })
    return _validate_consistency(rows)


def _write_split(root: Path, name: str, rows: list[dict]) -> dict:
    split = root / name
    images = split / "images"
    images.mkdir(parents=True)
    annotations = []
    integrity_rows = []
    for index, row in enumerate(rows):
        suffix = row["image"].suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            suffix = ".png"
        filename = f"{index:07d}-{row['sha256'][:12]}{suffix}"
        target = images / filename
        shutil.copy2(row["image"], target)
        annotations.append({
            "image_path": f"images/{filename}",
            "plate_text": row["plate_text"],
        })
        copied_digest = _sha256(target)
        if copied_digest != row["sha256"]:
            raise ValueError("Copied training crop hash mismatch")
        integrity_rows.append({
            "image_path": f"images/{filename}",
            "plate_text": row["plate_text"],
            "sha256": copied_digest,
        })
    annotation_path = split / "annotations.csv"
    with annotation_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "plate_text"],
        )
        writer.writeheader()
        writer.writerows(annotations)
    return {
        "samples": len(rows),
        "annotation_sha256": _sha256(annotation_path),
        "images_fingerprint": _canonical_sha256(integrity_rows),
    }


def prepare(
    source_manifest: Path,
    output: Path,
    validation_ratio: float,
    seed: int,
) -> dict:
    source_manifest = source_manifest.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    rows = _load_rows(source_manifest)
    declared_splits = {
        str(row.get("split", "")).strip().lower()
        for row in rows
        if str(row.get("split", "")).strip()
    }
    if declared_splits:
        if declared_splits != {"train", "validation"}:
            raise ValueError(
                "Declared operator splits must include train and validation"
            )
        train = [row for row in rows if row["split"] == "train"]
        validation = [
            row for row in rows if row["split"] == "validation"
        ]
    else:
        train, validation = grouped_train_validation_split(
            rows,
            # A normalized plate can appear in several camera tracks. Keep the
            # identity whole so the same vehicle cannot leak across both splits.
            group_key="plate_text",
            validation_ratio=validation_ratio,
            seed=seed,
        )
    if not train or not validation:
        raise ValueError(
            "Train and validation each require at least one usable crop"
        )
    output.mkdir(parents=True)
    split_integrity = {
        "train": _write_split(output, "train", train),
        "validation": _write_split(output, "val", validation),
    }
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    train_identities = {row["plate_text"] for row in train}
    validation_identities = {row["plate_text"] for row in validation}
    source_licenses = sorted({row["source_license"] for row in rows})
    distribution_allowed = all(
        row.get("distribution_allowed") is True
        for row in rows
    )
    ownership_attested = all(
        row.get("ownership_attested") is True
        for row in rows
    )
    license_evidence_values = sorted({
        str(row.get("license_evidence", "")).strip()
        for row in rows
        if str(row.get("license_evidence", "")).strip()
    })
    dataset_fingerprint = _canonical_sha256(split_integrity)
    manifest = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_license": source_licenses[0],
        "source_license_values": source_licenses,
        "ownership_attested": ownership_attested,
        "distribution_allowed": distribution_allowed,
        "license_evidence_values": license_evidence_values,
        "activation_policy": (
            "independent-golden-and-real-camera-pass"
            if distribution_allowed
            else "shadow-only-until-rights-attested"
        ),
        "golden_benchmark_data": False,
        "source_manifest": source_manifest.name,
        "source_manifest_sha256": _sha256(source_manifest),
        "train_samples": len(train),
        "validation_samples": len(validation),
        "train_groups": len(train_groups),
        "validation_groups": len(validation_groups),
        "group_overlap": len(train_groups & validation_groups),
        "train_plate_identities": len(train_identities),
        "validation_plate_identities": len(validation_identities),
        "plate_identity_overlap": len(
            train_identities & validation_identities
        ),
        "integrity": {
            "schema": 1,
            "algorithm": "sha256",
            "dataset_fingerprint": dataset_fingerprint,
            "splits": split_integrity,
        },
        "seed": int(seed),
    }
    (output / "dataset-license.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepare licensed BC Vision crops for CCT training",
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args(argv)
    result = prepare(
        source_manifest=args.source_manifest,
        output=args.output,
        validation_ratio=max(
            0.05,
            min(0.50, float(args.validation_ratio)),
        ),
        seed=int(args.seed),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
