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


ALLOWED_SOURCE_LICENSES = {
    "bcvision-company-owned",
    "operator-confirmed-company-owned",
    "cc0-1.0",
    "cc-by-4.0",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_rows(manifest: Path) -> list[dict]:
    rows = []
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for number, raw in enumerate(csv.DictReader(handle), 2):
            source = str(raw.get("source_license", "")).strip().lower()
            if source not in ALLOWED_SOURCE_LICENSES:
                raise ValueError(
                    f"Unapproved source license at CSV line {number}: "
                    f"{source or 'missing'}"
                )
            usage = str(raw.get("usage", "train")).strip().lower()
            if usage in {"golden", "benchmark", "test"}:
                raise ValueError(
                    f"Golden/benchmark crop cannot enter training at line {number}"
                )
            image = (
                manifest.parent / str(raw.get("image_path", ""))
            ).resolve()
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
                "sha256": _sha256(image),
            })
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
    by_digest = {}
    for row in rows:
        previous = by_digest.get(row["sha256"])
        if previous and (
            previous["plate_text"] != row["plate_text"]
            or previous["group_id"] != row["group_id"]
        ):
            raise ValueError(
                "The same crop has conflicting label or group metadata"
            )
        by_digest[row["sha256"]] = row
    return list(by_digest.values())


def _write_split(root: Path, name: str, rows: list[dict]) -> None:
    split = root / name
    images = split / "images"
    images.mkdir(parents=True)
    annotations = []
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
    with (split / "annotations.csv").open(
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
    train, validation = grouped_train_validation_split(
        rows,
        # A normalized plate can appear in several camera tracks. Keep the
        # identity whole so the same vehicle cannot leak across both splits.
        group_key="plate_text",
        validation_ratio=validation_ratio,
        seed=seed,
    )
    output.mkdir(parents=True)
    _write_split(output, "train", train)
    _write_split(output, "val", validation)
    train_groups = {row["group_id"] for row in train}
    validation_groups = {row["group_id"] for row in validation}
    train_identities = {row["plate_text"] for row in train}
    validation_identities = {row["plate_text"] for row in validation}
    manifest = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_license": "operator-confirmed-company-owned",
        "source_license_values": sorted({
            row["source_license"] for row in rows
        }),
        "golden_benchmark_data": False,
        "source_manifest": str(source_manifest),
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
