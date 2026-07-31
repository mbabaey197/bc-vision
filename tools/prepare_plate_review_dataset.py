"""Prepare an integrity-bound CCT dataset from an operator review export."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.dataset_split import grouped_train_validation_split
from app.ai.plate_rules import normalize_plate, plausible_plate


ALLOWED_STATUSES = frozenset({
    "confirmed",
    "excluded",
    "pending",
    "unreadable",
})
REQUIRED_COLUMNS = frozenset({
    "file_name",
    "plate",
    "sha256",
    "status",
})
REVIEW_EXPORT_KIND = "bcvision-operator-plate-review"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


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


def _review_metadata(review_page: Path) -> dict:
    text = review_page.read_text(encoding="utf-8")
    match = re.search(
        r"const META=(\{.*?\});\s*const SAMPLES=",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("Review page metadata is missing")
    try:
        metadata = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("Review page metadata is invalid") from exc
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema")) is not int
        or metadata.get("schema") != 1
        or metadata.get("training_source") != "operator-confirmed-only"
        or metadata.get("model_suggestions_are_labels") is not False
        or not str(metadata.get("source_id", "")).strip()
    ):
        raise ValueError("Review page contract is invalid")
    source_license = str(
        metadata.get("source_license", "")
    ).strip().lower()
    evidence = str(metadata.get("license_evidence", "")).strip()
    if source_license == "operator-confirmed-company-owned":
        if (
            metadata.get("ownership_attested") is not True
            or metadata.get("distribution_allowed") is not True
            or not evidence
        ):
            raise ValueError(
                "Company-owned review data requires ownership evidence"
            )
    elif source_license == "operator-confirmed-rights-unverified":
        if (
            metadata.get("ownership_attested") is not False
            or metadata.get("distribution_allowed") is not False
            or evidence
        ):
            raise ValueError(
                "Rights-unverified review data must be non-distributable"
            )
    else:
        raise ValueError("Review page source license is not approved")
    return metadata


def _review_rows(review_csv: Path, expected_count: int) -> list[dict]:
    with review_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or not REQUIRED_COLUMNS.issubset(reader.fieldnames)
            or len(reader.fieldnames) != len(set(reader.fieldnames))
        ):
            raise ValueError("Review CSV columns are invalid")
        rows = list(reader)
    if len(rows) != int(expected_count):
        raise ValueError("Review CSV row count does not match review page")
    names = set()
    digests = set()
    for number, row in enumerate(rows, 2):
        if None in row or not any(
            str(value or "").strip() for value in row.values()
        ):
            raise ValueError(f"Malformed review CSV row at line {number}")
        name = str(row.get("file_name", "")).strip()
        if (
            not name
            or Path(name).name != name
            or name in names
        ):
            raise ValueError(
                f"Unsafe or duplicate image name at line {number}"
            )
        digest = str(row.get("sha256", "")).strip().upper()
        if (
            re.fullmatch(r"[0-9A-F]{64}", digest) is None
            or digest in digests
        ):
            raise ValueError(
                f"Invalid or duplicate image digest at line {number}"
            )
        status = str(row.get("status", "")).strip().lower()
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Invalid review status at line {number}")
        plate = normalize_plate(row.get("plate", ""))
        if status == "confirmed":
            if not plausible_plate(plate):
                raise ValueError(
                    f"Confirmed plate is invalid at line {number}"
                )
        elif plate:
            raise ValueError(
                f"Non-confirmed row carries a plate at line {number}"
            )
        row["file_name"] = name
        row["sha256"] = digest
        row["status"] = status
        row["plate"] = plate
        names.add(name)
        digests.add(digest)
    return rows


def _archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    members = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        relative = PurePosixPath(info.filename)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.name
            or (info.external_attr >> 16) & 0o170000 == 0o120000
        ):
            raise ValueError("Source archive contains an unsafe entry")
        if relative.name in members:
            raise ValueError(
                "Source archive contains duplicate image base names"
            )
        members[relative.name] = info
    return members


def _verified_confirmed_rows(
    rows: list[dict],
    source_archive: Path,
) -> list[dict]:
    confirmed = []
    with zipfile.ZipFile(source_archive) as archive:
        members = _archive_members(archive)
        if set(members) != {
            str(row["file_name"])
            for row in rows
        }:
            raise ValueError(
                "Source archive inventory does not match review CSV"
            )
        for row in rows:
            data = archive.read(members[row["file_name"]])
            if _sha256_bytes(data) != row["sha256"]:
                raise ValueError(
                    f"Source image hash mismatch: {row['file_name']}"
                )
            try:
                with Image.open(BytesIO(data)) as image:
                    width, height = image.size
                    image.verify()
            except Exception as exc:
                raise ValueError(
                    f"Source image is unreadable: {row['file_name']}"
                ) from exc
            if row["status"] == "confirmed":
                if min(width, height) < 12:
                    raise ValueError(
                        f"Confirmed image is too small: {row['file_name']}"
                    )
                confirmed.append({
                    **row,
                    "data": data,
                    # Duplicate observations of one real plate identity must
                    # never cross Train and Validation.
                    "group_id": row["plate"],
                })
    if len(confirmed) < 2:
        raise ValueError("At least two confirmed plate identities are required")
    if len({row["plate"] for row in confirmed}) < 2:
        raise ValueError("At least two confirmed plate identities are required")
    return confirmed


def _write_split(root: Path, name: str, rows: list[dict]) -> dict:
    split = root / name
    images = split / "images"
    images.mkdir(parents=True)
    annotations = []
    integrity_rows = []
    for index, row in enumerate(rows):
        suffix = Path(row["file_name"]).suffix.lower()
        if suffix not in {".bmp", ".jpeg", ".jpg", ".png", ".webp"}:
            suffix = ".jpg"
        filename = f"{index:07d}-{row['sha256'][:12]}{suffix}"
        relative = f"images/{filename}"
        target = images / filename
        target.write_bytes(row["data"])
        if _sha256(target) != row["sha256"]:
            raise ValueError("Copied training crop hash mismatch")
        annotations.append({
            "image_path": relative,
            "plate_text": row["plate"],
        })
        integrity_rows.append({
            "image_path": relative,
            "plate_text": row["plate"],
            "sha256": row["sha256"],
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


def prepare_review_dataset(
    review_csv: Path,
    source_archive: Path,
    review_page: Path,
    output: Path,
    *,
    validation_ratio: float = 0.20,
    seed: int = 20260730,
) -> dict:
    review_csv = review_csv.resolve()
    source_archive = source_archive.resolve()
    review_page = review_page.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    metadata = _review_metadata(review_page)
    if _sha256(source_archive) != str(
        metadata.get("source_archive_sha256", "")
    ).strip().upper():
        raise ValueError("Source archive hash does not match review page")
    rows = _review_rows(
        review_csv,
        expected_count=int(metadata.get("image_count", -1)),
    )
    confirmed = _verified_confirmed_rows(rows, source_archive)
    train, validation = grouped_train_validation_split(
        confirmed,
        group_key="plate",
        validation_ratio=max(0.05, min(0.50, float(validation_ratio))),
        seed=int(seed),
    )
    if not train or not validation:
        raise ValueError("Train and validation each require usable crops")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    ))
    try:
        split_integrity = {
            "train": _write_split(temporary, "train", train),
            "validation": _write_split(temporary, "val", validation),
        }
        train_groups = {row["group_id"] for row in train}
        validation_groups = {row["group_id"] for row in validation}
        train_identities = {row["plate"] for row in train}
        validation_identities = {row["plate"] for row in validation}
        status_counts = Counter(row["status"] for row in rows)
        dataset_fingerprint = _canonical_sha256(split_integrity)
        source_license = str(metadata["source_license"]).strip().lower()
        distributable = metadata["distribution_allowed"] is True
        manifest = {
            "schema": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_license": source_license,
            "source_license_values": [source_license],
            "ownership_attested": metadata["ownership_attested"] is True,
            "distribution_allowed": distributable,
            "license_evidence_values": [
                str(metadata["license_evidence"]).strip(),
            ] if str(metadata.get("license_evidence", "")).strip() else [],
            "activation_policy": (
                "independent-golden-and-real-camera-pass"
                if distributable
                else "shadow-only-until-rights-attested"
            ),
            "golden_benchmark_data": False,
            "source_manifest": review_csv.name,
            "source_manifest_sha256": _sha256(review_csv),
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
            "review_export": {
                "kind": REVIEW_EXPORT_KIND,
                "source_id": str(metadata["source_id"]),
                "source_archive_sha256": _sha256(source_archive),
                "review_page_sha256": _sha256(review_page),
                "review_csv_sha256": _sha256(review_csv),
                "image_count": len(rows),
                "confirmed_samples": status_counts["confirmed"],
                "confirmed_plate_identities": len({
                    row["plate"] for row in confirmed
                }),
                "pending_samples": status_counts["pending"],
                "unreadable_samples": status_counts["unreadable"],
                "excluded_samples": status_counts["excluded"],
                "model_suggestions_used_as_labels": False,
            },
            "seed": int(seed),
        }
        (temporary / "dataset-license.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a company/operator-confirmed CCT dataset from a "
            "BC Vision plate-review CSV"
        ),
    )
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--review-page", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260730)
    args = parser.parse_args(argv)
    result = prepare_review_dataset(
        review_csv=args.review_csv,
        source_archive=args.source_archive,
        review_page=args.review_page,
        output=args.output,
        validation_ratio=args.validation_ratio,
        seed=args.seed,
    )
    print(json.dumps({
        "train_samples": result["train_samples"],
        "validation_samples": result["validation_samples"],
        "train_plate_identities": result["train_plate_identities"],
        "validation_plate_identities": (
            result["validation_plate_identities"]
        ),
        "plate_identity_overlap": result["plate_identity_overlap"],
        "dataset_fingerprint": result["integrity"][
            "dataset_fingerprint"
        ],
        "distribution_allowed": result["distribution_allowed"],
        "activation_policy": result["activation_policy"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
