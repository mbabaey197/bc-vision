"""Train, export and verify a BC Vision FastPlateOCR CCT candidate.

The command is deliberately offline with respect to training data.  It accepts
only a dataset carrying an explicit commercially compatible provenance
manifest, exports fixed-batch uint8 NHWC ONNX, and measures exact held-out
accuracy before producing candidate metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.onnx_cct import (
    CCT_FUSION_GEOMETRIC_MEAN,
    CCT_FUSION_IDENTITY,
    CCT_PREPROCESS_DUAL_VIEW,
    CCT_PREPROCESS_LEGACY,
    infer_cct_session,
)
from app.ai.evaluation import character_distance
from app.ai.plate_rules import normalize_plate, plausible_plate


ALLOWED_DATA_LICENSES = {
    "synthetic-bcvision-company-owned",
    "bcvision-company-owned",
    "operator-confirmed-company-owned",
    "operator-confirmed-rights-unverified",
    "cc0-1.0",
    "gpl-3.0-ir-lpr-research-only",
}
ALLOWED_FONT_LICENSES = {
    "apache-2.0",
    "bcvision-company-owned",
    "bsd-3-clause",
    "cc0-1.0",
    "dejavu-font-license",
    "ofl-1.1",
}
EXCLUDED_PRETRAINED_LAYERS = {
    "plate",
    "region",
    "region_pre_pool_transformer_block_1",
    "region_seq_pool",
}
CHECKPOINT_DISTRIBUTABLE_LICENSES = {
    "apache-2.0",
    "bcvision-company-owned",
    "cc0-1.0",
    "mit",
    "operator-confirmed-company-owned",
    "synthetic-bcvision-company-owned",
}
CHECKPOINT_RESEARCH_LICENSES = {
    "gpl-3.0-ir-lpr-research-only",
}
_SYNTHETIC_IMAGE_NAME = re.compile(
    r"^[0-9]{6}-[0-9]{2}-[a-z_]+\.jpg$"
)


def _training_plate_config(root: Path, preprocess_profile: str) -> Path:
    name = (
        "iran_plate_letterbox_config.yaml"
        if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
        else "iran_plate_config.yaml"
    )
    return root / "training" / "cct" / name


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


def _require_nonnegative_int(payload: dict, key: str) -> int:
    value = payload.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"Dataset manifest count is invalid: {key}")
    return value


def _require_count_map(payload: dict, key: str) -> dict[str, int]:
    value = payload.get(key)
    if (
        not isinstance(value, dict)
        or any(not isinstance(name, str) or not name for name in value)
        or any(type(count) is not int or count < 0 for count in value.values())
    ):
        raise ValueError(f"Dataset manifest count map is invalid: {key}")
    return value


def _contained_image(
    images_root: Path,
    raw_path: str,
    *,
    split_name: str,
    line_number: int,
) -> tuple[Path, str]:
    text = str(raw_path or "").strip()
    relative = Path(text)
    if (
        not text
        or relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
        or relative.parts[0] != "images"
    ):
        raise ValueError(
            f"{split_name} image path must be relative at line {line_number}"
        )
    split_root = images_root.parent.resolve()
    candidate = split_root / relative
    for index in range(1, len(relative.parts) + 1):
        if split_root.joinpath(*relative.parts[:index]).is_symlink():
            raise ValueError(
                f"{split_name} image path traverses a symlink at line "
                f"{line_number}"
            )
    image = candidate.resolve()
    images_root = images_root.resolve()
    try:
        image.relative_to(images_root)
    except ValueError as exc:
        raise ValueError(
            f"{split_name} image path escapes images root at line "
            f"{line_number}"
        ) from exc
    if not image.is_file() or image.is_symlink():
        raise ValueError(
            f"{split_name} image is missing or unsafe at line {line_number}"
        )
    return image, image.relative_to(split_root).as_posix()


def _load_synthetic_metadata(
    split_dir: Path,
    annotation_rows: list[dict],
) -> None:
    metadata_path = split_dir / "samples.jsonl"
    if not metadata_path.is_file() or metadata_path.is_symlink():
        raise ValueError("Synthetic split samples.jsonl is required")
    metadata_rows = []
    with metadata_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(
                    f"Synthetic metadata has a blank row at line {number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Synthetic metadata JSON is invalid at line {number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Synthetic metadata row is invalid at line {number}"
                )
            metadata_rows.append(row)
    if len(metadata_rows) != len(annotation_rows):
        raise ValueError(
            "Synthetic metadata count does not match annotations"
        )
    forbidden = {
        "capture_path",
        "frame_path",
        "golden",
        "is_golden",
        "original_image_path",
        "real_image_path",
        "source_image",
        "source_path",
        "video_path",
    }
    required = {
        "condition_profile",
        "difficulty",
        "image_path",
        "jpeg_quality",
        "plate_style",
        "plate_text",
        "sample_seed",
    }
    for index, (metadata, annotation) in enumerate(
        zip(metadata_rows, annotation_rows, strict=True),
        1,
    ):
        if not required.issubset(metadata):
            raise ValueError(
                f"Synthetic metadata is incomplete at row {index}"
            )
        if any(
            key in metadata
            for key in forbidden
        ):
            raise ValueError(
                f"Synthetic metadata references real/Golden media at row "
                f"{index}"
            )
        jpeg_quality = metadata.get("jpeg_quality")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError(
                f"Synthetic metadata JPEG quality is invalid at row {index}"
            )
        for key in (
            "condition_profile",
            "difficulty",
            "image_path",
            "plate_style",
            "plate_text",
            "quality_score",
            "sample_seed",
            "simulated_source_width",
        ):
            if str(metadata.get(key, "")) != str(annotation.get(key, "")):
                raise ValueError(
                    f"Synthetic metadata does not match annotations at row "
                    f"{index}"
                )


def _preflight_split(
    dataset: Path,
    split_name: str,
    directory_name: str,
    *,
    synthetic_only: bool,
    manifest: dict,
) -> dict:
    split_dir = (dataset / directory_name).resolve()
    if (dataset / directory_name).is_symlink():
        raise ValueError(f"Dataset split cannot be a symlink: {split_name}")
    try:
        split_dir.relative_to(dataset)
    except ValueError as exc:
        raise ValueError(f"Dataset split escapes root: {split_name}") from exc
    annotation_path = split_dir / "annotations.csv"
    images_root = split_dir / "images"
    if synthetic_only:
        expected_entries = {"annotations.csv", "images", "samples.jsonl"}
        if {entry.name for entry in split_dir.iterdir()} != expected_entries:
            raise ValueError(
                f"{split_name} synthetic split contains unexpected artifacts"
            )
    if (
        not annotation_path.is_file()
        or annotation_path.is_symlink()
        or not images_root.is_dir()
        or images_root.is_symlink()
    ):
        raise ValueError(
            f"{split_name} annotations and images directory are required"
        )
    rows = []
    referenced_paths = set()
    digests = set()
    identities = set()
    with annotation_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "plate_text"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{split_name} annotations are missing required columns"
            )
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(
                f"{split_name} annotations contain duplicate columns"
            )
        if synthetic_only:
            synthetic_fields = {
                "condition_profile",
                "difficulty",
                "plate_style",
                "quality_score",
                "sample_seed",
                "simulated_source_width",
            }
            if not synthetic_fields.issubset(reader.fieldnames):
                raise ValueError(
                    f"{split_name} synthetic annotations are incomplete"
                )
        for line_number, raw in enumerate(reader, 2):
            if None in raw or not any(
                str(value or "").strip() for value in raw.values()
            ):
                raise ValueError(
                    f"{split_name} annotations contain a malformed row at "
                    f"line {line_number}"
                )
            image, relative = _contained_image(
                images_root,
                raw.get("image_path", ""),
                split_name=split_name,
                line_number=line_number,
            )
            if relative in referenced_paths:
                raise ValueError(
                    f"{split_name} image path is duplicated at line "
                    f"{line_number}"
                )
            plate = normalize_plate(raw.get("plate_text", ""))
            if not plausible_plate(plate):
                raise ValueError(
                    f"{split_name} plate label is implausible at line "
                    f"{line_number}"
                )
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None or decoded.size == 0 or min(
                decoded.shape[:2]
            ) < 12:
                raise ValueError(
                    f"{split_name} image is unreadable at line {line_number}"
                )
            digest = _sha256(image)
            if digest in digests:
                raise ValueError(
                    f"{split_name} contains a duplicate image digest"
                )
            if synthetic_only:
                profile = str(raw.get("condition_profile", "")).strip()
                difficulty = str(raw.get("difficulty", "")).strip()
                style = str(raw.get("plate_style", "")).strip()
                profiles = {
                    str(value)
                    for value in manifest.get("condition_profiles", [])
                }
                try:
                    sample_seed = int(
                        str(raw.get("sample_seed", "")).strip()
                    )
                    quality_score = float(
                        str(raw.get("quality_score", "")).strip()
                    )
                    simulated_source_width = int(
                        str(
                            raw.get("simulated_source_width", "")
                        ).strip()
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"{split_name} synthetic values are invalid at line "
                        f"{line_number}"
                    ) from exc
                if (
                    not profile
                    or profile not in profiles
                    or difficulty not in {"easy", "medium", "hard"}
                    or not style
                    or sample_seed < 0
                    or str(sample_seed)
                    != str(raw.get("sample_seed", "")).strip()
                    or not 0.0 <= quality_score <= 1.0
                    or simulated_source_width <= 0
                    or str(simulated_source_width)
                    != str(
                        raw.get("simulated_source_width", "")
                    ).strip()
                    or image.suffix.lower() != ".jpg"
                    or not _SYNTHETIC_IMAGE_NAME.fullmatch(image.name)
                    or not image.name.endswith(f"-{profile}.jpg")
                    or relative != f"images/{image.name}"
                    or decoded.shape[:2]
                    != (
                        _require_nonnegative_int(
                            manifest,
                            "output_height",
                        ),
                        _require_nonnegative_int(
                            manifest,
                            "output_width",
                        ),
                    )
                ):
                    raise ValueError(
                        f"{split_name} synthetic row is not generator-shaped "
                        f"at line {line_number}"
                    )
            referenced_paths.add(relative)
            digests.add(digest)
            identities.add(plate)
            rows.append({
                **{
                    key: str(value or "")
                    for key, value in raw.items()
                },
                "image_path": relative,
                "plate_text": plate,
                "sha256": digest,
            })
    if not rows:
        raise ValueError(f"{split_name} split has no samples")
    inventory = set()
    for entry in images_root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"{split_name} images cannot contain symlinks")
        if entry.is_file():
            resolved = entry.resolve()
            try:
                resolved.relative_to(images_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"{split_name} image inventory escapes its root"
                ) from exc
            inventory.add(resolved.relative_to(split_dir).as_posix())
    if inventory != referenced_paths:
        raise ValueError(
            f"{split_name} image inventory does not exactly match annotations"
        )
    if synthetic_only:
        _load_synthetic_metadata(split_dir, rows)
    condition_counts = {}
    difficulty_counts = {}
    if synthetic_only:
        condition_counts = {
            str(profile): 0
            for profile in manifest["condition_profiles"]
        }
        difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
        for row in rows:
            profile = row["condition_profile"]
            condition_counts[profile] = condition_counts.get(profile, 0) + 1
            difficulty = row["difficulty"]
            difficulty_counts[difficulty] = (
                difficulty_counts.get(difficulty, 0) + 1
            )
    integrity_rows = [
        {
            "image_path": row["image_path"],
            "plate_text": row["plate_text"],
            "sha256": row["sha256"],
        }
        for row in rows
    ]
    return {
        "name": split_name,
        "directory": directory_name,
        "annotations": annotation_path,
        "annotation_sha256": _sha256(annotation_path),
        "images_fingerprint": _canonical_sha256(integrity_rows),
        "samples": len(rows),
        "identities": identities,
        "digests": digests,
        "rows": rows,
        "condition_counts": condition_counts,
        "difficulty_counts": difficulty_counts,
    }


def _declared_split_count(
    manifest: dict,
    split_name: str,
    *,
    synthetic_only: bool,
    research_only: bool,
) -> int:
    if synthetic_only:
        key = {
            "train": "train_images",
            "validation": "validation_images",
            "test": "test_images",
        }[split_name]
        return _require_nonnegative_int(manifest, key)
    if research_only:
        declared = manifest.get("split_samples")
        if not isinstance(declared, dict):
            raise ValueError("Research dataset split_samples are required")
        return _require_nonnegative_int(declared, split_name)
    key = {
        "train": "train_samples",
        "validation": "validation_samples",
        "test": "test_samples",
    }[split_name]
    return _require_nonnegative_int(manifest, key)


def _preflight_dataset(
    dataset: Path,
    manifest: dict,
    *,
    synthetic_only: bool,
    research_only: bool,
) -> dict:
    dataset = dataset.resolve()
    split_dirs = {
        "train": "train",
        "validation": "val",
        "test": "test",
    }
    if synthetic_only:
        allowed = {"dataset-license.json", "train", "val", "test"}
        extras = {
            child.name for child in dataset.iterdir()
            if child.name not in allowed
        }
        if extras:
            raise ValueError(
                "Synthetic dataset contains non-generator root artifacts"
            )
    split_summaries = {}
    for split_name, directory_name in split_dirs.items():
        split_dir = dataset / directory_name
        annotations = split_dir / "annotations.csv"
        declared = _declared_split_count(
            manifest,
            split_name,
            synthetic_only=synthetic_only,
            research_only=research_only,
        ) if (
            split_name != "test"
            or synthetic_only
            or research_only
            or "test_samples" in manifest
        ) else 0
        exists = annotations.is_file()
        if split_name in {"train", "validation"} and not exists:
            raise ValueError(f"{split_name} split is required")
        if bool(declared) != exists:
            raise ValueError(
                f"{split_name} declared count does not match split presence"
            )
        if not exists:
            if split_dir.exists() and any(split_dir.iterdir()):
                raise ValueError(
                    f"{split_name} data exists without trusted annotations"
                )
            continue
        summary = _preflight_split(
            dataset,
            split_name,
            directory_name,
            synthetic_only=synthetic_only,
            manifest=manifest,
        )
        if summary["samples"] != declared:
            raise ValueError(
                f"{split_name} declared sample count does not match data"
            )
        split_summaries[split_name] = summary

    names = sorted(split_summaries)
    identity_overlaps = {}
    digest_overlaps = {}
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            key = f"{left}_{right}"
            identity_overlaps[key] = len(
                split_summaries[left]["identities"]
                & split_summaries[right]["identities"]
            )
            digest_overlaps[key] = len(
                split_summaries[left]["digests"]
                & split_summaries[right]["digests"]
            )
    if any(identity_overlaps.values()):
        raise ValueError("Plate identity overlap across dataset splits")
    if any(digest_overlaps.values()):
        raise ValueError("Image digest overlap across dataset splits")

    if synthetic_only:
        expected_identity_overlaps = {
            "train_validation": identity_overlaps.get(
                "train_validation", 0
            ),
            "train_test": identity_overlaps.get("train_test", 0),
            "validation_test": identity_overlaps.get(
                "validation_test", 0
            ),
        }
        declared_identity_overlaps = _require_count_map(
            manifest,
            "identity_overlaps",
        )
        if (
            _require_nonnegative_int(manifest, "identity_overlap") != 0
            or declared_identity_overlaps != expected_identity_overlaps
        ):
            raise ValueError(
                "Synthetic identity-overlap declaration is inconsistent"
            )
        for split_name, prefix in (
            ("train", "train"),
            ("validation", "validation"),
            ("test", "test"),
        ):
            expected_unique = _require_nonnegative_int(
                manifest,
                f"{prefix}_unique_plates",
            )
            actual_unique = len(
                split_summaries.get(split_name, {}).get("identities", set())
            )
            if expected_unique != actual_unique:
                raise ValueError(
                    f"{split_name} unique-plate count does not match data"
                )
            expected_conditions = _require_count_map(
                manifest,
                f"{prefix}_conditions",
            )
            expected_difficulty = _require_count_map(
                manifest,
                f"{prefix}_difficulty",
            )
            actual_conditions = split_summaries.get(
                split_name,
                {},
            ).get("condition_counts", {})
            actual_difficulty = split_summaries.get(
                split_name,
                {},
            ).get("difficulty_counts", {})
            if (
                expected_conditions != actual_conditions
                or expected_difficulty != actual_difficulty
            ):
                raise ValueError(
                    f"{split_name} synthetic profile counts do not match data"
                )
    else:
        if _require_nonnegative_int(
            manifest,
            "plate_identity_overlap",
        ) != 0:
            raise ValueError(
                "Dataset must declare zero plate identity overlap"
            )
        if not research_only:
            if _require_nonnegative_int(manifest, "group_overlap") != 0:
                raise ValueError("Dataset must declare zero group overlap")
            for split_name, prefix in (
                ("train", "train"),
                ("validation", "validation"),
            ):
                expected_unique = _require_nonnegative_int(
                    manifest,
                    f"{prefix}_plate_identities",
                )
                actual_unique = len(
                    split_summaries[split_name]["identities"]
                )
                if expected_unique != actual_unique:
                    raise ValueError(
                        f"{split_name} unique-plate count does not match data"
                    )

    public_splits = {
        name: {
            "samples": summary["samples"],
            "annotation_sha256": summary["annotation_sha256"],
            "images_fingerprint": summary["images_fingerprint"],
        }
        for name, summary in sorted(split_summaries.items())
    }
    dataset_fingerprint = _canonical_sha256(public_splits)
    declared_integrity = manifest.get("integrity")
    if not synthetic_only and not research_only:
        expected = {
            "schema": 1,
            "algorithm": "sha256",
            "dataset_fingerprint": dataset_fingerprint,
            "splits": public_splits,
        }
        if declared_integrity != expected:
            raise ValueError(
                "Prepared dataset integrity declaration does not match files"
            )
    return {
        "schema": 1,
        "algorithm": "sha256",
        "dataset_fingerprint": dataset_fingerprint,
        "identity_overlaps": identity_overlaps,
        "digest_overlaps": digest_overlaps,
        "splits": public_splits,
    }


def _checkpoint_contract(
    artifact: Path | None,
    provenance_path: Path | None,
    *,
    role: str,
) -> dict | None:
    if artifact is None:
        if provenance_path is not None:
            raise ValueError(f"{role} provenance was supplied without artifact")
        return None
    artifact = Path(artifact).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if provenance_path is None:
        raise ValueError(f"{role} checkpoint provenance is required")
    provenance_path = Path(provenance_path).resolve()
    if not provenance_path.is_file():
        raise FileNotFoundError(provenance_path)
    try:
        payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{role} checkpoint provenance is invalid") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{role} checkpoint provenance is invalid")
    license_name = str(payload.get("source_license", "")).strip().lower()
    allowed = (
        CHECKPOINT_DISTRIBUTABLE_LICENSES
        | CHECKPOINT_RESEARCH_LICENSES
    )
    actual_digest = _sha256(artifact)
    if (
        type(payload.get("schema")) is not int
        or payload.get("schema") != 1
        or payload.get("artifact_type")
        != "fastplateocr-cct-checkpoint"
        or str(payload.get("artifact_sha256", "")).strip().upper()
        != actual_digest
        or payload.get("golden_benchmark_data") is not False
        or payload.get("training_data_provenance_verified") is not True
        or license_name not in allowed
    ):
        raise ValueError(f"{role} checkpoint provenance contract is invalid")
    distributable = payload.get("distribution_allowed")
    if not isinstance(distributable, bool):
        raise ValueError(
            f"{role} checkpoint distribution policy must be explicit"
        )
    if distributable:
        if license_name not in CHECKPOINT_DISTRIBUTABLE_LICENSES:
            raise ValueError(
                f"{role} checkpoint license is not distributable"
            )
        if license_name in {
            "bcvision-company-owned",
            "operator-confirmed-company-owned",
            "synthetic-bcvision-company-owned",
        }:
            if payload.get("ownership_attested") is not True:
                raise ValueError(
                    f"{role} checkpoint ownership attestation is required"
                )
        elif not str(payload.get("license_evidence", "")).strip():
            raise ValueError(
                f"{role} checkpoint license evidence is required"
            )
    return {
        "schema": 1,
        "artifact_type": "fastplateocr-cct-checkpoint",
        "artifact_sha256": actual_digest,
        "source_license": license_name,
        "distribution_allowed": distributable,
        "golden_benchmark_data": False,
        "training_data_provenance_verified": True,
        "ownership_attested": bool(
            payload.get("ownership_attested", False)
        ),
        "license_evidence": str(
            payload.get("license_evidence", "")
        ).strip(),
        "provenance_sha256": _sha256(provenance_path),
    }


def _dataset_contract(dataset: Path) -> dict:
    dataset = Path(dataset).resolve()
    manifest_path = dataset / "dataset-license.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("Dataset provenance manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Dataset provenance manifest must be an object")
    license_name = str(
        manifest.get("source_license", "")
    ).strip().lower()
    if license_name not in ALLOWED_DATA_LICENSES:
        raise ValueError(
            f"Dataset license is not approved for BC Vision: "
            f"{license_name or 'missing'}"
        )
    if manifest.get("golden_benchmark_data") is not False:
        raise ValueError(
            "Dataset manifest must explicitly forbid Golden benchmark data"
        )
    research_only = (
        license_name == "gpl-3.0-ir-lpr-research-only"
    )
    rights_unverified = (
        license_name == "operator-confirmed-rights-unverified"
    )
    synthetic_only = (
        license_name == "synthetic-bcvision-company-owned"
    )
    if research_only and (
        manifest.get("research_only") is not True
        or manifest.get("distribution_allowed") is not False
        or str(manifest.get("activation_policy", "")).strip().lower()
        != "shadow-only"
        or manifest.get("official_test_split") is not True
    ):
        raise ValueError(
            "IR-LPR must remain research-only, non-distributable and "
            "Shadow-only with an independent test split"
        )
    if rights_unverified and (
        manifest.get("ownership_attested") is not False
        or manifest.get("distribution_allowed") is not False
        or str(manifest.get("activation_policy", "")).strip().lower()
        != "shadow-only-until-rights-attested"
    ):
        raise ValueError(
            "Unverified operator data must remain non-distributable and "
            "Shadow-only"
        )
    if (
        license_name in {
            "bcvision-company-owned",
            "operator-confirmed-company-owned",
        }
        and (
            manifest.get("ownership_attested") is not True
            or manifest.get("distribution_allowed") is not True
        )
    ):
        raise ValueError(
            "Company-owned dataset requires an explicit ownership "
            "attestation and distribution policy"
        )
    if license_name == "synthetic-bcvision-company-owned":
        font_license = str(
            manifest.get("font_license", "")
        ).strip().lower()
        schema = manifest.get("schema")
        condition_profiles = manifest.get("condition_profiles")
        font_path = str(manifest.get("font_path", "")).strip()
        font_digest = str(manifest.get("font_sha256", "")).strip()
        fallback_path = manifest.get("fallback_font_path")
        fallback_digest = manifest.get("fallback_font_sha256")
        fallback_license = manifest.get("fallback_font_license")
        fallback_values = (
            fallback_path,
            fallback_digest,
            fallback_license,
        )
        renderer_digest = str(
            manifest.get("renderer_sha256", "")
        ).strip()
        layout_profiles = manifest.get("layout_profiles")
        forbidden_media_fields = {
            "capture_path",
            "frame_path",
            "golden_path",
            "real_image_path",
            "source_image",
            "source_path",
            "source_video",
            "video_path",
        }
        if (
            type(schema) is not int
            or schema not in {2, 3}
            or manifest.get("third_party_plate_dataset") is not False
            or manifest.get("procedural_only") is not True
            or manifest.get("real_plate_pixels_used") is not False
            or manifest.get("generator")
            != "generate_cct_synthetic_dataset.py"
            or manifest.get("activation_policy")
            != "shadow-only-until-independent-real-camera-pass"
            or not isinstance(condition_profiles, list)
            or not condition_profiles
            or len(condition_profiles) != len(set(condition_profiles))
            or any(
                not isinstance(profile, str) or not profile
                for profile in condition_profiles
            )
            or not font_path
            or re.fullmatch(r"[0-9A-Fa-f]{64}", font_digest) is None
            or any(key in manifest for key in forbidden_media_fields)
            or _require_nonnegative_int(manifest, "output_width") != 128
            or _require_nonnegative_int(manifest, "output_height") != 64
            or _require_nonnegative_int(manifest, "label_slots") != 8
            or font_license not in ALLOWED_FONT_LICENSES
            or (
                schema == 3
                and (
                    manifest.get("layout_profile")
                    != "iran-national-photo-reference-v1"
                    or manifest.get("renderer")
                    != "iran_plate_renderer.py"
                    or re.fullmatch(
                        r"[0-9A-Fa-f]{64}",
                        renderer_digest,
                    )
                    is None
                    or renderer_digest.upper()
                    != _sha256(
                        Path(__file__).with_name(
                            "iran_plate_renderer.py"
                        )
                    )
                    or not isinstance(layout_profiles, dict)
                    or layout_profiles
                    != {
                        "private": "iran-national-photo-reference-v1",
                        "special": "legacy-procedural-v2",
                    }
                    or _require_nonnegative_int(
                        manifest,
                        "base_plate_width",
                    )
                    <= 0
                    or _require_nonnegative_int(
                        manifest,
                        "base_plate_height",
                    )
                    <= 0
                    or (
                        any(value is not None for value in fallback_values)
                        and (
                            not all(
                                isinstance(value, str) and value.strip()
                                for value in fallback_values
                            )
                            or re.fullmatch(
                                r"[0-9A-Fa-f]{64}",
                                str(fallback_digest),
                            )
                            is None
                            or str(fallback_license).strip().lower()
                            not in ALLOWED_FONT_LICENSES
                        )
                    )
                )
            )
        ):
            raise ValueError(
                "Synthetic data provenance or font license is not approved"
            )
    integrity = _preflight_dataset(
        dataset,
        manifest,
        synthetic_only=synthetic_only,
        research_only=research_only,
    )
    train = dataset / "train" / "annotations.csv"
    validation = dataset / "val" / "annotations.csv"
    test = dataset / "test" / "annotations.csv"
    return {
        "dataset": dataset,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256(manifest_path),
        "manifest": manifest,
        "train": train,
        "validation": validation,
        "test": test if test.is_file() else None,
        "research_only": research_only,
        "rights_unverified": rights_unverified,
        "synthetic_only": synthetic_only,
        "integrity": integrity,
    }


def _deployment_policy(
    contract: dict,
    checkpoint_contracts: tuple[dict, ...] = (),
) -> dict:
    checkpoint_distributable = all(
        row.get("distribution_allowed") is True
        for row in checkpoint_contracts
    )
    if checkpoint_contracts and not checkpoint_distributable:
        return {
            "usage_scope": "research-shadow-only",
            "distribution_allowed": False,
            "activation_allowed": False,
            "activation_gate": "checkpoint-license-and-real-camera-pass",
        }
    if contract["research_only"] or contract.get(
        "rights_unverified",
        False,
    ):
        return {
            "usage_scope": "research-shadow-only",
            "distribution_allowed": False,
            "activation_allowed": False,
            "activation_gate": (
                "rights-attestation-and-real-camera-pass"
                if contract.get("rights_unverified", False)
                else "commercial-license-and-real-camera-pass"
            ),
        }
    if contract["synthetic_only"]:
        return {
            "usage_scope": "production-candidate",
            "distribution_allowed": True,
            "activation_allowed": False,
            "activation_gate": "independent-real-camera-pass",
        }
    return {
        "usage_scope": "production-candidate",
        "distribution_allowed": True,
        "activation_allowed": False,
        "activation_gate": "independent-golden-and-real-camera-pass",
    }


def _assert_dataset_unchanged(contract: dict) -> None:
    current = _dataset_contract(contract["dataset"])
    if (
        current["manifest_sha256"] != contract["manifest_sha256"]
        or current["integrity"] != contract["integrity"]
    ):
        raise ValueError("Dataset changed after integrity preflight")


def _is_excluded_pretrained_layer(name: str) -> bool:
    return (
        name in EXCLUDED_PRETRAINED_LAYERS
        or name.startswith("region_")
    )


def _copy_pretrained_backbone(source_model, target_model) -> list[str]:
    """Copy only shape-compatible feature layers, never OCR/region heads."""
    source_layers = [
        layer
        for layer in source_model.layers
        if layer.get_weights()
        and not _is_excluded_pretrained_layer(layer.name)
    ]
    target_layers = [
        layer
        for layer in target_model.layers
        if layer.get_weights()
        and not _is_excluded_pretrained_layer(layer.name)
    ]
    if len(source_layers) != len(target_layers):
        raise ValueError(
            "Pretrained backbone architecture has a different number "
            "of weighted feature layers"
        )
    planned_weights = []
    transferred = []
    copied_elements = 0
    target_elements = 0
    for source_layer, target_layer in zip(
        source_layers,
        target_layers,
        strict=True,
    ):
        target_weights = target_layer.get_weights()
        source_weights = source_layer.get_weights()
        if (
            type(source_layer).__name__ != type(target_layer).__name__
            or len(source_weights) != len(target_weights)
        ):
            raise ValueError(
                "Pretrained backbone architecture does not match "
                f"at layer: {target_layer.name}"
            )
        replacement = []
        for source, target in zip(
            source_weights,
            target_weights,
            strict=True,
        ):
            target_elements += int(target.size)
            if source.shape == target.shape:
                replacement.append(source)
                copied_elements += int(target.size)
            else:
                replacement.append(target)
        planned_weights.append((target_layer, replacement))
        transferred.append(target_layer.name)
    transfer_ratio = (
        copied_elements / target_elements
        if target_elements
        else 0.0
    )
    if transfer_ratio < 0.95:
        raise ValueError(
            "Pretrained backbone architecture does not match: "
            f"only {transfer_ratio:.1%} of feature parameters are compatible"
        )
    for target_layer, replacement in planned_weights:
        target_layer.set_weights(replacement)
    if not transferred:
        raise ValueError("Pretrained model supplied no transferable backbone")
    return transferred


def _prepare_pretrained_backbone(
    source_path: Path,
    model_config_path: Path,
    plate_config_path: Path,
    output: Path,
) -> tuple[Path, list[str]]:
    from fast_plate_ocr.train.model.config import (
        load_plate_config_from_yaml,
    )
    from fast_plate_ocr.train.model.model_builders import build_model
    from fast_plate_ocr.train.model.model_schema import (
        load_model_config_from_yaml,
    )
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    plate_config = load_plate_config_from_yaml(plate_config_path)
    model_config = load_model_config_from_yaml(model_config_path)
    source_model = load_keras_model(source_path, plate_config)
    target_model = build_model(
        model_config,
        plate_config,
        enable_region_head=False,
    )
    transferred = _copy_pretrained_backbone(source_model, target_model)
    initialized = output / "pretrained-backbone.keras"
    target_model.save(initialized)
    return initialized, transferred


def _run_official_training(
    model_config: Path,
    plate_config: Path,
    train_annotations: Path,
    validation_annotations: Path,
    output: Path,
    initialized_weights: Path | None,
    checkpoint_metric: str,
    augmentation_path: Path | None,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Path:
    from fast_plate_ocr.cli.train import train as train_command

    arguments = [
        "--model-config-file",
        str(model_config),
        "--plate-config-file",
        str(plate_config),
        "--annotations",
        str(train_annotations),
        "--val-annotations",
        str(validation_annotations),
        "--validate-dataset",
        "error",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output / "keras-runs"),
        "--early-stopping-patience",
        str(max(4, min(12, epochs // 3))),
        "--early-stopping-metric",
        (
            "val_plate_char_acc"
            if checkpoint_metric == "char"
            else "val_plate_acc"
        ),
        "--label-smoothing",
        "0.01",
        "--weight-decay",
        "0.0005",
        "--lr",
        str(learning_rate),
        "--seed",
        str(seed),
        "--workers",
        "1",
        "--no-use-multiprocessing",
    ]
    if initialized_weights is not None:
        arguments.extend(["--weights-path", str(initialized_weights)])
    if augmentation_path is not None:
        arguments.extend([
            "--augmentation-path",
            str(augmentation_path),
        ])
    train_command.main(args=arguments, standalone_mode=False)
    candidates = sorted(
        (output / "keras-runs").rglob("best.keras"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError("FastPlateOCR produced no best.keras checkpoint")
    return candidates[-1]


def _export_onnx(
    checkpoint: Path,
    plate_config: Path,
    output: Path,
    variant: str,
) -> Path:
    from fast_plate_ocr.cli.export import export as export_command

    export_dir = output / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_command.main(
        args=[
            "--model",
            str(checkpoint),
            "--format",
            "onnx",
            "--plate-config-file",
            str(plate_config),
            "--save-dir",
            str(export_dir),
            "--no-dynamic-batch",
            "--onnx-input-dtype",
            "uint8",
            "--onnx-data-format",
            "channels_last",
            "--no-simplify",
        ],
        standalone_mode=False,
    )
    exported = export_dir / checkpoint.with_suffix(".onnx").name
    if not exported.is_file():
        raise RuntimeError("FastPlateOCR ONNX export is missing")
    candidate = output / f"bcvision-cct-{variant}.onnx"
    shutil.copy2(exported, candidate)
    return candidate


def _validation_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if (
            not reader.fieldnames
            or not {"image_path", "plate_text"}.issubset(
                reader.fieldnames
            )
        ):
            raise ValueError("Validation annotations are incomplete")
        images_root = path.parent / "images"
        for line_number, row in enumerate(reader, 2):
            if None in row:
                raise ValueError(
                    f"Malformed validation row at line {line_number}"
                )
            image, _relative = _contained_image(
                images_root,
                row.get("image_path", ""),
                split_name="validation",
                line_number=line_number,
            )
            expected = normalize_plate(row["plate_text"])
            if not plausible_plate(expected):
                raise ValueError(
                    f"Implausible validation label at line {line_number}"
                )
            rows.append({
                "image": image,
                "expected": expected,
                "condition_profile": str(
                    row.get("condition_profile", "unknown")
                ),
                "difficulty": str(
                    row.get("difficulty", "unknown")
                ),
                "plate_style": str(
                    row.get("plate_style", "unknown")
                ),
            })
    if not rows:
        raise ValueError("Validation dataset has no usable rows")
    return rows


def _empty_metric_bucket() -> dict:
    return {
        "samples": 0,
        "raw_exact_matches": 0,
        "raw_character_matches": 0,
        "raw_character_total": 0,
        "raw_character_distance": 0,
        "position_matches": [0] * 8,
        "accepted_samples": 0,
        "accepted_exact_matches": 0,
    }


def _update_metric_bucket(
    bucket: dict,
    expected: str,
    raw: str,
    result: dict,
) -> None:
    bucket["samples"] += 1
    bucket["raw_exact_matches"] += raw == expected
    bucket["raw_character_matches"] += sum(
        observed == target
        for observed, target in zip(raw, expected)
    )
    bucket["raw_character_total"] += len(expected)
    bucket["raw_character_distance"] += character_distance(raw, expected)
    for index, target in enumerate(expected):
        if index < len(raw) and raw[index] == target:
            bucket["position_matches"][index] += 1
    accepted = bool(result["accepted"])
    bucket["accepted_samples"] += accepted
    bucket["accepted_exact_matches"] += (
        accepted and result["plate_norm"] == expected
    )


def _finalize_metric_bucket(bucket: dict) -> dict:
    samples = int(bucket["samples"])
    accepted = int(bucket["accepted_samples"])
    if samples <= 0:
        raise ValueError("Cannot finalize an empty metric bucket")
    return {
        "samples": samples,
        "raw_exact_matches": int(bucket["raw_exact_matches"]),
        "raw_exact_accuracy": round(
            bucket["raw_exact_matches"] / samples,
            6,
        ),
        "raw_character_accuracy": round(
            bucket["raw_character_matches"]
            / bucket["raw_character_total"],
            6,
        ),
        "raw_mean_character_error": round(
            bucket["raw_character_distance"] / samples,
            6,
        ),
        "raw_position_accuracy": [
            round(matches / samples, 6)
            for matches in bucket["position_matches"]
        ],
        "accepted_samples": accepted,
        "accepted_exact_matches": int(
            bucket["accepted_exact_matches"]
        ),
        "accepted_exact_accuracy": round(
            bucket["accepted_exact_matches"] / samples,
            6,
        ),
        "accepted_precision": (
            round(bucket["accepted_exact_matches"] / accepted, 6)
            if accepted
            else 0.0
        ),
        "rejection_rate": round((samples - accepted) / samples, 6),
    }


def _benchmark(
    model: Path,
    validation: Path,
    alphabet: str,
    preprocess_profile=CCT_PREPROCESS_LEGACY,
) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    rows = _validation_rows(validation)
    spec = {
        "input_width": 128,
        "input_height": 64,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "image_color_mode": "rgb",
        "keep_aspect_ratio": False,
        "interpolation": "linear",
        "padding_color": [114, 114, 114],
        "alphabet": alphabet,
        "max_plate_slots": 8,
        "beam_width": 16,
        "top_k": 5,
        "preprocess_profile": preprocess_profile,
        "fusion_method": (
            CCT_FUSION_GEOMETRIC_MEAN
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else CCT_FUSION_IDENTITY
        ),
        "min_confidence": 0.58,
        "min_position_confidence": (
            0.50
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.42
        ),
        "min_position_margin": (
            0.08
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.06
        ),
        "min_hypothesis_margin": (
            0.03
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.025
        ),
        "min_view_agreement": (
            0.75
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.0
        ),
    }
    images = []
    for row in rows:
        image = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        if image is None or not image.size:
            raise ValueError(f"Unreadable validation image: {row['image']}")
        images.append(image)
    for image in images[: min(10, len(images))]:
        infer_cct_session(
            session,
            input_meta.name,
            image,
            spec,
        )

    overall = _empty_metric_bucket()
    grouped = {
        field: {}
        for field in (
            "condition_profile",
            "difficulty",
            "plate_style",
        )
    }
    started = time.perf_counter()
    for row, image in zip(rows, images, strict=True):
        result = infer_cct_session(
            session,
            input_meta.name,
            image,
            spec,
        )
        hypotheses = result["hypotheses"]
        raw = (
            hypotheses[0]["plate_norm"]
            if hypotheses
            else ""
        )
        _update_metric_bucket(
            overall,
            expected=row["expected"],
            raw=raw,
            result=result,
        )
        for field, values in grouped.items():
            key = row[field]
            bucket = values.setdefault(key, _empty_metric_bucket())
            _update_metric_bucket(
                bucket,
                expected=row["expected"],
                raw=raw,
                result=result,
            )
    elapsed = time.perf_counter() - started
    output_shape = [
        dimension if isinstance(dimension, int) else str(dimension)
        for dimension in session.get_outputs()[0].shape
    ]
    metrics = _finalize_metric_bucket(overall)
    metrics["validation_samples"] = metrics.pop("samples")
    return {
        **metrics,
        "by_condition_profile": {
            key: _finalize_metric_bucket(bucket)
            for key, bucket in sorted(
                grouped["condition_profile"].items()
            )
        },
        "by_difficulty": {
            key: _finalize_metric_bucket(bucket)
            for key, bucket in sorted(grouped["difficulty"].items())
        },
        "by_plate_style": {
            key: _finalize_metric_bucket(bucket)
            for key, bucket in sorted(grouped["plate_style"].items())
        },
        "elapsed_seconds": round(elapsed, 6),
        "mean_latency_ms": round(elapsed * 1000 / len(rows), 6),
        "input_name": input_meta.name,
        "input_shape": [
            dimension if isinstance(dimension, int) else str(dimension)
            for dimension in input_meta.shape
        ],
        "input_type": input_meta.type,
        "output_shape": output_shape,
        "providers": session.get_providers(),
    }


def train_and_export(
    dataset: Path,
    output: Path,
    variant: str,
    pretrained_backbone: Path | None,
    epochs: int,
    batch_size: int,
    seed: int,
    resume_checkpoint: Path | None = None,
    checkpoint_metric: str = "char",
    augmentation_path: Path | None = None,
    learning_rate: float = 0.0005,
    pretrained_provenance: Path | None = None,
    resume_provenance: Path | None = None,
    preprocess_profile: str = CCT_PREPROCESS_LEGACY,
) -> dict:
    dataset = dataset.resolve()
    output = output.resolve()
    if pretrained_backbone is not None and resume_checkpoint is not None:
        raise ValueError(
            "Choose either pretrained backbone transfer or resume checkpoint"
        )
    if checkpoint_metric not in {"char", "exact"}:
        raise ValueError("Checkpoint metric must be char or exact")
    if preprocess_profile not in {
        CCT_PREPROCESS_LEGACY,
        CCT_PREPROCESS_DUAL_VIEW,
    }:
        raise ValueError("Unsupported CCT preprocess profile")
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    contract = _dataset_contract(dataset)
    pretrained_contract = _checkpoint_contract(
        pretrained_backbone,
        pretrained_provenance,
        role="Pretrained backbone",
    )
    resume_contract = _checkpoint_contract(
        resume_checkpoint,
        resume_provenance,
        role="Resume",
    )
    checkpoint_contracts = tuple(
        row for row in (pretrained_contract, resume_contract)
        if row is not None
    )
    root = Path(__file__).resolve().parents[1]
    plate_config = _training_plate_config(
        root,
        preprocess_profile,
    )
    model_config = (
        root
        / "training"
        / "cct"
        / f"cct_{variant}_v2_model_config.yaml"
    )
    if not model_config.is_file() or not plate_config.is_file():
        raise FileNotFoundError("BC Vision CCT configuration is missing")
    if augmentation_path is not None and not augmentation_path.is_file():
        raise FileNotFoundError(augmentation_path)
    if not 0 < learning_rate <= 0.1:
        raise ValueError("Learning rate must be between zero and 0.1")
    output.mkdir(parents=True)
    initialized_weights = resume_checkpoint
    transferred_layers = []
    if pretrained_backbone is not None:
        initialized_weights, transferred_layers = (
            _prepare_pretrained_backbone(
                source_path=pretrained_backbone,
                model_config_path=model_config,
                plate_config_path=plate_config,
                output=output,
            )
        )

    checkpoint = _run_official_training(
        model_config=model_config,
        plate_config=plate_config,
        train_annotations=contract["train"],
        validation_annotations=contract["validation"],
        output=output,
        initialized_weights=initialized_weights,
        checkpoint_metric=checkpoint_metric,
        augmentation_path=augmentation_path,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    _assert_dataset_unchanged(contract)
    model = _export_onnx(
        checkpoint=checkpoint,
        plate_config=plate_config,
        output=output,
        variant=variant,
    )
    alphabet = "0123456789ابپتثجدزژسشصطعفقکگلمنوهیDS_"
    metrics = _benchmark(
        model=model,
        validation=contract["validation"],
        alphabet=alphabet,
        preprocess_profile=preprocess_profile,
    )
    test_metrics = (
        _benchmark(
            model=model,
            validation=contract["test"],
            alphabet=alphabet,
            preprocess_profile=preprocess_profile,
        )
        if contract["test"] is not None
        else None
    )
    _assert_dataset_unchanged(contract)
    deployment = _deployment_policy(
        contract,
        checkpoint_contracts=checkpoint_contracts,
    )
    metadata = {
        "schema": 1,
        "runtime": "fast-plate-ocr-cct",
        "variant": f"cct-{variant}-v2",
        "model_path": model.name,
        "sha256": _sha256(model),
        "size": model.stat().st_size,
        "alphabet": alphabet,
        "max_plate_slots": 8,
        "input_width": 128,
        "input_height": 64,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "image_color_mode": "rgb",
        "keep_aspect_ratio": False,
        "interpolation": "linear",
        "padding_color": [114, 114, 114],
        "preprocess_profile": preprocess_profile,
        "fusion_method": (
            CCT_FUSION_GEOMETRIC_MEAN
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else CCT_FUSION_IDENTITY
        ),
        "min_confidence": 0.58,
        "min_position_confidence": (
            0.50
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.42
        ),
        "min_position_margin": (
            0.08
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.06
        ),
        "min_hypothesis_margin": (
            0.03
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.025
        ),
        "min_view_agreement": (
            0.75
            if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
            else 0.0
        ),
        "beam_width": 16,
        "top_k": 5,
        "dataset_integrity": contract["integrity"],
        "dataset_manifest_sha256": contract["manifest_sha256"],
        **deployment,
        "training": {
            "keras_backend": os.environ.get("KERAS_BACKEND", ""),
            "dataset_license": contract["manifest"],
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "pretrained_backbone_path": (
                pretrained_backbone.name
                if pretrained_backbone
                else ""
            ),
            "pretrained_backbone_sha256": (
                _sha256(pretrained_backbone)
                if pretrained_backbone
                else ""
            ),
            "pretrained_transferred_layers": transferred_layers,
            "pretrained_excluded_layers": sorted(
                EXCLUDED_PRETRAINED_LAYERS
            ),
            "resume_checkpoint_path": (
                resume_checkpoint.name
                if resume_checkpoint
                else ""
            ),
            "resume_checkpoint_sha256": (
                _sha256(resume_checkpoint)
                if resume_checkpoint
                else ""
            ),
            "initialization_provenance": {
                "pretrained_backbone": pretrained_contract,
                "resume_checkpoint": resume_contract,
            },
            "checkpoint_metric": checkpoint_metric,
            "augmentation_path": (
                augmentation_path.name
                if augmentation_path
                else "fastplateocr-default"
            ),
            "learning_rate": float(learning_rate),
            "checkpoint": checkpoint.name,
            "input_preprocess_profile": (
                "letterbox-v1"
                if preprocess_profile == CCT_PREPROCESS_DUAL_VIEW
                else "stretch-v1"
            ),
            "plate_config": plate_config.name,
            "plate_config_sha256": _sha256(plate_config),
        },
        "validation": metrics,
        "test": test_metrics,
    }
    (output / "candidate-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and verify BC Vision FastPlateOCR CCT",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["xs", "s"],
        required=True,
    )
    parser.add_argument(
        "--pretrained-backbone",
        type=Path,
        help=(
            "Optional FastPlateOCR model used only for compatible feature "
            "layers; OCR and region heads are always excluded"
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Resume all compatible model weights from a previous "
            "FastPlateOCR .keras checkpoint"
        ),
    )
    parser.add_argument(
        "--pretrained-provenance",
        type=Path,
        help=(
            "Required hash-bound license contract for --pretrained-backbone"
        ),
    )
    parser.add_argument(
        "--resume-provenance",
        type=Path,
        help="Required hash-bound license contract for --resume-checkpoint",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=["char", "exact"],
        default="char",
        help=(
            "Use character accuracy while learning from scratch; switch to "
            "exact plate accuracy only for a mature model"
        ),
    )
    parser.add_argument(
        "--augmentation-path",
        type=Path,
        help="Optional Albumentations YAML used for this training stage",
    )
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument(
        "--preprocess-profile",
        choices=[
            CCT_PREPROCESS_LEGACY,
            CCT_PREPROCESS_DUAL_VIEW,
        ],
        default=CCT_PREPROCESS_LEGACY,
    )
    args = parser.parse_args(argv)
    result = train_and_export(
        dataset=args.dataset,
        output=args.output,
        variant=args.variant,
        pretrained_backbone=(
            args.pretrained_backbone.resolve()
            if args.pretrained_backbone
            else None
        ),
        resume_checkpoint=(
            args.resume_checkpoint.resolve()
            if args.resume_checkpoint
            else None
        ),
        pretrained_provenance=(
            args.pretrained_provenance.resolve()
            if args.pretrained_provenance
            else None
        ),
        resume_provenance=(
            args.resume_provenance.resolve()
            if args.resume_provenance
            else None
        ),
        checkpoint_metric=args.checkpoint_metric,
        augmentation_path=(
            args.augmentation_path.resolve()
            if args.augmentation_path
            else None
        ),
        learning_rate=float(args.learning_rate),
        epochs=max(1, min(200, int(args.epochs))),
        batch_size=max(4, min(256, int(args.batch_size))),
        seed=int(args.seed),
        preprocess_profile=args.preprocess_profile,
    )
    print(json.dumps({
        "variant": result["variant"],
        "model_path": result["model_path"],
        "sha256": result["sha256"],
        "size": result["size"],
        "validation": result["validation"],
        "test": result["test"],
        "usage_scope": result["usage_scope"],
        "distribution_allowed": result["distribution_allowed"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
