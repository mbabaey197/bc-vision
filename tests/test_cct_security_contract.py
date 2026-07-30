import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from tools.generate_cct_synthetic_dataset import generate
from tools.prepare_cct_dataset import prepare
from tools.train_fastplate_cct import (
    _checkpoint_contract,
    _dataset_contract,
    _deployment_policy,
)


PLATES = (
    "31ط55674",
    "55ت63974",
    "84ب57133",
    "21ک12345",
)


def _write_image(path: Path, value: int) -> None:
    image = np.full((32, 128, 3), value, dtype=np.uint8)
    image[:, ::7, 1] = (value + 31) % 255
    assert cv2.imwrite(str(path), image)


def _prepared_dataset(tmp_path: Path) -> Path:
    rows = []
    for index, plate in enumerate(PLATES):
        image = tmp_path / f"source-{index}.png"
        _write_image(image, 30 + index * 45)
        rows.append({
            "image_path": image.name,
            "plate_text": plate,
            "group_id": plate,
            "source_license": "bcvision-company-owned",
            "usage": "train",
        })
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dataset = tmp_path / "prepared"
    prepare(
        source_manifest=source,
        output=dataset,
        validation_ratio=0.5,
        seed=20260730,
    )
    return dataset


def _annotation_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_annotations(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _font() -> Path:
    candidates = (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    pytest.skip("No Persian-capable test font is installed")


def _synthetic_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "synthetic"
    generate(
        output=dataset,
        train_plates=10,
        validation_plates=3,
        test_plates=2,
        views_per_plate=1,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260730,
        jpeg_quality=82,
    )
    return dataset


def _checkpoint_manifest(
    artifact: Path,
    *,
    source_license: str,
    distribution_allowed: bool,
    **extra,
) -> dict:
    return {
        "schema": 1,
        "artifact_type": "fastplateocr-cct-checkpoint",
        "artifact_sha256": hashlib.sha256(
            artifact.read_bytes()
        ).hexdigest().upper(),
        "source_license": source_license,
        "golden_benchmark_data": False,
        "training_data_provenance_verified": True,
        "distribution_allowed": distribution_allowed,
        **extra,
    }


def test_prepared_dataset_binds_annotations_images_and_counts(tmp_path):
    dataset = _prepared_dataset(tmp_path)

    contract = _dataset_contract(dataset)

    assert (
        contract["integrity"]["dataset_fingerprint"]
        == contract["manifest"]["integrity"]["dataset_fingerprint"]
    )
    assert (
        contract["integrity"]["splits"]
        == contract["manifest"]["integrity"]["splits"]
    )
    assert contract["integrity"]["digest_overlaps"] == {
        "train_validation": 0,
    }
    assert contract["integrity"]["identity_overlaps"] == {
        "train_validation": 0,
    }


def test_dataset_preflight_rejects_declared_count_tampering(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    manifest_path = dataset / "dataset-license.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_samples"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="declared sample count"):
        _dataset_contract(dataset)


def test_dataset_preflight_rejects_path_traversal(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    annotations = dataset / "train" / "annotations.csv"
    rows = _annotation_rows(annotations)
    rows[0]["image_path"] = "../../source-0.png"
    _write_annotations(annotations, rows)

    with pytest.raises(ValueError, match="must be relative"):
        _dataset_contract(dataset)


def test_dataset_preflight_rejects_symlinked_image(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    annotations = dataset / "train" / "annotations.csv"
    image = dataset / "train" / _annotation_rows(annotations)[0]["image_path"]
    replacement = image.with_name("replacement.png")
    replacement.write_bytes(image.read_bytes())
    image.unlink()
    image.symlink_to(replacement.name)

    with pytest.raises(ValueError, match="symlink"):
        _dataset_contract(dataset)


def test_dataset_preflight_rejects_implausible_label(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    annotations = dataset / "train" / "annotations.csv"
    rows = _annotation_rows(annotations)
    rows[0]["plate_text"] = "NOT-A-PLATE"
    _write_annotations(annotations, rows)

    with pytest.raises(ValueError, match="implausible"):
        _dataset_contract(dataset)


def test_dataset_preflight_rejects_cross_split_identity_overlap(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    train = _annotation_rows(dataset / "train" / "annotations.csv")
    validation_path = dataset / "val" / "annotations.csv"
    validation = _annotation_rows(validation_path)
    validation[0]["plate_text"] = train[0]["plate_text"]
    _write_annotations(validation_path, validation)

    with pytest.raises(ValueError, match="identity overlap"):
        _dataset_contract(dataset)


def test_dataset_preflight_rejects_cross_split_digest_overlap(tmp_path):
    dataset = _prepared_dataset(tmp_path)
    train_row = _annotation_rows(
        dataset / "train" / "annotations.csv"
    )[0]
    validation_row = _annotation_rows(
        dataset / "val" / "annotations.csv"
    )[0]
    train_image = dataset / "train" / train_row["image_path"]
    validation_image = dataset / "val" / validation_row["image_path"]
    shutil.copyfile(train_image, validation_image)

    with pytest.raises(ValueError, match="digest overlap"):
        _dataset_contract(dataset)


def test_prepare_rejects_cc_by_without_attribution_support(tmp_path):
    image = tmp_path / "plate.png"
    _write_image(image, 100)
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate.png,31ط55674,31ط55674,cc-by-4.0,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unapproved source license"):
        prepare(
            source_manifest=source,
            output=tmp_path / "prepared",
            validation_ratio=0.5,
            seed=1,
        )


def test_synthetic_contract_rejects_non_generator_artifacts(tmp_path):
    dataset = _synthetic_dataset(tmp_path)
    assert _dataset_contract(dataset)["synthetic_only"] is True
    (dataset / "golden").mkdir()

    with pytest.raises(ValueError, match="non-generator root artifacts"):
        _dataset_contract(dataset)


def test_synthetic_contract_rejects_split_media_artifact(tmp_path):
    dataset = _synthetic_dataset(tmp_path)
    (dataset / "train" / "golden-video.mp4").write_bytes(b"not training data")

    with pytest.raises(ValueError, match="unexpected artifacts"):
        _dataset_contract(dataset)


def test_synthetic_contract_binds_condition_counts(tmp_path):
    dataset = _synthetic_dataset(tmp_path)
    manifest_path = dataset / "dataset-license.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["train_conditions"]["clean"] += 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="profile counts"):
        _dataset_contract(dataset)


def test_synthetic_contract_rejects_real_media_metadata(tmp_path):
    dataset = _synthetic_dataset(tmp_path)
    metadata_path = dataset / "train" / "samples.jsonl"
    rows = [
        json.loads(line)
        for line in metadata_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[0]["source_path"] = ["golden", "01.mp4"]
    metadata_path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="real/Golden media"):
        _dataset_contract(dataset)


def test_checkpoint_requires_hash_bound_provenance(tmp_path):
    artifact = tmp_path / "checkpoint.keras"
    artifact.write_bytes(b"checkpoint")

    with pytest.raises(ValueError, match="provenance is required"):
        _checkpoint_contract(artifact, None, role="Resume")

    provenance = tmp_path / "checkpoint.provenance.json"
    payload = _checkpoint_manifest(
        artifact,
        source_license="mit",
        distribution_allowed=True,
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="license evidence"):
        _checkpoint_contract(artifact, provenance, role="Resume")

    payload["license_evidence"] = "SPDX MIT; upstream release manifest"
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    contract = _checkpoint_contract(artifact, provenance, role="Resume")
    assert contract["artifact_sha256"] == payload["artifact_sha256"]
    assert contract["distribution_allowed"] is True

    artifact.write_bytes(b"changed checkpoint")
    with pytest.raises(ValueError, match="contract is invalid"):
        _checkpoint_contract(artifact, provenance, role="Resume")


def test_company_checkpoint_requires_ownership_attestation(tmp_path):
    artifact = tmp_path / "checkpoint.keras"
    artifact.write_bytes(b"company checkpoint")
    provenance = tmp_path / "checkpoint.provenance.json"
    payload = _checkpoint_manifest(
        artifact,
        source_license="bcvision-company-owned",
        distribution_allowed=True,
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="ownership attestation"):
        _checkpoint_contract(artifact, provenance, role="Pretrained")

    payload["ownership_attested"] = True
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    contract = _checkpoint_contract(
        artifact,
        provenance,
        role="Pretrained",
    )
    assert contract["distribution_allowed"] is True


def test_research_checkpoint_locks_candidate_distribution(tmp_path):
    artifact = tmp_path / "checkpoint.keras"
    artifact.write_bytes(b"research checkpoint")
    provenance = tmp_path / "checkpoint.provenance.json"
    payload = _checkpoint_manifest(
        artifact,
        source_license="gpl-3.0-ir-lpr-research-only",
        distribution_allowed=False,
    )
    provenance.write_text(json.dumps(payload), encoding="utf-8")
    checkpoint = _checkpoint_contract(
        artifact,
        provenance,
        role="Resume",
    )

    policy = _deployment_policy(
        {"research_only": False, "synthetic_only": False},
        checkpoint_contracts=(checkpoint,),
    )
    assert policy["distribution_allowed"] is False
    assert policy["activation_allowed"] is False
