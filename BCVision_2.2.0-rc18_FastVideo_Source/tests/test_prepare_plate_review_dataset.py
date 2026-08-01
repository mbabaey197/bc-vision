import csv
import hashlib
import json
from pathlib import Path
import zipfile

from PIL import Image
import pytest

from tools.build_plate_label_review import build_review_page
from tools.prepare_plate_review_dataset import prepare_review_dataset


def _image(path: Path, color) -> None:
    Image.new("RGB", (128, 32), color).save(path, quality=92)


def _fixture(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    labels = {
        "a.jpg": "12ع34567",
        "b.jpg": "23ع45678",
        "c.jpg": "34ع56789",
        "d.jpg": "45د67890",
        "e.jpg": "",
    }
    for index, name in enumerate(labels):
        _image(images / name, (80 + index * 20, 100, 40))
    archive = tmp_path / "source.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(images.iterdir()):
            handle.write(path, arcname=f"nested/{path.name}")
    review_page = tmp_path / "review.html"
    build_review_page(
        images,
        review_page,
        source_archive=archive,
        ownership_evidence="user-attestation-test-company-owned",
    )
    review_csv = tmp_path / "review.csv"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["file_name", "sha256", "status", "plate"],
        )
        writer.writeheader()
        for name, plate in labels.items():
            writer.writerow({
                "file_name": name,
                "sha256": hashlib.sha256(
                    (images / name).read_bytes()
                ).hexdigest().upper(),
                "status": "confirmed" if plate else "pending",
                "plate": plate,
            })
    return archive, review_page, review_csv


def test_review_csv_prepares_identity_isolated_company_dataset(tmp_path):
    archive, review_page, review_csv = _fixture(tmp_path)
    output = tmp_path / "dataset"

    manifest = prepare_review_dataset(
        review_csv,
        archive,
        review_page,
        output,
        validation_ratio=0.25,
        seed=7,
    )

    assert manifest["source_license"] == (
        "operator-confirmed-company-owned"
    )
    assert manifest["ownership_attested"] is True
    assert manifest["distribution_allowed"] is True
    assert manifest["train_samples"] == 3
    assert manifest["validation_samples"] == 1
    assert manifest["plate_identity_overlap"] == 0
    assert manifest["review_export"]["confirmed_samples"] == 4
    assert manifest["review_export"][
        "model_suggestions_used_as_labels"
    ] is False
    assert len(list((output / "train" / "images").iterdir())) == 3
    assert len(list((output / "val" / "images").iterdir())) == 1
    saved = json.loads(
        (output / "dataset-license.json").read_text(encoding="utf-8")
    )
    assert saved["integrity"] == manifest["integrity"]


def test_review_dataset_rejects_archive_not_bound_to_review_page(tmp_path):
    archive, review_page, review_csv = _fixture(tmp_path)
    with zipfile.ZipFile(archive, "a") as handle:
        handle.writestr("extra.jpg", b"changed")

    with pytest.raises(ValueError, match="archive hash"):
        prepare_review_dataset(
            review_csv,
            archive,
            review_page,
            tmp_path / "dataset",
        )


def test_review_dataset_rejects_unconfirmed_plate_text(tmp_path):
    archive, review_page, review_csv = _fixture(tmp_path)
    rows = list(csv.DictReader(
        review_csv.open(encoding="utf-8-sig", newline="")
    ))
    rows[-1]["plate"] = "56ع78901"
    with review_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="Non-confirmed"):
        prepare_review_dataset(
            review_csv,
            archive,
            review_page,
            tmp_path / "dataset",
        )
