import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ai.plate_rules import ALLOWED_PLATE_LETTERS
from tools.generate_cct_synthetic_dataset import (
    _unique_plates,
    generate,
)
from tools.prepare_cct_dataset import _load_rows, prepare
from tools.train_fastplate_cct import (
    _copy_pretrained_backbone,
    _dataset_contract,
)


def _font() -> Path:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    if not path.is_file():
        pytest.skip("DejaVu Sans is not installed")
    return path


def _plate_texts(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["plate_text"]
            for row in csv.DictReader(handle)
        }


def test_synthetic_cct_dataset_has_disjoint_plate_identities(tmp_path):
    output = tmp_path / "synthetic"

    manifest = generate(
        output=output,
        train_plates=20,
        validation_plates=5,
        views_per_plate=2,
        font=_font(),
        font_license="DejaVu-font-license",
        seed=20260728,
        jpeg_quality=82,
    )

    train = _plate_texts(output / "train" / "annotations.csv")
    validation = _plate_texts(output / "val" / "annotations.csv")
    assert manifest["identity_overlap"] == 0
    assert train.isdisjoint(validation)
    assert manifest["train_images"] == 40
    assert manifest["validation_images"] == 10
    assert _dataset_contract(output)["manifest"]["golden_benchmark_data"] is False


def test_synthetic_generator_rejects_unapproved_font_license(tmp_path):
    with pytest.raises(ValueError, match="Font license is not approved"):
        generate(
            output=tmp_path / "synthetic",
            train_plates=20,
            validation_plates=5,
            views_per_plate=1,
            font=_font(),
            font_license="gpl-3.0",
            seed=20260728,
            jpeg_quality=82,
        )


def test_synthetic_plate_generation_balances_every_supported_letter():
    import random

    plates = _unique_plates(
        len(ALLOWED_PLATE_LETTERS),
        random.Random(20260728),
    )

    assert {plate[2] for plate in plates} == set(
        ALLOWED_PLATE_LETTERS
    )


def test_company_crop_import_rejects_golden_benchmark_rows(tmp_path):
    image = tmp_path / "plate.png"
    assert cv2.imwrite(
        str(image),
        np.zeros((32, 128, 3), dtype=np.uint8),
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate.png,31ط55674,track-1,bcvision-company-owned,golden\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Golden/benchmark"):
        _load_rows(source)


def test_company_crop_import_rejects_unapproved_license(tmp_path):
    image = tmp_path / "plate.png"
    assert cv2.imwrite(
        str(image),
        np.zeros((32, 128, 3), dtype=np.uint8),
    )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate.png,31ط55674,track-1,gpl-3.0,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unapproved source license"):
        _load_rows(source)


def test_company_crop_import_rejects_conflicting_track_labels(tmp_path):
    for index in range(2):
        image = tmp_path / f"plate-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), index, dtype=np.uint8),
        )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate-0.png,31ط55674,track-a,bcvision-company-owned,train\n"
        "plate-1.png,55ط63974,track-a,bcvision-company-owned,train\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="multiple plate identities"):
        _load_rows(source)


def test_training_contract_rejects_golden_manifest(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    (dataset / "dataset-license.json").write_text(
        json.dumps({
            "source_license": "bcvision-company-owned",
            "golden_benchmark_data": True,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Golden benchmark"):
        _dataset_contract(dataset)


def test_training_contract_rejects_synthetic_font_license(tmp_path):
    dataset = tmp_path / "dataset"
    (dataset / "train").mkdir(parents=True)
    (dataset / "val").mkdir()
    for split in ("train", "val"):
        (dataset / split / "annotations.csv").write_text(
            "image_path,plate_text\n",
            encoding="utf-8",
        )
    (dataset / "dataset-license.json").write_text(
        json.dumps({
            "source_license": "synthetic-bcvision-company-owned",
            "third_party_plate_dataset": False,
            "golden_benchmark_data": False,
            "font_license": "gpl-3.0",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="font license"):
        _dataset_contract(dataset)


def test_company_dataset_keeps_plate_identity_in_one_split(tmp_path):
    for index in range(4):
        image = tmp_path / f"plate-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), index, dtype=np.uint8),
        )
    source = tmp_path / "source.csv"
    source.write_text(
        "image_path,plate_text,group_id,source_license,usage\n"
        "plate-0.png,31ط55674,track-a,bcvision-company-owned,train\n"
        "plate-1.png,31ط55674,track-b,bcvision-company-owned,train\n"
        "plate-2.png,55ط63974,track-c,bcvision-company-owned,train\n"
        "plate-3.png,84ب57133,track-d,bcvision-company-owned,train\n",
        encoding="utf-8",
    )

    manifest = prepare(
        source,
        tmp_path / "prepared",
        validation_ratio=0.5,
        seed=20260728,
    )

    assert manifest["plate_identity_overlap"] == 0


class _FakeLayer:
    def __init__(self, name, weights):
        self.name = name
        self._weights = [np.asarray(weight) for weight in weights]

    def get_weights(self):
        return [weight.copy() for weight in self._weights]

    def set_weights(self, weights):
        self._weights = [np.asarray(weight).copy() for weight in weights]


class _FakeModel:
    def __init__(self, layers):
        self.layers = layers


def test_pretrained_transfer_copies_backbone_but_never_ocr_head():
    source = _FakeModel([
        _FakeLayer("backbone", [np.full((2, 2), 7)]),
        _FakeLayer("plate", [np.full((2, 3), 9)]),
    ])
    target = _FakeModel([
        _FakeLayer("backbone", [np.zeros((2, 2))]),
        _FakeLayer("plate", [np.zeros((2, 3))]),
    ])

    transferred = _copy_pretrained_backbone(source, target)

    assert transferred == ["backbone"]
    assert np.all(target.layers[0].get_weights()[0] == 7)
    assert np.all(target.layers[1].get_weights()[0] == 0)


def test_pretrained_transfer_rejects_different_backbone_shape():
    source = _FakeModel([
        _FakeLayer("backbone", [np.zeros((3, 2))]),
    ])
    target = _FakeModel([
        _FakeLayer("backbone", [np.zeros((2, 2))]),
    ])

    with pytest.raises(ValueError, match="does not match"):
        _copy_pretrained_backbone(source, target)


def test_training_rejects_pretrained_and_resume_together(tmp_path):
    from tools.train_fastplate_cct import train_and_export

    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    pretrained = tmp_path / "pretrained.keras"
    resume = tmp_path / "resume.keras"
    pretrained.write_bytes(b"pretrained")
    resume.write_bytes(b"resume")

    with pytest.raises(ValueError, match="either pretrained backbone"):
        train_and_export(
            dataset=dataset,
            output=output,
            variant="xs",
            pretrained_backbone=pretrained,
            resume_checkpoint=resume,
            checkpoint_metric="char",
            epochs=4,
            batch_size=4,
            seed=1,
        )
