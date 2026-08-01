import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import cv2
import numpy as np
import pytest

from tools.prepare_ir_lpr_dataset import main, prepare_ir_lpr
from tools.train_fastplate_cct import _dataset_contract


def _write_fixture(root: Path, name: str, labels: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    image_path = root / f"{name}.jpg"
    value = 80 + sum(ord(character) for character in name) % 140
    image = np.full((100, 320, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "filename").text = image_path.name
    plate = ET.SubElement(annotation, "object")
    ET.SubElement(plate, "name").text = "license_plate"
    plate_box = ET.SubElement(plate, "bndbox")
    for key, value in {
        "xmin": 12,
        "ymin": 20,
        "xmax": 308,
        "ymax": 82,
    }.items():
        ET.SubElement(plate_box, key).text = str(value)
    for index, label in enumerate(labels):
        item = ET.SubElement(annotation, "object")
        ET.SubElement(item, "name").text = label
        box = ET.SubElement(item, "bndbox")
        left = 22 + index * 34
        for key, value in {
            "xmin": left,
            "ymin": 30,
            "xmax": left + 22,
            "ymax": 72,
        }.items():
            ET.SubElement(box, key).text = str(value)
    ET.ElementTree(annotation).write(
        root / f"{name}.xml",
        encoding="utf-8",
        xml_declaration=True,
    )


def _sources(tmp_path: Path) -> dict[str, Path]:
    values = {
        "train": ["3", "1", "T", "5", "5", "6", "7", "4"],
        "validation": ["5", "5", "T", "6", "3", "9", "7", "4"],
        "test": ["8", "4", "B", "5", "7", "1", "3", "3"],
    }
    sources = {}
    for split, labels in values.items():
        root = tmp_path / split
        _write_fixture(root, f"{split}-plate", labels)
        sources[split] = root
    return sources


def test_ir_lpr_requires_explicit_research_acceptance(tmp_path):
    sources = _sources(tmp_path)
    with pytest.raises(ValueError, match="explicit GPL-3.0"):
        prepare_ir_lpr(
            plate_sources=sources,
            car_sources=None,
            output=tmp_path / "prepared",
            accept_gpl_research_only=False,
        )


def test_ir_lpr_prepares_isolated_ocr_and_detector_splits(tmp_path):
    sources = _sources(tmp_path)
    result = prepare_ir_lpr(
        plate_sources=sources,
        car_sources=sources,
        output=tmp_path / "prepared",
        accept_gpl_research_only=True,
    )

    assert result["ocr"]["split_samples"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    assert result["ocr"]["plate_identity_overlap"] == 0
    assert result["ocr"]["distribution_allowed"] is False
    assert result["ocr"]["activation_policy"] == "shadow-only"
    assert result["detector"]["splits"]["test"]["plates"] == 1
    validation = (
        tmp_path
        / "prepared"
        / "ocr"
        / "val"
        / "annotations.csv"
    )
    with validation.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["plate_text"] == "55ت63974"


def test_ir_lpr_excludes_plate_identity_leak_from_later_split(tmp_path):
    sources = _sources(tmp_path)
    _write_fixture(
        sources["validation"],
        "validation-leak",
        ["3", "1", "T", "5", "5", "6", "7", "4"],
    )
    result = prepare_ir_lpr(
        plate_sources=sources,
        car_sources=None,
        output=tmp_path / "prepared",
        accept_gpl_research_only=True,
    )

    excluded = result["ocr"]["excluded"]["validation"]
    assert excluded["plate_identity_overlap"] == 1
    assert result["ocr"]["split_samples"]["validation"] == 1


def test_ir_lpr_keeps_distinct_views_of_identity_within_train(tmp_path):
    sources = _sources(tmp_path)
    _write_fixture(
        sources["train"],
        "second-train-view",
        ["3", "1", "T", "5", "5", "6", "7", "4"],
    )
    result = prepare_ir_lpr(
        plate_sources=sources,
        car_sources=None,
        output=tmp_path / "prepared",
        accept_gpl_research_only=True,
    )

    assert result["ocr"]["split_samples"]["train"] == 2
    assert (
        result["ocr"]["excluded"]["train"]["plate_identity_overlap"]
        == 0
    )


def test_ir_lpr_maps_verbose_disabled_veteran_label():
    from tools.prepare_ir_lpr_dataset import _character

    assert _character("ژ (معلولین و جانبازان)") == "ژ"


def test_ir_lpr_accepts_official_split_zip_archives(tmp_path):
    sources = _sources(tmp_path / "sources")
    archives = {}
    for split, source in sources.items():
        archive = tmp_path / f"plate-{split}.zip"
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in source.iterdir():
                bundle.write(path, arcname=f"nested/{path.name}")
        archives[split] = archive

    result = prepare_ir_lpr(
        plate_sources=archives,
        car_sources=None,
        output=tmp_path / "prepared",
        accept_gpl_research_only=True,
    )

    assert result["ocr"]["split_samples"]["test"] == 1


def test_ir_lpr_cli_accepts_research_only_flag(tmp_path, capsys):
    sources = _sources(tmp_path / "sources")

    assert main([
        "--plate-train",
        str(sources["train"]),
        "--plate-validation",
        str(sources["validation"]),
        "--plate-test",
        str(sources["test"]),
        "--output",
        str(tmp_path / "prepared"),
        "--accept-gpl-3.0-research-only",
    ]) == 0

    assert '"activation_policy": "shadow-only"' in capsys.readouterr().out


def test_ir_lpr_cct_contract_is_shadow_only(tmp_path):
    sources = _sources(tmp_path / "sources")
    prepared = tmp_path / "prepared"
    result = prepare_ir_lpr(
        plate_sources=sources,
        car_sources=None,
        output=prepared,
        accept_gpl_research_only=True,
    )
    dataset = prepared / "ocr"
    manifest = result["ocr"]

    contract = _dataset_contract(dataset)
    assert contract["research_only"] is True
    assert contract["test"].name == "annotations.csv"

    manifest["distribution_allowed"] = True
    (dataset / "dataset-license.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-distributable"):
        _dataset_contract(dataset)
