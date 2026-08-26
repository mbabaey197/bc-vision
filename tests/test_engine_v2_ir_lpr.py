from __future__ import annotations

from pathlib import Path

import pytest

from app.engine_v2 import ir_lpr
from app.engine_v2.ir_lpr import load_ir_lpr


def _write_annotation(path: Path, image_name: str) -> None:
    labels = tuple("12") + ("ب",) + tuple("34567")
    objects = [
        (
            "<object><name>کل ناحیه پلاک</name><bndbox>"
            "<xmin>0</xmin><ymin>0</ymin><xmax>200</xmax><ymax>60</ymax>"
            "</bndbox></object>"
        )
    ]
    for index, label in enumerate(labels):
        xmin = 5 + index * 23
        xmax = xmin + 18
        objects.append(
            f"<object><name>{label}</name><bndbox>"
            f"<xmin>{xmin}</xmin><ymin>10</ymin>"
            f"<xmax>{xmax}</xmax><ymax>50</ymax>"
            "</bndbox></object>"
        )
    path.write_text(
        "<annotation>"
        f"<filename>{image_name}</filename>"
        "<size><width>200</width><height>60</height></size>"
        + "".join(objects)
        + "</annotation>",
        encoding="utf-8",
    )


def test_ir_lpr_pascal_voc_reader_reconstructs_plate_and_official_split(
    tmp_path: Path,
) -> None:
    split = tmp_path / "validation"
    split.mkdir()
    image = split / "sample.jpg"
    image.write_bytes(b"not-decoded-by-indexer")
    _write_annotation(split / "sample.xml", image.name)

    index = load_ir_lpr(tmp_path)

    assert len(index.samples) == 1
    sample = index.samples[0]
    assert sample.expected_plate == "12ب34567"
    assert sample.source_split == "validation"
    assert sample.calibration_split == "holdout"
    assert sample.plate_bbox.width == 200
    assert len(index.fingerprint_sha256) == 64


def test_ir_lpr_reader_can_record_invalid_annotations_without_silently_using_them(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train"
    split.mkdir()
    image = split / "good.jpg"
    image.write_bytes(b"good")
    _write_annotation(split / "good.xml", image.name)
    (split / "bad.jpg").write_bytes(b"bad")
    (split / "bad.xml").write_text("<annotation>", encoding="utf-8")

    index = load_ir_lpr(tmp_path, strict=False)

    assert len(index.samples) == 1
    assert index.skipped_annotations == (
        ("train/bad.xml", "no element found: line 1, column 12"),
    )


def test_ir_lpr_reader_finds_nested_annotations_and_sibling_images(
    tmp_path: Path,
) -> None:
    annotations = tmp_path / "validation" / "XML_val1"
    images = tmp_path / "validation" / "JPEGImages"
    annotations.mkdir(parents=True)
    images.mkdir(parents=True)
    image = images / "sample.jpg"
    image.write_bytes(b"indexed-by-name")
    _write_annotation(annotations / "sample.xml", image.name)

    index = load_ir_lpr(tmp_path)

    assert index.samples[0].image_path == image.resolve()
    assert index.samples[0].source_split == "validation"
    assert index.samples[0].calibration_split == "holdout"


def test_ir_lpr_reader_rejects_unknown_labels_instead_of_dropping_them(
    tmp_path: Path,
) -> None:
    split = tmp_path / "train"
    split.mkdir()
    image = split / "sample.jpg"
    image.write_bytes(b"image")
    annotation = split / "sample.xml"
    _write_annotation(annotation, image.name)
    contents = annotation.read_text(encoding="utf-8").replace(
        "</annotation>",
        "<object><name>mystery</name><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>2</xmax><ymax>2</ymax>"
        "</bndbox></object></annotation>",
    )
    annotation.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported IR-LPR object label"):
        load_ir_lpr(tmp_path)


def test_ir_lpr_reader_uses_dataset_root_name_as_official_split(
    tmp_path: Path,
) -> None:
    split = tmp_path / "testset"
    split.mkdir()
    image = split / "sample.jpg"
    image.write_bytes(b"image")
    _write_annotation(split / "sample.xml", image.name)

    index = load_ir_lpr(split)

    assert index.samples[0].source_split == "test"
    assert index.samples[0].calibration_split == "holdout"


def test_ir_lpr_reader_derives_missing_official_image_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "test"
    split.mkdir()
    image = split / "sample.jpg"
    image.write_bytes(b"image-size-is-read-through-the-helper")
    annotation = split / "sample.xml"
    _write_annotation(annotation, image.name)
    annotation.write_text(
        annotation.read_text(encoding="utf-8").replace(
            "<size><width>200</width><height>60</height></size>",
            "",
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ir_lpr, "_image_dimensions", lambda _path: (200, 60))

    index = load_ir_lpr(tmp_path)

    assert len(index.samples) == 1
    assert index.samples[0].image_width == 200
    assert index.samples[0].image_height == 60


def test_ir_lpr_reader_does_not_rescan_flat_directory_for_exact_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "train"
    split.mkdir()
    image = split / "sample.jpg"
    image.write_bytes(b"image")
    _write_annotation(split / "sample.xml", image.name)
    original_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == split:
            raise AssertionError("exact sibling lookup must not enumerate the directory")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)

    index = load_ir_lpr(tmp_path)

    assert len(index.samples) == 1
