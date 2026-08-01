import csv
import json

import cv2
import numpy as np
import pytest

from tools.prepare_ppyoloe_r_dataset import prepare_dataset


FIELDS = [
    "image_path",
    "split",
    "source_license",
    "is_golden",
    "is_negative",
    "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
]


def _image(path, value=120):
    image = np.full((120, 240, 3), value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def _csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(image, split, negative=False, golden=False):
    return {
        "image_path": image,
        "split": split,
        "source_license": "operator-confirmed-company-owned",
        "is_golden": int(golden),
        "is_negative": int(negative),
        "x1": "" if negative else 40,
        "y1": "" if negative else 45,
        "x2": "" if negative else 200,
        "y2": "" if negative else 40,
        "x3": "" if negative else 205,
        "y3": "" if negative else 82,
        "x4": "" if negative else 42,
        "y4": "" if negative else 88,
    }


def test_detector_dataset_keeps_plates_and_hard_negatives(tmp_path):
    _image(tmp_path / "train.jpg", 100)
    _image(tmp_path / "negative.jpg", 20)
    _image(tmp_path / "validation.jpg", 180)
    annotations = tmp_path / "annotations.csv"
    _csv(annotations, [
        _row("train.jpg", "train"),
        _row("negative.jpg", "train", negative=True),
        _row("validation.jpg", "validation"),
    ])

    result = prepare_dataset(annotations, tmp_path / "output")

    assert result["identity_overlap"] == 0
    assert result["splits"]["train"] == {
        "images": 2,
        "plates": 1,
        "hard_negatives": 1,
    }
    payload = json.loads(
        (tmp_path / "output" / "train" / "annotations.json").read_text(
            encoding="utf-8",
        )
    )
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == 1
    assert payload["categories"] == [{"id": 1, "name": "iran_plate"}]


def test_detector_dataset_rejects_golden_frame(tmp_path):
    _image(tmp_path / "golden.jpg")
    annotations = tmp_path / "annotations.csv"
    _csv(annotations, [_row("golden.jpg", "train", golden=True)])

    with pytest.raises(ValueError, match="Golden"):
        prepare_dataset(annotations, tmp_path / "output")


def test_detector_dataset_rejects_split_leak(tmp_path):
    _image(tmp_path / "same.jpg")
    annotations = tmp_path / "annotations.csv"
    _csv(annotations, [
        _row("same.jpg", "train"),
        _row("same.jpg", "validation"),
    ])

    with pytest.raises(ValueError, match="cannot cross"):
        prepare_dataset(annotations, tmp_path / "output")
