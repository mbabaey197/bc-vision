"""Prepare licensed Iranian plate quadrilaterals for PP-YOLOE-R training."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np


ALLOWED_LICENSES = {
    "operator-confirmed-company-owned",
    "company-owned",
    "cc0",
    "cc-by-4.0",
}
SPLITS = {"train", "validation"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bool(value) -> bool:
    return str(value or "").strip().lower() in {
        "1", "true", "yes", "y",
    }


def _corners(row: dict, width: int, height: int) -> list[float]:
    values = []
    for index in range(1, 5):
        try:
            x = float(row[f"x{index}"])
            y = float(row[f"y{index}"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Positive rows require x1,y1,...,x4,y4"
            ) from exc
        if not (
            np.isfinite(x)
            and np.isfinite(y)
            and 0 <= x < width
            and 0 <= y < height
        ):
            raise ValueError("Plate corner is outside the source image")
        values.extend((round(x, 3), round(y, 3)))
    polygon = np.asarray(values, dtype=np.float32).reshape(4, 2)
    if abs(float(cv2.contourArea(polygon))) < 40.0:
        raise ValueError("Plate quadrilateral area is too small")
    return values


def prepare_dataset(csv_path: Path, output: Path) -> dict:
    csv_path = csv_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Detector annotation CSV is empty")

    images = {}
    annotations = {"train": [], "validation": []}
    digests_by_split = {"train": set(), "validation": set()}
    licenses = set()
    for line_number, row in enumerate(rows, start=2):
        split = str(row.get("split", "")).strip().lower()
        license_name = str(
            row.get("source_license", "")
        ).strip().lower()
        if split not in SPLITS:
            raise ValueError(
                f"Invalid detector split at line {line_number}"
            )
        if license_name not in ALLOWED_LICENSES:
            raise ValueError(
                f"Unapproved source license at line {line_number}"
            )
        if _bool(row.get("is_golden")):
            raise ValueError(
                f"Golden/benchmark frame cannot enter training at line "
                f"{line_number}"
            )
        source = (csv_path.parent / str(row.get("image_path", ""))).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(
                f"Unreadable detector image at line {line_number}"
            )
        height, width = image.shape[:2]
        digest = _sha256(source)
        other = "validation" if split == "train" else "train"
        if digest in digests_by_split[other]:
            raise ValueError(
                "The same source image cannot cross train/validation"
            )
        digests_by_split[split].add(digest)
        key = (split, digest)
        metadata = images.get(key)
        if metadata is None:
            suffix = source.suffix.lower()
            if suffix not in {".jpg", ".jpeg", ".png", ".bmp"}:
                suffix = ".jpg"
            metadata = {
                "id": 0,
                "file_name": f"{digest[:24]}{suffix}",
                "width": int(width),
                "height": int(height),
                "source": source,
                "digest": digest,
                "license": license_name,
                "negative": _bool(row.get("is_negative")),
            }
            images[key] = metadata
        elif metadata["negative"] != _bool(row.get("is_negative")):
            raise ValueError(
                "One image cannot be both negative and plate-labelled"
            )
        licenses.add(license_name)
        if metadata["negative"]:
            continue
        corners = _corners(row, width, height)
        polygon = np.asarray(corners, dtype=np.float32).reshape(4, 2)
        x1, y1 = polygon.min(axis=0)
        x2, y2 = polygon.max(axis=0)
        annotations[split].append({
            "image_key": key,
            "segmentation": [corners],
            "area": round(abs(float(cv2.contourArea(polygon))), 3),
            "bbox": [
                round(float(x1), 3),
                round(float(y1), 3),
                round(float(x2 - x1), 3),
                round(float(y2 - y1), 3),
            ],
            "category_id": 1,
            "iscrowd": 0,
        })
    if not annotations["train"] or not annotations["validation"]:
        raise ValueError(
            "Train and validation each require at least one labelled plate"
        )

    output.mkdir(parents=True)
    summary = {}
    for split in ("train", "validation"):
        split_dir = output / split
        image_dir = split_dir / "images"
        image_dir.mkdir(parents=True)
        selected = [
            metadata
            for (row_split, _), metadata in images.items()
            if row_split == split
        ]
        selected.sort(key=lambda row: row["digest"])
        image_ids = {}
        coco_images = []
        for image_id, metadata in enumerate(selected, start=1):
            metadata["id"] = image_id
            image_ids[(split, metadata["digest"])] = image_id
            shutil.copy2(
                metadata["source"],
                image_dir / metadata["file_name"],
            )
            coco_images.append({
                "id": image_id,
                "file_name": metadata["file_name"],
                "width": metadata["width"],
                "height": metadata["height"],
                "sha256": metadata["digest"],
                "source_license": metadata["license"],
                "hard_negative": metadata["negative"],
            })
        coco_annotations = []
        for annotation_id, annotation in enumerate(
            annotations[split],
            start=1,
        ):
            item = dict(annotation)
            key = item.pop("image_key")
            item.update({
                "id": annotation_id,
                "image_id": image_ids[key],
            })
            coco_annotations.append(item)
        payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "iran_plate"}],
        }
        (split_dir / "annotations.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary[split] = {
            "images": len(coco_images),
            "plates": len(coco_annotations),
            "hard_negatives": sum(
                bool(row["negative"]) for row in selected
            ),
        }

    manifest = {
        "schema": 1,
        "task": "bcvision-ppyoloe-r-iranian-plate-detector",
        "source_license": sorted(licenses),
        "golden_data_included": False,
        "identity_overlap": len(
            digests_by_split["train"]
            & digests_by_split["validation"]
        ),
        "splits": summary,
    }
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = prepare_dataset(args.annotations, args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
