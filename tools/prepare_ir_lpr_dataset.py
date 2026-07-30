"""Prepare the official IR-LPR splits for BC Vision research training.

IR-LPR is distributed from https://github.com/mut-deep/IR-LPR under
GPL-3.0. This importer marks every derived dataset as research-only and
non-distributable. It never accepts BC Vision Golden data, never invents
missing OCR labels, and removes image/plate identity overlap from later
validation/test splits.

The official archives use LabelImg/VOC XML annotations. The importer accepts
either an extracted directory or a ZIP archive for each split.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
import zipfile

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.plate_rules import normalize_plate, plausible_plate


SOURCE_REPOSITORY = "https://github.com/mut-deep/IR-LPR"
SOURCE_LICENSE = "gpl-3.0-ir-lpr-research-only"
SOURCE_LICENSE_GIT_BLOB_SHA = (
    "f288702d2fa16d3cdf0035b15a9fcbc552cd88e7"
)
SPLIT_ORDER = ("train", "validation", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PLATE_OBJECTS = {
    "carplate",
    "iranplate",
    "licenseplate",
    "lp",
    "numberplate",
    "plate",
}

# IR-LPR Table 2 publishes these Latin equivalents for Persian classes.
# D and S are interpreted as د and س for the standard eight-slot layout;
# diplomatic/service layouts are non-standard and rejected by plausible_plate.
CHARACTER_ALIASES = {
    **{str(value): str(value) for value in range(10)},
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "a": "ا",
    "alef": "ا",
    "alf": "ا",
    "b": "ب",
    "be": "ب",
    "beh": "ب",
    "p": "پ",
    "pe": "پ",
    "peh": "پ",
    "t": "ت",
    "te": "ت",
    "teh": "ت",
    "th": "ث",
    "the": "ث",
    "theh": "ث",
    "j": "ج",
    "jeem": "ج",
    "jim": "ج",
    "d": "د",
    "dal": "د",
    "z": "ز",
    "zal": "ز",
    "ze": "ز",
    "zh": "ژ",
    "zhe": "ژ",
    "zheh": "ژ",
    "s": "س",
    "seen": "س",
    "sin": "س",
    "sh": "ش",
    "sheen": "ش",
    "shin": "ش",
    "sad": "ص",
    "saad": "ص",
    "ta": "ط",
    "taa": "ط",
    "tah": "ط",
    "o": "ع",
    "ain": "ع",
    "ayn": "ع",
    "ein": "ع",
    "f": "ف",
    "fa": "ف",
    "fe": "ف",
    "q": "ق",
    "qaf": "ق",
    "ghaf": "ق",
    "k": "ک",
    "kaf": "ک",
    "g": "گ",
    "gaf": "گ",
    "l": "ل",
    "lam": "ل",
    "m": "م",
    "meem": "م",
    "mim": "م",
    "n": "ن",
    "noon": "ن",
    "non": "ن",
    "v": "و",
    "w": "و",
    "vav": "و",
    "waw": "و",
    "h": "ه",
    "he": "ه",
    "heh": "ه",
    "y": "ی",
    "ya": "ی",
    "ye": "ی",
    "yeh": "ی",
}

VERBOSE_CHARACTER_ALIASES = {
    "ژمعلولینوجانبازان": "ژ",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _ascii_token(value: str) -> str:
    text = str(value or "").strip().lower()
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(
        character
        for character in normalized
        if character.isascii() and character.isalnum()
    )


def _character(value: str) -> str:
    normalized = normalize_plate(value)
    if len(normalized) == 1 and (
        normalized.isdigit()
        or "\u0600" <= normalized <= "\u06ff"
    ):
        return normalized
    if normalized in VERBOSE_CHARACTER_ALIASES:
        return VERBOSE_CHARACTER_ALIASES[normalized]
    return CHARACTER_ALIASES.get(_ascii_token(value), "")


def _is_plate_object(value: str) -> bool:
    return _ascii_token(value) in PLATE_OBJECTS


def _safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError("IR-LPR archive contains a symbolic link")
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError as exc:
                raise ValueError(
                    "IR-LPR archive contains an unsafe path"
                ) from exc
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _materialize(
    source: Path,
    stack: ExitStack,
    label: str,
) -> Path:
    source = source.resolve()
    if source.is_dir():
        return source
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(
            f"IR-LPR {label} must be an extracted directory or ZIP archive"
        )
    temporary = Path(stack.enter_context(
        tempfile.TemporaryDirectory(prefix=f"bcvision-ir-lpr-{label}-")
    ))
    _safe_extract(source, temporary)
    return temporary


def _image_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem.casefold(), []).append(path)
    return index


def _find_image(
    xml_path: Path,
    root: ET.Element,
    index: dict[str, list[Path]],
) -> Path:
    filename = str(root.findtext("filename") or "").strip()
    path_text = str(root.findtext("path") or "").strip()
    candidates = []
    if filename:
        candidates.append(xml_path.parent / Path(filename).name)
    if path_text:
        candidates.append(xml_path.parent / Path(path_text).name)
    for suffix in IMAGE_SUFFIXES:
        candidates.append(xml_path.with_suffix(suffix))
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    stem = Path(filename).stem if filename else xml_path.stem
    matches = index.get(stem.casefold(), [])
    if len(matches) != 1:
        raise ValueError(
            f"Cannot resolve one image for IR-LPR annotation: {xml_path}"
        )
    return matches[0].resolve()


def _objects(root: ET.Element) -> list[dict]:
    rows = []
    for item in root.findall("object"):
        name = str(item.findtext("name") or "").strip()
        box = item.find("bndbox")
        if not name or box is None:
            continue
        try:
            x1 = float(box.findtext("xmin"))
            y1 = float(box.findtext("ymin"))
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        rows.append({
            "name": name,
            "box": (x1, y1, x2, y2),
            "character": _character(name),
        })
    return rows


def _plate_text(objects: list[dict]) -> str:
    characters = [
        row
        for row in objects
        if row["character"] and not _is_plate_object(row["name"])
    ]
    characters.sort(
        key=lambda row: (
            (row["box"][0] + row["box"][2]) / 2,
            row["box"][1],
        )
    )
    plate = normalize_plate(
        "".join(row["character"] for row in characters)
    )
    return plate if plausible_plate(plate) else ""


def _plate_box(objects: list[dict]) -> tuple[float, float, float, float] | None:
    labelled = [
        row["box"]
        for row in objects
        if _is_plate_object(row["name"])
    ]
    if labelled:
        return max(
            labelled,
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
    characters = [
        row["box"] for row in objects if row["character"]
    ]
    if len(characters) < 7:
        return None
    return (
        min(box[0] for box in characters),
        min(box[1] for box in characters),
        max(box[2] for box in characters),
        max(box[3] for box in characters),
    )


def _annotation_rows(root: Path) -> list[dict]:
    index = _image_index(root)
    xml_files = sorted(root.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"IR-LPR split has no VOC XML annotations: {root}")
    rows = []
    for xml_path in xml_files:
        parsed = ET.parse(xml_path).getroot()
        image = _find_image(xml_path, parsed, index)
        objects = _objects(parsed)
        rows.append({
            "xml": xml_path,
            "image": image,
            "objects": objects,
            "plate_text": _plate_text(objects),
            "plate_box": _plate_box(objects),
        })
    return rows


def _copy_image(source: Path, target: Path) -> tuple[int, int]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"Unreadable IR-LPR image: {source}")
    height, width = image.shape[:2]
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return int(width), int(height)


def _deduplicate_splits(rows_by_split: dict[str, list[dict]]) -> tuple[dict, dict]:
    earlier_digests: dict[str, str] = {}
    earlier_identities: dict[str, str] = {}
    selected = {split: [] for split in SPLIT_ORDER}
    excluded = {
        split: {"duplicate_image": 0, "plate_identity_overlap": 0}
        for split in SPLIT_ORDER
    }
    for split in SPLIT_ORDER:
        split_digests = set()
        split_identities = set()
        for row in rows_by_split[split]:
            digest = _sha256(row["image"])
            plate = row["plate_text"]
            if digest in earlier_digests or digest in split_digests:
                excluded[split]["duplicate_image"] += 1
                continue
            if plate and plate in earlier_identities:
                excluded[split]["plate_identity_overlap"] += 1
                continue
            row = {**row, "sha256": digest}
            selected[split].append(row)
            split_digests.add(digest)
            if plate:
                split_identities.add(plate)
        earlier_digests.update({
            digest: split for digest in split_digests
        })
        earlier_identities.update({
            plate: split for plate in split_identities
        })
    return selected, excluded


def _prepare_ocr(
    roots: dict[str, Path],
    output: Path,
) -> dict:
    raw = {
        split: [
            row
            for row in _annotation_rows(roots[split])
            if row["plate_text"]
        ]
        for split in SPLIT_ORDER
    }
    selected, excluded = _deduplicate_splits(raw)
    counts = {}
    identities = {}
    for split in SPLIT_ORDER:
        directory_name = "val" if split == "validation" else split
        split_dir = output / directory_name
        images_dir = split_dir / "images"
        annotations = []
        for index, row in enumerate(selected[split]):
            suffix = row["image"].suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".jpg"
            filename = f"{index:07d}-{row['sha256'][:12]}{suffix}"
            _copy_image(row["image"], images_dir / filename)
            annotations.append({
                "image_path": f"images/{filename}",
                "plate_text": row["plate_text"],
            })
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "annotations.csv").open(
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
        counts[split] = len(annotations)
        identities[split] = {
            row["plate_text"] for row in selected[split]
        }
    if not all(counts.values()):
        raise ValueError(
            "IR-LPR OCR train, validation and test each require a usable "
            "standard Iranian plate"
        )
    manifest = {
        "schema": 1,
        "task": "bcvision-ir-lpr-cct-ocr-research",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
        "source_license_git_blob_sha": SOURCE_LICENSE_GIT_BLOB_SHA,
        "research_only": True,
        "distribution_allowed": False,
        "activation_policy": "shadow-only",
        "golden_benchmark_data": False,
        "official_test_split": True,
        "split_samples": counts,
        "excluded": excluded,
        "plate_identity_overlap": sum(
            len(identities[left] & identities[right])
            for index, left in enumerate(SPLIT_ORDER)
            for right in SPLIT_ORDER[index + 1:]
        ),
    }
    (output / "dataset-license.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def _prepare_detector(
    roots: dict[str, Path],
    output: Path,
) -> dict:
    raw = {
        split: [
            row
            for row in _annotation_rows(roots[split])
            if row["plate_box"] is not None
        ]
        for split in SPLIT_ORDER
    }
    selected, excluded = _deduplicate_splits(raw)
    counts = {}
    for split in SPLIT_ORDER:
        split_dir = output / split
        images_dir = split_dir / "images"
        coco_images = []
        coco_annotations = []
        for image_id, row in enumerate(selected[split], start=1):
            suffix = row["image"].suffix.lower()
            if suffix not in IMAGE_SUFFIXES:
                suffix = ".jpg"
            filename = f"{image_id:07d}-{row['sha256'][:12]}{suffix}"
            width, height = _copy_image(
                row["image"],
                images_dir / filename,
            )
            x1, y1, x2, y2 = row["plate_box"]
            x1 = max(0.0, min(float(width - 1), x1))
            y1 = max(0.0, min(float(height - 1), y1))
            x2 = max(x1 + 1.0, min(float(width), x2))
            y2 = max(y1 + 1.0, min(float(height), y2))
            polygon = [x1, y1, x2, y1, x2, y2, x1, y2]
            coco_images.append({
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
                "sha256": row["sha256"],
                "source_license": SOURCE_LICENSE,
            })
            coco_annotations.append({
                "id": image_id,
                "image_id": image_id,
                "segmentation": [polygon],
                "area": round((x2 - x1) * (y2 - y1), 3),
                "bbox": [
                    round(x1, 3),
                    round(y1, 3),
                    round(x2 - x1, 3),
                    round(y2 - y1, 3),
                ],
                "category_id": 1,
                "iscrowd": 0,
            })
        split_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "iran_plate"}],
        }
        (split_dir / "annotations.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        counts[split] = {
            "images": len(coco_images),
            "plates": len(coco_annotations),
        }
    if not all(row["plates"] for row in counts.values()):
        raise ValueError(
            "IR-LPR detector train, validation and test each require a "
            "labelled plate"
        )
    manifest = {
        "schema": 1,
        "task": "bcvision-ir-lpr-ppyoloe-r-detector-research",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_repository": SOURCE_REPOSITORY,
        "source_license": SOURCE_LICENSE,
        "source_license_git_blob_sha": SOURCE_LICENSE_GIT_BLOB_SHA,
        "research_only": True,
        "distribution_allowed": False,
        "activation_policy": "shadow-only",
        "golden_data_included": False,
        "official_test_split": True,
        "annotation_geometry": (
            "axis-aligned VOC boxes represented as zero-rotation polygons"
        ),
        "splits": counts,
        "excluded": excluded,
    }
    (output / "dataset-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def prepare_ir_lpr(
    *,
    plate_sources: dict[str, Path],
    car_sources: dict[str, Path] | None,
    output: Path,
    accept_gpl_research_only: bool,
) -> dict:
    if not accept_gpl_research_only:
        raise ValueError(
            "IR-LPR requires explicit GPL-3.0 research-only acceptance"
        )
    if set(plate_sources) != set(SPLIT_ORDER):
        raise ValueError("All three IR-LPR plate splits are required")
    if car_sources is not None and set(car_sources) != set(SPLIT_ORDER):
        raise ValueError("All three IR-LPR car splits are required")
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    with ExitStack() as stack:
        plates = {
            split: _materialize(
                plate_sources[split],
                stack,
                f"plate-{split}",
            )
            for split in SPLIT_ORDER
        }
        cars = (
            {
                split: _materialize(
                    car_sources[split],
                    stack,
                    f"car-{split}",
                )
                for split in SPLIT_ORDER
            }
            if car_sources is not None
            else None
        )
        output.mkdir(parents=True)
        result = {
            "ocr": _prepare_ocr(plates, output / "ocr"),
            "detector": (
                _prepare_detector(cars, output / "detector")
                if cars is not None
                else None
            ),
        }
    (output / "ir-lpr-import-report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare official IR-LPR data for non-distributable BC Vision "
            "Shadow research"
        ),
    )
    for prefix in ("plate", "car"):
        for split in SPLIT_ORDER:
            parser.add_argument(
                f"--{prefix}-{split}",
                type=Path,
                required=prefix == "plate",
            )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--accept-gpl-3.0-research-only",
        dest="accept_gpl_3_0_research_only",
        action="store_true",
    )
    args = parser.parse_args(argv)
    car_values = {
        split: getattr(args, f"car_{split}")
        for split in SPLIT_ORDER
    }
    any_car = any(car_values.values())
    if any_car and not all(car_values.values()):
        parser.error("Either provide all three car splits or none")
    result = prepare_ir_lpr(
        plate_sources={
            split: getattr(args, f"plate_{split}")
            for split in SPLIT_ORDER
        },
        car_sources=car_values if any_car else None,
        output=args.output,
        accept_gpl_research_only=(
            args.accept_gpl_3_0_research_only
        ),
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
