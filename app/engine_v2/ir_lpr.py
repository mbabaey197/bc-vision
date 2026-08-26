"""Strict reader for the public IR-LPR Pascal-VOC style annotations."""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .validator import IranianPlateValidator

IR_LPR_SPLIT_MAP = {
    "train": "train",
    "validation": "holdout",
    "val": "holdout",
    "test": "holdout",
}
_PLATE_REGION_LABELS = frozenset(
    {
        "کل ناحیه پلاک",
        "plate",
        "license_plate",
        "license plate",
    }
)
_SPECIAL_CHARACTER_LABELS = {
    "ژ (معلولین و جانبازان)": "ژ",
    "معلولین": "ژ",
    "disabled": "ژ",
    "alef": "ا",
    "he": "ه",
    "ye": "ی",
    "D": "D",
    "S": "S",
}
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
_SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "trainset": "train",
    "val": "validation",
    "validation": "validation",
    "validationset": "validation",
    "valset": "validation",
    "test": "test",
    "testing": "test",
    "testset": "test",
}
_UNSUPPORTED_LABELS = frozenset({"تشریفات", "protocol", "taxi", "police"})


@dataclass(frozen=True, slots=True)
class IRLPRBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def center(self) -> tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def contains_center(self, other: IRLPRBox) -> bool:
        center_x, center_y = other.center
        return self.xmin <= center_x <= self.xmax and self.ymin <= center_y <= self.ymax


@dataclass(frozen=True, slots=True)
class IRLPRCharacter:
    label: str
    bbox: IRLPRBox


@dataclass(frozen=True, slots=True)
class IRLPRSample:
    sample_id: str
    source_split: str
    calibration_split: str
    image_path: Path
    annotation_path: Path
    image_width: int
    image_height: int
    plate_bbox: IRLPRBox
    expected_plate: str
    characters: tuple[IRLPRCharacter, ...]


@dataclass(frozen=True, slots=True)
class IRLPRIndex:
    root: Path
    samples: tuple[IRLPRSample, ...]
    skipped_annotations: tuple[tuple[str, str], ...]
    fingerprint_sha256: str


def load_ir_lpr(
    root: str | Path,
    *,
    strict: bool = True,
) -> IRLPRIndex:
    """Load IR-LPR JPG/XML pairs without changing the source dataset."""

    source_root = Path(root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    samples: list[IRLPRSample] = []
    skipped: list[tuple[str, str]] = []
    image_index = _index_images(source_root)
    annotation_paths = sorted(source_root.rglob("*.xml"))
    if not annotation_paths:
        raise ValueError(f"no IR-LPR XML annotations found under {source_root}")
    for annotation_path in annotation_paths:
        try:
            source_split = _source_split(annotation_path, source_root)
            parsed = read_ir_lpr_annotation(
                annotation_path,
                source_split=source_split,
                dataset_root=source_root,
                image_index=image_index,
            )
        except (FileNotFoundError, TypeError, ValueError, ET.ParseError) as exc:
            if strict:
                raise ValueError(
                    f"invalid IR-LPR annotation {annotation_path}: {exc}"
                ) from exc
            skipped.append((str(annotation_path.relative_to(source_root)), str(exc)))
            continue
        samples.extend(parsed)
    if not samples:
        raise ValueError(f"no usable IR-LPR XML/JPG pairs found under {source_root}")
    identifiers = [sample.sample_id for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("IR-LPR sample identifiers are not unique")
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.sample_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sample.expected_plate.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256(sample.image_path).encode("ascii"))
        digest.update(_sha256(sample.annotation_path).encode("ascii"))
    return IRLPRIndex(
        source_root,
        tuple(samples),
        tuple(skipped),
        digest.hexdigest(),
    )


def read_ir_lpr_annotation(
    annotation_path: str | Path,
    *,
    source_split: str,
    dataset_root: str | Path | None = None,
    image_index: Mapping[str, tuple[Path, ...]] | None = None,
) -> tuple[IRLPRSample, ...]:
    path = Path(annotation_path).resolve()
    root = ET.parse(path).getroot()
    image_path = _resolve_image(
        path,
        root.findtext("filename"),
        image_index=image_index,
        source_split=source_split,
        dataset_root=Path(dataset_root).resolve() if dataset_root is not None else None,
    )
    raw_width = root.findtext("size/width")
    raw_height = root.findtext("size/height")
    if raw_width is None and raw_height is None:
        # The official IR-LPR archives omit Pascal-VOC's optional ``size``
        # element.  Derive the dimensions from the paired image so bounding
        # boxes are still validated against real pixels instead of trusting
        # arbitrary coordinates from the annotation.
        image_width, image_height = _image_dimensions(image_path)
    elif raw_width is None or raw_height is None:
        raise ValueError("image size must contain both width and height")
    else:
        image_width = _positive_xml_integer(raw_width, "image width")
        image_height = _positive_xml_integer(raw_height, "image height")
    validator = IranianPlateValidator()
    regions: list[IRLPRBox] = []
    characters: list[IRLPRCharacter] = []
    for element in root.findall("object"):
        raw_label = str(element.findtext("name") or "").strip()
        if not raw_label:
            raise ValueError("object label cannot be empty")
        bbox = _parse_box(element.find("bndbox"), image_width, image_height)
        if raw_label in _PLATE_REGION_LABELS:
            regions.append(bbox)
            continue
        label = _canonical_character(raw_label, validator)
        characters.append(IRLPRCharacter(label, bbox))

    if not regions:
        if not characters:
            raise ValueError("annotation contains no plate region or plate characters")
        regions = [_union(character.bbox for character in characters)]

    relative = (
        path.relative_to(Path(dataset_root).resolve()).as_posix()
        if dataset_root is not None
        else path.name
    )
    output: list[IRLPRSample] = []
    for region_index, region in enumerate(regions):
        members = tuple(
            sorted(
                (
                    character
                    for character in characters
                    if region.contains_center(character.bbox)
                ),
                key=lambda character: character.bbox.center[0],
            )
        )
        if not members:
            continue
        direct = "".join(character.label for character in members)
        reverse = "".join(character.label for character in reversed(members))
        direct_validation = validator.validate(direct)
        reverse_validation = validator.validate(reverse)
        if direct_validation.valid:
            expected = direct_validation.normalized
            ordered = members
        elif reverse_validation.valid:
            expected = reverse_validation.normalized
            ordered = tuple(reversed(members))
        else:
            raise ValueError(
                f"character boxes do not form a supported Iranian plate: {direct!r}"
            )
        sample_seed = f"{relative}#{region_index}"
        sample_id = hashlib.sha256(sample_seed.encode("utf-8")).hexdigest()[:24]
        output.append(
            IRLPRSample(
                sample_id=sample_id,
                source_split=source_split,
                calibration_split=IR_LPR_SPLIT_MAP[source_split],
                image_path=image_path,
                annotation_path=path,
                image_width=image_width,
                image_height=image_height,
                plate_bbox=region,
                expected_plate=expected,
                characters=ordered,
            )
        )
    if not output:
        raise ValueError("annotation contains no labelled plate instance")
    return tuple(output)


def _image_dimensions(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
    except (ImportError, OSError, ValueError) as exc:
        raise ValueError(f"could not read image dimensions from {path.name}") from exc
    if width < 1 or height < 1:
        raise ValueError(f"invalid image dimensions in {path.name}")
    return int(width), int(height)


def _source_split(annotation_path: Path, root: Path) -> str:
    relative_parts = (root.name, *annotation_path.relative_to(root).parts[:-1])
    exact = {
        mapped
        for part in relative_parts
        if (mapped := _SPLIT_ALIASES.get(part.lower())) is not None
    }
    if len(exact) == 1:
        return exact.pop()
    if len(exact) > 1:
        raise ValueError("annotation path contains conflicting dataset splits")

    inferred: set[str] = set()
    for part in relative_parts:
        tokens = (
            part.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace(".", "_")
            .split("_")
        )
        for token in tokens:
            normalized = token.rstrip("0123456789")
            mapped = _SPLIT_ALIASES.get(normalized)
            if mapped is not None:
                inferred.add(mapped)
    if len(inferred) == 1:
        return inferred.pop()
    if len(inferred) > 1:
        raise ValueError("annotation path contains ambiguous dataset split tokens")
    # A root containing only one official archive is commonly passed directly.
    # It is safe to tune on it, but never to claim it as unseen holdout data.
    return "train"


def _index_images(root: Path) -> dict[str, tuple[Path, ...]]:
    indexed: dict[str, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        indexed.setdefault(path.name.lower(), []).append(path.resolve())
    return {name: tuple(paths) for name, paths in indexed.items()}


def _resolve_image(
    annotation_path: Path,
    filename: str | None,
    *,
    image_index: Mapping[str, tuple[Path, ...]] | None = None,
    source_split: str | None = None,
    dataset_root: Path | None = None,
) -> Path:
    candidates: list[Path] = []
    if filename:
        candidates.append(annotation_path.parent / Path(filename).name)
    candidates.extend(annotation_path.with_suffix(suffix) for suffix in _IMAGE_SUFFIXES)
    # Official IR-LPR archives are large, flat directories.  Resolve the
    # overwhelmingly common exact sibling before constructing a directory
    # index; doing the reverse rescans every sibling for every annotation and
    # turns dataset loading into an O(n^2) operation.
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    by_lower_name = {
        child.name.lower(): child
        for child in annotation_path.parent.iterdir()
        if child.is_file()
    }
    for candidate in candidates:
        matched = by_lower_name.get(candidate.name.lower())
        if matched is not None:
            return matched.resolve()
    if image_index:
        matches: list[Path] = []
        names = ([Path(filename).name] if filename else []) + [
            annotation_path.with_suffix(suffix).name for suffix in _IMAGE_SUFFIXES
        ]
        for name in names:
            matches.extend(image_index.get(name.lower(), ()))
        matches = sorted(set(matches))
        if source_split is not None and dataset_root is not None:
            same_split = [
                match
                for match in matches
                if _source_split(match, dataset_root) == source_split
            ]
            if same_split:
                matches = same_split
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"multiple image pairs found for {annotation_path.name}: "
                f"{[str(match) for match in matches]}"
            )
    raise FileNotFoundError(f"image pair for {annotation_path.name} was not found")


def _canonical_character(raw_label: str, validator: IranianPlateValidator) -> str:
    if raw_label in _UNSUPPORTED_LABELS:
        raise ValueError(f"unsupported IR-LPR plate class label: {raw_label!r}")
    special = _SPECIAL_CHARACTER_LABELS.get(raw_label)
    if special is not None:
        return special
    normalized = validator.normalize(raw_label)
    if len(normalized) == 1 and (
        normalized.isascii()
        and normalized.isdigit()
        or normalized in validator.config.allowed_letters
    ):
        return normalized
    raise ValueError(f"unsupported IR-LPR object label: {raw_label!r}")


def _parse_box(
    element: ET.Element | None, image_width: int, image_height: int
) -> IRLPRBox:
    if element is None:
        raise ValueError("object bndbox is missing")
    xmin = _xml_integer(element.findtext("xmin"), "xmin")
    ymin = _xml_integer(element.findtext("ymin"), "ymin")
    xmax = _xml_integer(element.findtext("xmax"), "xmax")
    ymax = _xml_integer(element.findtext("ymax"), "ymax")
    if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
        raise ValueError("object bndbox is outside the declared image dimensions")
    return IRLPRBox(xmin, ymin, xmax, ymax)


def _union(boxes: Iterable[IRLPRBox]) -> IRLPRBox:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot build a plate region from no character boxes")
    return IRLPRBox(
        min(box.xmin for box in values),
        min(box.ymin for box in values),
        max(box.xmax for box in values),
        max(box.ymax for box in values),
    )


def _positive_xml_integer(value: str | None, name: str) -> int:
    number = _xml_integer(value, name)
    if number < 1:
        raise ValueError(f"{name} must be positive")
    return number


def _xml_integer(value: str | None, name: str) -> int:
    try:
        number = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "IR_LPR_SPLIT_MAP",
    "IRLPRBox",
    "IRLPRCharacter",
    "IRLPRIndex",
    "IRLPRSample",
    "load_ir_lpr",
    "read_ir_lpr_annotation",
]
