"""Generate a commercially safe synthetic Iranian-plate OCR dataset.

This tool does not download or mix any third-party plate dataset.  Generated
images are grouped by plate identity and train/validation identities are
disjoint, preventing near-duplicate leakage between the two splits.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.plate_rules import ALLOWED_PLATE_LETTERS


GENERATOR_SCHEMA = 1
DEFAULT_SEED = 20260728
ALLOWED_FONT_LICENSES = {
    "apache-2.0",
    "bcvision-company-owned",
    "bsd-3-clause",
    "cc0-1.0",
    "dejavu-font-license",
    "ofl-1.1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _default_font() -> Path:
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        Path(os.environ.get("WINDIR", "C:/Windows"))
        / "Fonts"
        / "tahomabd.ttf",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No Persian-capable font found; pass --font explicitly"
    )


def random_plate(
    rng: random.Random,
    letter: str | None = None,
) -> str:
    return (
        f"{rng.randrange(10, 100):02d}"
        + (letter or rng.choice(ALLOWED_PLATE_LETTERS))
        + f"{rng.randrange(0, 1000):03d}"
        + f"{rng.randrange(10, 100):02d}"
    )


def _unique_plates(count: int, rng: random.Random, excluded=None) -> list[str]:
    values = set(excluded or ())
    selected = []
    letters = list(ALLOWED_PLATE_LETTERS)
    rng.shuffle(letters)
    letter_index = 0
    while len(selected) < count:
        plate = random_plate(
            rng,
            letter=letters[letter_index % len(letters)],
        )
        letter_index += 1
        if plate in values:
            continue
        values.add(plate)
        selected.append(plate)
    return selected


def _centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
    fill,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2 - bounds[0]
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=fill)


def _base_plate(
    plate: str,
    font_path: Path,
    rng: random.Random,
) -> Image.Image:
    width, height = 280, 74
    background = rng.randint(238, 255)
    image = Image.new("RGB", (width, height), (background,) * 3)
    draw = ImageDraw.Draw(image)
    border = rng.randint(1, 4)
    draw.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=5,
        outline=(15, 18, 22),
        width=border,
    )
    blue_width = rng.randint(25, 31)
    draw.rectangle(
        (border, border, blue_width, height - border),
        fill=(7, rng.randint(75, 105), rng.randint(155, 205)),
    )
    small_font = ImageFont.truetype(str(font_path), 9)
    main_font = ImageFont.truetype(
        str(font_path),
        rng.randint(33, 39),
    )
    letter_font = ImageFont.truetype(
        str(font_path),
        rng.randint(31, 37),
    )
    _centered_text(
        draw,
        (1, 4, blue_width, 31),
        "IR",
        small_font,
        (255, 255, 255),
    )
    _centered_text(
        draw,
        (1, 29, blue_width, height - 4),
        "I.R.",
        small_font,
        (255, 255, 255),
    )

    region_left = width - rng.randint(49, 55)
    draw.line(
        (region_left, border, region_left, height - border),
        fill=(18, 20, 24),
        width=2,
    )
    _centered_text(
        draw,
        (region_left, 3, width - 3, 25),
        "IRAN",
        small_font,
        (15, 18, 22),
    )
    _centered_text(
        draw,
        (region_left, 20, width - 3, height - 3),
        plate[6:8],
        main_font,
        (12, 14, 18),
    )

    content_left = blue_width + 3
    content_right = region_left - 2
    spans = [
        (content_left, content_left + 62, plate[:2], main_font),
        (
            content_left + 60,
            content_left + 103,
            plate[2],
            letter_font,
        ),
        (
            content_left + 99,
            content_right,
            plate[3:6],
            main_font,
        ),
    ]
    for left, right, text, font in spans:
        _centered_text(
            draw,
            (left, 4, right, height - 4),
            text,
            font,
            (rng.randint(5, 22),) * 3,
        )
    return image


def _degrade(image: Image.Image, rng: random.Random) -> np.ndarray:
    array = np.asarray(image)
    height, width = array.shape[:2]
    margin_x = rng.randint(2, 12)
    margin_y = rng.randint(1, 5)
    source = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    target = np.float32([
        [rng.randint(0, margin_x), rng.randint(0, margin_y)],
        [width - 1 - rng.randint(0, margin_x), rng.randint(0, margin_y)],
        [
            width - 1 - rng.randint(0, margin_x),
            height - 1 - rng.randint(0, margin_y),
        ],
        [rng.randint(0, margin_x), height - 1 - rng.randint(0, margin_y)],
    ])
    transform = cv2.getPerspectiveTransform(source, target)
    array = cv2.warpPerspective(
        array,
        transform,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )
    result = Image.fromarray(array)
    result = ImageEnhance.Brightness(result).enhance(
        rng.uniform(0.58, 1.35)
    )
    result = ImageEnhance.Contrast(result).enhance(
        rng.uniform(0.68, 1.45)
    )
    if rng.random() < 0.55:
        result = result.filter(
            ImageFilter.GaussianBlur(rng.uniform(0.2, 1.25))
        )
    array = np.asarray(result).astype(np.float32)
    if rng.random() < 0.60:
        noise = np.random.default_rng(
            rng.randrange(0, 2 ** 32)
        ).normal(0, rng.uniform(1.5, 9.0), array.shape)
        array += noise
    array = np.clip(array, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        scale = rng.uniform(0.42, 0.78)
        small = cv2.resize(
            array,
            (max(32, int(width * scale)), max(14, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        array = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
    if rng.random() < 0.28:
        x = rng.randrange(20, width - 24)
        stripe_width = rng.randint(1, 4)
        opacity = rng.uniform(0.08, 0.25)
        overlay = array.copy()
        cv2.rectangle(
            overlay,
            (x, 3),
            (x + stripe_width, height - 4),
            (255, 255, 255),
            -1,
        )
        array = cv2.addWeighted(
            overlay,
            opacity,
            array,
            1.0 - opacity,
            0,
        )
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)


def _write_split(
    root: Path,
    name: str,
    plates: list[str],
    views_per_plate: int,
    font: Path,
    rng: random.Random,
    jpeg_quality: int,
) -> int:
    split = root / name
    images = split / "images"
    images.mkdir(parents=True, exist_ok=False)
    rows = []
    counter = 0
    for group_index, plate in enumerate(plates):
        for view in range(views_per_plate):
            image = _degrade(_base_plate(plate, font, rng), rng)
            filename = f"{group_index:06d}-{view:02d}.jpg"
            target = images / filename
            ok, payload = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
            )
            if not ok:
                raise RuntimeError(f"Failed to encode {target}")
            target.write_bytes(payload.tobytes())
            rows.append({
                "image_path": f"images/{filename}",
                "plate_text": plate,
            })
            counter += 1
    with (split / "annotations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", "plate_text"],
        )
        writer.writeheader()
        writer.writerows(rows)
    return counter


def generate(
    output: Path,
    train_plates: int,
    validation_plates: int,
    views_per_plate: int,
    font: Path,
    font_license: str,
    seed: int,
    jpeg_quality: int,
) -> dict:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    if not font.is_file():
        raise FileNotFoundError(font)
    normalized_font_license = str(font_license).strip().lower()
    if normalized_font_license not in ALLOWED_FONT_LICENSES:
        raise ValueError(
            "Font license is not approved for commercial synthetic data: "
            f"{normalized_font_license or 'missing'}"
        )
    output.mkdir(parents=True)
    rng = random.Random(int(seed))
    training = _unique_plates(train_plates, rng)
    validation = _unique_plates(
        validation_plates,
        rng,
        excluded=training,
    )
    train_count = _write_split(
        output,
        "train",
        training,
        views_per_plate,
        font,
        rng,
        jpeg_quality,
    )
    validation_count = _write_split(
        output,
        "val",
        validation,
        views_per_plate,
        font,
        rng,
        jpeg_quality,
    )
    manifest = {
        "schema": GENERATOR_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_license": "synthetic-bcvision-company-owned",
        "third_party_plate_dataset": False,
        "golden_benchmark_data": False,
        "generator": Path(__file__).name,
        "seed": int(seed),
        "font_path": str(font.resolve()),
        "font_sha256": _sha256(font),
        "font_license": normalized_font_license,
        "train_unique_plates": len(training),
        "validation_unique_plates": len(validation),
        "train_letter_counts": {
            letter: sum(plate[2] == letter for plate in training)
            for letter in ALLOWED_PLATE_LETTERS
        },
        "validation_letter_counts": {
            letter: sum(plate[2] == letter for plate in validation)
            for letter in ALLOWED_PLATE_LETTERS
        },
        "identity_overlap": len(set(training) & set(validation)),
        "views_per_plate": int(views_per_plate),
        "train_images": train_count,
        "validation_images": validation_count,
    }
    (output / "dataset-license.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Iranian CCT OCR dataset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-plates", type=int, default=3000)
    parser.add_argument("--validation-plates", type=int, default=500)
    parser.add_argument("--views-per-plate", type=int, default=4)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument(
        "--font-license",
        default="DejaVu-font-license",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    args = parser.parse_args(argv)
    manifest = generate(
        output=args.output,
        train_plates=max(20, int(args.train_plates)),
        validation_plates=max(5, int(args.validation_plates)),
        views_per_plate=max(1, min(12, int(args.views_per_plate))),
        font=(args.font or _default_font()).resolve(),
        font_license=args.font_license,
        seed=int(args.seed),
        jpeg_quality=max(45, min(98, int(args.jpeg_quality))),
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
