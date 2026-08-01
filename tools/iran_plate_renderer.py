"""Deterministic renderer for the white Iranian national plate layout.

The renderer contains geometry and drawing code only.  It deliberately ships
without a plate font or real plate pixels: callers must provide font files
whose training and distribution rights are known.  Real photographs may be
used for visual QA, but never as cut-and-paste glyph sources.
"""
from __future__ import annotations

from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter, ImageFont


BASE_PLATE_SIZE = (420, 90)
REFERENCE_RENDER_SCALE = 2
LAYOUT_PROFILE = "iran-national-photo-reference-v1"
SPECIAL_LAYOUT_PROFILE = "legacy-procedural-v2"
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"


# Coordinates are measured on the 840x184 calibration canvas.  They preserve
# the proportions seen across the supplied front/rear photographs: a narrow
# national band, six fixed main cells and a separately boxed two-digit region.
REFERENCE_GEOMETRY = {
    "canvas": (0, 0, 840, 180),
    "blue_band": (7, 7, 87, 172),
    "flag": (19, 18, 73, 51),
    "region_separator_x": 675,
    "iran_word": (686, 7, 834, 49),
    "prefix_cells": (
        (96, 22, 170, 171),
        (174, 22, 248, 171),
    ),
    "letter_cell": (270, 17, 410, 171),
    "serial_cells": (
        (418, 22, 500, 171),
        (504, 22, 586, 171),
        (590, 22, 670, 171),
    ),
    "region_cells": (
        (688, 47, 757, 171),
        (763, 47, 832, 171),
    ),
}


def _text_options(text: str) -> dict:
    if any("\u0600" <= character <= "\u06ff" for character in text):
        return {"direction": "rtl", "language": "fa"}
    return {}


def _font_signature(font, text: str) -> tuple:
    options = _text_options(text)
    mask = font.getmask(text, mode="L", **options)
    return mask.size, mask.getbbox(), bytes(mask)


def _font_supports(font, text: str) -> bool:
    missing = {
        _font_signature(font, "\u0378"),
        _font_signature(font, "\U0010ffff"),
    }
    return all(
        (
            (signature := _font_signature(font, character))[1] is not None
            and signature not in missing
        )
        for character in text
    )


def _select_font(
    text: str,
    primary_font: Path,
    fallback_font: Path | None,
    size: int,
):
    candidates = [primary_font]
    if fallback_font is not None and fallback_font != primary_font:
        candidates.append(fallback_font)
    for path in candidates:
        font = ImageFont.truetype(
            str(path),
            size,
            layout_engine=ImageFont.Layout.RAQM,
        )
        if _font_supports(font, text):
            return font
    raise ValueError(f"Font stack cannot render plate text: {text!r}")


def _text_mask(
    text: str,
    primary_font: Path,
    fallback_font: Path | None,
) -> Image.Image:
    font = _select_font(text, primary_font, fallback_font, 160)
    options = _text_options(text)
    bounds = font.getbbox(text, **options)
    width = max(1, bounds[2] - bounds[0])
    height = max(1, bounds[3] - bounds[1])
    mask = Image.new("L", (width + 16, height + 16), 0)
    draw = ImageDraw.Draw(mask)
    draw.text(
        (8 - bounds[0], 8 - bounds[1]),
        text,
        font=font,
        fill=255,
        **options,
    )
    content = mask.getbbox()
    if content is None:
        raise ValueError(f"Font stack produced no plate glyph: {text!r}")
    return mask.crop(content)


def _paste_fitted(
    image: Image.Image,
    text: str,
    box: tuple[int, int, int, int],
    *,
    primary_font: Path,
    fallback_font: Path | None,
    fill: tuple[int, int, int],
    width_fill: float,
    height_fill: float,
    emboss: bool = True,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    box_width = right - left
    box_height = bottom - top
    mask = _text_mask(text, primary_font, fallback_font)
    target_width = max(1, int(round(box_width * width_fill)))
    target_height = max(1, int(round(box_height * height_fill)))
    scale = min(
        target_width / max(1, mask.width),
        target_height / max(1, mask.height),
    )
    size = (
        max(1, int(round(mask.width * scale))),
        max(1, int(round(mask.height * scale))),
    )
    mask = mask.resize(size, Image.Resampling.LANCZOS)
    x = int(round(left + (box_width - size[0]) / 2))
    y = int(round(top + (box_height - size[1]) / 2))
    if emboss:
        rim = mask.filter(ImageFilter.MaxFilter(3))
        image.paste((126, 124, 116), (x + 1, y + 2), rim)
    image.paste(fill, (x, y), mask)
    return x, y, x + size[0], y + size[1]


def _draw_flag(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
) -> None:
    left, top, right, bottom = box
    stripe = max(1, (bottom - top) // 3)
    draw.rectangle((left, top, right, top + stripe), fill=(39, 151, 81))
    draw.rectangle(
        (left, top + stripe, right, top + 2 * stripe),
        fill=(248, 248, 245),
    )
    draw.rectangle(
        (left, top + 2 * stripe, right, bottom),
        fill=(218, 42, 49),
    )
    draw.rectangle(box, outline=(230, 230, 225), width=2)
    center = (left + right) // 2
    draw.ellipse(
        (
            center - 2,
            top + stripe + 3,
            center + 2,
            top + 2 * stripe - 3,
        ),
        fill=(198, 31, 39),
    )


def _draw_centered_ascii(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: tuple[int, int, int, int],
    font,
) -> None:
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2 - bounds[0]
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text((x, y), text, font=font, fill=(255, 255, 255))


def render_private_plate(
    plate: str,
    *,
    primary_font: Path,
    fallback_font: Path | None = None,
    background: tuple[int, int, int] = (246, 245, 238),
    foreground: tuple[int, int, int] = (13, 14, 15),
    letter_fill: tuple[int, int, int] | None = None,
    blue: tuple[int, int, int] = (4, 84, 180),
    rng: random.Random | None = None,
    calibration: bool = False,
) -> Image.Image:
    """Render an eight-slot private plate without using real plate pixels."""

    if len(plate) != 8 or not (
        plate[:2].isdigit()
        and plate[3:6].isdigit()
        and plate[6:8].isdigit()
    ):
        raise ValueError("Plate must use the canonical 2-letter-3-2 layout")
    primary_font = Path(primary_font)
    fallback_font = Path(fallback_font) if fallback_font is not None else None
    rng = rng or random.Random(0)

    width = BASE_PLATE_SIZE[0] * REFERENCE_RENDER_SCALE
    height = BASE_PLATE_SIZE[1] * REFERENCE_RENDER_SCALE
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (2, 2, width - 3, height - 3),
        radius=11,
        outline=(40, 42, 43),
        width=5,
    )
    draw.rounded_rectangle(
        (7, 7, width - 8, height - 8),
        radius=7,
        outline=(190, 188, 178),
        width=2,
    )
    band = REFERENCE_GEOMETRY["blue_band"]
    draw.rectangle(band, fill=blue)
    _draw_flag(draw, REFERENCE_GEOMETRY["flag"])

    auxiliary_path = fallback_font or primary_font
    auxiliary_font = _select_font(
        "I.R.IRAN",
        auxiliary_path,
        primary_font,
        21,
    )
    _draw_centered_ascii(draw, "I.R.", (10, 86, 84, 116), auxiliary_font)
    _draw_centered_ascii(draw, "IRAN", (10, 116, 84, 151), auxiliary_font)

    separator = REFERENCE_GEOMETRY["region_separator_x"]
    draw.line((separator, 5, separator, height - 6), fill=(45, 45, 43), width=4)
    _paste_fitted(
        image,
        "ایران",
        REFERENCE_GEOMETRY["iran_word"],
        primary_font=primary_font,
        fallback_font=fallback_font,
        fill=foreground,
        width_fill=0.72,
        height_fill=0.73,
    )

    offset_x = 0 if calibration else rng.randint(-2, 2)
    offset_y = 0 if calibration else rng.randint(-2, 2)

    def shifted(box):
        left, top, right, bottom = box
        return (
            left + offset_x,
            top + offset_y,
            right + offset_x,
            bottom + offset_y,
        )

    prefix = tuple(PERSIAN_DIGITS[int(value)] for value in plate[:2])
    serial = tuple(PERSIAN_DIGITS[int(value)] for value in plate[3:6])
    region = tuple(PERSIAN_DIGITS[int(value)] for value in plate[6:8])
    for text, box in zip(
        prefix,
        REFERENCE_GEOMETRY["prefix_cells"],
        strict=True,
    ):
        _paste_fitted(
            image,
            text,
            shifted(box),
            primary_font=primary_font,
            fallback_font=fallback_font,
            fill=foreground,
            width_fill=0.84,
            height_fill=0.86,
        )
    _paste_fitted(
        image,
        plate[2],
        shifted(REFERENCE_GEOMETRY["letter_cell"]),
        primary_font=primary_font,
        fallback_font=fallback_font,
        fill=letter_fill or foreground,
        width_fill=0.82,
        height_fill=0.88,
    )
    for text, box in zip(
        serial,
        REFERENCE_GEOMETRY["serial_cells"],
        strict=True,
    ):
        _paste_fitted(
            image,
            text,
            shifted(box),
            primary_font=primary_font,
            fallback_font=fallback_font,
            fill=foreground,
            width_fill=0.84,
            height_fill=0.86,
        )
    for text, box in zip(
        region,
        REFERENCE_GEOMETRY["region_cells"],
        strict=True,
    ):
        _paste_fitted(
            image,
            text,
            shifted(box),
            primary_font=primary_font,
            fallback_font=fallback_font,
            fill=foreground,
            width_fill=0.86,
            height_fill=0.88,
        )

    return image.resize(BASE_PLATE_SIZE, Image.Resampling.LANCZOS)
