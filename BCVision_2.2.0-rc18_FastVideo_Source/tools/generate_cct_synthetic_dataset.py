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
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import (
    Image,
    ImageDraw,
    ImageEnhance,
    ImageFilter,
    ImageFont,
    features as pillow_features,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.plate_rules import ALLOWED_PLATE_LETTERS
from tools.iran_plate_renderer import (
    BASE_PLATE_SIZE,
    LAYOUT_PROFILE,
    SPECIAL_LAYOUT_PROFILE,
    render_private_plate,
)


GENERATOR_SCHEMA = 3
DEFAULT_SEED = 20260728
OUTPUT_SIZE = (128, 64)
PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
FOCUS_LETTERS = ("ژ", "ث", "ا", "ف", "ک", "گ", "D", "S")
CONDITION_PROFILES = (
    "clean",
    "daylight",
    "night",
    "motion_blur",
    "perspective",
    "headlight_glare",
    "overexposed_defocus",
    "rain",
    "dirt",
    "low_resolution",
    "mixed_hard",
)
CONDITION_PROFILE_VERSIONS = {
    "overexposed_defocus": "rear-plate-overexposed-defocus-v1",
}
ALLOWED_FONT_LICENSES = {
    "apache-2.0",
    "bcvision-company-owned",
    "bsd-3-clause",
    "cc0-1.0",
    "dejavu-font-license",
    "ofl-1.1",
}

PLATE_STYLES = {
    "private": {
        "background": (248, 248, 244),
        "foreground": (12, 14, 18),
        "stripe": (7, 86, 178),
    },
    "public": {
        "background": (246, 194, 31),
        "foreground": (18, 16, 12),
        "stripe": (7, 86, 178),
    },
    "government": {
        "background": (174, 25, 34),
        "foreground": (255, 255, 255),
        "stripe": (7, 86, 178),
    },
    "military": {
        "background": (26, 86, 54),
        "foreground": (255, 255, 255),
        "stripe": (7, 86, 178),
    },
    "diplomatic": {
        "background": (28, 83, 153),
        "foreground": (255, 255, 255),
        "stripe": (245, 245, 245),
    },
    "service": {
        "background": (169, 41, 51),
        "foreground": (255, 255, 255),
        "stripe": (245, 245, 245),
    },
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
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "No approved default Persian-capable font found; pass --font and "
        "--font-license explicitly"
    )


def _validate_font_rendering(font_path: Path) -> None:
    _validate_font_stack(font_path)


def _font_signature(font, text: str) -> tuple:
    mask = font.getmask(
        text,
        mode="L",
        direction="rtl" if any(
            "\u0600" <= character <= "\u06ff"
            for character in text
        ) else None,
        language="fa",
    )
    return mask.size, mask.getbbox(), bytes(mask)


def _loaded_font(
    font_path: Path,
    size: int,
):
    return ImageFont.truetype(
        str(font_path),
        size,
        layout_engine=ImageFont.Layout.RAQM,
    )


def _font_has_text(font, text: str, missing_signatures: set[tuple]) -> bool:
    for character in text:
        signature = _font_signature(font, character)
        if signature[1] is None or signature in missing_signatures:
            return False
    return True


def _validate_font_stack(
    font_path: Path,
    fallback_font_path: Path | None = None,
) -> None:
    if not pillow_features.check_feature("raqm"):
        raise RuntimeError(
            "Pillow RAQM shaping support is required for Persian text"
        )
    try:
        fonts = [_loaded_font(font_path, 48)]
        if fallback_font_path is not None:
            fonts.append(_loaded_font(fallback_font_path, 48))
    except Exception as exc:
        raise ValueError("Unreadable primary or fallback font") from exc

    missing_signatures = {
        _font_signature(font, "\u0378")
        for font in fonts
    } | {
        _font_signature(font, "\U0010ffff")
        for font in fonts
    }
    required = set(
        "۰۱۲۳۴۵۶۷۸۹ایرانالف♿IR.ADS"
        + "".join(
            _display_letter(letter)
            for letter in ALLOWED_PLATE_LETTERS
        )
    )
    missing = sorted(
        character
        for character in required
        if not any(
            _font_has_text(font, character, missing_signatures)
            for font in fonts
        )
    )
    if missing:
        raise ValueError(
            "Font stack is missing required Iranian plate glyphs: "
            + "".join(missing)
        )
    try:
        shaped = next(
            _font_signature(font, "ایران")
            for font in fonts
            if _font_has_text(font, "ایران", missing_signatures)
        )
    except Exception as exc:
        raise ValueError(
            "Font stack cannot shape Persian text with RAQM"
        ) from exc
    if shaped[1] is None:
        raise ValueError("Font stack produced no shaped Persian text")


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


def _unique_plates(
    count: int,
    rng: random.Random,
    excluded=None,
    focus_letters: tuple[str, ...] = FOCUS_LETTERS,
    focus_multiplier: int = 3,
) -> list[str]:
    values = set(excluded or ())
    selected = []
    letters = list(ALLOWED_PLATE_LETTERS)
    rng.shuffle(letters)
    weighted_letters = list(ALLOWED_PLATE_LETTERS)
    approved_focus = [
        letter
        for letter in focus_letters
        if letter in ALLOWED_PLATE_LETTERS
    ]
    weighted_letters.extend(
        approved_focus * max(0, int(focus_multiplier) - 1)
    )
    rng.shuffle(weighted_letters)
    letter_index = 0
    while len(selected) < count:
        if letter_index < len(letters):
            letter = letters[letter_index]
        else:
            letter = weighted_letters[
                (letter_index - len(letters)) % len(weighted_letters)
            ]
        plate = random_plate(
            rng,
            letter=letter,
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
    text_options = {}
    if any("\u0600" <= character <= "\u06ff" for character in text):
        text_options = {"direction": "rtl", "language": "fa"}
    bounds = draw.textbbox(
        (0, 0),
        text,
        font=font,
        **text_options,
    )
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    x = left + (right - left - width) / 2 - bounds[0]
    y = top + (bottom - top - height) / 2 - bounds[1]
    draw.text(
        (x, y),
        text,
        font=font,
        fill=fill,
        **text_options,
    )


def _font_path_for_text(
    text: str,
    font_path: Path,
    fallback_font_path: Path | None,
) -> Path:
    candidates = [font_path]
    if fallback_font_path is not None and fallback_font_path != font_path:
        candidates.append(fallback_font_path)
    for candidate in candidates:
        font = _loaded_font(candidate, 48)
        missing_signatures = {
            _font_signature(font, "\u0378"),
            _font_signature(font, "\U0010ffff"),
        }
        if _font_has_text(font, text, missing_signatures):
            return candidate
    raise ValueError(f"Font stack cannot render plate text: {text!r}")


def _legacy_base_plate(
    plate: str,
    font_path: Path,
    rng: random.Random,
    fallback_font_path: Path | None = None,
) -> Image.Image:
    width, height = BASE_PLATE_SIZE
    letter = plate[2]
    style_name = _style_for_letter(letter)
    style = PLATE_STYLES[style_name]
    background = _jitter_color(style["background"], rng, 5)
    foreground = _jitter_color(style["foreground"], rng, 4)
    stripe = _jitter_color(style["stripe"], rng, 5)
    image = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(image)
    border = rng.randint(2, 4)
    draw.rounded_rectangle(
        (1, 1, width - 2, height - 2),
        radius=7,
        outline=foreground,
        width=border,
    )
    blue_width = rng.randint(39, 45)
    draw.rectangle(
        (border, border, blue_width, height - border),
        fill=stripe,
    )
    small_font = ImageFont.truetype(
        str(
            _font_path_for_text(
                "IR.ایران",
                font_path,
                fallback_font_path,
            )
        ),
        11,
        layout_engine=ImageFont.Layout.RAQM,
    )
    main_font = ImageFont.truetype(
        str(font_path),
        rng.randint(48, 54),
        layout_engine=ImageFont.Layout.RAQM,
    )
    letter_font = ImageFont.truetype(
        str(
            _font_path_for_text(
                _display_letter(letter),
                font_path,
                fallback_font_path,
            )
        ),
        rng.randint(35, 43) if letter == "ا" else rng.randint(46, 52),
        layout_engine=ImageFont.Layout.RAQM,
    )
    _centered_text(
        draw,
        (2, 4, blue_width, 34),
        "IR",
        small_font,
        style["background"] if style_name in {"diplomatic", "service"} else (255, 255, 255),
    )
    _centered_text(
        draw,
        (2, 31, blue_width, height - 4),
        "I.R.",
        small_font,
        style["background"] if style_name in {"diplomatic", "service"} else (255, 255, 255),
    )

    region_left = width - rng.randint(73, 79)
    draw.line(
        (region_left, border, region_left, height - border),
        fill=foreground,
        width=2,
    )
    _centered_text(
        draw,
        (region_left, 3, width - 3, 29),
        "ایران",
        small_font,
        foreground,
    )
    _centered_text(
        draw,
        (region_left, 23, width - 3, height - 3),
        plate[6:8].translate(PERSIAN_DIGITS),
        main_font,
        foreground,
    )

    content_left = blue_width + 7
    content_right = region_left - 4
    spans = [
        (
            content_left,
            content_left + 91,
            plate[:2].translate(PERSIAN_DIGITS),
            main_font,
        ),
        (
            content_left + 88,
            content_left + 151,
            _display_letter(letter),
            letter_font,
        ),
        (
            content_left + 146,
            content_right,
            plate[3:6].translate(PERSIAN_DIGITS),
            main_font,
        ),
    ]
    for left, right, text, font in spans:
        text_fill = (
            (20, 88, 182)
            if letter == "ژ" and text == _display_letter(letter)
            else foreground
        )
        _centered_text(
            draw,
            (left, 4, right, height - 4),
            text,
            font,
            text_fill,
        )
    return image


def _base_plate(
    plate: str,
    font_path: Path,
    rng: random.Random,
    fallback_font_path: Path | None = None,
    *,
    calibration: bool = False,
) -> Image.Image:
    letter = plate[2]
    style_name = _style_for_letter(letter)
    if style_name != "private":
        return _legacy_base_plate(
            plate,
            font_path,
            rng,
            fallback_font_path=fallback_font_path,
        )

    style = PLATE_STYLES[style_name]
    background = (
        style["background"]
        if calibration
        else _jitter_color(style["background"], rng, 4)
    )
    foreground = (
        style["foreground"]
        if calibration
        else _jitter_color(style["foreground"], rng, 3)
    )
    stripe = (
        style["stripe"]
        if calibration
        else _jitter_color(style["stripe"], rng, 4)
    )
    display_plate = plate[:2] + _display_letter(letter) + plate[3:]
    return render_private_plate(
        display_plate,
        primary_font=font_path,
        fallback_font=fallback_font_path,
        background=background,
        foreground=foreground,
        letter_fill=(
            (20, 88, 182)
            if letter == "ژ"
            else foreground
        ),
        blue=stripe,
        rng=rng,
        calibration=calibration,
    )


def _display_letter(letter: str) -> str:
    if letter == "ا":
        return "الف"
    if letter == "ژ":
        return "♿"
    return letter


def _style_for_letter(letter: str) -> str:
    if letter in {"ت", "ع", "ک"}:
        return "public"
    if letter == "ا":
        return "government"
    if letter in {"پ", "ث"}:
        return "military"
    if letter == "D":
        return "diplomatic"
    if letter == "S":
        return "service"
    return "private"


def _jitter_color(
    color: tuple[int, int, int],
    rng: random.Random,
    amount: int,
) -> tuple[int, int, int]:
    return tuple(
        max(0, min(255, channel + rng.randint(-amount, amount)))
        for channel in color
    )


def _sample_seed(master_seed: int, split: str, plate: str, view: int) -> int:
    payload = f"{master_seed}:{split}:{plate}:{view}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _perspective(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> tuple[np.ndarray, dict]:
    height, width = array.shape[:2]
    margin_x = max(1, int(width * rng.uniform(0.01, 0.12) * severity))
    margin_y = max(1, int(height * rng.uniform(0.01, 0.18) * severity))
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
    return array, {
        "perspective_x_px": margin_x,
        "perspective_y_px": margin_y,
    }


def _motion_blur(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> tuple[np.ndarray, dict]:
    length = max(3, int(round(rng.uniform(5, 21) * severity)))
    if length % 2 == 0:
        length += 1
    angle = rng.uniform(-18.0, 18.0)
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2
    radians = math.radians(angle)
    dx = math.cos(radians) * center
    dy = math.sin(radians) * center
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0,
        1,
    )
    total = float(kernel.sum())
    if total <= 0:
        kernel[length // 2, :] = 1.0
        total = float(length)
    kernel /= total
    return cv2.filter2D(array, -1, kernel), {
        "motion_blur_length": length,
        "motion_blur_angle": round(angle, 2),
    }


def _apply_vignette(
    array: np.ndarray,
    strength: float,
) -> np.ndarray:
    height, width = array.shape[:2]
    x = cv2.getGaussianKernel(width, width * 0.48)
    y = cv2.getGaussianKernel(height, height * 0.48)
    mask = y @ x.T
    mask /= max(float(mask.max()), 1e-6)
    mask = (1.0 - strength) + strength * mask
    return np.clip(array.astype(np.float32) * mask[..., None], 0, 255).astype(
        np.uint8
    )


def _add_glare(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> tuple[np.ndarray, dict]:
    height, width = array.shape[:2]
    overlay = np.zeros_like(array, dtype=np.float32)
    mask = np.zeros((height, width), dtype=np.float32)
    center = (
        rng.randint(int(width * 0.10), int(width * 0.90)),
        rng.randint(int(height * 0.10), int(height * 0.90)),
    )
    axes = (
        max(8, int(width * rng.uniform(0.05, 0.16) * severity)),
        max(4, int(height * rng.uniform(0.10, 0.32) * severity)),
    )
    cv2.ellipse(mask, center, axes, rng.uniform(-15, 15), 0, 360, 1.0, -1)
    sigma = max(2.0, max(axes) * 0.65)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=sigma, sigmaY=sigma)
    color = np.asarray(
        (rng.randint(225, 255), rng.randint(225, 252), rng.randint(205, 240)),
        dtype=np.float32,
    )
    overlay[:] = color
    alpha = np.clip(mask * rng.uniform(0.38, 0.72), 0.0, 0.78)[..., None]
    result = array.astype(np.float32) * (1.0 - alpha) + overlay * alpha
    return np.clip(result, 0, 255).astype(np.uint8), {
        "glare_center": [int(center[0]), int(center[1])],
        "glare_axes": [int(axes[0]), int(axes[1])],
    }


def _add_overexposed_defocus(
    array: np.ndarray,
    rng: random.Random,
) -> tuple[np.ndarray, dict]:
    """Model broad retroreflective washout without copying reference pixels."""

    draw = rng.random()
    if draw < 0.65:
        tier = "mild"
        blur_sigma = rng.uniform(0.40, 0.75)
        exposure_ev = rng.uniform(0.10, 0.35)
        bloom_strength = rng.uniform(0.05, 0.11)
    elif draw < 0.95:
        tier = "medium"
        blur_sigma = rng.uniform(0.65, 1.10)
        exposure_ev = rng.uniform(0.25, 0.55)
        bloom_strength = rng.uniform(0.08, 0.16)
    elif rng.random() < 0.5:
        tier = "hard-blur"
        blur_sigma = rng.uniform(1.10, 1.45)
        exposure_ev = rng.uniform(0.10, 0.40)
        bloom_strength = rng.uniform(0.05, 0.10)
    else:
        tier = "hard-exposure"
        blur_sigma = rng.uniform(0.40, 0.80)
        exposure_ev = rng.uniform(0.55, 0.75)
        bloom_strength = rng.uniform(0.08, 0.16)

    bloom_threshold = rng.uniform(0.76, 0.90)
    bloom_radius = rng.uniform(1.50, 4.00)
    contrast = rng.uniform(0.72, 0.94)
    black_lift = rng.uniform(0.01, 0.05)

    output_scale = max(
        array.shape[1] / OUTPUT_SIZE[0],
        array.shape[0] / OUTPUT_SIZE[1],
    )
    normalized = array.astype(np.float32) / 255.0
    linear = np.power(np.clip(normalized, 0.0, 1.0), 2.2)
    linear = cv2.GaussianBlur(
        linear,
        (0, 0),
        sigmaX=blur_sigma * output_scale,
        sigmaY=blur_sigma * output_scale,
    )
    luminance = (
        linear[..., 0] * 0.2126
        + linear[..., 1] * 0.7152
        + linear[..., 2] * 0.0722
    )
    highlights = np.clip(
        (luminance - bloom_threshold)
        / max(1e-6, 1.0 - bloom_threshold),
        0.0,
        1.0,
    )
    bloom = cv2.GaussianBlur(
        highlights,
        (0, 0),
        sigmaX=bloom_radius * output_scale,
        sigmaY=bloom_radius * output_scale,
    )[..., None]
    linear = 1.0 - (1.0 - linear) * (
        1.0 - np.clip(bloom * bloom_strength, 0.0, 0.22)
    )
    linear = np.clip(
        linear * (2.0 ** exposure_ev) + black_lift,
        0.0,
        1.0,
    )
    result = np.power(linear, 1.0 / 2.2)
    result = np.clip(
        (result - 0.5) * contrast + 0.5,
        0.0,
        1.0,
    )
    return np.round(result * 255.0).astype(np.uint8), {
        "degradation_profile_version": CONDITION_PROFILE_VERSIONS[
            "overexposed_defocus"
        ],
        "degradation_tier": tier,
        "defocus_sigma_output_px": round(blur_sigma, 4),
        "exposure_ev": round(exposure_ev, 4),
        "bloom_threshold": round(bloom_threshold, 4),
        "bloom_radius_output_px": round(bloom_radius, 4),
        "bloom_strength": round(bloom_strength, 4),
        "contrast_factor": round(contrast, 4),
        "black_lift": round(black_lift, 4),
    }


def _add_rain(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> tuple[np.ndarray, dict]:
    height, width = array.shape[:2]
    overlay = array.copy()
    count = max(6, int(rng.uniform(12, 30) * severity))
    slant = rng.randint(-5, 5)
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        length = rng.randint(4, max(5, int(height * 0.25)))
        cv2.line(
            overlay,
            (x, y),
            (max(0, min(width - 1, x + slant)), min(height - 1, y + length)),
            (205, 215, 225),
            rng.choice((1, 1, 2)),
        )
    overlay = cv2.GaussianBlur(overlay, (3, 3), 0.35)
    return cv2.addWeighted(
        overlay,
        rng.uniform(0.18, 0.38),
        array,
        rng.uniform(0.62, 0.82),
        0,
    ), {"rain_streaks": count}


def _add_dirt(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> tuple[np.ndarray, dict]:
    height, width = array.shape[:2]
    overlay = array.copy()
    count = max(3, int(rng.uniform(5, 15) * severity))
    for _ in range(count):
        center = (rng.randrange(width), rng.randrange(height))
        axes = (
            rng.randint(2, max(3, int(width * 0.035))),
            rng.randint(1, max(2, int(height * 0.11))),
        )
        tone = rng.randint(45, 115)
        color = (tone + rng.randint(4, 25), tone, max(20, tone - 25))
        cv2.ellipse(
            overlay,
            center,
            axes,
            rng.uniform(0, 180),
            0,
            360,
            color,
            -1,
        )
    alpha = min(0.38, 0.14 + 0.12 * severity)
    return cv2.addWeighted(overlay, alpha, array, 1.0 - alpha, 0), {
        "dirt_spots": count,
        "dirt_opacity": round(alpha, 3),
    }


def _add_shadow(
    array: np.ndarray,
    rng: random.Random,
    severity: float,
) -> np.ndarray:
    height, width = array.shape[:2]
    mask = np.ones((height, width), dtype=np.float32)
    polygon = np.asarray([
        [rng.randint(0, width // 3), 0],
        [rng.randint(width // 2, width - 1), 0],
        [rng.randint(width // 2, width - 1), height - 1],
        [rng.randint(0, width // 3), height - 1],
    ], dtype=np.int32)
    shadow = np.ones_like(mask) * rng.uniform(
        max(0.30, 0.62 - severity * 0.22),
        max(0.45, 0.82 - severity * 0.12),
    )
    cv2.fillPoly(mask, [polygon], float(shadow[0, 0]))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(5, width * 0.025))
    return np.clip(array.astype(np.float32) * mask[..., None], 0, 255).astype(
        np.uint8
    )


def _photometric(
    array: np.ndarray,
    rng: random.Random,
    brightness: tuple[float, float],
    contrast: tuple[float, float],
    color_shift: int,
) -> np.ndarray:
    result = Image.fromarray(array)
    result = ImageEnhance.Brightness(result).enhance(rng.uniform(*brightness))
    result = ImageEnhance.Contrast(result).enhance(rng.uniform(*contrast))
    value = np.asarray(result).astype(np.int16)
    if color_shift:
        warm = rng.randint(-color_shift, color_shift)
        value[..., 0] += warm
        value[..., 2] -= warm
    return np.clip(value, 0, 255).astype(np.uint8)


def _sensor_noise(
    array: np.ndarray,
    rng: random.Random,
    sigma: float,
) -> np.ndarray:
    noise = np.random.default_rng(
        rng.randrange(0, 2 ** 32)
    ).normal(0, sigma, array.shape)
    return np.clip(array.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def _quality_metrics(array: np.ndarray) -> dict:
    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    contrast = float(gray.std())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    exposure = float(gray.mean())
    exposure_score = max(0.0, 1.0 - abs(exposure - 128.0) / 128.0)
    contrast_score = min(1.0, contrast / 55.0)
    sharpness_score = min(1.0, sharpness / 500.0)
    score = (
        exposure_score * 0.25
        + contrast_score * 0.45
        + sharpness_score * 0.30
    )
    return {
        "quality_score": round(float(score), 4),
        "mean_luma": round(exposure, 2),
        "contrast_std": round(contrast, 2),
        "laplacian_variance": round(sharpness, 2),
        "clipped_highlight_fraction": round(
            float(np.mean(gray >= 254)),
            4,
        ),
    }


def _edge_retention(
    degraded: np.ndarray,
    clean: np.ndarray,
) -> float:
    def energy(value: np.ndarray) -> float:
        gray = cv2.cvtColor(value, cv2.COLOR_RGB2GRAY)
        horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return float(np.mean(cv2.magnitude(horizontal, vertical)))

    return min(1.0, energy(degraded) / max(energy(clean), 1e-6))


def _profile_conditions(profile: str, rng: random.Random) -> list[str]:
    if profile not in CONDITION_PROFILES:
        raise ValueError(f"Unknown condition profile: {profile}")
    if profile != "mixed_hard":
        return [profile]
    candidates = [
        "night",
        "motion_blur",
        "perspective",
        "headlight_glare",
        "rain",
        "dirt",
        "low_resolution",
    ]
    return rng.sample(candidates, rng.randint(3, 5))


def _degrade(
    image: Image.Image,
    rng: random.Random,
    profile: str = "mixed_hard",
    output_size: tuple[int, int] = OUTPUT_SIZE,
    return_metadata: bool = False,
):
    original = np.asarray(image)
    array = original.copy()
    conditions = _profile_conditions(profile, rng)
    severity = rng.uniform(0.72, 1.0) if profile == "mixed_hard" else rng.uniform(
        0.42,
        0.82,
    )
    metadata = {
        "condition_profile": profile,
        "conditions": conditions,
        "severity": round(severity, 4),
        "simulated_source_width": int(original.shape[1]),
        "quality_rescue": False,
    }

    if "clean" in conditions:
        array = _photometric(array, rng, (0.96, 1.05), (0.95, 1.08), 2)
    elif profile == "overexposed_defocus":
        if rng.random() < 0.45:
            array, values = _perspective(array, rng, severity * 0.18)
            metadata.update(values)
        array, values = _add_overexposed_defocus(array, rng)
        metadata.update(values)
        if rng.random() < 0.35:
            noise_sigma = rng.uniform(1.0, 3.0)
            array = _sensor_noise(array, rng, noise_sigma)
            metadata["sensor_noise_sigma"] = round(noise_sigma, 4)
    else:
        if "night" in conditions:
            array = _photometric(
                array,
                rng,
                (0.28, 0.62),
                (0.72, 1.18),
                15,
            )
            array = _apply_vignette(array, rng.uniform(0.20, 0.48))
            array = _sensor_noise(
                array,
                rng,
                rng.uniform(4.0, 13.0) * severity,
            )
            metadata["night"] = True
        elif "daylight" in conditions:
            array = _photometric(
                array,
                rng,
                (0.92, 1.34),
                (0.88, 1.28),
                10,
            )
        else:
            array = _photometric(
                array,
                rng,
                (0.68, 1.24),
                (0.72, 1.35),
                8,
            )

        if "perspective" in conditions:
            array, values = _perspective(array, rng, severity)
            metadata.update(values)
        elif rng.random() < 0.35:
            array, values = _perspective(array, rng, severity * 0.38)
            metadata.update(values)

        if "motion_blur" in conditions:
            array, values = _motion_blur(array, rng, severity)
            metadata.update(values)
        elif rng.random() < 0.25:
            radius = rng.uniform(0.15, 0.65)
            array = np.asarray(
                Image.fromarray(array).filter(ImageFilter.GaussianBlur(radius))
            )
            metadata["gaussian_blur_radius"] = round(radius, 3)

        if "headlight_glare" in conditions:
            array, values = _add_glare(array, rng, severity)
            metadata.update(values)
        if "rain" in conditions:
            array, values = _add_rain(array, rng, severity)
            metadata.update(values)
        if "dirt" in conditions:
            array, values = _add_dirt(array, rng, severity)
            metadata.update(values)
        if rng.random() < (0.45 if profile == "mixed_hard" else 0.14):
            array = _add_shadow(array, rng, severity)
            metadata["shadow"] = True

        array = _sensor_noise(
            array,
            rng,
            rng.uniform(1.0, 5.5) * severity,
        )

    if "low_resolution" in conditions:
        target_width = rng.randint(68, 155)
        target_height = max(18, int(round(array.shape[0] * target_width / array.shape[1])))
        small = cv2.resize(
            array,
            (target_width, target_height),
            interpolation=cv2.INTER_AREA,
        )
        array = cv2.resize(
            small,
            (array.shape[1], array.shape[0]),
            interpolation=rng.choice((cv2.INTER_LINEAR, cv2.INTER_CUBIC)),
        )
        metadata["simulated_source_width"] = target_width

    resized = cv2.resize(array, output_size, interpolation=cv2.INTER_AREA)
    clean = cv2.resize(original, output_size, interpolation=cv2.INTER_AREA)
    if profile == "overexposed_defocus":
        retention = _edge_retention(resized, clean)
        if retention < 0.20:
            for clean_weight in (0.12, 0.20, 0.28):
                candidate = cv2.addWeighted(
                    resized,
                    1.0 - clean_weight,
                    clean,
                    clean_weight,
                    0,
                )
                retention = _edge_retention(candidate, clean)
                resized = candidate
                if retention >= 0.20:
                    break
            metadata["quality_rescue"] = True
        metadata["edge_retention"] = round(retention, 4)
    metrics = _quality_metrics(resized)
    if metrics["quality_score"] < 0.12:
        resized = cv2.addWeighted(resized, 0.72, clean, 0.28, 0)
        metadata["quality_rescue"] = True
        metrics = _quality_metrics(resized)
    metadata.update(metrics)
    metadata["difficulty"] = (
        "hard"
        if profile in {
            "mixed_hard",
            "night",
            "motion_blur",
            "overexposed_defocus",
            "low_resolution",
        }
        else "easy"
        if profile == "clean"
        else "medium"
    )
    bgr = cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)
    if return_metadata:
        return bgr, metadata
    return bgr


def _write_split(
    root: Path,
    name: str,
    plates: list[str],
    views_per_plate: int,
    font: Path,
    fallback_font: Path | None,
    seed: int,
    jpeg_quality: int,
    profiles: tuple[str, ...],
) -> dict:
    split = root / name
    images = split / "images"
    images.mkdir(parents=True, exist_ok=False)
    rows = []
    sample_metadata = []
    counter = 0
    condition_counts = {profile: 0 for profile in profiles}
    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0}
    quality_scores = []
    rescued = 0
    for group_index, plate in enumerate(plates):
        for view in range(views_per_plate):
            sample_seed = _sample_seed(seed, name, plate, view)
            sample_rng = random.Random(sample_seed)
            profile = profiles[counter % len(profiles)]
            image, metadata = _degrade(
                _base_plate(
                    plate,
                    font,
                    sample_rng,
                    fallback_font_path=fallback_font,
                ),
                sample_rng,
                profile=profile,
                output_size=OUTPUT_SIZE,
                return_metadata=True,
            )
            filename = (
                f"{group_index:06d}-{view:02d}-{profile}.jpg"
            )
            target = images / filename
            effective_quality = int(jpeg_quality)
            if metadata["difficulty"] == "hard":
                effective_quality = max(
                    48,
                    effective_quality - sample_rng.randint(4, 18),
                )
            ok, payload = cv2.imencode(
                ".jpg",
                image,
                [cv2.IMWRITE_JPEG_QUALITY, effective_quality],
            )
            if not ok:
                raise RuntimeError(f"Failed to encode {target}")
            target.write_bytes(payload.tobytes())
            row = {
                "image_path": f"images/{filename}",
                "plate_text": plate,
                "condition_profile": profile,
                "difficulty": metadata["difficulty"],
                "quality_score": metadata["quality_score"],
                "simulated_source_width": metadata[
                    "simulated_source_width"
                ],
                "sample_seed": sample_seed,
                "plate_style": _style_for_letter(plate[2]),
            }
            rows.append(row)
            sample_metadata.append({
                **row,
                **metadata,
                "jpeg_quality": effective_quality,
            })
            condition_counts[profile] += 1
            difficulty_counts[metadata["difficulty"]] += 1
            quality_scores.append(float(metadata["quality_score"]))
            rescued += int(bool(metadata["quality_rescue"]))
            counter += 1
    with (split / "annotations.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "plate_text",
                "condition_profile",
                "difficulty",
                "quality_score",
                "simulated_source_width",
                "sample_seed",
                "plate_style",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with (split / "samples.jsonl").open(
        "w",
        encoding="utf-8",
    ) as handle:
        for row in sample_metadata:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {
        "images": counter,
        "condition_counts": condition_counts,
        "difficulty_counts": difficulty_counts,
        "mean_quality_score": round(
            sum(quality_scores) / max(1, len(quality_scores)),
            4,
        ),
        "quality_rescues": rescued,
    }


def generate(
    output: Path,
    train_plates: int,
    validation_plates: int,
    views_per_plate: int,
    font: Path,
    font_license: str,
    seed: int,
    jpeg_quality: int,
    profiles: tuple[str, ...] = CONDITION_PROFILES,
    focus_letters: tuple[str, ...] = FOCUS_LETTERS,
    focus_multiplier: int = 3,
    test_plates: int = 0,
    fallback_font: Path | None = None,
    fallback_font_license: str | None = None,
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
    normalized_fallback_license = str(
        fallback_font_license or ""
    ).strip().lower()
    if (fallback_font is None) != (not normalized_fallback_license):
        raise ValueError(
            "Fallback font and fallback font license must be supplied together"
        )
    if fallback_font is not None:
        fallback_font = Path(fallback_font)
        if not fallback_font.is_file():
            raise FileNotFoundError(fallback_font)
        if normalized_fallback_license not in ALLOWED_FONT_LICENSES:
            raise ValueError(
                "Fallback font license is not approved for commercial "
                "synthetic data: "
                f"{normalized_fallback_license or 'missing'}"
            )
    _validate_font_stack(font, fallback_font)
    profiles = tuple(dict.fromkeys(profiles))
    if not profiles:
        raise ValueError("At least one condition profile is required")
    unknown_profiles = sorted(set(profiles) - set(CONDITION_PROFILES))
    if unknown_profiles:
        raise ValueError(
            "Unknown condition profile(s): "
            + ", ".join(unknown_profiles)
        )
    invalid_focus = sorted(
        set(focus_letters) - set(ALLOWED_PLATE_LETTERS)
    )
    if invalid_focus:
        raise ValueError(
            "Unsupported focus letter(s): " + ", ".join(invalid_focus)
        )
    output.mkdir(parents=True)
    rng = random.Random(int(seed))
    training = _unique_plates(
        train_plates,
        rng,
        focus_letters=focus_letters,
        focus_multiplier=focus_multiplier,
    )
    validation = _unique_plates(
        validation_plates,
        rng,
        excluded=training,
        focus_letters=focus_letters,
        focus_multiplier=focus_multiplier,
    )
    test = _unique_plates(
        max(0, int(test_plates)),
        rng,
        excluded=training + validation,
        focus_letters=focus_letters,
        focus_multiplier=focus_multiplier,
    )
    train_result = _write_split(
        output,
        "train",
        training,
        views_per_plate,
        font,
        fallback_font,
        seed,
        jpeg_quality,
        profiles,
    )
    validation_result = _write_split(
        output,
        "val",
        validation,
        views_per_plate,
        font,
        fallback_font,
        seed,
        jpeg_quality,
        profiles,
    )
    test_result = (
        _write_split(
            output,
            "test",
            test,
            views_per_plate,
            font,
            fallback_font,
            seed,
            jpeg_quality,
            profiles,
        )
        if test
        else None
    )
    train_identities = set(training)
    validation_identities = set(validation)
    test_identities = set(test)
    identity_overlaps = {
        "train_validation": len(
            train_identities & validation_identities
        ),
        "train_test": len(train_identities & test_identities),
        "validation_test": len(
            validation_identities & test_identities
        ),
    }
    manifest = {
        "schema": GENERATOR_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_license": "synthetic-bcvision-company-owned",
        "third_party_plate_dataset": False,
        "golden_benchmark_data": False,
        "generator": Path(__file__).name,
        "renderer": "iran_plate_renderer.py",
        "renderer_sha256": _sha256(
            Path(__file__).with_name("iran_plate_renderer.py")
        ),
        "procedural_only": True,
        "real_plate_pixels_used": False,
        "activation_policy": (
            "shadow-only-until-independent-real-camera-pass"
        ),
        "seed": int(seed),
        "font_path": font.name,
        "font_sha256": _sha256(font),
        "font_license": normalized_font_license,
        "fallback_font_path": (
            fallback_font.name
            if fallback_font is not None
            else None
        ),
        "fallback_font_sha256": (
            _sha256(fallback_font)
            if fallback_font is not None
            else None
        ),
        "fallback_font_license": (
            normalized_fallback_license
            if fallback_font is not None
            else None
        ),
        "layout_profile": LAYOUT_PROFILE,
        "layout_profiles": {
            "private": LAYOUT_PROFILE,
            "special": SPECIAL_LAYOUT_PROFILE,
        },
        "base_plate_width": BASE_PLATE_SIZE[0],
        "base_plate_height": BASE_PLATE_SIZE[1],
        "output_width": OUTPUT_SIZE[0],
        "output_height": OUTPUT_SIZE[1],
        "label_slots": 8,
        "condition_profiles": list(profiles),
        "condition_profile_versions": {
            profile: CONDITION_PROFILE_VERSIONS.get(
                profile,
                "procedural-v1",
            )
            for profile in profiles
        },
        "focus_letters": list(focus_letters),
        "focus_multiplier": int(focus_multiplier),
        "train_unique_plates": len(training),
        "validation_unique_plates": len(validation),
        "test_unique_plates": len(test),
        "train_letter_counts": {
            letter: sum(plate[2] == letter for plate in training)
            for letter in ALLOWED_PLATE_LETTERS
        },
        "validation_letter_counts": {
            letter: sum(plate[2] == letter for plate in validation)
            for letter in ALLOWED_PLATE_LETTERS
        },
        "test_letter_counts": {
            letter: sum(plate[2] == letter for plate in test)
            for letter in ALLOWED_PLATE_LETTERS
        },
        "identity_overlap": sum(identity_overlaps.values()),
        "identity_overlaps": identity_overlaps,
        "views_per_plate": int(views_per_plate),
        "train_images": train_result["images"],
        "validation_images": validation_result["images"],
        "test_images": test_result["images"] if test_result else 0,
        "train_conditions": train_result["condition_counts"],
        "validation_conditions": validation_result[
            "condition_counts"
        ],
        "test_conditions": (
            test_result["condition_counts"] if test_result else {}
        ),
        "train_difficulty": train_result["difficulty_counts"],
        "validation_difficulty": validation_result[
            "difficulty_counts"
        ],
        "test_difficulty": (
            test_result["difficulty_counts"] if test_result else {}
        ),
        "train_mean_quality_score": train_result[
            "mean_quality_score"
        ],
        "validation_mean_quality_score": validation_result[
            "mean_quality_score"
        ],
        "test_mean_quality_score": (
            test_result["mean_quality_score"] if test_result else 0.0
        ),
        "train_quality_rescues": train_result["quality_rescues"],
        "validation_quality_rescues": validation_result[
            "quality_rescues"
        ],
        "test_quality_rescues": (
            test_result["quality_rescues"] if test_result else 0
        ),
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
    parser.add_argument(
        "--test-plates",
        type=int,
        default=0,
        help=(
            "Optional held-out synthetic test plate identities. Test images "
            "are never used for checkpoint selection."
        ),
    )
    parser.add_argument("--views-per-plate", type=int, default=4)
    parser.add_argument("--font", type=Path, default=None)
    parser.add_argument(
        "--font-license",
        default=None,
        help=(
            "Required when --font is supplied. The built-in default font "
            "uses the DejaVu font license."
        ),
    )
    parser.add_argument(
        "--fallback-font",
        type=Path,
        default=None,
        help=(
            "Optional approved font used only for glyphs missing from the "
            "primary plate font, such as the Latin national-band text."
        ),
    )
    parser.add_argument(
        "--fallback-font-license",
        default=None,
        help="Required together with --fallback-font.",
    )
    parser.add_argument(
        "--profiles",
        default=",".join(CONDITION_PROFILES),
        help=(
            "Comma-separated condition profiles. Available: "
            + ",".join(CONDITION_PROFILES)
        ),
    )
    parser.add_argument(
        "--focus-letters",
        default="".join(FOCUS_LETTERS),
        help="Letters to oversample after one balanced coverage cycle",
    )
    parser.add_argument(
        "--focus-multiplier",
        type=int,
        default=3,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--jpeg-quality", type=int, default=82)
    args = parser.parse_args(argv)
    if args.font is not None and not str(args.font_license or "").strip():
        parser.error("--font-license is required when --font is supplied")
    if args.font is None and str(args.font_license or "").strip():
        parser.error("--font-license may be set only together with --font")
    if args.fallback_font is not None and not str(
        args.fallback_font_license or ""
    ).strip():
        parser.error(
            "--fallback-font-license is required with --fallback-font"
        )
    if args.fallback_font is None and str(
        args.fallback_font_license or ""
    ).strip():
        parser.error(
            "--fallback-font-license may be set only with --fallback-font"
        )
    font_license = (
        args.font_license
        if args.font is not None
        else "DejaVu-font-license"
    )
    manifest = generate(
        output=args.output,
        train_plates=max(20, int(args.train_plates)),
        validation_plates=max(5, int(args.validation_plates)),
        views_per_plate=max(1, min(12, int(args.views_per_plate))),
        font=(args.font or _default_font()).resolve(),
        font_license=font_license,
        seed=int(args.seed),
        jpeg_quality=max(45, min(98, int(args.jpeg_quality))),
        profiles=tuple(
            item.strip()
            for item in str(args.profiles).split(",")
            if item.strip()
        ),
        focus_letters=tuple(dict.fromkeys(str(args.focus_letters))),
        focus_multiplier=max(1, min(12, int(args.focus_multiplier))),
        test_plates=max(0, int(args.test_plates)),
        fallback_font=(
            args.fallback_font.resolve()
            if args.fallback_font is not None
            else None
        ),
        fallback_font_license=args.fallback_font_license,
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
