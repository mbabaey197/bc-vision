"""Create a local, EXIF-free plate-renderer comparison sheet.

Reference images are QA inputs only.  This tool never adds them to a training
manifest and requires the caller to declare whether the supplied plate text is
an operator-confirmed label or an unrelated synthetic dummy identity.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.plate_rules import plausible_plate
from tools.generate_cct_synthetic_dataset import _base_plate
from tools.iran_plate_renderer import BASE_PLATE_SIZE


def parse_corners(value: str) -> np.ndarray:
    """Parse TL,TR,BR,BL pixel corners from ``x,y x,y x,y x,y``."""

    points = []
    for raw_point in str(value or "").split():
        values = raw_point.split(",")
        if len(values) != 2:
            raise ValueError("Each corner must use x,y syntax")
        points.append((float(values[0]), float(values[1])))
    if len(points) != 4:
        raise ValueError("Exactly four corners are required: TL TR BR BL")
    array = np.asarray(points, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Corner coordinates must be finite")
    return array


def rectify_plate(
    image: np.ndarray,
    corners: np.ndarray,
    output_size: tuple[int, int] = BASE_PLATE_SIZE,
) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Reference image is empty")
    corners = np.asarray(corners, dtype=np.float32)
    if corners.shape != (4, 2):
        raise ValueError("Corners must have shape (4, 2)")
    height, width = image.shape[:2]
    if (
        (corners[:, 0] < 0).any()
        or (corners[:, 0] >= width).any()
        or (corners[:, 1] < 0).any()
        or (corners[:, 1] >= height).any()
    ):
        raise ValueError("Reference corners fall outside the image")
    target_width, target_height = output_size
    target = np.asarray([
        [0, 0],
        [target_width - 1, 0],
        [target_width - 1, target_height - 1],
        [0, target_height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(corners, target)
    return cv2.warpPerspective(
        image,
        transform,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _header(
    image: Image.Image,
    title: str,
    x: int,
    width: int,
) -> None:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=18)
    bounds = draw.textbbox((0, 0), title, font=font)
    text_width = bounds[2] - bounds[0]
    draw.text(
        (x + (width - text_width) / 2, 8),
        title,
        font=font,
        fill=(30, 34, 39),
    )


def create_comparison(
    reference_image: Path,
    corners: np.ndarray,
    plate_text: str,
    label_status: str,
    primary_font: Path,
    fallback_font: Path | None,
    output: Path,
) -> Path:
    if label_status not in {"dummy", "operator-confirmed"}:
        raise ValueError("Label status must be dummy or operator-confirmed")
    if not plausible_plate(plate_text):
        raise ValueError("Comparison plate text is not a canonical plate")
    reference = cv2.imread(str(reference_image), cv2.IMREAD_COLOR)
    if reference is None:
        raise FileNotFoundError(reference_image)
    rectified = rectify_plate(reference, corners)
    rectified = cv2.cvtColor(rectified, cv2.COLOR_BGR2RGB)
    synthetic = _base_plate(
        plate_text,
        primary_font,
        random.Random(20260730),
        fallback_font_path=fallback_font,
        calibration=True,
    )

    panel_width, panel_height = BASE_PLATE_SIZE
    header_height = 38
    gutter = 18
    sheet = Image.new(
        "RGB",
        (
            panel_width * 2 + gutter * 3,
            panel_height + header_height + gutter,
        ),
        (239, 242, 246),
    )
    left_x = gutter
    right_x = panel_width + gutter * 2
    sheet.paste(Image.fromarray(rectified), (left_x, header_height))
    sheet.paste(synthetic, (right_x, header_height))
    _header(sheet, "REFERENCE - QA ONLY", left_x, panel_width)
    _header(
        sheet,
        (
            "SYNTHETIC - OPERATOR CONFIRMED"
            if label_status == "operator-confirmed"
            else "SYNTHETIC - DUMMY IDENTITY"
        ),
        right_x,
        panel_width,
    )
    draw = ImageDraw.Draw(sheet)
    draw.rectangle(
        (
            left_x - 1,
            header_height - 1,
            left_x + panel_width,
            header_height + panel_height,
        ),
        outline=(123, 132, 142),
        width=1,
    )
    draw.rectangle(
        (
            right_x - 1,
            header_height - 1,
            right_x + panel_width,
            header_height + panel_height,
        ),
        outline=(123, 132, 142),
        width=1,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG", optimize=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a rectified plate reference with the renderer",
    )
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument(
        "--corners",
        required=True,
        help='TL,TR,BR,BL as "x,y x,y x,y x,y"',
    )
    parser.add_argument("--plate-text", required=True)
    parser.add_argument(
        "--label-status",
        required=True,
        choices=("dummy", "operator-confirmed"),
    )
    parser.add_argument("--primary-font", type=Path, required=True)
    parser.add_argument("--fallback-font", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = create_comparison(
        reference_image=args.reference_image,
        corners=parse_corners(args.corners),
        plate_text=args.plate_text,
        label_status=args.label_status,
        primary_font=args.primary_font,
        fallback_font=args.fallback_font,
        output=args.output,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
