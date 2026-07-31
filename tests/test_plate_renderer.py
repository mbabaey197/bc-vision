from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest

from tools.compare_plate_renderer import (
    create_comparison,
    parse_corners,
    rectify_plate,
)
from tools.iran_plate_renderer import BASE_PLATE_SIZE


def _font() -> Path:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    )
    if not path.is_file():
        pytest.skip("DejaVu Sans is not installed")
    return path


def test_parse_reference_corners_requires_tl_tr_br_bl():
    points = parse_corners("10,20 110,21 109,60 9,59")

    assert points.shape == (4, 2)
    assert points.dtype == np.float32
    with pytest.raises(ValueError, match="Exactly four"):
        parse_corners("10,20 110,21 109,60")
    with pytest.raises(ValueError, match="x,y"):
        parse_corners("10,20 110 109,60 9,59")


def test_rectify_reference_produces_renderer_dimensions():
    image = np.zeros((100, 180, 3), dtype=np.uint8)
    image[20:80, 20:160] = (230, 230, 230)
    corners = parse_corners("20,20 159,20 159,79 20,79")

    rectified = rectify_plate(image, corners)

    assert rectified.shape == (
        BASE_PLATE_SIZE[1],
        BASE_PLATE_SIZE[0],
        3,
    )
    assert rectified.mean() > 200


def test_comparison_sheet_is_png_without_reference_metadata(tmp_path):
    reference = tmp_path / "reference.jpg"
    image = np.full((100, 180, 3), 190, dtype=np.uint8)
    assert cv2.imwrite(str(reference), image)
    output = tmp_path / "comparison.png"

    result = create_comparison(
        reference_image=reference,
        corners=parse_corners("20,20 159,20 159,79 20,79"),
        plate_text="28ف46195",
        label_status="dummy",
        primary_font=_font(),
        fallback_font=None,
        output=output,
    )

    assert result == output
    with Image.open(output) as rendered:
        assert rendered.format == "PNG"
        assert rendered.getexif() == {}
        assert rendered.size == (
            BASE_PLATE_SIZE[0] * 2 + 54,
            BASE_PLATE_SIZE[1] + 56,
        )
