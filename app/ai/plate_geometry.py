"""Fail-closed plate quadrilateral normalization shared by ANPR engines."""
from __future__ import annotations

import cv2
import numpy as np


def order_quad_points(points) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError("Plate quadrilateral must contain four finite points")
    if len(np.unique(values, axis=0)) != 4:
        raise ValueError("Plate quadrilateral points must be unique")
    hull = cv2.convexHull(values).reshape(-1, 2)
    if hull.shape != (4, 2) or cv2.contourArea(hull) <= 1.0:
        raise ValueError("Plate quadrilateral must be convex")

    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = values.sum(axis=1)
    differences = np.diff(values, axis=1).reshape(-1)
    indices = (
        int(np.argmin(sums)),
        int(np.argmin(differences)),
        int(np.argmax(sums)),
        int(np.argmax(differences)),
    )
    if len(set(indices)) != 4:
        raise ValueError("Plate quadrilateral ordering is ambiguous")
    for target, source in enumerate(indices):
        ordered[target] = values[source]
    if cv2.contourArea(ordered) <= 1.0:
        raise ValueError("Plate quadrilateral is degenerate")
    return ordered


def rectify_plate_quad(
    image,
    corners,
    *,
    expand_x=1.08,
    expand_y=1.16,
) -> np.ndarray | None:
    """Warp a valid plate quadrilateral without changing its aspect ratio."""

    if image is None or getattr(image, "size", 0) == 0:
        return None
    try:
        source = order_quad_points(corners)
    except (TypeError, ValueError):
        return None
    height, width = image.shape[:2]
    if (
        np.any(source[:, 0] < -0.25 * width)
        or np.any(source[:, 0] > 1.25 * width)
        or np.any(source[:, 1] < -0.25 * height)
        or np.any(source[:, 1] > 1.25 * height)
    ):
        return None

    center = source.mean(axis=0)
    horizontal = (
        (source[1] - source[0])
        + (source[2] - source[3])
    ) / 2.0
    vertical = (
        (source[3] - source[0])
        + (source[2] - source[1])
    ) / 2.0
    horizontal_norm = float(np.linalg.norm(horizontal))
    vertical_norm = float(np.linalg.norm(vertical))
    if horizontal_norm <= 1e-6 or vertical_norm <= 1e-6:
        return None
    basis = np.column_stack((
        horizontal / horizontal_norm,
        vertical / vertical_norm,
    )).astype(np.float32)
    if abs(float(np.linalg.det(basis))) < 0.12:
        return None
    local = np.linalg.solve(
        basis,
        (source - center).T,
    ).T
    local *= np.array(
        [float(expand_x), float(expand_y)],
        dtype=np.float32,
    )
    source = center + local @ basis.T
    source[:, 0] = np.clip(source[:, 0], 0, width - 1)
    source[:, 1] = np.clip(source[:, 1], 0, height - 1)
    if len(np.unique(source, axis=0)) != 4:
        return None

    top = float(np.linalg.norm(source[1] - source[0]))
    bottom = float(np.linalg.norm(source[2] - source[3]))
    left = float(np.linalg.norm(source[3] - source[0]))
    right = float(np.linalg.norm(source[2] - source[1]))
    plate_width = max(top, bottom)
    plate_height = max(left, right)
    if plate_height > plate_width:
        source = np.array(
            [source[3], source[0], source[1], source[2]],
            dtype=np.float32,
        )
        plate_width, plate_height = plate_height, plate_width
    ratio = plate_width / max(plate_height, 1e-9)
    if (
        plate_width < 12
        or plate_height < 5
        or not 1.7 <= ratio <= 8.8
    ):
        return None

    scale = max(
        1.0,
        64.0 / plate_width,
        24.0 / plate_height,
    )
    scale = min(
        scale,
        512.0 / plate_width,
        192.0 / plate_height,
    )
    target_width = max(1, int(round(plate_width * scale)))
    target_height = max(1, int(round(plate_height * scale)))
    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        return None
    crop = cv2.warpPerspective(
        image,
        matrix,
        (target_width, target_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop if crop is not None and crop.size else None
