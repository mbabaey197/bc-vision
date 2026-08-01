"""Conservative restoration for mildly blurred license-plate crops.

The restoration pass is intentionally conditional.  The dedicated character
model always reads the original crop first, and this module only produces a
second candidate when the original is soft or uncertain.  The caller must
still validate the restored candidate with the plate model.
"""
from __future__ import annotations

import math

import cv2
import numpy as np


def blur_score(image: np.ndarray) -> float:
    """Return Laplacian variance; lower values generally mean a softer crop."""

    if image is None or getattr(image, "size", 0) == 0:
        return 0.0
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def should_attempt_recovery(
    image: np.ndarray,
    text: str,
    confidence: float,
) -> bool:
    """Gate the extra inference pass to soft or low-confidence plate crops."""

    if image is None or getattr(image, "size", 0) == 0:
        return False
    height, width = image.shape[:2]
    if height < 12 or width < 40:
        return False
    sharpness = blur_score(image)
    # Extremely soft crops do not contain enough evidence for a trustworthy
    # reconstruction.  Refusing them is safer than hallucinating characters.
    if sharpness < 22.0:
        return False
    confidence = float(confidence)
    return (
        (sharpness < 340.0 and (not text or confidence < 0.82))
        or (not text and sharpness < 650.0)
        or (confidence < 0.72 and sharpness < 500.0)
    )


def _motion_angle(gray: np.ndarray) -> float:
    """Estimate the dominant local smear direction from the structure tensor."""

    work = gray.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=3)
    tensor = np.array(
        [
            [np.mean(gradient_x * gradient_x), np.mean(gradient_x * gradient_y)],
            [np.mean(gradient_x * gradient_y), np.mean(gradient_y * gradient_y)],
        ],
        dtype=np.float64,
    )
    _, vectors = np.linalg.eigh(tensor)
    direction = vectors[:, 0]
    return math.degrees(math.atan2(direction[1], direction[0]))


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    size = max(3, int(length) | 1)
    kernel = np.zeros((size, size), dtype=np.float32)
    center = (size - 1) / 2.0
    radius = center
    radians = math.radians(angle)
    dx = math.cos(radians) * radius
    dy = math.sin(radians) * radius
    cv2.line(
        kernel,
        (int(round(center - dx)), int(round(center - dy))),
        (int(round(center + dx)), int(round(center + dy))),
        1.0,
        1,
        cv2.LINE_AA,
    )
    total = float(kernel.sum())
    if total <= 0:
        kernel[int(center), int(center)] = 1.0
        total = 1.0
    return kernel / total


def _richardson_lucy(
    channel: np.ndarray,
    kernel: np.ndarray,
    iterations: int,
) -> np.ndarray:
    observed = channel.astype(np.float32) / 255.0
    estimate = np.maximum(observed, 1e-3)
    mirrored = cv2.flip(kernel, -1)
    for _ in range(iterations):
        blurred = cv2.filter2D(
            estimate,
            -1,
            kernel,
            borderType=cv2.BORDER_REFLECT,
        )
        relative = observed / np.maximum(blurred, 1e-4)
        correction = cv2.filter2D(
            relative,
            -1,
            mirrored,
            borderType=cv2.BORDER_REFLECT,
        )
        estimate = np.clip(estimate * correction, 0.0, 1.0)
    return np.round(estimate * 255.0).astype(np.uint8)


def recover_mild_blur(image: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deblur a crop without inventing pixels; AI validation happens later."""

    if image is None or getattr(image, "size", 0) == 0:
        return image, {"applied": False, "method": ""}
    gray = (
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.ndim == 3
        else image
    )
    sharpness = blur_score(gray)
    angle = _motion_angle(gray)
    length = 7 if sharpness < 180.0 else 5
    kernel = _motion_kernel(length, angle)
    if image.ndim == 2:
        restored = _richardson_lucy(image, kernel, iterations=8)
    else:
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        luminance[:, :, 0] = _richardson_lucy(
            luminance[:, :, 0],
            kernel,
            iterations=8,
        )
        restored = cv2.cvtColor(luminance, cv2.COLOR_YCrCb2BGR)
    return restored, {
        "applied": True,
        "method": "motion-deblur+ai-reread",
        "input_blur": round(sharpness, 3),
        "angle": round(angle, 2),
        "kernel": length,
    }
