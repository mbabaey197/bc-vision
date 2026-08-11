from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class QualityBreakdown:
    score: float
    sharpness: float
    exposure: float
    contrast: float


def evaluate_plate_quality(crop: np.ndarray) -> QualityBreakdown:
    if crop.size == 0:
        return QualityBreakdown(0.0, 0.0, 0.0, 0.0)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = gray.astype(np.uint8, copy=False)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = min(1.0, lap_var / 350.0)

    mean = float(gray.mean())
    exposure = max(0.0, 1.0 - abs(mean - 135.0) / 135.0)

    std = float(gray.std())
    contrast = min(1.0, std / 65.0)

    score = 0.55 * sharpness + 0.25 * exposure + 0.20 * contrast
    return QualityBreakdown(float(score), sharpness, exposure, contrast)
