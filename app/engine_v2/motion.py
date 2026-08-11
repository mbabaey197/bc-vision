from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class MotionGateConfig:
    pixel_threshold: int = 22
    changed_ratio_threshold: float = 0.018
    baseline_alpha: float = 0.03
    blur_kernel: int = 5


class AdaptiveMotionGate:
    """Very cheap wake-up gate for fixed cameras.

    It learns a slow background and emits only when enough pixels in the ROI
    change. This is intentionally not an object detector; it exists to keep
    expensive AI asleep while the lane is quiet.
    """

    def __init__(self, config: MotionGateConfig | None = None) -> None:
        self.config = config or MotionGateConfig()
        self._baseline: np.ndarray | None = None

    def reset(self) -> None:
        self._baseline = None

    def score(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> float:
        if roi is not None:
            x1, y1, x2, y2 = roi
            frame = frame[y1:y2, x1:x2]
        if frame.size == 0:
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        k = max(1, int(self.config.blur_kernel))
        if k % 2 == 0:
            k += 1
        if k > 1:
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        current = gray.astype(np.float32)

        if self._baseline is None or self._baseline.shape != current.shape:
            self._baseline = current.copy()
            return 0.0

        delta = cv2.absdiff(current, self._baseline)
        changed = delta >= float(self.config.pixel_threshold)
        ratio = float(np.count_nonzero(changed)) / float(changed.size)
        cv2.accumulateWeighted(current, self._baseline, self.config.baseline_alpha)
        return ratio

    def should_wake(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> bool:
        return self.score(frame, roi) >= self.config.changed_ratio_threshold
