from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _clamp(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True, slots=True)
class QualityBreakdown:
    score: float
    sharpness: float
    exposure: float
    contrast: float
    plate_size: float = 0.0
    detector_confidence: float = 0.0
    motion_blur: float = 0.0


@dataclass(slots=True)
class PlateFrame:
    crop: np.ndarray
    bbox: tuple[int, int, int, int]
    seq: int
    ts: float
    detector_confidence: float
    quality: QualityBreakdown


def evaluate_plate_quality(
    crop: np.ndarray,
    *,
    detector_confidence: float = 0.0,
    frame_shape: tuple[int, int] | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> QualityBreakdown:
    """Score an evidence crop without changing it.

    The score deliberately combines independent failure modes. A sharp but
    clipped crop, or a confident tiny detection, cannot dominate the ranking.
    All component scores are normalized to ``0..1`` so weights remain easy to
    calibrate against a real validation set later.
    """

    if crop.size == 0:
        return QualityBreakdown(0.0, 0.0, 0.0, 0.0)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    gray = np.nan_to_num(gray).astype(np.uint8, copy=False)

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = _clamp(lap_var / (lap_var + 140.0))

    mean = float(gray.mean())
    centered_exposure = max(0.0, 1.0 - abs(mean - 132.0) / 132.0)
    clipped_ratio = float(np.count_nonzero((gray <= 8) | (gray >= 247))) / float(gray.size)
    exposure = _clamp(centered_exposure * (1.0 - min(0.85, clipped_ratio)))

    std = float(gray.std())
    contrast = _clamp(std / (std + 38.0))

    gx = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))))
    gy = float(np.mean(np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))))
    directional_bias = abs(gx - gy) / max(1e-6, gx + gy)
    motion_blur = _clamp(0.70 * sharpness + 0.30 * (1.0 - directional_bias))

    size_score = 0.0
    if frame_shape is not None:
        frame_h, frame_w = frame_shape[:2]
        if bbox is None:
            crop_h, crop_w = gray.shape[:2]
            area = crop_h * crop_w
        else:
            x1, y1, x2, y2 = bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
        ratio = float(area) / float(max(1, frame_h * frame_w))
        # Roughly 3% of a full evidence frame is already a strong OCR crop.
        size_score = _clamp(ratio / 0.03)

    detector_score = _clamp(float(detector_confidence))
    score = (
        0.30 * sharpness
        + 0.15 * exposure
        + 0.12 * contrast
        + 0.12 * size_score
        + 0.18 * detector_score
        + 0.13 * motion_blur
    )
    return QualityBreakdown(
        score=_clamp(score),
        sharpness=sharpness,
        exposure=exposure,
        contrast=contrast,
        plate_size=size_score,
        detector_confidence=detector_score,
        motion_blur=motion_blur,
    )


class BestPlateFrameSelector:
    """Keep only a small, quality-ranked set of OCR candidates per track."""

    def __init__(self, capacity: int = 5, min_sequence_gap: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self.min_sequence_gap = max(0, int(min_sequence_gap))
        self._frames: list[PlateFrame] = []

    def __len__(self) -> int:
        return len(self._frames)

    @property
    def best(self) -> PlateFrame | None:
        return self._frames[0] if self._frames else None

    def clear(self) -> None:
        self._frames.clear()

    def add(
        self,
        crop: np.ndarray,
        *,
        bbox: tuple[int, int, int, int],
        seq: int,
        ts: float,
        detector_confidence: float,
        frame_shape: tuple[int, int],
    ) -> PlateFrame | None:
        if crop.size == 0:
            return None
        quality = evaluate_plate_quality(
            crop,
            detector_confidence=detector_confidence,
            frame_shape=frame_shape,
            bbox=bbox,
        )
        candidate = PlateFrame(
            crop=crop.copy(),
            bbox=bbox,
            seq=int(seq),
            ts=float(ts),
            detector_confidence=float(detector_confidence),
            quality=quality,
        )

        near_indices = [
            index for index, current in enumerate(self._frames)
            if abs(current.seq - candidate.seq) <= self.min_sequence_gap
        ]
        if near_indices:
            # Temporal diversity is a hard invariant: a new candidate may
            # replace all nearby weaker frames, but it may not coexist with a
            # nearby frame merely because a different nearby frame was weaker.
            if any(self._frames[index].quality.score >= quality.score for index in near_indices):
                return None
            for index in sorted(near_indices, reverse=True):
                del self._frames[index]

        self._frames.append(candidate)
        self._frames.sort(key=lambda frame: (frame.quality.score, frame.seq), reverse=True)
        del self._frames[self.capacity :]
        return candidate if any(frame is candidate for frame in self._frames) else None

    def selected(self, limit: int = 3, min_quality: float = 0.0) -> list[PlateFrame]:
        if int(limit) <= 0:
            return []
        return [
            frame for frame in self._frames
            if frame.quality.score >= float(min_quality)
        ][: int(limit)]
