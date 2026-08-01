"""Cheap per-camera motion wake-up and fixed-overlay suppression.

The live worker observes every submitted display frame at a small resolution.
Expensive detector/OCR inference can therefore stay throttled while an empty
scene is idle, but motion immediately wakes it.  Stable high-frequency pixels
inside common CCTV overlay bands are exposed as an exclusion mask; current
motion always clears the mask so a moving plate is never hidden by it.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class FrameActivity:
    motion_score: float
    moving: bool
    scene_change: bool
    wake_inference: bool
    exclusion_mask: np.ndarray | None


class FrameActivityAnalyzer:
    """Track low-resolution motion and temporally stable overlay edges."""

    def __init__(
        self,
        max_width=320,
        motion_threshold=0.012,
        scene_change_threshold=0.34,
        overlay_warmup_frames=8,
    ):
        self.max_width = max(96, int(max_width))
        self.motion_threshold = min(
            0.25,
            max(0.002, float(motion_threshold)),
        )
        self.scene_change_threshold = min(
            0.90,
            max(0.10, float(scene_change_threshold)),
        )
        self.overlay_warmup_frames = max(
            4,
            int(overlay_warmup_frames),
        )
        self._previous: np.ndarray | None = None
        self._static_edge_score: np.ndarray | None = None
        self._frames = 0

    @staticmethod
    def _overlay_zones(height: int, width: int) -> np.ndarray:
        """Return conservative bands used by single and four-view NVR feeds."""

        zones = np.zeros((height, width), dtype=np.uint8)
        horizontal = max(4, int(round(height * 0.11)))
        vertical = max(4, int(round(width * 0.08)))
        zones[:horizontal, :] = 255
        zones[-horizontal:, :] = 255
        zones[:, :vertical] = 255
        zones[:, -vertical:] = 255

        # Four-camera composites often repeat clock/name overlays around the
        # central split. Keep these bands narrow and let motion clear them.
        middle_y = height // 2
        middle_x = width // 2
        center_h = max(2, int(round(height * 0.035)))
        center_w = max(2, int(round(width * 0.025)))
        zones[
            max(0, middle_y - center_h):
            min(height, middle_y + center_h),
            :,
        ] = 255
        zones[
            :,
            max(0, middle_x - center_w):
            min(width, middle_x + center_w),
        ] = 255
        return zones

    def _small_gray(self, frame) -> np.ndarray:
        height, width = frame.shape[:2]
        scale = min(1.0, self.max_width / max(1, width))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (
                    max(1, int(round(width * scale))),
                    max(1, int(round(height * scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        gray = (
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if frame.ndim == 3
            else frame
        )
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def observe(self, frame) -> FrameActivity:
        if frame is None or getattr(frame, "size", 0) == 0:
            return FrameActivity(0.0, False, False, False, None)

        original_height, original_width = frame.shape[:2]
        gray = self._small_gray(frame)
        if (
            self._previous is None
            or self._previous.shape != gray.shape
        ):
            self._previous = gray
            self._static_edge_score = np.zeros(
                gray.shape,
                dtype=np.float32,
            )
            self._frames = 1
            return FrameActivity(1.0, True, True, True, None)

        difference = cv2.absdiff(gray, self._previous)
        motion = cv2.threshold(
            difference,
            16,
            255,
            cv2.THRESH_BINARY,
        )[1]
        motion = cv2.morphologyEx(
            motion,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )
        motion = cv2.dilate(
            motion,
            np.ones((5, 5), dtype=np.uint8),
            iterations=1,
        )
        motion_score = float(cv2.countNonZero(motion)) / max(
            1,
            motion.size,
        )
        scene_change = motion_score >= self.scene_change_threshold
        moving = motion_score >= self.motion_threshold

        edges = cv2.Canny(gray, 70, 160)
        unchanged = difference <= 4
        static_edges = np.logical_and(edges > 0, unchanged).astype(
            np.float32
        )
        if self._static_edge_score is None:
            self._static_edge_score = static_edges
        else:
            self._static_edge_score = (
                self._static_edge_score * 0.84
                + static_edges * 0.16
            )

        self._frames += 1
        exclusion_mask = None
        if self._frames >= self.overlay_warmup_frames:
            stable = (
                self._static_edge_score >= 0.72
            ).astype(np.uint8) * 255
            stable = cv2.dilate(
                stable,
                cv2.getStructuringElement(
                    cv2.MORPH_RECT,
                    (11, 5),
                ),
                iterations=2,
            )
            stable = cv2.bitwise_and(
                stable,
                self._overlay_zones(*gray.shape),
            )
            # A current moving object always wins over the learned static mask.
            stable[motion > 0] = 0
            exclusion_mask = cv2.resize(
                stable,
                (original_width, original_height),
                interpolation=cv2.INTER_NEAREST,
            )

        self._previous = gray
        return FrameActivity(
            motion_score=round(motion_score, 6),
            moving=moving,
            scene_change=scene_change,
            wake_inference=bool(moving or scene_change),
            exclusion_mask=exclusion_mask,
        )


def masked_bbox_ratio(mask, bbox) -> float:
    """Return the excluded-pixel share inside one clipped rectangle."""

    if mask is None or getattr(mask, "size", 0) == 0 or not bbox:
        return 0.0
    height, width = mask.shape[:2]
    x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mask[y1:y2, x1:x2]
    return float(np.count_nonzero(region)) / max(1, region.size)


def suppress_static_overlay_rows(
    rows,
    exclusion_mask,
    maximum_overlap=0.22,
) -> list[dict]:
    """Drop only detections dominated by the learned fixed-overlay mask."""

    selected = []
    for raw in rows:
        row = dict(raw)
        overlap = masked_bbox_ratio(exclusion_mask, row.get("bbox"))
        row["static_overlay_overlap"] = round(overlap, 5)
        if overlap >= float(maximum_overlap):
            continue
        selected.append(row)
    return selected
