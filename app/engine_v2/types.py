from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import numpy as np


class TrackPhase(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    TRACKING = "tracking"
    PLATE_FOUND = "plate_found"
    COLLECTING = "collecting"
    OCR = "ocr"
    # Compatibility alias for the first V2 slice. New code uses ``OCR``.
    OCR_PENDING = "ocr"
    VALIDATED = "validated"
    DONE = "done"


@dataclass(slots=True)
class FramePacket:
    camera_id: str
    seq: int
    ts: float
    frame: np.ndarray
    detector_frame: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def main_frame(self) -> np.ndarray:
        """High-resolution evidence frame.

        ``frame`` keeps its original name for compatibility with the first V2
        slice. Producers may supply an independently decoded detector frame.
        """

        return self.frame


@dataclass(slots=True)
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0
    track_hint: str | None = None


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float
    valid: bool = True
    character_confidences: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlateEvent:
    camera_id: str
    frame_seq: int
    ts: float
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    quality: float
    track_id: str | None = None
    episode_id: str | None = None
    observations: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


class PlateDetector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[PlateCandidate]: ...


class PlateOCR(Protocol):
    def read(self, plate_crop: np.ndarray) -> OCRResult: ...
