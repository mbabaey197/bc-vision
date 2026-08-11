from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence

import numpy as np


class TrackPhase(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    PLATE_FOUND = "plate_found"
    OCR_PENDING = "ocr_pending"
    DONE = "done"


@dataclass(slots=True)
class FramePacket:
    camera_id: str
    seq: int
    ts: float
    frame: np.ndarray
    detector_frame: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PlateCandidate:
    bbox: tuple[int, int, int, int]
    confidence: float
    class_id: int = 0


@dataclass(slots=True)
class OCRResult:
    text: str
    confidence: float
    valid: bool = True


@dataclass(slots=True)
class PlateEvent:
    camera_id: str
    frame_seq: int
    ts: float
    text: str
    confidence: float
    bbox: tuple[int, int, int, int]
    quality: float


class PlateDetector(Protocol):
    def detect(self, frame: np.ndarray) -> Sequence[PlateCandidate]: ...


class PlateOCR(Protocol):
    def read(self, plate_crop: np.ndarray) -> OCRResult: ...
