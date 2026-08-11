from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .motion import AdaptiveMotionGate
from .quality import evaluate_plate_quality
from .scheduler import LatestOnlyPriorityQueue
from .types import FramePacket, OCRResult, PlateCandidate, PlateDetector, PlateEvent, PlateOCR, TrackPhase


@dataclass(slots=True)
class EngineV2Config:
    idle_stride: int = 8
    active_stride: int = 2
    min_detector_confidence: float = 0.30
    min_ocr_confidence: float = 0.55
    min_quality: float = 0.18
    done_cooldown_frames: int = 25
    queue_size: int = 128


@dataclass(slots=True)
class CameraState:
    phase: TrackPhase = TrackPhase.IDLE
    last_detection_seq: int = -1
    last_done_seq: int = -10_000
    best_quality: float = 0.0
    best_crop: np.ndarray | None = None
    best_bbox: tuple[int, int, int, int] | None = None
    observations: list[OCRResult] = field(default_factory=list)

    def reset(self) -> None:
        self.phase = TrackPhase.IDLE
        self.best_quality = 0.0
        self.best_crop = None
        self.best_bbox = None
        self.observations.clear()


class EventDrivenANPREngine:
    """ANPR V2 core: camera producers + one shared detector/OCR path.

    This class deliberately contains no RTSP ownership and no per-camera model
    sessions. Camera readers push frames; the central queue keeps the newest
    frame for each camera and expensive inference is shared.
    """

    def __init__(
        self,
        detector: PlateDetector,
        ocr: PlateOCR,
        config: EngineV2Config | None = None,
        on_event: Callable[[PlateEvent], None] | None = None,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.config = config or EngineV2Config()
        self.on_event = on_event
        self.queue: LatestOnlyPriorityQueue[FramePacket] = LatestOnlyPriorityQueue(self.config.queue_size)
        self._states: dict[str, CameraState] = {}
        self._gates: dict[str, AdaptiveMotionGate] = {}
        self._rois: dict[str, tuple[int, int, int, int] | None] = {}

    def set_roi(self, camera_id: str, roi: tuple[int, int, int, int] | None) -> None:
        self._rois[camera_id] = roi

    def state_for(self, camera_id: str) -> CameraState:
        return self._states.setdefault(camera_id, CameraState())

    def submit_frame(self, packet: FramePacket) -> bool:
        state = self.state_for(packet.camera_id)
        gate = self._gates.setdefault(packet.camera_id, AdaptiveMotionGate())
        roi = self._rois.get(packet.camera_id)
        gate_frame = packet.detector_frame if packet.detector_frame is not None else packet.frame

        if state.phase is TrackPhase.DONE:
            if packet.seq - state.last_done_seq < self.config.done_cooldown_frames:
                return False
            state.reset()

        if state.phase is TrackPhase.IDLE:
            if packet.seq % max(1, self.config.idle_stride) != 0:
                return False
            if not gate.should_wake(gate_frame, roi):
                return False
            state.phase = TrackPhase.ACTIVE
            priority = 10
        else:
            if packet.seq % max(1, self.config.active_stride) != 0:
                return False
            priority = 20

        self.queue.submit(packet.camera_id, packet, priority=priority)
        return True

    def process_next(self) -> PlateEvent | None:
        packet = self.queue.pop()
        if packet is None:
            return None

        state = self.state_for(packet.camera_id)
        detector_frame = packet.detector_frame if packet.detector_frame is not None else packet.frame
        candidates = [
            c for c in self.detector.detect(detector_frame)
            if c.confidence >= self.config.min_detector_confidence
        ]
        state.last_detection_seq = packet.seq
        if not candidates:
            return None

        raw_candidate = max(candidates, key=lambda c: c.confidence)
        candidate = self._map_candidate(raw_candidate, detector_frame.shape[:2], packet.frame.shape[:2])
        crop = self._crop(packet.frame, candidate)
        quality = evaluate_plate_quality(crop).score
        if quality > state.best_quality:
            state.best_quality = quality
            state.best_crop = crop.copy() if crop.size else None
            state.best_bbox = candidate.bbox
        state.phase = TrackPhase.PLATE_FOUND

        if state.best_crop is None or state.best_quality < self.config.min_quality:
            return None

        state.phase = TrackPhase.OCR_PENDING
        result = self.ocr.read(state.best_crop)
        state.observations.append(result)
        if not result.valid or result.confidence < self.config.min_ocr_confidence or not result.text.strip():
            state.phase = TrackPhase.PLATE_FOUND
            return None

        event = PlateEvent(
            camera_id=packet.camera_id,
            frame_seq=packet.seq,
            ts=packet.ts,
            text=result.text.strip(),
            confidence=float(result.confidence),
            bbox=state.best_bbox or candidate.bbox,
            quality=float(state.best_quality),
        )
        state.phase = TrackPhase.DONE
        state.last_done_seq = packet.seq
        if self.on_event is not None:
            self.on_event(event)
        return event

    @staticmethod
    def _map_candidate(
        candidate: PlateCandidate,
        source_hw: tuple[int, int],
        target_hw: tuple[int, int],
    ) -> PlateCandidate:
        sh, sw = source_hw
        th, tw = target_hw
        if sw <= 0 or sh <= 0 or (sw == tw and sh == th):
            return candidate
        sx = tw / float(sw)
        sy = th / float(sh)
        x1, y1, x2, y2 = candidate.bbox
        return PlateCandidate(
            (round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)),
            candidate.confidence,
            candidate.class_id,
        )

    @staticmethod
    def _crop(frame: np.ndarray, candidate: PlateCandidate) -> np.ndarray:
        x1, y1, x2, y2 = candidate.bbox
        h, w = frame.shape[:2]
        x1, x2 = sorted((max(0, min(w, int(x1))), max(0, min(w, int(x2)))))
        y1, y2 = sorted((max(0, min(h, int(y1))), max(0, min(h, int(y2)))))
        return frame[y1:y2, x1:x2]
