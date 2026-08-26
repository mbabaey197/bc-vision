from __future__ import annotations

import numpy as np

from app.engine_v2 import (
    EngineV2Config,
    EventDrivenANPREngine,
    FramePacket,
    OCRResult,
    PlateCandidate,
)
from app.engine_v2.tcam import RecognitionPhase, TemporalFusionConfig

PLATE = "12ب34567"


class _Detector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        height, width = frame.shape[:2]
        return [
            PlateCandidate(
                (width // 4, height // 3, 3 * width // 4, 2 * height // 3), 0.95
            )
        ]


class _OCR:
    def __init__(self, confidence: float) -> None:
        self.confidence = confidence
        self.calls = 0

    def read(self, crop: np.ndarray) -> OCRResult:
        del crop
        self.calls += 1
        return OCRResult(PLATE, self.confidence, True)


def _frame(value: int) -> np.ndarray:
    image = np.full((120, 240, 3), value, dtype=np.uint8)
    image[40:80, 60:180:2] = min(255, value + 100)
    return image


def _engine(ocr: _OCR) -> EventDrivenANPREngine:
    return EventDrivenANPREngine(
        _Detector(),
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            load_control_enabled=False,
            track_temporal_fusion_enabled=True,
            temporal_fusion=TemporalFusionConfig(max_ocr_attempts=4),
        ),
    )


def test_runtime_ocr_is_immediate_but_75_percent_result_stays_provisional() -> None:
    ocr = _OCR(0.80)
    engine = _engine(ocr)
    assert engine.submit_frame(FramePacket("cam", 1, 1.0, _frame(0))) is False

    assert engine.submit_frame(FramePacket("cam", 2, 2.0, _frame(100))) is True
    assert engine.process_next() is None

    episode = next(iter(engine.state_for("cam").tracks.values()))
    assert ocr.calls == 1
    assert episode.recognition is not None
    assert episode.recognition.decision.phase is RecognitionPhase.PROVISIONAL
    assert episode.phase.value == "collecting"


def test_runtime_soft_locks_then_finalizes_once_without_more_ocr() -> None:
    ocr = _OCR(0.87)
    engine = _engine(ocr)
    engine.submit_frame(FramePacket("cam", 1, 1.00, _frame(0)))

    assert engine.submit_frame(FramePacket("cam", 2, 1.04, _frame(100))) is True
    assert engine.process_next() is None
    assert ocr.calls == 1

    # Adjacent evidence is correlated, so the adaptive scheduler skips it.
    assert engine.submit_frame(FramePacket("cam", 3, 1.08, _frame(105))) is True
    assert engine.process_next() is None
    assert ocr.calls == 1

    assert engine.submit_frame(FramePacket("cam", 4, 1.12, _frame(110))) is True
    assert engine.process_next() is None
    episode = next(iter(engine.state_for("cam").tracks.values()))
    assert episode.recognition is not None
    assert episode.recognition.decision.phase is RecognitionPhase.SOFT_LOCKED
    assert ocr.calls == 2

    # The hold expires on real time. Finalization is a zero-inference task.
    assert engine.submit_frame(FramePacket("cam", 5, 1.28, _frame(115))) is True
    event = engine.process_next()
    assert event is not None
    assert event.text == PLATE
    assert event.metadata["recognition_phase"] == "finalized"
    assert event.metadata["fusion_reason"] == "independent_temporal_consensus"
    assert event.metadata["finalization_reason"] == "soft_lock_hold_elapsed"
    assert event.metadata["ocr_attempts"] == 2
    assert ocr.calls == 2
    telemetry = engine.telemetry()
    assert telemetry["track_temporal_fusion_enabled"] is True
    assert telemetry["fusion_locked_tracks"] == 0
    assert telemetry["fusion_finalized_tracks"] == 1
    assert telemetry["fusion_ocr_attempts"] == 2

    assert engine.submit_frame(FramePacket("cam", 6, 1.32, _frame(120))) is True
    assert engine.process_next() is None
    assert ocr.calls == 2


def test_finalize_camera_commits_soft_lock_without_extra_inference() -> None:
    ocr = _OCR(0.97)
    engine = _engine(ocr)
    engine.submit_frame(FramePacket("cam", 1, 1.00, _frame(0)))
    engine.submit_frame(FramePacket("cam", 2, 1.04, _frame(100)))
    assert engine.process_next() is None

    episode = next(iter(engine.state_for("cam").tracks.values()))
    assert episode.recognition is not None
    assert episode.recognition.decision.phase is RecognitionPhase.SOFT_LOCKED
    assert ocr.calls == 1

    events = engine.finalize_camera("cam", final_seq=2, final_ts=1.04)

    assert len(events) == 1
    assert events[0].text == PLATE
    assert events[0].metadata["finalization_reason"] == "track_exit"
    assert ocr.calls == 1
    assert engine.finalize_camera("cam", final_seq=2, final_ts=1.04) == []
