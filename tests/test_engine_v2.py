from __future__ import annotations

import numpy as np

from app.engine_v2 import EngineV2Config, EventDrivenANPREngine, FramePacket, OCRResult, PlateCandidate
from app.engine_v2.scheduler import LatestOnlyPriorityQueue


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray):
        self.calls += 1
        h, w = frame.shape[:2]
        return [PlateCandidate((w // 4, h // 3, 3 * w // 4, 2 * h // 3), 0.95)]


class FakeOCR:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        self.calls += 1
        return OCRResult("12ب34567", 0.93, True)


def _frame(value: int = 0) -> np.ndarray:
    img = np.full((120, 240, 3), value, dtype=np.uint8)
    img[40:80, 60:180] = min(255, value + 100)
    return img


def test_latest_only_queue_replaces_stale_frame() -> None:
    q = LatestOnlyPriorityQueue[int](max_items=4)
    q.submit("cam-1", 1)
    q.submit("cam-1", 2)
    q.submit("cam-2", 3)
    assert len(q) == 2
    assert q.pop() == 2
    assert q.pop() == 3
    assert q.stats.replaced == 1


def test_latest_only_queue_discards_one_key_without_disturbing_other_work() -> None:
    q = LatestOnlyPriorityQueue[int](max_items=4)
    q.submit("stale-episode", 1)
    q.submit("other-camera", 2)

    assert q.discard("stale-episode") is True
    assert q.discard("stale-episode") is False
    assert len(q) == 1
    assert q.pop() == 2
    assert q.pop() is None
    assert q.stats.discarded == 1


def test_engine_sleeps_until_motion_then_emits_one_event() -> None:
    detector = FakeDetector()
    ocr = FakeOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            done_cooldown_frames=10,
            load_control_enabled=False,
        ),
    )

    # First frame establishes the motion baseline and should not wake AI.
    assert engine.submit_frame(FramePacket("cam-1", 1, 1.0, _frame(0))) is False
    assert engine.process_next() is None
    assert detector.calls == 0

    # Large scene change wakes the shared detector/OCR path.
    moving = _frame(90)
    assert engine.submit_frame(FramePacket("cam-1", 2, 2.0, moving)) is True
    event = engine.process_next()
    assert event is not None
    assert event.camera_id == "cam-1"
    assert event.text == "12ب34567"
    assert detector.calls == 1
    assert ocr.calls == 1

    # DONE is per-track: a cheap motion-gate probe may run the detector again,
    # but the completed episode cannot be sent to OCR twice.
    assert engine.submit_frame(FramePacket("cam-1", 3, 3.0, moving)) is True
    assert engine.process_next() is None
    assert detector.calls == 2
    assert ocr.calls == 1


def test_multiple_cameras_share_one_detector_instance() -> None:
    detector = FakeDetector()
    ocr = FakeOCR()
    engine = EventDrivenANPREngine(
        detector,
        ocr,
        EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_quality=0.0,
            min_candidates_before_ocr=1,
            # This test verifies shared model ownership, not host-dependent
            # adaptive load behavior.
            load_control_enabled=False,
            # This test verifies model sharing, not overlapping-camera dedup.
            cross_camera_duplicate_seconds=0.0,
        ),
    )

    for cam in ("cam-a", "cam-b"):
        engine.submit_frame(FramePacket(cam, 1, 1.0, _frame(0)))
        assert engine.submit_frame(FramePacket(cam, 2, 2.0, _frame(100))) is True

    e1 = engine.process_next()
    e2 = engine.process_next()
    assert {e1.camera_id, e2.camera_id} == {"cam-a", "cam-b"}
    assert detector.calls == 2
    assert ocr.calls == 2
