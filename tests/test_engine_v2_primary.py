from __future__ import annotations

import numpy as np
from pathlib import Path
import pytest

from app.ai.live_worker import LiveANPRWorker, _CameraState
from app.engine_v2.types import PlateEvent


def _event(text="12ب34567") -> PlateEvent:
    return PlateEvent(
        camera_id="7",
        frame_seq=11,
        ts=25.0,
        text=text,
        confidence=0.999,
        bbox=(10, 20, 90, 55),
        quality=0.91,
        track_id="v2-track-7",
        observations=3,
        metadata={"fusion_reason": "temporal_consensus"},
    )


def test_v2_callback_cannot_enter_durable_persistence_queue(monkeypatch):
    worker = LiveANPRWorker(max_workers=1, _defer_persistence_start=True)
    worker._states[7] = _CameraState()
    monkeypatch.setattr(
        worker,
        "_make_persistence_retry",
        lambda *args: pytest.fail("V2 shadow attempted persistence"),
    )
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    try:
        worker._ingest_engine_v2_event(_event(), frame)
    finally:
        worker._executor.shutdown(wait=True)

    assert worker._states[7].shadow_candidates == 1


def test_ready_v2_shadow_never_short_circuits_baseline_processing():
    source = Path("app/ai/live_worker.py").read_text(encoding="utf-8")
    start = source.index("            self._submit_engine_v2_shadow(")
    end = source.index("            activity = state.activity.observe", start)
    handoff = source[start:end]

    assert "if engine_v2_ready" not in handoff
    assert "return" not in handoff
