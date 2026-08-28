from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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


def test_v2_primary_event_uses_existing_durable_persistence_queue(monkeypatch):
    worker = LiveANPRWorker(max_workers=1, _defer_persistence_start=True)
    state = _CameraState()
    state.config = {
        "name": "Gate camera",
        "duplicate_seconds": 30,
    }
    worker._states[7] = state
    created = []
    queued = []
    drained = []
    marker = SimpleNamespace()
    monkeypatch.setattr(worker, "_engine_v2_shadow_enabled", lambda *_: True)
    monkeypatch.setattr(
        worker,
        "_make_persistence_retry",
        lambda *args: created.append(args) or marker,
    )
    monkeypatch.setattr(
        worker,
        "_enqueue_persistence_retry",
        lambda current_state, entry: queued.append((current_state, entry)),
    )
    monkeypatch.setattr(
        worker,
        "_drain_persistence_retry_locked",
        lambda current_state, lock: drained.append((current_state, lock)),
    )
    frame = np.zeros((100, 160, 3), dtype=np.uint8)
    try:
        worker._ingest_engine_v2_event(_event(), frame)
    finally:
        worker._executor.shutdown(wait=True)

    assert len(created) == 1
    args = created[0]
    assert args[0:2] == (7, "Gate camera")
    assert args[2]["plate_norm"] == "12ب34567"
    assert args[2]["valid"] is True
    assert args[2]["auto_confirmed"] is True
    assert args[2]["confirmation_source"] == "engine-v2-temporal-consensus"
    assert args[2]["engine_lane"] == "primary-v2"
    assert args[5] == "12ب34567"
    assert queued == [(state, marker)]
    assert len(drained) == 1


def test_v2_primary_rejects_invalid_plate_before_persistence(monkeypatch):
    worker = LiveANPRWorker(max_workers=1, _defer_persistence_start=True)
    worker._states[7] = _CameraState(config={"name": "Gate camera"})
    created = []
    monkeypatch.setattr(worker, "_engine_v2_shadow_enabled", lambda *_: True)
    monkeypatch.setattr(
        worker,
        "_make_persistence_retry",
        lambda *args: created.append(args),
    )
    try:
        worker._ingest_engine_v2_event(_event("INVALID"), None)
    finally:
        worker._executor.shutdown(wait=True)

    assert created == []
