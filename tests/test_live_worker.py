import time

import numpy as np

import app.ai.live_worker as live_worker


def test_roi_and_translation():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    source, x, y = live_worker.LiveANPRWorker._roi_frame(
        frame,
        {
            "roi_x": 10,
            "roi_y": 20,
            "roi_w": 50,
            "roi_h": 40,
        },
    )
    assert source.shape[:2] == (40, 100)
    assert (x, y) == (20, 20)
    row = live_worker.LiveANPRWorker._translate(
        {
            "bbox": (1, 2, 11, 12),
            "vehicle_bbox": (0, 0, 20, 20),
        },
        x,
        y,
    )
    assert row["bbox"] == (21, 22, 31, 32)
    assert row["vehicle_bbox"] == (20, 20, 40, 40)


def test_submit_is_non_blocking_and_drops_to_latest(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_load_config",
        lambda camera_id: {
            "id": camera_id,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 1,
            "lpr_confidence": 50,
            "duplicate_seconds": 10,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 100,
            "roi_h": 100,
        },
    )
    processed = []

    def fake_process(state, payload):
        processed.append(int(payload[2][0, 0, 0]))
        time.sleep(0.03)
        with worker._lock:
            next_payload = state.pending
            state.pending = None
            if next_payload is None:
                state.busy = False
            else:
                worker._executor.submit(
                    fake_process,
                    state,
                    next_payload,
                )

    monkeypatch.setattr(worker, "_process", fake_process)
    for value in range(5):
        worker.submit(
            1,
            "cam",
            np.full(
                (10, 10, 3),
                value,
                dtype=np.uint8,
            ),
        )
    time.sleep(0.15)
    worker.shutdown()
    assert processed[0] == 0
    assert processed[-1] == 4
    assert len(processed) <= 3


def test_slow_cpu_keeps_three_observations_for_consensus(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.90,
        "quality_score": 0.80,
        "bbox": (30, 30, 130, 65),
        "crop": frame[30:65, 30:130].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(result)],
    )
    persisted = []
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args: persisted.append(_args[3]),
    )
    clock = iter((0.0, 3.0, 3.0, 6.0, 6.0, 9.0))
    monkeypatch.setattr(
        live_worker.time,
        "perf_counter",
        lambda: next(clock),
    )

    for timestamp in (0.0, 3.0, 6.0):
        state.busy = True
        worker._process(
            state,
            (1, "CPU camera", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert len(persisted) == 1
    assert persisted[0]["plate_norm"] == "12ب34567"
    assert state.tracker.max_age_seconds >= 11.0


def test_latest_detection_is_available_for_live_overlay(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[4] = state
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [{
            "plate": "12-ب-345-67",
            "plate_norm": "12ب34567",
            "valid": True,
            "confidence": 0.91,
            "quality_score": 0.8,
            "bbox": (20, 25, 130, 60),
            "crop": frame[25:60, 20:130],
            "method": "test",
        }],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    state.busy = True
    worker._process(state, (4, "cam", frame, time.monotonic()))
    detections = worker.detections(4)
    worker.shutdown()

    assert detections[0]["bbox"] == (20, 25, 130, 60)
    assert detections[0]["plate"] == "12-ب-345-67"


def test_submit_adaptively_spaces_slow_cpu_inference(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(
        config={
            "id": 9,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 1,
        },
        config_loaded_at=100.0,
        processing_seconds_ema=2.0,
    )
    worker._states[9] = state
    submitted = []
    monkeypatch.setattr(
        worker,
        "_config",
        lambda _camera_id, current, _now: current.config,
    )
    monkeypatch.setattr(
        worker._executor,
        "submit",
        lambda _callback, _state, payload: submitted.append(payload),
    )
    times = iter((100.0, 100.4, 100.89, 100.91))
    monkeypatch.setattr(
        live_worker.time,
        "monotonic",
        lambda: next(times),
    )
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    worker.submit(9, "cam", frame)
    state.busy = False
    worker.submit(9, "cam", frame)
    worker.submit(9, "cam", frame)
    worker.submit(9, "cam", frame)
    worker.shutdown()

    # EMA=2s yields a 0.9s minimum interval. Frames inside that interval are
    # skipped, then the newest eligible frame is submitted.
    assert len(submitted) == 2
# RC7 regression coverage for adaptive live-frame processing.
