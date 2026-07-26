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
