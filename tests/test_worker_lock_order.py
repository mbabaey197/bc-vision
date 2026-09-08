import threading

import numpy as np

from app.ai import live_worker


def test_shadow_observation_does_not_invert_submission_lock_order(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {'lpr_confidence': 60}
    worker._states[1] = state
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(live_worker, 'process_frame', lambda *a, **k: [])
    monkeypatch.setattr(worker, '_selected_detector_variant', lambda: 'yolov8n')
    acquired = []

    def observe(*args):
        # Submit/status take the worker lock before reading retry entries.
        # A bounded probe reproduces the inverse wait without leaking threads.
        def submit_side():
            with worker._lock:
                got = state.model_switch_lock.acquire(timeout=0.3)
                acquired.append(got)
                if got:
                    try:
                        worker._retry_entries(state)
                    finally:
                        state.model_switch_lock.release()
        thread = threading.Thread(target=submit_side, daemon=True)
        thread.start()
        thread.join(2)
        assert not thread.is_alive()
        # The real observer consults the setting even when shadow is disabled.
        with worker._lock:
            pass

    monkeypatch.setattr(worker, '_observe_engine_v2_baseline', observe)
    try:
        worker._process(state, (1, 'test', frame, 1.0))
        assert acquired == [True]
        assert state.processed_frames == 1
        assert not state.busy
    finally:
        worker.shutdown()
