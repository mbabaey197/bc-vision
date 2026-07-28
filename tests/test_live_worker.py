import time
import sqlite3
import threading

import numpy as np

import app.ai.live_worker as live_worker


def test_unreadable_vehicle_event_is_upgraded_without_duplicate(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "events.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            plate_image_path TEXT,
            detector_method TEXT,
            ocr_confidence REAL,
            ocr_engine TEXT,
            ocr_alternative TEXT,
            ocr_disagreement INTEGER,
            vehicle_type TEXT,
            vehicle_color TEXT,
            vehicle_brand TEXT,
            vehicle_confidence REAL,
            direction TEXT,
            quality_score REAL,
            consensus_votes INTEGER,
            source TEXT,
            processing_ms REAL
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    settings = {
        "plate_path": str(tmp_path / "plates"),
        "snapshot_path": str(tmp_path / "vehicles"),
        "save_plate_images": "1",
        "save_snapshots": "1",
    }
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )
    frame = np.full((160, 260, 3), 175, dtype=np.uint8)
    base = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "valid": False,
        "confidence": 0.41,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "ocr_engine": "none",
        "ocr_alternative": "",
        "ocr_disagreement": False,
        "quality_score": 0.76,
        "bbox": (80, 95, 180, 125),
        "crop": frame[95:125, 80:180].copy(),
        "method": "test",
        "consensus_votes": 0,
    }
    event_id = worker._persist(3, "Gate", frame, base, 25.0)
    recognized = dict(base)
    recognized.update({
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "ocr_confidence": 0.90,
        "ocr_engine": "crnn-onnx",
        "ocr_alternative": "31-ط-558-74",
        "ocr_disagreement": True,
        "consensus_votes": 3,
    })
    updated_id = worker._persist(
        3,
        "Gate",
        frame,
        recognized,
        28.0,
        event_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT * FROM plate_events ORDER BY id"
        ).fetchall()
    assert updated_id == event_id
    assert len(rows) == 1
    assert rows[0]["plate_norm"] == "31ط55674"
    assert rows[0]["consensus_votes"] == 3
    assert rows[0]["ocr_engine"] == "crnn-onnx"
    assert rows[0]["ocr_alternative"] == "31-ط-558-74"
    assert rows[0]["ocr_disagreement"] == 1
    assert (tmp_path / "plates").is_dir()
    assert (tmp_path / "vehicles").is_dir()


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

    def fake_persist(
        _camera_id,
        _camera_name,
        _frame,
        saved_result,
        _processing_ms,
        event_id=None,
    ):
        persisted.append((saved_result, event_id))
        return event_id or 41

    monkeypatch.setattr(worker, "_persist", fake_persist)
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

    assert persisted[0][0]["capture_only"] is True
    recognized = [
        row for row, _event_id in persisted
        if not row.get("capture_only")
    ]
    assert len(recognized) == 1
    assert recognized[0]["plate_norm"] == "12ب34567"
    assert persisted[-1][1] == 41
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
    monkeypatch.setattr(worker, "_persist", lambda *_args: 1)
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


def test_every_received_frame_can_improve_pending_ocr_selection(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(
        busy=True,
        config={
            "id": 12,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 999,
        },
        config_loaded_at=200.0,
    )
    worker._states[12] = state
    monkeypatch.setattr(
        worker,
        "_config",
        lambda _camera_id, current, _now: current.config,
    )
    monkeypatch.setattr(
        worker,
        "_selection_score",
        lambda frame, _config: float(frame[0, 0, 0]),
    )
    times = iter((200.0, 200.1, 200.2))
    monkeypatch.setattr(
        live_worker.time,
        "monotonic",
        lambda: next(times),
    )

    for value in (20, 200, 80):
        worker.submit(
            12,
            "quality camera",
            np.full((20, 40, 3), value, dtype=np.uint8),
        )
    selected = int(state.pending[2][0, 0, 0])
    worker.shutdown()

    assert state.frame_counter == 3
    assert selected == 200
# RC7-RC9 regression coverage for adaptive live-frame processing.


def test_empty_inference_enters_backoff_and_clears_overlay(
    monkeypatch,
):
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
    worker._states[15] = state
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    detected = {
        "plate": "در حال بررسی",
        "plate_norm": "",
        "valid": False,
        "confidence": 0.51,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "quality_score": 0.7,
        "bbox": (20, 25, 130, 60),
        "crop": frame[25:60, 20:130],
        "method": "test",
        "whole_plate_ocr_attempted": True,
        "ocr_engine": "crnn-onnx",
        "ocr_alternative": "31-ط-556-74",
        "ocr_disagreement": True,
    }
    outputs = iter(([detected], [], [], []))
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: list(next(outputs)),
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    monkeypatch.setattr(worker, "_persist", lambda *_args: 1)

    for timestamp in (0.0, 1.0, 2.0, 3.0):
        state.busy = True
        worker._process(
            state,
            (15, "idle camera", frame.copy(), timestamp),
        )

    snapshot = worker.detection_snapshot(
        15,
        after_revision=3,
    )
    status = worker.status(15)
    worker.shutdown()

    assert state.detection_revision == 4
    assert snapshot["revision"] == 4
    assert snapshot["detections"] == []
    assert status["idle_mode"] is True
    assert status["no_plate_streak"] == 3
    assert status["next_inference_seconds"] >= 1.0
    assert status["ocr_ab"] == {
        "whole_plate_attempts": 1,
        "agreements": 0,
        "disagreements": 1,
        "crnn_selected": 1,
        "character_reader_selected": 0,
    }


def test_no_plate_backoff_grows_but_recognition_stays_responsive():
    delay = live_worker.LiveANPRWorker._post_inference_delay

    assert delay(0.25, 0) == 0.20
    assert delay(0.25, 1) == 0.40
    assert delay(0.25, 2) == 0.80
    assert delay(0.25, 3) == 1.60
    assert delay(0.25, 4) == 3.20
    assert delay(0.25, 10) == 3.20


def test_two_cameras_receive_independent_worker_slots(monkeypatch):
    monkeypatch.setattr(
        live_worker,
        "parallel_camera_limit",
        lambda: 2,
    )
    worker = live_worker.LiveANPRWorker()
    monkeypatch.setattr(
        worker,
        "_load_config",
        lambda camera_id: {
            "id": camera_id,
            "enabled": 1,
            "lpr_enabled": 1,
            "lpr_confidence": 50,
            "duplicate_seconds": 0,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 100,
            "roi_h": 100,
        },
    )
    active = 0
    maximum_active = 0
    engine_keys = []
    active_lock = threading.Lock()

    def process(_frame, _confidence, engine_key=None):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            engine_keys.append(engine_key)
        time.sleep(0.05)
        with active_lock:
            active -= 1
        return []

    monkeypatch.setattr(live_worker, "process_frame", process)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    worker.submit(1, "gate one", frame)
    worker.submit(2, "gate two", frame)
    time.sleep(0.14)
    first = worker.status(1)
    second = worker.status(2)
    worker.shutdown()

    assert maximum_active == 2
    assert sorted(engine_keys) == [1, 2]
    assert first["threads_per_camera"] == 2
    assert first["parallel_camera_limit"] == 2
    assert second["processed_frames"] == 1
