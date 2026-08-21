import time
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np

import app.ai.live_worker as live_worker
import app.media_storage as media_storage


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
            processing_ms REAL,
            media_status TEXT,
            media_error TEXT
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
        "plate_path": str(tmp_path / "تصاویر پلاک"),
        "snapshot_path": str(tmp_path / "تصاویر خودرو"),
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
        # The tracker can emit a best frame without retaining its crop.
        # Persistence must reconstruct the crop from the detector bbox.
        "crop": None,
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
    assert rows[0]["media_status"] == "complete"
    assert rows[0]["media_error"] == ""
    plate_path = Path(rows[0]["plate_image_path"])
    vehicle_path = Path(rows[0]["image_path"])
    assert plate_path.parent == tmp_path / "تصاویر پلاک"
    assert vehicle_path.parent == tmp_path / "تصاویر خودرو"
    for image_path in (plate_path, vehicle_path):
        payload = image_path.read_bytes()
        assert len(payload) > 0
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None
        assert decoded.size > 0


def test_confirmed_event_is_not_overwritten_by_different_plate(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "identity-events.db"
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
            review_status TEXT,
            source TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    first = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    first_id = worker._persist(8, "Gate", frame, first, 20.0)
    second = {
        **first,
        "plate": "98-م-765-43",
        "plate_norm": "98م76543",
    }
    second_id = worker._persist(
        8,
        "Gate",
        frame,
        second,
        20.0,
        first_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT id,plate_norm,review_status "
            "FROM plate_events ORDER BY id"
        ).fetchall()
    assert second_id != first_id
    assert [
        (row["id"], row["plate_norm"], row["review_status"])
        for row in rows
    ] == [
        (first_id, "12ب34567", "confirmed-ai"),
        (second_id, "", "suggested"),
    ]


def test_confirmed_event_cannot_be_downgraded_by_reviewable_result(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "immutable-identity-events.db"
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
            review_status TEXT,
            source TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    confirmed = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    event_id = worker._persist(
        8,
        "Gate",
        frame,
        confirmed,
        20.0,
    )
    reviewable = {
        **confirmed,
        "plate": "98-م-765-43",
        "plate_norm": "",
        "raw_guess_norm": "98م76543",
        "valid": False,
        "needs_review": True,
    }
    returned_id = worker._persist(
        8,
        "Gate",
        frame,
        reviewable,
        20.0,
        event_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT id,plate_text,plate_norm,review_status "
            "FROM plate_events ORDER BY id"
        ).fetchall()
    assert returned_id == event_id
    assert len(rows) == 1
    assert rows[0]["plate_norm"] == "12ب34567"
    assert rows[0]["plate_text"] == "12-ب-345-67"
    assert rows[0]["review_status"] == "confirmed-ai"


def test_recent_exact_event_is_reused_after_worker_state_restart(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "restart-dedup.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    settings = {
        "plate_path": str(tmp_path / "plates"),
        "snapshot_path": str(tmp_path / "vehicles"),
        "save_plate_images": "0",
        "save_snapshots": "0",
    }
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    first_worker = live_worker.LiveANPRWorker(max_workers=1)
    second_worker = live_worker.LiveANPRWorker(max_workers=1)
    for worker in (first_worker, second_worker):
        monkeypatch.setattr(
            worker,
            "_setting",
            lambda key, default="": settings.get(key, default),
        )

    first_id = first_worker._persist(
        camera_id,
        "Gate",
        frame,
        dict(result),
        20.0,
        duplicate_seconds=30,
    )
    second_id = second_worker._persist(
        camera_id,
        "Gate",
        frame,
        dict(result),
        20.0,
        duplicate_seconds=30,
    )
    first_worker.shutdown()
    second_worker.shutdown()

    with app.database.connect() as con:
        count = int(con.execute(
            "SELECT COUNT(*) FROM plate_events WHERE camera_id=?",
            (camera_id,),
        ).fetchone()[0])
    assert second_id == first_id
    assert count == 1


def test_media_encoder_failure_keeps_text_event_and_records_error(
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
            media_status TEXT,
            media_error TEXT
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
    monkeypatch.setattr(
        media_storage.cv2,
        "imencode",
        lambda *_args, **_kwargs: (False, None),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.91,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }

    event_id = worker._persist(8, "Gate", frame, result, 20.0)
    worker.shutdown()

    with connect() as con:
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
    assert row["plate_text"] == "12-ب-345-67"
    assert row["plate_norm"] == "12ب34567"
    assert row["plate_image_path"] == ""
    assert row["image_path"] == ""
    assert row["media_status"] == "error"
    assert "plate: JPEG encoder returned no data" in row["media_error"]
    assert "vehicle: JPEG encoder returned no data" in row["media_error"]
    assert list(tmp_path.rglob("*.tmp")) == []


def test_existing_event_keeps_original_observation_city(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "city-snapshot.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,location,city) "
            "VALUES(?,?,?,?)",
            ("Gate", "rtsp://gate", "ورودی شمالی", "تهران"),
        ).lastrowid)

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
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.91,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }

    event_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
    )
    with app.database.connect() as con:
        con.execute(
            "UPDATE cameras SET city='شیراز' WHERE id=?",
            (camera_id,),
        )
    worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
        event_id,
    )
    worker.shutdown()

    with app.database.connect() as con:
        city = con.execute(
            "SELECT city FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()[0]
    assert city == "تهران"


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


def test_operator_assisted_rows_replaces_only_unreadable_overlap():
    baseline = [{
        "bbox": (20, 20, 140, 60),
        "plate": "ناخوانا",
        "valid": False,
        "needs_review": True,
    }, {
        "bbox": (180, 20, 300, 60),
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "needs_review": False,
    }]
    shadow = [{
        "bbox": (22, 21, 142, 61),
        "plate": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": False,
    }, {
        "bbox": (182, 21, 302, 61),
        "plate": "12-ب-345-76",
        "raw_guess_norm": "12ب34576",
        "valid": False,
    }]

    selected = live_worker.operator_assisted_rows(baseline, shadow)

    assert len(selected) == 2
    assert selected[0]["raw_guess_norm"] == "31ط55674"
    assert selected[0]["assisted_candidate"] is True
    assert selected[0]["needs_review"] is True
    assert selected[1]["plate_norm"] == "12ب34567"


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
        _duplicate_seconds=0.0,
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

    # Provisional captures stay in tracker memory; only the strict consensus
    # becomes a durable row.
    assert all(not row.get("capture_only") for row, _event_id in persisted)
    recognized = [
        row for row, _event_id in persisted
        if not row.get("capture_only")
    ]
    assert len(recognized) == 1
    assert recognized[0]["plate_norm"] == "12ب34567"
    assert persisted[-1][1] is None
    assert state.emitted_events == 1
    # Slow inference must preserve consecutive observations without leaving a
    # physical track open long enough to absorb a later vehicle.
    assert state.tracker.max_age_seconds == 6.0


def test_fragmented_continuous_plate_reuses_one_event_after_cooldown(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "confidence": 0.92,
        "ocr_confidence": 0.90,
        "quality_score": 0.82,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(detected)],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    writes = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        saved_id = int(event_id) if event_id is not None else 71
        writes.append((saved_id, event_id, dict(result)))
        return saved_id

    monkeypatch.setattr(worker, "_persist", persist)
    # The 40-second inference gap forces a new tracker id and exceeds the
    # configured cooldown, but no empty observation ever ended the visit.
    for timestamp in (0.0, 0.2, 0.4, 40.0, 40.2, 40.4):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert [event_id for _saved, event_id, _row in writes] == [None, 71]
    assert {saved for saved, _event_id, _row in writes} == {71}
    assert state.emitted_events == 1
    assert state.seen["31ط55674"] == 40.4


def test_same_plate_after_confirmed_absence_creates_new_event(
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
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "confidence": 0.92,
        "ocr_confidence": 0.90,
        "quality_score": 0.82,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    outputs = iter(
        [[dict(detected)] for _ in range(3)]
        + [[] for _ in range(3)]
        + [[dict(detected)] for _ in range(3)]
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    inserted = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        _result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        if event_id is None:
            inserted.append(80 + len(inserted))
            return inserted[-1]
        return int(event_id)

    monkeypatch.setattr(worker, "_persist", persist)
    # The return happens before the tracker's normal expiry.  Three empty
    # observations must retire the old one-shot track so a new visit can emit.
    timestamps = (0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.2, 2.4)
    for timestamp in timestamps:
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert inserted == [80, 81]
    assert state.emitted_events == 2


def test_provisional_capture_waits_for_one_final_row(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.92,
            "ocr_confidence": 0.90,
            "quality_score": 0.82,
            "bbox": (25, 30, 155, 68),
            "crop": frame[30:68, 25:155].copy(),
            "method": "test",
        }],
    )
    writes = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        writes.append((dict(result), event_id))
        return event_id or 91

    monkeypatch.setattr(worker, "_persist", persist)

    state.busy = True
    worker._process(state, (1, "Gate", frame, 0.0))
    assert writes == []

    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [],
    )
    state.busy = True
    worker._process(state, (1, "Gate", frame, 6.0))
    worker.shutdown()

    assert len(writes) == 1
    assert writes[0][0]["provisional"] is False
    assert writes[0][0]["valid"] is False
    assert writes[0][0]["needs_review"] is True
    assert writes[0][1] is None
    assert state.emitted_events == 1


def test_unknown_fragment_cannot_erase_live_review_candidate(monkeypatch):
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
    candidate = {
        "plate": "31-ط-556-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
    }
    state.visits.register(
        candidate,
        77,
        0.0,
        allow_candidate=True,
    )
    state.track_event_ids[1] = 77
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    unknown = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "raw_guess_text": "",
        "raw_guess_norm": "",
        "valid": False,
        "needs_review": True,
        "confidence": 0.35,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "quality_score": 0.66,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(unknown)],
    )
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown fragment must not downgrade candidate")
        ),
    )

    for timestamp in (0.0, 0.4, 0.8):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert state.visits.event_refs == {"31ط55674": 77}
    assert state.emitted_events == 0


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


def test_motion_wakes_camera_during_long_empty_scene_backoff(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "id": 27,
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.config_loaded_at = time.monotonic()
    state.next_inference_at = time.monotonic() + 30.0
    worker._states[27] = state
    empty = np.zeros((120, 240, 3), dtype=np.uint8)
    state.activity.observe(empty)
    entering = empty.copy()
    entering[30:100, 70:190] = 220
    processed = threading.Event()

    def record_process(current_state, payload):
        assert payload[5].wake_inference is True
        processed.set()
        with worker._lock:
            current_state.busy = False

    monkeypatch.setattr(worker, "_process", record_process)

    worker.submit(27, "entry camera", entering)
    assert processed.wait(0.5)
    assert state.motion_wakeups == 1
    assert state.burst_frames_remaining >= 4
    worker.shutdown()


def test_two_cameras_receive_independent_worker_slots(monkeypatch):
    monkeypatch.setattr(
        live_worker,
        "parallel_camera_limit",
        lambda: 2,
    )
    worker = live_worker.LiveANPRWorker()
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": (
            "yolo8n" if key == "anpr_detector_model" else default
        ),
    )
    monkeypatch.setattr(
        worker,
        "_load_config",
        lambda camera_id: {
            "id": camera_id,
            "rtsp_url": f"rtsp://camera/{camera_id}",
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
    detector_variants = []
    active_lock = threading.Lock()

    def process(_frame, _confidence, engine_key=None, **kwargs):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            engine_keys.append(engine_key)
            detector_variants.append(kwargs.get("detector_variant"))
        time.sleep(0.05)
        with active_lock:
            active -= 1
        return []

    monkeypatch.setattr(live_worker, "process_frame", process)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    worker.submit(1, "gate one", frame)
    worker.submit(2, "gate two", frame)
    deadline = time.monotonic() + 2.0
    while True:
        first = worker.status(1)
        second = worker.status(2)
        if (
            first["processed_frames"] == 1
            and second["processed_frames"] == 1
        ):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    worker.shutdown()

    assert maximum_active == 2
    assert sorted(engine_keys) == [1, 2]
    assert detector_variants == ["yolov8n", "yolov8n"]
    assert first["threads_per_camera"] == 2
    assert first["parallel_camera_limit"] == 2
    assert first["anpr_engine"] == {
        "mode": "baseline",
        "detector_variant": "yolov8n",
        "exclusive_detector": True,
        "candidate_inference": False,
    }
    assert first["shadow"]["enabled"] is False
    assert second["processed_frames"] == 1


def test_detector_selection_cache_can_be_invalidated(monkeypatch):
    from app.ai import onnx_detector

    cleared = []
    monkeypatch.setattr(
        onnx_detector,
        "clear_detector_sessions",
        lambda: cleared.append(True),
    )
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {"duplicate_seconds": 27}
    observation = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "quality_score": 0.82,
        "bbox": (20, 20, 140, 55),
        "crop": np.zeros((35, 120, 3), dtype=np.uint8),
    }
    state.tracker.update([observation], timestamp=0.0)
    state.tracker.update([observation], timestamp=0.2)
    old_tracker = state.tracker
    state.visits.register(
        {**observation, "track_id": 3},
        41,
        10.0,
    )
    state.track_event_ids[3] = 41
    state.latest_detections = [{"plate": "31-ط-556-74"}]
    state.processed_frames = 12
    state.detected_candidates = 5
    state.emitted_events = 2
    state.frame_counter = 40
    worker._states[7] = state
    worker._model_state = {"detector_ready": True}
    worker._model_state_at = 123.0
    worker._model_state_variant = "yolov8n"

    worker.invalidate_model_cache()
    worker.shutdown()

    assert worker._model_state == {}
    assert worker._model_state_at == 0.0
    assert worker._model_state_variant == ""
    assert cleared == [True]
    assert state.tracker is not old_tracker
    assert state.tracker.emit_cooldown == 27
    assert state.tracker.update([observation], timestamp=0.4) == []
    # Exact durable visit identity survives a detector switch, while all
    # model-specific track bindings are reset.
    assert state.seen == {"31ط55674": 10.0}
    assert state.visits.event_refs == {"31ط55674": 41}
    assert state.visits.track_keys == {}
    assert state.track_event_ids == {}
    assert state.latest_detections == []
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.emitted_events == 0
    assert state.frame_counter == 0


def test_inflight_old_detector_result_is_discarded_on_switch(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[9] = state
    frame = np.zeros((100, 180, 3), dtype=np.uint8)
    stale = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "quality_score": 0.82,
        "bbox": (20, 20, 140, 55),
        "crop": frame[20:55, 20:140].copy(),
    }

    def switch_during_inference(*_args, **_kwargs):
        worker.invalidate_model_cache()
        return [stale]

    monkeypatch.setattr(live_worker, "process_frame", switch_during_inference)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale detector result must not be persisted")
        ),
    )

    worker._process(
        state,
        (9, "gate", frame, 1.0, 1.0, None, 0),
    )
    worker.shutdown()

    assert state.busy is False
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.tracker.active_track_ids() == set()
    assert state.seen == {}
    assert state.latest_detections == []


def test_removed_camera_discards_inflight_result_before_persistence(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[44] = state
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()

    def process(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        return [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "valid": True,
            "confidence": 0.92,
            "quality_score": 0.82,
            "bbox": (25, 30, 155, 68),
            "crop": frame[30:68, 25:155].copy(),
            "method": "test",
        }]

    monkeypatch.setattr(live_worker, "process_frame", process)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a retired camera must not persist")
        ),
    )
    thread = threading.Thread(
        target=worker._process,
        args=(state, (44, "Gate", frame, 1.0)),
    )
    thread.start()
    assert entered.wait(1.0)

    worker.remove(44)
    release.set()
    thread.join(timeout=2.0)
    worker.shutdown()

    assert not thread.is_alive()
    assert state.retired is True
    assert state.busy is False
    assert 44 not in worker._states


def test_selected_inference_failure_reaches_camera_last_error(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[12] = state
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )

    def fail(*_args, **kwargs):
        assert kwargs["engine_key"] == 12
        assert kwargs["detector_variant"] == "yolov8n"
        raise RuntimeError("selected YOLO inference failed")

    monkeypatch.setattr(live_worker, "process_frame", fail)

    worker._process(
        state,
        (12, "gate", np.zeros((100, 180, 3), dtype=np.uint8), 1.0),
    )
    worker.shutdown()

    assert state.busy is False
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.last_error == "RuntimeError: selected YOLO inference failed"


def test_launcher_preparation_transition_invalidates_model_status_cache(
    monkeypatch,
):
    from app.ai import model_manager

    worker = live_worker.LiveANPRWorker(max_workers=1)
    worker._model_state = {
        "selected_detector": "yolo11n",
        "detector_ready": True,
        "hezar_ready": True,
        "preparation_state": "",
        "preparation_error": "",
    }
    worker._model_state_at = time.monotonic()
    worker._model_state_variant = "yolo11n"
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolo11n",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_STATE_ENV,
        "error",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_ERROR_ENV,
        "ValueError: model hash mismatch",
    )
    calls = []

    def status(selected_detector=None):
        calls.append(selected_detector)
        return {
            "selected_detector": selected_detector,
            "detector_ready": True,
            "hezar_ready": True,
            "preparation_state": "error",
            "preparation_error": "ValueError: model hash mismatch",
        }

    monkeypatch.setattr(model_manager, "model_status", status)

    current = worker._models()
    worker.shutdown()

    assert calls == ["yolo11n"]
    assert current["ready"] is True
    assert current["preparation_state"] == "error"
    assert current["preparation_error"] == "ValueError: model hash mismatch"


def test_video_pass_drain_promotes_worker_pending_frame(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[31] = state
    token = worker.begin_video_pass(31)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    state.pending = (
        31,
        "uploaded video",
        frame,
        1.0,
        1.0,
        None,
        token["detector_generation"],
    )
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )
    monkeypatch.setattr(live_worker, "process_frame", lambda *_a, **_k: [])

    drained = worker.drain_video_pass(31, token, timeout=1.0)
    worker.shutdown()

    assert drained["ok"] is True
    assert drained["error"] == ""
    assert drained["processed_frames"] == 1
    assert state.pending is None
    assert state.busy is False


def test_video_pass_drain_remembers_error_cleared_by_later_success(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[32] = state
    token = worker.begin_video_pass(32)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("selected YOLO failed once")
        ),
    )
    state.busy = True
    worker._process(state, (32, "video", frame, 1.0))
    assert state.processing_errors == 1
    assert state.last_error == "RuntimeError: selected YOLO failed once"

    monkeypatch.setattr(live_worker, "process_frame", lambda *_a, **_k: [])
    state.busy = True
    worker._process(state, (32, "video", frame, 2.0))
    assert state.last_error == ""

    drained = worker.drain_video_pass(32, token, timeout=1.0)
    worker.shutdown()

    assert drained["ok"] is False
    assert drained["error"] == "RuntimeError: selected YOLO failed once"
