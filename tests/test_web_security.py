import os
import time
from pathlib import Path

import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.database as database
import app.main as main


def _as_role(monkeypatch, role):
    monkeypatch.setattr(main, "auth", lambda request: role)
    monkeypatch.setattr(
        main,
        "current_user",
        lambda request: {
            "username": role,
            "role": role,
            "is_admin": 1 if role == "admin" else 0,
        },
    )


def test_operator_cannot_change_system_or_license_settings(monkeypatch):
    _as_role(monkeypatch, "operator")
    writes = []
    monkeypatch.setattr(main, "set_setting", lambda key, value: writes.append((key, value)))

    with TestClient(main.app) as client:
        ai_response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "gpu",
                "ai_quality": "accuracy",
                "ai_confidence": "99",
                "ai_frames": "20",
            },
            follow_redirects=False,
        )
        license_response = client.post(
            "/license/deactivate",
            follow_redirects=False,
        )
        camera_response = client.post(
            "/cameras/1/delete",
            follow_redirects=False,
        )

    assert ai_response.status_code == 403
    assert license_response.status_code == 403
    assert camera_response.status_code == 403
    assert writes == []


def test_system_role_can_change_ai_settings(monkeypatch):
    _as_role(monkeypatch, "system")
    writes = []
    monkeypatch.setattr(main, "set_setting", lambda key, value: writes.append((key, value)))

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert ("ai_accelerator", "cpu") in writes
    assert ("ai_confidence", 80) in writes


def test_storage_children_must_be_distinct_and_below_root(tmp_path):
    root = tmp_path / "bcvision-data"
    paths = main._storage_paths(
        root,
        root / "snapshots",
        root / "plates",
        root / "videos",
        root / "backups",
    )
    assert paths[0] == root.resolve()

    with pytest.raises(ValueError):
        main._storage_paths(
            root,
            tmp_path / "outside",
            root / "plates",
            root / "videos",
            root / "backups",
        )

    with pytest.raises(ValueError):
        main._storage_paths(
            root,
            root / "shared",
            root / "shared",
            root / "videos",
            root / "backups",
        )


def test_retention_cleanup_refuses_files_outside_storage_root(tmp_path):
    root = tmp_path / "bcvision-data"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    protected = outside / "must-stay.txt"
    protected.write_text("keep", encoding="utf-8")
    old = time.time() - 10 * 86400
    os.utime(protected, (old, old))

    removed = main._cleanup_old_files(outside, 1, root)

    assert removed == 0
    assert protected.read_text(encoding="utf-8") == "keep"


def test_media_path_check_is_not_vulnerable_to_prefix_collision(tmp_path, monkeypatch):
    _as_role(monkeypatch, "guard")
    allowed = tmp_path / "images"
    sibling = tmp_path / "images-private"
    allowed.mkdir()
    sibling.mkdir()
    secret = sibling / "secret.jpg"
    secret.write_bytes(b"secret")
    settings = {
        "snapshot_path": str(allowed),
        "plate_path": str(tmp_path / "plates"),
        "video_path": str(tmp_path / "videos"),
    }
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )

    with TestClient(main.app) as client:
        response = client.get("/media", params={"path": str(secret)})

    assert response.status_code == 404


def test_legacy_video_upload_ignores_traversal_filename(tmp_path, monkeypatch):
    _as_role(monkeypatch, "operator")
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("../../owned.mp4", b"not-a-real-video", "video/mp4")},
        )

    assert response.status_code == 200
    assert not (tmp_path / "owned.mp4").exists()
    saved = list(video_dir.iterdir())
    assert len(saved) == 1
    assert saved[0].name.endswith(".mp4")
    assert ".." not in saved[0].name


def test_oversized_video_upload_is_removed(tmp_path, monkeypatch):
    _as_role(monkeypatch, "operator")
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )
    monkeypatch.setattr(main, "MAX_VIDEO_UPLOAD_BYTES", 3)

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("large.mp4", b"four", "video/mp4")},
        )

    assert response.status_code == 200
    assert "بیشتر از ۲ گیگابایت" in response.text
    assert list(video_dir.iterdir()) == []


def test_camera_video_upload_registers_live_source_without_batch_processing(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    db_path = tmp_path / "upload.db"
    video_dir = tmp_path / "videos"
    source_path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 90),
    )
    assert writer.isOpened()
    for value in (20, 80, 140):
        writer.write(np.full((90, 160, 3), value, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as con:
        camera_id = con.execute(
            "INSERT INTO cameras(name,rtsp_url,location,enabled,is_demo) "
            "VALUES('Gate','rtsp://gate','Gate',1,0)"
        ).lastrowid

    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )
    calls = {"get": [], "remove": []}
    monkeypatch.setattr(
        main.manager,
        "get",
        lambda *args: calls["get"].append(args),
    )
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda camera_id: calls["remove"].append(camera_id),
    )

    started = time.monotonic()
    with TestClient(main.app) as client, source_path.open("rb") as source:
        response = client.post(
            "/cameras/video-upload",
            data={"camera_id": str(camera_id)},
            files={"video": ("traffic.avi", source, "video/x-msvideo")},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )
    elapsed = time.monotonic() - started

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["redirect"] == "/dashboard?video=1"
    assert elapsed < 3
    with database.connect() as con:
        uploaded = con.execute(
            "SELECT * FROM cameras WHERE rtsp_url LIKE 'video://%'"
        ).fetchall()
    assert len(uploaded) == 1
    assert uploaded[0]["name"] == "ویدئو: traffic"
    assert calls["get"][0][0] == uploaded[0]["id"]
    assert calls["get"][0][1].startswith("video://")


def test_uploaded_video_flows_through_worker_to_sqlite_and_dashboard(
    tmp_path,
    monkeypatch,
):
    import app.ai.live_worker as live_worker

    _as_role(monkeypatch, "operator")
    db_path = tmp_path / "end-to-end.db"
    video_dir = tmp_path / "videos"
    source_path = tmp_path / "traffic.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 90),
    )
    assert writer.isOpened()
    for value in (30, 60, 90, 120):
        writer.write(np.full((90, 160, 3), value, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as con:
        con.execute(
            "UPDATE settings SET value=? WHERE key='plate_path'",
            (str(tmp_path / "plates"),),
        )
        con.execute(
            "UPDATE settings SET value=? WHERE key='snapshot_path'",
            (str(tmp_path / "snapshots"),),
        )
        source_camera_id = con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,enabled,is_demo,lpr_enabled,"
            "lpr_confidence,frame_step,duplicate_seconds"
            ") VALUES(?,?,?,?,?,?,?,?,?)",
            ("Gate", "rtsp://gate", "Gate", 1, 0, 1, 50, 1, 0),
        ).lastrowid

    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )
    frame_result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.90,
        "detector_confidence": 0.90,
        "ocr_confidence": 0.90,
        "quality_score": 0.85,
        "bbox": (30, 30, 130, 65),
        "method": "end-to-end-test",
    }

    def detect(frame, *_args, **_kwargs):
        result = dict(frame_result)
        result["crop"] = frame[30:65, 30:130].copy()
        return [result]

    monkeypatch.setattr(live_worker, "process_frame", detect)
    monkeypatch.setattr(
        live_worker.worker,
        "_models",
        lambda: {
            "ready": True,
            "detector_ready": True,
            "easyocr_ready": True,
        },
    )

    virtual_id = None
    try:
        with TestClient(main.app) as client, source_path.open("rb") as source:
            response = client.post(
                "/cameras/video-upload",
                data={"camera_id": str(source_camera_id)},
                files={"video": ("traffic.avi", source, "video/x-msvideo")},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert response.status_code == 200
            virtual_id = int(response.json()["camera_id"])

            for _ in range(250):
                with database.connect() as con:
                    event = con.execute(
                        "SELECT * FROM plate_events "
                        "WHERE camera_id=? ORDER BY id DESC LIMIT 1",
                        (virtual_id,),
                    ).fetchone()
                stream = main.manager.streams.get(virtual_id)
                if event and stream and stream.latest:
                    break
                time.sleep(0.02)

            dashboard = client.get("/dashboard")

        assert event is not None
        assert event["plate_norm"] == "12ب34567"
        assert dashboard.status_code == 200
        assert f"id='anpr-{virtual_id}'" in dashboard.text
        assert "ویدئو: traffic" in dashboard.text
        assert main.manager.streams[virtual_id].latest.startswith(b"\xff\xd8")
    finally:
        if virtual_id is not None:
            main.manager.remove(virtual_id)
            live_worker.stop_live_camera(virtual_id)


def test_events_escape_stored_watchlist_html(tmp_path, monkeypatch):
    db_path = tmp_path / "security.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    payload = "<img src=x onerror=alert(1)>"
    with database.connect() as con:
        con.execute(
            "INSERT INTO plate_watchlist("
            "plate_text,plate_norm,status,owner_name"
            ") VALUES(?,?,?,?)",
            ("12ب34567", "12ب34567", "allowed", payload),
        )
        con.execute(
            "INSERT INTO plate_events("
            "plate_text,plate_norm,confidence,camera_name"
            ") VALUES(?,?,?,?)",
            ("12ب34567", "12ب34567", 0.9, "Gate"),
        )
    _as_role(monkeypatch, "guard")

    with TestClient(main.app) as client:
        response = client.get("/events")

    assert response.status_code == 200
    assert payload not in response.text
    assert "&lt;img src=x onerror=alert(1)&gt;" in response.text


def test_public_health_does_not_expose_license_or_customer_data():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "bc-vision"
    assert "license" not in payload
    assert "customer" not in response.text.lower()
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("=HYPERLINK('https://example.invalid')", "'=HYPERLINK('https://example.invalid')"),
        ("+cmd", "'+cmd"),
        ("@SUM(1,1)", "'@SUM(1,1)"),
        ("camera-1", "camera-1"),
    ],
)
def test_csv_cells_neutralize_spreadsheet_formulas(value, expected):
    assert main._csv_cell(value) == expected
