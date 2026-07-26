import os
import time
from pathlib import Path

import pytest
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
