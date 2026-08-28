import asyncio
import os
import sqlite3
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

import app.database as database
import app.main as main
from app import config
from app.file_identity import path_file_identity


@pytest.fixture(autouse=True)
def _disable_network_camera_autostart(monkeypatch):
    """Keep web tests from opening their synthetic RTSP URLs.

    Lifespan behavior is exercised explicitly by the lifecycle tests below.
    Other tests populate isolated databases with placeholder ``rtsp://``
    values; allowing every TestClient to open them leaks capture-construction
    threads on Windows because OpenCV cannot interrupt that constructor.
    Individual lifecycle tests can, and do, replace this default stub.
    """

    monkeypatch.setattr(main.manager, "start_enabled_cameras", lambda: 0)


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


def test_authenticated_shell_shows_installed_release_number():
    response = main.page("داشبورد", "<main>ok</main>", "admin")
    rendered = response.body.decode("utf-8")

    assert "نسخه نصب‌شده" in rendered
    assert main.APP_RELEASE_LABEL in rendered


def test_update_quiescence_stops_cameras_then_anpr_worker(monkeypatch):
    from app.ai import live_worker

    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("cameras-stop") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout: calls.append(
            f"anpr-stop-{retry_timeout:g}"
        ) or True,
    )
    monkeypatch.setattr(
        main,
        "require_media_writes_quiescent",
        lambda: calls.append("media-idle"),
    )

    asyncio.run(main._quiesce_services_for_update())

    assert calls == ["cameras-stop", "anpr-stop-5", "media-idle"]


def test_failed_update_quiescence_restores_live_services(monkeypatch):
    from app.ai import live_worker

    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("cameras-stop") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout: calls.append("anpr-stop") or False,
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("anpr-start"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("cameras-start"),
    )

    with pytest.raises(main.UpdatePackageError, match="سرویس پلاک‌خوان"):
        asyncio.run(main._quiesce_services_for_update())

    assert calls == [
        "cameras-stop",
        "anpr-stop",
        "anpr-start",
        "cameras-start",
    ]


def _insert_admin(password, *, must_change_password=False):
    with database.connect() as connection:
        connection.execute(
            "INSERT INTO users(username,password_hash,display_name,is_admin,"
            "role,is_active,must_change_password,session_version) "
            "VALUES(?,?,?,1,'admin',1,?,0)",
            (
                "admin",
                main.hash_password(password),
                "مدیر سیستم",
                1 if must_change_password else 0,
            ),
        )


def _allow_test_upload_handoff(monkeypatch):
    class Reservation:
        def grow(self, _size):
            return None

        def close(self, **_kwargs):
            return None

    class Lease:
        def close(self):
            return None

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )
    monkeypatch.setattr(
        main,
        "pin_media_paths",
        lambda *_args, **_kwargs: Lease(),
    )


def _camera_row(tmp_path, monkeypatch):
    db_path = tmp_path / "camera-lifecycle.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as connection:
        return int(connection.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled) VALUES(?,?,1)",
            ("Original", "rtsp://original"),
        ).lastrowid)


def _new_camera_direct(rtsp_url):
    return main.new_cam(
        object(),
        name="Manual video",
        rtsp_url=rtsp_url,
        location="",
        city="",
        enabled="1",
        is_demo=0,
        sort_order=0,
        lpr_enabled="1",
        lpr_confidence=60,
        frame_step=5,
        duplicate_seconds=30,
        roi_x=0,
        roi_y=0,
        roi_w=100,
        roi_h=100,
        line_y=50,
    )


def _edit_camera_direct(camera_id, rtsp_url="rtsp://edited"):
    return main.edit_cam(
        camera_id,
        object(),
        name="Edited while uploading",
        rtsp_url=rtsp_url,
        location="",
        city="",
        enabled="1",
        is_demo=0,
        sort_order=0,
        lpr_enabled="1",
        lpr_confidence=60,
        frame_step=5,
        duplicate_seconds=30,
        roi_x=0,
        roi_y=0,
        roi_w=100,
        roi_h=100,
        line_y=50,
    )


def _manual_video_camera_mutation(operation, camera_id):
    url="  ViDeO:///manual-upload-bypass.avi"
    if operation == "new":
        return _new_camera_direct(url)
    return _edit_camera_direct(camera_id, url)


@pytest.mark.parametrize("operation", ["new", "edit"])
def test_camera_manager_rejects_manual_video_url_without_db_or_stream_write(
    tmp_path,
    monkeypatch,
    operation,
):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "system")
    remove_calls = []
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda removed_id: remove_calls.append(int(removed_id)) or True,
    )
    with database.connect() as connection:
        before = [dict(row) for row in connection.execute(
            "SELECT * FROM cameras ORDER BY id"
        ).fetchall()]

    response = _manual_video_camera_mutation(operation, camera_id)

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    with database.connect() as connection:
        after = [dict(row) for row in connection.execute(
            "SELECT * FROM cameras ORDER BY id"
        ).fetchall()]
    assert after == before
    assert remove_calls == []


@pytest.mark.parametrize("operation", ["new", "edit"])
def test_video_operator_cannot_use_camera_forms_to_create_virtual_owner(
    tmp_path,
    monkeypatch,
    operation,
):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "operator")
    assert main.has_permission(object(), "video.process") is True
    assert main.has_permission(object(), "camera.manage") is False
    remove_calls = []
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda removed_id: remove_calls.append(int(removed_id)) or True,
    )
    with database.connect() as connection:
        before = [dict(row) for row in connection.execute(
            "SELECT * FROM cameras ORDER BY id"
        ).fetchall()]

    response = _manual_video_camera_mutation(operation, camera_id)

    assert response.status_code == 403
    with database.connect() as connection:
        after = [dict(row) for row in connection.execute(
            "SELECT * FROM cameras ORDER BY id"
        ).fetchall()]
    assert after == before
    assert remove_calls == []


def test_camera_delete_preserves_db_owner_when_stream_will_not_stop(
    tmp_path,
    monkeypatch,
):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "system")
    monkeypatch.setattr(main.manager, "remove", lambda _camera_id: False)

    response = main.delete_cam(camera_id, object())

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    with database.connect() as connection:
        row = connection.execute(
            "SELECT name,rtsp_url FROM cameras WHERE id=?",
            (camera_id,),
        ).fetchone()
    assert tuple(row) == ("Original", "rtsp://original")


def test_camera_edit_preserves_db_config_when_stream_will_not_stop(
    tmp_path,
    monkeypatch,
):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "system")
    monkeypatch.setattr(main.manager, "remove", lambda _camera_id: False)

    response = main.edit_cam(
        camera_id,
        object(),
        name="Changed",
        rtsp_url="rtsp://changed",
        location="",
        city="",
        enabled="1",
        is_demo=0,
        sort_order=0,
        lpr_enabled="1",
        lpr_confidence=60,
        frame_step=5,
        duplicate_seconds=30,
        roi_x=0,
        roi_y=0,
        roi_w=100,
        roi_h=100,
        line_y=50,
    )

    assert response.status_code == 303
    assert "error=" in response.headers["location"]
    with database.connect() as connection:
        row = connection.execute(
            "SELECT name,rtsp_url FROM cameras WHERE id=?",
            (camera_id,),
        ).fetchone()
    assert tuple(row) == ("Original", "rtsp://original")


def test_camera_toggle_disables_stream_and_hides_from_dashboard_query(
    tmp_path,
    monkeypatch,
):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "system")
    removed = []
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda value: removed.append(int(value)) or True,
    )

    response = main.toggle_camera(
        camera_id,
        SimpleNamespace(client=None),
    )

    assert response.status_code == 303
    assert removed == [camera_id]
    with database.connect() as connection:
        assert connection.execute(
            "SELECT enabled FROM cameras WHERE id=?",
            (camera_id,),
        ).fetchone()["enabled"] == 0
    assert list(main.camera_rows(enabled_only=True)) == []


def test_camera_toggle_enables_and_starts_camera(tmp_path, monkeypatch):
    camera_id = _camera_row(tmp_path, monkeypatch)
    _as_role(monkeypatch, "system")
    with database.connect() as connection:
        connection.execute(
            "UPDATE cameras SET enabled=0 WHERE id=?",
            (camera_id,),
        )
    starts = []
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: starts.append(True) or 1,
    )

    response = main.toggle_camera(
        camera_id,
        SimpleNamespace(client=None),
    )

    assert response.status_code == 303
    assert starts == [True]
    with database.connect() as connection:
        assert connection.execute(
            "SELECT enabled FROM cameras WHERE id=?",
            (camera_id,),
        ).fetchone()["enabled"] == 1


def test_operator_cannot_change_system_or_camera_settings(monkeypatch):
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
        camera_response = client.post(
            "/cameras/1/delete",
            follow_redirects=False,
        )

    assert ai_response.status_code == 403
    assert camera_response.status_code == 403
    assert writes == []


def test_product_activation_routes_are_absent():
    paths = {route.path for route in main.app.routes}
    assert "/license" not in paths
    assert "/license/offline" not in paths
    assert "/license/online" not in paths
    assert "/license/deactivate" not in paths


def test_camera_creation_is_not_limited_by_product_plan(tmp_path, monkeypatch):
    db_path = tmp_path / "unlimited-cameras.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _as_role(monkeypatch, "system")
    with database.connect() as con:
        con.executemany(
            "INSERT INTO cameras(name,rtsp_url,location,enabled,is_demo) "
            "VALUES(?,?,?,?,?)",
            [
                (f"Camera {index}", f"rtsp://camera/{index}", "Gate", 1, 0)
                for index in range(64)
            ],
        )

    with TestClient(main.app) as client:
        response = client.post(
            "/cameras/new",
            data={
                "name": "Camera 65",
                "rtsp_url": "rtsp://camera/65",
                "location": "Gate",
                "city": "",
                "enabled": "1",
                "is_demo": "0",
                "sort_order": "65",
                "lpr_enabled": "1",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    with database.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM cameras").fetchone()[0] == 65


def test_system_role_can_change_ai_settings(monkeypatch):
    from app.ai import live_worker

    _as_role(monkeypatch, "system")
    writes = []
    switches = []
    audits = []
    monkeypatch.setattr(main, "set_setting", lambda key, value: writes.append((key, value)))
    monkeypatch.setattr(
        live_worker,
        "switch_live_anpr_detector",
        lambda variant, persist_setting=None: (
            switches.append(variant),
            persist_setting("anpr_detector_model", variant),
        ),
    )
    monkeypatch.setattr(
        main,
        "audit",
        lambda _request, action, details="": audits.append(
            (action, details)
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
                "anpr_detector_model": "yolo11n",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert ("ai_accelerator", "cpu") in writes
    assert ("ai_confidence", 80) in writes
    assert ("anpr_detector_model", "yolo11n") in writes
    assert switches == ["yolo11n"]
    assert audits == [
        (
            "anpr_detector_switch",
            "yolo11n; execution=exclusive-baseline",
        ),
        (
            "anpr_engine_v2_shadow",
            "disabled; persistence=false; mode=shadow-v2; "
            "primary=baseline; detector=yolo11n",
        ),
    ]


def test_ai_settings_reject_unknown_detector_model(monkeypatch):
    _as_role(monkeypatch, "system")
    writes = []
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: writes.append((key, value)),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
                "anpr_detector_model": "unverified-model",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert writes == []


def test_training_requires_explicit_rights_attestation(monkeypatch):
    _as_role(monkeypatch, "system")
    calls = []
    audits = []
    monkeypatch.setattr(
        main,
        "start_training",
        lambda **kwargs: calls.append(kwargs) or {
            "run_id": 7,
            "status": "queued",
        },
    )
    monkeypatch.setattr(
        main,
        "audit",
        lambda _request, action, details="": audits.append(
            (action, details)
        ),
    )
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": (
            "cpu" if key == "ai_accelerator" else default
        ),
    )

    rejected = main.start_ai_training(object(), 12, None)
    accepted = main.start_ai_training(object(), 12, "1")

    assert rejected.status_code == 303
    assert rejected.headers["location"].startswith("/settings?error=")
    assert calls == [{
        "device": "cpu",
        "epochs": 12,
        "rights_attested": True,
        "attested_by": "system",
    }]
    assert accepted.status_code == 303
    assert audits == [(
        "anpr_training_start",
        "run=7; epochs=12; rights_attested_by=system",
    )]


def test_ai_settings_reject_unready_yolox_without_partial_writes(monkeypatch):
    from app.ai import live_worker, model_manager

    _as_role(monkeypatch, "system")
    writes = []
    monkeypatch.setattr(main, "set_setting", lambda key, value: writes.append((key, value)))
    monkeypatch.setattr(
        model_manager,
        "model_status",
        lambda selected_detector=None: {"detector_yolox_ready": False},
    )
    monkeypatch.setattr(
        live_worker,
        "switch_live_anpr_detector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unready YOLOX must not reach the worker")
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
                "anpr_detector_model": "yolox",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert writes == []


def test_yolox_worker_recheck_failure_has_no_partial_ai_writes(monkeypatch):
    from app.ai import live_worker, model_manager

    _as_role(monkeypatch, "system")
    writes = []
    monkeypatch.setattr(main, "set_setting", lambda key, value: writes.append((key, value)))
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": "yolo11n" if key == "anpr_detector_model" else default,
    )
    monkeypatch.setattr(
        model_manager,
        "model_status",
        lambda selected_detector=None: {"detector_yolox_ready": True},
    )
    monkeypatch.setattr(
        live_worker,
        "switch_live_anpr_detector",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FileNotFoundError("YOLOX changed during activation")
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
                "anpr_detector_model": "yolox",
            },
            follow_redirects=False,
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert writes == []


def test_unchanged_detector_setting_is_not_rewritten_outside_worker(
    monkeypatch,
):
    _as_role(monkeypatch, "system")
    writes = []
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": (
            "yolo11n" if key == "anpr_detector_model" else default
        ),
    )
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: writes.append((key, value)),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/ai",
            data={
                "ai_accelerator": "cpu",
                "ai_quality": "balanced",
                "ai_confidence": "80",
                "ai_frames": "7",
                "anpr_detector_model": "yolo11n",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert all(key != "anpr_detector_model" for key, _value in writes)


def test_uploaded_video_playback_endpoint(monkeypatch):
    _as_role(monkeypatch, "operator")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "set_playback",
        lambda camera_id, action: calls.append(
            (camera_id, action)
        ) or True,
    )
    monkeypatch.setattr(main, "audit", lambda *_args: None)

    with TestClient(main.app) as client:
        pause = client.post(
            "/api/cameras/12/playback",
            json={"action": "pause"},
        )
        play = client.post(
            "/api/cameras/12/playback",
            json={"action": "play"},
        )

    assert pause.status_code == 200
    assert play.status_code == 200
    assert calls == [(12, "pause"), (12, "play")]


def test_guard_cannot_control_uploaded_video_playback(monkeypatch):
    _as_role(monkeypatch, "guard")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "set_playback",
        lambda camera_id, action: calls.append((camera_id, action)) or True,
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/api/cameras/12/playback",
            json={"action": "pause"},
        )

    assert response.status_code == 403
    assert calls == []


@pytest.mark.parametrize(
    "return_to",
    [
        "https://evil.invalid/dashboard?events_camera=7",
        "//evil.invalid/dashboard?events_camera=7",
        "/dashboard/../dashboard?events_camera=7",
        "/dashboard#https://evil.invalid",
        "/dashboard\\evil?events_camera=7",
    ],
)
def test_dashboard_return_target_rejects_nonlocal_or_noncanonical_urls(
    return_to,
):
    assert main._safe_dashboard_return_to(return_to) == "/dashboard"
    assert main._safe_dashboard_return_to(
        return_to,
        corrected=True,
    ) == "/dashboard?corrected=1"


def test_dashboard_return_target_keeps_only_valid_scope_fields():
    assert main._safe_dashboard_return_to(
        "/dashboard?video=2&events_camera=0007&events_after=-5"
        "&events_snapshot=9&events_page=2&next=https://evil.invalid"
    ) == (
        "/dashboard?video=1&events_camera=7"
        "&events_snapshot=9&events_page=2"
    )


def test_live_route_passes_persisted_video_markers_to_stream_manager(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "live-marker.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _as_role(monkeypatch, "operator")
    with database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,enabled,video_anpr_started,"
            "video_anpr_completed"
            ") VALUES('Completed video','video:///tmp/done.avi',1,1,1)"
        ).lastrowid)
    calls = []
    fake_stream = SimpleNamespace(frames=lambda: iter(()))
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: 0,
    )
    monkeypatch.setattr(
        main.manager,
        "get",
        lambda *args: calls.append(args) or fake_stream,
    )

    with TestClient(main.app) as client:
        response = client.get(f"/live/{camera_id}")

    assert response.status_code == 200
    assert len(calls) == 1
    assert calls[0][0:3] == (
        camera_id,
        "video:///tmp/done.avi",
        "Completed video",
    )
    assert calls[0][-2:] == (True, True)


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


def _storage_form(root):
    return {
        "storage_root": str(root),
        "snapshot_path": str(root / "snapshots"),
        "plate_path": str(root / "plates"),
        "video_path": str(root / "videos"),
        "backup_path": str(root / "backups"),
        "save_snapshots": "1",
        "save_plate_images": "1",
        "save_videos": None,
        "max_storage_gb": 0,
        "storage_full_action": "delete_oldest",
        "retention_snapshots_days": 90,
        "retention_plates_days": 90,
        "retention_videos_days": 7,
        "retention_events_days": 0,
    }


def _set_current_storage(root):
    database.set_settings_for_database(
        database.DB_PATH,
        {
            "storage_root": root,
            "snapshot_path": root / "snapshots",
            "plate_path": root / "plates",
            "video_path": root / "videos",
            "backup_path": root / "backups",
            "media_roots_history": "[]",
            "max_storage_gb": 0,
        },
        checkpoint_wal=False,
    )


def test_event_retention_preserves_operator_feedback_truth(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "retention-feedback.db"
    storage = tmp_path / "storage"
    for child in ("snapshots", "plates", "videos", "backups"):
        (storage / child).mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _set_current_storage(storage)
    database.set_setting("retention_events_days", "1")
    with database.connect() as con:
        protected_id = int(con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm,created_at) "
            "VALUES(?,?,?)",
            ("31-ط-556-74", "31ط55674", "2000-01-01 00:00:00"),
        ).lastrowid)
        expired_id = int(con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm,created_at) "
            "VALUES(?,?,?)",
            ("12-ب-345-67", "12ب34567", "2000-01-01 00:00:00"),
        ).lastrowid)
        feedback_id = int(con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,status,training_status"
            ") VALUES(?,?,?,?,?,'confirmed','pending')",
            (
                protected_id,
                "31-ط-556-74",
                "31ط55674",
                "31-ط-556-74",
                "31ط55674",
            ),
        ).lastrowid)

    main.run_retention_cleanup()

    with database.connect() as con:
        event_ids = {
            int(row[0])
            for row in con.execute("SELECT id FROM plate_events").fetchall()
        }
        feedback = con.execute(
            "SELECT event_id,training_status FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()
    assert protected_id in event_ids
    assert expired_id not in event_ids
    assert tuple(feedback) == (protected_id, "pending")


def _isolate_storage_mutation_gate(monkeypatch):
    restart_required = threading.Event()
    monkeypatch.setattr(
        main,
        "_STORAGE_MUTATION_GATE",
        main.WriterPreferredGate(),
    )
    monkeypatch.setattr(
        main,
        "_STORAGE_RESTART_REQUIRED",
        restart_required,
    )
    return restart_required


def _prepare_storage_migration(tmp_path, monkeypatch):
    from app.ai import live_worker

    restart_required = _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    storage_config = tmp_path / "storage_config.json"
    original_config = '{"storage_root":"old"}'
    storage_config.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(main.manager, "stop_all", lambda: True)
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: True,
    )
    monkeypatch.setattr(
        live_worker,
        "backup_live_anpr_outbox",
        lambda target: Path(target).write_bytes(b"durable-retry")
        or Path(target),
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("restart-cameras"),
    )
    return restart_required, storage_config, original_config, calls


def test_storage_migration_publish_falls_back_without_hardlinks(
    tmp_path,
    monkeypatch,
):
    staged = tmp_path / "staged.db"
    target = tmp_path / "published.db"
    staged.write_bytes(b"complete-snapshot")
    details = staged.lstat()
    staged_identity = path_file_identity(staged, details=details)
    monkeypatch.setattr(
        main.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("hardlinks unsupported")
        ),
    )

    published_identity = main._publish_staged_file(
        staged,
        target,
        staged_identity,
    )

    current = target.lstat()
    assert published_identity == path_file_identity(target, details=current)
    assert published_identity != staged_identity
    assert current.st_nlink == 1
    assert target.read_bytes() == b"complete-snapshot"
    assert not staged.exists()


@pytest.mark.parametrize(
    "path",
    ["/live/7", "/logout"],
)
def test_storage_gate_includes_side_effecting_get_routes(path):
    request = SimpleNamespace(
        method="GET",
        url=SimpleNamespace(path=path),
    )

    assert main._uses_storage_mutation_gate(request) is True


def test_streamed_csv_export_is_not_a_storage_mutation():
    request = SimpleNamespace(
        method="GET",
        url=SimpleNamespace(path="/events/export.csv"),
    )

    assert main._uses_storage_mutation_gate(request) is False


@pytest.mark.parametrize("path", sorted(main._VIDEO_UPLOAD_PATHS))
def test_video_upload_uses_global_post_parse_mutation_gate(path):
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path=path),
    )

    assert main._uses_storage_mutation_gate(request) is True


def test_only_storage_migration_uses_exclusive_mutation_gate():
    storage_request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/settings/storage"),
    )
    display_request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/settings/display"),
    )

    assert main._uses_exclusive_storage_gate(storage_request) is True
    assert main._uses_exclusive_storage_gate(display_request) is False


def test_low_privilege_storage_request_never_queues_writer(
    monkeypatch,
):
    _as_role(monkeypatch, "guard")
    gate = main.WriterPreferredGate()
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    request = main.Request({
        "type": "http",
        "method": "POST",
        "scheme": "http",
        "path": "/settings/storage",
        "raw_path": b"/settings/storage",
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "state": {},
    })

    asyncio.run(main._storage_mutation_dependency(request))

    assert gate.snapshot() == (0, False, 0)


@pytest.mark.parametrize("stop_result", [False, None])
def test_storage_migration_aborts_before_snapshots_unless_stream_stop_is_true(
    tmp_path,
    monkeypatch,
    stop_result,
):
    from app.ai import live_worker

    restart_required = _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    monkeypatch.setattr(
        main,
        "STORAGE_CONFIG_PATH",
        tmp_path / "storage_config.json",
    )
    database.init_db()
    _set_current_storage(old_root)
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(main.manager, "stop_all", lambda: stop_result)
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda **_kwargs: calls.append("shutdown") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("restart-cameras"),
    )
    monkeypatch.setattr(
        main,
        "create_database_backup",
        lambda _target: calls.append("database-snapshot"),
    )

    response = main.save_storage_settings(
        object(),
        **_storage_form(tmp_path / "new"),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert "shutdown" not in calls
    assert "database-snapshot" not in calls
    assert calls == []
    assert restart_required.is_set()


def test_same_root_media_path_change_quiesces_background_writers(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    _isolate_storage_mutation_gate(monkeypatch)
    root = tmp_path / "storage"
    root.mkdir()
    db_path = root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DATA_DIR", root)
    database.init_db()
    _set_current_storage(root)
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: calls.append("worker") or True,
    )
    monkeypatch.setattr(
        main,
        "require_media_writes_quiescent",
        lambda: calls.append("quiescent"),
    )
    monkeypatch.setattr(
        main,
        "run_retention_cleanup",
        lambda: calls.append("retention"),
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("restart-cameras"),
    )
    form = _storage_form(root)
    form["snapshot_path"] = str(root / "snapshots-v2")

    response = main.save_storage_settings(object(), **form)

    assert response.status_code == 303
    assert "saved=1" in response.headers["location"]
    assert calls == [
        "streams",
        "worker",
        "quiescent",
        "retention",
        "restart-worker",
        "restart-cameras",
    ]
    assert database.get_setting("snapshot_path", "") == str(
        (root / "snapshots-v2").resolve()
    )


def test_storage_migration_quiesces_and_copies_retry_outbox(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    restart_required = _isolate_storage_mutation_gate(monkeypatch)

    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage_config = bootstrap / "storage_config.json"
    storage_config.write_text(
        '{"storage_root":"old"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    (old_root / ".secret").write_bytes(b"stable-secret")
    _as_role(monkeypatch, "system")
    calls = []
    real_backup = main.create_database_backup

    def backup_database(target):
        calls.append("database")
        return real_backup(target)

    def backup_outbox(target):
        calls.append("outbox")
        Path(target).write_bytes(b"durable-retry")
        return Path(target)

    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: calls.append("worker") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "backup_live_anpr_outbox",
        backup_outbox,
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(main, "create_database_backup", backup_database)
    new_root = tmp_path / "new"

    response = main.save_storage_settings(object(), **_storage_form(new_root))

    assert response.status_code == 303
    assert "restart=1" in response.headers["location"]
    assert calls == ["streams", "worker", "database", "outbox"]
    assert (new_root / "bcvision-retry.db").read_bytes() == b"durable-retry"
    assert (new_root / ".secret").read_bytes() == b"stable-secret"
    with sqlite3.connect(new_root / "bcvision.db") as con:
        moved_root = con.execute(
            "SELECT value FROM settings WHERE key='storage_root'"
        ).fetchone()[0]
    assert moved_root == str(new_root.resolve())
    assert main.json.loads(storage_config.read_text(encoding="utf-8")) == {
        "storage_root": str(new_root.resolve())
    }
    marker = storage_config.with_name(
        main.STORAGE_MIGRATION_MARKER_NAME
    )
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    assert restart_required.is_set()


def test_storage_migration_rolls_back_if_outbox_snapshot_fails(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    restart_required = _isolate_storage_mutation_gate(monkeypatch)

    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage_config = bootstrap / "storage_config.json"
    original_config = '{"storage_root":"old"}'
    storage_config.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: calls.append("worker") or True,
    )

    rollback_artifacts = [
        "bcvision.db",
        "bcvision.db-wal",
        "bcvision.db-shm",
        "bcvision.db-journal",
        "bcvision-retry.db",
        "bcvision-retry.db-wal",
        "bcvision-retry.db-shm",
        "bcvision-retry.db-journal",
    ]

    def fail_outbox_snapshot(target):
        root = Path(target).parent
        for name in rollback_artifacts:
            (root / name).write_bytes(b"partial-migration")
        raise OSError("outbox unavailable")

    monkeypatch.setattr(
        live_worker,
        "backup_live_anpr_outbox",
        fail_outbox_snapshot,
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("restart-cameras"),
    )
    new_root = tmp_path / "new"

    response = main.save_storage_settings(object(), **_storage_form(new_root))

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert calls == [
        "streams",
        "worker",
        "restart-worker",
        "restart-cameras",
    ]
    assert all(
        not (new_root / name).exists()
        for name in rollback_artifacts
    )
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not restart_required.is_set()


def test_storage_migration_rolls_back_if_worker_does_not_quiesce(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    restart_required = _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    storage_config = tmp_path / "storage_config.json"
    original_config = '{"storage_root":"old"}'
    storage_config.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams") or True,
    )
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: calls.append("worker") or False,
    )
    monkeypatch.setattr(
        live_worker,
        "start_live_anpr_worker",
        lambda: calls.append("restart-worker"),
    )
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: calls.append("restart-cameras"),
    )
    backups = []
    monkeypatch.setattr(
        main,
        "create_database_backup",
        lambda target: backups.append(Path(target)),
    )
    new_root = tmp_path / "new"

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert calls == ["streams", "worker"]
    assert backups == []
    assert not (new_root / "bcvision.db").exists()
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert restart_required.is_set()


def test_storage_migration_refuses_active_training(tmp_path, monkeypatch):
    restart_required = _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    storage_config = tmp_path / "storage_config.json"
    original_config = '{"storage_root":"old"}'
    storage_config.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    with database.connect() as con:
        con.execute(
            "INSERT INTO anpr_training_runs(status) VALUES('running')"
        )
    _as_role(monkeypatch, "system")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams"),
    )
    new_root = tmp_path / "new"

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert calls == []
    assert not (new_root / "bcvision.db").exists()
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not restart_required.is_set()


@pytest.mark.parametrize(
    "orphan_name",
    [
        "bcvision.db-wal",
        "bcvision.db-shm",
        "bcvision.db-journal",
        "bcvision-retry.db-wal",
        "bcvision-retry.db-shm",
        "bcvision-retry.db-journal",
    ],
)
def test_storage_migration_rejects_orphan_sqlite_sidecars_before_quiesce(
    tmp_path,
    monkeypatch,
    orphan_name,
):
    restart_required = _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    storage_config = tmp_path / "storage_config.json"
    original_config = '{"storage_root":"old"}'
    storage_config.write_text(original_config, encoding="utf-8")
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    _as_role(monkeypatch, "system")
    new_root = tmp_path / "new"
    new_root.mkdir()
    orphan = new_root / orphan_name
    orphan.write_bytes(b"orphaned-sqlite-state")
    calls = []
    monkeypatch.setattr(
        main.manager,
        "stop_all",
        lambda: calls.append("streams"),
    )

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert calls == []
    assert orphan.read_bytes() == b"orphaned-sqlite-state"
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not restart_required.is_set()


def test_storage_migration_rejects_dangling_secret_symlink_without_following(
    tmp_path,
    monkeypatch,
):
    restart_required, storage_config, original_config, calls = (
        _prepare_storage_migration(tmp_path, monkeypatch)
    )
    new_root = tmp_path / "new"
    new_root.mkdir()
    outside = tmp_path / "outside-secret"
    dangling = new_root / ".secret"
    try:
        dangling.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert os.path.lexists(dangling)
    assert dangling.is_symlink()
    assert not outside.exists()
    assert calls == []
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not restart_required.is_set()


def test_storage_migration_preserves_published_root_if_replace_reports_late_error(
    tmp_path,
    monkeypatch,
):
    restart_required, storage_config, _original_config, calls = (
        _prepare_storage_migration(tmp_path, monkeypatch)
    )
    original_replace = Path.replace

    def replace_then_raise(path, target):
        result = original_replace(path, target)
        if Path(target) == storage_config:
            raise OSError("replace reported a late error")
        return result

    monkeypatch.setattr(Path, "replace", replace_then_raise)
    new_root = tmp_path / "new"

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert main.json.loads(storage_config.read_text(encoding="utf-8")) == {
        "storage_root": str(new_root.resolve())
    }
    assert (new_root / "bcvision.db").is_file()
    assert (new_root / "bcvision-retry.db").is_file()
    marker = storage_config.with_name(
        main.STORAGE_MIGRATION_MARKER_NAME
    )
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    assert calls == []
    assert restart_required.is_set()


def test_storage_migration_preserves_foreign_config_temp_collision(
    tmp_path,
    monkeypatch,
):
    restart_required, storage_config, original_config, calls = (
        _prepare_storage_migration(tmp_path, monkeypatch)
    )
    monkeypatch.setattr(main.secrets, "token_hex", lambda _size=8: "fixed")
    collision = storage_config.with_name(
        f".{storage_config.name}.fixed.tmp"
    )
    collision.write_bytes(b"foreign-config-temp")
    new_root = tmp_path / "new"

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert collision.read_bytes() == b"foreign-config-temp"
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not (new_root / "bcvision.db").exists()
    assert not (new_root / "bcvision-retry.db").exists()
    assert calls == ["restart-worker", "restart-cameras"]
    assert not restart_required.is_set()


def test_storage_migration_never_deletes_foreign_sidecar_from_failing_snapshot(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    restart_required, storage_config, original_config, calls = (
        _prepare_storage_migration(tmp_path, monkeypatch)
    )
    new_root = tmp_path / "new"
    foreign_sidecar = new_root / "bcvision-retry.db-wal"

    def fail_outbox(_target):
        foreign_sidecar.write_bytes(b"foreign-sidecar")
        raise OSError("outbox snapshot failed")

    monkeypatch.setattr(
        live_worker,
        "backup_live_anpr_outbox",
        fail_outbox,
    )

    response = main.save_storage_settings(
        object(),
        **_storage_form(new_root),
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?error=")
    assert foreign_sidecar.read_bytes() == b"foreign-sidecar"
    assert storage_config.read_text(encoding="utf-8") == original_config
    assert not (new_root / "bcvision.db").exists()
    assert calls == ["restart-worker", "restart-cameras"]
    assert not restart_required.is_set()


def test_successful_storage_migration_serializes_and_rejects_late_mutation(
    tmp_path,
    monkeypatch,
):
    from app.ai import live_worker

    _isolate_storage_mutation_gate(monkeypatch)
    old_root = tmp_path / "old"
    old_root.mkdir()
    old_db = old_root / "bcvision.db"
    monkeypatch.setattr(database, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DB_PATH", old_db)
    monkeypatch.setattr(main, "DATA_DIR", old_root)
    storage_config = tmp_path / "storage_config.json"
    storage_config.write_text(
        '{"storage_root":"old"}',
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "STORAGE_CONFIG_PATH", storage_config)
    database.init_db()
    _set_current_storage(old_root)
    with database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled) VALUES(?,?,1)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
    _as_role(monkeypatch, "system")
    backup_entered = threading.Event()
    release_backup = threading.Event()
    real_backup = main.create_database_backup

    def blocked_database_backup(target):
        backup_entered.set()
        assert release_backup.wait(5.0)
        return real_backup(target)

    def backup_outbox(target):
        Path(target).write_bytes(b"durable-retry")
        return Path(target)

    monkeypatch.setattr(main, "create_database_backup", blocked_database_backup)
    monkeypatch.setattr(main.manager, "stop_all", lambda: True)
    monkeypatch.setattr(main.manager, "start_enabled_cameras", lambda: 0)
    monkeypatch.setattr(live_worker, "start_live_anpr_worker", lambda: None)
    monkeypatch.setattr(
        live_worker,
        "shutdown_live_anpr_worker",
        lambda retry_timeout=5.0: True,
    )
    monkeypatch.setattr(
        live_worker,
        "backup_live_anpr_outbox",
        backup_outbox,
    )
    stream_starts = []

    class _EmptyStream:
        @staticmethod
        def frames():
            return iter(())

    monkeypatch.setattr(
        main.manager,
        "get",
        lambda *args, **kwargs: stream_starts.append((args, kwargs))
        or _EmptyStream(),
    )
    display_writes = []
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: display_writes.append((key, value)),
    )
    new_root = tmp_path / "new"
    migration_form = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in _storage_form(new_root).items()
        if value is not None
    }
    responses = {}
    errors = []

    def request_migration(client):
        try:
            responses["migration"] = client.post(
                "/settings/storage",
                data=migration_form,
                follow_redirects=False,
            )
        except Exception as exc:
            errors.append(exc)

    def request_display_change(client):
        try:
            responses["display"] = client.post(
                "/settings/display",
                data={
                    "dashboard_grid": "3",
                    "dashboard_event_rows": "12",
                    "live_fps": "5",
                    "stream_width": "640",
                    "jpeg_quality": "70",
                },
                follow_redirects=False,
            )
        except Exception as exc:
            errors.append(exc)

    def request_live_stream(client):
        try:
            responses["live"] = client.get(
                f"/live/{camera_id}",
                follow_redirects=False,
            )
        except Exception as exc:
            errors.append(exc)

    with TestClient(main.app) as client:
        migration_thread = threading.Thread(
            target=request_migration,
            args=(client,),
        )
        migration_thread.start()
        assert backup_entered.wait(2.0)
        display_thread = threading.Thread(
            target=request_display_change,
            args=(client,),
        )
        display_thread.start()
        live_thread = threading.Thread(
            target=request_live_stream,
            args=(client,),
        )
        live_thread.start()
        time.sleep(0.05)
        assert display_thread.is_alive()
        assert live_thread.is_alive()
        release_backup.set()
        migration_thread.join(timeout=5.0)
        display_thread.join(timeout=5.0)
        live_thread.join(timeout=5.0)

    assert not migration_thread.is_alive()
    assert not display_thread.is_alive()
    assert not live_thread.is_alive()
    assert errors == []
    assert responses["migration"].status_code == 303
    assert responses["display"].status_code == 503
    assert responses["display"].json()["error"] == (
        "storage-restart-required"
    )
    assert responses["live"].status_code == 503
    assert responses["live"].json()["error"] == (
        "storage-restart-required"
    )
    assert display_writes == []
    assert stream_starts == []


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


def test_media_response_holds_read_pin_until_file_is_sent(tmp_path, monkeypatch):
    _as_role(monkeypatch, "guard")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    image = snapshots / "event.jpg"
    image.write_bytes(b"jpeg-evidence")
    closed = []

    class Lease:
        def close(self):
            closed.append(True)

    settings = {
        "snapshot_path": str(snapshots),
        "plate_path": str(tmp_path / "plates"),
        "video_path": str(tmp_path / "videos"),
    }
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        main,
        "pin_media_paths",
        lambda paths: Lease() if tuple(paths) == (image,) else None,
    )

    with TestClient(main.app) as client:
        response = client.get("/media", params={"path": str(image)})

    assert response.status_code == 200
    assert response.content == b"jpeg-evidence"
    assert closed == [True]


def test_media_response_releases_read_pin_when_send_fails(tmp_path):
    image = tmp_path / "event.jpg"
    image.write_bytes(b"jpeg-evidence")
    closed = []

    class Lease:
        def close(self):
            closed.append(True)

    response = main._PinnedFileResponse(image, read_pin=Lease())
    scope = {
        "type": "http",
        "method": "GET",
        "scheme": "http",
        "path": "/media",
        "raw_path": b"/media",
        "query_string": b"",
        "headers": [],
    }

    async def receive():
        return {"type": "http.disconnect"}

    async def send(_message):
        raise RuntimeError("client disconnected")

    async def scenario():
        with pytest.raises(RuntimeError, match="client disconnected"):
            await response(scope,receive,send)

    asyncio.run(scenario())
    assert closed == [True]


def test_legacy_video_upload_ignores_traversal_filename(tmp_path, monkeypatch):
    _as_role(monkeypatch, "operator")
    _allow_test_upload_handoff(monkeypatch)
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )

    async def reject_invalid_video(*_args, **_kwargs):
        raise RuntimeError("invalid video fixture")

    monkeypatch.setattr(
        main,
        "run_module_job_subprocess",
        reject_invalid_video,
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("../../owned.mp4", b"not-a-real-video", "video/mp4")},
        )

    assert response.status_code == 200
    assert not (tmp_path / "owned.mp4").exists()
    saved = list(video_dir.iterdir())
    # Invalid input and all partial outputs are removed after processing fails.
    assert saved == []


def test_cancelled_video_test_waits_for_worker_and_preserves_partial_evidence(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    videos = tmp_path / "videos"
    plates = tmp_path / "plates"
    snapshots = tmp_path / "snapshots"
    roots = {
        "video_path": videos,
        "plate_path": plates,
        "snapshot_path": snapshots,
    }
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, _default: roots[setting_key],
    )

    class Reservation:
        def grow(self, _size):
            return None

        def close(self, **_kwargs):
            return None

    class Lease:
        def close(self):
            return None

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )
    monkeypatch.setattr(main, "pin_media_paths", lambda _paths: Lease())
    monkeypatch.setattr(
        main,
        "fsync_parent_directory",
        lambda _path: None,
    )

    class Upload:
        filename = "cancelled.mp4"
        consumed = False

        async def read(self, _size):
            if not self.consumed:
                self.consumed = True
                return b"video"
            return b""

    started = threading.Event()
    stopped = threading.Event()

    async def slow_process(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.sleep(3600)
        finally:
            # The real subprocess helper reaches this point only after the
            # child has been terminated and reaped.
            stopped.set()

    monkeypatch.setattr(main, "run_module_job_subprocess", slow_process)

    async def scenario():
        task = asyncio.create_task(
            main.ai_video_test_upload(SimpleNamespace(), Upload())
        )
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())

    assert not list(videos.rglob("*"))
    assert not list(plates.rglob("*"))
    assert not list(snapshots.rglob("*"))


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

    assert response.status_code == 413
    assert response.json()["error"] == "video-upload-too-large"
    assert not video_dir.exists()


def test_video_test_displays_plate_crop_and_text_on_one_row(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    db_path = tmp_path / "video-test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    storage = tmp_path / "storage"
    videos = storage / "videos"
    plates = storage / "plates"
    snapshots = storage / "snapshots"
    backups = storage / "backups"
    for folder in (videos, plates, snapshots, backups):
        folder.mkdir(parents=True)
    _set_current_storage(storage)
    settings = {
        "storage_root": str(storage),
        "video_path": str(videos),
        "plate_path": str(plates),
        "snapshot_path": str(snapshots),
        "backup_path": str(backups),
    }
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )

    async def fake_process(
        module_name,
        function_name,
        video_path,
        plate_dir,
        snapshot_dir,
        **kwargs,
    ):
        from app.ai.video_test import serialize_process_video_result

        assert module_name == "app.ai.video_test"
        assert function_name == "process_video_transport"
        assert Path(video_path).is_file()
        assert kwargs["frame_step"] == 1
        assert kwargs["include_candidate_shadow"] is False
        assert not Path(plate_dir).exists()
        assert not Path(snapshot_dir).exists()
        image = np.full((48, 120, 3), 180, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        jpeg = bytes(encoded)
        return serialize_process_video_result(
            {
                "frames": 12,
                "fps": 8.0,
                "width": 1920,
                "height": 1080,
                "duration": 1.5,
                "detector_variant": "yolov8n",
                "detector_execution_mode": "exclusive-baseline",
            },
            [{
                "plate": "31-ط-556-74",
                "plate_norm": "31ط55674",
                "plate_path": "",
                "image_path": "",
                "media_status": "complete",
                "confidence": 0.91,
                "video_second": 0.75,
                "ocr_engine": "hezar-crnn-fa-v2-onnx",
                "method": "yolov8n-plate-onnx",
                "engine_lane": "baseline",
                "valid": True,
                "needs_review": False,
                "_transport_plate_jpeg": jpeg,
                "_transport_vehicle_jpeg": jpeg,
                "_transport_plate_error": "",
                "_transport_vehicle_error": "",
            }],
        )

    monkeypatch.setattr(main, "run_module_job_subprocess", fake_process)

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("golden.mp4", b"fixture", "video/mp4")},
        )

    assert response.status_code == 200
    assert "تصویر پلاک / متن تشخیص‌داده‌شده" in response.text
    assert "31-ط-556-74" in response.text
    assert "YOLOv8n" in response.text
    assert "Shadow/Next اجرا نشدند" in response.text
    assert "hezar-crnn-fa-v2-onnx" in response.text
    assert "/media?path=" in response.text
    assert "تصویر خودرو" in response.text
    with database.connect() as con:
        archived = con.execute(
            "SELECT plate_norm,plate_image_path,image_path,video_path,"
            "source,plate_region FROM plate_events ORDER BY id"
        ).fetchall()
    assert len(archived) == 1
    assert archived[0]["plate_norm"] == "31ط55674"
    assert archived[0]["plate_region"] == "74"
    assert Path(archived[0]["plate_image_path"]).is_file()
    assert Path(archived[0]["plate_image_path"]).is_relative_to(plates)
    assert Path(archived[0]["image_path"]).is_file()
    assert Path(archived[0]["image_path"]).is_relative_to(snapshots)
    assert Path(archived[0]["video_path"]).is_file()
    assert archived[0]["source"] == "video-test"


def test_video_test_displays_selected_detector_inference_error(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    _allow_test_upload_handoff(monkeypatch)
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda _setting_key, _default: video_dir,
    )
    async def fail_inference(*_args, **_kwargs):
        raise RuntimeError("YOLOv8n inference failed: invalid tensor")

    monkeypatch.setattr(
        main,
        "run_module_job_subprocess",
        fail_inference,
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("failure.mp4", b"fixture", "video/mp4")},
        )

    assert response.status_code == 200
    assert "خطا: YOLOv8n inference failed: invalid tensor" in response.text
    assert "هیچ پلاکی در این ویدئو تشخیص داده نشد" not in response.text
    assert list(video_dir.rglob("*")) == []


def test_video_test_timeout_is_localized_and_removes_source(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    _allow_test_upload_handoff(monkeypatch)
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda _setting_key, _default: video_dir,
    )

    async def time_out(*_args, **_kwargs):
        raise main.SubprocessJobTimeout("child exceeded deadline")

    monkeypatch.setattr(main, "run_module_job_subprocess", time_out)

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("timeout.mp4", b"fixture", "video/mp4")},
        )

    assert response.status_code == 200
    assert "زمان پردازش از حد امن عبور کرد" in response.text
    assert list(video_dir.rglob("*")) == []


def test_auto_confirmed_event_requires_operator_action_before_training(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "operator-review.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    captured = []
    monkeypatch.setattr(
        main,
        "capture_feedback_sample",
        lambda feedback_id: captured.append(feedback_id),
    )
    _as_role(monkeypatch, "operator")
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events("
            "plate_text,plate_norm,confidence,review_status,"
            "confirmation_source,experimental,raw_guess_text,"
            "raw_guess_norm,raw_guess_confidence,raw_guess_engine,"
            "model_revision"
            ") VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                "31-ط-556-74",
                "31ط55674",
                0.69,
                "auto-confirmed",
                "ai-auto-guess",
                1,
                "31-ط-556-74",
                "31ط55674",
                0.93,
                "fast-plate-ocr-cct",
                "rc15-stage4",
            ),
        ).lastrowid

    with database.connect() as con:
        dashboard_row = con.execute(
            "SELECT id,plate_text,camera_name,confidence,created_at,"
            "image_path,plate_image_path,review_status "
            "FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
    return_to = (
        "/dashboard?video=1&events_camera=7&events_after=3"
        "&events_snapshot=9&events_page=2&ignored=https://evil.invalid"
    )
    dashboard_html = main.dashboard_event_row(dashboard_row, return_to)
    response = main.correct_event_plate(
        event_id,
        SimpleNamespace(client=None),
        "31 ط 556 ایران 74",
        return_to,
    )
    retry_response = main.correct_event_plate(
        event_id,
        SimpleNamespace(client=None),
        "31 ط 556 ایران 74",
        return_to,
    )

    assert "تأیید خودکار مدل" in dashboard_html
    assert response.status_code == 303
    assert response.headers["location"] == (
        "/dashboard?video=1&events_camera=7&events_after=3"
        "&events_snapshot=9&events_page=2&corrected=1"
    )
    with database.connect() as con:
        event = con.execute(
            "SELECT review_status,confirmation_source,"
            "operator_reviewed,experimental FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
        feedback = con.execute(
            "SELECT id,observed_norm,corrected_norm,exact_match,"
            "submitted_by FROM anpr_feedback WHERE event_id=?",
            (event_id,),
        ).fetchone()
        feedback_count = con.execute(
            "SELECT COUNT(*) FROM anpr_feedback WHERE event_id=?",
            (event_id,),
        ).fetchone()[0]
    assert dict(event) == {
        "review_status": "confirmed",
        "confirmation_source": "operator",
        "operator_reviewed": 1,
        "experimental": 0,
    }
    assert feedback["observed_norm"] == "31ط55674"
    assert feedback["corrected_norm"] == "31ط55674"
    assert feedback["exact_match"] == 1
    assert feedback["submitted_by"] == "operator"
    assert retry_response.status_code == 303
    assert feedback_count == 1
    assert captured == [feedback["id"], feedback["id"]]

    changed_response = main.correct_event_plate(
        event_id,
        SimpleNamespace(client=None),
        "31 ط 557 ایران 74",
        return_to,
    )
    with database.connect() as con:
        history = con.execute(
            "SELECT id,status,corrected_norm FROM anpr_feedback "
            "WHERE event_id=? ORDER BY id",
            (event_id,),
        ).fetchall()
    assert changed_response.status_code == 303
    assert [row["status"] for row in history] == [
        "superseded",
        "confirmed",
    ]
    assert history[-1]["corrected_norm"] == "31ط55774"
    assert captured[-1] == history[-1]["id"]


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
    monkeypatch.setattr(
        main.manager,
        "start_enabled_cameras",
        lambda: 0,
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

    with TestClient(main.app) as client, source_path.open("rb") as source:
        # This assertion guards the upload request itself. Application
        # startup/shutdown also reconciles storage and drains camera threads;
        # those independent lifecycle costs vary substantially across OpenCV
        # builds, especially on Windows.
        started = time.monotonic()
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
    assert payload["redirect"] == (
        "/dashboard?video=1&events_camera="
        f"{payload['camera_id']}"
    )
    assert elapsed < 3
    with database.connect() as con:
        uploaded = con.execute(
            "SELECT * FROM cameras WHERE rtsp_url LIKE 'video://%'"
        ).fetchall()
    assert len(uploaded) == 1
    assert uploaded[0]["name"] == "ویدئو: traffic"
    assert calls["get"][0][0] == uploaded[0]["id"]
    assert calls["get"][0][1].startswith("video://")


def _fake_camera_upload_dependencies(monkeypatch, video_dir):
    from app.ai import video_test

    _allow_test_upload_handoff(monkeypatch)
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda _setting_key, _default: video_dir,
    )
    monkeypatch.setattr(main.manager, "start_enabled_cameras", lambda: 0)
    monkeypatch.setattr(main.manager, "stop_all", lambda: True)

    class VideoProbe:
        def __init__(self, _target):
            pass

        def info(self):
            return {"frames": 1, "fps": 1.0, "duration": 1.0}

        def close(self):
            return None

    monkeypatch.setattr(video_test, "VideoTester", VideoProbe)


class _MemoryVideoUpload:
    filename = "new.avi"

    def __init__(self):
        self._chunks = [b"video", b""]

    async def read(self, _size):
        return self._chunks.pop(0)


def _run_camera_upload(source_id):
    request = SimpleNamespace(
        headers={"x-requested-with": "XMLHttpRequest"},
    )
    return asyncio.run(
        main.cameras_video_upload(
            request,
            camera_id=source_id,
            video=_MemoryVideoUpload(),
        )
    )


@pytest.mark.parametrize("operation", ["delete", "edit"])
def test_video_upload_serializes_with_old_camera_mutation(
    tmp_path,
    monkeypatch,
    operation,
):
    _as_role(monkeypatch, "system")
    db_path = tmp_path / f"upload-vs-{operation}.db"
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as connection:
        source_id = int(connection.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled,lpr_enabled) "
            "VALUES('Gate','rtsp://gate',1,1)"
        ).lastrowid)
        old_id = int(connection.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled,lpr_enabled) "
            "VALUES('Old','video:///old.avi',1,1)"
        ).lastrowid)

    _fake_camera_upload_dependencies(monkeypatch, video_dir)
    upload_inside_handoff = threading.Event()
    release_upload = threading.Event()
    remove_calls = []

    def block_new_stream(*_args):
        upload_inside_handoff.set()
        assert release_upload.wait(3.0)

    monkeypatch.setattr(main.manager, "get", block_new_stream)
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda camera_id: remove_calls.append(int(camera_id)) or True,
    )
    upload_responses = []
    mutation_responses = []

    upload_thread = threading.Thread(
        target=lambda: upload_responses.append(_run_camera_upload(source_id))
    )
    upload_thread.start()
    assert upload_inside_handoff.wait(2.0)

    def run_mutation():
        if operation == "delete":
            response = main.delete_cam(old_id, object())
        else:
            response = _edit_camera_direct(old_id)
        mutation_responses.append(response)

    mutation_thread = threading.Thread(target=run_mutation)
    mutation_thread.start()
    mutation_thread.join(0.15)
    blocked_behind_upload = mutation_thread.is_alive()
    calls_before_release = list(remove_calls)

    release_upload.set()
    upload_thread.join(3.0)
    mutation_thread.join(3.0)

    assert blocked_behind_upload is True
    assert calls_before_release == []
    assert len(upload_responses) == len(mutation_responses) == 1
    assert upload_responses[0].status_code == 200
    assert mutation_responses[0].status_code == 303
    with database.connect() as connection:
        video_rows = connection.execute(
            "SELECT id FROM cameras WHERE rtsp_url LIKE 'video://%' "
            "ORDER BY id"
        ).fetchall()
    assert len(video_rows) == 1
    assert int(video_rows[-1]["id"]) != old_id


def test_multiple_test_video_uploads_are_kept_as_separate_cameras(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "system")
    db_path = tmp_path / "multiple-test-videos.db"
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _fake_camera_upload_dependencies(monkeypatch, video_dir)
    monkeypatch.setattr(main.manager, "get", lambda *_args: None)

    first = _run_camera_upload(0)
    second = _run_camera_upload(0)

    assert first.status_code == second.status_code == 200
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id,name,enabled FROM cameras "
            "WHERE rtsp_url LIKE 'video://%' ORDER BY id"
        ).fetchall()
    assert len(rows) == 2
    assert all(int(row["enabled"]) == 1 for row in rows)


def test_video_upload_preserves_old_owner_when_stream_will_not_stop(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "system")
    db_path = tmp_path / "upload-old-stop-failure.db"
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as connection:
        source_id = int(connection.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled,lpr_enabled) "
            "VALUES('Gate','rtsp://gate',1,1)"
        ).lastrowid)
        old_id = int(connection.execute(
            "INSERT INTO cameras(name,rtsp_url,enabled,lpr_enabled) "
            "VALUES('Old','video:///old.avi',1,1)"
        ).lastrowid)

    _fake_camera_upload_dependencies(monkeypatch, video_dir)
    monkeypatch.setattr(main.manager, "get", lambda *_args: None)
    remove_calls = []
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda camera_id: remove_calls.append(int(camera_id)) or False,
    )

    response = _run_camera_upload(source_id)

    assert response.status_code == 200
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT id,rtsp_url FROM cameras WHERE rtsp_url LIKE 'video://%' "
            "ORDER BY id"
        ).fetchall()
    new_id = int(rows[-1]["id"])
    assert [int(row["id"]) for row in rows] == [old_id, new_id]
    assert rows[0]["rtsp_url"] == "video:///old.avi"
    assert remove_calls == []


def test_failed_virtual_stream_start_preserves_previous_camera(
    tmp_path,
    monkeypatch,
):
    _as_role(monkeypatch, "operator")
    db_path = tmp_path / "upload-failure.db"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    old_video = video_dir / "old.avi"
    old_video.write_bytes(b"old")
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
        source_id = int(con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,city,enabled,is_demo,lpr_enabled"
            ") VALUES(?,?,?,?,?,?,?)",
            ("Gate", "rtsp://gate", "Gate", "تهران", 1, 0, 1),
        ).lastrowid)
        old_id = int(con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,city,enabled,is_demo,lpr_enabled"
            ") VALUES(?,?,?,?,?,?,?)",
            (
                "Old video", f"video://{old_video}",
                "old", "تهران", 1, 1, 1,
            ),
        ).lastrowid)

    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda setting_key, default: video_dir,
    )
    removed = []
    monkeypatch.setattr(
        main.manager,
        "get",
        lambda *_args: (_ for _ in ()).throw(
            RuntimeError("stream start failed")
        ),
    )
    monkeypatch.setattr(
        main.manager,
        "remove",
        lambda camera_id: removed.append(camera_id) or True,
    )

    with TestClient(main.app) as client, source_path.open("rb") as source:
        response = client.post(
            "/cameras/video-upload",
            data={"camera_id": str(source_id)},
            files={"video": ("traffic.avi", source, "video/x-msvideo")},
            headers={"X-Requested-With": "XMLHttpRequest"},
        )

    assert response.status_code == 500
    with database.connect() as con:
        virtual_rows = con.execute(
            "SELECT id,rtsp_url FROM cameras "
            "WHERE rtsp_url LIKE 'video://%'"
        ).fetchall()
    assert [row["id"] for row in virtual_rows] == [old_id]
    assert virtual_rows[0]["rtsp_url"] == f"video://{old_video}"
    assert old_id not in removed
    assert list(video_dir.iterdir()) == [old_video]


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
            "name,rtsp_url,location,city,enabled,is_demo,lpr_enabled,"
            "lpr_confidence,frame_step,duplicate_seconds"
            ") VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "Gate", "rtsp://gate", "Gate", "کرج",
                1, 0, 1, 50, 1, 0,
            ),
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
    virtual_id = None
    try:
        with TestClient(main.app) as client, source_path.open("rb") as source:
            # The lifespan replaces a worker that a preceding TestClient has
            # already stopped. Patch the active worker, not the pre-lifespan
            # instance; otherwise the Windows suite can silently run the real
            # model readiness path and finish the video without a test event.
            monkeypatch.setattr(
                live_worker.worker,
                "_models",
                lambda: {
                    "ready": True,
                    "detector_ready": True,
                    "crnn_ready": True,
                    "cnn_ready": True,
                },
            )
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
                if (
                    event
                    and event["plate_norm"] == "12ب34567"
                    and stream
                    and stream.latest
                ):
                    break
                time.sleep(0.02)

            stream = main.manager.streams.get(virtual_id)
            for _ in range(250):
                if stream and stream.state.ended:
                    break
                time.sleep(0.02)
            assert stream is not None
            assert stream.state.ended is True
            dashboard = client.get(response.json()["redirect"])

            assert event is not None
            assert event["plate_norm"] == "12ب34567"
            assert event["plate_region"] == "67"
            assert event["city"] == "کرج"
            assert event["media_status"] == "complete"
            assert Path(event["plate_image_path"]).is_file()
            assert Path(event["image_path"]).is_file()
            assert Path(event["video_path"]).is_file()
            plate_media = client.get(
                "/media",
                params={"path": event["plate_image_path"]},
            )
            vehicle_media = client.get(
                "/media",
                params={"path": event["image_path"]},
            )
            assert plate_media.status_code == 200
            assert vehicle_media.status_code == 200
            assert plate_media.content.startswith(b"\xff\xd8")
            assert vehicle_media.content.startswith(b"\xff\xd8")
            with database.connect() as con:
                count = con.execute(
                    "SELECT COUNT(*) FROM plate_events WHERE camera_id=?",
                    (virtual_id,),
                ).fetchone()[0]
            assert count == 1
            assert "تصویر پلاک / پلاک خوانده‌شده" in dashboard.text
            assert "پلاک‌های این ویدئو" in dashboard.text
            assert "پاک‌کردن نمایش" in dashboard.text
            assert ">۱</b>" in dashboard.text
            assert "بازپخش فقط نمایشی است" in dashboard.text

            with database.connect() as con:
                marker = con.execute(
                    "SELECT video_anpr_started,video_anpr_completed,"
                    "video_anpr_completed_at FROM cameras WHERE id=?",
                    (virtual_id,),
                ).fetchone()
            assert marker[0:2] == (1, 1)
            assert marker[2]
            first_pass_last_frame_at = stream.state.last_frame_at

            replay = client.post(
                f"/api/cameras/{virtual_id}/playback",
                json={"action": "play"},
            )
            assert replay.status_code == 200
            for _ in range(250):
                if (
                    stream.state.ended
                    and stream.state.last_frame_at
                    > first_pass_last_frame_at
                ):
                    break
                time.sleep(0.02)
            assert stream.state.ended is True
            assert stream.state.last_frame_at > first_pass_last_frame_at
            time.sleep(0.50)

            status = client.get(
                f"/api/cameras/{virtual_id}/status"
            ).json()
            assert status["anpr_preview_only"] is True
            assert status["anpr_completed"] is True
            assert status["anpr_interrupted"] is False
            with database.connect() as con:
                replay_count = con.execute(
                    "SELECT COUNT(*) FROM plate_events WHERE camera_id=?",
                    (virtual_id,),
                ).fetchone()[0]
            assert replay_count == 1
            latest_preview = main.manager.streams[virtual_id].latest
        assert dashboard.status_code == 200
        assert f"id='anpr-{virtual_id}'" in dashboard.text
        assert "ویدئو: traffic" in dashboard.text
        assert latest_preview.startswith(b"\xff\xd8")
        assert virtual_id not in main.manager.streams
    finally:
        if virtual_id is not None:
            main.manager.remove(virtual_id)
            live_worker.stop_live_camera(virtual_id)


def test_uploaded_video_model_failure_at_eof_remains_incomplete(
    tmp_path,
    monkeypatch,
):
    import app.ai.live_worker as live_worker

    _as_role(monkeypatch, "operator")
    db_path = tmp_path / "failed-video.db"
    video_dir = tmp_path / "videos"
    source_path = tmp_path / "failure.avi"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 90),
    )
    assert writer.isOpened()
    for value in (30, 90, 150):
        writer.write(np.full((90, 160, 3), value, dtype=np.uint8))
    writer.release()

    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    with database.connect() as con:
        source_camera_id = int(con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,enabled,is_demo,lpr_enabled,lpr_confidence,"
            "frame_step,duplicate_seconds"
            ") VALUES('Gate','rtsp://gate',1,0,1,50,1,0)"
        ).lastrowid)
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda _setting_key, _default: video_dir,
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("selected YOLO inference failed at EOF")
        ),
    )

    virtual_id = None
    try:
        with TestClient(main.app) as client, source_path.open("rb") as source:
            response = client.post(
                "/cameras/video-upload",
                data={"camera_id": str(source_camera_id)},
                files={"video": ("failure.avi", source, "video/x-msvideo")},
                headers={"X-Requested-With": "XMLHttpRequest"},
            )
            assert response.status_code == 200
            virtual_id = int(response.json()["camera_id"])
            stream = main.manager.streams[virtual_id]
            for _ in range(300):
                if stream.state.ended:
                    break
                time.sleep(0.01)

            assert stream.state.ended is True
            status = client.get(
                f"/api/cameras/{virtual_id}/status"
            ).json()
            with database.connect() as con:
                marker = con.execute(
                    "SELECT video_anpr_started,video_anpr_completed,"
                    "video_anpr_completed_at FROM cameras WHERE id=?",
                    (virtual_id,),
                ).fetchone()

        assert tuple(marker) == (1, 0, "")
        assert status["anpr_preview_only"] is True
        assert status["anpr_completed"] is False
        assert status["anpr_interrupted"] is True
        assert (
            status["anpr"]["last_error"]
            == "RuntimeError: selected YOLO inference failed at EOF"
        )
        assert (
            status["anpr_marker_error"]
            == "RuntimeError: selected YOLO inference failed at EOF"
        )
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


def test_dashboard_recent_events_returns_only_after_new_commit(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "dashboard-events.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _as_role(monkeypatch, "operator")
    with database.connect() as con:
        event_id = con.execute(
            "INSERT INTO plate_events("
            "plate_text,plate_norm,confidence,camera_name"
            ") VALUES(?,?,?,?)",
            ("12-ب-345-67", "12ب34567", 0.91, "Gate"),
        ).lastrowid

    with TestClient(main.app) as client:
        fresh = client.get("/api/dashboard/recent-events?after=0")
        unchanged = client.get(
            f"/api/dashboard/recent-events?after={event_id}"
        )
        initial_updated = fresh.json()["latest_updated"]
        with database.connect() as con:
            con.execute(
                "UPDATE plate_events SET plate_text=?,updated_at=? "
                "WHERE id=?",
                ("12-ب-345-68", "2099-01-01 00:00:00", event_id),
            )
        refreshed_same_id = client.get(
            "/api/dashboard/recent-events",
            params={
                "after": event_id,
                "after_updated": initial_updated,
            },
        )

    assert fresh.status_code == 200
    assert fresh.json()["latest_id"] == event_id
    assert "Gate" in fresh.json()["rows_html"]
    assert "۱۲" in fresh.json()["rows_html"]
    assert fresh.json()["latest_updated"]
    assert unchanged.json() == {
        "latest_id": event_id,
        "rows_html": "",
    }
    assert refreshed_same_id.json()["latest_id"] == event_id
    assert refreshed_same_id.json()["latest_updated"] == (
        "2099-01-01 00:00:00"
    )
    assert "۶۸" in refreshed_same_id.json()["rows_html"]


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


def test_first_run_setup_creates_one_admin_without_a_default_password(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "first-run-setup.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()

    with TestClient(main.app) as client:
        login_before_setup = client.get("/login", follow_redirects=False)
        setup_page = client.get("/setup", follow_redirects=False)
        weak = client.post(
            "/setup",
            data={
                "display_name": "مدیر",
                "password": "too-short",
                "password_confirm": "too-short",
            },
            follow_redirects=False,
        )
        mismatch = client.post(
            "/setup",
            data={
                "display_name": "مدیر",
                "password": "unique-first-run-password",
                "password_confirm": "different-first-run-password",
            },
            follow_redirects=False,
        )
        created = client.post(
            "/setup",
            data={
                "display_name": "مدیر اصلی",
                "password": "unique-first-run-password",
                "password_confirm": "unique-first-run-password",
            },
            follow_redirects=False,
        )
        repeated = client.post(
            "/setup",
            data={
                "display_name": "مهاجم",
                "password": "another-first-run-password",
                "password_confirm": "another-first-run-password",
            },
            follow_redirects=False,
        )
        login = client.post(
            "/login",
            data={
                "username": "admin",
                "password": "unique-first-run-password",
                "next": "/dashboard",
            },
            follow_redirects=False,
        )

    assert login_before_setup.status_code == 302
    assert login_before_setup.headers["location"] == "/setup"
    assert setup_page.status_code == 200
    assert "هیچ رمز پیش‌فرضی وجود ندارد" in setup_page.text
    assert weak.headers["location"] == "/setup?error=weak"
    assert mismatch.headers["location"] == "/setup?error=mismatch"
    assert created.headers["location"] == "/login?created=1"
    assert repeated.headers["location"] == "/login"
    assert login.headers["location"] == "/dashboard"
    with database.connect() as connection:
        rows = connection.execute(
            "SELECT username,password_hash,must_change_password FROM users"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0]["username"] == "admin"
    assert rows[0]["must_change_password"] == 0
    assert main.verify_password(
        "unique-first-run-password",
        rows[0]["password_hash"],
    )


def test_legacy_admin_is_confined_until_password_is_changed(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "bootstrap-password.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _insert_admin(
        "legacy-admin-password",
        must_change_password=True,
    )

    with TestClient(main.app) as client:
        login_response = client.post(
            "/login",
            data={
                "username": "admin",
                "password": "legacy-admin-password",
                "next": "/dashboard",
            },
            follow_redirects=False,
        )
        confined_page = client.get("/dashboard", follow_redirects=False)
        confined_api = client.get(
            "/api/storage/status",
            follow_redirects=False,
        )
        settings_page = client.get(
            "/settings?password_required=1",
            follow_redirects=False,
        )
        short_change = client.post(
            "/settings/display",
            data={
                "dashboard_grid": "2",
                "dashboard_event_rows": "12",
                "live_fps": "5",
                "stream_width": "640",
                "jpeg_quality": "70",
                "new_password": "short",
            },
            follow_redirects=False,
        )
        changed = client.post(
            "/settings/display",
            data={
                "dashboard_grid": "2",
                "dashboard_event_rows": "12",
                "live_fps": "5",
                "stream_width": "640",
                "jpeg_quality": "70",
                "new_password": "new-admin-password",
            },
            follow_redirects=False,
        )
        dashboard = client.get("/dashboard", follow_redirects=False)

    assert login_response.status_code == 303
    assert login_response.headers["location"] == (
        "/settings?password_required=1"
    )
    assert confined_page.status_code == 303
    assert confined_page.headers["location"] == (
        "/settings?password_required=1"
    )
    assert confined_api.status_code == 403
    assert confined_api.json()["error"] == "password-change-required"
    assert settings_page.status_code == 200
    assert "تعویض رمز در اولین ورود اجباری است" not in settings_page.text
    assert short_change.status_code == 303
    assert "password_required=1" in short_change.headers["location"]
    assert changed.status_code == 303
    assert dashboard.status_code == 200
    with database.connect() as con:
        admin = con.execute(
            "SELECT password_hash,must_change_password FROM users "
            "WHERE username='admin'"
        ).fetchone()
    assert admin["must_change_password"] == 0
    assert main.verify_password(
        "new-admin-password",
        admin["password_hash"],
    )


def test_password_change_revokes_other_sessions_but_keeps_current_one(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "revocable-sessions.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _insert_admin("session-start-password")

    with TestClient(main.app) as current, TestClient(main.app) as stale:
        for client in (current, stale):
            response = client.post(
                "/login",
                data={
                    "username": "admin",
                    "password": "session-start-password",
                    "next": "/dashboard",
                },
                follow_redirects=False,
            )
            assert response.status_code == 303

        changed = current.post(
            "/settings/display",
            data={
                "dashboard_grid": "2",
                "dashboard_event_rows": "12",
                "live_fps": "5",
                "stream_width": "640",
                "jpeg_quality": "70",
                "new_password": "revoked-session-password",
            },
            follow_redirects=False,
        )
        current_dashboard = current.get(
            "/dashboard",
            follow_redirects=False,
        )
        stale_dashboard = stale.get(
            "/dashboard",
            follow_redirects=False,
        )

    assert changed.status_code == 303
    assert current_dashboard.status_code == 200
    assert stale_dashboard.status_code == 302
    assert stale_dashboard.headers["location"] == "/login"
    with database.connect() as con:
        generation = con.execute(
            "SELECT session_version FROM users WHERE username='admin'"
        ).fetchone()[0]
    assert generation == 1


def test_logout_revokes_the_exact_copied_session_token(tmp_path, monkeypatch):
    db_path = tmp_path / "logout-revocation.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    monkeypatch.setattr(main, "DB_PATH", db_path)
    database.init_db()
    _insert_admin("logout-start-password")

    with TestClient(main.app) as current:
        current.post(
            "/login",
            data={"username": "admin", "password": "logout-start-password"},
            follow_redirects=False,
        )
        current.post(
            "/settings/display",
            data={
                "dashboard_grid": "2",
                "dashboard_event_rows": "12",
                "live_fps": "5",
                "stream_width": "640",
                "jpeg_quality": "70",
                "new_password": "logout-revocation-password",
            },
            follow_redirects=False,
        )
        copied_token = current.cookies.get(main.COOKIE_NAME)
        assert copied_token
        logged_out = current.get("/logout", follow_redirects=False)

    with TestClient(main.app) as replay:
        replay.cookies.set(main.COOKIE_NAME, copied_token)
        rejected = replay.get("/dashboard", follow_redirects=False)

    assert logged_out.status_code == 302
    assert rejected.status_code == 302
    assert rejected.headers["location"] == "/login"
    with database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM revoked_sessions"
        ).fetchone()[0] == 1


@pytest.mark.parametrize(
    "path",
    ["/cameras/video-upload", "/ai/video-test/upload"],
)
def test_video_upload_declared_oversize_is_rejected_before_parsing(path):
    with TestClient(main.app) as client:
        response = client.post(
            path,
            content=b"not-a-multipart-body",
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "content-length": str(
                    main._max_declared_video_upload_bytes() + 1
                ),
            },
        )

    assert response.status_code == 413
    assert response.json()["error"] == "video-upload-too-large"


def test_oversize_preflight_does_not_wait_for_storage_migration(monkeypatch):
    gate = main.WriterPreferredGate()
    writer = gate.queue_exclusive()
    assert gate.try_acquire_exclusive(writer) is True
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    responses = []

    with TestClient(main.app) as client:
        thread = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/ai/video-test/upload",
                    content=b"not-a-multipart-body",
                    headers={
                        "content-type": "multipart/form-data; boundary=x",
                        "content-length": str(
                            main._max_declared_video_upload_bytes() + 1
                        ),
                    },
                )
            )
        )
        thread.start()
        thread.join(1.0)
        completed_before_release = not thread.is_alive()
        gate.release_exclusive(writer)
        thread.join(2.0)

    assert completed_before_release is True
    assert len(responses) == 1
    assert responses[0].status_code == 413


def test_chunked_oversize_does_not_hold_or_wait_for_storage_gate(monkeypatch):
    _as_role(monkeypatch, "operator")
    gate = main.WriterPreferredGate()
    writer = gate.queue_exclusive()
    assert gate.try_acquire_exclusive(writer) is True
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    monkeypatch.setattr(main, "MAX_VIDEO_UPLOAD_BYTES", 8)
    responses = []

    with TestClient(main.app) as client:
        thread = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/ai/video-test/upload",
                    content=iter((b"1234", b"56789")),
                    headers={
                        "content-type": (
                            "multipart/form-data; boundary=x"
                        ),
                        "transfer-encoding": "chunked",
                    },
                )
            )
        )
        thread.start()
        thread.join(1.0)
        completed_before_release = not thread.is_alive()
        gate.release_exclusive(writer)
        thread.join(2.0)

    assert completed_before_release is True
    assert len(responses) == 1
    assert responses[0].status_code == 413
    assert responses[0].json()["error"] == "video-upload-too-large"


def test_unauthenticated_video_upload_body_is_never_received():
    consumed = []

    def hostile_body():
        consumed.append(True)
        yield b"x" * 1024

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            content=hostile_body(),
            headers={
                "content-type": "multipart/form-data; boundary=x",
                "transfer-encoding": "chunked",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["location"] == "/login"
    assert consumed == []


def test_unauthenticated_storage_migration_body_is_never_received():
    consumed = []

    def hostile_body():
        consumed.append(True)
        yield b"storage_root=" + b"x" * 1024

    with TestClient(main.app) as client:
        response = client.post(
            "/settings/storage",
            content=hostile_body(),
            headers={
                "content-type": "application/x-www-form-urlencoded",
                "transfer-encoding": "chunked",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert response.headers["location"] == "/login"
    assert consumed == []


def test_slow_form_body_does_not_acquire_mutation_gate(monkeypatch):
    gate = main.WriterPreferredGate()
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    first_chunk = threading.Event()
    release_body = threading.Event()
    responses = []

    def slow_body():
        first_chunk.set()
        yield b"username=nobody&"
        release_body.wait(timeout=5)
        yield b"password=wrong&next=%2Fdashboard"

    with TestClient(main.app) as client:
        thread = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/login",
                    content=slow_body(),
                    headers={
                        "content-type": (
                            "application/x-www-form-urlencoded"
                        ),
                        "transfer-encoding": "chunked",
                    },
                    follow_redirects=False,
                )
            )
        )
        thread.start()
        assert first_chunk.wait(1.0)
        time.sleep(0.05)
        state_while_uploading = gate.snapshot()
        release_body.set()
        thread.join(2.0)

    assert state_while_uploading == (0, False, 0)
    assert len(responses) == 1
    assert responses[0].status_code == 303


def test_streaming_response_releases_gate_before_body_iteration(monkeypatch):
    gate = main.WriterPreferredGate()
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    body_iterated = []

    async def body():
        body_iterated.append(True)
        yield b"frame"

    async def scenario():
        request = main.Request({
            "type": "http",
            "method": "GET",
            "scheme": "http",
            "path": "/live/7",
            "raw_path": b"/live/7",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
            "state": {},
        })
        await main._storage_mutation_dependency(request)
        assert gate.snapshot() == (1, False, 0)

        async def call_next(_request):
            return main.StreamingResponse(body())

        response = await main.security_headers(request, call_next)
        assert isinstance(response, main.StreamingResponse)

    asyncio.run(scenario())

    assert body_iterated == []
    assert gate.snapshot() == (0, False, 0)
    writer = gate.queue_exclusive()
    assert gate.try_acquire_exclusive(writer) is True
    gate.release_exclusive(writer)


def test_parsed_video_upload_waits_for_exclusive_storage_gate(monkeypatch):
    _as_role(monkeypatch, "operator")
    gate = main.WriterPreferredGate()
    writer = gate.queue_exclusive()
    assert gate.try_acquire_exclusive(writer) is True
    monkeypatch.setattr(main, "_STORAGE_MUTATION_GATE", gate)
    async def reject_invalid_video(*_args, **_kwargs):
        raise RuntimeError("invalid video fixture")

    monkeypatch.setattr(
        main,
        "run_module_job_subprocess",
        reject_invalid_video,
    )
    responses = []

    with TestClient(main.app) as client:
        thread = threading.Thread(
            target=lambda: responses.append(
                client.post(
                    "/ai/video-test/upload",
                    files={"video": ("small.mp4", b"x", "video/mp4")},
                    follow_redirects=False,
                )
            )
        )
        thread.start()
        thread.join(0.25)
        blocked_before_release = thread.is_alive()
        gate.release_exclusive(writer)
        thread.join(2.0)

    assert blocked_before_release is True
    assert len(responses) == 1
    assert responses[0].status_code == 200
    assert "خطا" in responses[0].text


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


def test_engine_v3_selects_yolo11n_once_without_overriding_later_choice(
    monkeypatch,
):
    settings = {
        "anpr_detector_model": "yolov8n",
    }

    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: settings.__setitem__(key, str(value)),
    )

    assert main._migrate_anpr_v3_detector_selection() is True
    assert settings["anpr_detector_model"] == "yolo11n"
    assert settings[main._ANPR_V3_DETECTOR_MIGRATION] == "1"

    settings["anpr_detector_model"] = "yolov8n"
    assert main._migrate_anpr_v3_detector_selection() is False
    assert settings["anpr_detector_model"] == "yolov8n"


def test_v2_safety_migration_demotes_existing_primary_setting(monkeypatch):
    settings = {"anpr_engine_v2_shadow": "1"}
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: settings.__setitem__(key, str(value)),
    )

    assert main._migrate_anpr_v2_to_safe_shadow() is True
    assert settings["anpr_engine_v2_shadow"] == "0"
    assert settings[main._ANPR_V2_SAFE_SHADOW_MIGRATION] == "1"
    assert main._migrate_anpr_v2_to_safe_shadow() is False


def test_live_restore_migration_undoes_forced_yolo11_once(monkeypatch):
    settings = {
        main._ANPR_V3_DETECTOR_MIGRATION: "1",
        "anpr_detector_model": "yolo11n",
    }
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        main,
        "set_setting",
        lambda key, value: settings.__setitem__(key, str(value)),
    )

    assert main._migrate_anpr_live_detector_restore() is True
    assert settings["anpr_detector_model"] == "yolov8n"
    assert settings[main._ANPR_LIVE_RESTORE_MIGRATION] == "1"

    settings["anpr_detector_model"] = "yolo11n"
    assert main._migrate_anpr_live_detector_restore() is False
    assert settings["anpr_detector_model"] == "yolo11n"
