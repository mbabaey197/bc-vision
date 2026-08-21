import os
import time
from pathlib import Path
from types import SimpleNamespace

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
                "anpr_detector_model": "yolov8n",
            },
            follow_redirects=False,
        )

    assert response.status_code == 302
    assert ("ai_accelerator", "cpu") in writes
    assert ("ai_confidence", 80) in writes
    assert ("anpr_detector_model", "yolov8n") in writes
    assert switches == ["yolov8n"]
    assert audits == [(
        "anpr_detector_switch",
        "yolov8n; execution=exclusive-baseline",
    )]


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

    def fake_process(video_path, plate_dir, snapshot_dir, **kwargs):
        assert Path(video_path).is_file()
        assert kwargs["frame_step"] == 1
        assert kwargs["include_candidate_shadow"] is False
        Path(plate_dir).mkdir(parents=True)
        Path(snapshot_dir).mkdir(parents=True)
        crop = Path(plate_dir) / "plate-1.jpg"
        crop.write_bytes(b"jpeg")
        vehicle = Path(snapshot_dir) / "vehicle-1.jpg"
        vehicle.write_bytes(b"jpeg")
        return (
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
                "plate_path": str(crop),
                "image_path": str(vehicle),
                "media_status": "complete",
                "confidence": 0.91,
                "video_second": 0.75,
                "ocr_engine": "hezar-crnn-fa-v2-onnx",
                "method": "yolov8n-plate-onnx",
                "engine_lane": "baseline",
                "valid": True,
                "needs_review": False,
            }],
        )

    monkeypatch.setattr(
        "app.ai.video_test.process_video",
        fake_process,
    )

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
    video_dir = tmp_path / "videos"
    monkeypatch.setattr(
        main,
        "_configured_storage_child",
        lambda _setting_key, _default: video_dir,
    )
    monkeypatch.setattr(
        "app.ai.video_test.process_video",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("YOLOv8n inference failed: invalid tensor")
        ),
    )

    with TestClient(main.app) as client:
        response = client.post(
            "/ai/video-test/upload",
            files={"video": ("failure.mp4", b"fixture", "video/mp4")},
        )

    assert response.status_code == 200
    assert "خطا: YOLOv8n inference failed: invalid tensor" in response.text
    assert "هیچ پلاکی در این ویدئو تشخیص داده نشد" not in response.text


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
    assert captured == [feedback["id"]]


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
        lambda camera_id: removed.append(camera_id),
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
                if event and event["plate_norm"] == "12ب34567" and stream:
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
        assert dashboard.status_code == 200
        assert f"id='anpr-{virtual_id}'" in dashboard.text
        assert "ویدئو: traffic" in dashboard.text
        assert main.manager.streams[virtual_id].viewer_count == 0
        assert main.manager.streams[virtual_id].latest is None
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
