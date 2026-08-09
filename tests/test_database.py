import importlib
import sqlite3


def test_old_database_migrates_without_data_loss(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "old.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE users(
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password_hash TEXT,
            display_name TEXT,
            is_admin INTEGER
        );
        CREATE TABLE audit_logs(
            id INTEGER PRIMARY KEY,
            username TEXT,
            action TEXT,
            details TEXT,
            ip_address TEXT,
            created_at TEXT
        );
        CREATE TABLE settings(
            key TEXT PRIMARY KEY,
            value TEXT
        );
        CREATE TABLE cameras(
            id INTEGER PRIMARY KEY,
            name TEXT,
            rtsp_url TEXT,
            location TEXT,
            enabled INTEGER,
            is_demo INTEGER,
            sort_order INTEGER,
            created_at TEXT
        );
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY,
            plate_text TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            created_at TEXT
        );
        CREATE TABLE plate_watchlist(
            id INTEGER PRIMARY KEY,
            plate_text TEXT,
            plate_norm TEXT UNIQUE,
            status TEXT,
            owner_name TEXT,
            phone TEXT,
            vehicle_model TEXT,
            vehicle_color TEXT,
            notes TEXT,
            created_at TEXT
        );
        INSERT INTO cameras(
            id,name,rtsp_url,location,enabled,is_demo,sort_order
        ) VALUES(7,'North Gate','rtsp://north','ورودی شمالی',1,0,0);
        INSERT INTO plate_events(
            id,plate_text,confidence,camera_id
        ) VALUES(1,'۱۲-ب-۳۴۵-۶۷',0.8,7);
        INSERT INTO users(
            id,username,password_hash,display_name,is_admin
        ) VALUES(1,'existing','x','Existing',1);
        """)

    import app.config
    monkeypatch.setattr(app.config, "DB_PATH", db_path)
    import app.database
    importlib.reload(app.database)
    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        con.row_factory = sqlite3.Row
        columns = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(plate_events)"
            )
        }
        assert {
            "plate_norm",
            "direction",
            "quality_score",
            "consensus_votes",
            "source",
            "ocr_engine",
            "ocr_alternative",
            "ocr_disagreement",
            "raw_guess_text",
            "raw_guess_norm",
            "raw_guess_confidence",
            "raw_guess_engine",
            "raw_guess_reason",
            "model_revision",
            "experimental",
            "confirmation_source",
            "operator_reviewed",
            "city",
            "plate_region",
            "media_status",
            "media_error",
            "updated_at",
        } <= columns
        feedback_columns = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(anpr_feedback)"
            )
        }
        assert {
            "event_id",
            "observed_norm",
            "corrected_norm",
            "submitted_by",
            "status",
            "sample_path",
            "sample_sha256",
            "dataset_split",
            "training_status",
            "trained_run_id",
            "observed_engine",
            "observed_confidence",
            "observed_model_revision",
            "character_distance",
            "exact_match",
        } <= feedback_columns
        assert con.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' AND name='anpr_training_runs'"
        ).fetchone()[0] == 1
        training_columns = {
            row[1]
            for row in con.execute(
                "PRAGMA table_info(anpr_training_runs)"
            )
        }
        assert {
            "baseline_mean_character_error",
            "candidate_mean_character_error",
            "baseline_sha256",
            "candidate_checkpoint_path",
            "candidate_checkpoint_sha256",
            "promotion_report",
            "dataset_manifest_path",
            "dataset_manifest_sha256",
        } <= training_columns
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=1"
        ).fetchone()
        assert row["plate_text"] == "۱۲-ب-۳۴۵-۶۷"
        assert row["plate_norm"] == "12ب34567"
        assert row["plate_region"] == "67"
        assert row["city"] == ""
        assert row["media_status"] == "missing"
        assert row["updated_at"] == row["created_at"]
        camera_columns = {
            item[1]
            for item in con.execute("PRAGMA table_info(cameras)")
        }
        assert {
            "city",
            "video_anpr_started",
            "video_anpr_completed",
            "video_anpr_completed_at",
        } <= camera_columns
        assert tuple(con.execute(
            "SELECT city,video_anpr_started,video_anpr_completed,"
            "video_anpr_completed_at FROM cameras WHERE id=7"
        ).fetchone()) == ("", 0, 0, "")
        assert con.execute(
            "SELECT COUNT(*) FROM users "
            "WHERE username='existing'"
        ).fetchone()[0] == 1
        assert con.execute(
            "SELECT value FROM settings "
            "WHERE key='anpr_detector_model'"
        ).fetchone()[0] == "yolo11n"
        indexes = {
            row[1]
            for row in con.execute(
                "PRAGMA index_list(plate_events)"
            )
        }
        assert "idx_plate_events_plate_norm" in indexes
        assert "idx_plate_events_city_created" in indexes
        assert "idx_plate_events_region_created" in indexes
        assert "idx_plate_events_updated_at" in indexes


def test_new_database_has_no_automatic_demo_camera(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "fresh.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)

    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM cameras"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT value FROM settings "
            "WHERE key='migration_remove_builtin_demo_camera_v1'"
        ).fetchone()[0] == "1"
        assert con.execute(
            "SELECT value FROM settings "
            "WHERE key='migration_video_anpr_markers_rc29_v1'"
        ).fetchone()[0] == "1"


def test_legacy_video_marker_backfill_is_one_time_and_preserves_interrupts(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "legacy-video.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE settings(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE cameras(
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_demo INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        INSERT INTO cameras(id,name,rtsp_url)
        VALUES(7,'Legacy upload','video:///archive/already-seen.avi');
        INSERT INTO cameras(id,name,rtsp_url)
        VALUES(8,'Gate','rtsp://gate');
        """)

    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        legacy = con.execute(
            "SELECT video_anpr_started,video_anpr_completed,"
            "video_anpr_completed_at FROM cameras WHERE id=7"
        ).fetchone()
        live = con.execute(
            "SELECT video_anpr_started,video_anpr_completed,"
            "video_anpr_completed_at FROM cameras WHERE id=8"
        ).fetchone()
        marker = con.execute(
            "SELECT value FROM settings "
            "WHERE key='migration_video_anpr_markers_rc29_v1'"
        ).fetchone()
        con.execute(
            "INSERT INTO cameras("
            "id,name,rtsp_url,video_anpr_started,video_anpr_completed"
            ") VALUES(9,'Interrupted RC29 upload',"
            "'video:///archive/interrupted.avi',1,0)"
        )

    assert legacy[0:2] == (1, 1)
    assert legacy[2]
    assert live == (0, 0, "")
    assert marker == ("1",)

    # A later startup must not turn an interrupted RC29 pass into a completed
    # one merely because its durable started marker is already present.
    app.database.init_db()
    with sqlite3.connect(db_path) as con:
        interrupted = con.execute(
            "SELECT video_anpr_started,video_anpr_completed,"
            "video_anpr_completed_at FROM cameras WHERE id=9"
        ).fetchone()
        assert interrupted == (1, 0, "")


def test_media_migration_checks_real_files_before_marking_complete(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "media.db"
    vehicle = tmp_path / "vehicle.jpg"
    missing_plate = tmp_path / "missing-plate.jpg"
    vehicle.write_bytes(b"vehicle")
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        con.execute(
            "INSERT INTO plate_events("
            "plate_text,image_path,plate_image_path,media_status"
            ") VALUES(?,?,?,'pending')",
            ("ناخوانا", str(vehicle), str(missing_plate)),
        )

    app.database.init_db()
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            "SELECT media_status,media_error FROM plate_events"
        ).fetchone()

    assert row[0] == "partial"
    assert "پیدا نشد" in row[1]


def test_builtin_demo_migration_preserves_user_cameras(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "existing.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        con.execute(
            "DELETE FROM settings WHERE "
            "key='migration_remove_builtin_demo_camera_v1'"
        )
        con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,enabled,is_demo,sort_order"
            ") VALUES(?,?,?,?,?,?)",
            (
                "دوربین آزمایشی",
                "demo://camera-1",
                "نمایش نمونه",
                1,
                1,
                1,
            ),
        )
        con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,enabled,is_demo,sort_order"
            ") VALUES(?,?,?,?,?,?)",
            (
                "نمونه کاربر",
                "demo://custom",
                "پارکینگ",
                1,
                1,
                2,
            ),
        )

    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        rows = con.execute(
            "SELECT name,rtsp_url FROM cameras ORDER BY id"
        ).fetchall()
        assert rows == [("نمونه کاربر", "demo://custom")]

        # The marker makes the migration one-time and avoids deleting a camera
        # that a user might deliberately create later with matching fields.
        con.execute(
            "INSERT INTO cameras("
            "name,rtsp_url,location,enabled,is_demo,sort_order"
            ") VALUES(?,?,?,?,?,?)",
            (
                "دوربین آزمایشی",
                "demo://camera-1",
                "نمایش نمونه",
                1,
                1,
                3,
            ),
        )

    app.database.init_db()

    with sqlite3.connect(db_path) as con:
        assert con.execute(
            "SELECT COUNT(*) FROM cameras"
        ).fetchone()[0] == 2


def test_connect_enables_concurrency_safety_pragmas(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "pragmas.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with app.database.connect() as con:
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert con.execute("PRAGMA busy_timeout").fetchone()[0] == 20000
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_database_backup_is_atomic_and_valid(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "live.db"
    backup_path = tmp_path / "backups" / "snapshot.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?)",
            ("backup_marker", "preserved"),
        )

    result = app.database.backup_database(backup_path)

    assert result == backup_path.resolve()
    assert backup_path.is_file()
    assert list(backup_path.parent.glob(".*.tmp")) == []
    with sqlite3.connect(backup_path) as snapshot:
        assert snapshot.execute("PRAGMA quick_check").fetchone()[0] == "ok"
        assert snapshot.execute(
            "SELECT value FROM settings WHERE key='backup_marker'"
        ).fetchone()[0] == "preserved"


def test_settings_can_be_updated_in_migrated_database(
    tmp_path,
    monkeypatch,
):
    import app.database

    source = tmp_path / "source.db"
    migrated = tmp_path / "new-root" / "bcvision.db"
    monkeypatch.setattr(app.database, "DB_PATH", source)
    app.database.init_db()
    app.database.backup_database(migrated)

    app.database.set_settings_for_database(
        migrated,
        {
            "storage_root": migrated.parent,
            "video_path": migrated.parent / "videos",
        },
    )

    with sqlite3.connect(source) as old_db:
        assert old_db.execute(
            "SELECT value FROM settings WHERE key='storage_root'"
        ).fetchone()[0] != str(migrated.parent)
    with sqlite3.connect(migrated) as new_db:
        assert new_db.execute(
            "SELECT value FROM settings WHERE key='storage_root'"
        ).fetchone()[0] == str(migrated.parent)
        assert new_db.execute(
            "SELECT value FROM settings WHERE key='video_path'"
        ).fetchone()[0] == str(migrated.parent / "videos")
