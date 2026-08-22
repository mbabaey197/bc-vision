import importlib
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.security import hash_password


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
        assert con.execute(
            "SELECT COUNT(*) FROM sqlite_master "
            "WHERE type='table' "
            "AND name='anpr_persistence_receipts'"
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
        assert {"must_change_password", "session_version"} <= {
            row[1] for row in con.execute("PRAGMA table_info(users)")
        }
        assert {
            "token_hash",
            "expires_at",
            "revoked_at",
        } <= {
            row[1]
            for row in con.execute("PRAGMA table_info(revoked_sessions)")
        }
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
        receipt_indexes = {
            row[1]
            for row in con.execute(
                "PRAGMA index_list(anpr_persistence_receipts)"
            )
        }
        assert "idx_anpr_receipts_event" in receipt_indexes


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
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        assert con.execute(
            "SELECT value FROM settings WHERE key="
            "'migration_legacy_admin_password_change_rc30_v1'"
        ).fetchone()[0] == "1"


def test_legacy_admin_is_confined_once_but_new_password_is_not_reflagged(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "legacy-password.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        con.execute(
            "DELETE FROM settings WHERE key="
            "'migration_legacy_admin_password_change_rc30_v1'"
        )
        con.execute(
            "INSERT INTO users(username,password_hash,display_name,is_admin,"
            "role,is_active,must_change_password,session_version) "
            "VALUES(?,?,?,1,'admin',1,0,0)",
            ("admin", hash_password("legacy-admin-password"), "مدیر"),
        )

    app.database.init_db()
    with app.database.connect() as con:
        assert con.execute(
            "SELECT must_change_password FROM users WHERE username='admin'"
        ).fetchone()[0] == 1
        con.execute(
            "UPDATE users SET password_hash=?,must_change_password=0 "
            "WHERE username='admin'",
            (hash_password("a-unique-strong-password"),),
        )

    app.database.init_db()
    with app.database.connect() as con:
        assert con.execute(
            "SELECT must_change_password FROM users WHERE username='admin'"
        ).fetchone()[0] == 0


def test_duplicate_current_feedback_is_migrated_to_one_truth_row(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "duplicate-feedback.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        con.execute("DROP INDEX idx_anpr_feedback_current_event")
        event_id = con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("12ب34567", "12ب34567"),
        ).lastrowid
        con.executemany(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,corrected_text,corrected_norm,status"
            ") VALUES(?,?,?,?,'confirmed')",
            [
                (event_id, "12ب34567", "12ب34567", "12ب34567"),
                (event_id, "12ب34567", "12ب34568", "12ب34568"),
            ],
        )

    app.database.init_db()

    with app.database.connect() as con:
        rows = con.execute(
            "SELECT status,corrected_norm FROM anpr_feedback "
            "WHERE event_id=? ORDER BY id",
            (event_id,),
        ).fetchall()
        indexes = {
            row[1]
            for row in con.execute("PRAGMA index_list(anpr_feedback)")
        }
        with pytest.raises(sqlite3.IntegrityError):
            con.execute(
                "INSERT INTO anpr_feedback("
                "event_id,observed_text,corrected_text,corrected_norm,status"
                ") VALUES(?,?,?,?,'confirmed')",
                (event_id, "12ب34567", "12ب34569", "12ب34569"),
            )

    assert [tuple(row) for row in rows] == [
        ("superseded", "12ب34567"),
        ("confirmed", "12ب34568"),
    ]
    assert "idx_anpr_feedback_current_event" in indexes


def test_legacy_receipt_foreign_key_migrates_to_retention_tombstone(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "legacy-receipt.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
        event_id = int(con.execute(
            "INSERT INTO plate_events(camera_id,plate_norm) VALUES(?,?)",
            (camera_id, "31ط55674"),
        ).lastrowid)
        con.execute("DROP TABLE anpr_persistence_receipts")
        con.execute("""
            CREATE TABLE anpr_persistence_receipts(
                persistence_key TEXT PRIMARY KEY,
                event_id INTEGER NOT NULL,
                committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(event_id) REFERENCES plate_events(id)
                    ON DELETE CASCADE
            )
        """)
        con.execute(
            "INSERT INTO anpr_persistence_receipts(persistence_key,event_id) "
            "VALUES(?,?)",
            ("legacy-token", event_id),
        )

    app.database.init_db()

    with app.database.connect() as con:
        assert con.execute(
            "PRAGMA foreign_key_list(anpr_persistence_receipts)"
        ).fetchall() == []
        assert tuple(con.execute(
            "SELECT persistence_key,event_id "
            "FROM anpr_persistence_receipts"
        ).fetchone()) == ("legacy-token", event_id)
        con.execute("DELETE FROM plate_events WHERE id=?", (event_id,))
        assert con.execute(
            "SELECT event_id FROM anpr_persistence_receipts "
            "WHERE persistence_key='legacy-token'"
        ).fetchone()[0] == event_id


def test_legacy_feedback_cascade_migrates_to_retained_truth(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "legacy-feedback.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        event_id = int(con.execute(
            "INSERT INTO plate_events(plate_text,plate_norm) VALUES(?,?)",
            ("31-ط-556-74", "31ط55674"),
        ).lastrowid)
        feedback_id = int(con.execute(
            "INSERT INTO anpr_feedback("
            "event_id,observed_text,observed_norm,corrected_text,"
            "corrected_norm,status,training_status,sample_path,"
            "sample_sha256"
            ") VALUES(?,?,?,?,?,'confirmed','ready',?,?)",
            (
                event_id,
                "31-ط-556-74",
                "31ط55674",
                "31-ط-556-74",
                "31ط55674",
                "/retained/sample.png",
                "A" * 64,
            ),
        ).lastrowid)

    # Recreate the exact pre-migration constraint while retaining every
    # current column, then exercise the public init migration.
    with sqlite3.connect(db_path) as con:
        schema = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='anpr_feedback'"
        ).fetchone()[0]
        legacy_schema = schema.replace(
            "event_id INTEGER,",
            "event_id INTEGER NOT NULL,",
        ).replace("ON DELETE SET NULL", "ON DELETE CASCADE")
        columns = [
            row[1]
            for row in con.execute(
                "PRAGMA table_info(anpr_feedback)"
            ).fetchall()
        ]
        column_sql = ",".join(f'"{column}"' for column in columns)
        con.execute(
            "ALTER TABLE anpr_feedback "
            "RENAME TO anpr_feedback_retained_source"
        )
        con.execute(legacy_schema)
        con.execute(
            f"INSERT INTO anpr_feedback({column_sql}) "
            f"SELECT {column_sql} FROM anpr_feedback_retained_source"
        )
        con.execute("DROP TABLE anpr_feedback_retained_source")

    app.database.init_db()

    with app.database.connect() as con:
        event_column = next(
            row
            for row in con.execute(
                "PRAGMA table_info(anpr_feedback)"
            ).fetchall()
            if row[1] == "event_id"
        )
        feedback_fk = con.execute(
            "PRAGMA foreign_key_list(anpr_feedback)"
        ).fetchall()
        assert int(event_column[3]) == 0
        assert any(str(row[6]).upper() == "SET NULL" for row in feedback_fk)
        con.execute("DELETE FROM plate_events WHERE id=?", (event_id,))
        retained = con.execute(
            "SELECT event_id,training_status,sample_path,sample_sha256 "
            "FROM anpr_feedback WHERE id=?",
            (feedback_id,),
        ).fetchone()
    assert tuple(retained) == (
        None,
        "ready",
        "/retained/sample.png",
        "A" * 64,
    )


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


def test_database_backup_never_overwrites_existing_target(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "live.db"
    backup_path = tmp_path / "backups" / "snapshot.db"
    backup_path.parent.mkdir()
    backup_path.write_bytes(b"foreign-backup")
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with pytest.raises(FileExistsError):
        app.database.backup_database(backup_path)

    assert backup_path.read_bytes() == b"foreign-backup"
    assert list(backup_path.parent.glob(".*.tmp")) == []


def test_database_backup_does_not_follow_destination_symlink(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "live.db"
    backup_path = tmp_path / "backups" / "snapshot.db"
    backup_path.parent.mkdir()
    external = tmp_path / "external.db"
    external.write_bytes(b"external")
    try:
        backup_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()

    with pytest.raises(FileExistsError):
        app.database.backup_database(backup_path)

    assert backup_path.is_symlink()
    assert external.read_bytes() == b"external"


def test_database_backup_preserves_foreign_temp_collision(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "live.db"
    backup_path = tmp_path / "backups" / "snapshot.db"
    backup_path.parent.mkdir()
    collision = backup_path.parent / ".snapshot.db.fixed.tmp"
    collision.write_bytes(b"foreign-temp")
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    monkeypatch.setattr(
        app.database,
        "uuid4",
        lambda: SimpleNamespace(hex="fixed"),
    )
    app.database.init_db()

    with pytest.raises(FileExistsError):
        app.database.backup_database(backup_path)

    assert collision.read_bytes() == b"foreign-temp"
    assert not backup_path.exists()


def test_database_backup_falls_back_when_hardlinks_are_unavailable(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "live.db"
    backup_path = tmp_path / "backups" / "snapshot.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    monkeypatch.setattr(
        app.database.os,
        "link",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("hardlinks unsupported")
        ),
    )
    app.database.init_db()

    result = app.database.backup_database(backup_path)

    assert result == backup_path.resolve()
    assert backup_path.stat().st_nlink == 1
    with sqlite3.connect(backup_path) as snapshot:
        assert snapshot.execute("PRAGMA quick_check").fetchone()[0] == "ok"


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
    assert not Path(str(migrated) + "-wal").exists()
    assert not Path(str(migrated) + "-shm").exists()


def test_settings_batch_rolls_back_if_update_is_interrupted(tmp_path):
    import app.database

    database = tmp_path / "atomic-settings.db"
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE settings(key TEXT PRIMARY KEY,value TEXT)"
        )
        connection.executemany(
            "INSERT INTO settings(key,value) VALUES(?,?)",
            (("storage_root", "/old"), ("video_path", "/old/videos")),
        )

    class InterruptedValues(dict):
        def items(self):
            yield "storage_root", "/new"
            raise RuntimeError("simulated power loss")

    with pytest.raises(RuntimeError, match="simulated power loss"):
        app.database.set_settings_for_database(
            database,
            InterruptedValues(),
        )

    with sqlite3.connect(database) as connection:
        assert dict(connection.execute("SELECT key,value FROM settings")) == {
            "storage_root": "/old",
            "video_path": "/old/videos",
        }
