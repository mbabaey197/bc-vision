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
        INSERT INTO plate_events(
            id,plate_text,confidence
        ) VALUES(1,'۱۲-ب-۳۴۵-۶۷',0.8);
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
        } <= columns
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=1"
        ).fetchone()
        assert row["plate_text"] == "۱۲-ب-۳۴۵-۶۷"
        assert row["plate_norm"] == "12ب34567"
        assert con.execute(
            "SELECT COUNT(*) FROM users "
            "WHERE username='existing'"
        ).fetchone()[0] == 1
        indexes = {
            row[1]
            for row in con.execute(
                "PRAGMA index_list(plate_events)"
            )
        }
        assert "idx_plate_events_plate_norm" in indexes


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
