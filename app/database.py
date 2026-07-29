import sqlite3
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from app.config import DB_PATH
from app.security import hash_password


def connect():
    con = sqlite3.connect(
        DB_PATH,
        timeout=20,
        check_same_thread=False,
    )
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    con.execute("PRAGMA busy_timeout=20000")
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def backup_database(destination):
    """Create an atomic, integrity-checked snapshot of the live database."""
    target = Path(destination).expanduser().resolve()
    source_path = Path(DB_PATH).expanduser().resolve()
    if target == source_path:
        raise ValueError("Backup destination must differ from the live database.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(
        f".{target.name}.{uuid4().hex}.tmp"
    )
    try:
        with closing(connect()) as source, closing(
            sqlite3.connect(temporary)
        ) as snapshot:
            source.backup(snapshot)
            result = snapshot.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise sqlite3.DatabaseError(
                    "SQLite backup integrity check failed."
                )
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def set_settings_for_database(database_path, values):
    """Update settings in a selected database without changing global state."""
    path = Path(database_path).expanduser().resolve()
    with sqlite3.connect(path, timeout=20) as con:
        con.executemany(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ((key, str(value)) for key, value in values.items()),
        )


def _add_missing_columns(con, table, migrations):
    columns = {
        row[1]
        for row in con.execute(
            f"PRAGMA table_info({table})"
        ).fetchall()
    }
    for name, sql_type in migrations.items():
        if name not in columns:
            con.execute(
                f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}"
            )


def _backfill_plate_norm(con):
    try:
        from app.ai.plate_rules import normalize_plate
    except Exception:
        return
    rows = con.execute(
        "SELECT id,plate_text FROM plate_events "
        "WHERE plate_text IS NOT NULL AND plate_text<>'' "
        "AND (plate_norm IS NULL OR plate_norm='')"
    ).fetchall()
    for row in rows:
        normalized = normalize_plate(row["plate_text"])
        if normalized:
            con.execute(
                "UPDATE plate_events SET plate_norm=? WHERE id=?",
                (normalized, row["id"]),
            )


def init_db():
    with connect() as con:
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA wal_autocheckpoint=1000")
        con.executescript("""
        CREATE TABLE IF NOT EXISTS users(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            display_name TEXT NOT NULL,
            is_admin INTEGER NOT NULL DEFAULT 1,
            role TEXT NOT NULL DEFAULT 'admin',
            is_active INTEGER NOT NULL DEFAULT 1,
            failed_attempts INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT,
            last_login TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS audit_logs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT NOT NULL,
            details TEXT DEFAULT '',
            ip_address TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cameras(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_demo INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT DEFAULT '',
            confidence REAL DEFAULT 0,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS plate_watchlist(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT NOT NULL,
            plate_norm TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL DEFAULT 'allowed',
            owner_name TEXT DEFAULT '',
            phone TEXT DEFAULT '',
            vehicle_model TEXT DEFAULT '',
            vehicle_color TEXT DEFAULT '',
            notes TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS anpr_feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            observed_text TEXT NOT NULL,
            observed_norm TEXT NOT NULL DEFAULT '',
            observed_engine TEXT NOT NULL DEFAULT '',
            observed_confidence REAL NOT NULL DEFAULT 0,
            observed_model_revision TEXT NOT NULL DEFAULT '',
            corrected_text TEXT NOT NULL,
            corrected_norm TEXT NOT NULL,
            character_distance INTEGER NOT NULL DEFAULT 0,
            exact_match INTEGER NOT NULL DEFAULT 0,
            plate_image_path TEXT DEFAULT '',
            image_path TEXT DEFAULT '',
            submitted_by TEXT DEFAULT '',
            status TEXT NOT NULL DEFAULT 'confirmed',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(event_id) REFERENCES plate_events(id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS anpr_training_runs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'queued',
            device TEXT NOT NULL DEFAULT 'auto',
            epochs INTEGER NOT NULL DEFAULT 12,
            train_samples INTEGER NOT NULL DEFAULT 0,
            validation_samples INTEGER NOT NULL DEFAULT 0,
            baseline_accuracy REAL DEFAULT 0,
            candidate_accuracy REAL DEFAULT 0,
            candidate_path TEXT DEFAULT '',
            candidate_sha256 TEXT DEFAULT '',
            message TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
            applied_at TEXT,
            applied_by TEXT DEFAULT ''
        );
        """)

        _add_missing_columns(con, "users", {
            "role": "TEXT NOT NULL DEFAULT 'admin'",
            "is_active": "INTEGER NOT NULL DEFAULT 1",
            "failed_attempts": "INTEGER NOT NULL DEFAULT 0",
            "locked_until": "TEXT",
            "last_login": "TEXT",
            "created_at": "TEXT",
        })
        con.execute(
            "UPDATE users SET role='admin' "
            "WHERE is_admin=1 AND (role IS NULL OR role='')"
        )
        con.execute(
            "UPDATE users SET created_at=CURRENT_TIMESTAMP "
            "WHERE created_at IS NULL OR created_at=''"
        )

        _add_missing_columns(con, "plate_events", {
            "plate_norm": "TEXT DEFAULT ''",
            "plate_image_path": "TEXT",
            "video_path": "TEXT",
            "video_second": "REAL DEFAULT 0",
            "detector_method": "TEXT",
            "ocr_confidence": "REAL DEFAULT 0",
            "ocr_engine": "TEXT DEFAULT ''",
            "ocr_alternative": "TEXT DEFAULT ''",
            "ocr_disagreement": "INTEGER NOT NULL DEFAULT 0",
            "vehicle_type": "TEXT DEFAULT 'نامشخص'",
            "vehicle_color": "TEXT DEFAULT 'نامشخص'",
            "vehicle_brand": "TEXT DEFAULT 'نامشخص'",
            "vehicle_confidence": "REAL DEFAULT 0",
            "direction": "TEXT DEFAULT 'stationary'",
            "quality_score": "REAL DEFAULT 0",
            "consensus_votes": "INTEGER DEFAULT 1",
            "source": "TEXT DEFAULT 'video'",
            "processing_ms": "REAL DEFAULT 0",
            "review_status": "TEXT NOT NULL DEFAULT 'confirmed-ai'",
            "confirmation_source": "TEXT NOT NULL DEFAULT 'ai-strict'",
            "operator_reviewed": "INTEGER NOT NULL DEFAULT 0",
            "raw_guess_text": "TEXT NOT NULL DEFAULT ''",
            "raw_guess_norm": "TEXT NOT NULL DEFAULT ''",
            "raw_guess_confidence": "REAL NOT NULL DEFAULT 0",
            "raw_guess_engine": "TEXT NOT NULL DEFAULT ''",
            "raw_guess_reason": "TEXT NOT NULL DEFAULT ''",
            "model_revision": "TEXT NOT NULL DEFAULT ''",
            "experimental": "INTEGER NOT NULL DEFAULT 0",
        })
        _add_missing_columns(con, "anpr_feedback", {
            "sample_path": "TEXT DEFAULT ''",
            "sample_sha256": "TEXT DEFAULT ''",
            "dataset_split": "TEXT DEFAULT ''",
            "training_status": "TEXT DEFAULT 'pending'",
            "trained_run_id": "INTEGER",
            "observed_engine": "TEXT NOT NULL DEFAULT ''",
            "observed_confidence": "REAL NOT NULL DEFAULT 0",
            "observed_model_revision": "TEXT NOT NULL DEFAULT ''",
            "character_distance": "INTEGER NOT NULL DEFAULT 0",
            "exact_match": "INTEGER NOT NULL DEFAULT 0",
        })
        _backfill_plate_norm(con)
        con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_plate_events_created_at
            ON plate_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_plate_events_plate_norm
            ON plate_events(plate_norm);
        CREATE INDEX IF NOT EXISTS idx_plate_events_camera_created
            ON plate_events(camera_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_observed
            ON anpr_feedback(observed_norm,status);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_event
            ON anpr_feedback(event_id);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_training
            ON anpr_feedback(training_status,dataset_split);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_model_revision
            ON anpr_feedback(observed_model_revision,created_at);
        CREATE INDEX IF NOT EXISTS idx_anpr_training_runs_created
            ON anpr_training_runs(created_at);
        """)

        _add_missing_columns(con, "cameras", {
            "lpr_enabled": "INTEGER NOT NULL DEFAULT 1",
            "lpr_confidence": "INTEGER NOT NULL DEFAULT 60",
            "frame_step": "INTEGER NOT NULL DEFAULT 5",
            "duplicate_seconds": "REAL NOT NULL DEFAULT 30",
            "roi_x": "INTEGER NOT NULL DEFAULT 0",
            "roi_y": "INTEGER NOT NULL DEFAULT 0",
            "roi_w": "INTEGER NOT NULL DEFAULT 100",
            "roi_h": "INTEGER NOT NULL DEFAULT 100",
            "line_y": "INTEGER NOT NULL DEFAULT 50",
        })

        if con.execute(
            "SELECT 1 FROM users WHERE username=?",
            ("admin",),
        ).fetchone() is None:
            con.execute(
                "INSERT INTO users(" 
                "username,password_hash,display_name,is_admin"
                ") VALUES(?,?,?,1)",
                (
                    "admin",
                    hash_password("123456"),
                    "مدیر سیستم",
                ),
            )

        defaults = {
            "company_name": "گیلاس آبی البرز",
            "dashboard_grid": "2",
            "live_fps": "5",
            "stream_width": "640",
            "jpeg_quality": "70",
            "storage_root": str(DB_PATH.parent),
            "snapshot_path": str(DB_PATH.parent / "snapshots"),
            "plate_path": str(DB_PATH.parent / "plates"),
            "video_path": str(DB_PATH.parent / "videos"),
            "backup_path": str(DB_PATH.parent / "backups"),
            "save_snapshots": "1",
            "save_plate_images": "1",
            "save_videos": "0",
            "max_storage_gb": "0",
            "storage_full_action": "delete_oldest",
            "retention_snapshots_days": "90",
            "retention_plates_days": "90",
            "retention_videos_days": "7",
            "retention_events_days": "0",
            "anpr_auto_confirm_guesses": "1",
        }
        for key, value in defaults.items():
            con.execute(
                "INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)",
                (key, value),
            )

        migration_key = "migration_remove_builtin_demo_camera_v1"
        if con.execute(
            "SELECT 1 FROM settings WHERE key=?",
            (migration_key,),
        ).fetchone() is None:
            con.execute(
                "DELETE FROM cameras WHERE "
                "name=? AND rtsp_url=? AND location=? AND is_demo=1",
                (
                    "دوربین آزمایشی",
                    "demo://camera-1",
                    "نمایش نمونه",
                ),
            )
            con.execute(
                "INSERT INTO settings(key,value) VALUES(?,?)",
                (migration_key, "1"),
            )


def get_setting(key, default=""):
    with connect() as con:
        row = con.execute(
            "SELECT value FROM settings WHERE key=?",
            (key,),
        ).fetchone()
        return row["value"] if row else default


def set_setting(key, value):
    with connect() as con:
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
