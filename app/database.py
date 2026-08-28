import os
import sqlite3
import stat
import time
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from app.config import DB_PATH
from app.file_identity import descriptor_file_identity, path_file_identity


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
    """Create a no-clobber, integrity-checked snapshot of the live database."""

    requested = Path(destination).expanduser()
    target = requested.parent.resolve() / requested.name
    source_path = Path(DB_PATH).expanduser().resolve()
    if target == source_path:
        raise ValueError("Backup destination must differ from the live database.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    temporary_identity = None
    published_identity = None
    try:
        for _attempt in range(8):
            candidate = target.with_name(
                f".{target.name}.{uuid4().hex}.tmp"
            )
            try:
                descriptor = os.open(
                    candidate,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_BINARY", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                    0o600,
                )
            except FileExistsError:
                continue
            temporary = candidate
            try:
                opened = os.fstat(descriptor)
                temporary_identity = descriptor_file_identity(
                    descriptor,
                    details=opened,
                )
            finally:
                os.close(descriptor)
            break
        if temporary is None or temporary_identity is None:
            raise FileExistsError("could not reserve database backup temporary")
        with closing(connect()) as source, closing(
            sqlite3.connect(temporary)
        ) as snapshot:
            snapshot.execute("PRAGMA synchronous=FULL")
            source.backup(snapshot)
            result = snapshot.execute("PRAGMA quick_check").fetchone()
            if not result or result[0] != "ok":
                raise sqlite3.DatabaseError(
                    "SQLite backup integrity check failed."
                )
        with temporary.open("r+b") as snapshot_file:
            os.fsync(snapshot_file.fileno())
        completed = temporary.lstat()
        if (
            not stat.S_ISREG(completed.st_mode)
            or int(completed.st_nlink) != 1
            or path_file_identity(temporary, details=completed)
            != temporary_identity
        ):
            raise OSError("database backup temporary changed before publish")
        hardlinked = False
        try:
            os.link(temporary, target, follow_symlinks=False)
            hardlinked = True
        except FileExistsError:
            raise
        except OSError as link_error:
            try:
                current = target.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (
                stat.S_ISREG(current.st_mode)
                and int(current.st_nlink) == 2
                and path_file_identity(target, details=current)
                == temporary_identity
            ):
                hardlinked = True
            elif current is not None:
                raise FileExistsError(
                    f"database backup target already exists: {target}"
                ) from link_error
            else:
                target_descriptor = None
                try:
                    target_descriptor = os.open(
                        target,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        | getattr(os, "O_BINARY", 0)
                        | getattr(os, "O_NOFOLLOW", 0),
                        0o600,
                    )
                    created = os.fstat(target_descriptor)
                    published_identity = descriptor_file_identity(
                        target_descriptor,
                        details=created,
                    )
                    if (
                        not stat.S_ISREG(created.st_mode)
                        or int(created.st_nlink) != 1
                    ):
                        raise OSError("database backup target is unsafe")
                    with temporary.open("rb") as source_file:
                        source_details = os.fstat(source_file.fileno())
                        if (
                            not stat.S_ISREG(source_details.st_mode)
                            or int(source_details.st_nlink) != 1
                            or descriptor_file_identity(
                                source_file.fileno(),
                                details=source_details,
                            ) != temporary_identity
                        ):
                            raise OSError(
                                "database backup temporary changed during copy"
                            )
                        while chunk := source_file.read(1024 * 1024):
                            remaining = memoryview(chunk)
                            while remaining:
                                written = os.write(
                                    target_descriptor,
                                    remaining,
                                )
                                if written <= 0:
                                    raise OSError(
                                        "database backup copy made no progress"
                                    )
                                remaining = remaining[written:]
                    os.fsync(target_descriptor)
                    copied = os.fstat(target_descriptor)
                    if int(copied.st_size) != int(completed.st_size):
                        raise OSError("database backup copy is incomplete")
                finally:
                    if target_descriptor is not None:
                        os.close(target_descriptor)
                temporary.unlink()
                temporary = None
        if hardlinked:
            published = target.lstat()
            if (
                not stat.S_ISREG(published.st_mode)
                or int(published.st_nlink) != 2
                or path_file_identity(target, details=published)
                != temporary_identity
            ):
                raise OSError("database backup target changed during publish")
            published_identity = temporary_identity
            temporary.unlink()
            temporary = None
        from app.storage_policy import fsync_parent_directory

        fsync_parent_directory(target)
        final = target.lstat()
        if (
            not stat.S_ISREG(final.st_mode)
            or int(final.st_nlink) != 1
            or path_file_identity(target, details=final)
            != published_identity
        ):
            raise OSError("database backup target failed final validation")
    except Exception:
        if published_identity is not None:
            try:
                current = target.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (
                stat.S_ISREG(current.st_mode)
                and int(current.st_nlink) == 1
                and path_file_identity(target, details=current)
                == published_identity
            ):
                target.unlink()
        if temporary is not None and temporary_identity is not None:
            try:
                current = temporary.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (
                stat.S_ISREG(current.st_mode)
                and int(current.st_nlink) == 1
                and path_file_identity(temporary, details=current)
                == temporary_identity
            ):
                temporary.unlink()
        raise
    return target


def set_settings_for_database(
    database_path,
    values,
    *,
    checkpoint_wal=True,
):
    """Update settings in a selected database without changing global state."""
    path = Path(database_path).expanduser().resolve()
    with closing(sqlite3.connect(path, timeout=20)) as con:
        # Configuration commits must survive power loss even when the live DB
        # normally uses the lower-latency WAL synchronous=NORMAL setting.
        con.execute("PRAGMA synchronous=FULL")
        con.executemany(
            "INSERT INTO settings(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ((key, str(value)) for key, value in values.items()),
        )
        con.commit()
        journal_mode = str(
            con.execute("PRAGMA journal_mode").fetchone()[0]
        ).lower()
        if checkpoint_wal and journal_mode == "wal":
            checkpoint = con.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if checkpoint and int(checkpoint[0]) != 0:
                raise sqlite3.OperationalError(
                    "migrated settings WAL checkpoint was busy"
                )
            # This helper edits an offline migration copy. Leave every
            # committed setting in the main database file and remove the WAL
            # transport contract before the config pointer can be switched.
            # The normal startup path enables WAL again after restart.
            selected_mode = str(
                con.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
            ).lower()
            if selected_mode != "delete":
                raise sqlite3.OperationalError(
                    "migrated settings database could not leave WAL mode"
                )
        checked = con.execute("PRAGMA quick_check").fetchone()
        if not checked or checked[0] != "ok":
            raise sqlite3.DatabaseError(
                "settings database integrity check failed"
            )
    if checkpoint_wal:
        from app.storage_policy import fsync_parent_directory

        with path.open("r+b") as database_file:
            os.fsync(database_file.fileno())
        fsync_parent_directory(path)


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


def _require_legacy_admin_password_change(con):
    """Confine the pre-setup administrator once on upgrade.

    Older releases created a predictable bootstrap account.  The plaintext
    credential is deliberately absent from current source.  A one-time
    migration confines any existing ``admin`` account until its owner replaces
    the old credential; first-run databases have no account and use /setup.
    """

    marker = "migration_legacy_admin_password_change_rc30_v1"
    if con.execute(
        "SELECT 1 FROM settings WHERE key=?",
        (marker,),
    ).fetchone() is not None:
        return
    con.execute(
        "UPDATE users SET must_change_password=1,"
        "session_version=session_version+1 "
        "WHERE username='admin' AND is_active=1"
    )
    con.execute(
        "INSERT INTO settings(key,value) VALUES(?,?)",
        (marker, "1"),
    )


def _supersede_duplicate_current_feedback(con):
    """Keep exactly one current operator truth row for each event."""

    con.execute(
        "UPDATE anpr_feedback AS older SET status='superseded' "
        "WHERE older.status='confirmed' AND EXISTS("
        "SELECT 1 FROM anpr_feedback AS newer "
        "WHERE newer.event_id=older.event_id "
        "AND newer.status='confirmed' AND newer.id>older.id)"
    )


def _migrate_persistence_receipts_to_tombstones(con):
    """Remove legacy cascade semantics while preserving idempotency keys."""

    if not con.execute(
        "PRAGMA foreign_key_list(anpr_persistence_receipts)"
    ).fetchall():
        return
    # A receipt must outlive retention of its event; otherwise a delayed
    # outbox replay can resurrect (or duplicate) a deliberately expired row.
    # SQLite cannot drop a foreign key in place, so rebuild the small table in
    # the surrounding init_db transaction.
    con.execute("DROP TABLE IF EXISTS anpr_persistence_receipts_legacy")
    con.execute(
        "ALTER TABLE anpr_persistence_receipts "
        "RENAME TO anpr_persistence_receipts_legacy"
    )
    con.execute("""
        CREATE TABLE anpr_persistence_receipts(
            persistence_key TEXT PRIMARY KEY,
            event_id INTEGER NOT NULL,
            committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    con.execute("""
        INSERT INTO anpr_persistence_receipts(
            persistence_key,event_id,committed_at
        )
        SELECT persistence_key,event_id,committed_at
        FROM anpr_persistence_receipts_legacy
    """)
    con.execute("DROP TABLE anpr_persistence_receipts_legacy")


def _migrate_feedback_to_retained_truth(con):
    """Detach confirmed operator truth from event-retention lifecycle."""

    table_info = con.execute(
        "PRAGMA table_info(anpr_feedback)"
    ).fetchall()
    event_column = next(
        (row for row in table_info if row[1] == "event_id"),
        None,
    )
    foreign_keys = con.execute(
        "PRAGMA foreign_key_list(anpr_feedback)"
    ).fetchall()
    has_set_null_event_fk = any(
        row[3] == "event_id"
        and row[2] == "plate_events"
        and str(row[6]).upper() == "SET NULL"
        for row in foreign_keys
    )
    if (
        event_column is not None
        and not int(event_column[3] or 0)
        and has_set_null_event_fk
    ):
        return

    # SQLite cannot alter a foreign key or NOT NULL constraint in place.
    # All current columns have already been backfilled by init_db before this
    # migration runs, so rebuild the table in the same connection.
    con.execute("DROP TABLE IF EXISTS anpr_feedback_event_legacy")
    con.execute(
        "ALTER TABLE anpr_feedback "
        "RENAME TO anpr_feedback_event_legacy"
    )
    con.execute("""
        CREATE TABLE anpr_feedback(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER,
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
            sample_path TEXT DEFAULT '',
            sample_sha256 TEXT DEFAULT '',
            dataset_split TEXT DEFAULT '',
            training_status TEXT DEFAULT 'pending',
            trained_run_id INTEGER,
            FOREIGN KEY(event_id) REFERENCES plate_events(id)
                ON DELETE SET NULL
        )
    """)
    con.execute("""
        INSERT INTO anpr_feedback(
            id,event_id,observed_text,observed_norm,observed_engine,
            observed_confidence,observed_model_revision,corrected_text,
            corrected_norm,character_distance,exact_match,plate_image_path,
            image_path,submitted_by,status,created_at,sample_path,
            sample_sha256,dataset_split,training_status,trained_run_id
        )
        SELECT
            id,event_id,observed_text,observed_norm,observed_engine,
            observed_confidence,observed_model_revision,corrected_text,
            corrected_norm,character_distance,exact_match,plate_image_path,
            image_path,submitted_by,status,created_at,sample_path,
            sample_sha256,dataset_split,training_status,trained_run_id
        FROM anpr_feedback_event_legacy
    """)
    con.execute("DROP TABLE anpr_feedback_event_legacy")


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


def _backfill_event_metadata(con):
    city_marker = "migration_backfill_event_city_rc17"
    if con.execute(
        "SELECT 1 FROM settings WHERE key=?",
        (city_marker,),
    ).fetchone() is None:
        con.execute(
            "UPDATE plate_events SET city=COALESCE(("
            "SELECT city FROM cameras c "
            "WHERE c.id=plate_events.camera_id),'') "
            "WHERE city IS NULL OR TRIM(city)=''"
        )
        con.execute(
            "INSERT INTO settings(key,value) VALUES(?,?)",
            (city_marker, "1"),
        )
    con.execute(
        "UPDATE plate_events SET plate_region=SUBSTR("
        "COALESCE(NULLIF(plate_norm,''),raw_guess_norm),-2) "
        "WHERE LENGTH(COALESCE(NULLIF(plate_norm,''),raw_guess_norm))=8 "
        "AND SUBSTR(COALESCE(NULLIF(plate_norm,''),raw_guess_norm),-2) "
        "GLOB '[0-9][0-9]' "
        "AND (plate_region IS NULL OR plate_region='')"
    )
    media_rows = con.execute(
        "SELECT id,image_path,plate_image_path,media_error "
        "FROM plate_events WHERE media_status IS NULL "
        "OR media_status='' OR media_status='pending'"
    ).fetchall()
    for row in media_rows:
        requested = [
            str(value)
            for value in (
                row["image_path"],
                row["plate_image_path"],
            )
            if value
        ]
        present = []
        for value in requested:
            try:
                path = Path(value)
                present.append(
                    path.is_file() and path.stat().st_size > 0
                )
            except OSError:
                present.append(False)
        if not requested:
            status = "missing"
        elif all(present):
            status = "complete"
        elif any(present):
            status = "partial"
        else:
            status = "missing"
        error = str(row["media_error"] or "")
        if requested and not all(present) and not error:
            error = (
                "یک یا چند فایل تصویر تاریخی در مسیر ثبت‌شده "
                "پیدا نشد."
            )
        con.execute(
            "UPDATE plate_events SET media_status=?,media_error=? "
            "WHERE id=?",
            (status, error, row["id"]),
        )
    con.execute(
        "UPDATE plate_events SET created_at=CURRENT_TIMESTAMP "
        "WHERE created_at IS NULL OR created_at=''"
    )
    con.execute(
        "UPDATE plate_events SET updated_at="
        "COALESCE(created_at,CURRENT_TIMESTAMP) "
        "WHERE updated_at IS NULL OR updated_at=''"
    )


def _mark_pre_rc29_uploaded_videos_completed(con):
    """Prevent legacy uploaded files from being ANPR-processed again."""

    migration_key = "migration_video_anpr_markers_rc29_v1"
    if con.execute(
        "SELECT 1 FROM settings WHERE key=?",
        (migration_key,),
    ).fetchone() is not None:
        return
    # Before RC29 uploaded videos had no durable pass marker. They were already
    # processed when originally uploaded, so the only safe upgrade behavior is
    # preview-only. The migration key is written even on a fresh empty database;
    # videos added later can therefore remain started-but-incomplete after a
    # decoder/app interruption without a later init incorrectly completing them.
    con.execute(
        "UPDATE cameras SET video_anpr_started=1,"
        "video_anpr_completed=1,"
        "video_anpr_completed_at="
        "COALESCE(NULLIF(video_anpr_completed_at,''),CURRENT_TIMESTAMP) "
        "WHERE rtsp_url LIKE 'video://%'"
    )
    con.execute(
        "INSERT INTO settings(key,value) VALUES(?,?)",
        (migration_key, "1"),
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
            must_change_password INTEGER NOT NULL DEFAULT 0,
            session_version INTEGER NOT NULL DEFAULT 0,
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
        CREATE TABLE IF NOT EXISTS revoked_sessions(
            token_hash TEXT PRIMARY KEY,
            expires_at INTEGER NOT NULL,
            revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_revoked_sessions_expiry
            ON revoked_sessions(expires_at);
        CREATE TABLE IF NOT EXISTS settings(
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cameras(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            rtsp_url TEXT NOT NULL DEFAULT '',
            location TEXT NOT NULL DEFAULT '',
            city TEXT NOT NULL DEFAULT '',
            enabled INTEGER NOT NULL DEFAULT 1,
            is_demo INTEGER NOT NULL DEFAULT 0,
            sort_order INTEGER NOT NULL DEFAULT 0,
            video_anpr_started INTEGER NOT NULL DEFAULT 0,
            video_anpr_completed INTEGER NOT NULL DEFAULT 0,
            video_anpr_completed_at TEXT NOT NULL DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT DEFAULT '',
            confidence REAL DEFAULT 0,
            camera_id INTEGER,
            camera_name TEXT,
            city TEXT DEFAULT '',
            plate_region TEXT DEFAULT '',
            image_path TEXT,
            plate_image_path TEXT,
            media_status TEXT NOT NULL DEFAULT 'pending',
            media_error TEXT DEFAULT '',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
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
            event_id INTEGER,
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
                ON DELETE SET NULL
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
        CREATE TABLE IF NOT EXISTS anpr_persistence_receipts(
            persistence_key TEXT PRIMARY KEY,
            event_id INTEGER NOT NULL,
            committed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS media_acceptance_intents(
            acceptance_id TEXT PRIMARY KEY,
            target_path TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'pending',
            device INTEGER,
            inode INTEGER,
            size_bytes INTEGER,
            owner_kind TEXT NOT NULL DEFAULT '',
            owner_id TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            accepted_at TEXT,
            CHECK(state IN ('pending','accepted'))
        );
        """)

        _add_missing_columns(con, "users", {
            "role": "TEXT NOT NULL DEFAULT 'admin'",
            "is_active": "INTEGER NOT NULL DEFAULT 1",
            "failed_attempts": "INTEGER NOT NULL DEFAULT 0",
            "locked_until": "TEXT",
            "last_login": "TEXT",
            "must_change_password": "INTEGER NOT NULL DEFAULT 0",
            "session_version": "INTEGER NOT NULL DEFAULT 0",
            "created_at": "TEXT",
        })
        con.execute(
            "DELETE FROM revoked_sessions WHERE expires_at < ?",
            (int(time.time()),),
        )
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
            "city": "TEXT DEFAULT ''",
            "plate_region": "TEXT DEFAULT ''",
            "media_status": "TEXT NOT NULL DEFAULT 'pending'",
            "media_error": "TEXT DEFAULT ''",
            "updated_at": "TEXT",
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
        _migrate_feedback_to_retained_truth(con)
        _supersede_duplicate_current_feedback(con)
        _add_missing_columns(con, "anpr_training_runs", {
            "baseline_mean_character_error": "REAL DEFAULT 0",
            "candidate_mean_character_error": "REAL DEFAULT 0",
            "baseline_sha256": "TEXT DEFAULT ''",
            "candidate_checkpoint_path": "TEXT DEFAULT ''",
            "candidate_checkpoint_sha256": "TEXT DEFAULT ''",
            "promotion_report": "TEXT DEFAULT ''",
            "dataset_manifest_path": "TEXT DEFAULT ''",
            "dataset_manifest_sha256": "TEXT DEFAULT ''",
        })
        _migrate_persistence_receipts_to_tombstones(con)
        _backfill_plate_norm(con)
        con.executescript("""
        CREATE INDEX IF NOT EXISTS idx_plate_events_created_at
            ON plate_events(created_at);
        CREATE INDEX IF NOT EXISTS idx_plate_events_plate_norm
            ON plate_events(plate_norm);
        CREATE INDEX IF NOT EXISTS idx_plate_events_camera_created
            ON plate_events(camera_id,created_at);
        CREATE INDEX IF NOT EXISTS idx_plate_events_city_created
            ON plate_events(city,created_at,id);
        CREATE INDEX IF NOT EXISTS idx_plate_events_region_created
            ON plate_events(plate_region,created_at,id);
        CREATE INDEX IF NOT EXISTS idx_plate_events_updated_at
            ON plate_events(updated_at,id);
        CREATE INDEX IF NOT EXISTS idx_anpr_receipts_event
            ON anpr_persistence_receipts(event_id);
        CREATE INDEX IF NOT EXISTS idx_media_acceptance_state
            ON media_acceptance_intents(state,created_at);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_observed
            ON anpr_feedback(observed_norm,status);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_event
            ON anpr_feedback(event_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_anpr_feedback_current_event
            ON anpr_feedback(event_id) WHERE status='confirmed';
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_training
            ON anpr_feedback(training_status,dataset_split);
        CREATE INDEX IF NOT EXISTS idx_anpr_feedback_model_revision
            ON anpr_feedback(observed_model_revision,created_at);
        CREATE INDEX IF NOT EXISTS idx_anpr_training_runs_created
            ON anpr_training_runs(created_at);
        """)

        _add_missing_columns(con, "cameras", {
            "city": "TEXT NOT NULL DEFAULT ''",
            "lpr_enabled": "INTEGER NOT NULL DEFAULT 1",
            "lpr_confidence": "INTEGER NOT NULL DEFAULT 60",
            "frame_step": "INTEGER NOT NULL DEFAULT 5",
            "duplicate_seconds": "REAL NOT NULL DEFAULT 30",
            "video_anpr_started": "INTEGER NOT NULL DEFAULT 0",
            "video_anpr_completed": "INTEGER NOT NULL DEFAULT 0",
            "video_anpr_completed_at": "TEXT NOT NULL DEFAULT ''",
            "roi_x": "INTEGER NOT NULL DEFAULT 0",
            "roi_y": "INTEGER NOT NULL DEFAULT 0",
            "roi_w": "INTEGER NOT NULL DEFAULT 100",
            "roi_h": "INTEGER NOT NULL DEFAULT 100",
            "line_y": "INTEGER NOT NULL DEFAULT 50",
        })
        _mark_pre_rc29_uploaded_videos_completed(con)
        _backfill_event_metadata(con)

        _require_legacy_admin_password_change(con)

        defaults = {
            "company_name": "گیلاس آبی البرز",
            "dashboard_grid": "2",
            "dashboard_event_rows": "12",
            "live_fps": "5",
            "stream_width": "640",
            "jpeg_quality": "70",
            "storage_root": str(DB_PATH.parent),
            "snapshot_path": str(DB_PATH.parent / "snapshots"),
            "plate_path": str(DB_PATH.parent / "plates"),
            "video_path": str(DB_PATH.parent / "videos"),
            "backup_path": str(DB_PATH.parent / "backups"),
            "media_roots_history": "[]",
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
            "anpr_detector_model": "yolov8n",
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
