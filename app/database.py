import sqlite3
from app.config import DB_PATH
from app.security import hash_password

def connect():
    con = sqlite3.connect(DB_PATH, timeout=20, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db():
    with connect() as con:
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
        """)
        user_columns={r[1] for r in con.execute("PRAGMA table_info(users)").fetchall()}
        user_migrations={
            "role":"TEXT NOT NULL DEFAULT 'admin'",
            "is_active":"INTEGER NOT NULL DEFAULT 1",
            "failed_attempts":"INTEGER NOT NULL DEFAULT 0",
            "locked_until":"TEXT",
            "last_login":"TEXT",
            "created_at":"TEXT"
        }
        for name,sql_type in user_migrations.items():
            if name not in user_columns:
                con.execute(f"ALTER TABLE users ADD COLUMN {name} {sql_type}")
        con.execute("UPDATE users SET role='admin' WHERE is_admin=1 AND (role IS NULL OR role='')")
        con.execute("UPDATE users SET created_at=CURRENT_TIMESTAMP WHERE created_at IS NULL OR created_at=''")

        # Backward-compatible event metadata migration.
        event_columns={r[1] for r in con.execute("PRAGMA table_info(plate_events)").fetchall()}
        for name,sql_type in {"plate_image_path":"TEXT","video_path":"TEXT","video_second":"REAL DEFAULT 0","detector_method":"TEXT","ocr_confidence":"REAL DEFAULT 0","vehicle_type":"TEXT DEFAULT 'نامشخص'","vehicle_color":"TEXT DEFAULT 'نامشخص'","vehicle_brand":"TEXT DEFAULT 'نامشخص'","vehicle_confidence":"REAL DEFAULT 0"}.items():
            if name not in event_columns:
                con.execute(f"ALTER TABLE plate_events ADD COLUMN {name} {sql_type}")
        camera_columns={r[1] for r in con.execute("PRAGMA table_info(cameras)").fetchall()}
        camera_migrations={
            "lpr_enabled":"INTEGER NOT NULL DEFAULT 1",
            "lpr_confidence":"INTEGER NOT NULL DEFAULT 60",
            "frame_step":"INTEGER NOT NULL DEFAULT 5",
            "duplicate_seconds":"REAL NOT NULL DEFAULT 30",
            "roi_x":"INTEGER NOT NULL DEFAULT 0",
            "roi_y":"INTEGER NOT NULL DEFAULT 0",
            "roi_w":"INTEGER NOT NULL DEFAULT 100",
            "roi_h":"INTEGER NOT NULL DEFAULT 100",
            "line_y":"INTEGER NOT NULL DEFAULT 50"
        }
        for name,sql_type in camera_migrations.items():
            if name not in camera_columns:
                con.execute(f"ALTER TABLE cameras ADD COLUMN {name} {sql_type}")
        if con.execute("SELECT 1 FROM users WHERE username=?", ("admin",)).fetchone() is None:
            con.execute("INSERT INTO users(username,password_hash,display_name,is_admin) VALUES(?,?,?,1)",
                        ("admin", hash_password("123456"), "مدیر سیستم"))
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
        }
        for k,v in defaults.items():
            con.execute("INSERT OR IGNORE INTO settings(key,value) VALUES(?,?)", (k,v))
        if con.execute("SELECT COUNT(*) c FROM cameras").fetchone()["c"] == 0:
            con.execute("INSERT INTO cameras(name,rtsp_url,location,enabled,is_demo,sort_order) VALUES(?,?,?,?,?,?)",
                        ("دوربین آزمایشی", "demo://camera-1", "نمایش نمونه", 1, 1, 1))

def get_setting(key, default=""):
    with connect() as con:
        row = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

def set_setting(key, value):
    with connect() as con:
        con.execute("INSERT INTO settings(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key,str(value)))
