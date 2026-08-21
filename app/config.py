from pathlib import Path
import json, os, sys

COMPANY_NAME = "گیلاس آبی البرز"
APP_NAME = "BC Vision"
APP_VERSION = "2.2.0-rc29.2"
HOST = "127.0.0.1"
PORT = 8000


def install_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_data_dir() -> Path:
    override = os.environ.get("BCVISION_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if os.name == "nt":
        base = Path(os.environ.get("PROGRAMDATA", Path.home() / "AppData" / "Local"))
        return base / "BCVision" / "data"
    if not getattr(sys, "frozen", False):
        return install_dir() / "data"
    return Path.home() / ".local" / "share" / "BCVision" / "data"

BASE_DIR = install_dir()
BOOTSTRAP_DIR = default_data_dir()
BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
STORAGE_CONFIG_PATH = BOOTSTRAP_DIR / "storage_config.json"

def _configured_root() -> Path:
    try:
        data = json.loads(STORAGE_CONFIG_PATH.read_text(encoding="utf-8"))
        value = str(data.get("storage_root", "")).strip()
        if value:
            return Path(value).expanduser()
    except Exception:
        pass
    return BOOTSTRAP_DIR

DATA_DIR = _configured_root()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "bcvision.db"
LOG_PATH = DATA_DIR / "BCVision.log"
SECRET_PATH = DATA_DIR / ".secret"
BACKUP_DIR = DATA_DIR / "backups"
SNAPSHOT_DIR = DATA_DIR / "snapshots"
PLATE_DIR = DATA_DIR / "plates"
VIDEO_DIR = DATA_DIR / "videos"
for folder in (BACKUP_DIR, SNAPSHOT_DIR, PLATE_DIR, VIDEO_DIR):
    folder.mkdir(parents=True, exist_ok=True)
