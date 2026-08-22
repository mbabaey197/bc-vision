from pathlib import Path
import json, os, secrets, stat, sys

from app.file_identity import descriptor_file_identity, path_file_identity

COMPANY_NAME = "گیلاس آبی البرز"
APP_NAME = "BC Vision"
APP_VERSION = "2.2.0-rc30"
HOST = "127.0.0.1"
PORT = 8000
MAX_STORAGE_CONFIG_BYTES = 64 * 1024
STORAGE_MIGRATION_MARKER_NAME = "storage_config.migrated"
STORAGE_MIGRATION_MARKER_PAYLOAD = b"BCVISION_STORAGE_POINTER_V1\n"


class StorageConfigurationError(RuntimeError):
    """Raised when a published storage-root pointer is not trustworthy."""


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
STORAGE_MIGRATION_MARKER_PATH = (
    BOOTSTRAP_DIR / STORAGE_MIGRATION_MARKER_NAME
)


def _storage_pointer_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise StorageConfigurationError(
                "Storage configuration contains a duplicate key."
            )
        result[key] = value
    return result


def _read_private_regular_file(
    path: Path,
    *,
    maximum_bytes: int,
    label: str,
) -> bytes | None:
    """Read a stable private file, returning ``None`` only when absent."""

    path = Path(path)
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise StorageConfigurationError(
            f"{label} could not be inspected."
        ) from exc
    if (
        not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise StorageConfigurationError(
            f"{label} is not a private regular file."
        )
    if int(before.st_size) <= 0 or int(before.st_size) > maximum_bytes:
        raise StorageConfigurationError(
            f"{label} size is invalid."
        )
    try:
        before_identity = path_file_identity(path, details=before)
    except OSError as exc:
        raise StorageConfigurationError(
            f"{label} could not be identified."
        ) from exc

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        opened_identity = descriptor_file_identity(
            descriptor,
            details=opened,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or opened_identity != before_identity
        ):
            raise StorageConfigurationError(
                f"{label} changed while it was opened."
            )
        chunks = []
        remaining = maximum_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(8192, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        after_identity = descriptor_file_identity(
            descriptor,
            details=after,
        )
    except StorageConfigurationError:
        raise
    except OSError as exc:
        raise StorageConfigurationError(
            f"{label} could not be read."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    if (
        len(payload) > maximum_bytes
        or int(after.st_size) != len(payload)
        or after_identity != opened_identity
        or int(after.st_nlink) != int(opened.st_nlink)
        or int(getattr(after, "st_mtime_ns", 0))
        != int(getattr(opened, "st_mtime_ns", 0))
        or (
            os.name != "nt"
            and int(getattr(after, "st_ctime_ns", 0))
            != int(getattr(opened, "st_ctime_ns", 0))
        )
    ):
        raise StorageConfigurationError(
            f"{label} changed while it was read."
        )

    try:
        current = path.lstat()
    except OSError as exc:
        raise StorageConfigurationError(
            f"{label} changed after it was read."
        ) from exc
    try:
        current_identity = path_file_identity(path, details=current)
    except OSError as exc:
        raise StorageConfigurationError(
            f"{label} changed after it was read."
        ) from exc
    if (
        not stat.S_ISREG(current.st_mode)
        or int(current.st_nlink) != 1
        or current_identity != after_identity
        or int(current.st_size) != int(after.st_size)
        or int(getattr(current, "st_mtime_ns", 0))
        != int(getattr(after, "st_mtime_ns", 0))
        or (
            os.name != "nt"
            and int(getattr(current, "st_ctime_ns", 0))
            != int(getattr(after, "st_ctime_ns", 0))
        )
    ):
        raise StorageConfigurationError(
            f"{label} changed after it was read."
        )
    return payload


def _read_storage_pointer(path: Path) -> dict | None:
    """Read one stable, private pointer or return ``None`` when truly absent."""

    payload = _read_private_regular_file(
        path,
        maximum_bytes=MAX_STORAGE_CONFIG_BYTES,
        label="Storage configuration",
    )
    if payload is None:
        return None
    try:
        decoded = payload.decode("utf-8")
        data = json.loads(decoded, object_pairs_hook=_storage_pointer_object)
    except StorageConfigurationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise StorageConfigurationError(
            "Storage configuration is not valid UTF-8 JSON."
        ) from exc
    if not isinstance(data, dict) or set(data) != {"storage_root"}:
        raise StorageConfigurationError(
            "Storage configuration must contain only storage_root."
        )
    return data


def _read_storage_migration_marker(path: Path) -> bool | None:
    payload = _read_private_regular_file(
        path,
        maximum_bytes=len(STORAGE_MIGRATION_MARKER_PAYLOAD),
        label="Storage migration marker",
    )
    if payload is None:
        return None
    if payload != STORAGE_MIGRATION_MARKER_PAYLOAD:
        raise StorageConfigurationError(
            "Storage migration marker content is invalid."
        )
    return True


def _fsync_storage_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = None
    try:
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        os.fsync(descriptor)
    except OSError as exc:
        raise StorageConfigurationError(
            "Storage migration marker directory could not be synchronized."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _unlink_owned_marker_temporary(
    path: Path,
    identity: tuple[int, int],
) -> None:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise StorageConfigurationError(
            "Storage migration marker temporary could not be inspected."
        ) from exc
    if (
        not stat.S_ISREG(details.st_mode)
        or path_file_identity(path, details=details) != identity
    ):
        raise StorageConfigurationError(
            "Storage migration marker temporary was replaced and preserved."
        )
    try:
        path.unlink()
    except OSError as exc:
        raise StorageConfigurationError(
            "Storage migration marker temporary could not be removed."
        ) from exc


def _create_storage_migration_marker(path: Path) -> None:
    """Publish the exact marker without replacing any existing entry."""

    path = Path(path)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    temporary = None
    descriptor = None
    identity = None
    for _attempt in range(8):
        candidate = path.with_name(
            f".{path.name}.{secrets.token_hex(8)}.tmp"
        )
        try:
            descriptor = os.open(candidate, flags, 0o600)
        except FileExistsError:
            continue
        except OSError as exc:
            raise StorageConfigurationError(
                "Storage migration marker temporary could not be created."
            ) from exc
        temporary = candidate
        break
    if temporary is None or descriptor is None:
        raise StorageConfigurationError(
            "Storage migration marker temporary name could not be reserved."
        )

    try:
        opened = os.fstat(descriptor)
        identity = descriptor_file_identity(descriptor, details=opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
        ):
            raise StorageConfigurationError(
                "Storage migration marker temporary is unsafe."
            )
        remaining = memoryview(STORAGE_MIGRATION_MARKER_PAYLOAD)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("marker write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        completed = os.fstat(descriptor)
        if (
            not stat.S_ISREG(completed.st_mode)
            or int(completed.st_nlink) != 1
            or descriptor_file_identity(
                descriptor,
                details=completed,
            ) != identity
            or int(completed.st_size)
            != len(STORAGE_MIGRATION_MARKER_PAYLOAD)
        ):
            raise StorageConfigurationError(
                "Storage migration marker temporary changed while written."
            )
    except (StorageConfigurationError, OSError) as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        if identity is not None:
            _unlink_owned_marker_temporary(temporary, identity)
            temporary = None
        if isinstance(exc, StorageConfigurationError):
            raise
        raise StorageConfigurationError(
            "Storage migration marker temporary could not be written."
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None

    try:
        current = temporary.lstat()
        if (
            not stat.S_ISREG(current.st_mode)
            or int(current.st_nlink) != 1
            or path_file_identity(temporary, details=current) != identity
            or int(current.st_size)
            != len(STORAGE_MIGRATION_MARKER_PAYLOAD)
        ):
            raise StorageConfigurationError(
                "Storage migration marker temporary changed before publish."
            )
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _unlink_owned_marker_temporary(temporary, identity)
            temporary = None
            _fsync_storage_directory(path.parent)
            if _read_storage_migration_marker(path) is not True:
                raise StorageConfigurationError(
                    "Storage migration marker disappeared during creation."
                )
            return
        except OSError as exc:
            raise StorageConfigurationError(
                "Storage migration marker could not be published."
            ) from exc

        published = path.lstat()
        if (
            not stat.S_ISREG(published.st_mode)
            or int(published.st_nlink) != 2
            or path_file_identity(path, details=published) != identity
        ):
            raise StorageConfigurationError(
                "Storage migration marker changed while published."
            )
        _unlink_owned_marker_temporary(temporary, identity)
        temporary = None
        _fsync_storage_directory(path.parent)
        if _read_storage_migration_marker(path) is not True:
            raise StorageConfigurationError(
                "Storage migration marker could not be verified."
            )
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None and identity is not None:
            _unlink_owned_marker_temporary(temporary, identity)


def _ensure_storage_migration_marker(path: Path) -> None:
    if _read_storage_migration_marker(path) is True:
        return
    _create_storage_migration_marker(path)


def ensure_storage_migration_marker(
    path: Path = STORAGE_MIGRATION_MARKER_PATH,
) -> None:
    """Durably publish or validate the post-migration bootstrap sentinel."""

    _ensure_storage_migration_marker(Path(path))


def _configured_root(
    pointer_path: Path = STORAGE_CONFIG_PATH,
    bootstrap_dir: Path = BOOTSTRAP_DIR,
    marker_path: Path | None = None,
) -> Path:
    pointer_path = Path(pointer_path)
    if marker_path is None:
        marker_path = pointer_path.with_name(STORAGE_MIGRATION_MARKER_NAME)
    marker_path = Path(marker_path)
    data = _read_storage_pointer(pointer_path)
    if data is None:
        if _read_storage_migration_marker(marker_path) is True:
            raise StorageConfigurationError(
                "Storage configuration is missing after storage migration."
            )
        return Path(bootstrap_dir).resolve(strict=False)

    value = data["storage_root"]
    if not isinstance(value, str) or not value or value != value.strip():
        raise StorageConfigurationError(
            "Storage configuration has an invalid storage_root."
        )
    try:
        value.encode("utf-8", errors="strict")
        candidate = Path(value)
        if not candidate.is_absolute():
            raise StorageConfigurationError(
                "Configured storage_root must be absolute."
            )
        resolved = candidate.resolve(strict=False)
    except StorageConfigurationError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError) as exc:
        raise StorageConfigurationError(
            "Configured storage_root could not be resolved."
        ) from exc
    if candidate != resolved:
        raise StorageConfigurationError(
            "Configured storage_root must be canonical."
        )
    if resolved == Path(resolved.anchor):
        raise StorageConfigurationError(
            "Configured storage_root cannot be a filesystem root."
        )
    try:
        root_details = resolved.lstat()
    except OSError as exc:
        raise StorageConfigurationError(
            "Configured storage_root is unavailable."
        ) from exc
    if stat.S_ISLNK(root_details.st_mode) or not stat.S_ISDIR(
        root_details.st_mode
    ):
        raise StorageConfigurationError(
            "Configured storage_root is not a real directory."
        )

    database_path = resolved / "bcvision.db"
    try:
        database_details = database_path.lstat()
    except OSError as exc:
        raise StorageConfigurationError(
            "Configured storage database is unavailable."
        ) from exc
    if (
        not stat.S_ISREG(database_details.st_mode)
        or int(database_details.st_nlink) != 1
    ):
        raise StorageConfigurationError(
            "Configured storage database is not a private regular file."
        )
    ensure_storage_migration_marker(marker_path)
    return resolved

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
