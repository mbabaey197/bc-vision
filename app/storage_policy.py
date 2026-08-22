"""Thread-safe storage quota enforcement for managed media files.

Configured snapshot, plate, video and safe same-root historical directories
are managed here. External historical roots are accounting-only. Database,
secret, retry-outbox and backup files deliberately remain outside the delete
candidate set, even when they live below the same storage root.
"""
from __future__ import annotations

import heapq
import json
import os
import sqlite3
import stat
import threading
import time
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.config import DATA_DIR, PLATE_DIR, SNAPSHOT_DIR, VIDEO_DIR

GIB = 1024 * 1024 * 1024
DEFAULT_MAX_SCAN_ENTRIES = 200_000
DEFAULT_MAX_DELETE_FILES = 256
MAX_HISTORY_ROOTS = 1024
MAX_PROTECTED_WRITE_PATHS = 64
_CACHE_SECONDS = 1.0
_QUARANTINE_DIRECTORY = ".bcvision-media-quarantine"
_QUARANTINE_JOURNAL = "journal.json"
_QUARANTINE_VERSION = 3
_SUPPORTED_QUARANTINE_VERSIONS = frozenset({1, 2, 3})


class StoragePolicyError(RuntimeError):
    """Base error for an invalid or unsafe storage policy operation."""


class StorageWriteRejected(StoragePolicyError):
    """Raised before or during a managed media write that exceeds policy."""


class _StorageCommitMarkerError(StoragePolicyError):
    """A retryable failure before a reservation became durably committed."""


@dataclass(frozen=True)
class StoragePolicyConfig:
    storage_root: Path
    media_roots: tuple[Path, ...]
    read_only_history_roots: tuple[Path, ...]
    invalid_history_roots: tuple[str, ...]
    limit_bytes: int
    action: str


@dataclass(frozen=True)
class StorageStatus:
    storage_root: str
    media_roots: tuple[str, ...]
    action: str
    limit_bytes: int
    managed_bytes: int
    reserved_bytes: int
    over_limit: bool
    write_blocked: bool
    usage_complete: bool
    scanned_files: int
    read_only_history_roots: tuple[str, ...] = ()
    invalid_history_roots: tuple[str, ...] = ()
    deleted_files: int = 0
    deleted_bytes: int = 0
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "storage_root": self.storage_root,
            "media_roots": list(self.media_roots),
            "action": self.action,
            "limit_bytes": self.limit_bytes,
            "managed_bytes": self.managed_bytes,
            "reserved_bytes": self.reserved_bytes,
            "over_limit": self.over_limit,
            "write_blocked": self.write_blocked,
            "usage_complete": self.usage_complete,
            "scanned_files": self.scanned_files,
            "read_only_history_roots": list(self.read_only_history_roots),
            "invalid_history_roots": list(self.invalid_history_roots),
            "deleted_files": self.deleted_files,
            "deleted_bytes": self.deleted_bytes,
            "error": self.error,
        }


@dataclass(frozen=True)
class _FileRecord:
    path: Path
    size: int
    mtime_ns: int
    device: int
    inode: int
    nlink: int
    deletable: bool


@dataclass(frozen=True)
class _QuarantineMove:
    original: Path
    quarantine_name: str
    size: int
    mtime_ns: int
    device: int
    inode: int
    nlink: int


@dataclass
class _Inventory:
    managed_bytes: int
    scanned_files: int
    complete: bool
    oldest: list[_FileRecord]
    expires_at: float


@dataclass
class _ReservationState:
    signature: tuple[str, ...]
    target: Path
    protected_paths: tuple[Path, ...]
    protected_paths_existed: tuple[bool, ...]
    expected_bytes: int
    original_size: int
    target_existed: bool
    original_file_identity: tuple[int, int] | None
    owned_file_identities: set[tuple[int, int]]
    config: StoragePolicyConfig
    acceptance_id: str | None = None
    quarantine_id: str | None = None
    quarantine_moves: list[_QuarantineMove] | None = None


_LOCK = threading.RLock()
_CACHE: dict[tuple[str, ...], _Inventory] = {}
_RESERVATIONS: dict[str, _ReservationState] = {}
_PIN_LEASES: dict[str, tuple[Path, ...]] = {}


class WriterPreferredGate:
    """Non-blocking reader/writer gate for use from async middleware.

    Normal mutations acquire a shared slot. A storage migration queues an
    exclusive ticket; once queued, later readers wait so a continuous stream
    of normal requests cannot starve the migration.
    """

    def __init__(self):
        self._state_lock = threading.Lock()
        self._readers = 0
        self._writer = None
        self._waiting_writers = deque()

    def try_acquire_shared(self) -> bool:
        with self._state_lock:
            if self._writer is not None or self._waiting_writers:
                return False
            self._readers += 1
            return True

    def release_shared(self) -> None:
        with self._state_lock:
            if self._readers <= 0:
                raise RuntimeError("shared storage gate is not held")
            self._readers -= 1

    def queue_exclusive(self):
        ticket = object()
        with self._state_lock:
            self._waiting_writers.append(ticket)
        return ticket

    def try_acquire_exclusive(self, ticket) -> bool:
        with self._state_lock:
            if self._writer is ticket:
                return True
            if (
                self._writer is not None
                or self._readers
                or not self._waiting_writers
                or self._waiting_writers[0] is not ticket
            ):
                return False
            self._waiting_writers.popleft()
            self._writer = ticket
            return True

    def cancel_exclusive(self, ticket) -> None:
        with self._state_lock:
            if self._writer is ticket:
                self._writer = None
                return
            try:
                self._waiting_writers.remove(ticket)
            except ValueError:
                pass

    def release_exclusive(self, ticket) -> None:
        with self._state_lock:
            if self._writer is not ticket:
                raise RuntimeError("exclusive storage gate is not held")
            self._writer = None

    def snapshot(self) -> tuple[int, bool, int]:
        """Expose immutable state for deterministic concurrency tests."""

        with self._state_lock:
            return (
                self._readers,
                self._writer is not None,
                len(self._waiting_writers),
            )


def _setting(key: str, default: str) -> str:
    # Import lazily so tests and storage migration can replace DB_PATH without
    # leaving a stale module-level database handle in this policy module.
    from app.database import get_setting

    try:
        return str(get_setting(key, default))
    except sqlite3.OperationalError as exc:
        # Several isolated engine tests intentionally use the minimal legacy
        # plate_events schema. Production startup always creates settings;
        # only that explicitly absent legacy table may use safe defaults.
        if "no such table: settings" in str(exc).lower():
            return str(default)
        raise StoragePolicyError("storage settings are unavailable") from exc


def _resolve_path(value) -> Path:
    return Path(str(value)).expanduser().resolve(strict=False)


def _is_below(path: Path, root: Path) -> bool:
    return path != root and path.is_relative_to(root)


def _paths_overlap(first: Path, second: Path) -> bool:
    return bool(
        first == second
        or first.is_relative_to(second)
        or second.is_relative_to(first)
    )


def _normalise_config(
    *,
    storage_root=None,
    media_roots: Iterable[os.PathLike | str] | None = None,
    limit_bytes: int | None = None,
    action: str | None = None,
) -> StoragePolicyConfig:
    root = _resolve_path(
        storage_root
        if storage_root is not None
        else _setting("storage_root", str(DATA_DIR))
    )
    using_configured_roots = media_roots is None
    raw_roots = (
        (
            _setting("snapshot_path", str(SNAPSHOT_DIR)),
            _setting("plate_path", str(PLATE_DIR)),
            _setting("video_path", str(VIDEO_DIR)),
        )
        if using_configured_roots
        else media_roots
    )
    roots: list[Path] = []
    for raw in raw_roots:
        candidate = _resolve_path(raw)
        if not _is_below(candidate, root):
            raise StoragePolicyError(
                "managed media directory must be below storage root"
            )
        if candidate in roots:
            raise StoragePolicyError(
                "managed media directories must be distinct"
            )
        roots.append(candidate)
    if not roots:
        raise StoragePolicyError("at least one managed media directory is required")
    quarantine_root = root / _QUARANTINE_DIRECTORY
    for media_root in roots:
        if _paths_overlap(media_root, quarantine_root):
            raise StoragePolicyError(
                "managed media directory overlaps quota quarantine"
            )
    for index, media_root in enumerate(roots):
        for other in roots[index + 1 :]:
            if _paths_overlap(media_root, other):
                raise StoragePolicyError(
                    "managed media directories must not overlap"
                )

    read_only_history: list[Path] = []
    invalid_history: list[str] = []
    if using_configured_roots:
        backup = _resolve_path(_setting("backup_path", str(root / "backups")))
        if _paths_overlap(backup, quarantine_root):
            raise StoragePolicyError(
                "backup directory overlaps quota quarantine"
            )
        for media_root in roots:
            if _paths_overlap(media_root, backup):
                raise StoragePolicyError(
                    "managed media directory overlaps backup directory"
                )
        history_raw = _setting("media_roots_history", "[]")
        try:
            if len(history_raw.encode("utf-8")) > 1024 * 1024:
                raise ValueError("media roots history is too large")
            history_payload = json.loads(history_raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            history_payload = []
            invalid_history.append("<invalid-media-roots-history>")
        if not isinstance(history_payload, list):
            invalid_history.append("<invalid-media-roots-history>")
            history_payload = []
        if len(history_payload) > MAX_HISTORY_ROOTS:
            invalid_history.append(
                f"<media-roots-history-overflow:{len(history_payload)}>"
            )
            history_payload = history_payload[:MAX_HISTORY_ROOTS]
        for raw_history in history_payload:
            value = str(raw_history or "").strip()
            if not value:
                continue
            try:
                candidate = _resolve_path(value)
            except (OSError, RuntimeError, ValueError):
                invalid_history.append(value)
                continue
            if candidate == Path(candidate.anchor):
                invalid_history.append(value)
                continue
            if _paths_overlap(candidate, quarantine_root):
                invalid_history.append(value)
                continue
            if _paths_overlap(candidate, backup):
                invalid_history.append(value)
                continue
            if candidate in roots or candidate in read_only_history:
                continue
            if any(_paths_overlap(candidate, item) for item in roots):
                invalid_history.append(value)
                continue
            if any(
                _paths_overlap(candidate, item)
                for item in read_only_history
            ):
                invalid_history.append(value)
                continue
            if _is_below(candidate, root):
                roots.append(candidate)
            else:
                # Previous media roots on another storage volume are counted
                # read-only. They can block a strict quota, but this process
                # will never delete outside the current storage root.
                read_only_history.append(candidate)

    if limit_bytes is None:
        raw_limit = _setting("max_storage_gb", "0")
        try:
            configured_gib = int(raw_limit)
        except (OverflowError, TypeError, ValueError):
            raise StoragePolicyError(
                "configured media storage limit is invalid"
            ) from None
        if configured_gib < 0:
            raise StoragePolicyError(
                "configured media storage limit is invalid"
            )
        byte_limit = configured_gib * GIB
    else:
        try:
            byte_limit = int(limit_bytes)
        except (OverflowError, TypeError, ValueError):
            raise StoragePolicyError(
                "media storage byte limit is invalid"
            ) from None
        if byte_limit < 0:
            raise StoragePolicyError(
                "media storage byte limit is invalid"
            )
    selected_action = str(
        action
        if action is not None
        else _setting("storage_full_action", "delete_oldest")
    ).strip()
    if selected_action not in {"delete_oldest", "stop", "alert"}:
        # A corrupted setting must never silently enable deletion.
        selected_action = "stop"
    return StoragePolicyConfig(
        storage_root=root,
        media_roots=tuple(roots),
        read_only_history_roots=tuple(read_only_history),
        invalid_history_roots=tuple(invalid_history),
        limit_bytes=byte_limit,
        action=selected_action,
    )


def configured_policy(**overrides) -> StoragePolicyConfig:
    """Return a validated policy; byte limits may be injected by tests."""

    return _normalise_config(**overrides)


def validate_storage_layout(
    storage_root,
    media_roots: Iterable[os.PathLike | str],
    backup_path,
    history_roots: Iterable[os.PathLike | str] | None = None,
) -> tuple[Path, ...]:
    """Validate proposed UI paths with the same safety rules as enforcement."""

    config = configured_policy(
        storage_root=storage_root,
        media_roots=media_roots,
        limit_bytes=0,
        action="stop",
    )
    backup = _resolve_path(backup_path)
    if not _is_below(backup, config.storage_root):
        raise StoragePolicyError("backup directory must be below storage root")
    quarantine_root = config.storage_root / _QUARANTINE_DIRECTORY
    if _paths_overlap(backup, quarantine_root):
        raise StoragePolicyError("backup directory overlaps quota quarantine")
    for media_root in config.media_roots:
        if _paths_overlap(media_root, backup):
            raise StoragePolicyError(
                "managed media directory overlaps backup directory"
            )
    if history_roots is None:
        history_raw = _setting("media_roots_history", "[]")
        try:
            if len(history_raw.encode("utf-8")) > 1024 * 1024:
                raise ValueError("media roots history is too large")
            payload = json.loads(history_raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise StoragePolicyError("media roots history is invalid") from exc
        if not isinstance(payload, list):
            raise StoragePolicyError("media roots history is invalid")
        history_roots = payload
    elif isinstance(history_roots, (str, bytes)):
        raise StoragePolicyError("media roots history is invalid")
    bounded_history = []
    for raw_history in history_roots:
        bounded_history.append(raw_history)
        if len(bounded_history) > MAX_HISTORY_ROOTS:
            raise StoragePolicyError("media roots history is too large")
    accepted_history: list[Path] = []
    for raw_history in bounded_history:
        value = str(raw_history or "").strip()
        if not value:
            continue
        try:
            history = _resolve_path(value)
        except (OSError, RuntimeError, ValueError) as exc:
            raise StoragePolicyError("media roots history is invalid") from exc
        if history == Path(history.anchor):
            raise StoragePolicyError("media roots history is unsafe")
        if _paths_overlap(history, quarantine_root):
            raise StoragePolicyError(
                "media history overlaps quota quarantine"
            )
        if _paths_overlap(history, backup):
            raise StoragePolicyError(
                "backup directory overlaps media history"
            )
        if history in config.media_roots or history in accepted_history:
            continue
        if any(
            _paths_overlap(history, media_root)
            for media_root in config.media_roots
        ):
            raise StoragePolicyError(
                "media history overlaps managed media directory"
            )
        if any(
            _paths_overlap(history, accepted)
            for accepted in accepted_history
        ):
            raise StoragePolicyError("media history directories overlap")
        accepted_history.append(history)
    return config.media_roots


def _signature(config: StoragePolicyConfig) -> tuple[str, ...]:
    return tuple(
        sorted(
            [f"w:{path}" for path in config.media_roots]
            + [f"r:{path}" for path in config.read_only_history_roots]
        )
    )


def _signature_roots(signature: tuple[str, ...]) -> tuple[Path, ...]:
    return tuple(Path(value[2:]) for value in signature)


def _accounting_roots(config: StoragePolicyConfig) -> tuple[Path, ...]:
    return config.media_roots + config.read_only_history_roots


def _configs_overlap(
    first: StoragePolicyConfig,
    second: StoragePolicyConfig,
) -> bool:
    return any(
        _paths_overlap(first_root, second_root)
        for first_root in _accounting_roots(first)
        for second_root in _accounting_roots(second)
    )


def _config_covers_path(config: StoragePolicyConfig, path: Path) -> bool:
    return any(
        path == root or _is_below(path, root)
        for root in _accounting_roots(config)
    )


def _signature_covers_path(signature: tuple[str, ...], path: Path) -> bool:
    return any(
        path == root or _is_below(path, root)
        for root in _signature_roots(signature)
    )


def _active_reserved_bytes_for_config(
    config: StoragePolicyConfig,
    *,
    exclude: _ReservationState | None = None,
) -> int:
    return sum(
        max(0, state.expected_bytes - state.original_size)
        for state in _RESERVATIONS.values()
        if state is not exclude and _config_covers_path(config, state.target)
    )


def _protected_for_config(config: StoragePolicyConfig) -> set[Path]:
    protected: set[Path] = set()
    for state in _RESERVATIONS.values():
        for path in (state.target, *state.protected_paths):
            if any(_is_below(path, root) for root in config.media_roots):
                protected.add(path)
    for paths in _PIN_LEASES.values():
        for path in paths:
            if any(
                path == root or _is_below(path, root)
                for root in _accounting_roots(config)
            ):
                protected.add(path)
    return protected


def _path_is_protected(path: Path, protected_paths: set[Path]) -> bool:
    return any(
        path == protected or path.is_relative_to(protected)
        for protected in protected_paths
    )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        str(directory),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_parent_directory(path: os.PathLike | str) -> None:
    """Durably publish a file entry before committing quota replacement."""

    _fsync_directory(Path(path).parent)


def _quarantine_root(config: StoragePolicyConfig, *, create: bool) -> Path:
    root = config.storage_root / _QUARANTINE_DIRECTORY
    try:
        root_stat = root.lstat()
    except FileNotFoundError:
        if not create:
            return root
        root.mkdir(mode=0o700)
        _fsync_directory(config.storage_root)
        root_stat = root.lstat()
    except OSError as exc:
        raise StoragePolicyError("quota quarantine is unavailable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise StoragePolicyError("quota quarantine path is unsafe")
    if not _is_below(root.resolve(strict=True), config.storage_root):
        raise StoragePolicyError("quota quarantine escapes storage root")
    return root


def _atomic_journal_write(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            try:
                os.chmod(temporary, 0o600)
            except OSError:
                pass
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _quarantine_paths(
    config: StoragePolicyConfig,
    transaction_id: str,
    *,
    create: bool,
) -> tuple[Path, Path, Path]:
    root = _quarantine_root(config, create=create)
    transaction = root / transaction_id
    files = transaction / "files"
    journal = transaction / _QUARANTINE_JOURNAL
    if create and not transaction.exists():
        transaction.mkdir(mode=0o700)
        files.mkdir(mode=0o700)
        _fsync_directory(root)
    for directory in (transaction, files):
        if not directory.exists():
            continue
        directory_stat = directory.lstat()
        if (
            stat.S_ISLNK(directory_stat.st_mode)
            or not stat.S_ISDIR(directory_stat.st_mode)
            or not _is_below(directory.resolve(strict=True), root)
        ):
            raise StoragePolicyError("quota transaction path is unsafe")
    return transaction, files, journal


def _journal_payload(state: _ReservationState, status: str) -> dict:
    return {
        "version": _QUARANTINE_VERSION,
        "status": status,
        "transaction_id": state.quarantine_id,
        "target": str(state.target),
        "expected_bytes": state.expected_bytes,
        "original_size": state.original_size,
        "target_existed": state.target_existed,
        "original_file_identity": (
            {
                "device": state.original_file_identity[0],
                "inode": state.original_file_identity[1],
            }
            if state.original_file_identity is not None
            else None
        ),
        "acceptance_id": state.acceptance_id or "",
        "protected_paths": [
            {"path": str(path), "existed": existed}
            for path, existed in zip(
                state.protected_paths,
                state.protected_paths_existed,
                strict=True,
            )
        ],
        "owned_file_identities": [
            {"device": device, "inode": inode}
            for device, inode in sorted(state.owned_file_identities)
        ],
        "moves": [
            {
                "original": str(move.original),
                "quarantine_name": move.quarantine_name,
                "size": move.size,
                "mtime_ns": move.mtime_ns,
                "device": move.device,
                "inode": move.inode,
                "nlink": move.nlink,
            }
            for move in (state.quarantine_moves or [])
        ],
    }


def _ensure_quarantine_transaction(state: _ReservationState) -> tuple[Path, Path]:
    if state.quarantine_id is None:
        state.quarantine_id = uuid4().hex
        state.quarantine_moves = []
    _, files, journal = _quarantine_paths(
        state.config,
        state.quarantine_id,
        create=True,
    )
    if not journal.exists():
        _atomic_journal_write(journal, _journal_payload(state, "pending"))
    return files, journal


def _quarantine_record_locked(
    state: _ReservationState,
    record: _FileRecord,
    inventory: _Inventory,
) -> bool:
    if not _record_is_safe(record, state.config):
        return False
    files, journal = _ensure_quarantine_transaction(state)
    quarantine_name = f"{len(state.quarantine_moves or []):04d}-{uuid4().hex}"
    move = _QuarantineMove(
        original=record.path,
        quarantine_name=quarantine_name,
        size=record.size,
        mtime_ns=record.mtime_ns,
        device=record.device,
        inode=record.inode,
        nlink=record.nlink,
    )
    assert state.quarantine_moves is not None
    state.quarantine_moves.append(move)
    # The durable plan is published before the rename. On a crash, recovery
    # can distinguish both "planned only" and "already quarantined" states.
    _atomic_journal_write(journal, _journal_payload(state, "pending"))
    quarantine_path = files / quarantine_name
    try:
        os.replace(record.path, quarantine_path)
    except OSError:
        state.quarantine_moves.pop()
        _atomic_journal_write(journal, _journal_payload(state, "pending"))
        return False
    # Once rename succeeds the durable journal entry must never be removed.
    # A directory-fsync failure leaves a recoverable pending transaction.
    try:
        _fsync_directory(record.path.parent)
        _fsync_directory(files)
    except OSError:
        pass
    _account_record_removed(record, inventory)
    return True


def _move_record(move: _QuarantineMove) -> _FileRecord:
    return _FileRecord(
        path=move.original,
        size=move.size,
        mtime_ns=move.mtime_ns,
        device=move.device,
        inode=move.inode,
        nlink=move.nlink,
        deletable=True,
    )


def _safe_quarantine_file(path: Path, move: _QuarantineMove) -> bool:
    try:
        current = path.lstat()
    except OSError:
        return False
    return bool(
        stat.S_ISREG(current.st_mode)
        and int(current.st_dev) == move.device
        and int(current.st_ino) == move.inode
    )


def _cleanup_transaction_directories(
    transaction: Path,
    files: Path,
    journal: Path,
) -> bool:
    try:
        if files.exists():
            files.rmdir()
        journal.unlink(missing_ok=True)
        transaction.rmdir()
        root = transaction.parent
        try:
            root.rmdir()
        except OSError:
            pass
        return True
    except OSError:
        return False


def _finish_quarantine_locked(
    state: _ReservationState,
    *,
    commit: bool,
) -> bool:
    if state.quarantine_id is None:
        return True
    transaction, files, journal = _quarantine_paths(
        state.config,
        state.quarantine_id,
        create=False,
    )
    moves = state.quarantine_moves or []
    if commit:
        try:
            _atomic_journal_write(journal, _journal_payload(state, "committed"))
        except OSError as exc:
            # The caller must not accept/persist the target unless the commit
            # marker is durable. A pending marker authorizes crash recovery to
            # remove that target.
            raise _StorageCommitMarkerError(
                "media write commit marker could not be persisted"
            ) from exc
        complete = True
        for move in moves:
            quarantine_path = files / move.quarantine_name
            if not quarantine_path.exists():
                continue
            if not _safe_quarantine_file(quarantine_path, move):
                complete = False
                continue
            try:
                quarantine_path.unlink()
            except OSError:
                complete = False
        if complete:
            try:
                _fsync_directory(files)
            except OSError:
                return False
            return _cleanup_transaction_directories(
                transaction,
                files,
                journal,
            )
        return False

    # A pending reservation never owns a newly-created target. Remove it
    # before restoring evicted evidence so a crash cannot leave both the
    # orphan and the restored files over quota. Existing targets are not
    # recoverable without a content backup and are therefore left untouched.
    owned_identities = state.owned_file_identities
    for protected, existed in zip(
        state.protected_paths,
        state.protected_paths_existed,
        strict=True,
    ):
        if existed:
            continue
        try:
            protected = _safe_target(protected, state.config)
            try:
                protected_stat = protected.lstat()
            except FileNotFoundError:
                protected_stat = None
            if protected_stat is not None:
                if (
                    not stat.S_ISREG(protected_stat.st_mode)
                    or int(protected_stat.st_nlink) != 1
                ):
                    return False
                identity = (
                    int(protected_stat.st_dev),
                    int(protected_stat.st_ino),
                )
                if identity in owned_identities:
                    protected.unlink()
                    _fsync_directory(protected.parent)
        except (OSError, StoragePolicyError):
            return False

    if not state.target_existed:
        try:
            target = _safe_target(state.target, state.config)
            try:
                target_stat = target.lstat()
            except FileNotFoundError:
                target_stat = None
            if target_stat is not None:
                if (
                    not stat.S_ISREG(target_stat.st_mode)
                    or int(target_stat.st_nlink) != 1
                ):
                    return False
                identity = (
                    int(target_stat.st_dev),
                    int(target_stat.st_ino),
                )
                if identity in owned_identities:
                    target.unlink()
                    _fsync_directory(target.parent)
        except (OSError, StoragePolicyError):
            return False

    complete = True
    for move in reversed(moves):
        quarantine_path = files / move.quarantine_name
        if not quarantine_path.exists():
            continue
        if not _safe_quarantine_file(quarantine_path, move):
            complete = False
            continue
        try:
            restored = _safe_target(move.original, state.config)
            if restored.exists():
                complete = False
                continue
            restored.parent.mkdir(parents=True, exist_ok=True)
            os.replace(quarantine_path, restored)
            _account_record_restored(_move_record(move))
            try:
                _fsync_directory(restored.parent)
                _fsync_directory(files)
            except OSError:
                complete = False
        except (OSError, StoragePolicyError):
            complete = False
    if complete:
        return _cleanup_transaction_directories(
            transaction,
            files,
            journal,
        )
    return False


def _parse_journal_move(item: object, config: StoragePolicyConfig) -> _QuarantineMove:
    if not isinstance(item, dict):
        raise StoragePolicyError("quota journal move is invalid")
    quarantine_name = str(item.get("quarantine_name", ""))
    if (
        not quarantine_name
        or Path(quarantine_name).name != quarantine_name
        or quarantine_name in {".", ".."}
    ):
        raise StoragePolicyError("quota journal filename is invalid")
    original = _safe_target(Path(str(item.get("original", ""))), config)
    move = _QuarantineMove(
        original=original,
        quarantine_name=quarantine_name,
        size=max(0, int(item.get("size", 0))),
        mtime_ns=int(item.get("mtime_ns", 0)),
        device=int(item.get("device", -1)),
        inode=int(item.get("inode", -1)),
        nlink=int(item.get("nlink", 0)),
    )
    if move.device < 0 or move.inode < 0 or move.nlink != 1:
        raise StoragePolicyError("quota journal identity is invalid")
    return move


def _is_journal_temp_name(name: str) -> bool:
    prefix = f".{_QUARANTINE_JOURNAL}."
    suffix = ".tmp"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    token = name[len(prefix) : -len(suffix)]
    return len(token) == 32 and all(
        character in "0123456789abcdef" for character in token
    )


def _clean_journal_temps(
    transaction: Path,
    files: Path,
    journal: Path,
) -> tuple[bool, bool]:
    """Remove only validated regular atomic-journal leftovers.

    Returns ``(safe, transaction_removed)``. With no published journal, a
    transaction is removable only when its quarantine files directory is
    empty; a rename cannot precede the first durable journal publication.
    """

    try:
        with os.scandir(transaction) as iterator:
            entries = list(iterator)
    except OSError:
        return False, False
    temporary_paths: list[Path] = []
    unexpected = []
    for entry in entries:
        if entry.name in {"files", _QUARANTINE_JOURNAL}:
            continue
        if not _is_journal_temp_name(entry.name):
            unexpected.append(entry.name)
            continue
        try:
            entry_stat = entry.stat(follow_symlinks=False)
        except OSError:
            return False, False
        if not stat.S_ISREG(entry_stat.st_mode) or int(entry_stat.st_nlink) != 1:
            return False, False
        temporary_paths.append(Path(entry.path))
    if unexpected:
        return False, False
    if journal.exists():
        try:
            for temporary in temporary_paths:
                temporary.unlink()
            if temporary_paths:
                _fsync_directory(transaction)
            return True, False
        except OSError:
            return False, False

    try:
        if files.exists():
            with os.scandir(files) as iterator:
                if next(iterator, None) is not None:
                    return False, False
        for temporary in temporary_paths:
            temporary.unlink()
        if files.exists():
            files.rmdir()
        transaction.rmdir()
        root = transaction.parent
        try:
            root.rmdir()
        except OSError:
            pass
        return True, True
    except OSError:
        return False, False


def _recover_stale_quarantine_locked(
    config: StoragePolicyConfig,
    *,
    max_transactions: int = 64,
    max_moves: int = DEFAULT_MAX_DELETE_FILES,
) -> bool:
    root = _quarantine_root(config, create=False)
    if not root.exists():
        return True
    active_ids = {
        state.quarantine_id
        for state in _RESERVATIONS.values()
        if state.quarantine_id
    }
    try:
        with os.scandir(root) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError:
        return False
    complete = True
    processed = 0
    for entry in entries:
        if entry.name in active_ids:
            continue
        if processed >= max(1, int(max_transactions)):
            complete = False
            break
        processed += 1
        try:
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                complete = False
                continue
            transaction_id = entry.name
            if len(transaction_id) != 32 or any(
                character not in "0123456789abcdef"
                for character in transaction_id
            ):
                complete = False
                continue
            transaction, files, journal = _quarantine_paths(
                config,
                transaction_id,
                create=False,
            )
            temps_safe, transaction_removed = _clean_journal_temps(
                transaction,
                files,
                journal,
            )
            if not temps_safe:
                complete = False
                continue
            if transaction_removed:
                continue
            journal_stat = journal.lstat()
            if (
                not stat.S_ISREG(journal_stat.st_mode)
                or journal_stat.st_size > 1024 * 1024
            ):
                complete = False
                continue
            payload = json.loads(journal.read_text(encoding="utf-8"))
            journal_version = (
                payload.get("version")
                if isinstance(payload, dict)
                else None
            )
            if (
                not isinstance(payload, dict)
                or journal_version not in _SUPPORTED_QUARANTINE_VERSIONS
                or payload.get("transaction_id") != transaction_id
                or payload.get("status") not in {"pending", "committed"}
                or not isinstance(payload.get("moves"), list)
                or len(payload["moves"]) > max(1, int(max_moves))
            ):
                complete = False
                continue
            moves = [
                _parse_journal_move(item, config)
                for item in payload["moves"]
            ]
            if journal_version in {2, 3}:
                target = _safe_target(
                    Path(str(payload.get("target", ""))),
                    config,
                )
                original_size = max(0, int(payload.get("original_size", 0)))
                if journal_version == 3:
                    expected_bytes = int(payload.get("expected_bytes", -1))
                    if expected_bytes < 0:
                        raise StoragePolicyError(
                            "quota journal expected size is invalid"
                        )
                else:
                    expected_bytes = 0
                target_existed = payload.get("target_existed")
                if not isinstance(target_existed, bool):
                    raise StoragePolicyError(
                        "quota journal target state is invalid"
                    )
                original_identity_payload = payload.get(
                    "original_file_identity"
                )
                if journal_version == 3:
                    if original_identity_payload is None:
                        original_file_identity = None
                    elif isinstance(original_identity_payload, dict):
                        original_file_identity = (
                            int(original_identity_payload.get("device", -1)),
                            int(original_identity_payload.get("inode", -1)),
                        )
                        if (
                            original_file_identity[0] < 0
                            or original_file_identity[1] < 0
                        ):
                            raise StoragePolicyError(
                                "quota journal original identity is invalid"
                            )
                    else:
                        raise StoragePolicyError(
                            "quota journal original identity is invalid"
                        )
                    acceptance_value = str(
                        payload.get("acceptance_id", "") or ""
                    ).strip().lower()
                    if acceptance_value and (
                        len(acceptance_value) != 32
                        or any(
                            character not in "0123456789abcdef"
                            for character in acceptance_value
                        )
                    ):
                        raise StoragePolicyError(
                            "quota journal acceptance id is invalid"
                        )
                    acceptance_id = acceptance_value or None
                else:
                    original_file_identity = None
                    acceptance_id = None
                protected_payload = payload.get("protected_paths")
                if (
                    not isinstance(protected_payload, list)
                    or len(protected_payload) > MAX_PROTECTED_WRITE_PATHS
                ):
                    raise StoragePolicyError(
                        "quota journal protected paths are invalid"
                    )
                protected_paths = []
                protected_paths_existed = []
                for item in protected_payload:
                    if not isinstance(item, dict) or not isinstance(
                        item.get("existed"),
                        bool,
                    ):
                        raise StoragePolicyError(
                            "quota journal protected path is invalid"
                        )
                    protected_paths.append(
                        _safe_target(Path(str(item.get("path", ""))), config)
                    )
                    protected_paths_existed.append(bool(item["existed"]))
                identities_payload = payload.get("owned_file_identities")
                if (
                    not isinstance(identities_payload, list)
                    or len(identities_payload) > MAX_PROTECTED_WRITE_PATHS + 1
                ):
                    raise StoragePolicyError(
                        "quota journal owned files are invalid"
                    )
                owned_file_identities = set()
                for item in identities_payload:
                    if not isinstance(item, dict):
                        raise StoragePolicyError(
                            "quota journal owned file is invalid"
                        )
                    identity = (
                        int(item.get("device", -1)),
                        int(item.get("inode", -1)),
                    )
                    if identity[0] < 0 or identity[1] < 0:
                        raise StoragePolicyError(
                            "quota journal owned file identity is invalid"
                        )
                    owned_file_identities.add(identity)
            else:
                # Version 1 did not track the write target. Preserve backwards
                # recovery semantics while still restoring its moved files.
                target = config.media_roots[0] / ".recovery-placeholder"
                expected_bytes = 0
                original_size = 1
                target_existed = True
                original_file_identity = None
                protected_paths = []
                protected_paths_existed = []
                owned_file_identities = set()
                acceptance_id = None
            if payload["status"] == "pending":
                missing = any(
                    not (files / move.quarantine_name).exists()
                    and not move.original.exists()
                    for move in moves
                )
                if missing:
                    complete = False
                    continue
            state = _ReservationState(
                signature=_signature(config),
                target=target,
                protected_paths=tuple(protected_paths),
                protected_paths_existed=tuple(protected_paths_existed),
                expected_bytes=expected_bytes,
                original_size=original_size,
                target_existed=target_existed,
                original_file_identity=original_file_identity,
                owned_file_identities=owned_file_identities,
                config=config,
                acceptance_id=acceptance_id,
                quarantine_id=transaction_id,
                quarantine_moves=moves,
            )
            commit = payload["status"] == "committed"
            if not commit and state.acceptance_id is not None:
                # SQLite accepted the exact inode in the same FULL-synchronous
                # transaction as its owner. Complete the filesystem side of
                # that transaction instead of rolling back a referenced file.
                commit = _accepted_intent_matches(state)
            if not _finish_quarantine_locked(state, commit=commit):
                complete = False
            elif state.acceptance_id is not None:
                try:
                    from app.media_acceptance import discard_intent

                    discard_intent(state.acceptance_id)
                except Exception:
                    # A stale oracle row is harmless and can be retried later;
                    # the journal decision above is already durable.
                    pass
        except (
            OSError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            sqlite3.Error,
            StoragePolicyError,
        ):
            complete = False
    return complete


def _scan_inventory(
    config: StoragePolicyConfig,
    *,
    max_scan_entries: int,
    max_delete_files: int,
) -> _Inventory:
    managed_bytes = 0
    scanned_files = 0
    scanned_entries = 0
    complete = not bool(config.invalid_history_roots)
    oldest_heap: list[tuple[int, int, _FileRecord]] = []
    sequence = 0
    seen_files: set[tuple[int, int]] = set()
    active_owned_identities = {
        identity
        for state in _RESERVATIONS.values()
        for identity in state.owned_file_identities
    }

    writable_roots = set(config.media_roots)
    for media_root in _accounting_roots(config):
        deletable = media_root in writable_roots
        if not media_root.exists():
            if not deletable:
                # A historical volume may be temporarily unmounted. Treating
                # it as zero would admit writes that violate quota on remount.
                complete = False
            continue
        try:
            root_stat = media_root.lstat()
        except OSError:
            complete = False
            continue
        if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
            complete = False
            continue
        stack = [media_root]
        while stack:
            directory = stack.pop()
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        scanned_entries += 1
                        if scanned_entries > max_scan_entries:
                            complete = False
                            stack.clear()
                            break
                        try:
                            if entry.is_symlink():
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(Path(entry.path))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            file_stat = entry.stat(follow_symlinks=False)
                        except OSError:
                            complete = False
                            continue
                        identity = (int(file_stat.st_dev), int(file_stat.st_ino))
                        if identity in seen_files:
                            continue
                        seen_files.add(identity)
                        # Reservations account their full expected delta
                        # separately. Excluding the currently-written inode
                        # prevents a cache refresh from counting partial bytes
                        # once in inventory and again as reserved capacity.
                        if identity in active_owned_identities:
                            continue
                        record = _FileRecord(
                            path=Path(entry.path).resolve(strict=False),
                            size=max(0, int(file_stat.st_size)),
                            mtime_ns=int(file_stat.st_mtime_ns),
                            device=identity[0],
                            inode=identity[1],
                            nlink=max(1, int(file_stat.st_nlink)),
                            deletable=deletable,
                        )
                        managed_bytes += record.size
                        scanned_files += 1
                        if (
                            max_delete_files <= 0
                            or not record.deletable
                            or record.nlink != 1
                        ):
                            continue
                        sequence += 1
                        item = (-record.mtime_ns, sequence, record)
                        if len(oldest_heap) < max_delete_files:
                            heapq.heappush(oldest_heap, item)
                        elif record.mtime_ns < -oldest_heap[0][0]:
                            heapq.heapreplace(oldest_heap, item)
            except OSError:
                complete = False
    oldest = sorted(
        (item[2] for item in oldest_heap),
        key=lambda record: (record.mtime_ns, str(record.path)),
    )
    return _Inventory(
        managed_bytes=managed_bytes,
        scanned_files=scanned_files,
        complete=complete,
        oldest=oldest,
        expires_at=time.monotonic() + _CACHE_SECONDS,
    )


def _inventory(
    config: StoragePolicyConfig,
    *,
    force: bool = False,
    max_scan_entries: int = DEFAULT_MAX_SCAN_ENTRIES,
    max_delete_files: int = DEFAULT_MAX_DELETE_FILES,
) -> _Inventory:
    signature = _signature(config)
    recovery_complete = _recover_stale_quarantine_locked(config)
    cached = _CACHE.get(signature)
    active = any(
        _config_covers_path(config, state.target)
        for state in _RESERVATIONS.values()
    )
    if cached is not None and (
        active or (not force and cached.expires_at > time.monotonic())
    ):
        cached.complete = cached.complete and recovery_complete
        return cached
    current = _scan_inventory(
        config,
        max_scan_entries=max(1, int(max_scan_entries)),
        max_delete_files=max(0, int(max_delete_files)),
    )
    current.complete = current.complete and recovery_complete
    _CACHE[signature] = current
    return current


def _record_is_safe(record: _FileRecord, config: StoragePolicyConfig) -> bool:
    if not record.deletable or record.nlink != 1:
        return False
    try:
        current = record.path.lstat()
    except OSError:
        return False
    if not stat.S_ISREG(current.st_mode):
        return False
    if (int(current.st_dev), int(current.st_ino)) != (
        record.device,
        record.inode,
    ):
        return False
    if int(current.st_nlink) != 1:
        return False
    resolved = record.path.resolve(strict=True)
    return any(_is_below(resolved, root) for root in config.media_roots)


def _cache_accounts_for(signature: tuple[str, ...], path: Path) -> bool:
    return any(
        path == root or _is_below(path, root)
        for root in _signature_roots(signature)
    )


def _account_record_removed(
    record: _FileRecord,
    primary_inventory: _Inventory,
) -> None:
    primary_inventory.managed_bytes = max(
        0,
        primary_inventory.managed_bytes - record.size,
    )
    for signature, cached in _CACHE.items():
        if cached is primary_inventory:
            continue
        if _cache_accounts_for(signature, record.path):
            cached.managed_bytes = max(0, cached.managed_bytes - record.size)
            cached.oldest = [
                item for item in cached.oldest if item.path != record.path
            ]


def _account_record_restored(record: _FileRecord) -> None:
    for signature, cached in _CACHE.items():
        if not _cache_accounts_for(signature, record.path):
            continue
        cached.managed_bytes += record.size
        if (
            record.deletable
            and record.nlink == 1
            and all(item.path != record.path for item in cached.oldest)
        ):
            cached.oldest.append(record)
            cached.oldest.sort(
                key=lambda item: (item.mtime_ns, str(item.path))
            )
            del cached.oldest[DEFAULT_MAX_DELETE_FILES:]


def _delete_oldest_locked(
    config: StoragePolicyConfig,
    inventory: _Inventory,
    *,
    bytes_needed: int,
    protected_paths: set[Path],
    cutoff_ns: int | None = None,
    max_delete_files: int = DEFAULT_MAX_DELETE_FILES,
) -> tuple[int, int]:
    removed_files = 0
    removed_bytes = 0
    for record in inventory.oldest:
        if removed_files >= max(0, int(max_delete_files)):
            break
        if cutoff_ns is not None and record.mtime_ns >= cutoff_ns:
            break
        if _path_is_protected(record.path, protected_paths):
            continue
        if not _record_is_safe(record, config):
            continue
        try:
            record.path.unlink()
        except OSError:
            continue
        removed_files += 1
        removed_bytes += record.size
        _account_record_removed(record, inventory)
        if cutoff_ns is None and removed_bytes >= max(0, bytes_needed):
            break
    inventory.oldest = [
        record for record in inventory.oldest if record.path.exists()
    ]
    inventory.expires_at = time.monotonic() + _CACHE_SECONDS
    return removed_files, removed_bytes


def _quarantine_oldest_locked(
    state: _ReservationState,
    inventory: _Inventory,
    *,
    bytes_needed: int,
    protected_paths: set[Path],
    max_files: int = DEFAULT_MAX_DELETE_FILES,
) -> tuple[int, int]:
    moved_files = 0
    moved_bytes = 0
    already_moved = len(state.quarantine_moves or [])
    remaining = max(0, int(max_files) - already_moved)
    for record in list(inventory.oldest):
        if moved_files >= remaining or moved_bytes >= max(0, bytes_needed):
            break
        if _path_is_protected(record.path, protected_paths):
            continue
        if not _quarantine_record_locked(state, record, inventory):
            continue
        moved_files += 1
        moved_bytes += record.size
    inventory.oldest = [
        record for record in inventory.oldest if record.path.exists()
    ]
    inventory.expires_at = time.monotonic() + _CACHE_SECONDS
    return moved_files, moved_bytes


def _status_from_inventory(
    config: StoragePolicyConfig,
    inventory: _Inventory,
    *,
    deleted_files: int = 0,
    deleted_bytes: int = 0,
    error: str = "",
) -> StorageStatus:
    reserved = _active_reserved_bytes_for_config(config)
    effective = inventory.managed_bytes + reserved
    over_limit = bool(config.limit_bytes and effective > config.limit_bytes)
    blocked = bool(
        config.limit_bytes
        and config.action in {"stop", "delete_oldest"}
        and (effective >= config.limit_bytes or not inventory.complete)
    )
    return StorageStatus(
        storage_root=str(config.storage_root),
        media_roots=tuple(str(path) for path in config.media_roots),
        action=config.action,
        limit_bytes=config.limit_bytes,
        managed_bytes=inventory.managed_bytes,
        reserved_bytes=reserved,
        over_limit=over_limit,
        write_blocked=blocked,
        usage_complete=inventory.complete,
        scanned_files=inventory.scanned_files,
        read_only_history_roots=tuple(
            str(path) for path in config.read_only_history_roots
        ),
        invalid_history_roots=config.invalid_history_roots,
        deleted_files=deleted_files,
        deleted_bytes=deleted_bytes,
        error=error,
    )


def storage_status(
    *,
    force: bool = False,
    max_scan_entries: int = DEFAULT_MAX_SCAN_ENTRIES,
    max_delete_files: int = DEFAULT_MAX_DELETE_FILES,
    **policy_overrides,
) -> StorageStatus:
    """Return managed-media usage without deleting any data."""

    with _LOCK:
        config = configured_policy(**policy_overrides)
        inventory = _inventory(
            config,
            force=force,
            max_scan_entries=max_scan_entries,
            max_delete_files=max_delete_files,
        )
        return _status_from_inventory(config, inventory)


def enforce_storage_limit(
    *,
    max_scan_entries: int = DEFAULT_MAX_SCAN_ENTRIES,
    max_delete_files: int = DEFAULT_MAX_DELETE_FILES,
    **policy_overrides,
) -> StorageStatus:
    """Apply the configured full action once, with bounded deletions."""

    with _LOCK:
        config = configured_policy(**policy_overrides)
        inventory = _inventory(
            config,
            force=True,
            max_scan_entries=max_scan_entries,
            max_delete_files=max_delete_files,
        )
        deleted_files = 0
        deleted_bytes = 0
        error = ""
        if (
            config.limit_bytes
            and config.action == "delete_oldest"
            and inventory.complete
            and inventory.managed_bytes > config.limit_bytes
        ):
            deleted_files, deleted_bytes = _delete_oldest_locked(
                config,
                inventory,
                bytes_needed=inventory.managed_bytes - config.limit_bytes,
                protected_paths=_protected_for_config(config),
                max_delete_files=max_delete_files,
            )
            if inventory.managed_bytes > config.limit_bytes:
                error = "bounded cleanup could not reduce usage below limit"
        elif not inventory.complete:
            error = "managed media scan was incomplete"
        return _status_from_inventory(
            config,
            inventory,
            deleted_files=deleted_files,
            deleted_bytes=deleted_bytes,
            error=error,
        )


def delete_older_than(
    folder,
    cutoff_timestamp: float,
    *,
    storage_root,
    max_scan_entries: int = DEFAULT_MAX_SCAN_ENTRIES,
    max_delete_files: int = DEFAULT_MAX_DELETE_FILES,
) -> int:
    """Delete a bounded oldest-first batch below one validated media root."""

    with _LOCK:
        config = configured_policy(
            storage_root=storage_root,
            media_roots=(folder,),
            limit_bytes=0,
            action="delete_oldest",
        )
        inventory = _inventory(
            config,
            force=True,
            max_scan_entries=max_scan_entries,
            max_delete_files=max_delete_files,
        )
        if not inventory.complete:
            return 0
        removed, _ = _delete_oldest_locked(
            config,
            inventory,
            bytes_needed=0,
            protected_paths=_protected_for_config(config),
            cutoff_ns=int(float(cutoff_timestamp) * 1_000_000_000),
            max_delete_files=max_delete_files,
        )
        if not _RESERVATIONS:
            _CACHE.clear()
        return removed


def _safe_managed_path(
    target: Path,
    config: StoragePolicyConfig,
    *,
    allow_read_only: bool,
    allow_root: bool,
) -> Path:
    lexical = Path(os.path.abspath(os.fspath(target.expanduser())))
    allowed_roots = (
        _accounting_roots(config)
        if allow_read_only
        else config.media_roots
    )
    matching_root = next(
        (
            root
            for root in allowed_roots
            if (allow_root and lexical == root) or _is_below(lexical, root)
        ),
        None,
    )
    if matching_root is None:
        raise StorageWriteRejected(
            "media target is outside configured managed directories"
        )
    current = matching_root
    for component in lexical.relative_to(matching_root).parts:
        current = current / component
        try:
            current_stat = current.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise StorageWriteRejected(
                "media target path could not be validated"
            ) from exc
        if stat.S_ISLNK(current_stat.st_mode):
            raise StorageWriteRejected("media target path contains a symlink")
    resolved = lexical.resolve(strict=False)
    if not any(
        (allow_root and resolved == root) or _is_below(resolved, root)
        for root in allowed_roots
    ):
        raise StorageWriteRejected(
            "media target resolves outside configured managed directories"
        )
    return resolved


def _safe_target(target: Path, config: StoragePolicyConfig) -> Path:
    return _safe_managed_path(
        target,
        config,
        allow_read_only=False,
        allow_root=False,
    )


class MediaPinLease:
    """Read lease preventing quota/retention deletion of paths/subtrees."""

    def __init__(self, token: str | None):
        self._token = token
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._token is None:
            return
        with _LOCK:
            _PIN_LEASES.pop(self._token, None)

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        self.close()
        return False


def pin_media_paths(
    paths: Iterable[os.PathLike | str],
    **policy_overrides,
) -> MediaPinLease:
    """Pin managed files/directories while they are streamed or processed."""

    with _LOCK:
        config = configured_policy(**policy_overrides)
        resolved: list[Path] = []
        for path in paths:
            candidate = _safe_managed_path(
                Path(path),
                config,
                allow_read_only=True,
                allow_root=True,
            )
            if candidate not in resolved:
                resolved.append(candidate)
        if not resolved:
            return MediaPinLease(None)
        token = uuid4().hex
        _PIN_LEASES[token] = tuple(resolved)
        return MediaPinLease(token)


def _validated_committable_target(
    state: _ReservationState,
    actual_bytes: int | None,
) -> tuple[tuple[int, int], int]:
    """Validate exact ownership and size before making eviction permanent."""

    target = _safe_target(state.target, state.config)
    try:
        details = target.lstat()
    except OSError as exc:
        raise StorageWriteRejected(
            "media write target is missing before commit"
        ) from exc
    if not stat.S_ISREG(details.st_mode) or int(details.st_nlink) != 1:
        raise StorageWriteRejected(
            "media write target is not a private regular file"
        )
    identity = (int(details.st_dev), int(details.st_ino))
    if state.target_existed:
        owned = identity == state.original_file_identity
    else:
        owned = identity in state.owned_file_identities
    if not owned:
        raise StorageWriteRejected(
            "media write target ownership changed before commit"
        )
    size_bytes = max(0, int(details.st_size))
    if actual_bytes is not None and size_bytes != max(0, int(actual_bytes)):
        raise StorageWriteRejected(
            "media write target size changed before commit"
        )
    if size_bytes > state.expected_bytes:
        raise StorageWriteRejected(
            "media write exceeded its reserved byte count"
        )
    return identity, size_bytes


def _accepted_intent_matches(
    state: _ReservationState,
    *,
    actual_bytes: int | None = None,
) -> bool:
    """Resolve a pending journal using the SQLite commit oracle."""

    if state.acceptance_id is None:
        return False
    from app.media_acceptance import load_intent

    intent = load_intent(state.acceptance_id)
    if intent is None or intent.get("state") == "pending":
        return False
    if intent.get("state") != "accepted":
        raise StoragePolicyError("media acceptance state is invalid")
    identity, size_bytes = _validated_committable_target(
        state,
        actual_bytes,
    )
    try:
        intent_identity = (
            int(intent.get("device")),
            int(intent.get("inode")),
        )
        intent_size = int(intent.get("size_bytes"))
    except (TypeError, ValueError) as exc:
        raise StoragePolicyError(
            "accepted media identity is invalid"
        ) from exc
    if (
        Path(str(intent.get("target_path", ""))).resolve(strict=False)
        != state.target
        or intent_identity != identity
        or intent_size != size_bytes
        or not str(intent.get("owner_kind") or "").strip()
        or not str(intent.get("owner_id") or "").strip()
    ):
        raise StoragePolicyError(
            "accepted media does not match the pending journal"
        )
    return True


class MediaWriteReservation:
    """A quota reservation that keeps concurrent writes within the limit."""

    def __init__(self, token: str | None):
        self._token = token
        self._closed = False

    def grow(self, expected_total_bytes: int) -> None:
        if self._closed or self._token is None:
            return
        with _LOCK:
            state = _RESERVATIONS.get(self._token)
            if state is None:
                return
            desired = max(0, int(expected_total_bytes))
            if desired <= state.expected_bytes:
                return
            _enforce_reservation_locked(state, desired)
            previous = state.expected_bytes
            state.expected_bytes = desired
            try:
                _, journal = _ensure_quarantine_transaction(state)
                _atomic_journal_write(
                    journal,
                    _journal_payload(state, "pending"),
                )
            except BaseException:
                state.expected_bytes = previous
                raise

    def claim_created_path(self, path) -> None:
        """Durably bind this reservation to a newly-created regular file."""

        if self._closed or self._token is None:
            raise StorageWriteRejected("media write reservation is closed")
        with _LOCK:
            state = _RESERVATIONS.get(self._token)
            if state is None:
                raise StorageWriteRejected("media write reservation is missing")
            candidate = _safe_target(Path(path), state.config)
            allowed = (state.target, *state.protected_paths)
            if candidate not in allowed:
                raise StorageWriteRejected(
                    "claimed media path is outside this reservation"
                )
            if candidate == state.target:
                existed = state.target_existed
            else:
                existed = state.protected_paths_existed[
                    state.protected_paths.index(candidate)
                ]
            if existed:
                raise StorageWriteRejected(
                    "pre-existing media path cannot be claimed"
                )
            try:
                claimed = candidate.lstat()
            except OSError as exc:
                raise StorageWriteRejected(
                    "claimed media file is unavailable"
                ) from exc
            if (
                not stat.S_ISREG(claimed.st_mode)
                or int(claimed.st_nlink) != 1
            ):
                raise StorageWriteRejected(
                    "claimed media path is not a private regular file"
                )
            identity = (int(claimed.st_dev), int(claimed.st_ino))
            if identity in state.owned_file_identities:
                return
            state.owned_file_identities.add(identity)
            try:
                _ensure_quarantine_transaction(state)
                _, _, journal = _quarantine_paths(
                    state.config,
                    state.quarantine_id,
                    create=False,
                )
                _atomic_journal_write(
                    journal,
                    _journal_payload(state, "pending"),
                )
            except Exception:
                state.owned_file_identities.discard(identity)
                raise

    def close(self, *, success: bool, actual_bytes: int | None = None) -> None:
        if self._closed:
            return
        if self._token is None:
            self._closed = True
            return
        with _LOCK:
            state = _RESERVATIONS.get(self._token)
            if state is None:
                self._closed = True
                return

            commit = bool(success)
            if state.acceptance_id:
                accepted = _accepted_intent_matches(
                    state,
                    actual_bytes=actual_bytes,
                )
                if commit and not accepted:
                    raise StorageWriteRejected(
                        "media database owner is not durably committed"
                    )
                # SQLite is the durable owner oracle. A caller can reach its
                # compensating close(False) path after an ambiguous database
                # commit; once the exact inode was accepted, rolling the
                # filesystem transaction back would delete DB-owned media.
                if accepted:
                    commit = True
            elif commit:
                _validated_committable_target(state, actual_bytes)

            def consume(*, account_success: bool) -> None:
                self._closed = True
                _RESERVATIONS.pop(self._token, None)
                signature = state.signature
                cached = _CACHE.get(signature)
                if cached is not None and account_success:
                    if actual_bytes is None:
                        try:
                            written = state.target.lstat()
                            actual = (
                                max(0, int(written.st_size))
                                if stat.S_ISREG(written.st_mode)
                                else 0
                            )
                        except OSError:
                            actual = 0
                    else:
                        actual = max(0, int(actual_bytes))
                    cached.managed_bytes = max(
                        0,
                        cached.managed_bytes - state.original_size + actual,
                    )
                    cached.scanned_files += int(
                        state.original_size == 0 and actual > 0
                    )
                if not any(
                    other.signature == signature
                    for other in _RESERVATIONS.values()
                ):
                    _CACHE.pop(signature, None)
                for cached_signature in list(_CACHE):
                    if _signature_covers_path(
                        cached_signature,
                        state.target,
                    ):
                        _CACHE.pop(cached_signature, None)

            try:
                completed = _finish_quarantine_locked(
                    state,
                    commit=commit,
                )
            except _StorageCommitMarkerError:
                for cached_signature in list(_CACHE):
                    if _signature_covers_path(
                        cached_signature,
                        state.target,
                    ):
                        _CACHE.pop(cached_signature, None)
                if state.acceptance_id and _accepted_intent_matches(
                    state,
                    actual_bytes=actual_bytes,
                ):
                    # SQLite is already the FULL-synchronous commit oracle.
                    # Release the process-local token so the next inventory
                    # can retry the pending marker through crash recovery.
                    # Keep the intent row until that journal is finalized.
                    consume(account_success=True)
                    return
                # Unaccepted writes remain retryable by their live caller.
                raise
            except Exception:
                # An unexpected finalizer failure is recovery-owned, but must
                # not strand an in-memory token forever.
                _CACHE.pop(state.signature, None)
                consume(account_success=False)
                raise
            if not commit and not completed:
                # The pending journal remains durable for startup recovery,
                # but callers must not mistake an incomplete rollback for a
                # fully removed upload/evidence file.
                consume(account_success=False)
                raise StoragePolicyError(
                    "media write rollback could not be completed"
                )
            consume(account_success=commit)
            if state.acceptance_id is not None:
                try:
                    from app.media_acceptance import discard_intent

                    discard_intent(state.acceptance_id)
                except Exception:
                    # Journal finalization is authoritative. Leaving a stale
                    # intent is safer than turning cleanup into a false write
                    # failure after the marker became durable.
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, _exc, _tb):
        self.close(success=exc_type is None)
        return False


def _enforce_reservation_locked(
    state: _ReservationState,
    desired_bytes: int,
) -> None:
    config = state.config
    if not config.limit_bytes or config.action == "alert":
        return
    inventory = _inventory(config)
    if not inventory.complete:
        raise StorageWriteRejected(
            "managed media usage could not be measured completely"
        )
    reserved_other = _active_reserved_bytes_for_config(
        config,
        exclude=state,
    )
    desired_delta = max(0, desired_bytes - state.original_size)
    prospective = inventory.managed_bytes + reserved_other + desired_delta
    if prospective <= config.limit_bytes:
        return
    if config.action == "stop":
        raise StorageWriteRejected("managed media storage limit reached")
    protected = _protected_for_config(config)
    protected.add(state.target)
    protected.update(state.protected_paths)
    _, removed_bytes = _quarantine_oldest_locked(
        state,
        inventory,
        bytes_needed=prospective - config.limit_bytes,
        protected_paths=protected,
    )
    if removed_bytes < prospective - config.limit_bytes:
        raise StorageWriteRejected(
            "bounded oldest-first cleanup could not free enough media space"
        )


def begin_media_write(
    target,
    expected_bytes: int = 0,
    *,
    protected_paths: Iterable[os.PathLike | str] = (),
    acceptance_id: str | None = None,
    **policy_overrides,
) -> MediaWriteReservation:
    """Reserve quota before a destination file or its parent is created."""

    with _LOCK:
        config = configured_policy(**policy_overrides)
        resolved_target = _safe_target(Path(target), config)
        normalized_acceptance_id = str(acceptance_id or "").strip().lower()
        if normalized_acceptance_id and (
            len(normalized_acceptance_id) != 32
            or any(
                character not in "0123456789abcdef"
                for character in normalized_acceptance_id
            )
        ):
            raise StorageWriteRejected("media acceptance id is invalid")
        protected_list = []
        protected_existed = []
        for raw_path in protected_paths:
            if len(protected_list) >= MAX_PROTECTED_WRITE_PATHS:
                raise StorageWriteRejected(
                    "too many protected media write paths"
                )
            path = _safe_target(Path(raw_path), config)
            try:
                path.lstat()
                existed = True
            except FileNotFoundError:
                existed = False
            except OSError as exc:
                raise StorageWriteRejected(
                    "protected media write path is unavailable"
                ) from exc
            protected_list.append(path)
            protected_existed.append(existed)
        protected = tuple(protected_list)
        current_signature = _signature(config)
        if any(
            active_state.signature != current_signature
            and _configs_overlap(active_state.config, config)
            for active_state in _RESERVATIONS.values()
        ):
            # A first scan under a different overlapping signature can see a
            # partial file and then count its reservation again. Fail closed
            # until the old configuration is quiescent instead of deleting
            # unrelated evidence based on a double count.
            raise StorageWriteRejected(
                "overlapping storage configuration has active writes"
            )
        requested_paths = {resolved_target, *protected}
        for active_state in _RESERVATIONS.values():
            active_paths = {
                active_state.target,
                *active_state.protected_paths,
            }
            if requested_paths & active_paths:
                raise StorageWriteRejected(
                    "media write path already has an active reservation"
                )
        inventory = _inventory(config)
        original_size = 0
        target_existed = False
        original_file_identity = None
        try:
            original_stat = resolved_target.lstat()
            target_existed = True
            if (
                stat.S_ISREG(original_stat.st_mode)
                and int(original_stat.st_nlink) == 1
            ):
                original_size = max(0, int(original_stat.st_size))
                original_file_identity = (
                    int(original_stat.st_dev),
                    int(original_stat.st_ino),
                )
        except FileNotFoundError:
            pass
        except OSError as exc:
            raise StorageWriteRejected(
                "media write target is unavailable"
            ) from exc
        token = uuid4().hex
        state = _ReservationState(
            signature=_signature(config),
            target=resolved_target,
            protected_paths=protected,
            protected_paths_existed=tuple(protected_existed),
            expected_bytes=0,
            original_size=original_size,
            target_existed=target_existed,
            original_file_identity=original_file_identity,
            owned_file_identities=set(),
            config=config,
            acceptance_id=normalized_acceptance_id or None,
        )
        # A zero-size initial reservation must still reject creation when the
        # stop policy is already at its limit.
        if (
            config.limit_bytes
            and config.action == "stop"
            and inventory.complete
            and inventory.managed_bytes
            + _active_reserved_bytes_for_config(config)
            >= config.limit_bytes
        ):
            raise StorageWriteRejected("managed media storage limit reached")
        desired_bytes = max(0, int(expected_bytes))
        state.expected_bytes = desired_bytes
        # Register before publishing the eager journal. Enforcement performs
        # another inventory pass; without this ordering it could mistake this
        # process's brand-new transaction for a stale crash remnant.
        _RESERVATIONS[token] = state
        try:
            # Publish a pending target plan even when no eviction is needed.
            # A process crash during upload/JPEG creation can then remove the
            # uncommitted target on the next inventory pass.
            _ensure_quarantine_transaction(state)
            if config.limit_bytes:
                _enforce_reservation_locked(state, desired_bytes)
        except BaseException:
            try:
                _finish_quarantine_locked(state, commit=False)
            finally:
                _RESERVATIONS.pop(token, None)
            raise
        return MediaWriteReservation(token)


def invalidate_storage_cache() -> None:
    """Forget cached usage after settings changes or external maintenance."""

    with _LOCK:
        _CACHE.clear()


def storage_activity_status(
    roots: Iterable[os.PathLike | str],
) -> dict[str, int]:
    """Return active quota writers/read pins touching selected roots."""

    resolved_roots = tuple(_resolve_path(root) for root in roots)
    with _LOCK:
        reservations = sum(
            1
            for state in _RESERVATIONS.values()
            if any(
                state.target == root or _is_below(state.target, root)
                for root in resolved_roots
            )
        )
        pins = sum(
            1
            for paths in _PIN_LEASES.values()
            if any(
                path == root or _is_below(path, root)
                for path in paths
                for root in resolved_roots
            )
        )
    return {
        "reservations": reservations,
        "pins": pins,
    }


def require_media_writes_quiescent() -> None:
    """Fail closed unless every process-local media write has finished."""

    with _LOCK:
        if _RESERVATIONS:
            raise StorageWriteRejected(
                "active media write reservations prevent storage migration"
            )
