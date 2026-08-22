"""SQLite commit oracle coordinating durable media with database owners."""
from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from app.file_identity import path_file_identity


class MediaAcceptanceError(RuntimeError):
    """Raised when a media intent cannot be validated or transitioned."""


def _canonical_path(value) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


def _acceptance_id(value) -> str:
    normalized = str(value or "").strip().lower()
    if len(normalized) != 32 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise MediaAcceptanceError("media acceptance id is invalid")
    return normalized


def require_full_synchronous(connection) -> None:
    """Require the next owner transaction to survive a power loss."""

    if connection.in_transaction:
        raise MediaAcceptanceError(
            "media owner transaction started before durable mode"
        )
    connection.execute("PRAGMA synchronous=FULL")


def create_intent(target) -> str:
    """Persist a pending file intent before its filesystem journal exists."""

    from app.database import connect

    acceptance_id = uuid4().hex
    target_path = _canonical_path(target)
    with connect() as connection:
        require_full_synchronous(connection)
        connection.execute(
            "INSERT INTO media_acceptance_intents("
            "acceptance_id,target_path,state) VALUES(?,?,'pending')",
            (acceptance_id, target_path),
        )
    return acceptance_id


def accept_intent(
    connection,
    acceptance_id,
    target,
    identity: tuple[int, int],
    size_bytes: int,
    *,
    owner_kind: str,
    owner_id,
) -> None:
    """Accept an exact inode inside the same transaction as its DB owner."""

    acceptance_id = _acceptance_id(acceptance_id)
    target_path = _canonical_path(target)
    device, inode = (int(identity[0]), int(identity[1]))
    size_bytes = int(size_bytes)
    owner_kind = str(owner_kind or "").strip()[:64]
    owner_id = str(owner_id or "").strip()[:160]
    if device < 0 or inode < 0 or size_bytes < 0:
        raise MediaAcceptanceError("media acceptance identity is invalid")
    if not owner_kind or not owner_id:
        raise MediaAcceptanceError("media acceptance owner is missing")
    row = connection.execute(
        "SELECT target_path,state,device,inode,size_bytes,owner_kind,owner_id "
        "FROM media_acceptance_intents WHERE acceptance_id=?",
        (acceptance_id,),
    ).fetchone()
    if row is None:
        raise MediaAcceptanceError("media acceptance intent is missing")
    expected = (
        target_path,
        "accepted",
        device,
        inode,
        size_bytes,
        owner_kind,
        owner_id,
    )
    current = (
        str(row["target_path"]),
        str(row["state"]),
        row["device"],
        row["inode"],
        row["size_bytes"],
        str(row["owner_kind"]),
        str(row["owner_id"]),
    )
    if row["state"] == "accepted":
        if current != expected:
            raise MediaAcceptanceError(
                "accepted media intent does not match its owner"
            )
        return
    if row["state"] != "pending" or str(row["target_path"]) != target_path:
        raise MediaAcceptanceError("media acceptance intent is inconsistent")
    updated = connection.execute(
        "UPDATE media_acceptance_intents SET state='accepted',device=?,"
        "inode=?,size_bytes=?,owner_kind=?,owner_id=?,"
        "accepted_at=CURRENT_TIMESTAMP "
        "WHERE acceptance_id=? AND state='pending' AND target_path=?",
        (
            device,
            inode,
            size_bytes,
            owner_kind,
            owner_id,
            acceptance_id,
            target_path,
        ),
    )
    if updated.rowcount != 1:
        raise MediaAcceptanceError("media acceptance transition was lost")


def load_intent(acceptance_id) -> dict | None:
    """Load the durable commit-oracle row used by storage recovery."""

    from app.database import connect

    acceptance_id = _acceptance_id(acceptance_id)
    with connect() as connection:
        row = connection.execute(
            "SELECT acceptance_id,target_path,state,device,inode,size_bytes,"
            "owner_kind,owner_id FROM media_acceptance_intents "
            "WHERE acceptance_id=?",
            (acceptance_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def discard_intent(acceptance_id) -> None:
    """Best-effort cleanup after the filesystem journal is finalized."""

    from app.database import connect

    acceptance_id = _acceptance_id(acceptance_id)
    with connect() as connection:
        connection.execute(
            "DELETE FROM media_acceptance_intents WHERE acceptance_id=?",
            (acceptance_id,),
        )


def discard_pending_intent(acceptance_id, target) -> bool:
    """Atomically revoke only an owner intent that is still pending.

    A subprocess result can race an owner transaction that is deciding
    whether to accept the media.  Restricting this delete to ``pending`` makes
    rollback lose safely to a committed owner instead of deleting its oracle
    row and allowing crash recovery to remove DB-owned evidence.
    """

    from app.database import connect

    acceptance_id = _acceptance_id(acceptance_id)
    target_path = _canonical_path(target)
    with connect() as connection:
        require_full_synchronous(connection)
        deleted = connection.execute(
            "DELETE FROM media_acceptance_intents "
            "WHERE acceptance_id=? AND target_path=? AND state='pending'",
            (acceptance_id, target_path),
        )
    return deleted.rowcount == 1


def current_identity(path) -> tuple[tuple[int, int], int]:
    """Return the exact private regular-file identity accepted by SQLite."""

    import stat

    details = Path(path).lstat()
    if not stat.S_ISREG(details.st_mode) or int(details.st_nlink) != 1:
        raise MediaAcceptanceError(
            "accepted media path is not a private regular file"
        )
    return (
        path_file_identity(path, details=details),
        max(0, int(details.st_size)),
    )


def durable_unlink(path) -> None:
    """Remove an unused intent-side artifact and persist its directory."""

    target = Path(path)
    target.unlink(missing_ok=True)
    if os.name != "nt" and target.parent.exists():
        from app.storage_policy import fsync_parent_directory

        fsync_parent_directory(target)
