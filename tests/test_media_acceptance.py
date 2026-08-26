import os

import pytest

from app import database, storage_policy
from app.media_acceptance import (
    accept_intent,
    create_intent,
    current_identity,
    require_full_synchronous,
)


def _policy(tmp_path):
    root = tmp_path / "storage"
    snapshots = root / "snapshots"
    plates = root / "plates"
    videos = root / "videos"
    for directory in (snapshots, plates, videos):
        directory.mkdir(parents=True)
    return root, snapshots, plates, videos, {
        "storage_root": root,
        "media_roots": (snapshots, plates, videos),
        "limit_bytes": 10,
        "action": "delete_oldest",
    }


def _database(tmp_path, monkeypatch):
    db_path = tmp_path / "acceptance.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db()
    return db_path


def _accepted_reservation(tmp_path, monkeypatch, *, target_name):
    _database(tmp_path, monkeypatch)
    root, snapshots, _plates, videos, policy = _policy(tmp_path)
    old = snapshots / "old.jpg"
    target = videos / target_name
    old.write_bytes(b"o" * 8)
    acceptance_id = create_intent(target)
    reservation = storage_policy.begin_media_write(
        target,
        6,
        acceptance_id=acceptance_id,
        **policy,
    )
    target.write_bytes(b"v" * 6)
    reservation.claim_created_path(target)
    identity, _ = current_identity(target)
    with database.connect() as connection:
        require_full_synchronous(connection)
        event_id = int(connection.execute(
            "INSERT INTO plate_events(plate_text,video_path) VALUES(?,?)",
            ("ناخوانا", str(target)),
        ).lastrowid)
        accept_intent(
            connection,
            acceptance_id,
            target,
            identity,
            6,
            owner_kind="plate-event",
            owner_id=event_id,
        )
    return root, old, videos, target, policy, acceptance_id, reservation


def test_accepted_sqlite_owner_commits_pending_journal_after_crash(
    tmp_path,
    monkeypatch,
):
    _database(tmp_path, monkeypatch)
    root, snapshots, _plates, videos, policy = _policy(tmp_path)
    old = snapshots / "old.jpg"
    target = videos / "accepted.mp4"
    old.write_bytes(b"o" * 8)
    acceptance_id = create_intent(target)
    reservation = storage_policy.begin_media_write(
        target,
        6,
        acceptance_id=acceptance_id,
        **policy,
    )
    target.write_bytes(b"v" * 6)
    reservation.claim_created_path(target)
    identity, _ = current_identity(target)
    with database.connect() as connection:
        require_full_synchronous(connection)
        event_id = int(connection.execute(
            "INSERT INTO plate_events(plate_text,video_path) VALUES(?,?)",
            ("ناخوانا", str(target)),
        ).lastrowid)
        accept_intent(
            connection,
            acceptance_id,
            target,
            identity,
            6,
            owner_kind="plate-event",
            owner_id=event_id,
        )

    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token)
    reservation._closed = True
    status = storage_policy.storage_status(force=True, **policy)

    assert status.usage_complete is True
    assert status.managed_bytes == 6
    assert target.read_bytes() == b"v" * 6
    assert not old.exists()
    assert not (root / ".bcvision-media-quarantine").exists()
    with database.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM media_acceptance_intents WHERE acceptance_id=?",
            (acceptance_id,),
        ).fetchone() is None
        owner = connection.execute(
            "SELECT video_path FROM plate_events WHERE video_path=?",
            (str(target),),
        ).fetchone()
        assert owner is not None and owner["video_path"] == str(target)


def test_accepted_sqlite_owner_overrides_live_close_false(
    tmp_path,
    monkeypatch,
):
    (
        root,
        old,
        _videos,
        target,
        policy,
        acceptance_id,
        reservation,
    ) = _accepted_reservation(
        tmp_path,
        monkeypatch,
        target_name="accepted-live-rollback.mp4",
    )

    reservation.close(success=False)

    assert reservation._closed is True
    assert reservation._token not in storage_policy._RESERVATIONS
    assert target.read_bytes() == b"v" * 6
    assert not old.exists()
    assert not (root / ".bcvision-media-quarantine").exists()
    status = storage_policy.storage_status(force=True, **policy)
    assert status.usage_complete is True
    assert status.managed_bytes == 6
    with database.connect() as connection:
        assert connection.execute(
            "SELECT 1 FROM media_acceptance_intents WHERE acceptance_id=?",
            (acceptance_id,),
        ).fetchone() is None
        owner = connection.execute(
            "SELECT video_path FROM plate_events WHERE video_path=?",
            (str(target),),
        ).fetchone()
        assert owner is not None and owner["video_path"] == str(target)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("foreign", "ownership changed"),
        ("missing", "missing before commit"),
        ("hardlink", "private regular file"),
        ("size", "accepted media does not match"),
    ],
)
def test_accepted_close_false_fails_closed_on_target_mismatch(
    tmp_path,
    monkeypatch,
    mutation,
    error,
):
    (
        root,
        old,
        videos,
        target,
        _policy_values,
        acceptance_id,
        reservation,
    ) = _accepted_reservation(
        tmp_path,
        monkeypatch,
        target_name=f"accepted-{mutation}.mp4",
    )
    extra_link = None
    if mutation == "foreign":
        replacement = videos / "foreign.mp4"
        replacement.write_bytes(b"f" * 6)
        replacement.replace(target)
    elif mutation == "missing":
        target.unlink()
    elif mutation == "hardlink":
        extra_link = videos / "accepted-hardlink-copy.mp4"
        try:
            os.link(target, extra_link)
        except OSError:
            extra_link.unlink(missing_ok=True)
            with database.connect() as connection:
                connection.execute(
                    "DELETE FROM plate_events WHERE video_path=?",
                    (str(target),),
                )
                connection.execute(
                    "DELETE FROM media_acceptance_intents "
                    "WHERE acceptance_id=?",
                    (acceptance_id,),
                )
            reservation.close(success=False)
            pytest.skip("hard links are unavailable")
    elif mutation == "size":
        target.write_bytes(b"s" * 5)

    with pytest.raises(storage_policy.StoragePolicyError, match=error):
        reservation.close(success=False)

    assert reservation._closed is False
    assert reservation._token in storage_policy._RESERVATIONS
    assert not old.exists()
    assert (root / ".bcvision-media-quarantine").exists()
    if mutation == "missing":
        assert not target.exists()
    elif mutation == "foreign":
        assert target.read_bytes() == b"f" * 6
    elif mutation == "size":
        assert target.read_bytes() == b"s" * 5
    else:
        assert target.read_bytes() == b"v" * 6
        assert extra_link is not None and extra_link.exists()
    with database.connect() as connection:
        intent = connection.execute(
            "SELECT state FROM media_acceptance_intents "
            "WHERE acceptance_id=?",
            (acceptance_id,),
        ).fetchone()
        assert intent is not None and intent["state"] == "accepted"

    # Release the synthetic DB owner (and the extra hard link) so the ordinary
    # unaccepted rollback can clean this test's live reservation. The mismatch
    # assertions above prove close(False) itself did not mutate either side.
    if extra_link is not None:
        extra_link.unlink()
    with database.connect() as connection:
        connection.execute(
            "DELETE FROM plate_events WHERE video_path=?",
            (str(target),),
        )
        connection.execute(
            "DELETE FROM media_acceptance_intents WHERE acceptance_id=?",
            (acceptance_id,),
        )
    reservation.close(success=False)
    assert old.read_bytes() == b"o" * 8
    assert not (root / ".bcvision-media-quarantine").exists()
    if mutation == "foreign":
        assert target.read_bytes() == b"f" * 6
    else:
        assert not target.exists()


def test_acceptance_backed_commit_rejects_uncommitted_database_owner(
    tmp_path,
    monkeypatch,
):
    _database(tmp_path, monkeypatch)
    _root, _snapshots, _plates, videos, policy = _policy(tmp_path)
    target = videos / "not-owned.mp4"
    acceptance_id = create_intent(target)
    reservation = storage_policy.begin_media_write(
        target,
        1,
        acceptance_id=acceptance_id,
        **policy,
    )
    target.write_bytes(b"x")
    reservation.claim_created_path(target)

    with pytest.raises(
        storage_policy.StorageWriteRejected,
        match="database owner",
    ):
        reservation.close(success=True, actual_bytes=1)

    assert reservation._closed is False
    reservation.close(success=False)
    assert not target.exists()


def test_accepted_identity_mismatch_fails_closed_without_deleting_either_side(
    tmp_path,
    monkeypatch,
):
    _database(tmp_path, monkeypatch)
    root, snapshots, _plates, videos, policy = _policy(tmp_path)
    old = snapshots / "old.jpg"
    target = videos / "mismatch.mp4"
    old.write_bytes(b"o" * 8)
    acceptance_id = create_intent(target)
    reservation = storage_policy.begin_media_write(
        target,
        6,
        acceptance_id=acceptance_id,
        **policy,
    )
    target.write_bytes(b"v" * 6)
    reservation.claim_created_path(target)
    identity, _ = current_identity(target)
    with database.connect() as connection:
        require_full_synchronous(connection)
        accept_intent(
            connection,
            acceptance_id,
            target,
            identity,
            6,
            owner_kind="video-test-run",
            owner_id="run-1",
        )
        connection.execute(
            "UPDATE media_acceptance_intents SET inode=inode+1 "
            "WHERE acceptance_id=?",
            (acceptance_id,),
        )

    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token)
    reservation._closed = True
    status = storage_policy.storage_status(force=True, **policy)

    assert status.usage_complete is False
    assert target.read_bytes() == b"v" * 6
    assert not old.exists()
    assert (root / ".bcvision-media-quarantine").exists()
