import asyncio
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest

from app import main, media_storage, storage_policy
from app.storage_policy import (
    StorageWriteRejected,
    WriterPreferredGate,
    begin_media_write,
    delete_older_than,
    enforce_storage_limit,
    fsync_parent_directory,
    invalidate_storage_cache,
    pin_media_paths,
    require_media_writes_quiescent,
    storage_activity_status,
    storage_status,
    validate_storage_layout,
)


def _roots(tmp_path):
    root = tmp_path / "data"
    snapshots = root / "snapshots"
    plates = root / "plates"
    videos = root / "videos"
    backups = root / "backups"
    for directory in (snapshots, plates, videos, backups):
        directory.mkdir(parents=True)
    return root, snapshots, plates, videos, backups


def _policy(root, snapshots, plates, videos, *, limit, action):
    return {
        "storage_root": root,
        "media_roots": (snapshots, plates, videos),
        "limit_bytes": limit,
        "action": action,
    }


def test_stop_rejects_before_creating_new_media_file(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    (snapshots / "existing.jpg").write_bytes(b"x" * 10)
    target = videos / "new.mp4"

    with pytest.raises(StorageWriteRejected):
        begin_media_write(
            target,
            1,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=10,
                action="stop",
            ),
        )

    assert not target.exists()
    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="stop",
        ),
    )
    assert status.managed_bytes == 10
    assert status.write_blocked is True


def test_overlapping_signature_reservations_share_one_limit(tmp_path):
    root = tmp_path / "data"
    old_media = root / "old-media"
    new_media = root / "new-media"
    old_media.mkdir(parents=True)
    new_media.mkdir(parents=True)
    old_policy = {
        "storage_root": root,
        "media_roots": (old_media,),
        "limit_bytes": 10,
        "action": "stop",
    }
    expanded_policy = {
        "storage_root": root,
        "media_roots": (old_media, new_media),
        "limit_bytes": 10,
        "action": "stop",
    }

    first = begin_media_write(old_media / "first.jpg", 6, **old_policy)
    try:
        status = storage_status(force=True, **expanded_policy)
        assert status.reserved_bytes == 6
        with pytest.raises(StorageWriteRejected):
            begin_media_write(
                new_media / "second.jpg",
                6,
                **expanded_policy,
            )
    finally:
        first.close(success=False)

    assert storage_activity_status((root,))["reservations"] == 0


def test_overlapping_policy_transition_never_double_counts_partial_write(
    tmp_path,
):
    root = tmp_path / "data"
    old_media = root / "old-media"
    new_media = root / "new-media"
    old_media.mkdir(parents=True)
    new_media.mkdir(parents=True)
    evidence = old_media / "evidence.jpg"
    first_target = old_media / "partial.mp4"
    evidence.write_bytes(b"e" * 6)
    old_policy = {
        "storage_root": root,
        "media_roots": (old_media,),
        "limit_bytes": 18,
        "action": "delete_oldest",
    }
    expanded_policy = {
        "storage_root": root,
        "media_roots": (old_media, new_media),
        "limit_bytes": 18,
        "action": "delete_oldest",
    }
    first = begin_media_write(first_target, 6, **old_policy)
    first_target.write_bytes(b"p" * 3)
    first.claim_created_path(first_target)
    try:
        with pytest.raises(
            StorageWriteRejected,
            match="overlapping storage configuration",
        ):
            begin_media_write(
                new_media / "second.mp4",
                6,
                **expanded_policy,
            )
        assert evidence.read_bytes() == b"e" * 6
    finally:
        first.close(success=False)

    assert not first_target.exists()
    assert evidence.read_bytes() == b"e" * 6


def test_cache_refresh_never_double_counts_claimed_partial_inode(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    evidence = snapshots / "old-evidence.jpg"
    target = videos / "growing.mp4"
    evidence.write_bytes(b"e" * 6)
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=12,
        action="delete_oldest",
    )
    reservation = begin_media_write(target, 0, **policy)
    target.write_bytes(b"p" * 3)
    reservation.claim_created_path(target)

    invalidate_storage_cache()
    reservation.grow(6)

    assert evidence.read_bytes() == b"e" * 6
    with target.open("ab") as stream:
        stream.write(b"f" * 3)
        stream.flush()
        os.fsync(stream.fileno())
    reservation.close(success=True, actual_bytes=6)
    status = storage_status(force=True, **policy)
    assert status.managed_bytes == 12
    assert evidence.read_bytes() == b"e" * 6
    assert target.read_bytes() == b"p" * 3 + b"f" * 3


def test_same_media_path_cannot_have_two_active_reservations(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = videos / "one-writer.mp4"
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="stop",
    )
    first = begin_media_write(target, 1, **policy)
    try:
        with pytest.raises(
            StorageWriteRejected,
            match="active reservation",
        ):
            begin_media_write(target, 1, **policy)
        target.write_bytes(b"a")
        first.claim_created_path(target)
        first.close(success=True, actual_bytes=1)
    finally:
        if not first._closed:
            first.close(success=False)

    assert target.read_bytes() == b"a"


def test_rollback_preserves_unclaimed_foreign_file_at_target(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = videos / "foreign.mp4"
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="stop",
    )
    reservation = begin_media_write(target, 1, **policy)

    # Simulate an external writer winning the pathname. Without an explicit
    # inode claim, rollback must not assume that this process owns the file.
    target.write_bytes(b"foreign")
    fsync_parent_directory(target)
    reservation.close(success=False)

    assert target.read_bytes() == b"foreign"
    assert not (root / ".bcvision-media-quarantine").exists()


def test_quiescence_check_rejects_active_media_write(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    reservation = begin_media_write(
        videos / "active.mp4",
        1,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="stop",
        ),
    )
    try:
        with pytest.raises(StorageWriteRejected, match="active media write"):
            require_media_writes_quiescent()
    finally:
        reservation.close(success=False)

    require_media_writes_quiescent()


def test_failed_delete_oldest_write_rolls_back_quarantined_media(tmp_path):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    outside = tmp_path / "outside.jpg"
    database = root / "bcvision.db"
    secret = root / ".secret"
    outbox = root / "bcvision-retry.db"
    backup = backups / "bcvision-old.db"
    for protected in (outside, database, secret, outbox, backup):
        protected.write_bytes(b"p" * 100)

    oldest = snapshots / "oldest.jpg"
    newer = plates / "newer.jpg"
    oldest.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    os.utime(oldest, (10, 10))
    os.utime(newer, (20, 20))
    outside_link = videos / "outside-link.jpg"
    try:
        outside_link.symlink_to(outside)
    except OSError:
        outside_link = None

    reservation = begin_media_write(
        videos / "incoming.mp4",
        3,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    journals = list(
        (root / ".bcvision-media-quarantine").glob("*/journal.json")
    )

    assert not oldest.exists()
    assert len(journals) == 1
    reservation.close(success=False)

    assert oldest.read_bytes() == b"a" * 6
    assert not (root / ".bcvision-media-quarantine").exists()
    assert newer.read_bytes() == b"b" * 6
    for protected in (outside, database, secret, outbox, backup):
        assert protected.read_bytes() == b"p" * 100
    if outside_link is not None:
        assert outside_link.is_symlink()


def test_successful_write_commits_quarantine_after_publish(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    oldest = snapshots / "oldest.jpg"
    newer = plates / "newer.jpg"
    target = videos / "new.mp4"
    oldest.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    os.utime(oldest, (10, 10))
    os.utime(newer, (20, 20))

    reservation = begin_media_write(
        target,
        3,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    assert not oldest.exists()
    target.write_bytes(b"n" * 3)
    reservation.claim_created_path(target)
    reservation.close(success=True, actual_bytes=3)

    assert not oldest.exists()
    assert newer.read_bytes() == b"b" * 6
    assert target.read_bytes() == b"n" * 3
    assert not (root / ".bcvision-media-quarantine").exists()
    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    assert status.managed_bytes == 9


def test_crash_pending_write_removes_target_and_restores_eviction(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    old_evidence = snapshots / "old-evidence.jpg"
    target = videos / "orphan.mp4"
    old_evidence.write_bytes(b"e" * 8)
    reservation = begin_media_write(
        target,
        6,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    assert not old_evidence.exists()
    target.write_bytes(b"o" * 6)
    reservation.claim_created_path(target)
    fsync_parent_directory(target)

    # Simulate abrupt process loss: the in-memory token disappears without a
    # success/failure close, while the durable pending journal remains.
    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token)
    reservation._closed = True

    recovered = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )

    assert recovered.usage_complete is True
    assert recovered.managed_bytes == 8
    assert old_evidence.read_bytes() == b"e" * 8
    assert not target.exists()
    assert not (root / ".bcvision-media-quarantine").exists()


def test_crash_pending_write_without_eviction_removes_target(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = videos / "orphan-without-eviction.mp4"
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="stop",
    )
    reservation = begin_media_write(target, 1, **policy)
    journals = list(
        (root / ".bcvision-media-quarantine").glob("*/journal.json")
    )
    assert len(journals) == 1
    target.write_bytes(b"x")
    reservation.claim_created_path(target)
    fsync_parent_directory(target)

    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token)
    reservation._closed = True
    recovered = storage_status(force=True, **policy)

    assert recovered.usage_complete is True
    assert recovered.managed_bytes == 0
    assert not target.exists()
    assert not (root / ".bcvision-media-quarantine").exists()


def test_failed_write_preserves_preexisting_zero_byte_target(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = videos / "existing-empty.jpg"
    target.touch()
    reservation = begin_media_write(
        target,
        1,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="stop",
        ),
    )

    reservation.close(success=False)

    assert target.is_file()
    assert target.stat().st_size == 0
    assert not (root / ".bcvision-media-quarantine").exists()


def test_crash_pending_write_removes_owned_temporary_file(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = plates / "evidence.jpg"
    temporary = plates / ".evidence.jpg.random.tmp"
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="stop",
    )
    reservation = begin_media_write(
        target,
        1,
        protected_paths=(temporary,),
        **policy,
    )
    temporary.write_bytes(b"partial")
    reservation.claim_created_path(temporary)
    fsync_parent_directory(temporary)

    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token)
    reservation._closed = True
    recovered = storage_status(force=True, **policy)

    assert recovered.usage_complete is True
    assert recovered.managed_bytes == 0
    assert not temporary.exists()
    assert not target.exists()


def test_commit_marker_failure_is_reported_and_target_remains_pending(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    target = videos / "not-accepted.mp4"
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="stop",
    )
    reservation = begin_media_write(target, 1, **policy)
    target.write_bytes(b"x")
    reservation.claim_created_path(target)
    fsync_parent_directory(target)
    real_write = storage_policy._atomic_journal_write

    def fail_commit_marker(path, payload):
        if payload.get("status") == "committed":
            raise OSError("simulated journal fsync failure")
        return real_write(path, payload)

    monkeypatch.setattr(
        storage_policy,
        "_atomic_journal_write",
        fail_commit_marker,
    )
    with pytest.raises(
        storage_policy.StoragePolicyError,
        match="commit marker",
    ):
        reservation.close(success=True, actual_bytes=1)
    assert target.is_file()
    assert reservation._closed is False
    assert reservation._token in storage_policy._RESERVATIONS

    monkeypatch.setattr(
        storage_policy,
        "_atomic_journal_write",
        real_write,
    )
    reservation.close(success=False)
    recovered = storage_status(force=True, **policy)
    assert recovered.usage_complete is True
    assert recovered.managed_bytes == 0
    assert not target.exists()


def test_stale_pending_quarantine_is_recovered_after_crash(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    oldest = snapshots / "oldest.jpg"
    newer = plates / "newer.jpg"
    oldest.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    os.utime(oldest, (10, 10))
    os.utime(newer, (20, 20))

    reservation = begin_media_write(
        videos / "new.mp4",
        3,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    assert not oldest.exists()
    # Simulate process loss: the in-memory reservation disappears while its
    # pending journal and quarantined evidence remain durable.
    storage_policy._RESERVATIONS.pop(reservation._token)
    invalidate_storage_cache()

    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )

    assert oldest.read_bytes() == b"a" * 6
    assert status.managed_bytes == 12
    assert not (root / ".bcvision-media-quarantine").exists()


def test_recovery_cleans_regular_atomic_journal_temp_leftover(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    oldest = snapshots / "oldest.jpg"
    newer = plates / "newer.jpg"
    oldest.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    os.utime(oldest, (10, 10))
    os.utime(newer, (20, 20))
    reservation = begin_media_write(
        videos / "new.mp4",
        3,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    transaction = next(
        (root / ".bcvision-media-quarantine").iterdir()
    )
    orphan = transaction / f".journal.json.{'f' * 32}.tmp"
    orphan.write_bytes(b"interrupted-journal-rewrite")
    storage_policy._RESERVATIONS.pop(reservation._token)
    invalidate_storage_cache()

    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )

    assert status.usage_complete is True
    assert oldest.read_bytes() == b"a" * 6
    assert not (root / ".bcvision-media-quarantine").exists()


def test_recovery_never_unlinks_symlink_disguised_as_journal_temp(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    quarantine = root / ".bcvision-media-quarantine"
    transaction = quarantine / ("a" * 32)
    files = transaction / "files"
    files.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.write_bytes(b"keep")
    disguised = transaction / f".journal.json.{'b' * 32}.tmp"
    try:
        disguised.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable")

    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="stop",
        ),
    )

    assert status.usage_complete is False
    assert disguised.is_symlink()
    assert outside.read_bytes() == b"keep"


def test_close_exception_never_strands_active_quarantine_token(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    oldest = snapshots / "oldest.jpg"
    newer = plates / "newer.jpg"
    oldest.write_bytes(b"a" * 6)
    newer.write_bytes(b"b" * 6)
    os.utime(oldest, (10, 10))
    os.utime(newer, (20, 20))
    reservation = begin_media_write(
        videos / "new.mp4",
        3,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    token = reservation._token
    real_finish = storage_policy._finish_quarantine_locked

    def fail_finish(*_args, **_kwargs):
        raise RuntimeError("injected finalization failure")

    monkeypatch.setattr(
        storage_policy,
        "_finish_quarantine_locked",
        fail_finish,
    )
    with pytest.raises(RuntimeError, match="injected finalization failure"):
        reservation.close(success=False)

    assert token not in storage_policy._RESERVATIONS
    monkeypatch.setattr(
        storage_policy,
        "_finish_quarantine_locked",
        real_finish,
    )
    invalidate_storage_cache()
    status = storage_status(
        force=True,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="delete_oldest",
        ),
    )
    assert status.usage_complete is True
    assert oldest.read_bytes() == b"a" * 6


def test_alert_reports_over_limit_without_deleting(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    media = videos / "keep.mp4"
    media.write_bytes(b"v" * 11)

    enforced = enforce_storage_limit(
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="alert",
        ),
    )
    reservation = begin_media_write(
        plates / "also-allowed.jpg",
        5,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="alert",
        ),
    )
    during_write = storage_status(
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=10,
            action="alert",
        )
    )
    reservation.close(success=False)

    assert enforced.over_limit is True
    assert enforced.deleted_files == 0
    assert during_write.over_limit is True
    assert during_write.reserved_bytes == 5
    assert media.read_bytes() == b"v" * 11


def test_delete_oldest_cleanup_is_bounded(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    files = []
    for index in range(3):
        path = snapshots / f"{index}.jpg"
        path.write_bytes(b"x" * 5)
        os.utime(path, (index + 1, index + 1))
        files.append(path)

    status = enforce_storage_limit(
        max_delete_files=1,
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=5,
            action="delete_oldest",
        ),
    )

    assert status.deleted_files == 1
    assert status.over_limit is True
    assert status.error
    assert not files[0].exists()
    assert files[1].exists() and files[2].exists()


def test_concurrent_reservations_cannot_oversubscribe_limit(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    barrier = threading.Barrier(2)
    release = threading.Event()
    outcomes = []

    def reserve(name):
        barrier.wait()
        try:
            reservation = begin_media_write(
                videos / name,
                6,
                **_policy(
                    root,
                    snapshots,
                    plates,
                    videos,
                    limit=10,
                    action="stop",
                ),
            )
        except StorageWriteRejected:
            outcomes.append("rejected")
            release.set()
            return
        outcomes.append("reserved")
        release.wait(2)
        reservation.close(success=False)

    threads = [
        threading.Thread(target=reserve, args=(f"{index}.mp4",))
        for index in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(3)

    assert sorted(outcomes) == ["rejected", "reserved"]


def test_pin_lease_protects_source_video_until_reader_releases_it(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    source = videos / "source.mp4"
    alternate = snapshots / "alternate.jpg"
    source.write_bytes(b"s" * 6)
    alternate.write_bytes(b"a" * 6)
    os.utime(source, (10, 10))
    os.utime(alternate, (20, 20))
    policy = _policy(
        root,
        snapshots,
        plates,
        videos,
        limit=10,
        action="delete_oldest",
    )

    with pin_media_paths((source,), **policy):
        reservation = begin_media_write(videos / "new.mp4", 3, **policy)
        assert source.read_bytes() == b"s" * 6
        assert not alternate.exists()
        reservation.close(success=False)
        assert alternate.read_bytes() == b"a" * 6

    reservation = begin_media_write(videos / "newer.mp4", 3, **policy)
    assert not source.exists()
    reservation.close(success=False)
    assert source.read_bytes() == b"s" * 6


def test_unlimited_write_still_protects_target_during_policy_transition(
    tmp_path,
):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    old = videos / "old.mp4"
    target = videos / "new.mp4"
    temporary = videos / ".new.mp4.atomic.tmp"
    old.write_bytes(b"o" * 3)
    reservation = begin_media_write(
        target,
        0,
        protected_paths=(temporary,),
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=0,
            action="delete_oldest",
        ),
    )
    assert reservation._token is not None
    temporary.write_bytes(b"partial")

    removed = delete_older_than(
        videos,
        10_000_000_000,
        storage_root=root,
    )

    assert removed == 1
    assert not old.exists()
    assert temporary.read_bytes() == b"partial"
    reservation.close(success=False)


def test_hardlinked_media_is_counted_but_never_claimed_as_freed(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    first = snapshots / "linked.jpg"
    second = plates / "linked-again.jpg"
    first.write_bytes(b"x" * 8)
    try:
        os.link(first, second)
    except OSError:
        pytest.skip("hardlinks are unavailable")

    status = enforce_storage_limit(
        **_policy(
            root,
            snapshots,
            plates,
            videos,
            limit=4,
            action="delete_oldest",
        )
    )

    assert status.managed_bytes == 8
    assert status.over_limit is True
    assert status.deleted_files == 0
    assert status.error
    assert first.read_bytes() == b"x" * 8
    assert second.read_bytes() == b"x" * 8


def test_hardlinked_overwrite_cannot_subtract_shared_original_size(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    first = snapshots / "linked.jpg"
    second = plates / "linked-again.jpg"
    first.write_bytes(b"x" * 8)
    try:
        os.link(first, second)
    except OSError:
        pytest.skip("hardlinks are unavailable")

    with pytest.raises(StorageWriteRejected):
        begin_media_write(
            first,
            9,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=10,
                action="stop",
            ),
        )

    assert first.read_bytes() == b"x" * 8
    assert second.read_bytes() == b"x" * 8


@pytest.mark.parametrize("relative", [".bcvision-media-quarantine", ".bcvision-media-quarantine/nested"])
def test_current_media_root_cannot_overlap_quarantine(
    tmp_path,
    relative,
):
    root, _snapshots, plates, videos, backups = _roots(tmp_path)
    unsafe = root / relative

    with pytest.raises(storage_policy.StoragePolicyError, match="quarantine"):
        validate_storage_layout(
            root,
            (unsafe, plates, videos),
            backups,
            history_roots=(),
        )


@pytest.mark.parametrize("relative", [".bcvision-media-quarantine", ".bcvision-media-quarantine/backups"])
def test_backup_root_cannot_overlap_quarantine(tmp_path, relative):
    root, snapshots, plates, videos, _backups = _roots(tmp_path)
    unsafe_backup = root / relative

    with pytest.raises(storage_policy.StoragePolicyError, match="quarantine"):
        validate_storage_layout(
            root,
            (snapshots, plates, videos),
            unsafe_backup,
            history_roots=(),
        )


def test_current_media_roots_cannot_be_ancestors_of_each_other(tmp_path):
    root, snapshots, _plates, videos, backups = _roots(tmp_path)
    nested_plates = snapshots / "plates"

    with pytest.raises(
        storage_policy.StoragePolicyError,
        match="must not overlap",
    ):
        validate_storage_layout(
            root,
            (snapshots, nested_plates, videos),
            backups,
            history_roots=(),
        )


def test_media_history_cannot_overlap_a_current_media_root(tmp_path):
    root, snapshots, plates, videos, backups = _roots(tmp_path)

    with pytest.raises(
        storage_policy.StoragePolicyError,
        match="history overlaps managed",
    ):
        validate_storage_layout(
            root,
            (snapshots, plates, videos),
            backups,
            history_roots=(snapshots / "legacy",),
        )


def test_backup_root_cannot_overlap_existing_media_history(tmp_path):
    root, snapshots, plates, videos, _backups = _roots(tmp_path)
    history = root / "legacy-media"

    with pytest.raises(
        storage_policy.StoragePolicyError,
        match="media history",
    ):
        validate_storage_layout(
            root,
            (snapshots, plates, videos),
            history / "backups",
            history_roots=(history,),
        )


def test_history_overlapping_quarantine_is_invalid_and_blocks_strict_write(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    unsafe = root / ".bcvision-media-quarantine" / "nested"
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps([str(unsafe)]),
        "max_storage_gb": "0",
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(force=True, limit_bytes=10, action="stop")

    assert status.usage_complete is False
    assert status.invalid_history_roots == (str(unsafe),)
    with pytest.raises(StorageWriteRejected):
        begin_media_write(videos / "blocked.mp4", 1, limit_bytes=10, action="stop")


@pytest.mark.parametrize("configured_limit", ["not-a-number", "-1"])
def test_corrupt_configured_limit_fails_closed(
    tmp_path,
    monkeypatch,
    configured_limit,
):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": "[]",
        "max_storage_gb": configured_limit,
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )
    target = videos / "new" / "blocked.mp4"

    with pytest.raises(
        storage_policy.StoragePolicyError,
        match="storage limit is invalid",
    ):
        begin_media_write(target, 1)

    assert not target.parent.exists()
    assert not target.exists()


def test_same_root_media_history_is_included_in_quota(tmp_path, monkeypatch):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    history = root / "legacy-plates"
    history.mkdir()
    (videos / "current.mp4").write_bytes(b"c" * 3)
    (history / "legacy.jpg").write_bytes(b"h" * 8)
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps([str(history)]),
        "max_storage_gb": "0",
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(force=True, limit_bytes=10, action="stop")

    assert status.managed_bytes == 11
    assert str(history.resolve()) in status.media_roots
    assert status.over_limit is True
    with pytest.raises(StorageWriteRejected):
        begin_media_write(videos / "blocked.mp4", 1, limit_bytes=10, action="stop")


def test_external_history_is_accounted_read_only_and_can_block(tmp_path, monkeypatch):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    external = tmp_path / "old-volume-media"
    external.mkdir()
    legacy = external / "legacy.mp4"
    legacy.write_bytes(b"h" * 11)
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps([str(external)]),
        "max_storage_gb": "0",
        "storage_full_action": "delete_oldest",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(
        force=True,
        limit_bytes=10,
        action="delete_oldest",
    )

    assert status.managed_bytes == 11
    assert status.read_only_history_roots == (str(external.resolve()),)
    with pytest.raises(StorageWriteRejected):
        begin_media_write(
            videos / "blocked.mp4",
            1,
            limit_bytes=10,
            action="delete_oldest",
        )
    assert legacy.read_bytes() == b"h" * 11


def test_missing_external_history_fails_closed_until_volume_returns(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    missing = tmp_path / "unmounted-old-media"
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps([str(missing)]),
        "max_storage_gb": "0",
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(force=True, limit_bytes=10, action="stop")

    assert status.usage_complete is False
    assert status.read_only_history_roots == (str(missing.resolve()),)
    with pytest.raises(StorageWriteRejected, match="measured completely"):
        begin_media_write(
            videos / "blocked.mp4",
            1,
            limit_bytes=10,
            action="stop",
        )


def test_unsafe_history_is_reported_and_strict_writes_fail_closed(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    unsafe = str(Path(root.anchor))
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps([unsafe]),
        "max_storage_gb": "0",
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(force=True, limit_bytes=10, action="stop")

    assert status.usage_complete is False
    assert status.invalid_history_roots == (unsafe,)
    with pytest.raises(StorageWriteRejected, match="measured completely"):
        begin_media_write(videos / "blocked.mp4", 1, limit_bytes=10, action="stop")


def test_history_overflow_is_reported_instead_of_silently_truncated(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, backups = _roots(tmp_path)
    history = [
        str(root / "history" / str(index))
        for index in range(storage_policy.MAX_HISTORY_ROOTS + 1)
    ]
    settings = {
        "storage_root": str(root),
        "snapshot_path": str(snapshots),
        "plate_path": str(plates),
        "video_path": str(videos),
        "backup_path": str(backups),
        "media_roots_history": json.dumps(history),
        "max_storage_gb": "0",
        "storage_full_action": "stop",
    }
    monkeypatch.setattr(
        storage_policy,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )

    status = storage_status(force=True, limit_bytes=10, action="stop")

    assert status.usage_complete is False
    assert status.invalid_history_roots == (
        f"<media-roots-history-overflow:{len(history)}>",
    )


def test_storage_mutation_gate_allows_readers_and_prefers_waiting_writer():
    gate = WriterPreferredGate()

    assert gate.try_acquire_shared() is True
    assert gate.try_acquire_shared() is True
    writer = gate.queue_exclusive()
    # Once a migration is queued, later normal mutations must not jump ahead.
    assert gate.try_acquire_shared() is False
    assert gate.try_acquire_exclusive(writer) is False
    gate.release_shared()
    assert gate.try_acquire_exclusive(writer) is False
    gate.release_shared()
    assert gate.try_acquire_exclusive(writer) is True
    assert gate.snapshot() == (0, True, 0)
    gate.release_exclusive(writer)
    assert gate.try_acquire_shared() is True
    gate.release_shared()


def test_fsync_parent_directory_targets_published_entry_parent(
    tmp_path,
    monkeypatch,
):
    target = tmp_path / "media" / "upload.mp4"
    calls = []
    monkeypatch.setattr(
        storage_policy,
        "_fsync_directory",
        lambda directory: calls.append(Path(directory)),
    )

    fsync_parent_directory(target)

    assert calls == [target.parent]


def test_symlinked_target_outside_media_root_is_rejected(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = videos / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(StorageWriteRejected):
        begin_media_write(
            link / "escaped.mp4",
            1,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=10,
                action="stop",
            ),
        )

    assert list(outside.iterdir()) == []


def test_unlimited_policy_still_rejects_target_outside_media_roots(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    outside = tmp_path / "outside" / "escaped.mp4"

    with pytest.raises(StorageWriteRejected, match="outside configured"):
        begin_media_write(
            outside,
            0,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=0,
                action="delete_oldest",
            ),
        )

    assert not outside.parent.exists()


def test_symlinked_target_is_rejected_even_when_it_points_inside(tmp_path):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    real = videos / "real"
    real.mkdir()
    link = videos / "alias"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(StorageWriteRejected, match="symlink"):
        begin_media_write(
            link / "unsafe.mp4",
            1,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=10,
                action="stop",
            ),
        )

    assert list(real.iterdir()) == []


def test_media_quota_failure_does_not_raise_or_discard_text_result(
    tmp_path,
    monkeypatch,
):
    def reject(*_args, **_kwargs):
        raise StorageWriteRejected("full")

    monkeypatch.setattr(media_storage, "begin_media_write", reject)
    image = np.full((20, 40, 3), 120, dtype=np.uint8)
    result = {"plate": "۱۲ب۳۴۵۶۷", "crop": image, "bbox": None}
    saved = media_storage.save_event_images(
        result,
        image,
        plate_target=tmp_path / "plate.jpg",
        vehicle_target=tmp_path / "vehicle.jpg",
    )

    assert result["plate"] == "۱۲ب۳۴۵۶۷"
    assert saved.media_status == "error"
    assert "storage policy rejected write" in saved.media_error


def test_video_stop_policy_rejects_before_destination_creation(
    tmp_path,
    monkeypatch,
):
    target_dir = tmp_path / "not-created"

    def reject(*_args, **_kwargs):
        raise StorageWriteRejected("full")

    class Upload:
        async def read(self, _size):
            raise AssertionError("quota must reject before reading the upload")

    monkeypatch.setattr(main, "begin_media_write", reject)

    with pytest.raises(ValueError, match="سهمیه ذخیره‌سازی"):
        asyncio.run(main._save_video_upload(Upload(), target_dir, ".mp4"))

    assert not target_dir.exists()


def test_cancelled_video_upload_removes_partial_file_and_reservation(
    tmp_path,
    monkeypatch,
):
    target_dir = tmp_path / "videos"
    closes = []

    class Reservation:
        def grow(self, _size):
            return None

        def close(self, *, success, actual_bytes=None):
            closes.append((success, actual_bytes))

    class Upload:
        calls = 0

        async def read(self, _size):
            self.calls += 1
            if self.calls == 1:
                return b"partial"
            raise asyncio.CancelledError()

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(main._save_video_upload(Upload(), target_dir, ".mp4"))

    assert list(target_dir.iterdir()) == []
    assert closes == [(False, None)]


def test_storage_status_api_reports_managed_over_limit_without_cleanup(
    tmp_path,
    monkeypatch,
):
    root, snapshots, plates, videos, _ = _roots(tmp_path)
    media = videos / "keep.mp4"
    media.write_bytes(b"x" * 11)
    settings = {
        "storage_root": str(root),
        "max_storage_gb": "1",
        "storage_full_action": "alert",
    }

    monkeypatch.setattr(main, "auth", lambda _request: "admin")
    monkeypatch.setattr(
        main,
        "get_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        main,
        "_path_usage",
        lambda path: {
            "ok": True,
            "path": str(path),
            "total": 100,
            "used": 11,
            "free": 89,
            "percent": 11,
        },
    )
    monkeypatch.setattr(
        main,
        "storage_status",
        lambda force=False: storage_status(
            force=force,
            **_policy(
                root,
                snapshots,
                plates,
                videos,
                limit=10,
                action="alert",
            ),
        ),
    )

    response = main.api_storage_status(object())
    payload = json.loads(response.body)

    assert payload["managed_bytes"] == 11
    assert payload["limit_bytes"] == 10
    assert payload["over_limit"] is True
    assert payload["write_blocked"] is False
    assert payload["policy_error"] == ""
    assert media.exists()


@pytest.fixture(autouse=True)
def _clear_policy_cache_after_test():
    yield
    invalidate_storage_cache()
