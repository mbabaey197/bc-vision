import os
from pathlib import Path

import numpy as np
import pytest

from app import database, storage_policy
from app.ai import video_test
from app.async_jobs import _validate_transport_value
from app.media_acceptance import load_intent, require_full_synchronous
from app.media_storage import (
    MediaTransportError,
    PendingMediaFile,
    write_jpeg_atomic,
)


@pytest.fixture
def transport_environment(tmp_path, monkeypatch):
    database_path = tmp_path / "transport.db"
    monkeypatch.setattr(database, "DB_PATH", database_path)
    database.init_db()

    root = (tmp_path / "storage").resolve()
    snapshots = root / "snapshots"
    plates = root / "plates"
    videos = root / "videos"
    for directory in (snapshots, plates, videos):
        directory.mkdir(parents=True)
    policy = storage_policy.configured_policy(
        storage_root=root,
        media_roots=(snapshots, plates, videos),
        limit_bytes=0,
        action="delete_oldest",
    )
    monkeypatch.setattr(
        storage_policy,
        "configured_policy",
        lambda **_overrides: policy,
    )
    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.clear()
        storage_policy._CACHE.clear()

    def create(name="plate.jpg"):
        image = np.full((32, 80, 3), 120, dtype=np.uint8)
        pending = write_jpeg_atomic(
            plates / name,
            image,
            defer_commit=True,
        )
        assert isinstance(pending, PendingMediaFile)
        return pending

    yield root, plates, create

    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.clear()
        storage_policy._CACHE.clear()


def _detach_child_reservation(pending):
    reservation = pending._reservation
    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(reservation._token, None)
    reservation._closed = True


def _transport(pending, *, info=None, event=None):
    payload = video_test.serialize_process_video_result(
        info or {"frames": np.int64(7)},
        [
            {
                "plate_path": str(pending.path),
                "score": np.float32(0.75),
                "shape": (np.int32(32), np.int32(80)),
                "source": Path("source.mp4"),
                "_pending_media": (pending,),
                **(event or {}),
            }
        ],
    )
    _validate_transport_value(payload)
    return payload


def test_video_result_transport_is_data_only_and_reconstructs_handles(
    transport_environment,
):
    _root, _plates, create = transport_environment
    pending = create()
    payload = _transport(pending)
    descriptor = payload["events"][0]["_pending_media"][0]

    assert type(payload["info"]["frames"]) is int
    assert type(payload["events"][0]["score"]) is float
    assert payload["events"][0]["shape"] == (32, 80)
    assert payload["events"][0]["source"] == "source.mp4"
    assert set(descriptor) == {
        "transport_type",
        "version",
        "path",
        "acceptance_id",
        "device",
        "inode",
        "size_bytes",
    }

    _detach_child_reservation(pending)
    info, events = video_test.restore_process_video_result(
        payload,
        allow_pending_media=True,
    )

    assert info == {"frames": 7}
    rebuilt = events[0]["_pending_media"][0]
    assert isinstance(rebuilt, PendingMediaFile)
    assert rebuilt.path == pending.path.resolve()
    assert rebuilt.identity == pending.identity
    assert rebuilt._reservation is None
    assert rebuilt._recovery_required is True


def test_unaccepted_parent_handle_rolls_back_only_through_durable_recovery(
    transport_environment,
):
    root, _plates, create = transport_environment
    pending = create("rollback.jpg")
    payload = _transport(pending)
    _detach_child_reservation(pending)
    _info, events = video_test.restore_process_video_result(
        payload,
        allow_pending_media=True,
    )
    rebuilt = events[0]["_pending_media"][0]

    rebuilt.rollback()

    assert not pending.path.exists()
    assert load_intent(pending.acceptance_id) is None
    assert not (root / ".bcvision-media-quarantine").exists()
    assert rebuilt._recovery_required is False


def test_accepted_parent_handle_commits_exact_child_inode_through_recovery(
    transport_environment,
):
    root, _plates, create = transport_environment
    pending = create("accepted.jpg")
    payload = _transport(pending)
    _detach_child_reservation(pending)
    _info, events = video_test.restore_process_video_result(
        payload,
        allow_pending_media=True,
    )
    rebuilt = events[0]["_pending_media"][0]

    with database.connect() as connection:
        require_full_synchronous(connection)
        rebuilt.accept(
            connection,
            owner_kind="video-test-run",
            owner_id="run-transport-1",
        )
    rebuilt.finalize()

    details = pending.path.lstat()
    assert (int(details.st_dev), int(details.st_ino)) == pending.identity
    assert int(details.st_nlink) == 1
    assert int(details.st_size) == pending.size_bytes
    assert load_intent(pending.acceptance_id) is None
    assert not (root / ".bcvision-media-quarantine").exists()
    assert rebuilt._recovery_required is False


def test_accepted_parent_recovery_preserves_foreign_substitution(
    transport_environment,
):
    root, plates, create = transport_environment
    pending = create("accepted-substitution.jpg")
    payload = _transport(pending)
    _detach_child_reservation(pending)
    _info, events = video_test.restore_process_video_result(
        payload,
        allow_pending_media=True,
    )
    rebuilt = events[0]["_pending_media"][0]
    with database.connect() as connection:
        require_full_synchronous(connection)
        rebuilt.accept(
            connection,
            owner_kind="video-test-run",
            owner_id="run-substitution",
        )
    replacement = plates / "foreign-accepted.bin"
    replacement.write_bytes(b"foreign-accepted-evidence")
    os.replace(replacement, pending.path)

    with pytest.raises(MediaTransportError, match="identity changed"):
        rebuilt.finalize()

    assert pending.path.read_bytes() == b"foreign-accepted-evidence"
    assert load_intent(pending.acceptance_id)["state"] == "accepted"
    assert (root / ".bcvision-media-quarantine").exists()


@pytest.mark.parametrize("mutation", ["foreign", "symlink", "hardlink"])
def test_parent_recovery_never_unlinks_substituted_or_shared_file(
    transport_environment,
    mutation,
):
    _root, plates, create = transport_environment
    pending = create(f"unsafe-{mutation}.jpg")
    payload = _transport(pending)
    _detach_child_reservation(pending)
    _info, events = video_test.restore_process_video_result(
        payload,
        allow_pending_media=True,
    )
    rebuilt = events[0]["_pending_media"][0]

    outside = plates / f"outside-{mutation}.bin"
    if mutation == "foreign":
        outside.write_bytes(b"foreign-evidence")
        os.replace(outside, pending.path)
    elif mutation == "symlink":
        outside.write_bytes(b"foreign-evidence")
        pending.path.unlink()
        try:
            pending.path.symlink_to(outside)
        except OSError:
            pytest.skip("symbolic links are unavailable")
    else:
        try:
            os.link(pending.path, outside)
        except OSError:
            pytest.skip("hard links are unavailable")

    with pytest.raises(
        MediaTransportError,
        match="identity changed|private regular file",
    ):
        rebuilt.rollback()

    if mutation == "foreign":
        assert pending.path.read_bytes() == b"foreign-evidence"
    elif mutation == "symlink":
        assert pending.path.is_symlink()
        assert outside.read_bytes() == b"foreign-evidence"
    else:
        assert pending.path.read_bytes() != b"foreign-evidence"
        assert outside.read_bytes() == pending.path.read_bytes()
        assert pending.path.stat().st_nlink >= 2
    assert load_intent(pending.acceptance_id)["state"] == "pending"


def test_reconstruction_fails_closed_if_recovery_wins_before_parent_accept(
    transport_environment,
):
    _root, _plates, create = transport_environment
    pending = create("lost-handoff.jpg")
    payload = _transport(pending)
    _detach_child_reservation(pending)

    # There is currently no cross-process lease covering this gap. Any
    # inventory pass is entitled to treat the child's pending journal as a
    # crashed write and roll it back before the parent decodes the result.
    status = storage_policy.storage_status(force=True)

    assert status.usage_complete is True
    assert not pending.path.exists()
    assert load_intent(pending.acceptance_id) is None
    with pytest.raises(MediaTransportError, match="disappeared"):
        video_test.restore_process_video_result(
            payload,
            allow_pending_media=True,
        )


def test_child_inventory_without_parent_registry_can_rollback_parent_upload(
    transport_environment,
):
    _root, plates, create = transport_environment
    parent_upload = create("parent-upload.mp4")
    parent_acceptance_id = parent_upload.acceptance_id
    parent_token = parent_upload._reservation._token

    # A spawned process has its own empty _RESERVATIONS dictionary. Removing
    # only that in-memory token models the child's view while leaving the
    # parent's durable journal and pending SQLite intent unchanged.
    with storage_policy._LOCK:
        storage_policy._RESERVATIONS.pop(parent_token)
    child_target = plates / "child-evidence.jpg"
    child_reservation = storage_policy.begin_media_write(child_target, 0)

    assert not parent_upload.path.exists()
    assert load_intent(parent_acceptance_id) is None
    child_reservation.close(success=False)
    parent_upload._reservation._closed = True


def test_transport_wrapper_converts_process_video_result(monkeypatch):
    monkeypatch.setattr(
        video_test,
        "process_video",
        lambda *_args, **_kwargs: (
            {"frames": np.int64(3)},
            [{"confidence": np.float32(0.5)}],
        ),
    )

    payload = video_test.process_video_transport(
        "input.mp4",
        "plates",
        "snapshots",
        detector_variant="yolo11n",
    )

    _validate_transport_value(payload)
    assert payload["info"] == {"frames": 3}
    assert payload["events"][0]["confidence"] == 0.5
