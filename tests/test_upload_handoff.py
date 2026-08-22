import asyncio

import pytest

import app.main as main
from app import storage_policy


def test_saved_upload_is_pinned_before_write_reservation_closes(
    tmp_path,
    monkeypatch,
):
    events = []

    class Reservation:
        def grow(self, size):
            events.append(("grow", size))

        def close(self, *, success, actual_bytes=None):
            events.append(("close", success, actual_bytes))
            if success:
                assert any(event[0] == "pin" for event in events)

    class Lease:
        def close(self):
            events.append(("unpin",))

    class Upload:
        def __init__(self):
            self._chunks = [b"video-bytes", b""]

        async def read(self, _size):
            return self._chunks.pop(0)

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda target, size: (
            events.append(("reserve", target, size)) or Reservation()
        ),
    )
    monkeypatch.setattr(
        main,
        "pin_media_paths",
        lambda paths: events.append(("pin", tuple(paths))) or Lease(),
    )
    monkeypatch.setattr(
        main,
        "fsync_parent_directory",
        lambda target: events.append(("fsync-parent", target)),
    )

    target, lease = asyncio.run(
        main._save_video_upload(
            Upload(),
            tmp_path / "videos",
            ".mp4",
            pin_after_save=True,
        )
    )

    assert target.read_bytes() == b"video-bytes"
    assert [event[0] for event in events] == [
        "reserve",
        "fsync-parent",
        "grow",
        "fsync-parent",
        "pin",
        "close",
    ]
    lease.close()
    assert events[-1] == ("unpin",)


def test_downstream_failure_removes_target_and_rolls_back_reservation(
    tmp_path,
    monkeypatch,
):
    events = []

    class Reservation:
        def grow(self, _size):
            pass

        def close(self, *, success, actual_bytes=None):
            events.append(("close", success, actual_bytes))

    class Lease:
        def close(self):
            events.append(("unpin",))

    class Upload:
        def __init__(self):
            self._chunks = [b"corrupt-video", b""]

        async def read(self, _size):
            return self._chunks.pop(0)

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda _target, _size: Reservation(),
    )
    monkeypatch.setattr(main, "pin_media_paths", lambda _paths: Lease())
    monkeypatch.setattr(
        main,
        "fsync_parent_directory",
        lambda target: events.append(("fsync-parent", target)),
    )

    pending = asyncio.run(
        main._stage_video_upload(
            Upload(),
            tmp_path / "videos",
            ".mp4",
        )
    )
    target = pending.target
    assert target.exists()

    pending.rollback()
    pending.close_pin()

    assert not target.exists()
    assert ("close", False, None) in events
    assert events[-1] == ("unpin",)


def test_claim_marker_failure_removes_exact_owned_upload(
    tmp_path,
    monkeypatch,
):
    closes = []

    class Reservation:
        def claim_created_path(self, _path):
            raise OSError("claim journal unavailable")

        def close(self, *, success, actual_bytes=None):
            closes.append((success, actual_bytes))

    class Upload:
        async def read(self, _size):
            raise AssertionError("claim must fail before the first read")

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda _target, _size: Reservation(),
    )

    video_dir = tmp_path / "videos"
    with pytest.raises(OSError, match="claim journal unavailable"):
        asyncio.run(
            main._stage_video_upload(
                Upload(),
                video_dir,
                ".mp4",
                create_pin=False,
            )
        )

    assert list(video_dir.iterdir()) == []
    assert closes == [(False, None)]


def test_upload_rollback_preserves_foreign_replacement(tmp_path, monkeypatch):
    class Reservation:
        def grow(self, _size):
            return None

        def close(self, *, success, actual_bytes=None):
            return None

    class Upload:
        def __init__(self):
            self._chunks = [b"owned", b""]

        async def read(self, _size):
            return self._chunks.pop(0)

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda _target, _size: Reservation(),
    )

    pending = asyncio.run(
        main._stage_video_upload(
            Upload(),
            tmp_path / "videos",
            ".mp4",
            create_pin=False,
        )
    )
    replacement = tmp_path / "replacement.mp4"
    replacement.write_bytes(b"foreign")
    replacement.replace(pending.target)

    pending.rollback()

    assert pending.target.read_bytes() == b"foreign"


def test_commit_marker_failure_keeps_upload_retryable(tmp_path, monkeypatch):
    root = tmp_path / "storage"
    video_dir = root / "videos"
    video_dir.mkdir(parents=True)
    policy = {
        "storage_root": root,
        "media_roots": (video_dir,),
        "limit_bytes": 1024,
        "action": "stop",
    }

    class Lease:
        def close(self):
            return None

    class Upload:
        def __init__(self):
            self._chunks = [b"video", b""]

        async def read(self, _size):
            return self._chunks.pop(0)

    monkeypatch.setattr(
        main,
        "begin_media_write",
        lambda target, size: storage_policy.begin_media_write(
            target,
            size,
            **policy,
        ),
    )
    monkeypatch.setattr(main, "pin_media_paths", lambda _paths: Lease())
    pending = asyncio.run(
        main._stage_video_upload(Upload(), video_dir, ".mp4")
    )
    real_write = storage_policy._atomic_journal_write

    def fail_commit_marker(path, payload):
        if payload.get("status") == "committed":
            raise OSError("commit marker unavailable")
        return real_write(path, payload)

    monkeypatch.setattr(
        storage_policy,
        "_atomic_journal_write",
        fail_commit_marker,
    )
    with pytest.raises(storage_policy.StoragePolicyError):
        pending.commit()
    assert pending.committed is False
    assert pending._reservation._closed is False

    monkeypatch.setattr(
        storage_policy,
        "_atomic_journal_write",
        real_write,
    )
    pending.commit()
    assert pending.committed is True
    assert pending.target.read_bytes() == b"video"
    pending.close_pin()
