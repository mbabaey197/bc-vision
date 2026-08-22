import importlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys

import pytest

import app.config as config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _import_config(bootstrap: Path):
    environment = os.environ.copy()
    environment.update({
        "BCVISION_DATA_DIR": str(bootstrap),
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; from app import config; "
                "print(json.dumps({'data_dir': str(config.DATA_DIR)}))"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )


def _assert_no_media_directories(*roots: Path):
    for root in roots:
        for name in ("backups", "snapshots", "plates", "videos"):
            assert not (root / name).exists()


def _assert_no_data_artifacts(*roots: Path):
    for root in roots:
        assert not (root / "bcvision.db").exists()
    _assert_no_media_directories(*roots)


def test_truly_absent_storage_pointer_uses_bootstrap_directory(tmp_path):
    bootstrap = tmp_path / "bootstrap"

    completed = _import_config(bootstrap)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["data_dir"]) == bootstrap.resolve()
    assert all(
        (bootstrap / name).is_dir()
        for name in ("backups", "snapshots", "plates", "videos")
    )


def test_valid_private_storage_pointer_selects_canonical_root(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    pointer = bootstrap / "storage_config.json"
    pointer.write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert Path(payload["data_dir"]) == storage.resolve()
    assert all(
        (storage / name).is_dir()
        for name in ("backups", "snapshots", "plates", "videos")
    )
    marker = bootstrap / config.STORAGE_MIGRATION_MARKER_NAME
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    marker_details = marker.lstat()
    assert stat.S_ISREG(marker_details.st_mode)
    assert marker_details.st_nlink == 1
    if os.name != "nt":
        assert stat.S_IMODE(marker_details.st_mode) & 0o077 == 0
    assert not list(
        bootstrap.glob(f".{config.STORAGE_MIGRATION_MARKER_NAME}.*.tmp")
    )

    restarted = _import_config(bootstrap)

    assert restarted.returncode == 0, restarted.stderr
    restarted_details = marker.lstat()
    assert (
        restarted_details.st_dev,
        restarted_details.st_ino,
    ) == (marker_details.st_dev, marker_details.st_ino)
    _assert_no_data_artifacts(bootstrap)


def test_deleted_pointer_after_valid_migration_never_bootstraps(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    pointer = bootstrap / "storage_config.json"
    pointer.write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    first = _import_config(bootstrap)

    assert first.returncode == 0, first.stderr
    marker = bootstrap / config.STORAGE_MIGRATION_MARKER_NAME
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    for name in ("backups", "snapshots", "plates", "videos"):
        (storage / name).rmdir()
    pointer.unlink()

    second = _import_config(bootstrap)

    assert second.returncode != 0
    assert "StorageConfigurationError" in second.stderr
    assert "missing after storage migration" in second.stderr
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_corrupt_existing_marker_aborts_without_replacement(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )
    marker = bootstrap / config.STORAGE_MIGRATION_MARKER_NAME
    marker.write_bytes(b"foreign-marker")

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    assert marker.read_bytes() == b"foreign-marker"
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_foreign_symlink_marker_is_never_followed_or_removed(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )
    external = tmp_path / "foreign-marker"
    external.write_bytes(config.STORAGE_MIGRATION_MARKER_PAYLOAD)
    marker = bootstrap / config.STORAGE_MIGRATION_MARKER_NAME
    try:
        marker.symlink_to(external)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    assert marker.is_symlink()
    assert external.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_hardlinked_marker_is_not_private(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )
    marker = bootstrap / config.STORAGE_MIGRATION_MARKER_NAME
    marker.write_bytes(config.STORAGE_MIGRATION_MARKER_PAYLOAD)
    try:
        os.link(marker, tmp_path / "foreign-marker-link")
    except OSError:
        pytest.skip("hard links are unavailable")

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    assert marker.read_bytes() == config.STORAGE_MIGRATION_MARKER_PAYLOAD
    assert marker.stat().st_nlink == 2
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_marker_creation_collision_preserves_foreign_entry(
    tmp_path,
    monkeypatch,
):
    marker = tmp_path / config.STORAGE_MIGRATION_MARKER_NAME
    real_link = os.link

    def collide(source, destination, *args, **kwargs):
        assert Path(destination) == marker
        marker.write_bytes(b"foreign-collision")
        raise FileExistsError("marker collision")

    monkeypatch.setattr(os, "link", collide)
    with pytest.raises(
        config.StorageConfigurationError,
        match="marker content is invalid",
    ):
        config._create_storage_migration_marker(marker)
    monkeypatch.setattr(os, "link", real_link)

    assert marker.read_bytes() == b"foreign-collision"
    assert not list(
        tmp_path.glob(f".{config.STORAGE_MIGRATION_MARKER_NAME}.*.tmp")
    )


def test_pointer_to_missing_storage_root_never_bootstraps_it(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "missing-storage"
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    assert not storage.exists()
    _assert_no_data_artifacts(bootstrap)


def test_pointer_to_root_without_database_never_creates_one(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap, storage)


def test_symlinked_storage_root_is_never_followed(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    real_storage = tmp_path / "real-storage"
    real_storage.mkdir()
    (real_storage / "bcvision.db").write_bytes(b"existing-database")
    linked_storage = tmp_path / "linked-storage"
    try:
        linked_storage.symlink_to(real_storage, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(linked_storage.absolute())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(real_storage)


def test_symlinked_storage_database_is_never_followed(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    external_database = tmp_path / "external.db"
    external_database.write_bytes(b"external-database")
    try:
        (storage / "bcvision.db").symlink_to(external_database)
    except OSError:
        pytest.skip("file symlinks are unavailable")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    assert external_database.read_bytes() == b"external-database"
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_hardlinked_storage_database_is_not_private(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    database = storage / "bcvision.db"
    database.write_bytes(b"existing-database")
    try:
        os.link(database, tmp_path / "database-copy.db")
    except OSError:
        pytest.skip("hard links are unavailable")
    (bootstrap / "storage_config.json").write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


@pytest.mark.parametrize(
    "payload",
    [
        b"{",
        b"{}",
        b'{"storage_root":""}',
        b'{"storage_root":"relative/path"}',
        b'{"storage_root":"~/storage"}',
        b'{"storage_root":"/"}',
        b'{"storage_root":"/tmp/storage","unexpected":true}',
        b'{"storage_root":"/tmp/first","storage_root":"/tmp/second"}',
        b'\xff',
    ],
)
def test_invalid_existing_pointer_fails_before_data_directories(
    tmp_path,
    payload,
):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "storage_config.json").write_bytes(payload)

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap, tmp_path / "storage")


def test_oversized_existing_pointer_fails_before_data_directories(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "storage_config.json").write_bytes(
        b" " * (config.MAX_STORAGE_CONFIG_BYTES + 1)
    )

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap)


def test_directory_storage_pointer_fails_before_data_directories(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    (bootstrap / "storage_config.json").mkdir()

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap)


def test_symlinked_storage_pointer_is_never_followed(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    external = tmp_path / "external.json"
    external.write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )
    try:
        (bootstrap / "storage_config.json").symlink_to(external)
    except OSError:
        pytest.skip("file symlinks are unavailable")

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap)
    _assert_no_media_directories(storage)


def test_hardlinked_storage_pointer_is_not_private(tmp_path):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    pointer = bootstrap / "storage_config.json"
    pointer.write_text(
        json.dumps({"storage_root": str((tmp_path / "storage").resolve())}),
        encoding="utf-8",
    )
    try:
        os.link(pointer, tmp_path / "pointer-copy.json")
    except OSError:
        pytest.skip("hard links are unavailable")

    completed = _import_config(bootstrap)

    assert completed.returncode != 0
    assert "StorageConfigurationError" in completed.stderr
    _assert_no_data_artifacts(bootstrap, tmp_path / "storage")


def test_unreadable_pointer_aborts_reload_before_target_creation(
    tmp_path,
    monkeypatch,
):
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    storage = tmp_path / "storage"
    storage.mkdir()
    (storage / "bcvision.db").write_bytes(b"existing-database")
    pointer = bootstrap / "storage_config.json"
    pointer.write_text(
        json.dumps({"storage_root": str(storage.resolve())}),
        encoding="utf-8",
    )
    original_data_dir = os.environ.get("BCVISION_DATA_DIR")
    real_open = os.open

    def denied_open(path, flags, *args, **kwargs):
        if Path(path) == pointer:
            raise PermissionError("pointer is unreadable")
        return real_open(path, flags, *args, **kwargs)

    try:
        monkeypatch.setenv("BCVISION_DATA_DIR", str(bootstrap))
        monkeypatch.setattr(os, "open", denied_open)
        with pytest.raises(RuntimeError, match="could not be read"):
            importlib.reload(config)
        _assert_no_data_artifacts(bootstrap)
        _assert_no_media_directories(storage)
    finally:
        monkeypatch.setattr(os, "open", real_open)
        if original_data_dir is None:
            monkeypatch.delenv("BCVISION_DATA_DIR", raising=False)
        else:
            monkeypatch.setenv("BCVISION_DATA_DIR", original_data_dir)
        importlib.reload(config)
