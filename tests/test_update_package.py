import hashlib
from pathlib import Path
import zipfile

import pytest

from app.update_package import (
    UpdatePackageError,
    stage_update_zip,
    validate_update_target,
)


def _bundle(path: Path, *, name="BCVision_RC31.1_Update.exe", payload=b"exe"):
    digest = hashlib.sha256(payload).hexdigest().upper()
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, payload)
        archive.writestr("SHA256SUMS.txt", f"{digest}  {name}\n")
    return digest


def test_stages_exact_verified_updater(tmp_path):
    archive = tmp_path / "update.zip"
    expected = _bundle(archive)
    result = stage_update_zip(archive, tmp_path / "staged")
    assert result.executable.read_bytes() == b"exe"
    assert result.version_label == "RC31.1"
    assert result.sha256 == expected


def test_rejects_zip_slip(tmp_path):
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../BCVision_RC31.1_Update.exe", b"exe")
        bundle.writestr("SHA256SUMS.txt", b"0" * 64)
    with pytest.raises(UpdatePackageError, match="unsafe"):
        stage_update_zip(archive, tmp_path / "staged")


def test_rejects_hash_mismatch(tmp_path):
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("BCVision_RC31.1_Update.exe", b"exe")
        bundle.writestr(
            "SHA256SUMS.txt",
            "0" * 64 + "  BCVision_RC31.1_Update.exe\n",
        )
    with pytest.raises(UpdatePackageError):
        stage_update_zip(archive, tmp_path / "staged")


def test_rejects_unlisted_extra_file(tmp_path):
    archive = tmp_path / "update.zip"
    digest = _bundle(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr("extra.dll", digest)
    with pytest.raises(UpdatePackageError, match="only one updater"):
        stage_update_zip(archive, tmp_path / "staged")


def test_target_must_be_newer_child_of_current_full_base(tmp_path):
    archive = tmp_path / "update.zip"
    _bundle(archive, name="BCVision_RC31.2_Update.exe")
    staged = stage_update_zip(archive, tmp_path / "staged")
    validate_update_target(staged, "2.2.0-rc31.1")
    with pytest.raises(UpdatePackageError, match="not newer"):
        validate_update_target(staged, "2.2.0-rc31.2")
    with pytest.raises(UpdatePackageError, match="different full base"):
        validate_update_target(staged, "2.2.0-rc30.9")
