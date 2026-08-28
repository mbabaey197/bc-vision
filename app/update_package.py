from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import shutil
import stat
import subprocess
import sys
import zipfile


MAX_UPDATE_ZIP_BYTES = 64 * 1024 * 1024
MAX_UPDATE_EXE_BYTES = 48 * 1024 * 1024
UPDATE_NAME = re.compile(
    r"^BCVision_RC(?P<rc>\d+(?:\.\d+)*)_Update\.exe$",
)
CHECKSUM_LINE = re.compile(
    r"^(?P<hash>[0-9A-Fa-f]{64})\s+\*?(?P<name>[^/\\]+)$",
)


class UpdatePackageError(ValueError):
    pass


@dataclass(frozen=True)
class StagedUpdate:
    executable: Path
    version_label: str
    sha256: str


def validate_update_target(update: StagedUpdate, current_version: str) -> None:
    match = re.fullmatch(
        r"(?P<core>\d+\.\d+\.\d+)-rc(?P<rc>\d+(?:\.\d+)*)",
        current_version,
    )
    if match is None:
        raise UpdatePackageError("Current application version is invalid")
    current_rc = tuple(int(part) for part in match.group("rc").split("."))
    target_rc = tuple(
        int(part) for part in update.version_label[2:].split(".")
    )
    if target_rc[:1] != current_rc[:1]:
        raise UpdatePackageError("Update belongs to a different full base")
    width = max(len(current_rc), len(target_rc))
    current_key = current_rc + (0,) * (width - len(current_rc))
    target_key = target_rc + (0,) * (width - len(target_rc))
    if target_key <= current_key:
        raise UpdatePackageError("Update is not newer than this version")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _safe_member(info: zipfile.ZipInfo) -> str:
    name = info.filename
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    if (
        not name
        or "\\" in name
        or path.is_absolute()
        or ".." in path.parts
        or len(path.parts) != 1
        or info.is_dir()
        or stat.S_ISLNK(mode)
    ):
        raise UpdatePackageError("Update ZIP contains an unsafe entry")
    return path.name


def _expected_hash(checksums: bytes, executable_name: str) -> str:
    try:
        text = checksums.decode("ascii")
    except UnicodeDecodeError as exc:
        raise UpdatePackageError("Checksum file is not ASCII") from exc
    matches = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parsed = CHECKSUM_LINE.fullmatch(line)
        if parsed and parsed.group("name") == executable_name:
            matches.append(parsed.group("hash").upper())
    if len(matches) != 1:
        raise UpdatePackageError(
            "Checksum file must contain exactly one updater hash",
        )
    return matches[0]


def stage_update_zip(
    archive: Path,
    staging_root: Path,
) -> StagedUpdate:
    archive = Path(archive)
    staging_root = Path(staging_root)
    if not archive.is_file() or archive.is_symlink():
        raise UpdatePackageError("Update ZIP is unavailable")
    size = archive.stat().st_size
    if size <= 0 or size > MAX_UPDATE_ZIP_BYTES:
        raise UpdatePackageError("Update ZIP size is invalid")
    try:
        bundle = zipfile.ZipFile(archive)
    except (OSError, zipfile.BadZipFile) as exc:
        raise UpdatePackageError("Update ZIP is corrupt") from exc
    with bundle:
        entries = {_safe_member(info): info for info in bundle.infolist()}
        if len(entries) != len(bundle.infolist()):
            raise UpdatePackageError("Update ZIP contains duplicate entries")
        executable_names = [
            name for name in entries if UPDATE_NAME.fullmatch(name)
        ]
        if set(entries) != {*executable_names, "SHA256SUMS.txt"}:
            raise UpdatePackageError(
                "Update ZIP must contain only one updater and checksums",
            )
        if len(executable_names) != 1:
            raise UpdatePackageError(
                "Update ZIP must contain exactly one updater",
            )
        executable_name = executable_names[0]
        executable_info = entries[executable_name]
        if (
            executable_info.file_size <= 0
            or executable_info.file_size > MAX_UPDATE_EXE_BYTES
            or executable_info.compress_size < 0
        ):
            raise UpdatePackageError("Updater size is invalid")
        checksums_info = entries["SHA256SUMS.txt"]
        if checksums_info.file_size <= 0 or checksums_info.file_size > 64 * 1024:
            raise UpdatePackageError("Checksum file size is invalid")
        expected = _expected_hash(
            bundle.read(checksums_info),
            executable_name,
        )
        token = secrets.token_hex(12)
        destination = staging_root / token
        destination.mkdir(parents=True, exist_ok=False)
        temporary = destination / (executable_name + ".tmp")
        executable = destination / executable_name
        try:
            with bundle.open(executable_info) as source, temporary.open("xb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            if temporary.stat().st_size != executable_info.file_size:
                raise UpdatePackageError("Updater extraction is incomplete")
            actual = _sha256(temporary)
            if actual != expected:
                raise UpdatePackageError("Updater SHA-256 does not match")
            temporary.replace(executable)
        except Exception:
            shutil.rmtree(destination, ignore_errors=True)
            raise
    label = UPDATE_NAME.fullmatch(executable_name).group("rc")
    return StagedUpdate(executable, f"RC{label}", expected)


def launch_staged_update(update: StagedUpdate) -> None:
    executable = str(update.executable.resolve())
    parameters = "/SP- /SILENT /SUPPRESSMSGBOXES /NORESTART"
    if os.name == "nt":
        import ctypes

        result = ctypes.windll.shell32.ShellExecuteW(
            None,
            "runas",
            executable,
            parameters,
            str(update.executable.parent),
            1,
        )
        if int(result) <= 32:
            raise OSError(f"Updater elevation failed ({int(result)})")
        return
    subprocess.Popen(
        [executable, *parameters.split()],
        cwd=update.executable.parent,
        close_fds=True,
    )


def exit_after_update_launch(delay: float = 2.0) -> None:
    import time

    time.sleep(delay)
    os._exit(0)
