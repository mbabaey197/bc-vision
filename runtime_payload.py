from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
from uuid import uuid4


MANIFEST_NAME = "runtime-manifest.json"
RUNTIME_ABI_FILE = "RUNTIME_ABI"
RUNTIME_ABI_INSTALL_FILE = "runtime-abi.txt"
CURRENT_MARKER = "current.txt"
PREVIOUS_MARKER = "previous.txt"
LAST_KNOWN_GOOD_MARKER = "last-known-good.txt"
PENDING_MARKER = "pending.txt"
FAILED_MARKER = "failed.txt"
VERSION_PATTERN = re.compile(
    r"^\d+\.\d+\.\d+-rc\d+(?:\.\d+)*$",
)
ABI_PATTERN = re.compile(r"^[1-9]\d*$")


class RuntimePayloadError(ValueError):
    """Raised when a runtime payload is incomplete or unsafe to activate."""


@dataclass(frozen=True)
class RuntimePayload:
    root: Path
    version: str
    runtime_abi: str
    file_count: int


class ExternalAppFinder(MetaPathFinder):
    """Load only ``app`` modules from a verified external payload first."""

    def __init__(self, payload_root: Path):
        self.payload_root = Path(payload_root).resolve()
        self.app_root = self.payload_root / "app"

    def find_spec(
        self,
        fullname: str,
        path=None,
        target=None,
    ) -> ModuleSpec | None:
        if fullname == "app":
            search_path = [str(self.payload_root)]
        elif fullname.startswith("app."):
            search_path = path
        else:
            return None
        spec = PathFinder.find_spec(fullname, search_path, target)
        if spec is None or spec.origin is None:
            return None
        try:
            origin = Path(spec.origin).resolve()
            origin.relative_to(self.app_root)
        except (OSError, ValueError):
            return None
        return spec


def install_runtime_importer(payload: RuntimePayload) -> ExternalAppFinder:
    """Put the verified app finder ahead of PyInstaller's frozen finder."""
    for finder in sys.meta_path:
        if isinstance(finder, ExternalAppFinder):
            if finder.payload_root == payload.root:
                return finder
            raise RuntimePayloadError("A different runtime is already active")
    finder = ExternalAppFinder(payload.root)
    sys.meta_path.insert(0, finder)
    return finder


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_marker(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return ""


def read_runtime_marker(install_root: Path, marker: str) -> str:
    """Read one runtime marker without exposing arbitrary filesystem paths."""
    if marker not in {
        CURRENT_MARKER,
        PREVIOUS_MARKER,
        LAST_KNOWN_GOOD_MARKER,
        PENDING_MARKER,
        FAILED_MARKER,
    }:
        raise RuntimePayloadError(f"Unsupported runtime marker: {marker!r}")
    return _read_marker(Path(install_root).resolve() / "runtime" / marker)


def atomic_write_runtime_marker(
    install_root: Path,
    marker: str,
    version: str,
) -> Path:
    """Durably replace a version marker in the installed runtime directory."""
    version = _validated_version(version)
    runtime_root = Path(install_root).resolve() / "runtime"
    runtime_root.mkdir(parents=True, exist_ok=True)
    if marker not in {
        CURRENT_MARKER,
        PREVIOUS_MARKER,
        LAST_KNOWN_GOOD_MARKER,
        PENDING_MARKER,
        FAILED_MARKER,
    }:
        raise RuntimePayloadError(f"Unsupported runtime marker: {marker!r}")
    destination = runtime_root / marker
    temporary = runtime_root / f".{marker}.{uuid4().hex}.tmp"
    try:
        with temporary.open("w", encoding="ascii", newline="\n") as handle:
            handle.write(version + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def remove_runtime_marker(install_root: Path, marker: str) -> None:
    """Remove a known marker; absence is already the requested state."""
    if marker not in {
        CURRENT_MARKER,
        PREVIOUS_MARKER,
        LAST_KNOWN_GOOD_MARKER,
        PENDING_MARKER,
        FAILED_MARKER,
    }:
        raise RuntimePayloadError(f"Unsupported runtime marker: {marker!r}")
    (Path(install_root).resolve() / "runtime" / marker).unlink(
        missing_ok=True,
    )


def _validated_version(value: str) -> str:
    value = str(value).strip()
    if not VERSION_PATTERN.fullmatch(value):
        raise RuntimePayloadError(f"Invalid runtime version: {value!r}")
    return value


def version_parts(value: str) -> tuple[tuple[int, int, int], tuple[int, ...]]:
    """Return numeric release components with unambiguous RC ordering."""
    value = _validated_version(value)
    base_text, rc_text = value.split("-rc", 1)
    base_tokens = base_text.split(".")
    rc_tokens = rc_text.split(".")
    if any(
        len(token) > 1 and token.startswith("0")
        for token in (*base_tokens, *rc_tokens)
    ):
        raise RuntimePayloadError("Version components cannot have leading zeros")
    return (
        tuple(int(token) for token in base_tokens),
        tuple(int(token) for token in rc_tokens),
    )


def _compare_numeric_parts(
    left: tuple[int, ...],
    right: tuple[int, ...],
) -> int:
    """Compare dotted numeric parts with absent trailing zeroes equivalent."""
    width = max(len(left), len(right))
    left_padded = left + (0,) * (width - len(left))
    right_padded = right + (0,) * (width - len(right))
    return (left_padded > right_padded) - (left_padded < right_padded)


def compare_runtime_versions(left: str, right: str) -> int:
    """Compare validated RC versions using dotted numeric ordering."""
    left_core, left_rc = version_parts(left)
    right_core, right_rc = version_parts(right)
    core_result = _compare_numeric_parts(left_core, right_core)
    if core_result:
        return core_result
    return _compare_numeric_parts(left_rc, right_rc)


def validate_fast_update_version(base_version: str, version: str) -> None:
    """Require a newer child RC in the exact full-runtime release train."""
    base_core, base_rc = version_parts(base_version)
    update_core, update_rc = version_parts(version)
    if update_core != base_core:
        raise RuntimePayloadError("Fast update and base release trains differ")
    if (
        len(update_rc) <= len(base_rc)
        or update_rc[:len(base_rc)] != base_rc
        or compare_runtime_versions(version, base_version) <= 0
    ):
        raise RuntimePayloadError(
            "Fast update must be a newer child of the full base version",
        )


def _validated_abi(value: str) -> str:
    value = str(value).strip()
    if not ABI_PATTERN.fullmatch(value):
        raise RuntimePayloadError(f"Invalid runtime ABI: {value!r}")
    return value


def _safe_manifest_path(value: str) -> PurePosixPath:
    value = str(value)
    path = PurePosixPath(value)
    if (
        "\\" in value
        or path.is_absolute()
        or not path.parts
        or path.parts[0] != "app"
        or any(
            part in {"", ".", ".."} or ":" in part
            for part in path.parts
        )
    ):
        raise RuntimePayloadError(f"Unsafe runtime path: {value!r}")
    return path


def _payload_source_files(app_root: Path) -> list[Path]:
    files = []
    for path in app_root.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.as_posix())


def build_runtime_payload(
    source_root: Path,
    output_root: Path,
    *,
    version: str | None = None,
    runtime_abi: str | None = None,
) -> RuntimePayload:
    """Build a small, complete application payload for one-click updates."""
    source_root = Path(source_root).resolve()
    output_root = Path(output_root).resolve()
    version = _validated_version(
        version or _read_marker(source_root / "VERSION"),
    )
    runtime_abi = _validated_abi(
        runtime_abi or _read_marker(source_root / RUNTIME_ABI_FILE),
    )
    app_root = source_root / "app"
    if not (app_root / "__init__.py").is_file():
        raise RuntimePayloadError("The app package is missing")

    destination = output_root / version
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    hashes: dict[str, str] = {}
    for source in _payload_source_files(app_root):
        relative = source.relative_to(source_root)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[relative.as_posix()] = _sha256(target)

    if not hashes:
        raise RuntimePayloadError("The runtime payload is empty")

    manifest = {
        "schema": 1,
        "version": version,
        "runtime_abi": runtime_abi,
        "files": hashes,
    }
    manifest_path = destination / MANIFEST_NAME
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return verify_runtime_payload(
        destination,
        expected_version=version,
        expected_abi=runtime_abi,
    )


def verify_runtime_payload(
    payload_root: Path,
    *,
    expected_version: str | None = None,
    expected_abi: str | None = None,
) -> RuntimePayload:
    payload_root = Path(payload_root).resolve()
    manifest_path = payload_root / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimePayloadError("Runtime manifest is unreadable") from exc

    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        raise RuntimePayloadError("Unsupported runtime manifest schema")
    version = _validated_version(manifest.get("version", ""))
    runtime_abi = _validated_abi(manifest.get("runtime_abi", ""))
    if payload_root.name != version:
        raise RuntimePayloadError("Runtime directory and version disagree")
    if expected_version is not None and version != _validated_version(
        expected_version,
    ):
        raise RuntimePayloadError("Runtime version does not match")
    if expected_abi is not None and runtime_abi != _validated_abi(
        expected_abi,
    ):
        raise RuntimePayloadError("Runtime ABI does not match")

    hashes = manifest.get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise RuntimePayloadError("Runtime manifest has no files")

    expected_paths: set[str] = set()
    for relative_value, expected_hash in hashes.items():
        relative = _safe_manifest_path(relative_value)
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise RuntimePayloadError("Runtime manifest has an invalid hash")
        path = payload_root.joinpath(*relative.parts)
        try:
            path.relative_to(payload_root)
        except ValueError as exc:
            raise RuntimePayloadError("Runtime file escapes its payload") from exc
        if not path.is_file() or path.is_symlink():
            raise RuntimePayloadError(f"Runtime file is missing: {relative}")
        if not hmac.compare_digest(_sha256(path), expected_hash):
            raise RuntimePayloadError(f"Runtime file is corrupt: {relative}")
        expected_paths.add(relative.as_posix())

    actual_paths = {
        path.relative_to(payload_root).as_posix()
        for path in _payload_source_files(payload_root / "app")
    }
    if actual_paths != expected_paths:
        raise RuntimePayloadError("Runtime payload contains unverified files")

    return RuntimePayload(
        root=payload_root,
        version=version,
        runtime_abi=runtime_abi,
        file_count=len(expected_paths),
    )


def _verified_installed_payload(
    install_root: Path,
    version: str,
    runtime_abi: str,
) -> RuntimePayload | None:
    try:
        version = _validated_version(version)
        return verify_runtime_payload(
            Path(install_root).resolve() / "runtime" / version,
            expected_version=version,
            expected_abi=runtime_abi,
        )
    except RuntimePayloadError:
        return None


def recover_pending_activation(install_root: Path) -> str:
    """Report an interrupted transaction without mutating Program Files.

    A confirmed transaction can briefly retain ``pending.txt`` if power was
    lost after the last-known-good marker was written. Every other valid
    pending version is returned so selection can skip it and use a verified
    last-known-good/previous payload in memory. Only the elevated installer
    owns installed marker mutation.
    """
    install_root = Path(install_root).resolve()
    runtime_abi = _read_marker(
        install_root / RUNTIME_ABI_INSTALL_FILE,
    )
    try:
        runtime_abi = _validated_abi(runtime_abi)
    except RuntimePayloadError:
        return ""

    pending = read_runtime_marker(install_root, PENDING_MARKER)
    if not pending:
        return ""
    try:
        pending = _validated_version(pending)
    except RuntimePayloadError:
        return ""

    current = read_runtime_marker(install_root, CURRENT_MARKER)
    last_good = read_runtime_marker(install_root, LAST_KNOWN_GOOD_MARKER)
    if (
        current == pending
        and last_good == pending
        and _verified_installed_payload(
            install_root,
            pending,
            runtime_abi,
        ) is not None
    ):
        return ""
    return pending


def select_runtime_payload(
    install_root: Path,
    *,
    requested_version: str | None = None,
) -> RuntimePayload | None:
    """Select a requested candidate or recover and select a stable payload."""
    install_root = Path(install_root).resolve()
    runtime_abi = _read_marker(
        install_root / RUNTIME_ABI_INSTALL_FILE,
    )
    try:
        runtime_abi = _validated_abi(runtime_abi)
    except RuntimePayloadError:
        return None

    if requested_version is not None:
        return _verified_installed_payload(
            install_root,
            requested_version,
            runtime_abi,
        )

    interrupted = recover_pending_activation(install_root)

    tried: set[str] = {interrupted} if interrupted else set()
    for pointer_name in (
        CURRENT_MARKER,
        LAST_KNOWN_GOOD_MARKER,
        PREVIOUS_MARKER,
    ):
        version = read_runtime_marker(install_root, pointer_name)
        if not version or version in tried:
            continue
        tried.add(version)
        payload = _verified_installed_payload(
            install_root,
            version,
            runtime_abi,
        )
        if payload is None:
            continue
        return payload
    return None
