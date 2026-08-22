import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime_payload import (
    CURRENT_MARKER,
    FAILED_MARKER,
    LAST_KNOWN_GOOD_MARKER,
    MANIFEST_NAME,
    PENDING_MARKER,
    PREVIOUS_MARKER,
    RuntimePayloadError,
    atomic_write_runtime_marker,
    build_runtime_payload,
    compare_runtime_versions,
    read_runtime_marker,
    recover_pending_activation,
    select_runtime_payload,
    validate_fast_update_version,
    verify_runtime_payload,
)


def _source(root: Path, version: str = "2.2.0-rc29.1") -> Path:
    (root / "app" / "ai").mkdir(parents=True)
    (root / "app" / "__init__.py").write_text("", encoding="utf-8")
    (root / "app" / "main.py").write_text(
        "VALUE = 'verified'\n",
        encoding="utf-8",
    )
    (root / "app" / "ai" / "README.md").write_text(
        "runtime asset\n",
        encoding="utf-8",
    )
    (root / "app" / "__pycache__").mkdir()
    (root / "app" / "__pycache__" / "ignored.pyc").write_bytes(b"pyc")
    (root / "VERSION").write_text(version + "\n", encoding="utf-8")
    (root / "RUNTIME_ABI").write_text("1\n", encoding="utf-8")
    return root


def test_build_runtime_payload_is_small_complete_and_verified(tmp_path):
    source = _source(tmp_path / "source")
    result = build_runtime_payload(source, tmp_path / "payloads")

    assert result.version == "2.2.0-rc29.1"
    assert result.runtime_abi == "1"
    assert result.file_count == 3
    assert (result.root / "app" / "main.py").is_file()
    assert not (result.root / "app" / "__pycache__").exists()
    manifest = json.loads(
        (result.root / MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert sorted(manifest["files"]) == [
        "app/__init__.py",
        "app/ai/README.md",
        "app/main.py",
    ]


def test_runtime_payload_rejects_corruption_and_unverified_files(tmp_path):
    source = _source(tmp_path / "source")
    result = build_runtime_payload(source, tmp_path / "payloads")
    (result.root / "app" / "main.py").write_text(
        "VALUE = 'tampered'\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimePayloadError, match="corrupt"):
        verify_runtime_payload(result.root)

    result = build_runtime_payload(source, tmp_path / "payloads")
    (result.root / "app" / "unverified.py").write_text(
        "UNVERIFIED = True\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimePayloadError, match="unverified"):
        verify_runtime_payload(result.root)


def test_runtime_selector_rolls_back_from_bad_current_payload(tmp_path):
    source = _source(tmp_path / "source")
    install = tmp_path / "installed"
    runtime = install / "runtime"
    current = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29.2",
        runtime_abi="1",
    )
    previous = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("1\n", encoding="utf-8")
    (runtime / "current.txt").write_text(
        current.version + "\n",
        encoding="utf-8",
    )
    (runtime / "previous.txt").write_text(
        previous.version + "\n",
        encoding="utf-8",
    )
    (current.root / "app" / "main.py").unlink()

    selected = select_runtime_payload(install)

    assert selected is not None
    assert selected.version == previous.version


def test_runtime_selector_rejects_wrong_abi_and_unsafe_pointer(tmp_path):
    source = _source(tmp_path / "source")
    install = tmp_path / "installed"
    runtime = install / "runtime"
    build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("2\n", encoding="utf-8")
    (runtime / "current.txt").write_text(
        "../../outside\n",
        encoding="utf-8",
    )

    assert select_runtime_payload(install) is None


def test_pending_activation_selects_last_good_without_marker_writes(tmp_path):
    source = _source(tmp_path / "source")
    install = tmp_path / "installed"
    runtime = install / "runtime"
    stable = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29",
        runtime_abi="1",
    )
    candidate = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("1\n", encoding="ascii")
    atomic_write_runtime_marker(install, PREVIOUS_MARKER, stable.version)
    atomic_write_runtime_marker(
        install,
        LAST_KNOWN_GOOD_MARKER,
        stable.version,
    )
    atomic_write_runtime_marker(install, PENDING_MARKER, candidate.version)
    atomic_write_runtime_marker(install, CURRENT_MARKER, candidate.version)

    assert recover_pending_activation(install) == candidate.version
    selected = select_runtime_payload(install)
    assert selected is not None
    assert selected.version == stable.version
    assert read_runtime_marker(install, CURRENT_MARKER) == candidate.version
    assert (
        read_runtime_marker(install, LAST_KNOWN_GOOD_MARKER)
        == stable.version
    )
    assert read_runtime_marker(install, FAILED_MARKER) == ""
    assert read_runtime_marker(install, PENDING_MARKER) == candidate.version


def test_confirmed_pending_activation_is_selected_without_marker_writes(
    tmp_path,
):
    source = _source(tmp_path / "source")
    install = tmp_path / "installed"
    candidate = build_runtime_payload(
        source,
        install / "runtime",
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("1\n", encoding="ascii")
    for marker in (CURRENT_MARKER, LAST_KNOWN_GOOD_MARKER, PENDING_MARKER):
        atomic_write_runtime_marker(install, marker, candidate.version)

    assert recover_pending_activation(install) == ""
    selected = select_runtime_payload(install)
    assert selected is not None
    assert selected.version == candidate.version
    assert read_runtime_marker(install, CURRENT_MARKER) == candidate.version
    assert read_runtime_marker(install, PENDING_MARKER) == candidate.version
    assert read_runtime_marker(install, FAILED_MARKER) == ""


def test_pending_recovery_survives_denied_program_files_writes(
    tmp_path,
    monkeypatch,
):
    source = _source(tmp_path / "source")
    install = tmp_path / "Program Files" / "BC Vision"
    runtime = install / "runtime"
    stable = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29",
        runtime_abi="1",
    )
    candidate = build_runtime_payload(
        source,
        runtime,
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("1\n", encoding="ascii")
    atomic_write_runtime_marker(install, CURRENT_MARKER, candidate.version)
    atomic_write_runtime_marker(install, PENDING_MARKER, candidate.version)
    atomic_write_runtime_marker(
        install,
        LAST_KNOWN_GOOD_MARKER,
        stable.version,
    )
    atomic_write_runtime_marker(install, PREVIOUS_MARKER, stable.version)

    def denied(*_args, **_kwargs):
        raise PermissionError("Program Files is not writable")

    monkeypatch.setattr("runtime_payload.atomic_write_runtime_marker", denied)
    monkeypatch.setattr("runtime_payload.remove_runtime_marker", denied)

    selected = select_runtime_payload(install)

    assert selected is not None
    assert selected.version == stable.version
    assert read_runtime_marker(install, CURRENT_MARKER) == candidate.version
    assert read_runtime_marker(install, PENDING_MARKER) == candidate.version


def test_requested_candidate_does_not_consume_pending_transaction(tmp_path):
    source = _source(tmp_path / "source")
    install = tmp_path / "installed"
    candidate = build_runtime_payload(
        source,
        install / "runtime",
        version="2.2.0-rc29.1",
        runtime_abi="1",
    )
    (install / "runtime-abi.txt").write_text("1\n", encoding="ascii")
    atomic_write_runtime_marker(install, PENDING_MARKER, candidate.version)

    selected = select_runtime_payload(
        install,
        requested_version=candidate.version,
    )

    assert selected is not None
    assert selected.version == candidate.version
    assert read_runtime_marker(install, PENDING_MARKER) == candidate.version


def test_fast_update_version_is_a_numeric_child_of_base():
    validate_fast_update_version("2.2.0-rc29", "2.2.0-rc29.1")
    validate_fast_update_version("2.2.0-rc29", "2.2.0-rc29.10")

    for invalid in (
        "2.2.0-rc28.9",
        "2.2.0-rc30",
        "2.3.0-rc29.1",
        "2.2.0-rc29.01",
        "2.2.0-rc29.0",
    ):
        with pytest.raises(RuntimePayloadError):
            validate_fast_update_version("2.2.0-rc29", invalid)


def test_runtime_versions_use_padded_dotted_numeric_ordering():
    assert compare_runtime_versions(
        "2.2.0-rc29.10",
        "2.2.0-rc29.9",
    ) > 0
    assert compare_runtime_versions(
        "2.2.0-rc29.1",
        "2.2.0-rc29.1.0",
    ) == 0
    assert compare_runtime_versions(
        "2.2.0-rc28.9",
        "2.2.0-rc29",
    ) < 0


def test_external_finder_wins_over_a_frozen_app_importer(tmp_path):
    source = _source(tmp_path / "source")
    payload = build_runtime_payload(source, tmp_path / "payloads")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["BCVISION_TEST_PAYLOAD"] = str(payload.root)
    script = r"""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import sys

from runtime_payload import install_runtime_importer, verify_runtime_payload

class BundledLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.VALUE = 'bundled'

class BundledFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'app' or fullname.startswith('app.'):
            return importlib.util.spec_from_loader(
                fullname,
                BundledLoader(),
                is_package=fullname == 'app',
            )
        return None

sys.meta_path.insert(0, BundledFinder())
payload = verify_runtime_payload(Path(os.environ['BCVISION_TEST_PAYLOAD']))
install_runtime_importer(payload)
from app.main import VALUE
assert VALUE == 'verified'
assert Path(sys.modules['app.main'].__file__).resolve().is_relative_to(
    payload.root,
)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_runtime_importer_does_not_expose_unmanifested_root_module(tmp_path):
    source = _source(tmp_path / "source")
    payload = build_runtime_payload(source, tmp_path / "payloads")
    (payload.root / "unverified_dependency.py").write_text(
        "raise RuntimeError('unverified root module executed')\n",
        encoding="utf-8",
    )
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["BCVISION_TEST_PAYLOAD"] = str(payload.root)
    script = r"""
import importlib
import os
from pathlib import Path
import sys

from runtime_payload import install_runtime_importer, verify_runtime_payload

payload = verify_runtime_payload(Path(os.environ['BCVISION_TEST_PAYLOAD']))
install_runtime_importer(payload)
assert str(payload.root) not in sys.path
try:
    importlib.import_module('unverified_dependency')
except ModuleNotFoundError:
    pass
else:
    raise AssertionError('unmanifested payload-root module was imported')
from app.main import VALUE
assert VALUE == 'verified'
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_release_workflows_separate_fast_and_full_runtime_paths():
    root = Path(__file__).resolve().parents[1]
    fast = (
        root / ".github" / "workflows" / "fast-one-click-update.yml"
    ).read_text(encoding="utf-8")
    full = (
        root / ".github" / "workflows" / "rc27-hosted-fallback.yml"
    ).read_text(encoding="utf-8")
    candidate = (
        root / ".github" / "workflows" / "windows-release-candidate.yml"
    ).read_text(encoding="utf-8")
    installer = (
        root / "installer" / "BCVision_Fast_Update.iss"
    ).read_text(encoding="utf-8")

    assert "timeout-minutes: 4" in fast
    assert "timeout-minutes: 20" in fast
    assert "timeout-minutes: 45" in fast
    assert "timeout-minutes: 1" in fast
    assert "BUILD_PORTABLE_EXE.bat" not in fast
    assert "Wait for successful exact-SHA ANPR validation" in fast
    assert "AddMinutes(18)" in fast
    assert "210" not in fast
    assert "  push:" in fast
    assert "      - VERSION" in fast
    assert "Verify real base Setup to transactional fast update" in fast
    assert "function Invoke-CheckedProcess" in fast
    assert "WaitForExit($TimeoutSeconds * 1000)" in fast
    assert "taskkill.exe /PID $process.Id /T /F" in fast
    assert "fast-base-integration-diagnostics" in fast
    assert "[byte[]](77, 90)" not in fast
    assert "RUNTIME_CONTRACT_ID.txt" in fast
    assert "git merge-base --is-ancestor" in fast
    assert "--require-newer-than" in fast
    assert "--paginate" in fast
    assert "exact-tag resume is refused" in fast
    assert "10MB" in fast
    assert "15MB" in fast
    assert "BC_Vision_v" not in fast
    assert "workflow_dispatch:" in full
    assert "  push:" in full
    assert '      - "RUNTIME_CONTRACT.json"' in full
    assert "workflow_dispatch:" in candidate
    assert "  push:" not in candidate
    assert "Compression=lzma2/fast" in installer
    assert "runtime-abi.txt" in installer
    assert "runtime-contract.txt" in installer
    assert "MyRuntimeContract" in installer
    assert "MyBaseVersion" in installer
    assert "last-known-good.txt" in installer
    assert "pending.txt" in installer
    assert "failed.txt" in installer
    assert "--runtime-candidate" in installer
    assert "--self-test-data-dir" in installer
    assert "MoveFileEx" in installer
    assert "CompareRuntimeVersions" in installer
    assert "CheckNoDowngrade" in installer
    assert "FindVerifiedRollback" in installer
    assert "No different verified runtime" in installer
    assert installer.count("AtomicWriteMarker('previous.txt'") == 1
    assert "Assert-ExactReleaseAssets" in fast
    assert "Get-FileHash $localChecksums" in fast
    assert "Refusing to mutate" in fast
    assert "v2.2.0-rc28.1" in full
    assert "BCVision_RC28.1_Setup.exe" in full
    assert "5032e5a5801af5368af4c85476cbefbd1cd563e1" in full
    assert "E64DFCC90D8C9D17742591C7254D7176CB088C60F38C58C5930D42CAA529C2EC" in full
    assert "RC28.1 Setup pinned SHA-256 verification failed" in full
    assert "Assert-ReleaseAssetsMatchChecksums" in full
    assert "Get-FileHash $localChecksums" in full
    assert "Refusing to mutate" in full
    assert "BCVISION_DATA_DIR" not in installer
    assert "bcvision.db" not in installer
    assert "bcvision/fast-release" in fast
    assert "bcvision/full-release" in full
    assert "statuses: write" in fast
    assert "statuses: write" in full
    assert 'state="pending"' in fast
    assert 'state="pending"' in full
    assert "Publish final fast-release status" in fast
    assert "Publish final full-release status" in full


def test_legacy_update_batch_only_launches_official_transactional_updater():
    root = Path(__file__).resolve().parents[1]
    updater = (root / "UPDATE_EXISTING_INSTALL.bat").read_text(
        encoding="utf-8",
    )
    lowered = updater.lower()

    assert "xcopy" not in lowered
    assert "copy /y" not in lowered
    assert "del /q" not in lowered
    assert "bcvision_%rc_label%_update.exe" in lowered
    assert 'start "" /wait "%updater%"' in lowered
    assert "update_result=%errorlevel%" in lowered
    assert "هیچ فایلی را مستقیم کپی نمی‌کند" in updater


def test_committed_runtime_contract_matches_stable_base_files():
    root = Path(__file__).resolve().parents[1]
    contract = json.loads(
        (root / "RUNTIME_CONTRACT.json").read_text(encoding="utf-8")
    )
    runtime_abi = (root / "RUNTIME_ABI").read_text(
        encoding="utf-8"
    ).strip()
    base_version = (root / "FAST_UPDATE_BASE_VERSION").read_text(
        encoding="utf-8"
    ).strip()
    assert "app/database.py" in contract["files"]
    completed = subprocess.run(
        [sys.executable, "scripts/verify_runtime_contract.py"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (
        f"ABI {runtime_abi} base {base_version} verified"
        in completed.stdout
    )


def test_release_contract_cli_enforces_monotonic_fast_versions():
    root = Path(__file__).resolve().parents[1]
    base_version = (root / "FAST_UPDATE_BASE_VERSION").read_text(
        encoding="utf-8"
    ).strip()
    command = [
        sys.executable,
        "scripts/verify_runtime_contract.py",
        "--validate-update-version",
        f"{base_version}.10",
        "--require-newer-than",
        f"{base_version}.9",
    ]
    accepted = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    rejected = subprocess.run(
        [*command[:-1], f"{base_version}.11"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert rejected.returncode != 0
    assert "must be newer" in rejected.stderr
