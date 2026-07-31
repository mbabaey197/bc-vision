import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.verify_windows_gui_subsystem import (
    IMAGE_SUBSYSTEM_WINDOWS_GUI,
    read_subsystem,
)


def test_launcher_self_test_uses_isolated_data_directory(tmp_path):
    output_path = tmp_path / "self-test.json"
    data_dir = tmp_path / "persistent-data"
    env = os.environ.copy()
    env.update({
        "BCVISION_DATA_DIR": str(data_dir),
        "BCVISION_SKIP_MODEL_PREP": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })

    completed = subprocess.run(
        [
            sys.executable,
            "launcher.py",
            "--self-test",
            "--self-test-output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result["ok"] is True
    expected_version = (
        Path(__file__).resolve().parents[1] / "VERSION"
    ).read_text(encoding="utf-8").strip()
    assert result["version"] == expected_version
    assert Path(result["data_dir"]).resolve() == data_dir.resolve()
    assert Path(result["database_path"]).is_file()
    assert result["database_ready"] is True
    assert result["public_key_ready"] is True
    assert result["web_app_ready"] is True


def test_release_version_metadata_stays_consistent():
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text(encoding="utf-8").strip()

    config = (root / "app" / "config.py").read_text(encoding="utf-8")
    assert f'APP_VERSION = "{version}"' in config

    for filename, prefix in (
        ("BCVision.iss", "BCVision_Setup_v"),
        ("BCVision_Update.iss", "BCVision_Update_v"),
    ):
        source = (
            root / "installer" / filename
        ).read_text(encoding="utf-8")
        assert f'#define MyAppVersion "{version}"' in source
        assert f"OutputBaseFilename={prefix}{version}" in source
        assert "VersionInfoVersion=2.2.0.17" in source
        assert "PrivilegesRequiredOverridesAllowed=commandline" in source


def test_windows_build_and_source_launch_are_windowless():
    root = Path(__file__).resolve().parents[1]
    build = (root / "BUILD_PORTABLE_EXE.bat").read_text(
        encoding="utf-8",
    )
    source_launcher = (root / "RUN_SOURCE.bat").read_text(
        encoding="utf-8",
    )

    assert "--windowed" in build
    assert "pythonw.exe" in source_launcher
    assert 'start "" /B' in source_launcher
    assert "python launcher.py" not in source_launcher


def test_pe_subsystem_reader_accepts_gui_executable(tmp_path):
    executable = tmp_path / "BCVision.exe"
    image = bytearray(512)
    image[:2] = b"MZ"
    image[0x3C:0x40] = (0x80).to_bytes(4, "little")
    image[0x80:0x84] = b"PE\0\0"
    optional_header = 0x80 + 4 + 20
    image[optional_header:optional_header + 2] = (0x20B).to_bytes(
        2,
        "little",
    )
    image[
        optional_header + 68:optional_header + 70
    ] = IMAGE_SUBSYSTEM_WINDOWS_GUI.to_bytes(2, "little")
    executable.write_bytes(image)

    assert read_subsystem(executable) == IMAGE_SUBSYSTEM_WINDOWS_GUI


def test_packaged_self_test_can_require_offline_anpr_models():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "launcher.py").read_text(encoding="utf-8")
    build = (root / "BUILD_PORTABLE_EXE.bat").read_text(encoding="utf-8")

    assert '"--verify-anpr" in sys.argv' in launcher
    assert "prepare_models(download=False)" in launcher
    assert "detect_plates_onnx(" in launcher
    assert "read_plate_crnn(" in launcher
    assert "warmup_cnn(" in launcher
    assert '--add-data ".model-seed;model-seed"' in build
    assert "--collect-all av" in build
    assert "--collect-all onnx" in build
    assert "--collect-all onnxruntime" in build
    assert "--hidden-import app.ai.next_engine" in build
    assert "--hidden-import app.ai.next_models" in build
    assert "--hidden-import app.ai.onnx_cct" in build
    assert "--hidden-import app.ai.review_policy" in build
    assert "--collect-all easyocr" not in build
    assert "--collect-all ultralytics" not in build
    assert 'copy /Y "THIRD_PARTY_NOTICES.md"' in build


def test_windows_gate_runs_detector_and_ocr_inside_installed_executable():
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "windows-release-candidate.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count('"--verify-anpr"') == 2
    assert "firstJson.anpr_ready" in workflow
    assert "updatedJson.anpr_ready" in workflow


def test_fast_updater_reuses_build_cache_and_skips_full_installer():
    root = Path(__file__).resolve().parents[1]
    build = (root / "BUILD_PORTABLE_EXE.bat").read_text(
        encoding="utf-8",
    )
    workflow = (
        root / ".github" / "workflows" / "windows-fast-updater.yml"
    ).read_text(encoding="utf-8")

    assert "BCVISION_INCREMENTAL_BUILD" in build
    assert "build_dependency_stamp.py check" in build
    assert 'set "BCVISION_PYINSTALLER_CLEAN=--clean"' in build
    assert 'set "BCVISION_PYINSTALLER_CLEAN="' in build
    assert "clean: false" in workflow
    assert 'BCVISION_INCREMENTAL_BUILD: "1"' in workflow
    assert "Build only the one-click updater" in workflow
    assert "installer\\BCVision_Update.iss" in workflow
    assert "installer\\BCVision.iss" not in workflow
    assert "BCVision_Setup_v" not in workflow
    assert "compression-level: 0" in workflow


def test_full_windows_release_is_manual_and_non_cancelling():
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "windows-release-candidate.yml"
    ).read_text(encoding="utf-8")

    trigger_block = workflow.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger_block
    assert "push:" not in trigger_block
    assert "pull_request:" not in trigger_block
    assert "cancel-in-progress: false" in workflow
    assert "ensure_inno_setup.ps1" in workflow


def test_fast_update_uses_reusable_verified_packaging_tools():
    root = Path(__file__).resolve().parents[1]
    inno = (root / "scripts" / "ensure_inno_setup.ps1").read_text(
        encoding="utf-8",
    )
    verifier = (root / "scripts" / "verify_fast_update.ps1").read_text(
        encoding="utf-8",
    )
    updater = (
        root / "installer" / "BCVision_Update.iss"
    ).read_text(encoding="utf-8")

    assert "RUNNER_TOOL_CACHE" in inno
    assert "Get-AuthenticodeSignature" in inno
    assert 'Status -ne "Valid"' in inno
    assert '"--verify-anpr"' in verifier
    assert "Database marker was not preserved" in verifier
    assert "AI model marker was not preserved" in verifier
    assert "Compression=lzma2/fast" in updater
    assert "SolidCompression=no" in updater
