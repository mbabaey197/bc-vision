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
    assert result["web_app_ready"] is True
    assert "public_key_ready" not in result


def test_release_version_metadata_stays_consistent():
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text(encoding="utf-8").strip()

    config = (root / "app" / "config.py").read_text(encoding="utf-8")
    assert f'APP_VERSION = "{version}"' in config

    base_version, rc_version = version.split("-rc", 1)
    version_info = f"{base_version}.{rc_version.replace('.', '')}"
    release_label = f"RC{rc_version}"
    for filename, suffix in (
        ("BCVision.iss", "Setup"),
        ("BCVision_Update.iss", "Update"),
    ):
        source = (
            root / "installer" / filename
        ).read_text(encoding="utf-8")
        assert f'#define MyAppVersion "{version}"' in source
        assert (
            f"OutputBaseFilename=BCVision_{release_label}_{suffix}"
            in source
        )
        assert f"VersionInfoVersion={version_info}" in source


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
    assert "read_plate_hezar_primary(" in launcher
    assert "read_plate_crnn(" in launcher
    assert "warmup_cnn(" in launcher
    assert '--add-data ".model-seed;model-seed"' in build
    assert "--collect-all av" in build
    assert "--collect-all onnx" in build
    assert "--collect-all onnxruntime" in build
    assert "--hidden-import app.ai.next_engine" in build
    assert "--hidden-import app.ai.next_models" in build
    assert "--hidden-import app.ai.onnx_cct" in build
    assert "--hidden-import app.ai.onnx_hezar" in build
    assert "--hidden-import app.ai.review_policy" in build
    assert "--collect-all easyocr" not in build
    assert "--collect-all ultralytics" not in build
    assert "--exclude-module hezar" in build
    assert 'copy /Y "THIRD_PARTY_NOTICES.md"' in build


def test_windows_gate_runs_detector_and_ocr_inside_installed_executable():
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "windows-release-candidate.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count('"--verify-anpr"') == 2
    assert "firstJson.anpr_ready" in workflow
    assert "updatedJson.anpr_ready" in workflow
