import json
import os
from pathlib import Path
import subprocess
import sys


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
