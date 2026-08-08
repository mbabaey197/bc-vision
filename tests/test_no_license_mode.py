import json
import os
import subprocess
import sys


def test_packaged_default_runs_without_license(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BCVISION_DATA_DIR"] = str(tmp_path)
    env.pop("BCVISION_NO_LICENSE", None)
    env.pop("BCVISION_LICENSE_REGRESSION", None)
    license_path = tmp_path / "license.dat"
    state_path = tmp_path / ".license-state.dat"
    key_path = tmp_path / ".license-state.key"
    license_path.write_bytes(b"stale-invalid-license")
    state_path.write_bytes(b"stale-invalid-state")
    key_path.write_bytes(b"stale-invalid-key")
    script = r"""
import json
from app import license
from app.config import LICENSE_PATH

status = license.status()
capacity_ok, capacity_message = license.camera_capacity(4095, 1)
result = {
    "license_exists": LICENSE_PATH.exists(),
    "license_bytes": LICENSE_PATH.read_bytes().decode("ascii"),
    "state_bytes": LICENSE_PATH.with_name(".license-state.dat").read_bytes().decode("ascii"),
    "key_bytes": LICENSE_PATH.with_name(".license-state.key").read_bytes().decode("ascii"),
    "valid": status.get("valid"),
    "mode": status.get("mode"),
    "expires_at": status.get("expires_at"),
    "camera_limit": status.get("camera_limit"),
    "anpr": license.has_feature("anpr"),
    "future_feature": license.has_feature("future.feature"),
    "runtime": license.runtime_camera_allowed(999999),
    "capacity_ok": capacity_ok,
    "capacity_message": capacity_message,
}
print(json.dumps(result))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout.strip())
    assert result == {
        "license_exists": True,
        "license_bytes": "stale-invalid-license",
        "state_bytes": "stale-invalid-state",
        "key_bytes": "stale-invalid-key",
        "valid": True,
        "mode": "no-license",
        "expires_at": None,
        "camera_limit": 2147483647,
        "anpr": True,
        "future_feature": True,
        "runtime": True,
        "capacity_ok": True,
        "capacity_message": "",
    }


def test_old_disable_environment_cannot_restore_packaged_enforcement(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["BCVISION_DATA_DIR"] = str(tmp_path)
    env["BCVISION_NO_LICENSE"] = "0"
    env.pop("BCVISION_LICENSE_REGRESSION", None)
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app import license; s=license.status(); "
            "raise SystemExit(0 if s.get('mode') == 'no-license' "
            "and license.runtime_camera_allowed(999999) else 1)",
        ],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
