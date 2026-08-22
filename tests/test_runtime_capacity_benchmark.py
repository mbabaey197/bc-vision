import json
from pathlib import Path
import subprocess
import sys


def test_transport_capacity_benchmark_emits_fail_closed_schema(tmp_path):
    output = tmp_path / "capacity.json"
    completed = subprocess.run(
        [
            sys.executable,
            "tools/benchmark_runtime_capacity.py",
            "transport",
            "--output",
            str(output),
            "--ref",
            "unit-test",
            "--camera-counts",
            "1",
            "--frames-per-camera",
            "2",
            "--source-fps",
            "100",
            "--source-width",
            "160",
            "--source-height",
            "96",
            "--dashboard-fps",
            "100",
            "--preview-width",
            "128",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["schema"] == 1
    assert result["mode"] == "transport"
    assert result["implementation_ref"] == "unit-test"
    assert result["accuracy"]["evaluable"] is False
    assert len(result["scenarios"]) == 1
    scenario = result["scenarios"][0]
    assert scenario["camera_count"] == 1
    assert scenario["decoded_frames"] >= 2
    assert scenario["published_frames"] >= 2
    assert scenario["anpr_submissions"] >= 2
    assert scenario["event_count"] is None
    assert scenario["event_count_evaluable"] is False
    assert scenario["estimated_source_frame_drop"] >= 0
    assert scenario["jpeg_encodes"] >= 1
