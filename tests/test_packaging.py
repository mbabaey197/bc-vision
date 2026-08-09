import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

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


def test_runtime_candidate_is_rejected_outside_isolated_self_test():
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    for arguments in (
        ["--runtime-candidate", "2.2.0-rc29.1"],
        ["--self-test", "--runtime-candidate", "2.2.0-rc29.1"],
    ):
        completed = subprocess.run(
            [sys.executable, "launcher.py", *arguments],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )

        assert completed.returncode != 0
        assert "explicit --self-test-data-dir" in completed.stderr


def test_release_version_metadata_stays_consistent():
    root = Path(__file__).resolve().parents[1]
    version = (root / "VERSION").read_text(encoding="utf-8").strip()

    config = (root / "app" / "config.py").read_text(encoding="utf-8")
    assert f'APP_VERSION = "{version}"' in config

    full_base_version = (root / "FAST_UPDATE_BASE_VERSION").read_text(
        encoding="utf-8",
    ).strip()
    base_version, rc_version = full_base_version.split("-rc", 1)
    assert "." not in rc_version
    version_info = f"{base_version}.{rc_version}0"
    release_label = f"RC{rc_version}"
    for filename, suffix in (
        ("BCVision.iss", "Setup"),
        ("BCVision_Update.iss", "Update"),
    ):
        source = (
            root / "installer" / filename
        ).read_text(encoding="utf-8")
        assert f'#define MyAppVersion "{full_base_version}"' in source
        assert (
            f"OutputBaseFilename=BCVision_{release_label}_{suffix}"
            in source
        )
        assert f"VersionInfoVersion={version_info}" in source

    # RC28.1 shipped with Windows file version 2.2.0.281. Reserve the final
    # digit zero for the immutable RC29 base; the fast workflow's existing
    # dot-removal mapping then gives RC29.1 the next version, 2.2.0.291.
    previous_version_info = (2, 2, 0, 281)
    base_version_info = tuple(int(part) for part in version_info.split("."))
    first_fast_version_info = (
        *base_version_info[:3],
        int(f"{rc_version}1"),
    )
    assert previous_version_info < base_version_info < first_fast_version_info

    fast_workflow = (
        root / ".github" / "workflows" / "fast-one-click-update.yml"
    ).read_text(encoding="utf-8")
    assert (
        '$versionInfo = "$releaseBase.$($rc.Replace(\'.\', \'\'))"'
        in fast_workflow
    )


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
    assert 'detector_variant="yolo11n"' in launcher
    assert 'detector_variant="yolov8n"' in launcher
    assert 'models["detector_yolov8n"]' in launcher
    assert 'before["detector_yolov8n_ready"]' in launcher
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
    assert "scripts\\build_runtime_payload.py" in build
    assert 'copy /Y "RUNTIME_ABI"' in build
    assert '"RUNTIME_CONTRACT.json"' in build
    assert "runtime-contract.json" in build
    assert "runtime\\current.txt" in build
    assert "runtime\\last-known-good.txt" in build


def test_packaged_launcher_prefers_verified_external_runtime():
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "launcher.py").read_text(encoding="utf-8")

    assert "select_runtime_payload(" in launcher
    assert "requested_version=requested" in launcher
    assert "recover_pending_activation(BASE)" in launcher
    assert "install_runtime_importer(payload)" in launcher
    assert "ACTIVE_RUNTIME = activate_runtime_payload()" in launcher
    assert '"runtime_source"' in launcher
    assert '"--runtime-candidate"' in launcher
    assert "APP_VERSION == requested_candidate" in launcher
    assert '"candidate_ready"' in launcher


def test_full_installers_preserve_verified_newer_runtime_pointers():
    root = Path(__file__).resolve().parents[1]
    guard = (
        root / "installer" / "Runtime_Pointer_Guard.iss"
    ).read_text(encoding="utf-8")
    for filename in ("BCVision.iss", "BCVision_Update.iss"):
        installer = (root / "installer" / filename).read_text(
            encoding="utf-8",
        )
        assert 'Excludes: "runtime\\current.txt,' in installer
        assert "runtime\\last-known-good.txt" in installer
        assert '#include "Runtime_Pointer_Guard.iss"' in installer

    assert "VerifyInstalledRuntimeCandidate" in guard
    assert "RuntimeVersionToActivate" in guard
    assert "--runtime-candidate" in guard
    assert "--self-test-data-dir" in guard
    assert "Downgrade was refused" in guard
    assert "AtomicWriteRuntimeMarker" in guard
    assert "'pending.txt', RuntimeVersionToActivate" in guard
    assert "retained for safe launcher fallback" in guard


def test_windows_gate_runs_detector_and_ocr_inside_installed_executable():
    root = Path(__file__).resolve().parents[1]
    workflow = (
        root / ".github" / "workflows" / "windows-release-candidate.yml"
    ).read_text(encoding="utf-8")
    full_release = (
        root / ".github" / "workflows" / "rc27-hosted-fallback.yml"
    ).read_text(encoding="utf-8")

    assert workflow.count('"--verify-anpr"') == 2
    assert "firstJson.anpr_ready" in workflow
    assert "updatedJson.anpr_ready" in workflow
    assert '"${{ github.ref }}" -ne "refs/heads/main"' in full_release
    assert "RUNTIME_CONTRACT.json" in full_release
    assert "RUNTIME_CONTRACT_ID.txt" in full_release
    assert "storage_config.json" in full_release
    assert "preserve-yolo11n.marker" in full_release
    assert "preserve-yolov8n.marker" in full_release
    assert "v2.2.0-rc28.1" in full_release
    assert "BCVision_RC28.1_Setup.exe" in full_release
    assert "SHA256SUMS.txt" in full_release
    assert "5032e5a5801af5368af4c85476cbefbd1cd563e1" in full_release
    assert "E64DFCC90D8C9D17742591C7254D7176CB088C60F38C58C5930D42CAA529C2EC" in full_release
    assert "immutable RC28.1 to RC29 upgrade" in full_release
    assert "Publish or safely resume exact-SHA full release" in full_release
    assert "bcvision/full-release" in full_release


def test_release_workflows_do_not_persist_checkout_credentials_or_job_tokens():
    root = Path(__file__).resolve().parents[1]
    workflow_names = (
        "anpr-pr-validation.yml",
        "anpr-tests.yml",
        "fast-one-click-update.yml",
        "rc27-hosted-fallback.yml",
        "windows-release-candidate.yml",
    )

    for workflow_name in workflow_names:
        workflow_path = root / ".github" / "workflows" / workflow_name
        workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))
        assert workflow["permissions"]["contents"] == "read"
        for job_name, job in workflow["jobs"].items():
            assert "GH_TOKEN" not in (job.get("env") or {}), (
                workflow_name,
                job_name,
            )
            for step in job.get("steps", []):
                uses = str(step.get("uses", ""))
                if uses.startswith("actions/checkout@"):
                    assert step.get("with", {}).get(
                        "persist-credentials"
                    ) is False, (workflow_name, job_name)
                if "GH_TOKEN" in (step.get("env") or {}):
                    assert "gh " in str(step.get("run", "")), (
                        workflow_name,
                        job_name,
                        step.get("name"),
                    )


def test_full_release_publishes_verified_artifact_on_separate_write_job():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((
        root / ".github" / "workflows" / "rc27-hosted-fallback.yml"
    ).read_text(encoding="utf-8"))
    jobs = workflow["jobs"]
    build = jobs["build-and-verify"]
    publish = jobs["publish-release"]
    status = jobs["publish-status"]

    assert build["permissions"] == {
        "contents": "read",
        "statuses": "write",
    }
    assert publish["permissions"] == {
        "actions": "read",
        "contents": "write",
    }
    assert publish["needs"] == "build-and-verify"
    assert status["needs"] == "publish-release"
    assert status["permissions"] == {"statuses": "write"}

    build_uses = [str(step.get("uses", "")) for step in build["steps"]]
    publish_uses = [str(step.get("uses", "")) for step in publish["steps"]]
    assert "actions/upload-artifact@v4" in build_uses
    assert "actions/download-artifact@v4" in publish_uses
    for step in build["steps"]:
        command = str(step.get("run", ""))
        assert "gh release create" not in command
        assert "gh release upload" not in command
        assert "gh release edit" not in command
    assert any(
        "Assert-ReleaseAssetsMatchChecksums" in str(step.get("run", ""))
        and "gh release edit" in str(step.get("run", ""))
        for step in publish["steps"]
    )


def test_fast_release_grants_write_only_to_isolated_publish_job():
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((
        root / ".github" / "workflows" / "fast-one-click-update.yml"
    ).read_text(encoding="utf-8"))
    jobs = workflow["jobs"]

    assert jobs["build-update"]["permissions"] == {
        "contents": "read",
        "statuses": "write",
    }
    assert jobs["validation-gate"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert jobs["base-integration"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert jobs["publish"]["permissions"] == {
        "actions": "read",
        "contents": "write",
    }
    status = jobs["publish-status"]
    assert status["needs"] == [
        "build-update",
        "validation-gate",
        "base-integration",
        "publish",
    ]
    assert status["if"] == "always()"
    assert status["timeout-minutes"] == 2
    assert status["permissions"] == {"statuses": "write"}
    assert "GH_TOKEN" not in (status.get("env") or {})
    assert len(status["steps"]) == 1
    status_step = status["steps"][0]
    assert status_step["env"] == {"GH_TOKEN": "${{ github.token }}"}
    status_script = status_step["run"]
    for dependency in (
        "needs.build-update.result",
        "needs.validation-gate.result",
        "needs.base-integration.result",
        "needs.publish.result",
    ):
        assert dependency in status_script
    assert '$eligible -eq "false"' in status_script
    assert '$baseIntegrationResult -eq "skipped"' in status_script
    assert '$eligible -eq "true"' in status_script
    assert '$publishResult -eq "success"' in status_script
    assert "steps.release.outcome" not in status_script
    assert all(
        step.get("name") != "Publish final fast-release status"
        for step in jobs["publish"]["steps"]
    )
    assert [
        name
        for name, job in jobs.items()
        if (job.get("permissions") or {}).get("contents") == "write"
    ] == ["publish"]


def test_release_resume_requires_a_resolvable_exact_sha_tag():
    root = Path(__file__).resolve().parents[1]
    workflows = (
        ("rc27-hosted-fallback.yml", "publish-release"),
        ("fast-one-click-update.yml", "publish"),
    )
    guard = "if ($releaseExists -and -not $tagExists)"
    public_resume = "if ($releaseExists) {"

    for workflow_name, job_name in workflows:
        workflow = yaml.safe_load((
            root / ".github" / "workflows" / workflow_name
        ).read_text(encoding="utf-8"))
        publish_script = next(
            str(step.get("run", ""))
            for step in workflow["jobs"][job_name]["steps"]
            if "Publish or safely resume" in str(step.get("name", ""))
        )
        assert guard in publish_script
        assert "has no resolvable exact tag" in publish_script
        assert publish_script.index(guard) < publish_script.index(public_resume)
