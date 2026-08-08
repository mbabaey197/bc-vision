from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_windows_build_embeds_and_verifies_no_license_mode():
    build = (ROOT / "BUILD_PORTABLE_EXE.bat").read_text(encoding="utf-8")
    workflow = (
        ROOT / ".github" / "workflows" / "windows-fast-updater.yml"
    ).read_text(encoding="utf-8")
    verifier = (ROOT / "scripts" / "verify_fast_update.ps1").read_text(
        encoding="utf-8"
    )

    assert "--hidden-import app.no_license" in build
    assert "BCVISION_NO_LICENSE:" not in workflow
    assert 'BCVISION_LICENSE_REGRESSION: "1"' in workflow
    assert 'license_public_key.pem was not found' not in build
    assert 'copy /Y "license_public_key.pem"' not in build
    assert verifier.count('"--verify-no-license"') == 2
    assert verifier.count("no_license_ready") == 2


def test_rc28_installer_names_and_versions_are_consistent():
    version = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    full = (ROOT / "installer" / "BCVision.iss").read_text(encoding="utf-8")
    update = (ROOT / "installer" / "BCVision_Update.iss").read_text(
        encoding="utf-8"
    )

    assert version == "2.2.0-rc28"
    for source in (full, update):
        assert '#define MyAppVersion "2.2.0-rc28"' in source
        assert "VersionInfoVersion=2.2.0.28" in source
    assert "OutputBaseFilename=BCVision_Update_v2.2.0-rc28" in update
