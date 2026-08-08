from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_product_activation_modules_and_assets_are_absent():
    for relative in (
        "app/license.py",
        "app/license_format.py",
        "app/offline_license_policy.py",
        "license_public_key.pem",
        "tools/generate_license.py",
        "tools/init_license_keys.py",
        "tests/test_license.py",
        "tests/test_offline_license_policy.py",
    ):
        assert not (ROOT / relative).exists(), relative


def test_runtime_and_packaging_have_no_product_activation_gate():
    checks = {
        "app/main.py": (
            "from app.license",
            "license.manage",
            "@app.get('/license')",
            "@app.post('/license')",
            "camera_limit",
        ),
        "app/config.py": (
            "LICENSE_PATH",
            "LEGACY_LICENSE_PATH",
            "TRIAL_PATH",
            "PUBLIC_KEY_PATH",
            "offline_license_policy",
        ),
        "launcher.py": ("PUBLIC_KEY_PATH", "public_key_ready"),
        "BUILD_PORTABLE_EXE.bat": (
            "--hidden-import app.license",
            "license_public_key.pem",
        ),
        ".github/workflows/rc27-hosted-fallback.yml": (
            "BCVISION_EXPERIMENTAL_NO_LICENSE",
        ),
    }
    for relative, forbidden in checks.items():
        source = (ROOT / relative).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{relative}: {token}"


def test_legacy_source_snapshot_has_no_product_activation_code():
    legacy = ROOT / "BCVision_2.2.0-rc18_FastVideo_Source"
    if not legacy.exists():
        return
    for relative in (
        "app/license.py",
        "app/license_format.py",
        "app/offline_license_policy.py",
        "license_public_key.pem",
        "tools/generate_license.py",
        "tools/init_license_keys.py",
        "tests/test_license.py",
        "tests/test_offline_license_policy.py",
    ):
        assert not (legacy / relative).exists(), relative
    source = (legacy / "app" / "main.py").read_text(encoding="utf-8")
    assert "from app.license" not in source
    assert "@app.get('/license')" not in source
    assert "camera_limit" not in source
