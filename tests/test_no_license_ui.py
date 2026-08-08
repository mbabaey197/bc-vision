from pathlib import Path


def test_license_activation_ui_and_camera_gate_are_removed():
    source = (Path(__file__).parents[1] / "app" / "main.py").read_text(
        encoding="utf-8"
    )

    forbidden = (
        "from app.license",
        "license_status()",
        "href='/license'",
        "@app.get('/license')",
        "@app.post('/license')",
        "مدیریت لایسنس",
        "فعال‌سازی آنلاین",
        "فعال‌سازی آفلاین",
        "محدودیت لایسنس",
        "license.manage",
    )
    for token in forbidden:
        assert token not in source

    assert "بدون محدودیت نرم‌افزاری" in source
    assert "بدون لایسنس" in source
    assert "تمام قابلیت‌ها فعال هستند" in source


def test_config_does_not_install_a_license_route_policy():
    source = (Path(__file__).parents[1] / "app" / "config.py").read_text(
        encoding="utf-8"
    )

    assert "offline_license_policy" not in source
    assert "install_offline_license_policy" not in source
