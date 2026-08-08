from fastapi.responses import HTMLResponse

from app.offline_license_policy import _offline_page


def test_offline_license_page_removes_online_form():
    def endpoint():
        return HTMLResponse(
            "<p>فعال‌سازی امن آنلاین یا آفلاین BC Vision</p>"
            "<div class='card'><h3>فعال‌سازی آنلاین</h3>"
            "<form><input></form></div>"
            "<label>محتوای فایل license.json</label>"
        )

    response = _offline_page(endpoint)()
    text = response.body.decode("utf-8")
    assert "فعال‌سازی آنلاین" not in text
    assert "license.json" not in text
    assert "license.dat" in text
    assert "کاملاً آفلاین" in text


def test_license_activation_routes_are_not_exposed():
    import app.main as main

    paths = {
        route.path
        for route in main.app.routes
    }
    for path in (
        "/license",
        "/license/online",
        "/license/offline",
        "/license/deactivate",
    ):
        assert path not in paths
