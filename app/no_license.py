from __future__ import annotations

import os
import sys


_TEST_SWITCH = "BCVISION_LICENSE_REGRESSION"


class _AllFeatures(list):
    def __contains__(self, item) -> bool:
        return bool(str(item or "").strip())


def _legacy_regression_requested() -> bool:
    """Expose the preserved implementation only to source pytest runs."""
    return (
        os.environ.get(_TEST_SWITCH, "").strip() == "1"
        and "pytest" in sys.modules
        and not getattr(sys, "frozen", False)
    )


def install_no_license_mode() -> bool:
    """Temporarily remove all runtime license enforcement.

    The legacy implementation stays in app.license solely for source regression
    tests. A packaged customer runtime cannot re-enable it with an environment
    variable. No license file is created, migrated, validated, or required.
    """
    if _legacy_regression_requested():
        return False

    from app import license as license_module

    if getattr(license_module, "_bcvision_no_license_mode", False):
        return True

    features = _AllFeatures(sorted(
        {
            feature
            for rows in license_module.PLAN_FEATURES.values()
            for feature in rows
        }
    ))

    def status() -> dict:
        return {
            "valid": True,
            "mode": "no-license",
            "plan": "enterprise",
            "customer": "نسخه بدون لایسنس",
            "license_id": "DISABLED-RC28",
            "issued_at": None,
            "expires_at": None,
            "days_left": 99999,
            "camera_limit": 2147483647,
            "features": features,
            "license_enforcement": False,
            "message": "سیستم لایسنس موقتاً غیرفعال است",
        }

    def camera_capacity(current_count: int, requested: int = 1):
        del current_count, requested
        return True, ""

    def runtime_camera_allowed(camera_id: int) -> bool:
        del camera_id
        return True

    def has_feature(name: str) -> bool:
        return bool(str(name or "").strip())

    def install_license(raw):
        del raw
        return True, "سیستم لایسنس غیرفعال است و نیازی به فعال‌سازی نیست"

    def activate_online(server_url: str, activation_code: str):
        del server_url, activation_code
        return True, "سیستم لایسنس غیرفعال است و نیازی به فعال‌سازی نیست"

    def deactivate_local():
        return True, "سیستم لایسنس از قبل غیرفعال است"

    license_module.status = status
    license_module.camera_capacity = camera_capacity
    license_module.runtime_camera_allowed = runtime_camera_allowed
    license_module.has_feature = has_feature
    license_module.install_license = install_license
    license_module.activate_online = activate_online
    license_module.deactivate_local = deactivate_local
    license_module._bcvision_no_license_mode = True
    return True
