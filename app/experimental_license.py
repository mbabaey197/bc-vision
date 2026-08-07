from __future__ import annotations

import os


def install_experimental_license_override() -> None:
    if os.environ.get("BCVISION_EXPERIMENTAL_NO_LICENSE", "1") == "0":
        return
    try:
        from app import license as license_module
    except Exception:
        return

    if getattr(license_module, "_bcvision_experimental_no_license", False):
        return

    def status() -> dict:
        features = sorted(
            {
                feature
                for rows in license_module.PLAN_FEATURES.values()
                for feature in rows
            }
        )
        return {
            "valid": True,
            "mode": "experimental",
            "plan": "experimental",
            "customer": "نسخه آزمایشی",
            "license_id": "RC27-EXPERIMENTAL",
            "issued_at": "—",
            "expires_at": "دائمی",
            "days_left": 99999,
            "camera_limit": 4096,
            "features": features,
            "message": "نسخه آزمایشی بدون نیاز به لایسنس فعال است",
        }

    def camera_capacity(current_count: int, requested: int = 1):
        del current_count, requested
        return True, ""

    def runtime_camera_allowed(camera_id: int) -> bool:
        del camera_id
        return True

    def has_feature(name: str) -> bool:
        return name in status()["features"]

    license_module.status = status
    license_module.camera_capacity = camera_capacity
    license_module.runtime_camera_allowed = runtime_camera_allowed
    license_module.has_feature = has_feature
    license_module._bcvision_experimental_no_license = True
