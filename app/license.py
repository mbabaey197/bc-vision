from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import secrets
import subprocess
import uuid
from datetime import date, timedelta
from pathlib import Path

from app.config import LICENSE_PATH, PUBLIC_KEY_PATH, TRIAL_PATH

PLAN_CAMERA_LIMITS = {
    "trial": 2,
    "basic": 2,
    "professional": 8,
    "enterprise": 64,
}
PLAN_FEATURES = {
    "trial": ["anpr", "events", "reports"],
    "basic": ["anpr", "events", "reports"],
    "professional": [
        "anpr",
        "events",
        "reports",
        "vehicle_ai",
        "watchlist",
        "api",
    ],
    "enterprise": [
        "anpr",
        "events",
        "reports",
        "vehicle_ai",
        "watchlist",
        "api",
        "gate",
        "multi_site",
        "priority_support",
    ],
}

_STATE_PATH = LICENSE_PATH.with_name(".license-state.json")
_STATE_KEY_PATH = LICENSE_PATH.with_name(".license-state.key")
_CLOCK_ROLLBACK_TOLERANCE_DAYS = 1


def _hidden_process_kwargs() -> dict:
    """Prevent Windows hardware-ID probes from flashing a console window."""
    if platform.system().lower() != "windows":
        return {}
    kwargs = {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
    }
    startupinfo_type = getattr(subprocess, "STARTUPINFO", None)
    if startupinfo_type is not None:
        startupinfo = startupinfo_type()
        startupinfo.dwFlags |= getattr(
            subprocess,
            "STARTF_USESHOWWINDOW",
            0,
        )
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        kwargs["startupinfo"] = startupinfo
    return kwargs


def _windows_hardware_uuid() -> str:
    if platform.system().lower() != "windows":
        return ""
    commands = [
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_ComputerSystemProduct).UUID",
        ],
        ["wmic", "csproduct", "get", "uuid"],
    ]
    for command in commands:
        try:
            output = subprocess.check_output(
                command,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
                **_hidden_process_kwargs(),
            )
            values = [
                item.strip()
                for item in output.splitlines()
                if item.strip() and "uuid" not in item.lower()
            ]
            if values:
                value = values[0].upper()
                if value not in {
                    "00000000-0000-0000-0000-000000000000",
                    "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",
                }:
                    return value
        except Exception:
            pass
    return ""


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32].upper()


def machine_id_candidates() -> list[str]:
    """Return stable and legacy IDs so existing licenses keep working."""
    system = platform.system()
    machine = platform.machine()
    hardware_uuid = _windows_hardware_uuid()
    candidates: list[str] = []
    if hardware_uuid:
        candidates.append(_digest([system, machine, hardware_uuid]))
    # Compatibility ID used by earlier BC Vision releases.
    legacy_parts = [system, machine, platform.node(), str(uuid.getnode())]
    if hardware_uuid:
        legacy_parts.append(hardware_uuid)
    candidates.append(_digest(legacy_parts))
    # Non-Windows or restricted systems still receive a deterministic fallback.
    candidates.append(_digest([system, machine, platform.node(), str(uuid.getnode())]))
    return list(dict.fromkeys(candidates))


def machine_id() -> str:
    return machine_id_candidates()[0]


def _canonical(payload: dict) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verify_signature(payload: dict, signature_b64: str) -> bool:
    if not PUBLIC_KEY_PATH.exists():
        return False
    try:
        from cryptography.hazmat.primitives import serialization

        public_key = serialization.load_pem_public_key(
            PUBLIC_KEY_PATH.read_bytes()
        )
        public_key.verify(
            base64.b64decode(signature_b64, validate=True),
            _canonical(payload),
        )
        return True
    except Exception:
        return False


def _state_key() -> bytes:
    _STATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _STATE_KEY_PATH.exists():
        key = _STATE_KEY_PATH.read_bytes()
        if len(key) >= 32:
            return key
    key = secrets.token_bytes(32)
    temp = _STATE_KEY_PATH.with_suffix(".tmp")
    temp.write_bytes(key)
    temp.replace(_STATE_KEY_PATH)
    try:
        os.chmod(_STATE_KEY_PATH, 0o600)
    except OSError:
        pass
    return key


def _signed_state(data: dict) -> dict:
    signature = hmac.new(
        _state_key(),
        _canonical(data),
        hashlib.sha256,
    ).hexdigest()
    return {"data": data, "signature": signature}


def _write_state(data: dict, path: Path = _STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(_signed_state(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp.replace(path)


def _read_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        data = document["data"]
        signature = str(document["signature"])
        expected = hmac.new(
            _state_key(),
            _canonical(data),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        return data
    except Exception:
        return None


def _clock_is_valid(today: date, license_id: str) -> bool:
    state = _read_state(_STATE_PATH)
    if state is not None:
        try:
            last_seen = date.fromisoformat(str(state["last_seen"]))
            if today < last_seen - timedelta(
                days=_CLOCK_ROLLBACK_TOLERANCE_DAYS
            ):
                return False
        except Exception:
            return False
    _write_state(
        {
            "last_seen": today.isoformat(),
            "license_id": str(license_id or ""),
            "machine_id": machine_id(),
        }
    )
    return True


def _invalid_trial(message: str) -> dict:
    return {
        "valid": False,
        "mode": "invalid",
        "plan": "trial",
        "customer": "نسخه آزمایشی",
        "license_id": "TRIAL",
        "issued_at": "—",
        "expires_at": "—",
        "days_left": 0,
        "camera_limit": PLAN_CAMERA_LIMITS["trial"],
        "features": PLAN_FEATURES["trial"],
        "message": message,
    }


def _trial() -> dict:
    today = date.today()
    current_machine = machine_id()
    if TRIAL_PATH.exists():
        trial = _read_state(TRIAL_PATH)
        if trial is None:
            return _invalid_trial("اطلاعات نسخه آزمایشی دست‌کاری یا خراب شده است")
        try:
            started = date.fromisoformat(str(trial["started"]))
            last_seen = date.fromisoformat(str(trial["last_seen"]))
            stored_machine = str(trial["machine_id"]).upper()
        except Exception:
            return _invalid_trial("اطلاعات نسخه آزمایشی ناقص است")
        if stored_machine not in machine_id_candidates():
            return _invalid_trial("نسخه آزمایشی متعلق به این دستگاه نیست")
        if today < last_seen - timedelta(days=_CLOCK_ROLLBACK_TOLERANCE_DAYS):
            return _invalid_trial("تاریخ سیستم به عقب بازگردانده شده است")
    else:
        started = today
    expires = started + timedelta(days=30)
    active = today <= expires
    _write_state(
        {
            "started": started.isoformat(),
            "last_seen": today.isoformat(),
            "machine_id": current_machine,
        },
        TRIAL_PATH,
    )
    return {
        "valid": active,
        "mode": "trial",
        "plan": "trial",
        "customer": "نسخه آزمایشی",
        "license_id": "TRIAL",
        "issued_at": started.isoformat(),
        "expires_at": expires.isoformat(),
        "days_left": max(0, (expires - today).days),
        "camera_limit": PLAN_CAMERA_LIMITS["trial"],
        "features": PLAN_FEATURES["trial"],
        "message": (
            "نسخه آزمایشی فعال است"
            if active
            else "مهلت نسخه آزمایشی پایان یافته است"
        ),
    }


def status() -> dict:
    if not LICENSE_PATH.exists():
        return _trial()
    try:
        document = json.loads(LICENSE_PATH.read_text(encoding="utf-8"))
        payload = document["payload"]
        signature = document["signature"]
        if not isinstance(payload, dict) or not _verify_signature(
            payload,
            signature,
        ):
            return {
                **_invalid_trial("امضای لایسنس معتبر نیست"),
                "mode": "invalid",
            }
        allowed = [
            str(item).upper()
            for item in payload.get("machine_ids", [])
            if item
        ]
        single = str(payload.get("machine_id", "")).upper()
        if single:
            allowed.append(single)
        if not set(machine_id_candidates()).intersection(allowed):
            return {
                **_invalid_trial("لایسنس متعلق به این دستگاه نیست"),
                "mode": "invalid",
            }
        plan = str(payload.get("plan", "basic")).lower()
        if plan not in PLAN_CAMERA_LIMITS:
            return {
                **_invalid_trial("پلن لایسنس شناخته‌شده نیست"),
                "mode": "invalid",
            }
        expires_raw = payload.get("expires_at")
        perpetual = not expires_raw or str(expires_raw).lower() in {
            "never",
            "perpetual",
            "lifetime",
        }
        expires = None if perpetual else date.fromisoformat(str(expires_raw))
        today = date.today()
        license_id = str(payload.get("license_id", "—"))
        if not _clock_is_valid(today, license_id):
            return {
                **_invalid_trial("تاریخ سیستم به عقب بازگردانده شده است"),
                "mode": "invalid",
            }
        if expires and today > expires:
            return {
                **_invalid_trial("اعتبار لایسنس پایان یافته است"),
                "mode": "expired",
            }
        camera_limit = int(
            payload.get("camera_limit")
            or PLAN_CAMERA_LIMITS.get(plan, 2)
        )
        if camera_limit < 1 or camera_limit > 4096:
            return {
                **_invalid_trial("محدودیت دوربین لایسنس معتبر نیست"),
                "mode": "invalid",
            }
        features = payload.get("features") or PLAN_FEATURES[plan]
        if not isinstance(features, list) or not all(
            isinstance(item, str) for item in features
        ):
            return {
                **_invalid_trial("قابلیت‌های لایسنس معتبر نیست"),
                "mode": "invalid",
            }
        return {
            "valid": True,
            "mode": "licensed",
            "plan": plan,
            "customer": payload.get("customer", "—"),
            "license_id": license_id,
            "issued_at": payload.get("issued_at", "—"),
            "expires_at": "دائمی" if perpetual else expires.isoformat(),
            "days_left": (
                99999
                if perpetual
                else max(0, (expires - today).days)
            ),
            "camera_limit": camera_limit,
            "features": list(dict.fromkeys(features)),
            "message": (
                "لایسنس دائمی معتبر است"
                if perpetual
                else "لایسنس معتبر است"
            ),
        }
    except Exception:
        return {
            **_invalid_trial("فایل لایسنس خراب یا ناقص است"),
            "mode": "invalid",
        }


def camera_capacity(current_count: int, requested: int = 1) -> tuple[bool, str]:
    """Backend license gate shared by every camera-creation path."""
    result = status()
    if not result.get("valid"):
        return False, str(result.get("message") or "لایسنس معتبر نیست")
    try:
        current = max(0, int(current_count))
        added = max(1, int(requested))
        limit = int(result["camera_limit"])
    except Exception:
        return False, "اطلاعات ظرفیت لایسنس معتبر نیست"
    if current + added > limit:
        return (
            False,
            f"حداکثر تعداد دوربین در پلن {result['plan']} برابر {limit} است.",
        )
    return True, ""


def install_license(raw: str) -> tuple[bool, str]:
    try:
        document = json.loads(raw)
        if (
            not isinstance(document.get("payload"), dict)
            or not document.get("signature")
        ):
            return False, "ساختار لایسنس صحیح نیست"
        LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = LICENSE_PATH.with_suffix(".tmp")
        temp.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        old = LICENSE_PATH.read_bytes() if LICENSE_PATH.exists() else None
        old_state = _STATE_PATH.read_bytes() if _STATE_PATH.exists() else None
        temp.replace(LICENSE_PATH)
        _STATE_PATH.unlink(missing_ok=True)
        result = status()
        if not result["valid"]:
            if old is None:
                LICENSE_PATH.unlink(missing_ok=True)
            else:
                LICENSE_PATH.write_bytes(old)
            if old_state is None:
                _STATE_PATH.unlink(missing_ok=True)
            else:
                _STATE_PATH.write_bytes(old_state)
            return False, result["message"]
        return True, "لایسنس با موفقیت فعال شد"
    except Exception:
        return False, "متن لایسنس JSON معتبر نیست"


def activate_online(
    server_url: str,
    activation_code: str,
) -> tuple[bool, str]:
    server_url = server_url.strip().rstrip("/")
    activation_code = activation_code.strip()
    if not server_url.startswith("https://"):
        return False, "سرور فعال‌سازی باید از HTTPS معتبر استفاده کند"
    if not activation_code:
        return False, "کد فعال‌سازی وارد نشده است"
    try:
        import httpx

        response = httpx.post(
            server_url + "/api/v1/activate",
            json={
                "activation_code": activation_code,
                "machine_id": machine_id(),
                "machine_ids": machine_id_candidates(),
                "product": "bc-vision",
            },
            timeout=15,
            follow_redirects=False,
        )
        if response.status_code != 200:
            try:
                message = (
                    response.json().get("message")
                    or response.json().get("detail")
                )
            except Exception:
                message = None
            return False, message or (
                f"خطای سرور فعال‌سازی ({response.status_code})"
            )
        data = response.json()
        raw = json.dumps(data.get("license") or data, ensure_ascii=False)
        return install_license(raw)
    except Exception:
        return False, "ارتباط امن با سرور فعال‌سازی برقرار نشد"


def deactivate_local() -> tuple[bool, str]:
    if not LICENSE_PATH.exists():
        return False, "لایسنسی برای حذف وجود ندارد"
    try:
        LICENSE_PATH.unlink()
        _STATE_PATH.unlink(missing_ok=True)
        return True, "لایسنس از این دستگاه حذف شد"
    except Exception:
        return False, "حذف فایل لایسنس امکان‌پذیر نیست"


def has_feature(name: str) -> bool:
    result = status()
    return bool(result.get("valid")) and name in result.get("features", [])
