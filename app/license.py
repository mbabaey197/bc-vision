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

from app.config import (
    LEGACY_LICENSE_PATH,
    LICENSE_PATH,
    PUBLIC_KEY_PATH,
    TRIAL_PATH,
)
from app.license_format import canonical, decode_document, encode_document

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
_ALL_FEATURES = {
    feature
    for features in PLAN_FEATURES.values()
    for feature in features
}
_STATE_PATH = LICENSE_PATH.with_name(".license-state.dat")
_STATE_KEY_PATH = LICENSE_PATH.with_name(".license-state.key")
_CLOCK_ROLLBACK_TOLERANCE_DAYS = 1


def _hidden_process_kwargs() -> dict:
    if platform.system().lower() != "windows":
        return {}
    kwargs = {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
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


def _windows_probe(script: str) -> str:
    if platform.system().lower() != "windows":
        return ""
    try:
        output = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
            **_hidden_process_kwargs(),
        )
        values = [
            value.strip()
            for value in output.splitlines()
            if value.strip()
            and value.strip().lower() not in {
                "uuid",
                "serialnumber",
                "processorid",
                "machineguid",
            }
        ]
        return values[0].upper() if values else ""
    except Exception:
        return ""


def _windows_hardware_uuid() -> str:
    value = _windows_probe(
        "(Get-CimInstance Win32_ComputerSystemProduct).UUID"
    )
    if value in {
        "00000000-0000-0000-0000-000000000000",
        "FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF",
    }:
        return ""
    return value


def _hardware_claims() -> dict[str, str]:
    claims = {
        "system": platform.system().upper(),
        "machine": platform.machine().upper(),
        "uuid": _windows_hardware_uuid(),
        "board": _windows_probe(
            "(Get-CimInstance Win32_BaseBoard | Select-Object -First 1).SerialNumber"
        ),
        "bios": _windows_probe(
            "(Get-CimInstance Win32_BIOS | Select-Object -First 1).SerialNumber"
        ),
        "cpu": _windows_probe(
            "(Get-CimInstance Win32_Processor | Select-Object -First 1).ProcessorId"
        ),
        "machine_guid": _windows_probe(
            "(Get-ItemProperty 'HKLM:\\SOFTWARE\\Microsoft\\Cryptography').MachineGuid"
        ),
        "disk": _windows_probe(
            "(Get-CimInstance Win32_DiskDrive | Where-Object {$_.SerialNumber} | Select-Object -First 1).SerialNumber"
        ),
        "node": platform.node().upper(),
        "mac": str(uuid.getnode()),
    }
    invalid = {
        "",
        "NONE",
        "UNKNOWN",
        "DEFAULT STRING",
        "TO BE FILLED BY O.E.M.",
        "SYSTEM SERIAL NUMBER",
    }
    return {
        key: value.strip().upper()
        for key, value in claims.items()
        if str(value).strip().upper() not in invalid
    }


def _digest(parts: list[str]) -> str:
    return hashlib.sha256(
        "|".join(parts).encode("utf-8")
    ).hexdigest()[:32].upper()


def machine_id_candidates() -> list[str]:
    claims = _hardware_claims()
    system = claims.get("system", platform.system().upper())
    machine = claims.get("machine", platform.machine().upper())
    candidates: list[str] = []

    strong_values = [
        claims.get("uuid", ""),
        claims.get("board", ""),
        claims.get("bios", ""),
        claims.get("cpu", ""),
    ]
    strong_values = [value for value in strong_values if value]
    if strong_values:
        candidates.append(_digest([system, machine, *strong_values]))
    if claims.get("uuid"):
        candidates.append(_digest([system, machine, claims["uuid"]]))
    if claims.get("board") and claims.get("cpu"):
        candidates.append(
            _digest([system, machine, claims["board"], claims["cpu"]])
        )
    if claims.get("machine_guid") and claims.get("uuid"):
        candidates.append(
            _digest(
                [
                    system,
                    machine,
                    claims["uuid"],
                    claims["machine_guid"],
                ]
            )
        )

    legacy_parts = [
        platform.system(),
        platform.machine(),
        platform.node(),
        str(uuid.getnode()),
    ]
    if claims.get("uuid"):
        legacy_parts.append(claims["uuid"])
    candidates.append(_digest(legacy_parts))
    candidates.append(
        _digest(
            [
                platform.system(),
                platform.machine(),
                platform.node(),
                str(uuid.getnode()),
            ]
        )
    )
    return list(dict.fromkeys(candidates))


def machine_id() -> str:
    return machine_id_candidates()[0]


def machine_request() -> dict:
    return {
        "product": "bc-vision",
        "request_version": 2,
        "machine_id": machine_id(),
        "machine_ids": machine_id_candidates(),
    }


def _canonical(payload: dict) -> bytes:
    return canonical(payload)


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
            canonical(payload),
        )
        return True
    except Exception:
        return False


def _state_key(create: bool) -> bytes | None:
    _STATE_KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _STATE_KEY_PATH.exists():
        key = _STATE_KEY_PATH.read_bytes()
        return key if len(key) >= 32 else None
    if not create:
        return None
    key = secrets.token_bytes(32)
    temp = _STATE_KEY_PATH.with_suffix(".tmp")
    temp.write_bytes(key)
    temp.replace(_STATE_KEY_PATH)
    try:
        os.chmod(_STATE_KEY_PATH, 0o600)
    except OSError:
        pass
    return key


def _write_state(data: dict, path: Path = _STATE_PATH) -> None:
    key = _state_key(create=True)
    if key is None:
        raise RuntimeError("license state key is unavailable")
    signature = hmac.new(key, canonical(data), hashlib.sha256).hexdigest()
    document = {"data": data, "signature": signature}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(document, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    temp.replace(path)


def _read_state(path: Path) -> dict | None:
    key = _state_key(create=False)
    if key is None or not path.exists():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        data = document["data"]
        signature = str(document["signature"])
        expected = hmac.new(key, canonical(data), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        return data
    except Exception:
        return None


def _invalid(message: str, mode: str = "invalid") -> dict:
    return {
        "valid": False,
        "mode": mode,
        "plan": "none",
        "customer": "—",
        "license_id": "—",
        "issued_at": "—",
        "expires_at": "—",
        "days_left": 0,
        "camera_limit": 0,
        "features": [],
        "message": message,
    }


def _validate_document(document: dict) -> tuple[bool, str, dict]:
    try:
        payload = document["payload"]
        signature = document["signature"]
    except Exception:
        return False, "ساختار لایسنس صحیح نیست", {}
    if not isinstance(payload, dict) or not _verify_signature(payload, signature):
        return False, "امضای لایسنس معتبر نیست", {}
    product = str(payload.get("product", "bc-vision")).lower()
    if product != "bc-vision":
        return False, "این لایسنس متعلق به BC Vision نیست", {}
    allowed = [
        str(value).strip().upper()
        for value in payload.get("machine_ids", [])
        if str(value).strip()
    ]
    single = str(payload.get("machine_id", "")).strip().upper()
    if single:
        allowed.append(single)
    if not set(machine_id_candidates()).intersection(allowed):
        return False, "لایسنس متعلق به این دستگاه نیست", {}
    plan = str(payload.get("plan", "")).lower()
    if plan not in {"basic", "professional", "enterprise"}:
        return False, "پلن لایسنس شناخته‌شده نیست", {}
    license_id = str(payload.get("license_id", "")).strip()
    if not license_id:
        return False, "شناسه لایسنس موجود نیست", {}
    camera_limit = int(
        payload.get("camera_limit") or PLAN_CAMERA_LIMITS[plan]
    )
    if camera_limit < 1 or camera_limit > 4096:
        return False, "محدودیت دوربین لایسنس معتبر نیست", {}
    features = payload.get("features") or PLAN_FEATURES[plan]
    if (
        not isinstance(features, list)
        or not all(isinstance(item, str) for item in features)
        or not set(features).issubset(_ALL_FEATURES)
    ):
        return False, "قابلیت‌های لایسنس معتبر نیست", {}
    issued_raw = str(payload.get("issued_at", "")).strip()
    issued = date.fromisoformat(issued_raw)
    today = date.today()
    if issued > today + timedelta(days=1):
        return False, "تاریخ صدور لایسنس معتبر نیست", {}
    expires_raw = payload.get("expires_at")
    perpetual = not expires_raw or str(expires_raw).lower() in {
        "never",
        "perpetual",
        "lifetime",
    }
    expires = None if perpetual else date.fromisoformat(str(expires_raw))
    if expires and today > expires:
        return False, "اعتبار لایسنس پایان یافته است", {}
    result = {
        "valid": True,
        "mode": "licensed",
        "plan": plan,
        "customer": str(payload.get("customer", "—")),
        "license_id": license_id,
        "issued_at": issued.isoformat(),
        "expires_at": "دائمی" if perpetual else expires.isoformat(),
        "days_left": 99999 if perpetual else max(0, (expires - today).days),
        "camera_limit": camera_limit,
        "features": list(dict.fromkeys(features)),
        "message": "لایسنس دائمی معتبر است" if perpetual else "لایسنس معتبر است",
        "payload": payload,
    }
    return True, "", result


def _license_digest() -> str:
    return hashlib.sha256(LICENSE_PATH.read_bytes()).hexdigest().upper()


def _write_license_state(payload: dict) -> None:
    _write_state(
        {
            "license_id": str(payload["license_id"]),
            "license_digest": _license_digest(),
            "machine_id": machine_id(),
            "last_seen": date.today().isoformat(),
        }
    )


def _check_license_state(payload: dict) -> tuple[bool, str]:
    state = _read_state(_STATE_PATH)
    if state is None:
        return False, "وضعیت محلی لایسنس حذف یا دست‌کاری شده است"
    try:
        if str(state["license_id"]) != str(payload["license_id"]):
            return False, "وضعیت محلی با لایسنس مطابقت ندارد"
        if str(state["license_digest"]).upper() != _license_digest():
            return False, "فایل لایسنس پس از فعال‌سازی تغییر کرده است"
        if str(state["machine_id"]).upper() not in machine_id_candidates():
            return False, "وضعیت لایسنس متعلق به این دستگاه نیست"
        last_seen = date.fromisoformat(str(state["last_seen"]))
        today = date.today()
        if today < last_seen - timedelta(days=_CLOCK_ROLLBACK_TOLERANCE_DAYS):
            return False, "تاریخ سیستم به عقب بازگردانده شده است"
        if today > last_seen:
            state["last_seen"] = today.isoformat()
            _write_state(state)
        return True, ""
    except Exception:
        return False, "وضعیت محلی لایسنس خراب است"


def _decode_installed() -> dict:
    return decode_document(
        LICENSE_PATH.read_bytes(),
        machine_id_candidates(),
        PUBLIC_KEY_PATH.read_bytes(),
    )


def _migrate_legacy_license() -> bool:
    if not LEGACY_LICENSE_PATH.exists() or LICENSE_PATH.exists():
        return False
    try:
        document = json.loads(
            LEGACY_LICENSE_PATH.read_text(encoding="utf-8")
        )
        valid, _, result = _validate_document(document)
        if not valid:
            return False
        token = encode_document(
            document,
            machine_id_candidates(),
            PUBLIC_KEY_PATH.read_bytes(),
        )
        LICENSE_PATH.write_text(token, encoding="ascii")
        _write_license_state(result["payload"])
        LEGACY_LICENSE_PATH.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _trial_enabled() -> bool:
    return os.environ.get("BCVISION_ENABLE_TRIAL", "0") == "1"


def _trial() -> dict:
    if not _trial_enabled():
        return _invalid("لایسنس آفلاین نصب نشده است", mode="unlicensed")
    today = date.today()
    current_machine = machine_id()
    trial = _read_state(TRIAL_PATH)
    if trial is None:
        if TRIAL_PATH.exists():
            return _invalid("اطلاعات نسخه آزمایشی دست‌کاری شده است")
        trial = {
            "started": today.isoformat(),
            "last_seen": today.isoformat(),
            "machine_id": current_machine,
        }
        _write_state(trial, TRIAL_PATH)
    try:
        started = date.fromisoformat(str(trial["started"]))
        last_seen = date.fromisoformat(str(trial["last_seen"]))
        if str(trial["machine_id"]).upper() not in machine_id_candidates():
            return _invalid("نسخه آزمایشی متعلق به این دستگاه نیست")
        if today < last_seen - timedelta(days=_CLOCK_ROLLBACK_TOLERANCE_DAYS):
            return _invalid("تاریخ سیستم به عقب بازگردانده شده است")
    except Exception:
        return _invalid("اطلاعات نسخه آزمایشی خراب است")
    expires = started + timedelta(days=30)
    active = today <= expires
    trial["last_seen"] = max(today, last_seen).isoformat()
    _write_state(trial, TRIAL_PATH)
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
        "message": "نسخه آزمایشی فعال است" if active else "مهلت نسخه آزمایشی پایان یافته است",
    }


def status() -> dict:
    if not LICENSE_PATH.exists():
        _migrate_legacy_license()
    if not LICENSE_PATH.exists():
        return _trial()
    if not PUBLIC_KEY_PATH.exists():
        return _invalid("کلید عمومی لایسنس موجود نیست")
    try:
        document = _decode_installed()
    except Exception:
        return _invalid("فایل license.dat معتبر نیست یا به این دستگاه تعلق ندارد")
    valid, message, result = _validate_document(document)
    if not valid:
        mode = "expired" if "پایان" in message else "invalid"
        return _invalid(message, mode=mode)
    state_valid, state_message = _check_license_state(result["payload"])
    if not state_valid:
        return _invalid(state_message)
    result.pop("payload", None)
    return result


def camera_capacity(current_count: int, requested: int = 1) -> tuple[bool, str]:
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
        return False, f"حداکثر تعداد دوربین در پلن {result['plan']} برابر {limit} است."
    return True, ""


def runtime_camera_allowed(camera_id: int) -> bool:
    result = status()
    if not result.get("valid") or "anpr" not in result.get("features", []):
        return False
    try:
        from app.database import connect

        with connect() as connection:
            rows = connection.execute(
                "SELECT id FROM cameras WHERE enabled=1 AND lpr_enabled=1 "
                "ORDER BY sort_order,id LIMIT ?",
                (int(result["camera_limit"]),),
            ).fetchall()
        return int(camera_id) in {int(row["id"]) for row in rows}
    except Exception:
        return False


def install_license(raw: str | bytes) -> tuple[bool, str]:
    old_license = LICENSE_PATH.read_bytes() if LICENSE_PATH.exists() else None
    old_state = _STATE_PATH.read_bytes() if _STATE_PATH.exists() else None
    old_key = _STATE_KEY_PATH.read_bytes() if _STATE_KEY_PATH.exists() else None
    try:
        raw_bytes = raw if isinstance(raw, bytes) else str(raw).encode("utf-8")
        stripped = raw_bytes.strip()
        if stripped.startswith(b"{"):
            document = json.loads(stripped.decode("utf-8"))
            token = encode_document(
                document,
                machine_id_candidates(),
                PUBLIC_KEY_PATH.read_bytes(),
            )
        else:
            token = stripped.decode("ascii")
            document = decode_document(
                token,
                machine_id_candidates(),
                PUBLIC_KEY_PATH.read_bytes(),
            )
        valid, message, result = _validate_document(document)
        if not valid:
            return False, message
        LICENSE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temp = LICENSE_PATH.with_suffix(".tmp")
        temp.write_text(token, encoding="ascii")
        temp.replace(LICENSE_PATH)
        _STATE_PATH.unlink(missing_ok=True)
        _STATE_KEY_PATH.unlink(missing_ok=True)
        _write_license_state(result["payload"])
        final = status()
        if not final.get("valid"):
            raise ValueError(final.get("message") or "license activation failed")
        LEGACY_LICENSE_PATH.unlink(missing_ok=True)
        return True, "لایسنس آفلاین با موفقیت فعال شد"
    except Exception as exc:
        if old_license is None:
            LICENSE_PATH.unlink(missing_ok=True)
        else:
            LICENSE_PATH.write_bytes(old_license)
        if old_state is None:
            _STATE_PATH.unlink(missing_ok=True)
        else:
            _STATE_PATH.write_bytes(old_state)
        if old_key is None:
            _STATE_KEY_PATH.unlink(missing_ok=True)
        else:
            _STATE_KEY_PATH.write_bytes(old_key)
        message = str(exc).strip()
        return False, message or "فایل لایسنس آفلاین معتبر نیست"


def activate_online(server_url: str, activation_code: str) -> tuple[bool, str]:
    del server_url, activation_code
    return False, "فعال‌سازی آنلاین در این نسخه غیرفعال است؛ فایل license.dat را وارد کنید"


def deactivate_local() -> tuple[bool, str]:
    if not LICENSE_PATH.exists() and not LEGACY_LICENSE_PATH.exists():
        return False, "لایسنسی برای حذف وجود ندارد"
    try:
        LICENSE_PATH.unlink(missing_ok=True)
        LEGACY_LICENSE_PATH.unlink(missing_ok=True)
        _STATE_PATH.unlink(missing_ok=True)
        _STATE_KEY_PATH.unlink(missing_ok=True)
        return True, "لایسنس آفلاین از این دستگاه حذف شد"
    except Exception:
        return False, "حذف فایل لایسنس امکان‌پذیر نیست"


def has_feature(name: str) -> bool:
    result = status()
    return bool(result.get("valid")) and name in result.get("features", [])


# RC28 validation builds run without activation while the ANPR pipeline is
# repaired. The original implementation above remains intact for later use.
try:
    from app.no_license import install_no_license_mode

    install_no_license_mode()
except ImportError:
    pass

