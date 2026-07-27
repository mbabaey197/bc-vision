from __future__ import annotations
import base64, hashlib, json, os, platform, subprocess, uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from app.config import LICENSE_PATH, PUBLIC_KEY_PATH, TRIAL_PATH

PLAN_CAMERA_LIMITS = {"trial": 2, "basic": 2, "professional": 8, "enterprise": 64}
PLAN_FEATURES = {
    "trial": ["anpr", "events", "reports"],
    "basic": ["anpr", "events", "reports"],
    "professional": ["anpr", "events", "reports", "vehicle_ai", "watchlist", "api"],
    "enterprise": ["anpr", "events", "reports", "vehicle_ai", "watchlist", "api", "gate", "multi_site", "priority_support"],
}

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


def machine_id() -> str:
    parts = [platform.system(), platform.machine(), platform.node(), str(uuid.getnode())]
    if platform.system().lower() == "windows":
        commands = [
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_ComputerSystemProduct).UUID"],
            ["wmic", "csproduct", "get", "uuid"],
        ]
        for cmd in commands:
            try:
                out = subprocess.check_output(
                    cmd,
                    text=True,
                    stderr=subprocess.DEVNULL,
                    timeout=5,
                    **_hidden_process_kwargs(),
                )
                vals = [x.strip() for x in out.splitlines() if x.strip() and "uuid" not in x.lower()]
                if vals:
                    parts.append(vals[0]); break
            except Exception:
                pass
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:32].upper()

def _canonical(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")

def _verify_signature(payload: dict, signature_b64: str) -> bool:
    if not PUBLIC_KEY_PATH.exists(): return False
    try:
        from cryptography.hazmat.primitives import serialization
        public_key = serialization.load_pem_public_key(PUBLIC_KEY_PATH.read_bytes())
        public_key.verify(base64.b64decode(signature_b64), _canonical(payload))
        return True
    except Exception: return False

def _trial() -> dict:
    today = date.today(); started=today
    if TRIAL_PATH.exists():
        try: started = date.fromisoformat(json.loads(TRIAL_PATH.read_text(encoding="utf-8"))["started"])
        except Exception: pass
    else:
        TRIAL_PATH.write_text(json.dumps({"started": started.isoformat(), "machine_id": machine_id()}), encoding="utf-8")
    expires = started + timedelta(days=30); active = today <= expires
    return {"valid":active,"mode":"trial","plan":"trial","customer":"نسخه آزمایشی","license_id":"TRIAL",
            "issued_at":started.isoformat(),"expires_at":expires.isoformat(),"days_left":max(0,(expires-today).days),
            "camera_limit":PLAN_CAMERA_LIMITS["trial"],"features":PLAN_FEATURES["trial"],
            "message":"نسخه آزمایشی فعال است" if active else "مهلت نسخه آزمایشی پایان یافته است"}

def status() -> dict:
    if not LICENSE_PATH.exists(): return _trial()
    try:
        doc=json.loads(LICENSE_PATH.read_text(encoding="utf-8")); payload,signature=doc["payload"],doc["signature"]
        if not _verify_signature(payload,signature): return {**_trial(),"valid":False,"mode":"invalid","message":"امضای لایسنس معتبر نیست"}
        allowed=[str(x).upper() for x in payload.get("machine_ids",[]) if x]
        single=str(payload.get("machine_id","")).upper()
        if single: allowed.append(single)
        if machine_id() not in allowed: return {**_trial(),"valid":False,"mode":"invalid","message":"لایسنس متعلق به این دستگاه نیست"}
        plan=str(payload.get("plan","basic")).lower()
        expires_raw=payload.get("expires_at")
        perpetual=not expires_raw or str(expires_raw).lower() in {"never","perpetual","lifetime"}
        expires=None if perpetual else date.fromisoformat(str(expires_raw))
        if expires and date.today()>expires: return {**_trial(),"valid":False,"mode":"expired","message":"اعتبار لایسنس پایان یافته است"}
        features=payload.get("features") or PLAN_FEATURES.get(plan,PLAN_FEATURES["basic"])
        return {"valid":True,"mode":"licensed","plan":plan,"customer":payload.get("customer","—"),
                "license_id":payload.get("license_id","—"),"issued_at":payload.get("issued_at","—"),
                "expires_at":"دائمی" if perpetual else expires.isoformat(),"days_left":99999 if perpetual else max(0,(expires-date.today()).days),
                "camera_limit":int(payload.get("camera_limit") or PLAN_CAMERA_LIMITS.get(plan,2)),"features":features,
                "message":"لایسنس دائمی معتبر است" if perpetual else "لایسنس معتبر است"}
    except Exception:
        return {**_trial(),"valid":False,"mode":"invalid","message":"فایل لایسنس خراب یا ناقص است"}

def install_license(raw: str) -> tuple[bool,str]:
    try:
        doc=json.loads(raw)
        if not isinstance(doc.get("payload"),dict) or not doc.get("signature"): return False,"ساختار لایسنس صحیح نیست"
        LICENSE_PATH.parent.mkdir(parents=True,exist_ok=True)
        temp=LICENSE_PATH.with_suffix('.tmp'); temp.write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
        old=LICENSE_PATH.read_bytes() if LICENSE_PATH.exists() else None
        temp.replace(LICENSE_PATH); result=status()
        if not result["valid"]:
            if old is None: LICENSE_PATH.unlink(missing_ok=True)
            else: LICENSE_PATH.write_bytes(old)
            return False,result["message"]
        return True,"لایسنس با موفقیت فعال شد"
    except Exception: return False,"متن لایسنس JSON معتبر نیست"

def activate_online(server_url: str, activation_code: str) -> tuple[bool,str]:
    server_url=server_url.strip().rstrip('/'); activation_code=activation_code.strip()
    if not server_url.startswith(('https://','http://')): return False,"آدرس سرور فعال‌سازی معتبر نیست"
    if not activation_code: return False,"کد فعال‌سازی وارد نشده است"
    try:
        import httpx
        r=httpx.post(server_url+'/api/v1/activate',json={"activation_code":activation_code,"machine_id":machine_id(),"product":"bc-vision"},timeout=15)
        if r.status_code!=200:
            try: msg=r.json().get('message') or r.json().get('detail')
            except Exception: msg=None
            return False,msg or f"خطای سرور فعال‌سازی ({r.status_code})"
        data=r.json(); raw=json.dumps(data.get('license') or data,ensure_ascii=False)
        return install_license(raw)
    except Exception as exc: return False,"ارتباط با سرور فعال‌سازی برقرار نشد"

def deactivate_local() -> tuple[bool,str]:
    if not LICENSE_PATH.exists(): return False,"لایسنسی برای حذف وجود ندارد"
    try: LICENSE_PATH.unlink(); return True,"لایسنس از این دستگاه حذف شد"
    except Exception: return False,"حذف فایل لایسنس امکان‌پذیر نیست"

def has_feature(name: str) -> bool:
    s=status(); return bool(s.get('valid')) and name in s.get('features',[])
