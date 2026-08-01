import base64, hashlib, hmac, os, secrets, time
from fastapi import Request
from app.config import SECRET_PATH

COOKIE_NAME = "gilaslpr_session"

def _secret() -> bytes:
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not SECRET_PATH.exists():
        with SECRET_PATH.open("wb") as f:
            f.write(secrets.token_bytes(32))
    with SECRET_PATH.open("rb") as f:
        return f.read()

def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return base64.urlsafe_b64encode(salt + digest).decode("ascii")

def verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.urlsafe_b64decode(encoded.encode("ascii"))
        salt, expected = raw[:16], raw[16:]
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False

def create_token(username: str, hours: int = 12) -> str:
    exp = int(time.time()) + hours * 3600
    payload = f"{username}|{exp}".encode()
    sig = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + b"|" + sig).decode()

def read_token(request: Request) -> str | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        return None
    try:
        raw = base64.urlsafe_b64decode(token.encode())
        username, exp, sig = raw.split(b"|", 2)
        payload = username + b"|" + exp
        if not hmac.compare_digest(sig, hmac.new(_secret(), payload, hashlib.sha256).digest()):
            return None
        if int(exp) < int(time.time()):
            return None
        return username.decode()
    except Exception:
        return None
