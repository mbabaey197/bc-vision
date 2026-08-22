import base64
import binascii
import hashlib
import hmac
import os
import secrets
import stat
import threading
import time

from fastapi import Request

from app.config import SECRET_PATH
from app.file_identity import descriptor_file_identity, path_file_identity

COOKIE_NAME = "gilaslpr_session"
_SECRET_SIZE = 32
_SECRET_LOCK = threading.Lock()


def _secret_file_value(path) -> bytes | None:
    """Read one stable private key without following filesystem aliases."""

    path = os.fspath(path)
    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return None
    if (
        not stat.S_ISREG(before.st_mode)
        or int(before.st_nlink) != 1
        or int(before.st_size) != _SECRET_SIZE
        or (
            os.name != "nt"
            and int(before.st_mode) & (stat.S_IRWXG | stat.S_IRWXO)
        )
    ):
        return None
    try:
        before_identity = path_file_identity(path, details=before)
    except OSError:
        return None
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        identity = descriptor_file_identity(descriptor, details=opened)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or int(opened.st_size) != _SECRET_SIZE
            or identity != before_identity
        ):
            return None
        value = b""
        while len(value) <= _SECRET_SIZE:
            chunk = os.read(descriptor, _SECRET_SIZE + 1 - len(value))
            if not chunk:
                break
            value += chunk
        after = os.fstat(descriptor)
        after_identity = descriptor_file_identity(descriptor, details=after)
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(path)
    except OSError:
        return None
    try:
        current_identity = path_file_identity(path, details=current)
    except OSError:
        return None
    if (
        len(value) != _SECRET_SIZE
        or not stat.S_ISREG(after.st_mode)
        or int(after.st_nlink) != 1
        or int(after.st_size) != _SECRET_SIZE
        or after_identity != identity
        or current_identity != identity
        or not stat.S_ISREG(current.st_mode)
        or int(current.st_nlink) != 1
        or int(current.st_size) != _SECRET_SIZE
    ):
        return None
    return value


def _unlink_owned_secret_temporary(path, identity) -> bool:
    try:
        details = os.lstat(path)
        current_identity = path_file_identity(path, details=details)
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if (
        not stat.S_ISREG(details.st_mode)
        or int(details.st_nlink) != 1
        or current_identity != identity
    ):
        return False
    os.unlink(path)
    return True


def _fsync_secret_directory(path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(
        os.fspath(path),
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_secret_atomically(value: bytes) -> None:
    """Publish a complete session key or leave the old file untouched."""

    if len(value) != _SECRET_SIZE:
        raise ValueError("session secret must contain exactly 32 bytes")
    SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    descriptor = None
    identity = None
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        for _attempt in range(8):
            candidate = SECRET_PATH.with_name(
                f".{SECRET_PATH.name}.{secrets.token_hex(8)}.tmp"
            )
            try:
                descriptor = os.open(candidate, flags, 0o600)
            except FileExistsError:
                continue
            temporary = candidate
            break
        if temporary is None or descriptor is None:
            raise FileExistsError(
                "could not reserve a private session-secret temporary"
            )
        opened = os.fstat(descriptor)
        identity = descriptor_file_identity(descriptor, details=opened)
        if not stat.S_ISREG(opened.st_mode) or int(opened.st_nlink) != 1:
            raise OSError("session-secret temporary is unsafe")
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            if hasattr(os, "fchmod"):
                try:
                    os.fchmod(handle.fileno(), 0o600)
                except OSError:
                    if os.name != "nt":
                        raise
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        current = os.lstat(temporary)
        if (
            not stat.S_ISREG(current.st_mode)
            or int(current.st_nlink) != 1
            or int(current.st_size) != _SECRET_SIZE
            or path_file_identity(temporary, details=current) != identity
        ):
            raise OSError("session-secret temporary changed before publish")
        try:
            os.replace(temporary, SECRET_PATH)
        except OSError:
            # Some filesystems can report a late rename error. Accept it only
            # if the exact generated inode and bytes are already published.
            try:
                published = os.lstat(SECRET_PATH)
            except OSError:
                raise
            if (
                not stat.S_ISREG(published.st_mode)
                or int(published.st_nlink) != 1
                or path_file_identity(
                    SECRET_PATH,
                    details=published,
                ) != identity
                or _secret_file_value(SECRET_PATH) != value
            ):
                raise
        published = os.lstat(SECRET_PATH)
        if (
            not stat.S_ISREG(published.st_mode)
            or int(published.st_nlink) != 1
            or path_file_identity(
                SECRET_PATH,
                details=published,
            ) != identity
            or _secret_file_value(SECRET_PATH) != value
        ):
            raise OSError("published session secret failed identity validation")
        _fsync_secret_directory(SECRET_PATH.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None and identity is not None:
            _unlink_owned_secret_temporary(temporary, identity)

def _secret() -> bytes:
    with _SECRET_LOCK:
        value = _secret_file_value(SECRET_PATH)
        if value is None:
            # A crash during legacy direct creation could leave an empty or
            # truncated key. Symlinks, hardlinks and permissive legacy keys
            # are rotated rather than trusted for signing sessions.
            value = secrets.token_bytes(_SECRET_SIZE)
            _replace_secret_atomically(value)
        return value

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
    except (binascii.Error, TypeError, UnicodeError, ValueError):
        return False

def create_token(
    username: str,
    hours: int = 12,
    *,
    session_version: int = 0,
) -> str:
    exp = int(time.time()) + hours * 3600
    encoded_username = base64.urlsafe_b64encode(
        str(username).encode("utf-8")
    ).decode("ascii")
    version = max(0, int(session_version))
    nonce = secrets.token_hex(16)
    payload = (
        f"v3|{encoded_username}|{version}|{exp}|{nonce}"
    ).encode("ascii")
    sig = hmac.new(_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(payload + b"|" + sig).decode()


def _canonical_token_bytes(request: Request) -> bytes | None:
    token = request.cookies.get(COOKIE_NAME)
    if not token or len(token) > 4096:
        return None
    try:
        encoded = token.encode("ascii")
        raw = base64.b64decode(
            encoded,
            altchars=b"-_",
            validate=True,
        )
    except (binascii.Error, TypeError, UnicodeError, ValueError):
        return None
    canonical = base64.urlsafe_b64encode(raw)
    if not hmac.compare_digest(encoded, canonical):
        return None
    return raw


def read_session_details(request: Request) -> tuple[str, int, int] | None:
    """Return username, revocable generation and signed expiry epoch."""

    try:
        raw = _canonical_token_bytes(request)
        if raw is None:
            return None
        if raw.startswith(b"v3|"):
            marker, encoded_username, version, exp, nonce, sig = raw.split(
                b"|",
                5,
            )
            payload = b"|".join(
                (marker, encoded_username, version, exp, nonce)
            )
            username = base64.urlsafe_b64decode(
                encoded_username
            ).decode("utf-8")
            session_version = int(version)
            if len(nonce) != 32 or any(
                character not in b"0123456789abcdef" for character in nonce
            ):
                return None
        elif raw.startswith(b"v2|"):
            marker, encoded_username, version, exp, sig = raw.split(b"|", 4)
            payload = b"|".join(
                (marker, encoded_username, version, exp)
            )
            username = base64.urlsafe_b64decode(
                encoded_username
            ).decode("utf-8")
            session_version = int(version)
        else:
            # Accept pre-migration sessions as generation zero. The next
            # password change increments the database generation and revokes
            # every legacy token immediately.
            username_bytes, exp, sig = raw.split(b"|", 2)
            payload = username_bytes + b"|" + exp
            username = username_bytes.decode("utf-8")
            session_version = 0
        expected = hmac.new(_secret(), payload, hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expected):
            return None
        expiry = int(exp)
        if (
            expiry < int(time.time())
            or session_version < 0
            or not username
            or len(username) > 256
        ):
            return None
        return username, session_version, expiry
    except (binascii.Error, TypeError, UnicodeError, ValueError):
        return None


def read_session(request: Request) -> tuple[str, int] | None:
    """Return the authenticated username and revocable session generation."""

    details = read_session_details(request)
    return details[:2] if details else None


def session_fingerprint(request: Request) -> str:
    """Return a non-reversible identifier for the exact presented token."""

    raw = _canonical_token_bytes(request)
    if raw is None:
        return ""
    return hashlib.sha256(raw).hexdigest()


def read_token(request: Request) -> str | None:
    session = read_session(request)
    return session[0] if session else None
