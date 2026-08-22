from concurrent.futures import ThreadPoolExecutor
import os
import stat
from types import SimpleNamespace

import pytest

from app import security


def _request_with_token(token):
    return SimpleNamespace(cookies={security.COOKIE_NAME: token})


def test_session_secret_repairs_truncated_file_atomically(tmp_path, monkeypatch):
    secret_path = tmp_path / ".secret"
    secret_path.write_bytes(b"")
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)

    value = security._secret()

    assert len(value) == 32
    assert secret_path.read_bytes() == value
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_first_use_publishes_one_complete_secret(
    tmp_path,
    monkeypatch,
):
    secret_path = tmp_path / ".secret"
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)

    with ThreadPoolExecutor(max_workers=12) as executor:
        values = list(executor.map(lambda _index: security._secret(), range(48)))

    assert len(set(values)) == 1
    assert len(values[0]) == 32
    assert secret_path.read_bytes() == values[0]
    assert not list(tmp_path.glob(".*.tmp"))


def test_session_secret_replaces_symlink_without_touching_external_file(
    tmp_path,
    monkeypatch,
):
    external = tmp_path / "external-key"
    external.write_bytes(b"e" * 32)
    secret_path = tmp_path / ".secret"
    try:
        secret_path.symlink_to(external)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)

    value = security._secret()

    assert value != b"e" * 32
    assert external.read_bytes() == b"e" * 32
    assert not secret_path.is_symlink()
    assert secret_path.read_bytes() == value


def test_session_secret_rotates_hardlinked_key(tmp_path, monkeypatch):
    shared = tmp_path / "shared-key"
    shared.write_bytes(b"h" * 32)
    secret_path = tmp_path / ".secret"
    try:
        os.link(shared, secret_path)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)

    value = security._secret()

    assert value != b"h" * 32
    assert shared.read_bytes() == b"h" * 32
    assert secret_path.read_bytes() == value
    assert secret_path.stat().st_nlink == 1


@pytest.mark.skipif(os.name == "nt", reason="POSIX permission contract")
def test_session_secret_rotates_group_readable_key(tmp_path, monkeypatch):
    secret_path = tmp_path / ".secret"
    secret_path.write_bytes(b"p" * 32)
    secret_path.chmod(0o644)
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)

    value = security._secret()

    assert value != b"p" * 32
    assert stat.S_IMODE(secret_path.stat().st_mode) & 0o077 == 0


def test_session_secret_temp_collision_is_preserved(tmp_path, monkeypatch):
    secret_path = tmp_path / ".secret"
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)
    monkeypatch.setattr(
        security.secrets,
        "token_hex",
        lambda _size=8: "fixed",
    )
    collision = tmp_path / "..secret.fixed.tmp"
    collision.write_bytes(b"foreign-temp")

    with pytest.raises(FileExistsError):
        security._replace_secret_atomically(b"s" * 32)

    assert collision.read_bytes() == b"foreign-temp"
    assert not secret_path.exists()


def test_session_secret_accepts_only_exact_late_replace_result(
    tmp_path,
    monkeypatch,
):
    secret_path = tmp_path / ".secret"
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)
    real_replace = security.os.replace

    def replace_then_raise(source, target):
        real_replace(source, target)
        raise OSError("late replace error")

    monkeypatch.setattr(security.os, "replace", replace_then_raise)

    security._replace_secret_atomically(b"a" * 32)

    assert secret_path.read_bytes() == b"a" * 32
    assert not list(tmp_path.glob(".*.tmp"))


def test_session_secret_rejects_temp_inode_substitution(
    tmp_path,
    monkeypatch,
):
    secret_path = tmp_path / ".secret"
    monkeypatch.setattr(security, "SECRET_PATH", secret_path)
    real_replace = security.os.replace

    def substitute_before_replace(source, target):
        Path = type(secret_path)
        source_path = Path(source)
        source_path.unlink()
        source_path.write_bytes(b"f" * 32)
        real_replace(source_path, target)

    monkeypatch.setattr(security.os, "replace", substitute_before_replace)

    with pytest.raises(OSError, match="identity validation"):
        security._replace_secret_atomically(b"a" * 32)

    assert secret_path.read_bytes() == b"f" * 32


def test_versioned_session_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "SECRET_PATH", tmp_path / ".secret")

    token = security.create_token("operator", session_version=7)

    details = security.read_session_details(_request_with_token(token))
    assert details is not None
    assert details[:2] == ("operator", 7)
    assert details[2] >= int(security.time.time())
    assert security.read_session(_request_with_token(token)) == (
        "operator",
        7,
    )
    assert security.read_token(_request_with_token(token)) == "operator"


def test_each_login_token_has_a_distinct_nonce(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "SECRET_PATH", tmp_path / ".secret")

    first = security.create_token("operator", session_version=3)
    second = security.create_token("operator", session_version=3)

    assert first != second
    assert security.read_session(_request_with_token(first)) == (
        "operator",
        3,
    )
    assert security.read_session(_request_with_token(second)) == (
        "operator",
        3,
    )


def test_versioned_session_rejects_signature_tampering(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(security, "SECRET_PATH", tmp_path / ".secret")
    token = security.create_token("operator", session_version=2)
    raw = bytearray(security.base64.urlsafe_b64decode(token.encode("ascii")))
    raw[-1] ^= 1
    tampered = security.base64.urlsafe_b64encode(bytes(raw)).decode("ascii")

    assert security.read_session(_request_with_token(tampered)) is None


def test_session_rejects_noncanonical_base64_aliases(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "SECRET_PATH", tmp_path / ".secret")
    token = security.create_token("operator", session_version=2)
    aliased = token[:3] + "!" + token[3:]

    assert security.read_session(_request_with_token(aliased)) is None
    assert security.session_fingerprint(_request_with_token(aliased)) == ""
    assert security.session_fingerprint(_request_with_token(token))


def test_legacy_session_maps_to_generation_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(security, "SECRET_PATH", tmp_path / ".secret")
    expiry = str(int(security.time.time()) + 300).encode("ascii")
    payload = b"legacy-admin|" + expiry
    signature = security.hmac.new(
        security._secret(),
        payload,
        security.hashlib.sha256,
    ).digest()
    token = security.base64.urlsafe_b64encode(
        payload + b"|" + signature
    ).decode("ascii")

    assert security.read_session(_request_with_token(token)) == (
        "legacy-admin",
        0,
    )
