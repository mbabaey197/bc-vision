from __future__ import annotations

import base64
import hashlib
import json
import os
from typing import Iterable

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes

PREFIX = "BCV2."
_FORMAT_VERSION = 2
_INFO = b"BC Vision offline license v2"


def canonical(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def public_key_fingerprint(public_key_pem: bytes) -> str:
    return hashlib.sha256(public_key_pem).hexdigest().upper()


def _derive_key(machine_id: str, fingerprint: str, salt: bytes) -> bytes:
    material = (
        str(machine_id).strip().upper().encode("utf-8")
        + b"|"
        + fingerprint.encode("ascii")
    )
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=_INFO,
    ).derive(material)


def encode_document(
    document: dict,
    machine_ids: Iterable[str],
    public_key_pem: bytes,
) -> str:
    fingerprint = public_key_fingerprint(public_key_pem)
    plaintext = canonical(document)
    entries = []
    seen = set()
    for raw_machine_id in machine_ids:
        machine_id = str(raw_machine_id).strip().upper()
        if not machine_id or machine_id in seen:
            continue
        seen.add(machine_id)
        salt = os.urandom(16)
        nonce = os.urandom(12)
        aad = f"BCV2|{fingerprint}".encode("ascii")
        ciphertext = AESGCM(
            _derive_key(machine_id, fingerprint, salt)
        ).encrypt(nonce, plaintext, aad)
        entries.append(
            {
                "s": base64.urlsafe_b64encode(salt).decode("ascii"),
                "n": base64.urlsafe_b64encode(nonce).decode("ascii"),
                "c": base64.urlsafe_b64encode(ciphertext).decode("ascii"),
            }
        )
    if not entries:
        raise ValueError("at least one machine id is required")
    envelope = {
        "v": _FORMAT_VERSION,
        "k": fingerprint,
        "e": entries,
    }
    encoded = base64.urlsafe_b64encode(canonical(envelope)).decode("ascii")
    return PREFIX + encoded.rstrip("=")


def decode_document(
    raw: str | bytes,
    machine_ids: Iterable[str],
    public_key_pem: bytes,
) -> dict:
    if isinstance(raw, bytes):
        text = raw.decode("ascii")
    else:
        text = str(raw)
    text = "".join(text.strip().split())
    if not text.startswith(PREFIX):
        raise ValueError("unsupported license format")
    token = text[len(PREFIX):]
    token += "=" * (-len(token) % 4)
    envelope = json.loads(base64.urlsafe_b64decode(token).decode("utf-8"))
    if int(envelope.get("v", 0)) != _FORMAT_VERSION:
        raise ValueError("unsupported license version")
    fingerprint = public_key_fingerprint(public_key_pem)
    if str(envelope.get("k", "")).upper() != fingerprint:
        raise ValueError("license public key mismatch")
    aad = f"BCV2|{fingerprint}".encode("ascii")
    candidates = [
        str(value).strip().upper()
        for value in machine_ids
        if str(value).strip()
    ]
    for entry in envelope.get("e", []):
        try:
            salt = base64.urlsafe_b64decode(entry["s"])
            nonce = base64.urlsafe_b64decode(entry["n"])
            ciphertext = base64.urlsafe_b64decode(entry["c"])
        except Exception:
            continue
        for machine_id in candidates:
            try:
                plaintext = AESGCM(
                    _derive_key(machine_id, fingerprint, salt)
                ).decrypt(nonce, ciphertext, aad)
                document = json.loads(plaintext.decode("utf-8"))
                if not isinstance(document, dict):
                    raise ValueError("invalid license document")
                return document
            except Exception:
                continue
    raise ValueError("license is not bound to this machine")
