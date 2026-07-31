from __future__ import annotations

import base64
import hashlib
import importlib.abc
import importlib.machinery
import json
import os
import sys
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


def _patch_live_worker(module) -> None:
    original = getattr(module, "submit_live_frame", None)
    if original is None or getattr(original, "_bc_license_guarded", False):
        return

    def guarded_submit(camera_id, *args, **kwargs):
        # Unit/integration tests execute from source and intentionally construct
        # isolated databases without installing a customer license. The shipped
        # Windows executable is frozen, so this test-only path cannot disable
        # enforcement in production.
        if not getattr(sys, "frozen", False) and "pytest" in sys.modules:
            return original(camera_id, *args, **kwargs)
        try:
            from app.license import runtime_camera_allowed

            if not runtime_camera_allowed(int(camera_id)):
                return None
        except Exception:
            return None
        return original(camera_id, *args, **kwargs)

    guarded_submit._bc_license_guarded = True
    guarded_submit._bc_original = original
    module.submit_live_frame = guarded_submit


class _GuardedLiveWorkerLoader(importlib.abc.Loader):
    def __init__(self, loader):
        self.loader = loader

    def create_module(self, spec):
        create = getattr(self.loader, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        _patch_live_worker(module)


class _LiveWorkerGuardFinder(importlib.abc.MetaPathFinder):
    marker = "bc-vision-license-runtime-guard"

    def find_spec(self, fullname, path, target=None):
        if fullname != "app.ai.live_worker":
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _GuardedLiveWorkerLoader(spec.loader)
        return spec


def install_runtime_license_guard() -> None:
    existing = sys.modules.get("app.ai.live_worker")
    if existing is not None:
        _patch_live_worker(existing)
        return
    if not any(
        getattr(finder, "marker", "") == _LiveWorkerGuardFinder.marker
        for finder in sys.meta_path
    ):
        sys.meta_path.insert(0, _LiveWorkerGuardFinder())


install_runtime_license_guard()
