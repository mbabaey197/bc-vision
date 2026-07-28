"""Verified RC13 model bundle activation and fail-safe engine selection.

The next ANPR engine is intentionally dormant until a signed manifest and both
ONNX files are present.  A failed verification or runtime rollback always
returns the application to the RC12 baseline.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

from .model_manager import verify_file


MANIFEST_SCHEMA = 1
RUNTIME_SCHEMA = 1
ENGINE_MODES = {"baseline", "shadow", "next"}
REQUIRED_MODELS = ("detector", "ocr")
MANIFEST_CACHE_SECONDS = 30.0
_cache_lock = threading.RLock()
_verified_cache: tuple[tuple, float, dict] | None = None


def _data_root() -> Path:
    from app.config import DATA_DIR

    return Path(DATA_DIR)


def next_models_root() -> Path:
    return _data_root() / "models" / "next"


def next_manifest_path() -> Path:
    configured = os.environ.get(
        "BCVISION_NEXT_MANIFEST",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return next_models_root() / "active-models.json"


def model_public_key_path() -> Path:
    configured = os.environ.get(
        "BCVISION_ANPR_MODEL_PUBLIC_KEY",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return next_models_root() / "model_public_key.pem"


def runtime_state_path() -> Path:
    return next_models_root() / "runtime-state.json"


def canonical_manifest_bytes(payload: dict) -> bytes:
    unsigned = deepcopy(payload)
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verify_signature(payload: dict) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    signature_text = str(payload.get("signature", "")).strip()
    if not signature_text:
        raise ValueError("Next-model manifest is not signed")
    public_key = serialization.load_pem_public_key(
        model_public_key_path().read_bytes()
    )
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("Next-model public key is not Ed25519")
    try:
        signature = base64.b64decode(
            signature_text,
            validate=True,
        )
    except Exception as exc:
        raise ValueError("Invalid next-model signature encoding") from exc
    public_key.verify(signature, canonical_manifest_bytes(payload))


def _safe_model_path(root: Path, filename: str) -> Path:
    root = root.resolve()
    candidate = (root / filename).resolve()
    candidate.relative_to(root)
    return candidate


def _file_fingerprint(paths) -> tuple:
    values = []
    for path in paths:
        candidate = Path(path)
        stat = candidate.stat()
        values.append(
            (
                str(candidate.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        )
    return tuple(values)


def verified_next_manifest() -> dict:
    """Return a verified manifest with resolved paths or raise."""

    global _verified_cache
    path = next_manifest_path()
    with _cache_lock:
        if _verified_cache is not None:
            fingerprint, verified_at, cached = _verified_cache
            try:
                current = _file_fingerprint(
                    [item[0] for item in fingerprint]
                )
            except OSError:
                current = ()
            if (
                str(path.resolve()) == fingerprint[0][0]
                and
                current == fingerprint
                and time.monotonic() - verified_at
                < MANIFEST_CACHE_SECONDS
            ):
                return deepcopy(cached)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema", 0)) != MANIFEST_SCHEMA:
        raise ValueError("Unsupported next-model manifest schema")
    if str(payload.get("engine", "")) != "bcvision-rc13":
        raise ValueError("Unexpected next-model engine identifier")
    _verify_signature(payload)

    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("Next-model manifest has no model map")
    root = path.parent
    resolved = {}
    for name in REQUIRED_MODELS:
        spec = models.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing next-model entry: {name}")
        filename = str(spec.get("filename", "")).strip()
        digest = str(spec.get("sha256", "")).strip().upper()
        size = int(spec.get("size", 0))
        if (
            not filename
            or len(digest) != 64
            or size <= 0
        ):
            raise ValueError(f"Invalid next-model entry: {name}")
        model_path = _safe_model_path(root, filename)
        if not verify_file(model_path, digest, size):
            raise ValueError(
                f"Next-model SHA-256 verification failed: {name}"
            )
        resolved[name] = {
            **spec,
            "path": str(model_path),
            "sha256": digest,
            "size": size,
        }
    result = deepcopy(payload)
    result["manifest_path"] = str(path)
    result["models"] = resolved
    fingerprint = _file_fingerprint([
        path,
        model_public_key_path(),
        *(
            resolved[name]["path"]
            for name in REQUIRED_MODELS
        ),
    ])
    with _cache_lock:
        _verified_cache = (
            fingerprint,
            time.monotonic(),
            deepcopy(result),
        )
    return result


def next_models_status() -> dict:
    try:
        manifest = verified_next_manifest()
        return {
            "ready": True,
            "release_id": str(manifest.get("release_id", "")),
            "manifest_path": manifest["manifest_path"],
            "detector_path": manifest["models"]["detector"]["path"],
            "ocr_path": manifest["models"]["ocr"]["path"],
            "error": "",
        }
    except Exception as exc:
        return {
            "ready": False,
            "release_id": "",
            "manifest_path": str(next_manifest_path()),
            "detector_path": "",
            "ocr_path": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_runtime_state() -> dict:
    try:
        payload = json.loads(
            runtime_state_path().read_text(encoding="utf-8")
        )
        if int(payload.get("schema", 0)) == RUNTIME_SCHEMA:
            return payload
    except Exception:
        pass
    return {
        "schema": RUNTIME_SCHEMA,
        "mode": "baseline",
        "previous_mode": "baseline",
    }


def requested_engine_mode() -> str:
    state = _read_runtime_state()
    if bool(state.get("rollback_lock")):
        return "baseline"
    configured = os.environ.get(
        "BCVISION_ANPR_MODE",
        "",
    ).strip().lower()
    if configured:
        return configured if configured in ENGINE_MODES else "baseline"
    mode = str(state.get("mode", "baseline")).lower()
    return mode if mode in ENGINE_MODES else "baseline"


def engine_mode() -> str:
    requested = requested_engine_mode()
    if requested == "baseline":
        return requested
    return requested if next_models_status()["ready"] else "baseline"


def _write_engine_mode(
    mode: str,
    reason: str,
    rollback_lock: bool,
) -> dict:
    selected = str(mode).strip().lower()
    if selected not in ENGINE_MODES:
        raise ValueError("Unknown ANPR engine mode")
    if selected != "baseline" and not next_models_status()["ready"]:
        raise ValueError("Verified RC13 models are not ready")
    previous = requested_engine_mode()
    payload = {
        "schema": RUNTIME_SCHEMA,
        "mode": selected,
        "previous_mode": previous,
        "reason": str(reason),
        "rollback_lock": bool(rollback_lock),
        "changed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = runtime_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def set_engine_mode(mode: str, reason="operator") -> dict:
    return _write_engine_mode(
        mode,
        reason=str(reason),
        rollback_lock=False,
    )


def rollback_to_baseline(reason: str) -> dict:
    return _write_engine_mode(
        "baseline",
        reason=str(reason),
        rollback_lock=True,
    )
