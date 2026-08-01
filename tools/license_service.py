from __future__ import annotations

import base64
import json
import os
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable

from cryptography.hazmat.primitives import serialization

from app.license_format import canonical, encode_document

LIMITS = {"basic": 2, "professional": 8, "enterprise": 64}
FEATURES = {
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
ALL_FEATURES = tuple(dict.fromkeys(item for values in FEATURES.values() for item in values))


@dataclass(frozen=True)
class LicenseRequest:
    customer: str
    machine_ids: tuple[str, ...]
    plan: str = "basic"
    days: int = 365
    perpetual: bool = False
    camera_limit: int = 0
    features: tuple[str, ...] = ()
    contract: str = ""


@dataclass(frozen=True)
class IssuedLicense:
    license_id: str
    output_path: Path
    payload: dict


def normalize_machine_ids(values: Iterable[str]) -> tuple[str, ...]:
    normalized = []
    for value in values:
        item = str(value).strip().upper()
        if item and item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("At least one machine ID is required")
    return tuple(normalized)


def machine_ids_from_request_file(path: Path) -> tuple[str, ...]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if str(request.get("product", "")).lower() != "bc-vision":
        raise ValueError("Machine request is not for BC Vision")
    values = list(request.get("machine_ids", []))
    if request.get("machine_id"):
        values.append(request["machine_id"])
    return normalize_machine_ids(values)


def _load_matching_private_key(private_path: Path, public_path: Path):
    if not private_path.is_file():
        raise FileNotFoundError(
            "Private key not found. Run tools/init_license_keys.py once on the isolated licensing computer."
        )
    if not public_path.is_file():
        raise FileNotFoundError("Public key file not found")
    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(), password=None
    )
    derived_public = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_pem = public_path.read_bytes()
    if derived_public != public_pem:
        raise ValueError("Private key does not match the BC Vision public key")
    return private_key, public_pem


def issue_license(
    request: LicenseRequest,
    *,
    private_key_path: Path,
    public_key_path: Path,
    output_path: Path,
    today: date | None = None,
) -> IssuedLicense:
    customer = request.customer.strip()
    if not customer:
        raise ValueError("Customer name is required")
    machine_ids = normalize_machine_ids(request.machine_ids)
    if request.plan not in LIMITS:
        raise ValueError("Unknown license plan")
    camera_limit = request.camera_limit or LIMITS[request.plan]
    if camera_limit < 1 or camera_limit > 4096:
        raise ValueError("Camera limit must be between 1 and 4096")
    selected_features = tuple(dict.fromkeys(request.features or tuple(FEATURES[request.plan])))
    unknown = set(selected_features) - set(ALL_FEATURES)
    if unknown:
        raise ValueError(f"Unknown features: {', '.join(sorted(unknown))}")
    if request.days < 1 and not request.perpetual:
        raise ValueError("Days must be positive")

    private_key, public_pem = _load_matching_private_key(
        private_key_path, public_key_path
    )
    issue_date = today or date.today()
    payload = {
        "format_version": 2,
        "product": "bc-vision",
        "license_id": secrets.token_hex(16).upper(),
        "customer": customer,
        "contract": request.contract.strip(),
        "machine_id": machine_ids[0],
        "machine_ids": list(machine_ids),
        "plan": request.plan,
        "camera_limit": camera_limit,
        "features": list(selected_features),
        "issued_at": issue_date.isoformat(),
        "expires_at": (
            "perpetual"
            if request.perpetual
            else (issue_date + timedelta(days=request.days)).isoformat()
        ),
        "nonce": base64.urlsafe_b64encode(os.urandom(18)).decode("ascii"),
    }
    document = {
        "payload": payload,
        "signature": base64.b64encode(
            private_key.sign(canonical(payload))
        ).decode("ascii"),
    }
    token = encode_document(document, machine_ids, public_pem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(token, encoding="ascii")
    return IssuedLicense(payload["license_id"], output_path, payload)


class LicenseLedger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self):
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS issued_licenses (
                    license_id TEXT PRIMARY KEY,
                    customer TEXT NOT NULL,
                    contract TEXT NOT NULL,
                    plan TEXT NOT NULL,
                    camera_limit INTEGER NOT NULL,
                    issued_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    machine_ids_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    output_path TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def record(self, issued: IssuedLicense) -> None:
        payload = issued.payload
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO issued_licenses (
                    license_id, customer, contract, plan, camera_limit,
                    issued_at, expires_at, machine_ids_json, features_json,
                    output_path
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    issued.license_id,
                    payload["customer"],
                    payload["contract"],
                    payload["plan"],
                    payload["camera_limit"],
                    payload["issued_at"],
                    payload["expires_at"],
                    json.dumps(payload["machine_ids"], ensure_ascii=False),
                    json.dumps(payload["features"], ensure_ascii=False),
                    str(issued.output_path),
                ),
            )

    def list_recent(self, limit: int = 200) -> list[dict]:
        with self._connect() as db:
            rows = db.execute(
                """
                SELECT * FROM issued_licenses
                ORDER BY created_at DESC, license_id DESC
                LIMIT ?
                """,
                (max(1, min(int(limit), 2000)),),
            ).fetchall()
        return [dict(row) for row in rows]
