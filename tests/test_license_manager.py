from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from tools.license_service import (
    LicenseLedger,
    LicenseRequest,
    issue_license,
    machine_ids_from_request_file,
    normalize_machine_ids,
)


def _write_keypair(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    private = Ed25519PrivateKey.generate()
    private_path = tmp_path / "private.pem"
    public_path = tmp_path / "public.pem"
    private_path.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        private.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    return private_path, public_path


def test_normalize_machine_ids_deduplicates_and_uppercases():
    assert normalize_machine_ids([" abc ", "ABC", "def"]) == ("ABC", "DEF")


def test_machine_request_file_is_validated(tmp_path: Path):
    request = tmp_path / "machine.json"
    request.write_text(
        json.dumps(
            {
                "product": "bc-vision",
                "machine_id": "alpha",
                "machine_ids": ["beta", "ALPHA"],
            }
        ),
        encoding="utf-8",
    )
    assert machine_ids_from_request_file(request) == ("BETA", "ALPHA")


def test_issue_license_and_record_ledger(tmp_path: Path):
    private_path, public_path = _write_keypair(tmp_path)
    output = tmp_path / "issued" / "customer.license.dat"
    issued = issue_license(
        LicenseRequest(
            customer="Test Customer",
            contract="C-100",
            machine_ids=("MACHINE-A",),
            plan="professional",
            days=30,
            camera_limit=4,
            features=("anpr", "reports", "api"),
        ),
        private_key_path=private_path,
        public_key_path=public_path,
        output_path=output,
        today=date(2026, 7, 31),
    )
    assert output.is_file()
    assert output.read_text(encoding="ascii").strip()
    assert issued.payload["customer"] == "Test Customer"
    assert issued.payload["expires_at"] == "2026-08-30"
    assert issued.payload["camera_limit"] == 4

    ledger = LicenseLedger(tmp_path / "manager.db")
    ledger.record(issued)
    rows = ledger.list_recent()
    assert len(rows) == 1
    assert rows[0]["license_id"] == issued.license_id
    assert rows[0]["customer"] == "Test Customer"


def test_issue_license_rejects_mismatched_public_key(tmp_path: Path):
    private_path, _ = _write_keypair(tmp_path / "one")
    _, wrong_public = _write_keypair(tmp_path / "two")
    with pytest.raises(ValueError, match="does not match"):
        issue_license(
            LicenseRequest(customer="Customer", machine_ids=("MACHINE",)),
            private_key_path=private_path,
            public_key_path=wrong_public,
            output_path=tmp_path / "license.dat",
        )


def test_issue_license_rejects_empty_customer(tmp_path: Path):
    private_path, public_path = _write_keypair(tmp_path)
    with pytest.raises(ValueError, match="Customer name"):
        issue_license(
            LicenseRequest(customer="   ", machine_ids=("MACHINE",)),
            private_key_path=private_path,
            public_key_path=public_path,
            output_path=tmp_path / "license.dat",
        )
