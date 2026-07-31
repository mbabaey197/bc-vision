from __future__ import annotations

import argparse
import base64
import json
import os
import secrets
from datetime import date, timedelta
from pathlib import Path

from cryptography.hazmat.primitives import serialization

from app.license_format import canonical, encode_document

ROOT = Path(__file__).resolve().parent.parent
KEY_DIR = Path(__file__).resolve().parent / "keys"
DEFAULT_PRIVATE = KEY_DIR / "license_private_key.pem"
DEFAULT_PUBLIC = ROOT / "license_public_key.pem"
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


def _load_private_key(path: Path):
    if not path.is_file():
        raise SystemExit(
            "Private key not found. Run tools/init_license_keys.py once on the "
            "isolated licensing computer."
        )
    return serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )


def _machine_ids(args) -> list[str]:
    values = [str(value).strip().upper() for value in (args.machine or [])]
    if args.request:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        if str(request.get("product", "")).lower() != "bc-vision":
            raise SystemExit("Machine request is not for BC Vision")
        values.extend(
            str(value).strip().upper()
            for value in request.get("machine_ids", [])
        )
        if request.get("machine_id"):
            values.append(str(request["machine_id"]).strip().upper())
    values = list(dict.fromkeys(value for value in values if value))
    if not values:
        raise SystemExit("Provide --machine or --request")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BC Vision offline license.dat generator"
    )
    parser.add_argument("--machine", action="append")
    parser.add_argument("--request", help="Machine request JSON file")
    parser.add_argument("--customer", required=True)
    parser.add_argument(
        "--plan",
        choices=sorted(LIMITS),
        default="basic",
    )
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--perpetual", action="store_true")
    parser.add_argument("--cameras", type=int, default=0)
    parser.add_argument("--feature", action="append", default=[])
    parser.add_argument("--contract", default="")
    parser.add_argument("--private-key", default=str(DEFAULT_PRIVATE))
    parser.add_argument("--public-key", default=str(DEFAULT_PUBLIC))
    parser.add_argument("--output", default="license.dat")
    args = parser.parse_args()

    machine_ids = _machine_ids(args)
    private_path = Path(args.private_key)
    public_path = Path(args.public_key)
    private_key = _load_private_key(private_path)
    derived_public = private_key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    if not public_path.is_file():
        raise SystemExit("Public key file not found")
    public_pem = public_path.read_bytes()
    if derived_public != public_pem:
        raise SystemExit("Private key does not match the BC Vision public key")

    camera_limit = args.cameras or LIMITS[args.plan]
    if camera_limit < 1 or camera_limit > 4096:
        raise SystemExit("Camera limit must be between 1 and 4096")
    selected_features = args.feature or FEATURES[args.plan]
    unknown = set(selected_features) - {
        item for values in FEATURES.values() for item in values
    }
    if unknown:
        raise SystemExit(f"Unknown features: {', '.join(sorted(unknown))}")
    if args.days < 1 and not args.perpetual:
        raise SystemExit("Days must be positive")

    today = date.today()
    payload = {
        "format_version": 2,
        "product": "bc-vision",
        "license_id": secrets.token_hex(16).upper(),
        "customer": args.customer.strip(),
        "contract": args.contract.strip(),
        "machine_id": machine_ids[0],
        "machine_ids": machine_ids,
        "plan": args.plan,
        "camera_limit": camera_limit,
        "features": list(dict.fromkeys(selected_features)),
        "issued_at": today.isoformat(),
        "expires_at": (
            "perpetual"
            if args.perpetual
            else (today + timedelta(days=args.days)).isoformat()
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
    output = Path(args.output)
    output.write_text(token, encoding="ascii")
    print(f"Created: {output.resolve()}")
    print(f"License ID: {payload['license_id']}")
    print(f"Machines: {len(machine_ids)}")
    print("Offline license generated; no activation server is used.")


if __name__ == "__main__":
    main()
