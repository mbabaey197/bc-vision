from __future__ import annotations
import argparse, base64, json, secrets
from datetime import date, timedelta
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

ROOT = Path(__file__).resolve().parent.parent
KEY_DIR = Path(__file__).resolve().parent / "keys"
PRIVATE = KEY_DIR / "license_private_key.pem"
PUBLIC = ROOT / "license_public_key.pem"


def ensure_keys():
    KEY_DIR.mkdir(parents=True, exist_ok=True)
    if PRIVATE.exists():
        key = serialization.load_pem_private_key(PRIVATE.read_bytes(), password=None)
    else:
        key = Ed25519PrivateKey.generate()
        PRIVATE.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
        PUBLIC.write_bytes(key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo))
    return key


def canonical(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def main():
    ap = argparse.ArgumentParser(description="BC Vision offline license generator")
    ap.add_argument("--machine", required=True)
    ap.add_argument("--customer", required=True)
    ap.add_argument("--plan", choices=["basic", "professional", "enterprise"], default="basic")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--cameras", type=int, default=0)
    ap.add_argument("--output", default="license.json")
    a = ap.parse_args()
    limits = {"basic": 2, "professional": 8, "enterprise": 64}
    payload = {
        "license_id": secrets.token_hex(8).upper(), "customer": a.customer,
        "machine_id": a.machine.upper(), "plan": a.plan,
        "camera_limit": a.cameras or limits[a.plan],
        "issued_at": date.today().isoformat(),
        "expires_at": (date.today() + timedelta(days=a.days)).isoformat(),
    }
    key = ensure_keys()
    doc = {"payload": payload, "signature": base64.b64encode(key.sign(canonical(payload))).decode("ascii")}
    Path(a.output).write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Created: {a.output}")
    print(f"Public key: {PUBLIC}")
    print("Keep tools/keys/license_private_key.pem secret and never commit it.")

if __name__ == "__main__":
    main()
