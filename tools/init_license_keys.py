from __future__ import annotations

import argparse
import os
from pathlib import Path

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PRIVATE = Path(__file__).resolve().parent / "keys" / "license_private_key.pem"
DEFAULT_PUBLIC = ROOT / "license_public_key.pem"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize the BC Vision offline licensing keypair once"
    )
    parser.add_argument("--private-key", default=str(DEFAULT_PRIVATE))
    parser.add_argument("--public-key", default=str(DEFAULT_PUBLIC))
    args = parser.parse_args()

    private_path = Path(args.private_key)
    public_path = Path(args.public_key)
    if private_path.exists() or public_path.exists():
        raise SystemExit(
            "Refusing to overwrite an existing licensing key. Back up the "
            "current keypair; replacing it invalidates issued licenses."
        )

    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    public_path.write_bytes(
        key.public_key().public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
    )
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        pass
    print(f"Private key: {private_path.resolve()}")
    print(f"Public key: {public_path.resolve()}")
    print("Back up the private key offline. Never copy it to customer systems.")


if __name__ == "__main__":
    main()
