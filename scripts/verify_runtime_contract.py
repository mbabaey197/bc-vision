from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "RUNTIME_CONTRACT.json"
REQUIRED_FILES = {
    "BUILD_PORTABLE_EXE.bat",
    "app/database.py",
    "launcher.py",
    "requirements-ai-lock.txt",
    "requirements-lock.txt",
    "runtime_payload.py",
}


def _sha256(path: Path) -> str:
    # Runtime compatibility is independent of checkout newline settings.
    content = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def contract_id(contract: dict) -> str:
    canonical = json.dumps(
        contract,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_payload import (  # noqa: E402
    compare_runtime_versions,
    validate_fast_update_version,
)


def verify_contract(contract_path: Path = CONTRACT_PATH) -> dict:
    try:
        contract = json.loads(Path(contract_path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("RUNTIME_CONTRACT.json is unreadable") from exc
    if not isinstance(contract, dict) or contract.get("schema") != 1:
        raise ValueError("Unsupported runtime contract schema")

    abi = str(contract.get("runtime_abi", ""))
    installed_abi = (ROOT / "RUNTIME_ABI").read_text(
        encoding="utf-8-sig",
    ).strip()
    if abi != installed_abi or not re.fullmatch(r"[1-9]\d*", abi):
        raise ValueError("Runtime ABI and contract disagree")

    base_version = str(contract.get("base_version", ""))
    expected_base = (ROOT / "FAST_UPDATE_BASE_VERSION").read_text(
        encoding="utf-8-sig",
    ).strip()
    if base_version != expected_base:
        raise ValueError("Fast-update base and contract disagree")

    files = contract.get("files")
    if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
        raise ValueError("Runtime contract file set is incomplete")
    for relative, expected_hash in files.items():
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise ValueError(f"Invalid contract hash for {relative}")
        actual_hash = _sha256(ROOT / relative)
        if actual_hash != expected_hash:
            raise ValueError(
                f"Runtime file changed without a new full base: {relative}"
            )
    return contract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-id", action="store_true")
    parser.add_argument("--write-id", type=Path)
    parser.add_argument("--contract", type=Path, default=CONTRACT_PATH)
    parser.add_argument("--validate-update-version")
    parser.add_argument("--require-newer-than", action="append", default=[])
    args = parser.parse_args()
    contract = verify_contract(args.contract)
    identifier = contract_id(contract)
    if args.require_newer_than and not args.validate_update_version:
        parser.error(
            "--require-newer-than requires --validate-update-version",
        )
    if args.validate_update_version:
        validate_fast_update_version(
            str(contract["base_version"]),
            args.validate_update_version,
        )
        for previous in args.require_newer_than:
            validate_fast_update_version(
                str(contract["base_version"]),
                previous,
            )
            if compare_runtime_versions(
                args.validate_update_version,
                previous,
            ) <= 0:
                raise ValueError(
                    f"Fast update {args.validate_update_version} must be "
                    f"newer than existing release {previous}",
                )
    if args.write_id is not None:
        args.write_id.parent.mkdir(parents=True, exist_ok=True)
        args.write_id.write_text(identifier + "\n", encoding="ascii")
    if args.contract_id:
        print(identifier)
        return 0
    print(
        f"runtime contract ABI {contract['runtime_abi']} "
        f"base {contract['base_version']} verified"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
