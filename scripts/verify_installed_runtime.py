from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_payload import select_runtime_payload  # noqa: E402
from runtime_payload import (  # noqa: E402
    FAILED_MARKER,
    LAST_KNOWN_GOOD_MARKER,
    PENDING_MARKER,
    read_runtime_marker,
)


def _marker(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig").strip()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify an installed BC Vision versioned runtime",
    )
    parser.add_argument("--install-dir", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--abi", required=True)
    parser.add_argument("--previous")
    parser.add_argument("--last-known-good")
    parser.add_argument("--failed")
    parser.add_argument(
        "--allow-pending",
        action="store_true",
    )
    args = parser.parse_args()

    install_dir = args.install_dir.resolve()
    selected = select_runtime_payload(install_dir)
    if selected is None:
        raise SystemExit("No verified installed runtime was selected")
    if selected.version != args.version:
        raise SystemExit(
            f"Expected runtime {args.version}, selected {selected.version}"
        )
    if selected.runtime_abi != args.abi:
        raise SystemExit(
            f"Expected ABI {args.abi}, selected {selected.runtime_abi}"
        )
    if _marker(install_dir / "runtime" / "current.txt") != args.version:
        raise SystemExit("The current runtime pointer was not activated")
    if args.previous is not None:
        previous = _marker(
            install_dir / "runtime" / "previous.txt",
        )
        if previous != args.previous:
            raise SystemExit(
                f"Expected previous runtime {args.previous}, got {previous}"
            )
    if args.last_known_good is not None:
        last_good = read_runtime_marker(
            install_dir,
            LAST_KNOWN_GOOD_MARKER,
        )
        if last_good != args.last_known_good:
            raise SystemExit(
                "Expected last-known-good runtime "
                f"{args.last_known_good}, got {last_good}",
            )
    pending = read_runtime_marker(install_dir, PENDING_MARKER)
    if pending and not args.allow_pending:
        raise SystemExit(f"Pending runtime transaction remains: {pending}")
    if args.failed is not None:
        failed = read_runtime_marker(install_dir, FAILED_MARKER)
        if failed != args.failed:
            raise SystemExit(
                f"Expected failed runtime {args.failed}, got {failed}",
            )
    print(
        f"verified installed runtime {selected.version} "
        f"ABI {selected.runtime_abi} ({selected.file_count} files)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
