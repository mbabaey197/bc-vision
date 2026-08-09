from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_payload import build_runtime_payload  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a versioned BC Vision fast-update payload",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fast_update_payload"),
    )
    args = parser.parse_args()
    result = build_runtime_payload(ROOT, args.output)
    print(
        f"runtime payload {result.version} ABI {result.runtime_abi}: "
        f"{result.file_count} files at {result.root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
