"""Skip repeated package resolution when pinned Windows locks are unchanged."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys


def dependency_fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    digest.update(sys.version.encode("utf-8"))
    for path in paths:
        resolved = Path(path)
        digest.update(resolved.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(resolved.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest().upper()


def stamp_matches(stamp: Path, paths: list[Path]) -> bool:
    stamp = Path(stamp)
    if not stamp.is_file():
        return False
    return stamp.read_text(encoding="ascii").strip() == (
        dependency_fingerprint(paths)
    )


def write_stamp(stamp: Path, paths: list[Path]) -> None:
    stamp = Path(stamp)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    temporary = stamp.with_suffix(stamp.suffix + ".tmp")
    temporary.write_text(
        dependency_fingerprint(paths) + "\n",
        encoding="ascii",
    )
    temporary.replace(stamp)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "write"))
    parser.add_argument("stamp", type=Path)
    parser.add_argument("locks", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.action == "check":
        return 0 if stamp_matches(args.stamp, args.locks) else 1
    write_stamp(args.stamp, args.locks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
