from __future__ import annotations

import struct
import sys
from pathlib import Path


IMAGE_SUBSYSTEM_WINDOWS_GUI = 2


def read_subsystem(executable: Path) -> int:
    with executable.open("rb") as handle:
        if handle.read(2) != b"MZ":
            raise ValueError("missing DOS executable signature")

        handle.seek(0x3C)
        pe_offset_raw = handle.read(4)
        if len(pe_offset_raw) != 4:
            raise ValueError("truncated DOS header")
        pe_offset = struct.unpack("<I", pe_offset_raw)[0]

        handle.seek(pe_offset)
        if handle.read(4) != b"PE\0\0":
            raise ValueError("missing PE signature")

        optional_header = pe_offset + 4 + 20
        handle.seek(optional_header)
        magic_raw = handle.read(2)
        if len(magic_raw) != 2:
            raise ValueError("truncated PE optional header")
        magic = struct.unpack("<H", magic_raw)[0]
        if magic not in (0x10B, 0x20B):
            raise ValueError(f"unsupported PE optional-header magic: {magic:#x}")

        handle.seek(optional_header + 68)
        subsystem_raw = handle.read(2)
        if len(subsystem_raw) != 2:
            raise ValueError("truncated PE subsystem field")
        return struct.unpack("<H", subsystem_raw)[0]


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_windows_gui_subsystem.py <executable>")
        return 2

    executable = Path(sys.argv[1])
    try:
        subsystem = read_subsystem(executable)
    except (OSError, ValueError) as exc:
        print(f"PE verification failed: {exc}")
        return 1

    if subsystem != IMAGE_SUBSYSTEM_WINDOWS_GUI:
        print(
            "BCVision.exe is not a windowless GUI executable: "
            f"subsystem={subsystem}"
        )
        return 1

    print(
        "BCVision.exe uses IMAGE_SUBSYSTEM_WINDOWS_GUI "
        "(no console window)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
