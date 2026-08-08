"""Export the pinned official Hezar Persian plate CRNN to ONNX.

This development utility does not redistribute model weights. It downloads
the selected Hezar model into a caller-controlled cache, exports a fixed CPU
inference graph, and verifies numerical parity with ONNX Runtime.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.hezar_export import (  # noqa: E402
    export_pinned_model,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Hezar Persian plate CRNN to ONNX",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = export_pinned_model(
        args.output.resolve(),
        args.cache_dir.resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
