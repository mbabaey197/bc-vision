"""Build a hash-verified, non-training ANPR Golden Dataset manifest."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ai.golden import validate_golden_manifest
from app.ai.plate_rules import normalize_plate, plausible_plate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().lower()


def prepare(input_csv: Path, output: Path) -> dict:
    input_csv = Path(input_csv).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Golden output already exists; choose a new folder: {output}"
        )
    lowered_parts = {
        part.lower()
        for part in output.parts
    }
    if {"anpr-training", "training-data"} & lowered_parts:
        raise ValueError("Golden data cannot be placed in a training folder")

    media_root = output / "media"
    media_root.mkdir(parents=True)
    rows = []
    with input_csv.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            sample_id = str(row.get("id", "")).strip()
            source_value = str(row.get("media_path", "")).strip()
            if not sample_id or not source_value:
                raise ValueError(f"Missing id/media_path on CSV row {index}")
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", sample_id):
                raise ValueError(f"Unsafe sample id on CSV row {index}")
            source = Path(source_value).expanduser()
            if not source.is_absolute():
                source = (input_csv.parent / source).resolve()
            if not source.is_file():
                raise FileNotFoundError(source)

            readable_value = str(row.get("readable", "1")).strip().lower()
            readable = readable_value not in {"0", "false", "no", "خیر"}
            expected = normalize_plate(row.get("expected_plate", ""))
            if readable != plausible_plate(expected):
                raise ValueError(
                    f"Invalid readable/plate label on CSV row {index}"
                )
            slices = sorted({
                value.strip().lower()
                for value in str(row.get("slices", "")).split("|")
                if value.strip()
            })
            media_kind = str(
                row.get("media_kind", "frame")
            ).strip().lower()
            if media_kind not in {"frame", "crop"}:
                raise ValueError(
                    f"Invalid media_kind on CSV row {index}"
                )
            digest = _sha256(source)
            suffix = source.suffix.lower() or ".bin"
            target = media_root / f"{digest[:24]}-{sample_id}{suffix}"
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copy2(source, temporary)
            if _sha256(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"Media copy hash mismatch: {source}")
            os.replace(temporary, target)
            prepared = {
                "id": sample_id,
                "frame_index": int(row.get("frame_index") or -1),
                "sha256": digest,
                "expected_plate": expected,
                "readable": readable,
                "slices": slices,
                "label_source": "operator",
                "training_allowed": False,
            }
            prepared[
                "crop_path" if media_kind == "crop" else "frame_path"
            ] = str(target.relative_to(output))
            rows.append(prepared)

    payload = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label_source": "operator",
        "training_allowed": False,
        "samples": rows,
    }
    manifest = output / "manifest.json"
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    status = validate_golden_manifest(
        payload,
        output,
        verify_media=True,
    )
    status["manifest"] = str(manifest)
    return status


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare BC Vision's operator-labelled Golden Dataset. "
            "CSV columns: id,media_path,media_kind(frame|crop),frame_index,"
            "expected_plate,readable,slices (pipe-separated)."
        ),
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    status = prepare(args.input_csv, args.output)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
