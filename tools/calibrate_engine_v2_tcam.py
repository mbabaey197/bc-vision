from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.engine_v2.calibration import (
    CALIBRATION_SCHEMA,
    CalibrationRequirements,
    analyze_static_ocr,
    calibrate,
    load_calibration_dataset,
)
from app.engine_v2.inference import InferenceConfig, SharedInferenceBackend
from app.engine_v2.ir_lpr import load_ir_lpr
from app.engine_v2.model_adapters import CTCPlateOCR, CTCPlateOCRConfig
from app.engine_v2.quality import evaluate_plate_quality


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or not image.size:
        raise ValueError(f"OpenCV could not decode {path}")
    return image


def collect_ir_lpr(args: argparse.Namespace) -> int:
    dataset_root = Path(args.dataset_root).resolve()
    model_path = Path(args.ocr_model).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(output_path)
    report_output = (
        Path(args.static_report_output).resolve()
        if args.static_report_output is not None
        else None
    )
    if report_output is not None and report_output.exists() and not args.overwrite:
        raise FileExistsError(report_output)
    if not 0.0 <= args.assumed_detector_confidence <= 1.0:
        raise ValueError("assumed detector confidence must be within 0..1")
    index = load_ir_lpr(dataset_root, strict=not args.skip_invalid)
    backend_config = InferenceConfig(
        model_path=model_path,
        backend=args.backend,
        device=args.device,
        allow_fallback=args.allow_inference_fallback,
    )
    ocr_config = CTCPlateOCRConfig(
        beam_width=args.beam_width,
        top_k=args.top_k,
        constrain_iranian_layout=args.constrain_iranian_layout,
    )
    tracks: list[dict[str, object]] = []
    with SharedInferenceBackend(backend_config) as backend:
        ocr = CTCPlateOCR(backend, ocr_config)
        for sample in index.samples:
            image = _read_image(sample.image_path)
            box = sample.plate_bbox
            crop = image[box.ymin : box.ymax, box.xmin : box.xmax]
            result = ocr.read(crop)
            quality = evaluate_plate_quality(
                crop,
                detector_confidence=args.assumed_detector_confidence,
                frame_shape=image.shape[:2],
                bbox=(box.xmin, box.ymin, box.xmax, box.ymax),
            )
            candidates = result.metadata.get("candidates", [])
            tracks.append(
                {
                    "track_id": sample.sample_id,
                    "split": sample.calibration_split,
                    "profile": args.profile,
                    "expected_plate": sample.expected_plate,
                    "observations": [
                        {
                            "seq": 1,
                            "ts": 0.0,
                            "text": result.text,
                            "confidence": result.confidence,
                            "quality": quality.score,
                            "plate_width": box.width,
                            "plate_height": box.height,
                            "valid": result.valid,
                            "character_confidences": list(result.character_confidences),
                            "candidates": candidates
                            if isinstance(candidates, list)
                            else [],
                        }
                    ],
                }
            )
        runtime_metadata = asdict(backend.metadata)

    payload = {
        "schema": CALIBRATION_SCHEMA,
        "dataset_id": f"ir-lpr-{index.fingerprint_sha256[:16]}",
        "label_scope": "exhaustive",
        "metadata": {
            "source": "mut-deep/IR-LPR",
            "source_kind": "single-image-static-ocr",
            "ir_lpr_fingerprint_sha256": index.fingerprint_sha256,
            "ocr_model_sha256": _sha256(model_path),
            "runtime": runtime_metadata,
            "ocr_config": asdict(ocr_config),
            "profile_assignment": args.profile,
            "assumed_detector_confidence": args.assumed_detector_confidence,
            "limitations": [
                "no_temporal_frames",
                "no_tracking_evidence",
                "no_verified-negative-tracks",
                "day-night-profile-is-operator-assigned",
                "ground-truth-crop-not-detector-output",
            ],
            "skipped_annotations": list(index.skipped_annotations),
        },
        "tracks": tracks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    static_report = None
    if report_output is not None:
        static_report = analyze_static_ocr(
            load_calibration_dataset(output_path)
        ).as_dict()
        report_output.parent.mkdir(parents=True, exist_ok=True)
        report_output.write_text(
            json.dumps(static_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output": str(output_path),
                "static_report_output": (
                    str(report_output) if report_output is not None else None
                ),
                "tracks": len(tracks),
                "profile": args.profile,
                "skipped": len(index.skipped_annotations),
                "promotion_eligible": False,
                "reason": "IR-LPR is static positive OCR evidence, not temporal/negative camera evidence",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def analyze_static(args: argparse.Namespace) -> int:
    dataset = load_calibration_dataset(args.dataset)
    report = analyze_static_ocr(
        dataset,
        confidence_bin_count=args.confidence_bin_count,
    )
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = report.as_dict()
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def search_thresholds(args: argparse.Namespace) -> int:
    dataset = load_calibration_dataset(args.dataset)
    grids: dict[str, dict[str, list[object]]] = _default_grids()
    if args.grid is not None:
        raw_grid: Any = json.loads(Path(args.grid).read_text(encoding="utf-8"))
        if not isinstance(raw_grid, dict) or any(
            not isinstance(value, dict) for value in raw_grid.values()
        ):
            raise TypeError("grid JSON must map profile names to field/value arrays")
        grids = raw_grid
    requirements = CalibrationRequirements(
        target_exact_accuracy=args.target_exact_accuracy,
        minimum_event_recall=args.minimum_event_recall,
        minimum_event_precision=args.minimum_event_precision,
        maximum_false_accept_rate=args.maximum_false_accept_rate,
        maximum_wrong_event_rate=args.maximum_wrong_event_rate,
        maximum_mean_character_error_rate=args.maximum_cer,
        minimum_train_tracks=args.minimum_train_tracks,
        minimum_holdout_tracks=args.minimum_holdout_tracks,
        maximum_grid_candidates=args.maximum_grid_candidates,
    )
    report = calibrate(dataset, grids=grids, requirements=requirements)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.as_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    return 0 if report.valid else 2


def merge_traces(args: argparse.Namespace) -> int:
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)
    tracks: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for source_index, raw_path in enumerate(args.input):
        path = Path(raw_path).resolve()
        dataset = load_calibration_dataset(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        for track in payload["tracks"]:
            copied = dict(track)
            copied["track_id"] = f"s{source_index}-{track['track_id']}"
            tracks.append(copied)
        sources.append(
            {
                "dataset_id": dataset.dataset_id,
                "fingerprint_sha256": dataset.fingerprint_sha256,
                "path": str(path),
                "tracks": len(dataset.tracks),
                "metadata": dict(dataset.metadata),
            }
        )
    merged = {
        "schema": CALIBRATION_SCHEMA,
        "dataset_id": args.dataset_id,
        "label_scope": "exhaustive",
        "metadata": {"merged_sources": sources},
        "tracks": tracks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Parse the exact persisted bytes so invalid source combinations fail now.
    verified = load_calibration_dataset(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "dataset_id": verified.dataset_id,
                "fingerprint_sha256": verified.fingerprint_sha256,
                "tracks": len(verified.tracks),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect IR-LPR OCR traces and calibrate Engine V2 TCAM policies."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser(
        "collect-ir-lpr",
        help="run the configured OCR model on labelled IR-LPR plate crops",
    )
    collect.add_argument("--dataset-root", required=True)
    collect.add_argument("--ocr-model", required=True)
    collect.add_argument("--output", required=True)
    collect.add_argument("--static-report-output")
    collect.add_argument("--backend", default="auto")
    collect.add_argument("--device", default="AUTO")
    collect.add_argument("--profile", choices=("day", "night"), default="day")
    collect.add_argument("--beam-width", type=int, default=8)
    collect.add_argument("--top-k", type=int, default=3)
    collect.add_argument(
        "--constrain-iranian-layout",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    collect.add_argument("--assumed-detector-confidence", type=float, default=0.0)
    collect.add_argument("--allow-inference-fallback", action="store_true")
    collect.add_argument("--skip-invalid", action="store_true")
    collect.add_argument("--overwrite", action="store_true")
    collect.set_defaults(handler=collect_ir_lpr)

    static = subparsers.add_parser(
        "analyze-static",
        help="measure exact-match, CER, and confidence calibration on labelled crops",
    )
    static.add_argument("--dataset", required=True)
    static.add_argument("--output", required=True)
    static.add_argument("--confidence-bin-count", type=int, default=10)
    static.add_argument("--overwrite", action="store_true")
    static.set_defaults(handler=analyze_static)

    search = subparsers.add_parser(
        "search", help="select policies on train and evaluate once on holdout"
    )
    search.add_argument("--dataset", required=True)
    search.add_argument("--grid")
    search.add_argument("--output", required=True)
    search.add_argument("--target-exact-accuracy", type=float, default=0.99)
    search.add_argument("--minimum-event-recall", type=float, default=0.99)
    search.add_argument("--minimum-event-precision", type=float, default=0.995)
    search.add_argument("--maximum-false-accept-rate", type=float, default=0.001)
    search.add_argument("--maximum-wrong-event-rate", type=float, default=0.001)
    search.add_argument("--maximum-cer", type=float, default=0.01)
    search.add_argument("--minimum-train-tracks", type=int, default=50)
    search.add_argument("--minimum-holdout-tracks", type=int, default=50)
    search.add_argument("--maximum-grid-candidates", type=int, default=50_000)
    search.set_defaults(handler=search_thresholds)

    merge = subparsers.add_parser(
        "merge", help="merge immutable static and temporal trace files"
    )
    merge.add_argument("--input", action="append", required=True)
    merge.add_argument("--dataset-id", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--overwrite", action="store_true")
    merge.set_defaults(handler=merge_traces)
    return parser


def _default_grids() -> dict[str, dict[str, list[object]]]:
    common: dict[str, list[object]] = {
        "provisional_confidence": [0.72, 0.75, 0.78],
        "lock_confidence": [0.84, 0.86, 0.88],
        "express_lock_confidence": [0.93, 0.95, 0.97],
        "min_slot_confidence": [0.76, 0.78, 0.82],
        "min_slot_margin": [0.12, 0.16, 0.20],
        "min_ocr_quality": [0.28, 0.32, 0.36],
        "soft_lock_hold_seconds": [0.08, 0.12, 0.20],
    }
    return {"day": dict(common), "night": dict(common)}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
