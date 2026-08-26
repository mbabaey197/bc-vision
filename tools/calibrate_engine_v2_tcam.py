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
from app.engine_v2.types import OCRResult

_CCT_RUNTIME = "fast-plate-ocr-cct"
_HEZAR_RUNTIME = "hezar-v2"
_CCT_REQUIRED_SPEC_KEYS = frozenset(
    {
        "alphabet",
        "max_plate_slots",
        "input_width",
        "input_height",
        "input_layout",
        "input_dtype",
        "image_color_mode",
    }
)


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


def _verify_model_file(
    model_path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    model_name: str,
) -> None:
    digest = expected_sha256.strip().lower()
    if expected_size < 1 or len(digest) != 64:
        raise ValueError(f"{model_name} has an invalid pinned model contract")
    if model_path.stat().st_size != expected_size or _sha256(model_path) != digest:
        raise ValueError(
            f"{model_name} does not match its pinned size/SHA-256 contract"
        )


def _load_hezar_spec(
    model_path: Path,
    *,
    beam_width: int | None,
    top_k: int | None,
) -> dict[str, object]:
    # Load the contract lazily so the calibration CLI remains importable in
    # minimal Engine V2 test environments without the production AI package.
    from app.ai.model_manager import HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE
    from app.ai.onnx_hezar import HEZAR_V2_SPEC

    _verify_model_file(
        model_path,
        expected_size=int(HEZAR_ONNX_SIZE),
        expected_sha256=str(HEZAR_ONNX_SHA256),
        model_name="Hezar v2",
    )
    spec = dict(HEZAR_V2_SPEC)
    if beam_width is not None:
        spec["beam_width"] = beam_width
    if top_k is not None:
        spec["top_k"] = top_k
    if int(spec.get("beam_width", 10)) < 1:
        raise ValueError("Hezar beam_width must be positive")
    if not 1 <= int(spec.get("top_k", 5)) <= int(spec.get("beam_width", 10)):
        raise ValueError("Hezar top_k must be within 1..beam_width")
    return spec


def _load_cct_spec(
    model_path: Path,
    manifest_path: Path | None,
    *,
    beam_width: int | None,
    top_k: int | None,
) -> tuple[dict[str, object], Path]:
    source = (
        manifest_path.resolve()
        if manifest_path is not None
        else model_path.parent.joinpath("active-models.json").resolve()
    )
    payload: Any = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("CCT manifest root must be an object")
    if isinstance(payload.get("models"), dict):
        raw_spec = payload["models"].get("ocr")
    else:
        raw_spec = payload
    if not isinstance(raw_spec, dict):
        raise TypeError("CCT manifest must contain models.ocr")
    spec = dict(raw_spec)
    if str(spec.get("runtime", "")).strip().lower() != _CCT_RUNTIME:
        raise ValueError(f"CCT manifest runtime must be {_CCT_RUNTIME!r}")
    missing = sorted(_CCT_REQUIRED_SPEC_KEYS - set(spec))
    if missing:
        raise ValueError(f"CCT manifest is missing contract keys: {missing}")
    expected_size = spec.get("size")
    expected_sha256 = str(spec.get("sha256", "")).strip().lower()
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or len(expected_sha256) != 64
    ):
        raise ValueError("CCT manifest must bind model size and SHA-256")
    _verify_model_file(
        model_path,
        expected_size=expected_size,
        expected_sha256=expected_sha256,
        model_name="CCT",
    )
    if beam_width is not None:
        spec["beam_width"] = beam_width
    if top_k is not None:
        spec["top_k"] = top_k
    if int(spec.get("beam_width", 16)) < 1:
        raise ValueError("CCT beam_width must be positive")
    if not 1 <= int(spec.get("top_k", 5)) <= int(spec.get("beam_width", 16)):
        raise ValueError("CCT top_k must be within 1..beam_width")
    return spec, source


def _hypothesis_candidates(
    hypotheses: object,
    *,
    accepted: bool,
) -> list[dict[str, object]]:
    if not accepted or not isinstance(hypotheses, list):
        return []
    candidates: list[dict[str, object]] = []
    for hypothesis in hypotheses:
        if not isinstance(hypothesis, dict):
            continue
        text = str(
            hypothesis.get("plate_norm") or hypothesis.get("plate") or ""
        ).strip()
        if not text:
            continue
        raw_positions = hypothesis.get("positions", {})
        positions = raw_positions if isinstance(raw_positions, dict) else {}
        character_confidences = tuple(
            float(value.get("confidence", 0.0))
            for _, value in sorted(positions.items(), key=lambda item: int(item[0]))
            if isinstance(value, dict)
        )
        confidence = float(hypothesis.get("confidence", 0.0))
        candidates.append(
            {
                "text": text,
                "confidence": confidence,
                "weight": confidence,
                "character_confidences": character_confidences,
            }
        )
    return candidates


def _cct_result_to_ocr(payload: dict[str, object]) -> OCRResult:
    raw_hypotheses = payload.get("hypotheses", [])
    accepted = bool(payload.get("accepted", False))
    candidates = _hypothesis_candidates(raw_hypotheses, accepted=accepted)
    text = str(
        payload.get("raw_plate_norm")
        or payload.get("plate_norm")
        or (candidates[0]["text"] if candidates else "")
    ).strip()
    character_confidences = (
        tuple(candidates[0]["character_confidences"]) if candidates else ()
    )
    return OCRResult(
        text=text,
        confidence=float(payload.get("confidence", 0.0)),
        valid=accepted,
        character_confidences=character_confidences,
        metadata={
            "candidates": candidates,
            "decoder": _CCT_RUNTIME,
            "rejection_reason": str(payload.get("reason", "")),
        },
    )


def _hezar_result_to_ocr(payload: dict[str, object]) -> OCRResult:
    raw_hypotheses = payload.get("hypotheses", [])
    hypotheses = raw_hypotheses if isinstance(raw_hypotheses, list) else []
    accepted = bool(payload.get("accepted", False))
    candidates = _hypothesis_candidates(hypotheses, accepted=accepted)
    raw_details = payload.get("position_details", [])
    position_details = raw_details if isinstance(raw_details, list) else []
    character_confidences = tuple(
        float(item.get("probability", 0.0))
        for item in position_details
        if isinstance(item, dict)
    )
    raw_text = str(
        payload.get("plate_norm")
        or (hypotheses[0].get("plate_norm", "") if hypotheses else "")
    ).strip()
    return OCRResult(
        text=raw_text,
        confidence=float(payload.get("confidence", 0.0)),
        valid=accepted,
        character_confidences=character_confidences,
        metadata={
            "candidates": candidates,
            "decoder": _HEZAR_RUNTIME,
            "rejection_reason": str(payload.get("reason", "")),
        },
    )


class _BackendSessionFacade:
    """Expose ONNX Runtime's small ``run`` surface over Engine V2."""

    def __init__(self, backend: SharedInferenceBackend) -> None:
        self.backend = backend

    def run(
        self,
        output_names: list[str] | tuple[str, ...] | None,
        input_feed: dict[str, object],
    ) -> list[object]:
        return self.backend.infer(input_feed, output_names)


class _CCTPlateOCR:
    def __init__(
        self,
        backend: SharedInferenceBackend,
        spec: dict[str, object],
    ) -> None:
        self.backend = backend
        self.spec = spec

    def read(self, crop: np.ndarray) -> OCRResult:
        # Import lazily: the complete BC Vision tree contains the production
        # CCT adapter, while isolated Engine V2 unit tests intentionally do not.
        from app.ai.onnx_cct import infer_cct_session

        result = infer_cct_session(
            _BackendSessionFacade(self.backend),
            self.backend.input_names[0],
            crop,
            self.spec,
        )
        if not isinstance(result, dict):
            raise TypeError("CCT inference result must be an object")
        return _cct_result_to_ocr(result)


class _HezarPlateOCR:
    def __init__(
        self,
        backend: SharedInferenceBackend,
        spec: dict[str, object],
    ) -> None:
        self.backend = backend
        self.spec = spec

    def read(self, crop: np.ndarray) -> OCRResult:
        from app.ai.onnx_hezar import (
            accept_hypotheses,
            ctc_beam_hypotheses,
            prepare_hezar_input,
        )

        tensor = prepare_hezar_input(crop, self.spec)
        if tensor is None:
            raise ValueError("Empty Hezar OCR crop")
        outputs = self.backend.infer({self.backend.input_names[0]: tensor})
        if not outputs:
            raise ValueError("Hezar inference returned no outputs")
        logits = np.asarray(outputs[0])
        if logits.ndim == 3:
            logits = logits[0]
        if bool(self.spec.get("reverse_output_digits", False)):
            logits = logits[::-1]
        labels = list(self.spec.get("labels") or [])
        hypotheses = ctc_beam_hypotheses(
            logits,
            labels=labels,
            blank_index=int(self.spec.get("blank_index", 0)),
            beam_width=int(self.spec.get("beam_width", 10)),
            top_k=int(self.spec.get("top_k", 5)),
        )
        result = accept_hypotheses(
            hypotheses,
            min_confidence=float(self.spec.get("min_confidence", 0.56)),
            min_position_margin=float(self.spec.get("min_position_margin", 0.12)),
        )
        if not isinstance(result, dict):
            raise TypeError("Hezar inference result must be an object")
        return _hezar_result_to_ocr(result)


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
    cct_spec: dict[str, object] | None = None
    hezar_spec: dict[str, object] | None = None
    ctc_config: CTCPlateOCRConfig | None = None
    ocr_config_metadata: dict[str, object]
    if args.ocr_runtime == "hezar":
        hezar_spec = _load_hezar_spec(
            model_path,
            beam_width=args.beam_width,
            top_k=args.top_k,
        )
        ocr_config_metadata = {
            "runtime": _HEZAR_RUNTIME,
            "contract": "app.ai.onnx_hezar.HEZAR_V2_SPEC",
            "spec": hezar_spec,
        }
    elif args.ocr_runtime == "cct":
        cct_spec, cct_manifest = _load_cct_spec(
            model_path,
            Path(args.ocr_manifest) if args.ocr_manifest is not None else None,
            beam_width=args.beam_width,
            top_k=args.top_k,
        )
        ocr_config_metadata = {
            "runtime": _CCT_RUNTIME,
            "manifest": str(cct_manifest),
            "spec": cct_spec,
        }
    else:
        beam_width = args.beam_width if args.beam_width is not None else 8
        top_k = args.top_k if args.top_k is not None else 3
        ctc_config = CTCPlateOCRConfig(
            beam_width=beam_width,
            top_k=top_k,
            constrain_iranian_layout=args.constrain_iranian_layout,
        )
        ocr_config_metadata = {
            "runtime": "ctc",
            "config": asdict(ctc_config),
        }
    tracks: list[dict[str, object]] = []
    with SharedInferenceBackend(backend_config) as backend:
        if cct_spec is not None:
            ocr = _CCTPlateOCR(backend, cct_spec)
        elif hezar_spec is not None:
            ocr = _HezarPlateOCR(backend, hezar_spec)
        elif ctc_config is not None:
            ocr = CTCPlateOCR(backend, ctc_config)
        else:  # pragma: no cover - guarded by argparse choices
            raise AssertionError("unconfigured OCR runtime")
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
        runtime_metadata = {
            "session": asdict(backend.metadata),
            "telemetry": asdict(backend.telemetry_snapshot()),
        }

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
            "ocr_config": ocr_config_metadata,
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
    collect.add_argument(
        "--ocr-runtime",
        choices=("hezar", "cct", "ctc"),
        default="hezar",
        help=(
            "Hezar v2 is the current production baseline; CCT is a signed "
            "Shadow candidate and ctc is the legacy generic adapter"
        ),
    )
    collect.add_argument(
        "--ocr-manifest",
        help="CCT active-models.json; defaults to the model's sibling file",
    )
    collect.add_argument("--output", required=True)
    collect.add_argument("--static-report-output")
    collect.add_argument("--backend", default="auto")
    collect.add_argument("--device", default="AUTO")
    collect.add_argument("--profile", choices=("day", "night"), default="day")
    collect.add_argument("--beam-width", type=int)
    collect.add_argument("--top-k", type=int)
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
