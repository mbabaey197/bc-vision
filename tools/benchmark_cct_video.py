"""Benchmark a BC Vision CCT OCR candidate on a fixed operator-labelled video."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.detector import detect_plates, detector_status
from app.ai.evaluation import character_distance
from app.ai.onnx_cct import (
    _validate_session_contract,
    accept_cct_hypotheses,
    decode_cct_hypotheses,
    prepare_cct_input,
)
from app.ai.pipeline import PlateConsensusTracker, image_quality
from app.ai.plate_rules import (
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)
from app.cpu_budget import threads_per_camera


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _serializable(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"crop", "capture_frame", "quality"}
    }


def _persist_emitted_rows(
    rows,
    *,
    artifact_dir: Path | None,
    start_index: int,
    frame_number: int,
    fps: float,
) -> list[dict]:
    serialized = []
    crop_dir = artifact_dir / "crops" if artifact_dir else None
    if crop_dir is not None:
        crop_dir.mkdir(parents=True, exist_ok=True)
    for offset, raw in enumerate(rows, start=1):
        result = dict(raw)
        result["frame"] = int(frame_number)
        result["video_second"] = round(
            float(result.get("last_seen", frame_number / fps)),
            3,
        )
        crop = result.get("crop")
        if (
            crop_dir is not None
            and crop is not None
            and getattr(crop, "size", 0)
        ):
            filename = f"plate-{start_index + offset:04d}.jpg"
            crop_path = crop_dir / filename
            if not cv2.imwrite(
                str(crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            ):
                raise OSError(f"Could not save benchmark crop: {crop_path}")
            result["crop_path"] = str(Path("crops") / filename)
        serialized.append(_serializable(result))
    return serialized


def _raw_guess_from_hypotheses(
    hypotheses: list[dict],
    *,
    accepted: bool,
    reason: str,
    frame_number: int,
    fps: float,
    detection: dict,
) -> dict | None:
    if not hypotheses:
        return None
    best = hypotheses[0]
    normalized = normalize_plate(
        best.get("plate_norm") or best.get("plate")
    )
    if not plausible_plate(normalized):
        return None
    return {
        "plate": format_iran_plate(normalized),
        "plate_norm": normalized,
        "confidence": float(best.get("confidence", 0.0)),
        "accepted": bool(accepted),
        "reason": str(reason or ""),
        "frame": int(frame_number),
        "video_second": round(frame_number / fps, 3),
        "detector_confidence": float(
            detection.get("confidence", 0.0)
        ),
        "bbox": detection.get("bbox"),
        "method": str(detection.get("method", "")),
        "crop": detection.get("crop"),
    }


def _update_nearest_raw(
    nearest: dict[str, dict],
    observation: dict,
    truth_plates: set[str],
) -> None:
    observed = str(observation["plate_norm"])
    for truth in truth_plates:
        candidate = {
            **observation,
            "truth": format_iran_plate(truth),
            "truth_norm": truth,
            "character_distance": character_distance(observed, truth),
        }
        previous = nearest.get(truth)
        candidate_rank = (
            int(candidate["character_distance"]),
            -float(candidate["confidence"]),
            int(candidate["frame"]),
        )
        previous_rank = (
            int(previous["character_distance"]),
            -float(previous["confidence"]),
            int(previous["frame"]),
        ) if previous else None
        if previous_rank is None or candidate_rank < previous_rank:
            crop = candidate.get("crop")
            if crop is not None and getattr(crop, "size", 0):
                candidate["crop"] = crop.copy()
            nearest[truth] = candidate


def _persist_nearest_raw(
    nearest: dict[str, dict],
    *,
    artifact_dir: Path | None,
) -> list[dict]:
    raw_dir = artifact_dir / "raw-nearest" if artifact_dir else None
    if raw_dir is not None:
        raw_dir.mkdir(parents=True, exist_ok=True)
    serialized = []
    for index, truth in enumerate(sorted(nearest), start=1):
        result = dict(nearest[truth])
        crop = result.get("crop")
        if (
            raw_dir is not None
            and crop is not None
            and getattr(crop, "size", 0)
        ):
            filename = f"truth-{index:02d}.jpg"
            crop_path = raw_dir / filename
            if not cv2.imwrite(
                str(crop_path),
                crop,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            ):
                raise OSError(
                    f"Could not save nearest raw crop: {crop_path}"
                )
            result["crop_path"] = str(
                Path("raw-nearest") / filename
            )
        serialized.append(_serializable(result))
    return serialized


def _session_options(ort):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads_per_camera()
    options.inter_op_num_threads = 1
    if hasattr(ort, "ExecutionMode"):
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if hasattr(ort, "GraphOptimizationLevel"):
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
    add_entry = getattr(options, "add_session_config_entry", None)
    if callable(add_entry):
        add_entry("session.intra_op.allow_spinning", "0")
        add_entry("session.inter_op.allow_spinning", "0")
    return options


def benchmark(
    video_path: Path,
    model_path: Path,
    metadata_path: Path,
    frame_step: int,
    truth_plates: list[str],
    expected_video_sha256="",
    artifact_dir: Path | None = None,
) -> dict:
    import onnxruntime as ort

    actual_video_sha256 = _sha256(video_path)
    expected_digest = str(expected_video_sha256).strip().upper()
    if expected_digest and actual_video_sha256 != expected_digest:
        raise ValueError(
            "Video SHA-256 does not match the fixed benchmark input"
        )
    spec = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_model_digest = str(spec.get("sha256", "")).strip().upper()
    if expected_model_digest and _sha256(model_path) != expected_model_digest:
        raise ValueError("CCT model SHA-256 does not match metadata")
    session = ort.InferenceSession(
        str(model_path),
        sess_options=_session_options(ort),
        providers=["CPUExecutionProvider"],
    )
    _validate_session_contract(session, spec)
    input_name = session.get_inputs()[0].name
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Video cannot be opened: {video_path}")
    fps = max(float(capture.get(cv2.CAP_PROP_FPS)), 1.0)
    tracker = PlateConsensusTracker(
        min_votes=3,
        max_age_seconds=2.5,
        emit_cooldown=2.5,
        emit_unreadable=False,
    )
    frame_number = 0
    processed_frames = 0
    detection_count = 0
    emitted = []
    normalized_truth = {
        normalize_plate(value)
        for value in truth_plates
        if plausible_plate(value)
    }
    raw_guess_count = 0
    raw_guess_frequencies: Counter[str] = Counter()
    nearest_raw: dict[str, dict] = {}
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_number += 1
            if frame_number % max(1, int(frame_step)):
                continue
            processed_frames += 1
            rows = []
            for item in detect_plates(frame, min_confidence=0.20):
                detection_count += 1
                crop = item["crop"]
                tensor = prepare_cct_input(crop, spec)
                if tensor is None:
                    continue
                output = session.run(
                    None,
                    {input_name: tensor},
                )[0]
                hypotheses = decode_cct_hypotheses(
                    output,
                    alphabet=str(spec["alphabet"]),
                    max_plate_slots=int(spec["max_plate_slots"]),
                    beam_width=int(spec.get("beam_width", 16)),
                    top_k=int(spec.get("top_k", 5)),
                )
                ocr = accept_cct_hypotheses(
                    hypotheses,
                    min_confidence=float(
                        spec.get("min_confidence", 0.58)
                    ),
                    min_position_confidence=float(
                        spec.get("min_position_confidence", 0.42)
                    ),
                    min_position_margin=float(
                        spec.get("min_position_margin", 0.06)
                    ),
                    min_hypothesis_margin=float(
                        spec.get("min_hypothesis_margin", 0.025)
                    ),
                )
                quality = image_quality(crop)
                raw_guess = _raw_guess_from_hypotheses(
                    hypotheses,
                    accepted=bool(ocr["accepted"]),
                    reason=str(ocr.get("reason", "")),
                    frame_number=frame_number,
                    fps=fps,
                    detection=item,
                )
                if raw_guess is not None:
                    raw_guess_count += 1
                    raw_guess_frequencies[
                        str(raw_guess["plate_norm"])
                    ] += 1
                    _update_nearest_raw(
                        nearest_raw,
                        raw_guess,
                        normalized_truth,
                    )
                normalized = normalize_plate(
                    ocr.get("plate_norm", "")
                )
                valid = bool(ocr["accepted"]) and plausible_plate(normalized)
                ocr_confidence = float(ocr["confidence"])
                confidence = min(
                    1.0,
                    0.34 * float(item["confidence"])
                    + 0.56 * ocr_confidence
                    + 0.10 * float(quality["score"])
                    + (0.08 if valid else 0.0),
                )
                rows.append({
                    "plate": (
                        format_iran_plate(normalized)
                        if valid
                        else (
                            raw_guess["plate"]
                            if raw_guess is not None
                            else "ناخوانا"
                        )
                    ),
                    "plate_norm": normalized if valid else "",
                    "confidence": confidence,
                    "detector_confidence": float(item["confidence"]),
                    "ocr_confidence": ocr_confidence,
                    "quality_score": quality["score"],
                    "quality": quality,
                    "bbox": item["bbox"],
                    "crop": crop,
                    "method": item["method"],
                    "ocr_engine": "fast-plate-ocr-cct",
                    "plate_hypotheses": hypotheses,
                    "position_hypotheses": [],
                    "valid": valid,
                    "best_effort": bool(not valid and raw_guess),
                    "needs_review": not valid,
                    "read_status": (
                        "confirmed-ai"
                        if valid
                        else "experimental-guess"
                        if raw_guess is not None
                        else "unreadable"
                    ),
                    "raw_guess_text": (
                        raw_guess["plate"]
                        if raw_guess is not None
                        else ""
                    ),
                    "raw_guess_norm": (
                        raw_guess["plate_norm"]
                        if raw_guess is not None
                        else ""
                    ),
                    "raw_guess_confidence": (
                        raw_guess["confidence"]
                        if raw_guess is not None
                        else 0.0
                    ),
                    "raw_guess_engine": "fast-plate-ocr-cct",
                    "raw_guess_reason": str(ocr.get("reason", "")),
                    "experimental": not valid,
                    "hypotheses_accepted_for_consensus": valid,
                })
            emitted.extend(
                _persist_emitted_rows(
                    tracker.update(
                        rows,
                        timestamp=frame_number / fps,
                    ),
                    artifact_dir=artifact_dir,
                    start_index=len(emitted),
                    frame_number=frame_number,
                    fps=fps,
                )
            )
    finally:
        capture.release()
    emitted.extend(
        _persist_emitted_rows(
            tracker.flush(),
            artifact_dir=artifact_dir,
            start_index=len(emitted),
            frame_number=frame_number,
            fps=fps,
        )
    )
    for truth, row in nearest_raw.items():
        row["exact_observations"] = int(
            raw_guess_frequencies.get(truth, 0)
        )
    nearest_raw_rows = _persist_nearest_raw(
        nearest_raw,
        artifact_dir=artifact_dir,
    )
    raw_exact_truth = sorted(
        truth
        for truth, row in nearest_raw.items()
        if int(row["character_distance"]) == 0
    )
    emitted_norms = {
        normalize_plate(result.get("plate_norm", ""))
        for result in emitted
        if plausible_plate(result.get("plate_norm", ""))
    }
    matched_truth = sorted(normalized_truth & emitted_norms)
    missed_truth = sorted(normalized_truth - emitted_norms)
    unmatched_emitted = sorted(emitted_norms - normalized_truth)
    return {
        "video": str(video_path),
        "video_sha256": actual_video_sha256,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "artifact_dir": str(artifact_dir) if artifact_dir else "",
        "frames": frame_number,
        "processed_frames": processed_frames,
        "detections": detection_count,
        "detector": detector_status(),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "truth_count": len(normalized_truth),
        "matched_truth_count": len(matched_truth),
        "matched_truth": [
            format_iran_plate(value)
            for value in matched_truth
        ],
        "truth_recall": round(
            len(matched_truth) / len(normalized_truth),
            6,
        ) if normalized_truth else 0.0,
        "missed_truth": [
            format_iran_plate(value)
            for value in missed_truth
        ],
        "raw_guess_observation_count": raw_guess_count,
        "raw_guess_unique_count": len(raw_guess_frequencies),
        "raw_exact_truth_count": len(raw_exact_truth),
        "raw_exact_truth": [
            format_iran_plate(value)
            for value in raw_exact_truth
        ],
        "nearest_raw_by_truth": nearest_raw_rows,
        "top_raw_guesses": [
            {
                "plate": format_iran_plate(plate),
                "plate_norm": plate,
                "observations": count,
            }
            for plate, count in raw_guess_frequencies.most_common(25)
        ],
        "emitted_unique_count": len(emitted_norms),
        "unmatched_emitted_unique_count": len(unmatched_emitted),
        "unmatched_emitted": [
            format_iran_plate(value)
            for value in unmatched_emitted
        ],
        "emitted_count": len(emitted),
        "emitted": emitted,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark a CCT ONNX candidate on BC Vision video",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--truth", action="append", default=[])
    parser.add_argument("--expected-video-sha256", default="")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        help="Optional directory for the exact plate crop behind each row",
    )
    args = parser.parse_args(argv)
    result = benchmark(
        args.video.resolve(),
        args.ocr_model.resolve(),
        args.metadata.resolve(),
        args.frame_step,
        list(args.truth),
        expected_video_sha256=args.expected_video_sha256,
        artifact_dir=(
            args.artifacts_dir.resolve()
            if args.artifacts_dir is not None
            else None
        ),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "frames": result["frames"],
        "processed_frames": result["processed_frames"],
        "detections": result["detections"],
        "elapsed_seconds": result["elapsed_seconds"],
        "truth_count": result["truth_count"],
        "matched_truth_count": result["matched_truth_count"],
        "matched_truth": result["matched_truth"],
        "truth_recall": result["truth_recall"],
        "raw_exact_truth_count": result["raw_exact_truth_count"],
        "unmatched_emitted_unique_count": (
            result["unmatched_emitted_unique_count"]
        ),
        "emitted_count": result["emitted_count"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
