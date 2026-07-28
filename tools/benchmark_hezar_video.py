"""Benchmark an exported Hezar ONNX OCR with BC Vision's current detector.

This hybrid benchmark isolates the OCR change before a trained OBB detector is
available.  It never promotes a model and reports only exact matches for truth
plates supplied by the operator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.detector import detect_plates
from app.ai.onnx_hezar import (
    accept_hypotheses,
    ctc_beam_hypotheses,
    prepare_hezar_input,
)
from app.ai.pipeline import (
    PlateConsensusTracker,
    image_quality,
)
from app.ai.plate_rules import (
    format_iran_plate,
    normalize_plate,
    plausible_plate,
)


def _serializable(result: dict) -> dict:
    return {
        key: value
        for key, value in result.items()
        if key not in {"crop", "capture_frame", "quality"}
    }


def benchmark(
    video_path: Path,
    model_path: Path,
    metadata_path: Path,
    frame_step: int,
    truth_plates: list[str],
) -> dict:
    import onnxruntime as ort

    spec = json.loads(metadata_path.read_text(encoding="utf-8"))
    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
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
            for item in detect_plates(
                frame,
                min_confidence=0.20,
            ):
                detection_count += 1
                crop = item["crop"]
                tensor = prepare_hezar_input(crop, spec)
                if tensor is None:
                    continue
                logits = np.asarray(
                    session.run(None, {input_name: tensor})[0]
                )
                if logits.ndim == 3:
                    logits = logits[0]
                if bool(spec.get("reverse_output_digits", False)):
                    logits = logits[::-1]
                hypotheses = ctc_beam_hypotheses(
                    logits,
                    labels=list(spec["labels"]),
                    blank_index=int(spec["blank_index"]),
                    beam_width=int(spec.get("beam_width", 12)),
                    top_k=int(spec.get("top_k", 5)),
                )
                ocr = accept_hypotheses(
                    hypotheses,
                    min_confidence=float(
                        spec.get("min_confidence", 0.56)
                    ),
                    min_position_margin=float(
                        spec.get("min_position_margin", 0.12)
                    ),
                )
                quality = image_quality(crop)
                norm = normalize_plate(ocr.get("plate_norm", ""))
                valid = bool(ocr.get("accepted")) and plausible_plate(norm)
                ocr_confidence = float(ocr.get("confidence", 0.0))
                confidence = min(
                    1.0,
                    0.34 * float(item["confidence"])
                    + 0.56 * ocr_confidence
                    + 0.10 * float(quality["score"])
                    + (0.08 if valid else 0.0),
                )
                rows.append({
                    "plate": (
                        format_iran_plate(norm) if valid else "ناخوانا"
                    ),
                    "plate_norm": norm if valid else "",
                    "confidence": confidence,
                    "detector_confidence": float(item["confidence"]),
                    "ocr_confidence": ocr_confidence,
                    "quality_score": quality["score"],
                    "quality": quality,
                    "bbox": item["bbox"],
                    "crop": crop,
                    "method": item["method"],
                    "ocr_engine": "hezar-v2-onnx",
                    "plate_hypotheses": hypotheses,
                    "position_hypotheses": [],
                    "valid": valid,
                    "best_effort": False,
                    "needs_review": not valid,
                })
            emitted.extend(
                _serializable(result)
                for result in tracker.update(
                    rows,
                    timestamp=frame_number / fps,
                )
            )
    finally:
        capture.release()
    emitted.extend(
        _serializable(result)
        for result in tracker.flush()
    )
    normalized_truth = {
        normalize_plate(value)
        for value in truth_plates
        if plausible_plate(value)
    }
    emitted_norms = {
        normalize_plate(result.get("plate_norm", ""))
        for result in emitted
        if plausible_plate(result.get("plate_norm", ""))
    }
    matched_truth = sorted(normalized_truth & emitted_norms)
    return {
        "video": str(video_path),
        "model": str(model_path),
        "frames": frame_number,
        "processed_frames": processed_frames,
        "detections": detection_count,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "max_rss_kb": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "truth_count": len(normalized_truth),
        "matched_truth_count": len(matched_truth),
        "matched_truth": [
            format_iran_plate(value)
            for value in matched_truth
        ],
        "emitted_count": len(emitted),
        "emitted": emitted,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark Hezar ONNX on a BC Vision video",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--ocr-model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--truth", action="append", default=[])
    args = parser.parse_args(argv)
    model_path = args.ocr_model.resolve()
    metadata_path = (
        args.metadata.resolve()
        if args.metadata
        else model_path.with_suffix(".json")
    )
    result = benchmark(
        args.video.resolve(),
        model_path,
        metadata_path,
        args.frame_step,
        list(args.truth),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(
        {
            "frames": result["frames"],
            "processed_frames": result["processed_frames"],
            "detections": result["detections"],
            "elapsed_seconds": result["elapsed_seconds"],
            "truth_count": result["truth_count"],
            "matched_truth_count": result["matched_truth_count"],
            "matched_truth": result["matched_truth"],
            "emitted_count": result["emitted_count"],
        },
        ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
