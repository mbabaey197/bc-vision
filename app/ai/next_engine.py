"""Fail-safe baseline/shadow/next routing for the RC14 ANPR engine."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .activity import suppress_static_overlay_rows
from .next_models import (
    engine_mode,
    rollback_to_baseline,
    verified_next_manifest,
)
from .onnx_cct import cct_status, read_plate_cct
from .onnx_hezar import hezar_status, read_plate_hezar
from .onnx_obb import detect_plates_obb, obb_status
from .onnx_detector import detect_plates_onnx, detector_status


@dataclass
class EngineFrameResult:
    mode: str
    primary: list
    shadow: list
    primary_ms: float
    shadow_ms: float
    degraded: bool = False
    error: str = ""


def _read_candidate_ocr(crop, engine_key=None) -> tuple[dict, dict, str]:
    manifest = verified_next_manifest()
    runtime = str(
        manifest["models"]["ocr"].get(
            "runtime",
            "hezar-ctc-onnx",
        )
    ).strip().lower()
    if runtime == "fast-plate-ocr-cct":
        return (
            read_plate_cct(crop, engine_key=engine_key),
            cct_status(),
            runtime,
        )
    if runtime == "hezar-ctc-onnx":
        return (
            read_plate_hezar(crop, engine_key=engine_key),
            hezar_status(),
            runtime,
        )
    raise RuntimeError(f"Unsupported candidate OCR runtime: {runtime}")


def process_frame_next(
    frame,
    min_detection_confidence=0.25,
    engine_key=None,
    detections=None,
    exclusion_mask=None,
) -> list[dict]:
    from .pipeline import image_quality

    manifest = verified_next_manifest()
    model_revision = str(
        manifest.get("release_id")
        or manifest.get("engine")
        or "next-candidate"
    )
    detector_runtime = str(
        manifest["models"]["detector"].get(
            "runtime",
            "yolo26-obb-onnx",
        )
    ).strip().lower()
    results = []
    if detector_runtime in {
        "baseline-yolov8-onnx",
        "baseline-yolo11n-onnx",
    }:
        detector_variant = (
            "yolov8n"
            if detector_runtime == "baseline-yolov8-onnx"
            else "yolo11n"
        )
        expected_method = (
            "yolov8n-plate-onnx"
            if detector_variant == "yolov8n"
            else "yolo11n-plate-onnx"
        )
        detector_state = detector_status()
        reusable = (
            bool(detections)
            and detector_state.get("model_loaded")
            and detector_state.get("selected_variant")
            == detector_variant
            and all(
                row.get("method") == expected_method
                for row in detections
            )
        )
        if not reusable:
            detections = detect_plates_onnx(
                frame,
                min_confidence=min_detection_confidence,
                engine_key=engine_key,
                detector_variant=detector_variant,
                raise_on_error=True,
            )
        detector_state = detector_status()
    else:
        detections = detect_plates_obb(
            frame,
            min_confidence=min_detection_confidence,
            engine_key=engine_key,
        )
        detector_state = obb_status()
    detections = suppress_static_overlay_rows(
        detections,
        exclusion_mask,
    )
    if (
        detector_state.get("attempted")
        and not detector_state.get("model_loaded")
    ):
        raise RuntimeError(
            "RC14 detector failed: "
            + str(detector_state.get("error", "unknown error"))
        )
    for detection in detections:
        crop = detection.get("crop")
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        quality = image_quality(crop)
        ocr, ocr_state, ocr_engine = _read_candidate_ocr(
            crop,
            engine_key=engine_key,
        )
        if (
            ocr_state.get("attempted")
            and not ocr_state.get("model_loaded")
        ):
            raise RuntimeError(
                "RC14 OCR failed: "
                + str(ocr_state.get("error", "unknown error"))
            )
        valid = bool(ocr["accepted"])
        raw_guess = (
            ocr["hypotheses"][0]
            if ocr.get("hypotheses")
            else {}
        )
        raw_guess_norm = str(
            ocr.get("raw_plate_norm")
            or raw_guess.get("plate_norm")
            or ""
        )
        raw_guess_text = str(
            raw_guess.get("plate")
            or (
                ocr.get("plate")
                if valid
                else ""
            )
        )
        raw_guess_confidence = float(
            ocr.get(
                "confidence",
                raw_guess.get("confidence", 0.0),
            )
        )
        detector_confidence = float(
            detection.get(
                "detector_confidence",
                detection.get("confidence", 0.0),
            )
        )
        confidence = (
            0.30 * detector_confidence
            + 0.60 * float(ocr["confidence"])
            + 0.10 * float(quality["score"])
        )
        if not valid:
            confidence *= 0.45
        position_hypotheses = []
        for hypothesis in ocr["hypotheses"]:
            plate = hypothesis["plate_norm"]
            raw_positions = hypothesis.get("positions", {})
            position_hypotheses.append({
                "positions": {
                    position: {
                        "character": character,
                        "confidence": float(
                            raw_positions.get(
                                position,
                                {},
                            ).get(
                                "confidence",
                                hypothesis["confidence"],
                            )
                        ),
                    }
                    for position, character in enumerate(plate)
                },
                "coverage": len(plate),
                "score": hypothesis["confidence"],
                "engine": ocr_engine,
            })
        results.append({
            **detection,
            "plate": (
                ocr["plate"]
                if valid
                else raw_guess_text or "ناخوانا"
            ),
            "plate_norm": ocr["plate_norm"] if valid else "",
            "valid": valid,
            "confidence": round(min(1.0, confidence), 4),
            "detector_confidence": detector_confidence,
            "ocr_confidence": float(ocr["confidence"]),
            "ocr_engine": ocr_engine,
            "quality_score": float(quality["score"]),
            "quality": quality,
            "plate_hypotheses": ocr["hypotheses"],
            "position_hypotheses": position_hypotheses,
            "whole_plate_ocr_attempted": True,
            "generic_ocr_attempted": False,
            "needs_review": not valid,
            "best_effort": bool(not valid and raw_guess_text),
            "read_status": (
                "confirmed-ai"
                if valid
                else "experimental-guess"
                if raw_guess_text
                else "unreadable"
            ),
            "raw_guess_text": raw_guess_text,
            "raw_guess_norm": raw_guess_norm,
            "raw_guess_confidence": raw_guess_confidence,
            "raw_guess_engine": ocr_engine,
            "raw_guess_reason": str(ocr.get("reason", "")),
            "raw_model_confidence": float(
                ocr.get(
                    "uncalibrated_confidence",
                    raw_guess_confidence,
                )
            ),
            "preprocess_profile": str(
                ocr.get("preprocess_profile", "stretch-v1")
            ),
            "fusion_method": str(
                ocr.get("fusion_method", "identity-v1")
            ),
            "view_agreement": float(
                ocr.get("view_agreement", 1.0)
            ),
            "whole_view_agreement": bool(
                ocr.get("whole_view_agreement", True)
            ),
            "view_diagnostics": list(
                ocr.get("view_diagnostics", [])
            ),
            "association_plate_norm": str(
                ocr.get("association_plate_norm", "")
            ),
            "association_plate_strong": bool(
                ocr.get("association_plate_strong", False)
            ),
            "model_revision": model_revision,
            "detector_runtime": detector_runtime,
            "experimental": bool(not valid),
            "hypotheses_accepted_for_consensus": bool(
                ocr.get("temporal_consensus_eligible", valid)
            ),
            "next_engine": True,
        })
    results.sort(
        key=lambda row: (row["valid"], row["confidence"]),
        reverse=True,
    )
    return results


class EngineRouter:
    """Run candidate inference without allowing shadow failures to escape."""

    def __init__(self):
        self._lock = threading.RLock()
        self._status = {}

    def process(
        self,
        frame,
        baseline,
        min_detection_confidence=0.25,
        engine_key=None,
        mode=None,
        exclusion_mask=None,
    ) -> EngineFrameResult:
        selected = str(mode or engine_mode())
        baseline_started = time.monotonic()
        if selected == "next":
            try:
                primary = process_frame_next(
                    frame,
                    min_detection_confidence,
                    engine_key,
                    exclusion_mask=exclusion_mask,
                )
                elapsed = (time.monotonic() - baseline_started) * 1000
                result = EngineFrameResult(
                    mode="next",
                    primary=primary,
                    shadow=[],
                    primary_ms=elapsed,
                    shadow_ms=0.0,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                rollback_to_baseline("next-runtime-error: " + error)
                primary = baseline()
                elapsed = (time.monotonic() - baseline_started) * 1000
                result = EngineFrameResult(
                    mode="baseline",
                    primary=primary,
                    shadow=[],
                    primary_ms=elapsed,
                    shadow_ms=0.0,
                    degraded=True,
                    error=error,
                )
        else:
            primary = baseline()
            primary_ms = (
                time.monotonic() - baseline_started
            ) * 1000
            shadow = []
            shadow_ms = 0.0
            error = ""
            if selected == "shadow":
                shadow_started = time.monotonic()
                try:
                    shadow = process_frame_next(
                        frame,
                        min_detection_confidence,
                        engine_key,
                        detections=primary,
                        exclusion_mask=exclusion_mask,
                    )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                shadow_ms = (
                    time.monotonic() - shadow_started
                ) * 1000
            result = EngineFrameResult(
                mode=selected if selected == "shadow" else "baseline",
                primary=primary,
                shadow=shadow,
                primary_ms=primary_ms,
                shadow_ms=shadow_ms,
                error=error,
            )
        with self._lock:
            self._status[str(engine_key or "default")] = {
                "mode": result.mode,
                "primary_results": len(result.primary),
                "shadow_results": len(result.shadow),
                "primary_ms": round(result.primary_ms, 3),
                "shadow_ms": round(result.shadow_ms, 3),
                "degraded": result.degraded,
                "error": result.error,
            }
        return result

    def status(self, engine_key=None) -> dict:
        with self._lock:
            return dict(
                self._status.get(
                    str(engine_key or "default"),
                    {},
                )
            )


engine_router = EngineRouter()
