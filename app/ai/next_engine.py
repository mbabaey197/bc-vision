"""Fail-safe baseline/shadow/next routing for the RC13 ANPR engine."""
from __future__ import annotations

from dataclasses import dataclass
import threading
import time

from .next_models import engine_mode, rollback_to_baseline
from .onnx_hezar import hezar_status, read_plate_hezar
from .onnx_obb import detect_plates_obb, obb_status


@dataclass
class EngineFrameResult:
    mode: str
    primary: list
    shadow: list
    primary_ms: float
    shadow_ms: float
    degraded: bool = False
    error: str = ""


def process_frame_next(
    frame,
    min_detection_confidence=0.25,
    engine_key=None,
) -> list[dict]:
    from .pipeline import image_quality

    results = []
    detections = detect_plates_obb(
        frame,
        min_confidence=min_detection_confidence,
        engine_key=engine_key,
    )
    detector_state = obb_status()
    if (
        detector_state.get("attempted")
        and not detector_state.get("model_loaded")
    ):
        raise RuntimeError(
            "RC13 detector failed: "
            + str(detector_state.get("error", "unknown error"))
        )
    for detection in detections:
        crop = detection["crop"]
        quality = image_quality(crop)
        ocr = read_plate_hezar(crop, engine_key=engine_key)
        ocr_state = hezar_status()
        if (
            ocr_state.get("attempted")
            and not ocr_state.get("model_loaded")
        ):
            raise RuntimeError(
                "RC13 OCR failed: "
                + str(ocr_state.get("error", "unknown error"))
            )
        valid = bool(ocr["accepted"])
        confidence = (
            0.30 * float(detection["confidence"])
            + 0.60 * float(ocr["confidence"])
            + 0.10 * float(quality["score"])
        )
        if not valid:
            confidence *= 0.45
        position_hypotheses = []
        for hypothesis in ocr["hypotheses"]:
            plate = hypothesis["plate_norm"]
            position_hypotheses.append({
                "positions": {
                    position: {
                        "character": character,
                        "confidence": hypothesis["confidence"],
                    }
                    for position, character in enumerate(plate)
                },
                "coverage": len(plate),
                "score": hypothesis["confidence"],
                "engine": "hezar-ctc-onnx",
            })
        results.append({
            **detection,
            "plate": ocr["plate"],
            "plate_norm": ocr["plate_norm"],
            "valid": valid,
            "confidence": round(min(1.0, confidence), 4),
            "detector_confidence": float(detection["confidence"]),
            "ocr_confidence": float(ocr["confidence"]),
            "ocr_engine": "hezar-ctc-onnx",
            "quality_score": float(quality["score"]),
            "quality": quality,
            "plate_hypotheses": ocr["hypotheses"],
            "position_hypotheses": position_hypotheses,
            "whole_plate_ocr_attempted": True,
            "generic_ocr_attempted": False,
            "needs_review": not valid,
            "read_status": "confirmed" if valid else "unreadable",
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
    ) -> EngineFrameResult:
        selected = str(mode or engine_mode())
        baseline_started = time.monotonic()
        if selected == "next":
            try:
                primary = process_frame_next(
                    frame,
                    min_detection_confidence,
                    engine_key,
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
