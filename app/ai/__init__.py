"""BC Vision AI package bootstrap for the RC27 experimental ANPR path."""
from __future__ import annotations

import os


def _install_hezar_primary() -> None:
    if os.environ.get("BCVISION_HEZAR_OCR", "1") == "0":
        return
    try:
        from . import ocr as legacy_ocr
        from .hezar_ocr import read_plate_hezar
        from .plate_rules import format_iran_plate, plausible_plate
    except Exception:
        return

    current = legacy_ocr.read_plate_candidate
    if getattr(current, "_bcvision_hezar_primary", False):
        return

    def read_plate_candidate(image, engine_key=None, allow_legacy=True):
        text, confidence = read_plate_hezar(image, engine_key=engine_key)
        if plausible_plate(text):
            return (
                format_iran_plate(text),
                float(confidence),
                "hezar-crnn-v2",
            )
        return current(
            image,
            engine_key=engine_key,
            allow_legacy=allow_legacy,
        )

    read_plate_candidate._bcvision_hezar_primary = True
    legacy_ocr.read_plate_candidate = read_plate_candidate


def _install_detection_recovery() -> None:
    """Recover real plates missed by the ONNX detector in live mode.

    RC18 treated every neural miss as authoritative for live cameras and
    returned before the existing geometric fallback could run.  This wrapper
    keeps the fast ONNX result as first choice and only invokes a tightly
    bounded OpenCV fallback after a neural miss.
    """
    if os.environ.get("BCVISION_DETECTION_RECOVERY", "1") == "0":
        return
    try:
        from . import detector as detector_module
    except Exception:
        return

    current = detector_module.detect_plates
    if getattr(current, "_bcvision_rc27_recovery", False):
        return

    def detect_plates(
        frame,
        min_confidence: float = 0.25,
        max_results: int = 8,
        engine_key=None,
        exclusion_mask=None,
    ):
        rows = current(
            frame,
            min_confidence=min_confidence,
            max_results=max_results,
            engine_key=engine_key,
            exclusion_mask=exclusion_mask,
        )
        if rows or engine_key is None:
            return rows

        try:
            light_status = detector_module.onnx_detector_status()
            if not light_status.get("model_loaded"):
                return rows
            fallback = detector_module._opencv_candidates(
                frame,
                max_results=min(max_results, 3),
                exclusion_mask=exclusion_mask,
            )
            floor = min(
                0.45,
                max(0.12, float(min_confidence) * 0.72),
            )
            return [
                row for row in fallback
                if float(row.get("confidence", 0.0)) >= floor
            ][:max_results]
        except Exception:
            return rows

    detect_plates._bcvision_rc27_recovery = True
    detector_module.detect_plates = detect_plates


_install_hezar_primary()
_install_detection_recovery()
