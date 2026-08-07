"""BC Vision AI package bootstrap for the RC27 experimental OCR path."""
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
        text, confidence = read_plate_hezar(
            image,
            engine_key=engine_key,
        )
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


_install_hezar_primary()
