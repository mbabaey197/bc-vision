from __future__ import annotations

import os
import sys
import tempfile
import threading
from pathlib import Path

import cv2

from .plate_rules import format_iran_plate, normalize_plate, plausible_plate

_MODEL = None
_MODEL_ERROR = ""
_MODEL_LOCK = threading.Lock()


def _bundled_model_dir() -> Path:
    configured = os.environ.get("BCVISION_HEZAR_MODEL_DIR", "").strip()
    if configured:
        return Path(configured).expanduser()
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    candidates = [
        bundle_root / "hezar-model",
        Path(sys.executable).resolve().parent / "hezar-model",
        Path(__file__).resolve().parents[2] / ".hezar-model",
    ]
    for candidate in candidates:
        if (candidate / "model.pt").is_file() and (candidate / "model_config.yaml").is_file():
            return candidate
    return candidates[0]


def _load_model():
    global _MODEL, _MODEL_ERROR
    if _MODEL is not None:
        return _MODEL
    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL
        try:
            from hezar.models import Model
            model_dir = _bundled_model_dir()
            _MODEL = Model.load(str(model_dir), load_locally=True)
            _MODEL_ERROR = ""
            return _MODEL
        except Exception as exc:
            _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
            return None


def status() -> dict:
    return {
        "model_loaded": _MODEL is not None,
        "model_path": str(_bundled_model_dir()),
        "error": _MODEL_ERROR,
    }


def _extract_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)) and result:
        first = result[0]
        if isinstance(first, str):
            return first
        text = getattr(first, "text", None)
        if text is not None:
            return str(text)
        if isinstance(first, dict):
            for key in ("text", "prediction", "generated_text", "label"):
                if key in first:
                    return str(first[key])
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return ""


def read_plate_hezar(image, engine_key=None) -> tuple[str, float]:
    del engine_key
    if image is None or getattr(image, "size", 0) == 0:
        return "", 0.0
    model = _load_model()
    if model is None:
        return "", 0.0

    handle = tempfile.NamedTemporaryFile(
        suffix=".png",
        prefix="bcvision-hezar-",
        delete=False,
    )
    path = Path(handle.name)
    handle.close()
    try:
        if not cv2.imwrite(str(path), image):
            return "", 0.0
        raw = _extract_text(model.predict(str(path))).strip()
        normalized = normalize_plate(raw)
        if not plausible_plate(normalized):
            return "", 0.0
        return format_iran_plate(normalized), 0.95
    except Exception as exc:
        global _MODEL_ERROR
        _MODEL_ERROR = f"{type(exc).__name__}: {exc}"
        return "", 0.0
    finally:
        path.unlink(missing_ok=True)
