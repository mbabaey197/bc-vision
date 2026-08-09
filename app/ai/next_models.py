"""Verified RC13/RC14 model bundle activation and fail-safe engine selection.

The next ANPR engine is intentionally dormant until a signed manifest and both
ONNX files are present.  A failed verification or runtime rollback always
returns the application to the RC12 baseline.
"""
from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import threading
import time

from .model_manager import verify_file
from .plate_rules import ALLOWED_PLATE_LETTERS


MANIFEST_SCHEMA = 1
RUNTIME_SCHEMA = 1
ENGINE_MODES = {"baseline", "shadow", "next"}
REQUIRED_MODELS = ("detector", "ocr")
SUPPORTED_ENGINE_IDS = {
    "bcvision-rc13",
    "bcvision-rc14",
    "bcvision-rc15",
}
DETECTOR_RUNTIMES = {
    "baseline-yolov8-onnx",
    "baseline-yolo11n-onnx",
    "yolo26-obb-onnx",
    "ppyoloe-r-onnx",
}
BASELINE_DETECTOR_RUNTIMES = {
    "baseline-yolov8-onnx",
    "baseline-yolo11n-onnx",
}
OCR_RUNTIMES = {
    "hezar-ctc-onnx",
    "fast-plate-ocr-cct",
}
MANIFEST_CACHE_SECONDS = 30.0
_cache_lock = threading.RLock()
_verified_cache: tuple[tuple, float, dict] | None = None


def _data_root() -> Path:
    from app.config import DATA_DIR

    return Path(DATA_DIR)


def next_models_root() -> Path:
    return _data_root() / "models" / "next"


def next_manifest_path() -> Path:
    configured = os.environ.get(
        "BCVISION_NEXT_MANIFEST",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return next_models_root() / "active-models.json"


def model_public_key_path() -> Path:
    configured = os.environ.get(
        "BCVISION_ANPR_MODEL_PUBLIC_KEY",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return next_models_root() / "model_public_key.pem"


def runtime_state_path() -> Path:
    return next_models_root() / "runtime-state.json"


def canonical_manifest_bytes(payload: dict) -> bytes:
    unsigned = deepcopy(payload)
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _verify_signature(payload: dict) -> None:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey,
    )

    signature_text = str(payload.get("signature", "")).strip()
    if not signature_text:
        raise ValueError("Next-model manifest is not signed")
    public_key = serialization.load_pem_public_key(
        model_public_key_path().read_bytes()
    )
    if not isinstance(public_key, Ed25519PublicKey):
        raise ValueError("Next-model public key is not Ed25519")
    try:
        signature = base64.b64decode(
            signature_text,
            validate=True,
        )
    except Exception as exc:
        raise ValueError("Invalid next-model signature encoding") from exc
    public_key.verify(signature, canonical_manifest_bytes(payload))


def _safe_model_path(root: Path, filename: str) -> Path:
    root = root.resolve()
    candidate = (root / filename).resolve()
    candidate.relative_to(root)
    return candidate


def _validate_ocr_runtime(spec: dict, engine_id: str) -> str:
    runtime = str(
        spec.get("runtime", "hezar-ctc-onnx")
    ).strip().lower()
    if runtime not in OCR_RUNTIMES:
        raise ValueError(f"Unsupported next-model OCR runtime: {runtime}")
    if runtime == "fast-plate-ocr-cct":
        if engine_id not in {"bcvision-rc14", "bcvision-rc15"}:
            raise ValueError(
                "FastPlateOCR CCT requires the bcvision-rc14/rc15 engine"
            )
        alphabet = str(spec.get("alphabet", ""))
        slots = int(spec.get("max_plate_slots", 0))
        width = int(spec.get("input_width", 0))
        height = int(spec.get("input_height", 0))
        layout = str(spec.get("input_layout", "")).strip().lower()
        dtype = str(spec.get("input_dtype", "")).strip().lower()
        color_mode = str(
            spec.get("image_color_mode", "")
        ).strip().lower()
        keep_aspect_ratio = spec.get("keep_aspect_ratio")
        preprocess_profile = str(
            spec.get("preprocess_profile", "stretch-v1")
        ).strip().lower()
        fusion_method = str(
            spec.get(
                "fusion_method",
                (
                    "geometric-mean-v1"
                    if preprocess_profile
                    == "stretch-letterbox-geomean-v1"
                    else "identity-v1"
                ),
            )
        ).strip().lower()
        min_view_agreement = spec.get(
            "min_view_agreement",
            (
                0.75
                if preprocess_profile
                == "stretch-letterbox-geomean-v1"
                else 0.0
            ),
        )
        interpolation = str(
            spec.get("interpolation", "")
        ).strip().lower()
        padding_color = spec.get("padding_color")
        thresholds = {
            name: spec.get(name)
            for name in (
                "min_confidence",
                "min_position_confidence",
                "min_position_margin",
                "min_hypothesis_margin",
            )
        }
        numeric_thresholds = all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in thresholds.values()
        )
        beam_width = spec.get("beam_width")
        top_k = spec.get("top_k")
        if (
            slots != 8
            or alphabet
            != "0123456789" + ALLOWED_PLATE_LETTERS + "_"
            or len(set(alphabet)) != len(alphabet)
            or width != 128
            or height != 64
            or layout != "nhwc"
            or dtype != "uint8"
            or color_mode != "rgb"
            or keep_aspect_ratio is not False
            or preprocess_profile not in {
                "stretch-v1",
                "stretch-letterbox-geomean-v1",
            }
            or (
                preprocess_profile == "stretch-v1"
                and fusion_method != "identity-v1"
            )
            or (
                preprocess_profile
                == "stretch-letterbox-geomean-v1"
                and fusion_method != "geometric-mean-v1"
            )
            or not isinstance(min_view_agreement, (int, float))
            or isinstance(min_view_agreement, bool)
            or not math.isfinite(float(min_view_agreement))
            or not 0.0 <= float(min_view_agreement) <= 1.0
            or (
                preprocess_profile
                == "stretch-letterbox-geomean-v1"
                and float(min_view_agreement) < 0.75
            )
            or interpolation not in {
                "nearest",
                "linear",
                "cubic",
                "area",
                "lanczos4",
            }
            or not isinstance(padding_color, list)
            or len(padding_color) != 3
            or any(
                not isinstance(value, int) or not 0 <= value <= 255
                for value in padding_color
            )
            or not numeric_thresholds
            or not 0.5 <= float(thresholds["min_confidence"]) <= 1.0
            or not 0.3
            <= float(thresholds["min_position_confidence"])
            <= 1.0
            or not 0.01
            <= float(thresholds["min_position_margin"])
            <= 1.0
            or not 0.005
            <= float(thresholds["min_hypothesis_margin"])
            <= 1.0
            or not isinstance(beam_width, int)
            or isinstance(beam_width, bool)
            or not 2 <= beam_width <= 64
            or not isinstance(top_k, int)
            or isinstance(top_k, bool)
            or not 2 <= top_k <= min(10, beam_width)
        ):
            raise ValueError("Invalid signed FastPlateOCR CCT contract")
    return runtime


def _validate_detector_runtime(spec: dict, engine_id: str) -> str:
    runtime = str(
        spec.get("runtime", "yolo26-obb-onnx")
    ).strip().lower()
    if runtime not in DETECTOR_RUNTIMES:
        raise ValueError(
            f"Unsupported next-model detector runtime: {runtime}"
        )
    if runtime in BASELINE_DETECTOR_RUNTIMES:
        if (
            engine_id != "bcvision-rc15"
            or spec.get("reuse_verified_baseline") is not True
        ):
            raise ValueError(
                "Baseline detector reuse requires the bcvision-rc15 engine"
            )
        return runtime
    if runtime != "ppyoloe-r-onnx":
        return runtime
    if engine_id != "bcvision-rc15":
        raise ValueError(
            "PP-YOLOE-R requires the bcvision-rc15 engine"
        )
    width = spec.get("input_width")
    height = spec.get("input_height")
    mean = spec.get("mean")
    std = spec.get("std")
    score_threshold = spec.get("score_threshold")
    nms_threshold = spec.get("nms_threshold")
    max_results = spec.get("max_results")

    def numeric_triplet(values):
        return (
            isinstance(values, list)
            and len(values) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in values
            )
        )

    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or not 320 <= width <= 1280
        or not 320 <= height <= 1280
        or width % 32
        or height % 32
        or spec.get("keep_ratio") is not True
        or int(spec.get("pad_to_stride", 0)) != 32
        or not numeric_triplet(mean)
        or not numeric_triplet(std)
        or any(float(value) <= 0 for value in std)
        or not isinstance(score_threshold, (int, float))
        or isinstance(score_threshold, bool)
        or not 0.01 <= float(score_threshold) <= 0.95
        or not isinstance(nms_threshold, (int, float))
        or isinstance(nms_threshold, bool)
        or not 0.01 <= float(nms_threshold) <= 0.90
        or not isinstance(max_results, int)
        or isinstance(max_results, bool)
        or not 1 <= max_results <= 32
    ):
        raise ValueError("Invalid signed PP-YOLOE-R detector contract")
    return runtime


def _file_fingerprint(paths) -> tuple:
    values = []
    for path in paths:
        candidate = Path(path)
        stat = candidate.stat()
        values.append(
            (
                str(candidate.resolve()),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            )
        )
    return tuple(values)


def verified_next_manifest() -> dict:
    """Return a verified manifest with resolved paths or raise."""

    global _verified_cache
    path = next_manifest_path()
    with _cache_lock:
        if _verified_cache is not None:
            fingerprint, verified_at, cached = _verified_cache
            try:
                current = _file_fingerprint(
                    [item[0] for item in fingerprint]
                )
            except OSError:
                current = ()
            if (
                str(path.resolve()) == fingerprint[0][0]
                and
                current == fingerprint
                and time.monotonic() - verified_at
                < MANIFEST_CACHE_SECONDS
            ):
                return deepcopy(cached)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema", 0)) != MANIFEST_SCHEMA:
        raise ValueError("Unsupported next-model manifest schema")
    engine_id = str(payload.get("engine", ""))
    if engine_id not in SUPPORTED_ENGINE_IDS:
        raise ValueError("Unexpected next-model engine identifier")
    usage_scope = str(
        payload.get("usage_scope", "production-candidate")
    ).strip().lower()
    if usage_scope not in {
        "production-candidate",
        "research-shadow-only",
    }:
        raise ValueError("Unexpected next-model usage scope")
    if usage_scope == "research-shadow-only" and (
        payload.get("distribution_allowed") is not False
        or payload.get("activation_allowed") is not False
    ):
        raise ValueError(
            "Research-only model bundle must be non-distributable and "
            "Shadow-only"
        )
    _verify_signature(payload)

    models = payload.get("models")
    if not isinstance(models, dict):
        raise ValueError("Next-model manifest has no model map")
    root = path.parent
    resolved = {}
    for name in REQUIRED_MODELS:
        spec = models.get(name)
        if not isinstance(spec, dict):
            raise ValueError(f"Missing next-model entry: {name}")
        runtime = (
            _validate_ocr_runtime(spec, engine_id)
            if name == "ocr"
            else _validate_detector_runtime(spec, engine_id)
        )
        filename = str(spec.get("filename", "")).strip()
        digest = str(spec.get("sha256", "")).strip().upper()
        size = int(spec.get("size", 0))
        if (
            not filename
            or len(digest) != 64
            or size <= 0
        ):
            raise ValueError(f"Invalid next-model entry: {name}")
        if name == "detector" and runtime in BASELINE_DETECTOR_RUNTIMES:
            from .model_manager import detector_variant_spec

            baseline_variant = (
                "yolov8n"
                if runtime == "baseline-yolov8-onnx"
                else "yolo11n"
            )
            baseline = detector_variant_spec(baseline_variant)
            expected_filename = (
                "plate_yolov8n.onnx"
                if baseline_variant == "yolov8n"
                else "plate_yolo11n.onnx"
            )

            if (
                filename != expected_filename
                or digest != baseline["sha256"]
                or size != baseline["size"]
            ):
                raise ValueError(
                    "Baseline detector reuse must bind the verified "
                    "BC Vision detector"
                )
            model_path = Path(baseline["path"])
        else:
            model_path = _safe_model_path(root, filename)
        if not verify_file(model_path, digest, size):
            raise ValueError(
                f"Next-model SHA-256 verification failed: {name}"
            )
        resolved[name] = {
            **spec,
            "path": str(model_path),
            "sha256": digest,
            "size": size,
            "runtime": runtime,
        }
    result = deepcopy(payload)
    result["usage_scope"] = usage_scope
    # Activation is an explicit signed opt-in. Missing, malformed and merely
    # truthy values remain Shadow-only.
    result["activation_allowed"] = (
        payload.get("activation_allowed") is True
    )
    result["manifest_path"] = str(path)
    result["models"] = resolved
    fingerprint = _file_fingerprint([
        path,
        model_public_key_path(),
        *(
            resolved[name]["path"]
            for name in REQUIRED_MODELS
        ),
    ])
    with _cache_lock:
        _verified_cache = (
            fingerprint,
            time.monotonic(),
            deepcopy(result),
        )
    return result


def next_models_status() -> dict:
    try:
        manifest = verified_next_manifest()
        return {
            "ready": True,
            "release_id": str(manifest.get("release_id", "")),
            "manifest_path": manifest["manifest_path"],
            "detector_path": manifest["models"]["detector"]["path"],
            "detector_runtime": manifest["models"]["detector"][
                "runtime"
            ],
            "ocr_path": manifest["models"]["ocr"]["path"],
            "ocr_runtime": manifest["models"]["ocr"]["runtime"],
            "usage_scope": manifest["usage_scope"],
            "activation_allowed": manifest["activation_allowed"],
            "error": "",
        }
    except Exception as exc:
        return {
            "ready": False,
            "release_id": "",
            "manifest_path": str(next_manifest_path()),
            "detector_path": "",
            "detector_runtime": "",
            "ocr_path": "",
            "ocr_runtime": "",
            "usage_scope": "",
            "activation_allowed": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _read_runtime_state() -> dict:
    try:
        payload = json.loads(
            runtime_state_path().read_text(encoding="utf-8")
        )
        if int(payload.get("schema", 0)) == RUNTIME_SCHEMA:
            return payload
    except Exception:
        pass
    return {
        "schema": RUNTIME_SCHEMA,
        "mode": "baseline",
        "previous_mode": "baseline",
    }


def requested_engine_mode() -> str:
    state = _read_runtime_state()
    if bool(state.get("rollback_lock")):
        return "baseline"
    configured = os.environ.get(
        "BCVISION_ANPR_MODE",
        "",
    ).strip().lower()
    if configured:
        return configured if configured in ENGINE_MODES else "baseline"
    mode = str(state.get("mode", "baseline")).lower()
    return mode if mode in ENGINE_MODES else "baseline"


def engine_mode() -> str:
    requested = requested_engine_mode()
    if requested == "baseline":
        return requested
    status = next_models_status()
    if not status["ready"]:
        return "baseline"
    if (
        requested == "next"
        and (
            status.get("usage_scope") == "research-shadow-only"
            or status.get("activation_allowed") is not True
        )
    ):
        return "baseline"
    return requested


def _write_engine_mode(
    mode: str,
    reason: str,
    rollback_lock: bool,
) -> dict:
    selected = str(mode).strip().lower()
    if selected not in ENGINE_MODES:
        raise ValueError("Unknown ANPR engine mode")
    status = next_models_status()
    if selected != "baseline" and not status["ready"]:
        raise ValueError("Verified next-generation models are not ready")
    if (
        selected == "next"
        and (
            status.get("usage_scope") == "research-shadow-only"
            or status.get("activation_allowed") is not True
        )
    ):
        raise ValueError(
            "This model candidate can run only in Shadow mode"
        )
    previous = requested_engine_mode()
    payload = {
        "schema": RUNTIME_SCHEMA,
        "mode": selected,
        "previous_mode": previous,
        "reason": str(reason),
        "rollback_lock": bool(rollback_lock),
        "changed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = runtime_state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return payload


def set_engine_mode(mode: str, reason="operator") -> dict:
    return _write_engine_mode(
        mode,
        reason=str(reason),
        rollback_lock=False,
    )


def rollback_to_baseline(reason: str) -> dict:
    return _write_engine_mode(
        "baseline",
        reason=str(reason),
        rollback_lock=True,
    )
