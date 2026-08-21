"""Verified model bootstrap for BC Vision ANPR.

Large models are stored under the persistent data directory rather than in the
application tree so upgrades preserve them. Downloads are atomic and checked
against fixed SHA-256 values before a model can be loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.request

from .hezar_export import HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE

YOLO11N_DETECTOR_URL = (
    "https://huggingface.co/morsetechlab/"
    "yolov11-license-plate-detection/resolve/"
    "0f8dc030388b3660418ac7d8c37d3a40148064c1/"
    "license-plate-finetune-v1n.onnx?download=true"
)
YOLO11N_DETECTOR_SHA256 = (
    "693133A1DB97A3BA1E90068986F80AFB"
    "72C3FCDDB681E57181A89A9A3DC351D6"
)
YOLO11N_DETECTOR_SIZE = 10_481_682
YOLOV8N_DETECTOR_URL = (
    "https://huggingface.co/Dibachain/Platrix/resolve/"
    "4f5a43eae683e0b6ad977d4001e3967dcb96e295/"
    "plate_yolo.onnx?download=true"
)
YOLOV8N_DETECTOR_SHA256 = (
    "A54E475C402E6036BB5C70F1A6FF7517"
    "9E76098A5C8039BB5D148C0B6421F5C6"
)
YOLOV8N_DETECTOR_SIZE = 12_608_775
YOLOX_MANIFEST_SCHEMA_VERSION = 1
YOLOX_OUTPUT_FORMATS = {
    "raw-grid",
    "decoded-cxcywh",
    "nms-xyxy",
}
MODEL_PREPARATION_STATE_ENV = "BCVISION_MODEL_PREPARATION_STATE"
MODEL_PREPARATION_ERROR_ENV = "BCVISION_MODEL_PREPARATION_ERROR"
MODEL_PREPARATION_ATTEMPT_ENV = "BCVISION_MODEL_PREPARATION_ATTEMPT"

# Backward-compatible aliases describe the default YOLO11n detector. Older
# integrations and signed baseline manifests import these names directly.
DETECTOR_URL = YOLO11N_DETECTOR_URL
DETECTOR_SHA256 = YOLO11N_DETECTOR_SHA256
DETECTOR_SIZE = YOLO11N_DETECTOR_SIZE
DETECTOR_FALLBACK_URL = (
    "https://huggingface.co/Dibachain/Platrix/resolve/main/"
    "plate_yolo_fallback.onnx?download=true"
)
DETECTOR_FALLBACK_SHA256 = (
    "A6974FCB0A79755C270D50F1EBEFD4D9"
    "6D765C879A29051A19AAC00DFDA8B5AF"
)
DETECTOR_FALLBACK_SIZE = 12_265_080
CRNN_URL = (
    "https://huggingface.co/Dibachain/Platrix/resolve/main/"
    "ocr_crnn.onnx?download=true"
)
CRNN_SHA256 = (
    "45F8C45F29EB1EE91F6274CB8D9C328D"
    "A1A2050EA7D8596BAE61F4A6B9F9FB1E"
)
CRNN_SIZE = 10_452_525
CNN_URL = (
    "https://huggingface.co/Dibachain/Platrix/resolve/main/"
    "ocr_cnn.onnx?download=true"
)
CNN_SHA256 = (
    "7D573C51CC855A8E080F1F88597477F4"
    "FB5A2B9CAFA1BB125BD6038E441F5BCA"
)
CNN_SIZE = 2_226_402


def _data_dir() -> Path:
    try:
        from app.config import DATA_DIR
        return Path(DATA_DIR)
    except Exception:
        return Path.home() / ".bcvision"


def detector_path() -> Path:
    configured = os.environ.get(
        "BCVISION_PLATE_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "plate" / "plate_yolo11n.onnx"


def yolov8n_detector_path() -> Path:
    configured = os.environ.get(
        "BCVISION_PLATE_YOLOV8N_MODEL",
        os.environ.get("BCVISION_PLATE_YOLO8N_MODEL", ""),
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "plate" / "plate_yolov8n.onnx"


def yolox_manifest_path() -> Path:
    configured = os.environ.get(
        "BCVISION_YOLOX_MANIFEST",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "plate" / "yolox-custom.json"


def yolox_detector_path() -> Path:
    configured = os.environ.get(
        "BCVISION_YOLOX_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "plate" / "yolox-custom.onnx"


def _empty_yolox_spec(error="YOLOX manifest is not installed") -> dict:
    return {
        "variant": "yolox",
        "path": yolox_detector_path(),
        "manifest_path": yolox_manifest_path(),
        "manifest_digest": "",
        "sha256": "",
        "size": 0,
        "input_size": 640,
        "input_height": 640,
        "input_width": 640,
        "output_format": "raw-grid",
        "coordinate_space": "grid",
        "output_index": 0,
        "class_count": 1,
        "plate_class_id": 0,
        "strides": [8, 16, 32],
        "color": "bgr",
        "input_scale": 1.0,
        "letterbox_mode": "top-left",
        "scores_are_logits": False,
        "method": "yolox-custom-onnx",
        "model_revision": "",
        "ready": False,
        "error": str(error),
    }


def yolox_detector_spec(
    manifest_path_override: Path | None = None,
    model_path_override: Path | None = None,
) -> dict:
    """Return a fail-closed, hash-verified custom YOLOX contract.

    Custom weights are installation data and live outside the application
    tree.  The sidecar manifest is deliberately explicit because common
    YOLOX exports disagree about whether grid/stride decoding or NMS is
    embedded in the ONNX graph.
    """

    manifest_path = (
        Path(manifest_path_override)
        if manifest_path_override is not None
        else yolox_manifest_path()
    )
    if not manifest_path.is_file():
        return _empty_yolox_spec()
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        payload = json.loads(manifest_bytes.decode("utf-8"))
        if int(payload.get("schema_version", 0)) != YOLOX_MANIFEST_SCHEMA_VERSION:
            raise ValueError("unsupported YOLOX manifest schema")
        filename = str(payload.get("filename", "")).strip()
        digest = str(payload.get("sha256", "")).strip().upper()
        size = int(payload.get("size", 0))
        if not filename or Path(filename).name != filename:
            raise ValueError("YOLOX filename must be a plain file name")
        if not re.fullmatch(r"[0-9A-F]{64}", digest):
            raise ValueError("YOLOX SHA-256 is invalid")
        if size <= 0:
            raise ValueError("YOLOX model size is invalid")

        raw_input = payload.get("input_size", 640)
        if isinstance(raw_input, (list, tuple)) and len(raw_input) == 2:
            input_height, input_width = (int(raw_input[0]), int(raw_input[1]))
        else:
            input_height = input_width = int(raw_input)
        if not (
            160 <= input_height <= 1280
            and 160 <= input_width <= 1280
            and input_height % 32 == 0
            and input_width % 32 == 0
        ):
            raise ValueError("YOLOX input size must be 160..1280 and divisible by 32")

        output_format = str(payload.get("output_format", "")).strip().lower()
        if output_format not in YOLOX_OUTPUT_FORMATS:
            raise ValueError("YOLOX output_format must be explicit")
        coordinate_space = str(
            payload.get("coordinate_space", "")
        ).strip().lower()
        expected_space = (
            "grid" if output_format == "raw-grid" else "input-pixels"
        )
        if coordinate_space != expected_space:
            raise ValueError(
                f"YOLOX {output_format} coordinate_space must be {expected_space}"
            )
        class_count = int(payload.get("class_count", 1))
        plate_class_id = int(payload.get("plate_class_id", 0))
        if (
            not 1 <= class_count <= 256
            or not 0 <= plate_class_id < class_count
        ):
            raise ValueError("YOLOX class contract is invalid")
        strides = [int(value) for value in payload.get("strides", [8, 16, 32])]
        if (
            not strides
            or any(value <= 0 for value in strides)
            or len(set(strides)) != len(strides)
            or any(
                input_height % value != 0
                or input_width % value != 0
                for value in strides
            )
        ):
            raise ValueError("YOLOX strides are invalid")
        color = str(payload.get("color", "")).strip().lower()
        if color not in {"rgb", "bgr"}:
            raise ValueError("YOLOX color must be rgb or bgr")
        input_scale = float(payload.get("input_scale", 0.0))
        if not 0.0 < input_scale <= 1.0:
            raise ValueError("YOLOX input_scale is invalid")
        letterbox_mode = str(
            payload.get("letterbox_mode", "")
        ).strip().lower()
        if letterbox_mode not in {"center", "top-left"}:
            raise ValueError(
                "YOLOX letterbox_mode must be center or top-left"
            )

        configured_model = os.environ.get("BCVISION_YOLOX_MODEL", "").strip()
        model_path = (
            Path(model_path_override)
            if model_path_override is not None
            else (
                Path(configured_model).expanduser()
                if configured_model
                else manifest_path.parent / filename
            )
        )
        output_index = int(payload.get("output_index", 0))
        if output_index < 0:
            raise ValueError("YOLOX output_index is invalid")
        scores_are_logits = payload.get("scores_are_logits", False)
        if not isinstance(scores_are_logits, bool):
            raise ValueError("YOLOX scores_are_logits must be boolean")
        ready = verify_file(model_path, digest, size)
        contract_identity = json.dumps(
            {
                "input_height": input_height,
                "input_width": input_width,
                "output_format": output_format,
                "coordinate_space": coordinate_space,
                "output_index": output_index,
                "class_count": class_count,
                "plate_class_id": plate_class_id,
                "strides": strides,
                "color": color,
                "input_scale": input_scale,
                "letterbox_mode": letterbox_mode,
                "scores_are_logits": scores_are_logits,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        model_revision = hashlib.sha256(
            (digest + ":" + contract_identity).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "variant": "yolox",
            "path": model_path,
            "manifest_path": manifest_path,
            "manifest_digest": manifest_digest,
            "sha256": digest,
            "size": size,
            "input_size": (
                input_height
                if input_height == input_width
                else [input_height, input_width]
            ),
            "input_height": input_height,
            "input_width": input_width,
            "output_format": output_format,
            "coordinate_space": coordinate_space,
            "output_index": output_index,
            "class_count": class_count,
            "plate_class_id": plate_class_id,
            "strides": strides,
            "color": color,
            "input_scale": input_scale,
            "letterbox_mode": letterbox_mode,
            "scores_are_logits": scores_are_logits,
            "method": "yolox-custom-onnx",
            "model_revision": model_revision,
            "ready": ready,
            "error": "" if ready else "YOLOX model does not match its manifest",
        }
    except Exception as exc:
        return _empty_yolox_spec(f"{type(exc).__name__}: {exc}")


def normalize_detector_variant(value, default="yolo11n") -> str:
    aliases = {
        "yolo11": "yolo11n",
        "yolo11n": "yolo11n",
        "yolov11n": "yolo11n",
        "yolo8": "yolov8n",
        "yolo8n": "yolov8n",
        "yolov8": "yolov8n",
        "yolov8n": "yolov8n",
        "yolox": "yolox",
        "yolox-custom": "yolox",
        "custom-yolox": "yolox",
    }
    normalized_default = aliases.get(
        str(default or "").strip().lower(),
        "yolo11n",
    )
    return aliases.get(
        str(value or "").strip().lower(),
        normalized_default,
    )


def detector_variant_spec(variant=None) -> dict:
    selected = normalize_detector_variant(variant)
    if selected == "yolox":
        return yolox_detector_spec()
    if selected == "yolov8n":
        return {
            "variant": selected,
            "path": yolov8n_detector_path(),
            "sha256": YOLOV8N_DETECTOR_SHA256,
            "size": YOLOV8N_DETECTOR_SIZE,
            "input_size": 416,
            "method": "yolov8n-plate-onnx",
        }
    return {
        "variant": "yolo11n",
        "path": detector_path(),
        "sha256": DETECTOR_SHA256,
        "size": DETECTOR_SIZE,
        "input_size": 640,
        "method": "yolo11n-plate-onnx",
    }


def detector_fallback_path() -> Path:
    configured = os.environ.get(
        "BCVISION_PLATE_FALLBACK_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return (
        _data_dir()
        / "models"
        / "plate"
        / "plate_yolo_fallback.onnx"
    )


def crnn_path() -> Path:
    configured = os.environ.get(
        "BCVISION_CRNN_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "crnn" / "ocr_crnn.onnx"


def cnn_path() -> Path:
    configured = os.environ.get(
        "BCVISION_CNN_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "cnn" / "ocr_cnn.onnx"


def hezar_path() -> Path:
    configured = os.environ.get(
        "BCVISION_HEZAR_MODEL",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "hezar" / "crnn_fa_v2.onnx"


def _active_crnn_manifest_path() -> Path:
    return (
        _data_dir()
        / "models"
        / "crnn"
        / "active-model.json"
    )


def active_crnn_model() -> tuple[Path, str, int]:
    """Return the verified promoted model or the fixed vendor model."""

    manifest = _active_crnn_manifest_path()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        filename = str(payload.get("filename", "")).strip()
        digest = str(payload.get("sha256", "")).strip().upper()
        size = int(payload.get("size", 0))
        custom_root = (
            _data_dir()
            / "models"
            / "crnn"
            / "custom"
        ).resolve()
        candidate = (custom_root / filename).resolve()
        candidate.relative_to(custom_root)
        if (
            filename
            and digest
            and size > 0
            and verify_file(candidate, digest, size)
        ):
            return candidate, digest, size
    except Exception:
        pass
    return crnn_path(), CRNN_SHA256, CRNN_SIZE


def active_crnn_training_checkpoint() -> tuple[Path, str, int] | None:
    """Return the hash-verified state dict paired with a promoted CRNN."""

    manifest = _active_crnn_manifest_path()
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        filename = str(
            payload.get("training_checkpoint_filename", "")
        ).strip()
        digest = str(
            payload.get("training_checkpoint_sha256", "")
        ).strip().upper()
        size = int(payload.get("training_checkpoint_size", 0))
        custom_root = (
            _data_dir()
            / "models"
            / "crnn"
            / "custom"
        ).resolve()
        checkpoint = (custom_root / filename).resolve()
        checkpoint.relative_to(custom_root)
        if (
            filename
            and digest
            and size > 0
            and verify_file(checkpoint, digest, size)
        ):
            return checkpoint, digest, size
    except Exception:
        pass
    return None


def promote_crnn_candidate(
    candidate: Path,
    expected_sha256: str,
    source_run_id: int,
    training_checkpoint: Path | None = None,
    training_checkpoint_sha256: str = "",
) -> dict:
    candidate = Path(candidate)
    digest = str(expected_sha256).upper()
    size = candidate.stat().st_size if candidate.is_file() else 0
    if size <= 0 or not verify_file(candidate, digest, size):
        raise ValueError("Candidate CRNN SHA-256 verification failed")
    custom_root = (
        _data_dir()
        / "models"
        / "crnn"
        / "custom"
    )
    custom_root.mkdir(parents=True, exist_ok=True)
    filename = f"run-{int(source_run_id)}-{digest[:16]}.onnx"
    target = custom_root / filename
    if not verify_file(target, digest, size):
        temporary = target.with_suffix(".tmp")
        shutil.copy2(candidate, temporary)
        if not verify_file(temporary, digest, size):
            temporary.unlink(missing_ok=True)
            raise ValueError("Promoted CRNN copy verification failed")
        os.replace(temporary, target)
    checkpoint_target = None
    checkpoint_digest = ""
    checkpoint_size = 0
    if training_checkpoint is not None:
        training_checkpoint = Path(training_checkpoint)
        checkpoint_digest = str(
            training_checkpoint_sha256
        ).upper()
        checkpoint_size = (
            training_checkpoint.stat().st_size
            if training_checkpoint.is_file()
            else 0
        )
        if (
            checkpoint_size <= 0
            or not verify_file(
                training_checkpoint,
                checkpoint_digest,
                checkpoint_size,
            )
        ):
            raise ValueError(
                "Candidate CRNN training checkpoint verification failed"
            )
        checkpoint_target = custom_root / (
            f"run-{int(source_run_id)}-{checkpoint_digest[:16]}.pt"
        )
        if not verify_file(
            checkpoint_target,
            checkpoint_digest,
            checkpoint_size,
        ):
            temporary_checkpoint = checkpoint_target.with_suffix(".tmp")
            shutil.copy2(
                training_checkpoint,
                temporary_checkpoint,
            )
            if not verify_file(
                temporary_checkpoint,
                checkpoint_digest,
                checkpoint_size,
            ):
                temporary_checkpoint.unlink(missing_ok=True)
                raise ValueError(
                    "Promoted CRNN checkpoint copy verification failed"
                )
            os.replace(temporary_checkpoint, checkpoint_target)
    manifest = _active_crnn_manifest_path()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest.with_suffix(".tmp")
    temporary_manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "filename": filename,
                "sha256": digest,
                "size": size,
                "source_run_id": int(source_run_id),
                "training_checkpoint_filename": (
                    checkpoint_target.name
                    if checkpoint_target is not None
                    else ""
                ),
                "training_checkpoint_sha256": checkpoint_digest,
                "training_checkpoint_size": checkpoint_size,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest)
    try:
        from .onnx_crnn import clear_crnn_sessions

        clear_crnn_sessions()
    except Exception:
        pass
    return {
        "path": str(target),
        "sha256": digest,
        "size": size,
        "source_run_id": int(source_run_id),
        "training_checkpoint_path": (
            str(checkpoint_target)
            if checkpoint_target is not None
            else ""
        ),
        "training_checkpoint_sha256": checkpoint_digest,
    }


def packaged_seed_dir() -> Path | None:
    configured = os.environ.get(
        "BCVISION_PACKAGED_MODEL_SEED",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    bundle_root = getattr(sys, "_MEIPASS", "")
    if bundle_root:
        candidate = Path(bundle_root) / "model-seed"
        if candidate.is_dir():
            return candidate
    candidate = Path(__file__).resolve().parents[2] / "model-seed"
    return candidate if candidate.is_dir() else None


def _copy_verified(
    source: Path,
    target: Path,
    digest: str,
    size: int | None = None,
) -> bool:
    if not verify_file(source, digest, size):
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".seed")
    shutil.copy2(source, temporary)
    if not verify_file(temporary, digest, size):
        temporary.unlink(missing_ok=True)
        return False
    os.replace(temporary, target)
    return True


def sha256_file(
    path: Path,
    chunk_size=1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest().upper()


def verify_file(
    path: Path,
    expected_sha256: str,
    expected_size: int | None = None,
) -> bool:
    path = Path(path)
    if not path.is_file():
        return False
    if (
        expected_size is not None
        and path.stat().st_size != int(expected_size)
    ):
        return False
    return sha256_file(path) == expected_sha256.upper()


def install_yolox_model(
    source: Path,
    *,
    input_size=640,
    output_format="raw-grid",
    class_count=1,
    plate_class_id=0,
    strides=(8, 16, 32),
    color="bgr",
    input_scale=1.0,
    letterbox_mode="top-left",
    scores_are_logits=False,
    output_index=0,
) -> dict:
    """Atomically install custom YOLOX weights and their decode contract."""

    source = Path(source).expanduser()
    if not source.is_file() or source.suffix.lower() != ".onnx":
        raise FileNotFoundError(f"YOLOX ONNX model not found: {source}")
    size = source.stat().st_size
    if size <= 0:
        raise ValueError("YOLOX ONNX model is empty")
    digest = sha256_file(source)
    manifest = yolox_manifest_path()
    configured_target = os.environ.get("BCVISION_YOLOX_MODEL", "").strip()
    target = (
        Path(configured_target).expanduser()
        if configured_target
        else manifest.parent / f"yolox-{digest[:12].lower()}.onnx"
    )
    if configured_target and source.resolve() != target.resolve():
        raise ValueError(
            "BCVISION_YOLOX_MODEL is read-only during registration; "
            "place the model at that exact path first or unset the override"
        )
    target_was_present = target.exists()
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if source.resolve() != target.resolve():
        temp_model = None
        try:
            with tempfile.NamedTemporaryFile(
                "wb",
                delete=False,
                dir=target.parent,
                prefix=target.name + ".",
                suffix=".part",
            ) as output, source.open("rb") as input_file:
                temp_model = Path(output.name)
                shutil.copyfileobj(input_file, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
            if not verify_file(temp_model, digest, size):
                raise ValueError("Copied YOLOX model failed verification")
            os.replace(temp_model, target)
            temp_model = None
        finally:
            if temp_model is not None:
                temp_model.unlink(missing_ok=True)
    elif not verify_file(target, digest, size):
        raise ValueError("YOLOX model failed verification")

    payload = {
        "schema_version": YOLOX_MANIFEST_SCHEMA_VERSION,
        "filename": target.name,
        "sha256": digest,
        "size": size,
        "input_size": input_size,
        "output_format": str(output_format).strip().lower(),
        "coordinate_space": (
            "grid"
            if str(output_format).strip().lower() == "raw-grid"
            else "input-pixels"
        ),
        "output_index": int(output_index),
        "class_count": int(class_count),
        "plate_class_id": int(plate_class_id),
        "strides": [int(value) for value in strides],
        "color": str(color).strip().lower(),
        "input_scale": float(input_scale),
        "letterbox_mode": str(letterbox_mode).strip().lower(),
        "scores_are_logits": bool(scores_are_logits),
    }
    temp_manifest = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=manifest.parent,
            prefix=manifest.name + ".",
            suffix=".part",
        ) as output:
            temp_manifest = Path(output.name)
            json.dump(payload, output, ensure_ascii=False, indent=2)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        candidate = yolox_detector_spec(
            manifest_path_override=temp_manifest,
            model_path_override=target,
        )
        if not candidate.get("ready"):
            raise ValueError(candidate.get("error") or "Invalid YOLOX manifest")
        from .onnx_yolox import validate_yolox_model

        validate_yolox_model(candidate)
        os.replace(temp_manifest, manifest)
        temp_manifest = None
    except Exception:
        try:
            from .onnx_yolox import clear_yolox_session

            clear_yolox_session()
        except Exception:
            pass
        if (
            not target_was_present
            and source.resolve() != target.resolve()
        ):
            target.unlink(missing_ok=True)
        raise
    finally:
        if temp_manifest is not None:
            temp_manifest.unlink(missing_ok=True)
    return yolox_detector_spec()


def _download_verified(
    url: str,
    target: Path,
    expected_sha256: str,
    expected_size: int,
    timeout=90,
    attempts=3,
    retry_delay=2,
) -> Path:
    if not url.lower().startswith("https://"):
        raise ValueError("Only HTTPS model downloads are allowed")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(target, expected_sha256, expected_size):
        return target

    network_errors = (
        urllib.error.URLError,
        TimeoutError,
        ConnectionError,
        http.client.IncompleteRead,
        ssl.SSLError,
    )
    max_attempts = max(1, int(attempts))
    for attempt in range(max_attempts):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "BCVision-ANPR/2.2"},
        )
        temp_path = None
        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout,
            ) as response:
                with tempfile.NamedTemporaryFile(
                    "wb",
                    delete=False,
                    dir=target.parent,
                    prefix=target.name + ".",
                    suffix=".part",
                ) as output:
                    temp_path = Path(output.name)
                    total = 0
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > expected_size + 1024:
                            raise ValueError(
                                "Downloaded model exceeds the expected size"
                            )
                        output.write(chunk)
            if not verify_file(
                temp_path,
                expected_sha256,
                expected_size,
            ):
                raise ValueError(
                    "Downloaded model failed SHA-256 or size verification"
                )
            os.replace(temp_path, target)
            temp_path = None
            return target
        except network_errors:
            if attempt + 1 >= max_attempts:
                raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
        if retry_delay:
            time.sleep(float(retry_delay) * (attempt + 1))


def ensure_detector_model(download=True) -> Path:
    target = detector_path()
    if verify_file(target, DETECTOR_SHA256, DETECTOR_SIZE):
        return target
    source_dir = os.environ.get(
        "BCVISION_MODEL_SOURCE_DIR",
        "",
    ).strip()
    if source_dir:
        source = Path(source_dir) / "plate_yolo11n.onnx"
        if _copy_verified(
            source,
            target,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "plate" / "plate_yolo11n.onnx",
        target,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
    ):
        return target
    if not download:
        raise FileNotFoundError(
            f"Verified detector model not found: {target}"
        )
    return _download_verified(
        DETECTOR_URL,
        target,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
    )


def ensure_yolov8n_detector_model(download=True) -> Path:
    target = yolov8n_detector_path()
    if verify_file(
        target,
        YOLOV8N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SIZE,
    ):
        return target

    source_dir = os.environ.get(
        "BCVISION_MODEL_SOURCE_DIR",
        "",
    ).strip()
    if source_dir:
        for filename in (
            "plate_yolov8n.onnx",
            "plate_yolo.onnx",
        ):
            source = Path(source_dir) / filename
            if _copy_verified(
                source,
                target,
                YOLOV8N_DETECTOR_SHA256,
                YOLOV8N_DETECTOR_SIZE,
            ):
                return target

    # RC12-RC18 stored the same verified YOLOv8n graph under its upstream
    # filename. Reuse it atomically so an existing installation does not need
    # network access merely because the model now has an explicit name.
    legacy = _data_dir() / "models" / "plate" / "plate_yolo.onnx"
    if legacy.resolve() != target.resolve() and _copy_verified(
        legacy,
        target,
        YOLOV8N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SIZE,
    ):
        return target

    seed = packaged_seed_dir()
    if seed:
        for filename in (
            "plate_yolov8n.onnx",
            "plate_yolo.onnx",
        ):
            if _copy_verified(
                seed / "plate" / filename,
                target,
                YOLOV8N_DETECTOR_SHA256,
                YOLOV8N_DETECTOR_SIZE,
            ):
                return target

    if not download:
        raise FileNotFoundError(
            f"Verified YOLOv8n detector model not found: {target}"
        )
    return _download_verified(
        YOLOV8N_DETECTOR_URL,
        target,
        YOLOV8N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SIZE,
    )


def ensure_detector_fallback_model(download=True) -> Path:
    target = detector_fallback_path()
    if verify_file(
        target,
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
    ):
        return target
    source_dir = os.environ.get(
        "BCVISION_MODEL_SOURCE_DIR",
        "",
    ).strip()
    if source_dir:
        source = Path(source_dir) / "plate_yolo_fallback.onnx"
        if _copy_verified(
            source,
            target,
            DETECTOR_FALLBACK_SHA256,
            DETECTOR_FALLBACK_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "plate" / "plate_yolo_fallback.onnx",
        target,
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
    ):
        return target
    if not download:
        raise FileNotFoundError(
            f"Verified fallback detector model not found: {target}"
        )
    return _download_verified(
        DETECTOR_FALLBACK_URL,
        target,
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
    )


def ensure_crnn_model(download=True) -> Path:
    target = crnn_path()
    if verify_file(target, CRNN_SHA256, CRNN_SIZE):
        return target
    source_dir = os.environ.get(
        "BCVISION_CRNN_SOURCE_DIR",
        os.environ.get("BCVISION_MODEL_SOURCE_DIR", ""),
    ).strip()
    if source_dir:
        source = Path(source_dir) / "ocr_crnn.onnx"
        if _copy_verified(
            source,
            target,
            CRNN_SHA256,
            CRNN_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "crnn" / "ocr_crnn.onnx",
        target,
        CRNN_SHA256,
        CRNN_SIZE,
    ):
        return target
    if not download:
        raise FileNotFoundError(
            f"Verified CRNN ONNX model not found: {target}"
        )
    result = _download_verified(
        CRNN_URL,
        target,
        CRNN_SHA256,
        CRNN_SIZE,
    )
    try:
        from .onnx_crnn import clear_crnn_sessions

        clear_crnn_sessions()
    except Exception:
        pass
    return result


def ensure_cnn_model(download=True) -> Path:
    target = cnn_path()
    if verify_file(target, CNN_SHA256, CNN_SIZE):
        return target
    source_dir = os.environ.get(
        "BCVISION_CNN_SOURCE_DIR",
        os.environ.get("BCVISION_MODEL_SOURCE_DIR", ""),
    ).strip()
    if source_dir:
        source = Path(source_dir) / "ocr_cnn.onnx"
        if _copy_verified(
            source,
            target,
            CNN_SHA256,
            CNN_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "cnn" / "ocr_cnn.onnx",
        target,
        CNN_SHA256,
        CNN_SIZE,
    ):
        return target
    if not download:
        raise FileNotFoundError(
            f"Verified CNN ONNX model not found: {target}"
        )
    return _download_verified(
        CNN_URL,
        target,
        CNN_SHA256,
        CNN_SIZE,
    )


def ensure_hezar_model(download=True) -> Path:
    target = hezar_path()
    if verify_file(target, HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE):
        return target
    source_dir = os.environ.get(
        "BCVISION_HEZAR_SOURCE_DIR",
        os.environ.get("BCVISION_MODEL_SOURCE_DIR", ""),
    ).strip()
    if source_dir:
        source = Path(source_dir) / "crnn_fa_v2.onnx"
        if _copy_verified(
            source,
            target,
            HEZAR_ONNX_SHA256,
            HEZAR_ONNX_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "hezar" / "crnn_fa_v2.onnx",
        target,
        HEZAR_ONNX_SHA256,
        HEZAR_ONNX_SIZE,
    ):
        return target
    if not download:
        raise FileNotFoundError(
            f"Verified Hezar CRNN ONNX model not found: {target}"
        )
    from .hezar_export import export_pinned_model

    cache_dir = Path(os.environ.get(
        "BCVISION_HEZAR_CACHE_DIR",
        str(target.parent / "source-cache"),
    )).expanduser()
    export_pinned_model(target, cache_dir)
    if not verify_file(
        target,
        HEZAR_ONNX_SHA256,
        HEZAR_ONNX_SIZE,
    ):
        target.unlink(missing_ok=True)
        raise ValueError("Exported Hezar model verification failed")
    try:
        from .onnx_hezar import clear_hezar_sessions

        clear_hezar_sessions()
    except Exception:
        pass
    return target


def prepare_models(download=True) -> dict:
    detector = ensure_detector_model(download=download)
    detector_yolov8n = ensure_yolov8n_detector_model(
        download=download,
    )
    detector_fallback = ensure_detector_fallback_model(
        download=download,
    )
    crnn = ensure_crnn_model(download=download)
    cnn = ensure_cnn_model(download=download)
    hezar = ensure_hezar_model(download=download)
    return {
        "detector": str(detector),
        "detector_yolo11n": str(detector),
        "detector_yolov8n": str(detector_yolov8n),
        "detector_fallback": str(detector_fallback),
        "crnn": str(crnn),
        "cnn": str(cnn),
        "hezar": str(hezar),
    }


def prepare_seed(seed_dir: Path, download=True) -> dict:
    detector = ensure_detector_model(download=download)
    detector_yolov8n = ensure_yolov8n_detector_model(
        download=download,
    )
    detector_fallback = ensure_detector_fallback_model(
        download=download,
    )
    crnn = ensure_crnn_model(download=download)
    cnn = ensure_cnn_model(download=download)
    hezar = ensure_hezar_model(download=download)
    seed = Path(seed_dir)
    detector_target = seed / "plate" / "plate_yolo11n.onnx"
    if not _copy_verified(
        detector,
        detector_target,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
    ):
        raise ValueError("Detector seed verification failed")
    detector_yolov8n_target = (
        seed / "plate" / "plate_yolov8n.onnx"
    )
    if not _copy_verified(
        detector_yolov8n,
        detector_yolov8n_target,
        YOLOV8N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SIZE,
    ):
        raise ValueError("YOLOv8n detector seed verification failed")
    detector_fallback_target = (
        seed / "plate" / "plate_yolo_fallback.onnx"
    )
    if not _copy_verified(
        detector_fallback,
        detector_fallback_target,
        DETECTOR_FALLBACK_SHA256,
        DETECTOR_FALLBACK_SIZE,
    ):
        raise ValueError("Fallback detector seed verification failed")
    crnn_target = seed / "crnn" / "ocr_crnn.onnx"
    if not _copy_verified(
        crnn,
        crnn_target,
        CRNN_SHA256,
        CRNN_SIZE,
    ):
        raise ValueError("CRNN seed verification failed")
    cnn_target = seed / "cnn" / "ocr_cnn.onnx"
    if not _copy_verified(
        cnn,
        cnn_target,
        CNN_SHA256,
        CNN_SIZE,
    ):
        raise ValueError("CNN seed verification failed")
    hezar_target = seed / "hezar" / "crnn_fa_v2.onnx"
    if not _copy_verified(
        hezar,
        hezar_target,
        HEZAR_ONNX_SHA256,
        HEZAR_ONNX_SIZE,
    ):
        raise ValueError("Hezar seed verification failed")
    return {
        "detector": str(detector_target),
        "detector_yolo11n": str(detector_target),
        "detector_yolov8n": str(detector_yolov8n_target),
        "detector_fallback": str(detector_fallback_target),
        "crnn": str(crnn_target),
        "cnn": str(cnn_target),
        "hezar": str(hezar_target),
    }


def model_status(selected_detector=None) -> dict:
    selected_variant = normalize_detector_variant(selected_detector)
    selected_spec = detector_variant_spec(selected_variant)
    detector = detector_path()
    detector_yolov8n = yolov8n_detector_path()
    detector_fallback = detector_fallback_path()
    crnn = crnn_path()
    cnn = cnn_path()
    hezar = hezar_path()
    active_crnn, active_crnn_sha, active_crnn_size = (
        active_crnn_model()
    )
    detector_yolo11n_ready = verify_file(
        detector,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
    )
    detector_yolov8n_ready = verify_file(
        detector_yolov8n,
        YOLOV8N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SIZE,
    )
    detector_yolox_spec = (
        selected_spec
        if selected_variant == "yolox"
        else yolox_detector_spec()
    )
    detector_yolox_ready = bool(detector_yolox_spec.get("ready"))
    selected_ready = {
        "yolo11n": detector_yolo11n_ready,
        "yolov8n": detector_yolov8n_ready,
        "yolox": detector_yolox_ready,
    }[selected_variant]
    preparation_state = os.environ.get(
        MODEL_PREPARATION_STATE_ENV,
        "",
    ).strip().lower()
    preparation_error = os.environ.get(
        MODEL_PREPARATION_ERROR_ENV,
        "",
    ).strip()
    try:
        preparation_attempt = max(
            0,
            int(os.environ.get(MODEL_PREPARATION_ATTEMPT_ENV, "0")),
        )
    except (TypeError, ValueError):
        preparation_attempt = 0
    status = {
        "selected_detector": selected_variant,
        "selected_detector_method": selected_spec["method"],
        "selected_detector_input_size": selected_spec["input_size"],
        "preparation_state": preparation_state,
        "preparation_error": preparation_error,
        "preparation_attempt": preparation_attempt,
        # Backward-compatible fields describe the selected primary. With no
        # argument they retain their historical YOLO11n meaning.
        "detector_path": str(selected_spec["path"]),
        "detector_ready": selected_ready,
        "detector_yolo11n_path": str(detector),
        "detector_yolo11n_ready": detector_yolo11n_ready,
        "detector_yolov8n_path": str(detector_yolov8n),
        "detector_yolov8n_ready": detector_yolov8n_ready,
        "detector_yolox_path": str(detector_yolox_spec["path"]),
        "detector_yolox_manifest_path": str(
            detector_yolox_spec["manifest_path"]
        ),
        "detector_yolox_ready": detector_yolox_ready,
        "detector_yolox_error": detector_yolox_spec.get("error", ""),
        "detector_yolox_revision": detector_yolox_spec.get(
            "model_revision",
            "",
        ),
        "detector_fallback_path": str(detector_fallback),
        "detector_fallback_ready": verify_file(
            detector_fallback,
            DETECTOR_FALLBACK_SHA256,
            DETECTOR_FALLBACK_SIZE,
        ),
        "crnn_path": str(crnn),
        "crnn_ready": verify_file(
            crnn,
            CRNN_SHA256,
            CRNN_SIZE,
        ),
        "platrix_crnn_path": str(crnn),
        "platrix_crnn_ready": verify_file(
            crnn,
            CRNN_SHA256,
            CRNN_SIZE,
        ),
        "production_ocr_policy": "hezar-v2-then-fixed-platrix",
        "active_crnn_path": str(active_crnn),
        "active_crnn_sha256": active_crnn_sha,
        "active_crnn_ready": verify_file(
            active_crnn,
            active_crnn_sha,
            active_crnn_size,
        ),
        "custom_crnn_active": active_crnn.resolve() != crnn.resolve(),
        "cnn_path": str(cnn),
        "cnn_ready": verify_file(
            cnn,
            CNN_SHA256,
            CNN_SIZE,
        ),
        "hezar_path": str(hezar),
        "hezar_ready": verify_file(
            hezar,
            HEZAR_ONNX_SHA256,
            HEZAR_ONNX_SIZE,
        ),
        "easyocr_ready": False,
    }
    try:
        from .next_models import next_models_status

        next_status = next_models_status()
    except Exception as exc:
        next_status = {
            "ready": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    status["next_engine_ready"] = bool(next_status.get("ready"))
    status["next_engine"] = next_status
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Prepare verified BC Vision ANPR models"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only check model status",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Never access the network",
    )
    parser.add_argument(
        "--seed-dir",
        type=Path,
        help="Create a verified offline model seed for packaging",
    )
    parser.add_argument(
        "--install-yolox",
        type=Path,
        help="Atomically install a custom YOLOX ONNX model",
    )
    parser.add_argument(
        "--yolox-input-size",
        type=int,
        default=640,
        help="Square YOLOX input size (default: 640)",
    )
    parser.add_argument(
        "--yolox-output-format",
        choices=sorted(YOLOX_OUTPUT_FORMATS),
        default="raw-grid",
        help="Explicit ONNX output contract",
    )
    parser.add_argument(
        "--yolox-output-index",
        type=int,
        default=0,
        help="Zero-based graph output containing detections (default: 0)",
    )
    parser.add_argument(
        "--yolox-scores-are-logits",
        action="store_true",
        help="Apply sigmoid to objectness/class scores emitted as logits",
    )
    parser.add_argument("--yolox-class-count", type=int, default=1)
    parser.add_argument("--yolox-plate-class-id", type=int, default=0)
    parser.add_argument(
        "--yolox-strides",
        default="8,16,32",
        help="Comma-separated raw-grid strides",
    )
    parser.add_argument(
        "--yolox-color",
        choices=("bgr", "rgb"),
        default="bgr",
    )
    parser.add_argument(
        "--yolox-input-scale",
        type=float,
        default=1.0,
        help="Input multiplier; standard Megvii YOLOX uses 1.0",
    )
    parser.add_argument(
        "--yolox-letterbox",
        choices=("top-left", "center"),
        default="top-left",
    )
    args = parser.parse_args(argv)
    if args.install_yolox:
        strides = [
            int(value.strip())
            for value in str(args.yolox_strides).split(",")
            if value.strip()
        ]
        install_yolox_model(
            args.install_yolox,
            input_size=args.yolox_input_size,
            output_format=args.yolox_output_format,
            class_count=args.yolox_class_count,
            plate_class_id=args.yolox_plate_class_id,
            strides=strides,
            color=args.yolox_color,
            input_scale=args.yolox_input_scale,
            letterbox_mode=args.yolox_letterbox,
            scores_are_logits=args.yolox_scores_are_logits,
            output_index=args.yolox_output_index,
        )
    elif args.seed_dir:
        prepare_seed(
            args.seed_dir,
            download=not args.no_download,
        )
    elif not args.check:
        prepare_models(download=not args.no_download)
    status = model_status(
        selected_detector="yolox" if args.install_yolox else None
    )
    for key, value in status.items():
        print(f"{key.upper()}={value}")
    if args.install_yolox:
        return 0 if status["detector_yolox_ready"] else 1
    return 0 if (
        status["detector_ready"]
        and status["detector_yolo11n_ready"]
        and status["detector_yolov8n_ready"]
        and status["detector_fallback_ready"]
        and status["crnn_ready"]
        and status["cnn_ready"]
        and status["hezar_ready"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
