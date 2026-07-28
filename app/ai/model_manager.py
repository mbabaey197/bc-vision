"""Verified model bootstrap for BC Vision ANPR.

Large models are stored under the persistent data directory rather than in the
application tree so upgrades preserve them. Downloads are atomic and checked
against fixed SHA-256 values before a model can be loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

DETECTOR_URL = (
    "https://huggingface.co/Dibachain/Platrix/resolve/main/"
    "plate_yolo.onnx?download=true"
)
DETECTOR_SHA256 = (
    "A54E475C402E6036BB5C70F1A6FF7517"
    "9E76098A5C8039BB5D148C0B6421F5C6"
)
DETECTOR_SIZE = 12_608_775
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
    return _data_dir() / "models" / "plate" / "plate_yolo.onnx"


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


def promote_crnn_candidate(
    candidate: Path,
    expected_sha256: str,
    source_run_id: int,
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


def _download_verified(
    url: str,
    target: Path,
    expected_sha256: str,
    expected_size: int,
    timeout=90,
) -> Path:
    if not url.lower().startswith("https://"):
        raise ValueError("Only HTTPS model downloads are allowed")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify_file(target, expected_sha256, expected_size):
        return target

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
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def ensure_detector_model(download=True) -> Path:
    target = detector_path()
    if verify_file(target, DETECTOR_SHA256, DETECTOR_SIZE):
        return target
    source_dir = os.environ.get(
        "BCVISION_MODEL_SOURCE_DIR",
        "",
    ).strip()
    if source_dir:
        source = Path(source_dir) / "plate_yolo.onnx"
        if _copy_verified(
            source,
            target,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
        ):
            return target
    seed = packaged_seed_dir()
    if seed and _copy_verified(
        seed / "plate" / "plate_yolo.onnx",
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


def prepare_models(download=True) -> dict:
    detector = ensure_detector_model(download=download)
    detector_fallback = ensure_detector_fallback_model(
        download=download,
    )
    crnn = ensure_crnn_model(download=download)
    cnn = ensure_cnn_model(download=download)
    return {
        "detector": str(detector),
        "detector_fallback": str(detector_fallback),
        "crnn": str(crnn),
        "cnn": str(cnn),
    }


def prepare_seed(seed_dir: Path, download=True) -> dict:
    detector = ensure_detector_model(download=download)
    detector_fallback = ensure_detector_fallback_model(
        download=download,
    )
    crnn = ensure_crnn_model(download=download)
    cnn = ensure_cnn_model(download=download)
    seed = Path(seed_dir)
    detector_target = seed / "plate" / "plate_yolo.onnx"
    if not _copy_verified(
        detector,
        detector_target,
        DETECTOR_SHA256,
        DETECTOR_SIZE,
    ):
        raise ValueError("Detector seed verification failed")
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
    return {
        "detector": str(detector_target),
        "detector_fallback": str(detector_fallback_target),
        "crnn": str(crnn_target),
        "cnn": str(cnn_target),
    }


def model_status() -> dict:
    detector = detector_path()
    detector_fallback = detector_fallback_path()
    crnn = crnn_path()
    cnn = cnn_path()
    active_crnn, active_crnn_sha, active_crnn_size = (
        active_crnn_model()
    )
    status = {
        "detector_path": str(detector),
        "detector_ready": verify_file(
            detector,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
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
    args = parser.parse_args(argv)
    if args.seed_dir:
        prepare_seed(
            args.seed_dir,
            download=not args.no_download,
        )
    elif not args.check:
        prepare_models(download=not args.no_download)
    status = model_status()
    for key, value in status.items():
        print(f"{key.upper()}={value}")
    return 0 if (
        status["detector_ready"]
        and status["detector_fallback_ready"]
        and status["crnn_ready"]
        and status["cnn_ready"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
