"""Verified model bootstrap for BC Vision ANPR.

Large models are stored under the persistent data directory rather than in the
application tree so upgrades preserve them. Downloads are atomic and checked
against fixed SHA-256 values before a model can be loaded.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

DETECTOR_URL = (
    "https://huggingface.co/makhresearch/"
    "persian-license-plate-detector/resolve/main/"
    "best.pt?download=true"
)
DETECTOR_SHA256 = (
    "258104262d3a16a6bc613938cc1dd0198"
    "da8a7ddeab4843197666cb9ce0db756"
)
DETECTOR_SIZE = 119_237_050
EASYOCR_HASHES = {
    "arabic.pth": (
        "2A9AFD42C374DEB98AED0B53C9B77D75"
        "E1D00D4E0501F3B0276C54190C89B1A8"
    ),
    "craft_mlt_25k.pth": (
        "4A5EFBFB48B4081100544E75E1E2B57F"
        "8DE3D84F213004B14B85FD4B3748DB17"
    ),
}


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
    return _data_dir() / "models" / "plate" / "best.pt"


def easyocr_dir() -> Path:
    configured = os.environ.get(
        "BCVISION_EASYOCR_MODEL_DIR",
        "",
    ).strip()
    if configured:
        return Path(configured).expanduser()
    return _data_dir() / "models" / "easyocr"


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
        source = Path(source_dir) / "best.pt"
        if verify_file(
            source,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
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


def ensure_easyocr_models(download=True) -> Path:
    target = easyocr_dir()
    target.mkdir(parents=True, exist_ok=True)
    if all(
        verify_file(target / name, digest)
        for name, digest in EASYOCR_HASHES.items()
    ):
        return target

    source_dir = os.environ.get(
        "BCVISION_EASYOCR_SOURCE_DIR",
        "",
    ).strip()
    if source_dir:
        source = Path(source_dir)
        for name, digest in EASYOCR_HASHES.items():
            candidate = source / name
            if verify_file(candidate, digest):
                shutil.copy2(candidate, target / name)
        if all(
            verify_file(target / name, digest)
            for name, digest in EASYOCR_HASHES.items()
        ):
            return target

    if not download:
        raise FileNotFoundError(
            f"Verified EasyOCR models not found: {target}"
        )

    import easyocr
    easyocr.Reader(
        ["fa", "en"],
        gpu=False,
        verbose=False,
        model_storage_directory=str(target),
        user_network_directory=str(
            target / "user_network"
        ),
        download_enabled=True,
    )
    invalid = [
        name
        for name, digest in EASYOCR_HASHES.items()
        if not verify_file(target / name, digest)
    ]
    if invalid:
        raise ValueError(
            "EasyOCR model verification failed: "
            + ", ".join(invalid)
        )
    return target


def prepare_models(download=True) -> dict:
    detector = ensure_detector_model(download=download)
    ocr = ensure_easyocr_models(download=download)
    return {
        "detector": str(detector),
        "easyocr": str(ocr),
    }


def model_status() -> dict:
    detector = detector_path()
    ocr = easyocr_dir()
    return {
        "detector_path": str(detector),
        "detector_ready": verify_file(
            detector,
            DETECTOR_SHA256,
            DETECTOR_SIZE,
        ),
        "easyocr_path": str(ocr),
        "easyocr_ready": all(
            verify_file(ocr / name, digest)
            for name, digest in EASYOCR_HASHES.items()
        ),
    }


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
    args = parser.parse_args(argv)
    if not args.check:
        prepare_models(download=not args.no_download)
    status = model_status()
    for key, value in status.items():
        print(f"{key.upper()}={value}")
    return 0 if (
        status["detector_ready"]
        and status["easyocr_ready"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
