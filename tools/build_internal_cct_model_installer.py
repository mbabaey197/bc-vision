"""Build a private, one-click RC15 Shadow CCT model installer.

The generated CMD is intentionally not a public release artifact.  It embeds
the verified baseline detector and the research-only CCT OCR, signs a manifest
with an ephemeral Ed25519 key, installs files atomically under BC Vision's
persistent data root, and selects Shadow mode.  The private key is never
written to disk.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
)


BASELINE_DETECTOR_SHA256 = (
    "A54E475C402E6036BB5C70F1A6FF7517"
    "9E76098A5C8039BB5D148C0B6421F5C6"
)
BASELINE_DETECTOR_SIZE = 12_608_775


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_manifest_bytes(payload: dict) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("signature", None)
    return json.dumps(
        unsigned,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_block(name: str, content: bytes) -> str:
    encoded = base64.b64encode(content).decode("ascii")
    lines = "\r\n".join(
        encoded[index:index + 76]
        for index in range(0, len(encoded), 76)
    )
    return (
        f"::BCVISION_PAYLOAD_BEGIN:{name}\r\n"
        f"{lines}\r\n"
        f"::BCVISION_PAYLOAD_END:{name}\r\n"
    )


def _powershell_installer(
    detector_sha256: str,
    detector_size: int,
    ocr_sha256: str,
    ocr_size: int,
) -> str:
    return textwrap.dedent(
        rf"""
        param([Parameter(Mandatory=$true)][string]$SelfPath)
        $ErrorActionPreference = "Stop"

        function Write-EmbeddedPayload {{
            param(
                [Parameter(Mandatory=$true)][string]$Name,
                [Parameter(Mandatory=$true)][string]$Destination
            )
            $content = [IO.File]::ReadAllText($SelfPath)
            $escaped = [Regex]::Escape($Name)
            $pattern = "(?ms)^::BCVISION_PAYLOAD_BEGIN:$escaped\r?\n(.*?)^::BCVISION_PAYLOAD_END:$escaped\r?$"
            $match = [Regex]::Match($content, $pattern)
            if (-not $match.Success) {{
                throw "Embedded payload is missing: $Name"
            }}
            $encoded = $match.Groups[1].Value -replace "\s", ""
            [IO.File]::WriteAllBytes(
                $Destination,
                [Convert]::FromBase64String($encoded)
            )
        }}

        function Assert-File {{
            param(
                [Parameter(Mandatory=$true)][string]$Path,
                [Parameter(Mandatory=$true)][string]$Sha256,
                [Parameter(Mandatory=$true)][long]$Size
            )
            $item = Get-Item -LiteralPath $Path
            if ($item.Length -ne $Size) {{
                throw "Size verification failed: $Path"
            }}
            $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
            if ($actual -ne $Sha256) {{
                throw "SHA-256 verification failed: $Path"
            }}
        }}

        function Publish-Atomic {{
            param(
                [Parameter(Mandatory=$true)][string]$Source,
                [Parameter(Mandatory=$true)][string]$Destination
            )
            $parent = Split-Path -Parent $Destination
            New-Item -ItemType Directory -Path $parent -Force | Out-Null
            $temporary = "$Destination.rc15-new"
            Copy-Item -LiteralPath $Source -Destination $temporary -Force
            Move-Item -LiteralPath $temporary -Destination $Destination -Force
        }}

        $bootstrapRoot = Join-Path $env:ProgramData "BCVision\data"
        $dataRoot = $bootstrapRoot
        $storageConfig = Join-Path $bootstrapRoot "storage_config.json"
        if (Test-Path -LiteralPath $storageConfig) {{
            $storage = Get-Content -LiteralPath $storageConfig -Raw |
                ConvertFrom-Json
            if ($storage.storage_root) {{
                $dataRoot = [Environment]::ExpandEnvironmentVariables(
                    [string]$storage.storage_root
                )
            }}
        }}

        $stage = Join-Path $env:TEMP ("BCVision-RC15-Model-" + $PID)
        New-Item -ItemType Directory -Path $stage -Force | Out-Null
        try {{
            $detectorStage = Join-Path $stage "plate_yolo.onnx"
            $ocrStage = Join-Path $stage "rc15-cct-xs-ir-lpr-stage4.onnx"
            $manifestStage = Join-Path $stage "active-models.json"
            $keyStage = Join-Path $stage "model_public_key.pem"
            $noticeStage = Join-Path $stage "RC15-INTERNAL-MODEL-NOTICE.txt"
            $stateStage = Join-Path $stage "runtime-state.json"

            Write-EmbeddedPayload "detector" $detectorStage
            Write-EmbeddedPayload "ocr" $ocrStage
            Write-EmbeddedPayload "manifest" $manifestStage
            Write-EmbeddedPayload "public-key" $keyStage
            Write-EmbeddedPayload "notice" $noticeStage
            Write-EmbeddedPayload "runtime-state" $stateStage

            Assert-File $detectorStage "{detector_sha256}" {detector_size}
            Assert-File $ocrStage "{ocr_sha256}" {ocr_size}

            $plateRoot = Join-Path $dataRoot "models\plate"
            $nextRoot = Join-Path $dataRoot "models\next"
            $detectorTarget = Join-Path $plateRoot "plate_yolo.onnx"
            $detectorReady = $false
            if (Test-Path -LiteralPath $detectorTarget) {{
                try {{
                    Assert-File $detectorTarget "{detector_sha256}" {detector_size}
                    $detectorReady = $true
                }} catch {{
                    $detectorReady = $false
                }}
            }}
            if (-not $detectorReady) {{
                Publish-Atomic $detectorStage $detectorTarget
                Assert-File $detectorTarget "{detector_sha256}" {detector_size}
            }}

            New-Item -ItemType Directory -Path $nextRoot -Force | Out-Null
            Publish-Atomic $ocrStage (
                Join-Path $nextRoot "rc15-cct-xs-ir-lpr-stage4.onnx"
            )
            Publish-Atomic $keyStage (
                Join-Path $nextRoot "model_public_key.pem"
            )
            Publish-Atomic $noticeStage (
                Join-Path $nextRoot "RC15-INTERNAL-MODEL-NOTICE.txt"
            )
            Publish-Atomic $manifestStage (
                Join-Path $nextRoot "active-models.json"
            )
            Publish-Atomic $stateStage (
                Join-Path $nextRoot "runtime-state.json"
            )

            Assert-File (
                Join-Path $nextRoot "rc15-cct-xs-ir-lpr-stage4.onnx"
            ) "{ocr_sha256}" {ocr_size}
            Write-Host ""
            Write-Host "BC Vision RC15 experimental model installed." -ForegroundColor Green
            Write-Host "Mode: Shadow with operator confirmation"
            Write-Host "Data preserved at: $dataRoot"
            exit 0
        }} catch {{
            Write-Host ""
            Write-Host "Model installation failed: $($_.Exception.Message)" -ForegroundColor Red
            exit 1
        }} finally {{
            Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
        }}
        """
    ).strip() + "\r\n"


def build(
    model_path: Path,
    detector_path: Path,
    metadata_path: Path,
    output_path: Path,
) -> dict:
    model_path = model_path.resolve()
    detector_path = detector_path.resolve()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_sha256 = sha256_file(model_path)
    model_size = model_path.stat().st_size
    if (
        model_sha256 != str(metadata.get("sha256", "")).upper()
        or model_size != int(metadata.get("size", 0))
    ):
        raise ValueError("CCT model does not match candidate metadata")
    if (
        str(metadata.get("usage_scope")) != "research-shadow-only"
        or metadata.get("distribution_allowed") is not False
        or metadata.get("activation_allowed") is not False
    ):
        raise ValueError("Only the research Shadow candidate can be packed")

    detector_sha256 = sha256_file(detector_path)
    detector_size = detector_path.stat().st_size
    if (
        detector_sha256 != BASELINE_DETECTOR_SHA256
        or detector_size != BASELINE_DETECTOR_SIZE
    ):
        raise ValueError("Baseline detector verification failed")

    ocr_contract_keys = (
        "alphabet",
        "max_plate_slots",
        "input_width",
        "input_height",
        "input_layout",
        "input_dtype",
        "image_color_mode",
        "keep_aspect_ratio",
        "interpolation",
        "padding_color",
        "min_confidence",
        "min_position_confidence",
        "min_position_margin",
        "min_hypothesis_margin",
        "beam_width",
        "top_k",
        "preprocess_profile",
        "fusion_method",
        "min_view_agreement",
    )
    manifest = {
        "schema": 1,
        "engine": "bcvision-rc15",
        "release_id": "rc15-internal-cct-stage4-20260729",
        "usage_scope": "research-shadow-only",
        "distribution_allowed": False,
        "activation_allowed": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": {
            "detector": {
                "filename": "plate_yolo.onnx",
                "sha256": detector_sha256,
                "size": detector_size,
                "runtime": "baseline-yolov8-onnx",
                "reuse_verified_baseline": True,
            },
            "ocr": {
                "filename": "rc15-cct-xs-ir-lpr-stage4.onnx",
                "sha256": model_sha256,
                "size": model_size,
                "runtime": "fast-plate-ocr-cct",
                **{
                    key: metadata[key]
                    for key in ocr_contract_keys
                    if key in metadata
                },
            },
        },
        "provenance": {
            "dataset": "IR-LPR",
            "dataset_repository": "https://github.com/mut-deep/IR-LPR",
            "dataset_license": "GPL-3.0",
            "intended_use": "private-internal-evaluation",
            "operator_confirmation_required": True,
        },
    }
    private_key = Ed25519PrivateKey.generate()
    manifest["signature"] = base64.b64encode(
        private_key.sign(canonical_manifest_bytes(manifest))
    ).decode("ascii")
    public_key = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    manifest_bytes = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    runtime_state = json.dumps(
        {
            "schema": 1,
            "mode": "shadow",
            "previous_mode": "baseline",
            "reason": "RC15 internal operator-assisted CCT model pack",
            "rollback_lock": False,
            "changed_at": datetime.now(timezone.utc).isoformat(),
        },
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")
    notice = (
        "BC Vision RC15 internal evaluation model\n"
        "This package enables Shadow inference with operator confirmation.\n"
        "It is not approved for public or commercial distribution.\n"
        "Training source: IR-LPR (GPL-3.0), https://github.com/mut-deep/IR-LPR\n"
        "Automatic results can be wrong; operator corrections are authoritative.\n"
    ).encode("utf-8")
    ps1 = _powershell_installer(
        detector_sha256,
        detector_size,
        model_sha256,
        model_size,
    ).encode("utf-8")
    ps1_hash = hashlib.sha256(ps1).hexdigest().upper()
    bootstrap = textwrap.dedent(
        rf"""
        @echo off
        setlocal EnableExtensions
        chcp 65001 >nul
        set "BCVISION_SELF=%~f0"
        fltmc >nul 2>&1
        if errorlevel 1 (
          powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process -FilePath $env:ComSpec -ArgumentList '/c',([char]34 + $env:BCVISION_SELF + [char]34) -Verb RunAs"
          exit /b
        )
        set "BCVISION_PS1=%TEMP%\BCVision-RC15-Model-%RANDOM%.ps1"
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText($env:BCVISION_SELF);$m=[regex]::Match($s,'(?ms)^::BCVISION_PAYLOAD_BEGIN:installer-script\r?\n(.*?)^::BCVISION_PAYLOAD_END:installer-script\r?$');if(!$m.Success){{exit 2}};$b=[Convert]::FromBase64String(($m.Groups[1].Value-replace '\s',''));if(([BitConverter]::ToString([Security.Cryptography.SHA256]::Create().ComputeHash($b))-replace '-','')-ne '{ps1_hash}'){{exit 3}};[IO.File]::WriteAllBytes($env:BCVISION_PS1,$b)"
        if errorlevel 1 (
          echo Embedded installer verification failed.
          pause
          exit /b 1
        )
        powershell -NoProfile -ExecutionPolicy Bypass -File "%BCVISION_PS1%" -SelfPath "%BCVISION_SELF%"
        set "BCVISION_RESULT=%ERRORLEVEL%"
        del /q "%BCVISION_PS1%" >nul 2>&1
        echo.
        pause
        exit /b %BCVISION_RESULT%
        """
    ).lstrip().replace("\n", "\r\n")
    payloads = (
        _payload_block("installer-script", ps1)
        + _payload_block("detector", detector_path.read_bytes())
        + _payload_block("ocr", model_path.read_bytes())
        + _payload_block("manifest", manifest_bytes)
        + _payload_block("public-key", public_key)
        + _payload_block("notice", notice)
        + _payload_block("runtime-state", runtime_state)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes((bootstrap + payloads).encode("ascii"))
    return {
        "output": str(output_path),
        "size": output_path.stat().st_size,
        "sha256": sha256_file(output_path),
        "model_sha256": model_sha256,
        "detector_sha256": detector_sha256,
        "release_id": manifest["release_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--detector", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(
        args.model,
        args.detector,
        args.metadata,
        args.output,
    ), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
