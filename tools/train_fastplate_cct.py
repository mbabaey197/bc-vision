"""Train, export and verify a BC Vision FastPlateOCR CCT candidate.

The command is deliberately offline with respect to training data.  It accepts
only a dataset carrying an explicit commercially compatible provenance
manifest, exports fixed-batch uint8 NHWC ONNX, and measures exact held-out
accuracy before producing candidate metadata.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.onnx_cct import (
    accept_cct_hypotheses,
    decode_cct_hypotheses,
    prepare_cct_input,
)
from app.ai.plate_rules import normalize_plate


ALLOWED_DATA_LICENSES = {
    "synthetic-bcvision-company-owned",
    "bcvision-company-owned",
    "operator-confirmed-company-owned",
    "cc0-1.0",
    "cc-by-4.0",
}
ALLOWED_FONT_LICENSES = {
    "apache-2.0",
    "bcvision-company-owned",
    "bsd-3-clause",
    "cc0-1.0",
    "dejavu-font-license",
    "ofl-1.1",
}
EXCLUDED_PRETRAINED_LAYERS = {
    "plate",
    "region",
    "region_pre_pool_transformer_block_1",
    "region_seq_pool",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _dataset_contract(dataset: Path) -> dict:
    manifest_path = dataset / "dataset-license.json"
    if not manifest_path.is_file():
        raise ValueError("Dataset provenance manifest is required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    license_name = str(
        manifest.get("source_license", "")
    ).strip().lower()
    if license_name not in ALLOWED_DATA_LICENSES:
        raise ValueError(
            f"Dataset license is not approved for BC Vision: "
            f"{license_name or 'missing'}"
        )
    if bool(manifest.get("golden_benchmark_data", False)):
        raise ValueError(
            "Golden benchmark data must never be used for CCT training"
        )
    if license_name == "synthetic-bcvision-company-owned":
        font_license = str(
            manifest.get("font_license", "")
        ).strip().lower()
        if (
            bool(manifest.get("third_party_plate_dataset", True))
            or font_license not in ALLOWED_FONT_LICENSES
        ):
            raise ValueError(
                "Synthetic data provenance or font license is not approved"
            )
    train = dataset / "train" / "annotations.csv"
    validation = dataset / "val" / "annotations.csv"
    if not train.is_file() or not validation.is_file():
        raise ValueError("Train and validation annotations are required")
    return {
        "manifest": manifest,
        "train": train,
        "validation": validation,
    }


def _is_excluded_pretrained_layer(name: str) -> bool:
    return (
        name in EXCLUDED_PRETRAINED_LAYERS
        or name.startswith("region_")
    )


def _copy_pretrained_backbone(source_model, target_model) -> list[str]:
    """Copy only shape-compatible feature layers, never OCR/region heads."""
    source_layers = [
        layer
        for layer in source_model.layers
        if layer.get_weights()
        and not _is_excluded_pretrained_layer(layer.name)
    ]
    target_layers = [
        layer
        for layer in target_model.layers
        if layer.get_weights()
        and not _is_excluded_pretrained_layer(layer.name)
    ]
    if len(source_layers) != len(target_layers):
        raise ValueError(
            "Pretrained backbone architecture has a different number "
            "of weighted feature layers"
        )
    planned_weights = []
    transferred = []
    copied_elements = 0
    target_elements = 0
    for source_layer, target_layer in zip(
        source_layers,
        target_layers,
        strict=True,
    ):
        target_weights = target_layer.get_weights()
        source_weights = source_layer.get_weights()
        if (
            type(source_layer).__name__ != type(target_layer).__name__
            or len(source_weights) != len(target_weights)
        ):
            raise ValueError(
                "Pretrained backbone architecture does not match "
                f"at layer: {target_layer.name}"
            )
        replacement = []
        for source, target in zip(
            source_weights,
            target_weights,
            strict=True,
        ):
            target_elements += int(target.size)
            if source.shape == target.shape:
                replacement.append(source)
                copied_elements += int(target.size)
            else:
                replacement.append(target)
        planned_weights.append((target_layer, replacement))
        transferred.append(target_layer.name)
    transfer_ratio = (
        copied_elements / target_elements
        if target_elements
        else 0.0
    )
    if transfer_ratio < 0.95:
        raise ValueError(
            "Pretrained backbone architecture does not match: "
            f"only {transfer_ratio:.1%} of feature parameters are compatible"
        )
    for target_layer, replacement in planned_weights:
        target_layer.set_weights(replacement)
    if not transferred:
        raise ValueError("Pretrained model supplied no transferable backbone")
    return transferred


def _prepare_pretrained_backbone(
    source_path: Path,
    model_config_path: Path,
    plate_config_path: Path,
    output: Path,
) -> tuple[Path, list[str]]:
    from fast_plate_ocr.train.model.config import (
        load_plate_config_from_yaml,
    )
    from fast_plate_ocr.train.model.model_builders import build_model
    from fast_plate_ocr.train.model.model_schema import (
        load_model_config_from_yaml,
    )
    from fast_plate_ocr.train.utilities.utils import load_keras_model

    plate_config = load_plate_config_from_yaml(plate_config_path)
    model_config = load_model_config_from_yaml(model_config_path)
    source_model = load_keras_model(source_path, plate_config)
    target_model = build_model(
        model_config,
        plate_config,
        enable_region_head=False,
    )
    transferred = _copy_pretrained_backbone(source_model, target_model)
    initialized = output / "pretrained-backbone.keras"
    target_model.save(initialized)
    return initialized, transferred


def _run_official_training(
    model_config: Path,
    plate_config: Path,
    train_annotations: Path,
    validation_annotations: Path,
    output: Path,
    initialized_backbone: Path | None,
    epochs: int,
    batch_size: int,
    seed: int,
) -> Path:
    from fast_plate_ocr.cli.train import train as train_command

    arguments = [
        "--model-config-file",
        str(model_config),
        "--plate-config-file",
        str(plate_config),
        "--annotations",
        str(train_annotations),
        "--val-annotations",
        str(validation_annotations),
        "--validate-dataset",
        "error",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--output-dir",
        str(output / "keras-runs"),
        "--early-stopping-patience",
        str(max(4, min(12, epochs // 3))),
        "--early-stopping-metric",
        "val_plate_acc",
        "--label-smoothing",
        "0.01",
        "--weight-decay",
        "0.0005",
        "--lr",
        "0.0005",
        "--seed",
        str(seed),
        "--workers",
        "1",
        "--no-use-multiprocessing",
    ]
    if initialized_backbone is not None:
        arguments.extend(["--weights-path", str(initialized_backbone)])
    train_command.main(args=arguments, standalone_mode=False)
    candidates = sorted(
        (output / "keras-runs").rglob("best.keras"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if not candidates:
        raise RuntimeError("FastPlateOCR produced no best.keras checkpoint")
    return candidates[-1]


def _export_onnx(
    checkpoint: Path,
    plate_config: Path,
    output: Path,
    variant: str,
) -> Path:
    from fast_plate_ocr.cli.export import export as export_command

    export_dir = output / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)
    export_command.main(
        args=[
            "--model",
            str(checkpoint),
            "--format",
            "onnx",
            "--plate-config-file",
            str(plate_config),
            "--save-dir",
            str(export_dir),
            "--no-dynamic-batch",
            "--onnx-input-dtype",
            "uint8",
            "--onnx-data-format",
            "channels_last",
            "--no-simplify",
        ],
        standalone_mode=False,
    )
    exported = export_dir / checkpoint.with_suffix(".onnx").name
    if not exported.is_file():
        raise RuntimeError("FastPlateOCR ONNX export is missing")
    candidate = output / f"bcvision-cct-{variant}.onnx"
    shutil.copy2(exported, candidate)
    return candidate


def _validation_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image = (path.parent / row["image_path"]).resolve()
            expected = normalize_plate(row["plate_text"])
            if image.is_file() and len(expected) == 8:
                rows.append({"image": image, "expected": expected})
    if not rows:
        raise ValueError("Validation dataset has no usable rows")
    return rows


def _benchmark(
    model: Path,
    validation: Path,
    alphabet: str,
) -> dict:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model),
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    rows = _validation_rows(validation)
    spec = {
        "input_width": 128,
        "input_height": 64,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "image_color_mode": "rgb",
        "keep_aspect_ratio": False,
        "interpolation": "linear",
        "padding_color": [114, 114, 114],
    }
    tensors = []
    for row in rows:
        image = cv2.imread(str(row["image"]), cv2.IMREAD_COLOR)
        tensor = prepare_cct_input(image, spec)
        if tensor is None:
            raise ValueError(f"Unreadable validation image: {row['image']}")
        tensors.append(tensor)
    for tensor in tensors[: min(10, len(tensors))]:
        session.run(None, {input_meta.name: tensor})

    raw_exact = 0
    accepted_exact = 0
    accepted = 0
    started = time.perf_counter()
    for row, tensor in zip(rows, tensors, strict=True):
        output = session.run(None, {input_meta.name: tensor})[0]
        hypotheses = decode_cct_hypotheses(
            output,
            alphabet=alphabet,
            top_k=5,
        )
        result = accept_cct_hypotheses(hypotheses)
        raw = (
            hypotheses[0]["plate_norm"]
            if hypotheses
            else ""
        )
        raw_exact += raw == row["expected"]
        accepted += bool(result["accepted"])
        accepted_exact += (
            bool(result["accepted"])
            and result["plate_norm"] == row["expected"]
        )
    elapsed = time.perf_counter() - started
    output_shape = [
        dimension if isinstance(dimension, int) else str(dimension)
        for dimension in session.get_outputs()[0].shape
    ]
    return {
        "validation_samples": len(rows),
        "raw_exact_matches": raw_exact,
        "raw_exact_accuracy": round(raw_exact / len(rows), 6),
        "accepted_samples": accepted,
        "accepted_exact_matches": accepted_exact,
        "accepted_exact_accuracy": round(
            accepted_exact / len(rows),
            6,
        ),
        "accepted_precision": round(
            accepted_exact / accepted,
            6,
        ) if accepted else 0.0,
        "rejection_rate": round(
            (len(rows) - accepted) / len(rows),
            6,
        ),
        "elapsed_seconds": round(elapsed, 6),
        "mean_latency_ms": round(elapsed * 1000 / len(rows), 6),
        "input_name": input_meta.name,
        "input_shape": [
            dimension if isinstance(dimension, int) else str(dimension)
            for dimension in input_meta.shape
        ],
        "input_type": input_meta.type,
        "output_shape": output_shape,
        "providers": session.get_providers(),
    }


def train_and_export(
    dataset: Path,
    output: Path,
    variant: str,
    pretrained_backbone: Path | None,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict:
    dataset = dataset.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new directory: {output}"
        )
    output.mkdir(parents=True)
    contract = _dataset_contract(dataset)
    root = Path(__file__).resolve().parents[1]
    plate_config = root / "training" / "cct" / "iran_plate_config.yaml"
    model_config = (
        root
        / "training"
        / "cct"
        / f"cct_{variant}_v2_model_config.yaml"
    )
    if not model_config.is_file() or not plate_config.is_file():
        raise FileNotFoundError("BC Vision CCT configuration is missing")
    if (
        pretrained_backbone is not None
        and not pretrained_backbone.is_file()
    ):
        raise FileNotFoundError(pretrained_backbone)
    initialized_backbone = None
    transferred_layers = []
    if pretrained_backbone is not None:
        initialized_backbone, transferred_layers = (
            _prepare_pretrained_backbone(
                source_path=pretrained_backbone,
                model_config_path=model_config,
                plate_config_path=plate_config,
                output=output,
            )
        )

    checkpoint = _run_official_training(
        model_config=model_config,
        plate_config=plate_config,
        train_annotations=contract["train"],
        validation_annotations=contract["validation"],
        output=output,
        initialized_backbone=initialized_backbone,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model = _export_onnx(
        checkpoint=checkpoint,
        plate_config=plate_config,
        output=output,
        variant=variant,
    )
    alphabet = "0123456789ابپتثجدزژسشصطعفقکگلمنوهیDS_"
    metrics = _benchmark(
        model=model,
        validation=contract["validation"],
        alphabet=alphabet,
    )
    metadata = {
        "schema": 1,
        "runtime": "fast-plate-ocr-cct",
        "variant": f"cct-{variant}-v2",
        "model_path": str(model),
        "sha256": _sha256(model),
        "size": model.stat().st_size,
        "alphabet": alphabet,
        "max_plate_slots": 8,
        "input_width": 128,
        "input_height": 64,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "image_color_mode": "rgb",
        "keep_aspect_ratio": False,
        "interpolation": "linear",
        "padding_color": [114, 114, 114],
        "min_confidence": 0.58,
        "min_position_confidence": 0.42,
        "min_position_margin": 0.06,
        "min_hypothesis_margin": 0.025,
        "beam_width": 16,
        "top_k": 5,
        "training": {
            "keras_backend": os.environ.get("KERAS_BACKEND", ""),
            "dataset_license": contract["manifest"],
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "seed": int(seed),
            "pretrained_backbone_path": (
                str(pretrained_backbone.resolve())
                if pretrained_backbone
                else ""
            ),
            "pretrained_backbone_sha256": (
                _sha256(pretrained_backbone)
                if pretrained_backbone
                else ""
            ),
            "pretrained_transferred_layers": transferred_layers,
            "pretrained_excluded_layers": sorted(
                EXCLUDED_PRETRAINED_LAYERS
            ),
            "checkpoint": str(checkpoint),
        },
        "validation": metrics,
    }
    (output / "candidate-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Train and verify BC Vision FastPlateOCR CCT",
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant",
        choices=["xs", "s"],
        required=True,
    )
    parser.add_argument(
        "--pretrained-backbone",
        type=Path,
        help=(
            "Optional FastPlateOCR model used only for compatible feature "
            "layers; OCR and region heads are always excluded"
        ),
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260728)
    args = parser.parse_args(argv)
    result = train_and_export(
        dataset=args.dataset,
        output=args.output,
        variant=args.variant,
        pretrained_backbone=(
            args.pretrained_backbone.resolve()
            if args.pretrained_backbone
            else None
        ),
        epochs=max(4, min(200, int(args.epochs))),
        batch_size=max(4, min(256, int(args.batch_size))),
        seed=int(args.seed),
    )
    print(json.dumps({
        "variant": result["variant"],
        "model_path": result["model_path"],
        "sha256": result["sha256"],
        "size": result["size"],
        "validation": result["validation"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
