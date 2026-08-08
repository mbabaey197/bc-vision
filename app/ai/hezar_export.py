"""Pinned export of Hezar's Persian license-plate CRNN to ONNX.

This module is a build/source-install utility. Frozen BC Vision builds ship the
verified ONNX result in ``model-seed`` and do not import Hezar or PyTorch at
camera runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


HEZAR_MODEL_ID = "hezarai/crnn-fa-license-plate-recognition-v2"
HEZAR_REVISION = "0c48a86abe5bfb140ceeb160c028701028d236b9"
HEZAR_ONNX_SHA256 = (
    "57CB02BC10BDEBD14BE2AC50CD7C25D6"
    "57BDCDEE6EFE77A37A561B832206B0C8"
)
HEZAR_ONNX_SIZE = 37_146_355
HEZAR_SOURCE_FILES = {
    "model.pt": (
        "C20AD7BE2B1FE383DA6F22CBC7BDF8A9A"
        "37119F0B20235D736FAA59B731F6620",
        37_185_141,
    ),
    "model_config.yaml": (
        "AB953CFF1C9A969A0FEDF8BC58774900"
        "30C02F173DE323FCB569F73DAD724896",
        609,
    ),
    "preprocessor/image_processor_config.yaml": (
        "0EDD566B597FA28BFB9A788F04AA6889"
        "EB08BCF9BA80F142319AD2E81440515E",
        153,
    ),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _verify(path: Path, digest: str, size: int) -> bool:
    candidate = Path(path)
    return bool(
        candidate.is_file()
        and candidate.stat().st_size == int(size)
        and _sha256_file(candidate) == str(digest).upper()
    )


def verify_source_model(model_dir: Path) -> None:
    root = Path(model_dir)
    for relative, (digest, size) in HEZAR_SOURCE_FILES.items():
        candidate = root / relative
        if not _verify(candidate, digest, size):
            raise ValueError(
                "Hezar source model verification failed: " + relative
            )


def export_local_model(model_dir: Path, output: Path) -> dict:
    """Export already-downloaded, hash-verified Hezar files to ONNX."""

    import numpy as np
    import onnxruntime as ort
    import torch
    from hezar.models import Model

    verify_source_model(model_dir)

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # Hezar emits time,batch,classes. The runtime uses
            # batch,time,classes for a stable ONNX interface.
            return self.model(pixel_values)["logits"].permute(1, 0, 2)

    model = Model.load(
        str(Path(model_dir).resolve()),
        load_locally=True,
    )
    model.eval()
    wrapper = _LogitsWrapper(model).eval()
    config = model.config
    processor = model.preprocessor.image_processor.config
    width, height = (int(value) for value in processor.size)
    sample = torch.zeros(
        1,
        int(config.n_channels),
        height,
        width,
        dtype=torch.float32,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        reference = wrapper(sample).cpu().numpy()
    torch.onnx.export(
        wrapper,
        (sample,),
        str(output),
        input_names=["pixel_values"],
        output_names=["logits"],
        opset_version=18,
        dynamo=False,
    )
    session = ort.InferenceSession(
        str(output),
        providers=["CPUExecutionProvider"],
    )
    actual = session.run(
        None,
        {session.get_inputs()[0].name: sample.cpu().numpy()},
    )[0]
    maximum_error = float(np.max(np.abs(reference - actual)))
    if maximum_error > 1e-4:
        output.unlink(missing_ok=True)
        raise RuntimeError(
            "Hezar ONNX parity check failed: "
            f"max abs error {maximum_error}"
        )
    labels = [
        str(config.id2label[index])
        for index in range(len(config.id2label))
    ]
    raw_mean = processor.mean[0] if isinstance(
        processor.mean, list
    ) else processor.mean
    raw_std = processor.std[0] if isinstance(
        processor.std, list
    ) else processor.std
    metadata = {
        "source_model": HEZAR_MODEL_ID,
        "source_revision": HEZAR_REVISION,
        "filename": output.name,
        "sha256": _sha256_file(output),
        "size": output.stat().st_size,
        "input_height": height,
        "input_width": width,
        "channels": int(config.n_channels),
        "mean": float(raw_mean),
        "std": float(raw_std),
        "mirror": bool(processor.mirror),
        "labels": labels,
        "blank_index": int(config.blank_id),
        "reverse_output_digits": bool(config.reverse_output_digits),
        "output_shape": list(actual.shape),
        "maximum_parity_error": maximum_error,
    }
    output.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def export_pinned_model(output: Path, cache_dir: Path) -> dict:
    """Download one immutable Hezar revision, verify it, and export it."""

    from huggingface_hub import snapshot_download

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model_dir = Path(snapshot_download(
        repo_id=HEZAR_MODEL_ID,
        revision=HEZAR_REVISION,
        cache_dir=str(Path(cache_dir)),
        allow_patterns=list(HEZAR_SOURCE_FILES),
    ))
    verify_source_model(model_dir)
    with tempfile.TemporaryDirectory(
        prefix="hezar-export-",
        dir=output.parent,
    ) as temporary_dir:
        candidate = Path(temporary_dir) / output.name
        metadata = export_local_model(model_dir, candidate)
        if not _verify(
            candidate,
            HEZAR_ONNX_SHA256,
            HEZAR_ONNX_SIZE,
        ):
            raise ValueError(
                "Exported Hezar ONNX does not match the pinned artifact"
            )
        os.replace(candidate, output)
        metadata_path = candidate.with_suffix(".json")
        if metadata_path.is_file():
            os.replace(metadata_path, output.with_suffix(".json"))
    return metadata
