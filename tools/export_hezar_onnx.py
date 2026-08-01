"""Export the official Hezar Persian plate CRNN to a verified ONNX artifact.

This development utility does not redistribute model weights.  It downloads
the selected Hezar model into a caller-controlled cache, exports a fixed CPU
inference graph, and verifies numerical parity with ONNX Runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


DEFAULT_MODEL = "hezarai/crnn-fa-license-plate-recognition-v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest().upper()


def export_model(model_id: str, output: Path, cache_dir: Path) -> dict:
    import numpy as np
    import onnxruntime as ort
    import torch
    from hezar.models import Model

    class _LogitsWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            # Hezar returns time, batch, classes.  BC Vision uses the more
            # deployment-friendly batch, time, classes contract.
            return self.model(pixel_values)["logits"].permute(1, 0, 2)

    model = Model.load(model_id, cache_dir=str(cache_dir))
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
            f"ONNX parity check failed: max abs error {maximum_error}"
        )
    labels = [
        str(config.id2label[index])
        for index in range(len(config.id2label))
    ]
    mean = processor.mean[0] if isinstance(processor.mean, list) else processor.mean
    std = processor.std[0] if isinstance(processor.std, list) else processor.std
    metadata = {
        "source_model": model_id,
        "filename": output.name,
        "sha256": sha256_file(output),
        "size": output.stat().st_size,
        "input_height": height,
        "input_width": width,
        "channels": int(config.n_channels),
        "mean": float(mean),
        "std": float(std),
        "mirror": bool(processor.mirror),
        "labels": labels,
        "blank_index": int(config.blank_id),
        "reverse_output_digits": bool(config.reverse_output_digits),
        "output_shape": list(actual.shape),
        "maximum_parity_error": maximum_error,
    }
    metadata_path = output.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Export a Hezar Persian plate CRNN to ONNX",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    metadata = export_model(
        args.model,
        args.output.resolve(),
        args.cache_dir.resolve(),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
