from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from tools.calibrate_engine_v2_tcam import (
    _BackendSessionFacade,
    _CCTPlateOCR,
    _HezarPlateOCR,
    _cct_result_to_ocr,
    _load_cct_spec,
    _load_hezar_spec,
    build_parser,
)


class _Backend:
    input_names = ("input",)
    output_names = ("output",)

    def __init__(self, output: np.ndarray) -> None:
        self.output = output
        self.calls: list[tuple[dict[str, object], object]] = []

    def infer(self, input_feed, output_names=None):
        self.calls.append((dict(input_feed), output_names))
        return [self.output.copy()]


def _install_fake_ai_modules(monkeypatch, **hezar_attributes) -> None:
    package = ModuleType("app.ai")
    package.__path__ = []
    hezar = ModuleType("app.ai.onnx_hezar")
    for name, value in hezar_attributes.items():
        setattr(hezar, name, value)
    monkeypatch.setitem(sys.modules, "app.ai", package)
    monkeypatch.setitem(sys.modules, "app.ai.onnx_hezar", hezar)


def test_collect_ir_lpr_defaults_to_current_production_hezar() -> None:
    args = build_parser().parse_args(
        [
            "collect-ir-lpr",
            "--dataset-root",
            "dataset",
            "--ocr-model",
            "model.onnx",
            "--output",
            "trace.json",
        ]
    )

    assert args.ocr_runtime == "hezar"


def test_hezar_contract_uses_pinned_production_hash(
    tmp_path: Path,
    monkeypatch,
) -> None:
    payload = b"pinned-hezar-model"
    model = tmp_path / "crnn_fa_v2.onnx"
    model.write_bytes(payload)
    manager = ModuleType("app.ai.model_manager")
    manager.HEZAR_ONNX_SIZE = len(payload)
    manager.HEZAR_ONNX_SHA256 = hashlib.sha256(payload).hexdigest()
    _install_fake_ai_modules(
        monkeypatch,
        HEZAR_V2_SPEC={"beam_width": 10, "top_k": 5},
    )
    monkeypatch.setitem(sys.modules, "app.ai.model_manager", manager)

    spec = _load_hezar_spec(model, beam_width=12, top_k=4)

    assert spec["beam_width"] == 12
    assert spec["top_k"] == 4
    model.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="pinned size/SHA-256"):
        _load_hezar_spec(model, beam_width=None, top_k=None)


def test_cct_contract_requires_manifest_hash_and_runtime(tmp_path: Path) -> None:
    payload = b"signed-cct-model"
    model = tmp_path / "candidate.onnx"
    model.write_bytes(payload)
    manifest = tmp_path / "active-models.json"
    manifest.write_text(
        """{
          "models": {"ocr": {
            "runtime": "fast-plate-ocr-cct",
            "alphabet": "0123456789ب_",
            "max_plate_slots": 8,
            "input_width": 128,
            "input_height": 64,
            "input_layout": "nhwc",
            "input_dtype": "uint8",
            "image_color_mode": "rgb",
            "size": %d,
            "sha256": "%s"
          }}
        }"""
        % (len(payload), hashlib.sha256(payload).hexdigest()),
        encoding="utf-8",
    )

    spec, source = _load_cct_spec(
        model,
        manifest,
        beam_width=None,
        top_k=None,
    )

    assert spec["runtime"] == "fast-plate-ocr-cct"
    assert source == manifest.resolve()


def test_rejected_cct_guess_stays_diagnostic_only() -> None:
    result = _cct_result_to_ocr(
        {
            "accepted": False,
            "raw_plate_norm": "12ب34567",
            "confidence": 0.69,
            "reason": "view-disagreement",
            "hypotheses": [
                {
                    "plate_norm": "12ب34567",
                    "confidence": 0.91,
                    "positions": {index: {"confidence": 0.91} for index in range(8)},
                }
            ],
        }
    )

    assert result.text == "12ب34567"
    assert result.valid is False
    assert result.metadata["candidates"] == []


def test_cct_adapter_runs_through_engine_v2_backend(monkeypatch) -> None:
    seen = {}

    def infer_cct_session(session, input_name, image, spec):
        seen["output"] = session.run(None, {input_name: image})[0]
        seen["spec"] = spec
        return {
            "accepted": True,
            "raw_plate_norm": "12ب34567",
            "confidence": 0.91,
            "hypotheses": [{"plate_norm": "12ب34567", "confidence": 0.91}],
        }

    cct = ModuleType("app.ai.onnx_cct")
    cct.infer_cct_session = infer_cct_session
    package = ModuleType("app.ai")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "app.ai", package)
    monkeypatch.setitem(sys.modules, "app.ai.onnx_cct", cct)
    backend = _Backend(np.ones((1, 8, 37), dtype=np.float32))

    result = _CCTPlateOCR(backend, {"runtime": "fast-plate-ocr-cct"}).read(
        np.zeros((32, 96, 3), dtype=np.uint8)
    )

    assert result.valid is True
    assert len(backend.calls) == 1
    assert seen["output"].shape == (1, 8, 37)


def test_hezar_adapter_uses_production_preprocess_and_decoder(monkeypatch) -> None:
    decoder_input = {}

    def prepare_hezar_input(_crop, _spec):
        return np.ones((1, 1, 32, 384), dtype=np.float32)

    def ctc_beam_hypotheses(logits, **_kwargs):
        decoder_input["logits"] = logits.copy()
        return [{"plate_norm": "12ب34567", "confidence": 0.92}]

    def accept_hypotheses(hypotheses, **_kwargs):
        return {
            "accepted": True,
            "plate_norm": hypotheses[0]["plate_norm"],
            "confidence": hypotheses[0]["confidence"],
            "position_details": [{"probability": 0.9, "margin": 0.5} for _ in range(8)],
            "hypotheses": hypotheses,
        }

    _install_fake_ai_modules(
        monkeypatch,
        prepare_hezar_input=prepare_hezar_input,
        ctc_beam_hypotheses=ctc_beam_hypotheses,
        accept_hypotheses=accept_hypotheses,
    )
    output = np.asarray([[[1.0], [2.0], [3.0]]], dtype=np.float32)
    backend = _Backend(output)

    result = _HezarPlateOCR(
        backend,
        {
            "labels": ["", "1"],
            "blank_index": 0,
            "reverse_output_digits": True,
            "beam_width": 10,
            "top_k": 5,
        },
    ).read(np.zeros((32, 96, 3), dtype=np.uint8))

    assert result.valid is True
    assert result.text == "12ب34567"
    assert tuple(decoder_input["logits"][:, 0]) == (3.0, 2.0, 1.0)
    assert len(backend.calls) == 1


def test_backend_session_facade_preserves_output_selection() -> None:
    backend = _Backend(np.zeros((1,), dtype=np.float32))

    outputs = _BackendSessionFacade(backend).run(
        ["output"],
        {"input": np.zeros((1,), dtype=np.float32)},
    )

    assert len(outputs) == 1
    assert backend.calls[0][1] == ["output"]
