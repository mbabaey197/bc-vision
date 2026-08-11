from __future__ import annotations

import pytest

import app.engine_v2.factory as factory_module
from app.engine_v2.factory import SharedModelBundleConfig


class _ClosableBackend:
    def __init__(self, name: str, close_order: list[str]) -> None:
        self.name = name
        self.close_order = close_order
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1
        self.close_order.append(self.name)


@pytest.mark.parametrize("failing_adapter", ["yolo", "ctc"])
def test_build_shared_models_closes_both_backends_once_on_adapter_failure(
    monkeypatch: pytest.MonkeyPatch,
    failing_adapter: str,
) -> None:
    close_order: list[str] = []
    backends = [
        _ClosableBackend("detector", close_order),
        _ClosableBackend("ocr", close_order),
    ]
    backend_iterator = iter(backends)
    monkeypatch.setattr(
        factory_module,
        "SharedInferenceBackend",
        lambda _config: next(backend_iterator),
    )

    def build_yolo(*_args: object) -> object:
        if failing_adapter == "yolo":
            raise RuntimeError("synthetic YOLO adapter failure")
        return object()

    def build_ctc(*_args: object) -> object:
        if failing_adapter == "ctc":
            raise RuntimeError("synthetic CTC adapter failure")
        return object()

    monkeypatch.setattr(factory_module, "YOLOPlateDetector", build_yolo)
    monkeypatch.setattr(factory_module, "CTCPlateOCR", build_ctc)

    with pytest.raises(RuntimeError, match=f"synthetic {failing_adapter.upper()} adapter failure"):
        factory_module.build_shared_models(
            SharedModelBundleConfig("detector.onnx", "ocr.onnx")
        )

    assert [backend.close_calls for backend in backends] == [1, 1]
    assert close_order == ["ocr", "detector"]


def test_build_shared_models_transfers_cleanup_ownership_on_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_order: list[str] = []
    backends = [
        _ClosableBackend("detector", close_order),
        _ClosableBackend("ocr", close_order),
    ]
    backend_iterator = iter(backends)
    monkeypatch.setattr(
        factory_module,
        "SharedInferenceBackend",
        lambda _config: next(backend_iterator),
    )
    detector = object()
    ocr = object()
    monkeypatch.setattr(factory_module, "YOLOPlateDetector", lambda *_args: detector)
    monkeypatch.setattr(factory_module, "CTCPlateOCR", lambda *_args: ocr)

    bundle = factory_module.build_shared_models(
        SharedModelBundleConfig("detector.onnx", "ocr.onnx")
    )

    assert bundle.detector is detector
    assert bundle.ocr is ocr
    assert [backend.close_calls for backend in backends] == [0, 0]
    bundle.close()
    assert [backend.close_calls for backend in backends] == [1, 1]
    assert close_order == ["ocr", "detector"]
