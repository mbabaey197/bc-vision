from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any

import pytest

from app.engine_v2.inference import (
    BackendKind,
    InferenceConfig,
    InferenceUnavailableError,
    SharedInferenceBackend,
    rank_onnx_providers,
    rank_openvino_devices,
)


class FakePort:
    def __init__(self, name: str) -> None:
        self.name = name

    def get_any_name(self) -> str:
        return self.name


class FakeCompiledModel:
    def __init__(self) -> None:
        self.inputs = (FakePort("images"),)
        self.outputs = (FakePort("plates"),)
        self.calls = 0
        self.closed = False

    def __call__(self, input_feed: dict[str, Any]) -> dict[FakePort, Any]:
        self.calls += 1
        return {self.outputs[0]: input_feed["images"] + 1}

    def close(self) -> None:
        self.closed = True


class FakeOpenVINOCore:
    def __init__(self, fail_devices: set[str] | None = None) -> None:
        self.available_devices = ("CPU", "GPU.0")
        self.fail_devices = fail_devices or set()
        self.read_paths: list[str] = []
        self.compile_attempts: list[str] = []
        self.compiled = FakeCompiledModel()

    def read_model(self, path: str) -> object:
        self.read_paths.append(path)
        return object()

    def compile_model(
        self,
        *,
        model: object,
        device_name: str,
        config: dict[str, str],
    ) -> FakeCompiledModel:
        del model, config
        self.compile_attempts.append(device_name)
        if device_name in self.fail_devices:
            raise RuntimeError(f"cannot compile for {device_name}")
        return self.compiled


class FakeNode:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeORTSession:
    def __init__(self, providers: list[str]) -> None:
        self.providers = tuple(providers)
        self.closed = False

    def get_inputs(self) -> list[FakeNode]:
        return [FakeNode("images")]

    def get_outputs(self) -> list[FakeNode]:
        return [FakeNode("plates")]

    def get_providers(self) -> list[str]:
        return list(self.providers)

    def run(self, output_names: list[str] | None, input_feed: dict[str, Any]) -> list[Any]:
        del output_names
        if input_feed.get("fail"):
            raise ValueError("synthetic inference failure")
        return [input_feed["images"] + 1]

    def close(self) -> None:
        self.closed = True


class FakeSessionOptions:
    pass


def _missing_module(name: str) -> Any:
    raise ModuleNotFoundError(name)


def _ort_module(providers: tuple[str, ...]) -> Any:
    return SimpleNamespace(
        __version__="9.8.7-test",
        get_available_providers=lambda: list(providers),
        SessionOptions=FakeSessionOptions,
        GraphOptimizationLevel=SimpleNamespace(
            ORT_ENABLE_ALL="all",
            ORT_DISABLE_ALL="none",
        ),
    )


def test_provider_and_device_ranking_is_deterministic() -> None:
    assert rank_openvino_devices(("CPU", "NPU", "GPU.1", "GPU.0")) == (
        "GPU.1",
        "GPU.0",
        "CPU",
        "NPU",
    )
    assert rank_openvino_devices(("CPU", "GPU.0"), "CPU") == ("CPU",)
    assert rank_onnx_providers(
        ("CPUExecutionProvider", "CUDAExecutionProvider", "OpenVINOExecutionProvider")
    ) == (
        "OpenVINOExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    assert rank_onnx_providers(("CPUExecutionProvider", "FutureAcceleratorProvider")) == (
        "FutureAcceleratorProvider",
        "CPUExecutionProvider",
    )
    assert rank_onnx_providers(("AzureExecutionProvider", "CPUExecutionProvider")) == (
        "CPUExecutionProvider",
        "AzureExecutionProvider",
    )


def test_auto_prefers_direct_openvino_igpu_and_reuses_one_compiled_model() -> None:
    core = FakeOpenVINOCore()
    module = SimpleNamespace(__version__="2026.2-test", Core=FakeOpenVINOCore)
    loads: list[str] = []

    def loader(name: str) -> Any:
        loads.append(name)
        if name == "openvino":
            return module
        raise AssertionError(f"unexpected module load: {name}")

    backend = SharedInferenceBackend(
        InferenceConfig("plate-detector.onnx"),
        module_loader=loader,
        openvino_core_factory=lambda: core,
    )

    assert backend.metadata.backend is BackendKind.OPENVINO
    assert backend.metadata.device == "GPU.0"
    assert backend.metadata.available_devices == ("CPU", "GPU.0")
    assert backend.metadata.runtime_version == "2026.2-test"
    assert backend.input_names == ("images",)
    assert backend.output_names == ("plates",)
    assert core.read_paths == ["plate-detector.onnx"]
    assert core.compile_attempts == ["GPU.0"]
    assert loads == ["openvino"]

    assert backend.infer({"images": 10}) == [11]
    assert backend.run(["plates"], {"images": 20}) == [21]
    assert core.compiled.calls == 2

    backend.close()
    backend.close()
    assert core.compiled.closed is True


def test_openvino_falls_back_from_gpu_to_cpu_and_records_reason() -> None:
    core = FakeOpenVINOCore(fail_devices={"GPU.0"})
    module = SimpleNamespace(__version__="test", Core=FakeOpenVINOCore)
    backend = SharedInferenceBackend(
        InferenceConfig("ocr.onnx", backend="openvino"),
        module_loader=lambda name: module,
        openvino_core_factory=lambda: core,
    )

    assert core.compile_attempts == ["GPU.0", "CPU"]
    assert backend.metadata.device == "CPU"
    assert any("openvino/GPU.0" in item for item in backend.metadata.fallback_log)


def test_auto_uses_ranked_ort_openvino_provider_when_direct_runtime_is_absent() -> None:
    module = _ort_module(
        (
            "CPUExecutionProvider",
            "CUDAExecutionProvider",
            "OpenVINOExecutionProvider",
        )
    )
    factory_calls: list[dict[str, Any]] = []

    def loader(name: str) -> Any:
        if name in {"openvino", "openvino.runtime"}:
            raise ModuleNotFoundError(name)
        if name == "onnxruntime":
            return module
        raise AssertionError(name)

    def session_factory(model_path: str, **kwargs: Any) -> FakeORTSession:
        factory_calls.append({"model_path": model_path, **kwargs})
        return FakeORTSession(kwargs["providers"])

    backend = SharedInferenceBackend(
        InferenceConfig(
            "detector.onnx",
            intra_op_threads=3,
            inter_op_threads=1,
        ),
        module_loader=loader,
        ort_session_factory=session_factory,
    )

    assert backend.metadata.backend is BackendKind.ONNX_RUNTIME
    assert backend.metadata.providers == (
        "OpenVINOExecutionProvider",
        "CPUExecutionProvider",
    )
    assert backend.metadata.device == "AUTO"
    assert backend.metadata.available_providers == (
        "CPUExecutionProvider",
        "CUDAExecutionProvider",
        "OpenVINOExecutionProvider",
    )
    assert any(item.startswith("openvino:") for item in backend.metadata.fallback_log)
    assert len(factory_calls) == 1
    call = factory_calls[0]
    assert call["provider_options"] == [{"device_type": "AUTO"}, {}]
    assert call["sess_options"].intra_op_num_threads == 3
    assert call["sess_options"].inter_op_num_threads == 1
    assert call["sess_options"].graph_optimization_level == "all"
    assert backend.infer({"images": 4}) == [5]


def test_onnx_runtime_falls_back_to_next_provider_after_session_creation_failure() -> None:
    module = _ort_module(("CPUExecutionProvider", "CUDAExecutionProvider"))
    attempts: list[tuple[str, ...]] = []

    def factory(model_path: str, **kwargs: Any) -> FakeORTSession:
        del model_path
        providers = tuple(kwargs["providers"])
        attempts.append(providers)
        if providers[0] == "CUDAExecutionProvider":
            raise RuntimeError("CUDA device is busy")
        return FakeORTSession(list(providers))

    backend = SharedInferenceBackend(
        InferenceConfig("ocr.onnx", backend="onnxruntime"),
        module_loader=lambda name: module,
        ort_session_factory=factory,
    )

    assert attempts == [
        ("CUDAExecutionProvider", "CPUExecutionProvider"),
        ("CPUExecutionProvider",),
    ]
    assert backend.metadata.providers == ("CPUExecutionProvider",)
    assert backend.metadata.device == "CPU"
    assert any("CUDA device is busy" in item for item in backend.metadata.fallback_log)


def test_onnx_runtime_does_not_append_cpu_when_fallback_is_disabled() -> None:
    module = _ort_module(("CPUExecutionProvider", "CUDAExecutionProvider"))
    attempts: list[tuple[str, ...]] = []

    def factory(model_path: str, **kwargs: Any) -> FakeORTSession:
        del model_path
        providers = tuple(kwargs["providers"])
        attempts.append(providers)
        return FakeORTSession(list(providers))

    backend = SharedInferenceBackend(
        InferenceConfig(
            "detector.onnx",
            backend="onnxruntime",
            allow_fallback=False,
        ),
        module_loader=lambda name: module,
        ort_session_factory=factory,
    )

    assert attempts == [("CUDAExecutionProvider",)]
    assert backend.metadata.providers == ("CUDAExecutionProvider",)
    backend.close()


def test_shared_session_serializes_calls_and_collects_thread_safe_telemetry() -> None:
    module = _ort_module(("CPUExecutionProvider",))
    barrier = threading.Barrier(5)

    class ConcurrencySession(FakeORTSession):
        def __init__(self) -> None:
            super().__init__(["CPUExecutionProvider"])
            self.active = 0
            self.max_active = 0
            self.guard = threading.Lock()

        def run(
            self,
            output_names: list[str] | None,
            input_feed: dict[str, Any],
        ) -> list[Any]:
            del output_names
            with self.guard:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            time.sleep(0.02)
            with self.guard:
                self.active -= 1
            return [input_feed["images"]]

    session = ConcurrencySession()
    backend = SharedInferenceBackend(
        InferenceConfig("shared.onnx", backend="onnxruntime"),
        module_loader=lambda name: module,
        ort_session_factory=lambda model_path, **kwargs: session,
    )

    def invoke(value: int) -> int:
        barrier.wait()
        return backend.infer({"images": value})[0]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(invoke, value) for value in range(4)]
        barrier.wait()
        assert sorted(future.result() for future in futures) == [0, 1, 2, 3]

    telemetry = backend.telemetry
    assert session.max_active == 1
    assert telemetry.requests == 4
    assert telemetry.successful_requests == 4
    assert telemetry.failed_requests == 0
    assert telemetry.active_requests == 0
    assert telemetry.max_concurrent_requests == 4
    assert telemetry.average_latency_ms > 0
    assert telemetry.p95_latency_ms > 0
    assert telemetry.total_queue_wait_ms > 0


def test_inference_failures_are_counted_and_optional_runtimes_fail_cleanly() -> None:
    module = _ort_module(("CPUExecutionProvider",))
    session = FakeORTSession(["CPUExecutionProvider"])
    backend = SharedInferenceBackend(
        InferenceConfig("shared-ocr.onnx", backend="onnxruntime"),
        module_loader=lambda name: module,
        ort_session_factory=lambda model_path, **kwargs: session,
    )

    assert backend.infer({"images": 1}) == [2]
    with pytest.raises(ValueError, match="synthetic inference failure"):
        backend.infer({"images": 1, "fail": True})
    telemetry = backend.telemetry_snapshot()
    assert telemetry.requests == 2
    assert telemetry.successful_requests == 1
    assert telemetry.failed_requests == 1
    assert telemetry.last_error == "ValueError: synthetic inference failure"

    with pytest.raises(InferenceUnavailableError) as error:
        SharedInferenceBackend(
            InferenceConfig("missing.onnx"),
            module_loader=_missing_module,
        )
    assert "openvino" in str(error.value)
    assert "onnxruntime" in str(error.value)


def test_configuration_validation_does_not_import_optional_dependencies() -> None:
    with pytest.raises(ValueError, match="unsupported inference backend"):
        InferenceConfig("model.onnx", backend="tensorflow")
    with pytest.raises(ValueError, match="telemetry_window"):
        InferenceConfig("model.onnx", telemetry_window=0)
