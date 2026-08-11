"""Shared, optional-dependency inference runtime for ANPR Engine V2.

The module intentionally has no import-time dependency on OpenVINO or ONNX
Runtime.  A model backend is selected once, then the resulting session can be
shared by all camera jobs.  Calls are serialized by default because not every
OpenVINO/third-party session implementation guarantees that a single infer
request is safe to use concurrently.

This is a clean-room Engine V2 component and does not import the legacy ANPR
runtime.
"""

from __future__ import annotations

import importlib
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any, Callable, Mapping, Protocol, Sequence


DEFAULT_ONNX_PROVIDER_ORDER = (
    # Prefer Intel's execution provider when it is installed.  Direct
    # OpenVINO is attempted before ONNX Runtime in ``auto`` mode.
    "OpenVINOExecutionProvider",
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "MIGraphXExecutionProvider",
    "ROCMExecutionProvider",
    "DmlExecutionProvider",
    "CoreMLExecutionProvider",
    "QNNExecutionProvider",
    "XnnpackExecutionProvider",
    "ACLExecutionProvider",
    "ArmNNExecutionProvider",
    "CPUExecutionProvider",
)

# ONNX Runtime may expose service/control providers that are not local AI
# accelerators. They must not outrank a deterministic local CPU fallback.
_NON_ACCELERATOR_PROVIDERS = frozenset({"AzureExecutionProvider"})


class BackendKind(str, Enum):
    OPENVINO = "openvino"
    ONNX_RUNTIME = "onnxruntime"


class InferenceBackendError(RuntimeError):
    """Base error raised while preparing an Engine V2 inference backend."""


class InferenceUnavailableError(InferenceBackendError):
    """Raised when no requested optional inference runtime can be created."""


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration for one shared model session.

    ``backend="auto"`` first tries direct OpenVINO (GPU, then CPU), followed
    by ranked ONNX Runtime execution providers.  Selecting an explicit backend
    disables cross-runtime fallback but still permits device/provider fallback.
    """

    model_path: str | Path
    backend: str = "auto"
    device: str = "AUTO"
    provider_order: tuple[str, ...] = DEFAULT_ONNX_PROVIDER_ORDER
    provider_options: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    openvino_compile_config: Mapping[str, str] = field(default_factory=dict)
    intra_op_threads: int | None = None
    inter_op_threads: int | None = None
    graph_optimization: bool = True
    serialize_calls: bool = True
    allow_fallback: bool = True
    telemetry_window: int = 256

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        if backend not in {"auto", BackendKind.OPENVINO.value, BackendKind.ONNX_RUNTIME.value}:
            raise ValueError(f"unsupported inference backend: {self.backend!r}")
        if not str(self.model_path).strip():
            raise ValueError("model_path must not be empty")
        if self.intra_op_threads is not None and self.intra_op_threads < 1:
            raise ValueError("intra_op_threads must be positive")
        if self.inter_op_threads is not None and self.inter_op_threads < 1:
            raise ValueError("inter_op_threads must be positive")
        if self.telemetry_window < 1:
            raise ValueError("telemetry_window must be positive")


@dataclass(frozen=True, slots=True)
class InferenceMetadata:
    """Immutable facts about the selected shared model session."""

    model_path: str
    backend: BackendKind
    runtime_version: str | None
    device: str | None
    providers: tuple[str, ...]
    available_devices: tuple[str, ...]
    available_providers: tuple[str, ...]
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    initialization_ms: float
    fallback_log: tuple[str, ...]
    serialized_calls: bool


@dataclass(frozen=True, slots=True)
class InferenceTelemetry:
    """Point-in-time, thread-safe inference counters and latency summary."""

    requests: int
    successful_requests: int
    failed_requests: int
    active_requests: int
    max_concurrent_requests: int
    average_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    last_latency_ms: float
    average_queue_wait_ms: float
    total_queue_wait_ms: float
    last_error: str | None


class _SessionAdapter(Protocol):
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]: ...

    def close(self) -> None: ...


def rank_openvino_devices(
    available_devices: Sequence[str],
    preferred: str = "AUTO",
) -> tuple[str, ...]:
    """Rank OpenVINO devices with Intel GPU first and CPU second.

    OpenVINO device identifiers may include an index (for example ``GPU.0``).
    If device discovery is unavailable, ``AUTO`` is returned so OpenVINO can
    make the decision itself.
    """

    devices = _unique_strings(available_devices)
    requested = preferred.strip() or "AUTO"
    if requested.upper() != "AUTO":
        exact = [device for device in devices if device.upper() == requested.upper()]
        family = [
            device
            for device in devices
            if device not in exact and _device_family(device) == requested.upper()
        ]
        return tuple(exact + family)
    if not devices:
        return ("AUTO",)

    priorities = {"GPU": 0, "CPU": 1, "NPU": 2, "AUTO": 3}
    original_index = {device: index for index, device in enumerate(devices)}
    return tuple(
        sorted(
            devices,
            key=lambda device: (
                priorities.get(_device_family(device), 10),
                original_index[device],
            ),
        )
    )


def rank_onnx_providers(
    available_providers: Sequence[str],
    preferred_order: Sequence[str] = DEFAULT_ONNX_PROVIDER_ORDER,
) -> tuple[str, ...]:
    """Return installed ONNX Runtime providers in deterministic priority order."""

    available = _unique_strings(available_providers)
    preferred = _unique_strings(preferred_order)
    rank = {provider: index for index, provider in enumerate(preferred)}
    original_index = {provider: index for index, provider in enumerate(available)}

    def sort_key(provider: str) -> tuple[int, int]:
        if provider in _NON_ACCELERATOR_PROVIDERS:
            return (3, original_index[provider])
        if provider == "CPUExecutionProvider":
            return (2, original_index[provider])
        if provider in rank:
            return (0, rank[provider])
        # Unknown providers may be accelerators introduced by a newer ONNX
        # Runtime. Preserve their reported order, but still try them before CPU.
        return (1, original_index[provider])

    return tuple(
        sorted(
            available,
            key=sort_key,
        )
    )


class SharedInferenceBackend:
    """One model session shared safely by central detector or OCR workers.

    Dependencies and factories are injectable so provider selection can be
    tested without installing OpenVINO or ONNX Runtime.  In production callers
    normally pass only :class:`InferenceConfig`.
    """

    def __init__(
        self,
        config: InferenceConfig,
        *,
        module_loader: Callable[[str], Any] = importlib.import_module,
        openvino_core_factory: Callable[[], Any] | None = None,
        ort_session_factory: Callable[..., Any] | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.config = config
        self._module_loader = module_loader
        self._openvino_core_factory = openvino_core_factory
        self._ort_session_factory = ort_session_factory
        self._clock = clock
        self._run_lock = threading.RLock()
        self._telemetry_lock = threading.Lock()
        self._latencies_ms: deque[float] = deque(maxlen=config.telemetry_window)
        self._requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._active_requests = 0
        self._max_concurrent_requests = 0
        self._total_latency_ms = 0.0
        self._last_latency_ms = 0.0
        self._max_latency_ms = 0.0
        self._total_queue_wait_ms = 0.0
        self._last_error: str | None = None
        self._closed = False

        started = self._clock()
        adapter, selection = self._select_backend()
        initialization_ms = max(0.0, (self._clock() - started) * 1000.0)
        self._adapter = adapter
        self.metadata = InferenceMetadata(
            model_path=str(config.model_path),
            backend=selection.backend,
            runtime_version=selection.runtime_version,
            device=selection.device,
            providers=selection.providers,
            available_devices=selection.available_devices,
            available_providers=selection.available_providers,
            input_names=adapter.input_names,
            output_names=adapter.output_names,
            initialization_ms=initialization_ms,
            fallback_log=selection.fallback_log,
            serialized_calls=config.serialize_calls,
        )

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.metadata.input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self.metadata.output_names

    @property
    def telemetry(self) -> InferenceTelemetry:
        return self.telemetry_snapshot()

    def telemetry_snapshot(self) -> InferenceTelemetry:
        with self._telemetry_lock:
            latencies = tuple(self._latencies_ms)
            p95 = _percentile(latencies, 0.95)
            average = self._total_latency_ms / self._requests if self._requests else 0.0
            average_wait = self._total_queue_wait_ms / self._requests if self._requests else 0.0
            return InferenceTelemetry(
                requests=self._requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                active_requests=self._active_requests,
                max_concurrent_requests=self._max_concurrent_requests,
                average_latency_ms=average,
                p95_latency_ms=p95,
                max_latency_ms=self._max_latency_ms,
                last_latency_ms=self._last_latency_ms,
                average_queue_wait_ms=average_wait,
                total_queue_wait_ms=self._total_queue_wait_ms,
                last_error=self._last_error,
            )

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]:
        """Run one inference and update latency/error telemetry.

        The return value follows ONNX Runtime's convention: a list ordered by
        requested output names, or by model output order when no names are
        supplied.
        """

        if self._closed:
            raise InferenceBackendError("inference backend is closed")
        if not isinstance(input_feed, Mapping):
            raise TypeError("input_feed must be a mapping of input names to tensors")

        started = self._clock()
        with self._telemetry_lock:
            self._active_requests += 1
            self._max_concurrent_requests = max(
                self._max_concurrent_requests,
                self._active_requests,
            )

        lock_acquired = started
        succeeded = False
        error_text: str | None = None
        try:
            if self.config.serialize_calls:
                with self._run_lock:
                    lock_acquired = self._clock()
                    if self._closed:
                        raise InferenceBackendError("inference backend is closed")
                    result = self._adapter.infer(input_feed, output_names)
            else:
                lock_acquired = self._clock()
                result = self._adapter.infer(input_feed, output_names)
            succeeded = True
            return result
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            finished = self._clock()
            latency_ms = max(0.0, (finished - started) * 1000.0)
            queue_wait_ms = max(0.0, (lock_acquired - started) * 1000.0)
            with self._telemetry_lock:
                self._active_requests -= 1
                self._requests += 1
                self._last_latency_ms = latency_ms
                self._max_latency_ms = max(self._max_latency_ms, latency_ms)
                self._total_latency_ms += latency_ms
                self._total_queue_wait_ms += queue_wait_ms
                self._latencies_ms.append(latency_ms)
                if succeeded:
                    self._successful_requests += 1
                else:
                    self._failed_requests += 1
                    self._last_error = error_text

    def run(
        self,
        output_names: Sequence[str] | None,
        input_feed: Mapping[str, Any],
    ) -> list[Any]:
        """ONNX Runtime-compatible spelling used by model adapters."""

        return self.infer(input_feed, output_names)

    def close(self) -> None:
        if self._closed:
            return
        with self._run_lock:
            if self._closed:
                return
            self._adapter.close()
            self._closed = True

    def __enter__(self) -> SharedInferenceBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _select_backend(self) -> tuple[_SessionAdapter, _Selection]:
        requested = self.config.backend.strip().lower()
        backend_order = (
            (BackendKind.OPENVINO, BackendKind.ONNX_RUNTIME)
            if requested == "auto"
            else (BackendKind(requested),)
        )
        failures: list[str] = []
        for backend in backend_order:
            try:
                if backend is BackendKind.OPENVINO:
                    return self._create_openvino(failures)
                return self._create_onnx_runtime(failures)
            except Exception as exc:
                failures.append(f"{backend.value}: {type(exc).__name__}: {exc}")

        detail = "; ".join(failures) or "no backend candidates"
        raise InferenceUnavailableError(
            f"could not initialize inference model {str(self.config.model_path)!r}: {detail}"
        )

    def _create_openvino(self, previous_failures: list[str]) -> tuple[_SessionAdapter, _Selection]:
        module = self._load_openvino()
        core_factory = self._openvino_core_factory or getattr(module, "Core", None)
        if core_factory is None:
            raise InferenceBackendError("OpenVINO Core is unavailable")
        core = core_factory()
        available_devices = _read_openvino_devices(core)
        devices = rank_openvino_devices(available_devices, self.config.device)
        if not devices:
            raise InferenceBackendError(
                f"requested OpenVINO device {self.config.device!r} is not available"
            )
        if not self.config.allow_fallback:
            devices = devices[:1]

        model = core.read_model(str(self.config.model_path))
        failures = list(previous_failures)
        compile_config = dict(self.config.openvino_compile_config)
        for device in devices:
            try:
                compiled = core.compile_model(
                    model=model,
                    device_name=device,
                    config=compile_config,
                )
                adapter = _OpenVINOAdapter(compiled)
                selection = _Selection(
                    backend=BackendKind.OPENVINO,
                    runtime_version=_runtime_version(module),
                    device=device,
                    providers=(),
                    available_devices=available_devices,
                    available_providers=(),
                    fallback_log=tuple(failures),
                )
                return adapter, selection
            except Exception as exc:
                failures.append(f"openvino/{device}: {type(exc).__name__}: {exc}")
        raise InferenceBackendError("; ".join(failures))

    def _create_onnx_runtime(
        self,
        previous_failures: list[str],
    ) -> tuple[_SessionAdapter, _Selection]:
        module = self._module_loader("onnxruntime")
        provider_reader = getattr(module, "get_available_providers", None)
        if provider_reader is None:
            raise InferenceBackendError("onnxruntime.get_available_providers is unavailable")
        available_providers = tuple(provider_reader())
        providers = rank_onnx_providers(available_providers, self.config.provider_order)
        if not providers:
            raise InferenceBackendError("ONNX Runtime reports no execution providers")
        if not self.config.allow_fallback:
            providers = providers[:1]

        session_options = _make_ort_session_options(module, self.config)
        session_factory = self._ort_session_factory or getattr(module, "InferenceSession", None)
        if session_factory is None:
            raise InferenceBackendError("onnxruntime.InferenceSession is unavailable")

        failures = list(previous_failures)
        cpu_available = "CPUExecutionProvider" in available_providers
        for provider in providers:
            provider_chain = [provider]
            if (
                self.config.allow_fallback
                and provider != "CPUExecutionProvider"
                and cpu_available
            ):
                provider_chain.append("CPUExecutionProvider")
            provider_options = [self._provider_options(name) for name in provider_chain]
            kwargs: dict[str, Any] = {"providers": provider_chain}
            if session_options is not None:
                kwargs["sess_options"] = session_options
            if any(provider_options):
                kwargs["provider_options"] = provider_options
            try:
                session = session_factory(str(self.config.model_path), **kwargs)
                adapter = _ONNXRuntimeAdapter(session)
                actual_providers_reader = getattr(session, "get_providers", None)
                actual_providers = (
                    tuple(actual_providers_reader())
                    if callable(actual_providers_reader)
                    else tuple(provider_chain)
                )
                actual_primary = actual_providers[0] if actual_providers else provider
                selection = _Selection(
                    backend=BackendKind.ONNX_RUNTIME,
                    runtime_version=_runtime_version(module),
                    device=_device_for_ort_provider(
                        actual_primary,
                        self._provider_options(actual_primary),
                    ),
                    providers=actual_providers,
                    available_devices=(),
                    available_providers=available_providers,
                    fallback_log=tuple(failures),
                )
                return adapter, selection
            except Exception as exc:
                failures.append(f"onnxruntime/{provider}: {type(exc).__name__}: {exc}")
        raise InferenceBackendError("; ".join(failures))

    def _provider_options(self, provider: str) -> dict[str, str]:
        options = dict(self.config.provider_options.get(provider, {}))
        if provider == "OpenVINOExecutionProvider" and "device_type" not in options:
            requested = self.config.device.strip().upper() or "AUTO"
            options["device_type"] = requested
        return options

    def _load_openvino(self) -> Any:
        try:
            module = self._module_loader("openvino")
        except (ImportError, ModuleNotFoundError):
            module = self._module_loader("openvino.runtime")
        if getattr(module, "Core", None) is None:
            # Some distributions expose version information at the package root
            # but retain Core under openvino.runtime.
            runtime_module = self._module_loader("openvino.runtime")
            if getattr(runtime_module, "__version__", None) is None:
                try:
                    runtime_module.__version__ = module.__version__
                except (AttributeError, TypeError):
                    pass
            return runtime_module
        return module


@dataclass(frozen=True, slots=True)
class _Selection:
    backend: BackendKind
    runtime_version: str | None
    device: str | None
    providers: tuple[str, ...]
    available_devices: tuple[str, ...]
    available_providers: tuple[str, ...]
    fallback_log: tuple[str, ...]


class _ONNXRuntimeAdapter:
    def __init__(self, session: Any) -> None:
        self._session = session
        self.input_names = _ort_node_names(session, "get_inputs")
        self.output_names = _ort_node_names(session, "get_outputs")

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]:
        result = self._session.run(None if output_names is None else list(output_names), dict(input_feed))
        return list(result)

    def close(self) -> None:
        close = getattr(self._session, "close", None)
        if callable(close):
            close()


class _OpenVINOAdapter:
    def __init__(self, compiled_model: Any) -> None:
        self._compiled_model = compiled_model
        self._inputs = tuple(getattr(compiled_model, "inputs", ()) or ())
        self._outputs = tuple(getattr(compiled_model, "outputs", ()) or ())
        self.input_names = tuple(_port_name(port, index) for index, port in enumerate(self._inputs))
        self.output_names = tuple(_port_name(port, index) for index, port in enumerate(self._outputs))

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]:
        result = self._compiled_model(dict(input_feed))
        if isinstance(result, Mapping):
            ordered = self._mapping_outputs(result)
        elif isinstance(result, (list, tuple)):
            ordered = list(result)
        else:
            ordered = [result]

        if output_names is None:
            return ordered
        if not self.output_names:
            raise KeyError("OpenVINO model did not expose output names")
        by_name = dict(zip(self.output_names, ordered))
        return [by_name[name] for name in output_names]

    def _mapping_outputs(self, result: Mapping[Any, Any]) -> list[Any]:
        if self._outputs and all(port in result for port in self._outputs):
            return [result[port] for port in self._outputs]
        if self.output_names:
            named: dict[str, Any] = {}
            for index, (key, value) in enumerate(result.items()):
                named[_port_name(key, index)] = value
            if all(name in named for name in self.output_names):
                return [named[name] for name in self.output_names]
        return list(result.values())

    def close(self) -> None:
        close = getattr(self._compiled_model, "close", None)
        if callable(close):
            close()


def _unique_strings(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            result.append(text)
    return result


def _device_family(device: str) -> str:
    return device.upper().split(".", 1)[0].split(":", 1)[0]


def _read_openvino_devices(core: Any) -> tuple[str, ...]:
    available = getattr(core, "available_devices", ())
    if callable(available):
        available = available()
    return tuple(_unique_strings(tuple(available or ())))


def _runtime_version(module: Any) -> str | None:
    version = getattr(module, "__version__", None)
    return None if version is None else str(version)


def _make_ort_session_options(module: Any, config: InferenceConfig) -> Any | None:
    options_factory = getattr(module, "SessionOptions", None)
    if options_factory is None:
        return None
    options = options_factory()
    if config.intra_op_threads is not None:
        options.intra_op_num_threads = config.intra_op_threads
    if config.inter_op_threads is not None:
        options.inter_op_num_threads = config.inter_op_threads
    graph_levels = getattr(module, "GraphOptimizationLevel", None)
    if graph_levels is not None:
        level_name = "ORT_ENABLE_ALL" if config.graph_optimization else "ORT_DISABLE_ALL"
        level = getattr(graph_levels, level_name, None)
        if level is not None:
            options.graph_optimization_level = level
    return options


def _ort_node_names(session: Any, method_name: str) -> tuple[str, ...]:
    reader = getattr(session, method_name, None)
    if not callable(reader):
        return ()
    return tuple(str(node.name) for node in reader() if getattr(node, "name", None) is not None)


def _port_name(port: Any, index: int) -> str:
    for attribute in ("get_any_name", "any_name"):
        value = getattr(port, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value:
            return str(value)
    names = getattr(port, "names", None)
    if names:
        return str(sorted(str(name) for name in names)[0])
    if isinstance(port, str):
        return port
    return f"output_{index}"


def _device_for_ort_provider(provider: str, options: Mapping[str, str]) -> str | None:
    if provider == "OpenVINOExecutionProvider":
        return options.get("device_type", "AUTO")
    if provider == "CPUExecutionProvider":
        return "CPU"
    suffix = "ExecutionProvider"
    return provider[: -len(suffix)] if provider.endswith(suffix) else provider


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return float(ordered[index])


__all__ = [
    "BackendKind",
    "DEFAULT_ONNX_PROVIDER_ORDER",
    "InferenceBackendError",
    "InferenceConfig",
    "InferenceMetadata",
    "InferenceTelemetry",
    "InferenceUnavailableError",
    "SharedInferenceBackend",
    "rank_onnx_providers",
    "rank_openvino_devices",
]
