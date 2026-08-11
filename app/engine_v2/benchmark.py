"""Independent performance and accuracy harness for ANPR Engine V2.

The harness deliberately does not enable Engine V2 in production.  Synthetic
mode exercises scheduling, newest-frame replacement, accounting and output
formats; it is always labelled as non-production evidence.  Real accuracy
evidence is accepted only from an operator-verified manifest and an explicit
callable or command adapter.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import platform
import shlex
import statistics
import subprocess
import sys
import time
from typing import Any, Protocol

from .scheduler import LatestOnlyPriorityQueue
from .validator import IranianPlateValidator

try:  # ``resource`` is not available on Windows.
    import resource
except ImportError:  # pragma: no cover - exercised on Windows runners
    resource = None  # type: ignore[assignment]


ACCURACY_MANIFEST_SCHEMA = "bcvision.anpr.accuracy-manifest/v1"
REQUIRED_ACCURACY_CATEGORIES = (
    "clear_plate",
    "night",
    "overexposure",
    "motion_blur",
    "angled_plate",
    "multiple_vehicles",
    "fast_vehicle",
    "partial_dirty_plate",
)


@dataclass(frozen=True, slots=True)
class BenchmarkScenario:
    """A finite, repeatable producer/consumer workload.

    ``nominal_seconds`` controls workload size, not a sleep interval.  Throughput
    rates use observed wall time, while frame timestamps use the nominal clock.
    This keeps CI fast without presenting simulated time as measured throughput.
    """

    name: str
    camera_count: int
    active_cameras: int
    nominal_seconds: float = 5.0
    ticks_per_second: int = 10
    producer_burst: int = 2
    consumer_budget_per_tick: int | None = None
    max_frame_age_ms: float = 250.0
    queue_capacity: int = 128
    matrix: str = "custom"

    def __post_init__(self) -> None:
        if self.camera_count < 1:
            raise ValueError("camera_count must be at least 1")
        if not 0 <= self.active_cameras <= self.camera_count:
            raise ValueError("active_cameras must be between 0 and camera_count")
        if self.nominal_seconds <= 0:
            raise ValueError("nominal_seconds must be greater than zero")
        if self.ticks_per_second < 1:
            raise ValueError("ticks_per_second must be at least 1")
        if self.producer_burst < 1:
            raise ValueError("producer_burst must be at least 1")
        if self.consumer_budget_per_tick is not None and self.consumer_budget_per_tick < 1:
            raise ValueError("consumer_budget_per_tick must be at least 1")
        if self.max_frame_age_ms < 0:
            raise ValueError("max_frame_age_ms cannot be negative")
        if self.queue_capacity < 1:
            raise ValueError("queue_capacity must be at least 1")
        if not self.matrix.strip():
            raise ValueError("matrix cannot be empty")

    @property
    def idle_cameras(self) -> int:
        return self.camera_count - self.active_cameras

    @property
    def ticks(self) -> int:
        return max(1, int(math.ceil(self.nominal_seconds * self.ticks_per_second)))


@dataclass(frozen=True, slots=True)
class BenchmarkFrameJob:
    camera_id: str
    sequence: int
    nominal_timestamp: float
    produced_monotonic: float
    active: bool = True
    payload: Any = None


@dataclass(frozen=True, slots=True)
class AdapterTelemetry:
    detector_inferences: int = 0
    ocr_inferences: int = 0
    plate_events: int = 0
    decode_utilization_percent: float | None = None
    decode_utilization_kind: str = "unavailable"
    decode_utilization_source: str = "unavailable"

    def __post_init__(self) -> None:
        for field_name in ("detector_inferences", "ocr_inferences", "plate_events"):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} cannot be negative")
        value = self.decode_utilization_percent
        if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0):
            raise ValueError("decode_utilization_percent must be null or within 0..100")
        if self.decode_utilization_kind not in {"unavailable", "measured", "estimated"}:
            raise ValueError(
                "decode_utilization_kind must be unavailable, measured, or estimated"
            )
        if value is None and self.decode_utilization_kind != "unavailable":
            raise ValueError("measured/estimated decode utilization requires a numeric value")
        if value is not None and self.decode_utilization_kind == "unavailable":
            raise ValueError("numeric decode utilization must be labelled measured or estimated")
        if value is not None and not self.decode_utilization_source.strip():
            raise ValueError("decode_utilization_source is required when utilization is reported")


class PerformanceAdapter(Protocol):
    adapter_name: str
    evidence_kind: str
    production_evidence: bool

    def process(self, job: BenchmarkFrameJob) -> AdapterTelemetry | Mapping[str, Any] | None: ...


class SyntheticControlPlaneAdapter:
    """Deterministic fake used to validate harness behavior, never model quality."""

    adapter_name = "deterministic-synthetic-control-plane"
    evidence_kind = "synthetic-control-plane"
    production_evidence = False
    decode_utilization_kind = "unavailable"
    decode_utilization_source = "unavailable:synthetic-adapter-has-no-decoder"

    def __init__(self, *, ocr_every: int = 3, event_every: int = 2, cpu_work: int = 0) -> None:
        self.ocr_every = max(1, int(ocr_every))
        self.event_every = max(1, int(event_every))
        self.cpu_work = max(0, int(cpu_work))
        self._ocr_calls = 0
        self._checksum = 0

    def process(self, job: BenchmarkFrameJob) -> AdapterTelemetry:
        # Optional integer work gives smoke benchmarks a measurable CPU signal;
        # it does not pretend to be detector or OCR inference.
        checksum = self._checksum
        for offset in range(self.cpu_work):
            checksum = (checksum * 33 + job.sequence + offset) & 0xFFFFFFFF
        self._checksum = checksum

        ocr_calls = int(job.sequence % self.ocr_every == 0)
        if ocr_calls:
            self._ocr_calls += 1
        events = int(ocr_calls and self._ocr_calls % self.event_every == 0)
        return AdapterTelemetry(
            detector_inferences=1,
            ocr_inferences=ocr_calls,
            plate_events=events,
            decode_utilization_percent=None,
            decode_utilization_source=self.decode_utilization_source,
        )

    def prepare_scenario(self, scenario: BenchmarkScenario) -> None:
        del scenario
        self._ocr_calls = 0
        self._checksum = 0

    def observe_idle(self, camera_id: str, tick: int) -> AdapterTelemetry:
        # Idle cameras deliberately do no AI work in this control-plane fixture.
        del camera_id, tick
        return AdapterTelemetry(
            decode_utilization_source=self.decode_utilization_source,
        )


class CallablePerformanceAdapter:
    """Wrap a Python callable that accepts one :class:`BenchmarkFrameJob`."""

    def __init__(
        self,
        function: Callable[[BenchmarkFrameJob], Any],
        *,
        name: str | None = None,
        evidence_kind: str = "real-callable-adapter",
        production_evidence: bool = False,
    ) -> None:
        self._function = function
        self.adapter_name = name or getattr(function, "__name__", "callable-adapter")
        self.evidence_kind = evidence_kind
        self.production_evidence = bool(production_evidence)
        self.decode_utilization_kind = "unavailable"
        self.decode_utilization_source = "unavailable:adapter-did-not-report"

    def process(self, job: BenchmarkFrameJob) -> AdapterTelemetry:
        return _coerce_telemetry(self._function(job), self.decode_utilization_source)


class CommandPerformanceAdapter:
    """Invoke an explicit JSON-in/JSON-out command once per scheduled job.

    Process startup time is intentionally included in latency and CPU/RAM of the
    parent harness is not claimed to include child-process resource consumption.
    A long-lived callable adapter is preferred for production measurements.
    """

    evidence_kind = "external-command-adapter"
    # Parent-process resource counters exclude the spawned command. The adapter
    # is useful for contract testing, but cannot be resource/promotion evidence.
    production_evidence = False

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        name: str = "command-adapter",
        timeout_seconds: float = 30.0,
    ) -> None:
        self.command = _split_command(command)
        if not self.command:
            raise ValueError("adapter command cannot be empty")
        self.adapter_name = name
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.decode_utilization_kind = "unavailable"
        self.decode_utilization_source = "unavailable:adapter-did-not-report"

    def process(self, job: BenchmarkFrameJob) -> AdapterTelemetry:
        request = {
            "schema": "bcvision.anpr.performance-job/v1",
            "camera_id": job.camera_id,
            "sequence": job.sequence,
            "nominal_timestamp": job.nominal_timestamp,
            "active": job.active,
            "payload": job.payload,
        }
        completed = subprocess.run(
            self.command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
            env={
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"performance adapter exited with {completed.returncode}: "
                f"{completed.stderr.strip()[:500]}"
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError("performance adapter stdout must be one JSON object") from exc
        return _coerce_telemetry(payload, self.decode_utilization_source)


def _coerce_telemetry(value: Any, default_decode_source: str) -> AdapterTelemetry:
    if value is None:
        return AdapterTelemetry(decode_utilization_source=default_decode_source)
    if isinstance(value, AdapterTelemetry):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("performance adapter must return AdapterTelemetry, a mapping, or None")
    return AdapterTelemetry(
        detector_inferences=int(value.get("detector_inferences", 0)),
        ocr_inferences=int(value.get("ocr_inferences", 0)),
        plate_events=int(value.get("plate_events", value.get("events", 0))),
        decode_utilization_percent=(
            None
            if value.get("decode_utilization_percent") is None
            else float(value["decode_utilization_percent"])
        ),
        decode_utilization_kind=str(
            value.get("decode_utilization_kind", "unavailable")
        ),
        decode_utilization_source=str(
            value.get("decode_utilization_source", default_decode_source)
        ),
    )


def _split_command(command: str | Sequence[str]) -> list[str]:
    if not isinstance(command, str):
        tokens = [str(value) for value in command]
    else:
        tokens = shlex.split(command, posix=os.name != "nt")
        if os.name == "nt":
            tokens = [
                token[1:-1]
                if len(token) >= 2 and token[0] == token[-1] and token[0] in {'"', "'"}
                else token
                for token in tokens
            ]
    if not tokens:
        raise ValueError("adapter command cannot be empty")
    return tokens


class _ResourceSampler:
    def __init__(self) -> None:
        self._process = None
        self._psutil = None
        try:
            import psutil  # type: ignore[import-not-found]

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except (ImportError, OSError, RuntimeError):
            pass
        except Exception:
            # Some containerized runners expose a PID namespace that psutil
            # cannot resolve and raise a psutil-specific Error subclass.
            self._psutil = None
            self._process = None
        self.started_wall = 0.0
        self.started_process_cpu = 0.0
        self.started_psutil_cpu: float | None = None
        self.max_psutil_rss_bytes = 0
        self.max_resource_rss_bytes = 0
        self.resource_sampling_warnings: list[str] = []

    def _psutil_cpu_time(self) -> float | None:
        if self._process is None:
            return None
        try:
            values = self._process.cpu_times()
            return float(values.user + values.system)
        except Exception:
            self.resource_sampling_warnings.append("psutil_cpu_read_failed")
            return None

    def _psutil_rss_bytes(self) -> int:
        if self._process is None:
            return 0
        try:
            return max(0, int(self._process.memory_info().rss))
        except Exception:
            self.resource_sampling_warnings.append("psutil_rss_read_failed")
            return 0

    @staticmethod
    def _resource_rss_bytes() -> int:
        if resource is None:
            return 0
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.
        return value if sys.platform == "darwin" else value * 1024

    @staticmethod
    def _psutil_cpu_is_sane(
        psutil_delta: float | None,
        process_time_delta: float,
        wall: float,
    ) -> bool:
        if psutil_delta is None or not math.isfinite(psutil_delta) or psutil_delta < 0:
            return False
        logical_cpus = max(1, int(os.cpu_count() or 1))
        if psutil_delta > wall * logical_cpus * 1.5 + 0.05:
            return False
        if process_time_delta > 0 and psutil_delta == 0:
            return False
        if abs(psutil_delta - process_time_delta) > max(
            0.005,
            process_time_delta * 0.75,
        ):
            return False
        # Both counters measure the current process. This generous ratio still
        # catches PID namespace aliases that expose a static/unrelated process.
        if process_time_delta > 0.01:
            if psutil_delta < process_time_delta * 0.25:
                return False
            if psutil_delta > process_time_delta * 4.0 + 0.05:
                return False
        return True

    @staticmethod
    def _psutil_rss_is_sane(psutil_rss: int, resource_rss: int) -> bool:
        if psutil_rss <= 0:
            return False
        if resource_rss <= 0:
            return True
        # ru_maxrss is a lifetime peak, so current RSS can be lower. An extreme
        # mismatch is safer to report as a conservative maxrss fallback.
        ratio = psutil_rss / float(resource_rss)
        return 0.25 <= ratio <= 4.0

    def start(self) -> None:
        self.started_wall = time.perf_counter()
        self.started_process_cpu = float(time.process_time())
        self.started_psutil_cpu = self._psutil_cpu_time()
        self.sample_memory()

    def sample_memory(self) -> None:
        self.max_psutil_rss_bytes = max(
            self.max_psutil_rss_bytes,
            self._psutil_rss_bytes(),
        )
        self.max_resource_rss_bytes = max(
            self.max_resource_rss_bytes,
            self._resource_rss_bytes(),
        )

    def finish(self) -> dict[str, Any]:
        self.sample_memory()
        wall = max(time.perf_counter() - self.started_wall, 1e-9)
        process_time_delta = max(
            0.0,
            float(time.process_time()) - self.started_process_cpu,
        )
        finished_psutil_cpu = self._psutil_cpu_time()
        psutil_delta = (
            None
            if self.started_psutil_cpu is None or finished_psutil_cpu is None
            else finished_psutil_cpu - self.started_psutil_cpu
        )
        if self._psutil_cpu_is_sane(psutil_delta, process_time_delta, wall):
            cpu = float(psutil_delta)
            cpu_counter_source = "psutil.Process.cpu_times"
        else:
            cpu = process_time_delta
            cpu_counter_source = "time.process_time"
            if self._process is not None:
                self.resource_sampling_warnings.append(
                    "psutil_cpu_rejected_inconsistent_with_process_time"
                )

        if self._psutil_rss_is_sane(
            self.max_psutil_rss_bytes,
            self.max_resource_rss_bytes,
        ):
            max_rss_bytes = self.max_psutil_rss_bytes
            ram_source = "psutil.Process.rss"
        else:
            max_rss_bytes = self.max_resource_rss_bytes
            ram_source = (
                "resource.getrusage.maxrss"
                if resource is not None
                else "unavailable"
            )
            if self._process is not None:
                self.resource_sampling_warnings.append(
                    "psutil_rss_rejected_inconsistent_with_resource_maxrss"
                )

        process_cpu_percent = 100.0 * cpu / wall
        logical_cpus = max(1, int(os.cpu_count() or 1))
        return {
            "observed_wall_seconds": wall,
            "cpu_percent": process_cpu_percent / logical_cpus,
            "process_cpu_percent": process_cpu_percent,
            "cpu_percent_source": cpu_counter_source + "/wall-time/logical-cpu-count",
            "ram_mb": max_rss_bytes / (1024.0 * 1024.0) if max_rss_bytes else None,
            "ram_source": ram_source,
            "resource_sampling_warnings": sorted(set(self.resource_sampling_warnings)),
        }


def _percentile_95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return ordered[index]


def _adapter_property(adapter: Any, name: str, default: Any) -> Any:
    return getattr(adapter, name, default)


def _valid_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _performance_evidence_status(adapter: Any) -> dict[str, Any]:
    """Fail closed unless a real in-process adapter supplies verifiable files."""

    requested = bool(_adapter_property(adapter, "production_evidence", False))
    metadata = _adapter_property(adapter, "evidence_metadata", None)
    reasons: list[str] = []
    verified_files: list[dict[str, Any]] = []
    if not requested:
        reasons.append("adapter-did-not-request-production-evidence")
    if not isinstance(metadata, Mapping):
        reasons.append("evidence-metadata-missing")
        metadata = {}
    if str(metadata.get("resource_scope", "")) != "current-process":
        reasons.append("resource-scope-must-be-current-process")
    if metadata.get("uses_child_processes") is not False:
        reasons.append("child-process-resource-coverage-not-proven")
    if not str(metadata.get("execution_provider", "")).strip():
        reasons.append("execution-provider-missing")
    for group_name in ("input_files", "model_files"):
        entries = metadata.get(group_name)
        if not isinstance(entries, list) or not entries:
            reasons.append(f"{group_name}-missing")
            continue
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping):
                reasons.append(f"{group_name}-{index}-invalid")
                continue
            path = Path(str(entry.get("path", ""))).resolve()
            expected = str(entry.get("sha256", "")).lower()
            if not path.is_file() or not _valid_sha256(expected):
                reasons.append(f"{group_name}-{index}-identity-invalid")
                continue
            actual = _file_sha256(path)
            if actual != expected:
                reasons.append(f"{group_name}-{index}-sha256-mismatch")
                continue
            verified_files.append(
                {
                    "group": group_name,
                    "path": str(path),
                    "sha256": actual,
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "requested": requested,
        "valid": requested and not reasons,
        "reasons": sorted(set(reasons)),
        "execution_provider": str(metadata.get("execution_provider", "")),
        "resource_scope": str(metadata.get("resource_scope", "")),
        "verified_files": verified_files,
    }


def run_performance_scenario(
    scenario: BenchmarkScenario,
    adapter: PerformanceAdapter,
    *,
    evidence_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one finite scheduling workload and return JSON-safe measured metrics."""

    queue: LatestOnlyPriorityQueue[BenchmarkFrameJob] = LatestOnlyPriorityQueue(
        max_items=scenario.queue_capacity
    )
    sampler = _ResourceSampler()
    queue_depths: list[int] = [0]
    latencies_ms: list[float] = []
    decode_values: list[float] = []
    decode_kinds: set[str] = set()
    decode_sources: set[str] = set()
    sequence_by_camera = [0] * scenario.camera_count
    camera_metrics: dict[str, dict[str, Any]] = {
        f"camera-{index + 1:02d}": {
            "active": index < scenario.active_cameras,
            "produced_frames": 0,
            "processed_jobs": 0,
            "latest_frame_replacements": 0,
            "expired_stale_frames": 0,
            "detector_inferences": 0,
            "ocr_inferences": 0,
            "plate_events": 0,
            "latencies_ms": [],
        }
        for index in range(scenario.camera_count)
    }
    pending_camera_ids: set[str] = set()
    detector_inferences = 0
    ocr_inferences = 0
    plate_events = 0
    idle_detector_inferences = 0
    idle_ocr_inferences = 0
    expired_frames = 0
    produced_frames = 0
    idle_gate_observations = 0
    idle_observation_wall_seconds = 0.0
    idle_observation_cpu_seconds = 0.0
    processed_jobs = 0

    def account(
        telemetry: AdapterTelemetry,
        *,
        camera_id: str,
        idle: bool = False,
    ) -> None:
        nonlocal detector_inferences, ocr_inferences, plate_events
        nonlocal idle_detector_inferences, idle_ocr_inferences
        detector_inferences += telemetry.detector_inferences
        ocr_inferences += telemetry.ocr_inferences
        plate_events += telemetry.plate_events
        camera_row = camera_metrics[camera_id]
        camera_row["detector_inferences"] += telemetry.detector_inferences
        camera_row["ocr_inferences"] += telemetry.ocr_inferences
        camera_row["plate_events"] += telemetry.plate_events
        if idle:
            idle_detector_inferences += telemetry.detector_inferences
            idle_ocr_inferences += telemetry.ocr_inferences
        if telemetry.decode_utilization_percent is not None:
            decode_values.append(float(telemetry.decode_utilization_percent))
            decode_kinds.add(telemetry.decode_utilization_kind)
            decode_sources.add(telemetry.decode_utilization_source)

    def consume_one() -> bool:
        nonlocal expired_frames, processed_jobs
        job = queue.pop()
        queue_depths.append(len(queue))
        if job is None:
            return False
        pending_camera_ids.discard(job.camera_id)
        age_ms = (time.perf_counter() - job.produced_monotonic) * 1000.0
        if age_ms > scenario.max_frame_age_ms:
            expired_frames += 1
            camera_metrics[job.camera_id]["expired_stale_frames"] += 1
            return True
        telemetry = _coerce_telemetry(
            adapter.process(job),
            str(
                _adapter_property(
                    adapter,
                    "decode_utilization_source",
                    "unavailable:adapter-did-not-report",
                )
            ),
        )
        finished = time.perf_counter()
        account(telemetry, camera_id=job.camera_id)
        latency_ms = (finished - job.produced_monotonic) * 1000.0
        latencies_ms.append(latency_ms)
        camera_metrics[job.camera_id]["latencies_ms"].append(latency_ms)
        camera_metrics[job.camera_id]["processed_jobs"] += 1
        processed_jobs += 1
        return True

    camera_descriptors = tuple(
        {
            "camera_id": f"camera-{index + 1:02d}",
            "active": index < scenario.active_cameras,
        }
        for index in range(scenario.camera_count)
    )
    configure = getattr(adapter, "configure_scenario", None)
    if callable(configure):
        configure(scenario, camera_descriptors)
    prepare = getattr(adapter, "prepare_scenario", None)
    if callable(prepare):
        prepare(scenario)
    start_scenario = getattr(adapter, "start_scenario", None)
    stop_scenario = getattr(adapter, "stop_scenario", None)
    stream_lifecycle_included = all(
        callable(hook) for hook in (configure, start_scenario, stop_scenario)
    )
    sampler.start()
    if callable(start_scenario):
        start_scenario(scenario)
    for tick in range(scenario.ticks):
        nominal_timestamp = tick / float(scenario.ticks_per_second)
        # Rotate producer order so a constrained consumer budget cannot always
        # favor the first camera IDs.
        active_order = (
            [
                (tick + offset) % scenario.active_cameras
                for offset in range(scenario.active_cameras)
            ]
            if scenario.active_cameras
            else []
        )
        idle_order = list(range(scenario.active_cameras, scenario.camera_count))
        for camera_index in active_order + idle_order:
            camera_id = f"camera-{camera_index + 1:02d}"
            if camera_index >= scenario.active_cameras:
                idle_wall_started = time.perf_counter()
                idle_cpu_started = time.process_time()
                idle_gate_observations += 1
                observe_idle = getattr(adapter, "observe_idle", None)
                if callable(observe_idle):
                    account(
                        _coerce_telemetry(
                            observe_idle(camera_id, tick),
                            str(
                                _adapter_property(
                                    adapter,
                                    "decode_utilization_source",
                                    "unavailable:adapter-did-not-report",
                                )
                            ),
                        ),
                        camera_id=camera_id,
                        idle=True,
                    )
                idle_observation_wall_seconds += time.perf_counter() - idle_wall_started
                idle_observation_cpu_seconds += time.process_time() - idle_cpu_started
                continue
            for _ in range(scenario.producer_burst):
                sequence_by_camera[camera_index] += 1
                produced_frames += 1
                camera_metrics[camera_id]["produced_frames"] += 1
                if camera_id in pending_camera_ids:
                    camera_metrics[camera_id]["latest_frame_replacements"] += 1
                submitted_before = int(queue.stats.submitted)
                queue.submit(
                    camera_id,
                    BenchmarkFrameJob(
                        camera_id=camera_id,
                        sequence=sequence_by_camera[camera_index],
                        nominal_timestamp=nominal_timestamp,
                        produced_monotonic=time.perf_counter(),
                    ),
                    priority=10,
                )
                if int(queue.stats.submitted) > submitted_before:
                    pending_camera_ids.add(camera_id)
                queue_depths.append(len(queue))

        budget = scenario.consumer_budget_per_tick
        if budget is None:
            budget = max(1, scenario.active_cameras // 2) if scenario.active_cameras else 1
        for _ in range(budget):
            if not consume_one():
                break
        sampler.sample_memory()

    # Drain only the newest remaining frame per camera. Superseded heap entries
    # are ignored by LatestOnlyPriorityQueue.pop().
    while consume_one():
        pass
    if callable(stop_scenario):
        stop_scenario(scenario)
    measured = sampler.finish()
    wall = float(measured["observed_wall_seconds"])
    replaced_frames = int(queue.stats.replaced)
    stale_drops = replaced_frames + expired_frames
    if decode_values:
        decode_utilization: float | None = statistics.fmean(decode_values)
        decode_kind = (
            next(iter(decode_kinds))
            if len(decode_kinds) == 1
            else "mixed:" + ",".join(sorted(decode_kinds))
        )
        decode_source = (
            next(iter(decode_sources))
            if len(decode_sources) == 1
            else "mixed:" + ",".join(sorted(decode_sources))
        )
    else:
        decode_utilization = None
        decode_kind = str(
            _adapter_property(adapter, "decode_utilization_kind", "unavailable")
        )
        decode_source = (
            next(iter(decode_sources))
            if len(decode_sources) == 1
            else str(
                _adapter_property(
                    adapter,
                    "decode_utilization_source",
                    "unavailable:adapter-did-not-report",
                )
            )
        )

    evidence = dict(evidence_status or _performance_evidence_status(adapter))
    evidence["reasons"] = list(evidence.get("reasons", []))
    if scenario.idle_cameras and not stream_lifecycle_included:
        evidence["valid"] = False
        evidence["reasons"] = sorted(
            set(evidence["reasons"]) | {"idle-stream-lifecycle-not-instrumented"}
        )
    per_camera: dict[str, dict[str, Any]] = {}
    for camera_id, raw in camera_metrics.items():
        camera_latencies = raw.pop("latencies_ms")
        per_camera[camera_id] = {
            **raw,
            "average_latency_ms": (
                round(statistics.fmean(camera_latencies), 4) if camera_latencies else 0.0
            ),
            "p95_latency_ms": round(_percentile_95(camera_latencies), 4),
        }
    active_processed = [
        int(row["processed_jobs"])
        for row in per_camera.values()
        if row["active"]
    ]
    processed_sum = sum(active_processed)
    processed_square_sum = sum(value * value for value in active_processed)
    fairness_index = (
        (processed_sum * processed_sum)
        / (len(active_processed) * processed_square_sum)
        if active_processed and processed_square_sum
        else 1.0
    )
    capacity_evictions = int(getattr(queue.stats, "evicted", 0))
    capacity_rejections = int(queue.stats.dropped)
    total_capacity_drops = capacity_evictions + capacity_rejections
    total_drops = stale_drops + total_capacity_drops
    return {
        "scenario": scenario.name,
        "matrix": scenario.matrix,
        "camera_count": scenario.camera_count,
        "active_cameras": scenario.active_cameras,
        "idle_cameras": scenario.idle_cameras,
        "nominal_workload_seconds": scenario.nominal_seconds,
        "scenario_config": {
            "ticks_per_second": scenario.ticks_per_second,
            "ticks": scenario.ticks,
            "producer_burst": scenario.producer_burst,
            "consumer_budget_per_tick": scenario.consumer_budget_per_tick,
            "max_frame_age_ms": scenario.max_frame_age_ms,
            "queue_capacity": scenario.queue_capacity,
        },
        "observed_wall_seconds": round(wall, 6),
        "throughput_time_source": "observed-wall-time",
        "adapter_name": str(_adapter_property(adapter, "adapter_name", type(adapter).__name__)),
        "evidence_kind": str(_adapter_property(adapter, "evidence_kind", "unspecified-adapter")),
        "production_evidence": bool(evidence["valid"]),
        "evidence_validation": evidence,
        "cpu_percent": round(float(measured["cpu_percent"]), 4),
        "process_cpu_percent": round(float(measured["process_cpu_percent"]), 4),
        "cpu_percent_source": measured["cpu_percent_source"],
        "ram_mb": None if measured["ram_mb"] is None else round(float(measured["ram_mb"]), 4),
        "ram_source": measured["ram_source"],
        "resource_sampling_warnings": measured["resource_sampling_warnings"],
        "resource_scope": "current Python process; child processes excluded",
        "decode_utilization_percent": (
            None if decode_utilization is None else round(decode_utilization, 4)
        ),
        "decode_utilization_kind": decode_kind,
        "decode_utilization_source": decode_source,
        "decode_utilization_aggregation": "arithmetic mean of adapter job samples",
        "detector_inferences": detector_inferences,
        "detector_inferences_per_second": round(detector_inferences / wall, 4),
        "ocr_inferences": ocr_inferences,
        "ocr_inferences_per_second": round(ocr_inferences / wall, 4),
        "idle_detector_inferences": idle_detector_inferences,
        "idle_ocr_inferences": idle_ocr_inferences,
        "plate_events": plate_events,
        "plate_events_per_second": round(plate_events / wall, 4),
        "queue_depth_average": round(statistics.fmean(queue_depths), 4),
        "queue_depth_max": max(queue_depths),
        "queue_depth_sampling": "event-sampled",
        "produced_frames": produced_frames,
        "submitted_frames": int(queue.stats.submitted),
        "processed_jobs": processed_jobs,
        "latest_frame_replacements": replaced_frames,
        "expired_stale_frames": expired_frames,
        "dropped_stale_frames": stale_drops,
        "dropped_capacity_frames": total_capacity_drops,
        "evicted_capacity_frames": capacity_evictions,
        "rejected_capacity_frames": capacity_rejections,
        "total_drop_ratio": round(total_drops / max(1, produced_frames), 6),
        "deadline_miss_ratio": round(expired_frames / max(1, produced_frames), 6),
        "average_latency_ms": round(statistics.fmean(latencies_ms), 4) if latencies_ms else 0.0,
        "p95_latency_ms": round(_percentile_95(latencies_ms), 4),
        "latency_scope": "processed-survivors-only",
        "idle_gate_observations": idle_gate_observations,
        "idle_observation_wall_seconds": round(idle_observation_wall_seconds, 6),
        "idle_observation_cpu_seconds": round(idle_observation_cpu_seconds, 6),
        "idle_observation_system_cpu_percent": round(
            100.0
            * idle_observation_cpu_seconds
            / max(wall, 1e-9)
            / max(1, int(os.cpu_count() or 1)),
            6,
        ),
        "idle_observation_scope": (
            "synchronous observe_idle callback and scheduler bookkeeping only; "
            "background stream threads are represented by fixed-active scenario deltas"
        ),
        "idle_cost_scope": (
            "adapter-configure/start/stop plus observe_idle"
            if stream_lifecycle_included
            else "scheduler loop and optional observe_idle only; RTSP/decode lifecycle not measured"
        ),
        "idle_stream_lifecycle_included": stream_lifecycle_included,
        "scheduler_fairness": {
            "jain_index": round(fairness_index, 6),
            "processed_job_spread": (
                max(active_processed) - min(active_processed) if active_processed else 0
            ),
            "starved_active_cameras": sum(value == 0 for value in active_processed),
        },
        "per_camera": per_camera,
    }


STANDARD_CAMERA_COUNTS = (1, 4, 8, 16)


def _standard_camera_counts(include_32: bool) -> tuple[int, ...]:
    return STANDARD_CAMERA_COUNTS + ((32,) if include_32 else ())


def default_camera_scenarios(
    *,
    include_32: bool = False,
    active_cameras: int = 1,
    nominal_seconds: float = 5.0,
    ticks_per_second: int = 10,
    producer_burst: int = 2,
    consumer_budget_per_tick: int | None = None,
    max_frame_age_ms: float = 250.0,
) -> list[BenchmarkScenario]:
    """Return 1/4/8/16 camera cases with a fixed active count.

    Keeping active camera count fixed isolates the incremental cost of idle
    cameras. Callers can run an additional suite with another active count or
    ratio to characterize busy-site scaling.
    """

    scenarios = []
    for count in _standard_camera_counts(include_32):
        active = min(max(0, int(active_cameras)), count)
        idle = count - active
        scenarios.append(
            BenchmarkScenario(
                name=f"cameras-{count:02d}-active-{active:02d}-idle-{idle:02d}",
                camera_count=count,
                active_cameras=active,
                nominal_seconds=nominal_seconds,
                ticks_per_second=ticks_per_second,
                producer_burst=producer_burst,
                consumer_budget_per_tick=consumer_budget_per_tick,
                max_frame_age_ms=max_frame_age_ms,
                matrix="fixed-active-idle-scaling",
            )
        )
    return scenarios


def all_active_camera_scenarios(
    *,
    include_32: bool = False,
    nominal_seconds: float = 5.0,
    ticks_per_second: int = 10,
    producer_burst: int = 2,
    consumer_budget_per_tick: int | None = None,
    max_frame_age_ms: float = 250.0,
) -> list[BenchmarkScenario]:
    """Return the standard all-active busy scaling matrix."""

    return [
        BenchmarkScenario(
            name=f"busy-cameras-{count:02d}-active-{count:02d}",
            camera_count=count,
            active_cameras=count,
            nominal_seconds=nominal_seconds,
            ticks_per_second=ticks_per_second,
            producer_burst=producer_burst,
            consumer_budget_per_tick=consumer_budget_per_tick,
            max_frame_age_ms=max_frame_age_ms,
            matrix="all-active-busy-scaling",
        )
        for count in _standard_camera_counts(include_32)
    ]


def standard_camera_matrices(
    *,
    include_32: bool = False,
    fixed_active_cameras: int = 1,
    nominal_seconds: float = 5.0,
    ticks_per_second: int = 10,
    producer_burst: int = 2,
    consumer_budget_per_tick: int | None = None,
    max_frame_age_ms: float = 250.0,
) -> dict[str, list[BenchmarkScenario]]:
    """Build fixed-active idle and all-active busy matrices together."""

    common = {
        "include_32": include_32,
        "nominal_seconds": nominal_seconds,
        "ticks_per_second": ticks_per_second,
        "producer_burst": producer_burst,
        "consumer_budget_per_tick": consumer_budget_per_tick,
        "max_frame_age_ms": max_frame_age_ms,
    }
    return {
        "fixed_active_idle_scaling": default_camera_scenarios(
            active_cameras=fixed_active_cameras,
            **common,
        ),
        "all_active_busy_scaling": all_active_camera_scenarios(**common),
    }


def _system_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "pid": os.getpid(),
    }


def _idle_scaling_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    groups: dict[int, list[Mapping[str, Any]]] = {}
    for row in results:
        groups.setdefault(int(row["active_cameras"]), []).append(row)
    comparisons = []
    for active, rows in sorted(groups.items()):
        ordered = sorted(rows, key=lambda item: int(item["camera_count"]))
        if len(ordered) < 2:
            continue
        baseline = ordered[0]
        baseline_cpu = float(baseline["cpu_percent"])
        baseline_ram = baseline["ram_mb"]
        for row in ordered[1:]:
            comparisons.append(
                {
                    "active_cameras": active,
                    "baseline_camera_count": int(baseline["camera_count"]),
                    "camera_count": int(row["camera_count"]),
                    "additional_idle_cameras": int(row["idle_cameras"]) - int(baseline["idle_cameras"]),
                    "cpu_percent_delta": round(float(row["cpu_percent"]) - baseline_cpu, 4),
                    "ram_mb_delta": (
                        None
                        if baseline_ram is None or row["ram_mb"] is None
                        else round(float(row["ram_mb"]) - float(baseline_ram), 4)
                    ),
                    "detector_inference_delta": int(row["detector_inferences"])
                    - int(baseline["detector_inferences"]),
                    "idle_detector_inference_delta": int(row["idle_detector_inferences"])
                    - int(baseline["idle_detector_inferences"]),
                }
            )
    return {
        "method": (
            "fixed-active-camera comparison; deltas are measured within each scenario's "
            "reported idle_cost_scope and are not extrapolated"
        ),
        "comparisons": comparisons,
    }


def _busy_scaling_summary(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    all_active = sorted(
        (
            row
            for row in results
            if int(row["active_cameras"]) == int(row["camera_count"])
        ),
        key=lambda item: int(item["camera_count"]),
    )
    comparisons = []
    if len(all_active) >= 2:
        baseline = all_active[0]
        baseline_ram = baseline["ram_mb"]
        for row in all_active[1:]:
            comparisons.append(
                {
                    "baseline_camera_count": int(baseline["camera_count"]),
                    "camera_count": int(row["camera_count"]),
                    "additional_active_cameras": int(row["active_cameras"])
                    - int(baseline["active_cameras"]),
                    "cpu_percent_delta": round(
                        float(row["cpu_percent"]) - float(baseline["cpu_percent"]),
                        4,
                    ),
                    "ram_mb_delta": (
                        None
                        if baseline_ram is None or row["ram_mb"] is None
                        else round(float(row["ram_mb"]) - float(baseline_ram), 4)
                    ),
                    "detector_inferences_per_second_delta": round(
                        float(row["detector_inferences_per_second"])
                        - float(baseline["detector_inferences_per_second"]),
                        4,
                    ),
                    "ocr_inferences_per_second_delta": round(
                        float(row["ocr_inferences_per_second"])
                        - float(baseline["ocr_inferences_per_second"]),
                        4,
                    ),
                    "queue_depth_max_delta": int(row["queue_depth_max"])
                    - int(baseline["queue_depth_max"]),
                    "dropped_stale_frames_delta": int(row["dropped_stale_frames"])
                    - int(baseline["dropped_stale_frames"]),
                    "average_latency_ms_delta": round(
                        float(row["average_latency_ms"])
                        - float(baseline["average_latency_ms"]),
                        4,
                    ),
                    "p95_latency_ms_delta": round(
                        float(row["p95_latency_ms"])
                        - float(baseline["p95_latency_ms"]),
                        4,
                    ),
                    "plate_events_per_second_delta": round(
                        float(row["plate_events_per_second"])
                        - float(baseline["plate_events_per_second"]),
                        4,
                    ),
                }
            )
    return {
        "method": (
            "all-active-camera comparison against the smallest camera count; "
            "deltas are measured and not extrapolated"
        ),
        "comparisons": comparisons,
    }


def _close_performance_adapter(adapter: PerformanceAdapter) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _performance_report(
    results: Sequence[Mapping[str, Any]],
    adapter: PerformanceAdapter,
    *,
    matrices: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    rows = [dict(row) for row in results]
    matrix_rows = {
        name: [dict(row) for row in values]
        for name, values in (matrices or {}).items()
    }
    idle_rows = matrix_rows.get("fixed_active_idle_scaling", rows)
    busy_rows = matrix_rows.get("all_active_busy_scaling", rows)
    production_evidence = all(bool(row["production_evidence"]) for row in rows)
    return {
        "schema": "bcvision.anpr.performance-report/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": _system_metadata(),
        "adapter": {
            "name": str(_adapter_property(adapter, "adapter_name", type(adapter).__name__)),
            "evidence_kind": str(
                _adapter_property(adapter, "evidence_kind", "unspecified-adapter")
            ),
            "production_evidence": production_evidence,
        },
        "production_decision_allowed": False,
        "production_decision_reason": (
            "Harness reports evidence only and never switches the production engine."
            if production_evidence
            else "Synthetic/non-production adapter results cannot justify an engine switch."
        ),
        "scenarios": rows,
        "performance_matrices": matrix_rows,
        "idle_camera_scaling": _idle_scaling_summary(idle_rows),
        "busy_camera_scaling": _busy_scaling_summary(busy_rows),
    }


def run_performance_suite(
    scenarios: Sequence[BenchmarkScenario],
    adapter: PerformanceAdapter,
) -> dict[str, Any]:
    if not scenarios:
        raise ValueError("at least one performance scenario is required")
    evidence = _performance_evidence_status(adapter)
    try:
        results = [
            run_performance_scenario(scenario, adapter, evidence_status=evidence)
            for scenario in scenarios
        ]
    finally:
        _close_performance_adapter(adapter)
    return _performance_report(results, adapter)


def run_standard_performance_matrices(
    adapter: PerformanceAdapter,
    *,
    include_32: bool = False,
    fixed_active_cameras: int = 1,
    nominal_seconds: float = 5.0,
    ticks_per_second: int = 10,
    producer_burst: int = 2,
    consumer_budget_per_tick: int | None = None,
    max_frame_age_ms: float = 250.0,
) -> dict[str, Any]:
    """Run the standard idle and busy matrices with one shared adapter lifecycle."""

    scenario_matrices = standard_camera_matrices(
        include_32=include_32,
        fixed_active_cameras=fixed_active_cameras,
        nominal_seconds=nominal_seconds,
        ticks_per_second=ticks_per_second,
        producer_burst=producer_burst,
        consumer_budget_per_tick=consumer_budget_per_tick,
        max_frame_age_ms=max_frame_age_ms,
    )
    evidence = _performance_evidence_status(adapter)
    results_by_matrix: dict[str, list[dict[str, Any]]] = {}
    try:
        for matrix_name, scenarios in scenario_matrices.items():
            results_by_matrix[matrix_name] = [
                run_performance_scenario(
                    scenario,
                    adapter,
                    evidence_status=evidence,
                )
                for scenario in scenarios
            ]
    finally:
        _close_performance_adapter(adapter)
    flattened = [
        row
        for matrix_name in (
            "fixed_active_idle_scaling",
            "all_active_busy_scaling",
        )
        for row in results_by_matrix[matrix_name]
    ]
    return _performance_report(
        flattened,
        adapter,
        matrices=results_by_matrix,
    )


PERFORMANCE_CSV_FIELDS = (
    "scenario",
    "matrix",
    "camera_count",
    "active_cameras",
    "idle_cameras",
    "nominal_workload_seconds",
    "observed_wall_seconds",
    "adapter_name",
    "evidence_kind",
    "production_evidence",
    "cpu_percent",
    "process_cpu_percent",
    "cpu_percent_source",
    "ram_mb",
    "ram_source",
    "resource_sampling_warnings",
    "decode_utilization_percent",
    "decode_utilization_kind",
    "decode_utilization_source",
    "detector_inferences_per_second",
    "ocr_inferences_per_second",
    "queue_depth_average",
    "queue_depth_max",
    "dropped_stale_frames",
    "latest_frame_replacements",
    "expired_stale_frames",
    "average_latency_ms",
    "p95_latency_ms",
    "plate_events_per_second",
    "idle_detector_inferences",
    "idle_ocr_inferences",
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_performance_outputs(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    _write_json(Path(json_path), report)
    rows = report.get("scenarios")
    if not isinstance(rows, Sequence):
        raise ValueError("performance report has no scenarios")
    _write_csv(Path(csv_path), rows, PERFORMANCE_CSV_FIELDS)


@dataclass(frozen=True, slots=True)
class AccuracyManifest:
    path: Path
    dataset_id: str
    samples: tuple[dict[str, Any], ...]
    sha256: str
    negative_sample_count: int = 0
    verified_media_sha256s: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class AccuracyPrediction:
    plate: str = ""
    confidence: float | None = None
    accepted: bool | None = None
    events: tuple[Mapping[str, Any], ...] | None = None
    details: Mapping[str, Any] | None = None


class AccuracyAdapter(Protocol):
    adapter_name: str

    def predict(self, sample: Mapping[str, Any]) -> AccuracyPrediction | Mapping[str, Any] | str | None: ...


_GROUND_TRUTH_VALIDATOR = IranianPlateValidator()


def normalize_plate_text(value: Any) -> str:
    return _GROUND_TRUTH_VALIDATOR.normalize(str(value or ""))


def _verified_ground_truth_plate(value: Any, *, context: str) -> str:
    validation = _GROUND_TRUTH_VALIDATOR.validate(str(value or ""))
    if not validation.valid:
        raise ValueError(
            f"{context} is not a structurally valid Iranian plate: {validation.reason}"
        )
    return validation.normalized


def _manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_accuracy_manifest(
    path: str | Path,
    *,
    require_all_categories: bool = True,
    require_input_files: bool = True,
    require_negative_sample: bool = True,
) -> AccuracyManifest:
    manifest_path = Path(path).resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid accuracy manifest JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("accuracy manifest root must be an object")
    if payload.get("schema") != ACCURACY_MANIFEST_SCHEMA:
        raise ValueError(f"manifest schema must be {ACCURACY_MANIFEST_SCHEMA!r}")
    if payload.get("template") is True:
        raise ValueError("template manifests cannot be used as benchmark evidence")
    dataset_id = str(payload.get("dataset_id", "")).strip()
    if not dataset_id:
        raise ValueError("manifest dataset_id is required")
    raw_samples = payload.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("manifest samples must be a non-empty array")

    samples: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    categories: set[str] = set()
    readable_events_by_category = {
        category: 0 for category in REQUIRED_ACCURACY_CATEGORIES
    }
    has_multi_vehicle_event_set = False
    negative_sample_count = 0
    verified_media_sha256s: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_samples):
        if not isinstance(raw, Mapping):
            raise ValueError(f"sample {index} must be an object")
        if raw.get("enabled", True) is not True:
            continue
        sample = dict(raw)
        identifier = str(sample.get("id", "")).strip()
        if not identifier or identifier in identifiers:
            raise ValueError(f"sample {index} has a missing or duplicate id")
        identifiers.add(identifier)
        category = str(sample.get("category", "")).strip()
        if category not in REQUIRED_ACCURACY_CATEGORIES:
            raise ValueError(f"sample {identifier!r} has unsupported category {category!r}")
        categories.add(category)
        if sample.get("label_status") != "verified":
            raise ValueError(f"sample {identifier!r} must have label_status='verified'")
        has_single_label = "expected_plate" in sample
        has_event_labels = "expected_events" in sample
        if has_single_label == has_event_labels:
            raise ValueError(
                f"sample {identifier!r} must contain exactly one of expected_plate or expected_events"
            )
        expected_events: list[dict[str, Any]] = []
        if has_event_labels:
            raw_events = sample["expected_events"]
            if not isinstance(raw_events, list) or not raw_events:
                raise ValueError(f"sample {identifier!r} expected_events must be non-empty")
            for event_index, raw_event in enumerate(raw_events):
                if not isinstance(raw_event, Mapping):
                    raise ValueError(
                        f"sample {identifier!r} expected event {event_index} must be an object"
                    )
                plate = _verified_ground_truth_plate(
                    raw_event.get("plate"),
                    context=f"sample {identifier!r} expected event {event_index} plate",
                )
                event = dict(raw_event)
                event["plate"] = plate
                start_ms = event.get("start_ms")
                end_ms = event.get("end_ms")
                if start_ms is not None:
                    start_ms = float(start_ms)
                    if not math.isfinite(start_ms) or start_ms < 0:
                        raise ValueError(
                            f"sample {identifier!r} expected event {event_index} has invalid start_ms"
                        )
                    event["start_ms"] = start_ms
                if end_ms is not None:
                    end_ms = float(end_ms)
                    if not math.isfinite(end_ms) or end_ms < 0:
                        raise ValueError(
                            f"sample {identifier!r} expected event {event_index} has invalid end_ms"
                        )
                    event["end_ms"] = end_ms
                if start_ms is not None and end_ms is not None and end_ms < start_ms:
                    raise ValueError(
                        f"sample {identifier!r} expected event {event_index} ends before it starts"
                    )
                expected_events.append(event)
        elif sample["expected_plate"] is not None:
            plate = _verified_ground_truth_plate(
                sample["expected_plate"],
                context=f"sample {identifier!r} expected_plate",
            )
            expected_events.append({"plate": plate})
        sample["_expected_events"] = expected_events
        if not expected_events:
            negative_sample_count += 1
        readable_events_by_category[category] += len(expected_events)
        if category == "multiple_vehicles" and len(expected_events) >= 2:
            has_multi_vehicle_event_set = True
        raw_input_value = sample.get("input")
        if not isinstance(raw_input_value, Mapping):
            raise ValueError(f"sample {identifier!r} input must be an object")
        input_value = dict(raw_input_value)
        input_path = str(input_value.get("path", "")).strip()
        if not input_path:
            raise ValueError(f"sample {identifier!r} input.path is required")
        expected_media_sha256 = input_value.get("sha256")
        if expected_media_sha256 is not None:
            expected_media_sha256 = str(expected_media_sha256).strip().lower()
            if not _valid_sha256(expected_media_sha256):
                raise ValueError(
                    f"sample {identifier!r} input.sha256 must be 64 lowercase/uppercase hex characters"
                )
            input_value["sha256"] = expected_media_sha256
        if "://" not in input_path:
            resolved = (manifest_path.parent / input_path).resolve()
            if require_input_files and not resolved.is_file():
                raise FileNotFoundError(f"sample {identifier!r} input does not exist: {resolved}")
            if resolved.is_file() and expected_media_sha256 is not None:
                actual_media_sha256 = _file_sha256(resolved)
                if actual_media_sha256 != expected_media_sha256:
                    raise ValueError(
                        f"sample {identifier!r} input.sha256 mismatch: "
                        f"expected {expected_media_sha256}, got {actual_media_sha256}"
                    )
                sample["_verified_input_sha256"] = actual_media_sha256
                verified_media_sha256s.append((identifier, actual_media_sha256))
        sample["input"] = input_value
        sample["_manifest_directory"] = str(manifest_path.parent)
        samples.append(sample)
    if not samples:
        raise ValueError("manifest contains no enabled, verified samples")
    if require_all_categories:
        missing = sorted(set(REQUIRED_ACCURACY_CATEGORIES) - categories)
        if missing:
            raise ValueError("manifest is missing required categories: " + ", ".join(missing))
        empty_categories = sorted(
            category
            for category, event_count in readable_events_by_category.items()
            if event_count == 0
        )
        if empty_categories:
            raise ValueError(
                "manifest has no readable event labels for categories: "
                + ", ".join(empty_categories)
            )
        if not has_multi_vehicle_event_set:
            raise ValueError(
                "multiple_vehicles requires at least one sample with two or more expected_events"
            )
    if require_negative_sample and negative_sample_count < 1:
        raise ValueError(
            "manifest requires at least one verified negative sample with expected_plate=null"
        )
    return AccuracyManifest(
        path=manifest_path,
        dataset_id=dataset_id,
        samples=tuple(samples),
        sha256=_manifest_sha256(manifest_path),
        negative_sample_count=negative_sample_count,
        verified_media_sha256s=tuple(verified_media_sha256s),
    )


def _load_symbol(specification: str) -> Any:
    if ":" not in specification:
        raise ValueError("callable specification must be 'module:attribute' or '/path/file.py:attribute'")
    module_name, attribute_path = specification.rsplit(":", 1)
    if not module_name or not attribute_path:
        raise ValueError("callable specification has an empty module or attribute")
    source_path = Path(module_name)
    if source_path.suffix == ".py" or source_path.is_file():
        source_path = source_path.resolve()
        generated_name = "_bcvision_benchmark_adapter_" + hashlib.sha256(
            str(source_path).encode("utf-8")
        ).hexdigest()[:12]
        module_spec = importlib.util.spec_from_file_location(generated_name, source_path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load adapter module from {source_path}")
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    value = module
    for part in attribute_path.split("."):
        value = getattr(value, part)
    return value


def load_performance_adapter(
    specification: str,
    *,
    name: str | None = None,
    production_evidence: bool = False,
) -> PerformanceAdapter:
    target = _load_symbol(specification)
    if inspect.isclass(target):
        target = target()
    if hasattr(target, "process") and callable(target.process):
        if name:
            target.adapter_name = name
        return target
    if not callable(target):
        raise TypeError("performance adapter symbol must be callable or expose process(job)")
    return CallablePerformanceAdapter(
        target,
        name=name,
        production_evidence=production_evidence,
    )


class CallableAccuracyAdapter:
    def __init__(self, function: Callable[[Mapping[str, Any]], Any], *, name: str | None = None) -> None:
        self._function = function
        self.adapter_name = name or getattr(function, "__name__", "callable-accuracy-adapter")

    @classmethod
    def from_specification(cls, specification: str, *, name: str | None = None) -> "CallableAccuracyAdapter":
        target = _load_symbol(specification)
        if inspect.isclass(target):
            target = target()
        if hasattr(target, "predict") and callable(target.predict):
            function = target.predict
            adapter_name = name or getattr(target, "adapter_name", type(target).__name__)
            return cls(function, name=adapter_name)
        if not callable(target):
            raise TypeError("accuracy adapter symbol must be callable or expose predict(sample)")
        return cls(target, name=name)

    def predict(self, sample: Mapping[str, Any]) -> AccuracyPrediction:
        return _coerce_prediction(self._function(sample))


class CommandAccuracyAdapter:
    """Run a JSON-in/JSON-out command per verified manifest sample."""

    def __init__(
        self,
        command: str | Sequence[str],
        *,
        name: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.command = _split_command(command)
        if not self.command:
            raise ValueError("accuracy adapter command cannot be empty")
        self.adapter_name = name
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    def predict(self, sample: Mapping[str, Any]) -> AccuracyPrediction:
        input_mapping = sample.get("input", {})
        input_path = str(
            input_mapping.get("resolved_path", input_mapping.get("path", ""))
        )
        replacements = {
            "{engine}": self.adapter_name,
            "{sample_id}": str(sample.get("id", "")),
            "{input}": input_path,
        }
        command = []
        for token in self.command:
            rendered = token
            for marker, value in replacements.items():
                rendered = rendered.replace(marker, value)
            command.append(rendered)
        request = {
            "schema": "bcvision.anpr.accuracy-sample/v1",
            "engine": self.adapter_name,
            "sample": {key: value for key, value in sample.items() if not key.startswith("_")},
        }
        completed = subprocess.run(
            command,
            input=json.dumps(request, ensure_ascii=False),
            text=True,
            encoding="utf-8",
            errors="strict",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
            env={
                **os.environ,
                "BC_VISION_BENCHMARK_ENGINE": self.adapter_name,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            },
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"accuracy adapter {self.adapter_name!r} exited with {completed.returncode}: "
                f"{completed.stderr.strip()[:500]}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"accuracy adapter {self.adapter_name!r} stdout must be one JSON value"
            ) from exc
        return _coerce_prediction(result)


def _json_safe_details(value: Any) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    details = dict(value)
    for key in (
        "plate",
        "text",
        "predicted_plate",
        "plates",
        "events",
        "confidence",
        "accepted",
        "valid",
    ):
        details.pop(key, None)
    try:
        json.dumps(details, ensure_ascii=False)
    except (TypeError, ValueError):
        return {"note": "adapter details were not JSON-serializable"}
    return details or None


def _coerce_prediction(value: Any) -> AccuracyPrediction:
    if isinstance(value, AccuracyPrediction):
        return value
    if value is None:
        return AccuracyPrediction(plate="", accepted=False)
    if isinstance(value, str):
        return AccuracyPrediction(plate=value, accepted=bool(normalize_plate_text(value)))
    if not isinstance(value, Mapping):
        raise TypeError("accuracy adapter must return AccuracyPrediction, mapping, string, or None")
    raw_events = value.get("events", value.get("plates"))
    parsed_events: list[dict[str, Any]] = []
    if raw_events is not None:
        if not isinstance(raw_events, (list, tuple)):
            raise TypeError("accuracy prediction events/plates must be an array")
        for event_index, raw_event in enumerate(raw_events):
            event = {"plate": raw_event} if isinstance(raw_event, str) else raw_event
            if not isinstance(event, Mapping):
                raise TypeError(f"accuracy prediction event {event_index} must be an object or string")
            event_plate = normalize_plate_text(
                event.get("plate", event.get("text", event.get("predicted_plate", "")))
            )
            if not event_plate:
                continue
            normalized_event = dict(event)
            normalized_event["plate"] = event_plate
            timestamp_ms = normalized_event.get("timestamp_ms")
            if timestamp_ms is not None:
                timestamp_ms = float(timestamp_ms)
                if not math.isfinite(timestamp_ms) or timestamp_ms < 0:
                    raise ValueError(
                        f"accuracy prediction event {event_index} has invalid timestamp_ms"
                    )
                normalized_event["timestamp_ms"] = timestamp_ms
            event_confidence = normalized_event.get("confidence")
            if event_confidence is not None:
                event_confidence = float(event_confidence)
                if not math.isfinite(event_confidence) or not 0.0 <= event_confidence <= 1.0:
                    raise ValueError(
                        f"accuracy prediction event {event_index} confidence must be within 0..1"
                    )
                normalized_event["confidence"] = event_confidence
            parsed_events.append(normalized_event)
    plate = str(value.get("plate", value.get("text", value.get("predicted_plate", ""))) or "")
    if not plate and parsed_events:
        plate = str(parsed_events[0]["plate"])
    confidence_raw = value.get("confidence")
    confidence = None if confidence_raw is None else float(confidence_raw)
    if confidence is not None and (not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
        raise ValueError("accuracy prediction confidence must be null or within 0..1")
    accepted_raw = value.get("accepted", value.get("valid"))
    accepted = (
        bool(parsed_events or normalize_plate_text(plate))
        if accepted_raw is None
        else bool(accepted_raw)
    )
    return AccuracyPrediction(
        plate=plate,
        confidence=confidence,
        accepted=accepted,
        events=tuple(parsed_events) if raw_events is not None else None,
        details=_json_safe_details(value),
    )


def _character_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _prediction_events(prediction: AccuracyPrediction) -> list[dict[str, Any]]:
    if prediction.accepted is False:
        return []
    if prediction.events is not None:
        return [dict(event) for event in prediction.events]
    plate = normalize_plate_text(prediction.plate)
    if not plate:
        return []
    event: dict[str, Any] = {"plate": plate}
    if prediction.confidence is not None:
        event["confidence"] = prediction.confidence
    return [event]


def _event_matches(expected: Mapping[str, Any], predicted: Mapping[str, Any]) -> bool:
    if normalize_plate_text(expected.get("plate")) != normalize_plate_text(predicted.get("plate")):
        return False
    start_ms = expected.get("start_ms")
    end_ms = expected.get("end_ms")
    if start_ms is None and end_ms is None:
        return True
    timestamp_ms = predicted.get("timestamp_ms")
    if timestamp_ms is None:
        return False
    timestamp = float(timestamp_ms)
    if start_ms is not None and timestamp < float(start_ms):
        return False
    if end_ms is not None and timestamp > float(end_ms):
        return False
    return True


def _match_event_sets(
    expected_events: Sequence[Mapping[str, Any]],
    predicted_events: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    remaining = set(range(len(predicted_events)))
    matched_pairs: list[tuple[int, int]] = []
    for expected_index, expected in enumerate(expected_events):
        candidates = [
            predicted_index
            for predicted_index in remaining
            if _event_matches(expected, predicted_events[predicted_index])
        ]
        if not candidates:
            continue
        # Prefer the closest timestamp to the center of a labelled event window.
        start_ms = expected.get("start_ms")
        end_ms = expected.get("end_ms")
        if start_ms is not None or end_ms is not None:
            center = statistics.fmean(
                [
                    float(value)
                    for value in (start_ms, end_ms)
                    if value is not None
                ]
            )
            candidates.sort(
                key=lambda index: abs(float(predicted_events[index]["timestamp_ms"]) - center)
            )
        predicted_index = candidates[0]
        remaining.remove(predicted_index)
        matched_pairs.append((expected_index, predicted_index))

    expected_counts = Counter(
        normalize_plate_text(event.get("plate")) for event in expected_events
    )
    predicted_counts = Counter(
        normalize_plate_text(event.get("plate")) for event in predicted_events
    )
    duplicates = sum(
        max(0, predicted_count - max(1, expected_counts.get(plate, 0)))
        for plate, predicted_count in predicted_counts.items()
    )
    return {
        "matched_pairs": matched_pairs,
        "matched_events": len(matched_pairs),
        "missed_events": len(expected_events) - len(matched_pairs),
        "false_positive_events": len(remaining),
        "duplicate_events": duplicates,
        "exact_set_match": (
            bool(expected_events)
            and len(matched_pairs) == len(expected_events)
            and not remaining
        ),
    }


def _optional_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else round(numerator / denominator, 6)


def _score_accuracy_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = {
        "samples": 0,
        "positive_samples": 0,
        "negative_samples": 0,
        "exact_set_matches": 0,
        "false_accept_samples": 0,
        "readable_rejections": 0,
        "expected_events": 0,
        "predicted_events": 0,
        "matched_events": 0,
        "missed_events": 0,
        "false_positive_events": 0,
        "duplicate_events": 0,
    }
    character_error_rates: list[float] = []
    latencies: list[float] = []
    categories: dict[str, dict[str, Any]] = {
        category: dict(totals) for category in REQUIRED_ACCURACY_CATEGORIES
    }
    for row in rows:
        expected_events = list(row["expected_events"])
        predicted_events = list(row["predicted_events"])
        positive = bool(expected_events)
        false_accept_sample = not positive and bool(predicted_events)
        readable_rejection = positive and not predicted_events
        values = {
            "samples": 1,
            "positive_samples": int(positive),
            "negative_samples": int(not positive),
            "exact_set_matches": int(bool(row["exact_set_match"])),
            "false_accept_samples": int(false_accept_sample),
            "readable_rejections": int(readable_rejection),
            "expected_events": len(expected_events),
            "predicted_events": len(predicted_events),
            "matched_events": int(row["matched_events"]),
            "missed_events": int(row["missed_events"]),
            "false_positive_events": int(row["false_positive_events"]),
            "duplicate_events": int(row["duplicate_events"]),
        }
        bucket = categories[str(row["category"])]
        for key, amount in values.items():
            totals[key] += amount
            bucket[key] += amount
        predicted_plates = [normalize_plate_text(event.get("plate")) for event in predicted_events]
        for expected_event in expected_events:
            expected_plate = normalize_plate_text(expected_event.get("plate"))
            best_distance = (
                min(_character_distance(predicted, expected_plate) for predicted in predicted_plates)
                if predicted_plates
                else len(expected_plate)
            )
            character_error_rates.append(best_distance / max(1, len(expected_plate)))
        latencies.append(float(row["wall_latency_ms"]))

    def finalize(bucket: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(bucket)
        result["exact_set_accuracy"] = _optional_rate(
            int(bucket["exact_set_matches"]), int(bucket["positive_samples"])
        )
        result["false_accept_rate"] = _optional_rate(
            int(bucket["false_accept_samples"]), int(bucket["negative_samples"])
        )
        result["event_recall"] = _optional_rate(
            int(bucket["matched_events"]), int(bucket["expected_events"])
        )
        result["event_precision"] = _optional_rate(
            int(bucket["matched_events"]), int(bucket["predicted_events"])
        )
        return result

    result = finalize(totals)
    # Backward-readable names retain sample-level exact-set semantics.
    result["readable_samples"] = result["positive_samples"]
    result["exact_matches"] = result["exact_set_matches"]
    result["exact_accuracy"] = result["exact_set_accuracy"]
    result["false_accepts"] = result["false_accept_samples"]
    result["mean_character_error_rate"] = round(
        statistics.fmean(character_error_rates) if character_error_rates else 0.0,
        6,
    )
    result["average_latency_ms"] = (
        round(statistics.fmean(latencies), 4) if latencies else 0.0
    )
    result["p95_latency_ms"] = round(_percentile_95(latencies), 4)
    result["categories"] = {
        category: finalize(bucket) for category, bucket in categories.items()
    }
    return result


def run_accuracy_adapter(
    manifest: AccuracyManifest,
    adapter: AccuracyAdapter,
    *,
    engine_label: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for sample in manifest.samples:
        # Ground truth must never cross the adapter boundary. In particular,
        # expected_plate, label_status and operator notes stay scorer-only.
        adapter_input = _accuracy_adapter_input(sample)
        started = time.perf_counter()
        prediction = _coerce_prediction(adapter.predict(adapter_input))
        wall_latency_ms = (time.perf_counter() - started) * 1000.0
        expected_events = [dict(event) for event in sample["_expected_events"]]
        predicted_events = _prediction_events(prediction)
        matching = _match_event_sets(expected_events, predicted_events)
        expected_plates = [str(event["plate"]) for event in expected_events]
        predicted_plates = [str(event["plate"]) for event in predicted_events]
        rows.append(
            {
                "engine": engine_label,
                "adapter_name": adapter.adapter_name,
                "sample_id": sample["id"],
                "category": sample["category"],
                "input_path": sample["input"]["path"],
                "expected_plate": expected_plates[0] if expected_plates else "",
                "predicted_plate": predicted_plates[0] if predicted_plates else "",
                "expected_events": expected_events,
                "predicted_events": predicted_events,
                "accepted": bool(predicted_events),
                "exact_match": matching["exact_set_match"],
                "exact_set_match": matching["exact_set_match"],
                "false_accept": bool(not expected_events and predicted_events),
                "matched_events": matching["matched_events"],
                "missed_events": matching["missed_events"],
                "false_positive_events": matching["false_positive_events"],
                "duplicate_events": matching["duplicate_events"],
                "confidence": prediction.confidence,
                "wall_latency_ms": round(wall_latency_ms, 4),
                "details": prediction.details,
            }
        )
    return {
        "engine": engine_label,
        "adapter_name": adapter.adapter_name,
        "metrics": _score_accuracy_rows(rows),
        "predictions": rows,
    }


def _accuracy_adapter_input(sample: Mapping[str, Any]) -> dict[str, Any]:
    input_value = dict(sample["input"])
    raw_path = str(input_value["path"])
    if "://" not in raw_path:
        input_value["resolved_path"] = str(
            (Path(str(sample["_manifest_directory"])) / raw_path).resolve()
        )
    request = {
        "id": str(sample["id"]),
        "category": str(sample["category"]),
        "input": input_value,
    }
    # Optional adapter_input is explicitly inference-side configuration. The
    # loader never copies arbitrary manifest fields into the adapter request.
    if isinstance(sample.get("adapter_input"), Mapping):
        request["adapter_input"] = dict(sample["adapter_input"])
    return request


def _category_accuracy_regressions(
    v1_categories: Mapping[str, Mapping[str, Any]],
    v2_categories: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    regressions: list[dict[str, Any]] = []
    for category in REQUIRED_ACCURACY_CATEGORIES:
        v1 = v1_categories[category]
        v2 = v2_categories[category]
        metrics: list[str] = []
        deltas: dict[str, float | int | None] = {}

        for metric in ("exact_set_accuracy", "event_recall"):
            before = v1.get(metric)
            after = v2.get(metric)
            delta = (
                None
                if before is None or after is None
                else round(float(after) - float(before), 6)
            )
            deltas[metric] = delta
            if (before is None) != (after is None) or (
                delta is not None and delta < 0
            ):
                metrics.append(metric)

        before_false_accept = v1.get("false_accept_rate")
        after_false_accept = v2.get("false_accept_rate")
        false_accept_delta = (
            None
            if before_false_accept is None or after_false_accept is None
            else round(
                float(after_false_accept) - float(before_false_accept),
                6,
            )
        )
        deltas["false_accept_rate"] = false_accept_delta
        if (before_false_accept is None) != (after_false_accept is None) or (
            false_accept_delta is not None and false_accept_delta > 0
        ):
            metrics.append("false_accept_rate")

        for metric in ("false_positive_events", "duplicate_events"):
            delta = int(v2[metric]) - int(v1[metric])
            deltas[metric] = delta
            if delta > 0:
                metrics.append(metric)

        if metrics:
            regressions.append(
                {
                    "category": category,
                    "metrics": metrics,
                    "deltas": deltas,
                }
            )
    return regressions


def compare_accuracy_adapters(
    manifest: AccuracyManifest,
    v1_adapter: AccuracyAdapter,
    v2_adapter: AccuracyAdapter,
) -> dict[str, Any]:
    """Run V1 and V2 independently over exactly the same verified samples."""

    v1 = run_accuracy_adapter(manifest, v1_adapter, engine_label="v1")
    v2 = run_accuracy_adapter(manifest, v2_adapter, engine_label="v2")
    v1_metrics = v1["metrics"]
    v2_metrics = v2["metrics"]
    v1_false_accept_rate = v1_metrics["false_accept_rate"]
    v2_false_accept_rate = v2_metrics["false_accept_rate"]
    false_accept_not_worse = (
        v1_false_accept_rate is None and v2_false_accept_rate is None
    ) or (
        v1_false_accept_rate is not None
        and v2_false_accept_rate is not None
        and float(v2_false_accept_rate) <= float(v1_false_accept_rate)
    )
    category_regressions = _category_accuracy_regressions(
        v1_metrics["categories"],
        v2_metrics["categories"],
    )
    accuracy_not_worse = (
        float(v2_metrics["exact_accuracy"]) >= float(v1_metrics["exact_accuracy"])
        and float(v2_metrics["event_recall"]) >= float(v1_metrics["event_recall"])
        and false_accept_not_worse
        and int(v2_metrics["false_positive_events"])
        <= int(v1_metrics["false_positive_events"])
        and int(v2_metrics["duplicate_events"]) <= int(v1_metrics["duplicate_events"])
        and float(v2_metrics["mean_character_error_rate"])
        <= float(v1_metrics["mean_character_error_rate"])
        and not category_regressions
    )
    return {
        "schema": "bcvision.anpr.accuracy-comparison/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": manifest.dataset_id,
        "manifest_path": str(manifest.path),
        "manifest_sha256": manifest.sha256,
        "negative_sample_count": manifest.negative_sample_count,
        "verified_media_sha256s": [
            {"sample_id": sample_id, "sha256": sha256}
            for sample_id, sha256 in manifest.verified_media_sha256s
        ],
        "required_categories": list(REQUIRED_ACCURACY_CATEGORIES),
        "same_manifest_for_both_engines": True,
        "v1": v1,
        "v2": v2,
        "comparison": {
            "v2_accuracy_not_worse": accuracy_not_worse,
            "category_regressions": category_regressions,
            "exact_accuracy_delta": round(
                float(v2_metrics["exact_accuracy"]) - float(v1_metrics["exact_accuracy"]), 6
            ),
            "false_accept_rate_delta": round(
                float(v2_metrics["false_accept_rate"])
                - float(v1_metrics["false_accept_rate"]),
                6,
            )
            if v1_false_accept_rate is not None and v2_false_accept_rate is not None
            else None,
            "event_recall_delta": round(
                float(v2_metrics["event_recall"]) - float(v1_metrics["event_recall"]),
                6,
            ),
            "false_positive_event_delta": int(v2_metrics["false_positive_events"])
            - int(v1_metrics["false_positive_events"]),
            "duplicate_event_delta": int(v2_metrics["duplicate_events"])
            - int(v1_metrics["duplicate_events"]),
            "average_latency_ms_delta": round(
                float(v2_metrics["average_latency_ms"])
                - float(v1_metrics["average_latency_ms"]),
                4,
            ),
        },
        "production_decision_allowed": False,
        "production_decision_reason": (
            "Accuracy comparison alone is insufficient; verified real performance and resource "
            "benchmarks are also required, and this harness never switches engines."
        ),
    }


ACCURACY_CSV_FIELDS = (
    "engine",
    "adapter_name",
    "sample_id",
    "category",
    "input_path",
    "expected_plate",
    "predicted_plate",
    "expected_events",
    "predicted_events",
    "accepted",
    "exact_match",
    "false_accept",
    "matched_events",
    "missed_events",
    "false_positive_events",
    "duplicate_events",
    "confidence",
    "wall_latency_ms",
    "details",
)


def write_accuracy_outputs(
    report: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    _write_json(Path(json_path), report)
    rows: list[dict[str, Any]] = []
    for engine in ("v1", "v2"):
        section = report.get(engine)
        if not isinstance(section, Mapping) or not isinstance(section.get("predictions"), list):
            raise ValueError(f"accuracy report has no {engine} predictions")
        for raw in section["predictions"]:
            row = dict(raw)
            for field in ("expected_events", "predicted_events", "details"):
                if row.get(field) is not None:
                    row[field] = json.dumps(row[field], ensure_ascii=False, sort_keys=True)
            rows.append(row)
    _write_csv(Path(csv_path), rows, ACCURACY_CSV_FIELDS)


def scenario_from_mapping(value: Mapping[str, Any]) -> BenchmarkScenario:
    """Small public helper for callable adapters and external configuration."""

    allowed = {field.name for field in inspect.signature(BenchmarkScenario).parameters.values()}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("unknown scenario fields: " + ", ".join(sorted(unknown)))
    return BenchmarkScenario(**dict(value))


def report_as_json(value: Any) -> str:
    """Return stable JSON for tests, logs and command adapter implementations."""

    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
