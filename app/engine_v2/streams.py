from __future__ import annotations

import importlib.util
import json
import math
import os
import platform
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Protocol

import numpy as np

from .types import FramePacket


class StreamRole(str, Enum):
    MAIN = "main"
    SUB = "sub"


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One decoded video frame, independent of a decoder implementation."""

    image: np.ndarray
    captured_at: float | None = None
    monotonic_at: float | None = None
    source_pts: float | None = None
    key_frame: bool | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("decoded frame image must be a numpy array")
        if self.image.ndim not in (2, 3) or self.image.size == 0:
            raise ValueError("decoded frame image must be a non-empty 2D or 3D array")


@dataclass(frozen=True, slots=True)
class StreamSpec:
    camera_id: str
    role: StreamRole
    url: str = field(repr=False)
    transport: str = "tcp"
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    options: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not self.url.strip():
            raise ValueError(f"{self.role.value} stream URL must not be empty")
        if self.transport not in {"tcp", "udp", "udp_multicast", "http", "https"}:
            raise ValueError(f"unsupported RTSP transport: {self.transport}")
        if self.connect_timeout_seconds <= 0 or self.read_timeout_seconds <= 0:
            raise ValueError("stream timeouts must be positive")


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    initial_delay_seconds: float = 0.25
    maximum_delay_seconds: float = 8.0
    multiplier: float = 2.0
    jitter_ratio: float = 0.15

    def __post_init__(self) -> None:
        if self.initial_delay_seconds < 0:
            raise ValueError("initial reconnect delay cannot be negative")
        if self.maximum_delay_seconds < self.initial_delay_seconds:
            raise ValueError("maximum reconnect delay cannot be below the initial delay")
        if self.multiplier < 1:
            raise ValueError("reconnect multiplier must be at least 1")
        if not 0 <= self.jitter_ratio <= 1:
            raise ValueError("reconnect jitter must be between 0 and 1")

    def delay_for(self, consecutive_failure: int, random_value: float = 0.5) -> float:
        attempt = max(1, int(consecutive_failure))
        base = min(
            self.maximum_delay_seconds,
            self.initial_delay_seconds * (self.multiplier ** (attempt - 1)),
        )
        centered_random = min(1.0, max(0.0, float(random_value))) * 2.0 - 1.0
        return max(0.0, base * (1.0 + centered_random * self.jitter_ratio))


@dataclass(frozen=True, slots=True)
class RTSPProducerConfig:
    camera_id: str
    main_url: str = field(repr=False)
    sub_url: str = field(repr=False)
    transport: str = "tcp"
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    maximum_main_frame_age_seconds: float = 1.0
    require_main_frame: bool = True
    copy_main_frames: bool = True
    hardware_decode: str | None = "auto"
    backend_preference: tuple[str, ...] = ("pyav", "ffmpeg", "opencv")
    reconnect: ReconnectPolicy = field(default_factory=ReconnectPolicy)
    main_options: Mapping[str, str] = field(default_factory=dict, repr=False)
    sub_options: Mapping[str, str] = field(default_factory=dict, repr=False)
    daemon_threads: bool = True

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not self.main_url.strip() or not self.sub_url.strip():
            raise ValueError("both main_url and sub_url are required")
        if self.maximum_main_frame_age_seconds <= 0:
            raise ValueError("maximum main-frame age must be positive")
        supported = {"pyav", "ffmpeg", "opencv"}
        if not self.backend_preference:
            raise ValueError("at least one decoder backend is required")
        unknown = set(self.backend_preference) - supported
        if unknown:
            raise ValueError(f"unsupported decoder backend(s): {', '.join(sorted(unknown))}")

    def stream_spec(self, role: StreamRole) -> StreamSpec:
        if role is StreamRole.MAIN:
            url, options = self.main_url, self.main_options
        else:
            url, options = self.sub_url, self.sub_options
        return StreamSpec(
            camera_id=self.camera_id,
            role=role,
            url=url,
            transport=self.transport,
            connect_timeout_seconds=self.connect_timeout_seconds,
            read_timeout_seconds=self.read_timeout_seconds,
            options=dict(options),
        )


class ProducerActivity(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class ProducerCadencePolicy:
    """Thread-safe producer admission target supplied by the V2 runtime.

    Decoders must continue draining RTSP continuously. This policy controls
    only which newest decoded sub-frames are admitted to the central runtime.
    Active cameras admit cheap tracking frames between detector-due frames;
    idle cameras retain a safety floor for the motion gate so a fast vehicle is
    not missed merely because the system is under load.
    """

    target_detector_fps: float
    activity: ProducerActivity = ProducerActivity.IDLE
    tracking_frames_between_detection: int = 0
    minimum_active_detector_fps: float = 0.5
    minimum_idle_admission_fps: float = 2.0
    maximum_admission_fps: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.activity, ProducerActivity):
            try:
                object.__setattr__(self, "activity", ProducerActivity(self.activity))
            except (TypeError, ValueError) as exc:
                raise ValueError("activity must be 'idle' or 'active'") from exc
        if not math.isfinite(self.target_detector_fps) or self.target_detector_fps <= 0:
            raise ValueError("target_detector_fps must be finite and positive")
        if self.tracking_frames_between_detection < 0:
            raise ValueError("tracking_frames_between_detection cannot be negative")
        if (
            not math.isfinite(self.minimum_active_detector_fps)
            or self.minimum_active_detector_fps <= 0
        ):
            raise ValueError("minimum_active_detector_fps must be finite and positive")
        if (
            not math.isfinite(self.minimum_idle_admission_fps)
            or self.minimum_idle_admission_fps <= 0
        ):
            raise ValueError("minimum_idle_admission_fps must be finite and positive")
        if not math.isfinite(self.maximum_admission_fps) or self.maximum_admission_fps <= 0:
            raise ValueError("maximum_admission_fps must be finite and positive")
        if self.maximum_admission_fps < max(
            self.minimum_active_detector_fps,
            self.minimum_idle_admission_fps,
        ):
            raise ValueError("maximum_admission_fps cannot be below a safety floor")

    @property
    def effective_detector_fps(self) -> float:
        target = float(self.target_detector_fps)
        if self.activity is ProducerActivity.IDLE:
            target = max(target, float(self.minimum_idle_admission_fps))
            maximum_detector_fps = float(self.maximum_admission_fps)
        else:
            target = max(target, float(self.minimum_active_detector_fps))
            maximum_detector_fps = float(self.maximum_admission_fps) / (
                self.tracking_frames_between_detection + 1
            )
        return min(target, maximum_detector_fps)

    @property
    def target_admission_fps(self) -> float:
        multiplier = (
            1
            if self.activity is ProducerActivity.IDLE
            else self.tracking_frames_between_detection + 1
        )
        return min(
            float(self.maximum_admission_fps),
            self.effective_detector_fps * multiplier,
        )


@dataclass(frozen=True, slots=True)
class FrameAdmissionDecision:
    admit: bool
    adaptive: bool
    detector_due: bool = True
    policy: ProducerCadencePolicy | None = None


class AdaptiveFrameAdmissionController:
    """Lock-protected, non-blocking newest-frame cadence controller.

    ``decide`` never sleeps. A denied frame is discarded immediately while the
    decoder loop continues reading, preventing RTSP buffers from accumulating
    latency. An activity transition forces the next current frame through and
    marks it detector-due.
    """

    def __init__(self, policy: ProducerCadencePolicy | None = None) -> None:
        self._policy = policy
        self._last_admitted_at: float | None = None
        self._last_detector_at: float | None = None
        self._tracking_since_detector = 0
        self._force_next = policy is not None
        self._lock = threading.RLock()

    @property
    def policy(self) -> ProducerCadencePolicy | None:
        with self._lock:
            return self._policy

    def update(self, policy: ProducerCadencePolicy | None) -> None:
        if policy is not None and not isinstance(policy, ProducerCadencePolicy):
            raise TypeError("adaptive cadence policy must be ProducerCadencePolicy or None")
        with self._lock:
            previous = self._policy
            if previous == policy:
                return
            self._policy = policy
            if policy is None:
                self._reset_timing_locked(force=False)
                return
            if previous is None or previous.activity is not policy.activity:
                self._force_next = True
                self._tracking_since_detector = 0

    def reset_timing(self) -> None:
        with self._lock:
            self._reset_timing_locked(force=self._policy is not None)

    def detector_unaccepted(self) -> None:
        """Retry detector intent at the next cadence slot after sink pressure."""

        with self._lock:
            policy = self._policy
            if policy is None:
                return
            self._last_detector_at = None
            self._tracking_since_detector = policy.tracking_frames_between_detection

    def decide(self, monotonic_at: float) -> FrameAdmissionDecision:
        now = float(monotonic_at)
        if not math.isfinite(now):
            raise ValueError("admission timestamp must be finite")
        with self._lock:
            policy = self._policy
            if policy is None:
                return FrameAdmissionDecision(True, False)

            if self._last_admitted_at is not None and now < self._last_admitted_at:
                self._reset_timing_locked(force=True)

            interval = 1.0 / policy.target_admission_fps
            admission_due = (
                self._force_next
                or self._last_admitted_at is None
                or now - self._last_admitted_at + 1e-12 >= interval
            )
            if not admission_due:
                return FrameAdmissionDecision(False, True, False, policy)

            force = self._force_next
            self._force_next = False
            self._last_admitted_at = now
            if policy.activity is ProducerActivity.IDLE:
                detector_due = True
            else:
                detector_interval = 1.0 / policy.effective_detector_fps
                time_due = (
                    self._last_detector_at is None
                    or now - self._last_detector_at + 1e-12 >= detector_interval
                )
                gap_ready = (
                    self._tracking_since_detector
                    >= policy.tracking_frames_between_detection
                )
                detector_due = force or (time_due and gap_ready)

            if detector_due:
                self._last_detector_at = now
                self._tracking_since_detector = 0
            else:
                self._tracking_since_detector += 1
            return FrameAdmissionDecision(True, True, detector_due, policy)

    def _reset_timing_locked(self, *, force: bool) -> None:
        self._last_admitted_at = None
        self._last_detector_at = None
        self._tracking_since_detector = 0
        self._force_next = force


@dataclass(frozen=True, slots=True)
class HardwareDecodePlan:
    accelerator: str | None = None
    device: str | None = None
    reason: str = "software decode"

    @property
    def enabled(self) -> bool:
        return self.accelerator is not None

    @classmethod
    def software(cls, reason: str = "software decode") -> HardwareDecodePlan:
        return cls(reason=reason)


def probe_ffmpeg_hardware_accelerators(ffmpeg_binary: str | None = None) -> frozenset[str]:
    """Return FFmpeg hardware accelerators without making it a dependency.

    Probe failures intentionally degrade to software decoding.
    """

    binary = ffmpeg_binary or shutil.which("ffmpeg")
    if not binary:
        return frozenset()
    try:
        result = subprocess.run(
            [binary, "-hide_banner", "-hwaccels"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    accelerators = {
        line.strip().lower()
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("hardware acceleration")
    }
    return frozenset(accelerators)


def select_hardware_decode(
    requested: str | None = "auto",
    *,
    system: str | None = None,
    available_accelerators: Sequence[str] | None = None,
    device_exists: Callable[[str], bool] = os.path.exists,
    ffmpeg_binary: str | None = None,
) -> HardwareDecodePlan:
    """Select a verified hardware path, preferring Intel Quick Sync/iGPU."""

    normalized = (requested or "off").strip().lower()
    if normalized in {"off", "none", "software", "disabled"}:
        return HardwareDecodePlan.software("hardware decoding disabled")

    available = {
        item.strip().lower()
        for item in (
            available_accelerators
            if available_accelerators is not None
            else probe_ffmpeg_hardware_accelerators(ffmpeg_binary)
        )
    }
    if normalized != "auto":
        if available and normalized not in available:
            return HardwareDecodePlan.software(f"requested accelerator {normalized} is unavailable")
        return HardwareDecodePlan(normalized, reason="explicit hardware accelerator")

    os_name = (system or platform.system()).strip().lower()
    if os_name == "linux":
        render_device = "/dev/dri/renderD128"
        candidates = (
            ("qsv", render_device),
            ("vaapi", render_device),
            ("cuda", "/dev/nvidia0"),
            ("vdpau", None),
        )
    elif os_name == "windows":
        candidates = (("qsv", None), ("d3d11va", None), ("dxva2", None), ("cuda", None))
    elif os_name in {"darwin", "macos"}:
        candidates = (("videotoolbox", None),)
    else:
        candidates = (("qsv", None), ("vaapi", None), ("cuda", None))

    for accelerator, device in candidates:
        if accelerator not in available:
            continue
        if device is not None and not device_exists(device):
            continue
        reason = "Intel Quick Sync available" if accelerator == "qsv" else "hardware accelerator available"
        return HardwareDecodePlan(accelerator, device, reason)
    return HardwareDecodePlan.software("no usable hardware accelerator detected")


class DecoderSession(Protocol):
    backend_name: str
    hardware_accelerator: str | None

    def read(self) -> DecodedFrame | None: ...

    def close(self) -> None: ...


class DecoderFactory(Protocol):
    def open(self, spec: StreamSpec) -> DecoderSession: ...


class DecoderOpenError(RuntimeError):
    pass


class _PrefetchedSession:
    def __init__(self, session: DecoderSession, first_frame: DecodedFrame) -> None:
        self._session = session
        self._first_frame: DecodedFrame | None = first_frame
        self.backend_name = session.backend_name
        self.hardware_accelerator = session.hardware_accelerator

    def read(self) -> DecodedFrame | None:
        if self._first_frame is not None:
            frame, self._first_frame = self._first_frame, None
            return frame
        return self._session.read()

    def close(self) -> None:
        self._first_frame = None
        self._session.close()


class AutoDecoderFactory:
    """Lazy optional decoder selection with hardware-to-software fallback."""

    def __init__(
        self,
        *,
        backend_preference: Sequence[str] = ("pyav", "ffmpeg", "opencv"),
        hardware_decode: str | None = "auto",
        hardware_plan: HardwareDecodePlan | None = None,
        ffmpeg_binary: str | None = None,
        ffprobe_binary: str | None = None,
    ) -> None:
        self.backend_preference = tuple(backend_preference)
        self.ffmpeg_binary = ffmpeg_binary or shutil.which("ffmpeg")
        self.ffprobe_binary = ffprobe_binary or shutil.which("ffprobe")
        self.hardware_plan = hardware_plan or select_hardware_decode(
            hardware_decode,
            ffmpeg_binary=self.ffmpeg_binary,
        )

    def open(self, spec: StreamSpec) -> DecoderSession:
        plans = [self.hardware_plan] if self.hardware_plan.enabled else []
        plans.append(HardwareDecodePlan.software("automatic software fallback"))
        failures: list[str] = []

        for plan in plans:
            for backend in self.backend_preference:
                if not self._backend_available(backend):
                    failures.append(f"{backend}: unavailable")
                    continue
                session: DecoderSession | None = None
                try:
                    session = self._open_backend(backend, spec, plan)
                    first_frame = session.read()
                    if first_frame is None:
                        raise DecoderOpenError("decoder ended before its first frame")
                    return _PrefetchedSession(session, first_frame)
                except Exception as exc:
                    failures.append(
                        f"{backend}/{plan.accelerator or 'software'}: {type(exc).__name__}"
                    )
                    if session is not None:
                        try:
                            session.close()
                        except Exception:
                            pass

        summary = "; ".join(failures) if failures else "no decoder candidates"
        raise DecoderOpenError(f"unable to open video decoder ({summary})")

    def _backend_available(self, backend: str) -> bool:
        if backend == "pyav":
            return importlib.util.find_spec("av") is not None
        if backend == "opencv":
            return importlib.util.find_spec("cv2") is not None
        if backend == "ffmpeg":
            return self.ffmpeg_binary is not None and self.ffprobe_binary is not None
        return False

    def _open_backend(
        self,
        backend: str,
        spec: StreamSpec,
        plan: HardwareDecodePlan,
    ) -> DecoderSession:
        if backend == "pyav":
            # PyAV's input options are not the FFmpeg CLI's -hwaccel API. Keep
            # this path truthful and portable until an explicit PyAV HW device
            # context is available; the outer loop will try FFmpeg/OpenCV HW
            # first and return to PyAV during the software-fallback pass.
            if plan.enabled:
                raise DecoderOpenError("PyAV hardware device context is unavailable")
            return _PyAVSession(spec, plan)
        if backend == "opencv":
            return _OpenCVSession(spec, plan)
        if backend == "ffmpeg" and self.ffmpeg_binary and self.ffprobe_binary:
            return _FFmpegSession(spec, plan, self.ffmpeg_binary, self.ffprobe_binary)
        raise DecoderOpenError(f"decoder backend {backend} is unavailable")


class _PyAVSession:
    backend_name = "pyav"

    def __init__(self, spec: StreamSpec, plan: HardwareDecodePlan) -> None:
        import av  # type: ignore[import-not-found]

        options = {
            "rtsp_transport": spec.transport,
            "stimeout": str(round(spec.connect_timeout_seconds * 1_000_000)),
            "rw_timeout": str(round(spec.read_timeout_seconds * 1_000_000)),
            **dict(spec.options),
        }
        try:
            self._container = av.open(
                spec.url,
                mode="r",
                options=options,
                timeout=(spec.connect_timeout_seconds, spec.read_timeout_seconds),
            )
        except TypeError:
            self._container = av.open(spec.url, mode="r", options=options)
        self._frames = iter(self._container.decode(video=0))
        self.hardware_accelerator = None
        self._closed = False

    def read(self) -> DecodedFrame | None:
        if self._closed:
            return None
        try:
            frame = next(self._frames)
        except StopIteration:
            return None
        source_pts = float(frame.time) if frame.time is not None else None
        return DecodedFrame(
            image=frame.to_ndarray(format="bgr24"),
            captured_at=time.time(),
            monotonic_at=time.monotonic(),
            source_pts=source_pts,
            key_frame=bool(frame.key_frame),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._container.close()


class _OpenCVSession:
    backend_name = "opencv"

    def __init__(self, spec: StreamSpec, plan: HardwareDecodePlan) -> None:
        import cv2  # type: ignore[import-not-found]

        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend(
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, round(spec.connect_timeout_seconds * 1000)]
            )
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend(
                [cv2.CAP_PROP_READ_TIMEOUT_MSEC, round(spec.read_timeout_seconds * 1000)]
            )
        if plan.enabled and hasattr(cv2, "CAP_PROP_HW_ACCELERATION"):
            params.extend(
                [cv2.CAP_PROP_HW_ACCELERATION, getattr(cv2, "VIDEO_ACCELERATION_ANY", 1)]
            )
        requested_hardware = plan.enabled and hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
        used_hardware_parameters = requested_hardware
        try:
            self._capture = cv2.VideoCapture(spec.url, cv2.CAP_FFMPEG, params)
        except (TypeError, cv2.error):
            used_hardware_parameters = False
            self._capture = cv2.VideoCapture(spec.url, cv2.CAP_FFMPEG)
        if not self._capture.isOpened():
            self._capture.release()
            raise DecoderOpenError("OpenCV could not open the stream")
        actual_hardware = 0.0
        if used_hardware_parameters:
            try:
                actual_hardware = self._capture.get(cv2.CAP_PROP_HW_ACCELERATION)
            except cv2.error:
                actual_hardware = 0.0
        self.hardware_accelerator = plan.accelerator if actual_hardware > 0 else None
        self._closed = False

    def read(self) -> DecodedFrame | None:
        if self._closed:
            return None
        ok, image = self._capture.read()
        if not ok or image is None:
            return None
        return DecodedFrame(
            image=image,
            captured_at=time.time(),
            monotonic_at=time.monotonic(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._capture.release()


class _FFmpegSession:
    backend_name = "ffmpeg"

    def __init__(
        self,
        spec: StreamSpec,
        plan: HardwareDecodePlan,
        ffmpeg_binary: str,
        ffprobe_binary: str,
    ) -> None:
        self._closed = False
        width, height = self._probe_dimensions(spec, ffprobe_binary)
        self._shape = (height, width, 3)
        self._frame_bytes = height * width * 3

        command = [ffmpeg_binary, "-nostdin", "-hide_banner", "-loglevel", "error"]
        if plan.enabled:
            command.extend(["-hwaccel", plan.accelerator or "auto"])
            if plan.device:
                command.extend(["-hwaccel_device", plan.device])
        if spec.transport in {"tcp", "udp", "udp_multicast"}:
            command.extend(["-rtsp_transport", spec.transport])
        command.extend(["-rw_timeout", str(round(spec.read_timeout_seconds * 1_000_000))])
        for key, value in spec.options.items():
            command.extend([f"-{key}", str(value)])
        command.extend(
            ["-i", spec.url, "-map", "0:v:0", "-an", "-sn", "-dn", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
        )
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        if self._process.stdout is None:
            self.close()
            raise DecoderOpenError("FFmpeg stdout is unavailable")
        self.hardware_accelerator = plan.accelerator

    @staticmethod
    def _probe_dimensions(spec: StreamSpec, ffprobe_binary: str) -> tuple[int, int]:
        command = [ffprobe_binary, "-v", "error"]
        if spec.transport in {"tcp", "udp", "udp_multicast"}:
            command.extend(["-rtsp_transport", spec.transport])
        command.extend(
            [
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height",
                "-of",
                "json",
                spec.url,
            ]
        )
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=spec.connect_timeout_seconds,
        )
        if result.returncode != 0:
            raise DecoderOpenError("FFprobe could not inspect the stream")
        try:
            stream = json.loads(result.stdout)["streams"][0]
            width, height = int(stream["width"]), int(stream["height"])
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise DecoderOpenError("FFprobe returned invalid stream dimensions") from exc
        if width <= 0 or height <= 0:
            raise DecoderOpenError("stream dimensions must be positive")
        return width, height

    def read(self) -> DecodedFrame | None:
        if self._closed or self._process.stdout is None:
            return None
        data = bytearray()
        while len(data) < self._frame_bytes:
            chunk = self._process.stdout.read(self._frame_bytes - len(data))
            if not chunk:
                return None
            data.extend(chunk)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self._shape).copy()
        return DecodedFrame(
            image=image,
            captured_at=time.time(),
            monotonic_at=time.monotonic(),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.stdout is not None:
            try:
                process.stdout.close()
            except OSError:
                pass
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=0.5)


@dataclass(frozen=True, slots=True)
class MainFrameSnapshot:
    camera_id: str
    seq: int
    captured_at: float
    monotonic_at: float
    frame: np.ndarray
    source_pts: float | None = None
    backend_name: str | None = None
    hardware_accelerator: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def crop_from_detector_bbox(
        self,
        bbox: tuple[int, int, int, int],
        detector_shape: tuple[int, ...],
        *,
        padding_ratio: float = 0.0,
        copy: bool = True,
    ) -> np.ndarray | None:
        """Map a detector-space box to this main frame and return a high-res crop."""

        if len(detector_shape) < 2:
            raise ValueError("detector_shape must contain height and width")
        detector_h, detector_w = int(detector_shape[0]), int(detector_shape[1])
        if detector_h <= 0 or detector_w <= 0:
            raise ValueError("detector dimensions must be positive")
        if padding_ratio < 0:
            raise ValueError("padding_ratio cannot be negative")

        x1, y1, x2, y2 = (float(value) for value in bbox)
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        pad_x = (x2 - x1) * padding_ratio
        pad_y = (y2 - y1) * padding_ratio
        main_h, main_w = self.frame.shape[:2]
        mapped_x1 = max(0, min(main_w, round((x1 - pad_x) * main_w / detector_w)))
        mapped_x2 = max(0, min(main_w, round((x2 + pad_x) * main_w / detector_w)))
        mapped_y1 = max(0, min(main_h, round((y1 - pad_y) * main_h / detector_h)))
        mapped_y2 = max(0, min(main_h, round((y2 + pad_y) * main_h / detector_h)))
        if mapped_x2 <= mapped_x1 or mapped_y2 <= mapped_y1:
            return None
        crop = self.frame[mapped_y1:mapped_y2, mapped_x1:mapped_x2]
        return crop.copy() if copy else crop


class LatestMainFrameCache:
    """A one-slot, thread-safe cache; every main frame replaces the previous one."""

    def __init__(self, camera_id: str, *, copy_frames: bool = True) -> None:
        self.camera_id = camera_id
        self.copy_frames = copy_frames
        self._latest: MainFrameSnapshot | None = None
        self._lock = threading.Lock()

    def put(
        self,
        seq: int,
        frame: DecodedFrame,
        *,
        backend_name: str | None = None,
        hardware_accelerator: str | None = None,
    ) -> MainFrameSnapshot:
        if frame.captured_at is None or frame.monotonic_at is None:
            raise ValueError("cache input must have normalized timestamps")
        image = frame.image.copy() if self.copy_frames else frame.image
        snapshot = MainFrameSnapshot(
            camera_id=self.camera_id,
            seq=int(seq),
            captured_at=float(frame.captured_at),
            monotonic_at=float(frame.monotonic_at),
            frame=image,
            source_pts=frame.source_pts,
            backend_name=backend_name,
            hardware_accelerator=hardware_accelerator,
            metadata=dict(frame.metadata),
        )
        with self._lock:
            self._latest = snapshot
        return snapshot

    def latest(
        self,
        *,
        reference_monotonic: float | None = None,
        maximum_age_seconds: float | None = None,
        copy: bool = False,
    ) -> MainFrameSnapshot | None:
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return None
        if maximum_age_seconds is not None:
            reference = time.monotonic() if reference_monotonic is None else reference_monotonic
            if reference - snapshot.monotonic_at > maximum_age_seconds:
                return None
        if copy:
            return replace(snapshot, frame=snapshot.frame.copy(), metadata=dict(snapshot.metadata))
        return snapshot

    def clear(self) -> None:
        with self._lock:
            self._latest = None


@dataclass(frozen=True, slots=True)
class StreamStats:
    connection_attempts: int = 0
    successful_connections: int = 0
    reconnects: int = 0
    decoded_frames: int = 0
    end_of_streams: int = 0
    errors: int = 0
    backend_name: str | None = None
    hardware_accelerator: str | None = None
    last_frame_at: float | None = None


@dataclass(frozen=True, slots=True)
class ProducerStats:
    main: StreamStats
    sub: StreamStats
    packets_emitted: int = 0
    packets_dropped_without_main: int = 0
    packets_dropped_stale_main: int = 0
    packets_dropped_by_admission: int = 0
    detector_due_packets: int = 0
    tracking_only_packets: int = 0
    cadence_provider_errors: int = 0
    sink_rejections: int = 0
    sink_errors: int = 0


@dataclass(slots=True)
class _MutableStreamStats:
    connection_attempts: int = 0
    successful_connections: int = 0
    reconnects: int = 0
    decoded_frames: int = 0
    end_of_streams: int = 0
    errors: int = 0
    backend_name: str | None = None
    hardware_accelerator: str | None = None
    last_frame_at: float | None = None

    def snapshot(self) -> StreamStats:
        return StreamStats(
            connection_attempts=self.connection_attempts,
            successful_connections=self.successful_connections,
            reconnects=self.reconnects,
            decoded_frames=self.decoded_frames,
            end_of_streams=self.end_of_streams,
            errors=self.errors,
            backend_name=self.backend_name,
            hardware_accelerator=self.hardware_accelerator,
            last_frame_at=self.last_frame_at,
        )


@dataclass(frozen=True, slots=True)
class StreamError:
    camera_id: str
    role: StreamRole
    stage: str
    exception_type: str
    message: str
    consecutive_failures: int


class DualStreamRTSPProducer:
    """Producer-only dual-stream reader for the central Engine V2 scheduler.

    The main stream is only cached. Sub-stream frames are emitted as detector
    packets paired with the latest sufficiently fresh main frame. No detector,
    OCR session, tracking, or inference work is owned by this class.
    """

    def __init__(
        self,
        config: RTSPProducerConfig,
        on_packet: Callable[[FramePacket], bool | None],
        *,
        decoder_factory: DecoderFactory | None = None,
        on_error: Callable[[StreamError], None] | None = None,
        cadence_provider: Callable[[str], ProducerCadencePolicy | None] | None = None,
        admission_controller: AdaptiveFrameAdmissionController | None = None,
        wall_clock: Callable[[], float] = time.time,
        monotonic_clock: Callable[[], float] = time.monotonic,
        random_value: Callable[[], float] | None = None,
        wait_for_stop: Callable[[threading.Event, float], bool] | None = None,
    ) -> None:
        self.config = config
        self.on_packet = on_packet
        self.on_error = on_error
        self._cadence_provider = cadence_provider
        self._cadence_provider_generation = 0
        self._cadence_provider_lock = threading.Lock()
        self.admission_controller = admission_controller or AdaptiveFrameAdmissionController()
        self.decoder_factory = decoder_factory or AutoDecoderFactory(
            backend_preference=config.backend_preference,
            hardware_decode=config.hardware_decode,
        )
        self.main_frame_cache = LatestMainFrameCache(
            config.camera_id,
            copy_frames=config.copy_main_frames,
        )
        self._wall_clock = wall_clock
        self._monotonic_clock = monotonic_clock
        self._random_value = random_value or (lambda: 0.5)
        self._wait_for_stop = wait_for_stop or (lambda event, delay: event.wait(delay))
        self._stop_event = threading.Event()
        # Serialize complete lifecycle operations. Without this guard a new
        # start() could replace ``_threads`` after an older stop() had copied
        # the previous threads but before it cleared them, leaving untracked
        # decoder threads running.
        self._operation_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._session_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._sessions: dict[StreamRole, DecoderSession] = {}
        self._threads: dict[StreamRole, threading.Thread] = {}
        # Initialized here so packets produced by a manually driven/test
        # instance still carry an identity. Every successful start() rotates it
        # again to mark a new decoder lifecycle for the central runtime.
        self._producer_epoch = uuid.uuid4().hex
        self._stream_stats = {
            StreamRole.MAIN: _MutableStreamStats(),
            StreamRole.SUB: _MutableStreamStats(),
        }
        self._seq = {StreamRole.MAIN: 0, StreamRole.SUB: 0}
        self._packets_emitted = 0
        self._packets_dropped_without_main = 0
        self._packets_dropped_stale_main = 0
        self._packets_dropped_by_admission = 0
        self._detector_due_packets = 0
        self._tracking_only_packets = 0
        self._cadence_provider_errors = 0
        self._sink_rejections = 0
        self._sink_errors = 0

    @property
    def running(self) -> bool:
        with self._lifecycle_lock:
            return bool(self._threads) and any(thread.is_alive() for thread in self._threads.values())

    @property
    def producer_epoch(self) -> str:
        return self._producer_epoch

    @property
    def cadence_policy(self) -> ProducerCadencePolicy | None:
        return self.admission_controller.policy

    def update_adaptive_cadence(self, policy: ProducerCadencePolicy | None) -> None:
        """Atomically update admission cadence without pausing either decoder."""

        self.admission_controller.update(policy)

    def set_cadence_provider(
        self,
        provider: Callable[[str], ProducerCadencePolicy | None] | None,
    ) -> None:
        """Replace the runtime callback; unbinding restores safe pass-through."""

        with self._cadence_provider_lock:
            self._cadence_provider = provider
            self._cadence_provider_generation += 1
        if provider is None:
            self.admission_controller.update(None)

    @property
    def stats(self) -> ProducerStats:
        with self._stats_lock:
            return ProducerStats(
                main=self._stream_stats[StreamRole.MAIN].snapshot(),
                sub=self._stream_stats[StreamRole.SUB].snapshot(),
                packets_emitted=self._packets_emitted,
                packets_dropped_without_main=self._packets_dropped_without_main,
                packets_dropped_stale_main=self._packets_dropped_stale_main,
                packets_dropped_by_admission=self._packets_dropped_by_admission,
                detector_due_packets=self._detector_due_packets,
                tracking_only_packets=self._tracking_only_packets,
                cadence_provider_errors=self._cadence_provider_errors,
                sink_rejections=self._sink_rejections,
                sink_errors=self._sink_errors,
            )

    def start(self) -> bool:
        """Start both readers. Returns False when already running."""

        with self._operation_lock:
            with self._lifecycle_lock:
                if any(thread.is_alive() for thread in self._threads.values()):
                    return False
                self._stop_event.clear()
                self.main_frame_cache.clear()
                self._producer_epoch = uuid.uuid4().hex
                self.admission_controller.reset_timing()
                # Sequence numbers intentionally survive stop/start. The
                # central runtime rejects out-of-order packets, so resetting a
                # long-running camera to one would make it look stale until it
                # caught up with its previous sequence.
                self._threads = {}
                for role in (StreamRole.MAIN, StreamRole.SUB):
                    thread = threading.Thread(
                        target=self._run_stream,
                        args=(self.config.stream_spec(role),),
                        name=f"anpr-v2-{self.config.camera_id}-{role.value}",
                        daemon=self.config.daemon_threads,
                    )
                    self._threads[role] = thread
                    thread.start()
        return True

    def stop(self, timeout_seconds: float = 5.0) -> bool:
        """Stop readers, close blocking decoders, and wait for thread exit."""

        if timeout_seconds < 0:
            raise ValueError("stop timeout cannot be negative")
        with self._operation_lock:
            self._stop_event.set()
            with self._session_lock:
                sessions = list(self._sessions.values())
            for session in sessions:
                try:
                    session.close()
                except Exception:
                    pass

            deadline = self._monotonic_clock() + timeout_seconds
            current = threading.current_thread()
            with self._lifecycle_lock:
                threads = list(self._threads.values())
            for thread in threads:
                if thread is current:
                    continue
                remaining = max(0.0, deadline - self._monotonic_clock())
                thread.join(remaining)
            stopped = all(thread is current or not thread.is_alive() for thread in threads)
            if stopped:
                with self._lifecycle_lock:
                    self._threads.clear()
            return stopped

    def __enter__(self) -> DualStreamRTSPProducer:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run_stream(self, spec: StreamSpec) -> None:
        consecutive_failures = 0
        while not self._stop_event.is_set():
            session: DecoderSession | None = None
            decoded_any = False
            self._update_stream_stats(spec.role, connection_attempts=1)
            try:
                session = self.decoder_factory.open(spec)
                if self._stop_event.is_set():
                    session.close()
                    break
                with self._session_lock:
                    self._sessions[spec.role] = session
                self._set_connection(spec.role, session)

                while not self._stop_event.is_set():
                    decoded = session.read()
                    if decoded is None:
                        self._update_stream_stats(spec.role, end_of_streams=1)
                        break
                    decoded_any = True
                    consecutive_failures = 0
                    frame = self._normalize_timestamps(decoded)
                    seq = self._next_seq(spec.role)
                    self._record_frame(spec.role, frame.captured_at)
                    if spec.role is StreamRole.MAIN:
                        self.main_frame_cache.put(
                            seq,
                            frame,
                            backend_name=session.backend_name,
                            hardware_accelerator=session.hardware_accelerator,
                        )
                    else:
                        self._emit_detector_packet(seq, frame, session)
            except Exception as exc:
                # Decoder close is allowed to interrupt a blocking read. That is
                # an expected part of stop(), not a camera/decoder failure.
                if not self._stop_event.is_set():
                    self._update_stream_stats(spec.role, errors=1)
                    self._notify_error(
                        spec,
                        "open" if session is None else "read",
                        exc,
                        consecutive_failures + 1,
                    )
            finally:
                with self._session_lock:
                    if session is not None and self._sessions.get(spec.role) is session:
                        del self._sessions[spec.role]
                if session is not None:
                    try:
                        session.close()
                    except Exception:
                        pass

            if self._stop_event.is_set():
                break
            consecutive_failures = 1 if decoded_any else consecutive_failures + 1
            delay = self.config.reconnect.delay_for(consecutive_failures, self._random_value())
            self._update_stream_stats(spec.role, reconnects=1)
            if self._wait_for_stop(self._stop_event, delay):
                break

    def _normalize_timestamps(self, frame: DecodedFrame) -> DecodedFrame:
        captured_at = self._wall_clock() if frame.captured_at is None else float(frame.captured_at)
        monotonic_at = (
            self._monotonic_clock() if frame.monotonic_at is None else float(frame.monotonic_at)
        )
        if captured_at == frame.captured_at and monotonic_at == frame.monotonic_at:
            return frame
        return replace(frame, captured_at=captured_at, monotonic_at=monotonic_at)

    def _next_seq(self, role: StreamRole) -> int:
        with self._stats_lock:
            self._seq[role] += 1
            return self._seq[role]

    def _emit_detector_packet(
        self,
        seq: int,
        sub_frame: DecodedFrame,
        session: DecoderSession,
    ) -> None:
        assert sub_frame.captured_at is not None
        assert sub_frame.monotonic_at is not None
        self._refresh_cadence_policy()
        main = self.main_frame_cache.latest()
        if main is None:
            with self._stats_lock:
                self._packets_dropped_without_main += 1
            if self.config.require_main_frame:
                return
        else:
            main_age = sub_frame.monotonic_at - main.monotonic_at
            # Independent RTSP readers can advance at different rates. Reject
            # both an old main frame and a main frame implausibly far in the
            # future; accepting a large negative age can crop a different car.
            if abs(main_age) > self.config.maximum_main_frame_age_seconds:
                with self._stats_lock:
                    self._packets_dropped_stale_main += 1
                if self.config.require_main_frame:
                    return
                main = None

        try:
            admission = self.admission_controller.decide(sub_frame.monotonic_at)
        except (TypeError, ValueError) as exc:
            # A malformed external timestamp/policy must not tear down a healthy
            # decoder. Fall back to pass-through and surface the contract error.
            self._record_cadence_error(exc)
            admission = FrameAdmissionDecision(True, False)
        if not admission.admit:
            with self._stats_lock:
                self._packets_dropped_by_admission += 1
            return

        full_resolution_frame = main.frame if main is not None else sub_frame.image
        metadata: dict[str, Any] = {
            "producer_epoch": self._producer_epoch,
            "stream_role": StreamRole.SUB.value,
            "sub_source_pts": sub_frame.source_pts,
            "sub_monotonic_ts": sub_frame.monotonic_at,
            "sub_backend": session.backend_name,
            "sub_hardware_accelerator": session.hardware_accelerator,
            "main_fallback_to_sub": main is None,
            **dict(sub_frame.metadata),
        }
        if admission.adaptive and admission.policy is not None:
            policy = admission.policy
            metadata.update(
                {
                    "adaptive_admission": True,
                    "detector_due": admission.detector_due,
                    "producer_activity": policy.activity.value,
                    "producer_target_detector_fps": policy.target_detector_fps,
                    "producer_effective_detector_fps": policy.effective_detector_fps,
                    "producer_target_admission_fps": policy.target_admission_fps,
                    "tracking_frames_between_detection": (
                        policy.tracking_frames_between_detection
                    ),
                }
            )
        if main is not None:
            metadata.update(
                {
                    "main_seq": main.seq,
                    "main_ts": main.captured_at,
                    "main_monotonic_ts": main.monotonic_at,
                    "main_source_pts": main.source_pts,
                    "main_age_seconds": sub_frame.monotonic_at - main.monotonic_at,
                    "main_detector_skew_ms": (
                        sub_frame.monotonic_at - main.monotonic_at
                    ) * 1_000.0,
                    "main_backend": main.backend_name,
                    "main_hardware_accelerator": main.hardware_accelerator,
                }
            )
        packet = FramePacket(
            camera_id=self.config.camera_id,
            seq=seq,
            ts=sub_frame.captured_at,
            frame=full_resolution_frame,
            detector_frame=sub_frame.image,
            metadata=metadata,
        )
        try:
            accepted = self.on_packet(packet)
        except Exception as exc:
            if admission.adaptive and admission.detector_due:
                self.admission_controller.detector_unaccepted()
            with self._stats_lock:
                self._sink_errors += 1
            self._notify_error(
                self.config.stream_spec(StreamRole.SUB),
                "sink",
                exc,
                0,
            )
            return
        if accepted is False and admission.adaptive and admission.detector_due:
            self.admission_controller.detector_unaccepted()
        with self._stats_lock:
            self._packets_emitted += 1
            if admission.adaptive:
                if admission.detector_due:
                    self._detector_due_packets += 1
                else:
                    self._tracking_only_packets += 1
            if accepted is False and (not admission.adaptive or admission.detector_due):
                self._sink_rejections += 1

    def _refresh_cadence_policy(self) -> None:
        with self._cadence_provider_lock:
            provider = self._cadence_provider
            generation = self._cadence_provider_generation
        if provider is None:
            return
        try:
            policy = provider(self.config.camera_id)
        except Exception as exc:
            self._record_cadence_error(exc)
            return
        # Do not let a slow callback resurrect a policy after another thread
        # has unbound or replaced that provider.
        with self._cadence_provider_lock:
            if (
                generation != self._cadence_provider_generation
                or provider is not self._cadence_provider
            ):
                return
            self.admission_controller.update(policy)

    def _record_cadence_error(self, exc: Exception) -> None:
        with self._stats_lock:
            self._cadence_provider_errors += 1
        self._notify_error(
            self.config.stream_spec(StreamRole.SUB),
            "cadence",
            exc,
            0,
        )

    def _set_connection(self, role: StreamRole, session: DecoderSession) -> None:
        with self._stats_lock:
            stats = self._stream_stats[role]
            stats.successful_connections += 1
            stats.backend_name = session.backend_name
            stats.hardware_accelerator = session.hardware_accelerator

    def _record_frame(self, role: StreamRole, captured_at: float | None) -> None:
        with self._stats_lock:
            stats = self._stream_stats[role]
            stats.decoded_frames += 1
            stats.last_frame_at = captured_at

    def _update_stream_stats(self, role: StreamRole, **increments: int) -> None:
        with self._stats_lock:
            stats = self._stream_stats[role]
            for name, increment in increments.items():
                setattr(stats, name, getattr(stats, name) + int(increment))

    def _notify_error(
        self,
        spec: StreamSpec,
        stage: str,
        exc: Exception,
        consecutive_failures: int,
    ) -> None:
        if self.on_error is None:
            return
        message = str(exc)
        for url in (spec.url, self.config.main_url, self.config.sub_url):
            message = message.replace(url, "<redacted-stream-url>")
        error = StreamError(
            camera_id=spec.camera_id,
            role=spec.role,
            stage=stage,
            exception_type=type(exc).__name__,
            message=message,
            consecutive_failures=consecutive_failures,
        )
        try:
            self.on_error(error)
        except Exception:
            pass


__all__ = [
    "AdaptiveFrameAdmissionController",
    "AutoDecoderFactory",
    "DecodedFrame",
    "DecoderFactory",
    "DecoderOpenError",
    "DecoderSession",
    "DualStreamRTSPProducer",
    "HardwareDecodePlan",
    "FrameAdmissionDecision",
    "LatestMainFrameCache",
    "MainFrameSnapshot",
    "ProducerStats",
    "ProducerActivity",
    "ProducerCadencePolicy",
    "RTSPProducerConfig",
    "ReconnectPolicy",
    "StreamError",
    "StreamRole",
    "StreamSpec",
    "StreamStats",
    "probe_ffmpeg_hardware_accelerators",
    "select_hardware_decode",
]
