"""Concrete offline adapters for same-input V1/V2 ANPR accuracy runs.

This module deliberately lives outside both production ANPR implementations.
The legacy adapter imports :mod:`app.ai.video_test` only when a sample is run;
the V2 adapter owns one shared detector/OCR bundle for its whole lifetime.
Neither adapter changes production routing or writes benchmark media into the
application's event archive.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import importlib
import inspect
import math
from pathlib import Path
import platform
import re
import tempfile
from typing import Any

import cv2
import numpy as np

from app.engine_v2.factory import SharedModelBundleConfig, build_engine_v2
from app.engine_v2.model_adapters import (
    CTCPlateOCRConfig,
    YOLOPlateDetectorConfig,
)
from app.engine_v2.runtime import EngineV2Config
from app.engine_v2.types import FramePacket, PlateEvent


_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".tif", ".tiff"}
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _resolved_input(sample: Mapping[str, Any]) -> tuple[Path, str]:
    input_value = sample.get("input")
    if not isinstance(input_value, Mapping):
        raise ValueError("accuracy sample input must be an object")
    requested_window = {
        key: input_value.get(key)
        for key in ("start_ms", "end_ms")
        if input_value.get(key) is not None
    }
    if requested_window:
        raise ValueError(
            "built-in V1/V2 accuracy adapters do not silently slice input.start_ms/end_ms; "
            "use a pre-clipped content-addressed media file so both engines receive exactly "
            f"the same bytes (requested {requested_window})"
        )
    raw_path = str(input_value.get("resolved_path", input_value.get("path", ""))).strip()
    if not raw_path:
        raise ValueError("accuracy sample input path is required")
    if "://" in raw_path:
        raise ValueError("built-in accuracy adapters accept local files only")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"accuracy input does not exist: {path}")
    media_type = str(input_value.get("media_type", "")).strip().lower()
    if not media_type:
        media_type = "image" if path.suffix.lower() in _IMAGE_SUFFIXES else "video"
    return path, media_type


def _safe_sample_id(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value or "sample")).strip("-.")
    return cleaned[:64] or "sample"


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return max(0.0, min(1.0, parsed))


@dataclass(frozen=True, slots=True)
class LegacyVideoAccuracyConfig:
    """Explicit arguments passed to the unchanged legacy ``process_video``."""

    frame_step: int = 1
    max_events: int = 100
    min_confidence: float = 0.20
    duplicate_seconds: float = 2.5
    detector_variant: str = "yolo11n"
    roi: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")
        if self.max_events < 1:
            raise ValueError("max_events must be at least 1")
        if not 0.0 <= float(self.min_confidence) <= 1.0:
            raise ValueError("min_confidence must be within 0..1")
        if self.duplicate_seconds < 0:
            raise ValueError("duplicate_seconds cannot be negative")
        if not self.detector_variant.strip():
            raise ValueError("detector_variant is required")
        if self.roi is not None and len(self.roi) != 4:
            raise ValueError("roi must contain x,y,width,height percentages")


class LegacyVideoAccuracyAdapter:
    """Thin, lazy adapter over the unchanged production video test function."""

    def __init__(
        self,
        config: LegacyVideoAccuracyConfig | None = None,
        *,
        name: str = "legacy-process-video",
        _process_video: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or LegacyVideoAccuracyConfig()
        self.adapter_name = name
        self._process_video = _process_video
        self._legacy_model_identity: dict[str, Any] | None = None
        self._has_predicted = False

    def _processor(self) -> Callable[..., Any]:
        if self._process_video is None:
            # Keep imports of all legacy model/database modules outside module
            # import and adapter construction. This also ensures the generic
            # benchmark CLI does not initialize V1 unless explicitly selected.
            module = importlib.import_module("app.ai.video_test")
            processor = getattr(module, "process_video", None)
            if not callable(processor):
                raise TypeError("app.ai.video_test.process_video is not callable")
            self._process_video = processor
        return self._process_video

    def predict(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        path, media_type = _resolved_input(sample)
        sample_id = _safe_sample_id(sample.get("id"))
        with tempfile.TemporaryDirectory(prefix=f"bcvision-v1-{sample_id}-") as temporary:
            root = Path(temporary)
            plate_dir = root / "plates"
            snapshot_dir = root / "snapshots"
            info, raw_events = self._processor()(
                video_path=str(path),
                plate_dir=plate_dir,
                snapshot_dir=snapshot_dir,
                frame_step=self.config.frame_step,
                max_events=self.config.max_events,
                min_confidence=self.config.min_confidence,
                duplicate_seconds=self.config.duplicate_seconds,
                roi=self.config.roi,
                include_candidate_shadow=False,
                detector_variant=self.config.detector_variant,
            )
            events = self._timestamped_events(raw_events, info)
            self._has_predicted = True

        confidences = [event["confidence"] for event in events if "confidence" in event]
        return {
            "plate": events[0]["plate"] if events else "",
            "confidence": max(confidences) if confidences else None,
            "accepted": bool(events),
            "events": events,
            "run_metadata": {
                "sample_id": str(sample.get("id", "")),
                "input_sha256": _sha256_file(path),
                "media_type": media_type,
                "legacy_info": _jsonable(info),
                "temporary_outputs_persisted": False,
            },
            "adapter_config": _jsonable(self.config),
        }

    @staticmethod
    def _timestamped_events(raw_events: Any, info: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
            raise TypeError("legacy process_video events must be an array")
        info_value = info if isinstance(info, Mapping) else {}
        try:
            fps = float(info_value.get("fps", 0.0))
        except (TypeError, ValueError):
            fps = 0.0
        timestamped: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_events):
            if not isinstance(raw, Mapping):
                continue
            # Legacy process_video deliberately persists capture-only /
            # unreadable rows for operator review. They are not accepted plate
            # events and must not become false-positive predictions in an
            # accuracy comparison.
            if (
                bool(raw.get("capture_only"))
                or bool(raw.get("unreadable_final"))
                or ("valid" in raw and not bool(raw.get("valid")))
            ):
                continue
            plate = str(raw.get("plate_norm") or raw.get("plate") or "").strip()
            if not plate:
                continue
            timestamp_ms: float | None = None
            source = "unavailable"
            # process_video stores video_second rounded to two decimals, while
            # its 1-based frame counter and FPS retain the exact timestamp it
            # used internally. Prefer the lossless frame/FPS reconstruction.
            if fps > 0:
                try:
                    frame_number = float(raw.get("frame"))
                    if math.isfinite(frame_number) and frame_number >= 0:
                        timestamp_ms = frame_number * 1000.0 / fps
                        source = "legacy-frame/fps"
                except (TypeError, ValueError):
                    pass
            if timestamp_ms is None:
                try:
                    video_second = float(raw.get("video_second"))
                    if math.isfinite(video_second) and video_second >= 0:
                        timestamp_ms = video_second * 1000.0
                        source = "legacy-video_second"
                except (TypeError, ValueError):
                    pass
            event: dict[str, Any] = {
                "plate": plate,
                "legacy_event_index": index,
                "timestamp_source": source,
            }
            if timestamp_ms is not None:
                event["timestamp_ms"] = round(timestamp_ms, 6)
            confidence = _probability(raw.get("confidence"))
            if confidence is not None:
                event["confidence"] = confidence
            for field in ("frame", "track_id", "bbox", "engine_lane", "detector_variant"):
                if raw.get(field) is not None:
                    event[field] = _jsonable(raw[field])
            timestamped.append(event)
        return timestamped

    def reproducibility_metadata(self) -> dict[str, Any]:
        # Hashing model files belongs to report/evidence construction, not the
        # measured per-sample inference latency. The CLI calls this after all
        # predictions. Paths are captured only after V1 has actually run.
        if self._has_predicted and self._legacy_model_identity is None:
            self._legacy_model_identity = self._capture_model_identity()
        return {
            "schema": "bcvision.anpr.accuracy-adapter-metadata/v1",
            "adapter": "legacy-process-video",
            "implementation": "app.ai.video_test.process_video",
            "settings": _jsonable(self.config),
            "temporary_output_policy": "isolated-per-sample-and-deleted",
            "model_identity": self._legacy_model_identity
            or {
                "detector_variant": self.config.detector_variant,
                "files": [],
                "capture_status": "pending-first-prediction",
            },
            "decode": {
                "library": "OpenCV VideoCapture through unchanged VideoTester",
                "backend": "legacy OpenCV default",
            },
        }

    def _capture_model_identity(self) -> dict[str, Any]:
        """Resolve model-manager paths without preparing/downloading models."""

        try:
            manager = importlib.import_module("app.ai.model_manager")
        except Exception as exc:
            return {
                "detector_variant": self.config.detector_variant,
                "files": [],
                "capture_status": f"model-manager-unavailable:{type(exc).__name__}:{exc}",
            }

        files: list[dict[str, Any]] = []

        def record(
            role: str,
            raw_path: Any,
            *,
            expected_sha256: Any = None,
            expected_size: Any = None,
        ) -> None:
            if raw_path is None or not str(raw_path).strip():
                return
            path = Path(raw_path).expanduser().resolve()
            exists = path.is_file()
            actual_sha256 = _sha256_file(path) if exists else None
            expected_digest = str(expected_sha256 or "").strip().lower() or None
            try:
                normalized_expected_size = (
                    None if expected_size is None else int(expected_size)
                )
            except (TypeError, ValueError):
                normalized_expected_size = None
            files.append(
                {
                    "role": role,
                    "path": str(path),
                    "exists": exists,
                    "size": path.stat().st_size if exists else None,
                    "sha256": actual_sha256,
                    "expected_sha256": expected_digest,
                    "expected_size": normalized_expected_size,
                    "matches_expected_sha256": (
                        None
                        if actual_sha256 is None or expected_digest is None
                        else actual_sha256.lower() == expected_digest
                    ),
                }
            )

        errors: list[str] = []
        try:
            spec = manager.detector_variant_spec(self.config.detector_variant)
            record(
                "detector-selected",
                spec.get("path"),
                expected_sha256=spec.get("sha256"),
                expected_size=spec.get("size"),
            )
        except Exception as exc:
            errors.append(f"detector-selected:{type(exc).__name__}:{exc}")
        try:
            record(
                "detector-fallback",
                manager.detector_fallback_path(),
                expected_sha256=getattr(manager, "DETECTOR_FALLBACK_SHA256", None),
                expected_size=getattr(manager, "DETECTOR_FALLBACK_SIZE", None),
            )
        except Exception as exc:
            errors.append(f"detector-fallback:{type(exc).__name__}:{exc}")
        try:
            crnn_path, crnn_digest, crnn_size = manager.active_crnn_model()
            record(
                "ocr-crnn-active",
                crnn_path,
                expected_sha256=crnn_digest,
                expected_size=crnn_size,
            )
        except Exception as exc:
            errors.append(f"ocr-crnn-active:{type(exc).__name__}:{exc}")
        for role, path_function, digest_name, size_name in (
            ("ocr-cnn-fallback", "cnn_path", "CNN_SHA256", "CNN_SIZE"),
            ("ocr-hezar-primary", "hezar_path", "HEZAR_ONNX_SHA256", "HEZAR_ONNX_SIZE"),
        ):
            try:
                record(
                    role,
                    getattr(manager, path_function)(),
                    expected_sha256=getattr(manager, digest_name, None),
                    expected_size=getattr(manager, size_name, None),
                )
            except Exception as exc:
                errors.append(f"{role}:{type(exc).__name__}:{exc}")
        return {
            "detector_variant": self.config.detector_variant,
            "files": files,
            "execution_provider_contract": "CPUExecutionProvider",
            "device_contract": "CPU",
            "capture_status": "resolved-read-only-model-manager-paths",
            "errors": errors,
        }

    def close(self) -> None:
        # The unchanged process_video function closes its VideoTester in a
        # finally block and all temporary directories are scoped per predict.
        return None


@dataclass(frozen=True, slots=True)
class V2OfflineAccuracyConfig:
    detector_model: str | Path
    ocr_model: str | Path
    backend: str = "auto"
    device: str = "AUTO"
    detector_frame_size: tuple[int, int] = (640, 360)  # width, height
    detector_input_size: tuple[int, int] = (320, 320)  # width, height
    detector_confidence: float = 0.25
    min_ocr_confidence: float = 0.55
    duplicate_seconds: float = 2.5
    frame_step: int = 1
    max_frames: int | None = None
    fallback_fps: float = 25.0
    opencv_threads: int = 1
    allow_inference_fallback: bool = True
    allow_capture_backend_fallback: bool = False
    detector_intra_op_threads: int | None = None
    ocr_intra_op_threads: int | None = None
    inter_op_threads: int | None = 1

    def __post_init__(self) -> None:
        if not str(self.detector_model).strip() or not str(self.ocr_model).strip():
            raise ValueError("detector_model and ocr_model are required")
        if self.backend.strip().lower() not in {"auto", "openvino", "onnxruntime"}:
            raise ValueError("backend must be auto, openvino, or onnxruntime")
        for name, size in {
            "detector_frame_size": self.detector_frame_size,
            "detector_input_size": self.detector_input_size,
        }.items():
            if len(size) != 2 or min(int(value) for value in size) < 1:
                raise ValueError(f"{name} must contain positive width,height values")
        for name, value in {
            "detector_confidence": self.detector_confidence,
            "min_ocr_confidence": self.min_ocr_confidence,
        }.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within 0..1")
        if self.duplicate_seconds < 0:
            raise ValueError("duplicate_seconds cannot be negative")
        if self.frame_step < 1:
            raise ValueError("frame_step must be at least 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive when supplied")
        if self.fallback_fps <= 0:
            raise ValueError("fallback_fps must be positive")
        if self.opencv_threads < 1:
            raise ValueError("opencv_threads must be positive")


@dataclass(frozen=True, slots=True)
class _DecodedFrame:
    index: int
    timestamp_seconds: float
    timestamp_source: str
    frame: np.ndarray


class EngineV2OfflineAccuracyAdapter:
    """Offline same-input adapter backed by one shared Engine V2 model bundle."""

    def __init__(
        self,
        config: V2OfflineAccuracyConfig,
        *,
        name: str = "engine-v2-offline",
        _bundle_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self.adapter_name = name
        self._closed = False
        self._current_events: list[PlateEvent] = []
        self._event_keys: set[tuple[Any, ...]] = set()
        self._decode_runs: list[dict[str, Any]] = []
        self._previous_opencv_threads = (
            int(cv2.getNumThreads()) if callable(getattr(cv2, "getNumThreads", None)) else None
        )
        cv2.setNumThreads(int(config.opencv_threads))
        if callable(getattr(cv2, "setRNGSeed", None)):
            cv2.setRNGSeed(0)

        detector_path = Path(config.detector_model).resolve()
        ocr_path = Path(config.ocr_model).resolve()
        if not detector_path.is_file():
            self._restore_opencv_threads()
            raise FileNotFoundError(f"V2 detector model does not exist: {detector_path}")
        if not ocr_path.is_file():
            self._restore_opencv_threads()
            raise FileNotFoundError(f"V2 OCR model does not exist: {ocr_path}")
        try:
            self._model_files = (
                {
                    "role": "detector",
                    "path": str(detector_path),
                    "sha256": _sha256_file(detector_path),
                },
                {
                    "role": "ocr",
                    "path": str(ocr_path),
                    "sha256": _sha256_file(ocr_path),
                },
            )
        except Exception:
            self._restore_opencv_threads()
            raise

        detector_width, detector_height = config.detector_input_size
        self._model_config = SharedModelBundleConfig(
            detector_model=detector_path,
            ocr_model=ocr_path,
            backend=config.backend,
            device=config.device,
            detector_intra_op_threads=config.detector_intra_op_threads,
            ocr_intra_op_threads=config.ocr_intra_op_threads,
            inter_op_threads=config.inter_op_threads,
            allow_fallback=config.allow_inference_fallback,
            detector=YOLOPlateDetectorConfig(
                input_size=(detector_height, detector_width),
                confidence_threshold=config.detector_confidence,
                num_classes=1,
            ),
            ocr=CTCPlateOCRConfig(),
        )
        self._engine_config = EngineV2Config(
            idle_stride=1,
            active_stride=1,
            min_detector_confidence=config.detector_confidence,
            min_ocr_confidence=config.min_ocr_confidence,
            load_control_enabled=False,
            same_camera_duplicate_seconds=config.duplicate_seconds,
        )
        factory = _bundle_factory or build_engine_v2
        try:
            self._bundle = factory(
                self._model_config,
                self._engine_config,
                self._capture_event,
            )
        except Exception:
            self._restore_opencv_threads()
            raise
        self.engine = self._bundle.engine
        models = getattr(self._bundle, "models", None)
        summary = getattr(models, "summary", None)
        try:
            self._shared_model_summary = _jsonable(summary()) if callable(summary) else None
        except Exception:
            try:
                self._bundle.close()
            finally:
                self._restore_opencv_threads()
            raise

    def predict(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        if self._closed:
            raise RuntimeError("Engine V2 accuracy adapter is closed")
        path, media_type = _resolved_input(sample)
        camera_id = "accuracy-" + _safe_sample_id(sample.get("id"))
        self.engine.reset_runtime_state()
        self._current_events = []
        self._event_keys = set()
        decode_metadata: dict[str, Any] = {
            "sample_id": str(sample.get("id", "")),
            "media_type": media_type,
            "timestamp_sources": {},
        }
        decoded_count = 0
        submitted_count = 0
        last_sequence = 0
        last_timestamp = 0.0
        baseline_primed = False

        for decoded in self._decode(path, media_type, decode_metadata):
            decoded_count += 1
            if decoded.index % self.config.frame_step != 0:
                continue
            if not baseline_primed:
                self._prime_motion_baseline(camera_id, decoded.frame, decoded.timestamp_seconds)
                baseline_primed = True
            sequence = decoded.index + 1
            detector_frame = self._detector_frame(decoded.frame)
            accepted = self.engine.submit_frame(
                FramePacket(
                    camera_id=camera_id,
                    seq=sequence,
                    ts=decoded.timestamp_seconds,
                    frame=np.ascontiguousarray(decoded.frame),
                    detector_frame=detector_frame,
                    metadata={
                        "offline_accuracy": True,
                        "source_frame_index": decoded.index,
                        "timestamp_source": decoded.timestamp_source,
                        "detector_frame_derived_from_main": True,
                    },
                )
            )
            if accepted:
                self.engine.process_available(limit=256)
            submitted_count += 1
            last_sequence = sequence
            last_timestamp = decoded.timestamp_seconds
            if self.config.max_frames is not None and submitted_count >= self.config.max_frames:
                decode_metadata["stopped_at_max_frames"] = True
                break

        if decoded_count == 0:
            raise ValueError(f"OpenCV decoded no frames from {path}")
        # Finish ordinary detector jobs before EOF forces the already-harvested
        # candidates into OCR. finalize_camera intentionally never fabricates a
        # detector frame and may discard a still-pending camera job.
        self._collect_returned(self.engine.process_available(limit=4096))
        eof_finalizer_used = self._finalize_camera(
            camera_id,
            final_seq=last_sequence,
            final_ts=last_timestamp,
        )
        # The finalizer is expected to drain its forced OCR work. This extra
        # public drain is conservative for older V2 snapshots without it.
        self._collect_returned(self.engine.process_available(limit=4096))

        decode_metadata["frames_decoded"] = decoded_count
        decode_metadata["frames_submitted"] = submitted_count
        decode_metadata["eof_finalizer_used"] = eof_finalizer_used
        decode_metadata["input_sha256"] = _sha256_file(path)
        decode_metadata["timestamp_sources"] = dict(
            sorted(Counter(decode_metadata.pop("_timestamp_source_rows", [])).items())
        )
        self._decode_runs.append(_jsonable(decode_metadata))
        events = sorted(
            (self._event_mapping(event) for event in self._current_events),
            key=lambda event: (
                float(event.get("timestamp_ms", 0.0)),
                int(event.get("frame_seq", 0)),
                str(event.get("plate", "")),
            ),
        )
        confidences = [event["confidence"] for event in events if "confidence" in event]
        return {
            "plate": events[0]["plate"] if events else "",
            "confidence": max(confidences) if confidences else None,
            "accepted": bool(events),
            "events": events,
            "run_metadata": _jsonable(decode_metadata),
            "adapter_config": _jsonable(self.config),
        }

    def _prime_motion_baseline(
        self,
        camera_id: str,
        main_frame: np.ndarray,
        timestamp_seconds: float,
    ) -> None:
        detector_width, detector_height = self.config.detector_frame_size
        baseline = np.zeros((detector_height, detector_width, 3), dtype=np.uint8)
        self.engine.submit_frame(
            FramePacket(
                camera_id=camera_id,
                seq=0,
                ts=max(0.0, timestamp_seconds),
                frame=np.ascontiguousarray(main_frame),
                detector_frame=baseline,
                metadata={
                    "offline_accuracy": True,
                    "accuracy_motion_baseline": "zero-detector-frame-no-ai",
                    "detector_frame_derived_from_main": False,
                },
            )
        )
        self.engine.process_available(limit=8)

    def _detector_frame(self, main_frame: np.ndarray) -> np.ndarray:
        width, height = self.config.detector_frame_size
        main_height, main_width = main_frame.shape[:2]
        interpolation = (
            cv2.INTER_AREA if width <= main_width and height <= main_height else cv2.INTER_LINEAR
        )
        return np.ascontiguousarray(
            cv2.resize(main_frame, (width, height), interpolation=interpolation)
        )

    def _decode(
        self,
        path: Path,
        media_type: str,
        metadata: dict[str, Any],
    ) -> Iterator[_DecodedFrame]:
        is_image = media_type in {"image", "still", "photo"} or (
            media_type not in {"video", "clip"} and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if is_image:
            encoded = np.fromfile(path, dtype=np.uint8)
            frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if frame is None or not frame.size:
                raise ValueError(f"OpenCV could not decode image: {path}")
            metadata.update(
                {
                    "decode_backend": "opencv-imdecode",
                    "hardware_acceleration": "not-applicable",
                    "fps": None,
                }
            )
            metadata.setdefault("_timestamp_source_rows", []).append("image-zero")
            yield _DecodedFrame(0, 0.0, "image-zero", np.ascontiguousarray(frame))
            return

        capture, open_metadata = self._open_video_capture(path)
        metadata.update(open_metadata)
        try:
            raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
            fps = raw_fps if math.isfinite(raw_fps) and raw_fps > 0 else self.config.fallback_fps
            metadata["reported_fps"] = raw_fps if math.isfinite(raw_fps) else None
            metadata["fps"] = fps
            frame_index = 0
            previous_ms = -1.0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame is None or not getattr(frame, "size", 0):
                    raise ValueError(f"OpenCV returned an empty video frame at index {frame_index}")
                fallback_ms = frame_index * 1000.0 / fps
                try:
                    position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                except Exception:
                    position_ms = float("nan")
                if (
                    math.isfinite(position_ms)
                    and position_ms >= 0.0
                    and (frame_index == 0 or position_ms > previous_ms + 1e-6)
                ):
                    timestamp_ms = position_ms
                    timestamp_source = "opencv-pos-msec-pts"
                else:
                    timestamp_ms = max(fallback_ms, previous_ms + 1000.0 / fps)
                    timestamp_source = "frame-index/fps-fallback"
                previous_ms = timestamp_ms
                metadata.setdefault("_timestamp_source_rows", []).append(timestamp_source)
                yield _DecodedFrame(
                    frame_index,
                    timestamp_ms / 1000.0,
                    timestamp_source,
                    np.ascontiguousarray(frame),
                )
                frame_index += 1
        finally:
            capture.release()

    def _open_video_capture(self, path: Path) -> tuple[Any, dict[str, Any]]:
        api = int(getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY))
        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_NONE"):
            params.extend(
                [int(cv2.CAP_PROP_HW_ACCELERATION), int(cv2.VIDEO_ACCELERATION_NONE)]
            )
        if hasattr(cv2, "CAP_PROP_N_THREADS"):
            params.extend([int(cv2.CAP_PROP_N_THREADS), 1])
        try:
            capture = cv2.VideoCapture(str(path), api, params)
        except (TypeError, cv2.error):
            capture = cv2.VideoCapture(str(path), api)
        fallback_used = False
        if not capture.isOpened() and self.config.allow_capture_backend_fallback:
            capture.release()
            capture = cv2.VideoCapture(str(path), cv2.CAP_ANY)
            fallback_used = True
        if not capture.isOpened():
            capture.release()
            raise ValueError(
                "OpenCV FFmpeg software decoder could not open video; "
                "capture fallback is disabled for deterministic accuracy runs: "
                f"{path}"
            )
        backend_name = "unknown"
        get_backend_name = getattr(capture, "getBackendName", None)
        if callable(get_backend_name):
            try:
                backend_name = str(get_backend_name())
            except Exception:
                pass
        return capture, {
            "decode_backend": backend_name,
            "requested_backend": "FFMPEG",
            "capture_backend_fallback_used": fallback_used,
            "hardware_acceleration": "disabled/software",
            "decoder_threads_requested": 1,
        }

    def _finalize_camera(self, camera_id: str, *, final_seq: int, final_ts: float) -> bool:
        finalizer = getattr(self.engine, "finalize_camera", None)
        if not callable(finalizer):
            return False
        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(finalizer)
            parameters = signature.parameters
            accepts_keywords = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            if accepts_keywords or "final_seq" in parameters:
                kwargs["final_seq"] = final_seq
            if accepts_keywords or "final_ts" in parameters:
                kwargs["final_ts"] = final_ts
        except (TypeError, ValueError):
            # Some extension-backed callables do not expose a signature. The
            # stable V2 API accepts these keywords; let its own exception remain
            # visible instead of silently skipping EOF finalization.
            kwargs = {"final_seq": final_seq, "final_ts": final_ts}
        self._collect_returned(finalizer(camera_id, **kwargs))
        return True

    def _capture_event(self, event: PlateEvent) -> None:
        key = self._event_key(event)
        if key not in self._event_keys:
            self._event_keys.add(key)
            self._current_events.append(event)

    def _collect_returned(self, value: Any) -> None:
        if value is None:
            return
        events = value if isinstance(value, (list, tuple)) else (value,)
        for event in events:
            if isinstance(event, PlateEvent) or (
                hasattr(event, "text") and hasattr(event, "ts") and hasattr(event, "frame_seq")
            ):
                self._capture_event(event)

    @staticmethod
    def _event_key(event: Any) -> tuple[Any, ...]:
        return (
            getattr(event, "episode_id", None),
            str(getattr(event, "text", "")),
            int(getattr(event, "frame_seq", 0)),
            round(float(getattr(event, "ts", 0.0)), 9),
        )

    @staticmethod
    def _event_mapping(event: Any) -> dict[str, Any]:
        confidence = _probability(getattr(event, "confidence", None))
        mapped: dict[str, Any] = {
            "plate": str(getattr(event, "text", "")),
            "timestamp_ms": round(max(0.0, float(getattr(event, "ts", 0.0))) * 1000.0, 6),
            "frame_seq": int(getattr(event, "frame_seq", 0)),
            "bbox": _jsonable(getattr(event, "bbox", ())),
            "quality": float(getattr(event, "quality", 0.0)),
            "track_id": getattr(event, "track_id", None),
            "episode_id": getattr(event, "episode_id", None),
            "observations": int(getattr(event, "observations", 1)),
        }
        if confidence is not None:
            mapped["confidence"] = confidence
        return mapped

    def reproducibility_metadata(self) -> dict[str, Any]:
        return {
            "schema": "bcvision.anpr.accuracy-adapter-metadata/v1",
            "adapter": "engine-v2-offline-shared-inference",
            "models": _jsonable(self._model_files),
            "requested_runtime": {
                "backend": self.config.backend,
                "device": self.config.device,
                "allow_fallback": self.config.allow_inference_fallback,
            },
            "selected_shared_model_runtime": self._shared_model_summary,
            "sessions": {"service_total": 2, "per_camera": 0},
            "config": {
                "adapter": _jsonable(self.config),
                "model_bundle": _jsonable(self._model_config),
                "engine": _jsonable(self._engine_config),
            },
            "decode": {
                "library": "OpenCV",
                "opencv_version": cv2.__version__,
                "video_backend": "FFMPEG",
                "video_hardware_acceleration": "disabled/software",
                "video_decoder_threads": 1,
                "opencv_compute_threads": self.config.opencv_threads,
                "detector_frame_size": list(self.config.detector_frame_size),
                "detector_frame_source": "resize of the same decoded main frame",
                "timestamp_policy": "CAP_PROP_POS_MSEC PTS, then frame-index/FPS fallback",
                "observed_runs": _jsonable(self._decode_runs),
            },
            "host": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "eof": {
                "preferred_api": "engine.finalize_camera(camera_id, final_seq=..., final_ts=...)",
                "available": callable(getattr(self.engine, "finalize_camera", None)),
            },
        }

    def _restore_opencv_threads(self) -> None:
        if self._previous_opencv_threads is not None:
            cv2.setNumThreads(self._previous_opencv_threads)
            self._previous_opencv_threads = None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._bundle.close()
        finally:
            self._restore_opencv_threads()

    def __enter__(self) -> "EngineV2OfflineAccuracyAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "EngineV2OfflineAccuracyAdapter",
    "LegacyVideoAccuracyAdapter",
    "LegacyVideoAccuracyConfig",
    "V2OfflineAccuracyConfig",
]
