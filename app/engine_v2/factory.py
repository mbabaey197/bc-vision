from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .inference import InferenceConfig, SharedInferenceBackend
from .model_adapters import (
    CTCPlateOCR,
    CTCPlateOCRConfig,
    YOLOPlateDetector,
    YOLOPlateDetectorConfig,
)
from .runtime import EngineV2Config, EventDrivenANPREngine
from .types import PlateEvent


@dataclass(frozen=True, slots=True)
class SharedModelBundleConfig:
    detector_model: str | Path
    ocr_model: str | Path
    backend: str = "auto"
    device: str = "AUTO"
    detector_intra_op_threads: int | None = None
    ocr_intra_op_threads: int | None = None
    inter_op_threads: int | None = 1
    allow_fallback: bool = True
    provider_options: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    detector: YOLOPlateDetectorConfig = field(
        default_factory=lambda: YOLOPlateDetectorConfig(input_size=(320, 320), num_classes=1)
    )
    ocr: CTCPlateOCRConfig = field(default_factory=CTCPlateOCRConfig)


class SharedModelBundle:
    """Exactly one detector session and one OCR session for the whole service."""

    def __init__(
        self,
        detector_backend: SharedInferenceBackend,
        ocr_backend: SharedInferenceBackend,
        detector: YOLOPlateDetector,
        ocr: CTCPlateOCR,
    ) -> None:
        self.detector_backend = detector_backend
        self.ocr_backend = ocr_backend
        self.detector = detector
        self.ocr = ocr

    def close(self) -> None:
        # Closing is idempotent in SharedInferenceBackend.
        self.ocr_backend.close()
        self.detector_backend.close()

    def __enter__(self) -> SharedModelBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def summary(self) -> dict[str, object]:
        return {
            "detector": self.detector_backend.metadata,
            "ocr": self.ocr_backend.metadata,
            "session_count": 2,
            "sessions_per_camera": 0,
        }


@dataclass(slots=True)
class EngineV2RuntimeBundle:
    engine: EventDrivenANPREngine
    models: SharedModelBundle

    def close(self) -> None:
        self.engine.ocr_worker.stop()
        self.models.close()

    def __enter__(self) -> EngineV2RuntimeBundle:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def build_shared_models(config: SharedModelBundleConfig) -> SharedModelBundle:
    common = {
        "backend": config.backend,
        "device": config.device,
        "inter_op_threads": config.inter_op_threads,
        "allow_fallback": config.allow_fallback,
        "provider_options": config.provider_options,
    }
    # Treat model/session/adapter construction as one transaction. ExitStack
    # registers each backend immediately after successful creation, so a failure
    # in either adapter closes both sessions exactly once. On success the
    # callbacks are detached and ownership moves to SharedModelBundle.
    with ExitStack() as cleanup:
        detector_backend = SharedInferenceBackend(
            InferenceConfig(
                model_path=config.detector_model,
                intra_op_threads=config.detector_intra_op_threads,
                **common,
            )
        )
        cleanup.callback(detector_backend.close)
        ocr_backend = SharedInferenceBackend(
            InferenceConfig(
                model_path=config.ocr_model,
                intra_op_threads=config.ocr_intra_op_threads,
                **common,
            )
        )
        cleanup.callback(ocr_backend.close)
        detector = YOLOPlateDetector(detector_backend, config.detector)
        ocr = CTCPlateOCR(ocr_backend, config.ocr)
        cleanup.pop_all()
    return SharedModelBundle(detector_backend, ocr_backend, detector, ocr)


def build_engine_v2(
    models: SharedModelBundleConfig,
    engine: EngineV2Config | None = None,
    on_event: Callable[[PlateEvent], None] | None = None,
) -> EngineV2RuntimeBundle:
    shared = build_shared_models(models)
    try:
        runtime = EventDrivenANPREngine(
            shared.detector,
            shared.ocr,
            engine,
            on_event,
        )
    except Exception:
        shared.close()
        raise
    return EngineV2RuntimeBundle(runtime, shared)
