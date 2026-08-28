"""Side-effect-free Engine V2 runner for opt-in live-camera evaluation.

The production live worker owns persistence and operator-visible baseline
events.  This module deliberately owns neither: it keeps one detector session
and one OCR session for the whole process, consumes newest live frames on a
background thread, and exposes only transient overlays and A/B telemetry.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Callable, Protocol

import numpy as np

from .types import FramePacket, OCRResult, PlateEvent
from .validator import IranianPlateValidator


SHADOW_OCR_ENGINE = "engine-v2-hezar-shadow"
SHADOW_POLICY_NAME = "ir-lpr-static-shadow-2026-08-26"
SHADOW_EXPRESS_CONFIDENCE = 0.999
SHADOW_EXPRESS_MIN_SLOT_CONFIDENCE = 0.98


class _EngineLike(Protocol):
    def submit_frame(self, packet: FramePacket) -> bool: ...

    def process_available(self, limit: int = 128) -> list[PlateEvent]: ...

    def finalize_camera(
        self,
        camera_id: str,
        *,
        final_seq: int | None = None,
        final_ts: float | None = None,
    ) -> list[PlateEvent]: ...

    def telemetry(self) -> dict[str, object]: ...


class _RuntimeLike(Protocol):
    engine: _EngineLike

    def close(self) -> None: ...


RuntimeFactory = Callable[[str], _RuntimeLike]


@dataclass(slots=True)
class _ComparisonRecord:
    ts: float
    text: str


@dataclass(slots=True)
class _CameraShadowState:
    seq: int = 0
    epoch: int = 0
    frames_submitted: int = 0
    frames_admitted: int = 0
    events: int = 0
    errors: int = 0
    agreements: int = 0
    disagreements: int = 0
    baseline_only: int = 0
    v2_only: int = 0
    detection_revision: int = 0
    last_error: str = ""
    last_event: dict[str, object] | None = None
    pending_baseline: list[_ComparisonRecord] = field(default_factory=list)
    pending_v2: list[_ComparisonRecord] = field(default_factory=list)
    detections: list[dict[str, object]] = field(default_factory=list)


@dataclass(slots=True)
class _LiveRuntime:
    engine: _EngineLike
    detector_backend: object
    ocr_backend: object

    def close(self) -> None:
        worker = getattr(self.engine, "ocr_worker", None)
        stop = getattr(worker, "stop", None)
        if callable(stop):
            stop()
        for backend in (self.ocr_backend, self.detector_backend):
            close = getattr(backend, "close", None)
            if callable(close):
                close()


class HezarV2PlateOCR:
    """Exact production Hezar v2 preprocessing/decoding over a shared backend."""

    def __init__(self, backend: object, spec: dict[str, object] | None = None) -> None:
        from app.ai.onnx_hezar import HEZAR_V2_SPEC

        self.backend = backend
        self.spec = dict(HEZAR_V2_SPEC if spec is None else spec)

    @staticmethod
    def _candidates(hypotheses: object, *, accepted: bool) -> list[dict[str, object]]:
        if not accepted or not isinstance(hypotheses, list):
            return []
        output: list[dict[str, object]] = []
        for item in hypotheses:
            if not isinstance(item, dict):
                continue
            text = str(item.get("plate_norm") or item.get("plate") or "").strip()
            if not text:
                continue
            confidence = _unit(item.get("confidence", 0.0))
            output.append(
                {
                    "text": text,
                    "confidence": confidence,
                    "weight": confidence,
                    "character_confidences": (),
                }
            )
        return output

    def read(self, crop: np.ndarray) -> OCRResult:
        from app.ai.onnx_hezar import (
            accept_hypotheses,
            ctc_beam_hypotheses,
            prepare_hezar_input,
        )

        tensor = prepare_hezar_input(crop, self.spec)
        if tensor is None:
            return OCRResult("", 0.0, False, metadata={"reason": "empty_crop"})
        input_names = tuple(getattr(self.backend, "input_names", ()))
        if not input_names:
            raise ValueError("Hezar backend exposes no input")
        outputs = self.backend.infer({input_names[0]: tensor})
        if not outputs:
            raise ValueError("Hezar inference returned no outputs")
        logits = np.asarray(outputs[0])
        if logits.ndim == 3:
            logits = logits[0]
        if bool(self.spec.get("reverse_output_digits", False)):
            logits = logits[::-1]
        hypotheses = ctc_beam_hypotheses(
            logits,
            labels=list(self.spec.get("labels") or []),
            blank_index=int(self.spec.get("blank_index", 0)),
            beam_width=int(self.spec.get("beam_width", 10)),
            top_k=int(self.spec.get("top_k", 5)),
        )
        result = accept_hypotheses(
            hypotheses,
            min_confidence=float(self.spec.get("min_confidence", 0.56)),
            min_position_margin=float(self.spec.get("min_position_margin", 0.12)),
        )
        accepted = bool(result.get("accepted", False))
        raw_details = result.get("position_details", [])
        details = raw_details if isinstance(raw_details, list) else []
        character_confidences = tuple(
            _unit(item.get("probability", 0.0))
            for item in details
            if isinstance(item, dict)
        )
        raw_hypotheses = result.get("hypotheses", [])
        candidates = self._candidates(raw_hypotheses, accepted=accepted)
        text = str(result.get("plate_norm") or "").strip()
        if not text and isinstance(raw_hypotheses, list) and raw_hypotheses:
            first = raw_hypotheses[0]
            if isinstance(first, dict):
                text = str(first.get("plate_norm") or "").strip()
        return OCRResult(
            text=text,
            confidence=_unit(result.get("confidence", 0.0)),
            valid=accepted,
            character_confidences=character_confidences,
            metadata={
                "candidates": candidates,
                "decoder": SHADOW_OCR_ENGINE,
                "shadow": True,
            },
        )


def _default_runtime_factory(detector_variant: str) -> _LiveRuntime:
    from app.ai.hezar_export import HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE
    from app.ai.model_manager import (
        detector_variant_spec,
        hezar_path,
        verify_file,
    )

    from .inference import InferenceConfig, SharedInferenceBackend
    from .model_adapters import YOLOPlateDetector, YOLOPlateDetectorConfig
    from .runtime import EngineV2Config, EventDrivenANPREngine
    from .tcam import TemporalFusionConfig

    detector_spec = detector_variant_spec(detector_variant)
    detector_path = detector_spec["path"]
    if not verify_file(
        detector_path,
        str(detector_spec["sha256"]),
        int(detector_spec["size"]),
    ):
        raise FileNotFoundError(f"Verified Shadow detector not found: {detector_path}")
    ocr_path = hezar_path()
    if not verify_file(ocr_path, HEZAR_ONNX_SHA256, HEZAR_ONNX_SIZE):
        raise FileNotFoundError(f"Verified Shadow Hezar model not found: {ocr_path}")

    backend = os.environ.get("BCVISION_ENGINE_V2_BACKEND", "auto").strip() or "auto"
    device = os.environ.get("BCVISION_ENGINE_V2_DEVICE", "AUTO").strip() or "AUTO"
    common = {
        "backend": backend,
        "device": device,
        "inter_op_threads": 1,
        "allow_fallback": True,
    }
    detector_backend = SharedInferenceBackend(
        InferenceConfig(model_path=detector_path, **common)
    )
    try:
        ocr_backend = SharedInferenceBackend(
            InferenceConfig(model_path=ocr_path, **common)
        )
    except Exception:
        detector_backend.close()
        raise
    try:
        input_size = int(detector_spec.get("input_size", 640))
        detector = YOLOPlateDetector(
            detector_backend,
            YOLOPlateDetectorConfig(
                input_size=(input_size, input_size),
                confidence_threshold=0.25,
                iou_threshold=0.45,
                num_classes=1,
                max_detections=12,
            ),
        )
        ocr = HezarV2PlateOCR(ocr_backend)
        engine = EventDrivenANPREngine(
            detector,
            ocr,
            EngineV2Config(
                min_detector_confidence=0.30,
                load_control_enabled=True,
                track_temporal_fusion_enabled=True,
                temporal_fusion=TemporalFusionConfig(
                    express_lock_confidence=SHADOW_EXPRESS_CONFIDENCE,
                    express_min_slot_confidence=(
                        SHADOW_EXPRESS_MIN_SLOT_CONFIDENCE
                    ),
                ),
                default_temporal_fusion_profile="day",
            ),
        )
    except Exception:
        ocr_backend.close()
        detector_backend.close()
        raise
    return _LiveRuntime(engine, detector_backend, ocr_backend)


class EngineV2LiveShadow:
    """Newest-frame, background Shadow service with no persistence path."""

    def __init__(
        self,
        runtime_factory: RuntimeFactory | None = None,
        *,
        match_window_seconds: float = 4.0,
        detection_ttl_seconds: float = 3.0,
        retry_seconds: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._runtime_factory = runtime_factory or _default_runtime_factory
        self._match_window = max(0.1, float(match_window_seconds))
        self._detection_ttl = max(0.1, float(detection_ttl_seconds))
        self._retry_seconds = max(0.1, float(retry_seconds))
        self._clock = clock
        self._validator = IranianPlateValidator()
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._runtime: _RuntimeLike | None = None
        self._runtime_variant = ""
        self._requested_variant = "yolo11n"
        self._enabled = False
        self._initializing = False
        self._rebuild_requested = False
        self._last_init_attempt = -1e12
        self._last_error = ""
        self._states: dict[str, _CameraShadowState] = {}
        self._pending: dict[str, FramePacket] = {}
        self._finalize: set[str] = set()

    def configure(self, enabled: bool, detector_variant: str = "yolo11n") -> None:
        variant = str(detector_variant or "yolo11n").strip().lower()
        if variant not in {"yolo11n", "yolov8n"}:
            variant = "yolo11n"
        with self._lock:
            changed = variant != self._requested_variant
            self._requested_variant = variant
            self._enabled = bool(enabled)
            if changed:
                self._rebuild_requested = True
            if not self._enabled:
                self._pending.clear()
                self._rebuild_requested = self._runtime is not None
                for state in self._states.values():
                    if state.detections:
                        state.detections.clear()
                        state.detection_revision += 1
            self._ensure_thread_locked()
        self._wake.set()

    def submit(
        self,
        camera_id: int | str,
        frame: np.ndarray,
        *,
        ts: float | None = None,
        roi: tuple[int, int, int, int] | None = None,
        illumination_profile: str = "day",
    ) -> bool:
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        camera_key = str(camera_id)
        with self._lock:
            if not self._enabled or self._stop.is_set():
                return False
            state = self._states.setdefault(camera_key, _CameraShadowState())
            state.seq += 1
            state.frames_submitted += 1
            metadata: dict[str, object] = {
                "producer_epoch": f"live-shadow-{state.epoch}",
                "illumination_profile": str(illumination_profile or "day"),
            }
            if roi is not None:
                metadata["motion_roi"] = tuple(int(value) for value in roi)
            self._pending[camera_key] = FramePacket(
                camera_id=camera_key,
                seq=state.seq,
                ts=self._clock() if ts is None else float(ts),
                frame=np.ascontiguousarray(frame.copy()),
                metadata=metadata,
            )
            self._ensure_thread_locked()
        self._wake.set()
        return True

    def observe_baseline(
        self,
        camera_id: int | str,
        rows: list[dict] | tuple[dict, ...],
        *,
        ts: float | None = None,
    ) -> None:
        timestamp = self._clock() if ts is None else float(ts)
        camera_key = str(camera_id)
        with self._lock:
            if not self._enabled:
                return
            state = self._states.setdefault(camera_key, _CameraShadowState())
            self._expire_locked(state, timestamp)
            seen: set[str] = set()
            for row in rows:
                if not isinstance(row, dict) or not bool(row.get("valid")):
                    continue
                value = row.get("plate_norm") or row.get("plate") or ""
                validated = self._validator.validate(str(value))
                if not validated.valid or validated.normalized in seen:
                    continue
                seen.add(validated.normalized)
                self._match_or_queue_locked(
                    state,
                    side="baseline",
                    record=_ComparisonRecord(timestamp, validated.normalized),
                )

    def remove_camera(self, camera_id: int | str) -> None:
        camera_key = str(camera_id)
        with self._lock:
            state = self._states.setdefault(camera_key, _CameraShadowState())
            state.epoch += 1
            state.seq = 0
            self._pending.pop(camera_key, None)
            state.pending_baseline.clear()
            state.pending_v2.clear()
            if state.detections:
                state.detections.clear()
                state.detection_revision += 1
            if self._runtime is not None:
                self._finalize.add(camera_key)
        self._wake.set()

    def detections(self, camera_id: int | str) -> list[dict[str, object]]:
        now = self._clock()
        with self._lock:
            state = self._states.get(str(camera_id))
            if state is None or not self._enabled:
                return []
            state.detections = [
                row
                for row in state.detections
                if now - float(row.get("_shadow_seen_at", now)) <= self._detection_ttl
            ]
            return [
                {key: value for key, value in row.items() if key != "_shadow_seen_at"}
                for row in state.detections
            ]

    def status(self, camera_id: int | str) -> dict[str, object]:
        now = self._clock()
        with self._lock:
            state = self._states.get(str(camera_id)) or _CameraShadowState()
            self._expire_locked(state, now)
            telemetry = self._runtime_telemetry_locked()
            return {
                "enabled": self._enabled,
                "ready": self._runtime is not None and not self._rebuild_requested,
                "initializing": self._initializing,
                "side_effects": False,
                "persistence": False,
                "detector_variant": self._requested_variant,
                "ocr_engine": SHADOW_OCR_ENGINE,
                "policy": SHADOW_POLICY_NAME,
                "express_lock_confidence": SHADOW_EXPRESS_CONFIDENCE,
                "express_min_slot_confidence": (
                    SHADOW_EXPRESS_MIN_SLOT_CONFIDENCE
                ),
                "frames": state.frames_submitted,
                "admitted_frames": state.frames_admitted,
                "events": state.events,
                "errors": state.errors,
                "last_error": state.last_error or self._last_error,
                "last_event": dict(state.last_event) if state.last_event else None,
                "detection_revision": state.detection_revision,
                "agreements": state.agreements,
                "disagreements": state.disagreements,
                "baseline_only": state.baseline_only,
                "v2_only": state.v2_only,
                "pending_baseline": len(state.pending_baseline),
                "pending_v2": len(state.pending_v2),
                "telemetry": telemetry,
            }

    def shutdown(self, *, wait: bool = True) -> None:
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if wait and thread is not None and thread is not threading.current_thread():
            thread.join(timeout=10.0)
        with self._lock:
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            runtime.close()

    def _ensure_thread_locked(self) -> None:
        if self._stop.is_set():
            return
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(
                target=self._run,
                name="bc-engine-v2-shadow",
                daemon=True,
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.5)
            self._wake.clear()
            try:
                self._run_cycle()
            except Exception as exc:  # final containment boundary
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"

    def _run_cycle(self) -> None:
        runtime_to_close: _RuntimeLike | None = None
        with self._lock:
            if self._runtime is not None and (
                not self._enabled
                or self._rebuild_requested
                or self._runtime_variant != self._requested_variant
            ):
                runtime_to_close = self._runtime
                self._runtime = None
                self._runtime_variant = ""
                self._rebuild_requested = False
        if runtime_to_close is not None:
            runtime_to_close.close()

        with self._lock:
            enabled = self._enabled
            runtime = self._runtime
            variant = self._requested_variant
            can_retry = self._clock() - self._last_init_attempt >= self._retry_seconds
            if enabled and runtime is None and can_retry:
                self._initializing = True
                self._last_init_attempt = self._clock()
        if enabled and runtime is None and can_retry:
            try:
                created = self._runtime_factory(variant)
            except Exception as exc:
                with self._lock:
                    self._initializing = False
                    self._last_error = f"{type(exc).__name__}: {exc}"
                return
            with self._lock:
                self._initializing = False
                if self._enabled and variant == self._requested_variant:
                    self._runtime = created
                    self._runtime_variant = variant
                    self._rebuild_requested = False
                    self._last_error = ""
                    runtime = created
                else:
                    runtime = None
            if runtime is None:
                created.close()
                return

        with self._lock:
            runtime = self._runtime
            if runtime is None:
                return
            packets = list(self._pending.values())
            self._pending.clear()
            finalizations = tuple(self._finalize)
            self._finalize.clear()

        for packet in packets:
            try:
                roi = packet.metadata.get("motion_roi")
                if isinstance(roi, tuple) and len(roi) == 4:
                    set_roi = getattr(runtime.engine, "set_roi", None)
                    if callable(set_roi):
                        set_roi(packet.camera_id, roi)
                admitted = bool(runtime.engine.submit_frame(packet))
                with self._lock:
                    state = self._states.setdefault(
                        packet.camera_id, _CameraShadowState()
                    )
                    state.frames_admitted += int(admitted)
            except Exception as exc:
                self._record_error(packet.camera_id, exc)

        try:
            events = runtime.engine.process_available(limit=max(32, len(packets) * 4))
        except Exception as exc:
            with self._lock:
                self._last_error = f"{type(exc).__name__}: {exc}"
                for packet in packets:
                    state = self._states.setdefault(
                        packet.camera_id, _CameraShadowState()
                    )
                    state.errors += 1
                    state.last_error = self._last_error
            events = []
        for event in events:
            self._record_event(event)

        for camera_id in finalizations:
            try:
                events = runtime.engine.finalize_camera(camera_id)
            except Exception as exc:
                self._record_error(camera_id, exc)
                continue
            for event in events:
                self._record_event(event)

        with self._lock:
            has_more = bool(self._pending or self._finalize)
        if has_more:
            self._wake.set()

    def _record_error(self, camera_id: str, exc: Exception) -> None:
        error = f"{type(exc).__name__}: {exc}"
        with self._lock:
            state = self._states.setdefault(str(camera_id), _CameraShadowState())
            state.errors += 1
            state.last_error = error
            self._last_error = error

    def _record_event(self, event: PlateEvent) -> None:
        validated = self._validator.validate(event.text)
        if not validated.valid:
            return
        now = self._clock()
        camera_id = str(event.camera_id)
        overlay = {
            "bbox": tuple(int(value) for value in event.bbox),
            "plate": validated.normalized,
            "plate_norm": validated.normalized,
            "confidence": _unit(event.confidence),
            "track_id": event.track_id or event.episode_id or "",
            "tracking_engine": "engine-v2-track-centric",
            "valid": True,
            "best_effort": False,
            "needs_review": True,
            "ocr_engine": SHADOW_OCR_ENGINE,
            "ocr_alternative": "",
            "ocr_disagreement": False,
            "raw_guess_text": validated.normalized,
            "raw_guess_confidence": _unit(event.confidence),
            "raw_guess_reason": str(event.metadata.get("fusion_reason", "")),
            "model_revision": SHADOW_POLICY_NAME,
            "engine_lane": "shadow-v2",
            "experimental": True,
            "_shadow_seen_at": now,
        }
        with self._lock:
            state = self._states.setdefault(camera_id, _CameraShadowState())
            state.events += 1
            state.last_event = {
                key: value for key, value in overlay.items() if key != "_shadow_seen_at"
            }
            state.detections = [
                row
                for row in state.detections
                if now - float(row.get("_shadow_seen_at", now))
                <= self._detection_ttl
            ][-31:]
            state.detections.append(overlay)
            state.detection_revision += 1
            self._expire_locked(state, event.ts)
            self._match_or_queue_locked(
                state,
                side="v2",
                record=_ComparisonRecord(float(event.ts), validated.normalized),
            )

    def _match_or_queue_locked(
        self,
        state: _CameraShadowState,
        *,
        side: str,
        record: _ComparisonRecord,
    ) -> None:
        opposite = state.pending_v2 if side == "baseline" else state.pending_baseline
        matches = [
            (abs(candidate.ts - record.ts), index)
            for index, candidate in enumerate(opposite)
            if abs(candidate.ts - record.ts) <= self._match_window
        ]
        if matches:
            _, index = min(matches)
            other = opposite.pop(index)
            if other.text == record.text:
                state.agreements += 1
            else:
                state.disagreements += 1
            return
        target = state.pending_baseline if side == "baseline" else state.pending_v2
        if not any(item.text == record.text for item in target):
            target.append(record)

    def _expire_locked(self, state: _CameraShadowState, now: float) -> None:
        baseline_kept = [
            row for row in state.pending_baseline if now - row.ts <= self._match_window
        ]
        v2_kept = [
            row for row in state.pending_v2 if now - row.ts <= self._match_window
        ]
        state.baseline_only += len(state.pending_baseline) - len(baseline_kept)
        state.v2_only += len(state.pending_v2) - len(v2_kept)
        state.pending_baseline = baseline_kept
        state.pending_v2 = v2_kept

    def _runtime_telemetry_locked(self) -> dict[str, object]:
        if self._runtime is None:
            return {}
        try:
            raw = self._runtime.engine.telemetry()
        except Exception as exc:
            self._last_error = f"{type(exc).__name__}: {exc}"
            return {}
        keep = {
            "frames_received",
            "detector_inferences",
            "detector_mean_ms",
            "ocr_inferences",
            "ocr_mean_ms",
            "dropped_stale_frames",
            "queue_replaced",
            "events",
            "duplicates_suppressed",
            "fusion_ocr_attempts",
            "active_cameras",
            "idle_cameras",
            "load_level",
        }
        return {
            key: _json_safe(value)
            for key, value in raw.items()
            if key in keep
        }


def _unit(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(number):
        return 0.0
    return max(0.0, min(1.0, number))


def _json_safe(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {key: _json_safe(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


live_shadow = EngineV2LiveShadow()


def configure_live_shadow(enabled: bool, detector_variant: str = "yolo11n") -> None:
    live_shadow.configure(enabled, detector_variant)


def submit_live_shadow_frame(
    camera_id: int | str,
    frame: np.ndarray,
    *,
    ts: float | None = None,
    roi: tuple[int, int, int, int] | None = None,
    detector_variant: str = "yolo11n",
    illumination_profile: str = "day",
) -> bool:
    live_shadow.configure(True, detector_variant)
    return live_shadow.submit(
        camera_id,
        frame,
        ts=ts,
        roi=roi,
        illumination_profile=illumination_profile,
    )


def observe_live_shadow_baseline(
    camera_id: int | str,
    rows: list[dict] | tuple[dict, ...],
    *,
    ts: float | None = None,
) -> None:
    live_shadow.observe_baseline(camera_id, rows, ts=ts)


def live_shadow_status(camera_id: int | str) -> dict[str, object]:
    return live_shadow.status(camera_id)


def live_shadow_detections(camera_id: int | str) -> list[dict[str, object]]:
    return live_shadow.detections(camera_id)


def stop_live_shadow_camera(camera_id: int | str) -> None:
    live_shadow.remove_camera(camera_id)


def shutdown_live_shadow() -> None:
    live_shadow.shutdown(wait=True)


__all__ = [
    "EngineV2LiveShadow",
    "HezarV2PlateOCR",
    "SHADOW_EXPRESS_CONFIDENCE",
    "SHADOW_EXPRESS_MIN_SLOT_CONFIDENCE",
    "SHADOW_OCR_ENGINE",
    "SHADOW_POLICY_NAME",
    "configure_live_shadow",
    "live_shadow_detections",
    "live_shadow_status",
    "observe_live_shadow_baseline",
    "shutdown_live_shadow",
    "stop_live_shadow_camera",
    "submit_live_shadow_frame",
]
