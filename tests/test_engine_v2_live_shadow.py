from __future__ import annotations

import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.engine_v2.live_shadow import EngineV2LiveShadow, _default_runtime_factory
from app.engine_v2.types import PlateEvent


def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.01)
    raise AssertionError("condition was not reached before timeout")


class _FakeEngine:
    def __init__(self):
        self.packets = []
        self.events = []
        self.rois = {}
        self.finalized = []

    def set_roi(self, camera_id, roi):
        self.rois[camera_id] = roi

    def submit_frame(self, packet):
        self.packets.append(packet)
        return True

    def process_available(self, limit=128):
        events, self.events = self.events[:limit], self.events[limit:]
        return events

    def finalize_camera(self, camera_id, **_kwargs):
        self.finalized.append(camera_id)
        return []

    def telemetry(self):
        return {
            "frames_received": len(self.packets),
            "detector_inferences": len(self.packets),
            "secret_internal_value": "must-not-leak",
        }


class _FakeRuntime:
    def __init__(self):
        self.engine = _FakeEngine()
        self.closed = False

    def close(self):
        self.closed = True


def _event(ts=10.0, text="12ب34567"):
    return PlateEvent(
        camera_id="7",
        frame_seq=1,
        ts=ts,
        text=text,
        confidence=0.995,
        bbox=(10, 20, 90, 55),
        quality=0.91,
        track_id="track-7",
        metadata={"fusion_reason": "temporal_lock"},
    )


def test_live_shadow_is_opt_in_and_has_no_persistence_path():
    created = []

    def factory(variant):
        runtime = _FakeRuntime()
        created.append((variant, runtime))
        return runtime

    shadow = EngineV2LiveShadow(factory, retry_seconds=0.01)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    try:
        assert shadow.submit(7, frame, ts=10.0) is False
        assert created == []

        shadow.configure(True, "yolov8n")
        assert shadow.submit(7, frame, ts=10.0, roi=(2, 3, 100, 70))
        runtime = _wait_for(lambda: created and created[0][1])
        _wait_for(lambda: runtime.engine.packets)

        assert created[0][0] == "yolov8n"
        assert runtime.engine.rois == {"7": (2, 3, 100, 70)}
        status = shadow.status(7)
        assert status["ready"] is True
        assert status["side_effects"] is False
        assert status["persistence"] is False
        assert status["frames"] == 1
        assert status["admitted_frames"] == 1
        assert "secret_internal_value" not in status["telemetry"]
    finally:
        shadow.shutdown()

    assert created[0][1].closed is True


def test_live_shadow_publishes_transient_v2_overlay_and_ab_agreement():
    runtime = _FakeRuntime()
    shadow = EngineV2LiveShadow(
        lambda _variant: runtime,
        retry_seconds=0.01,
        detection_ttl_seconds=5.0,
    )
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    stamp = time.monotonic()
    try:
        shadow.configure(True)
        assert shadow.submit(7, frame, ts=stamp)
        _wait_for(lambda: runtime.engine.packets)
        runtime.engine.events.append(_event(ts=stamp))
        assert shadow.submit(7, frame, ts=stamp + 0.1)
        _wait_for(lambda: shadow.status(7)["events"] == 1)

        rows = shadow.detections(7)
        assert len(rows) == 1
        assert rows[0]["plate_norm"] == "12ب34567"
        assert rows[0]["engine_lane"] == "shadow-v2"
        assert rows[0]["experimental"] is True

        shadow.observe_baseline(
            7,
            [{"plate_norm": "12ب34567", "valid": True}],
            ts=stamp + 0.2,
        )
        status = shadow.status(7)
        assert status["agreements"] == 1
        assert status["disagreements"] == 0
        assert status["pending_baseline"] == 0
        assert status["pending_v2"] == 0
    finally:
        shadow.shutdown()


def test_live_shadow_model_failure_stays_inside_shadow_lane():
    attempts = []

    def failing_factory(variant):
        attempts.append(variant)
        raise RuntimeError("model unavailable")

    shadow = EngineV2LiveShadow(failing_factory, retry_seconds=60.0)
    try:
        shadow.configure(True)
        accepted = shadow.submit(
            3,
            np.zeros((20, 40, 3), dtype=np.uint8),
        )
        assert accepted is True
        _wait_for(lambda: shadow.status(3)["last_error"])
        status = shadow.status(3)
        assert attempts == ["yolo11n"]
        assert status["ready"] is False
        assert "model unavailable" in status["last_error"]
        assert status["side_effects"] is False
        assert shadow.detections(3) == []
    finally:
        shadow.shutdown()


def test_disabling_shadow_closes_runtime_and_clears_overlay():
    runtime = _FakeRuntime()
    shadow = EngineV2LiveShadow(lambda _variant: runtime, retry_seconds=0.01)
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    try:
        shadow.configure(True)
        shadow.submit(7, frame, ts=10.0)
        _wait_for(lambda: runtime.engine.packets)
        runtime.engine.events.append(_event())
        shadow.submit(7, frame, ts=10.1)
        _wait_for(lambda: shadow.detections(7))

        shadow.configure(False)
        _wait_for(lambda: runtime.closed)
        assert shadow.status(7)["enabled"] is False
        assert shadow.detections(7) == []
    finally:
        shadow.shutdown()


def test_shadow_module_has_no_database_or_persistence_dependency():
    source = Path("app/engine_v2/live_shadow.py").read_text(encoding="utf-8")

    assert "app.database" not in source
    assert "._persist" not in source
    assert "set_setting" not in source
    assert '"persistence": False' in source


def test_live_worker_observes_v2_before_baseline_persistence():
    source = Path("app/ai/live_worker.py").read_text(encoding="utf-8")
    process_source = source[source.index("    def _process(") :]

    observe_at = process_source.index("self._observe_engine_v2_baseline(")
    persist_at = process_source.index(
        "self._enqueue_persistence_retry(",
        observe_at,
    )
    assert observe_at < persist_at
    assert "submit_live_shadow_frame" in source
    assert "configure_live_engine_v2_shadow" in source
    shadow_helpers = source[
        source.index("    def _submit_engine_v2_shadow(") :
        source.index("    def _shadow_status(")
    ]
    assert "state.last_error" not in shadow_helpers


def test_default_runtime_opens_exactly_one_detector_and_one_hezar_session():
    opened = []

    class Backend:
        def __init__(self, config):
            self.config = config
            self.input_names = ("input",)
            self.closed = False
            opened.append(self)

        def close(self):
            self.closed = True

    class DetectorConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Detector:
        def __init__(self, backend, config):
            self.backend = backend
            self.config = config

    class EngineConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FusionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class Engine:
        def __init__(self, detector, ocr, config):
            self.detector = detector
            self.ocr = ocr
            self.config = config

    def module(name, **members):
        result = ModuleType(name)
        for key, value in members.items():
            setattr(result, key, value)
        return result

    modules = {
        "app.ai.hezar_export": module(
            "app.ai.hezar_export",
            HEZAR_ONNX_SHA256="ocr-sha",
            HEZAR_ONNX_SIZE=22,
        ),
        "app.ai.model_manager": module(
            "app.ai.model_manager",
            detector_variant_spec=lambda variant: {
                "path": Path("detector.onnx"),
                "sha256": "detector-sha",
                "size": 11,
                "input_size": 640,
                "variant": variant,
            },
            hezar_path=lambda: Path("hezar.onnx"),
            verify_file=lambda *_args: True,
        ),
        "app.ai.onnx_hezar": module(
            "app.ai.onnx_hezar",
            HEZAR_V2_SPEC={"labels": ["-"]},
        ),
        "app.engine_v2.inference": module(
            "app.engine_v2.inference",
            InferenceConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            SharedInferenceBackend=Backend,
        ),
        "app.engine_v2.model_adapters": module(
            "app.engine_v2.model_adapters",
            YOLOPlateDetector=Detector,
            YOLOPlateDetectorConfig=DetectorConfig,
        ),
        "app.engine_v2.runtime": module(
            "app.engine_v2.runtime",
            EngineV2Config=EngineConfig,
            EventDrivenANPREngine=Engine,
        ),
        "app.engine_v2.tcam": module(
            "app.engine_v2.tcam",
            TemporalFusionConfig=FusionConfig,
        ),
    }

    with patch.dict("sys.modules", modules):
        runtime = _default_runtime_factory("yolo11n")

    assert len(opened) == 2
    assert opened[0].config.model_path == Path("detector.onnx")
    assert opened[1].config.model_path == Path("hezar.onnx")
    assert runtime.engine.config.kwargs["track_temporal_fusion_enabled"] is True
    fusion = runtime.engine.config.kwargs["temporal_fusion"]
    assert fusion.kwargs["express_lock_confidence"] == 0.999
    assert fusion.kwargs["express_min_slot_confidence"] == 0.98

    runtime.close()
    assert all(backend.closed for backend in opened)


def test_transient_overlay_buffer_is_bounded():
    shadow = EngineV2LiveShadow(lambda _variant: _FakeRuntime())
    try:
        shadow.configure(True)
        for index in range(40):
            event = _event(ts=time.monotonic())
            event.frame_seq = index
            shadow._record_event(event)
        assert len(shadow.detections(7)) == 32
    finally:
        shadow.shutdown()
