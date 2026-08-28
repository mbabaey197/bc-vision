import hashlib
import sys
import threading
import time
import types

import numpy as np

from app.ai import live_worker, model_manager, ocr, onnx_crnn, onnx_hezar
from app.ai.activity import FrameActivity
from app.ai.pipeline import process_frame


PLATE = "31-ط-556-74"
OTHER_PLATE = "31-ط-558-74"


def _rejected_hezar():
    return {
        "accepted": False,
        "plate_norm": "",
        "confidence": 0.0,
        "hypotheses": [],
    }


def test_accepted_hezar_never_calls_platrix_or_custom_ocr(monkeypatch):
    monkeypatch.setenv("BCVISION_OCR_ENGINE", "cnn")
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: {
            "accepted": True,
            "plate_norm": "31ط55674",
            "confidence": 0.93,
            "hypotheses": [{"plate_norm": "31ط55674"}],
        },
    )
    forbidden = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("non-primary OCR must not run")
    )
    monkeypatch.setattr(ocr, "read_plate_platrix", forbidden)
    monkeypatch.setattr(ocr, "read_plate_crnn", forbidden)
    monkeypatch.setattr(ocr, "read_plate_cnn", forbidden)

    result = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8)
    )

    assert result == (PLATE, 0.93, "hezar-crnn-fa-v2-onnx")


def test_rejected_hezar_uses_fixed_platrix_once(monkeypatch):
    calls = []
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: _rejected_hezar(),
    )
    monkeypatch.setattr(
        ocr,
        "hezar_status",
        lambda: {"error": "strict-decoder-rejected"},
    )
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: calls.append(1) or (PLATE, 0.78),
    )
    monkeypatch.setattr(
        ocr,
        "read_plate_crnn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("promoted custom CRNN must not run")
        ),
    )

    result = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8)
    )

    assert result == (PLATE, 0.78, "platrix-crnn-onnx")
    assert calls == [1]


def test_missing_hezar_model_still_reaches_fixed_platrix(
    tmp_path,
    monkeypatch,
):
    missing = tmp_path / "missing-hezar.onnx"
    monkeypatch.setattr(model_manager, "hezar_path", lambda: missing)
    monkeypatch.setattr(
        onnx_hezar,
        "_verified_primary_path",
        lambda: (_ for _ in ()).throw(FileNotFoundError("missing Hezar")),
    )
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: (PLATE, 0.81),
    )

    result = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8)
    )

    assert result == (PLATE, 0.81, "platrix-crnn-onnx")
    assert "FileNotFoundError" in onnx_hezar.hezar_status()["error"]


def test_low_confidence_platrix_stays_unreadable(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: _rejected_hezar(),
    )
    monkeypatch.setattr(ocr, "hezar_status", lambda: {"error": ""})
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: (PLATE, ocr.PLATRIX_MIN_CONFIDENCE - 0.01),
    )
    monkeypatch.setattr(ocr, "get_crnn_status", lambda: {"error": ""})
    monkeypatch.setattr(
        ocr,
        "read_plate_cnn",
        lambda *_args, **_kwargs: ("", 0.0),
    )
    monkeypatch.setattr(ocr, "get_cnn_status", lambda: {"error": ""})

    assert ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8)
    ) == ("", 0.0, "none")


def test_rejected_whole_plate_models_use_character_cnn(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: _rejected_hezar(),
    )
    monkeypatch.setattr(ocr, "hezar_status", lambda: {"error": "rejected"})
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: ("", 0.0),
    )
    monkeypatch.setattr(ocr, "get_crnn_status", lambda: {"error": "rejected"})
    monkeypatch.setattr(
        ocr,
        "read_plate_cnn",
        lambda *_args, **_kwargs: (PLATE, 0.72),
    )

    candidate = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8),
        allow_legacy=True,
        include_evidence=True,
    )

    assert candidate["plate"] == PLATE
    assert candidate["plate_norm"] == "31ط55674"
    assert candidate["confidence"] == 0.72
    assert candidate["engine"] == "cnn-onnx"
    assert candidate["hypotheses"][-1]["plate_norm"] == "31ط55674"


def test_character_cnn_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: _rejected_hezar(),
    )
    monkeypatch.setattr(ocr, "hezar_status", lambda: {"error": "rejected"})
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: ("", 0.0),
    )
    monkeypatch.setattr(ocr, "get_crnn_status", lambda: {"error": "rejected"})
    monkeypatch.setattr(
        ocr,
        "read_plate_cnn",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("character CNN must stay disabled")
        ),
    )

    assert ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8),
        allow_legacy=False,
    ) == ("", 0.0, "none")


def test_rejected_hezar_top_k_is_retained_as_temporal_evidence(
    monkeypatch,
):
    monkeypatch.setattr(
        ocr,
        "read_plate_hezar_primary",
        lambda *_args, **_kwargs: {
            "accepted": False,
            "plate_norm": "",
            "confidence": 0.48,
            "hypotheses": [
                {
                    "plate_norm": "31ط55874",
                    "confidence": 0.48,
                    "score": 1e-30,
                },
                {
                    "plate_norm": "31ط55674",
                    "confidence": 0.44,
                    "score": 1e-40,
                },
                {
                    "plate_norm": "31ط55974",
                    "confidence": 0.20,
                    "score": 1e-50,
                },
            ],
        },
    )
    monkeypatch.setattr(ocr, "hezar_status", lambda: {"error": ""})
    monkeypatch.setattr(
        ocr,
        "read_plate_platrix",
        lambda *_args, **_kwargs: ("", 0.0),
    )
    monkeypatch.setattr(ocr, "get_crnn_status", lambda: {"error": ""})

    candidate = ocr.read_plate_candidate(
        np.zeros((32, 160, 3), dtype=np.uint8),
        include_evidence=True,
    )

    assert candidate["plate"] == ""
    assert candidate["plate_norm"] == ""
    assert candidate["engine"] == "none"
    assert [
        row["plate_norm"] for row in candidate["hypotheses"]
    ] == ["31ط55874", "31ط55674", "31ط55974"]
    assert [
        row["temporal_evidence"]
        for row in candidate["hypotheses"]
    ] == [True, True, False]
    assert [
        row["score"] for row in candidate["hypotheses"]
    ] == [0.48, 0.44, 0.20]
    assert [
        row["ctc_path_score"] for row in candidate["hypotheses"]
    ] == [1e-30, 1e-40, 1e-50]


def test_platrix_production_runtime_ignores_promoted_custom_crnn(
    tmp_path,
    monkeypatch,
):
    vendor = tmp_path / "platrix.onnx"
    vendor.write_bytes(b"fixed-platrix")
    custom = tmp_path / "custom.onnx"
    custom.write_bytes(b"weaker-custom")
    vendor_digest = hashlib.sha256(vendor.read_bytes()).hexdigest()
    custom_digest = hashlib.sha256(custom.read_bytes()).hexdigest()
    monkeypatch.setattr(model_manager, "crnn_path", lambda: vendor)
    monkeypatch.setattr(model_manager, "CRNN_SHA256", vendor_digest)
    monkeypatch.setattr(model_manager, "CRNN_SIZE", vendor.stat().st_size)
    monkeypatch.setattr(
        model_manager,
        "active_crnn_model",
        lambda: (custom, custom_digest, custom.stat().st_size),
    )
    created = []

    class Options:
        def add_session_config_entry(self, *_args):
            return None

    class Session:
        def __init__(self, path, **_kwargs):
            created.append(path)

        def get_inputs(self):
            return [types.SimpleNamespace(name="plate")]

        def run(self, _outputs, inputs):
            assert inputs["plate"].shape == (1, 1, 32, 128)
            return [np.zeros(
                (1, 2, len(onnx_crnn.CRNN_LABELS) + 1),
                dtype=np.float32,
            )]

    monkeypatch.setitem(sys.modules, "onnxruntime", types.SimpleNamespace(
        SessionOptions=Options,
        InferenceSession=Session,
        ExecutionMode=types.SimpleNamespace(ORT_SEQUENTIAL=0),
        GraphOptimizationLevel=types.SimpleNamespace(ORT_ENABLE_ALL=1),
    ))
    monkeypatch.setattr(
        onnx_crnn,
        "ctc_greedy_decode",
        lambda _logits: ("31ط55674", 0.84),
    )
    onnx_crnn.clear_crnn_sessions()

    first = onnx_crnn.read_plate_platrix(
        np.zeros((32, 160, 3), dtype=np.uint8),
        engine_key=live_worker.ENGINE_V3_INFERENCE_KEY,
    )
    second = onnx_crnn.read_plate_platrix(
        np.zeros((32, 160, 3), dtype=np.uint8),
        engine_key=live_worker.ENGINE_V3_INFERENCE_KEY,
    )

    assert first == second == (PLATE, 0.84)
    assert created == [str(vendor)]
    assert str(custom) not in created
    assert onnx_crnn.get_crnn_status()["model_path"] == str(vendor)
    onnx_crnn.clear_crnn_sessions()


def test_production_pipeline_ignores_detector_attached_custom_ocr(monkeypatch):
    crop = np.full((40, 180, 3), 170, dtype=np.uint8)
    monkeypatch.setattr(
        "app.ai.pipeline.detect_plates",
        lambda *_args, **_kwargs: [{
            "crop": crop,
            "bbox": (10, 20, 190, 60),
            "confidence": 0.91,
            "method": "yolox-custom-onnx",
            "direct_text": OTHER_PLATE,
            "direct_ocr_confidence": 0.99,
            "direct_ocr_attempted": True,
            "plate_hypotheses": [{
                "plate_norm": "31ط55874",
                "confidence": 0.99,
            }],
        }],
    )
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            PLATE,
            0.82,
            "hezar-crnn-fa-v2-onnx",
        ),
    )

    row = process_frame(
        np.zeros((100, 220, 3), dtype=np.uint8),
        detector_variant="yolox",
    )[0]

    assert row["plate"] == PLATE
    assert row["ocr_engine"] == "hezar-crnn-fa-v2-onnx"
    assert row["ocr_disagreement"] is False
    assert row["dedicated_ocr_ignored"] is True
    assert all(
        hypothesis["plate_norm"] != "31ط55874"
        for hypothesis in row["plate_hypotheses"]
    )


def test_live_worker_keeps_camera_state_but_shares_model_sessions(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(config={
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    })
    worker._states[12] = state
    observed = []
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolox",
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **kwargs: observed.append(kwargs) or [],
    )
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    state.busy = True
    worker._process(state, (12, "cam", frame, 1.0))
    worker.shutdown()

    assert observed[0]["engine_key"] == 12
    assert observed[0]["inference_key"] == live_worker.ENGINE_V3_INFERENCE_KEY
    assert observed[0]["detector_variant"] == "yolox"


def test_live_empty_new_revision_resets_old_tracker_before_update(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(config={
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    })
    state.detector_model_revision = "revision-a"
    observation = {
        "plate": PLATE,
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.9,
        "quality_score": 0.8,
        "bbox": (20, 20, 140, 55),
        "crop": np.zeros((35, 120, 3), dtype=np.uint8),
    }
    state.tracker.update([observation], timestamp=0.0)
    old_tracker = state.tracker
    worker._states[18] = state
    monkeypatch.setattr(worker, "_selected_detector_variant", lambda: "yolox")

    def empty_new_revision(*_args, **kwargs):
        kwargs["runtime_metadata"].update(
            detector_variant="yolox",
            detector_model_revision="revision-b",
        )
        return []

    monkeypatch.setattr(live_worker, "process_frame", empty_new_revision)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("old revision must not be flushed")
        ),
    )

    state.busy = True
    worker._process(
        state,
        (18, "camera", np.zeros((80, 160, 3), dtype=np.uint8), 1.0),
    )
    worker.shutdown()

    assert state.tracker is not old_tracker
    assert state.tracker.active_track_ids() == set()
    assert state.detector_model_revision == "revision-b"


def test_detector_switch_waits_for_old_commit_before_persisting_setting(
    monkeypatch,
):
    from app.ai import onnx_detector

    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    worker._states[19] = state
    monkeypatch.setattr(onnx_detector, "clear_detector_sessions", lambda: None)
    switch_attempted = threading.Event()

    class ObservedRLock:
        def __init__(self):
            self.lock = threading.RLock()

        def acquire(self, *args, **kwargs):
            if threading.current_thread().name == "model-switch":
                switch_attempted.set()
            return self.lock.acquire(*args, **kwargs)

        def release(self):
            self.lock.release()

        def __enter__(self):
            self.acquire()
            return self

        def __exit__(self, *_args):
            self.release()

    state.model_switch_lock = ObservedRLock()
    old_guard_passed = threading.Event()
    release_old_commit = threading.Event()
    order = []

    def old_commit():
        with state.model_switch_lock:
            assert worker._detector_generation == 0
            old_guard_passed.set()
            release_old_commit.wait(1.0)
            order.append("old-event")

    old_thread = threading.Thread(target=old_commit, name="old-commit")
    old_thread.start()
    assert old_guard_passed.wait(1.0)

    switch_thread = threading.Thread(
        target=lambda: worker.invalidate_model_cache(
            "yolo11n",
            persist_setting=lambda _key, _value: order.append("setting"),
        ),
        name="model-switch",
    )
    switch_thread.start()
    assert switch_attempted.wait(1.0)
    assert order == []
    assert worker._detector_generation == 0

    release_old_commit.set()
    old_thread.join(1.0)
    switch_thread.join(1.0)
    worker.shutdown()

    assert not old_thread.is_alive()
    assert not switch_thread.is_alive()
    assert order == ["old-event", "setting"]
    assert worker._detector_generation == 1


def test_queued_worker_claims_newest_pending_frame_and_preserves_wake(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    config = {
        "id": 17,
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    old_mask = np.full((20, 40), 1, dtype=np.uint8)
    new_mask = np.full((20, 40), 2, dtype=np.uint8)
    activities = iter((
        FrameActivity(0.8, True, False, True, old_mask),
        FrameActivity(0.1, False, False, False, new_mask),
    ))
    state = live_worker._CameraState(config=config)
    state.activity = types.SimpleNamespace(observe=lambda _frame: next(activities))
    worker._states[17] = state
    monkeypatch.setattr(
        worker,
        "_config",
        lambda _camera_id, current, _now: current.config,
    )
    monkeypatch.setattr(worker, "_selection_score", lambda *_args: 1.0)
    monkeypatch.setattr(worker, "_selected_detector_variant", lambda: "yolox")
    monkeypatch.setattr(
        live_worker.LiveANPRWorker,
        "_post_inference_delay",
        staticmethod(lambda *_args: 99.0),
    )

    blocker_started = threading.Event()
    release_blocker = threading.Event()

    def blocker():
        blocker_started.set()
        release_blocker.wait(2.0)

    worker._executor.submit(blocker)
    assert blocker_started.wait(1.0)
    real_submit = worker._executor.submit
    submitted_callbacks = []

    def tracked_submit(callback, *args, **kwargs):
        submitted_callbacks.append(callback)
        return real_submit(callback, *args, **kwargs)

    monkeypatch.setattr(worker._executor, "submit", tracked_submit)
    observed = []
    inference_started = threading.Event()

    def process(frame, *_args, **kwargs):
        observed.append((int(frame[0, 0, 0]), kwargs.get("exclusion_mask")))
        inference_started.set()
        return []

    monkeypatch.setattr(live_worker, "process_frame", process)
    old_frame = np.full((20, 40, 3), 11, dtype=np.uint8)
    new_frame = np.full((20, 40, 3), 22, dtype=np.uint8)

    worker.submit(17, "queued camera", old_frame)
    worker.submit(17, "queued camera", new_frame)
    with worker._lock:
        assert state.busy is True
        assert int(state.pending[2][0, 0, 0]) == 22
        state.burst_frames_remaining = 0
    assert len(submitted_callbacks) == 1

    release_blocker.set()
    assert inference_started.wait(1.0)
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with worker._lock:
            if not state.busy:
                break
        time.sleep(0.005)

    with worker._lock:
        assert state.busy is False
        assert state.pending is None
        assert state.next_inference_at - time.monotonic() < 0.25
    worker.shutdown()

    assert len(observed) == 1
    assert observed[0][0] == 22
    assert observed[0][1] is new_mask


def test_pipeline_routes_one_shared_key_to_detector_and_ocr(monkeypatch):
    detector_keys = []
    ocr_keys = []
    crop = np.full((40, 180, 3), 170, dtype=np.uint8)

    def detect(_frame, **kwargs):
        detector_keys.append(kwargs.get("engine_key"))
        return [{
            "crop": crop,
            "bbox": (10, 20, 190, 60),
            "confidence": 0.91,
            "method": "yolox-custom-onnx",
        }]

    def read(_crop, engine_key=None, **_kwargs):
        ocr_keys.append(engine_key)
        return PLATE, 0.82, "hezar-crnn-fa-v2-onnx"

    monkeypatch.setattr("app.ai.pipeline.detect_plates", detect)
    monkeypatch.setattr("app.ai.pipeline.read_plate_candidate", read)
    frame = np.zeros((100, 220, 3), dtype=np.uint8)

    for camera_id in (1, 2):
        process_frame(
            frame,
            engine_key=camera_id,
            inference_key=live_worker.ENGINE_V3_INFERENCE_KEY,
            detector_variant="yolox",
        )

    assert detector_keys == [
        live_worker.ENGINE_V3_INFERENCE_KEY,
        live_worker.ENGINE_V3_INFERENCE_KEY,
    ]
    assert ocr_keys == detector_keys


def test_pipeline_attributes_builtin_detector_revision_without_yolox_prefix(
    monkeypatch,
):
    crop = np.full((40, 180, 3), 170, dtype=np.uint8)

    def detect(_frame, runtime_metadata=None, **_kwargs):
        runtime_metadata.update(
            detector_variant="yolo11n",
            detector_model_revision="fixed-sha-prefix",
        )
        return [{
            "crop": crop,
            "bbox": (10, 20, 190, 60),
            "confidence": 0.91,
            "method": "yolo11n-plate-onnx",
        }]

    monkeypatch.setattr("app.ai.pipeline.detect_plates", detect)
    monkeypatch.setattr(
        "app.ai.pipeline.read_plate_candidate",
        lambda *_args, **_kwargs: (
            PLATE,
            0.82,
            "hezar-crnn-fa-v2-onnx",
        ),
    )

    row = process_frame(
        np.zeros((100, 220, 3), dtype=np.uint8),
        detector_variant="yolo11n",
        runtime_metadata={},
    )[0]

    assert row["detector_model_revision"] == "fixed-sha-prefix"
    assert row["model_revision"].startswith("yolo11n:fixed-sha-prefix+")
    assert not row["model_revision"].startswith("yolox:")
