import numpy as np

from app.ai import next_engine


def test_shadow_failure_never_changes_baseline_output(monkeypatch):
    expected = [{"plate": "31-ط-556-74"}]

    def fail(*_args, **_kwargs):
        raise RuntimeError("candidate failed")

    monkeypatch.setattr(next_engine, "process_frame_next", fail)
    router = next_engine.EngineRouter()

    result = router.process(
        np.zeros((32, 64, 3), dtype=np.uint8),
        baseline=lambda: expected,
        mode="shadow",
    )

    assert result.primary is expected
    assert result.shadow == []
    assert result.mode == "shadow"
    assert "candidate failed" in result.error


def test_next_runtime_failure_rolls_back_and_uses_baseline(monkeypatch):
    rollbacks = []

    def fail(*_args, **_kwargs):
        raise RuntimeError("invalid model output")

    monkeypatch.setattr(next_engine, "process_frame_next", fail)
    monkeypatch.setattr(
        next_engine,
        "rollback_to_baseline",
        lambda reason: rollbacks.append(reason),
    )
    router = next_engine.EngineRouter()

    result = router.process(
        np.zeros((32, 64, 3), dtype=np.uint8),
        baseline=lambda: [{"plate": "baseline"}],
        mode="next",
    )

    assert result.primary == [{"plate": "baseline"}]
    assert result.mode == "baseline"
    assert result.degraded is True
    assert rollbacks and "invalid model output" in rollbacks[0]


def test_shadow_cct_reuses_baseline_detector_crop(monkeypatch):
    import app.ai.pipeline as pipeline

    crop = np.full((32, 96, 3), 180, dtype=np.uint8)
    baseline_row = {
        "crop": crop,
        "bbox": (10, 12, 106, 44),
        "confidence": 0.81,
        "detector_confidence": 0.73,
        "method": "yolov8-onnx-light",
    }
    monkeypatch.setattr(
        next_engine,
        "verified_next_manifest",
        lambda: {
            "engine": "bcvision-rc15",
            "release_id": "rc15-internal-cct-stage4",
            "models": {
                "detector": {
                    "runtime": "baseline-yolov8-onnx",
                },
                "ocr": {
                    "runtime": "fast-plate-ocr-cct",
                },
            },
        },
    )
    monkeypatch.setattr(
        next_engine,
        "detector_status",
        lambda: {
            "attempted": True,
            "model_loaded": True,
            "error": "",
        },
    )
    monkeypatch.setattr(
        next_engine,
        "detect_plates_obb",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("OBB detector must not run")
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "image_quality",
        lambda _crop: {"score": 0.82},
    )
    monkeypatch.setattr(
        next_engine,
        "_read_candidate_ocr",
        lambda *_args, **_kwargs: (
            {
                "accepted": True,
                "plate": "31-ط-556-74",
                "plate_norm": "31ط55674",
                "raw_plate_norm": "31ط55674",
                "confidence": 0.94,
                "reason": "",
                "hypotheses": [{
                    "plate": "31-ط-556-74",
                    "plate_norm": "31ط55674",
                    "confidence": 0.94,
                    "positions": {},
                }],
            },
            {
                "attempted": True,
                "model_loaded": True,
                "error": "",
            },
            "fast-plate-ocr-cct",
        ),
    )

    result = next_engine.process_frame_next(
        np.zeros((80, 160, 3), dtype=np.uint8),
        detections=[baseline_row],
    )

    assert len(result) == 1
    assert result[0]["plate_norm"] == "31ط55674"
    assert result[0]["detector_confidence"] == 0.73
    assert result[0]["detector_runtime"] == "baseline-yolov8-onnx"
    assert result[0]["crop"] is crop
