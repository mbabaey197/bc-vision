import os
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("BCVISION_RUN_AI_INTEGRATION") != "1",
    reason="AI integration runtime is disabled",
)


def test_verified_models_and_real_engines_load(tmp_path):
    import onnx
    import onnxruntime
    import torch

    from app.ai.model_manager import model_status, prepare_models
    from app.ai.onnx_cnn import get_cnn_status, warmup_cnn
    from app.ai.onnx_crnn import (
        get_crnn_status,
        read_plate_crnn,
    )
    from app.ai.onnx_detector import (
        detect_plates_onnx,
        detector_status,
    )
    from app.ai.onnx_hezar import (
        hezar_status,
        read_plate_hezar_primary,
    )

    prepared = prepare_models(download=True)
    status = model_status(selected_detector="yolo11n")
    assert status["detector_ready"]
    assert status["detector_yolo11n_ready"]
    assert status["detector_yolov8n_ready"]
    assert status["detector_fallback_ready"]
    assert status["hezar_ready"]
    assert status["crnn_ready"]
    assert status["cnn_ready"]
    assert prepared["detector"] == status["detector_path"]
    assert (
        prepared["detector_yolo11n"]
        == status["detector_yolo11n_path"]
    )
    assert (
        prepared["detector_yolov8n"]
        == status["detector_yolov8n_path"]
    )
    assert (
        prepared["detector_fallback"]
        == status["detector_fallback_path"]
    )
    assert prepared["hezar"] == status["hezar_path"]
    assert prepared["crnn"] == status["crnn_path"]
    assert prepared["cnn"] == status["cnn_path"]

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert float(tensor.sum().item()) == 10.0
    assert torch.__version__
    assert onnx.__version__
    assert onnxruntime.__version__

    blank = np.full((320, 640, 3), 255, dtype=np.uint8)
    output = detect_plates_onnx(
        blank,
        min_confidence=0.05,
        max_results=2,
        engine_key="integration-yolo11n",
        detector_variant="yolo11n",
    )
    assert isinstance(output, list)
    detector_state = detector_status()
    assert detector_state["model_loaded"] is True
    assert detector_state["selected_variant"] == "yolo11n"

    output = detect_plates_onnx(
        blank,
        min_confidence=0.05,
        max_results=2,
        engine_key="integration-yolov8n",
        detector_variant="yolov8n",
    )
    assert isinstance(output, list)
    detector_state = detector_status()
    assert detector_state["model_loaded"] is True
    assert detector_state["selected_variant"] == "yolov8n"

    read_plate_hezar_primary(
        np.zeros((32, 384, 3), dtype=np.uint8),
        engine_key="integration-test",
    )
    assert hezar_status()["model_loaded"] is True

    read_plate_crnn(
        np.zeros((32, 128, 3), dtype=np.uint8),
        engine_key="integration-test",
    )
    assert get_crnn_status()["model_loaded"] is True
    assert warmup_cnn("integration-test")["model_loaded"] is True
    assert get_cnn_status()["model_loaded"] is True

    from app.ai.training_worker import train_candidate
    from app.ai.training_manifest import operator_dataset_fingerprint

    samples = []
    for index, (label, split) in enumerate((
        ("12ب34567", "train"),
        ("31ط55674", "train"),
        ("55د63921", "validation"),
    )):
        image_path = tmp_path / f"sample-{index}.png"
        image = np.full((48, 180, 3), 220 - index * 20, dtype=np.uint8)
        cv2.rectangle(image, (2, 2), (177, 45), (25, 25, 25), 2)
        assert cv2.imwrite(str(image_path), image)
        samples.append({
            "feedback_id": index + 1,
            "image_path": str(image_path),
            "plate": label,
            "group_id": label,
            "sha256": hashlib.sha256(
                image_path.read_bytes()
            ).hexdigest().upper(),
            "split": split,
        })
    manifest = tmp_path / "dataset.json"
    manifest.write_text(
        json.dumps({
            "schema": 2,
            "training_source": "operator-confirmed-only",
            "golden_benchmark_data": False,
            "dataset_fingerprint": operator_dataset_fingerprint(samples),
            "samples": samples,
        }, ensure_ascii=False),
        encoding="utf-8",
    )
    trained = train_candidate(
        manifest,
        tmp_path / "candidate",
        device="cpu",
        epochs=4,
    )
    candidate = Path(trained["candidate_path"])
    assert candidate.is_file()
    assert hashlib.sha256(candidate.read_bytes()).hexdigest().upper() == (
        trained["candidate_sha256"]
    )
    assert 0.0 <= trained["candidate_accuracy"] <= 1.0
