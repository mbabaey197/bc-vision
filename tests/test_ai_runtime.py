import os

import cv2
import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("BCVISION_RUN_AI_INTEGRATION") != "1",
    reason="AI integration runtime is disabled",
)


def test_verified_models_and_real_engines_load():
    import easyocr
    import torch
    import torchvision
    import ultralytics

    from app.ai.detector import detector_status, load_model
    from app.ai.model_manager import model_status, prepare_models
    from app.ai.ocr import _get_easyocr_reader

    prepared = prepare_models(download=True)
    status = model_status()
    assert status["detector_ready"]
    assert status["easyocr_ready"]
    assert prepared["detector"] == status["detector_path"]
    assert prepared["easyocr"] == status["easyocr_path"]

    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    assert float(tensor.sum().item()) == 10.0
    assert torch.__version__
    assert torchvision.__version__
    assert ultralytics.__version__
    assert easyocr.__version__

    model = load_model()
    assert model is not None
    assert detector_status()["model_loaded"]

    blank = np.full((320, 640, 3), 255, dtype=np.uint8)
    model.predict(
        blank,
        verbose=False,
        conf=0.05,
        imgsz=640,
        max_det=2,
    )

    reader = _get_easyocr_reader()
    output = reader.readtext(
        cv2.cvtColor(blank, cv2.COLOR_BGR2GRAY),
        detail=1,
        paragraph=False,
    )
    assert isinstance(output, list)
