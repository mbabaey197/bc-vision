import cv2
import numpy as np

from app.ai.detector import _iou, _nms, detect_plates, detector_status


def make_plate_scene(brightness=45, angle=0, blur=0):
    image = np.full((480, 800, 3), brightness, dtype=np.uint8)
    plate = np.full((84, 340, 3), 235, dtype=np.uint8)
    cv2.rectangle(plate, (2, 2), (337, 81), (20, 20, 20), 3)
    x = 28
    for width in [18, 17, 8, 18, 17, 18, 18, 17]:
        cv2.rectangle(
            plate,
            (x, 18),
            (x + width, 68),
            (15, 15, 15),
            -1,
        )
        x += width + 17
    canvas = np.zeros_like(image)
    y1, x1 = 205, 230
    canvas[y1:y1 + plate.shape[0], x1:x1 + plate.shape[1]] = plate
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[y1:y1 + plate.shape[0], x1:x1 + plate.shape[1]] = 255
    if angle:
        matrix = cv2.getRotationMatrix2D((400, 240), angle, 1.0)
        canvas = cv2.warpAffine(canvas, matrix, (800, 480))
        mask = cv2.warpAffine(mask, matrix, (800, 480))
    image[mask > 0] = canvas[mask > 0]
    if blur:
        image = cv2.GaussianBlur(image, (blur, blur), 0)
    return image


def test_fallback_detects_clear_plate(monkeypatch):
    monkeypatch.setattr("app.ai.detector.load_model", lambda: None)
    rows = detect_plates(make_plate_scene(), min_confidence=0.1)
    assert rows
    assert rows[0]["crop"].size > 0
    assert rows[0]["method"].startswith("opencv")


def test_fallback_handles_dark_rotated_blurred(monkeypatch):
    monkeypatch.setattr("app.ai.detector.load_model", lambda: None)
    for scene in [
        make_plate_scene(brightness=8),
        make_plate_scene(angle=8),
        make_plate_scene(blur=5),
    ]:
        rows = detect_plates(scene, min_confidence=0.08)
        assert rows


def test_nms_removes_overlaps():
    rows = [
        {"bbox": (0, 0, 100, 30), "confidence": 0.8},
        {"bbox": (5, 2, 102, 31), "confidence": 0.7},
        {"bbox": (200, 0, 300, 30), "confidence": 0.6},
    ]
    kept = _nms(rows)
    assert len(kept) == 2
    assert _iou(rows[0]["bbox"], rows[1]["bbox"]) > 0.7


def test_status_is_safe():
    status = detector_status()
    assert "model_exists" in status
