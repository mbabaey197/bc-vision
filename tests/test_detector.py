import cv2
import numpy as np

from app.ai.detector import (
    _iou,
    _nms,
    _plate_class_ids,
    _select_plate_sequence,
    detect_plates,
    detector_status,
)


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


def _characters(text, confidences):
    class_ids = {
        **{str(value): value for value in range(10)},
        "ب": 11,
        "ط": 19,
        "ق": 22,
    }
    return [
        {
            "class_id": class_ids[character],
            "confidence": confidence,
            "x_center": index * 20.0,
            "bbox": (
                index * 20.0,
                0.0,
                index * 20.0 + 12.0,
                24.0,
            ),
        }
        for index, (character, confidence) in enumerate(
            zip(text, confidences)
        )
    ]


def test_character_decoder_removes_low_confidence_country_strip_noise():
    text, confidence = _select_plate_sequence(
        _characters(
            "027ط25374",
            [0.17, 0.86, 0.85, 0.85, 0.85, 0.85, 0.85, 0.87, 0.86],
        )
    )
    assert text == "27-ط-253-74"
    assert confidence > 0.84


def test_character_decoder_selects_highest_confidence_plate_template():
    text, _ = _select_plate_sequence(
        _characters(
            "418ب987232",
            [0.18, 0.82, 0.86, 0.89, 0.85, 0.86, 0.86, 0.46, 0.87, 0.86],
        )
    )
    assert text == "18-ب-987-32"


def test_verified_model_uses_only_plate_class():
    model = type(
        "Model",
        (),
        {"names": {index: str(index) for index in range(32)}},
    )()
    assert _plate_class_ids(model) == [30]


def test_successful_empty_yolo_result_does_not_run_fallback(
    monkeypatch,
):
    class Model:
        names = {index: str(index) for index in range(32)}

        @staticmethod
        def predict(*_args, **_kwargs):
            return [type("Result", (), {"boxes": []})()]

    monkeypatch.setattr(
        "app.ai.detector.load_model",
        lambda: Model(),
    )
    monkeypatch.setattr(
        "app.ai.detector._opencv_candidates",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fallback should not run")
        ),
    )

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detect_plates(frame) == []
