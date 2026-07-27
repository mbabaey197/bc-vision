from concurrent.futures import ThreadPoolExecutor
import threading
import time

import cv2
import numpy as np

from app.ai.detector import (
    _choose_recovery_result,
    _configure_cpu_threads,
    _cpu_thread_limit,
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


def test_cpu_thread_limit_is_clamped(monkeypatch):
    monkeypatch.setenv("BCVISION_CPU_THREADS", "0")
    assert _cpu_thread_limit() == 1
    monkeypatch.setenv("BCVISION_CPU_THREADS", "99")
    assert _cpu_thread_limit() == 8


def test_cpu_thread_limit_is_applied(monkeypatch):
    applied = []
    monkeypatch.setenv("BCVISION_CPU_THREADS", "3")
    monkeypatch.setattr(
        "app.ai.detector.cv2.setNumThreads",
        lambda value: applied.append(("opencv", value)),
    )

    class Torch:
        @staticmethod
        def set_num_threads(value):
            applied.append(("torch", value))

    monkeypatch.setitem(__import__("sys").modules, "torch", Torch())
    assert _configure_cpu_threads() == 3
    assert applied == [("opencv", 3), ("torch", 3)]


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


def test_recovery_result_requires_evidence_before_changing_digit():
    original = ("18-ب-987-33", 0.80)
    assert _choose_recovery_result(
        original,
        ("18-ب-987-32", 0.84),
    ) == ("18-ب-987-32", 0.84, "recovered")
    assert _choose_recovery_result(
        original,
        ("18-ب-987-32", 0.82),
    ) == ("", 0.0, "ambiguous")
    assert _choose_recovery_result(
        original,
        ("18-ب-987-33", 0.86),
    ) == ("18-ب-987-33", 0.86, "agreement")


def test_parallel_detector_calls_are_serialized(monkeypatch):
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()

    class Model:
        names = {index: str(index) for index in range(32)}

        @staticmethod
        def predict(*_args, **_kwargs):
            nonlocal active, maximum_active
            with state_lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.03)
            with state_lock:
                active -= 1
            return [type("Result", (), {"boxes": []})()]

    model = Model()
    monkeypatch.setattr(
        "app.ai.detector.load_model",
        lambda: model,
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as executor:
        rows = list(executor.map(detect_plates, [frame, frame]))

    assert rows == [[], []]
    assert maximum_active == 1


def test_verified_model_uses_only_plate_class():
    model = type(
        "Model",
        (),
        {"names": {index: str(index) for index in range(32)}},
    )()
    assert _plate_class_ids(model) == [30]


def test_successful_empty_yolo_result_uses_geometry_fallback(
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
    candidate = {
        "crop": np.zeros((20, 80, 3), dtype=np.uint8),
        "bbox": (10, 10, 90, 30),
        "confidence": 0.6,
        "method": "opencv",
    }
    monkeypatch.setattr(
        "app.ai.detector._opencv_candidates",
        lambda *_args, **_kwargs: [candidate],
    )

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    assert detect_plates(frame) == [candidate]
