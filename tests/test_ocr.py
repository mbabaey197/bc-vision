import numpy as np

from app.ai.ocr import (
    _align_to_template,
    _assemble_detections,
    _variants,
    get_ocr_status,
    read_plate,
)
from app.ai.plate_rules import plausible_plate


def test_reassembles_real_split_output():
    detections = [
        {"text": "IR", "confidence": 0.4601, "x_center": 70.0},
        {"text": "ب١٢", "confidence": 0.9972, "x_center": 279.0},
        {"text": "٣٤٥", "confidence": 0.7839, "x_center": 516.0},
        {"text": "٧ع", "confidence": 0.2839, "x_center": 776.0},
    ]
    text, confidence = _assemble_detections(detections)
    assert text == "12-ب-345-67"
    assert plausible_plate(text)
    assert confidence > 0.60


def test_ignores_noise_tokens():
    detections = [
        {"text": "XX", "confidence": 0.98, "x_center": 20.0},
        {"text": "ب١٢", "confidence": 0.91, "x_center": 120.0},
        {"text": "٣٤٥", "confidence": 0.86, "x_center": 280.0},
        {"text": "٦٧", "confidence": 0.82, "x_center": 430.0},
        {"text": "ZZZ", "confidence": 0.99, "x_center": 590.0},
    ]
    text, _ = _assemble_detections(detections)
    assert text == "12-ب-345-67"


def test_position_based_confusion_repair():
    hypotheses = _align_to_template("I2B34S67")
    assert hypotheses
    assert hypotheses[0][0] == "12ب34567"


def test_invalid_noise_does_not_become_valid():
    text, confidence = _assemble_detections([
        {"text": "HELLO", "confidence": 0.99, "x_center": 10.0},
        {"text": "WORLD", "confidence": 0.99, "x_center": 100.0},
    ])
    assert not plausible_plate(text)
    assert confidence <= 0.45


def test_preprocessing_handles_extreme_images():
    for value in (0, 127, 255):
        image = np.full((48, 180, 3), value, dtype=np.uint8)
        variants = _variants(image)
        assert variants
        assert all(variant.size > 0 for variant in variants)


def test_empty_input_is_safe():
    assert read_plate(None) == ("", 0.0)
    assert isinstance(get_ocr_status(), dict)
