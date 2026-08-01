import numpy as np

from app.ai.onnx_crnn import CRNN_LABELS
from app.ai.onnx_hezar import (
    accept_hypotheses,
    ctc_beam_hypotheses,
    prepare_hezar_input,
)


def test_constrained_ctc_beam_decodes_iran_plate_layout():
    target = "31ط55674"
    blank = len(CRNN_LABELS)
    logits = np.full(
        (len(target) * 2, blank + 1),
        -9.0,
        dtype=np.float32,
    )
    for position, character in enumerate(target):
        logits[position * 2, CRNN_LABELS.index(character)] = 9.0
        logits[position * 2 + 1, blank] = 9.0

    rows = ctc_beam_hypotheses(logits, beam_width=6, top_k=3)

    assert rows
    assert rows[0]["plate_norm"] == target
    assert rows[0]["plate"] == "31-ط-556-74"


def test_close_ocr_alternatives_are_rejected_as_unreadable():
    result = accept_hypotheses(
        [
            {
                "plate_norm": "31ط55674",
                "plate": "31-ط-556-74",
                "confidence": 0.52,
            },
            {
                "plate_norm": "31ط56674",
                "plate": "31-ط-566-74",
                "confidence": 0.48,
            },
        ],
        min_confidence=0.50,
        min_position_margin=0.12,
    )

    assert result["accepted"] is False
    assert result["plate"] == "ناخوانا"
    assert result["plate_norm"] == ""


def test_hezar_blank_at_zero_keeps_real_label_indices():
    labels = ["", "۱", "۲", "۳", "ط", "۴", "۵", "۶", "۷"]
    target_indices = [2, 3, 4, 5, 6, 7, 8, 1]
    logits = np.full(
        (len(target_indices) * 2, len(labels)),
        -9.0,
        dtype=np.float32,
    )
    for position, index in enumerate(target_indices):
        logits[position * 2, index] = 9.0
        logits[position * 2 + 1, 0] = 9.0

    rows = ctc_beam_hypotheses(
        logits,
        labels=labels,
        blank_index=0,
        beam_width=6,
        top_k=3,
    )

    assert rows
    assert rows[0]["plate_norm"] == "23ط45671"


def test_hezar_preprocessing_supports_mirror_and_list_normalization():
    image = np.zeros((24, 64, 3), dtype=np.uint8)
    image[:, :32] = 255
    tensor = prepare_hezar_input(
        image,
        {
            "input_height": 24,
            "input_width": 64,
            "channels": 1,
            "mirror": True,
            "mean": [0.5],
            "std": [0.5],
        },
    )

    assert tensor.shape == (1, 1, 24, 64)
    assert float(tensor[0, 0, :, :8].mean()) < -0.9
    assert float(tensor[0, 0, :, -8:].mean()) > 0.9
