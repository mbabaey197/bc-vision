import numpy as np

from app.ai.onnx_crnn import CRNN_LABELS
from app.ai.onnx_hezar import (
    accept_hypotheses,
    ctc_beam_hypotheses,
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
