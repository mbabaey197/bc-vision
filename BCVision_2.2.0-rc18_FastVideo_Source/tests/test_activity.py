import cv2
import numpy as np

from app.ai.activity import (
    FrameActivityAnalyzer,
    masked_bbox_ratio,
    suppress_static_overlay_rows,
)


def _overlay_frame():
    frame = np.full((180, 320, 3), 80, dtype=np.uint8)
    cv2.putText(
        frame,
        "CAM 01 12:45:10",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return frame


def test_fixed_cctv_text_becomes_an_exclusion_mask():
    analyzer = FrameActivityAnalyzer(
        max_width=320,
        overlay_warmup_frames=6,
    )
    frame = _overlay_frame()
    activity = None
    for _index in range(9):
        activity = analyzer.observe(frame)

    assert activity is not None
    assert activity.exclusion_mask is not None
    assert masked_bbox_ratio(
        activity.exclusion_mask,
        (5, 2, 190, 34),
    ) > 0.22
    rows = suppress_static_overlay_rows(
        [{
            "bbox": (5, 2, 190, 34),
            "plate": "ناخوانا",
        }],
        activity.exclusion_mask,
    )
    assert rows == []


def test_current_motion_clears_overlay_mask_and_wakes_inference():
    analyzer = FrameActivityAnalyzer(
        max_width=320,
        overlay_warmup_frames=6,
    )
    static = _overlay_frame()
    for _index in range(9):
        analyzer.observe(static)

    moving = static.copy()
    cv2.rectangle(
        moving,
        (20, 4),
        (190, 45),
        (15, 210, 15),
        -1,
    )
    activity = analyzer.observe(moving)

    assert activity.moving is True
    assert activity.wake_inference is True
    assert activity.motion_score > 0.012
    assert masked_bbox_ratio(
        activity.exclusion_mask,
        (20, 4, 190, 45),
    ) < 0.22
