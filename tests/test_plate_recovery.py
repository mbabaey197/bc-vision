import cv2
import numpy as np

from app.ai.plate_recovery import (
    blur_score,
    recover_mild_blur,
    should_attempt_recovery,
)


def _plate_crop():
    image = np.full((48, 180, 3), 230, dtype=np.uint8)
    cv2.rectangle(image, (2, 2), (177, 45), (20, 20, 20), 2)
    for x in range(18, 164, 18):
        cv2.rectangle(image, (x, 10), (x + 7, 38), (15, 15, 15), -1)
    return image


def test_recovery_gate_rejects_missing_and_destroyed_images():
    assert not should_attempt_recovery(None, "", 0.0)
    destroyed = np.full((48, 180, 3), 127, dtype=np.uint8)
    assert blur_score(destroyed) == 0.0
    assert not should_attempt_recovery(destroyed, "", 0.0)


def test_mild_blur_is_restored_at_same_size():
    blurred = cv2.GaussianBlur(_plate_crop(), (7, 7), 0)
    assert should_attempt_recovery(blurred, "18-ب-987-32", 0.77)

    restored, metadata = recover_mild_blur(blurred)

    assert restored.shape == blurred.shape
    assert restored.dtype == np.uint8
    assert metadata["applied"]
    assert metadata["method"] == "motion-deblur+ai-reread"
    assert blur_score(restored) > blur_score(blurred)


def test_clear_confident_plate_skips_extra_work():
    clear = _plate_crop()
    assert not should_attempt_recovery(
        clear,
        "18-ب-987-32",
        0.90,
    )
