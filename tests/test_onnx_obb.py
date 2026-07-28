import cv2
import numpy as np

from app.ai.onnx_obb import decode_obb_output, rectify_plate


def test_obb_decoder_accepts_traditional_and_end_to_end_outputs():
    traditional = np.zeros((1, 6, 20), dtype=np.float32)
    traditional[0, :, 0] = [320, 240, 180, 55, 0.91, 0.15]
    decoded = decode_obb_output(traditional, min_confidence=0.5)
    assert len(decoded) == 1
    assert decoded[0]["confidence"] == np.float32(0.91)
    assert decoded[0]["corners"].shape == (4, 2)

    end_to_end = np.array(
        [[[220, 196, 380, 244, 0.88, 0, -0.2]]],
        dtype=np.float32,
    )
    decoded = decode_obb_output(end_to_end, min_confidence=0.5)
    assert len(decoded) == 1
    assert decoded[0]["class_id"] == 0
    assert decoded[0]["angle"] == np.float32(-0.2)
    center = decoded[0]["corners"].mean(axis=0)
    assert np.allclose(center, [300, 220], atol=0.01)


def test_perspective_rectification_returns_horizontal_plate():
    image = np.zeros((180, 320, 3), dtype=np.uint8)
    corners = np.array(
        [[70, 65], [255, 45], [265, 105], [75, 125]],
        dtype=np.float32,
    )
    cv2.fillConvexPoly(image, corners.astype(np.int32), (220, 220, 220))

    crop = rectify_plate(image, corners)

    assert crop is not None
    assert crop.shape[1] > crop.shape[0]
    assert float(crop.mean()) > 150
