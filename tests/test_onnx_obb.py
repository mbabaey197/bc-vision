import cv2
import numpy as np

from app.ai import onnx_obb
from app.ai.onnx_obb import (
    _ppyoloe_r_feeds,
    decode_obb_output,
    decode_ppyoloe_r_outputs,
    prepare_ppyoloe_r_input,
    rectify_plate,
)


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


def test_ppyoloe_r_decoder_accepts_official_boxes_and_scores():
    boxes = np.array(
        [[[
            70, 65, 255, 45, 265, 105, 75, 125,
        ], [
            72, 66, 253, 47, 262, 103, 77, 123,
        ]]],
        dtype=np.float32,
    )
    scores = np.array([[[0.91, 0.72]]], dtype=np.float32)

    decoded = decode_ppyoloe_r_outputs(
        boxes,
        scores,
        min_confidence=0.5,
        nms_threshold=0.1,
    )

    assert len(decoded) == 1
    assert decoded[0]["confidence"] == np.float32(0.91)
    assert decoded[0]["corners"].shape == (4, 2)


def test_ppyoloe_r_preprocessing_supplies_three_official_inputs():
    image = np.full((360, 640, 3), 128, dtype=np.uint8)
    spec = {
        "input_width": 640,
        "input_height": 640,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "pad_to_stride": 32,
    }

    tensor, im_shape, scale_factor = prepare_ppyoloe_r_input(
        image,
        spec,
    )
    feeds = _ppyoloe_r_feeds(
        ("image", "im_shape", "scale_factor"),
        tensor,
        im_shape,
        scale_factor,
    )

    assert tensor.shape == (1, 3, 384, 640)
    assert im_shape.tolist() == [[360.0, 640.0]]
    assert scale_factor.tolist() == [[1.0, 1.0]]
    assert set(feeds) == {"image", "im_shape", "scale_factor"}


def test_ppyoloe_r_runtime_rectifies_official_onnx_result(monkeypatch):
    corners = np.array(
        [[70, 65, 255, 45, 265, 105, 75, 125]],
        dtype=np.float32,
    )

    class Session:
        def run(self, output_names, feeds):
            assert output_names is None
            assert set(feeds) == {"image", "im_shape", "scale_factor"}
            return [
                corners[None],
                np.array([[[0.93]]], dtype=np.float32),
            ]

    entry = onnx_obb._SessionEntry(
        session=Session(),
        input_name="image",
        input_names=("image", "im_shape", "scale_factor"),
        run_lock=onnx_obb.threading.Lock(),
    )
    manifest = {
        "models": {
            "detector": {
                "path": "fixture.onnx",
                "runtime": "ppyoloe-r-onnx",
                "input_width": 640,
                "input_height": 640,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
                "pad_to_stride": 32,
                "score_threshold": 0.25,
                "nms_threshold": 0.1,
                "max_results": 8,
            },
        },
    }
    monkeypatch.setattr(
        onnx_obb,
        "_load_session",
        lambda *_args, **_kwargs: (entry, manifest),
    )
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        corners.reshape(4, 2).astype(np.int32),
        (220, 220, 220),
    )

    rows = onnx_obb.detect_plates_obb(
        frame,
        min_confidence=0.5,
    )

    assert len(rows) == 1
    assert rows[0]["method"] == "ppyoloe-r-onnx"
    assert rows[0]["bbox"] == (70, 45, 265, 125)
    assert rows[0]["crop"].shape[1] > rows[0]["crop"].shape[0]
