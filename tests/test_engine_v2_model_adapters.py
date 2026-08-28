from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from app.engine_v2.model_adapters import (
    CTCPlateOCR,
    CTCPlateOCRConfig,
    IRANIAN_PLATE_CHARSET,
    YOLOPlateDetector,
    YOLOPlateDetectorConfig,
)


class FakeBackend:
    """A deterministic stand-in for one shared OpenVINO/ONNX session."""

    def __init__(
        self,
        outputs: Sequence[Any],
        *,
        input_names: tuple[str, ...] = ("images",),
        output_names: tuple[str, ...] = ("predictions",),
    ) -> None:
        self.input_names = input_names
        self.output_names = output_names
        self.outputs = list(outputs)
        self.calls: list[tuple[dict[str, Any], tuple[str, ...] | None]] = []

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]:
        copied = {
            name: value.copy() if isinstance(value, np.ndarray) else value
            for name, value in input_feed.items()
        }
        requested = None if output_names is None else tuple(output_names)
        self.calls.append((copied, requested))
        if requested is not None:
            indices = [self.output_names.index(name) for name in requested]
            return [self.outputs[index] for index in indices]
        return list(self.outputs)


def _bgr_frame(height: int, width: int) -> np.ndarray:
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    return frame


def test_yolov8_channels_first_letterbox_rgb_and_maps_back() -> None:
    # Original (50, 20, 150, 80) becomes (50, 70, 150, 130) after the
    # 100px-high image is centered in a 200x200 detector tensor.
    rows = np.asarray(
        [
            [100.0, 100.0, 100.0, 60.0, 0.90],
            [20.0, 60.0, 10.0, 10.0, 0.10],
        ],
        dtype=np.float32,
    )
    backend = FakeBackend([rows.T[None, ...]])
    detector = YOLOPlateDetector(
        backend,
        YOLOPlateDetectorConfig(
            input_size=(200, 200),
            num_classes=1,
            output_name="predictions",
            confidence_threshold=0.25,
        ),
    )

    detections = detector.detect(_bgr_frame(100, 200))

    assert detector.backend is backend
    assert len(backend.calls) == 1
    feed, requested = backend.calls[0]
    tensor = feed["images"]
    assert requested == ("predictions",)
    assert tensor.shape == (1, 3, 200, 200)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    # Padding is the configured YOLO gray, while content is RGB rather than BGR.
    assert tensor[0, 0, 0, 0] == pytest.approx(114.0 / 255.0)
    assert tensor[0, :, 75, 50] == pytest.approx(np.asarray([30, 20, 10]) / 255.0)
    assert len(detections) == 1
    assert detections[0].bbox == (50, 20, 150, 80)
    assert detections[0].confidence == pytest.approx(0.90)
    assert detections[0].class_id == 0


def test_yolov5_rows_multiplies_objectness_filters_class_and_applies_nms() -> None:
    # [cx, cy, width, height, objectness, class0, class1]
    predictions = np.asarray(
        [
            [80.0, 50.0, 60.0, 20.0, 0.90, 0.80, 0.10],
            [82.0, 51.0, 60.0, 20.0, 0.95, 0.90, 0.05],
            [150.0, 50.0, 30.0, 20.0, 0.90, 0.10, 0.99],
            [20.0, 20.0, 10.0, 10.0, 0.40, 0.50, 0.10],
        ],
        dtype=np.float32,
    )[None, ...]
    backend = FakeBackend([predictions])
    detector = YOLOPlateDetector(
        backend,
        YOLOPlateDetectorConfig(
            input_size=(100, 200),
            num_classes=2,
            has_objectness=True,
            class_ids=(0,),
            confidence_threshold=0.50,
            iou_threshold=0.50,
        ),
    )

    detections = detector.detect(_bgr_frame(100, 200))

    assert len(detections) == 1
    assert detections[0].bbox == (52, 41, 112, 61)
    assert detections[0].confidence == pytest.approx(0.95 * 0.90)
    assert detections[0].class_id == 0


def test_yolo_end_to_end_xyxy_is_detected_when_heuristic_is_opted_in() -> None:
    predictions = np.asarray(
        [
            [10.0, 20.0, 70.0, 50.0, 0.90, 0.0],
            [10.0, 20.0, 70.0, 50.0, 0.85, 1.0],
            [12.0, 20.0, 68.0, 50.0, 0.80, 0.0],
        ],
        dtype=np.float32,
    )[None, ...]
    detector = YOLOPlateDetector(
        FakeBackend([predictions]),
        YOLOPlateDetectorConfig(
            input_size=(100, 100),
            iou_threshold=0.5,
            auto_format_policy="heuristic",
        ),
    )

    detections = detector.detect(_bgr_frame(100, 100))

    # Same-class overlap is suppressed, but an overlapping different class is
    # retained unless class-agnostic NMS is explicitly requested.
    assert [item.class_id for item in detections] == [0, 1]
    assert [item.confidence for item in detections] == pytest.approx([0.90, 0.85])
    assert [item.bbox for item in detections] == [(10, 20, 70, 50), (10, 20, 70, 50)]


def test_yolov5_single_class_six_channel_auto_prefers_raw_contract() -> None:
    # [cx, cy, width, height, objectness, class0]. Every row deliberately
    # satisfies the old end-to-end heuristic: xyxy-looking coordinates, a valid
    # score and an integer-like final value. Safe auto mode must still decode the
    # standard single-class YOLOv5 raw contract.
    predictions = np.asarray(
        [
            [10.0, 10.0, 20.0, 20.0, 0.90, 1.0],
            [30.0, 20.0, 40.0, 30.0, 0.80, 1.0],
        ],
        dtype=np.float32,
    )[None, ...]
    detector = YOLOPlateDetector(
        FakeBackend([predictions]),
        YOLOPlateDetectorConfig(
            input_size=(100, 100),
            confidence_threshold=0.25,
        ),
    )

    detections = detector.detect(_bgr_frame(100, 100))

    assert [item.class_id for item in detections] == [0, 0]
    assert [item.confidence for item in detections] == pytest.approx([0.90, 0.80])
    assert [item.bbox for item in detections] == [(0, 0, 20, 20), (10, 5, 50, 35)]


def test_yolo_auto_format_policy_is_validated_and_end_to_end_can_be_pinned() -> None:
    with pytest.raises(ValueError, match="auto_format_policy"):
        YOLOPlateDetectorConfig(auto_format_policy="guess")  # type: ignore[arg-type]

    predictions = np.asarray(
        [[10.0, 20.0, 70.0, 50.0, 0.90, 3.0]],
        dtype=np.float32,
    )[None, ...]
    detector = YOLOPlateDetector(
        FakeBackend([predictions]),
        YOLOPlateDetectorConfig(
            input_size=(100, 100),
            output_format="end_to_end",
        ),
    )

    detections = detector.detect(_bgr_frame(100, 100))
    assert len(detections) == 1
    assert detections[0].bbox == (10, 20, 70, 50)
    assert detections[0].class_id == 3


@pytest.mark.parametrize(
    ("shape", "layout"),
    [
        ((1, 5, 20), "auto"),
        ((1, 20, 5), "auto"),
        ((5, 20), "channels_first"),
        ((20, 5), "rows"),
    ],
)
def test_yolov11_common_output_shapes_are_supported(
    shape: tuple[int, ...],
    layout: str,
) -> None:
    rows = np.zeros((20, 5), dtype=np.float32)
    rows[0] = (50.0, 30.0, 40.0, 20.0, 0.75)
    if len(shape) == 3 and shape[1] == 5 or len(shape) == 2 and shape[0] == 5:
        output = rows.T.reshape(shape)
    else:
        output = rows.reshape(shape)
    detector = YOLOPlateDetector(
        FakeBackend([output]),
        YOLOPlateDetectorConfig(
            input_size=(60, 100),
            num_classes=1,
            output_layout=layout,  # type: ignore[arg-type]
        ),
    )

    detections = detector.detect(_bgr_frame(60, 100))

    assert len(detections) == 1
    assert detections[0].bbox == (30, 20, 70, 40)


def _logit_row(class_index: int, classes: int, strength: float = 8.0) -> np.ndarray:
    result = np.full(classes, -5.0, dtype=np.float32)
    result[class_index] = strength
    return result


def test_ctc_ocr_resizes_pads_normalizes_and_collapses_blank_and_repeats() -> None:
    charset = ("1", "2", "ب")
    # blank=0: 1, 1, blank, 2, 3, 3 -> "12ب".  The stronger second
    # alignment for the first token must become that token's confidence.
    logits = np.stack(
        [
            _logit_row(1, 4, 3.0),
            _logit_row(1, 4, 8.0),
            _logit_row(0, 4, 8.0),
            _logit_row(2, 4, 8.0),
            _logit_row(3, 4, 4.0),
            _logit_row(3, 4, 7.0),
        ]
    )[None, ...]
    backend = FakeBackend(
        [logits],
        input_names=("plate",),
        output_names=("logits",),
    )
    ocr = CTCPlateOCR(
        backend,
        CTCPlateOCRConfig(
            input_size=(8, 24),
            charset=charset,
            blank_index=0,
            input_name="plate",
            output_name="logits",
            preserve_aspect_ratio=True,
            mean=(0.0,),
            std=(1.0,),
            pad_value=0,
        ),
    )

    result = ocr.read(_bgr_frame(10, 20))

    assert ocr.backend is backend
    assert result.text == "12ب"
    assert result.valid is True
    assert result.confidence > 0.99
    assert len(result.character_confidences) == 3
    assert result.character_confidences[0] > 0.99
    assert result.metadata["tokens"] == ("1", "2", "ب")
    feed, requested = backend.calls[0]
    tensor = feed["plate"]
    assert requested == ("logits",)
    assert tensor.shape == (1, 1, 8, 24)
    expected_gray = (0.114 * 10 + 0.587 * 20 + 0.299 * 30) / 255.0
    assert tensor[0, 0, 4, 5] == pytest.approx(expected_gray)
    # 20x10 -> 16x8; left alignment leaves eight columns of padding.
    assert tensor[0, 0, 4, 20] == 0.0


def test_ctc_bct_output_supports_last_blank_and_probability_input() -> None:
    # Class order is A, B, blank.  Consecutive A is collapsed using max(.6,.9).
    time_class = np.asarray(
        [
            [0.60, 0.20, 0.20],
            [0.90, 0.05, 0.05],
            [0.05, 0.05, 0.90],
            [0.10, 0.80, 0.10],
        ],
        dtype=np.float32,
    )
    backend = FakeBackend([time_class.T[None, ...]])
    ocr = CTCPlateOCR(
        backend,
        CTCPlateOCRConfig(
            charset=("A", "B"),
            blank_index=2,
            output_layout="bct",
            activation="probabilities",
            mean=(0.0,),
            std=(1.0,),
        ),
    )

    result = ocr.read(_bgr_frame(16, 64))

    assert result.text == "AB"
    assert result.character_confidences == pytest.approx((0.90, 0.80))
    assert result.confidence == pytest.approx(0.85)


def test_ctc_auto_layout_handles_time_batch_class() -> None:
    logits = np.stack(
        [_logit_row(1, 3), _logit_row(0, 3), _logit_row(2, 3)],
        axis=0,
    )[:, None, :]
    ocr = CTCPlateOCR(
        FakeBackend([logits]),
        CTCPlateOCRConfig(charset=("7", "م"), blank_index=0),
    )

    assert ocr.read(_bgr_frame(12, 40)).text == "7م"


def test_ctc_empty_crop_skips_shared_backend_and_bad_class_count_is_rejected() -> None:
    backend = FakeBackend([np.zeros((1, 3, 3), dtype=np.float32)])
    ocr = CTCPlateOCR(
        backend,
        CTCPlateOCRConfig(charset=("1", "2", "3"), blank_index=0),
    )

    empty = ocr.read(np.empty((0, 0, 3), dtype=np.uint8))

    assert empty.valid is False
    assert empty.metadata["reason"] == "empty_crop"
    assert backend.calls == []
    with pytest.raises(ValueError, match=r"charset \+ blank requires 4"):
        ocr.read(_bgr_frame(10, 20))


def test_default_iranian_charset_matches_32_label_plate_crnn_contract() -> None:
    assert len(IRANIAN_PLATE_CHARSET) == 32
    assert set("0123456789").issubset(IRANIAN_PLATE_CHARSET)
    assert {"ا", "ب", "م", "ی", "پ", "ژ"}.issubset(IRANIAN_PLATE_CHARSET)
    assert CTCPlateOCRConfig().blank_index == 32
