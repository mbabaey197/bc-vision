"""Model-specific adapters for the shared Engine V2 inference backends.

The adapters in this module own image pre/post-processing only.  They receive
an already-created backend, so constructing a detector or OCR adapter never
creates a model session per camera.  A central worker can therefore share one
``SharedInferenceBackend`` instance across every camera.

Only NumPy is required here.  ONNX Runtime and OpenVINO remain optional and
are hidden behind the small :class:`InferenceBackend` protocol.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Protocol, Sequence

import numpy as np

from .types import OCRResult, PlateCandidate


class InferenceBackend(Protocol):
    """The subset of ``SharedInferenceBackend`` consumed by model adapters."""

    input_names: Sequence[str]
    output_names: Sequence[str]

    def infer(
        self,
        input_feed: Mapping[str, Any],
        output_names: Sequence[str] | None = None,
    ) -> list[Any]: ...


DetectorLayout = Literal["auto", "rows", "channels_first"]
DetectorOutputFormat = Literal["auto", "raw", "end_to_end"]
DetectorAutoFormatPolicy = Literal["prefer_raw", "heuristic"]
BoxFormat = Literal["auto", "xywh", "xyxy"]


@dataclass(frozen=True, slots=True)
class YOLOPlateDetectorConfig:
    """Pre/post-processing contract for a YOLOv5/v8/v11 plate model.

    ``input_size`` is ``(height, width)``.  ``output_layout="auto"`` handles
    the usual YOLOv5 row-major ``[1, predictions, channels]`` tensor and the
    YOLOv8/11 channels-first ``[1, channels, predictions]`` tensor.  Ambiguous
    custom exports can be pinned explicitly with ``output_layout``,
    ``output_format``, ``box_format`` and ``has_objectness``.

    Six-channel tensors are inherently ambiguous: raw single-class YOLOv5 uses
    ``[cx,cy,w,h,objectness,class_score]``, while end-to-end exports commonly
    use ``[x1,y1,x2,y2,score,class_id]``. Therefore ``output_format="auto"``
    defaults to the safe ``auto_format_policy="prefer_raw"``. End-to-end models
    should be pinned with ``output_format="end_to_end"``; the older numeric
    heuristic remains available as an explicit opt-in.
    """

    input_size: tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    class_ids: tuple[int, ...] | None = None
    num_classes: int | None = None
    max_detections: int = 100
    max_candidates: int = 3000
    input_name: str | None = None
    output_name: str | None = None
    output_index: int = 0
    output_layout: DetectorLayout = "auto"
    output_format: DetectorOutputFormat = "auto"
    auto_format_policy: DetectorAutoFormatPolicy = "prefer_raw"
    box_format: BoxFormat = "auto"
    has_objectness: bool | None = None
    class_agnostic_nms: bool = False
    coordinates_normalized: bool = False
    scale_up: bool = True
    pad_value: int = 114

    def __post_init__(self) -> None:
        height, width = self.input_size
        if height < 1 or width < 1:
            raise ValueError("YOLO input_size values must be positive")
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")
        if self.num_classes is not None and self.num_classes < 1:
            raise ValueError("num_classes must be positive when supplied")
        if self.max_detections < 1 or self.max_candidates < 1:
            raise ValueError("YOLO detection limits must be positive")
        if self.output_index < 0:
            raise ValueError("output_index must not be negative")
        if self.output_layout not in {"auto", "rows", "channels_first"}:
            raise ValueError(f"unsupported YOLO output_layout: {self.output_layout!r}")
        if self.output_format not in {"auto", "raw", "end_to_end"}:
            raise ValueError(f"unsupported YOLO output_format: {self.output_format!r}")
        if self.auto_format_policy not in {"prefer_raw", "heuristic"}:
            raise ValueError(
                f"unsupported YOLO auto_format_policy: {self.auto_format_policy!r}"
            )
        if self.box_format not in {"auto", "xywh", "xyxy"}:
            raise ValueError(f"unsupported YOLO box_format: {self.box_format!r}")
        if not 0 <= self.pad_value <= 255:
            raise ValueError("pad_value must be between 0 and 255")
        if self.class_ids is not None and any(value < 0 for value in self.class_ids):
            raise ValueError("class_ids must not contain negative values")


@dataclass(frozen=True, slots=True)
class _LetterboxTransform:
    original_height: int
    original_width: int
    input_height: int
    input_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.original_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.original_height


class YOLOPlateDetector:
    """A stateless, camera-agnostic plate detector over one shared backend."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: YOLOPlateDetectorConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or YOLOPlateDetectorConfig()
        self.input_name = _resolve_model_name(
            self.config.input_name,
            backend.input_names,
            "input",
        )
        self.output_name = _resolve_optional_output_name(
            self.config.output_name,
            backend.output_names,
        )

    def detect(self, frame: np.ndarray) -> list[PlateCandidate]:
        tensor, transform = self.preprocess(frame)
        outputs = self.backend.infer(
            {self.input_name: tensor},
            (self.output_name,) if self.output_name is not None else None,
        )
        output = _select_backend_output(
            outputs,
            name=self.output_name,
            index=self.config.output_index,
        )
        return self.postprocess(output, transform)

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, _LetterboxTransform]:
        """Letterbox a BGR frame and return an RGB NCHW float32 tensor."""

        bgr = _coerce_bgr_image(frame, name="detector frame")
        original_height, original_width = bgr.shape[:2]
        input_height, input_width = self.config.input_size
        scale = min(input_width / original_width, input_height / original_height)
        if not self.config.scale_up:
            scale = min(1.0, scale)
        resized_width = min(input_width, max(1, int(round(original_width * scale))))
        resized_height = min(input_height, max(1, int(round(original_height * scale))))
        left = (input_width - resized_width) // 2
        top = (input_height - resized_height) // 2

        # OpenCV frames are BGR.  Convert before resize and normalize to the
        # standard YOLO 0..1 input range.
        rgb = _to_unit_float(bgr[..., ::-1])
        resized = _resize_bilinear(rgb, resized_height, resized_width)
        canvas = np.full(
            (input_height, input_width, 3),
            float(self.config.pad_value) / 255.0,
            dtype=np.float32,
        )
        canvas[top : top + resized_height, left : left + resized_width] = resized
        tensor = np.ascontiguousarray(canvas.transpose(2, 0, 1)[None, ...], dtype=np.float32)
        transform = _LetterboxTransform(
            original_height=original_height,
            original_width=original_width,
            input_height=input_height,
            input_width=input_width,
            resized_height=resized_height,
            resized_width=resized_width,
            top=top,
            left=left,
        )
        return tensor, transform

    def postprocess(
        self,
        output: Any,
        transform: _LetterboxTransform,
    ) -> list[PlateCandidate]:
        rows, channels_first = _as_yolo_rows(
            output,
            layout=self.config.output_layout,
            num_classes=self.config.num_classes,
        )
        if rows.size == 0:
            return []

        boxes, scores, class_ids = self._decode_rows(rows, channels_first)
        if boxes.size == 0:
            return []

        finite = np.isfinite(boxes).all(axis=1) & np.isfinite(scores)
        valid = finite & (scores >= self.config.confidence_threshold)
        if self.config.class_ids is not None:
            allowed = np.asarray(self.config.class_ids, dtype=np.int64)
            valid &= np.isin(class_ids, allowed)
        boxes = boxes[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]
        if not len(scores):
            return []

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        positive = (widths > 0.0) & (heights > 0.0)
        boxes = boxes[positive]
        scores = scores[positive]
        class_ids = class_ids[positive]
        if not len(scores):
            return []

        order = np.argsort(-scores, kind="stable")[: self.config.max_candidates]
        boxes = boxes[order]
        scores = scores[order]
        class_ids = class_ids[order]
        kept = _non_maximum_suppression(
            boxes,
            scores,
            class_ids,
            iou_threshold=self.config.iou_threshold,
            class_agnostic=self.config.class_agnostic_nms,
            limit=self.config.max_detections,
        )

        detections: list[PlateCandidate] = []
        for index in kept:
            mapped = _map_box_to_original(boxes[index], transform)
            if mapped is None:
                continue
            detections.append(
                PlateCandidate(
                    bbox=mapped,
                    confidence=float(np.clip(scores[index], 0.0, 1.0)),
                    class_id=int(class_ids[index]),
                )
            )
        return detections

    def _decode_rows(
        self,
        rows: np.ndarray,
        channels_first: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if rows.shape[1] < 5:
            raise ValueError(
                f"YOLO output must expose at least 5 values per prediction; got {rows.shape}"
            )

        output_format = self.config.output_format
        if output_format == "auto":
            forced_raw = self.config.has_objectness is not None
            output_format = (
                "end_to_end"
                if (
                    not forced_raw
                    and self.config.auto_format_policy == "heuristic"
                    and _looks_like_end_to_end(rows)
                )
                else "raw"
            )

        if output_format == "end_to_end":
            if rows.shape[1] < 6:
                raise ValueError("end-to-end YOLO output requires [x1,y1,x2,y2,score,class]")
            boxes = rows[:, :4].astype(np.float32, copy=True)
            scores = rows[:, 4].astype(np.float32, copy=False)
            class_ids = np.rint(rows[:, 5]).astype(np.int64, copy=False)
            source_box_format: BoxFormat = "xyxy"
        else:
            has_objectness = self._resolve_objectness(rows.shape[1], channels_first)
            class_offset = 5 if has_objectness else 4
            class_count = self.config.num_classes
            class_end = rows.shape[1] if class_count is None else class_offset + class_count
            if class_end > rows.shape[1] or class_end <= class_offset:
                raise ValueError(
                    "YOLO class count does not match the prediction channel count: "
                    f"shape={rows.shape}, num_classes={class_count!r}, "
                    f"has_objectness={has_objectness}"
                )
            class_scores = rows[:, class_offset:class_end]
            class_ids = np.argmax(class_scores, axis=1).astype(np.int64, copy=False)
            scores = class_scores[np.arange(len(rows)), class_ids]
            if has_objectness:
                scores = scores * rows[:, 4]
            scores = scores.astype(np.float32, copy=False)
            boxes = rows[:, :4].astype(np.float32, copy=True)
            source_box_format = "xywh"

        box_format = self.config.box_format
        if box_format == "auto":
            box_format = source_box_format
        if box_format == "xywh":
            boxes = _xywh_to_xyxy(boxes)

        if self.config.coordinates_normalized:
            boxes[:, (0, 2)] *= float(self.config.input_size[1])
            boxes[:, (1, 3)] *= float(self.config.input_size[0])
        return boxes, scores, class_ids

    def _resolve_objectness(self, channel_count: int, channels_first: bool) -> bool:
        if self.config.has_objectness is not None:
            return self.config.has_objectness
        if self.config.num_classes is not None:
            without = 4 + self.config.num_classes
            with_objectness = 5 + self.config.num_classes
            if channel_count == without:
                return False
            if channel_count == with_objectness:
                return True
            raise ValueError(
                f"YOLO output has {channel_count} channels; expected {without} or "
                f"{with_objectness} for num_classes={self.config.num_classes}"
            )
        if channel_count == 5:
            return False  # common single-class YOLOv8/11 export
        # The normal layouts disambiguate the common defaults: v5 is rows-first
        # and v8/v11 is channels-first.  Pin has_objectness for custom exports.
        return not channels_first


# CTC class order is model-specific, so applications may replace this tuple in
# CTCPlateOCRConfig.  This 32-label default matches BC Vision's 33-class plate
# CRNN contract (32 labels plus a trailing CTC blank) without importing any
# legacy runtime module.  ASCII digits are accepted directly by Engine V2's
# IranianPlateValidator.
IRANIAN_PLATE_CHARSET: tuple[str, ...] = (
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "ا",
    "ب",
    "ت",
    "ث",
    "ج",
    "ح",
    "د",
    "ز",
    "س",
    "ش",
    "ص",
    "ط",
    "ع",
    "ق",
    "ل",
    "م",
    "ن",
    "ه",
    "و",
    "پ",
    "ژ",
    "ی",
)


OCRLayout = Literal["auto", "btc", "bct", "tbc", "tc", "ct"]
OCRColorMode = Literal["grayscale", "rgb"]
OCRTensorLayout = Literal["nchw", "nhwc"]
OCRPadAlignment = Literal["left", "center", "right"]
OCRActivation = Literal["softmax", "probabilities"]


@dataclass(frozen=True, slots=True)
class CTCPlateOCRConfig:
    """Input normalization and greedy CTC decoding configuration."""

    input_size: tuple[int, int] = (32, 128)
    charset: tuple[str, ...] = IRANIAN_PLATE_CHARSET
    blank_index: int = len(IRANIAN_PLATE_CHARSET)
    input_name: str | None = None
    output_name: str | None = None
    output_index: int = 0
    output_layout: OCRLayout = "auto"
    color_mode: OCRColorMode = "grayscale"
    tensor_layout: OCRTensorLayout = "nchw"
    # Defaults match the bundled 1x1x32x128 CRNN contract. Alternate models
    # can opt into aspect-preserving padding explicitly.
    preserve_aspect_ratio: bool = False
    pad_alignment: OCRPadAlignment = "left"
    pad_value: int = 0
    mean: tuple[float, ...] = (0.0,)
    std: tuple[float, ...] = (1.0,)
    activation: OCRActivation = "softmax"
    strict_class_count: bool = True
    min_confidence: float = 0.0

    def __post_init__(self) -> None:
        height, width = self.input_size
        if height < 1 or width < 1:
            raise ValueError("OCR input_size values must be positive")
        if not self.charset or any(not token for token in self.charset):
            raise ValueError("OCR charset must contain non-empty labels")
        if len(set(self.charset)) != len(self.charset):
            raise ValueError("OCR charset labels must be unique")
        if not 0 <= self.blank_index <= len(self.charset):
            raise ValueError("blank_index must address one of charset + blank classes")
        if self.output_index < 0:
            raise ValueError("output_index must not be negative")
        if self.output_layout not in {"auto", "btc", "bct", "tbc", "tc", "ct"}:
            raise ValueError(f"unsupported OCR output_layout: {self.output_layout!r}")
        if self.color_mode not in {"grayscale", "rgb"}:
            raise ValueError(f"unsupported OCR color_mode: {self.color_mode!r}")
        if self.tensor_layout not in {"nchw", "nhwc"}:
            raise ValueError(f"unsupported OCR tensor_layout: {self.tensor_layout!r}")
        if self.pad_alignment not in {"left", "center", "right"}:
            raise ValueError(f"unsupported OCR pad_alignment: {self.pad_alignment!r}")
        if self.activation not in {"softmax", "probabilities"}:
            raise ValueError(f"unsupported OCR activation: {self.activation!r}")
        if not 0 <= self.pad_value <= 255:
            raise ValueError("pad_value must be between 0 and 255")
        channels = 1 if self.color_mode == "grayscale" else 3
        if len(self.mean) not in {1, channels} or len(self.std) not in {1, channels}:
            raise ValueError("OCR mean/std must have one value or one value per channel")
        if any(value == 0 for value in self.std):
            raise ValueError("OCR std values must be non-zero")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")


class CTCPlateOCR:
    """Greedy CTC OCR adapter that reuses one central inference backend."""

    def __init__(
        self,
        backend: InferenceBackend,
        config: CTCPlateOCRConfig | None = None,
    ) -> None:
        self.backend = backend
        self.config = config or CTCPlateOCRConfig()
        self.input_name = _resolve_model_name(
            self.config.input_name,
            backend.input_names,
            "input",
        )
        self.output_name = _resolve_optional_output_name(
            self.config.output_name,
            backend.output_names,
        )

    def read(self, plate_crop: np.ndarray) -> OCRResult:
        crop = np.asarray(plate_crop)
        if crop.size == 0:
            return OCRResult(
                text="",
                confidence=0.0,
                valid=False,
                metadata={"reason": "empty_crop"},
            )
        tensor = self.preprocess(crop)
        outputs = self.backend.infer(
            {self.input_name: tensor},
            (self.output_name,) if self.output_name is not None else None,
        )
        output = _select_backend_output(
            outputs,
            name=self.output_name,
            index=self.config.output_index,
        )
        return self.decode(output)

    def preprocess(self, plate_crop: np.ndarray) -> np.ndarray:
        """Resize/pad a BGR crop and normalize it into the configured layout."""

        bgr = _coerce_bgr_image(plate_crop, name="plate crop")
        target_height, target_width = self.config.input_size
        source_height, source_width = bgr.shape[:2]

        if self.config.color_mode == "grayscale":
            unit_bgr = _to_unit_float(bgr)
            image = (
                0.114 * unit_bgr[..., 0]
                + 0.587 * unit_bgr[..., 1]
                + 0.299 * unit_bgr[..., 2]
            )[..., None]
        else:
            image = _to_unit_float(bgr[..., ::-1])

        if self.config.preserve_aspect_ratio:
            scale = min(target_width / source_width, target_height / source_height)
            resized_width = min(target_width, max(1, int(round(source_width * scale))))
            resized_height = min(target_height, max(1, int(round(source_height * scale))))
        else:
            resized_width = target_width
            resized_height = target_height
        resized = _resize_bilinear(image, resized_height, resized_width)

        if self.config.pad_alignment == "left":
            left = 0
        elif self.config.pad_alignment == "right":
            left = target_width - resized_width
        else:
            left = (target_width - resized_width) // 2
        top = (target_height - resized_height) // 2
        channels = resized.shape[2]
        canvas = np.full(
            (target_height, target_width, channels),
            float(self.config.pad_value) / 255.0,
            dtype=np.float32,
        )
        canvas[top : top + resized_height, left : left + resized_width] = resized

        mean = _channel_vector(self.config.mean, channels)
        std = _channel_vector(self.config.std, channels)
        normalized = (canvas - mean) / std
        if self.config.tensor_layout == "nchw":
            normalized = normalized.transpose(2, 0, 1)
        return np.ascontiguousarray(normalized[None, ...], dtype=np.float32)

    def decode(self, output: Any) -> OCRResult:
        logits = _as_time_class_logits(
            output,
            layout=self.config.output_layout,
            expected_classes=len(self.config.charset) + 1,
            strict_class_count=self.config.strict_class_count,
        )
        if logits.shape[0] == 0:
            return OCRResult(
                text="",
                confidence=0.0,
                valid=False,
                metadata={"reason": "empty_logits", "timesteps": 0},
            )
        if not 0 <= self.config.blank_index < logits.shape[1]:
            raise ValueError(
                f"blank_index {self.config.blank_index} is outside OCR output {logits.shape}"
            )

        probabilities = (
            _softmax(logits)
            if self.config.activation == "softmax"
            else _normalize_probabilities(logits)
        )
        best_indices = np.argmax(probabilities, axis=1)
        best_probabilities = probabilities[
            np.arange(probabilities.shape[0]),
            best_indices,
        ]

        tokens: list[str] = []
        confidences: list[float] = []
        previous = self.config.blank_index
        for class_index, probability in zip(best_indices, best_probabilities):
            index = int(class_index)
            confidence = float(probability)
            if index == self.config.blank_index:
                previous = self.config.blank_index
                continue
            if index == previous:
                # Confidence for a collapsed run is its strongest alignment,
                # not an arbitrary first time step.
                confidences[-1] = max(confidences[-1], confidence)
                continue
            charset_index = index if index < self.config.blank_index else index - 1
            if not 0 <= charset_index < len(self.config.charset):
                if self.config.strict_class_count:
                    raise ValueError(
                        f"OCR class index {index} has no charset label "
                        f"(blank={self.config.blank_index})"
                    )
                previous = index
                continue
            tokens.append(self.config.charset[charset_index])
            confidences.append(confidence)
            previous = index

        text = "".join(tokens)
        confidence = float(np.mean(confidences)) if confidences else 0.0
        valid = bool(text) and confidence >= self.config.min_confidence
        reason = "ok" if valid else ("empty_decode" if not text else "below_confidence")
        return OCRResult(
            text=text,
            confidence=confidence,
            valid=valid,
            character_confidences=tuple(confidences),
            metadata={
                "reason": reason,
                "timesteps": int(logits.shape[0]),
                "classes": int(logits.shape[1]),
                "blank_index": self.config.blank_index,
                "tokens": tuple(tokens),
            },
        )


def _resolve_model_name(explicit: str | None, names: Sequence[str], kind: str) -> str:
    available = tuple(str(name) for name in names)
    if explicit is not None:
        if available and explicit not in available:
            raise ValueError(f"model {kind} {explicit!r} is not in {available!r}")
        return explicit
    if not available:
        raise ValueError(f"inference backend exposes no model {kind} names")
    return available[0]


def _resolve_optional_output_name(
    explicit: str | None,
    names: Sequence[str],
) -> str | None:
    available = tuple(str(name) for name in names)
    if explicit is None:
        return None
    if available and explicit not in available:
        raise ValueError(f"model output {explicit!r} is not in {available!r}")
    return explicit


def _select_backend_output(outputs: Any, *, name: str | None, index: int) -> Any:
    if isinstance(outputs, Mapping):
        if name is not None:
            if name not in outputs:
                raise KeyError(f"inference result has no output named {name!r}")
            return outputs[name]
        values = list(outputs.values())
    elif isinstance(outputs, (list, tuple)):
        values = list(outputs)
    else:
        values = [outputs]
    if not values:
        raise ValueError("inference backend returned no outputs")
    # A named request returns one result regardless of its index in the full
    # model output list.
    selected_index = 0 if name is not None else index
    if selected_index >= len(values):
        raise IndexError(
            f"inference output_index {selected_index} exceeds {len(values)} returned outputs"
        )
    return values[selected_index]


def _coerce_bgr_image(image: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if array.ndim == 2:
        return np.repeat(array[..., None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] not in {1, 3, 4}:
        raise ValueError(f"{name} must have shape HxW, HxWx1, HxWx3, or HxWx4")
    if array.shape[2] == 1:
        return np.repeat(array, 3, axis=2)
    return array[..., :3]


def _to_unit_float(image: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(image), nan=0.0, posinf=255.0, neginf=0.0)
    result = array.astype(np.float32, copy=False)
    if np.issubdtype(array.dtype, np.integer) or float(np.max(result)) > 1.0:
        result = result / 255.0
    return np.clip(result, 0.0, 1.0)


def _resize_bilinear(image: np.ndarray, height: int, width: int) -> np.ndarray:
    """Small dependency-free bilinear resize using half-pixel coordinates."""

    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 2:
        source = source[..., None]
    source_height, source_width = source.shape[:2]
    if source_height == height and source_width == width:
        return source.copy()

    y = (np.arange(height, dtype=np.float32) + 0.5) * source_height / height - 0.5
    x = (np.arange(width, dtype=np.float32) + 0.5) * source_width / width - 0.5
    y0_unclipped = np.floor(y).astype(np.int64)
    x0_unclipped = np.floor(x).astype(np.int64)
    y1 = np.clip(y0_unclipped + 1, 0, source_height - 1)
    x1 = np.clip(x0_unclipped + 1, 0, source_width - 1)
    y0 = np.clip(y0_unclipped, 0, source_height - 1)
    x0 = np.clip(x0_unclipped, 0, source_width - 1)
    wy = np.clip(y - y0_unclipped, 0.0, 1.0)[:, None, None]
    wx = np.clip(x - x0_unclipped, 0.0, 1.0)[None, :, None]

    top = source[y0][:, x0] * (1.0 - wx) + source[y0][:, x1] * wx
    bottom = source[y1][:, x0] * (1.0 - wx) + source[y1][:, x1] * wx
    return np.asarray(top * (1.0 - wy) + bottom * wy, dtype=np.float32)


def _as_yolo_rows(
    output: Any,
    *,
    layout: DetectorLayout,
    num_classes: int | None,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(output)
    while array.ndim > 2:
        if array.shape[0] != 1:
            raise ValueError(f"YOLO adapter expects batch size 1; got output {array.shape}")
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"YOLO output must reduce to a 2D tensor; got {array.shape}")
    if 0 in array.shape:
        return np.empty((0, 6), dtype=np.float32), False

    channels_first = layout == "channels_first"
    if layout == "auto":
        first, second = array.shape
        expected_widths = {6}
        if num_classes is not None:
            expected_widths.update({4 + num_classes, 5 + num_classes})
        if first in expected_widths and second not in expected_widths:
            channels_first = True
        elif second in expected_widths and first not in expected_widths:
            channels_first = False
        else:
            # Real YOLO prediction counts are far larger than channel counts.
            # This conservative fallback avoids transposing small Nx6
            # end-to-end outputs.
            channels_first = first >= 5 and first < second and second > 16
    if channels_first:
        array = array.T
    return np.asarray(array, dtype=np.float32), channels_first


def _looks_like_end_to_end(rows: np.ndarray) -> bool:
    if rows.shape[1] != 6 or not len(rows):
        return False
    sample = rows[: min(len(rows), 64)]
    finite = np.isfinite(sample).all(axis=1)
    sample = sample[finite]
    if not len(sample):
        return False
    scores = sample[:, 4]
    classes = sample[:, 5]
    integer_classes = np.abs(classes - np.rint(classes)) <= 1e-4
    xyxy = (sample[:, 2] >= sample[:, 0]) & (sample[:, 3] >= sample[:, 1])
    probabilities = (scores >= 0.0) & (scores <= 1.0)
    return bool(np.mean(integer_classes & xyxy & probabilities) >= 0.8)


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def _non_maximum_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    *,
    iou_threshold: float,
    class_agnostic: bool,
    limit: int,
) -> list[int]:
    order = np.argsort(-scores, kind="stable")
    kept: list[int] = []
    while len(order) and len(kept) < limit:
        current = int(order[0])
        kept.append(current)
        if len(order) == 1:
            break
        remaining = order[1:]
        xx1 = np.maximum(boxes[current, 0], boxes[remaining, 0])
        yy1 = np.maximum(boxes[current, 1], boxes[remaining, 1])
        xx2 = np.minimum(boxes[current, 2], boxes[remaining, 2])
        yy2 = np.minimum(boxes[current, 3], boxes[remaining, 3])
        intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        current_area = max(
            0.0,
            float((boxes[current, 2] - boxes[current, 0]) * (boxes[current, 3] - boxes[current, 1])),
        )
        remaining_area = np.maximum(
            0.0,
            (boxes[remaining, 2] - boxes[remaining, 0])
            * (boxes[remaining, 3] - boxes[remaining, 1]),
        )
        union = current_area + remaining_area - intersection
        iou = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection),
            where=union > 0.0,
        )
        comparable = (
            np.ones(len(remaining), dtype=bool)
            if class_agnostic
            else class_ids[remaining] == class_ids[current]
        )
        order = remaining[~(comparable & (iou > iou_threshold))]
    return kept


def _map_box_to_original(
    box: np.ndarray,
    transform: _LetterboxTransform,
) -> tuple[int, int, int, int] | None:
    x1 = (float(box[0]) - transform.left) / transform.scale_x
    y1 = (float(box[1]) - transform.top) / transform.scale_y
    x2 = (float(box[2]) - transform.left) / transform.scale_x
    y2 = (float(box[3]) - transform.top) / transform.scale_y
    x1 = float(np.clip(x1, 0.0, transform.original_width))
    y1 = float(np.clip(y1, 0.0, transform.original_height))
    x2 = float(np.clip(x2, 0.0, transform.original_width))
    y2 = float(np.clip(y2, 0.0, transform.original_height))
    mapped = (
        int(math.floor(x1)),
        int(math.floor(y1)),
        int(math.ceil(x2)),
        int(math.ceil(y2)),
    )
    if mapped[2] <= mapped[0] or mapped[3] <= mapped[1]:
        return None
    return mapped


def _channel_vector(values: Sequence[float], channels: int) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if len(vector) == 1:
        vector = np.repeat(vector, channels)
    return vector.reshape(1, 1, channels)


def _as_time_class_logits(
    output: Any,
    *,
    layout: OCRLayout,
    expected_classes: int,
    strict_class_count: bool,
) -> np.ndarray:
    array = np.asarray(output, dtype=np.float32)
    if layout == "btc":
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"OCR btc output must have shape [1,time,classes]; got {array.shape}")
        result = array[0]
    elif layout == "bct":
        if array.ndim != 3 or array.shape[0] != 1:
            raise ValueError(f"OCR bct output must have shape [1,classes,time]; got {array.shape}")
        result = array[0].T
    elif layout == "tbc":
        if array.ndim != 3 or array.shape[1] != 1:
            raise ValueError(f"OCR tbc output must have shape [time,1,classes]; got {array.shape}")
        result = array[:, 0, :]
    elif layout == "tc":
        if array.ndim != 2:
            raise ValueError(f"OCR tc output must have shape [time,classes]; got {array.shape}")
        result = array
    elif layout == "ct":
        if array.ndim != 2:
            raise ValueError(f"OCR ct output must have shape [classes,time]; got {array.shape}")
        result = array.T
    else:
        result = _auto_time_class_logits(array, expected_classes)

    if result.ndim != 2:
        raise ValueError(f"OCR logits must reduce to [time,classes]; got {result.shape}")
    if strict_class_count and result.shape[1] != expected_classes:
        raise ValueError(
            f"OCR output exposes {result.shape[1]} classes, but charset + blank "
            f"requires {expected_classes}"
        )
    if result.shape[1] < 2:
        raise ValueError("OCR output must expose at least blank and one character class")
    return np.asarray(result, dtype=np.float32)


def _auto_time_class_logits(array: np.ndarray, expected_classes: int) -> np.ndarray:
    if array.ndim == 3:
        if array.shape[0] == 1:
            array = array[0]
        elif array.shape[1] == 1:
            array = array[:, 0, :]
        else:
            raise ValueError(f"OCR adapter expects batch size 1; got output {array.shape}")
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError(f"OCR output must reduce to a 2D tensor; got {array.shape}")
    if array.shape[1] == expected_classes:
        return array
    if array.shape[0] == expected_classes:
        return array.T
    # Preserve the conventional time-major orientation so the caller receives
    # a precise class-count error when strict validation is enabled.
    return array


def _softmax(logits: np.ndarray) -> np.ndarray:
    finite = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    shifted = finite - np.max(finite, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    totals = np.sum(exponentials, axis=1, keepdims=True)
    return np.divide(
        exponentials,
        totals,
        out=np.zeros_like(exponentials),
        where=totals > 0.0,
    )


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    totals = np.sum(clipped, axis=1, keepdims=True)
    return np.divide(
        clipped,
        totals,
        out=np.zeros_like(clipped),
        where=totals > 0.0,
    )


# Short aliases keep configuration call sites readable while the longer names
# remain explicit in documentation.
YOLODetectorConfig = YOLOPlateDetectorConfig
YOLOPlateDetectorAdapter = YOLOPlateDetector
CTCOCRConfig = CTCPlateOCRConfig
CTCOCRAdapter = CTCPlateOCR


__all__ = [
    "CTCOCRAdapter",
    "CTCOCRConfig",
    "CTCPlateOCR",
    "CTCPlateOCRConfig",
    "IRANIAN_PLATE_CHARSET",
    "InferenceBackend",
    "YOLODetectorConfig",
    "YOLOPlateDetector",
    "YOLOPlateDetectorAdapter",
    "YOLOPlateDetectorConfig",
]
