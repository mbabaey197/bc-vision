import hashlib
from pathlib import Path

import numpy as np
import pytest

from app.ai import next_engine
from app.ai import model_manager
from app.ai.onnx_cct import (
    CCT_FUSION_GEOMETRIC_MEAN,
    CCT_PREPROCESS_DUAL_VIEW,
    CCT_DEFAULT_ALPHABET,
    _validate_session_contract,
    accept_cct_hypotheses,
    decode_cct_hypotheses,
    fuse_cct_outputs,
    infer_cct_session,
    prepare_cct_input,
    prepare_cct_views,
)
from tools.benchmark_cct_video import (
    _detect_benchmark_frame,
    _persist_emitted_rows,
    _preflight_detector_models,
    _raw_guess_from_hypotheses,
    _update_nearest_raw,
)
from tools.render_cct_benchmark_report import render_report


def _position_probabilities(plate: str) -> np.ndarray:
    alphabet = CCT_DEFAULT_ALPHABET
    values = np.full(
        (1, 8, len(alphabet)),
        0.001 / max(1, len(alphabet) - 2),
        dtype=np.float32,
    )
    for position, character in enumerate(plate):
        target = alphabet.index(character)
        alternative_character = (
            "8" if position != 2 else "ب"
        )
        alternative = alphabet.index(alternative_character)
        values[0, position, target] = 0.94
        values[0, position, alternative] = 0.059
        values[0, position] /= values[0, position].sum()
    return values


def test_cct_decoder_accepts_exact_fixed_iranian_layout():
    hypotheses = decode_cct_hypotheses(
        _position_probabilities("31ط55674"),
        top_k=3,
    )
    result = accept_cct_hypotheses(hypotheses)

    assert result["accepted"] is True
    assert result["plate"] == "31-ط-556-74"
    assert result["plate_norm"] == "31ط55674"
    assert result["confidence"] > 0.90
    assert hypotheses[0]["score"] == hypotheses[0]["confidence"]
    assert hypotheses[0]["score"] > 0
    assert hypotheses[0]["log_score"] < 0


def test_cct_decoder_rejects_global_layout_conflict():
    values = _position_probabilities("31ط55674")
    values[0, 0] = 0.0
    values[0, 0, CCT_DEFAULT_ALPHABET.index("3")] = 0.35
    values[0, 0, CCT_DEFAULT_ALPHABET.index("ط")] = 0.65

    hypotheses = decode_cct_hypotheses(values)
    result = accept_cct_hypotheses(hypotheses)

    assert result["accepted"] is False
    assert "layout-conflict" in result["reason"]


def test_cct_preprocessing_matches_signed_nhwc_uint8_contract():
    image = np.zeros((24, 80, 3), dtype=np.uint8)
    image[:, :, 2] = 255
    tensor = prepare_cct_input(
        image,
        {
            "input_width": 128,
            "input_height": 64,
            "input_layout": "nhwc",
            "input_dtype": "uint8",
            "image_color_mode": "rgb",
            "keep_aspect_ratio": True,
            "padding_color": [114, 114, 114],
        },
    )

    assert tensor.shape == (1, 64, 128, 3)
    assert tensor.dtype == np.uint8
    assert int(tensor[0, 32, 64, 0]) == 255
    assert int(tensor[0, 32, 64, 2]) == 0


def _dual_view_spec():
    return {
        "alphabet": CCT_DEFAULT_ALPHABET,
        "max_plate_slots": 8,
        "input_width": 128,
        "input_height": 64,
        "input_layout": "nhwc",
        "input_dtype": "uint8",
        "image_color_mode": "rgb",
        "keep_aspect_ratio": False,
        "interpolation": "linear",
        "padding_color": [114, 114, 114],
        "preprocess_profile": CCT_PREPROCESS_DUAL_VIEW,
        "fusion_method": CCT_FUSION_GEOMETRIC_MEAN,
        "min_confidence": 0.58,
        "min_position_confidence": 0.50,
        "min_position_margin": 0.08,
        "min_hypothesis_margin": 0.03,
        "min_view_agreement": 0.75,
        "beam_width": 16,
        "top_k": 5,
    }


def test_cct_dual_views_preserve_fixed_contract_and_real_padding():
    image = np.full((20, 100, 3), 35, dtype=np.uint8)

    views = prepare_cct_views(image, _dual_view_spec())

    assert [view.name for view in views] == ["stretch", "letterbox"]
    assert all(view.tensor.shape == (1, 64, 128, 3) for view in views)
    assert all(view.tensor.dtype == np.uint8 for view in views)
    assert int(views[1].tensor[0, 0, 0, 0]) == 114
    assert not np.array_equal(views[0].tensor, views[1].tensor)


def test_cct_geometric_fusion_is_normalized_and_zero_safe():
    first = _position_probabilities("31ط55674")
    second = first.copy()
    second[0, 0] = 0.0
    second[0, 0, CCT_DEFAULT_ALPHABET.index("3")] = 0.80
    second[0, 0, CCT_DEFAULT_ALPHABET.index("8")] = 0.20

    fused = fuse_cct_outputs(
        [first, second],
        method=CCT_FUSION_GEOMETRIC_MEAN,
    )

    assert fused.shape == first.shape
    assert np.isfinite(fused).all()
    assert np.allclose(fused.sum(axis=2), 1.0)
    assert decode_cct_hypotheses(fused)[0]["plate_norm"] == "31ط55674"


class _InferenceSession:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def run(self, _outputs, feeds):
        tensor = feeds["input"]
        self.calls.append(tensor.copy())
        return [self.output.copy()]


class _SequentialInferenceSession:
    def __init__(self, outputs):
        self.outputs = [output.copy() for output in outputs]
        self.calls = []

    def run(self, _outputs, feeds):
        tensor = feeds["input"]
        self.calls.append(tensor.copy())
        return [self.outputs[len(self.calls) - 1].copy()]


def test_cct_dual_view_deduplicates_exact_two_to_one_input():
    session = _InferenceSession(_position_probabilities("31ط55674"))
    image = np.full((64, 128, 3), 90, dtype=np.uint8)

    result = infer_cct_session(
        session,
        "input",
        image,
        _dual_view_spec(),
    )

    assert len(session.calls) == 1
    assert result["inference_count"] == 1
    assert result["preprocess_profile"] == CCT_PREPROCESS_DUAL_VIEW
    assert result["whole_view_agreement"] is True
    assert result["plate_norm"] == "31ط55674"


def test_cct_legacy_profile_keeps_one_inference_call():
    session = _InferenceSession(_position_probabilities("31ط55674"))
    spec = _dual_view_spec()
    spec.pop("preprocess_profile")
    spec.pop("fusion_method")
    image = np.full((28, 110, 3), 90, dtype=np.uint8)

    result = infer_cct_session(session, "input", image, spec)

    assert len(session.calls) == 1
    assert result["preprocess_profile"] == "stretch-v1"
    assert result["fusion_method"] == "identity-v1"


def test_rejected_dual_view_read_is_review_only():
    session = _InferenceSession(_position_probabilities("31ط55674"))
    spec = _dual_view_spec()
    spec["min_confidence"] = 0.99
    image = np.full((28, 110, 3), 90, dtype=np.uint8)

    result = infer_cct_session(session, "input", image, spec)

    assert result["accepted"] is False
    assert result["raw_plate_norm"] == "31ط55674"
    assert result["temporal_consensus_eligible"] is False
    assert result["association_plate_norm"] == ""
    assert result["association_plate_strong"] is False


def test_dual_view_rejects_large_top_string_disagreement():
    preferred = "31ط55674"
    competing = "84ب92336"
    stretch = _position_probabilities(preferred)
    letterbox = np.full_like(stretch, 1e-6)
    for position, (preferred_char, competing_char) in enumerate(
        zip(preferred, competing, strict=True)
    ):
        preferred_index = CCT_DEFAULT_ALPHABET.index(preferred_char)
        competing_index = CCT_DEFAULT_ALPHABET.index(competing_char)
        if preferred_char == competing_char:
            letterbox[0, position, preferred_index] = 0.99
        else:
            letterbox[0, position, preferred_index] = 0.40
            letterbox[0, position, competing_index] = 0.60
        letterbox[0, position] /= letterbox[0, position].sum()
    session = _SequentialInferenceSession([stretch, letterbox])
    image = np.full((28, 110, 3), 90, dtype=np.uint8)

    result = infer_cct_session(
        session,
        "input",
        image,
        _dual_view_spec(),
    )

    assert len(session.calls) == 2
    assert result["raw_plate_norm"] == preferred
    assert result["accepted"] is False
    assert result["view_agreement"] < 0.75
    assert "view-disagreement" in result["reason"]
    assert result["temporal_consensus_eligible"] is False


def test_cct_unknown_preprocess_profile_fails_closed():
    spec = _dual_view_spec()
    spec["preprocess_profile"] = "pick-best-after-seeing-truth"

    with pytest.raises(ValueError, match="Unsupported CCT preprocess"):
        prepare_cct_views(np.zeros((24, 96, 3), dtype=np.uint8), spec)


def test_next_engine_routes_signed_cct_runtime(monkeypatch):
    expected = {
        "accepted": False,
        "plate": "ناخوانا",
        "plate_norm": "",
        "confidence": 0.0,
        "hypotheses": [],
    }
    monkeypatch.setattr(
        next_engine,
        "verified_next_manifest",
        lambda: {
            "models": {
                "ocr": {"runtime": "fast-plate-ocr-cct"}
            }
        },
    )
    monkeypatch.setattr(
        next_engine,
        "read_plate_cct",
        lambda *_args, **_kwargs: expected,
    )
    monkeypatch.setattr(
        next_engine,
        "cct_status",
        lambda: {"attempted": True, "model_loaded": True},
    )

    result, status, runtime = next_engine._read_candidate_ocr(
        np.zeros((32, 128, 3), dtype=np.uint8),
        engine_key="camera-1",
    )

    assert result is expected
    assert status["model_loaded"] is True
    assert runtime == "fast-plate-ocr-cct"


class _Meta:
    def __init__(self, shape, kind):
        self.shape = shape
        self.type = kind


class _Session:
    def __init__(self, input_shape, output_shape):
        self._inputs = [_Meta(input_shape, "tensor(uint8)")]
        self._outputs = [_Meta(output_shape, "tensor(float)")]

    def get_inputs(self):
        return self._inputs

    def get_outputs(self):
        return self._outputs


def test_cct_onnx_shape_must_match_signed_contract():
    spec = {
        "alphabet": CCT_DEFAULT_ALPHABET,
        "max_plate_slots": 8,
        "input_width": 128,
        "input_height": 64,
    }
    _validate_session_contract(
        _Session([1, 64, 128, 3], [1, 8, 37]),
        spec,
    )

    with np.testing.assert_raises_regex(
        ValueError,
        "does not match signed metadata",
    ):
        _validate_session_contract(
            _Session(["batch", 64, 128, 3], [1, 8, 37]),
            spec,
        )


def test_cct_benchmark_row_keeps_crop_next_to_recognized_text(tmp_path):
    rows = _persist_emitted_rows(
        [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "confidence": 0.91,
            "last_seen": 2.5,
            "crop": np.full((24, 96, 3), 180, dtype=np.uint8),
            "quality": {"score": 0.8},
        }],
        artifact_dir=tmp_path,
        start_index=0,
        frame_number=20,
        fps=8.0,
    )

    assert rows[0]["plate"] == "31-ط-556-74"
    assert rows[0]["frame"] == 20
    assert rows[0]["video_second"] == 2.5
    assert rows[0]["crop_path"] == str(
        Path("crops") / "plate-0001.jpg"
    )
    assert (tmp_path / rows[0]["crop_path"]).is_file()
    assert "crop" not in rows[0]


def test_cct_benchmark_keeps_rejected_raw_guess_separate_from_truth():
    crop = np.full((24, 96, 3), 180, dtype=np.uint8)
    observation = _raw_guess_from_hypotheses(
        [{
            "plate": "31-ط-556-75",
            "plate_norm": "31ط55675",
            "confidence": 0.73,
        }],
        accepted=False,
        reason="position-margin",
        frame_number=20,
        fps=8.0,
        detection={
            "confidence": 0.81,
            "bbox": (1, 2, 30, 12),
            "method": "yolov8-onnx-light",
            "crop": crop,
        },
    )
    nearest = {}
    _update_nearest_raw(nearest, observation, {"31ط55674"})

    assert observation["accepted"] is False
    assert observation["plate_norm"] == "31ط55675"
    assert nearest["31ط55674"]["character_distance"] == 1
    assert nearest["31ط55674"]["crop"] is not crop


def test_cct_benchmark_detector_preflight_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(
        model_manager,
        "detector_path",
        lambda: tmp_path / "missing-primary.onnx",
    )
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: tmp_path / "missing-fallback.onnx",
    )

    with pytest.raises(
        RuntimeError,
        match="Verified ONNX detector preflight failed",
    ):
        _preflight_detector_models()


def test_cct_benchmark_detector_preflight_checks_size_and_sha(
    tmp_path,
    monkeypatch,
):
    primary_payload = b"primary"
    fallback_payload = b"fallback"
    primary = tmp_path / "plate_yolo.onnx"
    fallback = tmp_path / "plate_yolo_fallback.onnx"
    primary.write_bytes(b"x" * len(primary_payload))
    fallback.write_bytes(fallback_payload + b"-wrong-size")
    monkeypatch.setattr(model_manager, "detector_path", lambda: primary)
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: fallback,
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SHA256",
        hashlib.sha256(primary_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SIZE",
        len(primary_payload),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SHA256",
        hashlib.sha256(fallback_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SIZE",
        len(fallback_payload),
    )

    status = _preflight_detector_models(
        allow_opencv_detector=True,
    )

    assert status["ready"] is False
    assert status["primary"]["size_matches"] is True
    assert status["primary"]["sha256_matches"] is False
    assert status["fallback"]["size_matches"] is False
    assert status["fallback"]["sha256_matches"] is False


def test_cct_benchmark_prepares_detectors_only_when_explicit(
    tmp_path,
    monkeypatch,
):
    primary_payload = b"primary"
    fallback_payload = b"fallback"
    primary = tmp_path / "plate_yolo.onnx"
    fallback = tmp_path / "plate_yolo_fallback.onnx"
    calls = []
    monkeypatch.setattr(model_manager, "detector_path", lambda: primary)
    monkeypatch.setattr(
        model_manager,
        "detector_fallback_path",
        lambda: fallback,
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SHA256",
        hashlib.sha256(primary_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_SIZE",
        len(primary_payload),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SHA256",
        hashlib.sha256(fallback_payload).hexdigest(),
    )
    monkeypatch.setattr(
        model_manager,
        "DETECTOR_FALLBACK_SIZE",
        len(fallback_payload),
    )

    def prepare_primary(download=True):
        calls.append(("primary", download))
        primary.write_bytes(primary_payload)
        return primary

    def prepare_fallback(download=True):
        calls.append(("fallback", download))
        fallback.write_bytes(fallback_payload)
        return fallback

    monkeypatch.setattr(
        model_manager,
        "ensure_detector_model",
        prepare_primary,
    )
    monkeypatch.setattr(
        model_manager,
        "ensure_detector_fallback_model",
        prepare_fallback,
    )

    unprepared = _preflight_detector_models(
        allow_opencv_detector=True,
    )
    assert unprepared["ready"] is False
    assert calls == []

    prepared = _preflight_detector_models(
        prepare_detector=True,
    )
    assert prepared["ready"] is True
    assert calls == [("primary", True), ("fallback", True)]


def test_cct_benchmark_strict_runtime_requires_loaded_onnx(monkeypatch):
    monkeypatch.setattr(
        "tools.benchmark_cct_video.detect_plates_onnx",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "tools.benchmark_cct_video.onnx_detector_status",
        lambda: {"model_loaded": False},
    )

    with pytest.raises(RuntimeError, match="onnx_model_loaded is false"):
        _detect_benchmark_frame(
            np.zeros((24, 96, 3), dtype=np.uint8),
            allow_opencv_detector=False,
        )


def test_cct_benchmark_opencv_detector_requires_explicit_mode(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "tools.benchmark_cct_video.detect_plates",
        lambda *_args, **_kwargs: calls.append("opencv") or [],
    )
    monkeypatch.setattr(
        "tools.benchmark_cct_video.detect_plates_onnx",
        lambda *_args, **_kwargs: calls.append("onnx") or [],
    )

    _detect_benchmark_frame(
        np.zeros((24, 96, 3), dtype=np.uint8),
        allow_opencv_detector=True,
    )

    assert calls == ["opencv"]


def test_cct_benchmark_strict_runtime_checks_every_inference(monkeypatch):
    statuses = iter([
        {"model_loaded": True},
        {"model_loaded": False},
    ])
    monkeypatch.setattr(
        "tools.benchmark_cct_video.detect_plates_onnx",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "tools.benchmark_cct_video.onnx_detector_status",
        lambda: next(statuses),
    )
    frame = np.zeros((24, 96, 3), dtype=np.uint8)

    assert _detect_benchmark_frame(
        frame,
        allow_opencv_detector=False,
    ) == []
    with pytest.raises(RuntimeError, match="onnx_model_loaded is false"):
        _detect_benchmark_frame(
            frame,
            allow_opencv_detector=False,
        )


def test_cct_html_report_embeds_crop_beside_the_same_row_text(tmp_path):
    crop_dir = tmp_path / "crops"
    crop_dir.mkdir()
    crop = crop_dir / "plate-0001.jpg"
    crop.write_bytes(b"jpeg-bytes")
    result = {
        "artifact_dir": str(tmp_path),
        "video": "01.mp4",
        "video_sha256": "ABC",
        "truth_count": 1,
        "matched_truth_count": 1,
        "matched_truth": ["31-ط-556-74"],
        "missed_truth": [],
        "processed_frames": 12,
        "detections": 3,
        "emitted_count": 1,
        "elapsed_seconds": 1.25,
        "raw_exact_truth_count": 0,
        "nearest_raw_by_truth": [{
            "truth": "31-ط-556-74",
            "plate": "31-ط-556-75",
            "plate_norm": "31ط55675",
            "crop_path": "crops/plate-0001.jpg",
            "character_distance": 1,
            "confidence": 0.73,
            "video_second": 2.5,
            "exact_observations": 0,
            "accepted": False,
        }],
        "emitted": [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "crop_path": "crops/plate-0001.jpg",
            "confidence": 0.91,
            "ocr_confidence": 0.88,
            "video_second": 2.5,
            "track_id": 7,
        }],
    }

    report = render_report([("CCT-S-v2", result)])

    assert "31-ط-556-74" in report
    assert "31-ط-556-75" in report
    assert "حدس آزمایشی" in report
    assert "مطابق Golden" in report
    assert "data:image/jpeg;base64," in report
    emitted_section = report.index(
        "خروجی‌های پذیرفته‌شدهٔ Tracker"
    )
    assert report.index(
        "data:image/jpeg;base64,",
        emitted_section,
    ) < report.index(
        "31-ط-556-74",
        emitted_section,
    )
