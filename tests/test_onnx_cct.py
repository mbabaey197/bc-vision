import numpy as np
from pathlib import Path

from app.ai import next_engine
from app.ai.onnx_cct import (
    CCT_DEFAULT_ALPHABET,
    _validate_session_contract,
    accept_cct_hypotheses,
    decode_cct_hypotheses,
    prepare_cct_input,
)
from tools.benchmark_cct_video import (
    _persist_emitted_rows,
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
