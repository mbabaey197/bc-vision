import hashlib

import cv2
import numpy as np

from app.ai import benchmark
from app.ai.benchmark import (
    assess_training_candidate,
    compare_crnn_candidate_on_golden,
)
from app.ai.golden import (
    REQUIRED_GOLDEN_SLICES,
    validate_golden_manifest,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _passing_golden_decision(digest="A" * 64):
    return {
        "promote": True,
        "reasons": [],
        "baseline_exact_accuracy": 0.90,
        "candidate_exact_accuracy": 0.95,
        "baseline_false_accept_rate": 0.0,
        "candidate_false_accept_rate": 0.0,
        "baseline_mean_character_error": 0.10,
        "candidate_mean_character_error": 0.05,
        "evaluation_kind": "verified-ocr-crop-golden",
        "golden_manifest_sha256": digest,
        "samples": 40,
    }


def test_three_known_video_labels_are_not_a_large_enough_golden_set(
    tmp_path,
):
    samples = []
    for index, plate in enumerate((
        "31ط55674",
        "55ط63974",
        "84ب57133",
    )):
        media = tmp_path / f"known-{index}.jpg"
        media.write_bytes(f"known-{index}".encode())
        samples.append({
            "id": f"known-{index}",
            "frame_path": media.name,
            "sha256": _digest(media),
            "expected_plate": plate,
            "readable": True,
            "slices": ["day"],
            "label_source": "operator",
            "training_allowed": False,
        })

    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "operator",
            "training_allowed": False,
            "samples": samples,
        },
        tmp_path,
    )

    assert status["ready"] is False
    assert "insufficient-total-samples" in status["errors"]
    assert "insufficient-unique-plates" in status["errors"]
    assert "insufficient-slice:night" in status["errors"]


def test_golden_contract_requires_all_real_world_slices(tmp_path):
    samples = []
    readable_slices = [
        label
        for label in REQUIRED_GOLDEN_SLICES
        if label != "unreadable"
    ]
    for index in range(40):
        media = tmp_path / f"sample-{index}.jpg"
        media.write_bytes(f"golden-media-{index}".encode())
        unreadable = index < 3
        plate_index = index % 20
        samples.append({
            "id": f"sample-{index}",
            "frame_path": media.name,
            "sha256": _digest(media),
            "expected_plate": (
                ""
                if unreadable
                else (
                    f"{10 + plate_index:02d}"
                    f"ب{100 + plate_index:03d}"
                    f"{20 + plate_index:02d}"
                )
            ),
            "readable": not unreadable,
            "slices": (
                ["unreadable"]
                if unreadable
                else [
                    readable_slices[
                        (index - 3) % len(readable_slices)
                    ]
                ]
            ),
            "label_source": "operator",
            "training_allowed": False,
        })

    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "operator",
            "training_allowed": False,
            "samples": samples,
        },
        tmp_path,
    )

    assert status["ready"] is True
    assert status["samples"] == 40
    assert status["unique_plates"] >= 20
    assert all(
        count >= 3
        for count in status["slice_counts"].values()
    )


def test_golden_rejects_non_operator_labels(tmp_path):
    media = tmp_path / "ai-labelled.jpg"
    media.write_bytes(b"not-operator-labelled")
    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "ai",
            "training_allowed": False,
            "samples": [{
                "id": "ai-labelled",
                "frame_path": media.name,
                "sha256": _digest(media),
                "expected_plate": "31ط55674",
                "readable": True,
                "slices": ["day"],
                "label_source": "ai",
                "training_allowed": False,
            }],
        },
        tmp_path,
    )

    assert status["ready"] is False
    assert "label-source-must-be-operator" in status["errors"]
    assert (
        "label-source-must-be-operator:ai-labelled"
        in status["errors"]
    )


def test_readable_row_cannot_satisfy_unreadable_slice(tmp_path):
    media = tmp_path / "readable.jpg"
    media.write_bytes(b"readable")
    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "operator",
            "training_allowed": False,
            "samples": [{
                "id": "readable",
                "frame_path": media.name,
                "sha256": _digest(media),
                "expected_plate": "31ط55674",
                "readable": True,
                "slices": ["unreadable"],
                "label_source": "operator",
                "training_allowed": False,
            }],
        },
        tmp_path,
    )

    assert status["ready"] is False
    assert (
        "unreadable-slice-mismatch:readable"
        in status["errors"]
    )


def test_golden_requires_explicit_boolean_readability(tmp_path):
    media = tmp_path / "readable.jpg"
    media.write_bytes(b"readable")
    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "operator",
            "training_allowed": False,
            "samples": [{
                "id": "readable",
                "frame_path": media.name,
                "sha256": _digest(media),
                "expected_plate": "31ط55674",
                "readable": "true",
                "slices": ["day"],
                "label_source": "operator",
                "training_allowed": False,
            }],
        },
        tmp_path,
    )

    assert status["ready"] is False
    assert (
        "readable-must-be-boolean:readable"
        in status["errors"]
    )


def test_training_promotion_is_blocked_without_golden_comparison():
    result = {
        "validation_samples": 12,
        "baseline_accuracy": 0.80,
        "candidate_accuracy": 0.85,
        "baseline_mean_character_error": 0.40,
        "candidate_mean_character_error": 0.30,
        "validation_regressions": 0,
        "baseline_sha256": "A" * 64,
        "initialization_mode": "active-model-distillation",
        "training_rights_verified": True,
    }

    blocked = assess_training_candidate(
        result,
        {
            "ready": False,
            "errors": ["manifest-missing"],
            "samples": 0,
            "unique_plates": 0,
            "slice_counts": {},
        },
    )
    assert blocked["promote"] is False
    assert "golden-not-ready" in blocked["reasons"]
    assert "golden-comparison-missing" in blocked["reasons"]

    result["golden_decision"] = _passing_golden_decision()
    accepted = assess_training_candidate(
        result,
        {
            "ready": True,
            "manifest_sha256": "A" * 64,
            "errors": [],
            "samples": 40,
            "unique_plates": 20,
            "slice_counts": {
                label: 3
                for label in REQUIRED_GOLDEN_SLICES
            },
        },
    )
    assert accepted["promote"] is True


def test_verified_ocr_crop_golden_can_promote_candidate(
    tmp_path,
    monkeypatch,
):
    rows = []
    expected = "31ط55674"
    for index in range(40):
        image = tmp_path / f"crop-{index}.png"
        assert cv2.imwrite(
            str(image),
            np.full((32, 128, 3), 80, dtype=np.uint8),
        )
        rows.append({
            "media_path": str(image),
            "media_kind": "ocr-crop",
            "expected_plate": expected,
            "slices": ["day"],
            "sha256": _digest(image),
        })

    def predict(session, image):
        if session == "baseline.onnx":
            return "31ط55673", 1.0
        return expected, 1.0

    monkeypatch.setattr(
        benchmark,
        "_predict_crnn_session",
        predict,
    )

    decision = compare_crnn_candidate_on_golden(
        tmp_path / "baseline.onnx",
        tmp_path / "candidate.onnx",
        {
            "ready": True,
            "manifest_sha256": "A" * 64,
            "rows": rows,
        },
        session_factory=lambda path: path.name,
    )

    assert decision["promote"] is True
    assert decision["baseline_exact_accuracy"] == 0.0
    assert decision["candidate_exact_accuracy"] == 1.0
    assert decision["samples"] == 40


def test_frame_golden_does_not_enter_ocr_crop_promotion():
    decision = compare_crnn_candidate_on_golden(
        "baseline.onnx",
        "candidate.onnx",
        {
            "ready": True,
            "manifest_sha256": "A" * 64,
            "rows": [{
                "media_kind": "frame",
                "media_path": "frame.jpg",
            }],
        },
    )

    assert decision == {
        "promote": False,
        "reasons": ["ocr-crop-media-required"],
        "golden_manifest_sha256": "A" * 64,
    }


def test_golden_rejects_duplicate_media_content(tmp_path):
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    first.write_bytes(b"same-crop")
    second.write_bytes(first.read_bytes())
    digest = _digest(first)
    status = validate_golden_manifest(
        {
            "schema": 2,
            "label_source": "operator",
            "training_allowed": False,
            "samples": [
                {
                    "id": "first",
                    "crop_path": first.name,
                    "sha256": digest,
                    "expected_plate": "31ط55674",
                    "readable": True,
                    "slices": ["day"],
                    "label_source": "operator",
                    "training_allowed": False,
                },
                {
                    "id": "second",
                    "crop_path": second.name,
                    "sha256": digest,
                    "expected_plate": "55ط63974",
                    "readable": True,
                    "slices": ["night"],
                    "label_source": "operator",
                    "training_allowed": False,
                },
            ],
        },
        tmp_path,
    )

    assert status["ready"] is False
    assert "duplicate-media-digest:second" in status["errors"]
