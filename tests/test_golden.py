import hashlib

from app.ai.benchmark import assess_training_candidate
from app.ai.golden import (
    REQUIRED_GOLDEN_SLICES,
    validate_golden_manifest,
)


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        })

    status = validate_golden_manifest(
        {
            "schema": 2,
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
        })

    status = validate_golden_manifest(
        {
            "schema": 2,
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

    result["golden_decision"] = {
        "promote": True,
        "reasons": [],
    }
    accepted = assess_training_candidate(
        result,
        {
            "ready": True,
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
