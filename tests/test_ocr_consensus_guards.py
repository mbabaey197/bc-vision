"""Regression cases from the RC32 live-reader audit (no real-model claims)."""

from unittest.mock import patch

import numpy as np
import pytest

from app.ai import ocr, onnx_cnn, pipeline


def observation(plate, confidence=0.9):
    return {
        "bbox": (30, 40, 190, 72),
        "plate": plate,
        "plate_norm": plate,
        "valid": True,
        "confidence": confidence,
        "ocr_confidence": confidence,
        "quality_score": 0.9,
        "detector_confidence": 0.95,
        "needs_review": False,
        "ocr_engine": "cnn-onnx",
    }


@pytest.mark.parametrize("confidence", [0.0, 0.1, float("nan"), float("inf")])
def test_weak_cnn_cannot_become_confirmed_through_pipeline(confidence):
    crop = np.random.default_rng(7).integers(0, 256, (32, 160, 3), dtype=np.uint8)
    frame = np.zeros((200, 400, 3), dtype=np.uint8)
    with (
        patch.object(
            ocr,
            "read_plate_hezar_primary",
            return_value={"accepted": False, "hypotheses": []},
        ),
        patch.object(ocr, "read_plate_platrix", return_value=("", 0.0)),
        patch.object(ocr, "read_plate_cnn", return_value=("31-ط-556-74", confidence)),
        patch.object(
            pipeline,
            "detect_plates",
            return_value=[
                {
                    "crop": crop,
                    "bbox": (30, 40, 190, 72),
                    "confidence": 0.95,
                    "method": "test",
                }
            ],
        ),
    ):
        row = pipeline.process_frame(frame, detector_variant="yolov8n")[0]
    assert not row["valid"]
    tracker = pipeline.PlateConsensusTracker(emit_unreadable=True)
    events = []
    for i in range(8):
        events.extend(
            tracker.update([dict(row)], timestamp=1 + i * 0.2, min_emit_confidence=0.6)
        )
    events.extend(tracker.flush())
    assert not any(e.get("valid") or e.get("auto_confirmed") for e in events)


def test_cnn_rejects_uniform_and_one_ambiguous_position():
    labels = onnx_cnn.CNN_LABELS
    uniform = np.full((8, len(labels)), 1 / len(labels))
    assert onnx_cnn._decode(uniform) == ("", 0.0)
    decisive = np.zeros_like(uniform)
    for i, char in enumerate("31ط55674"):
        decisive[i, labels.index(char)] = 1
    assert onnx_cnn._decode(decisive)[0] == "31ط55674"
    decisive[4] = uniform[4]
    assert onnx_cnn._decode(decisive) == ("", 0.0)


def test_low_absolute_ocr_cannot_gain_confidence_from_repetition():
    tracker = pipeline.PlateConsensusTracker(emit_unreadable=True)
    events = []
    for i in range(8):
        events.extend(
            tracker.update([observation("31ط55674", 0.1)], timestamp=1 + i * 0.2)
        )
    events.extend(tracker.flush())
    assert not any(e.get("valid") or e.get("auto_confirmed") for e in events)


def test_continuous_one_digit_change_keeps_track_and_marks_conflict():
    tracker = pipeline.PlateConsensusTracker()
    events, ids, conflicts = [], [], []
    for i in range(11):
        row = observation("31ط55674" if i < 3 else "31ط55874", 0.9)
        events.extend(tracker.update([row], timestamp=1 + i * 0.2))
        ids.append(row["track_id"])
        if row.get("identity_conflict"):
            conflicts.append(row)
    assert len(set(ids)) == 1
    assert len(events) == 1
    assert conflicts and all(r["needs_review"] for r in conflicts)


def test_similar_vehicle_after_observed_gap_still_gets_new_track():
    tracker = pipeline.PlateConsensusTracker()
    for ts in (1, 1.2, 1.4):
        first = observation("31ط55674")
        tracker.update([first], timestamp=ts)
    tracker.update([], timestamp=1.6)
    tracker.update([], timestamp=1.8)
    second = observation("31ط55874")
    tracker.update([second], timestamp=2)
    assert second["track_id"] != first["track_id"]


def test_strong_five_frame_position_votes_recover_distributed_errors():
    tracker = pipeline.PlateConsensusTracker()
    events = []
    for i, plate in enumerate(
        ("31ط55675", "31ط55684", "31ط55774", "31ط56674", "31ط65674")
    ):
        events.extend(tracker.update([observation(plate)], timestamp=1 + i * 0.2))
    assert len(events) == 1
    assert events[0]["plate_norm"] == "31ط55674"
    assert events[0]["consensus_kind"] == "position-recovery"
    assert events[0]["confidence"] <= 0.9


def test_repeated_timestamp_is_not_independent_consensus_evidence():
    tracker = pipeline.PlateConsensusTracker()
    events = []
    for ts in (1, 1, 1, 1.2):
        events.extend(tracker.update([observation("31ط55674")], timestamp=ts))
    assert events == []


@pytest.mark.parametrize("confidence", [0.0, 0.1, 0.54, 0.74])
def test_weak_frames_cannot_recover_unseen_whole_plate(confidence):
    tracker = pipeline.PlateConsensusTracker(emit_unreadable=True)
    events = []
    for i, plate in enumerate(
        ("31ط55675", "31ط55684", "31ط55774", "31ط56674", "31ط65674")
    ):
        events.extend(
            tracker.update([observation(plate, confidence)], timestamp=1 + i * 0.2)
        )
    events.extend(tracker.flush())
    assert not any(e.get("valid") or e.get("auto_confirmed") for e in events)


def test_stale_and_duplicate_frames_cannot_change_track_state():
    tracker = pipeline.PlateConsensusTracker()
    row = observation("31ط55674")
    tracker.update([row], timestamp=2)
    track = tracker._tracks[row["track_id"]]
    for ts in (2, 1.9, float("nan"), float("inf")):
        assert tracker.update([], timestamp=ts) == []
        assert tracker.update([observation("94ب12345")], timestamp=ts) == []
    assert len(tracker._tracks) == 1
    assert len(track.observations) == 1
    assert track.last_seen == 2
    assert track.misses == 0


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -1.0, 1.1])
def test_cnn_rejects_invalid_probability_arrays(bad_value):
    probs = np.full((8, len(onnx_cnn.CNN_LABELS)), bad_value)
    assert onnx_cnn._decode(probs) == ("", 0.0)


def test_cnn_rejects_close_runner_and_wrong_slot_class():
    labels = onnx_cnn.CNN_LABELS
    probs = np.zeros((8, len(labels)))
    for i, char in enumerate("31ط55674"):
        probs[i, labels.index(char)] = 1
    probs[0, labels.index("3")] = 0.55
    probs[0, labels.index("ط")] = 0.45
    assert onnx_cnn._decode(probs) == ("", 0.0)
    probs[0, labels.index("3")] = 0.1
    probs[0, labels.index("ط")] = 0.9
    assert onnx_cnn._decode(probs) == ("", 0.0)
