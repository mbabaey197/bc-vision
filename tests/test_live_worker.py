import os
import subprocess
import sys
import time
import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

import app.ai.live_worker as live_worker
import app.media_acceptance as media_acceptance
import app.media_storage as media_storage


def _allow_unit_media_writes(monkeypatch):
    counter = iter(range(1, 10_000))

    class Reservation:
        def close(self, **_kwargs):
            return None

    monkeypatch.setattr(
        media_storage,
        "begin_media_write",
        lambda *_args, **_kwargs: Reservation(),
    )
    monkeypatch.setattr(
        media_acceptance,
        "create_intent",
        lambda _target: f"{next(counter):032x}",
    )
    monkeypatch.setattr(
        media_acceptance,
        "accept_intent",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        media_acceptance,
        "discard_intent",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        media_acceptance,
        "load_intent",
        lambda *_args, **_kwargs: {"state": "pending"},
    )


def persistence_entry(
    worker,
    *,
    camera_id=1,
    track_id=1,
    plate="31ط55674",
    event_id=None,
    generation=0,
    detector_revision="",
    valid=True,
):
    frame = np.full((80, 180, 3), 120, dtype=np.uint8)
    result = {
        "plate": plate,
        "plate_norm": plate if valid else "",
        "raw_guess_norm": plate,
        "valid": valid,
        "needs_review": not valid,
        "confidence": 0.90,
        "track_id": track_id,
        "bbox": (20, 20, 160, 60),
    }
    return worker._make_persistence_retry(
        camera_id,
        f"camera-{camera_id}",
        result,
        frame,
        event_id,
        plate,
        10.0,
        12.0,
        30.0,
        generation,
        detector_revision,
    )


def test_below_camera_gate_is_kept_as_review_candidate():
    confirmed = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.71,
        "ocr_confidence": 0.64,
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
        "needs_review": False,
        "read_status": "confirmed-ai",
    }

    review = live_worker.camera_confidence_result(confirmed, 0.80)

    assert confirmed["valid"] is True
    assert review["valid"] is False
    assert review["plate_norm"] == ""
    assert review["raw_guess_norm"] == "31ط55674"
    assert review["needs_review"] is True
    assert review["read_status"] == "experimental-guess"
    assert review["raw_guess_reason"] == "below-camera-confidence"
    assert review["confirmation_source"] == "ai-suggestion"
    assert review["auto_confirmation_blocked"] == (
        "below-camera-confidence"
    )


def test_below_camera_gate_downgrades_auto_confirmed_review_result():
    assisted = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": False,
        "auto_confirmed": True,
        "needs_review": True,
        "confidence": 0.71,
        "ocr_confidence": 0.64,
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
        "read_status": "auto-confirmed",
    }

    review = live_worker.camera_confidence_result(assisted, 0.80)

    assert assisted["auto_confirmed"] is True
    assert review["auto_confirmed"] is False
    assert review["valid"] is False
    assert review["plate_norm"] == ""
    assert review["raw_guess_norm"] == "31ط55674"
    assert review["read_status"] == "experimental-guess"
    assert review["confirmation_source"] == "ai-suggestion"
    assert review["auto_confirmation_blocked"] == (
        "below-camera-confidence"
    )


@pytest.mark.parametrize(
    ("camera_gate", "confidence", "ocr_confidence", "review_only"),
    [
        (0.50, 0.92, 0.90, False),
        (0.85, 0.55, 0.45, True),
    ],
)
def test_failed_first_persist_retries_same_tracker_emission(
    monkeypatch,
    camera_gate,
    confidence,
    ocr_confidence,
    review_only,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": int(camera_gate * 100),
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "confidence": confidence,
        "detector_confidence": confidence,
        "ocr_confidence": ocr_confidence,
        "quality_score": 0.82,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(detected)],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    attempts = []
    persisted = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        attempts.append(dict(result))
        if len(attempts) == 1:
            raise OSError("temporary disk failure")
        persisted.append((dict(result), event_id))
        return event_id or 77

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert len(attempts) == 2
    assert len(persisted) == 1
    assert persisted[0][0]["needs_review"] is review_only
    assert state.emitted_events == 1
    assert state.visits.event_refs == {"31ط55674": 77}


def test_persist_failure_does_not_drop_later_result_in_same_batch(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((160, 420, 3), 120, dtype=np.uint8)
    first = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "ocr_confidence": 0.95,
        "detector_confidence": 0.97,
        "quality_score": 0.90,
        "bbox": (20, 30, 170, 70),
        "crop": frame[30:70, 20:170].copy(),
        "method": "test",
    }
    second = {
        **first,
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "bbox": (230, 80, 390, 120),
        "crop": frame[80:120, 230:390].copy(),
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(first), dict(second)],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    attempts = []
    next_ids = {
        "31ط55674": 71,
        "12ب34567": 72,
    }

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        plate = (
            result.get("plate_norm")
            or result.get("raw_guess_norm")
        )
        attempts.append((plate, event_id))
        if len(attempts) == 1:
            raise OSError("temporary disk failure")
        return event_id or next_ids[plate]

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.1, 0.2, 0.3):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert attempts == [
        ("31ط55674", None),
        ("12ب34567", None),
        ("31ط55674", None),
    ]
    assert not state.persistence_retry
    assert state.emitted_events == 2
    assert state.visits.event_refs == {
        "31ط55674": 71,
        "12ب34567": 72,
    }


def test_expired_track_persist_failure_retries_without_live_track(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    unreadable = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "valid": False,
        "confidence": 0.30,
        "detector_confidence": 0.82,
        "quality_score": 0.70,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    outputs = iter(([dict(unreadable)], [], []))
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )
    attempts = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        attempts.append((dict(result), event_id))
        if len(attempts) == 1:
            raise OSError("temporary disk failure")
        return event_id or 81

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 6.0, 6.1):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert len(attempts) == 2
    assert attempts[0][0]["track_id"] == attempts[1][0]["track_id"]
    assert state.tracker.active_track_ids() == set()
    assert not state.persistence_retry
    assert state.emitted_events == 1


def test_low_review_identity_migrates_to_clear_plate_without_duplicate(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 90,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-558-74",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "best_effort": True,
        "visit_identity_stable": False,
        "confidence": 0.48,
        "ocr_confidence": 0.48,
        "detector_confidence": 0.90,
        "quality_score": 0.80,
        "bbox": (20, 30, 170, 70),
        "crop": frame[30:70, 20:170].copy(),
        "method": "test",
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
        "hypotheses_accepted_for_consensus": True,
        "plate_hypotheses": [{
            "plate_norm": "31ط55874",
            "confidence": 0.20,
            "score": 0.20,
            "engine": "hezar-crnn-fa-v2-onnx",
            "temporal_evidence": False,
        }],
    }
    clear = {
        **review,
        "plate": "31-ط-526-74",
        "plate_norm": "31ط52674",
        "raw_guess_text": "31-ط-526-74",
        "raw_guess_norm": "31ط52674",
        "valid": True,
        "needs_review": False,
        "best_effort": False,
        "visit_identity_stable": True,
        "confidence": 0.98,
        "ocr_confidence": 0.96,
        "detector_confidence": 0.98,
        "quality_score": 0.95,
        "plate_hypotheses": [],
        "hypotheses_accepted_for_consensus": True,
    }
    outputs = iter(
        [[dict(review)] for _ in range(5)]
        + [[dict(clear)] for _ in range(3)]
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )
    writes = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        writes.append((dict(result), event_id))
        return event_id or 77

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    inserts = [row for row, event_id in writes if event_id is None]
    assert len(inserts) == 1
    assert inserts[0]["raw_guess_norm"] == "31ط55874"
    assert writes[-1][0]["valid"] is True
    assert writes[-1][0]["plate_norm"] == "31ط52674"
    assert writes[-1][1] == 77
    assert state.emitted_events == 1
    assert state.visits.event_refs == {"31ط52674": 77}


def test_fragmented_track_review_migrates_to_near_clear_identity(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 90,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    review = {
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-558-74",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "confidence": 0.48,
        "bbox": (20, 30, 170, 70),
        "crop": frame[30:70, 20:170].copy(),
    }
    clear = {
        **review,
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "needs_review": False,
        "confidence": 0.98,
        "ocr_confidence": 0.96,
        "bbox": (24, 30, 174, 70),
    }
    outputs = iter(([dict(review)], [dict(clear)]))
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )

    class FragmentedTracker:
        max_age_seconds = 2.4

        def __init__(self):
            self.calls = 0
            self.active = set()

        def update(self, rows, **_kwargs):
            self.calls += 1
            track_id = self.calls
            self.active = {track_id}
            row = rows[0]
            row["track_id"] = track_id
            row["tracking_bbox"] = row["bbox"]
            emitted = dict(row)
            emitted["capture_frame"] = frame.copy()
            return [emitted]

        def active_track_ids(self):
            return set(self.active)

        def retire_tracks(self, track_ids):
            self.active.difference_update(track_ids)

    state.tracker = FragmentedTracker()
    writes = []

    def persist(*args):
        event_id = args[5]
        writes.append((dict(args[3]), event_id))
        return event_id or 77

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.2):
        state.busy = True
        worker._process(state, (1, "Gate", frame.copy(), timestamp))
    worker.shutdown()

    assert len(writes) == 2
    assert writes[0][1] is None
    assert writes[1][1] == 77
    assert writes[1][0]["plate_norm"] == "31ط55674"
    assert state.emitted_events == 1
    assert state.visits.event_refs == {"31ط55674": 77}


def test_unstable_fragment_cannot_downgrade_confirmed_event(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 220, 3), 130, dtype=np.uint8)
    confirmed = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "visit_identity_stable": True,
        "confidence": 0.97,
        "ocr_confidence": 0.95,
        "bbox": (20, 30, 170, 70),
        "crop": frame[30:70, 20:170].copy(),
    }
    flicker = {
        **confirmed,
        "plate": "31-ط-558-74",
        "plate_norm": "",
        "raw_guess_norm": "31ط55874",
        "valid": False,
        "needs_review": True,
        "visit_identity_stable": False,
        "confidence": 0.43,
        "ocr_confidence": 0.41,
        "bbox": (24, 30, 174, 70),
    }
    outputs = iter(([dict(confirmed)], [dict(flicker)]))
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )

    class FragmentedTracker:
        max_age_seconds = 2.4

        def __init__(self):
            self.calls = 0
            self.active = set()

        def update(self, rows, **_kwargs):
            self.calls += 1
            track_id = self.calls
            self.active = {track_id}
            row = rows[0]
            row["track_id"] = track_id
            row["tracking_bbox"] = row["bbox"]
            emitted = dict(row)
            emitted["capture_frame"] = frame.copy()
            return [emitted]

        def active_track_ids(self):
            return set(self.active)

        def retire_tracks(self, track_ids):
            self.active.difference_update(track_ids)

    state.tracker = FragmentedTracker()
    writes = []

    def persist(*args):
        writes.append((dict(args[3]), args[5]))
        return args[5] or 77

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.2):
        state.busy = True
        worker._process(state, (1, "Gate", frame.copy(), timestamp))
    worker.shutdown()

    assert len(writes) == 1
    assert writes[0][0]["plate_norm"] == "31ط55674"
    assert writes[0][1] is None
    assert state.emitted_events == 1
    assert state.visits.event_refs == {"31ط55674": 77}
    assert state.visits.track_event_refs() == {2: 77}


def test_retry_queue_never_silently_evicts_distinct_events():
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()

    for track_id in range(1, 18):
        worker._enqueue_persistence_retry(
            state,
            persistence_entry(
                worker,
                track_id=track_id,
                plate=f"31ط{550 + track_id:03d}74",
            ),
        )

    assert len(state.persistence_retry) == 17
    assert {entry.result["track_id"] for entry in state.persistence_retry.values()} == set(
        range(1, 18)
    )
    state.persistence_retry.clear()
    worker.shutdown()


def test_retry_entry_keeps_compressed_frame_without_image_arrays():
    worker = live_worker.LiveANPRWorker(max_workers=1)
    frame = np.full((720, 1280, 3), 127, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.95,
        "track_id": 1,
        "bbox": (400, 400, 800, 520),
        "crop": frame[400:520, 400:800].copy(),
        "vehicle_crop": frame[250:650, 250:950].copy(),
    }

    entry = worker._make_persistence_retry(
        1,
        "Gate",
        result,
        frame,
        None,
        "31ط55674",
        10.0,
        12.0,
        30.0,
        0,
    )
    decoded = worker._decode_retry_frame(entry.frame)

    assert isinstance(entry.frame, bytes)
    assert len(entry.frame) < frame.nbytes
    assert decoded.shape == frame.shape
    assert "crop" not in entry.result
    assert "vehicle_crop" not in entry.result
    worker.shutdown()


def test_retry_backoff_is_bounded_after_many_failures():
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    entry = persistence_entry(worker)
    entry.attempts = 9_999
    before = time.time()

    worker._mark_retry_failure(
        state,
        entry,
        OSError("still unavailable"),
        update_outbox=False,
    )

    assert entry.attempts == 10_000
    assert before + 7.5 <= entry.next_attempt_at_epoch <= time.time() + 8.5
    worker.shutdown()


def test_same_track_strict_retry_can_replace_distant_review_guess():
    worker = live_worker.LiveANPRWorker(max_workers=1)
    review = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    review.result["raw_guess_reason"] = "strict-decoder-rejected"
    strict = persistence_entry(
        worker,
        track_id=1,
        plate="31ط52674",
        valid=True,
    )
    strict.observed_at = review.observed_at + 0.2

    assert worker._retry_can_follow(review, strict) is True

    review.result.pop("raw_guess_reason")
    assert worker._retry_can_follow(review, strict) is False
    worker.shutdown()


def test_non_due_retry_blocks_its_correction_but_not_other_track(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    review = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    review.next_attempt_at_epoch = time.time() + 60.0
    correction = persistence_entry(
        worker,
        track_id=2,
        plate="31ط55674",
        valid=True,
    )
    correction.result["bbox"] = (24, 20, 164, 60)
    independent = persistence_entry(
        worker,
        track_id=3,
        plate="12ب34567",
        valid=True,
    )
    independent.result["bbox"] = (300, 200, 440, 240)
    for entry in (review, correction, independent):
        worker._enqueue_persistence_retry(state, entry)
    calls = []
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *args: calls.append(args[3]["plate"]) or 88,
    )

    worker._drain_persistence_retry_locked(
        state,
        threading.RLock(),
    )

    assert calls == ["12ب34567"]
    assert list(state.persistence_retry) == [
        review.retry_key,
        correction.retry_key,
    ]
    state.persistence_retry.clear()
    worker.shutdown()


def test_retry_backpressure_stops_inference_before_frame_copy(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    worker._states[1] = state
    for track_id in range(1, 33):
        worker._enqueue_persistence_retry(
            state,
            persistence_entry(
                worker,
                track_id=track_id,
                plate=f"31ط{550 + track_id:03d}74",
            ),
        )
    monkeypatch.setattr(
        worker,
        "_config",
        lambda *_args: {
            "enabled": 1,
            "lpr_enabled": 1,
        },
    )
    monkeypatch.setattr(
        worker,
        "_selection_score",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("inference preparation must be backpressured")
        ),
    )

    worker.submit(
        1,
        "Gate",
        np.zeros((120, 220, 3), dtype=np.uint8),
    )

    assert state.persistence_backpressure is True
    assert state.persistence_backpressure_frames == 1
    assert state.busy is False
    assert state.pending is None
    state.persistence_retry.clear()
    worker.shutdown()


def test_unspooled_retry_keeps_fail_closed_backpressure(tmp_path, monkeypatch):
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=tmp_path / "retry.db",
    )
    state = live_worker._CameraState()
    monkeypatch.setattr(
        worker._outbox,
        "upsert",
        lambda *_args: (_ for _ in ()).throw(OSError("sidecar offline")),
    )

    worker._enqueue_persistence_retry(
        state,
        persistence_entry(worker, track_id=1),
    )
    state.persistence_backpressure = False

    assert worker._retry_backpressure_active(state) is True
    assert state.persistence_backpressure is True
    assert next(iter(state.persistence_retry.values())).durably_spooled is False
    state.persistence_retry.clear()
    worker.shutdown(retry_timeout=0.0)


def test_recovery_pages_backlog_and_blocks_new_inference(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "bounded-recovery.db"
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    state = live_worker._CameraState()
    for track_id in range(1, 41):
        plate = f"{10 + track_id:02d}ط{100 + track_id:03d}74"
        first._enqueue_persistence_retry(
            state,
            persistence_entry(
                first,
                camera_id=1,
                track_id=track_id,
                plate=plate,
            ),
        )
    assert first._outbox.pending_count() == 40
    monkeypatch.setattr(
        first,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )
    first.shutdown(retry_timeout=0.0)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    loaded = sum(
        len(second._retry_entries(recovered))
        for _camera_id, recovered in second._detached_retry_states
    )
    assert loaded == live_worker.PERSISTENCE_RETRY_LOW_COUNT
    assert second._outbox.pending_stats(1)[0] == 40
    active = live_worker._CameraState()
    second._states[1] = active
    monkeypatch.setattr(
        second,
        "_config",
        lambda *_args: {"enabled": 1, "lpr_enabled": 1},
    )
    monkeypatch.setattr(
        second,
        "_selection_score",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("backpressure must precede frame selection")
        ),
    )

    second.submit(1, "Gate", np.zeros((80, 180, 3), dtype=np.uint8))

    assert active.persistence_backpressure is True
    assert active.persistence_backpressure_frames == 1
    monkeypatch.setattr(
        second,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )
    second.shutdown(retry_timeout=0.0)


def test_submit_captures_wall_clock_at_frame_admission(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    captured = []
    monkeypatch.setattr(live_worker.time, "monotonic", lambda: 123.0)
    monkeypatch.setattr(live_worker.time, "time", lambda: 456.25)
    monkeypatch.setattr(
        worker,
        "_config",
        lambda *_args: {
            "enabled": 1,
            "lpr_enabled": 1,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 100,
            "roi_h": 100,
        },
    )
    monkeypatch.setattr(
        worker._executor,
        "submit",
        lambda _fn, _state, payload: captured.append(payload),
    )

    worker.submit(1, "Gate", np.zeros((80, 180, 3), dtype=np.uint8))

    assert captured[0][3] == 123.0
    assert captured[0][7] == 456.25
    worker._states[1].busy = False
    worker.shutdown()


def test_start_worker_replaces_transient_outbox_init_failure(monkeypatch):
    calls = []

    class BrokenWorker:
        _stopped = False
        _outbox_required = True
        _outbox = None

        def shutdown(self, retry_timeout=0.0):
            calls.append(retry_timeout)
            self._stopped = True
            return True

    replacement = object()
    monkeypatch.setattr(live_worker, "worker", BrokenWorker())
    monkeypatch.setattr(
        live_worker,
        "LiveANPRWorker",
        lambda **_kwargs: replacement,
    )

    assert live_worker.start_live_anpr_worker() is replacement
    assert live_worker.worker is replacement
    assert calls == [0.0]


def test_import_defers_global_outbox_and_retry_thread(tmp_path):
    environment = os.environ.copy()
    environment["BCVISION_DATA_DIR"] = str(tmp_path)
    project_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join([
                "import threading",
                "import app.ai.live_worker as live_worker",
                "assert live_worker.worker._persistence_started is False",
                "assert live_worker.worker._outbox is None",
                "assert live_worker.worker._retry_thread is None",
                "assert not any(",
                "    thread.name == 'bc-anpr-persistence-retry'",
                "    for thread in threading.enumerate()",
                ")",
            ]),
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert not list(tmp_path.glob("bcvision-retry.db*"))


def test_deferred_global_lifecycle_preserves_shutdown_and_backup(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "global-retry.db"
    deferred = live_worker.LiveANPRWorker(
        max_workers=1,
        background_retry=True,
        retry_outbox_path=outbox_path,
        _defer_persistence_start=True,
    )
    monkeypatch.setattr(live_worker, "worker", deferred)
    monkeypatch.setattr(live_worker, "_GLOBAL_OUTBOX_PATH", outbox_path)

    assert live_worker.shutdown_live_anpr_worker() is True
    assert deferred._persistence_started is False
    assert deferred._retry_thread is None
    assert not outbox_path.exists()

    assert live_worker.start_live_anpr_worker() is deferred
    assert live_worker.start_live_anpr_worker() is deferred
    assert deferred._persistence_started is True
    assert deferred._outbox is not None
    assert deferred._retry_thread is not None
    assert deferred._retry_thread.is_alive()
    assert outbox_path.is_file()

    assert live_worker.shutdown_live_anpr_worker() is True
    backup_path = tmp_path / "global-retry-backup.db"
    assert live_worker.backup_live_anpr_outbox(backup_path) == backup_path
    assert backup_path.is_file()

    restarted = live_worker.start_live_anpr_worker()
    assert restarted is live_worker.worker
    assert restarted is not deferred
    assert restarted._retry_thread is not None
    assert restarted._retry_thread.is_alive()
    assert live_worker.shutdown_live_anpr_worker() is True


def test_public_wrapper_activates_same_deferred_worker(tmp_path, monkeypatch):
    deferred = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=tmp_path / "wrapper-retry.db",
        _defer_persistence_start=True,
    )
    status_calls = []
    monkeypatch.setattr(live_worker, "worker", deferred)
    monkeypatch.setattr(
        deferred,
        "status",
        lambda camera_id: status_calls.append(camera_id) or {"ok": True},
    )

    assert live_worker.live_anpr_status(17) == {"ok": True}
    assert live_worker.worker is deferred
    assert deferred._persistence_started is True
    assert deferred._outbox is not None
    assert status_calls == [17]
    assert live_worker.shutdown_live_anpr_worker() is True


def test_video_pass_fails_if_persistence_backpressure_skipped_frames():
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(processed_frames=1)
    worker._states[1] = state
    token = worker.begin_video_pass(1)
    state.processed_frames = 2
    state.persistence_backpressure_frames += 1

    drained = worker.drain_video_pass(1, token, timeout=0.1)

    assert drained["ok"] is False
    assert "backlog skipped" in drained["error"]
    worker.shutdown()


def test_detached_retry_runs_without_shutdown_or_new_frame(monkeypatch):
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        background_retry=True,
    )
    state = live_worker._CameraState()
    worker._states[1] = state
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )
    worker._enqueue_persistence_retry(
        state,
        persistence_entry(worker),
    )

    assert worker.remove(1, retry_timeout=0.0) is False
    monkeypatch.setattr(worker, "_persist", lambda *args: args[5] or 77)
    worker._retry_wakeup.set()
    deadline = time.monotonic() + 2.0
    while state.persistence_retry and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not state.persistence_retry
    assert worker._detached_retry_states == []
    worker.shutdown()


def test_shutdown_fails_closed_while_retry_hydration_is_alive(
    tmp_path,
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        background_retry=True,
        retry_outbox_path=tmp_path / "shutdown-retry.db",
    )
    entered = threading.Event()
    release = threading.Event()

    def blocked_hydration():
        entered.set()
        release.wait(2.0)
        return 0

    monkeypatch.setattr(worker, "_hydrate_outbox_entries", blocked_hydration)
    worker._retry_wakeup.set()
    assert entered.wait(1.0)

    assert worker.shutdown(retry_timeout=0.0) is False
    assert worker._retry_thread.is_alive()
    with pytest.raises(RuntimeError, match="retry thread must stop"):
        worker.backup_retry_outbox(tmp_path / "unsafe-backup.db")

    release.set()
    worker._retry_thread.join(timeout=1.0)
    assert not worker._retry_thread.is_alive()
    assert worker.backup_retry_outbox(
        tmp_path / "safe-backup.db"
    ).is_file()


def test_hydration_rotates_stuck_page_for_independent_camera(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "fair-recovery.db"
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    monkeypatch.setattr(
        first,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    queued = live_worker._CameraState()
    for track_id in range(1, live_worker.PERSISTENCE_RETRY_LOW_COUNT + 1):
        first._enqueue_persistence_retry(
            queued,
            persistence_entry(
                first,
                camera_id=1,
                track_id=track_id,
            ),
        )
    first._enqueue_persistence_retry(
        queued,
        persistence_entry(
            first,
            camera_id=2,
            track_id=99,
            plate="12ب34567",
        ),
    )
    assert first._outbox.pending_count() == 17
    assert first.shutdown(retry_timeout=0.0) is True

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    persisted = []

    def persist(camera_id, *_args):
        if int(camera_id) == 1:
            raise OSError("camera one remains unavailable")
        persisted.append(int(camera_id))
        return 77

    monkeypatch.setattr(second, "_persist", persist)
    initial_states = list(second._detached_retry_states)
    assert sum(
        len(second._retry_entries(state))
        for _camera_id, state in initial_states
    ) == live_worker.PERSISTENCE_RETRY_LOW_COUNT
    for camera_id, state in initial_states:
        second._drain_persistence_retry_locked(
            state,
            second._event_commit_lock(camera_id),
            allow_retired=True,
        )

    assert second._hydrate_outbox_entries() == 1
    hydrated_count = sum(
        len(second._retry_entries(state))
        for _camera_id, state in second._detached_retry_states
    )
    assert hydrated_count <= live_worker.PERSISTENCE_RETRY_LOW_COUNT
    for camera_id, state in list(second._detached_retry_states):
        second._drain_persistence_retry_locked(
            state,
            second._event_commit_lock(camera_id),
            allow_retired=True,
        )

    assert persisted == [2]
    assert second._outbox.pending_stats(1)[0] == 16
    assert second._outbox.pending_stats(2)[0] == 0
    assert sum(
        len(second._retry_entries(state))
        for _camera_id, state in second._detached_retry_states
    ) < live_worker.PERSISTENCE_RETRY_LOW_COUNT
    for _camera_id, state in second._detached_retry_states:
        state.persistence_retry.clear()
    second.shutdown(retry_timeout=0.0)


def test_hydration_refreshes_quarantine_observability(tmp_path):
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=tmp_path / "quarantine-status.db",
    )
    queued = persistence_entry(worker)
    worker._outbox.upsert(worker._outbox_entry(queued))
    with sqlite3.connect(worker._outbox.path) as con:
        con.execute(
            "UPDATE retry_outbox SET result_json='not-json' "
            "WHERE retry_id=?",
            (queued.persistence_id,),
        )

    assert worker._hydrate_outbox_entries() == 0
    assert worker._outbox_quarantined == 1
    assert worker._outbox.quarantined_count() == 1
    worker.shutdown()


def test_durable_outbox_survives_a_new_worker_instance(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "retry-outbox.db"
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    state = live_worker._CameraState()
    queued = persistence_entry(first)
    first._enqueue_persistence_retry(state, queued)
    monkeypatch.setattr(
        first,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )
    first._drain_persistence_retry_locked(state, threading.RLock())
    assert queued.attempts == 1
    assert first._outbox.pending_count() == 1
    first.shutdown(retry_timeout=0.0)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    assert len(second._detached_retry_states) == 1
    camera_id, recovered = second._detached_retry_states[0]
    restored = next(iter(recovered.persistence_retry.values()))
    restored.next_attempt_at_epoch = 0.0
    frames = []

    def persist(*args):
        frames.append(args[2])
        return 77

    monkeypatch.setattr(second, "_persist", persist)
    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera_id),
        allow_retired=True,
    )

    assert frames[0].shape == (80, 180, 3)
    assert not recovered.persistence_retry
    assert second._outbox.pending_count() == 0
    second.shutdown()


def test_recovered_retry_keeps_original_media_policy_and_roots(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "media-policy-outbox.db"
    original_plate_root = (tmp_path / "old-plates").resolve()
    original_snapshot_root = (tmp_path / "old-vehicles").resolve()
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    monkeypatch.setattr(
        first,
        "_setting",
        lambda key, default="": {
            "plate_path": str(original_plate_root),
            "snapshot_path": str(original_snapshot_root),
            "save_plate_images": "1",
            "save_snapshots": "0",
        }.get(key, default),
    )
    state = live_worker._CameraState()
    first._enqueue_persistence_retry(state, persistence_entry(first))
    monkeypatch.setattr(
        first,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )
    first.shutdown(retry_timeout=0.0)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    monkeypatch.setattr(
        second,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "new-plates"),
            "snapshot_path": str(tmp_path / "new-vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "1",
        }.get(key, default),
    )
    camera_id, recovered = second._detached_retry_states[0]
    restored = next(iter(recovered.persistence_retry.values()))
    restored.next_attempt_at_epoch = 0.0
    captured = []
    monkeypatch.setattr(
        second,
        "_persist",
        lambda *args: captured.append(dict(args[3])) or 77,
    )

    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera_id),
        allow_retired=True,
    )

    assert captured[0]["_plate_root"] == str(original_plate_root)
    assert captured[0]["_snapshot_root"] == str(original_snapshot_root)
    assert captured[0]["_save_plate"] is True
    assert captured[0]["_save_vehicle"] is False
    assert captured[0]["_reuse_media_targets"] is True
    second.shutdown()


def test_same_track_retry_order_survives_outbox_recovery(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "ordered-outbox.db"
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    state = live_worker._CameraState()
    first._enqueue_persistence_retry(
        state,
        persistence_entry(
            first,
            track_id=1,
            plate="31ط55874",
            valid=False,
        ),
    )
    first._enqueue_persistence_retry(
        state,
        persistence_entry(
            first,
            track_id=1,
            plate="31ط55674",
            valid=True,
        ),
    )
    first.shutdown(retry_timeout=0.0)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    camera_id, recovered = second._detached_retry_states[0]
    event_ids = []

    def persist(*args):
        event_ids.append(args[5])
        return args[5] or 77

    monkeypatch.setattr(second, "_persist", persist)
    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera_id),
        allow_retired=True,
    )

    assert event_ids == [None, 77]
    assert second._outbox.pending_count() == 0
    second.shutdown()


def test_predecessor_receipt_repairs_successor_after_process_crash(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "predecessor-receipt.db"
    outbox_path = tmp_path / "predecessor-outbox.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    monkeypatch.setattr(
        first,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    state = live_worker._CameraState()
    review = persistence_entry(
        first,
        camera_id=camera_id,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    strict = persistence_entry(
        first,
        camera_id=camera_id,
        track_id=2,
        plate="31ط55674",
        valid=True,
    )
    strict.observed_at = review.observed_at + 0.2
    strict.result["bbox"] = (24, 20, 164, 60)
    first._enqueue_persistence_retry(state, review)
    first._enqueue_persistence_retry(state, strict)
    assert strict.predecessor_id == review.persistence_id

    review_result = {
        **review.result,
        "_persistence_id": review.persistence_id,
        "_observed_at_utc": review.observed_at_utc,
        "_plate_root": review.plate_root,
        "_snapshot_root": review.snapshot_root,
        "_save_plate": review.save_plate,
        "_save_vehicle": review.save_vehicle,
        "_reuse_media_targets": True,
    }
    review_id = first._persist(
        camera_id,
        "Gate",
        first._decode_retry_frame(review.frame),
        review_result,
        review.processing_ms,
        duplicate_seconds=30,
    )
    first._outbox.delete(review.persistence_id)
    # Emulate power loss after the predecessor transaction+ACK delete but
    # before the in-memory successor receives that event id.
    first._stopped = True
    first._executor.shutdown(wait=True, cancel_futures=False)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    camera, recovered = second._detached_retry_states[0]
    restored = next(iter(recovered.persistence_retry.values()))
    assert restored.predecessor_id == review.persistence_id
    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera),
        allow_retired=True,
    )
    second.shutdown()

    with app.database.connect() as con:
        events = con.execute(
            "SELECT id,plate_norm FROM plate_events"
        ).fetchall()
        receipts = con.execute(
            "SELECT persistence_key,event_id "
            "FROM anpr_persistence_receipts ORDER BY persistence_key"
        ).fetchall()
    assert [tuple(row) for row in events] == [(review_id, "31ط55674")]
    assert {tuple(row) for row in receipts} == {
        (review.persistence_id, review_id),
        (strict.persistence_id, review_id),
    }
    assert second._outbox.pending_count() == 0


def test_latest_replacement_timestamp_preserves_crash_predecessor(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "replacement-predecessor.db"
    outbox_path = tmp_path / "replacement-predecessor-outbox.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    monkeypatch.setattr(
        first,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    state = live_worker._CameraState()
    review = persistence_entry(
        first,
        camera_id=camera_id,
        track_id=7,
        valid=False,
    )
    review.observed_at = 0.0
    review.observed_at_utc = "2026-01-01T00:00:00.000000Z"
    first._enqueue_persistence_retry(state, review)

    refreshed_review = persistence_entry(
        first,
        camera_id=camera_id,
        track_id=7,
        valid=False,
    )
    refreshed_review.observed_at = 1.4
    refreshed_review.observed_at_utc = (
        "2026-01-01T00:00:01.400000Z"
    )
    first._enqueue_persistence_retry(state, refreshed_review)
    assert refreshed_review.persistence_id == review.persistence_id
    assert refreshed_review.observed_at == 1.4

    strict = persistence_entry(
        first,
        camera_id=camera_id,
        track_id=7,
        valid=True,
    )
    strict.observed_at = 2.0
    strict.observed_at_utc = "2026-01-01T00:00:02.000000Z"
    first._enqueue_persistence_retry(state, strict)
    assert strict.predecessor_id == refreshed_review.persistence_id

    review_result = {
        **refreshed_review.result,
        "_persistence_id": refreshed_review.persistence_id,
        "_observed_at_utc": refreshed_review.observed_at_utc,
        "_plate_root": refreshed_review.plate_root,
        "_snapshot_root": refreshed_review.snapshot_root,
        "_save_plate": refreshed_review.save_plate,
        "_save_vehicle": refreshed_review.save_vehicle,
        "_reuse_media_targets": True,
    }
    review_id = first._persist(
        camera_id,
        "Gate",
        first._decode_retry_frame(refreshed_review.frame),
        review_result,
        refreshed_review.processing_ms,
        duplicate_seconds=30,
    )
    first._outbox.delete(refreshed_review.persistence_id)
    # Crash after the refreshed review ACK but before strict.event_id is
    # copied into its durable row. The predecessor receipt must bridge it.
    first._stopped = True
    first._executor.shutdown(wait=True, cancel_futures=False)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    camera, recovered = second._detached_retry_states[0]
    restored = next(iter(recovered.persistence_retry.values()))
    assert restored.predecessor_id == refreshed_review.persistence_id
    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera),
        allow_retired=True,
    )
    second.shutdown()

    with app.database.connect() as con:
        rows = con.execute(
            "SELECT id,plate_norm,review_status FROM plate_events"
        ).fetchall()
        receipts = con.execute(
            "SELECT persistence_key,event_id "
            "FROM anpr_persistence_receipts"
        ).fetchall()
    assert [tuple(row) for row in rows] == [
        (review_id, "31ط55674", "confirmed-ai")
    ]
    assert {receipt["persistence_key"] for receipt in receipts} == {
        refreshed_review.persistence_id,
        strict.persistence_id,
    }
    assert second._outbox.pending_count() == 0


def test_receipt_prevents_duplicate_when_outbox_ack_delete_retries(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "receipt-replay.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=tmp_path / "receipt-outbox.db",
    )
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    state = live_worker._CameraState()
    queued = persistence_entry(worker, camera_id=camera_id)
    worker._enqueue_persistence_retry(state, queued)
    real_delete = worker._outbox.delete
    delete_calls = 0

    def fail_once(retry_id):
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise OSError("outbox temporarily unavailable")
        return real_delete(retry_id)

    monkeypatch.setattr(worker._outbox, "delete", fail_once)
    worker._drain_persistence_retry_locked(state, threading.RLock())

    with app.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 1
        assert con.execute(
            "SELECT COUNT(*) FROM anpr_persistence_receipts"
        ).fetchone()[0] == 1
    assert len(state.persistence_retry) == 1
    queued.next_attempt_at_epoch = 0.0

    worker._drain_persistence_retry_locked(state, threading.RLock())

    with app.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 1
        assert con.execute(
            "SELECT COUNT(*) FROM anpr_persistence_receipts"
        ).fetchone()[0] == 1
    assert not state.persistence_retry
    assert state.emitted_events == 1
    assert worker._outbox.pending_count() == 0
    worker.shutdown()


def test_committed_retry_token_does_not_swallow_same_key_improvement(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "committed-token-refresh.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=tmp_path / "committed-token-outbox.db",
    )
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    state = live_worker._CameraState()
    original = persistence_entry(worker, camera_id=camera_id)
    original.result.update({
        "confidence": 0.80,
        "ocr_confidence": 0.80,
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
    })
    worker._enqueue_persistence_retry(state, original)
    real_delete = worker._outbox.delete
    delete_calls = 0

    def fail_once(retry_id):
        nonlocal delete_calls
        delete_calls += 1
        if delete_calls == 1:
            raise OSError("outbox ACK unavailable")
        return real_delete(retry_id)

    monkeypatch.setattr(worker._outbox, "delete", fail_once)
    worker._drain_persistence_retry_locked(state, threading.RLock())
    assert original.primary_committed is True

    improved = persistence_entry(worker, camera_id=camera_id)
    improved.result.update({
        "confidence": 0.99,
        "ocr_confidence": 0.99,
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
    })
    original_token = original.persistence_id
    improved_token = improved.persistence_id
    worker._enqueue_persistence_retry(state, improved)
    assert improved.persistence_id == improved_token
    assert improved.persistence_id != original_token
    assert improved.event_id == original.event_id
    assert improved.predecessor_id == original_token
    assert worker._outbox.pending_count() == 2

    monkeypatch.setattr(worker._outbox, "delete", real_delete)
    for entry in state.persistence_retry.values():
        entry.next_attempt_at_epoch = 0.0
    worker._drain_persistence_retry_locked(state, threading.RLock())

    with app.database.connect() as con:
        row = con.execute(
            "SELECT confidence,ocr_confidence FROM plate_events"
        ).fetchone()
        receipts = con.execute(
            "SELECT persistence_key,event_id "
            "FROM anpr_persistence_receipts"
        ).fetchall()
    assert tuple(row) == (0.99, 0.99)
    assert {receipt["persistence_key"] for receipt in receipts} == {
        original_token,
        improved_token,
    }
    assert len({receipt["event_id"] for receipt in receipts}) == 1
    assert not state.persistence_retry
    assert worker._outbox.pending_count() == 0
    worker.shutdown()


def test_recovered_retry_never_mutates_new_camera_state(
    tmp_path,
    monkeypatch,
):
    outbox_path = tmp_path / "isolated-recovery.db"
    first = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    old_state = live_worker._CameraState()
    first._enqueue_persistence_retry(
        old_state,
        persistence_entry(first, camera_id=1, event_id=77),
    )
    first.shutdown(retry_timeout=0.0)

    second = live_worker.LiveANPRWorker(
        max_workers=1,
        retry_outbox_path=outbox_path,
    )
    new_state = live_worker._CameraState()
    current = {
        "plate": "12ب34567",
        "plate_norm": "12ب34567",
        "valid": True,
        "track_id": 2,
    }
    new_state.visits.register(current, 90, 100.0)
    new_state.track_event_ids = {2: 90}
    second._states[1] = new_state
    camera_id, recovered = second._detached_retry_states[0]
    monkeypatch.setattr(second, "_persist", lambda *args: args[5])

    second._drain_persistence_retry_locked(
        recovered,
        second._event_commit_lock(camera_id),
        allow_retired=True,
    )

    assert new_state.track_event_ids == {2: 90}
    assert new_state.visits.event_refs == {"12ب34567": 90}
    second.shutdown()


def test_failed_same_track_insert_precedes_queued_update(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()

    class ActiveTracker:
        @staticmethod
        def active_track_ids():
            return {1}

    state.tracker = ActiveTracker()
    insert = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    update = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55674",
        valid=True,
    )
    worker._enqueue_persistence_retry(state, insert)
    worker._enqueue_persistence_retry(state, update)
    event_ids = []

    def persist(*args):
        event_id = args[5]
        event_ids.append(event_id)
        if len(event_ids) == 1:
            raise OSError("database busy")
        return event_id or 77

    monkeypatch.setattr(worker, "_persist", persist)
    commit_lock = threading.RLock()

    worker._drain_persistence_retry_locked(state, commit_lock)
    assert event_ids == [None]
    assert len(state.persistence_retry) == 2

    worker._drain_persistence_retry_locked(state, commit_lock)

    assert event_ids == [None, None, 77]
    assert not state.persistence_retry
    assert state.visits.event_refs == {"31ط55674": 77}
    worker.shutdown()


def test_retired_retry_chain_propagates_saved_event_id(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(retired=True)
    insert = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    update = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55674",
        valid=True,
    )
    worker._enqueue_persistence_retry(state, insert)
    worker._enqueue_persistence_retry(state, update)
    event_ids = []

    def persist(*args):
        event_ids.append(args[5])
        return args[5] or 77

    monkeypatch.setattr(worker, "_persist", persist)
    worker._drain_persistence_retry_locked(
        state,
        threading.RLock(),
        allow_retired=True,
    )

    assert event_ids == [None, 77]
    assert not state.persistence_retry
    assert state.visits.event_refs == {}
    worker.shutdown()


def test_stale_retry_chain_does_not_mutate_current_generation(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.detector_model_revision = "detector-new"
    worker._detector_generation = 1
    insert = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
        generation=0,
        detector_revision="detector-old",
    )
    update = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55674",
        valid=True,
        generation=0,
        detector_revision="detector-old",
    )
    worker._enqueue_persistence_retry(state, insert)
    worker._enqueue_persistence_retry(state, update)
    event_ids = []

    def persist(*args):
        event_ids.append(args[5])
        return args[5] or 77

    monkeypatch.setattr(worker, "_persist", persist)
    worker._drain_persistence_retry_locked(state, threading.RLock())

    assert event_ids == [None, 77]
    assert not state.persistence_retry
    assert state.track_event_ids == {}
    assert state.visits.event_refs == {}
    assert state.emitted_events == 0
    worker.shutdown()


def test_stale_retry_failure_does_not_poison_current_error_state(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.detector_model_revision = "detector-new"
    worker._detector_generation = 1
    stale = persistence_entry(
        worker,
        generation=0,
        detector_revision="detector-old",
    )
    worker._enqueue_persistence_retry(state, stale)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )

    worker._drain_persistence_retry_locked(state, threading.RLock())

    assert stale.attempts == 1
    assert "database busy" in stale.last_error
    assert state.processing_errors == 0
    assert state.persistence_errors == 0
    assert state.last_error == ""
    state.persistence_retry.clear()
    worker.shutdown()


def test_failed_fragmented_review_blocks_clear_until_lineage_is_saved(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    review = persistence_entry(
        worker,
        track_id=1,
        plate="31ط55874",
        valid=False,
    )
    clear = persistence_entry(
        worker,
        track_id=2,
        plate="31ط55674",
        valid=True,
    )
    clear.result["bbox"] = (24, 20, 164, 60)
    worker._enqueue_persistence_retry(state, review)
    worker._enqueue_persistence_retry(state, clear)
    calls = []

    def persist(*args):
        identity = (
            args[3].get("plate_norm")
            or args[3].get("raw_guess_norm")
        )
        calls.append((identity, args[5]))
        if len(calls) == 1:
            raise OSError("database busy")
        return args[5] or 77

    monkeypatch.setattr(worker, "_persist", persist)
    lock = threading.RLock()

    worker._drain_persistence_retry_locked(state, lock)
    assert calls == [("31ط55874", None)]
    assert len(state.persistence_retry) == 2

    worker._drain_persistence_retry_locked(state, lock)

    assert calls == [
        ("31ط55874", None),
        ("31ط55874", None),
        ("31ط55674", 77),
    ]
    assert not state.persistence_retry
    assert state.emitted_events == 1
    assert state.visits.event_refs == {"31ط55674": 77}
    worker.shutdown()


def test_retry_preserves_event_id_without_poisoning_new_generation(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    entry = persistence_entry(
        worker,
        track_id=1,
        event_id=77,
        generation=0,
        detector_revision="detector-old",
    )
    entry.attempts = 1
    worker._enqueue_persistence_retry(state, entry)
    worker._detector_generation = 1
    state.detector_model_revision = "detector-new"
    revision_entry = persistence_entry(
        worker,
        track_id=2,
        plate="12ب34567",
        event_id=78,
        generation=1,
        detector_revision="detector-old",
    )
    revision_entry.attempts = 1
    worker._enqueue_persistence_retry(state, revision_entry)
    persisted_ids = []

    def persist(*args):
        persisted_ids.append(args[5])
        return args[5]

    monkeypatch.setattr(worker, "_persist", persist)
    worker._drain_persistence_retry_locked(
        state,
        threading.RLock(),
    )

    assert persisted_ids == [77, 78]
    assert not state.persistence_retry
    assert state.track_event_ids == {}
    assert state.visits.event_refs == {}
    assert state.emitted_events == 0
    worker.shutdown()


def test_video_drain_retries_last_event_without_another_frame(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[1] = state
    token = worker.begin_video_pass(1)
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "ocr_confidence": 0.95,
        "detector_confidence": 0.97,
        "quality_score": 0.90,
        "bbox": (20, 30, 160, 70),
        "crop": frame[30:70, 20:160].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(detected)],
    )
    attempts = []

    def persist(*args):
        attempts.append(args[5])
        if len(attempts) == 1:
            raise OSError("database busy")
        return args[5] or 77

    monkeypatch.setattr(worker, "_persist", persist)
    for timestamp in (0.0, 0.1, 0.2):
        state.busy = True
        worker._process(state, (1, "video", frame.copy(), timestamp))

    assert len(state.persistence_retry) == 1
    drained = worker.drain_video_pass(1, token, timeout=1.0)

    assert drained["ok"] is True
    assert attempts == [None, None]
    assert not state.persistence_retry
    assert state.visits.event_refs == {"31ط55674": 77}
    worker.shutdown()


def test_video_drain_timeout_keeps_failed_retry(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(processed_frames=1)
    worker._states[1] = state
    token = worker.begin_video_pass(1)
    worker._enqueue_persistence_retry(
        state,
        persistence_entry(worker),
    )
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )

    drained = worker.drain_video_pass(1, token, timeout=0.1)

    assert drained["ok"] is False
    assert drained["pending_retry_count"] == 1
    assert len(state.persistence_retry) == 1
    monkeypatch.setattr(worker, "_persist", lambda *args: args[5] or 77)
    assert worker.shutdown() is True


def test_remove_and_shutdown_flush_queued_events(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    removed = live_worker._CameraState()
    remaining = live_worker._CameraState()
    worker._states[1] = removed
    worker._states[2] = remaining
    worker._enqueue_persistence_retry(
        removed,
        persistence_entry(worker, camera_id=1, track_id=1),
    )
    worker._enqueue_persistence_retry(
        remaining,
        persistence_entry(
            worker,
            camera_id=2,
            track_id=2,
            plate="12ب34567",
        ),
    )
    persisted = []

    def persist(*args):
        persisted.append(args[0])
        return 70 + args[0]

    monkeypatch.setattr(worker, "_persist", persist)

    assert worker.remove(1) is True
    assert not removed.persistence_retry
    assert worker.shutdown() is True
    assert not remaining.persistence_retry
    assert persisted == [1, 2]


def test_remove_preserves_failed_retry_for_shutdown_flush(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    worker._states[1] = state
    worker._enqueue_persistence_retry(
        state,
        persistence_entry(worker),
    )
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args: (_ for _ in ()).throw(OSError("database busy")),
    )

    assert worker.remove(1, retry_timeout=0.0) is False
    assert len(state.persistence_retry) == 1
    assert worker._detached_retry_states == [(1, state)]
    detached_status = worker.status(1)
    assert detached_status["active"] is True
    assert detached_status["pending_retry_count"] == 1
    assert detached_status["pending_retry_bytes"] > 0

    monkeypatch.setattr(worker, "_persist", lambda *args: args[5] or 77)
    assert worker.shutdown() is True
    assert not state.persistence_retry
    assert worker._detached_retry_states == []


def test_unreadable_vehicle_event_is_upgraded_without_duplicate(
    tmp_path,
    monkeypatch,
):
    _allow_unit_media_writes(monkeypatch)
    db_path = tmp_path / "events.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            plate_image_path TEXT,
            detector_method TEXT,
            ocr_confidence REAL,
            ocr_engine TEXT,
            ocr_alternative TEXT,
            ocr_disagreement INTEGER,
            vehicle_type TEXT,
            vehicle_color TEXT,
            vehicle_brand TEXT,
            vehicle_confidence REAL,
            direction TEXT,
            quality_score REAL,
            consensus_votes INTEGER,
            source TEXT,
            processing_ms REAL,
            media_status TEXT,
            media_error TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    settings = {
        "plate_path": str(tmp_path / "تصاویر پلاک"),
        "snapshot_path": str(tmp_path / "تصاویر خودرو"),
        "save_plate_images": "1",
        "save_snapshots": "1",
    }
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )
    frame = np.full((160, 260, 3), 175, dtype=np.uint8)
    base = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "valid": False,
        "confidence": 0.41,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "ocr_engine": "none",
        "ocr_alternative": "",
        "ocr_disagreement": False,
        "quality_score": 0.76,
        "bbox": (80, 95, 180, 125),
        # The tracker can emit a best frame without retaining its crop.
        # Persistence must reconstruct the crop from the detector bbox.
        "crop": None,
        "method": "test",
        "consensus_votes": 0,
    }
    event_id = worker._persist(3, "Gate", frame, base, 25.0)
    recognized = dict(base)
    recognized.update({
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "ocr_confidence": 0.90,
        "ocr_engine": "crnn-onnx",
        "ocr_alternative": "31-ط-558-74",
        "ocr_disagreement": True,
        "consensus_votes": 3,
    })
    updated_id = worker._persist(
        3,
        "Gate",
        frame,
        recognized,
        28.0,
        event_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT * FROM plate_events ORDER BY id"
        ).fetchall()
    assert updated_id == event_id
    assert len(rows) == 1
    assert rows[0]["plate_norm"] == "31ط55674"
    assert rows[0]["consensus_votes"] == 3
    assert rows[0]["ocr_engine"] == "crnn-onnx"
    assert rows[0]["ocr_alternative"] == "31-ط-558-74"
    assert rows[0]["ocr_disagreement"] == 1
    assert rows[0]["media_status"] == "complete"
    assert rows[0]["media_error"] == ""
    plate_path = Path(rows[0]["plate_image_path"])
    vehicle_path = Path(rows[0]["image_path"])
    assert plate_path.parent == tmp_path / "تصاویر پلاک"
    assert vehicle_path.parent == tmp_path / "تصاویر خودرو"
    for image_path in (plate_path, vehicle_path):
        payload = image_path.read_bytes()
        assert len(payload) > 0
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None
        assert decoded.size > 0


def test_confirmed_event_is_not_overwritten_by_different_plate(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "identity-events.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            plate_image_path TEXT,
            review_status TEXT,
            source TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    first = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    first_id = worker._persist(8, "Gate", frame, first, 20.0)
    second = {
        **first,
        "plate": "98-م-765-43",
        "plate_norm": "98م76543",
    }
    second_id = worker._persist(
        8,
        "Gate",
        frame,
        second,
        20.0,
        first_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT id,plate_norm,review_status "
            "FROM plate_events ORDER BY id"
        ).fetchall()
    assert second_id != first_id
    assert [
        (row["id"], row["plate_norm"], row["review_status"])
        for row in rows
    ] == [
        (first_id, "12ب34567", "confirmed-ai"),
        (second_id, "", "suggested"),
    ]


def test_confirmed_event_cannot_be_downgraded_by_reviewable_result(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "immutable-identity-events.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            plate_image_path TEXT,
            review_status TEXT,
            source TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    confirmed = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    event_id = worker._persist(
        8,
        "Gate",
        frame,
        confirmed,
        20.0,
    )
    reviewable = {
        **confirmed,
        "plate": "98-م-765-43",
        "plate_norm": "",
        "raw_guess_norm": "98م76543",
        "valid": False,
        "needs_review": True,
    }
    returned_id = worker._persist(
        8,
        "Gate",
        frame,
        reviewable,
        20.0,
        event_id,
    )
    worker.shutdown()

    with connect() as con:
        rows = con.execute(
            "SELECT id,plate_text,plate_norm,review_status "
            "FROM plate_events ORDER BY id"
        ).fetchall()
    assert returned_id == event_id
    assert len(rows) == 1
    assert rows[0]["plate_norm"] == "12ب34567"
    assert rows[0]["plate_text"] == "12-ب-345-67"
    assert rows[0]["review_status"] == "confirmed-ai"


def test_recent_exact_event_is_reused_after_worker_state_restart(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "restart-dedup.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    settings = {
        "plate_path": str(tmp_path / "plates"),
        "snapshot_path": str(tmp_path / "vehicles"),
        "save_plate_images": "0",
        "save_snapshots": "0",
    }
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.94,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }
    first_worker = live_worker.LiveANPRWorker(max_workers=1)
    second_worker = live_worker.LiveANPRWorker(max_workers=1)
    for worker in (first_worker, second_worker):
        monkeypatch.setattr(
            worker,
            "_setting",
            lambda key, default="": settings.get(key, default),
        )

    first_id = first_worker._persist(
        camera_id,
        "Gate",
        frame,
        dict(result),
        20.0,
        duplicate_seconds=30,
    )
    second_id = second_worker._persist(
        camera_id,
        "Gate",
        frame,
        dict(result),
        20.0,
        duplicate_seconds=30,
    )
    first_worker.shutdown()
    second_worker.shutdown()

    with app.database.connect() as con:
        count = int(con.execute(
            "SELECT COUNT(*) FROM plate_events WHERE camera_id=?",
            (camera_id,),
        ).fetchone()[0])
    assert second_id == first_id
    assert count == 1


def test_stale_context_cannot_reuse_recent_new_generation_event(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "stale-retry-isolation.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    current = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "model_revision": "detector-new",
        "bbox": (60, 70, 160, 100),
    }
    current_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        current,
        20.0,
        duplicate_seconds=30,
    )
    stale = {
        **current,
        "confidence": 0.51,
        "model_revision": "detector-old",
        "_allow_recent_reuse": False,
    }

    stale_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        stale,
        20.0,
        duplicate_seconds=30,
    )
    worker.shutdown()

    with app.database.connect() as con:
        rows = con.execute(
            "SELECT id,confidence,model_revision FROM plate_events "
            "ORDER BY id"
        ).fetchall()
    assert stale_id != current_id
    assert [
        (row["id"], row["confidence"], row["model_revision"])
        for row in rows
    ] == [
        (current_id, 0.96, "detector-new"),
        (stale_id, 0.51, "detector-old"),
    ]


def test_retry_token_reuses_media_targets_and_committed_insert(
    tmp_path,
    monkeypatch,
):
    _allow_unit_media_writes(monkeypatch)
    import app.database

    db_path = tmp_path / "idempotent-retry.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
        con.execute("""
            CREATE TRIGGER fail_first_retry_insert
            BEFORE INSERT ON plate_events
            BEGIN
                SELECT RAISE(FAIL, 'simulated database outage');
            END
        """)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    plate_dir = tmp_path / "plates"
    vehicle_dir = tmp_path / "vehicles"
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(plate_dir),
            "snapshot_path": str(vehicle_dir),
            "save_plate_images": "1",
            "save_snapshots": "1",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
        "_persistence_id": "retry-token-77",
        "_allow_recent_reuse": False,
    }

    with pytest.raises(sqlite3.IntegrityError):
        worker._persist(
            camera_id,
            "Gate",
            frame,
            result,
            20.0,
            duplicate_seconds=30,
        )
    assert [path.name for path in plate_dir.iterdir()] == [
        "plate-live-retry-token-77.jpg"
    ]
    assert [path.name for path in vehicle_dir.iterdir()] == [
        "vehicle-live-retry-token-77.jpg"
    ]

    with app.database.connect() as con:
        con.execute("DROP TRIGGER fail_first_retry_insert")
    event_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
        duplicate_seconds=30,
    )
    replay_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        {**result, "confidence": 0.97},
        20.0,
        duplicate_seconds=30,
    )
    worker.shutdown()

    with app.database.connect() as con:
        rows = con.execute(
            "SELECT id,confidence FROM plate_events"
        ).fetchall()
        receipts = con.execute(
            "SELECT persistence_key,event_id "
            "FROM anpr_persistence_receipts"
        ).fetchall()
    assert replay_id == event_id
    assert [tuple(row) for row in rows] == [
        (event_id, 0.96)
    ]
    assert [tuple(row) for row in receipts] == [
        ("retry-token-77", event_id)
    ]
    assert len(list(plate_dir.iterdir())) == 1
    assert len(list(vehicle_dir.iterdir())) == 1


def test_event_and_persistence_receipt_commit_atomically(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "atomic-receipt.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
        con.execute("""
            CREATE TRIGGER fail_receipt_insert
            BEFORE INSERT ON anpr_persistence_receipts
            BEGIN
                SELECT RAISE(FAIL, 'receipt unavailable');
            END
        """)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
        "_persistence_id": "atomic-receipt-token",
    }

    with pytest.raises(sqlite3.IntegrityError):
        worker._persist(
            camera_id,
            "Gate",
            frame,
            result,
            20.0,
        )

    with app.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 0
        assert con.execute(
            "SELECT COUNT(*) FROM anpr_persistence_receipts"
        ).fetchone()[0] == 0
    worker.shutdown()


def test_missing_receipt_table_fails_closed_before_event_commit(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "missing-receipt.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
        con.execute("DROP TABLE anpr_persistence_receipts")
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
        "_persistence_id": "receipt-table-required",
    }

    with pytest.raises(sqlite3.OperationalError):
        worker._persist(
            camera_id,
            "Gate",
            np.full((120, 220, 3), 140, dtype=np.uint8),
            result,
            20.0,
        )

    with app.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 0
    worker.shutdown()


def test_receipt_tombstone_prevents_retention_replay_resurrection(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "receipt-tombstone.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
        "_persistence_id": "retained-tombstone",
    }
    event_id = worker._persist(camera_id, "Gate", frame, result, 20.0)
    with app.database.connect() as con:
        con.execute("DELETE FROM plate_events WHERE id=?", (event_id,))
        assert con.execute(
            "SELECT event_id FROM anpr_persistence_receipts "
            "WHERE persistence_key=?",
            ("retained-tombstone",),
        ).fetchone()[0] == event_id

    replay_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        {**result, "confidence": 0.50},
        20.0,
    )

    with app.database.connect() as con:
        assert con.execute(
            "SELECT COUNT(*) FROM plate_events"
        ).fetchone()[0] == 0
    assert replay_id == event_id
    worker.shutdown()


def test_no_downgrade_success_records_predecessor_receipt(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "no-downgrade-receipt.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    confirmed = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
    }
    event_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        confirmed,
        20.0,
    )
    persistence_id = "review-no-downgrade-receipt"
    review = {
        "plate": "12-ب-345-67",
        "plate_norm": "",
        "raw_guess_norm": "12ب34567",
        "raw_guess_text": "12-ب-345-67",
        "valid": False,
        "needs_review": True,
        "confidence": 0.44,
        "bbox": (62, 70, 162, 100),
        "_persistence_id": persistence_id,
    }

    assert worker._persist(
        camera_id,
        "Gate",
        frame,
        review,
        20.0,
        event_id=event_id,
    ) == event_id

    with app.database.connect() as con:
        events = con.execute(
            "SELECT id,plate_norm FROM plate_events"
        ).fetchall()
        receipt = con.execute(
            "SELECT event_id FROM anpr_persistence_receipts "
            "WHERE persistence_key=?",
            (persistence_id,),
        ).fetchone()
    assert [tuple(row) for row in events] == [(event_id, "31ط55674")]
    assert receipt["event_id"] == event_id
    worker.shutdown()


def test_older_same_identity_retry_cannot_downgrade_newer_event(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "older-same-identity.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url) VALUES(?,?)",
            ("Gate", "rtsp://gate"),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    current = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "ocr_engine": "hezar-crnn-fa-v2-onnx",
        "model_revision": "hezar-new",
        "bbox": (60, 70, 160, 100),
        "_observed_at_utc": "2026-01-02T03:04:06Z",
        "_persistence_id": "newer-observation",
    }
    event_id = worker._persist(camera_id, "Gate", frame, current, 20.0)
    older_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        {
            **current,
            "confidence": 0.51,
            "ocr_engine": "legacy-ocr",
            "model_revision": "legacy-model",
            "_observed_at_utc": "2026-01-02T03:04:05Z",
            "_persistence_id": "older-observation",
        },
        20.0,
        event_id=event_id,
    )

    with app.database.connect() as con:
        row = con.execute(
            "SELECT confidence,ocr_engine,model_revision,updated_at "
            "FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
        receipts = con.execute(
            "SELECT persistence_key,event_id FROM anpr_persistence_receipts"
        ).fetchall()
    assert older_id == event_id
    assert tuple(row) == (
        0.96,
        "hezar-crnn-fa-v2-onnx",
        "hezar-new",
        "2026-01-02 03:04:06.000000",
    )
    assert {tuple(receipt) for receipt in receipts} == {
        ("newer-observation", event_id),
        ("older-observation", event_id),
    }
    worker.shutdown()


def test_delayed_retry_uses_observation_time_for_recent_lookup(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "observation-time.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,duplicate_seconds) "
            "VALUES(?,?,?)",
            ("Gate", "rtsp://gate", 30),
        ).lastrowid)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": {
            "plate_path": str(tmp_path / "plates"),
            "snapshot_path": str(tmp_path / "vehicles"),
            "save_plate_images": "0",
            "save_snapshots": "0",
        }.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.96,
        "bbox": (60, 70, 160, 100),
    }
    current_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
        duplicate_seconds=30,
    )
    delayed_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        {
            **result,
            "_observed_at_utc": "2026-01-02 03:04:05.000000",
            "_persistence_id": "delayed-observation",
        },
        20.0,
        duplicate_seconds=30,
    )
    worker.shutdown()

    with app.database.connect() as con:
        rows = con.execute(
            "SELECT id,created_at FROM plate_events ORDER BY id"
        ).fetchall()
    assert delayed_id != current_id
    assert rows[1]["created_at"] == "2026-01-02 03:04:05.000000"


def test_media_encoder_failure_keeps_text_event_and_records_error(
    tmp_path,
    monkeypatch,
):
    db_path = tmp_path / "events.db"
    with sqlite3.connect(db_path) as con:
        con.executescript("""
        CREATE TABLE plate_events(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plate_text TEXT,
            plate_norm TEXT,
            confidence REAL,
            camera_id INTEGER,
            camera_name TEXT,
            image_path TEXT,
            plate_image_path TEXT,
            media_status TEXT,
            media_error TEXT
        );
        """)

    import app.database

    def connect():
        con = sqlite3.connect(db_path)
        con.row_factory = sqlite3.Row
        return con

    monkeypatch.setattr(app.database, "connect", connect)
    worker = live_worker.LiveANPRWorker(max_workers=1)
    settings = {
        "plate_path": str(tmp_path / "plates"),
        "snapshot_path": str(tmp_path / "vehicles"),
        "save_plate_images": "1",
        "save_snapshots": "1",
    }
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )
    monkeypatch.setattr(
        media_storage.cv2,
        "imencode",
        lambda *_args, **_kwargs: (False, None),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.91,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }

    event_id = worker._persist(8, "Gate", frame, result, 20.0)
    worker.shutdown()

    with connect() as con:
        row = con.execute(
            "SELECT * FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()
    assert row["plate_text"] == "12-ب-345-67"
    assert row["plate_norm"] == "12ب34567"
    assert row["plate_image_path"] == ""
    assert row["image_path"] == ""
    assert row["media_status"] == "error"
    assert "plate: JPEG encoder returned no data" in row["media_error"]
    assert "vehicle: JPEG encoder returned no data" in row["media_error"]
    assert list(tmp_path.rglob("*.tmp")) == []


def test_existing_event_keeps_original_observation_city(
    tmp_path,
    monkeypatch,
):
    import app.database

    db_path = tmp_path / "city-snapshot.db"
    monkeypatch.setattr(app.database, "DB_PATH", db_path)
    app.database.init_db()
    with app.database.connect() as con:
        camera_id = int(con.execute(
            "INSERT INTO cameras(name,rtsp_url,location,city) "
            "VALUES(?,?,?,?)",
            ("Gate", "rtsp://gate", "ورودی شمالی", "تهران"),
        ).lastrowid)

    worker = live_worker.LiveANPRWorker(max_workers=1)
    settings = {
        "plate_path": str(tmp_path / "plates"),
        "snapshot_path": str(tmp_path / "vehicles"),
        "save_plate_images": "1",
        "save_snapshots": "1",
    }
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": settings.get(key, default),
    )
    frame = np.full((120, 220, 3), 140, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.91,
        "bbox": (60, 70, 160, 100),
        "crop": frame[70:100, 60:160].copy(),
    }

    event_id = worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
    )
    with app.database.connect() as con:
        con.execute(
            "UPDATE cameras SET city='شیراز' WHERE id=?",
            (camera_id,),
        )
    worker._persist(
        camera_id,
        "Gate",
        frame,
        result,
        20.0,
        event_id,
    )
    worker.shutdown()

    with app.database.connect() as con:
        city = con.execute(
            "SELECT city FROM plate_events WHERE id=?",
            (event_id,),
        ).fetchone()[0]
    assert city == "تهران"


def test_roi_and_translation():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    source, x, y = live_worker.LiveANPRWorker._roi_frame(
        frame,
        {
            "roi_x": 10,
            "roi_y": 20,
            "roi_w": 50,
            "roi_h": 40,
        },
    )
    assert source.shape[:2] == (40, 100)
    assert (x, y) == (20, 20)
    row = live_worker.LiveANPRWorker._translate(
        {
            "bbox": (1, 2, 11, 12),
            "vehicle_bbox": (0, 0, 20, 20),
        },
        x,
        y,
    )
    assert row["bbox"] == (21, 22, 31, 32)
    assert row["vehicle_bbox"] == (20, 20, 40, 40)


def test_operator_assisted_rows_replaces_only_unreadable_overlap():
    baseline = [{
        "bbox": (20, 20, 140, 60),
        "plate": "ناخوانا",
        "valid": False,
        "needs_review": True,
    }, {
        "bbox": (180, 20, 300, 60),
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "needs_review": False,
    }]
    shadow = [{
        "bbox": (22, 21, 142, 61),
        "plate": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": False,
    }, {
        "bbox": (182, 21, 302, 61),
        "plate": "12-ب-345-76",
        "raw_guess_norm": "12ب34576",
        "valid": False,
    }]

    selected = live_worker.operator_assisted_rows(baseline, shadow)

    assert len(selected) == 2
    assert selected[0]["raw_guess_norm"] == "31ط55674"
    assert selected[0]["assisted_candidate"] is True
    assert selected[0]["needs_review"] is True
    assert selected[1]["plate_norm"] == "12ب34567"


def test_submit_is_non_blocking_and_drops_to_latest(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    monkeypatch.setattr(
        worker,
        "_load_config",
        lambda camera_id: {
            "id": camera_id,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 1,
            "lpr_confidence": 50,
            "duplicate_seconds": 10,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 100,
            "roi_h": 100,
        },
    )
    processed = []

    def fake_process(state, payload):
        processed.append(int(payload[2][0, 0, 0]))
        time.sleep(0.03)
        with worker._lock:
            next_payload = state.pending
            state.pending = None
            if next_payload is None:
                state.busy = False
            else:
                worker._executor.submit(
                    fake_process,
                    state,
                    next_payload,
                )

    monkeypatch.setattr(worker, "_process", fake_process)
    for value in range(5):
        worker.submit(
            1,
            "cam",
            np.full(
                (10, 10, 3),
                value,
                dtype=np.uint8,
            ),
        )
    time.sleep(0.15)
    worker.shutdown()
    assert processed[0] == 0
    assert processed[-1] == 4
    assert len(processed) <= 3


def test_slow_cpu_keeps_three_observations_for_consensus(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((90, 160, 3), 80, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.90,
        "quality_score": 0.80,
        "bbox": (30, 30, 130, 65),
        "crop": frame[30:65, 30:130].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(result)],
    )
    persisted = []

    def fake_persist(
        _camera_id,
        _camera_name,
        _frame,
        saved_result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        persisted.append((saved_result, event_id))
        return event_id or 41

    monkeypatch.setattr(worker, "_persist", fake_persist)
    clock = iter((0.0, 3.0, 3.0, 6.0, 6.0, 9.0))
    monkeypatch.setattr(
        live_worker.time,
        "perf_counter",
        lambda: next(clock),
    )

    for timestamp in (0.0, 3.0, 6.0):
        state.busy = True
        worker._process(
            state,
            (1, "CPU camera", frame.copy(), timestamp),
        )
    worker.shutdown()

    # Provisional captures stay in tracker memory; only the strict consensus
    # becomes a durable row.
    assert all(not row.get("capture_only") for row, _event_id in persisted)
    recognized = [
        row for row, _event_id in persisted
        if not row.get("capture_only")
    ]
    assert len(recognized) == 1
    assert recognized[0]["plate_norm"] == "12ب34567"
    assert persisted[-1][1] is None
    assert state.emitted_events == 1
    # Slow inference must preserve consecutive observations without leaving a
    # physical track open long enough to absorb a later vehicle.
    assert state.tracker.max_age_seconds == 6.0


def test_fragmented_continuous_plate_reuses_one_event_after_cooldown(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "confidence": 0.92,
        "ocr_confidence": 0.90,
        "quality_score": 0.82,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(detected)],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    writes = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        saved_id = int(event_id) if event_id is not None else 71
        writes.append((saved_id, event_id, dict(result)))
        return saved_id

    monkeypatch.setattr(worker, "_persist", persist)
    # The 40-second inference gap forces a new tracker id and exceeds the
    # configured cooldown, but no empty observation ever ended the visit.
    for timestamp in (0.0, 0.2, 0.4, 40.0, 40.2, 40.4):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert [event_id for _saved, event_id, _row in writes] == [None, 71]
    assert {saved for saved, _event_id, _row in writes} == {71}
    assert state.emitted_events == 1
    assert state.seen["31ط55674"] == 40.4


def test_same_plate_after_confirmed_absence_creates_new_event(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    detected = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": True,
        "confidence": 0.92,
        "ocr_confidence": 0.90,
        "quality_score": 0.82,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    outputs = iter(
        [[dict(detected)] for _ in range(3)]
        + [[] for _ in range(3)]
        + [[dict(detected)] for _ in range(3)]
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: next(outputs),
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    inserted = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        _result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        if event_id is None:
            inserted.append(80 + len(inserted))
            return inserted[-1]
        return int(event_id)

    monkeypatch.setattr(worker, "_persist", persist)
    # The return happens before the tracker's normal expiry.  Three empty
    # observations must retire the old one-shot track so a new visit can emit.
    timestamps = (0.0, 0.2, 0.4, 0.8, 1.2, 1.6, 2.0, 2.2, 2.4)
    for timestamp in timestamps:
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert inserted == [80, 81]
    assert state.emitted_events == 2


def test_provisional_capture_waits_for_one_final_row(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.92,
            "ocr_confidence": 0.90,
            "quality_score": 0.82,
            "bbox": (25, 30, 155, 68),
            "crop": frame[30:68, 25:155].copy(),
            "method": "test",
        }],
    )
    writes = []

    def persist(
        _camera_id,
        _camera_name,
        _frame,
        result,
        _processing_ms,
        event_id=None,
        _duplicate_seconds=0.0,
    ):
        writes.append((dict(result), event_id))
        return event_id or 91

    monkeypatch.setattr(worker, "_persist", persist)

    state.busy = True
    worker._process(state, (1, "Gate", frame, 0.0))
    assert writes == []

    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [],
    )
    state.busy = True
    worker._process(state, (1, "Gate", frame, 6.0))
    worker.shutdown()

    assert len(writes) == 1
    assert writes[0][0]["provisional"] is False
    assert writes[0][0]["valid"] is False
    assert writes[0][0]["needs_review"] is True
    assert writes[0][1] is None
    assert state.emitted_events == 1


def test_unknown_fragment_cannot_erase_live_review_candidate(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    candidate = {
        "plate": "31-ط-556-74",
        "plate_norm": "",
        "raw_guess_text": "31-ط-556-74",
        "raw_guess_norm": "31ط55674",
        "valid": False,
        "needs_review": True,
        "track_id": 1,
    }
    state.visits.register(
        candidate,
        77,
        0.0,
        allow_candidate=True,
    )
    state.track_event_ids[1] = 77
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    unknown = {
        "plate": "ناخوانا",
        "plate_norm": "",
        "raw_guess_text": "",
        "raw_guess_norm": "",
        "valid": False,
        "needs_review": True,
        "confidence": 0.35,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "quality_score": 0.66,
        "bbox": (25, 30, 155, 68),
        "crop": frame[30:68, 25:155].copy(),
        "method": "test",
    }
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [dict(unknown)],
    )
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unknown fragment must not downgrade candidate")
        ),
    )

    for timestamp in (0.0, 0.4, 0.8):
        state.busy = True
        worker._process(
            state,
            (1, "Gate", frame.copy(), timestamp),
        )
    worker.shutdown()

    assert state.visits.event_refs == {"31ط55674": 77}
    assert state.emitted_events == 0


def test_latest_detection_is_available_for_live_overlay(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[4] = state
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: [{
            "plate": "12-ب-345-67",
            "plate_norm": "12ب34567",
            "valid": True,
            "confidence": 0.91,
            "quality_score": 0.8,
            "bbox": (20, 25, 130, 60),
            "crop": frame[25:60, 20:130],
            "method": "test",
        }],
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    monkeypatch.setattr(worker, "_persist", lambda *_args: 1)
    state.busy = True
    worker._process(state, (4, "cam", frame, time.monotonic()))
    detections = worker.detections(4)
    worker.shutdown()

    assert detections[0]["bbox"] == (20, 25, 130, 60)
    assert detections[0]["plate"] == "12-ب-345-67"


def test_submit_adaptively_spaces_slow_cpu_inference(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(
        config={
            "id": 9,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 1,
        },
        config_loaded_at=100.0,
        processing_seconds_ema=2.0,
    )
    worker._states[9] = state
    submitted = []
    monkeypatch.setattr(
        worker,
        "_config",
        lambda _camera_id, current, _now: current.config,
    )
    monkeypatch.setattr(
        worker._executor,
        "submit",
        lambda _callback, _state, payload: submitted.append(payload),
    )
    times = iter((100.0, 100.4, 100.89, 100.91))
    monkeypatch.setattr(
        live_worker.time,
        "monotonic",
        lambda: next(times),
    )
    frame = np.zeros((20, 40, 3), dtype=np.uint8)

    worker.submit(9, "cam", frame)
    state.busy = False
    worker.submit(9, "cam", frame)
    worker.submit(9, "cam", frame)
    worker.submit(9, "cam", frame)
    worker.shutdown()

    # EMA=2s yields a 0.9s minimum interval. Frames inside that interval are
    # skipped, then the newest eligible frame is submitted.
    assert len(submitted) == 2


def test_every_received_frame_can_improve_pending_ocr_selection(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState(
        busy=True,
        config={
            "id": 12,
            "enabled": 1,
            "lpr_enabled": 1,
            "frame_step": 999,
        },
        config_loaded_at=200.0,
    )
    worker._states[12] = state
    monkeypatch.setattr(
        worker,
        "_config",
        lambda _camera_id, current, _now: current.config,
    )
    monkeypatch.setattr(
        worker,
        "_selection_score",
        lambda frame, _config: float(frame[0, 0, 0]),
    )
    times = iter((200.0, 200.1, 200.2))
    monkeypatch.setattr(
        live_worker.time,
        "monotonic",
        lambda: next(times),
    )

    for value in (20, 200, 80):
        worker.submit(
            12,
            "quality camera",
            np.full((20, 40, 3), value, dtype=np.uint8),
        )
    selected = int(state.pending[2][0, 0, 0])
    worker.shutdown()

    assert state.frame_counter == 3
    assert selected == 200
# RC7-RC9 regression coverage for adaptive live-frame processing.


def test_empty_inference_enters_backoff_and_clears_overlay(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[15] = state
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    detected = {
        "plate": "در حال بررسی",
        "plate_norm": "",
        "valid": False,
        "confidence": 0.51,
        "detector_confidence": 0.82,
        "ocr_confidence": 0.0,
        "quality_score": 0.7,
        "bbox": (20, 25, 130, 60),
        "crop": frame[25:60, 20:130],
        "method": "test",
        "whole_plate_ocr_attempted": True,
        "ocr_engine": "crnn-onnx",
        "ocr_alternative": "31-ط-556-74",
        "ocr_disagreement": True,
    }
    outputs = iter(([detected], [], [], []))
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: list(next(outputs)),
    )
    monkeypatch.setattr(
        live_worker,
        "apply_learned_correction",
        lambda result: result,
    )
    monkeypatch.setattr(worker, "_persist", lambda *_args: 1)

    for timestamp in (0.0, 1.0, 2.0, 3.0):
        state.busy = True
        worker._process(
            state,
            (15, "idle camera", frame.copy(), timestamp),
        )

    snapshot = worker.detection_snapshot(
        15,
        after_revision=3,
    )
    status = worker.status(15)
    worker.shutdown()

    assert state.detection_revision == 4
    assert snapshot["revision"] == 4
    assert snapshot["detections"] == []
    assert status["idle_mode"] is True
    assert status["no_plate_streak"] == 3
    assert status["next_inference_seconds"] >= 1.0
    assert status["ocr_ab"] == {
        "whole_plate_attempts": 1,
        "agreements": 0,
        "disagreements": 1,
        "crnn_selected": 1,
        "character_reader_selected": 0,
    }


def test_no_plate_backoff_grows_but_recognition_stays_responsive():
    delay = live_worker.LiveANPRWorker._post_inference_delay

    assert delay(0.25, 0) == 0.20
    assert delay(0.25, 1) == 0.40
    assert delay(0.25, 2) == 0.80
    assert delay(0.25, 3) == 1.60
    assert delay(0.25, 4) == 3.20
    assert delay(0.25, 10) == 3.20


def test_motion_wakes_camera_during_long_empty_scene_backoff(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "id": 27,
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 0,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.config_loaded_at = time.monotonic()
    state.next_inference_at = time.monotonic() + 30.0
    worker._states[27] = state
    empty = np.zeros((120, 240, 3), dtype=np.uint8)
    state.activity.observe(empty)
    entering = empty.copy()
    entering[30:100, 70:190] = 220
    processed = threading.Event()

    def record_process(current_state, payload):
        assert payload[5].wake_inference is True
        processed.set()
        with worker._lock:
            current_state.busy = False

    monkeypatch.setattr(worker, "_process", record_process)

    worker.submit(27, "entry camera", entering)
    assert processed.wait(0.5)
    assert state.motion_wakeups == 1
    assert state.burst_frames_remaining >= 4
    worker.shutdown()


def test_two_cameras_receive_independent_worker_slots(monkeypatch):
    monkeypatch.setattr(
        live_worker,
        "parallel_camera_limit",
        lambda: 2,
    )
    worker = live_worker.LiveANPRWorker()
    monkeypatch.setattr(
        worker,
        "_setting",
        lambda key, default="": (
            "yolo8n" if key == "anpr_detector_model" else default
        ),
    )
    monkeypatch.setattr(
        worker,
        "_load_config",
        lambda camera_id: {
            "id": camera_id,
            "rtsp_url": f"rtsp://camera/{camera_id}",
            "enabled": 1,
            "lpr_enabled": 1,
            "lpr_confidence": 50,
            "duplicate_seconds": 0,
            "roi_x": 0,
            "roi_y": 0,
            "roi_w": 100,
            "roi_h": 100,
        },
    )
    active = 0
    maximum_active = 0
    engine_keys = []
    detector_variants = []
    active_lock = threading.Lock()

    def process(_frame, _confidence, engine_key=None, **kwargs):
        nonlocal active, maximum_active
        with active_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            engine_keys.append(engine_key)
            detector_variants.append(kwargs.get("detector_variant"))
        time.sleep(0.05)
        with active_lock:
            active -= 1
        return []

    monkeypatch.setattr(live_worker, "process_frame", process)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)

    worker.submit(1, "gate one", frame)
    worker.submit(2, "gate two", frame)
    deadline = time.monotonic() + 2.0
    while True:
        first = worker.status(1)
        second = worker.status(2)
        if (
            first["processed_frames"] == 1
            and second["processed_frames"] == 1
        ):
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.01)
    worker.shutdown()

    assert maximum_active == 2
    assert sorted(engine_keys) == [1, 2]
    assert detector_variants == ["yolov8n", "yolov8n"]
    assert first["threads_per_camera"] == 2
    assert first["parallel_camera_limit"] == 2
    assert first["anpr_engine"] == {
        "mode": "primary-v2",
        "detector_variant": "yolov8n",
        "exclusive_detector": True,
        "candidate_inference": False,
        "baseline_fallback": True,
    }
    assert first["shadow"]["enabled"] is True
    assert second["processed_frames"] == 1


def test_detector_selection_cache_can_be_invalidated(monkeypatch):
    from app.ai import onnx_detector

    cleared = []
    monkeypatch.setattr(
        onnx_detector,
        "clear_detector_sessions",
        lambda: cleared.append(True),
    )
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {"duplicate_seconds": 27}
    observation = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "quality_score": 0.82,
        "bbox": (20, 20, 140, 55),
        "crop": np.zeros((35, 120, 3), dtype=np.uint8),
    }
    state.tracker.update([observation], timestamp=0.0)
    state.tracker.update([observation], timestamp=0.2)
    old_tracker = state.tracker
    state.visits.register(
        {**observation, "track_id": 3},
        41,
        10.0,
    )
    state.track_event_ids[3] = 41
    state.latest_detections = [{"plate": "31-ط-556-74"}]
    state.processed_frames = 12
    state.detected_candidates = 5
    state.emitted_events = 2
    state.frame_counter = 40
    worker._states[7] = state
    worker._model_state = {"detector_ready": True}
    worker._model_state_at = 123.0
    worker._model_state_variant = "yolov8n"

    worker.invalidate_model_cache()
    worker.shutdown()

    assert worker._model_state == {}
    assert worker._model_state_at == 0.0
    assert worker._model_state_variant == ""
    assert cleared == [True]
    assert state.tracker is not old_tracker
    assert state.tracker.emit_cooldown == 27
    assert state.tracker.update([observation], timestamp=0.4) == []
    # Exact durable visit identity survives a detector switch, while all
    # model-specific track bindings are reset.
    assert state.seen == {"31ط55674": 10.0}
    assert state.visits.event_refs == {"31ط55674": 41}
    assert state.visits.track_keys == {}
    assert state.track_event_ids == {}
    assert state.latest_detections == []
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.emitted_events == 0
    assert state.frame_counter == 0


def test_inflight_old_detector_result_is_discarded_on_switch(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[9] = state
    frame = np.zeros((100, 180, 3), dtype=np.uint8)
    stale = {
        "plate": "31-ط-556-74",
        "plate_norm": "31ط55674",
        "valid": True,
        "confidence": 0.91,
        "quality_score": 0.82,
        "bbox": (20, 20, 140, 55),
        "crop": frame[20:55, 20:140].copy(),
    }

    def switch_during_inference(*_args, **_kwargs):
        worker.invalidate_model_cache()
        return [stale]

    monkeypatch.setattr(live_worker, "process_frame", switch_during_inference)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("stale detector result must not be persisted")
        ),
    )

    worker._process(
        state,
        (9, "gate", frame, 1.0, 1.0, None, 0),
    )
    worker.shutdown()

    assert state.busy is False
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.tracker.active_track_ids() == set()
    assert state.seen == {}
    assert state.latest_detections == []


def test_removed_camera_discards_inflight_result_before_persistence(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 30,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[44] = state
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    entered = threading.Event()
    release = threading.Event()

    def process(*_args, **_kwargs):
        entered.set()
        assert release.wait(2.0)
        return [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "valid": True,
            "confidence": 0.92,
            "quality_score": 0.82,
            "bbox": (25, 30, 155, 68),
            "crop": frame[30:68, 25:155].copy(),
            "method": "test",
        }]

    monkeypatch.setattr(live_worker, "process_frame", process)
    monkeypatch.setattr(
        worker,
        "_persist",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("a retired camera must not persist")
        ),
    )
    thread = threading.Thread(
        target=worker._process,
        args=(state, (44, "Gate", frame, 1.0)),
    )
    thread.start()
    assert entered.wait(1.0)

    worker.remove(44)
    release.set()
    thread.join(timeout=2.0)
    worker.shutdown()

    assert not thread.is_alive()
    assert state.retired is True
    assert state.busy is False
    assert 44 not in worker._states


def test_selected_inference_failure_reaches_camera_last_error(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    state.busy = True
    worker._states[12] = state
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )

    def fail(*_args, **kwargs):
        assert kwargs["engine_key"] == 12
        assert kwargs["detector_variant"] == "yolov8n"
        raise RuntimeError("selected YOLO inference failed")

    monkeypatch.setattr(live_worker, "process_frame", fail)

    worker._process(
        state,
        (12, "gate", np.zeros((100, 180, 3), dtype=np.uint8), 1.0),
    )
    worker.shutdown()

    assert state.busy is False
    assert state.processed_frames == 0
    assert state.detected_candidates == 0
    assert state.last_error == "RuntimeError: selected YOLO inference failed"


def test_launcher_preparation_transition_invalidates_model_status_cache(
    monkeypatch,
):
    from app.ai import model_manager

    worker = live_worker.LiveANPRWorker(max_workers=1)
    worker._model_state = {
        "selected_detector": "yolo11n",
        "detector_ready": True,
        "hezar_ready": True,
        "preparation_state": "",
        "preparation_error": "",
    }
    worker._model_state_at = time.monotonic()
    worker._model_state_variant = "yolo11n"
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolo11n",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_STATE_ENV,
        "error",
    )
    monkeypatch.setenv(
        model_manager.MODEL_PREPARATION_ERROR_ENV,
        "ValueError: model hash mismatch",
    )
    calls = []

    def status(selected_detector=None):
        calls.append(selected_detector)
        return {
            "selected_detector": selected_detector,
            "detector_ready": True,
            "hezar_ready": True,
            "preparation_state": "error",
            "preparation_error": "ValueError: model hash mismatch",
        }

    monkeypatch.setattr(model_manager, "model_status", status)

    current = worker._models()
    worker.shutdown()

    assert calls == ["yolo11n"]
    assert current["ready"] is True
    assert current["preparation_state"] == "error"
    assert current["preparation_error"] == "ValueError: model hash mismatch"


def test_video_pass_drain_promotes_worker_pending_frame(monkeypatch):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[31] = state
    token = worker.begin_video_pass(31)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    state.pending = (
        31,
        "uploaded video",
        frame,
        1.0,
        1.0,
        None,
        token["detector_generation"],
    )
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )
    monkeypatch.setattr(live_worker, "process_frame", lambda *_a, **_k: [])

    drained = worker.drain_video_pass(31, token, timeout=1.0)
    worker.shutdown()

    assert drained["ok"] is True
    assert drained["error"] == ""
    assert drained["processed_frames"] == 1
    assert state.pending is None
    assert state.busy is False


def test_video_pass_drain_remembers_error_cleared_by_later_success(
    monkeypatch,
):
    worker = live_worker.LiveANPRWorker(max_workers=1)
    state = live_worker._CameraState()
    state.config = {
        "enabled": 1,
        "lpr_enabled": 1,
        "lpr_confidence": 50,
        "duplicate_seconds": 20,
        "roi_x": 0,
        "roi_y": 0,
        "roi_w": 100,
        "roi_h": 100,
    }
    worker._states[32] = state
    token = worker.begin_video_pass(32)
    frame = np.zeros((80, 160, 3), dtype=np.uint8)
    monkeypatch.setattr(
        worker,
        "_selected_detector_variant",
        lambda: "yolov8n",
    )
    monkeypatch.setattr(
        live_worker,
        "process_frame",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("selected YOLO failed once")
        ),
    )
    state.busy = True
    worker._process(state, (32, "video", frame, 1.0))
    assert state.processing_errors == 1
    assert state.last_error == "RuntimeError: selected YOLO failed once"

    monkeypatch.setattr(live_worker, "process_frame", lambda *_a, **_k: [])
    state.busy = True
    worker._process(state, (32, "video", frame, 2.0))
    assert state.last_error == ""

    drained = worker.drain_video_pass(32, token, timeout=1.0)
    worker.shutdown()

    assert drained["ok"] is False
    assert drained["error"] == "RuntimeError: selected YOLO failed once"
