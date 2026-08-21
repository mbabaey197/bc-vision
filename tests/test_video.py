from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import app.ai.video_test as video_test
from app.ai import model_manager
import app.media_storage as media_storage
import app.streams as streams
from app.streams import CameraStream


def _write_video(path: Path, frames=8):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (320, 180),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((180, 320, 3), 30, dtype=np.uint8)
        cv2.rectangle(
            frame,
            (80 + index, 80),
            (240 + index, 120),
            (235, 235, 235),
            -1,
        )
        writer.write(frame)
    writer.release()


def test_video_emits_one_consensus_event(tmp_path, monkeypatch):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path)
    calls = {"count": 0}
    variants = []

    def fake_process(frame, threshold, detector_variant=None):
        calls["count"] += 1
        variants.append(detector_variant)
        plate = (
            "12-ب-345-67"
            if calls["count"] != 2
            else "12-ب-345-76"
        )
        confidence = 0.74 if calls["count"] != 2 else 0.55
        return [{
            "plate": plate,
            "plate_norm": plate.replace("-", ""),
            "valid": True,
            "confidence": confidence,
            "detector_confidence": 0.8,
            "ocr_confidence": 0.7,
            "quality_score": 0.8,
            "bbox": (80, 80, 240, 120),
            "crop": None,
            "method": "test",
            "vehicle_type": "سواری",
            "vehicle_color": "سفید",
            "vehicle_brand": "نامشخص",
            "vehicle_confidence": 0.5,
            "vehicle_bbox": (30, 30, 290, 160),
        }]

    monkeypatch.setattr(
        video_test,
        "process_frame",
        fake_process,
    )
    info, events = video_test.process_video(
        video_path,
        tmp_path / "پلاک‌ها",
        tmp_path / "خودروها",
        frame_step=1,
        duplicate_seconds=20,
        min_confidence=0.5,
        detector_variant="yolo8n",
    )
    assert info["frames"] >= 8
    assert info["detector_variant"] == "yolov8n"
    assert set(variants) == {"yolov8n"}
    assert len(events) == 1
    assert events[0]["plate"] == "12-ب-345-67"
    assert events[0]["consensus_votes"] >= 2
    assert events[0]["media_status"] == "complete"
    assert events[0]["media_error"] == ""
    for key in ("plate_path", "image_path"):
        image_path = Path(events[0][key])
        payload = image_path.read_bytes()
        assert len(payload) > 0
        decoded = cv2.imdecode(
            np.frombuffer(payload, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None
        assert decoded.size > 0


def test_video_track_fragment_keeps_one_event_when_cooldown_is_zero(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "fragmented.avi"
    _write_video(video_path, frames=8)
    calls = {"count": 0}

    def fake_process(_frame, _threshold, detector_variant=None):
        calls["count"] += 1
        # The large position jump starts a new visual track even though the
        # exact same plate remains the active visit.
        bbox = (
            (20, 70, 140, 110)
            if calls["count"] <= 3
            else (180, 70, 300, 110)
        )
        return [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.91,
            "detector_confidence": 0.90,
            "ocr_confidence": 0.89,
            "quality_score": 0.82,
            "bbox": bbox,
            "crop": None,
            "method": "test",
        }]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        duplicate_seconds=0,
        min_confidence=0.5,
        detector_variant="yolo8n",
    )

    assert len(events) == 1
    assert events[0]["plate_norm"] == "31ط55674"


def test_video_short_unreadable_track_survives_mid_video_expiry(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "short-unreadable.avi"
    _write_video(video_path, frames=40)
    calls = {"count": 0}

    def fake_process(_frame, _threshold, detector_variant=None):
        calls["count"] += 1
        if calls["count"] > 2:
            return []
        return [{
            "plate": "ناخوانا",
            "plate_norm": "",
            "valid": False,
            "confidence": 0.42,
            "detector_confidence": 0.84,
            "ocr_confidence": 0.0,
            "quality_score": 0.72,
            "bbox": (80, 75, 240, 118),
            "crop": None,
            "method": "test",
        }]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        duplicate_seconds=0,
        detector_variant="yolo8n",
    )

    assert len(events) == 1
    assert events[0]["capture_only"] is True
    assert events[0]["provisional"] is False
    assert events[0]["needs_review"] is True


def test_video_unreadable_fragment_does_not_split_recognized_visit(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "ocr-gap.avi"
    _write_video(video_path, frames=18)
    calls = {"count": 0}

    def fake_process(_frame, _threshold, detector_variant=None):
        calls["count"] += 1
        if 4 <= calls["count"] <= 13:
            return [{
                "plate": "ناخوانا",
                "plate_norm": "",
                "valid": False,
                "confidence": 0.40,
                "detector_confidence": 0.82,
                "ocr_confidence": 0.0,
                "quality_score": 0.68,
                "bbox": (180, 70, 300, 110),
                "crop": None,
                "method": "test",
            }]
        bbox = (
            (20, 70, 140, 110)
            if calls["count"] <= 3
            else (180, 70, 300, 110)
        )
        return [{
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.91,
            "detector_confidence": 0.90,
            "ocr_confidence": 0.89,
            "quality_score": 0.82,
            "bbox": bbox,
            "crop": None,
            "method": "test",
        }]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        duplicate_seconds=0,
        detector_variant="yolo8n",
    )

    assert len(events) == 1
    assert events[0]["plate_norm"] == "31ط55674"
    assert events[0]["valid"] is True


def test_video_distinct_review_candidate_is_not_merged_with_known_plate(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "distinct-review-candidate.avi"
    _write_video(video_path, frames=13)
    calls = {"count": 0}

    def fake_process(_frame, _threshold, detector_variant=None):
        calls["count"] += 1
        if calls["count"] <= 3:
            return [{
                "plate": "31-ط-556-74",
                "plate_norm": "31ط55674",
                "raw_guess_text": "31-ط-556-74",
                "raw_guess_norm": "31ط55674",
                "valid": True,
                "confidence": 0.91,
                "detector_confidence": 0.90,
                "ocr_confidence": 0.89,
                "quality_score": 0.82,
                "bbox": (20, 70, 140, 110),
                "crop": None,
                "method": "test",
            }]
        return [{
            "plate": "ناخوانا",
            "plate_norm": "",
            "raw_guess_text": "12-ب-345-67",
            "raw_guess_norm": "12ب34567",
            "valid": False,
            "needs_review": True,
            "confidence": 0.45,
            "detector_confidence": 0.86,
            "ocr_confidence": 0.48,
            "quality_score": 0.70,
            "bbox": (180, 70, 300, 110),
            "crop": None,
            "method": "test",
        }]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        duplicate_seconds=0,
        min_confidence=0.5,
        detector_variant="yolo8n",
    )

    assert len(events) == 2
    assert events[0]["plate_norm"] == "31ط55674"
    assert events[1]["raw_guess_norm"] == "12ب34567"
    assert events[1]["needs_review"] is True


def test_video_unknown_fragment_cannot_erase_review_candidate(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "candidate-then-unknown.avi"
    _write_video(video_path, frames=20)
    calls = {"count": 0}

    def fake_process(_frame, _threshold, detector_variant=None):
        calls["count"] += 1
        if calls["count"] <= 10:
            return [{
                "plate": "31-ط-556-74",
                "plate_norm": "",
                "raw_guess_text": "31-ط-556-74",
                "raw_guess_norm": "31ط55674",
                "valid": False,
                "needs_review": True,
                "confidence": 0.45,
                "detector_confidence": 0.86,
                "ocr_confidence": 0.48,
                "quality_score": 0.70,
                "bbox": (20, 70, 140, 110),
                "crop": None,
                "method": "test",
            }]
        return [{
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
            "bbox": (180, 70, 300, 110),
            "crop": None,
            "method": "test",
        }]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        duplicate_seconds=0,
        detector_variant="yolo8n",
    )

    assert len(events) == 1
    assert events[0]["raw_guess_norm"] == "31ط55674"
    assert events[0]["needs_review"] is True


def test_video_max_events_caps_final_unreadable_rows(tmp_path, monkeypatch):
    video_path = tmp_path / "unreadable-cap.avi"
    _write_video(video_path, frames=1)

    def fake_process(_frame, _threshold, detector_variant=None):
        return [
            {
                "plate": "ناخوانا",
                "plate_norm": "",
                "valid": False,
                "confidence": 0.40,
                "detector_confidence": 0.82,
                "ocr_confidence": 0.0,
                "quality_score": 0.68,
                "bbox": (20 + index * 90, 70, 90 + index * 90, 105),
                "crop": None,
                "method": "test",
            }
            for index in range(3)
        ]

    monkeypatch.setattr(video_test, "process_frame", fake_process)

    _info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "vehicles",
        frame_step=1,
        max_events=1,
        duplicate_seconds=0,
        detector_variant="yolo8n",
    )

    assert len(events) == 1


def test_video_media_failure_keeps_result_and_reports_error(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        media_storage.cv2,
        "imencode",
        lambda *_args, **_kwargs: (False, None),
    )
    frame = np.full((100, 180, 3), 120, dtype=np.uint8)
    result = {
        "plate": "12-ب-345-67",
        "plate_norm": "12ب34567",
        "valid": True,
        "confidence": 0.9,
        "bbox": (40, 50, 140, 82),
        "crop": frame[50:82, 40:140].copy(),
    }

    event = video_test._save_event(
        result,
        frame,
        frame_no=10,
        fps=10.0,
        plate_dir=tmp_path / "plates",
        snapshot_dir=tmp_path / "snapshots",
        video_path=tmp_path / "source.mp4",
    )

    assert event["plate"] == "12-ب-345-67"
    assert event["plate_path"] == ""
    assert event["image_path"] == ""
    assert event["media_status"] == "error"
    assert "plate: JPEG encoder returned no data" in event["media_error"]
    assert "vehicle: JPEG encoder returned no data" in event["media_error"]


def test_video_shadow_request_is_disabled_for_exclusive_selection(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path, frames=3)
    monkeypatch.setattr(
        video_test,
        "process_frame",
        lambda *_args, **_kwargs: [],
    )

    info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "snapshots",
        frame_step=1,
        include_candidate_shadow=True,
    )

    assert events == []
    assert info["candidate_shadow_requested"] is True
    assert info["candidate_shadow_enabled"] is False
    assert info["exclusive_detector"] is True
    assert info["detector_execution_mode"] == "exclusive-baseline"
    assert "انحصاری" in info["candidate_shadow_error"]


def test_video_selected_inference_failure_is_not_treated_as_no_plate(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "failure.avi"
    _write_video(video_path, frames=2)

    def fail(*_args, **kwargs):
        assert kwargs["detector_variant"] == "yolov8n"
        raise RuntimeError("YOLOv8n inference failed")

    monkeypatch.setattr(video_test, "process_frame", fail)

    with pytest.raises(RuntimeError, match="YOLOv8n inference failed"):
        video_test.process_video(
            video_path,
            tmp_path / "plates",
            tmp_path / "snapshots",
            frame_step=1,
            detector_variant="yolov8n",
        )


def test_video_fails_closed_if_detector_revision_changes_mid_pass(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "revision-change.avi"
    _write_video(video_path, frames=2)
    revisions = iter(("revision-a", "revision-b"))
    monkeypatch.setattr(
        model_manager,
        "yolox_detector_spec",
        lambda: {
            "ready": True,
            "model_revision": "revision-a",
            "error": "",
        },
    )

    def rows(frame, *_args, **kwargs):
        revision = next(revisions)
        assert kwargs["expected_detector_revision"] == "revision-a"
        kwargs["runtime_metadata"].update(
            detector_variant="yolox",
            detector_model_revision=revision,
        )
        if revision == "revision-b":
            return []
        return [{
            "plate": "ناخوانا",
            "plate_norm": "",
            "valid": False,
            "confidence": 0.8,
            "quality_score": 0.5,
            "bbox": (80, 80, 240, 120),
            "crop": frame[80:120, 80:240].copy(),
            "method": "yolox-custom-onnx",
            "detector_model_revision": revision,
        }]

    monkeypatch.setattr(video_test, "process_frame", rows)

    with pytest.raises(RuntimeError, match="revision changed"):
        video_test.process_video(
            video_path,
            tmp_path / "plates",
            tmp_path / "snapshots",
            frame_step=1,
            detector_variant="yolox",
        )


def test_video_shadow_request_cannot_affect_selected_baseline_events(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "sample.avi"
    _write_video(video_path, frames=6)

    def selected_baseline(frame, *_args, **_kwargs):
        return {
            "plate": "31-ط-556-74",
            "plate_norm": "31ط55674",
            "raw_guess_text": "31-ط-556-74",
            "raw_guess_norm": "31ط55674",
            "valid": True,
            "confidence": 0.88,
            "detector_confidence": 0.90,
            "ocr_confidence": 0.86,
            "quality_score": 0.80,
            "bbox": (80, 80, 240, 120),
            "crop": frame[80:120, 80:240].copy(),
            "method": "yolov8n-plate-onnx",
            "ocr_engine": "hezar-crnn-fa-v2-onnx",
        }

    monkeypatch.setattr(
        video_test,
        "process_frame",
        lambda frame, *_args, **kwargs: [
            selected_baseline(frame, **kwargs)
        ],
    )

    info, events = video_test.process_video(
        video_path,
        tmp_path / "plates",
        tmp_path / "snapshots",
        frame_step=1,
        include_candidate_shadow=True,
        detector_variant="yolov8n",
    )

    assert len(events) == 1
    assert events[0]["plate_norm"] == "31ط55674"
    assert events[0]["engine_lane"] == "baseline"
    assert events[0]["detector_variant"] == "yolov8n"
    assert events[0]["detector_selection_exclusive"] is True
    assert not events[0].get("experimental")
    assert info["candidate_shadow_enabled"] is False


def test_uploaded_video_stream_pauses_at_end_for_stable_event_count(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "loop.avi"
    _write_video(video_path, frames=3)
    stream = CameraStream(
        91,
        f"video://{video_path}",
        "Uploaded video",
        fps=30,
    )
    published = []
    monkeypatch.setattr(stream, "_publish", lambda frame: published.append(frame))

    stream.start()
    for _ in range(50):
        if stream.state.ended:
            break
        time.sleep(0.02)

    assert len(published) == 3
    assert stream.state.ended is True
    assert stream.state.paused is True

    # Replaying is explicit and the decoder continues producing preview frames.
    # ANPR replay suppression is covered separately with the real publish path.
    assert stream.resume() is True
    for _ in range(50):
        if len(published) > 3:
            break
        time.sleep(0.02)
    stream.stop()
    stream.thread.join(timeout=2)

    assert len(published) > 3
    assert not stream.thread.is_alive()


def test_completed_uploaded_video_replays_preview_without_anpr_submission(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "one-anpr-pass.avi"
    _write_video(video_path, frames=4)
    marker_calls = []
    submitted = []
    stream = CameraStream(
        191,
        f"video://{video_path}",
        "One ANPR pass",
        fps=30,
        video_anpr_state_callback=(
            lambda camera_id, state: marker_calls.append(
                (camera_id, state)
            )
        ),
    )
    published = []
    original_publish = stream._publish

    def publish(frame):
        published.append(frame.copy())
        original_publish(frame)

    monkeypatch.setattr(stream, "_publish", publish)
    monkeypatch.setattr(stream, "_encode", lambda _frame: b"jpeg")
    monkeypatch.setattr(
        "app.ai.live_worker.submit_live_frame",
        lambda camera_id, name, frame: submitted.append(
            (camera_id, name, frame.copy())
        ),
    )
    monkeypatch.setattr(
        "app.ai.live_worker.drain_live_video_pass",
        lambda *_args, **_kwargs: {"ok": True, "error": ""},
    )

    try:
        stream.start()
        for _ in range(300):
            if stream.state.ended:
                break
            time.sleep(0.01)
        assert stream.state.ended is True
        time.sleep(0.15)
        first_preview_frames = len(published)
        first_anpr_frames = len(submitted)

        assert first_preview_frames == 4
        assert first_anpr_frames >= 1
        assert marker_calls == [
            (191, "started"),
            (191, "completed"),
        ]
        assert stream.state.anpr_preview_only is True
        assert stream.state.anpr_completed is True
        assert stream.state.anpr_interrupted is False

        assert stream.resume() is True
        for _ in range(300):
            if (
                stream.state.ended
                and len(published) > first_preview_frames
            ):
                break
            time.sleep(0.01)
        assert stream.state.ended is True
        time.sleep(0.15)

        assert len(published) == first_preview_frames + 4
        assert len(submitted) == first_anpr_frames
        assert marker_calls == [
            (191, "started"),
            (191, "completed"),
        ]
    finally:
        stream.stop()
        if stream.thread:
            stream.thread.join(timeout=2)
        if stream._anpr_thread:
            stream._anpr_thread.join(timeout=2)


def test_uploaded_video_failure_at_eof_stays_incomplete_and_fail_closed(
    tmp_path,
    monkeypatch,
):
    video_path = tmp_path / "failed-anpr-pass.avi"
    _write_video(video_path, frames=3)
    marker_calls = []
    submitted = []
    stream = CameraStream(
        193,
        f"video://{video_path}",
        "Failed ANPR pass",
        fps=30,
        video_anpr_state_callback=(
            lambda camera_id, state: marker_calls.append(
                (camera_id, state)
            )
        ),
    )
    monkeypatch.setattr(stream, "_encode", lambda _frame: b"jpeg")
    monkeypatch.setattr(
        "app.ai.live_worker.submit_live_frame",
        lambda *_args, **_kwargs: submitted.append(True),
    )
    monkeypatch.setattr(
        "app.ai.live_worker.drain_live_video_pass",
        lambda *_args, **_kwargs: {
            "ok": False,
            "error": "RuntimeError: selected YOLO inference failed",
        },
    )

    try:
        stream.start()
        for _ in range(300):
            if stream.state.ended:
                break
            time.sleep(0.01)

        assert stream.state.ended is True
        assert submitted
        assert marker_calls == [(193, "started")]
        assert stream.state.anpr_completed is False
        assert stream.state.anpr_preview_only is True
        assert stream.state.anpr_interrupted is True
        assert (
            stream.state.anpr_marker_error
            == "RuntimeError: selected YOLO inference failed"
        )

        first_submissions = len(submitted)
        assert stream.resume() is True
        for _ in range(300):
            if stream.state.ended:
                break
            time.sleep(0.01)
        time.sleep(0.05)

        assert len(submitted) == first_submissions
        assert marker_calls == [(193, "started")]
    finally:
        stream.stop()
        if stream.thread:
            stream.thread.join(timeout=2)
        if stream._anpr_thread:
            stream._anpr_thread.join(timeout=2)


def test_stream_stop_with_local_pending_frame_cannot_complete_marker(
    monkeypatch,
):
    marker_calls = []
    stream = CameraStream(
        194,
        "video:///tmp/pending-stop.avi",
        "Pending stop",
        video_anpr_state_callback=(
            lambda camera_id, state: marker_calls.append(
                (camera_id, state)
            )
        ),
    )
    assert stream._ensure_video_anpr_started() is True
    with stream._anpr_condition:
        stream._anpr_pending_frame = np.zeros(
            (24, 32, 3),
            dtype=np.uint8,
        )
    monkeypatch.setattr(
        "app.ai.live_worker.drain_live_video_pass",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("aborted local handoff must not reach worker drain")
        ),
    )

    stream.stop()
    stream._complete_video_anpr_pass()

    assert marker_calls == [(194, "started")]
    assert stream.state.anpr_completed is False
    assert stream.state.anpr_preview_only is True
    assert stream.state.anpr_interrupted is True
    assert "aborted by stream stop" in stream.state.anpr_marker_error


def test_stream_stop_during_worker_drain_cannot_complete_marker(
    monkeypatch,
):
    marker_calls = []
    drain_entered = threading.Event()
    allow_drain_return = threading.Event()
    stream = CameraStream(
        195,
        "video:///tmp/drain-stop.avi",
        "Drain stop",
        video_anpr_state_callback=(
            lambda camera_id, state: marker_calls.append(
                (camera_id, state)
            )
        ),
    )
    assert stream._ensure_video_anpr_started() is True

    def drain(*_args, **_kwargs):
        drain_entered.set()
        assert allow_drain_return.wait(2.0)
        return {"ok": True, "error": ""}

    monkeypatch.setattr(
        "app.ai.live_worker.drain_live_video_pass",
        drain,
    )
    completion = threading.Thread(
        target=stream._complete_video_anpr_pass,
    )
    completion.start()
    assert drain_entered.wait(1.0)

    stream.stop()
    allow_drain_return.set()
    completion.join(timeout=2.0)

    assert not completion.is_alive()
    assert marker_calls == [(195, "started")]
    assert stream.state.anpr_completed is False
    assert stream.state.anpr_preview_only is True
    assert "aborted by stream stop" in stream.state.anpr_marker_error


def test_stream_stop_cannot_interleave_with_completion_callback(
    monkeypatch,
):
    marker_calls = []
    callback_entered = threading.Event()
    allow_callback_return = threading.Event()

    def marker_callback(camera_id, state):
        marker_calls.append((camera_id, state))
        if state == "completed":
            callback_entered.set()
            assert allow_callback_return.wait(2.0)

    stream = CameraStream(
        196,
        "video:///tmp/callback-stop.avi",
        "Callback stop",
        video_anpr_state_callback=marker_callback,
    )
    assert stream._ensure_video_anpr_started() is True
    monkeypatch.setattr(
        "app.ai.live_worker.drain_live_video_pass",
        lambda *_args, **_kwargs: {"ok": True, "error": ""},
    )
    completion = threading.Thread(
        target=stream._complete_video_anpr_pass,
    )
    completion.start()
    assert callback_entered.wait(1.0)

    stopped = threading.Event()
    stopping = threading.Thread(
        target=lambda: (stream.stop(), stopped.set()),
    )
    stopping.start()
    # Completion already owns the condition lock, so stop cannot latch an
    # abort between the durable callback and the in-memory state commit.
    assert not stopped.wait(0.10)
    allow_callback_return.set()
    completion.join(timeout=2.0)
    stopping.join(timeout=2.0)

    assert not completion.is_alive()
    assert not stopping.is_alive()
    assert stopped.is_set()
    assert marker_calls == [(196, "started"), (196, "completed")]
    assert stream.state.anpr_completed is True
    assert stream.state.anpr_preview_only is True
    assert stream.state.anpr_interrupted is False
    assert stream.state.anpr_marker_error == ""


def test_decoder_reopen_after_cooldown_is_preview_only_for_anpr(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "decoder-reopen.avi"
    source.write_bytes(b"decoder fixture")
    frame = np.full((24, 32, 3), 90, dtype=np.uint8)
    captures = []

    class ReopeningCapture:
        def __init__(self, pass_number):
            self.pass_number = pass_number
            self.reads = 0
            self.released = False

        def isOpened(self):
            return True

        def set(self, *_args):
            return True

        def get(self, _property):
            return 120.0

        def read(self):
            self.reads += 1
            if self.pass_number == 1:
                if self.reads == 1:
                    return True, frame.copy()
                raise RuntimeError("decoder crashed after a published frame")
            if self.reads <= 2:
                return True, frame.copy()
            return False, None

        def release(self):
            self.released = True

    def open_capture(*_args):
        capture = ReopeningCapture(len(captures) + 1)
        captures.append(capture)
        return capture

    monkeypatch.setattr(streams.cv2, "VideoCapture", open_capture)
    submitted = []
    marker_calls = []
    monkeypatch.setattr(
        "app.ai.live_worker.submit_live_frame",
        lambda camera_id, name, submitted_frame: submitted.append(
            (camera_id, name, submitted_frame.copy())
        ),
    )
    stream = CameraStream(
        192,
        f"video://{source}",
        "Decoder reopen",
        fps=120,
        video_anpr_state_callback=(
            lambda camera_id, state: marker_calls.append(
                (camera_id, state)
            )
        ),
    )
    published = []
    original_publish = stream._publish

    def publish(published_frame):
        published.append(published_frame.copy())
        original_publish(published_frame)

    monkeypatch.setattr(stream, "_publish", publish)
    monkeypatch.setattr(stream, "_encode", lambda _frame: b"jpeg")

    try:
        stream.start()
        # The production retry path waits one second after the decoder error.
        # Reaching EOF on capture two proves the reopen happened after it.
        for _ in range(400):
            if stream.state.ended and len(captures) >= 2:
                break
            time.sleep(0.01)
        assert stream.state.ended is True
        time.sleep(0.10)
    finally:
        stream.stop()
        if stream.thread:
            stream.thread.join(timeout=2)
        if stream._anpr_thread:
            stream._anpr_thread.join(timeout=2)

    assert len(captures) == 2
    assert all(capture.released for capture in captures)
    assert len(published) == 3
    assert len(submitted) == 1
    assert marker_calls == [(192, "started")]
    assert stream.state.anpr_preview_only is True
    assert stream.state.anpr_completed is False
    assert stream.state.anpr_interrupted is True


def test_uploaded_video_uses_ffmpeg_fallback_when_opencv_has_no_frames(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "camera-export.mp4"
    source.write_bytes(b"codec-fixture")
    stream = CameraStream(
        92,
        f"video://{source}",
        "HEVC export",
        fps=30,
    )
    published = []

    class FailedCapture:
        def __init__(self, *_args):
            pass

        def isOpened(self):
            return False

        def release(self):
            pass

    class VideoFrame:
        def to_ndarray(self, format):
            assert format == "bgr24"
            return np.full((24, 32, 3), 90, dtype=np.uint8)

    class Container:
        streams = SimpleNamespace(
            video=[SimpleNamespace(average_rate=30)],
        )

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def decode(self, video):
            assert video == 0
            yield VideoFrame()

    monkeypatch.setattr(streams.cv2, "VideoCapture", FailedCapture)
    monkeypatch.setattr(
        streams.av,
        "open",
        lambda path: Container(),
    )

    def publish(frame):
        published.append(frame)
        stream.stop_event.set()

    monkeypatch.setattr(stream, "_publish", publish)
    stream._run()

    assert len(published) == 1
    assert published[0].shape == (24, 32, 3)


def test_uploaded_video_stream_produces_real_jpeg(tmp_path):
    video_path = tmp_path / "dashboard.avi"
    _write_video(video_path, frames=3)
    stream = CameraStream(
        93,
        f"video://{video_path}",
        "Dashboard upload",
        fps=30,
    )
    jpeg = None
    stream._register_viewer()
    try:
        stream.start()
        for _ in range(100):
            with stream.lock:
                jpeg = stream.latest
            if jpeg:
                break
            time.sleep(0.01)
    finally:
        stream.stop()
        stream.thread.join(timeout=2)
        stream._unregister_viewer()

    assert jpeg
    assert jpeg.startswith(b"\xff\xd8")
    assert jpeg.endswith(b"\xff\xd9")


def test_uploaded_video_can_pause_and_resume(tmp_path, monkeypatch):
    video_path = tmp_path / "playback.avi"
    _write_video(video_path, frames=20)
    stream = CameraStream(
        94,
        f"video://{video_path}",
        "Playback controls",
        fps=30,
    )
    published = []
    monkeypatch.setattr(
        stream,
        "_publish",
        lambda frame: published.append(frame),
    )

    stream.start()
    for _ in range(100):
        if len(published) >= 3:
            break
        time.sleep(0.01)
    assert stream.pause() is True
    paused_count = len(published)
    time.sleep(0.12)
    assert len(published) <= paused_count + 1
    assert stream.state.paused is True

    assert stream.resume() is True
    for _ in range(100):
        if len(published) >= paused_count + 3:
            break
        time.sleep(0.01)
    stream.stop()
    stream.thread.join(timeout=2)

    assert len(published) >= paused_count + 3
    assert stream.state.paused is False
