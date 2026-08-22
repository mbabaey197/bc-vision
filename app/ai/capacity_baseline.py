"""Repeatable 1/3/6-camera capacity evidence for the production ANPR path.

The capacity runner replays one immutable video through independent production
``CameraStream`` instances and the real shared ``LiveANPRWorker``.  It records
process CPU, decode, inference, preview JPEG, application-level frame
coalescing, FPS and persisted events.  A replay is capacity evidence only; a
99% accuracy claim remains blocked unless a separate independently labelled
passage evidence file satisfies :mod:`app.ai.pass_benchmark`.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


BASELINE_SCHEMA = 1
SUPPORTED_CAMERA_COUNTS = (1, 3, 6)
CAPACITY_EVIDENCE_KIND = "production-pipeline-capacity-baseline"


def _nonnegative_number(value) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if number < 0.0 or number == float("inf") or number != number:
        return 0.0
    return number


def _nonnegative_count(value) -> int:
    if isinstance(value, bool):
        return 0
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(0, number)


def _sum_nested(rows, group: str, field: str, *, integer=False):
    values = [
        (row.get(group) or {}).get(field, 0)
        for row in rows
        if isinstance(row, dict)
    ]
    if integer:
        return sum(_nonnegative_count(value) for value in values)
    return sum(_nonnegative_number(value) for value in values)


def aggregate_capacity_run(
    statuses,
    *,
    camera_count: int,
    wall_seconds: float,
    process_cpu_seconds: float,
    logical_cpu_count: int,
    source_frames: int,
    source_fps: float,
    completed: bool,
) -> dict:
    """Aggregate production status snapshots into one comparable run."""

    if camera_count not in SUPPORTED_CAMERA_COUNTS:
        raise ValueError("camera_count must be one of 1, 3 or 6")
    if not isinstance(statuses, list) or len(statuses) != camera_count:
        raise ValueError("one status snapshot is required for every camera")

    wall = max(1e-9, _nonnegative_number(wall_seconds))
    cpu = _nonnegative_number(process_cpu_seconds)
    logical = max(1, _nonnegative_count(logical_cpu_count))
    source_frame_count = _nonnegative_count(source_frames)
    expected_frames = source_frame_count * camera_count

    decoded = _sum_nested(
        statuses,
        "stream_metrics",
        "decoded_frames",
        integer=True,
    )
    decode_seconds = _sum_nested(
        statuses,
        "stream_metrics",
        "decode_seconds",
    )
    capture_failures = _sum_nested(
        statuses,
        "stream_metrics",
        "capture_failures",
        integer=True,
    )
    jpeg_attempts = _sum_nested(
        statuses,
        "stream_metrics",
        "jpeg_attempts",
        integer=True,
    )
    jpeg_frames = _sum_nested(
        statuses,
        "stream_metrics",
        "jpeg_frames",
        integer=True,
    )
    jpeg_seconds = _sum_nested(
        statuses,
        "stream_metrics",
        "jpeg_seconds",
    )
    jpeg_bytes = _sum_nested(
        statuses,
        "stream_metrics",
        "jpeg_bytes",
        integer=True,
    )
    anpr_queue_frames = _sum_nested(
        statuses,
        "stream_metrics",
        "anpr_queue_frames",
        integer=True,
    )
    stream_coalesced = _sum_nested(
        statuses,
        "stream_metrics",
        "anpr_queue_coalesced_frames",
        integer=True,
    )
    handed_to_worker = _sum_nested(
        statuses,
        "stream_metrics",
        "anpr_submitted_frames",
        integer=True,
    )

    received = _sum_nested(
        statuses,
        "anpr",
        "received_frames",
        integer=True,
    )
    inference_calls = _sum_nested(
        statuses,
        "anpr",
        "inference_calls",
        integer=True,
    )
    processed = _sum_nested(
        statuses,
        "anpr",
        "processed_frames",
        integer=True,
    )
    inference_seconds = _sum_nested(
        statuses,
        "anpr",
        "inference_seconds",
    )
    worker_coalesced = _sum_nested(
        statuses,
        "anpr",
        "coalesced_frames",
        integer=True,
    )
    emitted_events = _sum_nested(
        statuses,
        "anpr",
        "emitted_events",
        integer=True,
    )
    persistence_backpressure = _sum_nested(
        statuses,
        "anpr",
        "persistence_backpressure_frames",
        integer=True,
    )

    decode_shortfall = (
        max(0, expected_frames - decoded)
        if completed and expected_frames
        else 0
    )
    application_coalesced = stream_coalesced + worker_coalesced
    errors = sorted({
        str(error)
        for row in statuses
        for error in (
            row.get("error"),
            (row.get("anpr") or {}).get("last_error"),
        )
        if error
    })
    models = dict((statuses[0].get("anpr") or {}).get("models") or {})
    model_ready = bool(models.get("ready"))
    invalid_reasons = []
    if not completed:
        invalid_reasons.append("camera-replay-incomplete")
    if errors:
        invalid_reasons.append("camera-or-anpr-error")
    if not model_ready:
        invalid_reasons.append("production-models-not-ready")
    if expected_frames and decode_shortfall:
        invalid_reasons.append("decoded-frame-shortfall")
    if persistence_backpressure:
        invalid_reasons.append("persistence-backpressure")

    return {
        "camera_count": camera_count,
        "valid": not invalid_reasons,
        "invalid_reasons": invalid_reasons,
        "completed": bool(completed),
        "wall_seconds": round(wall, 6),
        "process_cpu_seconds": round(cpu, 6),
        # Like Task Manager's process figure, host percent is normalized to
        # all logical CPUs; core percent retains the useful 100%-per-core view.
        "process_cpu_core_percent": round(cpu / wall * 100.0, 6),
        "process_cpu_host_percent": round(
            cpu / (wall * logical) * 100.0,
            6,
        ),
        "logical_cpu_count": logical,
        "source": {
            "frames_per_camera": source_frame_count,
            "fps_per_camera": round(_nonnegative_number(source_fps), 6),
            "expected_decoded_frames": expected_frames,
        },
        "decode": {
            "frames": decoded,
            "seconds": round(decode_seconds, 6),
            "aggregate_fps": round(decoded / wall, 6),
            "per_camera_fps": round(decoded / wall / camera_count, 6),
            "mean_ms": round(
                decode_seconds * 1000.0 / max(1, decoded),
                6,
            ),
            "capture_failures": capture_failures,
        },
        "inference": {
            "received_frames": received,
            "calls": inference_calls,
            "processed_frames": processed,
            "seconds": round(inference_seconds, 6),
            "aggregate_fps": round(inference_calls / wall, 6),
            "per_camera_fps": round(
                inference_calls / wall / camera_count,
                6,
            ),
            "mean_ms": round(
                inference_seconds * 1000.0 / max(1, inference_calls),
                6,
            ),
        },
        "jpeg": {
            "attempts": jpeg_attempts,
            "frames": jpeg_frames,
            "seconds": round(jpeg_seconds, 6),
            "fps": round(jpeg_frames / wall, 6),
            "mean_ms": round(
                jpeg_seconds * 1000.0 / max(1, jpeg_attempts),
                6,
            ),
            "bytes": jpeg_bytes,
        },
        "frame_drop": {
            "decode_shortfall_frames": decode_shortfall,
            "stream_queue_coalesced_frames": stream_coalesced,
            "worker_coalesced_frames": worker_coalesced,
            "application_coalesced_frames": application_coalesced,
            "application_coalesced_rate": round(
                application_coalesced / max(1, decoded),
                6,
            ),
            "note": (
                "Application coalescing is intentional newest-frame sampling; "
                "it is not a measurement of RTSP network packet loss."
            ),
        },
        "handoff": {
            "queued_frames": anpr_queue_frames,
            "submitted_frames": handed_to_worker,
        },
        "events": {
            "emitted": emitted_events,
            "persistence_backpressure_frames": persistence_backpressure,
        },
        "errors": errors,
        "models": models,
        "cameras": statuses,
    }


def evaluate_passage_evidence(payload) -> dict:
    """Apply the existing fail-closed 99% gate to independent pass rows."""

    from .pass_benchmark import (
        VERIFIED_PRODUCTION_PASS,
        evaluate_accuracy_claim,
        score_passages,
    )

    if payload is None:
        return {
            "claim_ready": False,
            "reasons": ["independent-labelled-passage-dataset-required"],
            "evaluation_kind": VERIFIED_PRODUCTION_PASS,
        }
    rows = payload.get("passages") if isinstance(payload, dict) else payload
    metrics = score_passages(rows)
    decision = evaluate_accuracy_claim(
        metrics,
        evaluation_kind=VERIFIED_PRODUCTION_PASS,
    )
    decision["annotation_provenance"] = (
        dict(payload.get("annotation_provenance") or {})
        if isinstance(payload, dict)
        else {}
    )
    if not decision["annotation_provenance"]:
        decision["claim_ready"] = False
        decision["reasons"] = sorted(set(
            decision.get("reasons", [])
            + ["independent-annotation-provenance-required"]
        ))
    return decision


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_info(video: Path) -> dict:
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError("capacity baseline video could not be decoded")
    try:
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()
    if frames <= 0 or not 1.0 <= fps <= 120.0 or width <= 0 or height <= 0:
        raise ValueError("capacity baseline video metadata is invalid")
    return {
        "frames": frames,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_seconds": frames / fps,
    }


def _pipeline_revision(detector_variant: str, detector_revision: str) -> str:
    from app.config import APP_VERSION
    from .hezar_export import HEZAR_ONNX_SHA256
    from .model_manager import (
        CRNN_SHA256,
        YOLO11N_DETECTOR_SHA256,
        YOLOV8N_DETECTOR_SHA256,
    )

    detector = str(detector_variant).strip().lower()
    fixed_revision = {
        "yolo11n": YOLO11N_DETECTOR_SHA256,
        "yolov8n": YOLOV8N_DETECTOR_SHA256,
    }.get(detector, str(detector_revision).strip())
    identity = {
        "app_version": APP_VERSION,
        "detector": detector,
        "detector_revision": fixed_revision,
        "ocr_primary": HEZAR_ONNX_SHA256,
        "ocr_fallback": CRNN_SHA256,
        "ocr_policy": "hezar-v2-then-fixed-platrix",
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"bcvision-{APP_VERSION}-{digest[:16]}"


def _run_worker(args) -> dict:
    from app.config import DATA_DIR, VIDEO_DIR
    from app.database import connect, init_db, set_setting
    from app.streams import StreamManager
    from .live_worker import shutdown_live_anpr_worker
    from .model_manager import model_status, prepare_models

    source = Path(args.video).resolve()
    info = _source_info(source)
    init_db()
    set_setting("anpr_detector_model", args.detector)
    set_setting("live_fps", args.live_fps)
    set_setting("stream_width", args.stream_width)
    set_setting("jpeg_quality", args.jpeg_quality)
    if args.download_models:
        prepare_models(download=True)
    models = model_status(selected_detector=args.detector)
    if not (
        models.get("detector_ready")
        and models.get("hezar_ready")
        and models.get("platrix_crnn_ready")
    ):
        raise RuntimeError("verified production detector/OCR models are not ready")

    copied = VIDEO_DIR / ("capacity-input" + (source.suffix or ".mp4"))
    shutil.copyfile(source, copied)
    camera_ids = []
    with connect() as connection:
        for index in range(args.camera_count):
            cursor = connection.execute(
                "INSERT INTO cameras("
                "name,rtsp_url,location,city,enabled,is_demo,sort_order,"
                "lpr_enabled,lpr_confidence,duplicate_seconds"
                ") VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    f"capacity-{index + 1}",
                    "video://" + str(copied),
                    "isolated capacity baseline",
                    "",
                    1,
                    0,
                    index,
                    1,
                    args.lpr_confidence,
                    args.duplicate_seconds,
                ),
            )
            camera_ids.append(int(cursor.lastrowid))

    manager = StreamManager(stop_timeout=15.0)
    streams = []
    registered_viewers = []
    wall_started = time.perf_counter()
    cpu_started = time.process_time()
    try:
        for index, camera_id in enumerate(camera_ids):
            stream = manager.get(
                camera_id,
                "video://" + str(copied),
                f"capacity-{index + 1}",
                args.stream_width,
                args.live_fps,
                args.jpeg_quality,
            )
            streams.append(stream)
            register = getattr(stream, "_register_viewer", None)
            if args.viewers_per_camera and callable(register):
                for _viewer in range(args.viewers_per_camera):
                    register()
                    registered_viewers.append(stream)

        timeout = args.timeout_seconds
        if timeout <= 0:
            timeout = max(120.0, info["duration_seconds"] * 3.0 + 60.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if all(stream.state.ended for stream in streams):
                break
            if any(
                stream.state.last_error and not stream.state.online
                for stream in streams
            ):
                break
            time.sleep(0.05)

        wall_seconds = time.perf_counter() - wall_started
        process_cpu_seconds = time.process_time() - cpu_started
        statuses = [manager.status(camera_id) for camera_id in camera_ids]
        completed = all(
            stream.state.ended and stream.state.anpr_completed
            for stream in streams
        )
        result = aggregate_capacity_run(
            statuses,
            camera_count=args.camera_count,
            wall_seconds=wall_seconds,
            process_cpu_seconds=process_cpu_seconds,
            logical_cpu_count=os.cpu_count() or 1,
            source_frames=info["frames"],
            source_fps=info["fps"],
            completed=completed,
        )
        with connect() as connection:
            persisted_events = int(connection.execute(
                "SELECT COUNT(*) FROM plate_events"
            ).fetchone()[0])
        result["events"]["persisted"] = persisted_events
        result["events"]["count_match"] = (
            persisted_events == result["events"]["emitted"]
        )
        detector_revision = next((
            str((status.get("anpr") or {}).get("detector_model_revision") or "")
            for status in statuses
            if (status.get("anpr") or {}).get("detector_model_revision")
        ), "")
        result["pipeline_revision"] = _pipeline_revision(
            args.detector,
            detector_revision,
        )
        result["viewer_mode"] = {
            "requested_viewers_per_camera": args.viewers_per_camera,
            "viewer_aware_stream_supported": all(
                hasattr(stream, "_register_viewer") for stream in streams
            ),
        }
        result["data_dir_isolated"] = str(DATA_DIR) != ""
        return result
    finally:
        for stream in registered_viewers:
            unregister = getattr(stream, "_unregister_viewer", None)
            if callable(unregister):
                unregister()
        manager.stop_all()
        shutdown_live_anpr_worker(retry_timeout=15.0)


def _write_json_exclusive(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def _run_parent(args) -> dict:
    video = Path(args.video).expanduser().resolve()
    details = video.stat()
    if not video.is_file() or details.st_size <= 0:
        raise ValueError("capacity baseline requires a non-empty video file")
    counts = tuple(sorted(set(args.camera_counts)))
    if counts != SUPPORTED_CAMERA_COUNTS:
        raise ValueError("capacity baseline must run exactly 1, 3 and 6 cameras")

    passage_payload = None
    if args.passage_evidence:
        with Path(args.passage_evidence).open("r", encoding="utf-8") as source:
            passage_payload = json.load(source)

    runs = []
    with tempfile.TemporaryDirectory(prefix="bcvision-capacity-") as temporary:
        temporary_root = Path(temporary)
        for camera_count in counts:
            data_dir = temporary_root / f"data-{camera_count}"
            result_file = temporary_root / f"result-{camera_count}.json"
            environment = os.environ.copy()
            environment["BCVISION_DATA_DIR"] = str(data_dir)
            command = [
                sys.executable,
                "-m",
                "app.ai.capacity_baseline",
                "--worker",
                "--video",
                str(video),
                "--camera-count",
                str(camera_count),
                "--result-file",
                str(result_file),
                "--detector",
                args.detector,
                "--live-fps",
                str(args.live_fps),
                "--stream-width",
                str(args.stream_width),
                "--jpeg-quality",
                str(args.jpeg_quality),
                "--lpr-confidence",
                str(args.lpr_confidence),
                "--duplicate-seconds",
                str(args.duplicate_seconds),
                "--timeout-seconds",
                str(args.timeout_seconds),
                "--viewers-per-camera",
                str(args.viewers_per_camera),
            ]
            if args.download_models:
                command.append("--download-models")
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[2],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(300.0, args.timeout_seconds + 180.0),
            )
            if completed.returncode != 0 or not result_file.exists():
                raise RuntimeError(
                    f"{camera_count}-camera baseline failed: "
                    + (completed.stderr.strip() or completed.stdout.strip())
                )
            with result_file.open("r", encoding="utf-8") as source:
                runs.append(json.load(source))

    pipeline_revisions = sorted({
        str(run.get("pipeline_revision") or "") for run in runs
    })
    report = {
        "schema": BASELINE_SCHEMA,
        "evidence_kind": CAPACITY_EVIDENCE_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "name": video.name,
            "size_bytes": details.st_size,
            "sha256": _sha256(video),
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "logical_cpu_count": os.cpu_count() or 1,
        },
        "settings": {
            "camera_counts": list(counts),
            "detector": args.detector,
            "live_fps": args.live_fps,
            "stream_width": args.stream_width,
            "jpeg_quality": args.jpeg_quality,
            "lpr_confidence": args.lpr_confidence,
            "duplicate_seconds": args.duplicate_seconds,
            "viewers_per_camera": args.viewers_per_camera,
        },
        "pipeline_revisions": pipeline_revisions,
        "comparable": (
            len(pipeline_revisions) == 1
            and all(run.get("valid") for run in runs)
        ),
        "runs": runs,
        "passage_accuracy": evaluate_passage_evidence(passage_payload),
        "accuracy_note": (
            "Capacity replay event counts are diagnostics only. Passage-level "
            "exact accuracy requires independent labelled passages including "
            "miss, duplicate, unreadable, day/night, distance, angle and "
            "image-quality slices."
        ),
    }
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the BC Vision production 1/3/6-camera baseline",
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--camera-counts",
        nargs="+",
        type=int,
        default=list(SUPPORTED_CAMERA_COUNTS),
    )
    parser.add_argument("--detector", choices=("yolo11n", "yolov8n", "yolox"), default="yolo11n")
    parser.add_argument("--live-fps", type=int, default=5)
    parser.add_argument("--stream-width", type=int, default=640)
    parser.add_argument("--jpeg-quality", type=int, default=70)
    parser.add_argument("--lpr-confidence", type=int, default=60)
    parser.add_argument("--duplicate-seconds", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=0.0)
    parser.add_argument("--viewers-per-camera", type=int, choices=(0, 1), default=0)
    parser.add_argument("--download-models", action="store_true")
    parser.add_argument("--passage-evidence")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--camera-count", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--result-file", help=argparse.SUPPRESS)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if not 1 <= args.live_fps <= 30:
        raise ValueError("live_fps must be within 1..30")
    if not 160 <= args.stream_width <= 3840:
        raise ValueError("stream_width must be within 160..3840")
    if not 30 <= args.jpeg_quality <= 95:
        raise ValueError("jpeg_quality must be within 30..95")
    if not 1 <= args.lpr_confidence <= 99:
        raise ValueError("lpr_confidence must be within 1..99")
    if args.timeout_seconds < 0.0:
        raise ValueError("timeout_seconds must not be negative")

    if args.worker:
        if args.camera_count not in SUPPORTED_CAMERA_COUNTS:
            raise ValueError("worker camera_count must be 1, 3 or 6")
        if not args.result_file:
            raise ValueError("worker result_file is required")
        _write_json_exclusive(Path(args.result_file), _run_worker(args))
        return 0

    if not args.output:
        raise ValueError("--output is required")
    _write_json_exclusive(Path(args.output), _run_parent(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
