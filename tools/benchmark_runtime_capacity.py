"""Repeatable BC Vision runtime capacity benchmark.

This tool is measurement-only.  It does not change detector/OCR selection,
database state, product version, or runtime ABI.  Synthetic transport runs use
the real CameraStream scheduling/JPEG path with a paced local MJPEG source.
They are useful for isolating scheduling and preview cost, but they are not a
substitute for a target-host RTSP/H.264 benchmark or passage-level accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import time
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * max(0.0, min(100.0, percentile)) / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _host() -> dict:
    logical = max(1, int(os.cpu_count() or 1))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": logical,
    }


def _cpu_metrics(cpu_seconds: float, wall_seconds: float) -> dict:
    wall = max(float(wall_seconds), 1e-9)
    logical = max(1, int(os.cpu_count() or 1))
    core_equivalent = float(cpu_seconds) / wall * 100.0
    return {
        "process_cpu_seconds": round(float(cpu_seconds), 6),
        "core_equivalent_cpu_percent": round(core_equivalent, 3),
        "host_normalized_cpu_percent": round(core_equivalent / logical, 3),
    }


def _make_frame(width: int, height: int, phase: int):
    import cv2
    import numpy as np

    width = max(64, int(width))
    height = max(36, int(height))
    x = np.arange(width, dtype=np.uint16)
    y = np.arange(height, dtype=np.uint16)[:, None]
    frame = np.empty((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = ((x[None, :] + phase * 11) % 256).astype(np.uint8)
    frame[:, :, 1] = ((y + phase * 17) % 256).astype(np.uint8)
    frame[:, :, 2] = (((x[None, :] // 2) + (y // 2) + phase * 23) % 256).astype(np.uint8)
    plate_w = max(80, width // 7)
    plate_h = max(28, height // 16)
    px = int((phase * 37) % max(1, width - plate_w))
    py = max(4, height // 2)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (238, 238, 238), -1)
    cv2.rectangle(frame, (px, py), (px + plate_w, py + plate_h), (18, 18, 18), 2)
    cv2.putText(
        frame,
        f"12B{phase % 10}4567",
        (px + 5, min(height - 5, py + plate_h - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, min(0.8, plate_h / 42.0)),
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return frame


def _write_source_video(path: Path, width: int, height: int, frames: int = 12) -> dict:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 25.0, (int(width), int(height)))
    if not writer.isOpened():
        raise RuntimeError("OpenCV MJPEG benchmark source could not be created")
    try:
        for phase in range(max(4, int(frames))):
            writer.write(_make_frame(width, height, phase))
    finally:
        writer.release()
    data = path.read_bytes()
    return {
        "path": str(path),
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "codec": "MJPG/AVI",
    }


class _Counters:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.decoded_frames = 0
        self.decode_wall = []
        self.jpeg_frames = 0
        self.jpeg_wall = []
        self.anpr_submissions = 0
        self.published_frames = 0

    def add_decode(self, wall_seconds: float) -> None:
        with self.lock:
            self.decoded_frames += 1
            self.decode_wall.append(float(wall_seconds))

    def add_jpeg(self, wall_seconds: float) -> None:
        with self.lock:
            self.jpeg_frames += 1
            self.jpeg_wall.append(float(wall_seconds))

    def add_anpr(self) -> None:
        with self.lock:
            self.anpr_submissions += 1

    def add_publish(self) -> int:
        with self.lock:
            self.published_frames += 1
            return self.published_frames


class _PacedCapture:
    def __init__(self, inner, source_fps: float, counters: _Counters) -> None:
        self.inner = inner
        self.period = 1.0 / max(0.1, float(source_fps))
        self.counters = counters
        self.next_frame_at = time.perf_counter()
        self.closed = False

    def isOpened(self):
        return (not self.closed) and bool(self.inner.isOpened())

    def set(self, prop, value):
        try:
            return self.inner.set(prop, value)
        except Exception:
            return False

    def get(self, prop):
        try:
            return self.inner.get(prop)
        except Exception:
            return 0.0

    def read(self):
        if self.closed:
            return False, None
        now = time.perf_counter()
        wait = self.next_frame_at - now
        if wait > 0:
            time.sleep(wait)
        self.next_frame_at = max(self.next_frame_at + self.period, time.perf_counter())

        started = time.perf_counter()
        ok, frame = self.inner.read()
        if not ok or frame is None:
            try:
                import cv2
                self.inner.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.inner.read()
            except Exception:
                ok, frame = False, None
        elapsed = time.perf_counter() - started
        if ok and frame is not None:
            self.counters.add_decode(elapsed)
        return ok, frame

    def release(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.inner.release()
        except Exception:
            pass

    close = release


def _run_transport_scenario(
    *,
    camera_count: int,
    frames_per_camera: int,
    source_fps: float,
    dashboard_fps: int,
    source_width: int,
    source_height: int,
    preview_width: int,
    jpeg_quality: int,
    viewer: bool,
    source_video: Path,
) -> dict:
    import cv2
    from app import streams as streams_module
    from app.streams import CameraStream

    camera_count = max(1, int(camera_count))
    frames_per_camera = max(1, int(frames_per_camera))
    counters = _Counters()
    original_video_capture = streams_module.cv2.VideoCapture
    capture_lock = threading.Lock()
    captures = []

    def capture_factory(*_args, **_kwargs):
        inner = original_video_capture(str(source_video), cv2.CAP_FFMPEG)
        if not inner.isOpened():
            inner.release()
            inner = original_video_capture(str(source_video))
        capture = _PacedCapture(inner, source_fps, counters)
        with capture_lock:
            captures.append(capture)
        return capture

    streams_module.cv2.VideoCapture = capture_factory
    streams = []
    viewer_registered = []
    target_total = camera_count * frames_per_camera
    publish_lock = threading.Lock()
    per_camera_published = {index: 0 for index in range(camera_count)}

    try:
        for index in range(camera_count):
            stream = CameraStream(
                10000 + index,
                f"rtsp://benchmark/camera-{index}",
                f"Benchmark {index}",
                width=int(preview_width),
                fps=int(dashboard_fps),
                quality=int(jpeg_quality),
            )
            stream._live_overlays = lambda _frame: []
            stream._queue_anpr = lambda _frame, c=counters: c.add_anpr()

            original_encode = stream._encode

            def timed_encode(frame, *, _encode=original_encode, _c=counters):
                started = time.perf_counter()
                result = _encode(frame)
                _c.add_jpeg(time.perf_counter() - started)
                return result

            stream._encode = timed_encode
            original_publish = stream._publish

            def counted_publish(
                frame,
                *,
                _publish=original_publish,
                _stream=stream,
                _index=index,
            ):
                _publish(frame)
                with publish_lock:
                    per_camera_published[_index] += 1
                    local_count = per_camera_published[_index]
                counters.add_publish()
                if local_count >= frames_per_camera:
                    _stream.stop_event.set()

            stream._publish = counted_publish
            if viewer and hasattr(stream, "_register_viewer"):
                stream._register_viewer()
                viewer_registered.append(stream)
            streams.append(stream)

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        threads = [
            threading.Thread(
                target=stream._run,
                daemon=True,
                name=f"benchmark-camera-{index}",
            )
            for index, stream in enumerate(streams)
        ]
        for thread in threads:
            thread.start()
        timeout = max(10.0, frames_per_camera / max(0.1, source_fps) * 8.0 + 5.0)
        deadline = time.monotonic() + timeout
        for thread in threads:
            remaining = max(0.0, deadline - time.monotonic())
            thread.join(remaining)
        alive = [thread.name for thread in threads if thread.is_alive()]
        if alive:
            for stream in streams:
                stream.stop_event.set()
            for thread in threads:
                thread.join(2.0)
            raise RuntimeError(f"transport benchmark threads did not stop: {alive}")
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
    finally:
        streams_module.cv2.VideoCapture = original_video_capture
        for stream in viewer_registered:
            try:
                stream._unregister_viewer()
            except Exception:
                pass
        for capture in captures:
            capture.release()

    expected_frames = max(0, int(round(float(source_fps) * wall_seconds * camera_count)))
    estimated_drop = max(0, expected_frames - counters.decoded_frames)
    drop_rate = estimated_drop / max(1, expected_frames)
    decode_values = list(counters.decode_wall)
    jpeg_values = list(counters.jpeg_wall)
    viewer_supported = all(hasattr(stream, "_register_viewer") for stream in streams)
    return {
        "camera_count": camera_count,
        "viewer_requested": bool(viewer),
        "viewer_lifecycle_supported": bool(viewer_supported),
        "wall_seconds": round(wall_seconds, 6),
        **_cpu_metrics(cpu_seconds, wall_seconds),
        "source_fps_per_camera": float(source_fps),
        "dashboard_fps": int(dashboard_fps),
        "target_frames": target_total,
        "decoded_frames": counters.decoded_frames,
        "decode_fps_total": round(counters.decoded_frames / max(wall_seconds, 1e-9), 3),
        "decode_fps_per_camera": round(
            counters.decoded_frames / max(wall_seconds * camera_count, 1e-9), 3
        ),
        "decode_call_ms_p50": round(_percentile(decode_values, 50) * 1000.0, 3),
        "decode_call_ms_p95": round(_percentile(decode_values, 95) * 1000.0, 3),
        "decode_call_wall_seconds_total": round(sum(decode_values), 6),
        "published_frames": counters.published_frames,
        "published_fps_total": round(counters.published_frames / max(wall_seconds, 1e-9), 3),
        "anpr_submissions": counters.anpr_submissions,
        "anpr_submission_fps_total": round(
            counters.anpr_submissions / max(wall_seconds, 1e-9), 3
        ),
        "jpeg_encodes": counters.jpeg_frames,
        "jpeg_fps_total": round(counters.jpeg_frames / max(wall_seconds, 1e-9), 3),
        "jpeg_encode_ms_p50": round(_percentile(jpeg_values, 50) * 1000.0, 3),
        "jpeg_encode_ms_p95": round(_percentile(jpeg_values, 95) * 1000.0, 3),
        "jpeg_encode_wall_seconds_total": round(sum(jpeg_values), 6),
        "expected_source_frame_opportunities": expected_frames,
        "estimated_source_frame_drop": estimated_drop,
        "estimated_source_frame_drop_rate": round(drop_rate, 6),
        "event_count": None,
        "event_count_evaluable": False,
        "event_count_reason": "ANPR is deliberately stubbed in transport isolation",
    }


def run_transport(args) -> dict:
    camera_counts = [
        int(value.strip())
        for value in str(args.camera_counts).split(",")
        if value.strip()
    ]
    if not camera_counts or any(value <= 0 for value in camera_counts):
        raise ValueError("camera counts must be positive")

    with tempfile.TemporaryDirectory(prefix="bcvision-capacity-") as temp_dir:
        source_path = Path(temp_dir) / "paced-source.avi"
        source_info = _write_source_video(
            source_path,
            int(args.source_width),
            int(args.source_height),
            frames=min(16, max(8, int(args.frames_per_camera))),
        )
        scenarios = [
            _run_transport_scenario(
                camera_count=count,
                frames_per_camera=int(args.frames_per_camera),
                source_fps=float(args.source_fps),
                dashboard_fps=int(args.dashboard_fps),
                source_width=int(args.source_width),
                source_height=int(args.source_height),
                preview_width=int(args.preview_width),
                jpeg_quality=int(args.jpeg_quality),
                viewer=bool(args.viewer),
                source_video=source_path,
            )
            for count in camera_counts
        ]

    return {
        "schema": 1,
        "kind": "bcvision-runtime-capacity",
        "mode": "transport",
        "implementation_ref": str(args.ref),
        "host": _host(),
        "source": {
            **source_info,
            "path": "temporary-local-source",
            "source_width": int(args.source_width),
            "source_height": int(args.source_height),
            "source_fps": float(args.source_fps),
            "transport_emulation": (
                "CameraStream sees an RTSP URL while VideoCapture is backed by "
                "a paced local MJPEG/AVI decoder"
            ),
        },
        "preview": {
            "dashboard_fps": int(args.dashboard_fps),
            "width": int(args.preview_width),
            "jpeg_quality": int(args.jpeg_quality),
            "viewer_requested": bool(args.viewer),
        },
        "scenarios": scenarios,
        "accuracy": {
            "passage_level_exact_accuracy": None,
            "misses": None,
            "duplicates": None,
            "unreadable": None,
            "evaluable": False,
            "reason": (
                "Synthetic transport cannot establish passage-level ANPR accuracy; "
                "use an independently labelled production passage dataset."
            ),
        },
        "limitations": [
            "The source decoder is MJPEG/AVI, not production RTSP H.264/H.265.",
            "Estimated frame drop is source-opportunity loss, not a camera/NIC packet-loss measurement.",
            "Overlay tracking is excluded so JPEG resize/encode cost is isolated.",
            "ANPR inference is stubbed in transport mode; event accuracy is not evaluable.",
        ],
    }


def _timed_call(function, *args, **kwargs):
    started = time.perf_counter()
    value = function(*args, **kwargs)
    return value, time.perf_counter() - started


def run_inference(args) -> dict:
    import numpy as np

    from app.ai.hezar_export import HEZAR_ONNX_SHA256
    from app.ai.model_manager import (
        CRNN_SHA256,
        detector_variant_spec,
        prepare_models,
    )
    from app.ai.onnx_crnn import read_plate_crnn
    from app.ai.onnx_detector import detect_plates_onnx
    from app.ai.onnx_hezar import read_plate_hezar_primary

    camera_counts = [
        int(value.strip())
        for value in str(args.camera_counts).split(",")
        if value.strip()
    ]
    if not camera_counts or any(value <= 0 for value in camera_counts):
        raise ValueError("camera counts must be positive")

    prepare_models(download=True)
    detector_spec = detector_variant_spec(args.detector)
    if args.detector == "yolox" and not detector_spec.get("ready"):
        raise RuntimeError(detector_spec.get("error") or "YOLOX is not ready")

    detector_frame = _make_frame(int(args.source_width), int(args.source_height), 3)
    hezar_crop = np.zeros((32, 384, 3), dtype=np.uint8)
    platrix_crop = np.zeros((32, 128, 3), dtype=np.uint8)
    shared_key = "runtime-capacity-shared"

    detect_plates_onnx(
        detector_frame,
        min_confidence=0.05,
        max_results=2,
        engine_key=shared_key,
        detector_variant=args.detector,
    )
    read_plate_hezar_primary(hezar_crop, engine_key=shared_key)
    if args.include_platrix:
        read_plate_crnn(platrix_crop, engine_key=shared_key)

    scenarios = []
    iterations = max(1, int(args.iterations))
    for camera_count in camera_counts:
        detector_latencies = []
        hezar_latencies = []
        platrix_latencies = []
        errors = []
        data_lock = threading.Lock()
        barrier = threading.Barrier(camera_count)

        def worker(index: int) -> None:
            try:
                barrier.wait(timeout=10.0)
                for _iteration in range(iterations):
                    _, detector_elapsed = _timed_call(
                        detect_plates_onnx,
                        detector_frame,
                        min_confidence=0.05,
                        max_results=2,
                        engine_key=shared_key,
                        detector_variant=args.detector,
                    )
                    _, hezar_elapsed = _timed_call(
                        read_plate_hezar_primary,
                        hezar_crop,
                        engine_key=shared_key,
                    )
                    platrix_elapsed = None
                    if args.include_platrix:
                        _, platrix_elapsed = _timed_call(
                            read_plate_crnn,
                            platrix_crop,
                            engine_key=shared_key,
                        )
                    with data_lock:
                        detector_latencies.append(detector_elapsed)
                        hezar_latencies.append(hezar_elapsed)
                        if platrix_elapsed is not None:
                            platrix_latencies.append(platrix_elapsed)
            except BaseException as exc:
                with data_lock:
                    errors.append(f"camera-{index}: {type(exc).__name__}: {exc}")

        wall_started = time.perf_counter()
        cpu_started = time.process_time()
        threads = [
            threading.Thread(target=worker, args=(index,), daemon=True)
            for index in range(camera_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(max(30.0, iterations * 15.0))
        alive = [index for index, thread in enumerate(threads) if thread.is_alive()]
        if alive:
            raise RuntimeError(f"inference benchmark threads timed out: {alive}")
        cpu_seconds = time.process_time() - cpu_started
        wall_seconds = time.perf_counter() - wall_started
        if errors:
            raise RuntimeError("; ".join(errors))

        expected_calls = camera_count * iterations
        scenarios.append({
            "camera_count": camera_count,
            "iterations_per_camera": iterations,
            "wall_seconds": round(wall_seconds, 6),
            **_cpu_metrics(cpu_seconds, wall_seconds),
            "detector_calls": len(detector_latencies),
            "detector_calls_expected": expected_calls,
            "detector_fps": round(len(detector_latencies) / max(wall_seconds, 1e-9), 3),
            "detector_ms_p50": round(_percentile(detector_latencies, 50) * 1000.0, 3),
            "detector_ms_p95": round(_percentile(detector_latencies, 95) * 1000.0, 3),
            "hezar_calls": len(hezar_latencies),
            "hezar_reads_per_second": round(len(hezar_latencies) / max(wall_seconds, 1e-9), 3),
            "hezar_ms_p50": round(_percentile(hezar_latencies, 50) * 1000.0, 3),
            "hezar_ms_p95": round(_percentile(hezar_latencies, 95) * 1000.0, 3),
            "platrix_calls": len(platrix_latencies),
            "platrix_reads_per_second": round(len(platrix_latencies) / max(wall_seconds, 1e-9), 3),
            "platrix_ms_p50": round(_percentile(platrix_latencies, 50) * 1000.0, 3),
            "platrix_ms_p95": round(_percentile(platrix_latencies, 95) * 1000.0, 3),
            "event_count": None,
            "event_count_evaluable": False,
            "event_count_reason": "raw model microbenchmark does not run passage persistence",
        })

    return {
        "schema": 1,
        "kind": "bcvision-runtime-capacity",
        "mode": "inference",
        "implementation_ref": str(args.ref),
        "host": _host(),
        "model_contract": {
            "detector_variant": str(args.detector),
            "detector_sha256": str(detector_spec.get("sha256", "")),
            "detector_method": str(detector_spec.get("method", "")),
            "hezar_sha256": str(HEZAR_ONNX_SHA256),
            "hezar_role": "primary",
            "platrix_crnn_sha256": str(CRNN_SHA256) if args.include_platrix else "",
            "platrix_role": "fixed fallback after Hezar rejection" if args.include_platrix else "not measured",
            "shared_engine_key": shared_key,
        },
        "source_shape": [int(args.source_height), int(args.source_width), 3],
        "scenarios": scenarios,
        "accuracy": {
            "passage_level_exact_accuracy": None,
            "evaluable": False,
            "reason": "synthetic model inputs are a load microbenchmark, not labelled passages",
        },
        "limitations": [
            "This measures real ONNX detector/OCR execution but uses synthetic pixels.",
            "It does not include RTSP decode, tracking, persistence, or passage event accuracy.",
            "YOLOX is measured only when its hash-verified custom manifest/model is installed.",
        ],
    }


def _write_result(result: dict, output: Path) -> None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="BC Vision 1/3/6-camera capacity benchmark")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    transport = subparsers.add_parser("transport", help="Measure decode scheduling/JPEG/ANPR submission")
    transport.add_argument("--output", type=Path, required=True)
    transport.add_argument("--ref", default=os.environ.get("BCVISION_BENCHMARK_REF", "working-tree"))
    transport.add_argument("--camera-counts", default="1,3,6")
    transport.add_argument("--frames-per-camera", type=int, default=30)
    transport.add_argument("--source-fps", type=float, default=25.0)
    transport.add_argument("--source-width", type=int, default=1920)
    transport.add_argument("--source-height", type=int, default=1080)
    transport.add_argument("--dashboard-fps", type=int, default=5)
    transport.add_argument("--preview-width", type=int, default=640)
    transport.add_argument("--jpeg-quality", type=int, default=70)
    transport.add_argument("--viewer", action="store_true")

    inference = subparsers.add_parser("inference", help="Measure real detector/Hezar/Platrix runtime")
    inference.add_argument("--output", type=Path, required=True)
    inference.add_argument("--ref", default=os.environ.get("BCVISION_BENCHMARK_REF", "working-tree"))
    inference.add_argument("--camera-counts", default="1,3,6")
    inference.add_argument("--iterations", type=int, default=4)
    inference.add_argument("--source-width", type=int, default=1920)
    inference.add_argument("--source-height", type=int, default=1080)
    inference.add_argument("--detector", choices=("yolo11n", "yolov8n", "yolox"), default="yolo11n")
    inference.add_argument("--include-platrix", action="store_true")

    args = parser.parse_args(argv)
    if args.mode == "transport":
        result = run_transport(args)
    else:
        result = run_inference(args)
    _write_result(result, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
