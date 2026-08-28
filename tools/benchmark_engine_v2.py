"""CLI for the independent BC Vision ANPR Engine V2 benchmark harness."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine_v2.benchmark import (  # noqa: E402
    CallableAccuracyAdapter,
    CommandAccuracyAdapter,
    CommandPerformanceAdapter,
    SyntheticControlPlaneAdapter,
    all_active_camera_scenarios,
    compare_accuracy_adapters,
    default_camera_scenarios,
    load_accuracy_manifest,
    load_performance_adapter,
    run_performance_suite,
    run_standard_performance_matrices,
    write_accuracy_outputs,
    write_performance_outputs,
)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _unit_float(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be within 0..1")
    return parsed


def _width_height(value: str) -> tuple[int, int]:
    pieces = value.lower().replace(",", "x").split("x")
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT, for example 640x360")
    try:
        width, height = (int(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("width and height must be integers") from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("width and height must be positive")
    return width, height


def _percentage_roi(value: str) -> tuple[float, float, float, float]:
    pieces = value.split(",")
    if len(pieces) != 4:
        raise argparse.ArgumentTypeError("must be X,Y,WIDTH,HEIGHT percentages")
    try:
        roi = tuple(float(piece.strip()) for piece in pieces)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI values must be numbers") from exc
    x, y, width, height = roi
    if min(x, y, width, height) < 0 or x + width > 100 or y + height > 100:
        raise argparse.ArgumentTypeError("ROI must fit within the 0..100 percent frame")
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("ROI width and height must be positive")
    return roi  # type: ignore[return-value]


def _adapter_group(parser: argparse.ArgumentParser, prefix: str) -> None:
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        f"--{prefix}-callable",
        metavar="MODULE:CALLABLE",
        help="Python adapter called once per verified sample",
    )
    group.add_argument(
        f"--{prefix}-command",
        metavar="COMMAND",
        help=(
            "JSON-in/JSON-out command called once per verified sample; placeholders "
            "{engine}, {sample_id}, and {input} are supported"
        ),
    )
    built_in = "legacy-video" if prefix == "v1" else "engine-v2-offline"
    group.add_argument(
        f"--{prefix}-builtin",
        choices=(built_in,),
        help=(
            "use the unchanged app.ai.video_test.process_video adapter"
            if prefix == "v1"
            else "use the shared-model deterministic OpenCV Engine V2 adapter"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure Engine V2 scheduling/resource behavior or compare V1/V2 "
            "accuracy on one operator-verified manifest."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    performance = subparsers.add_parser(
        "performance",
        help="run 1/4/8/16 camera resource and newest-frame scenarios",
    )
    performance.add_argument("--output-dir", type=Path, required=True)
    performance.add_argument("--include-32", action="store_true")
    performance.add_argument(
        "--matrix",
        choices=("standard", "fixed-active", "all-active"),
        default="standard",
        help=(
            "standard runs both fixed-active idle scaling and all-active busy scaling; "
            "the other choices run one matrix only (default: standard)"
        ),
    )
    performance.add_argument(
        "--active-cameras",
        type=_nonnegative_int,
        default=1,
        help=(
            "active count for the fixed-active matrix; ignored by all-active "
            "(default: 1)"
        ),
    )
    performance.add_argument("--nominal-seconds", type=_positive_float, default=5.0)
    performance.add_argument("--ticks-per-second", type=int, default=10)
    performance.add_argument("--producer-burst", type=int, default=2)
    performance.add_argument("--consumer-budget", type=int)
    performance.add_argument("--max-frame-age-ms", type=float, default=250.0)
    performance.add_argument(
        "--paced",
        action="store_true",
        help=(
            "pace producer ticks against real monotonic time; required for any "
            "production-evidence classification"
        ),
    )
    performance_adapter = performance.add_mutually_exclusive_group()
    performance_adapter.add_argument(
        "--adapter-callable",
        metavar="MODULE:CALLABLE",
        help="long-lived real adapter; callable receives BenchmarkFrameJob",
    )
    performance_adapter.add_argument(
        "--adapter-command",
        metavar="COMMAND",
        help="JSON-in/JSON-out command per job (child resources are not included in parent CPU/RAM)",
    )
    performance.add_argument("--adapter-name", default="engine-v2")
    performance.add_argument("--adapter-timeout", type=_positive_float, default=30.0)
    performance.add_argument(
        "--synthetic-cpu-work",
        type=_nonnegative_int,
        default=0,
        help="stub integer operations per scheduled job; never production evidence",
    )

    accuracy = subparsers.add_parser(
        "compare-accuracy",
        help="run V1 and V2 adapters on exactly the same verified eight-category manifest",
    )
    accuracy.add_argument("--manifest", type=Path, required=True)
    accuracy.add_argument("--output-dir", type=Path, required=True)
    _adapter_group(accuracy, "v1")
    _adapter_group(accuracy, "v2")
    accuracy.add_argument("--v1-name", default="v1")
    accuracy.add_argument("--v2-name", default="v2")
    accuracy.add_argument("--adapter-timeout", type=_positive_float, default=60.0)
    accuracy.add_argument("--v1-frame-step", type=_positive_int, default=1)
    accuracy.add_argument("--v1-max-events", type=_positive_int, default=100)
    accuracy.add_argument("--v1-min-confidence", type=_unit_float, default=0.20)
    accuracy.add_argument("--v1-duplicate-seconds", type=float, default=2.5)
    accuracy.add_argument("--v1-detector-variant", default="yolo11n")
    accuracy.add_argument(
        "--v1-roi",
        type=_percentage_roi,
        help=(
            "legacy percentage ROI as X,Y,WIDTH,HEIGHT; rejected for built-in "
            "same-input comparisons until V2 has an equivalent ROI contract"
        ),
    )
    accuracy.add_argument(
        "--v2-detector-model",
        type=Path,
        help="detector ONNX/OpenVINO model; required with --v2-builtin",
    )
    accuracy.add_argument(
        "--v2-ocr-model",
        type=Path,
        help="OCR ONNX/OpenVINO model; required with --v2-builtin",
    )
    accuracy.add_argument(
        "--v2-backend",
        choices=("auto", "openvino", "onnxruntime"),
        default="auto",
    )
    accuracy.add_argument("--v2-device", default="AUTO")
    accuracy.add_argument(
        "--v2-detector-frame-size",
        type=_width_height,
        default=(640, 360),
        metavar="WIDTHxHEIGHT",
    )
    accuracy.add_argument(
        "--v2-detector-input-size",
        type=_width_height,
        default=(320, 320),
        metavar="WIDTHxHEIGHT",
    )
    accuracy.add_argument("--v2-detector-confidence", type=_unit_float, default=0.25)
    accuracy.add_argument("--v2-min-ocr-confidence", type=_unit_float, default=0.55)
    accuracy.add_argument("--v2-duplicate-seconds", type=float, default=2.5)
    accuracy.add_argument("--v2-frame-step", type=_positive_int, default=1)
    accuracy.add_argument(
        "--v2-max-frames",
        type=_positive_int,
        help=(
            "diagnostic truncation only; rejected by compare-accuracy because "
            "both engines must consume the complete input"
        ),
    )
    accuracy.add_argument("--v2-opencv-threads", type=_positive_int, default=1)
    accuracy.add_argument(
        "--v2-allow-capture-backend-fallback",
        action="store_true",
        help="allow CAP_ANY if deterministic FFmpeg software decode is unavailable",
    )
    accuracy.add_argument(
        "--v2-no-inference-fallback",
        action="store_true",
        help="fail instead of falling back to another inference provider/device",
    )
    accuracy.add_argument(
        "--allow-missing-input-files",
        action="store_true",
        help=(
            "disable strict evidence and permit URI/adapter-resolved inputs; "
            "intended only for non-promotional contract tests"
        ),
    )
    accuracy.add_argument(
        "--allow-partial-coverage",
        action="store_true",
        help=(
            "permit a verified subset of the eight accuracy categories or any "
            "label_scope=known_positives sample; default requires complete, "
            "exhaustive coverage"
        ),
    )
    accuracy.add_argument(
        "--allow-no-negative",
        action="store_true",
        help=(
            "permit a manifest without a verified empty/negative sample; "
            "default requires one"
        ),
    )
    return parser


def _validate_builtin_accuracy_comparison(args: argparse.Namespace) -> None:
    """Reject CLI settings that give the built-ins unequal effective input."""

    if args.v2_max_frames is not None:
        raise ValueError(
            "--v2-max-frames is forbidden for accuracy comparisons because V1 "
            "must not receive more of the input than V2"
        )
    if args.v1_builtin == "legacy-video" and args.v1_roi is not None:
        raise ValueError(
            "--v1-roi is forbidden for built-in accuracy comparisons until V2 "
            "has an equivalent ROI contract"
        )
    if (
        args.v1_builtin == "legacy-video"
        and args.v2_builtin == "engine-v2-offline"
        and args.v1_frame_step != args.v2_frame_step
    ):
        raise ValueError(
            "--v1-frame-step and --v2-frame-step must match for built-in "
            "same-input accuracy comparisons"
        )


def _make_accuracy_adapter(args: argparse.Namespace, prefix: str):
    callable_spec = getattr(args, f"{prefix}_callable")
    command = getattr(args, f"{prefix}_command")
    built_in = getattr(args, f"{prefix}_builtin")
    name = getattr(args, f"{prefix}_name")
    if callable_spec:
        return CallableAccuracyAdapter.from_specification(callable_spec, name=name)
    if command:
        return CommandAccuracyAdapter(
            command,
            name=name,
            timeout_seconds=args.adapter_timeout,
        )
    if prefix == "v1" and built_in == "legacy-video":
        from tools.anpr_accuracy_adapters import (  # noqa: PLC0415
            LegacyVideoAccuracyAdapter,
            LegacyVideoAccuracyConfig,
        )

        if args.v1_duplicate_seconds < 0:
            raise ValueError("v1-duplicate-seconds cannot be negative")
        return LegacyVideoAccuracyAdapter(
            LegacyVideoAccuracyConfig(
                frame_step=args.v1_frame_step,
                max_events=args.v1_max_events,
                min_confidence=args.v1_min_confidence,
                duplicate_seconds=args.v1_duplicate_seconds,
                detector_variant=args.v1_detector_variant,
                roi=args.v1_roi,
            ),
            name=name,
        )
    if prefix == "v2" and built_in == "engine-v2-offline":
        from tools.anpr_accuracy_adapters import (  # noqa: PLC0415
            EngineV2OfflineAccuracyAdapter,
            V2OfflineAccuracyConfig,
        )

        if args.v2_detector_model is None or args.v2_ocr_model is None:
            raise ValueError(
                "--v2-detector-model and --v2-ocr-model are required with --v2-builtin"
            )
        if args.v2_duplicate_seconds < 0:
            raise ValueError("v2-duplicate-seconds cannot be negative")
        return EngineV2OfflineAccuracyAdapter(
            V2OfflineAccuracyConfig(
                detector_model=args.v2_detector_model,
                ocr_model=args.v2_ocr_model,
                backend=args.v2_backend,
                device=args.v2_device,
                detector_frame_size=args.v2_detector_frame_size,
                detector_input_size=args.v2_detector_input_size,
                detector_confidence=args.v2_detector_confidence,
                min_ocr_confidence=args.v2_min_ocr_confidence,
                duplicate_seconds=args.v2_duplicate_seconds,
                frame_step=args.v2_frame_step,
                max_frames=args.v2_max_frames,
                opencv_threads=args.v2_opencv_threads,
                allow_inference_fallback=not args.v2_no_inference_fallback,
                allow_capture_backend_fallback=args.v2_allow_capture_backend_fallback,
            ),
            name=name,
        )
    raise AssertionError(f"unhandled {prefix} accuracy adapter selection")


def _accuracy_adapter_metadata(adapter: Any) -> Mapping[str, Any]:
    metadata = getattr(adapter, "reproducibility_metadata", None)
    if callable(metadata):
        value = metadata()
        if isinstance(value, Mapping):
            return dict(value)
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {
        "schema": "bcvision.anpr.accuracy-adapter-metadata/v1",
        "adapter": str(getattr(adapter, "adapter_name", type(adapter).__name__)),
        "kind": "external-generic-adapter",
        "reproducibility_details_provided": False,
    }


def _close_accuracy_adapter(adapter: Any) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


def _run_performance(args: argparse.Namespace) -> int:
    if args.ticks_per_second < 1 or args.producer_burst < 1:
        raise ValueError("ticks-per-second and producer-burst must be at least 1")
    if args.consumer_budget is not None and args.consumer_budget < 1:
        raise ValueError("consumer-budget must be at least 1")
    if args.max_frame_age_ms < 0:
        raise ValueError("max-frame-age-ms cannot be negative")
    if args.adapter_callable:
        adapter = load_performance_adapter(args.adapter_callable, name=args.adapter_name)
    elif args.adapter_command:
        adapter = CommandPerformanceAdapter(
            args.adapter_command,
            name=args.adapter_name,
            timeout_seconds=args.adapter_timeout,
        )
    else:
        adapter = SyntheticControlPlaneAdapter(cpu_work=args.synthetic_cpu_work)
    common = {
        "include_32": args.include_32,
        "nominal_seconds": args.nominal_seconds,
        "ticks_per_second": args.ticks_per_second,
        "producer_burst": args.producer_burst,
        "consumer_budget_per_tick": args.consumer_budget,
        "max_frame_age_ms": args.max_frame_age_ms,
        "realtime_pacing": args.paced,
    }
    if args.matrix == "standard":
        report = run_standard_performance_matrices(
            adapter,
            fixed_active_cameras=args.active_cameras,
            **common,
        )
    elif args.matrix == "fixed-active":
        report = run_performance_suite(
            default_camera_scenarios(
                active_cameras=args.active_cameras,
                **common,
            ),
            adapter,
        )
    else:
        report = run_performance_suite(
            all_active_camera_scenarios(**common),
            adapter,
        )
    write_performance_outputs(
        report,
        json_path=args.output_dir / "engine_v2_performance.json",
        csv_path=args.output_dir / "engine_v2_performance.csv",
    )
    print(args.output_dir / "engine_v2_performance.json")
    print(args.output_dir / "engine_v2_performance.csv")
    return 0


def _run_accuracy(args: argparse.Namespace) -> int:
    _validate_builtin_accuracy_comparison(args)
    manifest = load_accuracy_manifest(
        args.manifest,
        require_all_categories=not args.allow_partial_coverage,
        require_input_files=not args.allow_missing_input_files,
        require_negative_sample=not args.allow_no_negative,
    )
    adapters: list[Any] = []
    try:
        v1_adapter = _make_accuracy_adapter(args, "v1")
        adapters.append(v1_adapter)
        v2_adapter = _make_accuracy_adapter(args, "v2")
        adapters.append(v2_adapter)
        report = compare_accuracy_adapters(manifest, v1_adapter, v2_adapter)
        report["adapter_reproducibility"] = {
            "v1": _accuracy_adapter_metadata(v1_adapter),
            "v2": _accuracy_adapter_metadata(v2_adapter),
        }
    finally:
        for adapter in reversed(adapters):
            _close_accuracy_adapter(adapter)
    write_accuracy_outputs(
        report,
        json_path=args.output_dir / "v1_v2_accuracy.json",
        csv_path=args.output_dir / "v1_v2_accuracy_predictions.csv",
    )
    print(args.output_dir / "v1_v2_accuracy.json")
    print(args.output_dir / "v1_v2_accuracy_predictions.csv")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "performance":
        return _run_performance(args)
    if args.action == "compare-accuracy":
        return _run_accuracy(args)
    raise AssertionError(f"unhandled action: {args.action}")


if __name__ == "__main__":
    raise SystemExit(main())
