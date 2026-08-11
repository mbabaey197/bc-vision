"""CLI for the independent BC Vision ANPR Engine V2 benchmark harness."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

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
    accuracy.add_argument(
        "--allow-missing-input-files",
        action="store_true",
        help="permit URI/adapter-resolved inputs; labels must still be verified and all categories present",
    )
    return parser


def _make_accuracy_adapter(args: argparse.Namespace, prefix: str):
    callable_spec = getattr(args, f"{prefix}_callable")
    command = getattr(args, f"{prefix}_command")
    name = getattr(args, f"{prefix}_name")
    if callable_spec:
        return CallableAccuracyAdapter.from_specification(callable_spec, name=name)
    return CommandAccuracyAdapter(
        command,
        name=name,
        timeout_seconds=args.adapter_timeout,
    )


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
    manifest = load_accuracy_manifest(
        args.manifest,
        require_all_categories=True,
        require_input_files=not args.allow_missing_input_files,
    )
    report = compare_accuracy_adapters(
        manifest,
        _make_accuracy_adapter(args, "v1"),
        _make_accuracy_adapter(args, "v2"),
    )
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
