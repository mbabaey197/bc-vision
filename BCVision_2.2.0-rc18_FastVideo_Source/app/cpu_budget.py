"""CPU scheduling limits for BC Vision ANPR.

Each camera inference may use at most two native compute threads.  The number
of cameras allowed to infer concurrently is bounded separately so adding
cameras cannot consume every logical processor.
"""
from __future__ import annotations

import os

ANPR_THREADS_PER_CAMERA = 2
MAX_PARALLEL_CAMERAS = 4

_NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "TBB_NUM_THREADS",
)


def threads_per_camera() -> int:
    """Return the configured per-camera budget, hard-clamped to one or two."""
    try:
        configured = int(
            os.environ.get(
                "BCVISION_CPU_THREADS",
                str(ANPR_THREADS_PER_CAMERA),
            )
        )
    except (TypeError, ValueError):
        configured = ANPR_THREADS_PER_CAMERA
    return max(1, min(ANPR_THREADS_PER_CAMERA, configured))


def parallel_camera_limit(cpu_count: int | None = None) -> int:
    """Bound simultaneous camera inference while reserving host capacity."""
    logical = max(1, int(cpu_count or os.cpu_count() or 1))
    threads = threads_per_camera()
    # Keep two logical processors available for video decoding, the web
    # dashboard and Windows.  Very small systems still receive one ANPR slot.
    automatic = max(1, (max(0, logical - 2)) // threads)
    automatic = min(MAX_PARALLEL_CAMERAS, automatic)
    try:
        configured = int(
            os.environ.get(
                "BCVISION_PARALLEL_CAMERAS",
                str(automatic),
            )
        )
    except (TypeError, ValueError):
        configured = automatic
    return max(1, min(MAX_PARALLEL_CAMERAS, automatic, configured))


def configure_process_cpu_budget() -> int:
    """Apply the per-operation budget before native runtimes are imported."""
    limit = threads_per_camera()
    value = str(limit)
    for name in _NATIVE_THREAD_VARIABLES:
        os.environ[name] = value

    # Avoid hot spinning between the deliberately sparse ANPR passes.
    os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
    os.environ["KMP_BLOCKTIME"] = "0"
    return limit
