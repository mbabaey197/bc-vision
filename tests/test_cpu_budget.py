import os

from app.cpu_budget import (
    configure_process_cpu_budget,
    parallel_camera_limit,
    threads_per_camera,
)


def test_each_camera_is_hard_capped_at_two_threads(monkeypatch):
    monkeypatch.setenv("BCVISION_CPU_THREADS", "12")

    assert configure_process_cpu_budget() == 2
    assert threads_per_camera() == 2
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "TBB_NUM_THREADS",
    ):
        assert os.environ[name] == "2"
    assert os.environ["OMP_WAIT_POLICY"] == "PASSIVE"
    assert os.environ["KMP_BLOCKTIME"] == "0"


def test_per_camera_budget_can_be_reduced_to_one(monkeypatch):
    monkeypatch.setenv("BCVISION_CPU_THREADS", "1")
    assert configure_process_cpu_budget() == 1
    assert threads_per_camera() == 1


def test_parallel_camera_limit_reserves_host_capacity(monkeypatch):
    monkeypatch.setenv("BCVISION_CPU_THREADS", "2")
    monkeypatch.delenv("BCVISION_PARALLEL_CAMERAS", raising=False)

    assert parallel_camera_limit(cpu_count=4) == 1
    assert parallel_camera_limit(cpu_count=8) == 3
    assert parallel_camera_limit(cpu_count=32) == 4

    monkeypatch.setenv("BCVISION_PARALLEL_CAMERAS", "2")
    assert parallel_camera_limit(cpu_count=8) == 2
