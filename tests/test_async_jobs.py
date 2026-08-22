import asyncio
import multiprocessing
import threading
import time

import pytest

from app import async_jobs
from app.async_jobs import (
    SubprocessJobError,
    SubprocessJobTimeout,
    run_module_job_subprocess,
    run_to_thread_quiescent,
)


def test_quiescent_thread_returns_result():
    async def scenario():
        return await run_to_thread_quiescent(lambda value: value + 1, 41)

    assert asyncio.run(scenario()) == 42


def test_cancellation_waits_for_blocking_writer_to_stop():
    started = threading.Event()
    release = threading.Event()
    stopped = threading.Event()

    def writer():
        started.set()
        release.wait(timeout=5)
        stopped.set()

    async def scenario():
        task = asyncio.create_task(run_to_thread_quiescent(writer))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        assert not stopped.is_set()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stopped.is_set()

    asyncio.run(scenario())


def test_cancelled_worker_exception_is_consumed_after_quiescence():
    started = threading.Event()
    release = threading.Event()

    def failing_writer():
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("write failed after disconnect")

    async def scenario():
        task = asyncio.create_task(run_to_thread_quiescent(failing_writer))
        assert await asyncio.to_thread(started.wait, 2)
        task.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert isinstance(caught.value.__cause__, RuntimeError)

    asyncio.run(scenario())


def test_module_job_returns_result_and_removes_private_workspace(
    monkeypatch,
    tmp_path,
):
    real_mkdtemp = async_jobs.tempfile.mkdtemp

    def tracked_mkdtemp(*, prefix):
        return real_mkdtemp(prefix=prefix, dir=tmp_path)

    monkeypatch.setattr(async_jobs.tempfile, "mkdtemp", tracked_mkdtemp)

    async def scenario():
        return await run_module_job_subprocess(
            "builtins",
            "sum",
            [10, 20, 12],
            timeout_seconds=10,
        )

    assert asyncio.run(scenario()) == 42
    assert list(tmp_path.iterdir()) == []


def test_module_job_wraps_remote_error_with_diagnostics():
    async def scenario():
        return await run_module_job_subprocess(
            "math",
            "sqrt",
            -1,
            timeout_seconds=10,
        )

    with pytest.raises(SubprocessJobError) as caught:
        asyncio.run(scenario())
    assert caught.value.remote_exception_type == "ValueError"
    assert "math domain error" in str(caught.value)
    assert "ValueError" in caught.value.remote_traceback


def test_module_job_timeout_kills_and_reaps_worker():
    before = {child.pid for child in multiprocessing.active_children()}

    async def scenario():
        await run_module_job_subprocess(
            "time",
            "sleep",
            30,
            timeout_seconds=0.2,
            terminate_grace_seconds=0.1,
        )

    started_at = time.monotonic()
    with pytest.raises(SubprocessJobTimeout):
        asyncio.run(scenario())
    assert time.monotonic() - started_at < 5
    assert {
        child.pid for child in multiprocessing.active_children()
    } <= before


def test_module_job_cancellation_kills_and_reaps_worker():
    before = {child.pid for child in multiprocessing.active_children()}

    async def scenario():
        task = asyncio.create_task(
            run_module_job_subprocess(
                "time",
                "sleep",
                30,
                timeout_seconds=30,
                terminate_grace_seconds=0.1,
            )
        )
        deadline = asyncio.get_running_loop().time() + 5
        spawned_pid = None
        while asyncio.get_running_loop().time() < deadline:
            children = [
                child
                for child in multiprocessing.active_children()
                if child.pid not in before
            ]
            if children:
                spawned_pid = children[0].pid
                break
            await asyncio.sleep(0.01)
        assert spawned_pid is not None
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert spawned_pid not in {
            child.pid for child in multiprocessing.active_children()
        }

    asyncio.run(scenario())


def test_module_job_large_result_cannot_block_worker_reap():
    result_size = 16 * 1024 * 1024

    async def scenario():
        return await run_module_job_subprocess(
            "_operator",
            "mul",
            bytes(1),
            result_size,
            timeout_seconds=30,
            max_result_bytes=32 * 1024 * 1024,
        )

    result = asyncio.run(scenario())
    assert len(result) == result_size
    assert result[:16] == bytes(16)
