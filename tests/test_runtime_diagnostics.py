import asyncio
import threading
import time

from app import runtime_diagnostics as diagnostics


def test_stalled_operation_is_logged_without_event_loop_stall(tmp_path):
    async def run():
        monitor = diagnostics.RuntimeDiagnostics(tmp_path)
        await monitor.start()
        entered = threading.Event()
        release = threading.Event()

        @diagnostics.diagnosed('blocked-test', timeout=0)
        def blocked():
            entered.set()
            release.wait(3)

        thread = threading.Thread(target=blocked)
        thread.start()
        try:
            assert entered.wait(1)
            monitor.last_dump = -1000
            monitor._check()
            monitor._check()
            content = (tmp_path / 'BCVision-runtime.log').read_text()
            assert content.count('Stall:') == 1
            assert 'blocked-test' in content
            assert 'Thread ' in content
        finally:
            release.set()
            thread.join(2)
            await monitor.stop()
        assert not diagnostics._operations
        assert not monitor.thread.is_alive()
    asyncio.run(run())


def test_stalled_heartbeat_is_logged(tmp_path):
    async def run():
        monitor = diagnostics.RuntimeDiagnostics(tmp_path)
        await monitor.start()
        try:
            monitor.heartbeat = time.monotonic() - 20
            monitor.last_dump = -1000
            monitor._check()
            assert 'heartbeat_age=20.' in (tmp_path / 'BCVision-runtime.log').read_text()
        finally:
            await monitor.stop()
    asyncio.run(run())
