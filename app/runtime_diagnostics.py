"""Bounded persistent errors and thread stacks for stalls; never records frames."""
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from functools import wraps
import sys
import threading
import time
import traceback

logger = logging.getLogger('bcvision.runtime')
_lock = threading.Lock()
_operations = {}


def diagnosed(label, timeout=30.0):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            token = object()
            with _lock:
                _operations[token] = (label, time.monotonic(), timeout)
            try:
                return function(*args, **kwargs)
            except Exception:
                logger.exception('%s failed', label)
                raise
            finally:
                with _lock:
                    _operations.pop(token, None)
        return wrapped
    return decorate


class RuntimeDiagnostics:
    def __init__(self, directory):
        self.directory = directory
        self.stop_event = threading.Event()
        self.heartbeat = time.monotonic()
        self.last_dump = 0.0
        self.handler = None
        self.task = None
        self.thread = None

    async def start(self):
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.handler = RotatingFileHandler(
                self.directory / 'BCVision-runtime.log',
                maxBytes=2 * 1024 * 1024, backupCount=2, encoding='utf-8',
            )
            self.handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
            logger.addHandler(self.handler)
            logger.setLevel(logging.INFO)
            logging.getLogger('uvicorn.error').addHandler(self.handler)
        except OSError:
            # A diagnostic failure must not prevent camera/service startup.
            return
        self.task = asyncio.create_task(self._pulse())
        self.thread = threading.Thread(target=self._watch, name='bc-watchdog', daemon=True)
        self.thread.start()
        logger.info('Runtime diagnostics started')

    async def _pulse(self):
        while not self.stop_event.is_set():
            self.heartbeat = time.monotonic()
            await asyncio.sleep(1)

    def _check(self):
        now = time.monotonic()
        with _lock:
            slow = [label for label, started, limit in _operations.values()
                    if now - started >= limit]
        if now - self.heartbeat < 15 and not slow:
            return
        if now - self.last_dump < 60:
            return
        self.last_dump = now
        stacks = []
        for ident, frame in sys._current_frames().items():
            stacks.append('Thread %s:\n%s' % (
                ident, ''.join(traceback.format_stack(frame, limit=35)),
            ))
        logger.error('Stall: heartbeat_age=%.1fs operations=%s\n%s',
                     now - self.heartbeat, slow, '\n'.join(stacks))

    def _watch(self):
        while not self.stop_event.wait(2):
            try:
                self._check()
            except Exception:
                # Never kill processing because diagnostics could not be saved.
                pass

    async def stop(self):
        self.stop_event.set()
        if self.task is not None:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass
        if self.thread is not None:
            await asyncio.to_thread(self.thread.join, 3)
        if self.handler is not None and (self.thread is None or not self.thread.is_alive()):
            logger.removeHandler(self.handler)
            logging.getLogger('uvicorn.error').removeHandler(self.handler)
            self.handler.close()
