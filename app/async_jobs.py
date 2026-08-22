"""Cancellation-safe helpers for blocking jobs that mutate durable storage."""

import asyncio
import importlib
import inspect
import math
import multiprocessing
import operator
import os
import pickle
import re
import secrets
import stat
import tempfile
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import ParamSpec, TypeVar

from app.file_identity import descriptor_file_identity, path_file_identity

P = ParamSpec("P")
R = TypeVar("R")

_MODULE_NAME_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_RESULT_PROTOCOL_VERSION = 1
_DEFAULT_MAX_RESULT_BYTES = 64 * 1024 * 1024
_MAX_RESULT_BYTES = 512 * 1024 * 1024
_MIN_RESULT_BYTES = 1024 * 1024


class SubprocessJobError(RuntimeError):
    """A module job failed in its isolated worker process."""

    def __init__(
        self,
        message: str,
        *,
        remote_exception_type: str = "",
        remote_traceback: str = "",
    ) -> None:
        super().__init__(message)
        self.remote_exception_type = remote_exception_type
        self.remote_traceback = remote_traceback


class SubprocessJobTimeout(SubprocessJobError, TimeoutError):
    """A module job exceeded its wall-clock deadline and was killed."""


class SubprocessJobProtocolError(SubprocessJobError):
    """A worker exited without producing a trustworthy result envelope."""


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    inode: int

    @classmethod
    def from_path(cls, path, value: os.stat_result) -> "_FileIdentity":
        device, inode = path_file_identity(path, details=value)
        return cls(device=device, inode=inode)

    @classmethod
    def from_descriptor(
        cls,
        descriptor: int,
        value: os.stat_result,
    ) -> "_FileIdentity":
        device, inode = descriptor_file_identity(
            descriptor,
            details=value,
        )
        return cls(device=device, inode=inode)

    def matches_path(self, path, value: os.stat_result) -> bool:
        return path_file_identity(path, details=value) == (
            self.device,
            self.inode,
        )

    def matches_descriptor(
        self,
        descriptor: int,
        value: os.stat_result,
    ) -> bool:
        return descriptor_file_identity(
            descriptor,
            details=value,
        ) == (self.device, self.inode)


class _BoundedWriter:
    def __init__(self, raw, limit: int) -> None:
        self.raw = raw
        self.limit = limit
        self.written = 0

    def write(self, value: bytes) -> int:
        next_size = self.written + len(value)
        if next_size > self.limit:
            raise ValueError(
                f"module-job result exceeds {self.limit} bytes"
            )
        written = self.raw.write(value)
        self.written += written
        return written


class _BoundedReader:
    def __init__(self, raw, limit: int) -> None:
        self.raw = raw
        self.remaining = limit

    def _consume(self, value: bytes) -> bytes:
        self.remaining -= len(value)
        if self.remaining < 0:
            raise pickle.UnpicklingError(
                "module-job result exceeds its configured bound"
            )
        return value

    def read(self, size: int = -1) -> bytes:
        if size < 0 or size > self.remaining + 1:
            size = self.remaining + 1
        return self._consume(self.raw.read(size))

    def readline(self, size: int = -1) -> bytes:
        if size < 0 or size > self.remaining + 1:
            size = self.remaining + 1
        return self._consume(self.raw.readline(size))


class _SafeResultUnpickler(pickle.Unpickler):
    """Decode data-only pickle opcodes without importing reducer globals."""

    def find_class(self, module: str, name: str) -> object:
        raise pickle.UnpicklingError(
            f"module-job result requested forbidden global {module}.{name}"
        )

    def persistent_load(self, persistent_id: object) -> object:
        raise pickle.UnpicklingError(
            f"module-job result used forbidden persistent id {persistent_id!r}"
        )


def _regular_owned_file(
    value: os.stat_result,
    identity: _FileIdentity,
    *,
    path=None,
    descriptor: int | None = None,
) -> bool:
    return (
        stat.S_ISREG(value.st_mode)
        and int(value.st_nlink) == 1
        and (
            identity.matches_descriptor(descriptor, value)
            if descriptor is not None
            else identity.matches_path(path, value)
        )
    )


def _private_owned_directory(
    value: os.stat_result,
    identity: _FileIdentity,
    *,
    path,
) -> bool:
    return (
        stat.S_ISDIR(value.st_mode)
        and identity.matches_path(path, value)
        and (
            os.name == "nt"
            or stat.S_IMODE(value.st_mode) & 0o077 == 0
        )
    )


def _open_flags(base: int) -> int:
    return (
        base
        | getattr(os, "O_BINARY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _safe_error_text(error: BaseException, *, limit: int = 131_072) -> str:
    try:
        value = str(error)
    except BaseException:  # noqa: BLE001 - even hostile exception rendering
        value = "<exception message could not be rendered>"
    return value[:limit]


def _safe_traceback(error: BaseException, *, limit: int = 262_144) -> str:
    try:
        value = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
    except BaseException:  # noqa: BLE001 - traceback rendering can fail
        value = "<remote traceback could not be rendered>"
    return value[-limit:]


def _error_envelope(
    nonce: str,
    error: BaseException,
    *,
    phase: str,
) -> dict[str, object]:
    error_type = type(error)
    return {
        "version": _RESULT_PROTOCOL_VERSION,
        "nonce": nonce,
        "status": "error",
        "phase": phase,
        "exception_module": str(getattr(error_type, "__module__", "")),
        "exception_type": str(getattr(error_type, "__qualname__", "Error")),
        "message": _safe_error_text(error),
        "traceback": _safe_traceback(error),
    }


def _open_verified_result(
    result_path: str,
    directory_identity: _FileIdentity,
    identity: _FileIdentity,
    flags: int,
) -> int:
    directory = Path(result_path).parent
    directory_before = directory.lstat()
    if not _private_owned_directory(
        directory_before,
        directory_identity,
        path=directory,
    ):
        raise RuntimeError("module-job workspace identity changed")
    before = os.lstat(result_path)
    if not _regular_owned_file(before, identity, path=result_path):
        raise RuntimeError("module-job result path identity changed")
    descriptor = os.open(result_path, _open_flags(flags))
    try:
        opened = os.fstat(descriptor)
        directory_after = directory.lstat()
        after = os.lstat(result_path)
        if not (
            _private_owned_directory(
                directory_after,
                directory_identity,
                path=directory,
            )
            and _regular_owned_file(
                opened,
                identity,
                descriptor=descriptor,
            )
            and _regular_owned_file(after, identity, path=result_path)
        ):
            raise RuntimeError("module-job result path identity changed")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _write_result_envelope(
    result_path: str,
    directory_identity: _FileIdentity,
    identity: _FileIdentity,
    envelope: dict[str, object],
    max_result_bytes: int,
) -> None:
    descriptor = _open_verified_result(
        result_path,
        directory_identity,
        identity,
        os.O_WRONLY,
    )
    try:
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "wb") as result_file:
            descriptor = -1
            bounded_result = _BoundedWriter(result_file, max_result_bytes)
            pickle.dump(
                envelope,
                bounded_result,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            result_file.flush()
            os.fsync(result_file.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _resolve_module_function(module_name: str, function_name: str) -> Callable:
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    if not (inspect.isfunction(function) or inspect.isbuiltin(function)):
        raise TypeError(
            f"{module_name}.{function_name} is not a module-level function"
        )
    if (
        getattr(function, "__module__", None) != module_name
        or getattr(function, "__name__", None) != function_name
    ):
        raise TypeError(
            f"{module_name}.{function_name} is not a module-level function"
        )
    return function


def _validate_transport_value(
    value: object,
    *,
    active: set[int] | None = None,
    depth: int = 0,
) -> None:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return
    if depth >= 64:
        raise TypeError("module-job result nesting exceeds 64 levels")
    if not isinstance(value, (list, tuple, dict)):
        raise TypeError(
            "module-job results may contain only data primitives, lists, "
            "tuples, and string-keyed dictionaries"
        )
    active = active if active is not None else set()
    identity = id(value)
    if identity in active:
        raise TypeError("module-job result contains a reference cycle")
    active.add(identity)
    try:
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(
                        "module-job result dictionaries require string keys"
                    )
                _validate_transport_value(
                    item,
                    active=active,
                    depth=depth + 1,
                )
        else:
            for item in value:
                _validate_transport_value(
                    item,
                    active=active,
                    depth=depth + 1,
                )
    finally:
        active.remove(identity)


def _module_job_worker(
    module_name: str,
    function_name: str,
    args: tuple[object, ...],
    kwargs: dict[str, object],
    result_path: str,
    directory_identity: _FileIdentity,
    result_identity: _FileIdentity,
    nonce: str,
    max_result_bytes: int,
) -> None:
    """Spawn-compatible worker entry point (never pass a caller's callable)."""

    try:
        function = _resolve_module_function(module_name, function_name)
        value = function(*args, **kwargs)
    except BaseException as error:  # noqa: BLE001 - transport remote failures
        _write_result_envelope(
            result_path,
            directory_identity,
            result_identity,
            _error_envelope(nonce, error, phase="execute"),
            max_result_bytes,
        )
        return

    envelope: dict[str, object] = {
        "version": _RESULT_PROTOCOL_VERSION,
        "nonce": nonce,
        "status": "ok",
        "value": value,
    }
    try:
        _validate_transport_value(value)
        _write_result_envelope(
            result_path,
            directory_identity,
            result_identity,
            envelope,
            max_result_bytes,
        )
    except BaseException as error:  # noqa: BLE001 - report serialization
        # A return value can be unpickleable even though the job itself
        # succeeded.  Replace the partial payload with a small, deterministic
        # failure envelope instead of relying on Process exit state.
        _write_result_envelope(
            result_path,
            directory_identity,
            result_identity,
            _error_envelope(nonce, error, phase="serialize-result"),
            max_result_bytes,
        )


def _create_result_workspace() -> tuple[Path, _FileIdentity, Path, _FileIdentity]:
    directory = Path(tempfile.mkdtemp(prefix="bcvision-module-job-"))
    directory_identity: _FileIdentity | None = None
    result_path: Path | None = None
    result_identity: _FileIdentity | None = None
    try:
        directory_stat = directory.lstat()
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise RuntimeError("module-job workspace is not a directory")
        directory_identity = _FileIdentity.from_path(
            directory,
            directory_stat,
        )
        os.chmod(directory, 0o700)
        directory_after_chmod = directory.lstat()
        if not _private_owned_directory(
            directory_after_chmod,
            directory_identity,
            path=directory,
        ):
            raise RuntimeError("module-job workspace identity changed")

        result_path = directory / f"result-{secrets.token_hex(16)}.bin"
        descriptor = os.open(
            result_path,
            _open_flags(os.O_RDWR | os.O_CREAT | os.O_EXCL),
            0o600,
        )
        try:
            result_stat = os.fstat(descriptor)
            result_identity = _FileIdentity.from_descriptor(
                descriptor,
                result_stat,
            )
            directory_after_create = directory.lstat()
            if not (
                _private_owned_directory(
                    directory_after_create,
                    directory_identity,
                    path=directory,
                )
                and _regular_owned_file(
                    result_stat,
                    result_identity,
                    descriptor=descriptor,
                )
            ):
                raise RuntimeError(
                    "module-job result workspace identity changed"
                )
        finally:
            os.close(descriptor)
        return directory, directory_identity, result_path, result_identity
    except BaseException:
        if result_path is not None and result_identity is not None:
            _unlink_if_owned(result_path, result_identity)
        if directory_identity is not None:
            try:
                current = directory.lstat()
            except FileNotFoundError:
                pass
            else:
                if _private_owned_directory(
                    current,
                    directory_identity,
                    path=directory,
                ):
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
        raise


def _unlink_if_owned(path: Path, identity: _FileIdentity) -> None:
    try:
        value = path.lstat()
    except FileNotFoundError:
        return
    if _regular_owned_file(value, identity, path=path):
        path.unlink()


def _cleanup_result_workspace(
    directory: Path,
    directory_identity: _FileIdentity,
    result_path: Path,
    result_identity: _FileIdentity,
) -> None:
    _unlink_if_owned(result_path, result_identity)
    try:
        value = directory.lstat()
    except FileNotFoundError:
        return
    if _private_owned_directory(
        value,
        directory_identity,
        path=directory,
    ):
        try:
            directory.rmdir()
        except OSError:
            # Never recurse through or delete an unexpected foreign entry.
            pass


def _terminate_and_reap(
    process: multiprocessing.Process,
    terminate_grace_seconds: float,
) -> None:
    if process.exitcode is None:
        try:
            process.terminate()
        except ProcessLookupError:
            pass
        process.join(terminate_grace_seconds)
    if process.exitcode is None:
        kill = getattr(process, "kill", process.terminate)
        try:
            kill()
        except ProcessLookupError:
            pass
        # Do not release a storage gate until the OS child is fully reaped.
        process.join()
    else:
        process.join()


async def _wait_until_reaped(
    process: multiprocessing.Process,
    timeout_seconds: float,
) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    while process.is_alive():
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise asyncio.TimeoutError
        await asyncio.sleep(min(0.02, remaining))
    process.join()


def _read_result_envelope(
    result_path: Path,
    directory_identity: _FileIdentity,
    result_identity: _FileIdentity,
    max_result_bytes: int,
) -> object:
    descriptor = _open_verified_result(
        str(result_path),
        directory_identity,
        result_identity,
        os.O_RDONLY,
    )
    try:
        result_stat = os.fstat(descriptor)
        if int(result_stat.st_size) > max_result_bytes:
            raise SubprocessJobProtocolError(
                "module-job result exceeds its configured bound"
            )
        with os.fdopen(descriptor, "rb") as result_file:
            descriptor = -1
            bounded_result = _BoundedReader(result_file, max_result_bytes)
            envelope = _SafeResultUnpickler(bounded_result).load()
            if bounded_result.read(1):
                raise SubprocessJobProtocolError(
                    "module-job result contains trailing data"
                )
            return envelope
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _validate_job_name(module_name: str, function_name: str) -> None:
    if not isinstance(module_name, str) or not _MODULE_NAME_RE.fullmatch(
        module_name
    ):
        raise ValueError("module_name must be a dotted Python module name")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError("function_name must be one Python identifier")


async def run_module_job_subprocess(
    module_name: str,
    function_name: str,
    *args: object,
    timeout_seconds: float,
    terminate_grace_seconds: float = 0.5,
    max_result_bytes: int = _DEFAULT_MAX_RESULT_BYTES,
    **kwargs: object,
) -> object:
    """Run one module-level function in a killable spawned process.

    The worker is named by module and function, rather than accepting an
    arbitrary callable.  Its response is written to a private, pre-created
    identity-checked file, so a large response cannot deadlock process reap as
    a full ``multiprocessing.Queue`` or pipe can.  Timeout and cancellation
    both terminate, kill if necessary, and fully reap the child before this
    coroutine returns control to its caller.
    """

    _validate_job_name(module_name, function_name)
    timeout_seconds = float(timeout_seconds)
    terminate_grace_seconds = float(terminate_grace_seconds)
    if isinstance(max_result_bytes, bool):
        raise TypeError("max_result_bytes must be an integer byte count")
    try:
        max_result_bytes = operator.index(max_result_bytes)
    except TypeError as error:
        raise TypeError(
            "max_result_bytes must be an integer byte count"
        ) from error
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero")
    if (
        not math.isfinite(terminate_grace_seconds)
        or terminate_grace_seconds < 0
    ):
        raise ValueError(
            "terminate_grace_seconds must be finite and non-negative"
        )
    if not _MIN_RESULT_BYTES <= max_result_bytes <= _MAX_RESULT_BYTES:
        raise ValueError(
            f"max_result_bytes must be between {_MIN_RESULT_BYTES} and "
            f"{_MAX_RESULT_BYTES}"
        )

    workspace = _create_result_workspace()
    directory, directory_identity, result_path, result_identity = workspace
    nonce = secrets.token_hex(32)
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_module_job_worker,
        args=(
            module_name,
            function_name,
            tuple(args),
            dict(kwargs),
            str(result_path),
            directory_identity,
            result_identity,
            nonce,
            max_result_bytes,
        ),
        name=f"bcvision-job-{function_name}",
    )
    started = False
    try:
        try:
            process.start()
            started = True
        except BaseException as error:
            if process.pid is not None:
                _terminate_and_reap(process, terminate_grace_seconds)
                started = True
            raise SubprocessJobError(
                f"could not start module job {module_name}.{function_name}: "
                f"{_safe_error_text(error)}"
            ) from error

        try:
            await _wait_until_reaped(process, timeout_seconds)
        except asyncio.CancelledError as cancellation:
            try:
                _terminate_and_reap(process, terminate_grace_seconds)
            except BaseException as cleanup_error:
                raise cancellation from cleanup_error
            raise
        except asyncio.TimeoutError as error:
            _terminate_and_reap(process, terminate_grace_seconds)
            raise SubprocessJobTimeout(
                f"module job {module_name}.{function_name} exceeded "
                f"{timeout_seconds:g} seconds"
            ) from error

        exit_code = process.exitcode
        if exit_code != 0:
            raise SubprocessJobProtocolError(
                f"module job {module_name}.{function_name} exited with "
                f"code {exit_code} before returning a result"
            )
        try:
            envelope = _read_result_envelope(
                result_path,
                directory_identity,
                result_identity,
                max_result_bytes,
            )
        except SubprocessJobError:
            raise
        except BaseException as error:
            raise SubprocessJobProtocolError(
                f"module job {module_name}.{function_name} returned an "
                "invalid result"
            ) from error
        if not isinstance(envelope, dict):
            raise SubprocessJobProtocolError(
                "module-job result envelope is not a dictionary"
            )
        if (
            envelope.get("version") != _RESULT_PROTOCOL_VERSION
            or envelope.get("nonce") != nonce
        ):
            raise SubprocessJobProtocolError(
                "module-job result envelope identity is invalid"
            )
        status = envelope.get("status")
        if status == "ok" and "value" in envelope:
            return envelope["value"]
        if status == "error":
            exception_type = str(envelope.get("exception_type") or "Error")
            phase = str(envelope.get("phase") or "execute")
            message = str(envelope.get("message") or "remote job failed")
            raise SubprocessJobError(
                f"module job {module_name}.{function_name} failed during "
                f"{phase} ({exception_type}): {message}",
                remote_exception_type=exception_type,
                remote_traceback=str(envelope.get("traceback") or ""),
            )
        raise SubprocessJobProtocolError(
            "module-job result envelope status is invalid"
        )
    finally:
        if started and process.exitcode is None:
            _terminate_and_reap(process, terminate_grace_seconds)
        if started:
            process.close()
        _cleanup_result_workspace(
            directory,
            directory_identity,
            result_path,
            result_identity,
        )


async def run_to_thread_quiescent(
    function: Callable[P, R],
    *args: P.args,
    **kwargs: P.kwargs,
) -> R:
    """Run a blocking job without releasing its caller on cancellation.

    ``asyncio.to_thread`` cannot stop the underlying OS thread.  Propagating
    cancellation immediately would therefore let an outer storage gate or
    cleanup scope close while the worker is still writing.  This helper
    shields the worker, waits until it has really stopped, consumes its final
    result/exception, and only then re-raises the caller's cancellation.
    """

    worker = asyncio.create_task(
        asyncio.to_thread(function, *args, **kwargs)
    )
    cancellation: asyncio.CancelledError | None = None
    while True:
        try:
            result = await asyncio.shield(worker)
            break
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            if not worker.done():
                continue
            try:
                result = worker.result()
            except BaseException as worker_error:
                raise cancellation from worker_error
            break
        except BaseException as worker_error:
            if cancellation is not None:
                raise cancellation from worker_error
            raise

    if cancellation is not None:
        raise cancellation
    return result
