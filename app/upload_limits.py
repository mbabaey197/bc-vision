"""Fail-closed request-body limits for BC Vision HTTP mutations.

The application-level ``UploadFile`` size check happens after Starlette has
parsed the multipart envelope and may already have spooled a large request to
disk.  This middleware counts the raw ASGI ``http.request`` bytes instead.  It
must be installed as Starlette/FastAPI *user middleware* (via
``app.add_middleware``) so it can turn the private sentinel raised by the
wrapped ``receive`` callable into a 413 before Starlette's outer error
middleware sees it.

``max_body_bytes`` is the complete HTTP request-body limit, not just the video
file limit.  Callers should therefore include a small, bounded allowance for
the multipart boundary and part headers.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

ASGIMessage = dict[str, Any]
ASGIScope = dict[str, Any]
ASGIReceive = Callable[[], Awaitable[ASGIMessage]]
ASGISend = Callable[[ASGIMessage], Awaitable[None]]
ASGIApp = Callable[[ASGIScope, ASGIReceive, ASGISend], Awaitable[None]]
BodyLimit = int | Callable[[], int]

DEFAULT_VIDEO_UPLOAD_PATHS = frozenset(
    {
        "/cameras/video-upload",
        "/ai/video-test/upload",
    }
)


class _VideoUploadBodyTooLarge(Exception):
    """Private control-flow exception raised before an oversized chunk leaks."""


class _InvalidContentLength(Exception):
    """Private control-flow exception for malformed or inconsistent lengths."""


class _ResponseAlreadyStarted(Exception):
    """An unsafe downstream app started responding before consuming its body."""


def _json_body(error: str) -> bytes:
    return json.dumps(
        {"error": error},
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


async def _send_json_error(
    send: ASGISend,
    *,
    status_code: int,
    error: str,
) -> None:
    body = _json_body(error)
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode("ascii")),
            ],
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
            "more_body": False,
        }
    )


def _declared_content_length(scope: ASGIScope) -> int | None:
    """Return one unambiguous Content-Length value or fail closed.

    ASGI retains duplicate headers.  Accepting the first of conflicting
    values would make the result dependent on the proxy/server in front of
    BC Vision, so duplicate/comma-separated values must all agree.
    """

    raw_values: list[bytes] = []
    for name, value in scope.get("headers", ()):  # pragma: no branch
        if bytes(name).lower() == b"content-length":
            raw_values.extend(bytes(value).split(b","))
    if not raw_values:
        return None

    parsed: list[int] = []
    for raw_value in raw_values:
        value = raw_value.strip()
        if not value or not value.isdigit():
            raise _InvalidContentLength
        try:
            parsed.append(int(value))
        except ValueError:
            # Python limits enormous decimal conversions.  Treat an abusive
            # header as malformed rather than leaking a 500 response.
            raise _InvalidContentLength from None
    if len(set(parsed)) != 1:
        raise _InvalidContentLength
    return parsed[0]


class VideoUploadBodyLimitMiddleware:
    """Limit raw bodies, with a larger capped lane for video uploads.

    The limit may be a callable so tests and runtime configuration can change
    it without rebuilding FastAPI's middleware stack.  The chunk that crosses
    the boundary is never returned to Starlette's multipart parser.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: BodyLimit,
        max_other_body_bytes: BodyLimit | None = 1024 * 1024,
        max_concurrent: int = 2,
        paths: Collection[str] = DEFAULT_VIDEO_UPLOAD_PATHS,
        path_body_limits: Mapping[str, BodyLimit] | None = None,
    ) -> None:
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise ValueError("max_concurrent must be a positive integer")
        self.app = app
        self._max_body_bytes = max_body_bytes
        self._max_other_body_bytes = max_other_body_bytes
        self._paths = frozenset(str(path) for path in paths)
        self._path_body_limits = dict(path_body_limits or {})
        if not set(self._path_body_limits).issubset(self._paths):
            raise ValueError("path_body_limits must refer to upload paths")
        # A threading semaphore is not bound to one asyncio loop.  The single
        # middleware instance therefore enforces the same process-local cap
        # across all request threads/event loops without parking a worker.
        self._upload_slots = threading.BoundedSemaphore(max_concurrent)

    @staticmethod
    def _limit(configured: BodyLimit, label: str) -> int:
        value = configured() if callable(configured) else configured
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(
                f"{label} body limit must be a non-negative integer"
            )
        return value

    async def __call__(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
    ) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = str(scope.get("method", "")).upper()
        is_video_upload = method == "POST" and scope.get("path") in self._paths
        is_other_mutation = method in {"POST", "PUT", "PATCH", "DELETE"}
        if not is_video_upload and (
            not is_other_mutation or self._max_other_body_bytes is None
        ):
            await self.app(scope, receive, send)
            return

        if is_video_upload:
            configured_limit = self._path_body_limits.get(
                str(scope.get("path")),
                self._max_body_bytes,
            )
            limit = self._limit(configured_limit, "file upload")
            too_large_error = "video-upload-too-large"
        else:
            limit = self._limit(
                self._max_other_body_bytes,
                "standard request",
            )
            too_large_error = "request-body-too-large"
        try:
            declared_length = _declared_content_length(scope)
        except _InvalidContentLength:
            await _send_json_error(
                send,
                status_code=400,
                error="invalid-content-length",
            )
            return

        if declared_length is not None and declared_length > limit:
            await _send_json_error(
                send,
                status_code=413,
                error=too_large_error,
            )
            return

        if not is_video_upload:
            await self._handle_admitted_request(
                scope,
                receive,
                send,
                declared_length=declared_length,
                limit=limit,
                too_large_error=too_large_error,
            )
            return

        # This is intentionally non-blocking: queued large bodies can still be
        # read/spooled by the server while waiting.  Reject before the first
        # receive instead and let the client retry later.
        if not self._upload_slots.acquire(blocking=False):
            await _send_json_error(
                send,
                status_code=429,
                error="video-upload-busy",
            )
            return

        try:
            await self._handle_admitted_request(
                scope,
                receive,
                send,
                declared_length=declared_length,
                limit=limit,
                too_large_error=too_large_error,
            )
        finally:
            # asyncio.CancelledError inherits BaseException on supported
            # Python versions, so release must live in a real finally block.
            self._upload_slots.release()

    async def _handle_admitted_request(
        self,
        scope: ASGIScope,
        receive: ASGIReceive,
        send: ASGISend,
        *,
        declared_length: int | None,
        limit: int,
        too_large_error: str,
    ) -> None:
        received = 0
        body_failure: str | None = None
        response_started = False

        async def limited_receive() -> ASGIMessage:
            nonlocal body_failure, received
            message = await receive()
            if message.get("type") != "http.request":
                return message

            received += len(message.get("body", b""))
            # Check the actual byte count first.  If a lying Content-Length
            # and the raw cap are crossed by the same chunk, size is the
            # security-relevant failure and the response must remain 413.
            if received > limit:
                body_failure = too_large_error
                raise _VideoUploadBodyTooLarge
            if declared_length is not None:
                if received > declared_length:
                    body_failure = "content-length-mismatch"
                    raise _InvalidContentLength
                if (
                    not message.get("more_body", False)
                    and received != declared_length
                ):
                    body_failure = "content-length-mismatch"
                    raise _InvalidContentLength
            return message

        async def guarded_send(message: ASGIMessage) -> None:
            nonlocal response_started
            # FastAPI currently converts arbitrary errors raised while parsing
            # an endpoint body to its own 400 response.  Once our receive
            # wrapper has identified an over-limit body, suppress that response
            # and emit the authoritative 413 below.  This is why checking only
            # an exception around ``self.app`` is insufficient.
            if body_failure is not None:
                return
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except _VideoUploadBodyTooLarge:
            pass
        except _InvalidContentLength:
            pass

        if body_failure is not None and response_started:
            # This cannot happen for FastAPI UploadFile parameters, which are
            # parsed before entering the endpoint.  For a custom ASGI app that
            # responds before reading, abort rather than append a second HTTP
            # response to an already-started one.
            raise _ResponseAlreadyStarted(
                "video upload response started before request body was validated"
            )
        if body_failure == too_large_error:
            await _send_json_error(
                send,
                status_code=413,
                error=too_large_error,
            )
        elif body_failure == "content-length-mismatch":
            await _send_json_error(
                send,
                status_code=400,
                error="content-length-mismatch",
            )
