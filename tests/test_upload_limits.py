import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated

import pytest
from fastapi import FastAPI, File, UploadFile

from app.upload_limits import VideoUploadBodyLimitMiddleware

TARGET_PATH = "/ai/video-test/upload"


def _scope(*, path=TARGET_PATH, method="POST", headers=()):
    return {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "scheme": "http",
        "method": method,
        "root_path": "",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": list(headers),
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }


def _run_asgi(app, *, chunks, path=TARGET_PATH, method="POST", headers=()):
    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]
    if not messages:
        messages.append(
            {"type": "http.request", "body": b"", "more_body": False}
        )
    sent = []
    receive_calls = 0

    async def receive():
        nonlocal receive_calls
        receive_calls += 1
        if messages:
            return messages.pop(0)
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    asyncio.run(
        app(
            _scope(path=path, method=method, headers=headers),
            receive,
            send,
        )
    )
    return sent, receive_calls


def _response(sent):
    start = next(message for message in sent if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return start["status"], body


async def _invoke_asgi(app, receive, *, path=TARGET_PATH, headers=()):
    sent = []

    async def send(message):
        sent.append(message)

    await app(_scope(path=path, headers=headers), receive, send)
    return sent


def _consuming_app(observed):
    async def app(_scope, receive, send):
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] != "http.request":
                break
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        observed.append(bytes(body))
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    return app


def test_chunked_body_without_content_length_is_stopped_at_raw_limit():
    observed = []
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app(observed),
        max_body_bytes=8,
    )

    sent, receive_calls = _run_asgi(
        app,
        chunks=[b"1234", b"5678", b"9"],
        headers=[(b"transfer-encoding", b"chunked")],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "video-upload-too-large"}
    assert receive_calls == 3
    # The over-limit chunk is rejected before the application can consume it.
    assert observed == []


def test_underreported_content_length_cannot_bypass_raw_limit():
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app([]),
        max_body_bytes=8,
    )

    sent, _ = _run_asgi(
        app,
        chunks=[b"1234", b"56789"],
        headers=[(b"content-length", b"4")],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "video-upload-too-large"}


def test_underreported_content_length_below_cap_is_rejected_as_malformed():
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app([]),
        max_body_bytes=8,
    )

    sent, _ = _run_asgi(
        app,
        chunks=[b"123", b"4"],
        headers=[(b"content-length", b"3")],
    )

    status, body = _response(sent)
    assert status == 400
    assert json.loads(body) == {"error": "content-length-mismatch"}


def test_body_exactly_at_raw_limit_is_accepted():
    observed = []
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app(observed),
        max_body_bytes=8,
    )

    sent, _ = _run_asgi(
        app,
        chunks=[b"123", b"45678"],
        headers=[(b"content-length", b"8")],
    )

    status, _ = _response(sent)
    assert status == 204
    assert observed == [b"12345678"]


def test_declared_oversize_is_rejected_without_reading_request_body():
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app([]),
        max_body_bytes=8,
    )

    sent, receive_calls = _run_asgi(
        app,
        chunks=[b"unread"],
        headers=[(b"content-length", b"9")],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "video-upload-too-large"}
    assert receive_calls == 0


def test_non_target_request_is_not_changed_even_when_larger_than_limit():
    observed = []
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app(observed),
        max_body_bytes=2,
    )

    sent, _ = _run_asgi(
        app,
        path="/api/unrelated",
        chunks=[b"larger-than-two"],
    )

    status, _ = _response(sent)
    assert status == 204
    assert observed == [b"larger-than-two"]


def test_standard_mutation_has_its_own_raw_body_limit():
    observed = []
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app(observed),
        max_body_bytes=100,
        max_other_body_bytes=4,
    )

    sent, _ = _run_asgi(
        app,
        path="/settings/display",
        chunks=[b"1234", b"5"],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "request-body-too-large"}
    assert observed == []


def test_declared_standard_oversize_is_rejected_before_receive():
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app([]),
        max_body_bytes=100,
        max_other_body_bytes=4,
    )

    sent, receive_calls = _run_asgi(
        app,
        path="/login",
        chunks=[b"unread"],
        headers=[(b"content-length", b"5")],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "request-body-too-large"}
    assert receive_calls == 0


def test_get_on_target_path_is_not_changed():
    observed = []
    app = VideoUploadBodyLimitMiddleware(
        _consuming_app(observed),
        max_body_bytes=2,
    )

    sent, _ = _run_asgi(
        app,
        method="GET",
        chunks=[b"larger-than-two"],
    )

    status, _ = _response(sent)
    assert status == 204
    assert observed == [b"larger-than-two"]


def test_real_fastapi_multipart_parse_returns_413_not_internal_error():
    fastapi_app = FastAPI()
    endpoint_called = []

    @fastapi_app.post(TARGET_PATH)
    async def upload(video: Annotated[UploadFile, File()]):
        endpoint_called.append(video.filename)
        return {"ok": True}

    boundary = b"bcvision-boundary"
    multipart_body = b"".join(
        [
            b"--" + boundary + b"\r\n",
            b'Content-Disposition: form-data; name="video"; filename="x.mp4"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            b"0123456789",
            b"\r\n--" + boundary + b"--\r\n",
        ]
    )
    # Install as user middleware: this position lets it translate the receive
    # sentinel before Starlette's outer ServerErrorMiddleware emits a 500.
    fastapi_app.add_middleware(
        VideoUploadBodyLimitMiddleware,
        max_body_bytes=len(multipart_body) - 1,
    )

    sent, _ = _run_asgi(
        fastapi_app,
        chunks=[multipart_body[:32], multipart_body[32:]],
        headers=[
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
        ],
    )

    status, body = _response(sent)
    assert status == 413
    assert json.loads(body) == {"error": "video-upload-too-large"}
    assert endpoint_called == []


def test_raw_limit_can_include_multipart_envelope_overhead():
    fastapi_app = FastAPI()
    endpoint_called = []

    @fastapi_app.post(TARGET_PATH)
    async def upload(video: Annotated[UploadFile, File()]):
        endpoint_called.append((video.filename, await video.read()))
        return {"ok": True}

    boundary = b"bcvision-boundary"
    file_body = b"0123456789"
    multipart_body = b"".join(
        [
            b"--" + boundary + b"\r\n",
            b'Content-Disposition: form-data; name="video"; filename="x.mp4"\r\n',
            b"Content-Type: video/mp4\r\n\r\n",
            file_body,
            b"\r\n--" + boundary + b"--\r\n",
        ]
    )
    assert len(multipart_body) > len(file_body)
    fastapi_app.add_middleware(
        VideoUploadBodyLimitMiddleware,
        # The raw cap deliberately includes the boundary and part headers.
        max_body_bytes=len(multipart_body),
    )

    sent, _ = _run_asgi(
        fastapi_app,
        chunks=[multipart_body],
        headers=[
            (
                b"content-type",
                b"multipart/form-data; boundary=" + boundary,
            ),
            (b"content-length", str(len(multipart_body)).encode("ascii")),
        ],
    )

    status, body = _response(sent)
    assert status == 200
    assert json.loads(body) == {"ok": True}
    assert endpoint_called == [("x.mp4", file_body)]


def test_default_concurrency_limit_rejects_extra_slow_body_before_receive():
    async def scenario():
        release_bodies = asyncio.Event()
        two_receivers_started = asyncio.Event()
        receiver_calls = [0, 0, 0, 0]
        started_count = 0

        async def consuming_app(_scope, receive, send):
            await receive()
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        middleware = VideoUploadBodyLimitMiddleware(
            consuming_app,
            max_body_bytes=8,
            # Exercise the production default explicitly.
        )

        def slow_receive(index):
            async def receive():
                nonlocal started_count
                receiver_calls[index] += 1
                started_count += 1
                if started_count == 2:
                    two_receivers_started.set()
                await release_bodies.wait()
                return {
                    "type": "http.request",
                    "body": b"ok",
                    "more_body": False,
                }

            return receive

        first = asyncio.create_task(_invoke_asgi(middleware, slow_receive(0)))
        second = asyncio.create_task(_invoke_asgi(middleware, slow_receive(1)))
        await asyncio.wait_for(two_receivers_started.wait(), timeout=1)

        async def rejected_receive():
            receiver_calls[2] += 1
            return {
                "type": "http.request",
                "body": b"ok",
                "more_body": False,
            }

        rejected = await _invoke_asgi(middleware, rejected_receive)
        rejected_status, rejected_body = _response(rejected)
        assert rejected_status == 429
        assert rejected_body == b'{"error":"video-upload-busy"}'
        assert json.loads(rejected_body) == {"error": "video-upload-busy"}
        assert receiver_calls[2] == 0

        async def oversized_receive():
            receiver_calls[3] += 1
            return {
                "type": "http.request",
                "body": b"unread",
                "more_body": False,
            }

        oversized = await _invoke_asgi(
            middleware,
            oversized_receive,
            headers=[(b"content-length", b"9")],
        )
        oversized_status, oversized_body = _response(oversized)
        assert oversized_status == 413
        assert json.loads(oversized_body) == {
            "error": "video-upload-too-large"
        }
        assert receiver_calls[3] == 0

        release_bodies.set()
        first_sent, second_sent = await asyncio.gather(first, second)
        assert _response(first_sent)[0] == 204
        assert _response(second_sent)[0] == 204

    asyncio.run(scenario())


def test_concurrency_slot_is_released_after_downstream_error():
    async def scenario():
        calls = 0

        async def flaky_app(_scope, receive, send):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("downstream failed")
            await receive()
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        middleware = VideoUploadBodyLimitMiddleware(
            flaky_app,
            max_body_bytes=8,
            max_concurrent=1,
        )

        async def receive():
            return {
                "type": "http.request",
                "body": b"ok",
                "more_body": False,
            }

        with pytest.raises(RuntimeError, match="downstream failed"):
            await _invoke_asgi(middleware, receive)

        retry = await _invoke_asgi(middleware, receive)
        assert _response(retry)[0] == 204

    asyncio.run(scenario())


def test_concurrency_slot_is_released_after_cancellation():
    async def scenario():
        first_started = asyncio.Event()
        never_release = asyncio.Event()
        calls = 0

        async def cancellable_app(_scope, receive, send):
            nonlocal calls
            calls += 1
            if calls == 1:
                first_started.set()
                await never_release.wait()
            await receive()
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        middleware = VideoUploadBodyLimitMiddleware(
            cancellable_app,
            max_body_bytes=8,
            max_concurrent=1,
        )

        async def receive():
            return {
                "type": "http.request",
                "body": b"ok",
                "more_body": False,
            }

        cancelled = asyncio.create_task(_invoke_asgi(middleware, receive))
        await asyncio.wait_for(first_started.wait(), timeout=1)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled

        retry = await _invoke_asgi(middleware, receive)
        assert _response(retry)[0] == 204

    asyncio.run(scenario())


def test_busy_target_does_not_consume_capacity_for_non_target_request():
    async def scenario():
        target_started = asyncio.Event()
        release_target = asyncio.Event()
        observed_paths = []

        async def slow_app(scope, receive, send):
            observed_paths.append(scope["path"])
            if scope["path"] == TARGET_PATH:
                target_started.set()
                await release_target.wait()
            await receive()
            await send(
                {
                    "type": "http.response.start",
                    "status": 204,
                    "headers": [],
                }
            )
            await send({"type": "http.response.body", "body": b""})

        middleware = VideoUploadBodyLimitMiddleware(
            slow_app,
            max_body_bytes=2,
            max_concurrent=1,
        )

        async def receive():
            return {
                "type": "http.request",
                "body": b"body-larger-than-target-limit",
                "more_body": False,
            }

        target = asyncio.create_task(_invoke_asgi(middleware, receive))
        await asyncio.wait_for(target_started.wait(), timeout=1)

        unrelated = await _invoke_asgi(
            middleware,
            receive,
            path="/api/unrelated",
        )
        assert _response(unrelated)[0] == 204

        release_target.set()
        await target
        assert observed_paths == [TARGET_PATH, "/api/unrelated"]

    asyncio.run(scenario())


def test_concurrency_limit_is_shared_across_thread_event_loops():
    first_started = threading.Event()
    release_first = threading.Event()
    second_receive_calls = 0

    async def slow_app(_scope, receive, send):
        first_started.set()
        while not release_first.is_set():
            await asyncio.sleep(0.001)
        await receive()
        await send(
            {
                "type": "http.response.start",
                "status": 204,
                "headers": [],
            }
        )
        await send({"type": "http.response.body", "body": b""})

    middleware = VideoUploadBodyLimitMiddleware(
        slow_app,
        max_body_bytes=8,
        max_concurrent=1,
    )

    async def first_receive():
        return {
            "type": "http.request",
            "body": b"ok",
            "more_body": False,
        }

    async def second_receive():
        nonlocal second_receive_calls
        second_receive_calls += 1
        return {
            "type": "http.request",
            "body": b"ok",
            "more_body": False,
        }

    pool = ThreadPoolExecutor(max_workers=2)
    try:
        first = pool.submit(
            asyncio.run,
            _invoke_asgi(middleware, first_receive),
        )
        assert first_started.wait(timeout=1)
        second = pool.submit(
            asyncio.run,
            _invoke_asgi(middleware, second_receive),
        )
        second_sent = second.result(timeout=1)
        assert _response(second_sent)[0] == 429
        assert second_receive_calls == 0

        release_first.set()
        assert _response(first.result(timeout=1))[0] == 204
    finally:
        release_first.set()
        pool.shutdown(wait=True)
