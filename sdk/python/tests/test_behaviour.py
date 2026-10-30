"""Errors, retries, readiness, auth, the per-model bound and the async client."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import threading
import time

import pytest

from conftest import FAST_RETRY
from hypnos_client import (
    NO_RETRY,
    AsyncClient,
    AuthenticationError,
    BadRequestError,
    Client,
    InsufficientMemoryError,
    ModelBusyError,
    NotFoundError,
    RateLimitError,
    ServerError,
    ServiceUnavailableError,
)
from hypnos_client.errors import APIConnectionError

CHAT = "gemma4:26b-q4"
EMBED = "gemmaembedding2"
USER = [{"role": "user", "content": "hi"}]


def test_unknown_model_is_not_found(client: Client) -> None:
    with pytest.raises(NotFoundError) as err:
        client.chat("no-such-model", USER)
    assert err.value.status == 404
    assert err.value.type == "model_not_found"
    assert err.value.request_id and err.value.request_id.startswith("req-")


def test_bad_request(client: Client) -> None:
    with pytest.raises(BadRequestError):
        client.chat(CHAT, [])


def test_busy_is_retried(client: Client, server) -> None:
    server.fail(EMBED, "busy", times=2)
    assert client.embeddings(EMBED, "a")["data"]


def test_busy_without_retries(server) -> None:
    server.fail(EMBED, "busy")
    with Client(server.url, retry=NO_RETRY) as c, pytest.raises(ModelBusyError) as err:
        c.embeddings(EMBED, "a")
    assert err.value.status == 429 and err.value.type == "model_busy"


def test_memory_errors_are_retried(client: Client, server) -> None:
    server.fail(EMBED, "insufficient_memory", times=2)
    assert client.embeddings(EMBED, "a")["data"]


def test_memory_error_surfaces_after_the_retries(server) -> None:
    server.fail(EMBED, "insufficient_memory", times=10)
    with Client(server.url, retry=FAST_RETRY) as c, pytest.raises(InsufficientMemoryError) as err:
        c.embeddings(EMBED, "a")
    assert err.value.status == 507


def test_a_500_is_retried_once(client: Client, server) -> None:
    server.fail(EMBED, "server_error", times=1)
    assert client.embeddings(EMBED, "a")["data"]
    server.fail(EMBED, "server_error", times=2)
    with pytest.raises(ServerError):
        client.embeddings(EMBED, "a")


def test_not_ready(server) -> None:
    server.control("ready", {"ready": False})
    with Client(server.url, retry=NO_RETRY) as c:
        assert not c.is_ready()
        with pytest.raises(ServiceUnavailableError) as err:
            c.embeddings(EMBED, "a")
        assert err.value.retry_after == 5
        # Models and the memory status stay open while staging.
        assert c.models()["data"]
        threading.Timer(0.3, lambda: server.control("ready", {"ready": True})).start()
        c.wait_until_ready(timeout=5, interval=0.05)
        assert c.embeddings(EMBED, "a")["data"]


def test_unreachable_server() -> None:
    with Client("http://127.0.0.1:9", retry=NO_RETRY) as c, pytest.raises(APIConnectionError):
        c.health()
    with Client("http://127.0.0.1:9") as c:
        assert not c.is_ready()
        with pytest.raises(TimeoutError):
            c.wait_until_ready(timeout=0.2, interval=0.05)


def test_api_keys(keyed_server) -> None:
    with Client(keyed_server.url, api_key=None, retry=NO_RETRY) as anonymous:
        with pytest.raises(AuthenticationError) as err:
            anonymous.models()
        assert err.value.code == "invalid_api_key"
        assert anonymous.health()["status"] == "ok"
    with Client(keyed_server.url, api_key="secret") as c:
        assert c.models()["data"]


def test_rate_limit(keyed_server) -> None:
    with Client(keyed_server.url, api_key="limited", retry=NO_RETRY) as c:
        c.models()
        with pytest.raises(RateLimitError) as err:
            c.models()
    assert err.value.code == "requests_per_minute"
    assert err.value.retry_after is not None and err.value.retry_after > 0


def test_base_url_accepts_v1(server) -> None:
    with Client(server.url + "/v1/") as c:
        assert c.base_url == server.url
        assert c.models()["data"]


def test_per_model_bound(server) -> None:
    with Client(server.url, max_concurrency_per_model=1) as c:
        stream = c.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=1000)
        done = threading.Event()

        def second() -> None:
            c.chat(CHAT, USER)
            done.set()

        worker = threading.Thread(target=second)
        worker.start()
        # The second request waits for the open stream's permit.
        assert not done.wait(0.3)
        stream.close()
        assert done.wait(5)
        worker.join()
        # Another model is not affected by the bound.
        assert c.embeddings(EMBED, "a")["data"]


def test_the_sync_client_is_generated_from_the_async_one() -> None:
    script = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "unasync.py"
    assert subprocess.run([sys.executable, str(script), "--check"]).returncode == 0


@pytest.mark.anyio
async def test_async_client(server) -> None:
    async with AsyncClient(server.url, retry=FAST_RETRY) as c:
        assert await c.is_ready()
        reply = await c.chat(CHAT, USER)
        assert reply["choices"][0]["message"]["content"] == "echo: hi"
        stream = await c.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=3)
        assert "".join([p async for p in stream.text()]) == "1 2 3"
        assert stream.usage is not None
        server.fail(EMBED, "busy")
        assert (await c.embeddings(EMBED, ["a", "b"]))["data"]
        doc = await (await c.convert_document_stream("docling", b"%PDF", pages="1-2")).final()
        assert [p["page"] for p in doc["pages"]] == [1, 2]
        speech = await c.speech("kokoro-82m-q4", "one", response_format="pcm")
        assert len(speech.audio) == 2400


@pytest.mark.anyio
async def test_async_cancel_and_bound(server) -> None:
    async with AsyncClient(server.url, max_concurrency_per_model=1) as c:
        started = time.monotonic()
        stream = await c.chat_stream(CHAT, [{"role": "user", "content": "count"}], max_tokens=2000)
        async for i, _ in _enumerate(stream.text()):
            if i == 2:
                break
        await stream.aclose()
        assert time.monotonic() - started < 5
        # The permit came back with the closed stream.
        assert (await c.chat(CHAT, USER))["choices"]


async def _enumerate(it):
    i = 0
    async for x in it:
        yield i, x
        i += 1
