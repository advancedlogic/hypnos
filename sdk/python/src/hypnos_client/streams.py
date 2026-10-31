"""Streaming responses: chat (SSE), document conversion (SSE) and speech (chunked audio).

Each stream holds the HTTP response open; closing it (``close()``, leaving the ``with`` block,
or dropping it) closes the connection, which is how the server is told to stop (docs/api.md
section 17, "Cancellation"). A stream that fails after it started raises
:class:`~hypnos_client.errors.StreamError` from the iteration.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, cast

import httpx

from ._base import SpeechTelemetry, iter_events, stream_error, SSEDecoder
from .types import (
    ChatCompletion,
    ChatCompletionChunk,
    Document,
    DocumentPage,
    DocumentStarted,
    Telemetry,
    ToolCall,
    Usage,
)

Release = Callable[[], None]


class _ChatState:
    def __init__(self, request_id: Optional[str]) -> None:
        self.request_id = request_id
        self.id: Optional[str] = None
        self.model: Optional[str] = None
        self.created: Optional[int] = None
        self.text = ""
        # Tool calls by `index`: id and name from the first fragment that has them, the
        # argument fragments concatenated (Hypnos sends each call whole, OpenAI in pieces).
        self.calls: Dict[int, Dict[str, Any]] = {}
        self.finish_reason: Optional[str] = None
        self.usage: Optional[Usage] = None
        self.telemetry: Optional[Telemetry] = None
        self.done = False

    def add(self, data: str) -> Optional[ChatCompletionChunk]:
        if data == "[DONE]":
            self.done = True
            return None
        chunk = cast(ChatCompletionChunk, json.loads(data))
        self.id = chunk.get("id", self.id)
        self.model = chunk.get("model", self.model)
        self.created = chunk.get("created", self.created)
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            self.text += delta.get("content") or ""
            for part in delta.get("tool_calls") or []:
                self._add_call(part)
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]
        if "usage" in chunk:
            self.usage = chunk["usage"]
        if "hypnos_telemetry" in chunk:
            self.telemetry = chunk["hypnos_telemetry"]
        if self.finish_reason == "error" or "error" in chunk:
            self.done = True
            raise stream_error(cast(Dict[str, Any], chunk), self.request_id)
        return chunk

    def _add_call(self, part: Dict[str, Any]) -> None:
        index = part.get("index", len(self.calls))
        call = self.calls.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if part.get("id") and not call["id"]:
            call["id"] = part["id"]
        if part.get("type"):
            call["type"] = part["type"]
        function = part.get("function") or {}
        if function.get("name") and not call["function"]["name"]:
            call["function"]["name"] = function["name"]
        call["function"]["arguments"] += function.get("arguments") or ""

    def tool_calls(self) -> List[ToolCall]:
        return [cast(ToolCall, self.calls[i]) for i in sorted(self.calls)]

    def completion(self) -> ChatCompletion:
        message: Dict[str, Any] = {"role": "assistant", "content": self.text}
        if self.calls:
            message["tool_calls"] = self.tool_calls()
        result: Dict[str, Any] = {
            "id": self.id or "",
            "object": "chat.completion",
            "created": self.created or 0,
            "model": self.model or "",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason or "stop",
                }
            ],
            "usage": self.usage
            or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        if self.telemetry is not None:
            result["hypnos_telemetry"] = self.telemetry
        return cast(ChatCompletion, result)


class _DocumentState:
    def __init__(self, request_id: Optional[str]) -> None:
        self.request_id = request_id
        self.started: Optional[DocumentStarted] = None
        self.pages: List[DocumentPage] = []
        self.document: Optional[Document] = None
        self.done = False

    def add(self, data: str) -> Optional[Dict[str, Any]]:
        if data == "[DONE]":
            self.done = True
            return None
        event = json.loads(data)
        kind = event.get("object")
        if kind == "document.started":
            self.started = cast(DocumentStarted, event)
        elif kind == "document.page":
            page = dict(event)
            page.pop("object", None)
            self.pages.append(cast(DocumentPage, page))
        elif kind == "document":
            # The final event leaves `pages` empty: they already came one by one.
            doc = dict(event)
            if not doc.get("pages"):
                doc["pages"] = list(self.pages)
            self.document = cast(Document, doc)
        elif kind == "error":
            self.done = True
            raise stream_error(event, self.request_id)
        return cast(Dict[str, Any], event)


class _SyncSSE:
    def __init__(self, response: httpx.Response, release: Release) -> None:
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self._release = release
        self._closed = False

    def _data(self) -> Iterator[str]:
        try:
            yield from iter_events(self.response.iter_lines())
        except httpx.TransportError as e:
            from .errors import APIConnectionError

            raise APIConnectionError(f"the stream broke: {e}") from e

    def close(self) -> None:
        """Closes the connection; the server stops at the next token, page or chunk."""
        if not self._closed:
            self._closed = True
            self.response.close()
            self._release()

    def __enter__(self) -> "_SyncSSE":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class ChatStream(_SyncSSE):
    """``POST /v1/chat/completions`` with ``stream=True``.

    Iterate it for :class:`~hypnos_client.types.ChatCompletionChunk` dicts, or over
    :meth:`text` for the content pieces only. ``text_so_far``, ``tool_calls``, ``usage``,
    ``telemetry`` and ``finish_reason`` fill in as the chunks arrive; :meth:`final` includes the
    tool calls in ``message.tool_calls``.
    """

    def __init__(self, response: httpx.Response, release: Release) -> None:
        super().__init__(response, release)
        self._state = _ChatState(self.request_id)

    @property
    def text_so_far(self) -> str:
        return self._state.text

    @property
    def usage(self) -> Optional[Usage]:
        return self._state.usage

    @property
    def telemetry(self) -> Optional[Telemetry]:
        return self._state.telemetry

    @property
    def finish_reason(self) -> Optional[str]:
        return self._state.finish_reason

    @property
    def tool_calls(self) -> List[ToolCall]:
        """The tool calls received so far, accumulated by ``index`` (complete once the stream ended)."""
        return self._state.tool_calls()

    def __iter__(self) -> Iterator[ChatCompletionChunk]:
        if self._closed:  # already read (or closed): final() returns what was collected
            return
        try:
            for data in self._data():
                chunk = self._state.add(data)
                if chunk is not None:
                    yield chunk
                if self._state.done:
                    break
        finally:
            self.close()

    def text(self) -> Iterator[str]:
        """The ``delta.content`` pieces, as they arrive."""
        for chunk in self:
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece

    def final(self) -> ChatCompletion:
        """Reads the rest of the stream and returns it as a non-streaming completion."""
        for _ in self:
            pass
        return self._state.completion()


class DocumentStream(_SyncSSE):
    """``POST /v1/documents/convert`` with ``stream=True``.

    Iterating yields every event (``document.started``, ``document.page``, ``document``) as a
    dict; :meth:`pages` yields only the pages. :meth:`final` returns the whole document with
    its pages put back.
    """

    def __init__(self, response: httpx.Response, release: Release) -> None:
        super().__init__(response, release)
        self._state = _DocumentState(self.request_id)

    @property
    def started(self) -> Optional[DocumentStarted]:
        return self._state.started

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        if self._closed:  # already read (or closed): final() returns what was collected
            return
        try:
            for data in self._data():
                event = self._state.add(data)
                if event is not None:
                    yield event
                if self._state.done:
                    break
        finally:
            self.close()

    def pages(self) -> Iterator[DocumentPage]:
        for event in self:
            if event.get("object") == "document.page":
                yield self._state.pages[-1]

    def final(self) -> Document:
        for _ in self:
            pass
        if self._state.document is None:
            from .errors import StreamError

            raise StreamError(
                "the document stream ended without its final event",
                status=200,
                request_id=self.request_id,
            )
        return self._state.document


class SpeechStream:
    """``POST /v1/audio/speech`` with ``stream=True``: the audio as it is synthesized.

    For ``wav`` the first chunk starts with a header that declares an open-ended length; for
    ``pcm`` the chunks are raw 16-bit little-endian mono samples at 24 kHz.
    """

    def __init__(self, response: httpx.Response, release: Release) -> None:
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self.content_type = response.headers.get("content-type", "")
        self.telemetry = SpeechTelemetry.from_headers(response.headers)
        self._release = release
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        try:
            yield from self.response.iter_bytes()
        finally:
            self.close()

    def read(self) -> bytes:
        return b"".join(self)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.response.close()
            self._release()

    def __enter__(self) -> "SpeechStream":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# --- async -----------------------------------------------------------------------------------


class _AsyncSSE:
    def __init__(self, response: httpx.Response, release: Release) -> None:
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self._release = release
        self._closed = False

    async def _data(self) -> AsyncIterator[str]:
        decoder = SSEDecoder()
        try:
            async for line in self.response.aiter_lines():
                data = decoder.feed(line)
                if data is not None:
                    yield data
        except httpx.TransportError as e:
            from .errors import APIConnectionError

            raise APIConnectionError(f"the stream broke: {e}") from e
        tail = decoder.flush()
        if tail is not None:
            yield tail

    async def aclose(self) -> None:
        """Closes the connection; the server stops at the next token, page or chunk."""
        if not self._closed:
            self._closed = True
            await self.response.aclose()
            self._release()

    async def __aenter__(self) -> "_AsyncSSE":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


class AsyncChatStream(_AsyncSSE):
    """The async form of :class:`ChatStream`."""

    def __init__(self, response: httpx.Response, release: Release) -> None:
        super().__init__(response, release)
        self._state = _ChatState(self.request_id)

    @property
    def text_so_far(self) -> str:
        return self._state.text

    @property
    def usage(self) -> Optional[Usage]:
        return self._state.usage

    @property
    def telemetry(self) -> Optional[Telemetry]:
        return self._state.telemetry

    @property
    def finish_reason(self) -> Optional[str]:
        return self._state.finish_reason

    @property
    def tool_calls(self) -> List[ToolCall]:
        """The tool calls received so far, accumulated by ``index`` (complete once the stream ended)."""
        return self._state.tool_calls()

    async def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        if self._closed:  # already read (or closed): final() returns what was collected
            return
        try:
            async for data in self._data():
                chunk = self._state.add(data)
                if chunk is not None:
                    yield chunk
                if self._state.done:
                    break
        finally:
            await self.aclose()

    async def text(self) -> AsyncIterator[str]:
        async for chunk in self:
            for choice in chunk.get("choices", []):
                piece = choice.get("delta", {}).get("content")
                if piece:
                    yield piece

    async def final(self) -> ChatCompletion:
        async for _ in self:
            pass
        return self._state.completion()


class AsyncDocumentStream(_AsyncSSE):
    """The async form of :class:`DocumentStream`."""

    def __init__(self, response: httpx.Response, release: Release) -> None:
        super().__init__(response, release)
        self._state = _DocumentState(self.request_id)

    @property
    def started(self) -> Optional[DocumentStarted]:
        return self._state.started

    async def __aiter__(self) -> AsyncIterator[Dict[str, Any]]:
        if self._closed:  # already read (or closed): final() returns what was collected
            return
        try:
            async for data in self._data():
                event = self._state.add(data)
                if event is not None:
                    yield event
                if self._state.done:
                    break
        finally:
            await self.aclose()

    async def pages(self) -> AsyncIterator[DocumentPage]:
        async for event in self:
            if event.get("object") == "document.page":
                yield self._state.pages[-1]

    async def final(self) -> Document:
        async for _ in self:
            pass
        if self._state.document is None:
            from .errors import StreamError

            raise StreamError(
                "the document stream ended without its final event",
                status=200,
                request_id=self.request_id,
            )
        return self._state.document


class AsyncSpeechStream:
    """The async form of :class:`SpeechStream`."""

    def __init__(self, response: httpx.Response, release: Release) -> None:
        self.response = response
        self.request_id = response.headers.get("x-request-id")
        self.content_type = response.headers.get("content-type", "")
        self.telemetry = SpeechTelemetry.from_headers(response.headers)
        self._release = release
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        try:
            async for chunk in self.response.aiter_bytes():
                yield chunk
        finally:
            await self.aclose()

    async def read(self) -> bytes:
        return b"".join([c async for c in self])

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await self.response.aclose()
            self._release()

    async def __aenter__(self) -> "AsyncSpeechStream":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()
