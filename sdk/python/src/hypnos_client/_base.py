"""What the sync and async clients share: request building, error mapping, SSE parsing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import httpx

from .errors import APIError, StreamError, error_class

__version__ = "0.7.1"

DEFAULT_BASE_URL = "http://127.0.0.1:9999"
#: Chat, images and documents may pay a model load: keep at or above the server's
#: ``request_timeout_secs`` (900 s by default).
DEFAULT_TIMEOUT = httpx.Timeout(900.0, connect=10.0)

FileInput = Union[bytes, str, os.PathLike, Tuple[str, bytes]]


def base_url_from_env(base_url: Optional[str]) -> str:
    url = base_url or os.environ.get("HYPNOS_BASE_URL") or DEFAULT_BASE_URL
    url = url.rstrip("/")
    # Accept the OpenAI-style ".../v1" too; the routes below add it.
    return url[: -len("/v1")] if url.endswith("/v1") else url


def headers(api_key: Optional[str]) -> Dict[str, str]:
    h = {"User-Agent": f"hypnos-client-python/{__version__}", "Accept": "application/json"}
    key = api_key if api_key is not None else os.environ.get("HYPNOS_API_KEY")
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def drop_none(body: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in body.items() if v is not None}


def read_file(file: FileInput, filename: Optional[str]) -> Tuple[str, bytes]:
    """``(filename, bytes)`` from bytes, a path, or an explicit ``(name, bytes)`` pair."""
    if isinstance(file, tuple):
        return file
    if isinstance(file, (str, os.PathLike)):
        path = os.fspath(file)
        with open(path, "rb") as f:
            return filename or os.path.basename(path), f.read()
    return filename or "upload", bytes(file)


def form_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def parse_retry_after(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def api_error(response: httpx.Response, text: str) -> APIError:
    """The exception for an error response whose body has been read into ``text``."""
    body: Optional[Dict[str, Any]] = None
    message = text.strip() or response.reason_phrase
    err_type: Optional[str] = None
    code: Optional[str] = None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
            body = parsed
            err = parsed["error"]
            message = str(err.get("message", message))
            err_type = err.get("type")
            code = err.get("code")
    except ValueError:
        pass
    cls = error_class(response.status_code, err_type)
    return cls(
        message,
        status=response.status_code,
        type=err_type,
        code=code,
        request_id=response.headers.get("x-request-id"),
        retry_after=parse_retry_after(response.headers.get("retry-after")),
        body=body,
    )


def stream_error(payload: Dict[str, Any], request_id: Optional[str]) -> StreamError:
    err = payload.get("error") or {}
    return StreamError(
        str(err.get("message", "the stream failed")),
        status=200,
        type=err.get("type", "server_error"),
        request_id=request_id,
        body=payload,
    )


class SSEDecoder:
    """Feeds lines of a ``text/event-stream`` body and yields each event's ``data``.

    ``data:`` lines are joined with newlines, a blank line ends the event, ``:`` comments and
    other fields are ignored. The literal ``[DONE]`` comes back as is; the caller stops on it.
    """

    def __init__(self) -> None:
        self._data: List[str] = []

    def feed(self, line: str) -> Optional[str]:
        line = line.rstrip("\r")
        if line == "":
            if not self._data:
                return None
            data = "\n".join(self._data)
            self._data = []
            return data
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        if field == "data":
            self._data.append(value[1:] if value.startswith(" ") else value)
        return None

    def flush(self) -> Optional[str]:
        return self.feed("")


def iter_events(lines: Iterable[str]) -> Iterable[str]:
    decoder = SSEDecoder()
    for line in lines:
        data = decoder.feed(line)
        if data is not None:
            yield data
    tail = decoder.flush()
    if tail is not None:
        yield tail


@dataclass
class SpeechTelemetry:
    """The ``x-hypnos-*`` headers of a speech response (docs/api.md section 5)."""

    previous_tier: Optional[str] = None
    gpu_layers: Optional[int] = None
    switch_ms: Optional[float] = None
    inference_ms: Optional[float] = None
    audio_secs: Optional[float] = None
    rtf: Optional[float] = None

    @classmethod
    def from_headers(cls, h: httpx.Headers) -> "SpeechTelemetry":
        def num(name: str, kind: Any) -> Any:
            v = h.get(name)
            try:
                return kind(v) if v is not None else None
            except ValueError:
                return None

        return cls(
            previous_tier=h.get("x-hypnos-previous-tier"),
            gpu_layers=num("x-hypnos-gpu-layers", int),
            switch_ms=num("x-hypnos-switch-ms", float),
            inference_ms=num("x-hypnos-inference-ms", float),
            audio_secs=num("x-hypnos-audio-secs", float),
            rtf=num("x-hypnos-rtf", float),
        )


@dataclass
class Speech:
    """``POST /v1/audio/speech``: the audio bytes, their media type and the telemetry headers."""

    audio: bytes
    content_type: str
    telemetry: SpeechTelemetry
    request_id: Optional[str] = None


def chat_body(
    model: str,
    messages: List[Any],
    stream: bool,
    *,
    max_tokens: Optional[int],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    num_ctx: Optional[int],
    think: Optional[bool],
    reasoning_effort: Optional[str],
    tools: Optional[Sequence[Any]],
    tool_choice: Optional[Any],
    extra: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    body = drop_none(
        {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "seed": seed,
            "num_ctx": num_ctx,
            "think": think,
            "reasoning_effort": reasoning_effort,
            "tools": list(tools) if tools is not None else None,
            "tool_choice": tool_choice,
            "stream": stream,
        }
    )
    body.update(extra or {})
    return body


def document_form(
    model: str,
    *,
    stream: bool,
    filename: Optional[str],
    pages: Optional[str],
    ocr: Optional[bool],
    output_format: Optional[str],
    describe_pictures: Optional[str],
    return_pictures: Optional[bool],
) -> Dict[str, str]:
    fields = drop_none(
        {
            "model": model,
            "filename": filename,
            "pages": pages,
            "ocr": ocr,
            "output_format": output_format,
            "describe_pictures": describe_pictures,
            "return_pictures": return_pictures,
            "stream": stream,
        }
    )
    return {k: form_value(v) for k, v in fields.items()}
