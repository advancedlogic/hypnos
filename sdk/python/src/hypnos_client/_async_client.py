"""The asynchronous client. ``_client.py`` is generated from this file by
``scripts/unasync.py``: edit here, then run ``python scripts/unasync.py``."""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union, cast

import httpx

from ._base import (
    DEFAULT_TIMEOUT,
    FileInput,
    Speech,
    SpeechTelemetry,
    api_error,
    base_url_from_env,
    chat_body,
    document_form,
    drop_none,
    form_value,
    headers,
    read_file,
)
from .errors import APIConnectionError, APITimeoutError, HypnosError
from .retry import RetryPolicy
from .streams import AsyncChatStream, AsyncDocumentStream, AsyncSpeechStream
from .types import (
    Advice,
    AdvisorNeed,
    ChatCompletion,
    ChatMessage,
    Document,
    EmbeddingResponse,
    EntityResponse,
    Health,
    ImageResponse,
    KvParkStatus,
    LogList,
    LogRecord,
    MemoryStatus,
    MemoryTier,
    Modality,
    ModelCatalog,
    ModelList,
    ModelPoolChange,
    ModelSwitch,
    MoeOffload,
    MoeOffloadSwitch,
    Picture,
    PictureList,
    ReasoningEffort,
    TierMove,
    ToolChoice,
    ToolSpec,
    Transcription,
)


def _no_release() -> None:
    pass


class AsyncClient:
    """A client for one Hypnos server.

    ``base_url`` defaults to ``$HYPNOS_BASE_URL`` or ``http://127.0.0.1:9999`` (a trailing
    ``/v1`` is accepted), ``api_key`` to ``$HYPNOS_API_KEY``. ``max_concurrency_per_model``
    bounds the requests this client keeps in flight per model (a stream counts until it is
    closed); beyond the model's slots the server only queues them, and beyond its queue it
    answers 429. ``retry`` is the policy of docs/api.md section 17.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        *,
        api_key: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = DEFAULT_TIMEOUT,
        retry: Optional[RetryPolicy] = None,
        max_concurrency_per_model: Optional[int] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url_from_env(base_url)
        self.retry = retry or RetryPolicy()
        self._max_per_model = max_concurrency_per_model
        self._limits: Dict[str, asyncio.Semaphore] = {}
        self._limits_lock = threading.Lock()
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(timeout=timeout)
        self._headers = headers(api_key)

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "AsyncClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # --- plumbing ----------------------------------------------------------------------------

    def _limit(self, model: Optional[str]) -> Optional[asyncio.Semaphore]:
        if model is None or not self._max_per_model:
            return None
        with self._limits_lock:
            sem = self._limits.get(model)
            if sem is None:
                sem = asyncio.Semaphore(self._max_per_model)
                self._limits[model] = sem
            return sem

    async def _send(
        self,
        method: str,
        path: str,
        *,
        model: Optional[str] = None,
        json: Any = None,
        data: Optional[Mapping[str, str]] = None,
        files: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        accept: Optional[str] = None,
        retry: bool = True,
    ) -> "tuple[httpx.Response, Any]":
        """Sends the request, retrying per the policy, and returns the open response with a
        release function for the model's concurrency permit (call it when the body is done)."""
        sem = self._limit(model)
        if sem is not None:
            await sem.acquire()
        release = sem.release if sem is not None else _no_release
        hdrs = dict(self._headers)
        if accept:
            hdrs["Accept"] = accept
        attempt = 0
        try:
            while True:
                error: HypnosError
                try:
                    request = self._http.build_request(
                        method,
                        self.base_url + path,
                        json=json,
                        data=data,
                        files=files,
                        params=drop_none(params or {}),
                        headers=hdrs,
                    )
                    response = await self._http.send(request, stream=True)
                except httpx.TimeoutException as e:
                    error = APITimeoutError(f"no response in time: {e}", status=408)
                except httpx.TransportError as e:
                    error = APIConnectionError(f"{method} {path}: {e}")
                else:
                    if response.is_success:
                        return response, release
                    body = (await response.aread()).decode("utf-8", "replace")
                    await response.aclose()
                    error = api_error(response, body)
                wait = self.retry.delay(error, attempt) if retry else None
                if wait is None:
                    raise error
                attempt += 1
                await asyncio.sleep(wait)
        except BaseException:
            release()
            raise

    async def _json(self, method: str, path: str, **kw: Any) -> Any:
        response, release = await self._send(method, path, **kw)
        try:
            await response.aread()
            return response.json()
        finally:
            await response.aclose()
            release()

    # --- server ------------------------------------------------------------------------------

    async def health(self) -> Health:
        """``GET /health``: the server is up (it may still be staging models)."""
        return cast(Health, await self._json("GET", "/health", retry=False))

    async def is_ready(self) -> bool:
        """``GET /ready``: True once the startup models are staged."""
        try:
            response = await self._http.get(self.base_url + "/ready", headers=self._headers)
        except httpx.TransportError:
            return False
        return response.status_code == 200

    async def wait_until_ready(self, timeout: float = 600.0, interval: float = 1.0) -> None:
        """Polls ``/ready`` until it answers 200; raises ``TimeoutError`` after ``timeout``."""
        deadline = time.monotonic() + timeout
        while not await self.is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{self.base_url} was not ready within {timeout:.0f} s")
            await asyncio.sleep(interval)

    async def metrics(self) -> str:
        """``GET /metrics``: Prometheus text."""
        response, release = await self._send("GET", "/metrics", accept="text/plain")
        try:
            return (await response.aread()).decode()
        finally:
            await response.aclose()
            release()

    # --- models ------------------------------------------------------------------------------

    async def models(self) -> ModelList:
        """``GET /v1/models``: the models in service."""
        return cast(ModelList, await self._json("GET", "/v1/models"))

    async def models_catalog(self, modality: Optional[Modality] = None) -> ModelCatalog:
        """``GET /v1/models/catalog``: the curated catalog, configured or not, with each variant's
        files, what is installed and its config block. ``modality`` keeps the entries serving it
        (primary or additional). :class:`NotFoundError` when the server has no catalog file."""
        params = {"modality": modality}
        return cast(ModelCatalog, await self._json("GET", "/v1/models/catalog", params=params))

    async def advise(
        self,
        needs: Sequence[AdvisorNeed],
        *,
        vram_bytes: Optional[int] = None,
        ram_bytes: Optional[int] = None,
        unified_memory: Optional[bool] = None,
    ) -> Advice:
        """``POST /v1/advisor``: recommends a lineup from the catalog for a machine and a set of
        needs (``"chat"``, ``"vision"``, ``"voice"``, ``"transcription"``, ``"speech"``,
        ``"embeddings"``, ``"images"``, ``"documents"``, ``"entities"``): models and variants,
        context, slots, tier and priority, warnings, ``specs`` and a complete ``config`` file.

        ``vram_bytes``, ``ram_bytes`` and ``unified_memory`` describe the machine; each one left
        out is the server's own. :class:`BadRequestError` when no lineup fits or ``needs`` is
        empty, :class:`NotFoundError` without a catalog. A read-only query, retried like any
        other request.
        """
        body = {
            "needs": list(needs),
            **drop_none(
                {
                    "vram_bytes": vram_bytes,
                    "ram_bytes": ram_bytes,
                    "unified_memory": unified_memory,
                }
            ),
        }
        return cast(Advice, await self._json("POST", "/v1/advisor", json=body))

    @staticmethod
    def _model_source(catalog: Optional[str], config: Optional[str]) -> Dict[str, Any]:
        if (catalog is None) == (config is None):
            raise ValueError("pass exactly one of `catalog` or `config`")
        return {"catalog": catalog} if catalog is not None else {"config": config}

    async def add_model(
        self, catalog: Optional[str] = None, config: Optional[str] = None
    ) -> ModelPoolChange:
        """``POST /v1/models``: adds a model to the running pool, from the catalog
        (``"<id>[:<quant>]"``) or from one ``[[models]]`` TOML block (``config``); exactly one.

        Its files must already be on disk. :class:`BadRequestError` when a check fails or the id
        is taken, :class:`PermissionDeniedError` without an admin key. Never retried
        automatically (the change may have been applied when the connection broke).
        """
        body = self._model_source(catalog, config)
        return cast(
            ModelPoolChange, await self._json("POST", "/v1/models", json=body, retry=False)
        )

    async def replace_model(
        self,
        model: str,
        catalog: Optional[str] = None,
        config: Optional[str] = None,
        *,
        drain_secs: Optional[int] = None,
    ) -> ModelPoolChange:
        """``PUT /v1/models/{id}``: replaces ``model`` under the same id; running requests finish
        on the old one. :class:`ModelBusyError` when they outlast ``drain_secs`` (server default
        120; the old model stays), :class:`NotFoundError` for an unknown id. Never retried
        automatically."""
        body = {**self._model_source(catalog, config), **drop_none({"drain_secs": drain_secs})}
        return cast(
            ModelPoolChange,
            await self._json("PUT", f"/v1/models/{model}", json=body, retry=False),
        )

    async def remove_model(
        self, model: str, *, drain_secs: Optional[int] = None
    ) -> ModelPoolChange:
        """``DELETE /v1/models/{id}``: removes ``model`` once its running requests finish
        (:class:`ModelBusyError` past ``drain_secs``). Never retried automatically."""
        params = {"drain_secs": drain_secs}
        return cast(
            ModelPoolChange,
            await self._json("DELETE", f"/v1/models/{model}", params=params, retry=False),
        )

    async def enable_model(self, model: str) -> ModelSwitch:
        """``POST /v1/models/{id}/enable``."""
        return cast(ModelSwitch, await self._json("POST", f"/v1/models/{model}/enable"))

    async def disable_model(self, model: str) -> ModelSwitch:
        """``POST /v1/models/{id}/disable``; refused with :class:`ModelBusyError` while leased."""
        return cast(ModelSwitch, await self._json("POST", f"/v1/models/{model}/disable"))

    async def set_moe_offload(self, model: str, mode: MoeOffload) -> MoeOffloadSwitch:
        """``POST /v1/models/{id}/moe-offload``: ``"off"``, ``"auto"`` or ``"on"``.

        ``applies`` in the reply is ``"now"`` or ``"next_load"`` (the model was busy).
        """
        return cast(
            MoeOffloadSwitch,
            await self._json("POST", f"/v1/models/{model}/moe-offload", json={"mode": mode}),
        )

    # --- chat --------------------------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        num_ctx: Optional[int] = None,
        think: Optional[bool] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        tools: Optional[Sequence[Union[ToolSpec, Mapping[str, Any]]]] = None,
        tool_choice: Optional[Union[ToolChoice, Mapping[str, Any]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> ChatCompletion:
        """``POST /v1/chat/completions``. ``num_ctx`` and ``think`` are Hypnos extensions;
        ``reasoning_effort`` sets the thinking level (it takes precedence over ``think``) and the
        thoughts come back as ``message.reasoning_content``; ``tools`` and ``tool_choice`` turn on tool calling (docs/api.md 7.5; helpers in
        :mod:`hypnos_client.tools`); ``extra`` adds any other body field."""
        body = chat_body(
            model, list(messages), False, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed, num_ctx=num_ctx, think=think,
            reasoning_effort=reasoning_effort, tools=tools, tool_choice=tool_choice, extra=extra,
        )  # fmt: skip
        return cast(
            ChatCompletion,
            await self._json("POST", "/v1/chat/completions", model=model, json=body),
        )

    async def chat_stream(
        self,
        model: str,
        messages: Sequence[Union[ChatMessage, Mapping[str, Any]]],
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        num_ctx: Optional[int] = None,
        think: Optional[bool] = None,
        reasoning_effort: Optional[ReasoningEffort] = None,
        tools: Optional[Sequence[Union[ToolSpec, Mapping[str, Any]]]] = None,
        tool_choice: Optional[Union[ToolChoice, Mapping[str, Any]]] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> AsyncChatStream:
        """``POST /v1/chat/completions`` with ``stream: true``. Close the stream to cancel."""
        body = chat_body(
            model, list(messages), True, max_tokens=max_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k, seed=seed, num_ctx=num_ctx, think=think,
            reasoning_effort=reasoning_effort, tools=tools, tool_choice=tool_choice, extra=extra,
        )  # fmt: skip
        response, release = await self._send(
            "POST", "/v1/chat/completions", model=model, json=body, accept="text/event-stream"
        )
        return AsyncChatStream(response, release)

    # --- embeddings, audio, images, entities ---------------------------------------------------

    async def embeddings(self, model: str, input: Union[str, Sequence[str]]) -> EmbeddingResponse:
        """``POST /v1/embeddings``; ``data`` is in input order."""
        body = {"model": model, "input": input if isinstance(input, str) else list(input)}
        return cast(
            EmbeddingResponse, await self._json("POST", "/v1/embeddings", model=model, json=body)
        )

    async def transcribe(
        self,
        model: str,
        file: FileInput,
        *,
        filename: Optional[str] = None,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
    ) -> Transcription:
        """``POST /v1/audio/transcriptions`` (multipart). ``file`` is bytes, a path or
        ``(filename, bytes)``; WAV 16-bit PCM is the reliable format."""
        name, content = read_file(file, filename)
        fields = {k: form_value(v) for k, v in drop_none(
            {"model": model, "language": language, "prompt": prompt}).items()}  # fmt: skip
        return cast(
            Transcription,
            await self._json(
                "POST",
                "/v1/audio/transcriptions",
                model=model,
                data=fields,
                files={"file": (name, content)},
            ),
        )

    def _speech_body(
        self, model: str, input: str, voice: Optional[str], response_format: Optional[str],
        speed: Optional[float], stream: bool, language: Optional[str] = None,
    ) -> Dict[str, Any]:  # fmt: skip
        return drop_none(
            {"model": model, "input": input, "voice": voice,
             "response_format": response_format, "speed": speed, "stream": stream,
             "language": language}
        )  # fmt: skip

    async def speech(
        self,
        model: str,
        input: str,
        *,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> Speech:
        """``POST /v1/audio/speech``: WAV (or ``response_format="pcm"``) bytes plus the
        ``x-hypnos-*`` telemetry headers. Check ``content_type``: unsupported formats fall back
        to WAV."""
        body = self._speech_body(model, input, voice, response_format, speed, False, language)
        response, release = await self._send("POST", "/v1/audio/speech", model=model, json=body)
        try:
            audio = await response.aread()
        finally:
            await response.aclose()
            release()
        return Speech(
            audio=audio,
            content_type=response.headers.get("content-type", ""),
            telemetry=SpeechTelemetry.from_headers(response.headers),
            request_id=response.headers.get("x-request-id"),
        )

    async def speech_stream(
        self,
        model: str,
        input: str,
        *,
        voice: Optional[str] = None,
        response_format: Optional[str] = None,
        speed: Optional[float] = None,
        language: Optional[str] = None,
    ) -> AsyncSpeechStream:
        """``POST /v1/audio/speech`` with ``stream: true``: audio chunks as they are made."""
        body = self._speech_body(model, input, voice, response_format, speed, True, language)
        response, release = await self._send("POST", "/v1/audio/speech", model=model, json=body)
        return AsyncSpeechStream(response, release)

    async def images(
        self,
        model: str,
        prompt: str,
        *,
        n: Optional[int] = None,
        size: Optional[str] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
    ) -> ImageResponse:
        """``POST /v1/images/generations``: base64 PNGs in ``data[i]["b64_json"]``."""
        body = drop_none(
            {"model": model, "prompt": prompt, "n": n, "size": size,
             "negative_prompt": negative_prompt, "steps": steps, "seed": seed,
             "cfg_scale": cfg_scale}
        )  # fmt: skip
        return cast(
            ImageResponse,
            await self._json("POST", "/v1/images/generations", model=model, json=body),
        )

    async def entities(
        self,
        model: str,
        input: Union[str, Sequence[str]],
        labels: Sequence[str],
        *,
        threshold: Optional[float] = None,
        flat: Optional[bool] = None,
    ) -> EntityResponse:
        """``POST /v1/entities``: zero-shot entity extraction. ``start``/``end`` are character
        offsets, the same as Python string indexes."""
        body = drop_none(
            {"model": model, "input": input if isinstance(input, str) else list(input),
             "labels": list(labels), "threshold": threshold, "flat": flat}
        )  # fmt: skip
        return cast(EntityResponse, await self._json("POST", "/v1/entities", model=model, json=body))

    # --- documents ---------------------------------------------------------------------------

    def _document_request(
        self,
        model: str,
        file: Optional[FileInput],
        url: Optional[str],
        stream: bool,
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        if (file is None) == (url is None):
            raise ValueError("pass exactly one of `file` and `url`")
        filename = options.pop("filename", None)
        if file is not None:
            name, content = read_file(file, filename)
            fields = document_form(model, stream=stream, filename=name, **options)
            return {"data": fields, "files": {"file": (name, content)}}
        body = drop_none({"model": model, "url": url, "filename": filename, "stream": stream,
                          **options})  # fmt: skip
        return {"json": body}

    async def convert_document(
        self,
        model: str,
        file: Optional[FileInput] = None,
        *,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        pages: Optional[str] = None,
        ocr: Optional[bool] = None,
        output_format: Optional[str] = None,
        describe_pictures: Optional[str] = None,
        return_pictures: Optional[bool] = None,
    ) -> Document:
        """``POST /v1/documents/convert``: a file (multipart upload) or a ``url`` the server
        fetches, to Markdown. ``pages`` is ``"3"`` or ``"2-5"``."""
        req = self._document_request(
            model, file, url, False,
            dict(filename=filename, pages=pages, ocr=ocr, output_format=output_format,
                 describe_pictures=describe_pictures, return_pictures=return_pictures),
        )  # fmt: skip
        return cast(
            Document, await self._json("POST", "/v1/documents/convert", model=model, **req)
        )

    async def convert_document_stream(
        self,
        model: str,
        file: Optional[FileInput] = None,
        *,
        url: Optional[str] = None,
        filename: Optional[str] = None,
        pages: Optional[str] = None,
        ocr: Optional[bool] = None,
        output_format: Optional[str] = None,
        describe_pictures: Optional[str] = None,
        return_pictures: Optional[bool] = None,
    ) -> AsyncDocumentStream:
        """The streaming form of :meth:`convert_document`: one event per page."""
        req = self._document_request(
            model, file, url, True,
            dict(filename=filename, pages=pages, ocr=ocr, output_format=output_format,
                 describe_pictures=describe_pictures, return_pictures=return_pictures),
        )  # fmt: skip
        response, release = await self._send(
            "POST", "/v1/documents/convert", model=model, accept="text/event-stream", **req
        )
        return AsyncDocumentStream(response, release)

    async def document_pictures(self, document_id: str) -> PictureList:
        """``GET /v1/documents/{id}/pictures``: the records, without image data."""
        return cast(PictureList, await self._json("GET", f"/v1/documents/{document_id}/pictures"))

    async def document_picture(self, document_id: str, index: int) -> Picture:
        """``GET /v1/documents/{id}/pictures/{n}``: one record with ``image`` (base64)."""
        return cast(
            Picture, await self._json("GET", f"/v1/documents/{document_id}/pictures/{index}")
        )

    async def document_picture_bytes(self, document_id: str, index: int) -> "tuple[bytes, str]":
        """The picture's image bytes and media type (``?format=image``)."""
        response, release = await self._send(
            "GET",
            f"/v1/documents/{document_id}/pictures/{index}",
            params={"format": "image"},
            accept="*/*",
        )
        try:
            return await response.aread(), response.headers.get("content-type", "")
        finally:
            await response.aclose()
            release()

    async def delete_document(self, document_id: str) -> Dict[str, Any]:
        """``DELETE /v1/documents/{id}``: releases the stored pictures."""
        return cast(Dict[str, Any], await self._json("DELETE", f"/v1/documents/{document_id}"))

    # --- memory ------------------------------------------------------------------------------

    async def memory_status(self) -> MemoryStatus:
        """``GET /v1/memory/status``: the three tiers and every model's placement."""
        return cast(MemoryStatus, await self._json("GET", "/v1/memory/status"))

    async def set_tier(
        self, model: str, target_tier: MemoryTier, *, context_len: Optional[int] = None
    ) -> TierMove:
        """``POST /v1/memory/tier``: promote or demote a model by hand (blocks for the load)."""
        body = drop_none({"model": model, "target_tier": target_tier, "context_len": context_len})
        return cast(TierMove, await self._json("POST", "/v1/memory/tier", json=body))

    async def kv_park(self) -> KvParkStatus:
        """``GET /v1/memory/kv-park``: whether KV parking is on, its cap and the parked states."""
        return cast(KvParkStatus, await self._json("GET", "/v1/memory/kv-park"))

    async def set_kv_park(
        self, enabled: Optional[bool] = None, max_bytes: Optional[int] = None
    ) -> KvParkStatus:
        """``POST /v1/memory/kv-park``: turn KV parking on or off and/or change its cap."""
        body = drop_none({"enabled": enabled, "max_bytes": max_bytes})
        if not body:
            raise ValueError("pass `enabled`, `max_bytes` or both")
        return cast(KvParkStatus, await self._json("POST", "/v1/memory/kv-park", json=body))

    # --- request log -------------------------------------------------------------------------

    async def logs(
        self,
        *,
        model: Optional[str] = None,
        route: Optional[str] = None,
        status: Optional[str] = None,
        key: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> LogList:
        """``GET /v1/logs``, newest first; ``status`` is ``ok``, ``error``, ``4xx`` or ``5xx``."""
        params = {"model": model, "route": route, "status": status, "key": key, "limit": limit}
        return cast(LogList, await self._json("GET", "/v1/logs", params=params))

    async def log(self, request_id: str) -> LogRecord:
        """``GET /v1/logs/{id}``: one record by its ``x-request-id``."""
        return cast(LogRecord, await self._json("GET", f"/v1/logs/{request_id}"))

    async def clear_logs(self) -> Dict[str, Any]:
        return cast(Dict[str, Any], await self._json("DELETE", "/v1/logs"))

    async def set_log_level(self, level: str) -> Dict[str, Any]:
        """``POST /v1/logs/level``: ``off``, ``error``, ``warn``, ``info``, ``debug``, ``trace``."""
        return cast(
            Dict[str, Any], await self._json("POST", "/v1/logs/level", json={"level": level})
        )


__all__: List[str] = ["AsyncClient"]
