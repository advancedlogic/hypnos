# Hypnos HTTP API reference

This is the wire-level reference for anyone writing a client or an SDK against a Hypnos server.
It describes every route, every request and response field, the streaming formats, the error
shape and the behaviour a client has to handle (readiness, admission, quotas, cancellation).
The operator's side (sizing, knobs, metrics) is in [operating.md](operating.md); the general
introduction is the [README](../README.md).

Everything below is derived from the server source as of v0.1.0 (`backend/crates/hypnos-server`
for the routes, `backend/crates/hypnos-core/src/openai.rs` for the schemas). Where the API is a
Hypnos extension rather than OpenAI behaviour it is marked **Hypnos extension**.

## Contents

1. [Basics](#1-basics)
2. [Authentication and per-key limits](#2-authentication-and-per-key-limits)
3. [Server lifecycle: health, readiness, 503](#3-server-lifecycle-health-readiness-503)
4. [Errors](#4-errors)
5. [Telemetry on every response](#5-telemetry-on-every-response)
6. [Models](#6-models)
7. [Chat completions (text, vision, audio in, tools)](#7-chat-completions-text-vision-audio-in-tools)
8. [Embeddings](#8-embeddings)
9. [Audio transcriptions](#9-audio-transcriptions)
10. [Speech synthesis](#10-speech-synthesis)
11. [Image generation](#11-image-generation)
12. [Document conversion and pictures](#12-document-conversion-and-pictures)
13. [Entity extraction](#13-entity-extraction)
14. [Memory status and placement control](#14-memory-status-and-placement-control)
15. [Request log](#15-request-log)
16. [Metrics](#16-metrics)
17. [Writing a client: recommendations](#17-writing-a-client-recommendations)
18. [Appendix: shared types](#18-appendix-shared-types)

---

## 1. Basics

**Base URL.** `http://<host>:<port>`; the default port is 9999 (`server.port`, `--port`,
`HYPNOS_PORT`). All API routes live under `/v1`. With `[server.tls]` configured the scheme is
`https`. The playground, when present, is served at `/` and never interferes with `/v1`.

**OpenAI compatibility.** The chat, embeddings, transcription, speech, image and model-list
routes accept the OpenAI request bodies and answer with the OpenAI response bodies, so the
official OpenAI SDKs work unchanged with `base_url = "http://host:9999/v1"` and any non-empty API
key (the key is checked only when the server has keys configured). Hypnos adds fields on top; it
never removes an OpenAI field it supports. Unknown request fields are ignored, not rejected.

**Content types.** JSON bodies are `application/json`. Transcription and document conversion
also accept `multipart/form-data`. Speech answers with audio bytes, streaming routes with
`text/event-stream` (SSE) or a chunked audio body. Every JSON response is UTF-8.

**Request id.** Every response carries `x-request-id: req-<32 hex>`. Quote it when reporting a
problem; the request log (section 15) is keyed by it.

**CORS.** Browsers may call the API from any loopback origin by default; the operator can list
other origins (`server.allowed_origins`, `http://host:*` wildcards). The methods are GET, POST
and OPTIONS (DELETE is not in the CORS allow-list, so release stored pictures from a non-browser
client or same-origin). The telemetry headers of section 5 are exposed to browser scripts.

**Body and time limits** (defaults; the operator may change them in `[limits]`):

| Limit | Default | Effect when exceeded |
| :--- | ---: | :--- |
| `max_body_bytes` (any request) | 256 MiB | 413 from the body-limit layer |
| `max_audio_bytes` (upload or base64 `input_audio`) | 64 MiB | 400 `invalid_request_error` |
| `max_image_bytes` (data URL or remote download) | 32 MiB | 400 |
| `max_document_bytes` (`/v1/documents/convert`) | 100 MiB | 400 |
| `max_prompt_chars` (chat messages, embedding inputs, TTS input) | 1,000,000 | 400 |
| `max_tokens_cap` | 8192 | `max_tokens` is silently clamped |
| `max_images_per_request` (`n`) | 4 | 400 |
| `request_timeout_secs` | 900 s | 408 from the timeout layer (streams: applies to the headers only) |
| `max_queue_per_model` | 8 | 429 `model_busy` |
| `allow_remote_image_urls` | true | when false, `http(s)` image and document URLs answer 400; only `data:` URLs and uploads are accepted |

---

## 2. Authentication and per-key limits

When the server has API keys configured (`server.api_keys`, `[[server.keys]]` or the
`HYPNOS_API_KEYS` environment variable), every `/v1/*` request must carry

```text
Authorization: Bearer <key>
```

The scheme is matched case-insensitively and surrounding whitespace is trimmed. `/health`,
`/ready` and, unless the operator closes it, `/metrics` stay open. When no keys are configured
the header is ignored and may be absent.

**401** (missing, malformed or unknown key) uses the OpenAI error shape, so the OpenAI SDKs raise
their usual authentication error, and carries `WWW-Authenticate: Bearer realm="hypnos"`:

```json
{ "error": { "message": "Missing API key. Send it as `Authorization: Bearer <key>` (server.api_keys).",
             "type": "invalid_request_error", "param": null, "code": "invalid_api_key" } }
```

**Per-key limits.** Each key can have `requests_per_minute`, `max_concurrent` and
`tokens_per_day` (UTC day, prompt plus completion tokens of chat, embeddings and the other
token-counting routes; a chat stream is charged when it ends). A rejected call answers **429**
with `Retry-After: <seconds>`:

```json
{ "error": { "message": "...", "type": "rate_limit_error", "param": null,
             "code": "requests_per_minute" | "max_concurrent" | "tokens_per_day" } }
```

The concurrency permit is held for the whole response body, so a streaming chat or a long
document conversion counts as one in-flight request until the client has read the last byte.

The key's label (from the config, or `key-<8 hex>` for unlabelled keys) is attached to the request
log entry and to the `hypnos_api_key_*` metrics; the key itself is never logged.

---

## 3. Server lifecycle: health, readiness, 503

The port opens before the startup models are staged, which can take minutes for a large model.
During that window:

| Route | Behaviour while staging |
| :--- | :--- |
| `GET /health` | 200 `{"status":"ok","service":"hypnos","version":"0.4.0","uptime_secs":12}` |
| `GET /ready` | 503 `{"ready":false,"reason":"startup tier initialization in progress"}`; 200 `{"ready":true}` afterwards |
| `GET /v1/models`, `GET /v1/memory/status`, `/v1/logs*`, stored pictures | open |
| Every inference and placement route | 503 with `Retry-After: 5` and the body below |

```json
{ "error": { "message": "Hypnos is still pre-staging its startup models; retry shortly (GET /ready reports when it is done)",
             "type": "model_not_available" } }
```

A client that starts together with the server should poll `/ready` (or retry on this 503 with
the `Retry-After` delay) instead of queueing requests. A 503 `model_not_available` can also come
from a configuration problem with the requested model after startup; the message says which.

Once ready, the first request for a model that is not in GPU memory pays the promotion (seconds
for a model warm in RAM, tens of seconds from disk); `hypnos_telemetry.tier_switch_latency_ms`
(section 5) reports it. Clients should not time out below the operator's `request_timeout_secs`.

---

## 4. Errors

All errors share one body:

```json
{ "error": { "message": "<human readable>", "type": "<machine readable>" } }
```

Authentication and quota errors add `"param": null` and `"code"` (section 2).

| HTTP | `type` | When | Client action |
| ---: | :--- | :--- | :--- |
| 400 | `invalid_request_error` | malformed body, unknown field value, a limit of section 1 exceeded, modality mismatch (e.g. an image sent to a text-only model) | fix the request |
| 401 | `invalid_request_error` / `invalid_api_key` | see section 2 | fix the key |
| 404 | `model_not_found` | unknown model id | check `GET /v1/models` |
| 404 | `model_disabled` | the model is configured but switched off | `POST /v1/models/{id}/enable` (if allowed) or pick another |
| 404 | `not_found` | unknown document id or picture index, unknown log record | nothing to retry |
| 408 | (plain text) | `request_timeout_secs` elapsed before the response headers were sent | retry with a smaller request or a shorter `max_tokens` |
| 413 | (plain text) | body above `max_body_bytes` | shrink the upload |
| 429 | `model_busy` | more than `max_queue_per_model` requests queued or running for this model | back off and retry; there is no `Retry-After` on this one |
| 429 | `rate_limit_error` | per-key limit, see section 2 | wait `Retry-After` seconds |
| 429 | `model_busy` | `POST /v1/memory/tier` or `.../disable` while the model is leased | retry when idle |
| 499 | `cancelled` | the client disconnected or the server is shutting down (rarely seen by the client) | none |
| 503 | `model_not_available` | still staging (section 3), or the model cannot be served as configured | poll `/ready`, retry later |
| 504 | `timeout` | an engine-side timeout | retry |
| 507 | `insufficient_vram_or_ram` | the model does not fit next to the models currently leased | retry later; the operator can lower `min_gpu_layer_ratio` or priorities |
| 507 | `process_memory_cap` | loading would exceed `hardware.max_process_memory_bytes` | retry later |
| 507 | `device_out_of_memory` | the engine ran out of device memory mid-request; the server already evicted idle models and retried once (chat without `stream`, embeddings, images) | retry later |
| 500 | `server_error` | the engine failed (weights refused, CUDA error, decode failure) | retry once; report with `x-request-id` |

On a streaming chat response that fails after the headers were sent, the error arrives as a
final chunk with `finish_reason: "error"` (section 7.4); the HTTP status is already 200.

---

## 5. Telemetry on every response

**Hypnos extension.** Every successful JSON inference response (and the last chunk of a chat
stream) carries `hypnos_telemetry`:

```json
"hypnos_telemetry": {
  "previous_tier": "tier2_pinned_ram",
  "placement": { "primary_tier": "tier1_gpu_vram", "gpu_layers": 30, "pinned_ram_layers": 0,
                 "ssd_layers": 0, "vram_weight_bytes": 15814641056,
                 "vram_kv_and_workspace_bytes": 5301600256, "pinned_ram_bytes": 0,
                 "allocated_context_len": 16384 },
  "tier_switch_latency_ms": 3662.1,
  "inference_latency_ms": 1240.5,
  "evicted_models": ["moss-image"],
  "engine": { "queue_ms": 0.1, "prefill_ms": 120.0, "decode_ms": 1100.0, "ttft_ms": 125.0,
              "prefill_tokens_per_s": 676.0, "decode_tokens_per_s": 163.7,
              "cached_tokens": 0, "concurrent_slots": 1 },
  "tier_switch_phases": { "load_mode": "read", "weights_ms": 2700.0, "projector_ms": 400.0,
                          "context_ms": 500.0, "total_ms": 3650.0 }
}
```

| Field | Meaning |
| :--- | :--- |
| `previous_tier` | where the model was when the request arrived: `tier1_gpu_vram`, `tier2_pinned_ram`, `tier3_nvme_ssd` |
| `placement` | the model's current placement (see appendix) |
| `tier_switch_latency_ms` | time spent loading the model for this request; 0 on the fast path |
| `inference_latency_ms` | wall time of the engine call as the server measured it |
| `evicted_models` | models demoted to make room for this one |
| `engine` | engine timings; each field is present only when that modality measures it: `queue_ms`, `context_ms`, `prefill_ms`, `decode_ms`, `ttft_ms`, `prefill_tokens_per_s`, `decode_tokens_per_s`, `cached_tokens`, `concurrent_slots` (chat); `audio_secs`, `rtf`, `ttfa_ms` (audio); `pages`, `ms_per_page` (documents) |
| `tier_switch_phases` | only when this request loaded a llama.cpp model into VRAM: `load_mode` (`mmap` or `read`), `weights_ms`, `projector_ms`, `context_ms`, `total_ms`, and with KV parking (`/v1/memory/kv-park`, section 14) `kv_restore_ms` and `kv_restored_tokens` when parked conversations were restored |

Speech responses are audio, so the same facts travel as headers: `x-hypnos-previous-tier`
(spelled as in the JSON, `tier2_pinned_ram`), `x-hypnos-gpu-layers`, `x-hypnos-switch-ms`,
`x-hypnos-inference-ms`, `x-hypnos-audio-secs`, `x-hypnos-rtf`. A streaming speech response carries the first three only (the rest is unknown
when the headers go out).

A client may ignore all of this; an SDK should expose it as an optional typed object because it
is the only way to tell a cold request (seconds of tier switch) from a slow one.

---

## 6. Models

### `GET /v1/models`

Lists the models in service (disabled models are hidden). Open during startup staging.

```json
{ "object": "list", "data": [ {
    "id": "gemma4:26b-q4", "object": "model", "created": 1793481600, "owned_by": "hypnos",
    "modality": "text_generation",
    "current_tier": "tier1_gpu_vram",
    "placement": { "...": "see appendix" },
    "active_leases": 0 } ] }
```

`modality` is the model's primary modality: `text_generation`, `vision`, `speech_to_text`,
`text_to_speech`, `embeddings`, `image_generation`, `document_conversion`, `entity_extraction`.
A model can serve more than its primary modality (the default `gemma4:26b-q4` also does vision,
`gemma4:e4b-q4` does speech-to-text, text and vision); `GET /v1/memory/status` lists the full
`modalities` array per model.

The default lineup (`hypnos.example.toml`):

| id | serves | route |
| :--- | :--- | :--- |
| `gemma4:26b-q4` | text, vision | `/v1/chat/completions` |
| `gemma4:e4b-q4` | speech-to-text, text, vision | `/v1/audio/transcriptions`, `/v1/chat/completions` (incl. `input_audio`) |
| `gemmaembedding2` | embeddings | `/v1/embeddings` |
| `kokoro-82m-q4` | text-to-speech | `/v1/audio/speech` |
| `moss-image` | image generation | `/v1/images/generations` |
| `docling` | document conversion | `/v1/documents/convert` |
| `gliner2.5-multi` | entity extraction | `/v1/entities` |

A client should not hard-code ids; operators rename and replace models in the config.

### `GET /v1/models/catalog`

**Hypnos extension.** The model catalog (`models.catalog.toml` next to the server): models Hypnos
has been tested with, configured here or not, each with its quantization variants, the files a
variant needs, what of it is already in the server's models folder, and the `[[models]]` config
block that runs it. A client picks a variant, downloads its files into `models_dir` (each file:
`repo` + `remote`, see below) and adds `config_block` to the config. Open during startup staging.
404 `not_found` when the server has no catalog file. `?modality=vision` keeps the entries serving
that modality (primary or additional).

```json
{ "object": "list", "catalog_version": 1,
  "models_dir": "/srv/hypnos/models",
  "tested_architectures": ["gemma4", "gemma-embedding2", "lfm3"],
  "data": [ {
    "id": "lfm3-vl:2b", "name": "LFM3-VL 2B", "publisher": "Liquid AI", "license": "lfm1.0",
    "architecture": "lfm3", "modality": "vision", "additional_modalities": ["text_generation"],
    "engine": "ggml_llama", "capabilities": ["thinking"],
    "summary": "Small vision-language model: ...", "lineup": false,
    "configured": true, "enabled": true,
    "variants": [ {
      "quant": "Q8_0", "default": true, "size_bytes": 2225234528,
      "files": [
        { "role": "weights", "repo": "LiquidAI/LFM3-VL-2B-GGUF", "remote": "LFM3-VL-2B-Q8_0.gguf",
          "local": "LFM3-VL-2B-Q8_0.gguf", "size": 1846238496, "platform": "all", "present": true },
        { "role": "aux", "...": "..." } ],
      "installed": true, "outdated": false,
      "config_block": "# LFM3-VL 2B (Q8_0), Liquid AI, license lfm1.0. ...\n[[models]]\nid = \"lfm3-vl:2b\"\n..." } ] } ] }
```

| Field | Meaning |
| :--- | :--- |
| `architecture` | for `engine: "ggml_llama"`, the GGUF's `general.architecture`; always one of `tested_architectures` |
| `capabilities` | beyond the modalities: `thinking` (`reasoning_effort`, 7.1), `tools` (7.5) |
| `lineup` | part of the default lineup (`models.manifest`) |
| `configured`, `enabled` | the config has a `[[models]]` block with this id; `enabled` only when configured |
| `variants[].size_bytes` | this machine's files (`files[].platform`: `all`, `macos-arm64`, `other`) |
| `files[].role` | the config key the file fills: `weights`, `aux` (projector or VAE), `text_encoder`, `tokenizer`, `voices`; `data` = needed beside the weights, named by no key |
| `files[].repo`, `remote` | a Hugging Face repository (download `https://huggingface.co/<repo>/resolve/main/<remote>`) or an `https://` base URL `remote` is appended to; a `remote` ending in `/` is a whole folder |
| `files[].local` | path under `models_dir`; `entry`, when present, is the file inside a folder the config names |
| `files[].present`, `outdated` | on disk (a folder: not empty); `outdated` = present with another size than listed |
| `installed`, `outdated` | every file present; some present file has another size (re-download it) |
| `problem` | present only when the downloaded weights' GGUF reports another architecture than the entry |
| `config_block` | the `[[models]]` + `[models.arch]` TOML for this variant, paths relative to `models_dir`; `hypnos --catalog-block <id>[:<quant>]` prints the same |

### `POST /v1/models/{id}/enable`, `POST /v1/models/{id}/disable`

**Hypnos extension.** Takes a model out of service (it is evicted to disk and hidden from
`/v1/models`; requests answer 404 `model_disabled`) or puts it back (it loads on its next
request). Not persisted to the config file. Disable is refused with 429 `model_busy` while the
model is leased. Response:

```json
{ "model": "gemma4:e4b-q4", "enabled": false, "placement": { "...": "see appendix" } }
```

### `POST /v1/models/{id}/moe-offload`

**Hypnos extension.** Sets a Mixture-of-Experts model's expert offload: its expert tensors stay
in host memory and run on the CPU while attention, the dense layers and the KV cache stay on the
GPU, so the model needs far less VRAM (the 26B: about 2.8 GiB of weights instead of 14.7 GiB) at
the cost of speed (on Apple Silicon: decode about half, prefill about 14x slower). Default from
the model's `moe_offload` in the config; not persisted.

```json
POST /v1/models/gemma4:26b-q4/moe-offload
{ "mode": "auto" }
```

| `mode` | Effect |
| :--- | :--- |
| `"off"` | experts follow their layer (default) |
| `"auto"` | when the model does not fit the GPU whole, the experts of as few layers as needed move to the CPU before any layer split; discrete GPUs only (no effect on unified memory) |
| `"on"` | every layer's experts run on the CPU |

Response:

```json
{ "model": "gemma4:26b-q4", "moe_offload": "auto", "applies": "now",
  "placement": { "...": "see appendix" } }
```

`applies` is `"now"` when the model was idle or not loaded (an idle resident model is unloaded
to Tier 2, and its next request loads it with the new placement; with KV parking its
conversations come back too) and `"next_load"` when it was busy. A dense model accepts the mode
and ignores it. An unknown mode is 422, an unknown model 404. `/v1/memory/status` shows each
model's `moe_offload`, `expert_bytes` (0 for a dense model) and `placement.cpu_expert_layers`.

---

## 7. Chat completions (text, vision, audio in, tools)

### `POST /v1/chat/completions`

#### 7.1 Request

```json
{
  "model": "gemma4:26b-q4",
  "messages": [
    { "role": "system", "content": "You are terse." },
    { "role": "user", "content": "Explain sliding-window attention in two sentences." }
  ],
  "max_tokens": 256,
  "temperature": 0.7,
  "top_p": 0.95,
  "top_k": 40,
  "seed": 42,
  "stream": false,
  "num_ctx": 8192,
  "think": false
}
```

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `model` | string | required | a model whose modalities include `text_generation` (or `vision` / `speech_to_text` for media parts) |
| `messages` | array | required, non-empty | roles `system`, `user`, `assistant`, `tool`; `content` is a string or an array of parts (below), or `null` on an assistant message that only calls tools (7.5) |
| `max_tokens` | int | 512 | clamped to `limits.max_tokens_cap` (8192 by default) |
| `temperature`, `top_p`, `top_k`, `seed` | number | engine defaults | sampling; `seed` makes a run reproducible on the same build |
| `stream` | bool | false | SSE response, see 7.4 |
| `num_ctx` | int | estimated | **Hypnos extension**: context window to reserve for this request (prompt plus generation). Unset, the server estimates it from the prompt length, the media and `max_tokens`, with a floor of 2048. Ask for more when the conversation will grow; a value above the model's `allocated_context_len` forces a context rebuild |
| `think` | bool | false | **Hypnos extension**: turns on the model's thinking (Gemma 4's thinking channel, LFM3's `<think>` block; other models ignore it) without a budget. The thoughts come back as `reasoning_content` (7.2, 7.4) and count in `completion_tokens` and in latency. `reasoning_effort` takes precedence when both are sent |
| `reasoning_effort` | string | unset (`think` decides) | the thinking level: `"none"` turns thinking off (a thinking block the model opens anyway is closed at once), `"low"`, `"medium"`, `"high"` turn it on with a budget of 512, 2048 or 6144 thinking tokens; when the budget is spent the server closes the thinking block and the model answers. `"minimal"` is read as `"low"`. Only models with a thinking format use it (Gemma 4, LFM3 chat models); others ignore it. `max_tokens` covers thoughts plus answer, and the thoughts count in `completion_tokens` |
| `tools` | array | none | functions the model may call, see 7.5 |
| `tool_choice` | string or object | `"auto"` | `"none"`, `"auto"`, `"required"`, or `{"type": "function", "function": {"name": "..."}}`, see 7.5 |

Not supported (ignored if sent): `n`, `functions` / `function_call` (the legacy form of tools), `parallel_tool_calls`, `response_format`,
`logprobs`, `stop`, `presence_penalty`, `frequency_penalty`, `logit_bias`, `user`. There is no
`/v1/completions` route.

**Content parts.** A message's `content` may be an array:

```json
{ "role": "user", "content": [
  { "type": "text", "text": "What is in this picture, and what does the speaker say?" },
  { "type": "image_url", "image_url": { "url": "data:image/png;base64,iVBORw0...", "detail": "auto" } },
  { "type": "input_audio", "input_audio": { "data": "<base64 wav>", "format": "wav" } }
] }
```

| Part | Fields | Notes |
| :--- | :--- | :--- |
| `text` | `text` | |
| `image_url` | `image_url.url` (`data:` URL or `http(s)` URL), `image_url.detail` (ignored) | PNG, JPEG, WebP and the other formats the `image` crate decodes; remote URLs need `allow_remote_image_urls`; up to `max_image_bytes`. The model must serve `vision` |
| `input_audio` | `input_audio.data` (base64), `input_audio.format` (default `wav`) | WAV is the reliable format (16-bit PCM, mono preferred); other containers go through the engine's decoder on a best-effort basis. The model must serve `speech_to_text` |

Media parts are placed in the prompt where they appear in the message. The server's context
estimate budgets 576 tokens per image and 1024 per audio clip; pass `num_ctx` when sending many.

#### 7.2 Response (non-streaming)

```json
{
  "id": "chatcmpl-6c1d...", "object": "chat.completion", "created": 1793481600,
  "model": "gemma4:26b-q4",
  "choices": [ { "index": 0,
                 "message": { "role": "assistant", "content": "Sliding-window attention ..." },
                 "finish_reason": "stop" } ],
  "usage": { "prompt_tokens": 31, "completion_tokens": 58, "total_tokens": 89,
             "prompt_tokens_details": { "cached_tokens": 24 } },
  "hypnos_telemetry": { "...": "section 5" }
}
```

With thinking on (`reasoning_effort` other than `"none"`, or `think`) and a model that thinks,
`message.reasoning_content` holds the thoughts that came before the answer; the field is omitted
when there were none. `content` is the answer alone.

`finish_reason` is `stop` (end of turn), `length` (`max_tokens` reached) or `tool_calls` (the
model called tools; `message.tool_calls` lists them and `content` is usually empty, see 7.5).
`choices` always has one element. `prompt_tokens_details.cached_tokens` is present only when part of the prompt was
served from the execution slot's KV cache (prefix reuse): a client that keeps the conversation
prefix stable (same system prompt, append-only history) gets faster prefill.

#### 7.3 Concurrency

Each text model has `slots` execution slots (3 for the default 26B); requests beyond the slots
queue, and beyond `max_queue_per_model` are refused with 429 `model_busy`. Requests that share a
decode step show `concurrent_slots > 1` in the telemetry.

#### 7.4 Streaming (`"stream": true`)

The response is `text/event-stream`. Every event is a `data:` line holding one JSON chunk, with
the OpenAI shape, and the stream ends with the literal `data: [DONE]`. Keep-alive comments
(`: ...`) may appear between events; ignore them.

```text
data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":1793481600,"model":"gemma4:26b-q4","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…","choices":[{"index":0,"delta":{"content":"Sliding"},"finish_reason":null}]}

data: {"id":"chatcmpl-…","object":"chat.completion.chunk","created":…,"model":"…","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":31,"completion_tokens":58,"total_tokens":89},"hypnos_telemetry":{…}}

data: [DONE]
```

Sequence:

1. One chunk with `delta.role = "assistant"` and empty content (the OpenAI convention).
2. With thinking on, zero or more chunks with `delta.reasoning_content` (pieces of the thoughts),
   all before the first `delta.content`.
3. Zero or more chunks with `delta.content` (a token or a few characters; special markers and the
   thinking block's delimiters are never emitted).
4. With tools, one chunk per call with `delta.tool_calls` (7.5).
5. One final chunk with an empty `delta`, `finish_reason` `stop`, `length` or `tool_calls`, plus
   `usage` and `hypnos_telemetry` (no `stream_options.include_usage` needed; usage always comes).
6. `[DONE]`.

If generation fails after the stream started, step 5 is instead a chunk with
`finish_reason: "error"` and an `error` object at the top level
(`{"message": "...", "type": "server_error"}`), followed by `[DONE]`. A client must treat that
chunk as a failure even though the HTTP status was 200.

Closing the connection cancels generation at the next token and releases the model. The request
timeout applies to the headers only; a stream can run as long as `max_tokens` takes.

#### 7.5 Tool calling

OpenAI-compatible function calling. The client describes functions in `tools`; the model either
answers or asks for one or more calls; the client runs them and sends the results back as
`tool` messages; the model then answers with them. The server never runs a tool itself.

```json
{ "model": "gemma4:26b-q4",
  "messages": [ { "role": "user", "content": "What's the weather in Paris?" } ],
  "tools": [ { "type": "function", "function": {
      "name": "get_weather", "description": "Current weather for a city",
      "parameters": { "type": "object",
                      "properties": { "city": { "type": "string" },
                                      "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] } },
                      "required": ["city"] } } } ] }
```

The answer to that:

```json
{ "choices": [ { "index": 0, "finish_reason": "tool_calls",
    "message": { "role": "assistant", "content": "",
      "tool_calls": [ { "id": "call_1f0c…_1", "type": "function",
        "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } } ] } } ], … }
```

`arguments` is a JSON object serialized as a string, as in OpenAI. To continue, append the
assistant message (with its `tool_calls`; `content` may be `null`) and one `tool` message per
call, then send the conversation again with the same `tools`:

```json
{ "role": "assistant", "content": null, "tool_calls": [ { "id": "call_1f0c…_1", "type": "function",
    "function": { "name": "get_weather", "arguments": "{\"city\":\"Paris\"}" } } ] },
{ "role": "tool", "tool_call_id": "call_1f0c…_1", "content": "{\"temperature\": 18, \"condition\": \"sunny\"}" }
```

A `tool` message's `content` is a string (send JSON as a string; it reaches the model as text) or
an array of text parts; `name` is optional, the function is found from `tool_call_id`.

| `tool_choice` | Effect |
| :--- | :--- |
| `"auto"` (default) | the model decides whether to call |
| `"none"` | the tools are not shown to the model; it answers in text |
| `"required"` | the model must call one of the tools |
| `{"type": "function", "function": {"name": "x"}}` | the model must call `x` (400 if `x` is not in `tools`) |

**Streaming.** Each call arrives whole in one chunk, `delta.tool_calls` holding one element with
its `index` (0, 1, … in the order of the calls), `id`, `type` and the complete `function.name` and
`function.arguments`; OpenAI clients that concatenate argument fragments by `index` work
unchanged. Text the model writes before a call arrives as `delta.content` as usual. The final
chunk has `finish_reason: "tool_calls"`.

**Models.** Tool calling uses the model's native format: Gemma 4 (all the default chat models).
A request with `tools` (and `tool_choice` other than `"none"`) for a model without one is refused
with 400 `invalid_request_error`. `"required"` and a named function are enforced by starting the
model's answer with a call, not by a grammar: the arguments follow the schema as well as the model
manages. Validate them before running the tool.

---

## 8. Embeddings

### `POST /v1/embeddings`

```json
{ "model": "gemmaembedding2", "input": ["first text", "second text"], "encoding_format": "float" }
```

`input` is one string or an array of strings (`max_prompt_chars` in total). `encoding_format`
is accepted for compatibility; vectors are always returned as JSON float arrays (`base64` is not
implemented).

```json
{ "object": "list", "model": "gemmaembedding2",
  "data": [ { "object": "embedding", "index": 0, "embedding": [0.0123, -0.0456, "…"] },
            { "object": "embedding", "index": 1, "embedding": ["…"] } ],
  "usage": { "prompt_tokens": 14, "total_tokens": 14 },
  "hypnos_telemetry": { "…": "section 5" } }
```

`data` is in input order. The vector length is the model's (768 for the default embedding
model); a client should read it from the first response rather than assume it.

---

## 9. Audio transcriptions

### `POST /v1/audio/transcriptions`

**Multipart** (recommended; what the OpenAI SDKs send):

| Part | Required | Notes |
| :--- | :--- | :--- |
| `model` | yes | a model serving `speech_to_text` (`gemma4:e4b-q4`) |
| `file` | yes | the audio file; WAV (16-bit PCM) is the reliable format, up to `max_audio_bytes` |
| `language` | no | hint, e.g. `en`, `it` |
| `prompt` | no | text hint given to the model before the audio |

The multipart part names are the only ones read; `response_format`, `temperature` and
`timestamp_granularities` are ignored (the answer is always JSON without timestamps).

**JSON** is also accepted, with the same field names and the audio as an array of byte values in
`audio_bytes` (`{"model": "…", "audio_bytes": [82, 73, 70, 70, …], "filename": "clip.wav"}`).
It exists for completeness; use multipart.

```json
{ "text": "The quick brown fox …", "duration_secs": 4.21, "hypnos_telemetry": { "…": "" } }
```

`language` is echoed when given. `duration_secs` is read from the WAV header when the file is a
WAV. The telemetry's `engine.audio_secs` and `engine.rtf` (processing seconds per audio second)
say how fast the transcription ran.

The same model also accepts audio inside a chat message (`input_audio` part, section 7.1), which
is the way to ask questions about a recording instead of transcribing it.

---

## 10. Speech synthesis

### `POST /v1/audio/speech`

```json
{ "model": "kokoro-82m-q4", "input": "Hello from Hypnos.", "voice": "alloy",
  "response_format": "wav", "speed": 1.0, "stream": false }
```

| Field | Default | Notes |
| :--- | :--- | :--- |
| `model` | required | a model serving `text_to_speech` |
| `input` | required | text, up to `max_prompt_chars`; long inputs are synthesized in phrase chunks with 180 ms of silence between them |
| `voice` | `alloy` | OpenAI names map to Kokoro voices: `alloy`→`af_alloy`, `echo`→`am_echo`, `fable`→`bm_fable`, `onyx`→`am_onyx`, `nova`→`af_nova`, `shimmer`/`coral`→`af_bella`, `ash`→`am_adam`, `sage`→`af_sarah`, `ballad`→`bm_george`, `verse`→`am_michael`; any other name falls back to `af_heart`. Raw Kokoro ids (`af_heart`, `bf_emma`, …) are accepted as-is |
| `response_format` | `wav` | `wav` (16-bit PCM, mono, 24 kHz, `audio/wav`) or `pcm` (the same samples without a header, `audio/pcm`). `mp3`, `opus`, `aac`, `flac` are not implemented: the server falls back to WAV, so check the `Content-Type` |
| `speed` | 1.0 | playback rate multiplier |
| `stream` | false | **Hypnos extension**: chunked audio as it is synthesized, see below |

**Response.** The audio bytes with `Content-Type: audio/wav` or `audio/pcm`, plus the
`x-hypnos-*` headers of section 5. There is no JSON wrapper.

**Streaming.** With `"stream": true` the body is sent in chunks as each phrase is synthesized:
for `wav` the first chunk begins with a WAV header whose length fields declare an open-ended
stream (`0xFFFFFFFF`), so a player can start immediately; for `pcm` the chunks are raw 16-bit
little-endian mono samples at 24 kHz. The telemetry headers present at the start are
`x-hypnos-previous-tier`, `x-hypnos-gpu-layers` and `x-hypnos-switch-ms`; the rest of the
timings are in the request log only. If synthesis fails mid-stream the body simply ends early;
a client that needs certainty should compare the samples received with the input length or use
the non-streaming form.

---

## 11. Image generation

### `POST /v1/images/generations`

```json
{ "model": "moss-image", "prompt": "a lighthouse at dawn, oil painting",
  "negative_prompt": "text, watermark", "n": 1, "size": "1024x1024",
  "response_format": "b64_json", "steps": 20, "seed": 7, "cfg_scale": 3.0 }
```

| Field | Default | Notes |
| :--- | :--- | :--- |
| `model` | required | a model serving `image_generation` |
| `prompt` | required | |
| `negative_prompt` | none | **Hypnos extension** |
| `n` | 1 | up to `max_images_per_request` (4) |
| `size` | `1024x1024` | `WIDTHxHEIGHT`; each side is clamped to 256..2048 and rounded down to a multiple of 16 (`1000x700` becomes `992x688`) |
| `response_format` | `b64_json` | only `b64_json` is implemented; `url` is accepted but still answers base64 |
| `steps` | 20 | **Hypnos extension**, clamped to 1..100 |
| `seed` | random | **Hypnos extension** |
| `cfg_scale` | 3.0 | **Hypnos extension** |

`quality`, `style` and `user` are ignored.

```json
{ "created": 1793481600,
  "data": [ { "b64_json": "iVBORw0KGgo…" } ],
  "hypnos_telemetry": { "…": "section 5" } }
```

Images are PNG. One 1024x1024 generation takes tens of seconds on a 24 GB GPU and more than two
minutes when the generator runs partially offloaded; keep the client timeout at the server's
`request_timeout_secs` for this route.

---

## 12. Document conversion and pictures

**Hypnos extension.** Converts PDFs, images and the office and web formats docling reads into
Markdown (or docling's JSON), one page at a time, with optional OCR, picture descriptions by a
vision model and a retrievable picture store.

### `POST /v1/documents/convert`

**Multipart** parts, or the same names as **JSON** fields:

| Field | Type | Default | Notes |
| :--- | :--- | :--- | :--- |
| `model` | string | required | a model serving `document_conversion` (`docling`) |
| `file` | file part / string | one of `file`, `url` | multipart: the document; JSON: base64 or a `data:` URI. Up to `max_document_bytes` |
| `url` | string | | fetch the document from an `http(s)` URL (needs `allow_remote_image_urls`) |
| `filename` | string | from the part or the URL | its extension helps detect the format |
| `pages` | string | all | 1-based page or inclusive range: `"3"`, `"2-5"` |
| `ocr` | bool | true | read pages without a text layer (scans) with OCR |
| `output_format` | string | `markdown` | `markdown` or `json` (adds each page's docling document as `json`) |
| `stream` | bool | false | one SSE event per page, see below |
| `describe_pictures` | string | none | a vision model id (e.g. `gemma4:e4b-q4`); its description of every picture is inserted under the picture's `<!-- image -->` placeholder as `> **Picture:** …` |
| `return_pictures` | bool | false | keep the pictures in the server's picture store and return a document `id` plus per-page `picture_urls` |

Booleans in multipart parts accept `true/false`, `1/0`, `yes/no`, `on/off`.

**Response (non-streaming):**

```json
{
  "object": "document",
  "id": "doc-5f2c…",                       // return_pictures only
  "pictures_expire_at": "2026-10-30T19:42:11Z",   // return_pictures only
  "model": "docling", "filename": "report.pdf",
  "page_count": 12,
  "pages": [
    { "page": 2, "markdown": "## Results\n\n<!-- image -->\n\n> **Picture:** A bar chart …",
      "json": { "…": "output_format = json only" },
      "pictures": 1, "tables": 2, "ms": 310.4,
      "picture_urls": ["/v1/documents/doc-5f2c…/pictures/1"] }
  ],
  "markdown": "…the requested pages joined, with tables, numbered lists and sentences mended across page breaks…",
  "usage": { "pages": 4, "pictures": 3, "tables": 5, "pictures_described": 3 },
  "hypnos_telemetry": { "…": "engine.pages and engine.ms_per_page" }
}
```

`page_count` is the source document's page count; `pages` holds only the converted ones.
Formats without pages (DOCX, HTML, images) come back as one page numbered 1.

**Streaming** (`stream = true`): `text/event-stream`, one JSON object per `data:` line,
distinguished by `object`:

| `object` | When | Fields |
| :--- | :--- | :--- |
| `document.started` | once the document is read | `page_count`, `pages` (the page numbers that will follow), `id` (with `return_pictures`) |
| `document.page` | per converted page, as soon as it is ready | the page object above |
| `document` | last event | the non-streaming response (joined `markdown`, `usage`, telemetry) with `pages` empty: the pages already came as `document.page` events |
| `error` | instead of `document` if the conversion fails | `error.message`, `error.type` |

The stream then ends with the literal `data: [DONE]`, as chat streams do.

A request the engine refuses up front (bad file, unsupported format) gets its HTTP error status;
the stream only starts once the first event exists. Closing the connection stops the conversion at
the next page. The `document.page` events let a client show pages progressively; to store the
result, take the final `document` event and put the collected `document.page` events into its
`pages`.

### Pictures of a converted document

Available only for a conversion made with `return_pictures = true`, until
`pictures_expire_at` (the operator's `server.document_pictures_ttl_secs`, 1 h by default) or an
explicit delete, whichever comes first. The store also has a size cap; when it is full the oldest
documents go first. These routes stay open during startup staging and need no model.

| Route | Returns |
| :--- | :--- |
| `GET /v1/documents/{id}/pictures` | `{"object":"list","document_id":"…","expires_at":"…","data":[…]}`; the records carry no `image` |
| `GET /v1/documents/{id}/pictures/{n}` | one picture record **with** `image` (base64) |
| `GET /v1/documents/{id}/pictures/{n}?format=image` | the bare image bytes with their `Content-Type` (`image/png`) and `Cache-Control: private, no-store` |
| `DELETE /v1/documents/{id}` | `{"object":"document.deleted","id":"…","deleted":true}`; releases the pictures |

A picture record:

```json
{ "object": "document.picture", "document_id": "doc-5f2c…", "index": 1, "page": 2,
  "url": "/v1/documents/doc-5f2c…/pictures/1",
  "mime": "image/png", "bytes": 48213, "width": 640, "height": 420, "dpi": 144,
  "bbox": { "l": 72.0, "t": 540.2, "r": 380.5, "b": 318.9, "coord_origin": "BOTTOMLEFT" },
  "page_size": { "width": 612.0, "height": 792.0 },
  "caption": "Figure 3: revenue by quarter", "ref": "#/pictures/0",
  "image": "iVBORw0KGgo…" }
```

`index` is 1-based and counts pictures across the document in Markdown order; `bbox` and
`page_size` are in the page's units (PDF points for PDFs) with `coord_origin` saying whether `t`
and `b` count from the top or the bottom. `caption` and `ref` are present only when docling found
them. Unknown ids and indexes answer 404 `not_found`.

---

## 13. Entity extraction

### `POST /v1/entities`

**Hypnos extension.** Zero-shot named-entity extraction: the labels are free text.

```json
{ "model": "gliner2.5-multi",
  "input": ["Mario Rossi works in Milan for Ferrari.", "Tim Cook announced the iPhone 17 in Cupertino."],
  "labels": ["person", "city", "company", "product"],
  "threshold": 0.5, "flat": true }
```

| Field | Default | Notes |
| :--- | :--- | :--- |
| `model` | required | a model serving `entity_extraction` |
| `input` | required | one string or an array |
| `labels` | required | any phrases; more than 8 labels are run in families, long texts in windows, transparently |
| `threshold` | 0.5 | minimum span score |
| `flat` | model default | `true`: only non-overlapping spans (highest score wins); `false`: nested spans too |

```json
{ "object": "list", "model": "gliner2.5-multi",
  "data": [
    { "object": "entities", "index": 0, "entities": [
        { "text": "Mario Rossi", "label": "person", "start": 0, "end": 11, "score": 0.97 },
        { "text": "Milan", "label": "city", "start": 21, "end": 26, "score": 0.93 },
        { "text": "Ferrari", "label": "company", "start": 31, "end": 38, "score": 0.91 } ] },
    { "object": "entities", "index": 1, "entities": ["…"] } ],
  "usage": { "texts": 2, "characters": 86 },
  "hypnos_telemetry": { "…": "section 5" } }
```

`start` and `end` are character offsets (`[start, end)`, Unicode scalar values, not bytes and not
UTF-16 units): JavaScript clients must convert before slicing strings with astral characters.

---

## 14. Memory status and placement control

**Hypnos extension.** These are the routes the playground's tier visualizer uses.

### `GET /v1/memory/status`

Open during startup staging. The full snapshot of the three tiers (bytes unless stated):

```json
{
  "capabilities": { "accelerator": "NvidiaCuda", "topology": "DiscretePcie",
                    "device_name": "NVIDIA GeForce RTX 5090 Laptop GPU",
                    "total_vram_bytes": 25648562176, "usable_vram_budget_bytes": 25648562176,
                    "pinned_ram_budget_bytes": 34359738368, "physical_memory_bytes": 68719476736,
                    "simulated": false, "installed_memory_bytes": 68719476736,
                    "tier2_to_tier1_bandwidth_gbps": 25.0, "tier3_to_tier2_bandwidth_gbps": 6.0 },
  "tier1_vram_used_bytes": 22145000000, "tier1_vram_free_bytes": 3503562176,
  "tier1_vram_real_free_bytes": 1240000000, "tier1_vram_real_total_bytes": 25648562176,
  "tier1_vram_unaccounted_bytes": 2200000000, "tier1_vram_effective_free_bytes": 1240000000,
  "tier2_pinned_ram_used_bytes": 17000000000, "tier2_pinned_ram_free_bytes": 17359738368,
  "physical_pool_free_bytes": null,
  "process_footprint_bytes": 31500000000, "process_memory_bytes": 31500000000,
  "process_memory_cap_bytes": null,
  "gpu": { "utilization_percent": 87, "temperature_celsius": 71, "power_watts": 148.2,
           "sm_clock_mhz": 2400, "throttle_reasons": 0, "throttled": false,
           "ecc_uncorrected_errors": 0, "retired_pages_pending": false, "sampled_unix_ms": 1793481600123 },
  "models": [ {
      "id": "gemma4:26b-q4", "enabled": true, "modality": "text_generation",
      "modalities": ["text_generation", "vision"], "engine": "ggml_llama",
      "total_weight_bytes": 15814641056, "priority": "high",
      "current_tier": "tier1_gpu_vram", "placement": { "…": "appendix" },
      "moe_offload": "off", "expert_bytes": 12846382080,
      "active_leases": 1, "idle_seconds": 0.0, "total_requests_served": 812 } ],
  "recent_evictions": [ { "model": "moss-image", "for_model": "gemma4:26b-q4", "reason": "…",
                          "to_tier": "tier3_nvme_ssd", "unix_ms": 1793481500000 } ]
}
```

`tier1_vram_used_bytes` is what the planner has booked; `tier1_vram_real_free_bytes` is what the
device reports; `tier1_vram_unaccounted_bytes` is the difference the planner subtracts from what it
hands out; `tier1_vram_effective_free_bytes` is what a new promotion can use right now.
`kv_park` is the state of KV parking (`/v1/memory/kv-park` below): `{"enabled", "max_bytes", "used_bytes", "models":
[{"model", "slots", "tokens", "bytes", "parked_secs_ago"}]}`.
`recent_evictions` lists the last automatic evictions with their `reason` (`memory_budget`,
`full_gpu_placement`, `process_memory_cap`, `device_out_of_memory`). `gpu` is
present on NVIDIA devices with `nvidia-smi` available and is sampled at most every few seconds.
Additional fields may appear in later versions; clients should ignore unknown ones.

### `POST /v1/memory/tier`

Moves a model by hand.

```json
{ "model": "gemma4:26b-q4", "target_tier": "tier2_pinned_ram", "context_len": 16384 }
```

`target_tier` is `tier1_gpu_vram`, `tier2_pinned_ram` or `tier3_nvme_ssd`; `context_len`
(optional) is the context to allocate when promoting to Tier 1. Refused with 429 `model_busy`
while the model is leased, 507 when it does not fit, 404 when unknown or disabled. Response:
`{"model": "…", "placement": {…}}`. A promotion blocks for as long as the load takes.

### `GET /v1/memory/kv-park`, `POST /v1/memory/kv-park`

**Hypnos extension.** KV parking keeps a llama.cpp model's conversations when it leaves
Tier 1: each used slot's KV state is saved in host memory, and when the model comes back with
the same context layout (same `allocated_context_len`, slot count, KV type and weights) the slots
are restored before it takes requests. The next turn of a conversation then prefills only its new
tokens instead of the whole history (on Apple Silicon, a 12.6k-token conversation: 4.7 s -> 1.1 s
for the turn after a swap). Off by default (`hardware.kv_park`); `max_bytes` caps the host memory
the parked states may hold, the least recently parked models going first.

```json
POST /v1/memory/kv-park
{ "enabled": true, "max_bytes": 4294967296 }
```

Either field may be left out (an empty body is 400). Turning it off drops every parked state; a
lower cap drops the oldest until the rest fits. The change is not written back to the config.
Both routes answer the same object as `kv_park` in `/v1/memory/status`:

```json
{ "enabled": true, "max_bytes": 4294967296, "used_bytes": 360710144,
  "models": [ { "model": "gemma4:26b-q4", "slots": 1, "tokens": 12612,
                "bytes": 360710144, "parked_secs_ago": 4.2 } ] }
```

A parked state that does not fit the new context (the model came back with another window or
slot count) is dropped and the conversation prefills as before; parking never changes an answer.
A disabled model's parked state is dropped. Slots emptied by `server.kv_idle_timeout_secs` have
nothing to park.

---

## 15. Request log

**Hypnos extension.** A ring buffer (200 entries by default) of recent inference calls, meant
for the playground and for debugging. Open during startup staging. Polling it is not logged.

| Route | Body / query | Returns |
| :--- | :--- | :--- |
| `GET /v1/logs` | `?model=&route=&status=ok\|error\|4xx\|5xx&key=&limit=` | `{"level":"info","capacity":200,"stored":57,"records":[…]}`, newest first |
| `GET /v1/logs/{id}` | the `x-request-id` | one record, or 404 `not_found` |
| `DELETE /v1/logs` | | `{"cleared": true}` |
| `POST /v1/logs/level` | `{"level":"off\|error\|warn\|info\|debug\|trace"}` | `{"level":"debug"}`; changes what the next calls capture |

A record: `id`, `route`, `method`, `status`, `key` (label), `started_at` (RFC 3339),
`duration_ms`, `model`, `stream`, `tokens_in`, `tokens_out`, `cached_tokens`, `tokens_per_s`,
`inference_ms`, `tier_switch_ms`, `previous_tier`, `error`, `level`, `truncated`, and at `debug`
or `trace` also `input`, `output` (`debug`) and `request_payload`, `response_payload` (`trace`).
The operator's `request_log_redact` may blank or pattern-redact the captured text. Treat the
captured text as sensitive; do not build product features on this route.

---

## 16. Metrics

`GET /metrics` returns Prometheus text (`text/plain; version=0.0.4`). It is open unless the
operator sets `server.metrics_require_auth`. The families a client dashboard may care about:
`hypnos_http_requests_total{route,class}`, `hypnos_http_request_latency_ms_{sum,max}`,
`hypnos_chat_ttft_ms`, `hypnos_chat_decode_tokens_per_s`, `hypnos_chat_prefill_tokens_per_s`,
`hypnos_prefix_cache_{hits,misses,tokens_reused}_total`, `hypnos_tts_rtf`, `hypnos_tts_ttfa_ms`,
`hypnos_stt_rtf`, `hypnos_image_jct_s`, `hypnos_document_ms_per_page`,
`hypnos_document_pages_total`, `hypnos_scheduler_{promotions,evictions,load_failures,admissions_rejected}_total`,
`hypnos_model_load_failures_total{model}`, `hypnos_model_tier{model}`,
`hypnos_model_gpu_layers{model}`, `hypnos_model_active_leases{model}`,
`hypnos_model_requests_total{model}`, `hypnos_vram_bytes{kind}`, `hypnos_pinned_ram_bytes{kind}`,
`hypnos_gpu_*`, `hypnos_api_key_{requests,rejections,tokens}_total{key}`,
`hypnos_api_key_tokens_today{key}`, `hypnos_uptime_seconds`, `hypnos_process_start_time_seconds`.
The full list with help strings is in the endpoint's own output.

---

## 17. Writing a client: recommendations

These are the behaviours a well-behaved client or SDK should implement; they follow from the
sections above.

**Configuration.** Base URL (default `http://127.0.0.1:9999`), optional API key, a request
timeout per route family, a `User-Agent`. Keep the timeout for chat, images and documents at or
above the server's `request_timeout_secs` (900 s by default); a cold request pays a model load.

**Startup.** Poll `GET /ready` with a short delay until it answers 200, or treat 503
`model_not_available` with `Retry-After` as "wait and retry". `GET /v1/models` works before that
and tells the client what it may ask for.

**Retries.** Retry, with backoff, on: 429 `model_busy` (no `Retry-After`; start at 250 ms),
429 `rate_limit_error` (sleep `Retry-After`), 503 `model_not_available` (sleep `Retry-After`,
default 5 s), 507 (another model holds the memory; sleep seconds, not milliseconds), 504 and
network errors before any byte was received. Do not retry 400, 401, 404, 413, or a 500 more than
once. Never retry a request whose stream already produced output; surface the partial result and
the error.

**Streaming.** Parse SSE by lines: `data:` payload, blank line ends an event; ignore `:` comment
lines; stop on `[DONE]`. Deliver `delta.content` pieces as they arrive (and the thoughts in
`delta.reasoning_content` apart from them); collect `usage` and `hypnos_telemetry` from the
final chunk; map `finish_reason: "error"` to an exception. Cancel by
closing the connection, nothing else is needed. For document streams dispatch on `object`.

**Cancellation.** Closing the socket cancels any in-flight request at the next token, chunk, step
or page and releases the model. SDKs should expose an abort signal that does exactly that.

**Concurrency.** A single client may keep several chat requests in flight; they batch on the
server up to the model's `slots`. Beyond that they queue, and beyond `max_queue_per_model` they
fail with 429. An SDK that fans out work should bound its parallelism per model.

**Uploads.** Use multipart for audio and documents; JSON with base64 inflates the body by a
third and the server's limits apply to the decoded size. Remote URLs are fetched by the server
only when the operator allows it.

**Prefix reuse.** Send conversations with a stable prefix (same system prompt, append-only
history). `usage.prompt_tokens_details.cached_tokens` and `engine.cached_tokens` show when it
paid off.

**OpenAI SDK users.** Point the SDK at `http://host:9999/v1`; `chat.completions.create` (with
`stream=True`), `embeddings.create`, `audio.transcriptions.create`, `audio.speech.create` and
`images.generate` work as-is. `hypnos_telemetry` is available through the SDK's raw-response or
extra-fields mechanism. The Hypnos-only routes (documents, entities, memory, models
enable/disable, logs) need a plain HTTP call.

**Versioning.** The response bodies are additive: new fields appear, existing ones keep their
meaning. A client must ignore unknown fields. The server's version is in `GET /health` and is
the release tag.

---

## 18. Appendix: shared types

**`MemoryTier`**: `"tier1_gpu_vram"`, `"tier2_pinned_ram"`, `"tier3_nvme_ssd"`.

**`Modality`**: `"text_generation"`, `"vision"`, `"speech_to_text"`, `"text_to_speech"`,
`"embeddings"`, `"image_generation"`, `"document_conversion"`, `"entity_extraction"`.

**`EngineKind`**: `"ggml_llama"`, `"ggml_whisper"`, `"ggml_diffusion"`, `"onnx_runtime"`,
`"docling"`, `"gliner"`.

**`ModelPriority`**: `"low"`, `"normal"`, `"high"`, `"realtime_pinned"`.

**`LayerPlacementPlan`** (in `placement` fields):

| Field | Meaning |
| :--- | :--- |
| `primary_tier` | the tier the model is served from |
| `gpu_layers`, `pinned_ram_layers`, `ssd_layers` | how the model's layers are split across the tiers; a partial GPU placement has both `gpu_layers` and `pinned_ram_layers` above 0 |
| `vram_weight_bytes` | weights resident in VRAM |
| `vram_kv_and_workspace_bytes` | KV cache plus compute workspace booked in VRAM |
| `pinned_ram_bytes` | bytes pinned in host RAM |
| `allocated_context_len` | the context window the current placement funds, across all slots |
| `cpu_expert_layers` | layers (counted from the last) whose Mixture-of-Experts expert tensors run on the CPU from host memory while the rest of the layer is on the GPU (`moe_offload`, section 6); their bytes are in `pinned_ram_bytes`, not `vram_weight_bytes`. 0 for dense models and with `moe_offload: off` |

**`EngineTiming`**: see section 5. All durations are milliseconds; rates are per second; `rtf` is
processing seconds per audio second.

**Error body**: section 4. **Picture record**: section 12.
