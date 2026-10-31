# hypnos-client (Python)

Python client for the [Hypnos](https://github.com/advancedlogic/hypnos) multi-modal inference
server: every route of the [HTTP API](../../docs/api.md), sync and async, with streaming, the
retry policy the API recommends, readiness polling and typed responses.

```bash
pip install hypnos-client
```

```python
from hypnos_client import Client

with Client() as hypnos:        # http://127.0.0.1:9999, or $HYPNOS_BASE_URL / $HYPNOS_API_KEY
    hypnos.wait_until_ready()
    reply = hypnos.chat("gemma4:26b-q4", [{"role": "user", "content": "Hello"}])
    print(reply["choices"][0]["message"]["content"])
    print(reply.get("hypnos_telemetry", {}).get("tier_switch_latency_ms"))
```

`AsyncClient` has the same methods as coroutines (`async with AsyncClient() as hypnos: ...`).

## Routes

| Method | Route |
| :--- | :--- |
| `chat`, `chat_stream` | `POST /v1/chat/completions` (text, `image_url` and `input_audio` parts, tools) |
| `embeddings` | `POST /v1/embeddings` |
| `transcribe` | `POST /v1/audio/transcriptions` (multipart upload) |
| `speech`, `speech_stream` | `POST /v1/audio/speech` |
| `images` | `POST /v1/images/generations` |
| `convert_document`, `convert_document_stream` | `POST /v1/documents/convert` (upload or `url`) |
| `document_pictures`, `document_picture`, `document_picture_bytes`, `delete_document` | `/v1/documents/{id}/pictures[/{n}]`, `DELETE /v1/documents/{id}` |
| `entities` | `POST /v1/entities` |
| `models`, `enable_model`, `disable_model` | `/v1/models`, `/v1/models/{id}/enable\|disable` |
| `models_catalog` | `/v1/models/catalog` (optional `modality`) |
| `advise` | `POST /v1/advisor` (`needs`, optional `vram_bytes`, `ram_bytes`, `unified_memory`) |
| `add_model`, `replace_model`, `remove_model` | `POST /v1/models`, `PUT` / `DELETE /v1/models/{id}` (`catalog=` or `config=`, optional `drain_secs`; admin key) |
| `set_moe_offload` | `POST /v1/models/{id}/moe-offload` |
| `memory_status`, `set_tier` | `/v1/memory/status`, `/v1/memory/tier` |
| `kv_park`, `set_kv_park` | `GET`/`POST /v1/memory/kv-park` |
| `logs`, `log`, `clear_logs`, `set_log_level` | `/v1/logs*` |
| `health`, `is_ready`, `wait_until_ready`, `metrics` | `/health`, `/ready`, `/metrics` |

Responses are `TypedDict`s (`hypnos_client.types`): plain dicts with type hints. Fields the
server adds later pass through. `hypnos_telemetry` is present on every inference response;
speech returns a `Speech` with `audio`, `content_type` and the `x-hypnos-*` headers as
`telemetry`.

## Streaming and cancellation

```python
with hypnos.chat_stream("gemma4:26b-q4", messages, max_tokens=400) as stream:
    for piece in stream.text():
        print(piece, end="", flush=True)
print(stream.usage, stream.telemetry)
```

Leaving the `with` block (or `stream.close()`) closes the connection, and the server stops at
the next token. `stream.final()` reads the rest and returns a normal completion. A stream that
fails after it started raises `StreamError`. Document streams yield `document.started`,
`document.page` and `document` events; `final()` returns the whole document with its pages.

## Tool calling

`chat` and `chat_stream` take `tools` and `tool_choice` (`"auto"`, `"none"`, `"required"` or
`{"type": "function", "function": {"name": ...}}`). The server never runs a tool: the model asks
for calls, you run them and send the results back.

```python
import json
from hypnos_client import assistant_message, function_tool, parse_arguments, tool_message

weather = function_tool(
    "get_weather", "Current weather for a city",
    {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
)
messages = [{"role": "user", "content": "What's the weather in Paris?"}]
reply = hypnos.chat("gemma4:26b-q4", messages, tools=[weather])
if reply["choices"][0]["finish_reason"] == "tool_calls":
    messages.append(assistant_message(reply))         # keeps tool_calls, content None
    for call in reply["choices"][0]["message"]["tool_calls"]:
        args = parse_arguments(call)                  # {"city": "Paris"}; validate before use
        messages.append(tool_message(call["id"], json.dumps({"temperature": 18})))
    reply = hypnos.chat("gemma4:26b-q4", messages, tools=[weather])
print(reply["choices"][0]["message"]["content"])
```

A stream accumulates the calls by `index` in `stream.tool_calls`, and `stream.final()` returns
them in `message.tool_calls` with `finish_reason` `"tool_calls"`. A `tool_choice` naming a
function that is not in `tools` raises `BadRequestError`.

## Errors and retries

Every error is a subclass of `HypnosError`; HTTP errors are `APIError`s with `status`, `type`,
`code`, `request_id` and `retry_after`: `BadRequestError` (400, 413), `AuthenticationError`
(401), `PermissionDeniedError` (403), `NotFoundError` (404), `ModelBusyError` and `RateLimitError` (429),
`ServiceUnavailableError` (503), `InsufficientMemoryError` (507), `APITimeoutError` (408, 504),
`ServerError` (500), `StreamError` (a failed stream) and `APIConnectionError`.

The default `RetryPolicy` follows the API's recommendations: 429 `model_busy` from 250 ms
doubling, 429 `rate_limit_error` and 503 after `Retry-After`, 507 from 2 s doubling, 504 and
connection errors from 1 s, a 500 once; never 400, 401, 403 or 404, never a stream that already
produced output, and never the model pool changes (`add_model`, `replace_model`, `remove_model`:
a lost connection may have applied the change, and their 429 means the drain already timed
out). `Client(retry=RetryPolicy(max_retries=8))` or `retry=NO_RETRY` changes it.

## Concurrency

`Client(max_concurrency_per_model=3)` keeps at most three requests per model in flight from
this client (a stream counts until it is closed). Beyond a model's execution slots the server
only queues requests, and beyond its queue it answers 429.

## Examples

`examples/` has one script per model of the default lineup: `chat.py`, `vision.py`,
`transcribe.py`, `embeddings.py`, `speech.py`, `images.py`, `documents.py`, `entities.py`.

## Development

```bash
uv sync
uv run python scripts/unasync.py   # after editing _async_client.py: regenerates _client.py
uv run pytest                      # starts backend/…/examples/mock_server (built if missing)
```

The tests run against the real Hypnos router with mock engines (`cargo run -p hypnos-server
--example mock_server`), so they check the actual status codes, headers and stream framing.
Set `HYPNOS_MOCK_SERVER` to a built binary to skip the cargo build.
