# hypnos-client (Rust)

Async Rust client for the [Hypnos](https://github.com/advancedlogic/hypnos) multi-modal
inference server: every route of the [HTTP API](../../docs/api.md), with streaming, the retry
policy the API recommends, readiness polling and typed responses. Built on `reqwest` (rustls)
and `tokio`.

```toml
[dependencies]
hypnos-client = "0.2"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

```rust
use hypnos_client::{ChatMessage, ChatRequest, Client};
use std::time::Duration;

#[tokio::main]
async fn main() -> hypnos_client::Result<()> {
    let hypnos = Client::new()?; // http://127.0.0.1:9999, or $HYPNOS_BASE_URL / $HYPNOS_API_KEY
    hypnos.wait_until_ready(Duration::from_secs(600), Duration::from_secs(1)).await?;
    let reply = hypnos
        .chat(&ChatRequest::new("gemma4:26b-q4", vec![ChatMessage::user("Hello")]))
        .await?;
    println!("{}", reply.text());
    println!("{:?}", reply.hypnos_telemetry.map(|t| t.tier_switch_latency_ms));
    Ok(())
}
```

`Client::builder()` sets the base URL, API key, timeout, retry policy and per-model bound.
Clones share the connection pool and the bound.

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
| `advise` | `POST /v1/advisor` (`AdvisorRequest::new(&[needs])`, optional `.vram_bytes()`, `.ram_bytes()`, `.unified_memory()`) |
| `add_model`, `replace_model`, `remove_model` | `POST /v1/models`, `PUT` / `DELETE /v1/models/{id}` (`ModelSource::Catalog` or `ModelSource::Config`, optional `drain_secs`; admin key) |
| `set_moe_offload` | `POST /v1/models/{id}/moe-offload` |
| `memory_status`, `set_tier` | `/v1/memory/status`, `/v1/memory/tier` |
| `kv_park`, `set_kv_park` | `GET`/`POST /v1/memory/kv-park` |
| `logs`, `log`, `clear_logs`, `set_log_level` | `/v1/logs*` |
| `health`, `is_ready`, `wait_until_ready`, `metrics` | `/health`, `/ready`, `/metrics` |

Requests are builder structs (`ChatRequest::new(model, messages).max_tokens(256)`,
`DocumentRequest::from_path("docling", "report.pdf")?.pages("2-5")`, …). Responses are serde
structs; optional fields are `Option`s, `hypnos_telemetry` is an `Option<Telemetry>`, and the
top-level responses keep fields this version does not know in `extra`. Speech returns a
`Speech` with `audio`, `content_type` and the `x-hypnos-*` headers as `telemetry`. Entity
offsets count characters, not bytes: slice with `slice_chars(text, start, end)`.

## Streaming and cancellation

```rust
use futures::StreamExt;

let mut stream = hypnos.chat_stream(&request).await?;
{
    let mut text = stream.text();
    while let Some(piece) = text.next().await {
        print!("{}", piece?);
    }
}
println!("{:?} {:?}", stream.usage(), stream.telemetry());
```

Every stream implements `futures::Stream`. Dropping it closes the connection, and the server
stops at the next token, page or audio chunk. `final_completion()` / `final_document()` read the
rest and return the whole result. A stream that fails after it started yields `Error::Stream`.
Document streams yield `DocumentEvent::Started`, `Page` and `Done`; `final_document()` returns
the document with its pages put back.

## Tool calling

`ChatRequest::tools` describes functions, `tool_choice` takes `ToolChoice::Auto` (the default),
`None`, `Required` or `ToolChoice::function("name")`. The server never runs a tool: when the
model asks for calls, `finish_reason()` is `"tool_calls"` and `tool_calls()` lists them; append
`completion.to_message()` (the assistant message with its calls, `content` `null`) and one
`ChatMessage::tool(call_id, result)` per call, then send the conversation again with the same
tools.

```rust
use hypnos_client::{ChatMessage, ChatRequest, ToolSpec};
use serde_json::json;

let tools = vec![ToolSpec::function(
    "get_weather",
    "Current weather for a city",
    json!({"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}),
)];
let mut messages = vec![ChatMessage::user("What's the weather in Paris?")];
let reply = hypnos
    .chat(&ChatRequest::new("gemma4:26b-q4", messages.clone()).tools(tools.clone()))
    .await?;
if reply.finish_reason() == "tool_calls" {
    messages.push(reply.to_message());
    for call in reply.tool_calls() {
        let args: serde_json::Value = call.parse_arguments()?; // validate before running
        let result = json!({"city": args["city"], "temperature": 18}).to_string();
        messages.push(ChatMessage::tool(&call.id, result));
    }
    let answer = hypnos
        .chat(&ChatRequest::new("gemma4:26b-q4", messages).tools(tools))
        .await?;
    println!("{}", answer.text());
}
```

`ChatMessage::assistant_with_calls(content, calls)` builds the assistant message by hand. A
`ChatStream` accumulates `delta.tool_calls` by `index` into `stream.tool_calls()`, and
`final_completion()` returns them with the finish reason. A named `tool_choice` that is not in
`tools` is `Error::BadRequest`.

## Errors and retries

`Error` has one variant per error type: `BadRequest` (400, 413), `Authentication` (401),
`PermissionDenied` (403), `NotFound` (404), `ModelBusy` and `RateLimit` (429), `ServiceUnavailable` (503),
`InsufficientMemory` (507), `Timeout` (408, 504), `Server` (500), `Stream` (a failed stream),
`Connection`, plus `Decode`, `InvalidArgument` and `NotReady`. The server's details
(`status`, `kind`, `code`, `request_id`, `retry_after`) are in `error.api()`.

The default `RetryPolicy` follows the API's recommendations: 429 `model_busy` from 250 ms
doubling, 429 `rate_limit_error` and 503 after `Retry-After`, 507 from 2 s doubling, 504 and
connection errors from 1 s, a 500 once; never 400, 401, 403 or 404, never a stream that already
produced output, and never the model pool changes (`add_model`, `replace_model`, `remove_model`:
a lost connection may have applied the change, and their 429 means the drain already timed
out). `RetryPolicy::none()` turns it off.

## Concurrency

`Client::builder().max_concurrency_per_model(3)` keeps at most three requests per model in
flight from this client (a stream counts until it is dropped). Beyond a model's execution slots
the server only queues requests, and beyond its queue it answers 429.

## Examples

`examples/` has one program per model of the default lineup: `chat`, `vision`, `transcribe`,
`embeddings`, `speech`, `images`, `documents`, `entities` (`cargo run --example chat -- "Hi"`).

## Development

```bash
cargo test     # starts backend/…/examples/mock_server (built if missing), one per test
cargo clippy --all-targets -- -D warnings
```

The tests run against the real Hypnos router with mock engines (`cargo run -p hypnos-server
--example mock_server`), so they check the actual status codes, headers and stream framing.
Set `HYPNOS_MOCK_SERVER` to a built binary to skip the cargo build.
