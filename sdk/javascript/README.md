# @hypnos/client (JavaScript / TypeScript)

JavaScript and TypeScript client for the [Hypnos](https://github.com/advancedlogic/hypnos)
multi-modal inference server: every route of the [HTTP API](../../docs/api.md), with streaming,
cancellation, the retry policy the API recommends, readiness polling and typed responses.
No runtime dependencies: it uses `fetch`, `ReadableStream` and `AbortController`, so it runs in
Node 18+ and in browsers.

```bash
npm install @hypnos/client
```

```ts
import { Client } from "@hypnos/client";

const hypnos = new Client(); // http://127.0.0.1:9999, or $HYPNOS_BASE_URL / $HYPNOS_API_KEY
await hypnos.waitUntilReady();
const reply = await hypnos.chat("gemma4:26b-q4", [{ role: "user", content: "Hello" }]);
console.log(reply.choices[0].message.content);
console.log(reply.hypnos_telemetry?.tier_switch_latency_ms);
```

## Routes

| Method | Route |
| :--- | :--- |
| `chat`, `chatStream` | `POST /v1/chat/completions` (text, `image_url` and `input_audio` parts, tools) |
| `embeddings` | `POST /v1/embeddings` |
| `transcribe` | `POST /v1/audio/transcriptions` (multipart upload) |
| `speech`, `speechStream` | `POST /v1/audio/speech` |
| `images` | `POST /v1/images/generations` |
| `convertDocument`, `convertDocumentStream` | `POST /v1/documents/convert` (`file` upload or `url`) |
| `documentPictures`, `documentPicture`, `documentPictureBytes`, `deleteDocument` | `/v1/documents/{id}/pictures[/{n}]`, `DELETE /v1/documents/{id}` |
| `entities` | `POST /v1/entities` |
| `models`, `enableModel`, `disableModel` | `/v1/models`, `/v1/models/{id}/enable\|disable` |
| `modelsCatalog` | `/v1/models/catalog` (optional `modality`) |
| `advise` | `POST /v1/advisor` (`{ needs }`, optional `vramBytes`, `ramBytes`, `unifiedMemory`) |
| `addModel`, `replaceModel`, `removeModel` | `POST /v1/models`, `PUT` / `DELETE /v1/models/{id}` (`{ catalog }` or `{ config }`, optional `drainSecs`; admin key) |
| `setMoeOffload` | `POST /v1/models/{id}/moe-offload` |
| `memoryStatus`, `setTier` | `/v1/memory/status`, `/v1/memory/tier` |
| `kvPark`, `setKvPark` | `GET`/`POST /v1/memory/kv-park` |
| `logs`, `log`, `clearLogs`, `setLogLevel` | `/v1/logs*` |
| `health`, `isReady`, `waitUntilReady`, `metrics` | `/health`, `/ready`, `/metrics` |

Request and response objects keep the API's field names (`max_tokens`, `hypnos_telemetry`,
`b64_json`), so the types match docs/api.md and responses can be passed on as they are. Fields
the server adds later come through untyped. `hypnos_telemetry` is present on every inference
response; speech returns `{ audio, contentType, telemetry, requestId }` with the `x-hypnos-*`
headers parsed into `telemetry`.

Uploads (`transcribe`, `convertDocument`) take a `Blob`/`File`, a `Uint8Array`/`ArrayBuffer`,
or, in Node, a file path.

## Streaming and cancellation

```ts
const stream = await hypnos.chatStream("gemma4:26b-q4", messages, { max_tokens: 400 });
for await (const piece of stream.text()) process.stdout.write(piece);
console.log(stream.usage, stream.telemetry);
```

Breaking out of the loop, `stream.close()`, or aborting the `signal` passed in the options
closes the connection, and the server stops at the next token. Every method accepts
`{ signal }`. `stream.final()` reads the rest and returns a normal completion; after the stream
was read, it returns what was collected. A stream that fails after it started throws
`StreamError`. Document streams yield `document.started`, `document.page` and `document`
events; `final()` returns the whole document with its pages put back.

## Tool calling

`tools` and `tool_choice` (`"auto"`, `"none"`, `"required"` or
`{ type: "function", function: { name } }`) go in the options of `chat` and `chatStream`
([docs/api.md section 7.5](../../docs/api.md)). The server never runs a tool: when the model
calls one, `finish_reason` is `tool_calls` and `message.tool_calls` lists the calls. Append
`assistantMessage(reply)` and one `toolMessage(call.id, result)` per call, then ask again with
the same `tools`.

```ts
import { Client, assistantMessage, parseArguments, toolMessage } from "@hypnos/client";
import type { ChatMessage, ToolSpec } from "@hypnos/client";

const tools: ToolSpec[] = [{
  type: "function",
  function: {
    name: "get_weather",
    description: "Current weather for a city",
    parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
  },
}];
const messages: ChatMessage[] = [{ role: "user", content: "What's the weather in Paris?" }];

let reply = await hypnos.chat("gemma4:26b-q4", messages, { tools });
while (reply.choices[0].finish_reason === "tool_calls") {
  messages.push(assistantMessage(reply));
  for (const call of reply.choices[0].message.tool_calls ?? []) {
    const { city } = parseArguments<{ city: string }>(call); // validate before running the tool
    messages.push(toolMessage(call.id, JSON.stringify({ city, temperature: 18 })));
  }
  reply = await hypnos.chat("gemma4:26b-q4", messages, { tools });
}
console.log(reply.choices[0].message.content);
```

A `ChatStream` accumulates the calls in `stream.toolCalls` (by `index`, arguments
concatenated, as OpenAI clients do), and `stream.final()` returns them in the message with
`finish_reason: "tool_calls"`. A `tool_choice` naming a function that is not in `tools` is a
`BadRequestError`.

## Errors and retries

Every error extends `HypnosError`; HTTP errors are `APIError`s with `status`, `type`, `code`,
`requestId` and `retryAfter` (seconds): `BadRequestError` (400, 413), `AuthenticationError`
(401), `PermissionDeniedError` (403), `NotFoundError` (404), `ModelBusyError` and `RateLimitError` (429),
`ServiceUnavailableError` (503), `InsufficientMemoryError` (507), `APITimeoutError` (408, 504),
`ServerError` (500), `StreamError` (a failed stream) and `APIConnectionError`.

The default `RetryPolicy` follows the API's recommendations: 429 `model_busy` from 250 ms
doubling, 429 `rate_limit_error` and 503 after `Retry-After`, 507 from 2 s doubling, 504 and
connection errors from 1 s, a 500 once; never 400, 401, 403 or 404, never a stream that already
produced output, and never the model pool changes (`addModel`, `replaceModel`, `removeModel`: a
lost connection may have applied the change, and their 429 means the drain already timed out).
`new Client({ retry: new RetryPolicy({ maxRetries: 8 }) })` or
`retry: NO_RETRY` changes it. `timeoutMs` (default 900 000) bounds the wait for the response
headers; a cold request pays a model load.

## Concurrency

`new Client({ maxConcurrencyPerModel: 3 })` keeps at most three requests per model in flight
from this client (a stream counts until it is closed). Beyond a model's execution slots the
server only queues requests, and beyond its queue it answers 429.

## Entity offsets

`entities` reports `start` and `end` as Unicode code point offsets. JavaScript strings count
UTF-16 units, so `text.slice(start, end)` is wrong after an emoji or another astral character.
Use `sliceByCodePoints(text, start, end)`.

## Browsers

The server allows loopback origins by default; the operator adds others with
`server.allowed_origins`. Its CORS allow-list has GET, POST and OPTIONS only, so
`deleteDocument` and `clearLogs` (DELETE) work from Node or a same-origin page, not across
origins. Browsers do not let scripts set `User-Agent`; the client sends it in Node only. Keep
API keys out of public pages.

## Examples

`examples/` has one script per model of the default lineup: `chat.mjs`, `vision.mjs`,
`transcribe.mjs`, `embeddings.mjs`, `speech.mjs`, `images.mjs`, `documents.mjs`,
`entities.mjs`. Build first (`npm run build`), then `node examples/chat.mjs "Hello"`.

## Development

```bash
npm install
npm test        # builds, then runs node --test against backend/…/examples/mock_server
```

The tests run against the real Hypnos router with mock engines (`cargo run -p hypnos-server
--example mock_server`), so they check the actual status codes, headers and stream framing.
Set `HYPNOS_MOCK_SERVER` to a built binary to skip the cargo build.
