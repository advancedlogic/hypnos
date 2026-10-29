# Hypnos TODO Board

Each card is one feature, bug, or chore. Cards are referenced by their ID (e.g. `BUG-3`, `UI-2`).
IDs are never reused: when a card is done it moves to **Done** and keeps its ID.

## Conventions

### Labels (ID prefix = area)

| Prefix  | Color     | Badge | Area                                                      |
|---------|-----------|-------|-----------------------------------------------------------|
| `UI-n`  | 🟦 Blue    | `🟦`  | Frontend / Svelte playground (`frontend/`)                |
| `API-n` | 🟪 Purple  | `🟪`  | HTTP / OpenAI-compatible API surface (`hypnos-server`)    |
| `ENG-n` | 🟩 Green   | `🟩`  | Inference engines: llama.cpp, sd.cpp, ONNX, Kokoro (`hypnos-backends`) |
| `MEM-n` | 🟫 Brown   | `🟫`  | 3-tier memory, scheduler, KV cache (`hypnos-memory`, `hypnos-scheduler`, `hypnos-core`) |
| `PERF-n`| 🟧 Orange  | `🟧`  | Performance, benchmarks, throughput, latency              |
| `BUG-n` | 🟥 Red     | `🟥`  | Defects (any area; cross-reference the area in the body)  |
| `BUILD-n`| ⬛ Black  | `⬛`  | Build system, vendor patches, CMake, CI, run scripts      |
| `DOC-n` | ⬜ White   | `⬜`  | Documentation, README, guides                             |
| `TEST-n`| 🟨 Yellow  | `🟨`  | Tests, fixtures, test infrastructure                      |

### Priority

| Tag  | Meaning                                          |
|------|--------------------------------------------------|
| `P0` | Blocking / crash / data loss. Fix before anything else. |
| `P1` | Important, next up.                              |
| `P2` | Normal.                                          |
| `P3` | Nice to have / someday.                          |

### Status columns

- **Backlog**: accepted, not started.
- **In Progress**: being worked on now.
- **Done**: completed; include the commit hash or PR when available.

### Card format

```
- [ ] 🟦 **UI-1** `P2` Short imperative title
      Optional one or two lines of detail, acceptance criteria, or file pointers.
```

- Checkbox `[ ]` open, `[x]` done.
- Title is imperative and under ~70 chars.
- Body is indented 6 spaces under the title.
- When done, move the card to **Done**, tick the box, and append `(commit abc1234)`.

### Next IDs

| UI | API | ENG | MEM | PERF | BUG | BUILD | DOC | TEST |
|----|-----|-----|-----|------|-----|-------|-----|------|
| 2  | 2   | 1   | 1   | 5    | 1   | 1     | 1   | 1    |

---

## Backlog

- [ ] 🟦 **UI-1** `P1` Add request log inspector panel with per-call telemetry
      Backend: keep a bounded in-memory ring of per-request records (request id, route, model id,
      status, start/end time, processing ms, tokens in / out / cached, tokens/s, tier-switch ms,
      and the request + response payloads, truncated for media). Expose it via a new
      `GET /v1/logs` (list, filterable) and `GET /v1/logs/{id}` (full input/output), honouring a
      configurable log level (`error` / `warn` / `info` / `debug` / `trace`): `info` records
      metadata only, `debug` adds prompt + completion text, `trace` adds raw payloads.
      Frontend: a "Logs" panel in `App.svelte` listing calls newest-first with columns for model,
      route, duration, tokens in/out/cached and tokens/s, a level selector, filter by model/route,
      and a detail view showing the full input and output of the selected call.
      Related: `ExecutionTelemetry`, `metrics.rs`, `x-request-id` header.
- [ ] 🟪 **API-1** `P1` HTTPS and API-key authentication for the HTTP server
      Today the server binds loopback only and has no authentication (the README tells people to
      put a proxy in front). Add both natively so `server.host = "0.0.0.0"` is safe on its own:
      TLS termination in-process (rustls through `axum-server` or `tokio-rustls`) configured by
      `server.tls.cert_path` / `server.tls.key_path` (PEM; plain HTTP when unset), and bearer
      authentication for every `/v1/*` route from a `server.api_keys` list (env override
      `HYPNOS_API_KEYS`; `/health`, `/ready` and `/metrics` stay open or get their own flag).
      Compare keys in constant time, answer 401 in the OpenAI error shape, log rejections without
      the key. CORS already allows `Authorization`; the playground needs an API-key field stored
      per browser and sent on every request (including streamed and multipart ones).
      Related: `cors_layer`, `observe_request`, `ServerConfig`, `validate` warning on non-loopback host.

## In Progress

_No cards yet._

## Done

- [x] 🟧 **PERF-4** `P1` Continuous batching across execution slots
      Work item C of `docs/execution-efficiency-plan.md`: the engine worker admits jobs into free
      slots and builds one `llama_batch` per step (chunked prefill for new slots, one token per
      decoding slot), decodes once, samples per slot by batch index, streams per slot with
      back-pressure that never stalls the other slots. Config knob stays `slots`. Measured: four clients
      on gemma4:26b-q4 reach 2.18x single-client aggregate decode tokens/s (wall 4.9 s -> 2.1 s),
      single-client TTFT unchanged; the dense E4B only 1.10x, see the plan
      (`docs/bench/after-C-2026-10-29.md`). Commit `4f12845`.
- [x] 🟧 **PERF-3** `P1` Streaming text-to-speech
      Work item D of `docs/execution-efficiency-plan.md`: `synthesize_speech_stream` on the backend
      trait, Kokoro emits each chunk (and the first clause early) as ONNX returns it,
      `POST /v1/audio/speech` with `"stream": true` streams raw PCM or a live WAV (open-ended
      header). `hypnos_tts_ttfa_ms` records time to first audio. Measured: time to first audio 5.4 s -> 385 ms, total synthesis unchanged
      (`docs/bench/after-D-2026-10-29.md`). Commit `945b333`.
- [x] 🟧 **PERF-2** `P1` Engine worker: persistent llama.cpp context, slots, prefix reuse
      Work item B of `docs/execution-efficiency-plan.md`: one worker thread per resident llama.cpp
      model owning a single context (`n_seq_max = slots`), requests served through slots with
      longest-common-prefix KV reuse plus sliding-window checkpoints, multimodal chunks matched by
      id so an unchanged image is never re-encoded. Config: `slots` (default 1), `prefix_cache`
      (default true). Five-turn chat TTFT 300 ms -> 145 ms, vision follow-ups 300 ms -> 105 ms
      (`docs/bench/after-B-2026-10-29.md`). Commit `442c206`.
- [x] 🟧 **PERF-1** `P1` Execution telemetry, per-model `/metrics` histograms, bench script and baseline
      Work item A of `docs/execution-efficiency-plan.md`. Commit `e209a56`.
