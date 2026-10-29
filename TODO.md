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
| 2  | 1   | 1   | 1   | 3    | 1   | 1     | 1   | 1    |

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

## In Progress

- [ ] 🟧 **PERF-2** `P1` Engine worker: persistent llama.cpp context, slots, prefix reuse
      Work item B of `docs/execution-efficiency-plan.md`: one long-lived worker thread per resident
      llama.cpp model owning a single context (`n_seq_max = slots`), requests served through slots
      with longest-common-prefix KV reuse (`kv_cache_seq_rm` of the divergent tail), multimodal
      chunks matched by id so an unchanged image is never re-encoded. Config: `slots` (default 1),
      `prefix_cache` (default true). Done when bench scenarios 2 and 3 show only the new turn
      prefilled and one image encode across three turns.

## Done

- [x] 🟧 **PERF-1** `P1` Execution telemetry, per-model `/metrics` histograms, bench script and baseline
      Work item A of `docs/execution-efficiency-plan.md`. Commit `e209a56`.
