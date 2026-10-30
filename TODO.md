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
| 2  | 2   | 3   | 2   | 6    | 1   | 4     | 4   | 1    |

---

## Backlog

- [ ] ⬛ **BUILD-1** `P1` Publish prebuilt release binaries
      The 10-20 min native build plus vendoring, CMake/Ninja and the CUDA host-compiler pin stop
      most people. Release workflow producing archives for Windows x64 (CUDA), Linux x64 (CUDA)
      and macOS arm64 (Metal), each with the sd.cpp shared library, `espeak-ng-data`, the
      example config and the built frontend. Document the CUDA runtime / driver requirement.
      Archives are GPL-3.0-or-later as a whole (espeak-ng, see `THIRD-PARTY-NOTICES.md`): ship
      `LICENSE-*`, the GPL-3.0 text and the notices, and link the tagged source.
- [ ] ⬛ **BUILD-2** `P2` Provide a Docker image with CUDA support
      Multi-stage Dockerfile (build on a CUDA devel image, run on a CUDA runtime image), models
      mounted as a volume, config via mount or env. Publish to GHCR from the release workflow
      (BUILD-1) and add a `docker run --gpus all ...` quickstart to the README. Same GPL
      obligations as BUILD-1 for the image.
- [ ] ⬛ **BUILD-3** `P2` Re-enable CI
      `.github/workflows/ci.yml` is fully commented out. Restore the engine-free jobs: CPU-only
      `cargo build --workspace --no-default-features`, `cargo test --workspace`, scoped `cargo fmt
      --check` and clippy, plus `npm run build` for the frontend. Add a macOS Metal build job if
      runner time allows.
- [ ] 🟩 **ENG-1** `P2` Add Docling document conversion (PDF / images to Markdown)
      Serve Docling-style conversion in-process, without the Python library: run
      `granite-docling-258M` (GGUF + mmproj) through the existing llama.cpp + mtmd path on page
      images, then turn its DocTags output into Markdown / JSON. Needs a new modality
      (`document_conversion`), a Hypnos extension endpoint (e.g. `POST /v1/documents/convert`,
      multipart file or URL, `output_format`), PDF page rasterization (pdfium is BSD; avoid
      AGPL MuPDF) and per-page streaming. Check the pinned llama.cpp supports the model; add it
      to `models.manifest`, the example configs and the default profile.
- [ ] 🟩 **ENG-2** `P2` Add GLiNER zero-shot named-entity extraction
      Run GLiNER ONNX exports on CPU through ONNX Runtime, like Kokoro (`onnx_runtime` engine,
      `realtime_pinned`): tokenizer via the `tokenizers` crate, span decoding and thresholding
      (`gline-rs`, Apache-2.0, is a reference or a dependency). New modality
      (`entity_extraction`) and endpoint (e.g. `POST /v1/entities` with `model`, `text` or a
      list, `labels`, `threshold`) returning spans with label, offsets and score. Add a
      playground tab, the model to `models.manifest` and the example configs.

## In Progress

_No cards yet._

## Done

- [x] ⬜ **DOC-3** `P1` Record a demo GIF of the 3-tier visualizer for the README
      30-60 s screen recording of the playground while chat, speech and image requests move
      models between tiers (evictions and promotions visible). Keep it under ~5 MB, save under
      `assets/`, and replace the `DOC-3` placeholder comment at the top of `README.md`.
      Shipped: `assets/demo.gif` (46 s, 0.76 MB), recorded headless with Playwright from timed
      screenshots against a demo server on an M5 Max with `simulate_physical_memory_bytes` = 32 GB,
      so the 26B chat model and MOSS-Image evict each other. README placeholder replaced.
- [x] 🟫 **MEM-1** `P2` Read `[models.arch]` defaults from GGUF metadata
      Custom models must copy `num_layers`, `num_kv_heads` and `head_dim` from the GGUF by hand
      (README "Bring your own models"); the defaults (32 / 8 / 128) misplan anything else. Fill
      unset fields from `<arch>.block_count`, `.attention.head_count_kv`, `.attention.key_length`,
      `.context_length` and the sliding-window keys, keep explicit config values, and log what
      was detected. `gguf_inspector.rs` already parses the header.
      Shipped: std-only reader `hypnos-core/src/gguf.rs` (per-layer arrays booked at their max,
      bool `sliding_window_pattern` turned into `global_attn_every_n_layers`), `from_toml_str`
      records the arch keys the file wrote, `HypnosConfig::detect_arch_from_gguf` fills the rest
      before `validate` and logs applied / kept values. Code-built configs (default profile) are
      left alone. Detected values match the hand-tuned Gemma 4 entries.
- [x] 🟧 **PERF-5** `P1` Benchmark model swap latency: cold vs warm vs hot
      The 3-tier hierarchy is the headline claim but no bench measures it (baseline shows 9
      promotions averaging ~4.8 s each). Add a `bench.sh` scenario that times the first request
      to a large model (`gemma4:26b-q4`) starting from Tier 3 (NVMe mmap), Tier 2 (pinned RAM)
      and Tier 1 (resident), using `POST /v1/memory/tier` to stage it. Report tier-switch ms and
      TTFT per start tier on CUDA and Metal; save under `docs/bench/` and chart it for the README.
      Shipped: `bench.py` scenario 7 (`--scenarios 7`, `--swap-model`, `--swap-reps`,
      `--swap-files`, `--chart`), per-file page-cache eviction without root (`posix_fadvise` /
      `msync(MS_INVALIDATE)`) checked with `mincore`, SVG chart. Metal run in
      `docs/bench/swap-metal-2026-10-29.md` (Tier 3 2.29 s, Tier 2 0.99 s, Tier 1 84 ms TTFT),
      chart in the README. CUDA run still to do on the Windows box (no page-cache drop there).
- [x] ⬜ **DOC-2** `P1` Rewrite README around the pitch, a quickstart and bring-your-own models
      Lead with the one-line pitch (one OpenAI endpoint for every modality on one GPU, with a
      scheduler that manages VRAM), then a short demo GIF/video of the 3-tier visualizer, then a
      quickstart (binary or Docker, one model, one request). Show a minimal `[[models]]` config
      for an arbitrary GGUF (verify `HypnosConfig` accepts it without the default lineup) and
      move the default lineup, build-from-source and tuning details further down. Add a short
      comparison with Ollama / LocalAI / llama-swap and the PERF-5 swap chart.
      Shipped: pitch, measured results and comparison up front, then quickstart (from source until
      BUILD-1/2) and a validated single-GGUF config; details moved below. `HYPNOS_CONFIG` selects
      the backend config in `run.sh` / `run.bat`; an omitted `[hardware]` now auto-detects VRAM
      (it used to book a fixed 24 GB). Demo GIF split into DOC-3, swap chart waits on PERF-5.
- [x] ⬜ **DOC-1** `P1` Add a LICENSE file to the repository
      Without one nobody may legally use or contribute to the code. Pick a license (MIT and/or
      Apache-2.0 are the Rust norm), check compatibility with the vendored engines (llama.cpp,
      sd.cpp, ONNX Runtime, espeak-ng is GPL-3.0) and the model licenses, add `LICENSE`, set
      `license` in the workspace `Cargo.toml` and mention it in the README.
      Shipped: source stays `MIT OR Apache-2.0` (already the workspace `license`), `LICENSE-MIT`,
      `LICENSE-APACHE`, `THIRD-PARTY-NOTICES.md` (engines, crates, models; binaries are
      GPL-3.0-or-later as a whole because espeak-ng is linked statically), README "License"
      section, `license` in `frontend/package.json`.
- [x] 🟪 **API-1** `P1` HTTPS and API-key authentication for the HTTP server
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
      Shipped: `auth.rs` middleware (SHA-256 digests, constant-time match, 401 `invalid_api_key`
      with `WWW-Authenticate`), `server.api_keys` / `HYPNOS_API_KEYS` / `metrics_require_auth`,
      `[server.tls]` served by axum-server + rustls (ring), validation errors and exposure warnings,
      playground API-key field (`apiFetch` on all 11 calls). Verified over HTTPS with streamed TTS
      and multipart STT; the key never appears in logs. Commit `8a5658c`.
- [x] 🟦 **UI-1** `P1` Add request log inspector panel with per-call telemetry
      Backend: keep a bounded in-memory ring of per-request records (request id, route, model id,
      status, start/end time, processing ms, tokens in / out / cached, tokens/s, tier-switch ms,
      and the request + response payloads, truncated for media). Expose it via a new
      `GET /v1/logs` (list, filterable) and `GET /v1/logs/{id}` (full input/output), honouring a
      configurable log level (`error` / `warn` / `info` / `debug` / `trace`): `info` records
      metadata only, `debug` adds prompt + completion text, `trace` adds raw payloads.
      Frontend: a "Logs" panel in `App.svelte` listing calls newest-first with columns for model,
      route, duration, tokens in/out/cached and tokens/s, a level selector, filter by model/route,
      and a detail view showing the full input and output of the selected call.
      Shipped: `request_log.rs` ring with capture levels, `GET /v1/logs`, `GET /v1/logs/{id}`,
      `POST /v1/logs/level`, `DELETE /v1/logs`; "6. Request Logs" tab in the playground.
      Commit `b79886c`.
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
