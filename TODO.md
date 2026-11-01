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
| `DESK-n`| 🟦 Cyan    | `🩵`  | Tauri desktop app (`desktop/`)                            |

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

| UI | API | ENG | MEM | PERF | BUG | BUILD | DOC | TEST | DESK |
|----|-----|-----|-----|------|-----|-------|-----|------|------|
| 3  | 13  | 5   | 8   | 11   | 8   | 11    | 7   | 6    | 20   |

---

## Backlog

- [ ] 🟨 **TEST-5** `P2` Starvation under sustained load; real-engine drills and reproducible releases
      From TEST-4. (1) The stress test showed that when two resident models stay busy, a third
      model's requests all get 507 `insufficient_vram_or_ram` (neither can be evicted while
      leased): it starves. Let such a request wait (bounded by its deadline, PERF-10) for a
      model to become idle and be evicted, ahead of new work for the resident models, instead
      of refusing at once. (2) On a CUDA machine: an engine abort and supervisor restart drill
      (systemd / Docker), out-of-memory recovery under real pressure, and `hypnos measure-memory`
      to calibrate the CUDA working-memory values (TEST-3 measured Metal only). (3) Reproducible
      release archives: pinned toolchains in every CI job, a build that gives the same archive
      twice, checksums in the release.

- [ ] 🩵 **DESK-19** `P1` Test on real 16 GB and 32 GB Macs and ship a signed, notarized release
      Split from DESK-18 (it needs hardware and secrets): run the guided setup end to end on a
      16 GB and a 32 GB Mac (the advised lineup, the downloads, the demo, swaps under memory
      pressure, `hypnos measure-memory` there), not only with `simulate_physical_memory_bytes`,
      and record the results in `docs/apple-silicon.md`; add the `APPLE_*` and
      `WINDOWS_CERTIFICATE` repository secrets (docs/signing.md) and make a release with them so
      the installers are Developer ID signed and notarized.

- [ ] 🟩 **ENG-4** `P2` Add audio.cpp for text-to-speech and optional speech-to-text
      Vendor https://github.com/0xShug0/audio.cpp (pin + patches in `scripts/vendor.sh`) and link
      it in-process as a new engine for `POST /v1/audio/speech`, next to Kokoro. Optionally offer
      it for `POST /v1/audio/transcriptions` as an alternative to the llama.cpp/mtmd path,
      selected per model in the config. Check its ggml does not clash with llama.cpp's (build
      shared like sd.cpp if needed), then add models to `models.manifest` and the example configs.
      Decided 2026-10-30: behind an `audio-cpp` Cargo feature, off by default (not in `default`,
      the run scripts or the release archives at first; a build without it refuses a config that
      uses it, CI compiles it on one Linux job). Step 1 is a one-day spike before vendoring: build
      it standalone with CUDA, run one TTS and one STT model, measure quality and speed against
      Kokoro and Gemma 4 E4B transcription, check the license, the API stability and whether its
      ggml builds shared cleanly; close the card with the numbers if it does not clearly win.
      After TEST-1 and TEST-2.

- [ ] 🟪 **API-8** `P2` Publish the SDKs to PyPI, npm and crates.io
      `sdk/python` (`hypnos-client`), `sdk/javascript` (`@hypnos/client`) and `sdk/rust`
      (`hypnos-client`) are ready to publish but never were: claim the names (the npm scope
      `@hypnos` included), add the tokens as repository secrets (`PYPI_TOKEN` or trusted
      publishing, `NPM_TOKEN`, `CARGO_REGISTRY_TOKEN`) and a publish job in release.yml after
      `ci`, versioned with the tag. Then point the README's quick start at the packages.

- [ ] 🟧 **PERF-9** `P2` Three concurrent streams on the 26B decode at 34 tokens/s each on CUDA
      Scenario 4 of `docs/bench/cuda-2026-10-31.md` (RTX 5090 laptop, `slots = 3`, 16k, f16 KV):
      one client decodes at 152-160 tokens/s; three batched clients got 34 each in one run
      (aggregate 105, 0.69x of one, the fourth waiting 4 s for a slot) and 92-99 each in the next
      (223, 1.39x), so the batched shape is also unstable between runs. The old
      `slots = 4` q8_0 default gave 2.18x in the same scenario (`after-C-2026-10-29.md`, 68 per
      client). PERF-7 only measured one stream per slot shape; the batched multi-stream shape
      probably makes llama.cpp b11982 re-capture its CUDA graph every step again. Check with the
      graph counters / `GGML_CUDA_DISABLE_GRAPHS`, try 2 slots and 4 x 8k, and ship the shape that
      batches well (on Metal 4 slots gave 1.43x).
- [ ] 🟩 **ENG-6** `P1` Qwen chat format: Qwen 3.5 / 3.6 / 3.8 and the models built on them
      One renderer (`qwen_prompt`: ChatML, `<think>` thinking with the budget/close logic of
      `lfm3_prompt`, Qwen's `<tool_call>` JSON tool format in `tools.rs`, vision markers) and
      `qwen35`, `qwen35moe`, `qwen3vl` in `TESTED_GGUF_ARCHITECTURES`. Catalog: Qwen3.8-27B (19 GB
      Q4_K_M, vision, 256k, Apache-2.0), Qwen3.6-35B-A3B (20 GB, MoE: `moe_offload`), Qwen3.5
      0.8B / 2B / 4B / 9B, MiniCPM-V 4.6 (0.5 GB vision), Qwen3-ASR 0.6B / 1.7B (transcription,
      `qwen3vl`). Check Clef / Clef-Flash (Cloudflare, own `clef` arch) against the same renderer.
- [ ] 🟩 **ENG-7** `P1` Image models on the existing sd.cpp file set; per-model sampling defaults
      `diffusion.rs` uses one `DEFAULT_STEPS` / CFG for every model; turbo models want ~4-8 steps
      and CFG 1. Add `[models.arch]`-style `default_steps` / `default_cfg_scale` (sample method,
      flow shift if needed) and generalize the module docs beyond MOSS. Catalog, all DiT + LLM
      encoder + VAE: Z-Image-Turbo (3.9 GB Q4_K + Qwen3 4B, Apache-2.0), FLUX.2 klein 4B
      (Apache-2.0; the 9B is non-commercial), ERNIE-Image-Turbo (5 GB + Ministral 3 3B,
      Apache-2.0), FLUX.3 Image Klein (3.5 GB + Qwen3-VL 4B, FLUX.2 VAE; FLUX Kommunity license:
      check the terms first). Memory measured per model (TEST-3).
- [ ] 🟩 **ENG-8** `P2` Speech output through llama.cpp (LFM3-Audio, Qwen3-TTS)
      b11982's libmtmd generates audio (`mtmd-helper-gen.cpp`, `tools/tts`): an engine path for
      `text_to_speech` on `ggml_llama` models, WAV / streamed PCM like Kokoro. First LFM3-Audio
      (already in the catalog for transcription: 19 languages, `female` / `male`, 80 ms frames,
      model-card sampling), so one model hears, chats and speaks; then Qwen3-TTS 1.7B
      (`ggml-org/Qwen3-TTS-12Hz-1.7B-Base-GGUF`, 1 GB, 10 languages, voice cloning from a
      reference clip: a `voice` upload or path; check its license). Pocket TTS (CC-BY-4.0, gated)
      and Soprano 80M (Apache-2.0, English) need our own GGUF conversion. Kokoro stays the CPU
      default.
- [ ] 🟩 **ENG-9** `P3` More chat formats: Mistral, Nemotron, GLM, LFM2, Qwen 3
      One renderer each, only where a model earns it: Voxtral Mini 4B Realtime (`mistral3`,
      streaming transcription, Apache-2.0), Nemotron 3.5 Lightning 30B-A3B (`nemotron_h_moe`, 1M
      context, NVIDIA license), GLM-4.7-Flash 30B (MIT, tool use), Qwen3-Coder-30B-A3B
      (`qwen3moe`), LFM2.5-24B-A2B (`lfm2moe`).
- [ ] 🟩 **ENG-10** `P3` Image engine: optional VAE and other encoders; Parakeet transcription
      Iris-3B works in pixel space (no VAE; Apache-2.0), Krea 2 uses the Wan 2.1 VAE, LLaDA-Image
      a 9.7 GB diffusion-LLM encoder: make `aux_weights_path` optional and check each pipeline.
      NVIDIA Parakeet TDT 0.6B v3 (25 languages, fast) runs through libmtmd but is not a chat
      model: a transcription-only path.
- [ ] 🟪 **API-13** `P3` `POST /v1/images/edits` (OpenAI images edit)
      Image + prompt (+ mask) in, image out, for the sd.cpp models that edit: Qwen-Image 2.1
      (Qwen3-VL 8B encoder; Qwen Research license, likely non-commercial), Fibo 2.0 Edit Turbo,
      FLUX.2 / FLUX.3 reference images. Then `docs/api.md`, the mock server, the three SDKs, the
      playground and the desktop app's image tool.

## In Progress

_No cards yet._

## Done

- [x] 🟩 **ENG-5** `P1` Catalog: the other Gemma 4 sizes, Nemotron 3.5 Bolt E4B, LFM3-Omni 2B
      Architectures Hypnos already runs (`gemma4`, `lfm3`), so catalog entries + `[models.estimate]`
      + a smoke run each: `ggml-org/gemma-4-E2B-it-GGUF` (2.8 GB Q4_0, vision + audio in),
      `gemma-4-12B-it` (7.2 GB, vision), `gemma-4-31B-it` (18 GB, vision),
      `NVIDIA-Nemotron-3.5-Bolt-E4B-GGUF` (gemma4 arch, text, 8 GB Q8; check its template matches
      `gemma4_prompt`), `LFM3-Omni-2B-GGUF` (1.1 GB, text + vision + audio in, LFM 1.0 license).
      Measure `activation_workspace_bytes` with `hypnos measure-memory` (TEST-3).
      Done: entries for all five, each tested on the M5 Max (chat, thinking, vision, tools,
      transcription where it hears) and measured (`docs/bench/memory-2026-11-01.md`). The 12B /
      26B / 31B templates close an empty thought block when thinking is off and Bolt's puts text
      before tool calls: `Gemma4Template`, read from the GGUF template, renders both (this also
      corrects the lineup's 26B with thinking off). LFM3-Omni is non-reasoning (no `thinking`
      capability). The advisor now scores parameters (bytes / bits per weight) instead of file
      size, with transcription at half weight: it picks the 31B on large Macs, which is slower
      than the 26B MoE (speed is not scored).
- [x] ⬛ **BUILD-11** `P1` Cut the release workflow from ~3 h to under 1 h
      v0.4.0's release run took 3h09m: Windows CUDA 2h44m, Linux CUDA 1h42m, Docker 2h31m (it
      compiles the CUDA server again and only starts after CI); the CI checks finish in ~25 min.
      (1) `build-server.yml`: an explicit `CMAKE_CUDA_ARCHITECTURES` (Turing..Blackwell real +
      one PTX) instead of ggml's all-architectures default; (2) the Docker image copies the
      `linux-x64-cuda` archive instead of building; (3) `sccache` / `ccache` for the CMake C++
      and CUDA objects; (4) `timeout-minutes` on every CI job (~60) and apt steps (~10): v0.5.0's
      `desktop app (ubuntu-22.04)` hung 1h45m+ in `apt-get`. Document the supported GPUs.
      Done: `CUDA_ARCHITECTURES` 75-virtual;80-virtual;86-real;89-real;120a-real (the
      diffusion-rs patch forwards `CMAKE_CUDA_ARCHITECTURES`, `setup-build-env` writes it for the
      local GPU); `build-linux` / `build-windows` calls, Docker `SERVER=archive` after
      `build-linux`; sccache for the CMake builds; job and apt-step timeouts. Drops Maxwell,
      Pascal and Volta. Timings to confirm on the next release run.
- [x] ⬛ **BUILD-10** `P2` `run.sh build all` / `run.bat build all` builds backend, playground and desktop
      Today `build` (target `all` by default) compiles the backend and installs the playground's
      packages, and the desktop app is only built when named. An explicit `build all` builds
      everything in dependency order: the backend, the playground (`npm install` + `npm run
      build`, so `frontend/dist`, which the server serves at `/` and the macOS app bundles, is
      fresh), then the desktop app (which bundles that server; skipped when the backend failed).
      Plain `build` and the other actions keep `all` = backend + playground.
      Done 2026-11-01: `build_playground` (dependencies + `npm run build`) and the ordered
      `build all` in run.sh and run.bat (run.bat not run: no Windows here); usage, README,
      desktop/README.md and CLAUDE.md updated. (commit 91c6f81)

- [x] 🟥 **BUG-7** `P1` Tools for an LFM3 model fail mid-stream instead of a 400 (API, desktop)
      A streamed chat with `tools` for `lfm3:1.7b` answered 200 and then an error event ("has no
      tool-calling format in Hypnos"), because the engine checked the architecture after the SSE
      headers; the desktop's prompt-convention fallback waits for a 400 and never ran. Fixed:
      `ModelBackend::supports_tool_calls` (GGUF architecture, read once) is checked in the chat
      handler before loading or streaming. Test:
      `tools_for_a_model_without_a_tool_format_are_refused_before_streaming`. (commit cddd1a9)

- [x] 🟨 **TEST-4** `P2` Harden for operational reliability
      Build on what exists (auth, quotas, TLS, metrics, request log, the TEST-2 fault tests):
      a stress suite (many concurrent clients across modalities, long runs, memory never
      growing), concurrent cancellation at every stage (queued, loading, prefill, decode,
      streaming), out-of-memory recovery under real pressure, engine crash and supervisor
      restart drills, upgrade compatibility (old configs and overlays, `config_version`
      migrations, catalog versions), a security review (auth bypass, path handling in
      `POST /v1/models`, remote URL fetches, request log redaction) and repeatable releases
      (pinned toolchains, reproducible archives).
      Done 2026-11-01 (engine-free and review parts): a stress / cancellation test over the real
      router and scheduler (300 clients, three models in room for two, cancellations at every
      stage), upgrade-compatibility tests over the released example configs, and a security
      review with three fixes (SSRF guard on remote URLs, pool paths bound without models_dir, a
      warning for an exposed server without keys). Crash drills, out-of-memory under real
      pressure and reproducible release archives need a CUDA machine and the CI runners: moved to
      TEST-5 with the starvation the stress test found. (commit 157ba99)

- [x] 🟧 **PERF-10** `P2` Deadline-aware admission and predictable latency
      A request that succeeds in 12 s when 1 s was expected is a failure in practice. Estimate the
      wait at admission (queue depth x recent service time, plus a promotion if the model is cold)
      and expose it (`x-hypnos-estimated-wait-ms`, `/v1/memory/status`); accept an optional
      deadline (`x-hypnos-deadline-ms` or a body field) and refuse at once with 503 + `Retry-After`
      when it cannot be met instead of queueing. Overload policies per profile (shed or queue);
      degrading an answer (smaller `max_tokens`, no thinking) only when the request opts in, never
      silently. Track p50 / p95 / p99 for whole workflows
      (request id chains, the voice turn end to end), not only per engine, in `/metrics` and in
      bench.sh reports. Prefer predictable latency over peak tokens per second.
      Done 2026-11-01: wait estimate per model (service time x requests ahead + load),
      `x-hypnos-deadline-ms` -> 503 `deadline_unmeetable` + Retry-After, `limits.overload = "shed"`
      with the profile's objective, `x-hypnos-estimated-wait-ms`, end-to-end histograms per route
      and per `x-hypnos-workflow` with p50/p95/p99 in `/v1/memory/status`. Not done: opt-in
      degrading (does not shorten the queue), percentiles in bench.sh reports, SDK helpers for
      the headers (clients set them directly). (commit ae6f1ef)

- [x] 🟫 **MEM-6** `P2` Make scheduling workload-aware, not only memory-aware
      Residency and eviction today follow priority, recency and fit. Add what a decision costs
      and what comes next: per model an estimated promotion cost (from measured
      `tier_switch_phases`, by tier), the cost of restoring its cache (KV parking, MEM-4),
      expected request rate (decayed counters per model) and latency sensitivity (from the
      profile, MEM-7). Evict the model whose return is cheapest and least likely, not just the
      lowest priority. Sequences (speech-to-text -> chat -> speech, the voice turn; a document
      conversion followed by chat) are declared, not guessed: a client or the profile names the
      next stage and the scheduler pre-stages its model (Tier 2, or Tier 1 when it fits) while the
      current one runs (the pipeline lease of `docs/execution-efficiency-plan.md`, item E).
      Learning sequences from the request log is left for later: it is hard to test and makes
      the scheduler's choices hard to predict.
      Acceptance: scheduler tests with mock costs; a voice-turn benchmark (bench.sh) shows the
      second stage no longer paying a cold promotion.
      Done 2026-11-01: eviction within a priority by reload time x decayed request rate (measured
      reload times per model, shown in `/v1/memory/status`), `x-hypnos-next-model` pre-staging the
      declared next stage (voice turn on Apple Silicon: 157 ms cold load -> 0). KV-restore cost
      not weighed separately (parking restores in the load itself); latency sensitivity is the
      priority and the profile; no bench.sh scenario yet. (commit 636d670)

- [x] 🟫 **MEM-7** `P2` Add workload profiles with latency objectives
      Presets that fill in config values, not a new layer of behaviour. Named profiles that turn
      the knobs into outcomes: `interactive_chat`, `voice_assistant`,
      `document_pipeline`, `batch_processing` (`[server] profile = ...`, overridable per model).
      Each sets model priorities and pre-staging, queue behaviour (depth, fairness), slot counts,
      which swaps are acceptable (a voice assistant never evicts its STT/TTS models; batch accepts
      swaps for throughput) and a latency objective (first token / first audio / pages per
      minute) that MEM-6 and PERF-10 read. Explicit keys in `[[models]]` still win; `hypnos
      --check-config` prints the effective settings. Document each profile in
      `docs/operating.md` with the symptom it answers.
      Done 2026-11-01: `[server] profile` presets applied to the raw TOML (only missing keys:
      priority, initial_tier, keep_warm_in_ram, slots, default_context_len, max_queue_per_model),
      per role and merged for multi-role models; runtime and overlay models included; notes in
      the log and `--check-config`; profile and objective in `/v1/memory/status`; presets table
      in `docs/operating.md`. Per-model profile overrides are the explicit keys themselves.
      (commit 03da363)

- [x] 🟨 **TEST-3** `P1` Measure booked vs real memory per engine and fix the estimates
      Memory budgets are planner estimates, not OS-enforced limits, and some are off: LFM3's KV
      booking is about twice its use (hybrid layers booked as full attention), and the docling
      files on disk drifted from the release. Measure booked vs real (device and process
      footprint) per engine and per model under load: a `bench.sh` scenario and a `--measure`
      mode that loads each configured model alone and records the peak. Report the drift in
      `/v1/memory/status` and `/metrics` (`tier1_vram_unaccounted_bytes`, process footprint),
      alert on it, and correct the estimators (per-layer KV for hybrid / SWA layouts from the
      GGUF, workspace per engine) where they are wrong; record the numbers in `docs/bench/`.
      Done 2026-11-01: `hypnos measure-memory` (one model at a time, one request of its kind,
      booked vs real peak), the exact KV layout from the GGUF, workspace values from the measured
      peaks, Kokoro on the CPU, `hypnos_memory_bytes` on /metrics; before / after in
      `docs/bench/memory-2026-11-01.md` (all GPU models now 0.90-0.99 of their booking on Apple
      Silicon). Not done: a CUDA run, long-context runs. (commit c61ad92)

- [x] 🩵 **DESK-18** `P1` Guided setup: detect, recommend, install, demonstrate
      First run of the desktop app (and the playground) should need no knowledge of pinned
      memory, quantization or KV caches: detect the hardware, recommend a lineup (API-12),
      download it (the Models dialog's downloader), add it live (API-11), then run a short
      multimodal demo (ask about an image, speak the answer, transcribe it back) to prove it
      works. Finish distribution: the Developer ID / notarization wiring of BUILD-9 still needs
      the secrets and a release made with them. Test on real 16 GB and 32 GB Macs (startup,
      the recommended lineup, swaps under memory pressure), not only with
      `simulate_physical_memory_bytes`, and record the results in `docs/apple-silicon.md`.
      Done 2026-11-01: a three-step setup wizard in the desktop app (choose what to do, see the
      recommended lineup in plain words with the download size and free space, install: download,
      then rewrite the app's config with a backup or add the models live to a running server,
      then a demo: a chat answer, the logo described, a spoken answer transcribed back), opened on
      first run and from the Models dialog and the local server panel; an Advisor panel in the
      playground's catalog tab; the catalog route names the configured variant
      (`configured_quant`); the advisor sizes CPU models against free RAM (a 16 GB Mac asking
      for chat, vision, voice and documents was refused). Not run end to end with a real
      download yet. Real-Mac tests and signed releases moved to DESK-19. (commits 47a2594, f023b2b)

- [x] 🟪 **API-12** `P1` Hardware and model advisor
      "I have 24 GB of VRAM and 64 GB of RAM and need chat, voice and documents": recommend a
      lineup from the catalog (API-10), with variants, context sizes, slots, memory policy
      (`keep_warm_in_ram`, priorities, `moe_offload`) and the profile (MEM-7), and say whether the
      latency targets are feasible (from the bench reports' measured numbers). Reuse the planner:
      run `compute_optimal_placement` / the booking formula over candidate lineups instead of a
      second model of memory. Deliver as `hypnos advise --vram 24G --ram 64G --needs chat,voice,
      documents` (detected hardware by default), `POST /v1/advisor`, and a rendered config (or
      `POST /v1/models` calls, API-11) the user can apply.
      Done 2026-10-31: `hypnos-memory/src/advisor.rs` (budgets as the server derives them, each
      candidate placed with the scheduler's planner, catalog `[models.estimate]` for sizing
      without files, an exhaustive search over entries x variants scoring model size per need and
      interactive co-residency), `hypnos advise` and `POST /v1/advisor` (a lineup, tiers,
      priorities, slots, warnings, rough load times, a config file), SDK `advise`. Latency is only
      estimated from bandwidth (no per-machine benchmarks yet); the UIs come with DESK-18.
      (commits c5b9309, 699186f)

- [x] 🟪 **API-11** `P2` Add, remove and replace models at runtime (a dynamic model pool)
      Requests already route by model name (`acquire_model(id)` loads or evicts on demand); what
      is fixed is the set of models, read once from the config. Make it changeable while the
      server runs:
      - `POST /v1/models` with a catalog spec (`{"catalog": "lfm3-vl:2b:Q4_K_M"}`) or a raw
        `[[models]]` block: files must exist under `models_dir`, the GGUF's architecture must be
        one of `TESTED_GGUF_ARCHITECTURES` (others refused), the model must fit the budgets at
        all; it is registered without loading (first request loads it).
      - `DELETE /v1/models/{id}`: stop admission, wait for leases, evict, unregister.
      - `PUT /v1/models/{id}`: replace atomically (register the new one, switch the name, drain
        and unload the old one), so clients using the name never see an error.
      - Optional aliases (`chat` -> the current chat model) so clients keep a stable name.
      - `ModelScheduler`'s `backends` / `admission` maps become lock-protected (the memory
        manager's model map already is).
      - Persist the changes in an overlay file next to the config (`hypnos.models.toml`, applied
        at startup) instead of rewriting the user's config.
      - With API keys set, these routes need an admin key; paths stay inside `models_dir`.
      - `docs/api.md`, mock server, the three SDKs; the desktop Catalog's Add and the playground
        catalog tab use the live routes (no server restart).
      Done 2026-10-31: `POST /v1/models`, `PUT` / `DELETE /v1/models/{id}` (`pool.rs`: startup
      checks for a new block, a per-model gate in the scheduler so requests wait during a swap,
      overlay `<config>.models.toml`, admin keys), SDK methods with a PermissionDenied error and
      no automatic retries, live Add / Replace / Remove in the desktop Models dialog and the
      playground. Aliases left out (a replacement keeps the id). (commits 7bc0e97, 6b8ab9b,
      0f976a3)


- [x] 🟪 **API-10** `P2` Hugging Face model inventory so users can bring their own models
      A curated inventory of Hugging Face repos/files Hypnos can run, grouped by modality: text,
      vision, speech-to-text, text-to-speech, image generation, embedding. Each entry names the
      engine and format it needs (GGUF for llama.cpp incl. the mmproj for vision/audio-in, GGUF or
      safetensors for sd.cpp, ONNX + voices for Kokoro), quantizations, size and the matching
      `[[models]]` / `[models.arch]` block. Deliver as a machine-readable file next to
      `models.manifest`, a `GET /v1/models/catalog` route, `./run.sh models <hf-repo>` download
      with checksum, and a picker in the frontend / desktop settings that writes the config block.
      Decide first: curated list vs. live HF search filtered by `gguf` tag + pipeline type, and
      how unknown architectures are validated (`gguf_inspector.rs`) before they are offered.
      Decided 2026-10-31: a curated catalog (`backend/models.catalog.toml`), limited to
      architectures Hypnos has been tested with; a model whose GGUF reports another
      architecture is refused before it is offered.
      Done 2026-10-31: `backend/models.catalog.toml` (lineup + LFM3, variants per quantization;
      `hypnos-core/src/catalog.rs` checks it, renders `[[models]]` blocks, a test keeps the
      lineup equal to `models.manifest`), `GET /v1/models/catalog` (on disk, `outdated`, a GGUF of
      the wrong architecture as `problem`, the block), `hypnos --catalog-block|--catalog-files`,
      `./run.sh models <id>[:<quant>]` with a SHA-256 check (fetch-models.ps1 not run: no
      PowerShell on the Mac), SDK methods, a catalog tab in the playground and a Catalog tab plus
      a header button in the desktop Models dialog (Add for the app's own config).


- [x] 🟥 **BUG-6** `P1` Kokoro's fp16 graph returns all-NaN audio on Windows too; use fp32 everywhere
      Seen in the voice round-trip benchmark on the RTX 5090 laptop (2026-10-31): the two-sentence
      reply "That is one of the most iconic lines from John F. Kennedy's inaugural address. It
      serves as a powerful call to civic duty and personal responsibility." made
      `kokoro/model_fp16.onnx` return 250,200 non-finite samples (HTTP 500), while each sentence
      alone and a 628-character paragraph synthesized. Same failure as on Apple Silicon
      (`docs/apple-silicon.md`). The fp32 `model.onnx` synthesizes it (10.4 s of audio in 1.5 s,
      RTF 0.14, the same as fp16 on this CPU).
      Fixed: `kokoro/model.onnx` is the weights path on every platform (both example configs, the
      default profile, one manifest row for `all`, README, CLAUDE.md); `./run.sh models
      kokoro-82m-q4` fetches it (310 MiB). The fp16 file can be deleted. (commit dc71df9)

- [x] 🩵 **DESK-17** `P2` First-run model download in the desktop app
      A bundled server without weights still leaves users stuck (DESK-16). On first run, or from
      settings, show the lineup from `models.manifest` (31 GB for all of it) with size and
      modality per model and let people pick. Download into
      `~/Library/Application Support/Hypnos/models/` (macOS) with progress, resume and checksum
      (the same manifest `fetch-models.sh` uses), enable the downloaded models in `hypnos.toml`
      and disable the others, then start or restart the local server. Later the picker can take
      entries from the API-10 inventory.
      - Models directory in the settings: a folder picker (default `.../Hypnos/models/`, e.g. to
        put 31 GB on an external SSD) used by both the downloader and the server. The server has
        no such concept yet (each `weights_path` resolves from the config's folder), so add
        `[server] models_dir` plus `HYPNOS_MODELS_DIR` that relative weight paths resolve
        against (validation, `--check-config`, `docs/operating.md`, the example configs and the
        default profile). Changing it offers to move the existing files, shows the free space
        and restarts the local server; a missing folder (unplugged drive) is reported as such,
        not as missing weights.
      Done 2026-10-31: `[server] models_dir` / `HYPNOS_MODELS_DIR` (one error for a missing
      folder), the Models dialog (lineup, sizes and SHA-256 from the hosts, `.part` resume, Stop,
      free-space check, restart of the local server), Change… with moving the files. Tested
      through the UI on a clean HOME (commit 7ae3f94).

- [x] 🩵 **DESK-16** `P2` Bundle the server inside the macOS app, as Ollama does
      The `.dmg` ships `hypnos` (Metal build, about 136 MB with the dylib and espeak data), and
      `local_server.rs` starts it when no other server is configured, so app and server always
      match versions. Windows and Linux keep the standalone archives (CUDA vs CPU, runtime size).
      - Placement: `hypnos` in `Contents/MacOS` as a Tauri sidecar (`externalBin`, named
        `hypnos-aarch64-apple-darwin`), `libstable-diffusion.dylib` in `Contents/Frameworks`
        (`bundle.macOS.frameworks`). The binary finds the dylib through
        `@executable_path` only, so link it with an extra `-rpath @executable_path/../Frameworks`.
        No Mach-O files in `Resources`: codesign treats that folder as data.
      - Data paths: `espeak-ng-data` and the playground `frontend/dist` in `Resources`. Kokoro
        looks beside the binary (`kokoro.rs:243`) or in `PIPER_ESPEAKNG_DATA_DIRECTORY`, so the app
        sets that variable when it spawns the server; give the frontend path the same treatment
        (`hypnos-server/src/lib.rs:286` also resolves it from `current_exe`).
      - Config and models in `~/Library/Application Support/Hypnos/` (`hypnos.toml`, `models/`),
        never inside the bundle, which must stay read-only to keep its signature valid. The app
        writes a default `hypnos.toml` from `hypnos.macos.example.toml` on first run.
      - Signing: Tauri signs sidecars and frameworks in its pass with the hardened runtime; check
        that library validation passes inside the bundle and that the notarized `.dmg` still
        assesses as "Notarized Developer ID" (`docs/signing.md`).
      - Licensing: `hypnos` statically links espeak-ng (GPL-3.0). Shipping it as a separate
        program in the bundle should count as aggregation, so the app's own code stays
        MIT/Apache, but the bundle must carry `licenses/GPL-3.0.txt`, `THIRD-PARTY-NOTICES.md` and
        a pointer to the corresponding source, as the server archive does. Confirm that reading
        before the first release that bundles it.
      - release.yml: the desktop macOS job needs the Metal server build (artifact from the
        `macos-arm64-metal` leg) before `tauri build`. Update `desktop/README.md`, `docs/signing.md`.
      Done 2026-10-31. CI run 41207858535: the `.dmg` and the app inside are notarized and stapled
      ("Notarized Developer ID" with a download quarantine flag), the bundled server and its dylib
      notarized, the CI app starts its server on a clean HOME. Follow-ups on the way: ad-hoc
      builds without the hardened runtime, `build-server.yml` so the `.dmg` skips the CUDA
      builds, notarization with an App Store Connect API key, the `.dmg` notarized itself. (commits 3a3036b, 5ecb943, 0c53027, fbe02ed; the first says DESK-15, the ID the
      regenerate card already had)

- [x] 🩵 **DESK-15** `P2` Regenerate from any request or answer
      A refresh icon (on hover) under every user request and assistant answer: it asks that
      request again (for an answer, the request it replies to), drops everything after it and
      streams a new answer; a confirmation names the later turns that would be removed, and an
      `/image` request makes a new image. Replaces the text "Regenerate" link on the last answer.
      `ChatStore.regenerateFrom` / `laterTurns`; answers carry `at` so the speaker reads a
      regenerated answer from its start (the read-aloud session was keyed by index only).

- [x] 🩵 **DESK-14** `P2` Desktop: switches for KV parking and MoE offload
      The same two controls in the desktop app: a KV parking toggle and cap in the connection
      dialog (or the stats panel), the parked conversations in the stats panel, and a MoE offload
      selector per model in the stats panel's model list (MoE models only), all through the Rust
      SDK (`kv_park` / `set_kv_park`, and the MoE call MEM-5 adds). The stats panel's "Last
      response" shows the restored tokens and restore time when a swap brought conversations
      back. After MEM-5.
      Done: Rust commands `hypnos_kv_park`, `hypnos_set_kv_park`, `hypnos_set_moe_offload` on the
      SDK; the stats panel gets a KV parking section (toggle, cap with Set, parked conversations)
      and, on models with `expert_bytes`, a MoE `off` / `auto` / `on` selector with the CPU expert
      layers and the `applies` note; "Last response" shows restored tokens / restore time and the
      expert layers on the CPU. Ignored live test switches both on a server built from af9f8dd and
      restores them. Not yet seen in the window.

- [x] 🟦 **UI-2** `P2` Playground: switches for KV parking and MoE offload
      In the memory panel: a KV parking toggle and cap (`GET` / `POST /v1/memory/kv-park`), with
      what is parked per model (slots, tokens, bytes, age, from `kv_park` in
      `/v1/memory/status`); and per model card a MoE offload selector `off` / `auto` / `on`
      (`POST /v1/models/{id}/moe-offload`, shown only when `expert_bytes` > 0), with
      `placement.cpu_expert_layers` and whether the change applied now or at the next load. Show
      `kv_restored_tokens` / `kv_restore_ms` in the telemetry log when a load restored
      conversations. After MEM-5.
      Done: memory panel "KV parking" box (toggle, cap in GiB with Set, used / cap, parked
      conversations per model with tokens, bytes and age); model cards with expert tensors get a
      MoE offload `off` / `auto` / `on` selector, the expert layers on the CPU and the
      `applies` note; the telemetry log shows experts on the CPU and the restored tokens / restore
      time. Hidden against servers without the features (no `kv_park`, `expert_bytes` 0). Builds;
      not yet seen against a server built from af9f8dd.
      (commit 299d72d)

- [x] 🟫 **MEM-5** `P2` Expert offload for MoE models: experts on the CPU, the rest on the GPU
      Design in docs/moe.md. Step 1, benchmark on the RTX 5090 laptop: the 26B full GPU, a layer
      split, experts on the CPU for all 30 layers (`add_cpu_moe_override`) and for the last 15;
      decode tok/s, 4-client aggregate, 8k prefill, TTFT, VRAM and host RAM. Stop if experts on
      the CPU are not clearly better than the layer split at equal VRAM. Then a placement mode:
      `cpu_expert_layers` in `LayerPlacementPlan`, expert bytes per layer from the GGUF
      inspector, booked as host memory, per-layer `blk.N.ffn_*_exps` overrides in the llama
      backend, preferred over a layer split or an eviction (config `moe_offload`), shown in
      `/v1/memory/status` and telemetry. Gemma 4 26B: experts ~12.0 of ~13.6 GiB, ~0.8 GB read
      per decoded token. Discrete GPUs only.
      Done (step 1's CUDA benchmark still open): optional, off by default: per model
      `moe_offload = "off" | "auto" | "on"`, `POST /v1/models/{id}/moe-offload` at runtime (an idle
      resident model is unloaded, `applies: "now"`, else `"next_load"`); `moe_offload`,
      `expert_bytes` and `placement.cpu_expert_layers` in `/v1/memory/status`. `gguf.rs` sums the
      `*_exps` tensors (sizes from the data offsets) into `arch.expert_bytes`; the planner tries
      full-GPU placements with the last `n` layers' experts in host memory before a layer split
      (`on` = all, `auto` = smallest `n`, discrete GPUs only); `load_model` adds a tensor override
      that also matches Gemma 4's fused `ffn_gate_up_exps` (the bindings' pattern does not).
      Measured on Apple Silicon (26B): 12.0 GiB of experts in `CPU_REPACK`, 2.8 GiB of weights on
      the GPU, decode 47 vs 105 tok/s, prefill 300 vs 4,100 tok/s, identical answer; the runtime
      switch reloads it full-GPU. Docs: api.md section 6, operating.md, example configs, moe.md;
      SDKs `set_moe_offload` / `setMoeOffload`. CUDA (RTX 5090 laptop, 2026-10-31,
      `docs/bench/cuda-features-2026-10-31.md`): VRAM weights 15.8 -> 3.0 GB, decode 141 -> 36
      tok/s, prefill 4,100-5,800 -> 1,400-2,800 tok/s (2-3x, not 14x); KV parking there saves
      0.1-1.9 s per turn after a swap, the 3.5-3.8 s switch remains.

- [x] 🟧 **PERF-8** `P2` A repeated prompt (Regenerate) is prefilled in full
      Sending a byte-identical prompt that a slot has just answered gets `cached_tokens: 0` and a
      full prefill (1.2 s for 4.7k tokens on the 26B, Metal), with or without a swap; the next
      turn of the same conversation reuses fine. The engine must re-evaluate the last prompt token
      to get logits, so the divergence lands one position before the SWA checkpoint taken at the
      prompt end, and the sliding-window rule refuses the reuse. Fix: checkpoint one position
      earlier (or keep the last prompt token's logits), so Regenerate in the playground and the
      desktop costs one token of prefill. Found while measuring MEM-4
      (docs/bench/mem4-baseline-2026-10-30.md).
      Done: with sliding-window attention the text prefill now stops one token before the prompt
      end, takes the SWA checkpoint there (`checkpoint_slot`), and evaluates the last token in the
      next step; a multimodal prompt (libmtmd, one piece) is still checkpointed at its end.
      `plan_cache` is unchanged: a repeat reuses up to `len - 1` and restores the checkpoint.
      Measured on Metal (26B, 4.7k-token prompt): repeat 1,155 ms prefill (0 cached) -> 11 ms
      (4,731 cached), TTFT 59 ms, identical answer at temperature 0; the next turn and MEM-4
      unchanged. Unit test `a_repeated_prompt_resumes_from_the_checkpoint_before_its_last_token`.
      (commit 108d11d)

- [x] 🟫 **MEM-4** `P2` Keep each slot's KV state across a demotion
      Design in docs/pagination.md. On demotion the worker saves every slot's full sequence
      state (`state_seq_get`, flags 0) with its token keys, the SWA checkpoint and a context
      signature, outside the worker; on promotion, if the signature matches, the next request
      for that conversation is matched by longest prefix and restored (`state_seq_set`), with
      the same divergence rules as today's in-context reuse, else it prefills normally. A cap
      `hardware.kv_park_bytes` booked by the memory manager, LRU eviction, shown in
      `/v1/memory/status`; new timings `restore_ms`, `restored_tokens`. Size from the GGUF:
      ~360 MiB for an 8k conversation on the 26B (20 KiB/token + <=200 MiB SWA). Bench: N-token
      conversation, swap, next turn TTFT with and without (N = 2k, 8k, 16k). Not paged attention,
      no Tier 3 spill at first.
      Baseline measured (docs/bench/mem4-baseline-2026-10-30.md, Metal, `mem4_bench.py`): turn
      after a swap 1.4 / 2.6 / 4.7 s for 1.7k / 6.3k / 12.6k tokens, ~0.95 s of it the tier switch,
      the rest re-prefill; expected ~1.1 s with the restore (2.4x at 6k, 4.3x at 12k).
      Done: optional, off by default: `hardware.kv_park` (bool) and `kv_park_bytes` (4 GiB
      default) in the config, `GET` / `POST /v1/memory/kv-park` (`enabled`, `max_bytes`) at
      runtime, `kv_park` in `/v1/memory/status`. `hypnos-backends/src/kv_park.rs`: a shared pool
      with the cap and least-recently-parked eviction (off drops everything, a lower cap the
      oldest, a disabled model its own). The llama worker parks each used idle slot
      (`state_seq_get` + keys + SWA checkpoint) on `LlamaJob::Park` before it is dropped (demotion
      or a new layer split) and restores them right after the context exists when the
      `ParkSignature` matches; `tier_switch_phases.kv_restore_ms` / `kv_restored_tokens`.
      Differences from the design: eager restore (prefix reuse then picks the slot), own cap
      rather than the Tier 2 budget. Measured on Metal (26B): turn after a swap 1.40 / 2.56 /
      4.71 s -> 1.01 / 1.01 / 1.06 s at 1.7k / 6.3k / 12.6k tokens, restore 2-5 ms; identical
      answer at temperature 0; off and a too-small cap fall back to the full prefill. Docs: api.md
      section 14, operating.md, example configs; SDKs `kv_park` / `set_kv_park`.
      (commit f5b891a)

- [x] 🩵 **DESK-13** `P2` Request log viewer, as in the playground
      A "Logs" view (a tab beside the stats panel, or its own window) over the server's request
      log: `GET /v1/logs` refreshed every few seconds while open, one row per request (time,
      route, model, key label, status, duration, tokens in / out, tok/s, tier switch, previous
      tier, error), filters by model, route and status, and a click that opens the record
      (`GET /v1/logs/{id}`: input and output text, request and response payloads when the log
      level captures them, telemetry). The log level selector (`POST /v1/logs/level`: off ...
      trace) and Clear (`DELETE /v1/logs`) as in the playground's "Request Logs" tab. Calls go
      through the Rust SDK (`logs`, `log`, `set_log_level`, `clear_logs`) like the rest of the
      app; a row for the app's own last answer can be highlighted from its `x-request-id`.
      Copy a record as JSON. The local server's process output (DESK-5) stays in its own dialog.
      Done: Rust commands `hypnos_logs` (model / route / status filter, limit),
      `hypnos_log`, `hypnos_set_log_level`, `hypnos_clear_logs` on the SDK; the chat `done`
      event carries `requestId`. `Logs.svelte` replaces the chat (header button, Cmd/Ctrl+L,
      Esc closes): table refreshed every 3 s while open and visible (pause toggle), filters
      from the models and routes seen, level selector, Clear (confirmed), details panel with
      the fields, error, input / output, payloads, Copy JSON and a hint about debug / trace;
      "Log" under each answer opens its record. Ignored live test reads the log of the running
      server (the chat's record by its request id, filters) without changing it. Not yet seen
      in the window.

- [x] 🩵 **DESK-12** `P2` Menu bar / system tray icon
      The Hypnos logo in the macOS menu bar, the Windows notification area and the Linux
      AppIndicator area (Tauri `tray-icon`), coloured by server state: green ready, amber while
      models load, red when unreachable or the API key is refused. Menu: Open Hypnos, the
      server status and version, Start / Stop local server (DESK-5), Connection settings, Quit.
      A left click shows or hides the main window (option: a small panel under the icon placed
      with `tauri-plugin-positioner`, with a quick chat or the stats). Closing the window keeps
      the app in the tray, with a setting to quit instead; on macOS an option to hide the Dock
      icon (`accessory` activation policy). Optional: a system notification when a long answer
      or an image finishes while the window is hidden. Linux needs libayatana-appindicator
      (already in the CI and release deps).
      Done: `src-tauri/src/tray.rs` (Tauri `tray-icon`; the logo at 64 px with a status dot
      drawn per state; menu: status line, Open Hypnos, Stop the server / Start the server,
      Connection settings, Quit; left click toggles the window; `on_window_event` hides on
      close when `close_to_tray`; `set_dock_visibility` on macOS; `RunEvent::Reopen` shows the
      window). The server item follows the connection: "Stop the server" whenever a local
      server answers, including one the app did not start (`local_server_stop_port`: the
      process named `hypnos` listening on that port, SIGTERM / taskkill, wait until it is
      gone), "Start the server" on the connection's port, disabled for a remote server.
      Settings: keep running in the tray, hide the Dock icon (macOS), notifications
      (`tauri-plugin-notification`, only when the window is hidden or unfocused). Not done:
      the optional panel under the icon (`tauri-plugin-positioner`). Checked: the icon with a
      green dot in the macOS menu bar; ignored test stops a `hypnos` the app did not start and
      leaves another process alone. The menu itself not clicked through yet.

- [x] 🟪 **API-9** `P3` Tool calling on `/v1/chat/completions`
      Accept OpenAI `tools` / `tool_choice` and return `tool_calls` (and `finish_reason:
      tool_calls`), using llama.cpp's chat-template tool support for the models that have it;
      stream the calls as deltas. Lets DESK-11 (and any OpenAI client) use real tool calls
      instead of a prompt convention. docs/api.md, the mock server and the three SDKs follow.
      Done: `tools` / `tool_choice` on the request, `tool_calls` / `tool_call_id` / nullable
      `content` on messages, `tool_calls` on the response and as `delta.tool_calls` chunks
      (one per call, arguments whole), `finish_reason: "tool_calls"`. Gemma 4's native format in
      `hypnos-backends/src/tools.rs` (declarations, past calls and results rendered as the GGUF
      template does, a streaming `CallParser`); the engine passes the call markers through and
      stops at `<|tool_response>`; `required` / a named function are forced by prefilling the
      call's opening; other architectures answer 400. Mock server calls tools
      deterministically; docs/api.md 7.5; the three SDKs (options, types, `tool_message` /
      `toolMessage` / `ChatMessage::tool`, `assistant_message`, `parse_arguments`, streams
      accumulate calls by index) with tests (Python 43, JS 41, Rust 44 passing). Live on
      gemma4:e4b: a weather call, its result answered, required / named / none, streaming.
      The desktop image tool (DESK-11) uses native tool calls since the follow-up commit
      (prompt convention only as the fallback for models without a native format).

- [x] ⬛ **BUILD-9** `P3` Sign and notarize the desktop installers
      The release's desktop installers are ad-hoc signed on macOS and unsigned on Windows
      (BUILD-8), so users must confirm the first open. Add an Apple Developer ID certificate
      + notarization (`APPLE_CERTIFICATE`, `APPLE_ID`, `APPLE_TEAM_ID`, app-specific password)
      and a Windows code-signing certificate as repository secrets and wire them into the
      `desktop` job of release.yml (Tauri reads them from the environment).
      Done: `src-tauri/src/local_server.rs` (commands detect / start / stop / status / logs /
      probe; output kept in a 2000-line ring without ANSI codes and pushed as events; the
      config's folder is the working directory; loader paths as in run.sh; SIGTERM, then a
      kill after 15 s; stopped on `RunEvent::Exit`; refuses a port where Hypnos already
      answers, checked with `hypnos-client`). `LocalServer.svelte` from a header button (and
      "Start a local server" when unreachable): binary, config, port, start with the app,
      live log; starting it switches the connection to `http://127.0.0.1:<port>`. Ignored
      test `starts_logs_and_stops_a_real_server` (Tauri mock app + real binary) passes.
      (commit 5590302)
      Done (wiring; needs the secrets to take effect): the `desktop` job of release.yml passes
      `APPLE_CERTIFICATE`, `APPLE_CERTIFICATE_PASSWORD`, `APPLE_SIGNING_IDENTITY`, `APPLE_ID`,
      `APPLE_PASSWORD`, `APPLE_TEAM_ID` to the Tauri CLI (empty ones unset, so the ad-hoc path
      stays as it is) and sets `bundle.macOS.signingIdentity`; on Windows it imports
      `WINDOWS_CERTIFICATE` / `_PASSWORD` into the user store and passes the thumbprint, sha256
      and a DigiCert timestamp URL as `bundle.windows`. The release notes say signed or not
      from the secrets. Shell logic dry-run locally with and without fake secrets; nothing was
      signed for real (no certificates here). Secrets listed in desktop/README.md.

- [x] 🟥 **BUG-5** `P2` Release builds on macOS 27 / Xcode 27.2 cannot load stripped proc macros
      With Homebrew's rustc 1.98 and Xcode 27.2, `cargo build --release` fails with E0463
      "can't find crate for `rustversion`" (and `futures_macro`, `zerocopy_derive`): the
      release profile's default `strip = "debuginfo"` runs Xcode's `strip` on proc-macro
      dylibs and rustc can no longer read them; debug builds are fine. The desktop crate sets
      `[profile.release.build-override] strip = false` (DESK-9). Check whether the backend
      release build hits it too (it is pinned to the rustup toolchain; this machine has only
      Homebrew's), apply the same override there if so, and report it upstream.
      Done: the cause is dyld, not rustc: `dlopen` of the failing dylib says "mis-aligned
      LINKEDIT string pool". Xcode 27.2's `strip -S` leaves the string table misaligned for
      some lengths of the paths embedded in it, so the same crate fails in one target
      directory and loads in another (reproduced: desktop/src-tauri/target fails, a scratch
      target with the same sources and flags works; serde_derive happened to be fine). Both
      workspaces now set `[profile.release.build-override] strip = false` (build-time code
      only; shipped binaries stay stripped). Verified: desktop bundle builds; backend's
      engine-free crates build in release (the full engine build was not rerun). Still to do
      by hand: report it to Apple (Feedback Assistant) / rust-lang with the dlopen message.
      (commit 27c0260)

- [x] 🩵 **DESK-11** `P2` Image-generation tool in the chat
      The server ignores `tools`, so the app runs the tool loop: the system prompt describes
      `generate_image(prompt, size)` and the reply format for calling it; when the chat
      model's answer is such a call, the app calls `POST /v1/images/generations` on the
      `image_generation` model (MOSS-Image), shows the image inline with its prompt (save,
      copy, regenerate), and feeds a short result back so the model can comment. An explicit
      "Image" button (or `/image <prompt>`) skips the model. Show progress during the load
      and the steps (the model may need a tier switch and evict the chat model: say so).
      Generated images are saved with the chat. After DESK-6; API-9 would make the call
      detection robust.
      Done: `src-tauri/src/images.rs`: `TOOL_INSTRUCTION` (a ```generate_image block with
      JSON prompt and size), `ToolGate` (passes an ordinary answer on within its first
      characters, holds a tool call whole), lenient `parse_call`, `generate` (routed
      `image_generation` model, PNG -> 1024 px JPEG data URL), `hypnos_generate_image`,
      `save_data_url`. `hypnos_chat(tools)` adds the instruction only when a model
      generates images, emits `toolCall` / `image`, then a comment round without the
      instruction; earlier images go back to the model as their prompt. Chat: image with
      prompt, Save / Copy (PNG to the clipboard) / Regenerate; progress with elapsed time and
      the image model's tier; Image button and `/image`; a settings switch. Live test: the
      chat model wrote an apple prompt, MOSS-Image made it in 134 s, comment followed, no
      block leaked. (commit 2891d73)

- [x] 🩵 **DESK-10** `P2` Speaker button: read answers aloud
      (Text-to-speech; DESK-9 covers speech-to-text.) A speaker toggle in the chat header
      reads every new answer aloud as it streams: split into sentences, synthesize each with
      `POST /v1/audio/speech` on the `text_to_speech` model (Kokoro), play them in order in
      Rust (`rodio`) so playback starts after the first sentence. A play/stop button on each
      answer reads it on demand; voice and speed in the settings; markdown and code blocks
      are stripped or announced, not read symbol by symbol. Stop and new messages cancel
      playback. After DESK-6.
      Done: `src-tauri/src/tts.rs`: `speak(id, text, done, voice, speed)` cleans the
      markdown (code blocks announced, links as text, URLs as "a link", bullets, emphasis
      and table pipes dropped; an open fence silences the rest), sends each newly complete
      sentence to a worker that synthesizes it on the routed `text_to_speech` model and
      queues it on a rodio player thread; `speak_stop`; `speech-state` events. Chat: speaker
      toggle in the header (only answers streamed in this window are read), Read aloud / Stop
      reading per answer, stop on send and chat switch; voice (Kokoro ids) and speed in the
      connection dialog. Live test synthesizes and decodes each sentence (speakers not used);
      playback itself not yet heard. (commit 8cb22ca)

- [x] 🩵 **DESK-9** `P1` Microphone button: speak a message
      Mic button in the composer: press to record, press again (or release, hold mode) to
      stop. Record in Rust with `cpal` (16 kHz mono WAV; more reliable than getUserMedia across
      WKWebView, WebView2 and WebKitGTK), with a level meter and a timer; then
      `POST /v1/audio/transcriptions` on the model ModelRouter picks for `speech_to_text`, the
      text lands in the composer for review (option: send right away). macOS needs
      `NSMicrophoneUsageDescription` in the bundle's Info.plist; a denied permission shows how
      to enable it. After DESK-6.
      Done: `src-tauri/src/mic.rs`: `mic_start` (default input device, a recording thread
      that owns the cpal stream, mono mixdown for f32/i16/u16/i32, `mic-level` events every
      100 ms, 5-minute cap), `mic_stop` (16 kHz WAV, transcription on the routed
      `speech_to_text` model), `mic_cancel`. Composer: mic button (click to toggle, hold to
      talk), strip with timer, level meter, Cancel/Done, Esc, and "send when done"
      (localStorage). `Info.plist` carries `NSMicrophoneUsageDescription` (checked in the
      built bundle). Ignored test records 1.5 s from the real microphone. Found BUG-5 on the way.
      (commit ebc5173)

- [x] 🩵 **DESK-8** `P2` Extract a URL's content into the chat, and download it
      URLs in the composer (or a "Read URL" action) are fetched by the Rust side (reqwest,
      size and time limits, http(s) only) and converted to markdown with docling (HTML, PDF,
      office files); the result is added to the message context like a document attachment
      (DESK-7) with its title and source. Each extracted item gets "Download" (save as .md,
      or the original file, through the dialog plugin) and "Copy". Rust-side fetching works
      even when the server sets `allow_remote_image_urls = false`. After DESK-7.
      Done: `src-tauri/src/web.rs`: `url_extract` (reqwest, http(s) only, 50 MiB, 30 s, 5
      redirects; file name from the URL or the content type; title from `<title>` or the
      first heading; docling for HTML/PDF/office, plain text as is) returns a `web`
      attachment sent as `<web_page url title>`; `save_text` and `url_download` write to a
      save-dialog path. Composer: a "Read <link>" chip per URL typed; the chip's panel has
      Copy, Download .md, Download <original>. Documents get Copy and Download .md too. Live
      test passes (example.com, a PDF, both saves, a 404 as a sentence). (commit 70b75e1)

- [x] 🩵 **DESK-7** `P1` Attach images, documents, audio and video to a message
      Paperclip button + drag and drop + paste; chips with name, size and a remove button above
      the composer. Routing (in Rust, DESK-6): images -> `image_url` data URL parts (needs a
      `vision` model, else ModelRouter switches or the UI says why); PDF, DOCX, PPTX, XLSX,
      HTML, MD -> `POST /v1/documents/convert` (docling), the markdown goes into the message
      as a quoted context block (truncated with a note when it exceeds the context window);
      audio -> `input_audio` (WAV, converted when needed) when the chat model serves
      `speech_to_text`, else `/v1/audio/transcriptions` and the transcript as text; video ->
      a few frames sampled in the webview (video + canvas, no ffmpeg) as images plus its
      audio track decoded to WAV and transcribed. Limits from docs/api.md
      (`max_image_bytes`, `max_audio_bytes`, `max_document_bytes`) checked before upload.
      Attachments are saved with the chat (images as thumbnails, documents as their text).
      After DESK-6.
      Done: `src-tauri/src/attachments.rs` (`attachment_from_path`, `attachment_from_bytes`
      for paste over a raw IPC body): images -> 1024 px JPEG data URL (`image`), PDF/office/
      HTML -> docling via the SDK, text files read directly (both capped at 60k chars),
      audio -> symphonia decode to 16 kHz WAV -> transcription, video -> soundtrack
      transcript + 6 frames sampled in the webview (`video.js`, asset protocol allowed per
      file). `hypnos_chat` builds text + `image_url` parts and switches to a vision model
      (a `model` event, shown under the answer). Composer: paperclip (dialog plugin), drop
      overlay, paste, chips with progress and the extracted text. Deviation: audio is always
      sent as its transcript, never as `input_audio`, so it works with any chat model and
      persists as text. Live test passes (m4a transcript, PDF text, image colour + document
      in one chat); video frame sampling not yet tried in the window. (commit 0d08176)

- [x] 🩵 **DESK-6** `P1` Route every server call through the Rust SDK in `src-tauri`
      Add `hypnos-client` (`sdk/rust`, path dependency) to the Tauri crate and expose Tauri
      commands for each call the app makes: health/ready, models, memory status, chat (stream
      over a `tauri::ipc::Channel`, cancel by id), transcription, speech, images, document
      conversion. The webview stops talking HTTP: drop the JS SDK alias and
      `tauri-plugin-http` (the Rust client has no CORS or scope problem; `allow_server` goes),
      keep the connection (URL + keychain key) in Rust state. Add a `ModelRouter` that picks
      the model for each job from `/v1/memory/status` `modalities` (never hard-coded ids):
      the chat's model when it serves the modality, else the first enabled model that does,
      overridable per modality in the settings. Errors map to the same sentences as
      `errors.js`. Base for DESK-7..11.
      Done: `src-tauri/src/hypnos.rs` with `hypnos_connect` / `probe` / `test` / `models` /
      `memory_status` / `route` / `set_routes` / `chat` (Channel) / `chat_cancel`; the client
      and the routes live in Rust state, the key is read from the keychain there. JS SDK
      alias, `tauri-plugin-http`, `allow_server` and the HTTP permission are gone (the app is
      Tauri-only: no browser mode). `route` + 3 unit tests; errors become
      `{ kind, message, detail, retryAfterSecs }` (errors.js removed). Settings gain "Models
      for each job". The media commands (transcription, speech, images, documents) land with
      DESK-7..11, where their inputs live in Rust (file paths, the mic buffer, playback).
      Ignored live test `chats_and_cancels_through_the_sdk` passes against the running server;
      no lease is left after a cancel. (commit 42ab87c)

- [x] 🩵 **DESK-5** `P3` Start and stop a local Hypnos server from the app
      Optional: bundle or locate the `hypnos` binary and run it as a Tauri sidecar with a
      chosen config, show its log (`/v1/logs`), stop it on quit. Decide first whether the
      desktop app stays a pure client or ships the server (archive size, GPU builds per
      platform). After DESK-4.
      Decided 2026-10-30: it does not bundle the server (GPU builds per platform, 30+ MB of
      engines, model downloads); it finds an installed `hypnos` (a path chosen in the
      settings, this repo's `backend/target/release`, or PATH) and manages that process.

- [x] ⬜ **DOC-6** `P2` Document the desktop app
      `desktop/README.md` (prerequisites per OS, dev and build commands, connecting to a
      server), a section in the root README, and the new directory in CLAUDE.md. After DESK-3.
      Done: `desktop/README.md` (usage, shortcuts, installing, building per OS, how it reaches
      a remote server, layout); README: desktop rows in the run-script table, a "Desktop app"
      section, the installers under Prebuilt binaries, the repository tree; CLAUDE.md: the
      commands, the CI/release notes and a "Desktop app" architecture section. (commit 8c488f5)

- [x] ⬛ **BUILD-8** `P2` Build the desktop app from the run scripts and CI
      `./run.sh build desktop` / `run.bat build desktop` (npm install + `tauri build`) and
      `start desktop` for `tauri dev`; a desktop build job in ci.yml (macOS, Windows, Linux)
      and, later, signed bundles (.dmg, .msi, .AppImage) in release.yml. After DESK-1.
      Done: `run.sh` / `run.bat` take a `desktop` target, left out of `all`: `build desktop`
      (npm install + `tauri build --no-bundle`, HYPNOS_PROFILE=debug adds `--debug`), and
      `start`/`stop`/`restart`/`status`/`logs desktop` around the built binary; `start` never
      compiles, so development stays `npx tauri dev`. ci.yml: a desktop job on
      ubuntu-22.04/windows/macos-15 (fmt + clippy + tests on Linux). release.yml: .dmg,
      .msi + NSIS .exe, .deb + .AppImage named `hypnos-desktop-<tag>-<platform>.*`, carrying
      the tag's version, ad-hoc signed on macOS, attached to the release with the archives.
      Not run yet: the workflows (run ci by hand before the next tag) and run.bat on Windows.
      (commit 4627348)

- [x] 🩵 **DESK-4** `P2` Connection settings and server states
      Settings screen: server URL (default `http://localhost:9999`), API key kept in the OS
      keychain (not plain storage), "test connection". The UI shows distinct states for
      unreachable, warming up (503 `model_not_available` until ready), unauthorized (401) and
      ready, and recovers by itself when the server comes back. The `http:default` scope in
      `src-tauri/capabilities/default.json` allows loopback only (DESK-1): a remote server
      needs a wider scope (or a runtime one) limited to the configured origin.
      Done: `src/lib/Settings.svelte` (Cmd/Ctrl+, or a click on the status) with Test
      connection; URL in the store, key in the keychain per server origin (`keyring` 4,
      Rust commands `get_api_key` / `set_api_key`); `allow_server` adds a runtime capability
      scoping `http:default` to the server's origin (kept until restart: capabilities cannot
      be removed). The probe tells unreachable / unauthorized (401 on `GET /v1/models`) /
      warming up / ready apart and keeps polling, so the app recovers by itself. Not yet
      tested end to end against a non-loopback server. App icon and in-app logo from
      `assets/logo.png` on a white tile. (commit dc485e6)

- [x] 🩵 **DESK-3** `P1` Collapsible right panel with live Hypnos stats
      Polls `GET /v1/memory/status` (1-2 s, paused while collapsed or the window is hidden):
      Tier 1 / Tier 2 / Tier 3 usage bars and budgets, which models sit in which tier, leases,
      recent evictions with their reason; `/health` and `/ready`. Below it, the last chat
      response's `hypnos_telemetry`: TTFT, prefill/decode tokens/s, cached tokens, previous
      tier, tier-switch latency and phases, evicted models. Toggle button + shortcut
      (Cmd/Ctrl+.), collapsed state remembered. Reuse the visual language of the frontend's
      tier visualizer. After DESK-1.
      Done: `src/lib/StatsPanel.svelte` polls every 1.5 s while open and the window is visible
      (it unmounts when collapsed): server version/uptime/device, unified-memory bar on
      Apple Silicon or Tier 1/Tier 2 bars on discrete GPUs, Tier 3 mapped bytes, process
      memory against its cap, nvidia-smi line, enabled models grouped by tier (leases,
      partial GPU layers, requests, idle), last response's telemetry with load phases, and
      the last five evictions. The last response survives a restart (newest saved answer).
      (commit df02d19)

- [x] 🩵 **DESK-2** `P1` Chat view in the style of Ollama
      Model picker from `GET /v1/models` (chat-capable, enabled models only); streaming
      `POST /v1/chat/completions` over SSE with markdown and code-block rendering; Stop
      (aborts the request, which cancels server-side on disconnect), regenerate, new chat.
      Conversation list in a left sidebar, persisted locally (Tauri store plugin or SQLite);
      per-chat model and system prompt. Handle 429 `Busy` and 503 warm-up with a clear
      message instead of a raw error. After DESK-1.
      Done: `ChatStore` (`src/lib/chats.svelte.js`) keeps the chats in `tauri-plugin-store`
      (localStorage in a browser); the picker lists enabled models whose `modalities` include
      `text_generation`, with their tier; markdown via marked + DOMPurify with copy buttons
      on code blocks; per-answer footer with tok/s, TTFT, tokens and load tier; Cmd/Ctrl+N
      new chat; `src/lib/errors.js` turns SDK errors into sentences. (commit 4633077)

- [x] 🩵 **DESK-1** `P1` Scaffold the Tauri 2 desktop app in `desktop/`
      Tauri 2 + Vite + Svelte 5 (runes) + Tailwind, the same stack and versions as `frontend/`
      (`svelte` ^5, `tailwindcss` ^3, `@sveltejs/vite-plugin-svelte`), no other UI framework, with its own
      `package.json` and `src-tauri/` (not part of the backend workspace). Talk to Hypnos through
      `@hypnos/client` from `sdk/javascript` (Vite alias to its sources) so retries, error classes
      and streaming match docs/api.md. App shell: header with server status dot, main area,
      right panel slot (DESK-3). `npm run tauri dev` opens a window on macOS, Windows and Linux.
      Done: `desktop/` with header (server status dot: connecting / unreachable / warming up /
      ready, polled every 3 s), placeholder chat area, right panel toggled by button or
      Cmd/Ctrl+. (state kept in localStorage). Vite aliases `@hypnos/client` to the SDK's
      TypeScript sources, so no SDK build step. Requests go through `tauri-plugin-http`
      (the `tauri://localhost` origin fails the server's loopback-only CORS default); its
      scope allows loopback only for now. Dev port 1420. (commit ab56291)

- [x] 🟥 **BUG-4** `P1` A second server on the same machine deletes the first one's pictures
      Every instance without `server.document_pictures_dir` stores pictures under
      `<tmp>/hypnos-pictures`, and opening the store removed every document directory there, so
      starting a second instance (`--port`, API-6) wiped the pictures the first was still
      serving (404 on `/v1/documents/{id}/pictures/{n}`). Found by the Rust SDK's tests, which
      run one mock server per test in parallel. The store now removes only documents older
      than the TTL at open; test `a_second_store_on_the_same_directory_keeps_live_documents`.

- [x] 🟪 **API-7** `P2` Client SDKs in Python, Rust and JavaScript
      One package per language, generated from `docs/api.md`, published from this repo
      (`sdk/python`, `sdk/rust`, `sdk/javascript`): typed requests and responses for every route
      (OpenAI ones and the Hypnos extensions: documents and pictures, entities, memory status
      and tier control, enable/disable, logs), `hypnos_telemetry` as a typed optional, SSE
      parsing for chat and document streams with an abort/cancel handle, the retry policy of
      `docs/api.md` section 17 (429 `model_busy` backoff, `Retry-After` on 429/503, 507 wait),
      readiness polling, multipart uploads, per-model concurrency bound. Python: `httpx`
      sync + async, PyPI `hypnos-client`. Rust: `reqwest` + `serde`, crates.io `hypnos-client`.
      JavaScript/TypeScript: `fetch` + `ReadableStream`, npm `@hypnos/client`, works in Node
      and browsers. Each ships examples for all seven default models and a test suite against
      the mock-backend server (`backend/crates/hypnos-server/tests`) run in CI.
      Done 2026-10-30. Shared test target: `cargo run -p hypnos-server --example mock_server`,
      the real router, scheduler and auth in front of mock engines for the default lineup, with
      `/mock/fail`, `/mock/ready` and `/mock/reset` to inject errors and flip readiness; the CI
      job `sdk` (run for releases) tests the three SDKs against it. Python (`hypnos-client`,
      httpx, sync generated from async by `scripts/unasync.py`, 35 tests), JavaScript
      (`@hypnos/client`, TypeScript to ESM, no runtime dependencies, AbortSignal on every call,
      35 tests), Rust (`hypnos-client`, reqwest + tokio, builder requests, dropping a stream
      cancels it, 36 tests); an example per default model in each. Found on the way: the speech
      headers spelled the tier `Tier3NvmeSsd`; docs/api.md did not say that a document stream's
      last event has empty `pages` and ends with `[DONE]`; and a server starting next to another
      one deleted the other's stored pictures (BUG-4). Publishing to the registries is API-8.

- [x] 🟥 **BUG-3** `P1` An engine that runs out of device memory mid-request answers 500
      Seen in the TEST-1 confirmation soak (PERF-7): sd.cpp found 0 MB free for a 518 MB
      workspace and the client got a generic 500. Engines (hypnos-backends) now report
      `HypnosError::DeviceOutOfMemory`: sd.cpp through the out-of-memory lines its log callback
      sees during the job (model manager, ggml allocators, cudaMalloc, Metal), llama.cpp through
      `llama_decode` -2 (batch and embedding decodes). The server answers it with 507
      `device_out_of_memory`; for chat without `stream`, embeddings and images it first calls
      `ModelScheduler::reclaim_vram_for`, which evicts every idle model holding VRAM (reason
      `device_out_of_memory` in `recent_evictions`), and retries once when something was freed.
      Streams are not replayed. Test: `faults.rs`
      `a_device_out_of_memory_evicts_idle_models_and_retries_once_then_answers_507`. Also fixed
      the two lints a newer clippy rejects on main (unused `mut` in `migrate`, a `usize` cast).

- [x] ⬛ **BUILD-7** `P1` `run.sh` reports a backend that dies on start as running
      `wait_up` counted a live pid as up, so a backend killed half a second after the spawn
      printed RUNNING and the UI just said "not connected". It now waits for the port, and a
      process that exits first prints `EXITED during startup` with the log tail (or, with no
      output on macOS, the code-signature hint) and fails the command. `start` and `status` name
      other `hypnos` processes from this build outside the backend port (one left on the old
      8080 kept writing to `.run/backend.log`). Root cause of the silent kill: the
      diffusion-rs build script overwrote `libstable-diffusion.dylib` in place, and macOS
      kills a process that maps a re-written signed file ("Code Signature Invalid", cached per
      vnode); it now copies beside the target and renames over it
      (`vendor/patches/diffusion-rs.patch`).

- [x] 🟪 **API-6** `P2` Choose the port on the command line; default 9999
      `hypnos [config.toml] --port N` (also `--port=N`, `-p N`) and `HYPNOS_PORT` override
      `server.port`, so one config file can serve several instances. The default port moved from
      8080 to 9999 everywhere: `server.port`, the built-in profile, both example configs, the run
      scripts, the Vite proxy, the playground's backend URL, the Dockerfile, the scripts
      (smoke, bench, soak, package), `backend.http`, the README, the website and the docs.
      Done 2026-10-30; first shipped as 8888, changed the same day to 9999 (8888 is Jupyter's
      default and was taken on the development machine).

- [x] 🟧 **PERF-7** `P2` Gemma 4 26B decodes at 12 tokens/s on the RTX 5090 laptop; VRAM is full
      The request log and the soak show 12 tok/s for `gemma4:26b-q4` (README: ~100 tok/s on a
      24 GB CUDA GPU). `/v1/memory/status` has the device full: 21.8 GB booked plus 3.9 GB
      unaccounted on a 25.6 GB device, real free 0. On Windows (WDDM) CUDA allocations past the
      device spill into system memory instead of failing, which would explain the decode rate.
      Confirm with a smaller booking (`slots = 2` or a 8k context) and nvidia-smi, then either
      lower the default booking for 24 GB cards or make the planner keep a margin above the
      unaccounted reading (`vram_safety_margin_bytes` is 1 GiB).
      Also seen in the confirmation soak: MOSS-Image placed fully on the GPU by the booking, then
      sd.cpp failed with "reported free 0.00 MB" needing 518 MB of workspace: the planner should
      compare against the device's real free memory (it is read at every reconcile) and refuse or
      shrink a placement the device cannot hold, not only its own bookings.
      Resolved (`docs/bench/decode-slots-cuda-2026-10-30.md`): not the full device. The decode
      rate depends on the slot configuration: 1 x 16k, 1 x 64k, 4 x 4k and (with f16 KV) 2 x
      16k, 3 x 16k, 4 x 8k all decode at 142-157 tok/s; the old default 4 x 16k q8_0 at 8-10,
      and 2 x 16k q8_0 at 27. With CUDA graphs disabled the old default runs at 70: llama.cpp
      re-captures its CUDA graph on every step for those multi-stream cache shapes (no
      consecutive-updates bail-out in b11982), an upstream report. Defaults changed: the 26B
      runs with `slots = 3`, 16k, `kv_dtype = "f16"`, `activation_workspace_bytes` 2 GiB (the
      2.2 GB compute buffer was the "unaccounted" VRAM). 150 tok/s per stream measured after
      the change. The sd.cpp failure under load is the engine's runtime reserve, now in the
      booking for the 26B; the planner's refusal to place on a full device was correct.

- [x] ⬛ **BUILD-6** `P1` Cut the first release (v0.1.0) and prove the release pipeline
      No `v*` tag had ever been pushed: the archives, the Docker image on GHCR and the publish
      job have never run on GitHub, while the README's quickstart points at them. Push the tag,
      watch the three archive jobs, the Docker push and the publish job, fix what fails (the
      Linux link of `docling-cuda`, the Windows esaxx CRT, runner disk and time limits are the
      known risks), then download each archive and run `scripts/smoke_test.sh` against it on its
      platform. Afterwards the README's prebuilt-binaries section must match what the release
      actually contains.
      Found so far: the v0.1.0 release was made in the GitHub UI (that creates the tag and starts
      the run), and the publish step's `gh release create` fails on an existing release, so the
      archives were never attached; the step now uploads into an existing or draft release.
      Done 2026-10-30: v0.1.0 is published (https://github.com/advancedlogic/hypnos/releases/tag/v0.1.0)
      with linux-x64-cuda (792 MB), windows-x64-cuda (274 MB) + the cuda12-runtime zip,
      macos-arm64-metal (47 MB), SHA256SUMS, and the Docker image on GHCR, built from a390677
      (the publish-step fix; everything from API-5 on goes into the next tag). The UI-created
      release got its tag moved to main and the archives attached by the second run (the first
      run, from the morning commit, had attached its own; the fixed publish step replaced them).
      Windows archive checked here: checksum, contents (provider DLLs, sd.cpp DLL, espeak-ng
      data, playground), starts with the runtime zip beside it and validates its config. Linux
      CUDA took 106 min and Windows CUDA 191 min on the hosted runners; macOS 16 min.

- [x] 🟨 **TEST-1** `P1` Soak test: many clients for hours, memory and bookings must stay flat
      All measurements so far are single-client scenarios (`backend/scripts/bench.sh`). Run the
      bench scenarios in a loop with 4-8 concurrent clients across modalities for several hours
      against a release build, sampling `/v1/memory/status` (process footprint, VRAM booked vs.
      real free, unaccounted), `/metrics` (promotions, evictions, load failures, rejections),
      the picture store and the request-log ring. Pass: no growth after warm-up, no deadlock, no
      429 storm, every eviction explained. Script it as `backend/scripts/soak.sh` with a report
      like `docs/bench/`, and fix what it finds (file them as BUG cards).
      Shipped: `backend/scripts/soak.sh` / `soak.py` (stdlib, reuses bench.py's client): N
      workers pick chat, streaming chat, vision, speech, embeddings, entities and document
      conversion (pictures kept, then released) by weight, one worker generates an image every
      `--image-every` seconds to force evictions, a sampler reads `/v1/memory/status`, `/metrics`
      and the log ring; the report (`docs/bench/soak-2026-10-30.md`) carries pass/fail verdicts
      (footprint and unaccounted VRAM flat after the warm-up with an absolute floor, no 5xx
      other than 507 capacity refusals, 429 under 5 %, no stall, no load failures, every
      eviction explained) plus per-kind latency and the samples. Found and fixed BUG-1 (eviction
      racing a fast-path lease) in the two-minute shakedown and BUG-2 (fast path during a load)
      in the 45-minute run; filed PERF-7 (12 tok/s decode on a full device). 45 min x 4 clients:
      3,705 requests, 2 errors (BUG-2), footprint +1.3 % after warm-up. Confirmation run after
      BUG-2 (`soak-2026-10-30-after-bug2.md`, 12 min): 1,130 requests, no entities error, one
      image 500 from sd.cpp finding the device full (PERF-7), footprint +0.3 %.

- [x] 🟥 **BUG-2** `P1` A request could run on an engine that was still loading
      Found by the 45-minute soak (TEST-1): two entities requests answered 500 "model
      'gliner2.5-multi' is not loaded" in the 3.8 s between the scheduler booking Tier 1 for the
      model and the engine finishing its load. The booking is written before `apply_placement`
      runs, and the fast path trusted it. Fixed: both fast paths in `acquire_model` first ask
      the engine `is_resident()` (every engine implements it) and otherwise wait for the
      transition lock, which the load in progress holds. Scheduler test
      `no_fast_path_lease_while_the_engine_is_still_loading` with a load blocked on a gate.

- [x] 🟨 **TEST-2** `P2` Cover the engine failure paths and contain engine crashes
      The CUDA lifecycle test (`multi_modal_lifecycle.rs`) runs only by hand with real weights.
      Add engine-free tests for CUDA OOM mid-generation (mock backend failing on the n-th token:
      the lease is released, the slot cleared, the request answers 5xx), a weights file that
      changes or disappears under a resident model, and cancellation during a tier transition.
      A C++ abort inside llama.cpp / sd.cpp takes the whole process down: document the
      supervisor expectation (run scripts, Docker restart policy, systemd unit) and add a
      per-engine crash counter to `/metrics` so restarts are visible.
      Shipped: `hypnos-server/tests/faults.rs` runs the real router, scheduler and memory manager
      over a mock engine: a refused load answers 500 `server_error`, rolls the booking back and
      counts in `hypnos_model_load_failures_total{model}`, then the next request loads and
      serves; a failure mid-generation answers 500 (plain) or an error event followed by
      `[DONE]` (stream), releases the lease and leaves the model resident, and the next request
      is served without a reload. Scheduler tests: a request cancelled while another model's
      load holds the transition lock loads nothing, and load failures are counted per model.
      `/metrics` gains `hypnos_model_load_failures_total{model}` and
      `hypnos_process_start_time_seconds` (a restart after an engine abort shows as a change);
      the operator's guide lists both as alerts and describes the supervisor expectation.

- [x] 🟥 **BUG-1** `P1` Eviction can unload a model between a fast-path lease and its request
      Seen in the first soak run (TEST-1, 2026-10-30): an entities request answered 500 "model
      'gliner2.5-multi' is not loaded" at the moment an image request evicted idle models. The
      fast path in `acquire_model` reads the placement (read lock), then `increment_lease` takes
      the write lock: an eviction running under the transition lock lists the model as idle in
      between, demotes it, and the request runs on an unloaded engine. Make both sides atomic
      under the models lock: `increment_lease_if_resident` (check tier and context, then count)
      and `demote_if_idle` (check `active_leases == 0`, then apply the demotion) so an eviction
      skips a model that just got a lease. Mock test in `hypnos-scheduler` (`hypnos-scheduler`,
      `hypnos-memory`).
      Fixed: `TieredMemoryManager::increment_lease_if_resident` (tier, layers and context checked
      and the lease counted under the models write lock) is the fast path in `acquire_model`;
      `demote_if_idle` (lease count checked and the placement applied under the same lock) is
      what the eviction loops, `set_model_tier` and `set_model_enabled` use, skipping a candidate
      that got a lease meanwhile. Scheduler test `eviction_skips_a_model_that_was_leased_meanwhile`.

- [x] ⬜ **DOC-5** `P2` Write an operator's guide (sizing, knobs, telemetry, errors)
      `docs/operating.md` for people who run Hypnos rather than build it: sizing a box for a
      lineup (VRAM / pinned RAM budgets, slots x context), which knob to turn for which symptom
      (429s, evictions, slow first token, OOM), how to read `hypnos_telemetry` and the `/metrics`
      histograms, what every 4xx / 5xx and error `type` means and the fix, log levels and
      redaction, upgrade and rollback, and the Docker / systemd / Windows service recipes.
      Shipped: `docs/operating.md`: sizing (what Tier 1 / Tier 2 bookings add up to, the budget
      rule for pinned RAM), a symptom -> knob table, the telemetry fields and `/metrics` series
      with alerts worth having, every status / error `type` and its fix, log levels and
      redaction, exposure, upgrade and rollback with `config_version`, Docker / systemd (with
      `LimitMEMLOCK`) / Windows service recipes and the supervision model, a go-live checklist.
      Linked from the README's "Operating the server"; CLAUDE.md asks to keep it in step.

- [x] 🟧 **PERF-6** `P1` Tier 2 -> Tier 1 promotion of the 26B takes 23-59 s on Windows CUDA
      Seen while recording the demo on the RTX 5090 laptop (request log, 2026-10-30): the first
      chat after startup promoted `gemma4:26b-q4` from pinned RAM in 23.2 s, and the chat after an
      image generation in 58.7 s, against 0.99 s from Tier 2 on the M5 Max
      (`docs/bench/swap-metal-2026-10-29.md`). Find where the time goes (`hypnos_telemetry`
      has only the total switch; add phases: weight upload, context + KV allocation for
      `slots x default_context_len`, mmproj load, warm-up), check that the CUDA path uploads from
      the VirtualLock'd Tier 2 pages rather than re-reading the file through llama.cpp's own mmap,
      and whether MOSS-Image's unload (text encoder, VAE) is paid inside the next promotion. This
      is PERF-5's CUDA swap measurement (`bench.sh --scenarios 7`) with a target: Tier 2 should
      beat Tier 3 clearly, and a 16 GB upload over PCIe 4/5 is 2-4 s, not 20-60.
      Shipped (`docs/bench/swap-cuda-2026-10-30.md`): phases per Tier 1 load (weights, projector,
      context; log line + `hypnos_telemetry.tier_switch_phases`, `ModelBackend::last_load_phases`).
      Found: (1) a file mapping feeds CUDA tensor by tensor through pageable memory on Windows
      (6.5 s for 15.8 GB); llama.cpp's read path streams through pinned buffers at 5.5 GB/s
      (2.7 s): `models.use_mmap` unset now means read on CUDA, mmap on Metal / CPU. (2) Tier 2
      pinned only `weights_path`; the 1.2 GB projector was evicted from the page cache by the next
      load and cost 4-7.5 s per promotion (the 23-59 s cases): `aux_weights_path` and
      `text_encoder_path` are pinned with the weights. (3) With everything booked actually locked,
      the 32 GiB budget plus the process starved a 64 GB machine (134 s eviction while paging):
      MOSS-Image is no longer kept warm in the example profiles (sd.cpp reloads it in 0.9 s) and
      the manager warns when the pinned budget exceeds half of physical RAM. Result: 3.7-4.2 s
      per promotion, also right after an image (was 10.9 s), image request 11 s (was 49-144 s).

- [x] 🟫 **MEM-3** `P3` Version the config and the model manifest with a migration path
      `HypnosConfig` has no schema version; the first breaking change to a field or to
      `models.manifest` will break every deployed `hypnos.toml`. Add `config_version` (absent =
      1) with per-version migration notes in the README, warnings for deprecated keys instead of
      silent acceptance, and a `hypnos --check-config` mode that prints what `validate` and the
      GGUF detection would do without starting the server.
      Shipped: `config_version` on `HypnosConfig` (absent = 1, `CONFIG_VERSION` = 1): a newer
      file is refused by `from_toml_str` with both versions named, an older one goes through
      `HypnosConfig::migrate` (one step per version, notes returned; none yet). Keys no field
      consumes are found by diffing the raw TOML against the parsed config serialized back
      (`unknown_keys`, dotted paths with model indices) and reported as validation warnings.
      `hypnos --check-config [file]` runs the GGUF detection and validation, prints one summary
      line and exits 1 on errors; `--version` and `--help` too. `models.manifest` carries
      `# manifest-version: 1`, which both fetch scripts refuse when newer. README "Config
      versions" section with the migration table; example configs start with `config_version`.

- [x] 🟫 **MEM-2** `P3` Expose GPU device health on /metrics (utilisation, temperature, errors)
      `/metrics` covers bookings and placements but not the device itself. Through the existing
      probe path (`GpuMemoryProbe`, NVML / `nvidia-smi` on CUDA, IOKit on Apple Silicon) export
      utilisation, temperature, power, clock throttling and Xid / driver error counts, sampled
      with the VRAM reconcile; show the first three in the playground's tier visualizer header.
      Shipped: `GpuHealth` in `hypnos-memory/src/hardware.rs` (utilisation, temperature, power,
      SM clock, throttle reasons + `throttled`, ECC uncorrected errors, pending retired pages)
      sampled through `nvidia-smi --query-gpu` on NVIDIA (Metal / ROCm: none yet, the field is
      absent), cached in `TieredMemoryManager::gpu_health` and refreshed at most every 3 s from
      `status_snapshot`; `gpu` in `/v1/memory/status`, `hypnos_gpu_*` gauges on `/metrics`, and
      the visualizer header shows utilisation, temperature and power in red while throttled.
      Xid counts are not in nvidia-smi's query set; ECC + retired pages stand in. Parser test.

- [x] 🟪 **API-5** `P2` Per-key rate limits and quotas, and redaction in the request log
      API keys are all-or-nothing. Add per-key limits in `[server]` (requests per minute,
      concurrent requests, tokens per day) enforced in `auth::require_api_key` with 429 and
      `Retry-After`, a key label in the request log and `/metrics`, and an audit line per
      rejected call. The request log captures prompts and completions at `debug`: add
      `request_log_redact` (patterns or "all") and keep `info` the documented level for anything
      exposed beyond a trusted group. Until then the README should say to put a gateway in front.
      Shipped: `[[server.keys]]` (label, key, `requests_per_minute`, `max_concurrent`,
      `tokens_per_day`) and `[server.api_key_defaults]` for every key incl. plain `api_keys`
      (labelled `key-<8 hex of digest>`); `quota.rs` `KeyLimiter` (minute window, semaphore held
      through the response body via a guarded stream, UTC-day token quota fed by `observe_request`
      for plain responses and by the chat stream at its end); `require_api_key` answers 429
      `rate_limit_error` + `Retry-After` and logs an audit line with label and reason; the label
      is on every request-log record (`?key=`), in the playground's log table and on `/metrics`
      (`hypnos_api_key_requests_total`, `_rejections_total{reason}`, `_tokens_total`,
      `_tokens_today`). `server.request_log_redact` (patterns or `"all"`) applied in `RequestLog`
      on record and update. Tests: limiter, redaction, labelled keys, router test `tests/keys.rs`.
      README says to put a gateway in front for anything finer than per key.

- [x] ⬛ **BUILD-5** `P2` Publish the website on hypnos.maai.solutions (AWS S3)
      Serve the static site from DOC-4 (`website/`) at https://hypnos.maai.solutions. An S3
      bucket alone only gives HTTP on an AWS hostname, so the custom domain with HTTPS needs a
      private S3 bucket behind CloudFront (origin access control, `index.html` default root,
      cache policy for the static assets) with an ACM certificate issued in us-east-1 (DNS
      validation) and a DNS record for `hypnos` in the `maai.solutions` zone (Route 53 alias if
      the zone is there, otherwise a CNAME at the current DNS provider). Deliverable, one of:
      (a) Terraform under `infra/website/` (bucket, OAC, CloudFront distribution, ACM cert +
      validation records, DNS record; variables for domain, zone and region; `terraform output`
      with the distribution id) plus a `deploy.sh` that runs `aws s3 sync website/ s3://<bucket>
      --delete` and a CloudFront invalidation; or (b) `docs/website-deploy.md` with the same
      steps done by hand in the console / AWS CLI. Either way write `docs/website-deploy.md`
      (prerequisites: AWS account, CLI profile, where the zone lives; first deploy; updating;
      costs: S3 + CloudFront free tier covers a site this size). Optional follow-up: a
      `website.yml` GitHub Actions job that syncs on every push to `main` touching `website/`
      (OIDC role, no long-lived keys).
      Done 2026-10-30: https://hypnos.maai.solutions answers over HTTPS. Set up outside the
      repository (no Terraform or `docs/website-deploy.md` checked in); add them if the deploy
      is to be reproducible from the tree.

- [x] 🟪 **API-4** `P2` Listen before startup pre-staging so /ready answers 503 instead of nothing
      `main.rs` runs `initialize_startup_tiers` before binding the port: with the 26B model
      pre-staged into pinned RAM that is about 3 minutes during which `/health` and `/ready` are
      unreachable, so orchestrators and `run.sh status` see a dead process. Bind first, serve
      `/health` and a 503 `/ready` while staging runs in a background task (the `ready` flag
      already exists), and refuse `/v1/*` with 503 `model_not_available` until it finishes.
      Keep graceful shutdown working mid-staging.
      Shipped: `main.rs` binds and serves first and runs `initialize_startup_tiers` as a task that
      flips `ready` (a failure there is logged and the models load on demand); a shutdown during
      staging aborts the task and releases what was loaded. `require_ready` answers 503
      `model_not_available` + `Retry-After: 5` on the inference and placement routes until then;
      `/health`, `/ready` (503), `/v1/memory/status`, `GET /v1/models`, `/v1/logs` and the stored
      picture routes stay open. Router test `tests/ready.rs`; README and CLAUDE.md updated.

- [x] ⬛ **BUILD-4** `P3` Ship docling on CUDA in the release archives and the Docker image
      ENG-3 left docling's CUDA path behind the `docling-cuda` feature. ort's prebuilt GPU build of
      ONNX Runtime 1.31 is CUDA 13 only, while llama.cpp / sd.cpp target CUDA 12.8 (driver 570+).
      Shipping it means a CUDA 13 runtime and cuDNN 9 (CUDA 13) beside the CUDA 12 one, driver
      580+, the provider libraries (`libonnxruntime_providers_cuda.so`, `_shared`) copied by
      package-release.sh, a cuDNN base image for Docker, and Kokoro / GLiNER on that ONNX Runtime.
      Decide the driver floor first (or build ONNX Runtime against CUDA 12), then measure on an
      NVIDIA GPU whether TableFormer-heavy documents gain enough to justify it.
      Shipped: measured first on an RTX 5090 laptop against its CPU
      (`docs/bench/docling-cuda-2026-10-30.md`): text pages 133 -> 49 ms, an OCR'd scan 2.1 -> 1.75 s,
      a page with two tables 0.9 -> 1.07 s (TableFormer's decoder is latency-bound), 5.7 GB VRAM
      booked. Driver floor unchanged: the provider library (90 MB) is loaded only for a docling
      model with `device = "gpu"`, which then needs driver 580+, the CUDA 13 runtime and cuDNN 9
      (verified: CPU path runs with the libraries removed; the GPU path names the missing library).
      `--features docling-cuda` in the Linux / Windows release jobs, `run.sh` / `run.bat` and the
      Dockerfile (`--build-arg DOCLING_CUDA=1` adds the CUDA 13 runtime + cuDNN to the image);
      `package-release.sh` copies `onnxruntime_providers_cuda` + `_shared` and documents the
      requirements in README.txt; CI links the feature instead of only checking it. Arena growth
      back to power-of-two under the per-session cap (1.27 -> 1.07 s on table pages). Kokoro and
      GLiNER checked on the same ONNX Runtime build. Also fixed `fetch-models.sh` skipping every
      row of a CRLF checkout of `models.manifest`.

- [x] 🟪 **API-3** `P2` Allow enabling and disabling models without removing their config
      Per-model `enabled = true|false` in `[[models]]` (default true): a disabled model keeps its
      block but is not pre-staged by `initialize_startup_tiers`, is left out of `/v1/models` and
      the Tier 1 / Tier 2 budgets, and a request for it answers 404 with a "disabled" message
      instead of loading it. Validation still checks the block (ids, engine / modality combos)
      but tolerates missing weights for a disabled model. Runtime toggle, e.g.
      `POST /v1/models/{id}/enable` and `/disable` (or a `state` field next to
      `POST /v1/memory/tier`): disabling evicts the model to Tier 3 and is refused while it is
      leased, like a manual tier move; the change is not persisted to the TOML. Show the state in
      `/v1/memory/status` with a switch per model in the playground's tier visualizer, mirror the
      flag in `default_profile`, and cover the scheduler side in the mock-backend tests.
      Shipped: `ModelConfig.enabled` (default true; validation downgrades missing files to a
      warning when false), `ManagedModelMemory.enabled` as the runtime switch with
      `ModelTierSnapshot.enabled` in `/v1/memory/status`, `HypnosError::ModelDisabled` (404
      `model_disabled`) from `acquire_model` and `set_model_tier`, startup pre-staging skips
      disabled models, `/v1/models` hides them. `ModelScheduler::set_model_enabled` behind
      `POST /v1/models/{id}/enable` / `/disable` (disable evicts to Tier 3 under the transition
      lock, 429 while leased; enable just flips the switch). Playground: on/off checkbox per model
      card, "disabled" badge, tier buttons greyed. Scheduler mock test + config test; README,
      CLAUDE.md, example profiles and backend.http updated.

- [x] ⬜ **DOC-4** `P2` Create the Hypnos website as a single page under `website/`
      One static `website/index.html` (plus its CSS, JS and copied assets; no framework or build
      step) presenting Hypnos: what it is, the 3-tier memory hierarchy with the lease-based
      scheduler, the OpenAI-compatible endpoint table and model lineup from `README.md`, the
      `assets/demo.gif` demo, quick-start (`run.sh` / `run.bat`, release archives) and links to
      the GitHub repo and docs. Modern style: generous whitespace, one display and one body
      typeface, responsive down to phone width, light and dark themes. Palette taken from
      `assets/logo.png`: deep navy (about `#1b1b4f`) as the base, the logo's vivid orange (about
      `#f26b0f`) as the single accent, white and light greys for surfaces and text; check text
      contrast on both themes. Use the logo as favicon and in the header. Keep it servable from
      GitHub Pages or any static host.
      Shipped: `website/index.html` + `styles.css` + `script.js`, no build step; assets copied to
      `website/assets/` (logo, 256 px and 64 px favicon renders, demo GIF). Sections: hero with the
      demo, measured numbers, features, tier diagram, endpoint table, model lineup with a
      single-model config, quickstart tabs (prebuilt / Docker / source), comparison, footer with
      licenses. Palette sampled from the logo: navy `#0d0f43`, orange `#fd6000`; orange text uses
      `#c24a00` (light) / `#ff8a3d` (dark) so every pair clears WCAG AA (lowest 4.5:1). Space
      Grotesk + Inter from Google Fonts; theme follows the OS with a stored toggle; checked at
      1280 px and 390 px in both themes with headless Chrome. (commit 22fc5fa)

- [x] 🟪 **API-2** `P2` Return the pictures docling extracts from a document
      `return_pictures` on `POST /v1/documents/convert` keeps the pictures the engine cuts out
      (today they are dropped, or consumed by `describe_pictures`) in a temporary store keyed by a
      document id: one JSON file per picture with the base64 image and its metadata (mime type,
      pixel size, page, bounding box, page size, caption, docling ref). The response carries the
      `id` and each page its `picture_urls`; `GET /v1/documents/{id}/pictures` lists them,
      `GET /v1/documents/{id}/pictures/{n}` returns one (JSON, or the image with `?format=image`),
      `DELETE /v1/documents/{id}` releases them. TTL and size cap in `[server]`
      (`document_pictures_dir`, `_ttl_secs`, `_max_bytes`); stale directories are swept at startup.
      Playground: "Keep pictures" checkbox and a thumbnail strip under the result.
      Shipped: `return_pictures` (JSON and multipart) keeps what `split_off_images` now extracts
      per picture (data URI, mime, pixel size, dpi, bbox, page size, caption, docling ref);
      `hypnos-server/src/pictures.rs` writes `<dir>/<doc-id>/<n>.json` records, serves
      `GET /v1/documents/{id}/pictures` (metadata), `GET .../pictures/{n}` (JSON with base64
      `image`, or the bytes with `?format=image`) and `DELETE /v1/documents/{id}`; TTL sweep every
      minute, size cap drops the oldest documents, a failed conversion releases its own. The
      response carries `id` and `pictures_expire_at`, the stream's `document.started` the `id`,
      each page its `picture_urls`. Playground checkbox + thumbnail strip + release button.
      Verified end to end on a two-page PDF (bbox, caption and page size match the source).

- [x] 🟩 **ENG-3** `P3` Run document conversion on the GPU (Core ML, CUDA)
      Follow-up to ENG-1, opt-in per model. macOS: docling's `coreml` feature (unified memory,
      nothing extra to book). CUDA: create the ORT session on load and drop it on unload, book a
      measured VRAM size and cap the arena with `gpu_mem_limit`; ships ORT as shared libraries
      plus cuDNN 9 in the CUDA archives and the Docker image, and moves Kokoro to the same ORT.
      Measure first how much of a page is model inference versus parsing and rendering.
      Shipped: measured first (M5 Max, 15-page PDF). Inference is 25-60% of a page (layout,
      TableFormer), the rest PDF rendering and image decoding. ENG-1's macOS "CPU" path was in fact
      ort's default policy putting CoreML under docling: 12 s to load, ~240 ms a page, 4.2 GB.
      docling-onnx is now vendored with `PreferCPU` and `DOCLING_RS_NO_ARENA=1` is set: 0.3 s to
      load, ~90 ms a page (release), 1.3-1.6 GB, booking cut to 1.8 GiB. `device = "gpu"` for docling:
      CoreML in the Metal build (fp32 models, `./run.sh models docling-gpu`; 2 s load, 6.4 s for
      the 15 pages, 5.2 GB: slower than the CPU, only frees its cores), CUDA behind the opt-in
      `docling-cuda` feature (`DOCLING_RS_EP=cuda`, per-arena cap from the booking,
      same_as_requested growth, provider-library re-exec before the runtime starts), compile-checked
      in CI. Not shipped in the archives / Docker image: see BUILD-4.

- [x] 🟩 **ENG-2** `P2` Add GLiNER zero-shot named-entity extraction
      Run GLiNER ONNX exports on CPU through ONNX Runtime, like Kokoro (`onnx_runtime` engine,
      `realtime_pinned`): tokenizer via the `tokenizers` crate, span decoding and thresholding
      (`gline-rs`, Apache-2.0, is a reference or a dependency). New modality
      (`entity_extraction`) and endpoint (e.g. `POST /v1/entities` with `model`, `text` or a
      list, `labels`, `threshold`) returning spans with label, offsets and score. Add a
      playground tab, the model to `models.manifest` and the example configs.
      `gliner25-rs` (Apache-2.0, crates.io, used by idu) runs GLiNER2.5 boundary checkpoints and
      accepts our ort; it uses `tokenizers` 0.23 (docling 0.20): check both build together.
      Shipped: `gliner` engine (`hypnos-backends/src/entities.rs`) on `gliner25-rs` 0.5.6, vendored
      with `vendor/patches/gliner25-rs.patch`: ported to ort rc.15 (shared environment,
      `AllocationDevice<'static>`), ort features cut to the static CPU build (upstream turned on
      `load-dynamic` and every GPU provider, which would have made the whole binary load ONNX
      Runtime dynamically), `PreferCPU` (macOS CoreML took 7 GB and 5 s to load the encoder; now
      3 GB and 0.7 s). `POST /v1/entities` (`input` text or list, `labels`, `threshold`, `flat`)
      returns character offsets; more than 8 labels run as families, long texts in windows (1,700
      words, 11 labels: 1.5 s). Model `gliner2.5-multi` (fp32, `device = "cpu"`, ~3.2 GiB booked,
      on demand rather than pinned: the encoder is 1.1 GB). Playground tab 7, smoke test step 10.

- [x] 🟩 **ENG-1** `P2` Add document conversion with docling.rs (PDF / images to Markdown)
      Use the `docling` crate (docling-project's Rust port, MIT; ~2.55, all `docling-*` crates on
      one version) on CPU: layout + TableFormer (+ OCR, formulas) in ONNX, pure-Rust PDF parsing
      and rendering (lopdf, tiny-skia; no pdfium). It builds against our `ort =2.0.0-rc.15`.
      Convert one page per pass (`page_range(p, p)`) with a warm pipeline: a 6-page PDF peaks at
      12.7 GB in one call but 1.8-2.6 GB per page, so the model gets a fixed Tier 2 booking.
      Endpoint `POST /v1/documents/convert` (multipart file or URL, `pages`, `ocr`,
      `output_format` markdown | json) streams each page; stitch across pages (tables and lists
      that continue, repeated headers / footers, heading levels). New modality
      `document_conversion`, `onnx_runtime` engine, config entry with a model directory, the
      models in `models.manifest`. Port from idu `rust/src/docling_convert.rs`: OCR retry for
      PDFs with no text layer, `/aNNN` glyph-name cleanup, optional figure / table descriptions
      through a loaded vision model. First measure warm per-page time and the int8 layout model.
      Shipped: `docling` engine (`hypnos-backends/src/document.rs`, docling 2.55.4) and
      `POST /v1/documents/convert` (multipart or JSON with base64 / data URI / `url`; `pages`, `ocr`,
      `output_format` markdown | json, `stream` = one SSE event per page, `describe_pictures`).
      New `device = "cpu"`: Tier 1 books host RAM only and a CPU model short of RAM moves idle
      RAM-only models to Tier 3. Measured on an M5 Max: models load in ~12 s (with the tier switch,
      warmed by a blank image), ~0.24 s per page after that, TableFormer loads on the first table;
      footprint 4.2 GB against a 4.1 GB booking, 1.0 GB after eviction. `join_pages` mends tables,
      numbered lists and cut sentences; OCR is docling's own (pages without a text layer only).
      All ONNX engines now share docling-onnx's environment: ort rc.15 allows one per process, so
      Kokoro could not reload after an eviction before. `models.manifest` rows may name an https
      base URL (no HF token sent there). Playground tab 6 "Documents", smoke test step 9, example
      configs and default profile, notices (docling MIT, models Apache-2.0 / CDLA, MPL-2.0 crates).

- [x] ⬛ **BUILD-2** `P2` Provide a Docker image with CUDA support
      Multi-stage Dockerfile (build on a CUDA devel image, run on a CUDA runtime image), models
      mounted as a volume, config via mount or env. Publish to GHCR from the release workflow
      (BUILD-1) and add a `docker run --gpus all ...` quickstart to the README. Same GPL
      obligations as BUILD-1 for the image.
      Shipped: three-stage `Dockerfile` (node frontend build; `nvidia/cuda:12.8.1-devel-ubuntu24.04`
      runs vendor.sh, the release build and `package-release.sh`; the `-runtime-ubuntu24.04` image
      holds the archive at `/opt/hypnos`, listens on 0.0.0.0, `VOLUME` models, `HEALTHCHECK`
      /health). `HYPNOS_CONFIG` selects the config when no path is passed. `release.yml` builds it
      on every run and pushes `ghcr.io/advancedlogic/hypnos` (semver, `latest`, sha) on tags; the
      README has the `docker run --gpus all` quickstart. `JOBS=2` / `-j 2` keep nvcc within 16 GB
      (cargo -j is also each engine's CMake --parallel). Ubuntu 24.04 because ort's prebuilt ONNX
      Runtime needs glibc 2.38+ and GCC 14's libstdc++ (this also moved the Linux release archive
      to 24.04). Verified with a local arm64 build: links, serves /health, the playground and an
      embeddings request (CPU, stub libcuda); not yet run on an NVIDIA host.
- [x] ⬛ **BUILD-1** `P1` Publish prebuilt release binaries
      The 10-20 min native build plus vendoring, CMake/Ninja and the CUDA host-compiler pin stop
      most people. Release workflow producing archives for Windows x64 (CUDA), Linux x64 (CUDA)
      and macOS arm64 (Metal), each with the sd.cpp shared library, `espeak-ng-data`, the
      example config and the built frontend. Document the CUDA runtime / driver requirement.
      Archives are GPL-3.0-or-later as a whole (espeak-ng, see `THIRD-PARTY-NOTICES.md`): ship
      `LICENSE-*`, the GPL-3.0 text and the notices, and link the tagged source.
      Shipped: `release.yml` (tag `v*` publishes, manual run only builds) for linux-x64-cuda
      (ubuntu-24.04 + NVIDIA apt CUDA 12.8), windows-x64-cuda (Jimver CUDA 12.8, plus a separate
      cudart/cuBLAS DLL zip) and macos-arm64-metal, packaged by `backend/scripts/package-release.sh`
      with SHA256SUMS. Archives run in place: hypnos serves the bundled playground at `/`
      (`server.frontend_dir` or `frontend/` beside the binary), espeak-ng-data is found beside the
      executable (it used the build machine's path), Linux gets a `$ORIGIN` rpath, and
      `fetch-models` resolves the manifest relative to the script. GPL text in `licenses/`.
      macOS archive verified locally (playground, sd.cpp dylib, TTS); the CUDA jobs only run on a tag.
- [x] ⬛ **BUILD-3** `P2` Re-enable CI
      `.github/workflows/ci.yml` is fully commented out. Restore the engine-free jobs: CPU-only
      `cargo build --workspace --no-default-features`, `cargo test --workspace`, scoped `cargo fmt
      --check` and clippy, plus `npm run build` for the frontend. Add a macOS Metal build job if
      runner time allows.
      Shipped: `ci.yml` on push to main / PRs / manual: `lint` (scoped fmt + clippy), `build-test`
      matrix (Linux CPU, Windows CPU, macOS 15 Metal; build all targets + test) and `frontend`
      (`npm ci && npm run build`). The repo is public, so the macOS minutes are free.
- [x] ⬜ **DOC-3** `P1` Record a demo GIF of the 3-tier visualizer for the README
      30-60 s screen recording of the playground while chat, speech and image requests move
      models between tiers (evictions and promotions visible). Keep it under ~5 MB, save under
      `assets/`, and replace the `DOC-3` placeholder comment at the top of `README.md`.
      Shipped: `assets/demo.gif` (46 s, 0.76 MB), recorded headless with Playwright from timed
      screenshots against a demo server on an M5 Max with `simulate_physical_memory_bytes` = 32 GB,
      so the 26B chat model and MOSS-Image evict each other. README placeholder replaced.
      Re-recorded 2026-10-30 on the RTX 5090 laptop with the documents (pictures kept) and
      entities tabs in the sequence; the recorder now lives in `tools/demo-gif/` (Playwright +
      gifenc, 1.66 MB, 45 s playback of 95 s real time).
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
      chart in the README. CUDA run 2026-10-31 (`docs/bench/cuda-2026-10-31.md`: Tier 3 4.28 s,
      Tier 2 3.69 s, Tier 1 131 ms; no page-cache drop on Windows) drawn next to the Metal run in
      one chart (`--chart-compare`, `docs/bench/swap-2026-10-31.svg`).
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
