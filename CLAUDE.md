# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Hypnos is

An OpenAI-compatible multi-modal inference server in Rust (`backend/`) plus a Svelte 5 + Tailwind
playground (`frontend/`) and a Tauri 2 desktop chat client (`desktop/`). All engines (llama.cpp + libmtmd, stable-diffusion.cpp, ONNX Runtime +
espeak-ng, docling.rs) are compiled and linked **in-process**; there is no external runtime. The core idea is a
3-tier memory hierarchy (Tier 1 GPU VRAM, Tier 2 pinned host RAM, Tier 3 memory-mapped NVMe) with a
lease-based scheduler that promotes/evicts models on demand. See [README.md](README.md) for the
model lineup, endpoint table and production behaviour.

## Commands

All cargo commands run from `backend/`. Paths in `hypnos.example.toml` are relative to `backend/`.

```bash
# One-time machine setup (writes git-ignored <repo>/.cargo/config.toml: GPU arch, CUDA host cl.exe, rustc pin)
powershell -ExecutionPolicy Bypass -File backend\scripts\setup-build-env.ps1   # Windows
./backend/scripts/setup-build-env.sh                                            # Linux/macOS

# Recreate backend/vendor/ (git-ignored engine sources, >1 GB) from pinned commits + patches
./backend/scripts/vendor.sh            # add --force to wipe and re-clone

# Build + run (process manager; build and start are deliberately separate, start never compiles)
./run.sh build [backend|frontend]      # cargo build --release --bin hypnos + npm install
./run.sh [start|stop|restart|status|logs] [backend|frontend|all]
HYPNOS_PROFILE=debug ./run.sh build backend   # dev profile instead of release
run.bat <same actions>                 # Windows; auto-loads vcvars64 when rc.exe is missing

# Manual
cd backend && cargo run --bin hypnos -- hypnos.example.toml     # port 9999
cd frontend && npm run dev                                       # port 5173, proxies /v1 and /health to 9999

# Tests (engine-free: mock backend, no weights needed)
cd backend && cargo test --workspace
cd backend && cargo test -p hypnos-scheduler --test scheduler            # scheduler integration tests
cd backend && cargo test -p hypnos-scheduler --test scheduler <name>     # one test by name
cd backend && cargo test -- --ignored                                     # GPU lifecycle test, needs real weights in backend/models/
./backend/scripts/smoke_test.sh [http://localhost:9999]                   # one real request per modality against a running server
./backend/scripts/bench.sh [--host ...] [--out docs/bench/baseline-YYYY-MM-DD.md] [--scenarios 1,2,3]

# CPU-only build (no CUDA toolkit, what CI did)
cd backend && cargo build --workspace --no-default-features

# Apple Silicon (Metal); ./run.sh build does this automatically on Darwin/arm64
cd backend && cargo build --release -p hypnos-server --no-default-features --features metal

# Desktop app (Tauri 2; not part of `all`; needs the Tauri prerequisites, see desktop/README.md)
./run.sh build desktop                 # npm install + tauri build --no-bundle (an .app bundle on macOS)
./run.sh [start|stop|status|logs] desktop
cd desktop && npx tauri dev            # development: Vite on 1420 + app window, hot reload
cd desktop/src-tauri && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo test

# Model files (~31 GB, listed in backend/models.manifest) into backend/models/
./run.sh models [model-id]             # run.bat models on Windows

# Release archive from an existing release build + `npm run build` (what release.yml runs per platform)
./backend/scripts/package-release.sh macos-arm64-metal --version v0.1.0 --out dist

# Lint (scope to workspace crates; --all would format the vendored bindings)
cd backend && cargo fmt -p hypnos-core -p hypnos-memory -p hypnos-scheduler -p hypnos-backends -p hypnos-server -- --check
cd backend && cargo clippy -p hypnos-core -p hypnos-memory -p hypnos-scheduler --all-targets -- -D warnings
```

Build notes:
- Rust 1.92+ via rustup. Both run scripts and `.cargo/config.toml` pin the rustup toolchain because
  a standalone Rust install may shadow it on PATH.
- Both `backend/Cargo.toml` and `desktop/src-tauri/Cargo.toml` set
  `[profile.release.build-override] strip = false` (BUG-5): Xcode 27.2's `strip` can leave a
  proc-macro dylib with a misaligned LINKEDIT string pool (it depends on the embedded paths'
  length), dyld refuses it and rustc reports E0463 "can't find crate". Keep it until that is fixed.
- The first backend build compiles every native engine (10-20 min). CMake generator is forced to
  Ninja; the Visual Studio generator ignores `CUDAHOSTCXX`.
- `.github/workflows/ci.yml` runs fmt + clippy, the CPU (Linux, Windows) and Metal (macOS) builds
  and tests, the SDK tests, the desktop app build (three OSes) and the frontend build only for a release: `release.yml` calls it on a `v*` tag and
  the Docker push and the GitHub release wait for it. Pushes and PRs start nothing; run it by
  hand (Actions > ci > Run workflow) before tagging, and run fmt + clippy + tests locally.
  Clippy runs on a pinned Rust (1.98.0 in the lint job; the builds use current stable): bump it
  on purpose, with the fixes the new lints ask for.
- `.github/workflows/release.yml` builds the CUDA (Linux, Windows) and Metal archives (through
  `build-server.yml`, called twice: `build` for CUDA, `build-macos` for Metal, which the desktop
  job waits for because the macOS app bundles that server) and the
  desktop installers (.dmg, .msi/.exe, .deb/.AppImage) on a `v*` tag and publishes them as a
  GitHub release; a manual run only builds them as artifacts. With the `APPLE_*` secrets the
  macOS app and server (`scripts/sign-macos.sh`) get a Developer ID signature and notarization,
  without them an ad-hoc signature; Windows signs with `WINDOWS_CERTIFICATE`. `docs/signing.md`
  is the signing guide; the app's hardened-runtime entitlements are in
  `desktop/src-tauri/Entitlements.plist`.
- `backend/backend.http` has ready-made requests for the VS Code REST Client extension.

## Architecture

### Crate graph (`backend/crates/`)

```
hypnos-core      traits + schemas, no engine deps   (ModelBackend, GpuMemoryProbe, config, kv_cache, openai, request)
   ^
hypnos-memory    TieredMemoryManager, mmap weights, device detection, pinned RAM (VirtualLock/mlock)
   ^
hypnos-scheduler ModelScheduler + ActiveModelLease   (promotion, partial offload, eviction, admission)
   ^
hypnos-backends  engines implementing ModelBackend: llama.rs, diffusion.rs, kokoro.rs (+ media.rs, gguf_inspector.rs)
   ^
hypnos-server    axum router, SSE streaming, metrics, built-in default profile, `hypnos` binary
```

`hypnos-backends` has the GPU features `cuda` and `metal`; `hypnos-server` forwards both and turns
`cuda` on by default (the workspace dependency on `hypnos-backends` has default features off). On a
Mac always build with `--no-default-features --features metal`. `hypnos-core`, `-memory`,
`-scheduler` never link an engine, which is why their tests run anywhere and clippy is enforced only
on them.

### Request lifecycle

0. `main.rs` binds the port before `initialize_startup_tiers`, which runs as a background task
   and flips `AppState::ready`; until then `require_ready` answers 503 `model_not_available` on
   the inference and placement routes (`/health`, `/ready` (503), `/v1/memory/status`,
   `GET /v1/models` and `/v1/logs` stay open). With `server.api_keys` (or `HYPNOS_API_KEYS`) set, `auth::require_api_key` answers 401 for any
   `/v1/*` request without a matching `Authorization: Bearer` key, then applies the key's limits
   from `quota.rs` (requests per minute, concurrency held through the response body, tokens per
   UTC day: 429 with `Retry-After`; the key's label travels as `ApiKeyLabel` on the request and
   the response, into the request log and `/metrics`; `server.request_log_redact` is applied in
   `RequestLog`) (it runs inside CORS and
   `observe_request`; `/health`, `/ready` and, by default, `/metrics` stay open). `[server.tls]`
   makes `main.rs` serve HTTPS through axum-server + rustls (ring provider).
1. A `hypnos-server` handler builds a `RequestContext` (limits, cancellation token tied to client
   disconnect via `CancelOnDrop`, request id).
2. It calls `ModelScheduler::acquire_model(model_id, requested_ctx_len, batch, ctx)`:
   - per-model `Semaphore` admission (`limits.max_queue_per_model`); rejection is `HypnosError::Busy` -> HTTP 429;
   - fast path if the model is already in Tier 1 with a large enough `allocated_context_len`;
   - otherwise it takes `transition_lock`, asks `TieredMemoryManager` for a `LayerPlacementPlan`
     (full or partial GPU layers + KV reservation from `KvCacheEstimator`), evicts idle
     lower-priority models, calls `ModelBackend::apply_placement` to actually load/unload, and
     reconciles bookings against real free VRAM via `GpuMemoryProbe` (`LlamaGpuProbe`).
3. It returns an `ActiveModelLease` (RAII). While held, the model cannot be evicted; dropping it
   decrements the lease count. Streaming responses keep the lease alive until the stream ends.
4. The handler invokes the backend method (`chat_completion[_stream]`, embeddings, transcription,
   speech, image generation). Engines must poll `ctx.should_continue()` and stop at the next
   token/chunk/step on cancellation.
5. `ExecutionTelemetry` (previous tier, placement, tier-switch latency, evicted models, engine
   timings) is attached to responses and fed into `metrics.rs`.

### Placement model

`LayerPlacementPlan` (`hypnos-core/src/kv_cache.rs`) is the single shared description of where a
model lives: `gpu_layers` / `pinned_ram_layers` / `ssd_layers`, VRAM weight bytes, VRAM KV +
workspace bytes, pinned bytes, `allocated_context_len`. The memory manager computes it, the
scheduler applies it, the backend materialises it. Models carry a priority (`RealtimePinned`,
`High`, `Low`) and an `initial_tier` that drive startup pre-staging (`initialize_startup_tiers`)
and eviction order. `POST /v1/memory/tier` moves a model manually and is refused while it is leased.

### Apple Silicon

`docs/apple-silicon.md` is the design and status record (`docs/operating.md` is the operator's
guide: sizing, knobs per symptom, telemetry, errors, service recipes; keep it in step with new
knobs, metrics and error types; `docs/api.md` is the HTTP API reference for client authors:
keep it in step with new routes, fields, error types and stream events). On unified memory Tier 1 and Tier 2 share
one physical pool: Tier 1 is Metal's working set (adopted from the device once llama.cpp starts),
Tier 2 is what RAM leaves after it and a 4 GiB OS reserve, and placement is all-or-nothing (no
partial GPU splits). `backend/hypnos.macos.example.toml` is the macOS profile. `[hardware]` also
offers `simulate_physical_memory_bytes` (derive budgets as on a smaller machine) and
`max_process_memory_bytes` (soft cap checked before each load; evictions are listed with their
reason in `/v1/memory/status`).

### Configuration

`HypnosConfig` (`hypnos-core/src/config.rs`): `[server]`, `[limits]`, `[hardware]`, `[[models]]` with
nested `[models.arch]`. `server.models_dir` (or `HYPNOS_MODELS_DIR`) is the folder relative model
paths resolve against: `main.rs` calls `apply_models_dir` before the GGUF detection and
`validate`, which reports a missing folder once (`models_dir not found`); the example configs
and the default profile use `"models"` with bare file names (the manifest's `local` column). `config_version` (absent = 1, `CONFIG_VERSION` in `config.rs`): a newer file is refused by
`from_toml_str`, an older one goes through `HypnosConfig::migrate` (add a step per bump); keys no
field consumes are collected in `unknown_keys` and reported as warnings. `hypnos --check-config
[file]` runs detection and validation and exits. `HypnosConfig::validate` runs at startup and aborts on any error (missing
weights, duplicate ids, impossible budgets, bad engine/modality combos; a model with
`enabled = false` may have missing weights). A disabled model keeps its block but is skipped by
`initialize_startup_tiers`, refused by `acquire_model` with `ModelDisabled` (404) and hidden from
`/v1/models`; `POST /v1/models/{id}/enable|disable` flips the runtime switch held by
`TieredMemoryManager` (disable evicts to Tier 3 first). `hardware.kv_park` / `kv_park_bytes` (off by default; `POST
/v1/memory/kv-park` at runtime) is MEM-4: `hypnos-backends/src/kv_park.rs` holds a shared
`KvParkPool` (in `EngineOptions` and `AppState`); before a llama worker is dropped
(`apply_placement`: demotion or a new layer split) a `LlamaJob::Park` copies each used idle slot's
`state_seq_get` plus its keys and SWA checkpoint into the pool, and `spawn_worker` restores them
right after the context exists when the `ParkSignature` (weights, per-slot `n_ctx`, slots, KV type)
matches; `tier_switch_phases` reports `kv_restore_ms` / `kv_restored_tokens`. `moe_offload`
(`off` / `auto` / `on`, per model; `POST /v1/models/{id}/moe-offload` at runtime, which unloads an
idle resident model so the next load uses it) is MEM-5: `gguf.rs` sums the `*_exps` tensor bytes
into `arch.expert_bytes`, `compute_optimal_placement` tries full-GPU placements with the experts of
the last `n` layers in host memory (`LayerPlacementPlan::cpu_expert_layers`; `on` = all, `auto` =
the smallest `n` that fits, discrete GPUs only) before a layer split, and `load_model` adds the
`cpu_expert_pattern` tensor override (it includes Gemma 4's fused `ffn_gate_up_exps`, which the
bindings' `add_cpu_moe_override` misses). If the config path does not
exist, the server falls back to `default_profile::default_six_model_config()` in `hypnos-server`;
keep that profile in sync with `hypnos.example.toml` when adding models or fields.

### Engines (`hypnos-backends`)

- `llama.rs`: `ggml_llama`. Text, vision, audio-in (via mtmd), embeddings, tool calling (`tools.rs`:
  Gemma 4's native format, rendered by `gemma4_prompt` as its GGUF template does; with tools the
  engine passes `<|tool_call>` / `<tool_call|>` / `<|"|>` through and stops at `<|tool_response>`,
  and a `CallParser` turns the raw text into OpenAI `tool_calls`; `tool_choice` required / named
  is forced by prefilling the call's opening; other architectures refuse `tools` with 400). `engine.rs` /
  `worker.rs`: one worker thread per resident model owns a persistent context with `slots`
  execution slots (KV prefix reuse, continuous batching); `docs/execution-efficiency-plan.md`
  is the reference for that area. Loading (PERF-6): `models.use_mmap` unset means the weights
  are read through pinned staging buffers with asynchronous uploads on CUDA (llama.cpp's
  `LLAMA_LOAD_MODE_NONE`, 2.4x faster than mapping on Windows) and mapped on Metal / CPU
  placements; each Tier 1 load logs its phases and reports them as
  `hypnos_telemetry.tier_switch_phases` (`ModelBackend::last_load_phases`). Tier 2 pins the
  projector and text-encoder files with the weights (`ManagedModelMemory::aux_buffers`), so the
  pinned budget must cover `total_weight_bytes` as a whole; the manager warns when the budget
  exceeds half of physical RAM. Measurements: `docs/bench/swap-cuda-2026-10-30.md`.
- `diffusion.rs`: `ggml_diffusion` (stable-diffusion.cpp, MOSS-Image). sd.cpp is built as a
  **shared** library so its ggml does not clash with llama.cpp's; the vendored build script copies
  the DLL/.so next to the binary.
- `kokoro.rs`: `onnx_runtime` TTS on CPU (`AutoDevicePolicy::PreferCPU`) with espeak-ng G2P. Use the
  fp16 ONNX graph on Windows/Linux (the quantized exports crash ORT session creation on Windows) and
  the fp32 `model.onnx` on Apple Silicon, where fp16 overflows to an all-NaN (silent) waveform for
  some inputs; a non-finite waveform is returned as an error. The crate's `build.rs` copies
  `espeak-ng-data` next to the binary.
- `document.rs`: `docling` (docling.rs crates, every `docling-*` pinned to one version).
  `device = "cpu"`: the planner books its Tier 1 placement in host RAM only (`runs_on_cpu`), and
  a CPU model short of RAM pushes idle RAM-only models to Tier 3. PDFs go one page per conversion
  through a `WarmPipeline` (a whole document at once keeps every page's data until the end);
  `join_pages` mends tables, numbered lists and sentences across breaks. `split_off_images` strips
  the base64 images from the docling JSON and keeps each picture with its metadata (mime, size,
  bbox, caption) only for `describe_pictures` / `return_pictures`; `hypnos-server/src/pictures.rs`
  is the on-disk store behind `return_pictures` (`GET /v1/documents/{id}/pictures[/{n}]`,
  `DELETE /v1/documents/{id}`, TTL and size cap in `[server]`). docling reads its setup
  from the environment once per process (`configure_docling`: `DOCLING_RS_MODELS_DIR` from
  `weights_path`'s folder, `DOCLING_RS_EP` from `device`, `DOCLING_RS_NO_ARENA=1`), hence one
  docling model per server. `device = "gpu"`: CoreML in the Metal build, CUDA with the
  `docling-cuda` feature, which the run scripts, the release archives and the Docker image build
  (ort's prebuilt ORT GPU = CUDA 13 + cuDNN 9, loaded only for that device; the provider library
  ships beside the binary, `main` calls `prepare_process` before the runtime for its
  provider-library re-exec on Linux; measurements in `docs/bench/docling-cuda-2026-10-30.md`).
  `docling-onnx` is vendored with `PreferCPU`.
- `entities.rs`: `gliner` (vendored `gliner25-rs`, `device = "cpu"`). One worker thread owns the
  `BoundaryEngine` from load to unload; more than 8 labels run as families, long texts in windows.
- `onnx.rs`: ort rc.15 allows one ONNX Runtime environment per process, never rebuilt once
  dropped. Every ONNX engine takes `docling_onnx::environment()`; never call `ort::init().build()`.
  ort's default device policy registers CoreML on macOS: CPU engines set `PreferCPU` (Kokoro,
  gliner; GLiNER took 7 GB and 5 s to load without it, 3 GB and 0.7 s with it).

### Vendored engines (`backend/vendor/`)

Git-ignored except `patches/`. `scripts/vendor.sh` is the source of truth for pins (llama.cpp
b11982, sd.cpp master-1024, llama-cpp-rs 0.1.160, diffusion-rs commit, espeak-rs-sys 0.2.0,
gliner25-rs 0.5.6, docling-onnx 2.55.4) and
`vendor/VENDOR.md` explains each patch and how to bump a pin. Never edit vendor sources without
regenerating the corresponding patch file.

### Client SDKs (`sdk/`)

`sdk/python`, `sdk/javascript`, `sdk/rust`: one client per language for every route of
`docs/api.md`, with the same method names, error classes, retry policy (section 17) and
per-model concurrency bound. Each has its own package manifest and is not part of the backend
workspace. Their tests run against `backend/crates/hypnos-server/examples/mock_server.rs`: the
real router, scheduler, auth and request log in front of mock engines for the default lineup,
with `/mock/fail` (inject an engine error), `/mock/ready` and `/mock/reset`. A change to the
API means a change to `docs/api.md`, the mock server and the three SDKs. In Python, edit
`_async_client.py` and regenerate the sync client with `scripts/unasync.py`.

### Desktop app (`desktop/`)

Tauri 2 + Svelte 5 + Tailwind 3 (same versions as `frontend/`), its own `package.json` and a
`src-tauri/` crate outside the backend workspace. `src-tauri/src/local_server.rs` can start a
`hypnos` (detected or chosen), stream its output to the UI and stop it on quit, with health
checks through `hypnos-client` (`sdk/rust`, a path dependency). The macOS app bundles the
Metal server (DESK-16): `desktop/scripts/stage-server.sh` stages it in the git-ignored
`src-tauri/server/` and `tauri.server-macos.conf.json` (merged with `--config` only by
`run.sh build desktop`, release.yml and by hand, so dev builds and clippy need no server) puts
it in `Contents/MacOS` (sidecar), `Contents/Frameworks` (dylib, found through the server's
`@executable_path/../Frameworks` rpath) and `Contents/Resources` (espeak data and playground,
found by the server through `hypnos_core::config::beside_executable`). The bundled server runs
with a config the app writes to `~/Library/Application Support/Hypnos/`; `sync_models`
disables models with missing files before each start and enables them again later (it resolves
paths through `[server] models_dir` and changes nothing when that folder is missing).
`src-tauri/src/models.rs` + `Models.svelte` (DESK-17) download the manifest's models (sizes and
SHA-256 from the Hugging Face tree API and the GitHub release API, `.part` + `Range` resume,
`models_set_dir` writes `models_dir` into the app's own config and moves the files); the
bundle carries `Resources/server/models.manifest`. Ad-hoc
builds have no hardened runtime (`hardenedRuntime: false`; its library validation refuses an
ad-hoc dylib); a Developer ID build turns it on. Every server call is a command in `src-tauri/src/hypnos.rs` on `hypnos-client`
(chat tokens over a `tauri::ipc::Channel`, cancel by request id); the webview has no HTTP
permission, and an SDK API change can break the app. `route` picks the model per modality from
`/v1/memory/status` `modalities` (chat model, then the settings' choice, then a Tier 1 model),
never hard-coded ids. `attachments.rs` prepares files when they are attached (image ->
1024 px JPEG data URL, documents -> docling markdown, audio -> symphonia decode + transcription,
video -> soundtrack transcript, frames sampled in the webview through the asset protocol,
allowed per file); `hypnos_chat` turns them into text and `image_url` parts and switches to a
vision model when the new message has images. `web.rs` fetches URLs on the Rust side
(`reqwest`, http(s), 50 MiB, 30 s) and converts them with docling into a `web` attachment;
`save_text` / `url_download` write to a path from the save dialog. `mic.rs` records with `cpal`
on a thread of its own (CoreAudio streams are not `Send`), emits `mic-level`, and transcribes on
stop. `tts.rs` turns an answer into speakable sentences (`speech_text`, `sentences`), synthesizes
them in order and plays them on a rodio player thread; `speak` takes the text so far per call.
The request log view (`Logs.svelte`) uses `hypnos_logs` / `hypnos_log` / `hypnos_set_log_level` /
`hypnos_clear_logs`; chat `done` events carry the answer's `x-request-id` for its "Log" link.
`tray.rs` is the tray icon (status dot drawn on the logo, menu events `tray-server` /
`tray-open-settings` handled by the webview, close-to-tray in `on_window_event`, macOS Dock
visibility); `local_server_stop_port` stops any local process named `hypnos` on the connection's
port, so the tray can stop a server `run.sh` started. `images.rs` is the image tool: `generate_image` goes to the server as a real tool
(`image_tool_spec`, API-9), its call becomes an image, an `image` event and a `tool` result the
model comments on (second round with `tool_choice: none`); when the server refuses tools for the
model (400, no native format) `hypnos_chat` falls back to the prompt convention
(`TOOL_INSTRUCTION` plus a `ToolGate` that holds back an answer's start). The real-server tests are `#[ignore]`d (`HYPNOS_TEST_URL`;
`HYPNOS_TEST_BINARY` / `HYPNOS_TEST_CONFIG`). `get_api_key` / `set_api_key` keep the API key in
the OS keychain (`keyring`), keyed by origin; chats and the server URL live in `tauri-plugin-store`. `ChatStore`
(`src/lib/chats.svelte.js`) owns chats and streaming; `StatsPanel.svelte` polls
`/v1/memory/status` only while open. A change to `/v1/memory/status`, telemetry or errors in
`docs/api.md` may need a change here too. `desktop/README.md` is its guide.

### Frontend

Single-file app: `frontend/src/App.svelte` (Svelte 5 runes) renders the live 3-tier visualizer from
`/v1/memory/status` and a tabbed playground for each modality. The Vite dev server proxies `/v1`
and `/health` to the backend, so the UI uses relative URLs.

## Task tracking

`TODO.md` is a card board with prefixed IDs (`UI-n`, `API-n`, `ENG-n`, `MEM-n`, `PERF-n`, `BUG-n`,
`BUILD-n`, `DOC-n`, `TEST-n`) and `P0`-`P3` priorities; its Conventions section defines the card
format. "Add a card to TODO" appends to Backlog with the next free ID; "implement XY-n" moves the
card through In Progress to Done.
