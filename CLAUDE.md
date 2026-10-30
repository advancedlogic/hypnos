# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What Hypnos is

An OpenAI-compatible multi-modal inference server in Rust (`backend/`) plus a Svelte 5 + Tailwind
playground (`frontend/`). All engines (llama.cpp + libmtmd, stable-diffusion.cpp, ONNX Runtime +
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
cd backend && cargo run --bin hypnos -- hypnos.example.toml     # port 8080
cd frontend && npm run dev                                       # port 5173, proxies /v1 and /health to 8080

# Tests (engine-free: mock backend, no weights needed)
cd backend && cargo test --workspace
cd backend && cargo test -p hypnos-scheduler --test scheduler            # scheduler integration tests
cd backend && cargo test -p hypnos-scheduler --test scheduler <name>     # one test by name
cd backend && cargo test -- --ignored                                     # GPU lifecycle test, needs real weights in backend/models/
./backend/scripts/smoke_test.sh [http://localhost:8080]                   # one real request per modality against a running server
./backend/scripts/bench.sh [--host ...] [--out docs/bench/baseline-YYYY-MM-DD.md] [--scenarios 1,2,3]

# CPU-only build (no CUDA toolkit, what CI did)
cd backend && cargo build --workspace --no-default-features

# Apple Silicon (Metal); ./run.sh build does this automatically on Darwin/arm64
cd backend && cargo build --release -p hypnos-server --no-default-features --features metal

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
- The first backend build compiles every native engine (10-20 min). CMake generator is forced to
  Ninja; the Visual Studio generator ignores `CUDAHOSTCXX`.
- `.github/workflows/ci.yml` runs fmt + clippy, the CPU (Linux, Windows) and Metal (macOS) builds
  and tests, and the frontend build on every push to main and every PR.
- `.github/workflows/release.yml` builds the CUDA (Linux, Windows) and Metal archives on a `v*` tag
  and publishes them as a GitHub release; a manual run only builds them as artifacts.
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

0. With `server.api_keys` (or `HYPNOS_API_KEYS`) set, `auth::require_api_key` answers 401 for any
   `/v1/*` request without a matching `Authorization: Bearer` key (it runs inside CORS and
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

`docs/apple-silicon.md` is the design and status record. On unified memory Tier 1 and Tier 2 share
one physical pool: Tier 1 is Metal's working set (adopted from the device once llama.cpp starts),
Tier 2 is what RAM leaves after it and a 4 GiB OS reserve, and placement is all-or-nothing (no
partial GPU splits). `backend/hypnos.macos.example.toml` is the macOS profile. `[hardware]` also
offers `simulate_physical_memory_bytes` (derive budgets as on a smaller machine) and
`max_process_memory_bytes` (soft cap checked before each load; evictions are listed with their
reason in `/v1/memory/status`).

### Configuration

`HypnosConfig` (`hypnos-core/src/config.rs`): `[server]`, `[limits]`, `[hardware]`, `[[models]]` with
nested `[models.arch]`. `HypnosConfig::validate` runs at startup and aborts on any error (missing
weights, duplicate ids, impossible budgets, bad engine/modality combos; a model with
`enabled = false` may have missing weights). A disabled model keeps its block but is skipped by
`initialize_startup_tiers`, refused by `acquire_model` with `ModelDisabled` (404) and hidden from
`/v1/models`; `POST /v1/models/{id}/enable|disable` flips the runtime switch held by
`TieredMemoryManager` (disable evicts to Tier 3 first). If the config path does not
exist, the server falls back to `default_profile::default_six_model_config()` in `hypnos-server`;
keep that profile in sync with `hypnos.example.toml` when adding models or fields.

### Engines (`hypnos-backends`)

- `llama.rs`: `ggml_llama`. Text, vision, audio-in (via mtmd), embeddings. Currently one
  `LlamaContext` per request behind a mutex; `docs/execution-efficiency-plan.md` is the plan to
  move to persistent, batched contexts and is the reference for any work in that area.
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
  docling model per server. `device = "gpu"`: CoreML in the Metal build, CUDA only with the opt-in
  `docling-cuda` feature (prebuilt ORT GPU = CUDA 13; `main` calls `prepare_process` before the
  runtime for its provider-library re-exec). `docling-onnx` is vendored with `PreferCPU`.
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

### Frontend

Single-file app: `frontend/src/App.svelte` (Svelte 5 runes) renders the live 3-tier visualizer from
`/v1/memory/status` and a tabbed playground for each modality. The Vite dev server proxies `/v1`
and `/health` to the backend, so the UI uses relative URLs.

## Task tracking

`TODO.md` is a card board with prefixed IDs (`UI-n`, `API-n`, `ENG-n`, `MEM-n`, `PERF-n`, `BUG-n`,
`BUILD-n`, `DOC-n`, `TEST-n`) and `P0`-`P3` priorities; its Conventions section defines the card
format. "Add a card to TODO" appends to Backlog with the next free ID; "implement XY-n" moves the
card through In Progress to Done.
