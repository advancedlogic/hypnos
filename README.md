<p align="center">
  <img src="assets/logo.png" alt="Hypnos logo" width="200">
</p>

<h1 align="center">Hypnos</h1>

**Hypnos** is an OpenAI-compatible multi-modal Rust inference server (`backend/`) and Svelte 5 + Tailwind CSS interactive playground (`frontend/`) that dynamically manages model weights and KV caches across a **3-tier memory hierarchy**:

1. **Tier 1 — Hot (GPU VRAM / Wired Unified Memory):** Active model layers + dynamically budgeted KV cache + activation scratchpad.
2. **Tier 2 — Warm (Pinned Host RAM / `VirtualLock` / `mlock`):** Page-locked system RAM for ultra-fast PCIe DMA promotion (~25–55 GB/s on CUDA/ROCm) or zero-copy residency on Apple Silicon Unified Memory.
3. **Tier 3 — Cold (NVMe SSD via `memmap2`):** Memory-mapped `.gguf`, `.safetensors`, and `.onnx` model repositories.

---

## Repository Structure

```text
hypnos/
├── backend/                     # Rust Workspace (Axum OpenAI Server + 3-Tier Memory Scheduler)
│   ├── Cargo.toml
│   ├── hypnos.example.toml
│   └── crates/
│       ├── hypnos-core/         # Config, KV-Cache + Gemma 4 SWA Estimator, OpenAI Schemas
│       ├── hypnos-memory/       # CUDA / ROCm / Metal UMA Topology & OS Pinned RAM Manager
│       ├── hypnos-scheduler/    # Lease-based Admission Control, Partial Offload & Eviction
│       ├── hypnos-backends/     # Native llama.cpp+mtmd, stable-diffusion.cpp & ONNX Runtime engines
│       └── hypnos-server/       # Axum HTTP/SSE OpenAI Server & Multi-Modal Lifecycle Tests
└── frontend/                    # Svelte 5 (Runes) + Tailwind CSS Interactive Playground
    ├── package.json
    ├── vite.config.js
    └── src/
        └── App.svelte           # Live 3-Tier Visualizer + 5-Tab Modality Playground
```

---

## Default Multi-Modal Model Lineup

Every engine runs **inside the `hypnos` process**: llama.cpp + libmtmd, stable-diffusion.cpp and
ONNX Runtime + espeak-ng are compiled and linked natively (CUDA). No Ollama, LM Studio or Python
sidecar is involved.

| Model ID | Modalities Served | Engine | Files (under `backend/models/`) | Default Tier Policy |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma4:26b-q4`** | **Text Generation + Vision** (`/v1/chat/completions`) | `ggml_llama` | `gemma-4-26B-A4B-it-Q4_0.gguf` (13.6 GiB) + `mmproj-gemma-4-26B-A4B-it-BF16.gguf` (1.1 GiB) | `High` (Tier 2 Pinned RAM ↔ Tier 1 VRAM, hybrid 5:1 SWA KV cache) |
| **`gemma4:e4b-q4`** | **Audio / STT + Chat** (`/v1/audio/transcriptions`) | `ggml_llama` | `gemma-4-E4B-it-Q4_0.gguf` (4.3 GiB) + `mmproj-gemma-4-E4B-it-BF16.gguf` (0.9 GiB) | `High` (Tier 2 Pinned RAM ↔ Tier 1 VRAM) |
| **`kokoro-82m-q4`** | **Text-to-Speech** (`/v1/audio/speech`) | `onnx_runtime` | `kokoro/model_fp16.onnx` (155 MiB) + `kokoro/voices/*.bin` | `RealtimePinned` (always resident; runs on CPU) |
| **`gemmaembedding2`** | **Vector Embeddings** (`/v1/embeddings`) | `ggml_llama` | `embeddinggemma-2-Q8_0.gguf` (295 MiB) | `RealtimePinned` (always hot in Tier 1 VRAM) |
| **`moss-image`** | **Image Generation** (`/v1/images/generations`) | `ggml_diffusion` | `MOSS-Image-SFT-Q4_K.gguf` (3.3 GiB) + `Qwen3.5-9B-Q4_K_M.gguf` (5.7 GiB) + `flux2-vae.safetensors` + `moss_tokenizer.json` | `Low` (promoted to Tier 1 VRAM on demand, evicts idle models) |

### Model files

`./run.sh models` (Linux / macOS) or `run.bat models` (Windows) downloads everything below into
`backend/models/` with `curl`, about 31 GB. Files already present are skipped, an interrupted download
resumes on the next run, and `./run.sh models moss-image` fetches a single model. The file list lives
in [backend/models.manifest](backend/models.manifest); on Apple Silicon it picks the fp32 Kokoro graph.

Download sources (all public Hugging Face repositories):

| File | Source |
| :--- | :--- |
| `gemma-4-26B-A4B-it-Q4_0.gguf`, `mmproj-gemma-4-26B-A4B-it-BF16.gguf` | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| `gemma-4-E4B-it-Q4_0.gguf`, `mmproj-gemma-4-E4B-it-BF16.gguf` | `ggml-org/gemma-4-E4B-it-GGUF` |
| `embeddinggemma-2-Q8_0.gguf` | `ggml-org/embeddinggemma-2-GGUF` |
| `kokoro/model_fp16.onnx` (macOS: `kokoro/model.onnx`), `kokoro/voices/*.bin` | `onnx-community/Kokoro-82M-v1.0-ONNX` (`onnx/` and `voices/` folders) |
| `MOSS-Image-SFT-Q4_K.gguf` | `leejet/MOSS-Image-SFT-GGUF` |
| `Qwen3.5-9B-Q4_K_M.gguf` | `unsloth/Qwen3.5-9B-GGUF` |
| `flux2-vae.safetensors` | `Comfy-Org/MOSS-Image` (`vae/`) |
| `moss_tokenizer.json` | `OpenMOSS-Team/MOSS-Image-Base` (`tokenizer/tokenizer.json`) |

Notes:
- Kokoro's quantized exports (`model_q8f16.onnx`, `model_q4.onnx`) crash the statically linked
  ONNX Runtime CPU provider at session creation on Windows; use the fp16 graph.
- On Apple Silicon use the fp32 graph (`onnx/model.onnx`): the fp16 one overflows to a silent
  (all-NaN) waveform for some voice/text pairs there. Hypnos now rejects such output with an error.
- `kv_dtype = "q8_0"` and the sliding-window KV cache keep the Gemma 4 context budgets small; all
  budgets live in `backend/hypnos.example.toml`.

### Build prerequisites (native engines)

- Rust **1.92+** (rustup stable). `backend/.cargo/config.toml` pins the rustup toolchain's `rustc`
  because a standalone Rust install may shadow it on PATH.
- CMake 3.2x+ and **Ninja**, plus MSVC on Windows. With **CUDA 13.0** the host compiler must be an
  MSVC **14.4x** toolset (VS 2022); the `.cargo/config.toml` pins it via `CUDAHOSTCXX`.
- CUDA toolkit; `CUDAARCHS` in `.cargo/config.toml` selects the GPU architecture (default `120`,
  RTX 50 series).
- `backend/vendor/` holds the engine bindings with upstream llama.cpp **b11982** (EmbeddingGemma 2
  support) and stable-diffusion.cpp **master-1024** (MOSS-Image support); sd.cpp is built as a shared
  library (`stable-diffusion.dll`) so its ggml does not collide with llama.cpp's.
- First build compiles all engines (10–20 minutes). `backend/scripts/smoke_test.sh` exercises every
  endpoint against a running server.

### macOS (Apple Silicon)

M1 or later, macOS 14+. llama.cpp and stable-diffusion.cpp run on the GPU through Metal; there is
no CUDA and no Python involved. Design notes and open items: [docs/apple-silicon.md](docs/apple-silicon.md).

```bash
xcode-select --install          # clang, Metal compiler, frameworks
brew install cmake ninja
./backend/scripts/vendor.sh
./run.sh models                 # ~31 GB of model files into backend/models/
./run.sh build backend          # = cargo build --release -p hypnos-server --no-default-features --features metal
./run.sh start backend          # uses backend/hypnos.macos.example.toml
```

- A plain `cargo build` selects the default `cuda` feature and fails on a Mac; always pass
  `--no-default-features --features metal` (`run.sh` does this on Darwin/arm64).
- CPU and GPU share one pool of RAM. Tier 1 is the Metal working set (about 2/3 to 3/4 of RAM),
  Tier 2 is limited to what Tier 1 leaves free, and models are placed fully on the GPU or not at
  all. `sudo sysctl iogpu.wired_limit_mb=<MiB>` raises the GPU ceiling until the next reboot.
- What fits at once: 16 GB, `gemma4:e4b-q4` plus the small models; 32 GB, one of `gemma4:26b-q4`
  or `moss-image` plus the small models; 48 GB, both Gemma 4 models; 64 GB and up, the whole lineup.
- For manual debugging, use `brew install llama.cpp` instead of the Windows binaries in `backend/tools/`.

---

## Running Backend & Frontend

### One-Command Launch (Root Scripts)

`run.bat` (Windows) and `run.sh` (Linux / macOS / Git Bash) are small process managers. Building
and starting are separate steps: `build` compiles, `start` launches the compiled binary and never
invokes cargo.

```text
run.bat [build|start|stop|restart|status]      [backend|frontend|all]
./run.sh [build|start|stop|restart|status|logs] [backend|frontend|all]
```

| Command                     | Effect                                                        |
|-----------------------------|---------------------------------------------------------------|
| `... build`                 | `cargo build --release --bin hypnos` + `npm install` (first run compiles the native engines) |
| `... build backend`         | compile only the Rust server to `backend/target/release/hypnos[.exe]` |
| `run.bat` / `./run.sh`      | start backend (port 8080) and frontend (port 5173) from the built artifacts |
| `... start backend`         | start only the Rust server binary; fails with a hint if it is not built yet |
| `... restart frontend`      | stop + start only the Vite dev server                         |
| `... stop`                  | stop both (kills the whole process tree, frees the ports)     |
| `... status`                | show what is running, on which PID, and whether the backend is built |
| `./run.sh logs [target]`    | follow `.run/backend.log` / `.run/frontend.log`               |
| `... models [model id]`     | download the model files into `backend/models/` (all, or one model) |

`restart` does not rebuild: run `build` after changing Rust code, then `restart backend`.
`HYPNOS_PROFILE=debug` makes `build`/`start` use the dev profile (`backend/target/debug/`) instead.

On Windows each service gets its own console window; on Linux/macOS they run detached with logs
under `.run/`. Both scripts prefer the rustup toolchain in `~/.cargo/bin` (the native engines need
rustc 1.92 or newer).

### First-time build setup

The native engines need machine-specific settings (GPU architecture, CUDA host compiler, rustc pin).
Generate them once; the result lands in the git-ignored `<repo>/.cargo/config.toml`:

```powershell
# Windows
powershell -ExecutionPolicy Bypass -File backend\scripts\setup-build-env.ps1
```
```bash
# Linux / macOS
./backend/scripts/setup-build-env.sh
```

`backend/vendor/` (engine sources, git-ignored) is recreated from pinned commits plus patches with
`./backend/scripts/vendor.sh`; see [backend/vendor/VENDOR.md](backend/vendor/VENDOR.md). CPU-only
builds (no CUDA toolkit, e.g. CI) use `cargo build --workspace --no-default-features`.

### Operating the server

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health` | liveness: version and uptime |
| `GET /ready` | readiness: 503 until startup tier initialization has finished |
| `GET /metrics` | Prometheus text: per-route counters and latency, scheduler promotions/evictions/load failures/rejections, VRAM bookkeeping vs. real device memory, per-model tier and leases |
| `GET /v1/memory/status` | full 3-tier snapshot, including `tier1_vram_real_free_bytes` and `tier1_vram_unaccounted_bytes` |
| `POST /v1/memory/tier` | manual promote / demote (`{"model": "...", "target_tier": "tier2_pinned_ram"}`); refused with 429 while the model is leased |

Behaviour that matters in production:

- **Startup validation**: missing weight files, duplicate ids, impossible budgets or engine
  combinations abort startup with one error per problem; warnings are logged.
- **Real VRAM reconciliation**: after every load/unload the planner compares its bookings with the
  device's free memory (ggml backend API) and never hands out more than the device really has.
- **Admission and limits** (`[limits]` in `hypnos.toml`): per-model queue bound (HTTP 429 when
  exceeded), request timeout (HTTP 504), body/audio/image/prompt size limits, `max_tokens` cap.
- **Cancellation**: when a client disconnects, the generation is stopped at the next token
  (llama.cpp), chunk (Kokoro) or diffusion step (sd.cpp) and the lease is released.
- **Exposure**: binds `127.0.0.1` by default; CORS allows loopback origins only unless
  `server.allowed_origins` says otherwise. There is no authentication: put a reverse proxy in front
  for anything beyond the local machine.
- **Logs**: `server.log_format = "json"` (or env `HYPNOS_LOG_FORMAT=json`) for log shippers; every
  response carries an `x-request-id` header.
- **Shutdown**: Ctrl+C / SIGTERM finishes in-flight requests, then releases every engine.
- **Tests**: `cargo test --workspace` runs engine-free scheduler tests (mock backend: promotion,
  eviction order, rollback on load failure, admission, cancellation, shutdown) plus unit tests;
  `cargo test -- --ignored` runs the GPU lifecycle test against the real weights;
  `backend/scripts/smoke_test.sh` exercises every endpoint of a running server. CI
  (`.github/workflows/ci.yml`) builds and tests CPU-only on Ubuntu and Windows.

### Manual Launch
#### 1. Start the Rust Backend (`http://localhost:8080`)
```bash
cd backend
cargo test
cargo run --bin hypnos -- hypnos.example.toml
```

### 2. Start the Svelte 5 Playground (`http://localhost:5173`)
```bash
cd frontend
npm install
npm run dev
```
