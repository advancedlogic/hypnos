<p align="center">
  <img src="assets/logo.png" alt="Hypnos logo" width="200">
</p>

<h1 align="center">Hypnos</h1>

<p align="center">
  <b>One OpenAI-compatible endpoint for every modality on one GPU,<br>
  with a scheduler that decides what lives in VRAM so you don't have to.</b>
</p>

Hypnos serves chat, vision, speech-to-text, text-to-speech, embeddings, image generation, document
conversion and entity extraction from a single Rust process. llama.cpp, stable-diffusion.cpp, ONNX
Runtime, docling.rs and GLiNER are linked in, so there is no Python, no sidecar and no second server
competing for the same GPU. A lease-based scheduler moves
models between GPU memory, pinned RAM and NVMe on demand: the models you use all the time stay hot,
the big ones come in when asked for, and nothing is evicted while a request is using it.

- **One process, one port, one memory budget.** Every engine books its weights, KV cache and scratch
  space with the same planner, which checks its bookings against the GPU's real free memory after
  every load.
- **Three memory tiers.** Tier 1 is GPU VRAM (on Apple Silicon, Metal's share of unified memory),
  Tier 2 is page-locked host RAM that promotes over PCIe without touching the disk, and Tier 3 is the
  memory-mapped model file on NVMe.
- **Priorities, not timeouts.** `realtime_pinned` models never leave the GPU, `high` ones are kept
  warm in RAM, `low` ones are evicted first. Models that don't fully fit run with part of their
  layers on the GPU (CUDA).
- **Swaps that keep their place.** With KV parking (optional) a chat model that is evicted for
  another one saves its conversations' KV cache in host RAM and restores it when it comes back, so
  the next turn prefills only its new tokens. A Mixture-of-Experts model can keep its expert
  weights on the CPU (optional, per model) and need a fraction of the VRAM, to stay resident next to
  an image or document model instead of being swapped out.
- **Drop-in for OpenAI clients.** `/v1/chat/completions` (streaming, images, audio, tool calling), `/v1/embeddings`,
  `/v1/audio/transcriptions`, `/v1/audio/speech` (streaming), `/v1/images/generations`, with
  optional API keys and in-process HTTPS. Plus `/v1/documents/convert` (PDFs, images and office
  documents to Markdown with docling, page by page) and `/v1/entities` (zero-shot named-entity
  extraction with GLiNER2.5, labels chosen per request).
- **Observable.** A live playground shows each model's tier as requests come in, every response
  carries its placement and timing, `/metrics` is Prometheus, and a request log keeps recent calls.
- **Runs on CUDA (Windows, Linux) and Apple Silicon (Metal).**

<p align="center">
  <img src="assets/demo.gif" alt="Playground: a chat request promotes Gemma 4 26B from pinned RAM to the GPU, docling converts a PDF and keeps its pictures, GLiNER extracts entities, an image request evicts the 26B for MOSS-Image, Kokoro answers from a pinned model, and the next chat swaps the 26B back in" width="960">
</p>

<p align="center"><sub>The playground on an RTX 5090 laptop GPU (24 GB) with the default lineup: chat, a PDF
through docling with its pictures kept, entity extraction with GLiNER, an image, speech, chat. The 26B
model and MOSS-Image don't fit together, so each request evicts the other; Kokoro and the embedder stay
pinned. 95 s of real time with the waits shortened (<code>tools/demo-gif/</code> re-records it).</sub></p>

### Measured

Full reports in [docs/bench/](docs/bench/). All chat numbers are Gemma 4 26B-A4B Q4.

#### CUDA (24 GB GPUs)

| Scenario | Result |
| :--- | :--- |
| Chat decode | ~100 tokens/s per stream on a 24 GB desktop GPU; **150** on an RTX 5090 laptop with 3 slots at 16k and an f16 KV cache (4 slots at 16k with q8_0 made llama.cpp re-capture its CUDA graph every step: 8 tokens/s, see [docs/bench/decode-slots-cuda-2026-10-30.md](docs/bench/decode-slots-cuda-2026-10-30.md)) |
| Five-turn chat, time to first token per turn | 300 ms → **145 ms** with KV prefix reuse |
| Follow-up question about the same image | 300 ms → **105 ms** (image not re-encoded) |
| Four concurrent chat clients | **2.18x** the aggregate decode throughput of one (continuous batching) |
| Text-to-speech, time to first audio | 5.4 s → **385 ms** with streaming synthesis |
| Document conversion (docling), text page / page with two tables | **49 ms** / 1.07 s on the GPU, 133 ms / 0.9 s on the CPU (RTX 5090 laptop; [report](docs/bench/docling-cuda-2026-10-30.md)) |
| Model swap, Tier 2 → first token | **3.7-4.2 s** (RTX 5090 laptop, Windows; [report](docs/bench/swap-cuda-2026-10-30.md)) |

The Tier 2 promotion was 7.5 s warm and 23-59 s when another model had just been loaded before
the PERF-6 changes: the weights are now read through pinned staging buffers instead of a file
mapping (2.4x faster upload), and Tier 2 pins a model's projector and text-encoder files with its
weights. KV parking and MoE expert offload are not measured on CUDA yet.

<!-- PERF-5: add the CUDA run (bench.sh --scenarios 7) as a chart like the Metal one. -->

#### Apple Silicon (M5 Max, 128 GB, Metal)

Release build, `hypnos.macos.example.toml` ([report](docs/bench/metal-2026-10-31.md), KV parking,
MoE offload and slots in [docs/bench/metal-features-2026-10-31.md](docs/bench/metal-features-2026-10-31.md)):

| Scenario | Result |
| :--- | :--- |
| Chat decode | **113-117 tokens/s** per stream; prefill 3,600-4,000 tokens/s on a 6.4k-token prompt |
| Five-turn chat, time to first token per turn | 182 ms → **97-100 ms** with KV prefix reuse |
| Follow-up question about the same image | 164 ms → **90 ms** (image not re-encoded) |
| Four concurrent chat clients | **1.43x** the aggregate throughput of one with `slots = 4` (169 vs 118 tokens/s, slowest first token 274 ms); the shipped profile has `slots = 1`, where they queue (0.97x, slowest first token 3.4 s) |
| Text-to-speech, time to first audio | 5.0 s → **359 ms** with streaming synthesis (Kokoro on the CPU, RTF 0.13) |
| Voice round trip (11 s of speech → transcript → reply → speech) | **2.25 s** to the first audio; transcription 0.98 s (RTF 0.04) |
| Next chat turn after the model was swapped out, 1.7k / 6.3k / 12.6k-token conversation | 1.39 / 2.57 / 4.62 s → **1.04 / 1.08 / 1.12 s** with KV parking (restore 2-5 ms for 230-344 MiB) |
| MoE expert offload (`moe_offload = "on"`) | GPU weights 15.8 → **3.0 GB** (12.8 GB of experts in host memory), at 49 vs 113 tokens/s decode and 250 vs 3,800 tokens/s prefill: no use on unified memory, which is why `auto` never acts there |
| Entity extraction (GLiNER2.5), 1,700 words with 11 labels | **1.5 s** on the CPU, models loaded in under a second |

Model swap: time to the first token of the same model (15.8 GB) depending on where the scheduler
finds it, with the weight files evicted from the OS page cache for the Tier 3 runs as far as macOS
allows (19% stayed resident):

![Time to first token by start tier: Tier 3 2.32 s, Tier 2 987 ms, Tier 1 54 ms](docs/bench/swap-metal-2026-10-31.svg)

### How it compares

| | Hypnos | Ollama | LocalAI | llama-swap |
| :--- | :--- | :--- | :--- | :--- |
| Shape | one process, engines linked in | one server, a runner process per loaded model | API gateway over separate backend processes | proxy that starts and stops one server per model |
| Focus | every modality on one GPU at once | LLMs, easy model library | broadest set of backends and model formats | swapping any OpenAI-compatible server |
| Deciding what is loaded | shared memory planner, priorities, leases, warm RAM tier | load on request, unload after an idle timeout | per backend | one model (or group) at a time |

Pick Ollama if you want the easiest install and a curated model library, LocalAI if you need a
backend Hypnos doesn't have, and vLLM or SGLang for multi-user datacenter throughput. Hypnos is for
the case where one machine has to serve a whole multi-modal stack (a voice assistant, an agent with
speech, images and retrieval) and VRAM is the bottleneck.

---

## Quickstart

### Prebuilt binaries

Each [GitHub release](https://github.com/advancedlogic/hypnos/releases) has archives for Linux x64
(CUDA), Windows x64 (CUDA) and macOS arm64 (Metal), with the playground, the example config and a
model downloader included. Unpack one and run everything from that folder:

```bash
./scripts/fetch-models.sh     # the default lineup, ~31 GB into models/ (or one model id)
./hypnos                      # reads hypnos.toml; API and playground on http://127.0.0.1:9999
./hypnos --port 9000          # or HYPNOS_PORT=9000: overrides server.port without editing the file
```

On Windows: `powershell -ExecutionPolicy Bypass -File scripts\fetch-models.ps1`, then `.\hypnos.exe`.

| Archive | Needs |
| :--- | :--- |
| `linux-x64-cuda` | glibc 2.39+ (Ubuntu 24.04+), NVIDIA driver 570+, CUDA 12 runtime libraries (`libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`) and `libgomp1`. Document conversion on the GPU (docling `device = "gpu"`, off by default) also needs driver 580+, the CUDA 13 runtime and cuDNN 9 for CUDA 13 (`apt install cuda-cudart-13-0 libcublas-13-0 libcudnn9-cuda-13`) |
| `windows-x64-cuda` | NVIDIA driver 570+, the release's `windows-x64-cuda12-runtime.zip` unpacked beside `hypnos.exe` (or the CUDA Toolkit 12.x), the MSVC 2015–2022 redistributable. Document conversion on the GPU also needs driver 580+, the CUDA 13 runtime DLLs and cuDNN 9 for CUDA 13 on `PATH` (CUDA Toolkit 13 + cuDNN, or the pip wheels `nvidia-cuda-runtime-cu13`, `nvidia-cublas-cu13`, `nvidia-cudnn-cu13`) |
| `macos-arm64-metal` | Apple Silicon, macOS 14+; run `xattr -dr com.apple.quarantine .` once in the folder (the binaries are not notarized) |

The binaries link espeak-ng statically, so the archives are distributed under the GNU GPL v3 or
later (see [License](#license)); each one includes the license texts and links its source tag.

Releases after v0.2.0 also carry installers for the [desktop app](#desktop-app)
(`hypnos-desktop-*.dmg`, `.msi` / `.setup.exe`, `.deb` / `.AppImage`).

### Docker (CUDA)

Linux with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
mkdir -p models
docker run --rm -v ./models:/opt/hypnos/models --entrypoint ./scripts/fetch-models.sh \
  ghcr.io/advancedlogic/hypnos                       # ~31 GB, or append one model id
docker run --gpus all -p 9999:9999 -v ./models:/opt/hypnos/models ghcr.io/advancedlogic/hypnos
```

The playground and API are then on http://127.0.0.1:9999. The image's config is the CUDA example
(`/opt/hypnos/hypnos.toml`, listening on all interfaces inside the container); mount your own over
it, or mount one elsewhere and set `-e HYPNOS_CONFIG=/path/in/container.toml`. Set
`-e HYPNOS_API_KEYS=...` before publishing the port beyond localhost. Build it yourself with
`docker build -t hypnos .`; `--build-arg JOBS=N` sets the compile parallelism (default 2, which
fits 16 GB of RAM: some CUDA kernels need several GB each). The image is
GPL-3.0-or-later as a whole, like the archives; the license texts are in `/opt/hypnos`.

### From source

The first build compiles every native engine and takes 10–20 minutes.

**You need:** Rust 1.92+ through rustup, CMake 3.2x+, Ninja, Node.js (for the playground), and
either the CUDA toolkit with an NVIDIA GPU (Windows needs MSVC 2022; see
[Building from source](#building-from-source)) or an Apple Silicon Mac with the Xcode command line
tools.

```bash
git clone https://github.com/advancedlogic/hypnos.git && cd hypnos
./backend/scripts/setup-build-env.sh   # once per machine: GPU arch, toolchain pins
./backend/scripts/vendor.sh            # fetch the pinned engine sources into backend/vendor/
./run.sh models                        # the default lineup, ~31 GB into backend/models/
./run.sh build                         # backend + playground (Metal is picked automatically on Apple Silicon)
./run.sh start                         # API on http://localhost:9999, playground on http://localhost:5173
```

On Windows use `powershell -ExecutionPolicy Bypass -File backend\scripts\setup-build-env.ps1` and
`run.bat` with the same arguments; `vendor.sh` runs under Git Bash.

Startup refuses to run when a configured model's files are missing, so the default profile needs the
whole lineup. To try Hypnos with a single model, use a config of your own instead (next section).
On a Mac, [macOS (Apple Silicon)](#macos-apple-silicon) lists which models fit at once for each RAM
size.

Then send a request with any OpenAI client, or curl:

```bash
curl http://localhost:9999/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gemma4:e4b-q4", "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:9999/v1", api_key="unused")
print(client.chat.completions.create(
    model="gemma4:e4b-q4",
    messages=[{"role": "user", "content": "Hello!"}],
).choices[0].message.content)
```

`./backend/scripts/smoke_test.sh` sends one real request per modality to a running server.

---

## Bring your own models

Any GGUF that llama.cpp loads works for chat and embeddings; the chat template is read from the
file. A complete config with a single model looks like this (paths are relative to `backend/`):

```toml
# backend/my.toml
[[models]]
id = "qwen3:8b"                      # the "model" name clients send
modality = "text_generation"         # or "embeddings"; add additional_modalities = ["vision"] with an mmproj
engine = "ggml_llama"
weights_path = "models/Qwen3-8B-Q4_K_M.gguf"
# aux_weights_path = "models/mmproj-....gguf"   # vision / audio projector, if the model has one
priority = "high"                    # realtime_pinned | high | normal | low
initial_tier = "tier2_pinned_ram"    # stage in RAM at startup; default tier3_nvme_ssd loads on first use
enabled = true                       # false keeps the block but takes the model out of service

[models.arch]                        # optional; sizes the KV cache and the GPU/CPU layer split
default_context_len = 8192           # context reserved per load (layers, KV heads, head_dim,
                                     # max context and sliding window are read from the GGUF)
kv_dtype = "q8_0"                    # f16 | q8_0 | q4_0
```

Relative model paths resolve against the working directory (`backend/` here), or against
`[server] models_dir` when the config sets one: the example configs use `models_dir = "models"`
with bare file names, and `HYPNOS_MODELS_DIR` overrides it (models on another drive).

```bash
HYPNOS_CONFIG=my.toml ./run.sh start backend     # or: cd backend && ./target/release/hypnos my.toml
```

On Windows: `set HYPNOS_CONFIG=my.toml` and then `run.bat start backend`.

The port is `server.port` (9999 by default); `hypnos my.toml --port 9000` or `HYPNOS_PORT=9000` override
it without editing the file, so one config can serve several instances.

- `use_mmap` (unset = automatic) decides how llama.cpp reads the weights: through pinned staging
  buffers with asynchronous uploads when layers go to a CUDA device, a file mapping on Metal and
  for CPU placements. Every Tier 1 load reports its phases (weights, projector, context) in the
  log and in `hypnos_telemetry.tier_switch_phases`.
- `moe_offload = "auto"` on a Mixture-of-Experts model keeps its expert weights on the CPU when
  the GPU is short, `"on"` always ([MoE expert offload](#operating-the-server)); `[hardware] kv_park = true`
  keeps chat models' conversations across swaps ([KV parking](#operating-the-server)). Both are off
  by default and can be switched at runtime.
- `[models.arch]` fields you leave out are read from the GGUF header at startup
  (`<arch>.block_count`, `.attention.head_count_kv`, `.attention.key_length`, `.context_length`,
  `.attention.sliding_window[_pattern]`) and logged; values you set always win, and a mismatch with
  the file is logged too. Per-layer values (Gemma 4) are booked at their largest, so the plan errs
  on the safe side.
- Weight size is read from the file. `[server]`, `[limits]` and `[hardware]` are optional; GPU
  memory is detected, so you only set budgets to stay below what the device has.
- Speech, text-to-speech and image generation need the engine-specific fields shown in
  [backend/hypnos.example.toml](backend/hypnos.example.toml), which documents every option.
  Document conversion (`engine = "docling"`, `modality = "document_conversion"`) and entity
  extraction (`engine = "gliner"`, `modality = "entity_extraction"`) point `weights_path` at a
  model file inside their model folder (`docling/layout_heron_int8.onnx`,
  `gliner2.5-multi/encoder_fp32.onnx`)
  and run on the CPU with `device = "cpu"`, so they book host RAM rather than VRAM; the example
  config has both blocks. Startup validates the config and prints one error per problem.

---

### Config versions and `--check-config`

`config_version = 1` at the top of `hypnos.toml` names the schema the file was written for
(absent means 1). A file for a newer Hypnos is refused at startup with a message naming both
versions; an older file is migrated in memory, one step per version, and each step is logged, so a
breaking change to a key never breaks a running deployment silently. Keys no field reads (typos,
or keys a later version removed) are reported as warnings instead of being ignored.

```bash
hypnos --check-config hypnos.toml     # what startup would do: GGUF detection notes, warnings,
                                      # errors, then one summary line; exit code 1 on errors
```

`backend/models.manifest` carries `# manifest-version: 1` the same way: the fetch scripts refuse a
manifest with a newer layout than they read.

| Config version | Introduced | Migration |
| :--- | :--- | :--- |
| 1 | v0.1.0 | first versioned schema; files without `config_version` are version 1 |

## Default model lineup

Every engine runs **inside the `hypnos` process**: llama.cpp + libmtmd, stable-diffusion.cpp,
ONNX Runtime + espeak-ng, docling.rs and GLiNER2.5 are compiled and linked natively (CUDA or Metal). No Ollama,
LM Studio or Python sidecar is involved.

| Model ID | Modalities Served | Engine | Files (under `backend/models/`) | Default Tier Policy |
| :--- | :--- | :--- | :--- | :--- |
| **`gemma4:26b-q4`** | **Text Generation + Vision** (`/v1/chat/completions`) | `ggml_llama` | `gemma-4-26B-A4B-it-Q4_0.gguf` (13.6 GiB) + `mmproj-gemma-4-26B-A4B-it-BF16.gguf` (1.1 GiB) | `High` (Tier 2 Pinned RAM ↔ Tier 1 VRAM, hybrid 5:1 SWA KV cache) |
| **`gemma4:e4b-q4`** | **Audio / STT + Chat** (`/v1/audio/transcriptions`) | `ggml_llama` | `gemma-4-E4B-it-Q4_0.gguf` (4.3 GiB) + `mmproj-gemma-4-E4B-it-BF16.gguf` (0.9 GiB) | `High` (Tier 2 Pinned RAM ↔ Tier 1 VRAM) |
| **`kokoro-82m-q4`** | **Text-to-Speech** (`/v1/audio/speech`) | `onnx_runtime` | `kokoro/model_fp16.onnx` (155 MiB) + `kokoro/voices/*.bin` | `RealtimePinned` (always resident; runs on CPU) |
| **`gemmaembedding2`** | **Vector Embeddings** (`/v1/embeddings`) | `ggml_llama` | `embeddinggemma-2-Q8_0.gguf` (295 MiB) | `RealtimePinned` (always hot in Tier 1 VRAM) |
| **`moss-image`** | **Image Generation** (`/v1/images/generations`) | `ggml_diffusion` | `MOSS-Image-SFT-Q4_K.gguf` (3.3 GiB) + `Qwen3.5-9B-Q4_K_M.gguf` (5.7 GiB) + `flux2-vae.safetensors` + `moss_tokenizer.json` | `Low` (promoted to Tier 1 VRAM on demand, evicts idle models) |
| **`docling`** | **Document Conversion** (`/v1/documents/convert`) | `docling` | `docling/` (310 MiB: layout, TableFormer, OCR) | `Normal`, `device = "cpu"` (loaded on demand into host RAM in ~0.3 s, ~1.8 GiB booked, no VRAM) |
| **`gliner2.5-multi`** | **Entity Extraction** (`/v1/entities`) | `gliner` | `gliner2.5-multi/` (1.15 GiB: encoder, boundary heads, tokenizer) | `Normal`, `device = "cpu"` (loaded on demand in under a second, ~3.2 GiB booked, no VRAM) |

### Model files

`./run.sh models` (Linux / macOS) or `run.bat models` (Windows) downloads everything below into
`backend/models/` with `curl`, about 33 GB. Files already present are skipped, an interrupted download
resumes on the next run, and `./run.sh models moss-image` fetches a single model. The file list lives
in [backend/models.manifest](backend/models.manifest); on Apple Silicon it picks the fp32 Kokoro graph.

Download sources (public Hugging Face repositories, and one GitHub release):

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
| `docling/` (`layout_heron_int8.onnx`, `tableformer/`, `ocr_*`) | `docling-project/docling.rs` GitHub release `models-v1` |
| `gliner2.5-multi/` | `jugaadsrl/gliner2.5-multi-v1-onnx` (`fp32_25/` folder) |

Notes:
- Kokoro's quantized exports (`model_q8f16.onnx`, `model_q4.onnx`) crash the statically linked
  ONNX Runtime CPU provider at session creation on Windows; use the fp16 graph.
- On Apple Silicon use the fp32 graph (`onnx/model.onnx`): the fp16 one overflows to a silent
  (all-NaN) waveform for some voice/text pairs there. Hypnos now rejects such output with an error.
- `kv_dtype = "q8_0"` and the sliding-window KV cache keep the Gemma 4 context budgets small; all
  budgets live in `backend/hypnos.example.toml`.

---

## Building from source

### Prerequisites

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
`./backend/scripts/vendor.sh`, whose header lists the pins; the patches are in
[backend/vendor/patches/](backend/vendor/patches/). CPU-only builds (no CUDA toolkit, e.g. CI) use
`cargo build --workspace --no-default-features`.

---

## Running

### Run scripts

`run.bat` (Windows) and `run.sh` (Linux / macOS / Git Bash) are small process managers. Building
and starting are separate steps: `build` compiles, `start` launches the compiled binary and never
invokes cargo.

```text
run.bat [build|start|stop|restart|status]      [backend|frontend|desktop|all]
./run.sh [build|start|stop|restart|status|logs] [backend|frontend|desktop|all]
```

| Command                     | Effect                                                        |
|-----------------------------|---------------------------------------------------------------|
| `... build`                 | `cargo build --release --bin hypnos` + `npm install` (first run compiles the native engines) |
| `... build backend`         | compile only the Rust server to `backend/target/release/hypnos[.exe]` |
| `run.bat` / `./run.sh`      | start backend (port 9999) and frontend (port 5173) from the built artifacts |
| `... start backend`         | start only the Rust server binary; fails with a hint if it is not built yet |
| `... restart frontend`      | stop + start only the Vite dev server                         |
| `... stop`                  | stop both (kills the whole process tree, frees the ports)     |
| `... status`                | show what is running, on which PID, and whether the backend is built |
| `./run.sh logs [target]`    | follow `.run/backend.log` / `.run/frontend.log`               |
| `... models [model id]`     | download the model files into `backend/models/` (all, or one model) |
| `... build desktop`         | `npm install` + `tauri build --no-bundle` (an `.app` bundle on macOS) of the [desktop app](#desktop-app) |
| `... start desktop`         | open the built desktop app (`stop`, `status`, `logs desktop` too); `all` never includes it |

`restart` does not rebuild: run `build` after changing Rust code, then `restart backend`.
`HYPNOS_PROFILE=debug` makes `build`/`start` use the dev profile (`backend/target/debug/`) instead.

On Windows each service gets its own console window; on Linux/macOS they run detached with logs
under `.run/`. Both scripts prefer the rustup toolchain in `~/.cargo/bin` (the native engines need
rustc 1.92 or newer).

### Manual launch

#### 1. Start the Rust Backend (`http://localhost:9999`)
```bash
cd backend
cargo test
cargo run --bin hypnos -- hypnos.example.toml
```

#### 2. Start the Svelte 5 Playground (`http://localhost:5173`)
```bash
cd frontend
npm install
npm run dev
```

### Desktop app

`desktop/` is a Tauri 2 app for a running server, in the spirit of the Ollama app: a chat with
streaming answers and a model picker, and a collapsible right panel with the server's live
state (memory tiers, the models in each, the last answer's timings, evictions). It connects to
`http://localhost:9999` by default; the server URL and an API key (kept in the system keychain)
are set in its connection dialog. It does not contain the server.

```bash
./run.sh build desktop && ./run.sh start desktop   # or: cd desktop && npm install && npx tauri dev
```

Prerequisites per OS, shortcuts and how it reaches a remote server: [desktop/README.md](desktop/README.md).

---

## Limiting memory and simulating smaller machines

All in `[hardware]` (bytes; 0 = off / auto):

| Setting | Effect |
| :--- | :--- |
| `vram_budget_bytes` | Cap on Tier 1 (GPU memory). Clamped to what the device reports, so it can only lower the budget. |
| `pinned_ram_budget_bytes` | Cap on Tier 2 (warm host RAM). Clamped to physical RAM minus a 4 GiB reserve. |
| `simulate_physical_memory_bytes` | Behave like a machine with this much RAM (never more than installed): Tier 1 and Tier 2 are derived as there. On a Mac this includes Metal's GPU share, estimated at 2/3 of RAM up to 36 GB and 3/4 above. |
| `max_process_memory_bytes` | Soft cap on memory the `hypnos` process holds, measured from the OS before every model load. Idle models are evicted to stay under it; if that is not enough the request fails with HTTP 507 (`process_memory_cap`). |

The budgets steer the scheduler, they are not OS-enforced: memory an engine allocates beyond its
estimate is caught only by `max_process_memory_bytes`, and only at the next load. On macOS the process
figure is the `phys_footprint` plus the model weights llama.cpp maps into Metal (file-backed, so the
footprint leaves them out); on Linux the resident set; on Windows the working set. GPU memory on a
discrete card is limited by `vram_budget_bytes` only. `/v1/memory/status` reports
`process_memory_bytes` next to the budgets, and `recent_evictions` lists the scheduler's last automatic
evictions with their reason (`memory_budget`, `full_gpu_placement`, `process_memory_cap`). The
playground's memory panel shows the process bar, a "Simulated" tag and those evictions. Disk space is only consumed by `models` downloads, which
check free space first.

---

## Operating the server

[docs/operating.md](docs/operating.md) is the operator's guide: sizing a machine for a lineup,
which knob to turn for which symptom, reading the telemetry and `/metrics`, every error and its
fix, log levels and redaction, upgrades, and the Docker / systemd / Windows service recipes.
[docs/api.md](docs/api.md) is the HTTP API reference for client and SDK authors: every route,
field, streaming format, error code and the retry and cancellation behaviour a client needs.
Client SDKs implement it for [Python](sdk/python), [JavaScript/TypeScript](sdk/javascript) and
[Rust](sdk/rust): every route, streaming with cancellation, typed errors, the retry policy and
readiness polling, with an example per default model. They are not on the package registries
yet; install them from the repository.

| Endpoint | Purpose |
| :--- | :--- |
| `GET /health` | liveness: version and uptime |
| `GET /ready` | readiness: 503 until startup tier initialization has finished. The port is open from the start: `/health`, `/v1/memory/status` (where the staging shows), `/v1/models` and `/v1/logs` answer meanwhile, while the inference and placement routes answer 503 `model_not_available` with `Retry-After` instead of queueing behind a model load that can take minutes |
| `GET /metrics` | Prometheus text: per-route counters and latency, scheduler promotions/evictions/load failures/rejections, VRAM bookkeeping vs. real device memory, per-model tier and leases, per-model load failures (`hypnos_model_load_failures_total{model}`), the process start time (`hypnos_process_start_time_seconds`, a change means a restart), and per-model execution histograms (`hypnos_chat_ttft_ms`, `hypnos_chat_queue_ms`, `hypnos_chat_context_ms`, `hypnos_chat_prefill_tokens_per_s`, `hypnos_chat_decode_tokens_per_s`, `hypnos_tts_rtf`, `hypnos_tts_ttfa_ms`, `hypnos_stt_rtf`, `hypnos_image_jct_s`; each also exposes a `_last` gauge) and prefix-cache counters (`hypnos_prefix_cache_hits_total`, `_misses_total`, `_tokens_reused_total`) |
| `GET /v1/memory/status` | full 3-tier snapshot, including `tier1_vram_real_free_bytes` and `tier1_vram_unaccounted_bytes`, plus `gpu` (utilisation, temperature, power, SM clock, throttling, ECC errors, pending retired pages; NVIDIA through `nvidia-smi`, sampled at most every 3 s; also the `hypnos_gpu_*` gauges on `/metrics` and the visualizer's header line) |
| `POST /v1/memory/tier` | manual promote / demote (`{"model": "...", "target_tier": "tier2_pinned_ram"}`); refused with 429 while the model is leased |
| `POST /v1/models/{id}/disable`, `.../enable` | take a model out of service without removing its config, or put it back. Disabling evicts it to Tier 3 (refused with 429 while leased), hides it from `/v1/models` and makes requests for it answer 404 `model_disabled`; enabling lets it load on its next request. Same as `enabled = false` in its `[[models]]` block (which also tolerates missing weights), but not written back to the file. `/v1/memory/status` shows `enabled` per model, and the playground's model cards carry the switch |
| `GET /v1/memory/kv-park`, `POST /v1/memory/kv-park` | KV parking state (parked models, slots, tokens, bytes) and the runtime switch: `{"enabled": true, "max_bytes": 4294967296}`; turning it off drops every parked state |
| `POST /v1/models/{id}/moe-offload` | a Mixture-of-Experts model's expert offload at runtime: `{"mode": "off" \| "auto" \| "on"}`; an idle resident model is unloaded so its next load uses the new mode |
| `GET /v1/logs` | recent API calls, newest first (`?model=&route=&status=ok|error|4xx|5xx&limit=`): route, status, duration, model, tokens in / out / cached, tokens/s, tier switch; `GET /v1/logs/{id}` adds the prompt and completion text (level `debug`) and the request / response payloads with media replaced by placeholders (level `trace`) |
| `POST /v1/logs/level` | `{"level": "off|error|warn|info|debug|trace"}` changes what is captured at runtime (`server.request_log_level` sets the start value, `server.request_log_capacity` the ring size); `DELETE /v1/logs` empties the ring. The playground's "Request Logs" tab is a client of these |

Behaviour that matters in production:

- **Startup validation**: missing weight files, duplicate ids, impossible budgets or engine
  combinations abort startup with one error per problem; warnings are logged.
- **Real VRAM reconciliation**: after every load/unload the planner compares its bookings with the
  device's free memory (ggml backend API) and never hands out more than the device really has.
- **Admission and limits** (`[limits]` in `hypnos.toml`): per-model queue bound (HTTP 429 when
  exceeded), request timeout (HTTP 504), body/audio/image/prompt size limits, `max_tokens` cap.
- **Cancellation**: when a client disconnects, the generation is stopped at the next token
  (llama.cpp), chunk (Kokoro) or diffusion step (sd.cpp) and the lease is released.
- **Persistent contexts, slots and prefix reuse** (llama.cpp models): a resident model is served
  by one worker thread that keeps a single `llama_context` for the whole residency and runs
  requests through execution slots (`slots` per model, default 1; the planner books KV for every
  slot). Requests in different slots are decoded together, one `llama_batch` per step, so several
  chats share each forward pass; `hypnos_telemetry.engine.concurrent_slots` reports how many
  shared a request's steps. With `prefix_cache = true` (default) a
  prompt that extends what a slot processed last prefills only the new part, and an image or audio
  clip already in the cache is not re-encoded; `usage.prompt_tokens_details.cached_tokens` reports
  the reused positions. Idle slot caches are cleared after `server.kv_idle_timeout_secs`.
- **KV parking** (`hardware.kv_park`, off by default; `POST /v1/memory/kv-park` at runtime): when a
  llama.cpp model leaves Tier 1, or is reloaded with a different layer split, each used slot's KV
  state (with its token keys and sliding-window checkpoint) is copied to host RAM; when the model
  comes back with the same context layout (context length, slots, KV type, weights) the slots are
  restored before it takes requests, and prefix reuse carries on as if it had never left.
  `hardware.kv_park_bytes` (default 4 GiB) caps the parked states, the least recently parked going
  first; a disabled model's states are dropped. Responses that restored report `kv_restore_ms` and
  `kv_restored_tokens` in `hypnos_telemetry.tier_switch_phases`. Design and numbers:
  [docs/pagination.md](docs/pagination.md).
- **MoE expert offload** (`moe_offload` per model: `off` (default), `auto`, `on`;
  `POST /v1/models/{id}/moe-offload` at runtime): the expert tensors of a Mixture-of-Experts model
  stay in host memory and run on the CPU, while attention, the dense layers, the router and the KV
  cache stay on the GPU. Gemma 4 26B-A4B then needs 3.0 GB of weights on the GPU instead of 15.8 GB
  (projector included). `on` offloads every layer's experts; `auto` offloads the experts of as few
  of the last layers as needed when the model does not fit whole, before falling back to a layer split
  (discrete GPUs only). It costs speed: on an M5 Max decode went from ~113 to ~49 tokens/s and
  prefill from ~3,800 to ~250 tokens/s, which is why `auto` never acts on unified memory; the CUDA
  measurement is pending. The placement reports `cpu_expert_layers`. Design:
  [docs/moe.md](docs/moe.md).
- **Exposure**: binds `127.0.0.1` by default; CORS allows loopback origins only unless
  `server.allowed_origins` says otherwise. To serve beyond the local machine, set both:
  - `server.api_keys` (or env `HYPNOS_API_KEYS`, comma-separated, which replaces the list): every
    `/v1/*` route then needs `Authorization: Bearer <key>`, as OpenAI clients send it. A missing or
    wrong key gets HTTP 401 in the OpenAI error shape (`code: invalid_api_key`); rejections are
    logged with route and reason, never the key. Keys are held only as SHA-256 digests and compared
    in constant time. `/health` and `/ready` stay open for probes; `/metrics` too unless
    `server.metrics_require_auth = true`.
  - `[[server.keys]]` adds labelled keys with limits: `requests_per_minute`, `max_concurrent` (a
    stream counts until it ends) and `tokens_per_day` (prompt + completion, UTC day); unset ones
    take `[server.api_key_defaults]`, which also covers the plain `api_keys`. Over a limit the
    call gets HTTP 429 `rate_limit_error` with `Retry-After` and a log line naming the key's label
    and the limit. The label is on every request-log record (`GET /v1/logs?key=team-a`) and on
    `/metrics` (`hypnos_api_key_requests_total`, `_rejections_total{reason}`, `_tokens_total`,
    `_tokens_today`). A plain key is labelled `key-<8 hex of its digest>`.
  - `server.request_log_redact` keeps prompts and completions out of the request log: regular
    expressions whose matches become `[redacted]`, or `["all"]`. Keep the log level at `info` for
    anything exposed beyond a trusted group; `debug` and `trace` store what users send.
  - For quotas per user or organisation, billing, or anything beyond one key per client, put an
    API gateway in front: Hypnos authenticates and limits per key, nothing finer.
  - `[server.tls]` with `cert_path` / `key_path` (PEM) to serve HTTPS in-process (rustls); plain
    HTTP when absent. A bad certificate stops startup before any model loads.

  Startup warns when a non-loopback `host` has no keys, or keys but no TLS. The playground has an
  API-key field next to the backend URL; the key is stored in that browser only.
- **Logs**: `server.log_format = "json"` (or env `HYPNOS_LOG_FORMAT=json`) for log shippers; every
  response carries an `x-request-id` header.
- **Shutdown**: Ctrl+C / SIGTERM finishes in-flight requests, then releases every engine.
- **Tests**: `cargo test --workspace` runs engine-free scheduler tests (mock backend: promotion,
  eviction order, rollback on load failure, admission, cancellation, shutdown) plus unit tests;
  `cargo test -- --ignored` runs the GPU lifecycle test against the real weights;
  `backend/scripts/smoke_test.sh` exercises every endpoint of a running server. CI
  (`.github/workflows/ci.yml`) runs fmt + clippy, CPU (Linux, Windows) and Metal (macOS) builds
  and tests, and the frontend build on every push and pull request.
- **Telemetry**: every response's `hypnos_telemetry` carries the tier placement plus an `engine`
  object with the request's timing split (`queue_ms`, `context_ms`, `prefill_ms`, `decode_ms`,
  `ttft_ms`, `prefill_tokens_per_s`, `decode_tokens_per_s`; `audio_secs` and `rtf` for speech).
  `/v1/audio/speech` reports the same in `x-hypnos-inference-ms`, `x-hypnos-audio-secs` and
  `x-hypnos-rtf` headers.
- **Streaming speech**: `POST /v1/audio/speech` with `"stream": true` returns audio as it is
  synthesized, chunk by chunk: raw 16-bit little-endian PCM at 24 kHz for `response_format: "pcm"`,
  or a WAV whose header declares an open-ended length (players read to end of stream) for `wav`.
  The first chunk is cut at the first clause boundary so playback starts after one short synthesis;
  `hypnos_tts_ttfa_ms` on `/metrics` tracks the time to first audio.
- **Document conversion** (Hypnos extension): `POST /v1/documents/convert` takes a multipart
  upload (`file`) or a JSON body (`file` as base64 or a data URI, or `url`) and returns
  `{"object": "document", "pages": [...], "markdown": "...", "usage": {...}}`. Options: `pages`
  (`"3"`, `"2-5"`), `ocr` (default true: pages without a text layer are read with OCR),
  `output_format` (`json` adds each page's docling document), `stream` (one server-sent event per
  page, then the joined document), `describe_pictures` (a vision model such as `gemma4:e4b-q4`
  writes a description under every picture) and `return_pictures` (the pictures docling cut out
  stay on the server: the response carries the document `id` and each page its `picture_urls`;
  `GET /v1/documents/{id}/pictures` lists them with mime type, size, page, bounding box and
  caption, `GET /v1/documents/{id}/pictures/{n}` returns one as JSON with the base64 image, or the
  image itself with `?format=image`, and `DELETE /v1/documents/{id}` releases them; otherwise they
  expire after `server.document_pictures_ttl_secs`, 1 hour, within a store capped at
  `server.document_pictures_max_bytes`). PDFs are converted one page at a time through models
  that stay loaded (about 0.1 s a page on an M5 Max CPU; the first table loads TableFormer), so
  memory is the same for 5 pages or 500; the joined Markdown puts back tables, numbered lists and
  sentences that a page break cut. Images, DOCX, PPTX, XLSX, HTML and Markdown work too.
  `limits.max_document_bytes` (100 MiB) caps the input. `device = "gpu"` runs docling on CoreML
  (Metal builds) or CUDA (the `docling-cuda` feature: ONNX Runtime's CUDA provider, built by the
  run scripts, the release archives and the Docker image; at run time it needs driver 580+, the
  CUDA 13 runtime and cuDNN 9 for CUDA 13, see the release table above, or
  `docker build --build-arg DOCLING_CUDA=1`) and the fp32 models (`./run.sh models docling-gpu`).
  Measured on an RTX 5090 laptop against its CPU (`docs/bench/docling-cuda-2026-10-30.md`): pages
  of text go from 133 to 49 ms, a scanned page through OCR from 2.1 to 1.75 s, but a page with two
  tables from 0.9 to 1.07 s (TableFormer's decoder is latency-bound), for 5.7 GB of VRAM booked.
  Worth it for layout- and OCR-heavy documents on a GPU with room to spare; on Apple Silicon the
  CPU is faster and lighter, CoreML only frees the CPU cores.
- **Entity extraction** (Hypnos extension): `POST /v1/entities` with `{"model", "input": text or
  [texts], "labels": ["person", "medication", ...], "threshold": 0.5, "flat": true}` returns one
  entity list per text (`text`, `label`, character `start` / `end`, `score`). The labels are free
  text: GLiNER2.5 is zero-shot and multilingual. More than 8 labels run as separate passes so they
  do not compete for the same spans, and texts longer than 512 words are read in overlapping
  windows (1,700 words with 11 labels: 1.5 s on an M5 Max CPU).
- **Benchmark**: `backend/scripts/bench.sh [--host URL] [--out docs/bench/<name>.md]` runs the
  fixed execution scenarios from `docs/execution-efficiency-plan.md` (cold chat, five-turn chat,
  vision chat, concurrent clients, TTS, voice round trip) against a running server and prints a
  Markdown report; the current baseline is `docs/bench/baseline-2026-10-29.md`. Scenario 7
  (`--scenarios 7 [--chart out.svg]`, not in the default set) stages a model in Tier 3, 2 and 1
  with `POST /v1/memory/tier` and times the first request from each; before the Tier 3 runs it
  evicts the weight files (`--swap-files`) from the OS page cache on Linux and macOS.

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
│       ├── hypnos-backends/     # Native llama.cpp+mtmd, stable-diffusion.cpp, ONNX Runtime, docling & GLiNER engines
│       └── hypnos-server/       # Axum HTTP/SSE OpenAI Server & Multi-Modal Lifecycle Tests
├── sdk/                         # Client SDKs: python/, javascript/, rust/ (tested against a mock server)
├── desktop/                     # Tauri 2 + Svelte 5 desktop app: chat + live stats panel (desktop/README.md)
└── frontend/                    # Svelte 5 (Runes) + Tailwind CSS Interactive Playground
    ├── package.json
    ├── vite.config.js
    └── src/
        └── App.svelte           # Live 3-Tier Visualizer + Modality Playground tabs
```

---

## License

Hypnos's source code is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT), at your option. Unless you explicitly state otherwise, any contribution
you intentionally submit for inclusion in this work, as defined in the Apache-2.0 license, is dual
licensed as above, without any additional terms or conditions.

A `hypnos` binary built with the default engines statically links espeak-ng (GPL-3.0-or-later) for
text-to-speech and is therefore distributed under the GPL v3 or later as a whole. Model weights are
downloaded separately and are covered by their own licenses. See
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md) for every engine and its license; the GPL text is
[licenses/GPL-3.0.txt](licenses/GPL-3.0.txt).
