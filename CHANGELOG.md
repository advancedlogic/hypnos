# Changelog

Notable changes per release. The ids in parentheses are the cards in [TODO.md](TODO.md), where
each one is described in full. Releases are on
[GitHub](https://github.com/advancedlogic/hypnos/releases); versions follow
[Semantic Versioning](https://semver.org) (before 1.0, a minor version may change the API or the
config, and its notes say so).

## Unreleased

### Fixed
- Multipart uploads above 2 MB were refused (documents, transcription, image edits): the routes
  now use the configured body limit (BUG-8).

### Added
- Desktop: the Models dialog's **In use** tab lists what the server runs, by job, with each
  model's tier, Thinking, "Make default for…", Enable / Disable, and Replace / Remove for catalog
  models; the Catalog groups the models to add by job (DESK-28, DESK-29).
- A user manual for the desktop app with screenshots, `docs/desktop-manual.md` (DOC-7).
- `SECURITY.md`, `CONTRIBUTING.md` and this changelog (DOC-9).

### Build and CI
- CI runs on every push to main and every pull request; a dependency audit (cargo-deny, npm
  audit, pip-audit), Dependabot, and every action pinned to a commit (BUILD-14, BUILD-15).
- The engines' CMake builds are reused between CI runs while their pins, patches and settings are
  unchanged, so a Rust-only change does not recompile llama.cpp and stable-diffusion.cpp
  (BUILD-13); nvcc goes through sccache on Windows too (BUILD-12).

## 0.7.0 (2026-11-01)

- A tool registry in the desktop app: the chat model hands work to the other models (web pages,
  reading aloud, entities, describing images), several calls in parallel (DESK-23).
- A collapsible blue tool panel with each call's input and output (DESK-27).
- The image tool takes size, exclusions, count, steps and seed from the conversation, edits the
  previous or an attached image, and the server returns each image's seed (DESK-22).
- FLUX.3 Image Klein in the catalog: Q4_K, Q8_0 and the gated BF16 (ENG-11).
- A Hugging Face token for downloads from gated repositories (DESK-21).
- The Models dialog lists the pool, makes a model the default for a job and sets each model's
  thinking level; the chat header no longer has model or thinking pickers (DESK-20, DESK-26).
- Rust SDK: `Client::with_headers` for per-task headers.

## 0.6.0 (2026-11-01)

- Gemma 4 E2B / 12B / 31B, Nemotron 3.5 Bolt E4B and LFM3-Omni 2B in the catalog (ENG-5).
- Qwen 3.5 / 3.6 / 3.8 and MiniCPM-V 4.6 chat with Qwen XML tool calls; Qwen3-ASR
  transcription (ENG-6).
- Per-model image defaults; Z-Image-Turbo, FLUX.2 klein 4B, ERNIE-Image-Turbo (ENG-7).
- Speech output through llama.cpp: LFM3-Audio, LFM3-Omni, Qwen3-TTS with voice cloning (ENG-8).
- Chat families: Qwen3-Coder, Nemotron 3.5 Lightning, GLM-4.7-Flash, LFM2.5; LFM3 tool calls
  (ENG-9).
- Pixel-space image models and custom sigma schedules; Iris-3B Turbo (ENG-10).
- `POST /v1/images/edits` with reference images, in the SDKs and the playground (API-13).
- The model catalog by use in the README and on the website.
- A faster release workflow: fewer CUDA targets, the Docker image from the Linux archive, sccache,
  timeouts (BUILD-11).
- Includes everything tagged 0.5.0, whose release run failed and was never published.

## 0.5.0 (2026-10-31, tagged, not published)

- LFM3 (Liquid AI) text, vision and audio models, with a test profile.
- Thinking levels (`reasoning_effort`) with returned thoughts, in the API, the SDKs, the
  playground and the desktop app.
- A curated model catalog (`GET /v1/models/catalog`) and catalog downloads with SHA-256 checks
  (API-10).
- A runtime model pool: add, replace and remove models while the server runs (API-11).
- A hardware and model advisor (`hypnos advise`, `POST /v1/advisor`) and a guided setup in the
  desktop app (API-12, DESK-18).
- Memory estimates measured and corrected (`hypnos measure-memory`, the exact KV layout) (TEST-3).
- Workload profiles, cost-aware eviction, declared next stages, deadline-aware admission (MEM-6,
  MEM-7, PERF-10).
- Hardening: a stress and cancellation test, upgrade compatibility, an SSRF guard on remote URLs
  (TEST-4).
- `run.sh` / `run.bat build all` (BUILD-10).

## 0.4.0 (2026-10-31)

- The macOS app bundles the Metal server and starts it with itself (DESK-16).
- First-run model downloads and a models folder setting in the desktop app (DESK-17).
- Kokoro runs the fp32 ONNX graph on every platform, fixing silent or crashing speech (BUG-6).
- The microphone entitlement for the hardened runtime.

## 0.3.0 (2026-10-31)

- The desktop app (Tauri 2): an Ollama-style chat, a collapsible stats panel, connection settings
  with the API key in the keychain, a local server it can start and stop, attachments (images,
  documents, audio, video), reading URLs, the microphone and reading answers aloud, the image
  tool, the menu bar icon, the request log viewer, regenerating any request (DESK-1..15).
- OpenAI tool calling on `/v1/chat/completions` (API-9).
- KV parking: a chat model's conversations kept across a swap (MEM-4); optional MoE expert
  offload (MEM-5); switches for both in the playground and the desktop app.
- Regenerate reuses the KV cache instead of a full prefill (PERF-8).
- Release builds keep their build dependencies unstripped on Xcode 27.2 (BUG-5).

## 0.2.0 (2026-10-30)

- Client SDKs for Python, JavaScript / TypeScript and Rust, with a mock server for their tests
  (API-7).
- Per-key rate limits, quotas and request-log redaction (API-5); enabling and disabling models
  without removing their config (API-3); the pictures docling extracts from a document (API-2).
- `config_version` with migrations, unknown-key warnings and `hypnos --check-config` (MEM-3);
  GPU device health on `/v1/memory/status` and `/metrics` (MEM-2).
- 2x faster Tier 2 to Tier 1 promotions on CUDA, with the load phases in the telemetry (PERF-6);
  26B defaults that keep the CUDA graph (PERF-7).
- The server listens before pre-staging its models; the default port is 9999 (API-4, API-6).
- Fixes: an eviction could unload a model between a fast-path lease and its request (BUG-1); a
  fast-path lease while the engine was still loading (BUG-2); a 507 and one retry when an engine
  runs out of device memory (BUG-3); a second server deleting the first one's pictures (BUG-4).

## 0.1.0 (2026-10-30)

- The first release: an OpenAI-compatible multi-modal server with the three-tier memory
  hierarchy (GPU, pinned RAM, NVMe) and a lease-based scheduler; chat, vision and audio through
  llama.cpp, images through stable-diffusion.cpp, speech through Kokoro, documents through
  docling.rs (ENG-1, ENG-3), entities through GLiNER2.5 (ENG-2); the playground.
- Native Apple Silicon support (Metal and unified memory).
- HTTPS and API-key authentication (API-1); the request log with per-call telemetry (UI-1); model
  defaults read from GGUF metadata (MEM-1); smaller machines simulated, a process memory cap.
