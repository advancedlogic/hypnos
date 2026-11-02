# Card index

Work is tracked in [GitHub issues](https://github.com/advancedlogic/hypnos/issues) on the
[Hypnos board](https://github.com/users/advancedlogic/projects/4) since 2026-11-01. This file
used to be the card board; every card became an issue (the done ones closed), with its id in the
title. The `(XY-n)` in older commits and in [CHANGELOG.md](CHANGELOG.md) point here.

| Card | Issue | | Title |
|------|-------|-|-------|
| UI-1 | [#18](https://github.com/advancedlogic/hypnos/issues/18) | done | Add request log inspector panel with per-call telemetry |
| UI-2 | [#74](https://github.com/advancedlogic/hypnos/issues/74) | done | Playground: switches for KV parking and MoE offload |
| API-1 | [#19](https://github.com/advancedlogic/hypnos/issues/19) | done | HTTPS and API-key authentication for the HTTP server |
| API-2 | [#31](https://github.com/advancedlogic/hypnos/issues/31) | done | Return the pictures docling extracts from a document |
| API-3 | [#33](https://github.com/advancedlogic/hypnos/issues/33) | done | Allow enabling and disabling models without removing their config |
| API-4 | [#35](https://github.com/advancedlogic/hypnos/issues/35) | done | Listen before startup pre-staging so /ready answers 503 instead of nothing |
| API-5 | [#37](https://github.com/advancedlogic/hypnos/issues/37) | done | Per-key rate limits and quotas, and redaction in the request log |
| API-6 | [#48](https://github.com/advancedlogic/hypnos/issues/48) | done | Choose the port on the command line; default 9999 |
| API-7 | [#51](https://github.com/advancedlogic/hypnos/issues/51) | done | Client SDKs in Python, Rust and JavaScript |
| API-8 | [#118](https://github.com/advancedlogic/hypnos/issues/118) | open | Publish the SDKs to PyPI, npm and crates.io |
| API-9 | [#68](https://github.com/advancedlogic/hypnos/issues/68) | done | Tool calling on `/v1/chat/completions` |
| API-10 | [#80](https://github.com/advancedlogic/hypnos/issues/80) | done | Hugging Face model inventory so users can bring their own models |
| API-11 | [#81](https://github.com/advancedlogic/hypnos/issues/81) | done | Add, remove and replace models at runtime (a dynamic model pool) |
| API-12 | [#82](https://github.com/advancedlogic/hypnos/issues/82) | done | Hardware and model advisor |
| API-13 | [#97](https://github.com/advancedlogic/hypnos/issues/97) | done | `POST /v1/images/edits` (OpenAI images edit) |
| API-14 | [#122](https://github.com/advancedlogic/hypnos/issues/122) | open | The agent loop on the server |
| API-15 | [#128](https://github.com/advancedlogic/hypnos/issues/128) | open | Tracing with OpenTelemetry |
| API-16 | [#127](https://github.com/advancedlogic/hypnos/issues/127) | open | Graceful shutdown and upgrades without dropped requests |
| ENG-1 | [#28](https://github.com/advancedlogic/hypnos/issues/28) | done | Add document conversion with docling.rs (PDF / images to Markdown) |
| ENG-2 | [#29](https://github.com/advancedlogic/hypnos/issues/29) | done | Add GLiNER zero-shot named-entity extraction |
| ENG-3 | [#30](https://github.com/advancedlogic/hypnos/issues/30) | done | Run document conversion on the GPU (Core ML, CUDA) |
| ENG-4 | [#119](https://github.com/advancedlogic/hypnos/issues/119) | open | Add audio.cpp for text-to-speech and optional speech-to-text |
| ENG-5 | [#92](https://github.com/advancedlogic/hypnos/issues/92) | done | Catalog: the other Gemma 4 sizes, Nemotron 3.5 Bolt E4B, LFM3-Omni 2B |
| ENG-6 | [#93](https://github.com/advancedlogic/hypnos/issues/93) | done | Qwen chat format: Qwen 3.5 / 3.6 / 3.8 and the models built on them |
| ENG-7 | [#94](https://github.com/advancedlogic/hypnos/issues/94) | done | Image models on the existing sd.cpp file set; per-model sampling defaults |
| ENG-8 | [#95](https://github.com/advancedlogic/hypnos/issues/95) | done | Speech output through llama.cpp (LFM3-Audio, Qwen3-TTS) |
| ENG-9 | [#96](https://github.com/advancedlogic/hypnos/issues/96) | done | More chat formats: Mistral, Nemotron, GLM, LFM2, Qwen 3 |
| ENG-10 | [#98](https://github.com/advancedlogic/hypnos/issues/98) | done | Image engine: optional VAE and other encoders; Parakeet transcription |
| ENG-11 | [#101](https://github.com/advancedlogic/hypnos/issues/101) | done | FLUX.3 Image Klein in the catalog |
| ENG-12 | [#129](https://github.com/advancedlogic/hypnos/issues/129) | open | Survive an engine crash |
| MEM-1 | [#23](https://github.com/advancedlogic/hypnos/issues/23) | done | Read `[models.arch]` defaults from GGUF metadata |
| MEM-2 | [#38](https://github.com/advancedlogic/hypnos/issues/38) | done | Expose GPU device health on /metrics (utilisation, temperature, errors) |
| MEM-3 | [#39](https://github.com/advancedlogic/hypnos/issues/39) | done | Version the config and the model manifest with a migration path |
| MEM-4 | [#71](https://github.com/advancedlogic/hypnos/issues/71) | done | Keep each slot's KV state across a demotion |
| MEM-5 | [#73](https://github.com/advancedlogic/hypnos/issues/73) | done | Expert offload for MoE models: experts on the CPU, the rest on the GPU |
| MEM-6 | [#86](https://github.com/advancedlogic/hypnos/issues/86) | done | Make scheduling workload-aware, not only memory-aware |
| MEM-7 | [#85](https://github.com/advancedlogic/hypnos/issues/85) | done | Add workload profiles with latency objectives |
| PERF-1 | [#14](https://github.com/advancedlogic/hypnos/issues/14) | done | Execution telemetry, per-model `/metrics` histograms, bench script and baseline |
| PERF-2 | [#15](https://github.com/advancedlogic/hypnos/issues/15) | done | Engine worker: persistent llama.cpp context, slots, prefix reuse |
| PERF-3 | [#16](https://github.com/advancedlogic/hypnos/issues/16) | done | Streaming text-to-speech |
| PERF-4 | [#17](https://github.com/advancedlogic/hypnos/issues/17) | done | Continuous batching across execution slots |
| PERF-5 | [#22](https://github.com/advancedlogic/hypnos/issues/22) | done | Benchmark model swap latency: cold vs warm vs hot |
| PERF-6 | [#40](https://github.com/advancedlogic/hypnos/issues/40) | done | Tier 2 -> Tier 1 promotion of the 26B takes 23-59 s on Windows CUDA |
| PERF-7 | [#47](https://github.com/advancedlogic/hypnos/issues/47) | done | Gemma 4 26B decodes at 12 tokens/s on the RTX 5090 laptop; VRAM is full |
| PERF-8 | [#72](https://github.com/advancedlogic/hypnos/issues/72) | done | A repeated prompt (Regenerate) is prefilled in full |
| PERF-9 | [#117](https://github.com/advancedlogic/hypnos/issues/117) | open | Three concurrent streams on the 26B decode at 34 tokens/s each on CUDA |
| PERF-10 | [#87](https://github.com/advancedlogic/hypnos/issues/87) | done | Deadline-aware admission and predictable latency |
| BUG-1 | [#42](https://github.com/advancedlogic/hypnos/issues/42) | done | Eviction can unload a model between a fast-path lease and its request |
| BUG-2 | [#44](https://github.com/advancedlogic/hypnos/issues/44) | done | A request could run on an engine that was still loading |
| BUG-3 | [#50](https://github.com/advancedlogic/hypnos/issues/50) | done | An engine that runs out of device memory mid-request answers 500 |
| BUG-4 | [#52](https://github.com/advancedlogic/hypnos/issues/52) | done | A second server on the same machine deletes the first one's pictures |
| BUG-5 | [#66](https://github.com/advancedlogic/hypnos/issues/66) | done | Release builds on macOS 27 / Xcode 27.2 cannot load stripped proc macros |
| BUG-6 | [#79](https://github.com/advancedlogic/hypnos/issues/79) | done | Kokoro's fp16 graph returns all-NaN audio on Windows too; use fp32 everywhere |
| BUG-7 | [#89](https://github.com/advancedlogic/hypnos/issues/89) | done | Tools for an LFM3 model fail mid-stream instead of a 400 (API, desktop) |
| BUG-8 | [#106](https://github.com/advancedlogic/hypnos/issues/106) | done | Multipart uploads above 2 MB were refused (documents, transcription, image edits) |
| BUG-9 | [#114](https://github.com/advancedlogic/hypnos/issues/114) | done | A crafted GGUF header crashed the server (stack overflow) |
| BUILD-1 | [#26](https://github.com/advancedlogic/hypnos/issues/26) | done | Publish prebuilt release binaries |
| BUILD-2 | [#27](https://github.com/advancedlogic/hypnos/issues/27) | done | Provide a Docker image with CUDA support |
| BUILD-3 | [#25](https://github.com/advancedlogic/hypnos/issues/25) | done | Re-enable CI |
| BUILD-4 | [#34](https://github.com/advancedlogic/hypnos/issues/34) | done | Ship docling on CUDA in the release archives and the Docker image |
| BUILD-5 | [#36](https://github.com/advancedlogic/hypnos/issues/36) | done | Publish the website on hypnos.maai.solutions (AWS S3) |
| BUILD-6 | [#46](https://github.com/advancedlogic/hypnos/issues/46) | done | Cut the first release (v0.1.0) and prove the release pipeline |
| BUILD-7 | [#49](https://github.com/advancedlogic/hypnos/issues/49) | done | `run.sh` reports a backend that dies on start as running |
| BUILD-8 | [#57](https://github.com/advancedlogic/hypnos/issues/57) | done | Build the desktop app from the run scripts and CI |
| BUILD-9 | [#67](https://github.com/advancedlogic/hypnos/issues/67) | done | Sign and notarize the desktop installers |
| BUILD-10 | [#90](https://github.com/advancedlogic/hypnos/issues/90) | done | `run.sh build all` / `run.bat build all` builds backend, playground and desktop |
| BUILD-11 | [#91](https://github.com/advancedlogic/hypnos/issues/91) | done | Cut the release workflow from ~3 h to under 1 h |
| BUILD-12 | [#116](https://github.com/advancedlogic/hypnos/issues/116) | in progress | Cut the Windows CUDA release build (3h10m) to the Linux one's time |
| BUILD-13 | [#115](https://github.com/advancedlogic/hypnos/issues/115) | in progress | Recompile llama.cpp and sd.cpp only when something that shapes them changed |
| BUILD-14 | [#110](https://github.com/advancedlogic/hypnos/issues/110) | done | CI on pushes and pull requests |
| BUILD-15 | [#111](https://github.com/advancedlogic/hypnos/issues/111) | done | Supply-chain checks: dependency audit, update PRs, pinned actions |
| BUILD-16 | [#133](https://github.com/advancedlogic/hypnos/issues/133) | open | Release provenance: checksums, SBOM, attestations, a signed image |
| BUILD-17 | [#134](https://github.com/advancedlogic/hypnos/issues/134) | open | Major dependency upgrades, done on purpose |
| DOC-1 | [#20](https://github.com/advancedlogic/hypnos/issues/20) | done | Add a LICENSE file to the repository |
| DOC-2 | [#21](https://github.com/advancedlogic/hypnos/issues/21) | done | Rewrite README around the pitch, a quickstart and bring-your-own models |
| DOC-3 | [#24](https://github.com/advancedlogic/hypnos/issues/24) | done | Record a demo GIF of the 3-tier visualizer for the README |
| DOC-4 | [#32](https://github.com/advancedlogic/hypnos/issues/32) | done | Create the Hypnos website as a single page under `website/` |
| DOC-5 | [#41](https://github.com/advancedlogic/hypnos/issues/41) | done | Write an operator's guide (sizing, knobs, telemetry, errors) |
| DOC-6 | [#58](https://github.com/advancedlogic/hypnos/issues/58) | done | Document the desktop app |
| DOC-7 | [#107](https://github.com/advancedlogic/hypnos/issues/107) | done | A user manual for the desktop app, with screenshots |
| DOC-8 | [#126](https://github.com/advancedlogic/hypnos/issues/126) | open | Operating in production: alerts, dashboard, runbooks, support matrix |
| DOC-9 | [#112](https://github.com/advancedlogic/hypnos/issues/112) | done | SECURITY.md, CHANGELOG.md, CONTRIBUTING.md |
| TEST-1 | [#45](https://github.com/advancedlogic/hypnos/issues/45) | done | Soak test: many clients for hours, memory and bookings must stay flat |
| TEST-2 | [#43](https://github.com/advancedlogic/hypnos/issues/43) | done | Cover the engine failure paths and contain engine crashes |
| TEST-3 | [#84](https://github.com/advancedlogic/hypnos/issues/84) | done | Measure booked vs real memory per engine and fix the estimates |
| TEST-4 | [#88](https://github.com/advancedlogic/hypnos/issues/88) | done | Harden for operational reliability |
| TEST-5 | [#121](https://github.com/advancedlogic/hypnos/issues/121) | open | Starvation under sustained load; real-engine drills and reproducible releases |
| TEST-6 | [#113](https://github.com/advancedlogic/hypnos/issues/113) | done | Tests at real sizes, and fuzzing of the untrusted parsers |
| TEST-7 | [#131](https://github.com/advancedlogic/hypnos/issues/131) | open | Desktop UI smoke tests in CI |
| TEST-8 | [#130](https://github.com/advancedlogic/hypnos/issues/130) | open | A soak test: hours of mixed traffic |
| DESK-1 | [#53](https://github.com/advancedlogic/hypnos/issues/53) | done | Scaffold the Tauri 2 desktop app in `desktop/` |
| DESK-2 | [#54](https://github.com/advancedlogic/hypnos/issues/54) | done | Chat view in the style of Ollama |
| DESK-3 | [#55](https://github.com/advancedlogic/hypnos/issues/55) | done | Collapsible right panel with live Hypnos stats |
| DESK-4 | [#56](https://github.com/advancedlogic/hypnos/issues/56) | done | Connection settings and server states |
| DESK-5 | [#59](https://github.com/advancedlogic/hypnos/issues/59) | done | Start and stop a local Hypnos server from the app |
| DESK-6 | [#60](https://github.com/advancedlogic/hypnos/issues/60) | done | Route every server call through the Rust SDK in `src-tauri` |
| DESK-7 | [#61](https://github.com/advancedlogic/hypnos/issues/61) | done | Attach images, documents, audio and video to a message |
| DESK-8 | [#62](https://github.com/advancedlogic/hypnos/issues/62) | done | Extract a URL's content into the chat, and download it |
| DESK-9 | [#63](https://github.com/advancedlogic/hypnos/issues/63) | done | Microphone button: speak a message |
| DESK-10 | [#64](https://github.com/advancedlogic/hypnos/issues/64) | done | Speaker button: read answers aloud |
| DESK-11 | [#65](https://github.com/advancedlogic/hypnos/issues/65) | done | Image-generation tool in the chat |
| DESK-12 | [#69](https://github.com/advancedlogic/hypnos/issues/69) | done | Menu bar / system tray icon |
| DESK-13 | [#70](https://github.com/advancedlogic/hypnos/issues/70) | done | Request log viewer, as in the playground |
| DESK-14 | [#75](https://github.com/advancedlogic/hypnos/issues/75) | done | Desktop: switches for KV parking and MoE offload |
| DESK-15 | [#76](https://github.com/advancedlogic/hypnos/issues/76) | done | Regenerate from any request or answer |
| DESK-16 | [#77](https://github.com/advancedlogic/hypnos/issues/77) | done | Bundle the server inside the macOS app, as Ollama does |
| DESK-17 | [#78](https://github.com/advancedlogic/hypnos/issues/78) | done | First-run model download in the desktop app |
| DESK-18 | [#83](https://github.com/advancedlogic/hypnos/issues/83) | done | Guided setup: detect, recommend, install, demonstrate |
| DESK-19 | [#120](https://github.com/advancedlogic/hypnos/issues/120) | open | Test on real 16 GB and 32 GB Macs and ship a signed, notarized release |
| DESK-20 | [#99](https://github.com/advancedlogic/hypnos/issues/99) | done | The lineup shows the pool; choose the model for each job there |
| DESK-21 | [#100](https://github.com/advancedlogic/hypnos/issues/100) | done | A Hugging Face token for model downloads (gated repositories) |
| DESK-22 | [#102](https://github.com/advancedlogic/hypnos/issues/102) | done | The image tool takes its settings from the conversation |
| DESK-23 | [#103](https://github.com/advancedlogic/hypnos/issues/103) | done | A tool registry: the chat model orchestrates the specialist models |
| DESK-24 | [#123](https://github.com/advancedlogic/hypnos/issues/123) | open | Sub-agents as tools: research and documents first |
| DESK-25 | [#124](https://github.com/advancedlogic/hypnos/issues/124) | open | A `search_notes` tool over the chats and attached files |
| DESK-26 | [#104](https://github.com/advancedlogic/hypnos/issues/104) | done | No model or thinking picker in the chat header; thinking per model in Models |
| DESK-27 | [#105](https://github.com/advancedlogic/hypnos/issues/105) | done | A collapsible blue tool panel with each call's input and output |
| DESK-28 | [#108](https://github.com/advancedlogic/hypnos/issues/108) | done | The Catalog tab: pool and not-pool tabs, models grouped by job |
| DESK-29 | [#109](https://github.com/advancedlogic/hypnos/issues/109) | done | Models dialog: an "In use" tab from the server in place of Lineup |
| DESK-30 | [#132](https://github.com/advancedlogic/hypnos/issues/132) | open | Auto-update for the desktop app |
| DESK-31 | [#125](https://github.com/advancedlogic/hypnos/issues/125) | open | Show and accept a model's license before it downloads |
| DESK-32 | [#135](https://github.com/advancedlogic/hypnos/issues/135) | done | Bundle the server inside the Windows and Linux apps too |
| DESK-33 | [#136](https://github.com/advancedlogic/hypnos/issues/136) | open | The Linux AppImage with the server inside |
