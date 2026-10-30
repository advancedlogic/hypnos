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
| 2  | 6   | 5   | 4   | 6    | 1   | 7     | 6   | 3    |

---

## Backlog

- [ ] 🟩 **ENG-4** `P2` Add audio.cpp for text-to-speech and optional speech-to-text
      Vendor https://github.com/0xShug0/audio.cpp (pin + patches in `scripts/vendor.sh`) and link
      it in-process as a new engine for `POST /v1/audio/speech`, next to Kokoro. Optionally offer
      it for `POST /v1/audio/transcriptions` as an alternative to the llama.cpp/mtmd path,
      selected per model in the config. Check its ggml does not clash with llama.cpp's (build
      shared like sd.cpp if needed), then add models to `models.manifest` and the example configs.
- [ ] ⬛ **BUILD-5** `P2` Publish the website on hypnos.maai.solutions (AWS S3)
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
- [ ] 🟨 **TEST-1** `P1` Soak test: many clients for hours, memory and bookings must stay flat
      All measurements so far are single-client scenarios (`backend/scripts/bench.sh`). Run the
      bench scenarios in a loop with 4-8 concurrent clients across modalities for several hours
      against a release build, sampling `/v1/memory/status` (process footprint, VRAM booked vs.
      real free, unaccounted), `/metrics` (promotions, evictions, load failures, rejections),
      the picture store and the request-log ring. Pass: no growth after warm-up, no deadlock, no
      429 storm, every eviction explained. Script it as `backend/scripts/soak.sh` with a report
      like `docs/bench/`, and fix what it finds (file them as BUG cards).
- [ ] 🟨 **TEST-2** `P2` Cover the engine failure paths and contain engine crashes
      The CUDA lifecycle test (`multi_modal_lifecycle.rs`) runs only by hand with real weights.
      Add engine-free tests for CUDA OOM mid-generation (mock backend failing on the n-th token:
      the lease is released, the slot cleared, the request answers 5xx), a weights file that
      changes or disappears under a resident model, and cancellation during a tier transition.
      A C++ abort inside llama.cpp / sd.cpp takes the whole process down: document the
      supervisor expectation (run scripts, Docker restart policy, systemd unit) and add a
      per-engine crash counter to `/metrics` so restarts are visible.
- [ ] 🟪 **API-5** `P2` Per-key rate limits and quotas, and redaction in the request log
      API keys are all-or-nothing. Add per-key limits in `[server]` (requests per minute,
      concurrent requests, tokens per day) enforced in `auth::require_api_key` with 429 and
      `Retry-After`, a key label in the request log and `/metrics`, and an audit line per
      rejected call. The request log captures prompts and completions at `debug`: add
      `request_log_redact` (patterns or "all") and keep `info` the documented level for anything
      exposed beyond a trusted group. Until then the README should say to put a gateway in front.
- [ ] 🟫 **MEM-2** `P3` Expose GPU device health on /metrics (utilisation, temperature, errors)
      `/metrics` covers bookings and placements but not the device itself. Through the existing
      probe path (`GpuMemoryProbe`, NVML / `nvidia-smi` on CUDA, IOKit on Apple Silicon) export
      utilisation, temperature, power, clock throttling and Xid / driver error counts, sampled
      with the VRAM reconcile; show the first three in the playground's tier visualizer header.
- [ ] 🟫 **MEM-3** `P3` Version the config and the model manifest with a migration path
      `HypnosConfig` has no schema version; the first breaking change to a field or to
      `models.manifest` will break every deployed `hypnos.toml`. Add `config_version` (absent =
      1) with per-version migration notes in the README, warnings for deprecated keys instead of
      silent acceptance, and a `hypnos --check-config` mode that prints what `validate` and the
      GGUF detection would do without starting the server.
- [ ] ⬜ **DOC-5** `P2` Write an operator's guide (sizing, knobs, telemetry, errors)
      `docs/operating.md` for people who run Hypnos rather than build it: sizing a box for a
      lineup (VRAM / pinned RAM budgets, slots x context), which knob to turn for which symptom
      (429s, evictions, slow first token, OOM), how to read `hypnos_telemetry` and the `/metrics`
      histograms, what every 4xx / 5xx and error `type` means and the fix, log levels and
      redaction, upgrade and rollback, and the Docker / systemd / Windows service recipes.

## In Progress

- [ ] ⬛ **BUILD-6** `P1` Cut the first release (v0.1.0) and prove the release pipeline
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

## Done

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
