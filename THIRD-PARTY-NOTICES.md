# Third-Party Notices

Hypnos's own source code is licensed under either of [Apache License, Version 2.0](LICENSE-APACHE)
or [MIT license](LICENSE-MIT), at your option. The engines it compiles and links into the `hypnos`
binary keep their own licenses, listed below.

## Native engines (`backend/vendor/`)

The sources are fetched by `backend/scripts/vendor.sh` at the pinned versions; each ships its
upstream license file except espeak-ng, whose `COPYING` is left out of the `espeak-rs-sys` crate
package (the GPL-3.0-or-later notice is in each source file's header).

| Component | Used for | License |
| :--- | :--- | :--- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) (incl. ggml, libmtmd) | text, vision, audio-in, embeddings | MIT |
| [llama-cpp-rs](https://github.com/utilityai/llama-cpp-rs) | Rust bindings for llama.cpp | MIT OR Apache-2.0 |
| [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) | image generation | MIT |
| [diffusion-rs](https://github.com/newfla/diffusion-rs) | Rust bindings for stable-diffusion.cpp | MIT |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) via [`ort`](https://github.com/pykeio/ort) | Kokoro TTS, docling | MIT / MIT OR Apache-2.0 |
| [docling.rs](https://github.com/docling-project/docling.rs) (`docling`, `docling-pdf`, `docling-onnx`, crates.io) | document conversion | MIT |
| [espeak-ng](https://github.com/espeak-ng/espeak-ng) via `espeak-rs-sys` | grapheme-to-phoneme for TTS | **GPL-3.0-or-later** |

### espeak-ng and the GPL

espeak-ng is GPL-3.0-or-later; the `espeak-rs-sys` crate (MIT) builds it from source and links it
statically. Hypnos's code stays MIT OR Apache-2.0, both of which are GPL-3.0 compatible, but a
`hypnos` binary built with the default engines is a combined work that includes espeak-ng and is
therefore distributed under the terms of the **GNU GPL v3 or later**. Anyone redistributing such a
binary must provide the GPL text ([licenses/GPL-3.0.txt](licenses/GPL-3.0.txt)) and the corresponding
source (this repository plus the vendored sources at the pins in `backend/scripts/vendor.sh`). The
release archives built by `backend/scripts/package-release.sh` ship both license texts, these
notices and a link to their source tag. The phoneme data copied next to the binary
(`espeak-ng-data/`) is part of espeak-ng and under the same license.

## Rust crates

All other Rust dependencies (see `backend/Cargo.lock`) are under permissive licenses: MIT,
Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC, Zlib, Unicode-3.0, Unlicense, BSL-1.0, NCSA,
CDLA-Permissive-2.0 or dual/multi-licensed combinations of these. The exception is
`cssparser`, `cssparser-macros`, `dtoa-short` and `selectors` (docling's HTML reader), which are
MPL-2.0: unmodified, and their source is available from crates.io. Regenerate the list with `cargo metadata`, or
`cargo about` / `cargo deny check licenses` if installed.

## Frontend

The playground (`frontend/`) depends on Svelte, Vite and Tailwind CSS, all MIT licensed.

## Model weights

Model files are not part of this repository. `./run.sh models` downloads them from their Hugging
Face repositories and docling.rs's `models-v1` GitHub release (listed in the README and
`backend/models.manifest`), and each is governed by the license or terms of use published there.
The docling models are ONNX exports of docling-project's layout model (`docling-layout-heron`,
Apache-2.0) and TableFormer (`docling-models`, CDLA-Permissive-2.0 / Apache-2.0), plus RapidOCR's
PP-OCRv6 detector and recognizer (Apache-2.0). Check them before using a model commercially or
redistributing it.
