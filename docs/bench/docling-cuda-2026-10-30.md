# docling on CUDA vs the CPU (2026-10-30)

The measurement behind BUILD-4: whether document conversion on the GPU (`device = "gpu"` for the
`docling` model, ONNX Runtime's CUDA provider through the `docling-cuda` feature) gains enough
over the CPU path to ship in the release archives and the Docker image.

## Setup

- Laptop with an NVIDIA GeForce RTX 5090 Laptop GPU (24 GB), driver 591.91, CUDA 13.0 toolkit,
  cuDNN 9 for CUDA 13 (the `nvidia-cudnn-cu13` pip wheel on `PATH`), Windows 11.
- Hypnos at commit 1ba8df3 plus the BUILD-4 changes, `cargo build --release --features docling-cuda`
  (ONNX Runtime 1.31 prebuilt by ort, `x86_64-pc-windows-msvc+cuda13,tensorrt,nvrtx`).
- CPU path: `device = "cpu"`, the int8 layout model and fp16 TableFormer encoder (`./run.sh models
  docling`). GPU path: `device = "gpu"`, the fp32 graphs (`./run.sh models docling-gpu`),
  `activation_workspace_bytes = 5 GiB`.
- Three generated inputs, each converted three times after the models were loaded (the numbers
  are the second and third runs, `ms_per_page` from `hypnos_telemetry.engine`):
  - `text.pdf`: 10 pages of a heading, 38 lines of prose and one small picture (layout model only);
  - `tables.pdf`: 10 pages with two ruled 6 x 12 tables each plus a picture (layout + TableFormer,
    20 tables);
  - `scan.png`: one 1654 x 2339 image of 48 lines of text (layout + PP-OCRv6 detection and
    recognition).

## Results

| Input | CPU | CUDA | CUDA / CPU |
| :--- | ---: | ---: | ---: |
| text page (layout only) | 133 ms | 49 ms | 0.37x |
| scanned page (OCR) | 2.1 s | 1.75 s | 0.83x |
| page with two tables (TableFormer) | 0.90 s | 1.07 s | 1.19x |

| Memory | CPU | CUDA |
| :--- | ---: | ---: |
| process footprint while converting | 0.7-1.2 GB | 1.4 GB |
| VRAM booked by the planner | 0 | 5.7 GB |
| model load (first request) | 0.6 s | 1.5 s |

The table pages were also run with the CUDA arena settings the engine used to force:
`DOCLING_RS_CUDA_ARENA_EXTEND=same_as_requested` gave 1.27 s a page; the default power-of-two growth
1.07 s; lifting the per-session memory cap changed nothing. The engine now keeps the power-of-two
growth under the cap. TableFormer's decoder runs one small step per table cell with a KV cache that
grows each step, so it is bound by launch latency and synchronisation rather than compute, and the
GPU does not help it. The layout model (a single convolutional pass per page) and the OCR
recognition batches are where the GPU wins.

## Decision

Ship the CUDA provider: it costs 90 MB in the archive and nothing at run time unless a docling
model is configured for the GPU (verified: the CPU path runs with the provider libraries removed,
and `device = "gpu"` without them fails at load with ONNX Runtime naming the missing library). A
user who sets `device = "gpu"` additionally needs driver 580+, the CUDA 13 runtime and cuDNN 9 for
CUDA 13, documented in the archive's `README.txt` and the README's release table. The Docker image
carries the provider and gets the CUDA 13 runtime and cuDNN only with `--build-arg DOCLING_CUDA=1`
(about 1.5 GB more image). Kokoro and GLiNER were checked on the same ONNX Runtime build.

Worth switching on for layout- and OCR-heavy workloads (scans, long reports) on a GPU with 6 GB to
spare; table-heavy documents are faster on the CPU, and on Apple Silicon the CPU path remains the
better one (see `docs/apple-silicon.md`).

## Reproducing

```bash
./run.sh models docling-gpu                     # fp32 layout model and TableFormer encoder
cargo build --release -p hypnos-server --features docling-cuda   # what run.sh build does on CUDA hosts
# config: device = "gpu", weights_path = "models/docling/layout_heron.onnx",
#         arch.activation_workspace_bytes = 5368709120
# PATH: cudart64_13 / cublas64_13 (CUDA 13 toolkit) and cudnn64_9 + cudnn_*64_9 (cuDNN 9 for CUDA 13)
curl -F model=docling -F file=@tables.pdf http://localhost:8080/v1/documents/convert | jq .hypnos_telemetry.engine
```
