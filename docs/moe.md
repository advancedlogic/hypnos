# Expert offload for Mixture-of-Experts models (proposal)

Status: implemented (MEM-5), off by default: per model `moe_offload = "off" | "auto" | "on"`, or
`POST /v1/models/{id}/moe-offload` at runtime (docs/api.md section 6). The CUDA benchmark of
step 1 is still to run; what was measured on Apple Silicon is in "Measured" below.

## Measured (Apple Silicon, Metal, 2026-10-30)

`gemma4:26b-q4`, debug build, `moe_offload = "on"` then switched to `off` through the API:

| | `on` (experts on the CPU) | `off` |
| :--- | :--- | :--- |
| Expert bytes found in the GGUF | 12,846,382,080 (11.96 GiB) | |
| Weights on the GPU | 2.76 GiB (llama.cpp `CPU_REPACK` buffer: 12,251 MiB) | 14.73 GiB |
| Decode | 46.6-47.7 tok/s | 98.8-112.2 tok/s |
| Prefill, 2.6k-token prompt | 303 tok/s | 4,130 tok/s |
| Answer at temperature 0 | identical | |

On unified memory the offload saves nothing (one pool) and llama.cpp runs the experts' prefill on
the CPU, hence the 14x; `auto` therefore never acts on unified memory. On a discrete GPU `op_offload`
copies expert weights to the GPU for large batches, so the prefill penalty should be smaller: that
is the number step 1 still has to produce on the RTX 5090 laptop. One finding while implementing:
the bindings' `add_cpu_moe_override` pattern `\.ffn_(up|down|gate)_(ch|)exps` does not match Gemma
4's fused `ffn_gate_up_exps`, so it would have left two thirds of the experts on the GPU; Hypnos uses
its own pattern.

Original proposal: This replaces an earlier draft
that compared per-token expert streaming with layer-wise offload. It rightly rejected the
streaming, but it missed the placement worth having: the expert weights stay in host memory and
are computed on the CPU, while everything else stays on the GPU.

## 1. The model

`gemma4:26b-q4`, read from its GGUF:

| | |
| :--- | :--- |
| Layers | 30 (5 full-attention, 25 sliding-window) |
| Experts per layer | 128, of which 8 are used per token (`expert_used_count`) |
| Expert FFN width | 704 (a dense FFN of 2112 runs beside the experts in every layer) |
| One expert, one layer | `gate_up` 2816 x 1408 + `down` 704 x 2816 = 5.9 M parameters, about 3.3 MB in Q4_0 |
| All expert tensors (`ffn_*_exps`) | about 12.0 GiB |
| Everything else (attention 0.58, dense FFN 0.32, embeddings and norms 0.73) | about 1.6 GiB |

So 88 % of the weights are experts, and a decoded token reads only 30 x 8 x 3.3 MB, about
0.8 GB of them.

## 2. Today: layer-wise placement

`LayerPlacementPlan` places whole layers: `gpu_layers` on the GPU, the rest (`pinned_ram_layers`,
`ssd_layers`) computed on the CPU from host memory. When the 26B and its KV cache do not fit next
to another resident model, the planner either splits by layers, which puts attention, the dense
FFN and the KV cache of the CPU layers on the CPU, or evicts the model and swaps it back later
(about 1 s on Metal to 3.7 s on CUDA from Tier 2, plus the prefill of the conversation, see
`docs/pagination.md`).

On Apple Silicon placement is all-or-nothing (unified memory); this proposal is for discrete GPUs.

## 3. What does not work: streaming experts on demand

Fetching each token's routed experts over PCIe when they are not on the GPU stalls every layer of
every token on a transfer, breaks the decode graph into dynamic pieces, and gets worse with
batching (four decoding sequences can touch up to 32 experts per layer, a prefill batch nearly
all 128). The earlier draft's conclusion here stands, although its numbers did not: an expert is
about 3.3 MB, not 250 MB, and pinned PCIe transfers run near 25 GB/s (the 5.5 GB/s figure in the
repo is the whole load path, disk read included).

## 4. Proposal: experts in host memory, computed on the CPU

llama.cpp can place tensors by name (`tensor_buft_overrides`); the vendored bindings already offer
`LlamaModelParams::add_cpu_moe_override()`, which keeps `ffn_(up|down|gate)_(ch|)exps` in a CPU
buffer. The GPU then holds attention, the dense FFN, the router, embeddings and the KV cache; the
CPU runs the eight routed experts of each token where their weights already are. Nothing crosses
the bus per token except activations. For large prefill batches llama.cpp's `op_offload` copies
expert weights to the GPU for the matrix multiplications, where the copy pays for itself.

| | Full GPU | Layer split | Experts on CPU (all layers) |
| :--- | :--- | :--- | :--- |
| VRAM for weights | ~13.6 GiB | proportional to `gpu_layers` | ~1.6 GiB |
| KV cache | GPU | GPU and CPU by layer | GPU |
| Host RAM for weights | 0 | the CPU layers | ~12 GiB |
| Decode bound by | GPU | the slowest CPU layers | RAM bandwidth (~0.8 GB per token) |
| Measured decode | ~150 tok/s (RTX 5090 laptop) | to measure | to measure |

The point for Hypnos is co-residency: with about 1.6 GiB of weights plus its KV cache on the GPU,
the 26B can stay resident beside MOSS-Image or docling on a 16-24 GB card instead of being evicted
for each image request. A partial version, experts on the CPU for the last N layers only, trades
VRAM for speed in finer steps than the layer split.

Costs:

- Decode drops to what host RAM bandwidth allows: a few tens of tok/s on dual-channel DDR5 is the
  plausible range, to be measured.
- Concurrency gains shrink: CPU expert work grows with the number of decoding sequences.
- About 12 GiB of host memory holds the experts. The planner must count it against host RAM
  (the experts are read for every token; pinning them, as Tier 2 does for weights, keeps them
  from being paged out).
- CPU threads: decode now competes with the other CPU engines (Kokoro, docling, GLiNER).

## 5. Steps

1. **Benchmark** on the RTX 5090 laptop with `backend/scripts/bench.sh` scenarios (single-client
   decode, 4-client aggregate, 8k prefill, TTFT) for: full GPU, the current layer split at the
   same VRAM as option 3, experts on CPU for all layers, and for the last 15 layers. Record VRAM
   and host RAM used. Stop here if the expert-on-CPU decode is not clearly better than the layer
   split at equal VRAM.
2. **Placement mode**: `LayerPlacementPlan` gains the number of layers whose experts sit on the
   CPU (`cpu_expert_layers`), the GGUF inspector reports expert tensor bytes per layer, the
   planner books them as host memory and only the rest of those layers as VRAM, and the llama
   backend adds the matching overrides (one pattern per layer, `blk\.(N|...)\.ffn_.*_exps`).
3. **Planner policy**: for a model with experts, prefer moving experts to the CPU over a layer
   split when it does not fit; consider it before evicting a model that is in use. A per-model
   config switch (`moe_offload = "auto" | "never" | "always"`) for operators who prefer the swap.
4. Telemetry and docs: the placement shows `cpu_expert_layers` in `/v1/memory/status` and the
   response telemetry; `docs/operating.md` explains when it kicks in and what it costs.

## 6. Out of scope

- Per-token streaming of experts (section 3).
- Frequency-based expert caching on the GPU ("hot experts in VRAM"): it needs routing statistics
  and custom kernels; revisit only if the benchmark shows a large gap worth closing.
- Apple Silicon, where memory is shared and the question does not arise.
