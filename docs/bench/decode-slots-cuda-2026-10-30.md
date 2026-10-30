# Decode speed vs execution slots on CUDA (2026-10-30)

PERF-7: `gemma4:26b-q4` decoded at 12 tokens/s on an RTX 5090 laptop with the default lineup,
while the README's number for a 24 GB CUDA GPU is about 100. The first suspicion was a full
device (the planner's bookings plus 3.9 GB unaccounted left 0 bytes really free). It was not: the
decode rate depends on the execution-slot configuration, and a few combinations make llama.cpp
re-capture its CUDA graph on every step.

## Setup

- RTX 5090 Laptop GPU (24 GB), driver 591.91, Windows 11, CUDA 12.8 build, llama.cpp b11982.
- `gemma4:26b-q4` (Q4_0, 15.8 GB weights, 1.2 GB vision projector) fully on the GPU, flash
  attention on (auto), `kv_unified = false`, `n_seq_max = slots`, `n_batch = 2048`,
  `swa_full = false`. The embedder and Kokoro are pinned in VRAM as in the default lineup.
- One client, three consecutive requests of about 140 completion tokens each; `decode_tokens_per_s`
  from `hypnos_telemetry.engine`, runs 2 and 3 (run 1 pays the promotion and the first prefill).
- The server is restarted for every row with the slot count, the context per slot and the KV cache
  type changed in the 26B block; everything else as in `hypnos.example.toml`.

## Results

| slots x context (total KV positions) | KV cache | decode tok/s | prefill tok/s | VRAM in use (nvidia-smi) |
| :--- | :--- | ---: | ---: | ---: |
| 1 x 16384 (16k) | q8_0 | 152-154 | 663-677 | 23.6 GB |
| 1 x 65536 (64k) | q8_0 | 142 | 578-593 | 23.4 GB |
| 4 x 4096 (16k) | q8_0 | 146-148 | 621-678 | 23.5 GB |
| 2 x 16384 (32k) | q8_0 | **27** | 285-293 | 22.7 GB |
| 4 x 8192 (32k) | q8_0 | **15** | 165-178 | 23.6 GB |
| 4 x 16384 (64k), the old default | q8_0 | **8-10** | 84-113 | 21.9 GB |
| 4 x 16384 (64k), CUDA graphs disabled (`GGML_CUDA_DISABLE_GRAPHS=1`) | q8_0 | 68-71 | 593-647 | 23.5 GB |
| 2 x 16384 (32k) | f16 | 151-154 | 644-659 | 23.3 GB |
| 3 x 16384 (48k) | f16 | 151-152 | 660-663 | 23.4 GB |
| 4 x 8192 (32k) | f16 | 156-157 | 673-678 | 23.4 GB |
| 4 x 16384 (64k) | f16 | **12-13** | 124-128 | 23.3 GB |

## Reading

- The device being nearly full is not the cause: every row sits at 23-24 GB used, including the
  150 tok/s ones.
- The slow rows are the ones where a graph is re-captured: with CUDA graphs disabled the old
  default goes from 8 to 70 tokens/s (the plain launch overhead of a 30-layer MoE model), and the
  fast rows run with graphs and reach 150. Which combinations trip the re-capture follows the
  shape of the KV cache, not its size: a single stream of 64k is fine, four streams of 4k are
  fine, two or more streams of 16k with a q8_0 cache are not, and four streams of 16k are not
  even with f16. Something in llama.cpp's per-step graph (the flash-attention mask or KV view
  of a multi-stream cache) changes between steps for those shapes, and this llama.cpp has no
  "too many consecutive updates" bail-out, so it captures a new graph every token. That is a
  report for upstream; `GGML_CUDA_GRAPH_DEBUG`-style logging would name the node.
- The compute buffer at `n_batch = 2048` is 2.2 GB, which the old booking (0.75 GiB) did not
  cover: that is most of the "unaccounted" VRAM the planner reported.

## Changes

- `hypnos.example.toml` and the built-in profile: the 26B runs with `slots = 3`,
  `default_context_len = 16384`, `kv_dtype = "f16"` and `activation_workspace_bytes = 2 GiB`:
  150 tokens/s per stream, three streams batched, 20.8 GB booked. Four slots at 8k with f16 is
  the alternative when short contexts are enough.
- The planner's "device is full" behaviour was right: it refused further full placements on a
  device with 0 bytes free. The sd.cpp failure in the soak (a workspace allocation on a device
  with nothing free after the engine had been placed by its booking) needs the engine's own
  runtime reserve in the booking, which the 2 GiB workspace now covers for the 26B.

## Reproducing

```bash
# edit slots / default_context_len / kv_dtype in the 26B block, restart, then:
curl -s -X POST localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:26b-q4","messages":[{"role":"user","content":"Write a 120-word paragraph about tidal energy."}],"max_tokens":160}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['hypnos_telemetry']['engine'])"
```
