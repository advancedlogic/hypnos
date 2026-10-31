# KV parking and MoE offload on CUDA, 2026-10-31

RTX 5090 laptop GPU (24 GB), 63 GB RAM, Windows 11, release build 197ee08 + dc71df9 (Kokoro
fp32), `hypnos.example.toml` (`gemma4:26b-q4`: `slots = 3`, 16k, f16 KV, `moe_offload = "off"`),
the E4B parked in Tier 2. The Metal counterpart is
[metal-features-2026-10-31.md](metal-features-2026-10-31.md). WSL was shut down first: its VM
workers held ~5 GB of VRAM and the planner had answered by offloading the 26B's experts
(`cpu_expert_layers: 30`, 36 tok/s); nothing below is measured in that state.

## KV parking (MEM-4)

`backend/scripts/mem4_bench.py http://127.0.0.1:9999`: per conversation size, turn 1 with a
synthetic document, turn 2 warm (prefix reuse), `POST /v1/memory/tier` moves the idle 26B to
Tier 2, then turn 3. TTFT is measured by the client; the rest is the response telemetry.
`POST /v1/memory/kv-park` switched parking between the two runs.

| Prompt tokens | Warm turn | Turn after the swap, parking off | of which switch / prefill | Turn after the swap, parking on | of which switch / prefill | restored | restore | parked |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,690 | 219-228 ms | 4,078 ms | 3,654 / 352 ms (cached 0) | 3,953 ms | 3,711 / 181 ms (cached 1,632) | 14,331 tokens | 87 ms | 1,080 MiB |
| 6,317 | 215-242 ms | 4,751 ms | 3,505 / 1,184 ms (cached 0) | 4,085 ms | 3,828 / 182 ms (cached 6,271) | 20,686 tokens | 128 ms | 1,604 MiB |
| 12,629 | 232-252 ms | 5,925 ms | 3,556 / 2,318 ms (cached 0) | 4,006 ms | 3,745 / 201 ms (cached 12,573) | 20,685 tokens | 130 ms | 1,604 MiB |

Reading: parking works on CUDA as on Metal (every token of the conversation is back, the new
turn prefills ~60 tokens), but the gain is bounded by the tier switch, 3.5-3.8 s here against
0.9 s on the M5 Max: it saves 0.1 / 0.7 / 1.9 s per turn after a swap at the three sizes (1.03x,
1.16x, 1.48x) because this GPU prefills at 5.4k tokens/s. The parked state is all three slots
(`restored` is more than one conversation): 1.1-1.6 GB of host memory for the 26B, copied in
about 0.1 s. Worth leaving on when the machine has the RAM and conversations run long; the
default stays off.

## MoE expert offload (MEM-5)

`moe_offload` switched with `POST /v1/models/{id}/moe-offload`, the model reloaded from Tier 2,
two runs per row, temperature 0. "Short" is an 829-token prompt with a 110-150-token answer,
"long" a 10,537-token prompt with 32 output tokens.

| | `on` (experts on the CPU) | `off` |
| :--- | :--- | :--- |
| weights in VRAM | 2.97 GB | 15.81 GB |
| expert weights in host memory | 12.85 GB (30 of 30 layers) | 0 |
| KV cache + workspace booked | 7.43 GB | 3.91 GB (6.12 GB with the 10.5k context) |
| decode, short | 36-38 tok/s | 139-144 tok/s |
| decode, long context | 34-36 tok/s | 80-89 tok/s |
| prefill, short | 1,400 tok/s (TTFT 0.70 s) | 4,100-4,400 tok/s (TTFT 0.22 s) |
| prefill, long | 2,820-2,830 tok/s (TTFT 3.8 s) | 5,300-5,800 tok/s (TTFT 1.9-2.0 s) |
| load from Tier 2 | 4.5 s | 3.7-4.0 s |

Reading: on a discrete GPU the offload frees 12.8 GB of VRAM for a 3.9x slower decode and a
prefill that is only 2-3x slower (llama.cpp copies the experts to the GPU for large batches, so
the 14x prefill penalty of unified memory does not apply). It is the way to keep the 26B
resident next to the E4B and MOSS-Image on a 24 GB card instead of swapping (the voice round trip
spends 7 of its 8.5 s on two swaps), at the cost of 36 tok/s. `auto` does this only when the
model does not fit whole, which is also what the planner did on its own with WSL holding 5 GB:
`cpu_expert_layers` in the placement is the only sign of it, so check it before reading a
benchmark.
