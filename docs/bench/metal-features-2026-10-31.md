# KV parking, MoE offload and slots on Apple Silicon, 2026-10-31

| field | value |
| :--- | :--- |
| machine | Apple M5 Max, 128 GB unified memory (Metal working set 107.5 GiB) |
| server | hypnos 0.2.0, release build, `hypnos.macos.example.toml`, commit 6c6d3f7 |
| model | `gemma4:26b-q4` (Gemma 4 26B-A4B, Q4_0, 15.8 GB with the projector), 16k context |

The standard scenarios of the same day are in [metal-2026-10-31.md](metal-2026-10-31.md).

## KV parking (MEM-4)

`backend/scripts/mem4_bench.py`: a conversation over a generated document, a warm second turn,
a forced swap to Tier 2 (`POST /v1/memory/tier`), then the third turn. Parking switched with
`POST /v1/memory/kv-park` between the two runs.

| conversation | first turn | warm turn | turn after swap, parking off | turn after swap, parking on | tier switch | prefill after swap off / on | restored | parked state |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1.7k tokens | 452-523 ms | 108 ms | 1,387 ms | **1,044 ms** | 926-936 ms | 406 / 73 ms | 1,672 tokens in 2 ms | 230 MiB |
| 6.3k tokens | 1,596-1,773 ms | 116 ms | 2,573 ms | **1,080 ms** | 942-954 ms | 1,582 / 77 ms | 6,308 tokens in 3 ms | 278 MiB |
| 12.6k tokens | 3,762-3,817 ms | 137 ms | 4,615 ms | **1,122 ms** | 938-975 ms | 3,613 / 93 ms | 12,612 tokens in 5 ms | 344 MiB |

With parking the turn after a swap costs the tier switch plus a short prefill of the new turn,
whatever the length of the conversation. On this machine the default lineup fits in Tier 1, so
swaps happen only when forced; the feature is aimed at GPUs where models evict each other.

## MoE expert offload (MEM-5)

`moe_offload` switched with `POST /v1/models/{id}/moe-offload`, the model reloaded from Tier 2,
two runs per row.

| | `on` (experts on the CPU) | `off` |
| :--- | :--- | :--- |
| weights in the GPU working set | 2.97 GB | 15.81 GB |
| expert weights in host memory | 12.85 GB (30 of 30 layers) | 0 |
| KV cache + workspace | 1.74 GB | 1.74 GB |
| decode, 256 tokens | 48.5-48.9 tok/s | 111.8-113.0 tok/s |
| prefill, 6.4k-token prompt | 247-261 tok/s (TTFT 24.9-26.2 s) | 3,628-3,997 tok/s (TTFT 1.7-1.8 s) |

Unified memory gains nothing from the offload (one pool) and llama.cpp runs the experts' prefill
on the CPU, hence `auto` never acts here. The case for it is a discrete GPU; the CUDA measurement
is still to do (docs/moe.md, step 1).

## Slots and concurrent clients

bench.py scenario 4 (4 clients, 128 tokens each) on the macOS profile as shipped (`slots = 1`)
and with `slots = 4` on the 26B ([report](metal-slots4-2026-10-31.md)):

| | `slots = 1` | `slots = 4` |
| :--- | :--- | :--- |
| single-client decode | 116.9 tok/s | 117.7 tok/s |
| aggregate, 4 clients | 113.3 tok/s (0.97x) | 168.9 tok/s (**1.43x**) |
| slowest client TTFT | 3,443 ms (queued) | 274 ms |
| wall time | 4,518 ms | 3,032 ms |
