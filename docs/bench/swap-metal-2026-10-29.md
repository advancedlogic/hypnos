# Hypnos execution benchmark, 2026-10-29

| field | value |
| :--- | :--- |
| server | http://127.0.0.1:8080 (hypnos 0.1.0) |
| label | PERF-5, Apple M5 Max 128 GB |
| chat / stt / tts models | gemma4:26b-q4 / gemma4:e4b-q4 / kokoro-82m-q4 |
| VRAM total | 107.5 GiB |
| resident at start | gemma4:26b-q4=tier2_pinned_ram, gemma4:e4b-q4=tier3_nvme_ssd, gemmaembedding2=tier1_gpu_vram, kokoro-82m-q4=tier1_gpu_vram, moss-image=tier3_nvme_ssd |
| platform | darwin, python 3.13.12 |

Latencies are measured by this client over HTTP on the same machine. TTFT = request sent to first non-empty content delta. Engine columns come from `hypnos_telemetry.engine`: engine TTFT = lease granted to first visible token; queue = waiting for the model lock; context = llama.cpp context creation; prefill = prompt evaluation.


### 7. Model swap latency by start tier (gemma4:26b-q4)

| rep | start tier | staged as | telemetry previous tier | staging call ms | page cache resident | tier switch ms | engine TTFT ms | TTFT ms | total ms |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | tier3_nvme_ssd | tier3_nvme_ssd | tier3_nvme_ssd | 161 | 19% | 2264 | 89 | 2353 | 2362 |
| 2 | tier3_nvme_ssd | tier3_nvme_ssd | tier3_nvme_ssd | 129 | 19% | 2201 | 89 | 2291 | 2300 |
| 3 | tier3_nvme_ssd | tier3_nvme_ssd | tier3_nvme_ssd | 134 | 19% | 2197 | 90 | 2288 | 2297 |
| 1 | tier2_pinned_ram | tier2_pinned_ram | tier2_pinned_ram | 701 | 100% | 902 | 89 | 992 | 1000 |
| 2 | tier2_pinned_ram | tier2_pinned_ram | tier2_pinned_ram | 629 | 100% | 894 | 90 | 985 | 994 |
| 3 | tier2_pinned_ram | tier2_pinned_ram | tier2_pinned_ram | 624 | 100% | 918 | 87 | 1006 | 1015 |
| 1 | tier1_gpu_vram | tier1_gpu_vram | tier1_gpu_vram | 0 | 100% | 0 | 84 | 84 | 93 |
| 2 | tier1_gpu_vram | tier1_gpu_vram | tier1_gpu_vram | 0 | 100% | 0 | 83 | 84 | 93 |
| 3 | tier1_gpu_vram | tier1_gpu_vram | tier1_gpu_vram | 0 | 100% | 0 | 81 | 81 | 90 |

| start tier | runs | median tier switch ms | median TTFT ms | TTFT range ms | median page cache resident |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Tier 3 (NVMe mmap, cold) | 3 | 2201 | 2291 | 2288-2353 | 19% |
| Tier 2 (pinned RAM, warm) | 3 | 902 | 992 | 985-1006 | 100% |
| Tier 1 (resident, hot) | 3 | 0 | 84 | 81-84 | 100% |

Each run stages the model with `POST /v1/memory/tier`, then times one streamed chat request (8 output tokens). `tier switch ms` is the server's `hypnos_telemetry.tier_switch_latency_ms` (promotion inside `acquire_model`); TTFT is measured by this client and includes it. Before each Tier 3 run the weight files were evicted from the OS page cache (gemma-4-26B-A4B-it-Q4_0.gguf, mmproj-gemma-4-26B-A4B-it-BF16.gguf), so Tier 3 reads come from the SSD; `page cache resident` is measured with `mincore` just before the request.


### Server metrics after the run

Lines from `/metrics` for the engine, scheduler and cache metrics (only those the server exports).

```
hypnos_scheduler_promotions_total 9
hypnos_scheduler_evictions_total 0
hypnos_scheduler_load_failures_total 0
hypnos_scheduler_admissions_rejected_total 0
hypnos_scheduler_tier_switch_ms_total 10709
hypnos_prefix_cache_misses_total{model="gemma4:26b-q4"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_ttft_ms_sum{model="gemma4:26b-q4"} 877.269
hypnos_chat_ttft_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_ttft_ms_last{model="gemma4:26b-q4"} 80.801
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="50"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_queue_ms_sum{model="gemma4:26b-q4"} 0.079
hypnos_chat_queue_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_queue_ms_last{model="gemma4:26b-q4"} 0.009
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="50"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_context_ms_sum{model="gemma4:26b-q4"} 0.000
hypnos_chat_context_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_context_ms_last{model="gemma4:26b-q4"} 0.000
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_prefill_tokens_per_s_sum{model="gemma4:26b-q4"} 3433.676
hypnos_chat_prefill_tokens_per_s_count{model="gemma4:26b-q4"} 10
hypnos_chat_prefill_tokens_per_s_last{model="gemma4:26b-q4"} 391.811
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_decode_tokens_per_s_sum{model="gemma4:26b-q4"} 853.818
hypnos_chat_decode_tokens_per_s_count{model="gemma4:26b-q4"} 10
hypnos_chat_decode_tokens_per_s_last{model="gemma4:26b-q4"} 86.392
```

Changed during the run:

```
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="50"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_context_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_context_ms_last{model="gemma4:26b-q4"} 0.000
hypnos_chat_context_ms_sum{model="gemma4:26b-q4"} 0.000
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_decode_tokens_per_s_count{model="gemma4:26b-q4"} 10
hypnos_chat_decode_tokens_per_s_last{model="gemma4:26b-q4"} 86.392
hypnos_chat_decode_tokens_per_s_sum{model="gemma4:26b-q4"} 853.818
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_prefill_tokens_per_s_count{model="gemma4:26b-q4"} 10
hypnos_chat_prefill_tokens_per_s_last{model="gemma4:26b-q4"} 391.811
hypnos_chat_prefill_tokens_per_s_sum{model="gemma4:26b-q4"} 3433.676
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="50"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_queue_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_queue_ms_last{model="gemma4:26b-q4"} 0.009
hypnos_chat_queue_ms_sum{model="gemma4:26b-q4"} 0.079
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="100"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="1000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="10000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="200"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="2000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="300"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="3000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="30000"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="500"} 10
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="5000"} 10
hypnos_chat_ttft_ms_count{model="gemma4:26b-q4"} 10
hypnos_chat_ttft_ms_last{model="gemma4:26b-q4"} 80.801
hypnos_chat_ttft_ms_sum{model="gemma4:26b-q4"} 877.269
hypnos_prefix_cache_misses_total{model="gemma4:26b-q4"} 10
hypnos_scheduler_promotions_total 9
hypnos_scheduler_tier_switch_ms_total 10709
```

Total benchmark wall time: 17 s.
