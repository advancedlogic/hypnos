# Hypnos execution benchmark, 2026-10-31

| field | value |
| :--- | :--- |
| server | http://127.0.0.1:9999 (hypnos 0.2.0) |
| label | M5 Max 128 GB, Metal, slots = 4 |
| chat / stt / tts models | gemma4:26b-q4 / gemma4:e4b-q4 / kokoro-82m-q4 |
| VRAM total | 107.5 GiB |
| resident at start | docling=tier3_nvme_ssd, gemma4:26b-q4=tier2_pinned_ram, gemma4:e4b-q4=tier3_nvme_ssd, gemmaembedding2=tier1_gpu_vram, gliner2.5-multi=tier3_nvme_ssd, kokoro-82m-q4=tier1_gpu_vram, moss-image=tier3_nvme_ssd |
| platform | darwin, python 3.13.12 |

Latencies are measured by this client over HTTP on the same machine. TTFT = request sent to first non-empty content delta. Engine columns come from `hypnos_telemetry.engine`: engine TTFT = lease granted to first visible token; queue = waiting for the model lock; context = llama.cpp context creation; prefill = prompt evaluation.


### 1. Cold single chat (~512-token prompt, 256 output tokens)

| run | TTFT ms | total ms | prompt tok | cached tok | completion tok | decode tok/s (client) | engine TTFT ms | queue ms | context ms | prefill ms | prefill tok/s | decode tok/s (engine) | slots | tier switch ms | previous tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| cold | 1241 | 3066 | 504 | 0 | 211 | 115.6 | 184 | 0 | 0 | 142 | 3554 | 112.4 | 1 | 1056 | tier2_pinned_ram |


### 4. 4 concurrent chat clients on the same model (batching)

| run | TTFT ms | total ms | prompt tok | cached tok | completion tok | decode tok/s (client) | engine TTFT ms | queue ms | context ms | prefill ms | prefill tok/s | decode tok/s (engine) | slots | tier switch ms | previous tier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| solo | 88 | 1176 | 25 | 0 | 128 | 117.7 | 88 | 0 | 0 | 45 | 561 | 112.3 | 1 | 0 | tier1_gpu_vram |
| client 0 | 274 | 3031 | 25 | 0 | 128 | 46.4 | 274 | 0 | 0 | 184 | 136 | 44.6 | 4 | 0 | tier1_gpu_vram |
| client 1 | 274 | 3031 | 25 | 0 | 128 | 46.4 | 273 | 0 | 0 | 183 | 136 | 44.6 | 4 | 0 | tier1_gpu_vram |
| client 2 | 274 | 3031 | 25 | 0 | 128 | 46.4 | 274 | 0 | 0 | 184 | 136 | 44.6 | 4 | 0 | tier1_gpu_vram |
| client 3 | 252 | 3012 | 25 | 24 | 128 | 46.4 | 251 | 0 | 0 | 62 | 16 | 43.1 | 4 | 0 | tier1_gpu_vram |

| metric | value |
| :--- | :--- |
| concurrent wall time ms | 3032 |
| aggregate completion tokens | 512 |
| aggregate tokens/s (completion tokens / wall) | 168.9 |
| single-client decode tokens/s | 117.7 |
| aggregate / single-client | 1.43 |
| slowest client TTFT ms | 274 |


### Server metrics after the run

Lines from `/metrics` for the engine, scheduler and cache metrics (only those the server exports).

```
hypnos_scheduler_promotions_total 3
hypnos_scheduler_evictions_total 0
hypnos_scheduler_load_failures_total 0
hypnos_scheduler_admissions_rejected_total 0
hypnos_scheduler_tier_switch_ms_total 1480
hypnos_prefix_cache_hits_total{model="gemma4:26b-q4"} 1
hypnos_prefix_cache_misses_total{model="gemma4:26b-q4"} 5
hypnos_prefix_cache_tokens_reused_total{model="gemma4:26b-q4"} 24
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="100"} 1
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="200"} 2
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_ttft_ms_sum{model="gemma4:26b-q4"} 1343.665
hypnos_chat_ttft_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_ttft_ms_last{model="gemma4:26b-q4"} 273.582
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="50"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="100"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_queue_ms_sum{model="gemma4:26b-q4"} 0.488
hypnos_chat_queue_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_queue_ms_last{model="gemma4:26b-q4"} 0.136
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="50"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="100"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_context_ms_sum{model="gemma4:26b-q4"} 0.000
hypnos_chat_context_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_context_ms_last{model="gemma4:26b-q4"} 0.000
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 4
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 4
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 5
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 5
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_prefill_tokens_per_s_sum{model="gemma4:26b-q4"} 4540.056
hypnos_chat_prefill_tokens_per_s_count{model="gemma4:26b-q4"} 6
hypnos_chat_prefill_tokens_per_s_last{model="gemma4:26b-q4"} 136.144
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 4
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 4
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_decode_tokens_per_s_sum{model="gemma4:26b-q4"} 401.646
hypnos_chat_decode_tokens_per_s_count{model="gemma4:26b-q4"} 6
hypnos_chat_decode_tokens_per_s_last{model="gemma4:26b-q4"} 44.614
```

Changed during the run:

```
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="100"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="50"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_context_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_context_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_context_ms_last{model="gemma4:26b-q4"} 0.000
hypnos_chat_context_ms_sum{model="gemma4:26b-q4"} 0.000
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 4
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 4
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_decode_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_decode_tokens_per_s_count{model="gemma4:26b-q4"} 6
hypnos_chat_decode_tokens_per_s_last{model="gemma4:26b-q4"} 44.614
hypnos_chat_decode_tokens_per_s_sum{model="gemma4:26b-q4"} 401.646
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="10"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="100"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="1000"} 5
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="20"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="200"} 4
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="2000"} 5
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5"} 0
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="50"} 1
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="500"} 4
hypnos_chat_prefill_tokens_per_s_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_prefill_tokens_per_s_count{model="gemma4:26b-q4"} 6
hypnos_chat_prefill_tokens_per_s_last{model="gemma4:26b-q4"} 136.144
hypnos_chat_prefill_tokens_per_s_sum{model="gemma4:26b-q4"} 4540.056
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="100"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="200"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="50"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_queue_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_queue_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_queue_ms_last{model="gemma4:26b-q4"} 0.136
hypnos_chat_queue_ms_sum{model="gemma4:26b-q4"} 0.488
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="+Inf"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="100"} 1
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="1000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="10000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="200"} 2
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="2000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="300"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="3000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="30000"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="50"} 0
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="500"} 6
hypnos_chat_ttft_ms_bucket{model="gemma4:26b-q4",le="5000"} 6
hypnos_chat_ttft_ms_count{model="gemma4:26b-q4"} 6
hypnos_chat_ttft_ms_last{model="gemma4:26b-q4"} 273.582
hypnos_chat_ttft_ms_sum{model="gemma4:26b-q4"} 1343.665
hypnos_prefix_cache_hits_total{model="gemma4:26b-q4"} 1
hypnos_prefix_cache_misses_total{model="gemma4:26b-q4"} 5
hypnos_prefix_cache_tokens_reused_total{model="gemma4:26b-q4"} 24
hypnos_scheduler_promotions_total 3
hypnos_scheduler_tier_switch_ms_total 1480
```

Total benchmark wall time: 7 s.
