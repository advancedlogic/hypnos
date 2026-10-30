# Operating Hypnos

For the person who runs Hypnos rather than builds it: how to size a machine for a lineup, which
knob to turn for which symptom, how to read what the server tells you, what each error means, and
how to keep it running. The README covers installation and the API; `backend/hypnos.example.toml`
documents every option.

## 1. Sizing a machine

Hypnos books memory before it loads anything, so the question is whether the lineup's bookings fit
the budgets, not whether the machine "feels" big enough.

**Tier 1 (VRAM)** holds, per resident model, its weights plus the KV cache and compute workspace:

```
vram(model) = total_weight_bytes
            + slots x KV(default_context_len)      # chat models; KV per token from [models.arch]
            + arch.activation_workspace_bytes
```

`POST /v1/memory/status` reports the booking of every resident model (`placement`), what the
device really has free (`tier1_vram_real_free_bytes`) and what the planner did not book
(`tier1_vram_unaccounted_bytes`: driver, display, other processes). As a rule of thumb for the
default lineup on a 24 GB GPU: the 26B chat model is 13.6 GiB of weights plus about 0.9 GiB of KV
per slot at 16k context, so `slots = 4` fills the card and leaves the embedder (0.3 GiB) and Kokoro
(0.2 GiB) pinned; MOSS-Image (9.3 GiB) and the 26B cannot be resident together and evict each other.

**Tier 2 (pinned RAM)** holds the weight files of models staged there (`initial_tier =
"tier2_pinned_ram"`, or evicted with `keep_warm_in_ram = true`). Since PERF-6 the lock covers
every file of the model (weights, projector, text encoder), so `pinned_ram_budget_bytes` must
cover the sum of `total_weight_bytes` of the models you want warm at once, and that memory is
gone for the OS and for the Hypnos process itself (CUDA staging buffers, docling, GLiNER). Keep
the budget under half of physical RAM; the server warns at startup when it is not. On a 64 GB
machine with the default lineup, 24 GiB for the two chat models (with projectors) is comfortable;
32 GiB with the image model kept warm as well was not (`docs/bench/swap-cuda-2026-10-30.md`).

**Tier 3 (NVMe)** is the file. Promotion from Tier 3 reads it through the OS page cache: 4.6 s for
the 26B on this laptop when the cache is warm, disk speed when it is not.

Three startup sanity checks: `hypnos --check-config hypnos.toml` validates the file without
starting; the first lines of the log print the detected device, the budgets and every model's
booking; `/v1/memory/status` right after `/ready` shows the pre-staged models in their tiers.

## 2. Which knob for which symptom

| Symptom | Where to look | What to change |
| :--- | :--- | :--- |
| 429 `model_busy` | `/metrics` `hypnos_scheduler_admissions_rejected_total`; request log | More `slots` for that model (chat), or raise `limits.max_queue_per_model`; clients should back off on 429 |
| 429 `rate_limit_error` | log line "API request rejected: key limit", `hypnos_api_key_rejections_total{key,reason}` | The key's `requests_per_minute` / `max_concurrent` / `tokens_per_day` (`[[server.keys]]`) or the defaults |
| Models evict each other on every request | `recent_evictions` in `/v1/memory/status`, `hypnos_scheduler_evictions_total` | They do not fit Tier 1 together: lower `slots` or `default_context_len` on the big one, drop a model's priority to `low`, or move one to another machine |
| First token slow after idle | `hypnos_telemetry.tier_switch_latency_ms` and `tier_switch_phases` | The model was evicted: `priority = "realtime_pinned"` keeps it in VRAM, `keep_warm_in_ram = true` + Tier 2 budget makes the return 4 s instead of a disk read; check `use_mmap` is unset on CUDA |
| 507 `insufficient_vram_or_ram` | the log line names required and available bytes | The booking cannot fit even after evicting idle models: smaller context, fewer slots, lower `vram_safety_margin_bytes` if it is large, or a smaller quantisation |
| 507 `process_memory_cap` | `process_footprint_bytes`, `process_memory_cap_bytes` | Raise `hardware.max_process_memory_bytes`, or accept it as the guard it is |
| 507 `device_out_of_memory` | log line "Retrying after the engine ran out of device memory" or "no idle model holds VRAM"; `recent_evictions` with reason `device_out_of_memory`; `tier1_vram_unaccounted_bytes` | A placed model's engine found the device full mid-request (runtime buffers beyond the booking). Idle models were evicted and the request retried once; a 507 means nothing was left to evict. Raise that model's `activation_workspace_bytes` so the booking covers its real peak, or `vram_safety_margin_bytes` |
| Machine paging, everything slow | `free RAM` on the host; `tier2_pinned_ram_used_bytes` close to the budget | Lower `pinned_ram_budget_bytes`, `keep_warm_in_ram = false` on models that reload fast (the image model), fewer pre-staged models |
| 503 `model_not_available` right after start | `/ready` says 503 | Startup pre-staging is running (minutes with a large model); wait for `/ready` |
| GPU hot or throttling | `gpu.throttled`, `hypnos_gpu_temperature_celsius`, `hypnos_gpu_throttled` | Cooling, power limits; Hypnos only reports it |
| Document conversion slow | `hypnos_document_ms_per_page` | Table-heavy pages are CPU-bound (TableFormer); `device = "gpu"` helps layout- and OCR-heavy documents only (`docs/bench/docling-cuda-2026-10-30.md`) |
| Prompts repeat the same prefix but prefill is paid every time | `hypnos_prefix_cache_misses_total`, `cached_tokens` in the telemetry | `prefix_cache = true` and enough `slots` that the conversation's slot is not reused by another client between turns |

The limits in `[limits]` (`request_timeout_secs`, `max_body_bytes`, `max_prompt_chars`,
`max_tokens_cap`, `max_images_per_request`, `allow_remote_image_urls`) are safety rails, not
tuning: raise them when a legitimate request hits one, and keep `allow_remote_image_urls = false`
on a server that faces the internet (it makes the server fetch URLs that clients send).

## 3. Reading the telemetry

Every response carries `hypnos_telemetry`:

| Field | Meaning |
| :--- | :--- |
| `previous_tier` | Where the model was when the request arrived (`tier1_gpu_vram`: no load happened) |
| `placement` | The booking the model runs with: GPU / RAM / SSD layers, VRAM bytes, `allocated_context_len` |
| `tier_switch_latency_ms` | Time spent promoting (or expanding the context) before inference; 0 on the fast path |
| `tier_switch_phases` | When this request loaded the model: `load_mode`, `weights_ms`, `projector_ms`, `context_ms`, `total_ms` |
| `evicted_models` | What was evicted to make room |
| `inference_latency_ms` | Lease granted to response finished |
| `engine.queue_ms` | Waiting for an execution slot |
| `engine.prefill_ms`, `engine.decode_ms`, `engine.ttft_ms` | Prompt evaluation, generation, time to first token |
| `engine.prefill_tokens_per_s`, `engine.decode_tokens_per_s` | Throughput of this request |
| `engine.cached_tokens` | Prompt positions served from the slot's KV cache |
| `engine.concurrent_slots` | How many slots decoded together while this request ran |
| `engine.audio_secs`, `engine.rtf`, `engine.ttfa_ms` | Speech: audio length, real-time factor, time to first audio chunk |
| `engine.pages`, `engine.ms_per_page` | Document conversion |

`/metrics` (Prometheus text) has the same per model as histograms (`hypnos_chat_ttft_ms`,
`hypnos_chat_queue_ms`, `hypnos_chat_context_ms`, `hypnos_chat_prefill_tokens_per_s`,
`hypnos_chat_decode_tokens_per_s`, `hypnos_tts_rtf`, `hypnos_tts_ttfa_ms`, `hypnos_stt_rtf`,
`hypnos_image_jct_s`, `hypnos_document_ms_per_page`, each with a `_last` gauge), the scheduler
counters (`hypnos_scheduler_promotions_total`, `_evictions_total`, `_load_failures_total`,
`_admissions_rejected_total`), the memory gauges (`hypnos_vram_bytes{kind}`,
`hypnos_pinned_ram_bytes{kind}`, `hypnos_model_tier`, `hypnos_model_gpu_layers`,
`hypnos_model_active_leases`), per-route HTTP counters and latency, the GPU health gauges
(`hypnos_gpu_utilization_percent`, `_temperature_celsius`, `_power_watts`, `_sm_clock_mhz`,
`_throttled`), the prefix-cache counters and the per-key counters.

Alerts worth having: `hypnos_model_load_failures_total{model}` increasing (an engine refused its
weights: file gone or changed, CUDA out of memory; the per-model series says which),
`changes(hypnos_process_start_time_seconds[1h]) > 0` (the process restarted: an engine abort took
it down and the supervisor brought it back), `hypnos_gpu_throttled == 1` for minutes, `/ready` not
200 for longer than your pre-staging takes, `hypnos_http_requests_total{class="5xx"}` rate.

The request log (`GET /v1/logs`, the playground's "Request Logs" tab) keeps the last
`server.request_log_capacity` calls with route, status, model, key, tokens, tier switch and
timings; `?key=`, `?model=`, `?route=`, `?status=error` filter it.

## 4. Errors

All errors come in the OpenAI shape, `{"error": {"message", "type", ...}}`, so OpenAI SDKs raise
their usual exceptions.

| Status | `type` | Meaning | Fix |
| :--- | :--- | :--- | :--- |
| 400 | `invalid_request_error` | Bad parameter, empty messages, unsupported modality for the model, oversized input | Fix the request; the message names the field |
| 401 | `invalid_request_error` / code `invalid_api_key` | Missing, malformed or wrong bearer key | Send `Authorization: Bearer <key>` |
| 404 | `model_not_found` | No such model id in the config | `GET /v1/models` lists the ids |
| 404 | `model_disabled` | The model is configured but switched off | `POST /v1/models/{id}/enable`, or `enabled = true` |
| 404 | `not_found` | A stored document or log entry is gone | Pictures expire after `document_pictures_ttl_secs` |
| 429 | `model_busy` | `limits.max_queue_per_model` requests already queued or running for the model | Back off and retry; more slots or a higher limit |
| 429 | `rate_limit_error` (`requests_per_minute`, `max_concurrent`, `tokens_per_day`) | The key's limit | `Retry-After` says when |
| 499 | `cancelled` | The client went away | Nothing |
| 503 | `model_not_available` | Startup pre-staging still running, or a config-level refusal | Wait for `/ready` |
| 504 | `timeout` | `limits.request_timeout_secs` passed | Longer timeout, smaller request |
| 507 | `insufficient_vram_or_ram` | The booking does not fit even after evicting idle models | See sizing |
| 507 | `process_memory_cap` | Loading would cross `hardware.max_process_memory_bytes` | Raise the cap or free RAM |
| 507 | `device_out_of_memory` | The engine ran out of device memory while running, and evicting idle models freed nothing (or the retry failed too) | Retry later; see the symptom table |
| 500 | `server_error` | An engine failed (the log line has the engine's message) | Check the log; a load failure rolls the booking back |

## 5. Logs, levels and redaction

- `server.log_format = "json"` (or `HYPNOS_LOG_FORMAT=json`) for log shippers; `pretty` for a
  terminal. Startup prints every config warning and error, the device and the budgets.
- The port is `server.port` (9999 by default), overridden by `HYPNOS_PORT` and, above that, by
  `--port N` on the command line, so one config file can serve several instances.
- `server.request_log_level`: `info` keeps metadata only and is the right level for anything
  exposed beyond a trusted group; `debug` stores prompts and completions; `trace` stores payloads
  with media replaced by placeholders. `POST /v1/logs/level` changes it at runtime.
- `server.request_log_redact`: regular expressions whose matches become `[redacted]` in stored
  text, or `["all"]` to keep no text at any level.
- Rejections are logged with route, reason and the key's label, never the key.

## 6. Exposure

Bind `127.0.0.1` unless keys and TLS (or a terminating proxy) are in place. With keys, every
`/v1/*` route needs a bearer key; `/health` and `/ready` stay open, `/metrics` too unless
`server.metrics_require_auth = true`. Per-key limits are the only multi-tenancy Hypnos has: for
quotas per user or organisation, billing, or finer rules put an API gateway in front.

## 7. Upgrading and rolling back

- Release archives and the Docker image are versioned; `hypnos --version` prints the build.
- `hypnos.toml` carries `config_version`. A newer Hypnos migrates an older file in memory and
  logs each step; a file written for a newer Hypnos is refused with both versions named, so a
  rollback to an older binary needs the older config (keep the previous `hypnos.toml` next to the
  previous archive). Unknown keys are reported as warnings, not ignored.
- `backend/models.manifest` carries `manifest-version`; the fetch scripts refuse a newer layout.
- Run `hypnos --check-config hypnos.toml` with the new binary before switching traffic.
- Model files are addressed by path in the config: a new release that changes the lineup needs
  `scripts/fetch-models` for the new files; old files can stay.

## 8. Running as a service

**Docker** (CUDA):

```bash
docker run --gpus all -p 9999:9999 -v ./models:/opt/hypnos/models \
  -e HYPNOS_API_KEYS=... --restart unless-stopped ghcr.io/advancedlogic/hypnos:0.1.0
```

Mount your own `hypnos.toml` over `/opt/hypnos/hypnos.toml` or point `HYPNOS_CONFIG` at a
mounted file. The image has a `HEALTHCHECK` on `/health` with a 10-minute start period for the
pre-staging. `--build-arg DOCLING_CUDA=1` adds the CUDA 13 runtime for docling on the GPU.

**systemd** (Linux archive unpacked in `/opt/hypnos`):

```ini
[Unit]
Description=Hypnos inference server
After=network-online.target

[Service]
WorkingDirectory=/opt/hypnos
ExecStart=/opt/hypnos/hypnos /opt/hypnos/hypnos.toml
Environment=HYPNOS_API_KEYS=...
Environment=HYPNOS_LOG_FORMAT=json
Restart=on-failure
RestartSec=5
LimitMEMLOCK=infinity
KillSignal=SIGTERM
TimeoutStopSec=60

[Install]
WantedBy=multi-user.target
```

`LimitMEMLOCK` matters: Tier 2 uses `mlock`, and the default limit is small (64 KiB to 8 MiB
depending on the distribution). SIGTERM triggers the
graceful shutdown (in-flight requests finish, engines are released).

**Windows**: the archive runs from its folder (`hypnos.exe` reads `hypnos.toml`); to run it as a
service use a wrapper such as NSSM (`nssm install hypnos C:\hypnos\hypnos.exe hypnos.toml`, set
the startup directory to the archive folder and `HYPNOS_API_KEYS` in its environment) or a Task
Scheduler task that starts at boot. The process raises its own working-set limit for Tier 2.

**Supervision**: an engine abort inside llama.cpp or stable-diffusion.cpp (a C++ `abort`) takes
the process down; the restart policy above is what brings it back, and `/ready` tells the load
balancer when it is serving again. A load failure that the engine reports as an error (missing
file, CUDA out of memory) does not: the booking is rolled back and the request answers 500.

## 9. Before going live

1. `hypnos --check-config` is clean, and the startup log shows the budgets you expect.
2. `backend/scripts/smoke_test.sh http://host:9999` passes once per modality.
3. `backend/scripts/bench.sh` against the box gives the numbers you will compare against later
   (`docs/bench/` holds the reference runs).
4. Keys set, bound beyond loopback only behind TLS, request log at `info`, redaction on if the
   level is ever raised.
5. Alerts on `/ready`, load failures, 5xx rate and GPU throttling.
