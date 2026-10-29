# Execution Efficiency Plan

Implementation plan for the essential changes that move Hypnos from "one request, one context, one
lock" to a persistent, batched, streaming execution model, while keeping the 3-tier memory
scheduler as the thing that decides what is resident.

Status: **work items A and B done** (2026-10-29). A: engine timings in `hypnos_telemetry.engine`,
per-model histograms on `/metrics`, `backend/scripts/bench.sh`, baseline in
`docs/bench/baseline-2026-10-29.md`. B: one engine worker thread per resident llama.cpp model
(`hypnos-backends/src/engine.rs`, `worker.rs`) owning a persistent context, execution slots
(`slots`, default 1), prefix reuse with multimodal chunk ids (`prefix_cache`), results in
`docs/bench/after-B-2026-10-29.md`. D: `synthesize_speech_stream` on the backend trait, Kokoro
streams each chunk as ONNX Runtime returns it, `POST /v1/audio/speech` with `"stream": true`
returns raw PCM or a live WAV. C: the worker decodes every active slot in one `llama_batch` per
step (chunked prefill, per-slot sampling by batch index, non-blocking per-slot streaming);
`hypnos.example.toml` gives both Gemma chat models `slots = 4`. E is not started. The motivation
and the comparison with vLLM-Omni are summarised in section 1; the rest is the work. Line numbers
refer to the tree before A.

What C measured (`docs/bench/after-C-2026-10-29.md`, four concurrent clients, 128 tokens each):

| Model | Single client | Four clients, aggregate | Ratio | Slowest TTFT |
| :--- | :--- | :--- | :--- | :--- |
| gemma4:26b-q4 (MoE, 4 slots) | 114 tokens/s | 248 tokens/s, wall 2.1 s (was 4.9 s) | 2.18x | 171 ms (was 3.7 s) |
| gemma4:e4b-q4 (dense, 4 slots) | 94 tokens/s | 103 tokens/s | 1.10x | 102 ms |

Single-client TTFT and the prefix-cache results of B are unchanged. The 2.5x target of section 5
is not reached on either model, for different reasons. On the 26B a four-sequence step costs 1.7x a
single-sequence step: with a mixture of experts every extra sequence routes to more distinct
experts, so the weights read per step grow with the batch. On the E4B a four-sequence step costs
3.6x a single step although all 42 layers are on the GPU, so some per-token work is not batched at
all; the Gemma E-series per-layer token embeddings, which llama.cpp gathers per token outside the
GPU graph, are the prime suspect and need a look in the engine before `slots > 1` is worth
anything on that model. Also visible in scenario 6: with `slots = 4` the 26B books 3.5 GiB more KV,
so it and the E4B still evict each other on every voice turn (work item E's co-residency lease).

What D measured (600-character paragraph, Kokoro on CPU): time to first audio 5.4 s without
streaming, 385 ms with it (engine 364 ms), total synthesis unchanged at 5.5 s. Section 6 asked for
300 ms; the first chunk is already down to 37 phonemes (about 4 words, 2.7 s of audio) and a
shorter one gets choppy, so 385 ms is where this engine lands. Two details section 6 did not
anticipate: espeak-ng's phoneme output carries no sentence or punctuation markers, so the chunk
boundaries are decided on the text before phonemization (sentences, then the first clause or word
boundary after 28 characters); and chunk sizes grow geometrically (48, 192, then 510 phonemes) so
each chunk is synthesized before the audio already sent has finished playing.

What B measured against the baseline (same machine, `slots = 1`):

| Scenario | Baseline | After B |
| :--- | :--- | :--- |
| 2. five-turn chat, turns 2 to 5 | TTFT 297 to 325 ms, 0 cached | TTFT 142 to 162 ms, 517 to 676 positions cached per turn |
| 3. vision chat, turns 2 and 3 | TTFT 288 to 306 ms, image re-encoded | TTFT 101 to 114 ms, one image encode over three turns |
| 1. cold chat | context creation 62 ms per request | 0 ms (context persists) |
| 4. four concurrent clients | 0.92x of single-client | 0.93x (unchanged by design; C) |

Two things B had to add that section 4 did not foresee:

- **Sliding-window checkpoints.** The generated reply never re-tokenises to the same tokens the
  model sampled (first-token boundary, trimmed whitespace, swallowed channel markers), so the next
  turn always diverges inside the previous reply. With `swa_full = false` the local layers cannot
  be trimmed back to that point, which made every turn a miss. The worker now snapshots the
  sliding-window state at the end of each prompt (`llama_state_seq_get_data_ext` with the
  partial-only flag, about 26 MB for the 26B) and restores it when the divergence is at or after
  the checkpoint, the same mechanism llama-server uses.
- **Planning for the default window.** The scheduler used to size a residency for the request's
  own estimate, which would have rebuilt the context (and dropped every cache) each time a
  conversation grew. `acquire_model` now plans `max(needed, default_context_len)` whenever that
  fits with the same GPU split.

Left inside B's scope for later: a new image arriving mid-conversation still triggers a full
prefill (evaluating a single mtmd chunk needs `mtmd_helper_eval_chunk_single`, which the vendored
bindings do not expose); M-RoPE models never reuse; `slots > 1` is accepted by the config and
booked by the planner but served one request at a time until C.

What the baseline (RTX, 24 GB, Windows) says about each work item:

- **B, text.** On a resident model the engine reaches the first visible token in 250 to 450 ms:
  context creation 20 to 60 ms, prefill 160 to 300 ms at 1.7k to 3.6k tokens/s, then the first decode
  step. Turn five of the five-turn chat re-prefills 725 tokens in about 200 ms, so prefix reuse saves
  roughly 150 ms per turn here, not seconds. The 3x TTFT target in section 8 is therefore not
  reachable for text on this machine (the floor is context plus one decode step, about 100 ms);
  B's text win is the removed per-request context allocation plus the prefill, and it matters more
  on slower GPUs and longer histories.
- **B, vision.** Turns two and three of the vision chat prefill only 137 to 159 tokens yet take 190 to
  200 ms, against 160 ms for 517 text tokens: the image encode is most of it and is repeated every
  turn. The chunk-id matching keeps its full value.
- **C.** Four concurrent clients reach 0.92x of single-client decode throughput: the per-model lock
  serialises them completely. The 2.5x target stands.
- **D.** Time to first byte for the 600-character paragraph is 4.9 s (RTF 0.13 on CPU); with
  streaming it becomes the first sentence's synthesis time.
- **E.** The voice round trip reaches first audio after 12.4 s, of which 7.5 s is the 26B model being
  promoted again after the E4B speech model displaced it. The pipeline lease (co-residency) is worth
  more than the stage overlap; the overlap then removes the remaining sequential 4 s.

The first two runs of the bench script reported a constant 2.3 s client-side TTFT that the engine
never saw: Python's urllib consults the Windows proxy settings per request. The script now uses a
proxy-free opener and 127.0.0.1; the checked-in baseline is from the corrected script.

---

## 1. Why

Hypnos spends most of its engineering on *placement* (which model is in VRAM, how many layers,
what fits) and almost none on *execution* (what happens between the lease being granted and the
response being sent). The execution path today, per request:

| Step | Where | Cost |
| :--- | :--- | :--- |
| Lock the model for the whole request | `StdMutex<Option<LoadedLlama>>`, `llama.rs:586` | second user on the same model waits for the first to finish |
| Create a new `LlamaContext` (KV cache + compute buffers) | `run_generation`, `llama.rs:304` | allocation plus graph warm-up for every call, freed at the end |
| Re-encode every image and audio clip in the conversation | media collected from all messages, `llama.rs:559`; mtmd encodes in `eval_chunks`, `llama.rs:360` | one vision-tower pass per image per turn |
| Re-prefill the entire chat history | no KV survives the request | prefill cost grows linearly with conversation length |
| Decode one input at a time for embeddings | `llama.rs:871` | N small graphs instead of one full batch |
| Synthesize all TTS chunks, then return one WAV | `kokoro.rs:343` | time to first audio equals total synthesis time |
| Voice interaction is three HTTP calls | client-driven: transcribe, chat, speak | three leases, three full materialisations, no overlap, no co-residency guarantee |

vLLM-Omni's measured gains come from three things that do transfer to a single machine: engines
that persist between requests and batch across them, caches that outlive a request, and stages that
stream into one another. Its distribution machinery (multi-node connectors, tensor and expert
parallelism) does not apply here and is out of scope.

Useful facts established while scoping (so the plan does not reinvent them):

- The vendored `llama-cpp-2` already exposes `with_n_seq_max`, `kv_cache_seq_rm`, `kv_cache_seq_cp`,
  `state_seq_get` / `state_seq_set`, `state_seq_save_file` / `state_seq_load_file`, and
  `with_flash_attention_policy` (`vendor/llama-cpp-rs/llama-cpp-2/src/context/`).
- mtmd input chunks carry an `id` and bitmaps accept `set_id`; llama-server uses exactly this to
  match multimodal prompts against a cached prefix without re-encoding.
- stable-diffusion.cpp on the vendored pin already has `conditioning_cache_size` (default 4),
  `batch_count` (Hypnos already passes `n`, `diffusion.rs:252`), and a step-cache family
  (`sd_cache_params_t` with EasyCache, UCache, DBCache, TaylorSeer, Cache-DiT, Spectrum).
- Per-model admission is a `Semaphore` sized by `limits.max_queue_per_model`
  (`scheduler/lib.rs:72`); it bounds the queue but adds no concurrency.

---

## 2. Scope

Essential (this plan):

| # | Work item | Size | Depends on |
| :--- | :--- | :--- | :--- |
| A | Measurement baseline: TTFT, tokens per second, RTF, JCT metrics and a bench script | S | none |
| B | Per-model engine worker with a persistent context, slots and prefix reuse | L | A (to prove it) |
| C | Continuous batching across slots | M | B |
| D | Streaming TTS | S | none |
| E | In-process voice pipeline (transcribe, chat, speak) with sentence-level overlap and a pipeline lease | M | B, D |

Deferred (section 9): KV state offload to Tier 2/3, a standalone encoder-output cache, batched
embeddings, diffusion step-cache wiring, image previews, a realtime WebSocket API.

The acceptance bar for the whole plan is in section 8. Each work item below carries its own
done-when.

---

## 3. Work item A: measure first

### Problem

`/metrics` has route latency and scheduler counters (`hypnos-server/src/metrics.rs`) but none of
the numbers that the rest of this plan changes. Without them nobody can tell whether B helped.

### Change

Add to `metrics.rs`, per model id:

| Metric | Type | Source |
| :--- | :--- | :--- |
| `hypnos_chat_ttft_ms` | histogram | lease granted to first `Delta` sent |
| `hypnos_chat_prefill_tokens_per_s` | gauge (last) and histogram | prompt tokens / prefill wall time |
| `hypnos_chat_decode_tokens_per_s` | gauge and histogram | completion tokens / decode wall time |
| `hypnos_prefix_cache_tokens_reused_total` | counter | from B |
| `hypnos_prefix_cache_hits_total`, `_misses_total` | counter | from B |
| `hypnos_slots_active` | gauge | from B and C |
| `hypnos_tts_rtf` | histogram | synthesis seconds / audio seconds |
| `hypnos_tts_ttfa_ms` | histogram | time to first audio chunk, from D |
| `hypnos_stt_rtf` | histogram | transcription seconds / audio seconds |
| `hypnos_image_jct_s` | histogram | request to last PNG |

`GenStats` (`llama.rs`) already carries prompt and completion token counts; add prefill and decode
durations to it and to `ExecutionTelemetry` so the numbers also appear in `hypnos_telemetry` on the
response.

Add `backend/scripts/bench.sh` (or a `hypnos-bench` bin in `hypnos-server/src/bin/`) that runs a
fixed scenario against a live server and prints a table:

1. Cold single chat, 512-token prompt, 256 output tokens.
2. The same conversation extended over five turns (measures re-prefill, the B win).
3. Vision chat, one image, three turns (measures re-encoding, the B win with chunk ids).
4. Four concurrent chat clients on the same model (the C win).
5. TTS of a 600-character paragraph with time to first byte (the D win).
6. Voice round trip: 10 s WAV in, speech out, time to first audio (the E win).

Done when: the script runs on the current `main` and produces the baseline table checked into
`docs/bench/baseline-<date>.md`.

---

## 4. Work item B: engine worker, persistent context, slots, prefix reuse

This is the core of the plan. It replaces the "lock the model, build a context, run, drop" shape
with one long-lived worker per resident llama.cpp model that owns a single context and serves
requests through slots.

### 4.1 Design

```
HTTP handler ──acquire_model──▶ lease ──▶ backend.chat_completion_stream(req)
                                              │
                                              ▼  mpsc::Sender<EngineJob>
                                   ┌──────────────────────────┐
                                   │  engine worker thread    │  one per resident model
                                   │  owns LlamaContext       │  n_seq_max = slots
                                   │  slots[0..n]             │  each: seq_id, cache_tokens,
                                   │  loop:                   │        sampler, tx, rctx, state
                                   │    admit new jobs → slot │
                                   │    build one LlamaBatch  │  (C: all active slots per step)
                                   │    decode                │
                                   │    sample per slot       │
                                   │    stream pieces via tx  │
                                   └──────────────────────────┘
```

Key decisions:

- **One OS thread per resident model**, started in `apply_placement` when the model becomes
  resident and stopped when it is demoted. The thread owns the `LlamaContext`, so there is no mutex
  held across awaits and no `spawn_blocking` per request. Jobs arrive over an `mpsc` channel.
- **Fixed context size per residency.** The context is created once with
  `n_ctx = plan.allocated_context_len` (already decided by the scheduler) and
  `n_seq_max = slots`. The per-request `num_ctx` override becomes a *requirement* checked against the
  slot's share, not a context parameter. A request that needs more than the resident context takes
  the existing reload path (`acquire_model` already re-plans when `allocated_context_len < needed`).
- **Slots.** Each slot is a llama.cpp sequence id with its own `cache_tokens: Vec<LlamaToken>`
  describing what its KV currently holds, its own `LlamaSampler`, and the request's `tx` and
  `RequestContext`. `slots = 1` reproduces today's semantics plus prefix reuse, and is the safe
  default for large models on small GPUs.
- **Prefix reuse.** On admission, pick the free slot whose `cache_tokens` shares the longest common
  prefix with the new prompt tokens (llama-server's slot selection). Call
  `kv_cache_seq_rm(seq, n_matched, -1)` to drop the divergent tail and prefill only the remaining
  tokens. For multimodal prompts the comparison unit is the mtmd chunk: text chunks compare by
  token, image and audio chunks compare by chunk `id`, which is derived from the bitmap hash set
  through `MtmdBitmap::set_id`. A matched media chunk is therefore never re-encoded. This gives most
  of the benefit of a separate encoder cache for the dominant case (same images, growing
  conversation) at a fraction of the complexity.
- **Idle slots keep their KV** until another request needs the slot or `kv_idle_timeout_secs`
  passes, at which point `kv_cache_seq_rm(seq, -1, -1)` clears it. The config key finally does what
  its name says.
- **Cancellation** stays cooperative: the worker checks `rctx.should_continue()` per slot per step
  and frees the slot.

### 4.2 Changes by file

- `hypnos-backends/src/llama.rs`
  - New module `engine.rs` (or `llama/worker.rs`): `EngineWorker`, `Slot`, `EngineJob`
    (`Chat { tokens | mtmd chunks, params, tx, rctx }`, `Embed { .. }`, `Shutdown`), slot selection,
    prefix matching, prefill, per-slot sampling, piece streaming. `run_generation` is split: the
    prompt building and media loading stay in the async path, the context work moves into the
    worker.
  - `LoadedLlama` gains the worker handle (`JoinHandle` + `Sender<EngineJob>`); dropping it on
    demotion sends `Shutdown` and joins.
  - `GgmlLlamaBackend::generate_chat` sends a job and awaits the result instead of locking.
  - Embeddings go through the same worker (one context, `embeddings(true)` toggled per batch is
    not possible, so an embeddings model gets an embeddings-mode worker; a model config is one or
    the other, which is already the case).
- `hypnos-core/src/config.rs`: `ModelConfig.slots: u32` (default 1), `ModelConfig.prefix_cache: bool`
  (default true). Validation: `slots >= 1`; warn when `slots > 1` and `min_gpu_layer_ratio < 1.0`.
- `hypnos-core/src/kv_cache.rs`: `vram_runtime_overhead` must book KV for `slots × context`, not
  one context. With Gemma 4's SWA layout at q8_0 this is small, but it has to be in the plan so the
  reconciliation does not report it as unaccounted.
- `hypnos-scheduler/src/lib.rs`: `acquire_model` no longer implies exclusive access. The lease keeps
  the model resident; the worker serialises or batches. The admission semaphore stays as the queue
  bound.
- `hypnos-server/src/lib.rs`: unchanged API. `chat_completions` already streams from an `mpsc`
  receiver, which is exactly what the worker produces.

### 4.3 Risks

- **Memory for `slots × n_ctx`.** Mitigated by default `slots = 1` and by the planner booking it.
- **Multimodal prefix matching correctness.** Chunk ids must include the projector identity and
  image preprocessing parameters; if the mmproj changes on reload, all slots are cleared (they are,
  because the context is recreated).
- **Sampler state per slot** must be reset per request (new `LlamaSampler` each admission).
- **Embeddings model reuse** currently builds a context with `n_batch = n_ctx`; the worker keeps
  that.
- **Behavioural change**: a request that today would wait for the lock now waits in the worker
  queue. Same latency, different place; the admission semaphore still bounds it.

Done when: bench scenarios 1 to 3 show no per-request context allocation in the logs, scenario 2
turn five prefills only the new turn's tokens (visible in `hypnos_prefix_cache_tokens_reused_total`),
and scenario 3 shows one mtmd encode for the image across three turns. `cargo test --workspace`
passes with the mock backend tests extended for the worker lifecycle (start on promote, stop on
demote, pending jobs fail cleanly on demotion).

---

## 5. Work item C: continuous batching across slots

### Problem

With B in place, slots exist but the worker still processes one job at a time. llama.cpp can decode
tokens for several sequences in one `llama_decode`; the GPU cost of a decode step for four
sequences is close to the cost for one at these model sizes.

### Change

In the worker loop, every iteration:

1. Admit pending jobs into free slots (up to `slots`).
2. For each slot in `Prefill` state, add up to `n_batch - already_added` prompt tokens to the shared
   `LlamaBatch` (chunked prefill: a long prompt does not starve decoding slots).
3. For each slot in `Decode` state, add its last sampled token.
4. One `ctx.decode(&batch)`.
5. For every slot that had its last token with logits in this batch, sample, stream the piece,
   check stop conditions and the cancellation flag.

Fairness is round-robin over slots; priority is not needed inside one model. Multimodal chunks
are evaluated with `eval_chunks` for the owning slot's sequence id before it enters the shared
batch (mtmd requires a non-causal batch of its own for an image chunk, as the comment at
`llama.rs:288` already explains).

`n_batch` and `n_ubatch` are set once per context from the largest expected multimodal chunk
(today's logic at `llama.rs:287-291`, applied at context creation instead of per request).

Config: `slots` is the only knob. A suggested macOS and small-GPU default is 1, a 24 GB CUDA
default is 4 for the 26B and 2 for the E4B.

Done when: bench scenario 4 (four concurrent clients) shows aggregate decode tokens per second at
least 2.5 times the single-client number, with single-client TTFT unchanged within 10 percent.

---

## 6. Work item D: streaming TTS

### Problem

`synthesize` concatenates all chunks before returning. For a paragraph that is several seconds of
synthesis before the first byte, and it blocks the voice pipeline in E from overlapping.

### Change

- `ModelBackend::synthesize_speech_stream(req, rctx, tx: mpsc::Sender<AudioChunk>)` next to the
  existing method; `AudioChunk { pcm: Vec<f32>, sample_rate, last: bool }`.
- `kokoro.rs`: run the chunk loop as today but send each chunk's samples (and the inter-chunk
  silence) through `tx` as soon as `session.run` returns.
- `hypnos-server`: `audio_speech` streams when `response_format` is `pcm` (raw 16-bit little-endian
  at 24 kHz, `Content-Type: audio/pcm`, `Transfer-Encoding: chunked`) or when the request sets
  `stream: true`. For `wav` streaming, emit a header with the maximum data length (`0xFFFFFFFF`)
  which every mainstream player and browser accepts for live WAV; non-streaming `wav` is unchanged.
  The `ActiveModelLease` is held by the stream state the same way `ChatSseState` holds it today.
- Chunking granularity: `phonemize_chunks` splits at `MAX_PHONEMES` and sentence ends; keep that,
  but split the first chunk shorter (one clause) so time to first audio is minimal.

Done when: bench scenario 5 shows time to first audio under 300 ms on the reference machine and
total synthesis time unchanged; `hypnos_tts_ttfa_ms` is populated.

---

## 7. Work item E: in-process voice pipeline

### Problem

A voice turn is three round trips through the client. Nothing overlaps, the TTS model may have been
evicted between calls, and the user hears nothing until the whole LLM answer exists.

### Change

#### 7.1 Endpoint

`POST /v1/audio/conversations` (Hypnos extension; an OpenAI Realtime-style WebSocket is deferred):

- Request: multipart with `audio` (WAV, MP3, FLAC as today), `messages` (JSON, prior turns),
  `stt_model`, `chat_model`, `tts_model`, `voice`, `max_tokens`.
- Response: SSE events in order: `transcript` (final STT text), repeated `delta` (LLM text pieces),
  repeated `audio` (base64 PCM chunks with sequence numbers), `done` (usage and telemetry). SSE is
  chosen over a binary stream because the playground already consumes SSE and the three event
  kinds interleave.

#### 7.2 Stages

```
audio ──▶ [STT: gemma4:e4b via mtmd] ──text──▶ [LLM: gemma4:26b worker] ──deltas──▶
          sentence splitter ──sentences──▶ [TTS: kokoro worker, CPU] ──pcm──▶ SSE
```

- STT is one chat call on the E4B model with the audio attached (what `transcribe` does today).
- LLM deltas stream from the B worker. A sentence splitter buffers deltas and cuts at `.`, `!`, `?`
  followed by whitespace, or at a maximum of about 200 characters, and forwards each sentence to the
  TTS stage. The first cut is made earlier (first clause boundary or 60 characters) to minimise
  time to first audio.
- TTS runs on CPU (ONNX Runtime) so it overlaps with GPU decoding without contention. Sentences are
  synthesized in order; audio chunks are forwarded as D produces them.
- Back-pressure: bounded channels between stages; if the client stops reading, the LLM worker's
  `tx.blocking_send` fails and the slot is freed, as today.

#### 7.3 Pipeline lease

`Scheduler::acquire_pipeline(&[stt, chat, tts], ctx)`:

- Acquires leases in a fixed order (largest model first) to avoid lock-order problems.
- Before acquiring, asks the planner whether all three fit simultaneously with the chat model's
  requested context. If yes, holds all three leases for the whole request. If no, degrades to
  sequential mode: STT lease is released after the transcript, then the chat lease is acquired. The
  response includes which mode ran in `hypnos_telemetry`.
- Eviction candidates exclude any model under a pipeline lease (already true for ordinary leases via
  `idle_vram_eviction_candidates`; the pipeline simply holds more than one).

#### 7.4 Files

- `hypnos-server/src/lib.rs`: handler, SSE event types, sentence splitter (small pure function, unit
  tested).
- `hypnos-scheduler/src/lib.rs`: `acquire_pipeline`, planner query for co-residency.
- `hypnos-core/src/openai.rs`: request and event structs.
- `frontend/src/App.svelte`: a "voice" tab that records, posts, plays chunks as they arrive. Not
  essential for the backend acceptance but needed to exercise the path.

Done when: bench scenario 6 shows time to first audio below the time of a full LLM answer by a
clear margin (target: first audio within 1.5 s of the transcript on the reference machine, with the
LLM still generating), and three concurrent voice requests complete without eviction thrash
(`hypnos_scheduler_evictions_total` unchanged during the run when all three models fit).

---

## 8. Acceptance for the plan as a whole

Against the baseline from A, on the same machine and models:

| Scenario | Target |
| :--- | :--- |
| 2. Five-turn chat, turn five | prefill tokens equal to the new turn only; TTFT at least 3 times lower than baseline |
| 3. Vision chat, three turns | exactly one image encode in the logs; turn two and three TTFT close to text-only |
| 4. Four concurrent clients | aggregate decode tokens per second at least 2.5 times single-client; single-client TTFT within 10 percent |
| 5. TTS paragraph | time to first audio under 300 ms |
| 6. Voice round trip | first audio within 1.5 s of transcript while generation continues |
| all | `cargo test --workspace` green; `smoke_test.sh` green; no increase in `hypnos_vram_unaccounted_bytes` at idle |

---

## 9. Deferred, with the reason

- **KV state offload across tiers.** `state_seq_get` / `state_seq_set` make a slot's KV portable;
  parking idle sessions in Tier 2 pinned RAM or Tier 3 and restoring in milliseconds is the natural
  Hypnos extension and nobody else does it. Deferred because it needs session identity in the API
  (a `session_id` or conversation hash) and B's slot model first.
- **Standalone encoder-output cache** keyed by media hash, usable even when the text prefix differs.
  B's chunk-id matching covers the common case; the standalone cache also needs a binding for
  `mtmd_get_output_embd` and an embeddings-input batch path.
- **Batched embeddings** (multiple sequence ids per decode). Small and worthwhile; it slots into the
  B worker naturally once embeddings jobs flow through it.
- **Diffusion step cache and conditioning reuse.** sd.cpp already implements them; this is config
  plumbing (`sd_cache_params_t`, `conditioning_cache_size`) plus a quality check on MOSS-Image.
  Also freeing the 5.7 GiB text encoder after conditioning. Deferred only because it is independent
  of everything above and should not block it.
- **Image previews** via the existing progress callback (`sd_set_progress_callback`).
- **Realtime WebSocket API** with VAD and barge-in. E is the HTTP half of it; the duplex half is a
  separate project.

---

## 10. Suggested order and checkpoints

1. A, then commit the baseline table.
2. B with `slots = 1` only: persistent context plus prefix reuse. Re-run A. This is the first
   checkpoint and already delivers the largest single win for the playground's typical use.
3. B multimodal chunk matching. Re-run scenario 3.
4. D (independent, can be done in parallel with 2 and 3).
5. C. Re-run scenario 4.
6. E. Re-run scenario 6.
7. Update `README.md` (new config keys, new endpoint, new metrics) and `hypnos.example.toml`.
