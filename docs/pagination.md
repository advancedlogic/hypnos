# KV state preservation across tier switches (proposal)

Status: proposal, tracked as MEM-4 in `TODO.md`. This replaces an earlier draft titled
"Paginated KV Cache Offloading"; the mechanism below is per-sequence state save and restore,
not paged attention (llama.cpp has no block tables, see "Out of scope").

## 1. The problem

A resident llama.cpp model keeps a persistent context with `slots` execution slots, and a new
turn of a conversation reuses the slot's KV cache: only the tokens after the common prefix are
prefilled (`engine.rs`, PERF work in `docs/execution-efficiency-plan.md`). That reuse ends when
the model leaves Tier 1: demotion drops the engine worker, which frees the context and with it
every slot's KV cache (`llama.rs`, `apply_placement`). The next turn after a swap pays a full
prefill of the conversation, and its images and audio go through the encoders again.

This is the common case in the default lineup: an image request evicts `gemma4:26b-q4` for
`moss-image`, and the next chat turn brings the 26B back cold. Weights come back in about 1 s
(Metal) to 3.7 s (CUDA) from Tier 2; the prefill of a long conversation can cost as much again.

## 2. What already exists (while the model stays resident)

| Feature | Where | Effect |
| :--- | :--- | :--- |
| Prefix reuse per slot | `engine.rs` (`select`, `kv_cache_seq_rm`) | a new turn prefills only past the longest common prefix; the slot with the longest match is chosen |
| SWA checkpoint | `engine.rs` (`state_seq_get` with `PARTIAL_ONLY` at the end of the prompt) | when the new turn diverges inside the previous reply (re-tokenization differs), the sliding-window layers roll back to the checkpoint instead of a full prefill |
| Media in the prefix | mtmd chunk ids in the slot keys | an image or clip already in the prefix is not encoded again |
| Continuous batching | the worker's batch over all slots | concurrent requests share decode steps |

Measured (`docs/bench/`): turn-5 TTFT 313 ms -> 145 ms with prefix reuse, vision turns 2-3
306 ms -> 101 ms without re-encoding, 4-client aggregate 106 -> 248 tok/s. None of this survives
a demotion; the proposal extends it across one.

## 3. Proposal

### 3.1 Park on demotion

Before the worker tears down its context for a demotion (not for a disable or a shutdown), for
each slot holding tokens:

1. Save the slot's full sequence state: `state_seq_get(seq_id, flags = 0)`, which copies the KV
   cells of that sequence from the backend buffers into a host buffer (a synchronous copy; on
   Apple Silicon a memory copy, on CUDA a device-to-host transfer).
2. Keep with it what the restore needs: the slot's token keys (text tokens and media chunk ids),
   the SWA checkpoint and its position when there is one, and a context signature (model id and
   weights file identity, KV type, `n_ctx` per slot, slot count, SWA settings, engine build).
3. Hand the parked states to the backend's state, outside the worker, so they outlive it.

### 3.2 Restore on promotion

When the worker starts again and its context signature equals the parked one, the first request
for each parked state is matched exactly as today's slot selection does (longest common prefix
over the token keys), then:

- **Divergence at or after the end of the parked state**: `state_seq_set` the full state into a
  slot and prefill only the new tokens.
- **Divergence between the SWA checkpoint and the end** (the reply re-tokenized differently):
  restore the full state, `kv_cache_seq_rm` the tail for the full-attention layers and restore
  the SWA checkpoint for the sliding-window layers, then prefill from the checkpoint. This is the
  same rule the engine applies in-context today.
- **Earlier divergence, signature mismatch, or a failed `state_seq_set`**: drop the parked state
  and prefill normally. Restoring is an optimization; it never changes an answer.

A signature mismatch is expected and fine: a promotion with a different `allocated_context_len`
or slot count builds an incompatible context.

### 3.3 Memory and policy

Per-token state for Gemma 4 26B in f16 KV, from its GGUF: 5 full-attention layers x 2 KV heads x
(512 + 512) x 2 bytes = 20 KiB per token, growing with the context; 25 sliding-window layers x 8
KV heads x (256 + 256) x 2 bytes = 200 KiB per token, but only for the last 1,024 positions (at
most ~200 MiB). A parked 8k-token conversation is therefore about 360 MiB and a 16k one about
520 MiB (estimates: measure the real size of `state_seq_get`). Three busy slots of the 26B can
park around 1.5 GiB.

That competes with the Tier 2 budget, which already has to hold the weights. So:

- a cap, `hardware.kv_park_bytes` (default a few GiB, 0 disables), booked by the
  `TieredMemoryManager` next to pinned weights and reported in `/v1/memory/status`;
- least-recently-used eviction of parked states across models when the cap or a weight
  promotion needs the room;
- parked states are dropped when the model is disabled, its weights change, or the server stops.

Plain host memory is enough for a first version (the copy is synchronous either way); pinning it
only matters if the restore is made asynchronous later.

## 4. Expected gain and how to measure it

Restoring an 8k-token conversation copies about 360 MiB (tens of milliseconds on either
platform) instead of prefilling 8k tokens (about 2.6 s at the 3,190 tok/s measured in
`docs/bench/after-B-2026-10-29.md`). The new turn's own tokens are still prefilled, so TTFT drops
to the swap time plus a short prefill, not to the decode floor.

A bench scenario in `backend/scripts/bench.sh`: build a conversation of N tokens on the 26B
(N = 2k, 8k, 16k), force a swap with an image request, send the next turn, and record its TTFT,
`tier_switch_latency_ms` and `prefill_ms`, with and without parking. Report the parked bytes and
the restore time as new engine timings (`restore_ms`, `restored_tokens`).

## 5. Out of scope

- **Paged attention.** llama.cpp allocates one KV buffer per context and shares its cells among
  sequences; vLLM-style block tables would be an engine rewrite, and fragmentation is not the
  problem here.
- **Tier 3 spill.** Writing parked states to disk adds latency and wear for a case the RAM cap
  already covers; reconsider with numbers.
- **Persistence across restarts** and an explicit session header: the longest-prefix match finds
  the conversation without one.

## 6. Open questions

- Is a full state copy fast enough on CUDA to park three slots inside a swap without adding to
  the swap latency? If not, park in the background after the weights start loading elsewhere.
- Does restoring into a context created with the same signature but a different slot index work
  as-is, or does the seq id have to match?
- Should a request that waits on a cold model trigger the restore before admission, so its queue
  time hides the copy?
