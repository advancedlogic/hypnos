# MEM-4 baseline: the turn after a swap (2026-10-30)

What MEM-4 (keep a slot's KV state across a demotion, `docs/pagination.md`) would remove, measured
on today's server before any change. Apple Silicon (Metal, unified memory), `gemma4:26b-q4`,
`num_ctx` 20480, temperature 0, 40-token answers. Script: `backend/scripts/mem4_bench.py [base url]`.

Per conversation size: turn 1 with a synthetic document, turn 2 warm (in-memory prefix reuse),
then `POST /v1/memory/tier` moves the idle 26B to Tier 2 (what an eviction does to its context),
then turn 3. TTFT is measured by the client; the rest is the response telemetry.

| Prompt tokens | Turn 1 (cold prefill) | Turn 2 warm | Turn 3 after the swap | of which tier switch | of which prefill | cached |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1,690 | 707 ms | 104 ms (1,576 cached) | 1,401 ms | 952 ms | 410 ms | 0 |
| 6,325 | 2,178 ms | 114 ms (6,230 cached) | 2,555 ms | 929 ms | 1,583 ms | 0 |
| 12,629 | 3,971 ms | 122 ms (12,533 cached) | 4,705 ms | 948 ms | 3,704 ms | 0 |

Reading: after the swap the whole conversation is prefilled again (cached 0), about 3.4k tok/s.
MEM-4 would keep the tier switch and replace the prefill by a state restore (an estimated
0.1-0.45 GiB copy, not measured) plus a warm-sized prefill of the new turn (~70 ms), so turn 3
would land near 1.05-1.1 s at every size: about 1.3x faster at 1.7k tokens, 2.4x at 6.3k, 4.3x at
12.6k. On CUDA the tier switch is longer (~3.7 s), so the ratio is lower for the same seconds saved.

Side observation: after the manual move to Tier 2, turn 3 reported `previous_tier:
tier3_nvme_ssd`; to check separately (Tier 2 accounting on unified memory, or the telemetry).
