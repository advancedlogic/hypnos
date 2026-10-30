# Soak test 2026-10-30

RTX 5090 laptop, default lineup, confirmation run after BUG-2: 4 clients for 12 min against `http://127.0.0.1:8080` (mix `chat:3,stream:3,vision:1,tts:2,embed:2,entities:2,documents:1`, an image every 180 s, warm-up 3 min excluded from the growth verdicts).


### Verdict

| Check | Result | Detail |
| :--- | :--- | :--- |
| process footprint flat | pass | 35.75 -> 35.85 GB (+0.3% after warm-up, limit 5.0% or 0.54 GB) |
| VRAM unaccounted flat | pass | 0.54 -> 0.36 GB (-32.3% after warm-up, limit 25.0% or 1.07 GB) |
| no 5xx other than 507 | FAIL | 1 server errors |
| 507 capacity refusals | info | 0 (a model that did not fit while the others were leased: expected under load, not a fault) |
| 429 under 5% of requests | pass | 0 of 1130 |
| no stall | pass | longest gap between completions below 5 min |
| no load failures | pass | 0.0 during the run |
| every eviction explained | pass | 0 without a reason |


### Requests

| Kind | OK | Errors | p50 ms | p95 ms | TTFT/TTFB p50 | p95 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| chat | 244 | 0 | 4559 | 6987 |  |  |
| documents | 73 | 0 | 312 | 345 |  |  |
| embed | 158 | 0 | 48 | 65 |  |  |
| entities | 166 | 0 | 45 | 70 |  |  |
| image | 2 | 1 | 139531 | 145361 |  |  |
| stream | 245 | 0 | 4578 | 6893 | 723 | 1184 |
| tts | 161 | 0 | 801 | 1591 | 796 | 1590 |
| vision | 80 | 0 | 1999 | 2469 |  |  |

Completion tokens generated: 23343. Documents stored: 73, released by the client: 69.


### Errors

| Kind | Count | Message |
| :--- | :--- | :--- |
| image | 1 | `HTTP 500 on /v1/images/generations: {"error":{"message":"Backend execution error` |


### Samples

| min | footprint GB | VRAM booked | real free | unaccounted | pinned GB | promotions | evictions | load fail | rejected | 5xx | log | GPU C |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0.0 | 22.86 | 1.39 | 23.85 | 0.41 | 25.85 | 2 | 0 | 0 | 0 | 8 | 0 | 55 |
| 0.5 | 35.82 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 46 | 62 |
| 1.0 | 35.84 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 87 | 68 |
| 1.5 | 35.83 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 129 | 68 |
| 2.0 | 35.82 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 169 | 72 |
| 2.5 | 35.83 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 68 |
| 3.0 | 35.75 | 24.52 | 0.00 | 1.13 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 67 |
| 3.5 | 35.93 | 24.52 | 0.59 | 0.54 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 68 |
| 4.0 | 35.72 | 24.52 | 0.55 | 0.58 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 69 |
| 4.5 | 35.84 | 24.52 | 0.73 | 0.40 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 5.0 | 35.72 | 24.52 | 0.61 | 0.52 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 5.5 | 35.72 | 24.52 | 0.54 | 0.59 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 75 |
| 6.0 | 35.83 | 24.52 | 1.16 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 69 |
| 6.5 | 35.83 | 24.52 | 1.18 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 68 |
| 7.1 | 35.83 | 24.52 | 1.18 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 69 |
| 7.6 | 35.83 | 24.52 | 1.18 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 8.1 | 35.83 | 24.52 | 1.16 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 8.6 | 35.84 | 24.52 | 0.71 | 0.42 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 69 |
| 9.1 | 35.90 | 24.52 | 0.59 | 0.54 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 71 |
| 9.6 | 35.91 | 24.52 | 0.59 | 0.54 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 69 |
| 10.1 | 35.85 | 24.52 | 0.59 | 0.54 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 71 |
| 10.6 | 35.85 | 24.52 | 0.66 | 0.47 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 11.1 | 35.84 | 24.52 | 1.29 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |
| 11.6 | 35.85 | 24.52 | 1.30 | 0.00 | 28.44 | 6 | 0 | 0 | 0 | 9 | 200 | 70 |

### Findings

- No entities error in 1,130 requests: BUG-2 is confirmed fixed (the 45-minute run had two).
- The one 500 is an image generation: MOSS-Image was placed fully on the GPU (the booking fit the
  planner's view) and then stable-diffusion.cpp found 0 MB really free when it needed 518 MB of
  workspace for the text encoder, with the device already holding the chat model's KV. That is
  the planner trusting bookings on a device that is really full, filed under PERF-7; the request
  failed cleanly (500, lease released, nothing leaked) and the next image succeeded.
- Footprint flat at 35.8 GB (+0.3 %), unaccounted VRAM flat, no 429, no stall.

