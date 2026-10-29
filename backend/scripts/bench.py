#!/usr/bin/env python3
"""Execution-efficiency benchmark for a running Hypnos server (stdlib only).

    python backend/scripts/bench.py [--host http://127.0.0.1:8080] [--out docs/bench/baseline-YYYY-MM-DD.md]
                                    [--scenarios 1,2,3,4,5,6] [--clients 4]

Scenarios (see docs/execution-efficiency-plan.md, work item A):
  1. cold single chat, ~512-token prompt, 256 output tokens
  2. the same conversation extended over five turns          (re-prefill cost)
  3. vision chat, one image, three turns                      (re-encoding cost)
  4. N concurrent chat clients on the same model              (batching)
  5. TTS of a 600-character paragraph, time to first byte     (streaming TTS)
  6. voice round trip: WAV in -> transcript -> reply -> speech (pipeline overlap)

Every number is measured client-side from this process, plus whatever the server reports in
`hypnos_telemetry` (and in `/metrics` once the engine metrics exist). The script prints a
Markdown report and optionally writes it to --out.
"""

import argparse
import datetime as dt
import io
import json
import os
import struct
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
import zlib

TIMEOUT = 900
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIXTURES = os.path.join(ROOT, "backend", "tests", "fixtures")

PARAGRAPH = (
    "The lighthouse keeper logged the weather twice a day, at dawn and at dusk, in a ledger "
    "bound in green cloth. Wind direction, swell height, visibility in nautical miles, and the "
    "number of vessels sighted were recorded in the same careful hand for forty-one years. When "
    "the station was automated the ledgers were moved to the county archive, where a researcher "
    "later used them to reconstruct a century of coastal storms with a precision that satellite "
    "records could not match for the earlier decades. "
)
PARAGRAPH_600 = (
    "Hypnos keeps every model inside one process and moves weights between graphics memory, "
    "pinned host memory, and solid state storage as requests arrive. The scheduler decides what "
    "stays resident; the execution path decides how fast a resident model answers. This paragraph "
    "exists only to measure how long the speech engine takes before the first audible sample "
    "reaches the listener, and how much faster than real time the whole passage is rendered. "
    "Six hundred characters is roughly thirty seconds of speech at a natural pace, which is long "
    "enough to expose the difference between buffering everything and streaming each sentence."
)


# ----------------------------------------------------------------------------------------------
# HTTP helpers
# ----------------------------------------------------------------------------------------------


class Client:
    def __init__(self, host):
        self.host = host.rstrip("/")
        # No proxy handling: on Windows urllib's default opener consults the system proxy settings
        # (registry / PAC) for every request, which costs about two seconds per call to localhost
        # and would be misread as server time to first token.
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _req(self, path, data=None, headers=None, method=None):
        req = urllib.request.Request(self.host + path, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        return req

    def _open(self, req):
        return self.opener.open(req, timeout=TIMEOUT)

    def get_json(self, path):
        with self._open(self._req(path)) as r:
            return json.loads(r.read())

    def get_text(self, path):
        with self._open(self._req(path)) as r:
            return r.read().decode("utf-8", "replace")

    def post_json(self, path, body):
        data = json.dumps(body).encode()
        req = self._req(path, data, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with self._open(req) as r:
                raw = r.read()
                return json.loads(raw), (time.perf_counter() - t0) * 1000.0, dict(r.headers)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:300].decode('utf-8', 'replace')}")

    def post_binary(self, path, body, skip=0):
        """POST JSON, return (bytes, ttfb_ms, total_ms, headers). `ttfb` is measured at the first
        byte after `skip` bytes (so a streamed WAV header does not count as audio)."""
        data = json.dumps(body).encode()
        req = self._req(path, data, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with self._open(req) as r:
                first = r.read(skip + 1)
                ttfb = (time.perf_counter() - t0) * 1000.0
                rest = r.read()
                total = (time.perf_counter() - t0) * 1000.0
                return first + rest, ttfb, total, dict(r.headers)
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:300].decode('utf-8', 'replace')}")

    def post_multipart(self, path, fields, file_field, filename, file_bytes, content_type):
        boundary = "----hypnosbench%08x" % (int(time.time() * 1000) & 0xFFFFFFFF)
        buf = io.BytesIO()
        for k, v in fields.items():
            buf.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
        buf.write(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n".encode()
        )
        buf.write(file_bytes)
        buf.write(f"\r\n--{boundary}--\r\n".encode())
        req = self._req(path, buf.getvalue(), {"Content-Type": f"multipart/form-data; boundary={boundary}"})
        t0 = time.perf_counter()
        try:
            with self._open(req) as r:
                return json.loads(r.read()), (time.perf_counter() - t0) * 1000.0
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} on {path}: {e.read()[:300].decode('utf-8', 'replace')}")

    def stream_chat(self, body):
        """Streaming chat completion. Returns a dict with client-side timings and server telemetry."""
        body = dict(body, stream=True)
        data = json.dumps(body).encode()
        req = self._req("/v1/chat/completions", data, {"Content-Type": "application/json"})
        t0 = time.perf_counter()
        first_delta = None
        text = []
        usage = None
        telemetry = None
        finish = None
        error = None
        try:
            with self._open(req) as r:
                for raw in r:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if "error" in chunk:
                        error = chunk["error"].get("message", str(chunk["error"]))
                        break
                    choice = (chunk.get("choices") or [{}])[0]
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        if first_delta is None:
                            first_delta = time.perf_counter()
                        text.append(content)
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    if "usage" in chunk:
                        usage = chunk["usage"]
                    if "hypnos_telemetry" in chunk:
                        telemetry = chunk["hypnos_telemetry"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"HTTP {e.code} on /v1/chat/completions: {e.read()[:300].decode('utf-8', 'replace')}")
        t1 = time.perf_counter()
        if error:
            raise RuntimeError(error)
        usage = usage or {}
        ttft_ms = (first_delta - t0) * 1000.0 if first_delta else None
        total_ms = (t1 - t0) * 1000.0
        completion = usage.get("completion_tokens", 0)
        decode_ms = (t1 - first_delta) * 1000.0 if first_delta else None
        decode_tps = completion / (decode_ms / 1000.0) if decode_ms and decode_ms > 0 and completion > 1 else None
        return {
            "text": "".join(text),
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
            "completion_tokens": completion,
            "decode_tps": decode_tps,
            "finish_reason": finish,
            "telemetry": telemetry or {},
        }

    def metrics(self, prefixes):
        """Lines of /metrics whose name starts with one of `prefixes`."""
        try:
            body = self.get_text("/metrics")
        except Exception as e:  # noqa: BLE001
            return [f"# /metrics unavailable: {e}"]
        return [l for l in body.splitlines() if l and not l.startswith("#") and l.startswith(tuple(prefixes))]


# ----------------------------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------------------------


def make_png(w=256, h=256):
    """A red disc and a blue square on white, encoded as RGB PNG with the stdlib only."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            dx, dy = x - w * 0.35, y - h * 0.5
            if dx * dx + dy * dy < (w * 0.2) ** 2:
                rows += b"\xd8\x20\x20"
            elif w * 0.62 < x < w * 0.92 and h * 0.3 < y < h * 0.7:
                rows += b"\x20\x40\xd8"
            else:
                rows += b"\xff\xff\xff"

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(rows), 9))
        + chunk(b"IEND", b"")
    )


def wav_seconds(data):
    try:
        with wave.open(io.BytesIO(data)) as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:  # noqa: BLE001
        return None


def fmt(v, digits=0):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def eng(tel, key):
    """Engine timing from telemetry when the server reports it (added with work item A)."""
    return ((tel or {}).get("engine") or {}).get(key)


# ----------------------------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------------------------


class Report:
    def __init__(self):
        self.lines = []

    def h(self, text):
        self.lines += ["", f"### {text}", ""]

    def p(self, text):
        self.lines += [text, ""]

    def table(self, header, rows):
        self.lines.append("| " + " | ".join(header) + " |")
        self.lines.append("| " + " | ".join(":---" for _ in header) + " |")
        for r in rows:
            self.lines.append("| " + " | ".join(str(c) for c in r) + " |")
        self.lines.append("")

    def fail(self, what, err):
        self.p(f"**{what} failed:** `{err}`")
        print(f"  [FAIL] {what}: {err}", file=sys.stderr)


def chat_row(label, r):
    tel = r["telemetry"]
    return [
        label,
        fmt(r["ttft_ms"]),
        fmt(r["total_ms"]),
        fmt(r["prompt_tokens"]),
        fmt(r["cached_tokens"]),
        fmt(r["completion_tokens"]),
        fmt(r["decode_tps"], 1),
        fmt(eng(tel, "ttft_ms"), 0),
        fmt(eng(tel, "queue_ms"), 0),
        fmt(eng(tel, "context_ms"), 0),
        fmt(eng(tel, "prefill_ms"), 0),
        fmt(eng(tel, "prefill_tokens_per_s"), 0),
        fmt(eng(tel, "decode_tokens_per_s"), 1),
        fmt(tel.get("tier_switch_latency_ms"), 0) if tel else "n/a",
        tel.get("previous_tier", "n/a") if tel else "n/a",
    ]


CHAT_HEADER = [
    "run", "TTFT ms", "total ms", "prompt tok", "cached tok", "completion tok", "decode tok/s (client)",
    "engine TTFT ms", "queue ms", "context ms", "prefill ms",
    "prefill tok/s", "decode tok/s (engine)", "tier switch ms", "previous tier",
]


def scenario_1(c, rep, model):
    rep.h("1. Cold single chat (~512-token prompt, 256 output tokens)")
    prompt = "Summarize the following report in detail, then list three open questions.\n\n" + PARAGRAPH * 5
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 256, "temperature": 0}
    print("-- scenario 1: cold chat ...")
    r = c.stream_chat(body)
    rep.table(CHAT_HEADER, [chat_row("cold", r)])
    print(f"     ttft={fmt(r['ttft_ms'])}ms total={fmt(r['total_ms'])}ms prompt={r['prompt_tokens']} completion={r['completion_tokens']}")
    return r


def scenario_2(c, rep, model, turns=5):
    rep.h(f"2. Same conversation extended over {turns} turns (re-prefill cost)")
    messages = [
        {"role": "system", "content": "You are a concise technical writer."},
        {"role": "user", "content": "Here is a report. Give me its first key point in two sentences.\n\n" + PARAGRAPH * 5},
    ]
    follow_ups = [
        "Give me the next key point, two sentences.",
        "And the next one.",
        "Name one risk the report implies.",
        "Suggest one follow-up study in two sentences.",
        "Which of your points is strongest and why?",
    ]
    rows = []
    print(f"-- scenario 2: {turns}-turn conversation ...")
    for turn in range(1, turns + 1):
        body = {"model": model, "messages": messages, "max_tokens": 96, "temperature": 0}
        r = c.stream_chat(body)
        rows.append(chat_row(f"turn {turn}", r))
        print(f"     turn {turn}: ttft={fmt(r['ttft_ms'])}ms prompt={r['prompt_tokens']} completion={r['completion_tokens']}")
        messages = messages + [
            {"role": "assistant", "content": r["text"] or "(empty)"},
            {"role": "user", "content": follow_ups[(turn - 1) % len(follow_ups)]},
        ]
    rep.table(CHAT_HEADER, rows)
    rep.p("With the prefix cache, turn N prefills only the new turn (see `cached tok`); without it every turn re-prefills the whole history.")


def scenario_3(c, rep, model, turns=3):
    rep.h(f"3. Vision chat, one image, {turns} turns (re-encoding cost)")
    import base64

    png = base64.b64encode(make_png()).decode()
    image_part = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png}}
    questions = [
        "Describe this image in one sentence.",
        "What colour is the round shape? One word.",
        "How many shapes are there? Answer with a number.",
    ]
    messages = [{"role": "user", "content": [{"type": "text", "text": questions[0]}, image_part]}]
    rows = []
    print(f"-- scenario 3: vision, {turns} turns ...")
    for turn in range(1, turns + 1):
        body = {"model": model, "messages": messages, "max_tokens": 48, "temperature": 0}
        r = c.stream_chat(body)
        rows.append(chat_row(f"turn {turn}", r))
        print(f"     turn {turn}: ttft={fmt(r['ttft_ms'])}ms prompt={r['prompt_tokens']} -> {r['text'][:60]!r}")
        if turn < turns:
            messages = messages + [
                {"role": "assistant", "content": r["text"] or "(empty)"},
                {"role": "user", "content": questions[turn]},
            ]
    rep.table(CHAT_HEADER, rows)
    rep.p("The image is a 256x256 PNG (red disc, blue square). A cached image counts in `cached tok`; without the cache every turn re-runs the vision tower.")


def scenario_4(c, rep, model, clients=4):
    rep.h(f"4. {clients} concurrent chat clients on the same model (batching)")
    prompt = "Write a short story about a lighthouse keeper. Keep going until you are stopped."
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0}
    print("-- scenario 4: single-client reference ...")
    solo = c.stream_chat(body)
    results = [None] * clients
    errors = [None] * clients
    barrier = threading.Barrier(clients)

    def worker(i):
        try:
            barrier.wait()
            results[i] = c.stream_chat(dict(body, seed=i))
        except Exception as e:  # noqa: BLE001
            errors[i] = str(e)

    print(f"-- scenario 4: {clients} concurrent clients ...")
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(clients)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall_ms = (time.perf_counter() - t0) * 1000.0
    rows = [chat_row("solo", solo)]
    total_completion = 0
    for i, r in enumerate(results):
        if r is None:
            rows.append([f"client {i}", "error", errors[i] or "?"] + [""] * (len(CHAT_HEADER) - 3))
            continue
        total_completion += r["completion_tokens"] or 0
        rows.append(chat_row(f"client {i}", r))
    rep.table(CHAT_HEADER, rows)
    agg = total_completion / (wall_ms / 1000.0) if wall_ms > 0 else None
    ratio = agg / solo["decode_tps"] if agg and solo["decode_tps"] else None
    rep.table(
        ["metric", "value"],
        [
            ["concurrent wall time ms", fmt(wall_ms)],
            ["aggregate completion tokens", total_completion],
            ["aggregate tokens/s (completion tokens / wall)", fmt(agg, 1)],
            ["single-client decode tokens/s", fmt(solo["decode_tps"], 1)],
            ["aggregate / single-client", fmt(ratio, 2)],
            ["slowest client TTFT ms", fmt(max((r["ttft_ms"] or 0) for r in results if r), 0)],
        ],
    )
    print(f"     aggregate={fmt(agg, 1)} tok/s vs solo={fmt(solo['decode_tps'], 1)} tok/s (x{fmt(ratio, 2)}), wall={fmt(wall_ms)}ms")


def scenario_5(c, rep, model, voice):
    rep.h("5. TTS of a 600-character paragraph (time to first audio)")
    print("-- scenario 5: tts ...")
    rows = []
    for fmt_name, streaming in (("wav", False), ("pcm", False), ("wav", True), ("pcm", True)):
        body = {"model": model, "input": PARAGRAPH_600, "voice": voice, "response_format": fmt_name}
        if streaming:
            body["stream"] = True
        skip = 44 if (fmt_name == "wav" and streaming) else 0
        try:
            data, ttfa, total, headers = c.post_binary("/v1/audio/speech", body, skip=skip)
        except Exception as e:  # noqa: BLE001
            rows.append([fmt_name, "yes" if streaming else "no", len(PARAGRAPH_600), "error", str(e)[:60], "", "", "", ""])
            print(f"     {fmt_name} stream={streaming}: {e}")
            continue
        if fmt_name == "wav" and not streaming:
            audio_secs = wav_seconds(data)
        else:
            audio_secs = max(len(data) - skip, 0) / 2 / 24000.0
        infer_ms = headers.get("x-hypnos-inference-ms")
        synth_s = float(infer_ms) / 1000.0 if infer_ms else total / 1000.0
        rtf = synth_s / audio_secs if audio_secs else None
        label = "yes" if streaming else "no"
        rows.append([fmt_name, label, len(PARAGRAPH_600), fmt(ttfa), fmt(total), fmt(audio_secs, 2), fmt(synth_s, 2), fmt(rtf, 3), len(data)])
        print(f"     {fmt_name} stream={label}: ttfa={fmt(ttfa)}ms total={fmt(total)}ms audio={fmt(audio_secs, 2)}s rtf={fmt(rtf, 3)}")
    rep.table(["format", "stream", "chars", "TTFA ms", "total ms", "audio s", "synthesis s", "RTF", "bytes"], rows)
    rep.p("TTFA = time to the first audio byte (after the WAV header when streaming). Without streaming it equals the whole synthesis time. RTF = synthesis seconds / audio seconds (lower is faster).")


def scenario_6(c, rep, stt_model, chat_model, tts_model, voice):
    rep.h("6. Voice round trip: WAV in -> transcript -> reply -> speech out")
    path = os.path.join(FIXTURES, "jfk.wav")
    if not os.path.exists(path):
        rep.fail("scenario 6", f"fixture missing: {path}")
        return
    with open(path, "rb") as f:
        wav = f.read()
    audio_secs = wav_seconds(wav)
    print("-- scenario 6: voice round trip (3 HTTP calls, client-driven) ...")
    t0 = time.perf_counter()
    stt, stt_ms = c.post_multipart("/v1/audio/transcriptions", {"model": stt_model}, "file", "jfk.wav", wav, "audio/wav")
    transcript = (stt.get("text") or "").strip()
    stt_tel = stt.get("hypnos_telemetry") or {}
    print(f"     stt: {stt_ms:.0f}ms -> {transcript[:70]!r}")
    chat = c.stream_chat({
        "model": chat_model,
        "messages": [
            {"role": "system", "content": "Answer in two short spoken sentences."},
            {"role": "user", "content": transcript or "Say hello."},
        ],
        "max_tokens": 80,
        "temperature": 0,
    })
    t_chat_done = time.perf_counter()
    print(f"     llm: ttft={fmt(chat['ttft_ms'])}ms total={fmt(chat['total_ms'])}ms -> {chat['text'][:70]!r}")
    reply = chat["text"].strip() or "I did not catch that."
    speech, ttfb, tts_total, headers = c.post_binary("/v1/audio/speech", {"model": tts_model, "input": reply, "voice": voice})
    t_end = time.perf_counter()
    first_audio_ms = (t_chat_done - t0) * 1000.0 + ttfb
    out_secs = wav_seconds(speech)
    rep.table(
        ["stage", "ms"],
        [
            ["STT (whole clip, %.1f s of audio)" % (audio_secs or 0), fmt(stt_ms)],
            ["STT engine inference", fmt(stt_tel.get("inference_latency_ms"), 0)],
            ["STT RTF", fmt((stt_tel.get("inference_latency_ms") or stt_ms) / 1000.0 / audio_secs, 3) if audio_secs else "n/a"],
            ["LLM TTFT", fmt(chat["ttft_ms"])],
            ["LLM total", fmt(chat["total_ms"])],
            ["TTS time to first byte", fmt(ttfb)],
            ["TTS total (%.1f s of audio)" % (out_secs or 0), fmt(tts_total)],
            ["**time to first audio from request start**", fmt(first_audio_ms)],
            ["end to end", fmt((t_end - t0) * 1000.0)],
        ],
    )
    rep.p(f"Transcript: `{transcript[:160]}`  \nReply: `{reply[:160]}`")
    rep.p("Today the client drives three sequential calls; nothing overlaps and the first audio arrives only after the full reply is synthesized.")
    print(f"     first audio at {first_audio_ms:.0f}ms, end to end {(t_end - t0) * 1000.0:.0f}ms")


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:8080")
    ap.add_argument("--out", help="write the Markdown report here (directories are created)")
    ap.add_argument("--scenarios", default="1,2,3,4,5,6")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--chat-model", default="gemma4:26b-q4")
    ap.add_argument("--stt-model", default="gemma4:e4b-q4")
    ap.add_argument("--tts-model", default="kokoro-82m-q4")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--label", default="", help="free text recorded in the report header (e.g. git sha, 'after B')")
    args = ap.parse_args()

    c = Client(args.host)
    try:
        health = c.get_json("/health")
    except Exception as e:  # noqa: BLE001
        print(f"server not reachable at {args.host}: {e}", file=sys.stderr)
        return 1
    try:
        mem = c.get_json("/v1/memory/status")
    except Exception:  # noqa: BLE001
        mem = {}

    rep = Report()
    today = dt.date.today().isoformat()
    rep.lines += [f"# Hypnos execution benchmark, {today}", ""]
    vram_total = mem.get("tier1_vram_real_total_bytes") or mem.get("tier1_vram_total_bytes")
    rep.table(
        ["field", "value"],
        [
            ["server", f"{args.host} (hypnos {health.get('version', '?')})"],
            ["label", args.label or "baseline (current main)"],
            ["chat / stt / tts models", f"{args.chat_model} / {args.stt_model} / {args.tts_model}"],
            ["VRAM total", f"{vram_total / 2**30:.1f} GiB" if vram_total else "n/a"],
            ["resident at start", ", ".join(f"{m['id']}={m['current_tier']}" for m in mem.get("models", [])) or "n/a"],
            ["platform", f"{sys.platform}, python {sys.version.split()[0]}"],
        ],
    )
    rep.p("Latencies are measured by this client over HTTP on the same machine. TTFT = request sent to first non-empty content delta. Engine columns come from `hypnos_telemetry.engine`: engine TTFT = lease granted to first visible token; queue = waiting for the model lock; context = llama.cpp context creation; prefill = prompt evaluation.")

    wanted = {int(s) for s in args.scenarios.split(",") if s.strip()}
    metric_prefixes = ("hypnos_chat_", "hypnos_tts_", "hypnos_stt_", "hypnos_image_", "hypnos_prefix_cache_", "hypnos_slots_", "hypnos_scheduler_")
    before = c.metrics(metric_prefixes)

    steps = [
        (1, lambda: scenario_1(c, rep, args.chat_model)),
        (2, lambda: scenario_2(c, rep, args.chat_model)),
        (3, lambda: scenario_3(c, rep, args.chat_model)),
        (4, lambda: scenario_4(c, rep, args.chat_model, args.clients)),
        (5, lambda: scenario_5(c, rep, args.tts_model, args.voice)),
        (6, lambda: scenario_6(c, rep, args.stt_model, args.chat_model, args.tts_model, args.voice)),
    ]
    started = time.perf_counter()
    for n, fn in steps:
        if n not in wanted:
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            rep.fail(f"scenario {n}", e)

    after = c.metrics(metric_prefixes)
    rep.h("Server metrics after the run")
    rep.p("Lines from `/metrics` for the engine, scheduler and cache metrics (only those the server exports).")
    rep.lines += ["```", *(after or ["(none)"]), "```", ""]
    if before and after:
        changed = sorted(set(after) - set(before))
        if changed:
            rep.lines += ["Changed during the run:", "", "```", *changed, "```", ""]
    rep.p(f"Total benchmark wall time: {(time.perf_counter() - started):.0f} s.")

    text = "\n".join(rep.lines).rstrip() + "\n"
    print()
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"report written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
