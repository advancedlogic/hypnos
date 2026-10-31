#!/usr/bin/env python3
"""Execution-efficiency benchmark for a running Hypnos server (stdlib only).

    python backend/scripts/bench.py [--host http://127.0.0.1:9999] [--out docs/bench/baseline-YYYY-MM-DD.md]
                                    [--scenarios 1,2,3,4,5,6] [--clients 4]

Scenarios (see docs/execution-efficiency-plan.md, work item A):
  1. cold single chat, ~512-token prompt, 256 output tokens
  2. the same conversation extended over five turns          (re-prefill cost)
  3. vision chat, one image, three turns                      (re-encoding cost)
  4. N concurrent chat clients on the same model              (batching)
  5. TTS of a 600-character paragraph, time to first byte     (streaming TTS)
  6. voice round trip: WAV in -> transcript -> reply -> speech (pipeline overlap)
  7. model swap latency: first request to a large model staged in Tier 3 / Tier 2 / Tier 1
     (opt-in, not in the default set; --chart writes an SVG of the result)

Every number is measured client-side from this process, plus whatever the server reports in
`hypnos_telemetry` (and in `/metrics` once the engine metrics exist). The script prints a
Markdown report and optionally writes it to --out.
"""

import argparse
import ctypes
import ctypes.util
import datetime as dt
import io
import json
import mmap
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
# Weight files of the default chat model (gemma4:26b-q4), relative to backend/models/.
DEFAULT_SWAP_FILES = ("gemma-4-26B-A4B-it-Q4_0.gguf", "mmproj-gemma-4-26B-A4B-it-BF16.gguf")
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
        fmt(eng(tel, "concurrent_slots")),
        fmt(tel.get("tier_switch_latency_ms"), 0) if tel else "n/a",
        tel.get("previous_tier", "n/a") if tel else "n/a",
    ]


CHAT_HEADER = [
    "run", "TTFT ms", "total ms", "prompt tok", "cached tok", "completion tok", "decode tok/s (client)",
    "engine TTFT ms", "queue ms", "context ms", "prefill ms",
    "prefill tok/s", "decode tok/s (engine)", "slots", "tier switch ms", "previous tier",
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
# Page cache (scenario 7)
# ----------------------------------------------------------------------------------------------


class PageCache:
    """Residency check (`mincore`) and per-file eviction of the OS page cache, without root.

    Linux evicts with `posix_fadvise(DONTNEED)`, macOS with `msync(MS_INVALIDATE)`. Both only drop
    clean pages nobody holds wired, which is the state of a model Hypnos demoted to Tier 3. On
    Windows neither is available (the standby list needs admin rights to purge), so every method
    returns None and the report says the cache was not dropped.
    """

    def __init__(self):
        self.libc = None
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
            libc.mmap.restype = ctypes.c_void_p
            libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
            libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
            libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            self.libc = libc

    @property
    def supported(self):
        return self.libc is not None

    def _with_map(self, path, fn):
        fd = os.open(path, os.O_RDONLY)
        try:
            size = os.fstat(fd).st_size
            addr = self.libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
            if addr in (None, ctypes.c_void_p(-1).value):
                raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
            try:
                return fn(addr, size)
            finally:
                self.libc.munmap(addr, size)
        finally:
            os.close(fd)

    def resident_fraction(self, paths):
        """Share of the files' pages currently in the page cache (0..1), or None."""
        if not self.supported or not paths:
            return None
        total = resident = 0
        for path in paths:
            def count(addr, size):
                pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
                vec = ctypes.create_string_buffer(pages)
                if self.libc.mincore(addr, size, vec) != 0:
                    raise OSError(ctypes.get_errno(), "mincore failed")
                return pages, sum(1 for b in vec.raw if b & 1)

            p, r = self._with_map(path, count)
            total += p
            resident += r
        return resident / total if total else None

    def evict(self, paths):
        """Drops the files' clean pages from the page cache. Returns False when unsupported."""
        if not self.supported:
            return False
        for path in paths:
            if sys.platform == "darwin":
                self._with_map(path, lambda addr, size: self.libc.msync(addr, size, 2))  # MS_INVALIDATE
            else:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
        return True


def median(values):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


SWAP_TIERS = [
    ("tier3_nvme_ssd", "Tier 3 (NVMe mmap, cold)"),
    ("tier2_pinned_ram", "Tier 2 (pinned RAM, warm)"),
    ("tier1_gpu_vram", "Tier 1 (resident, hot)"),
]


def model_tier(c, model):
    for m in c.get_json("/v1/memory/status").get("models", []):
        if m["id"] == model:
            return m["current_tier"]
    raise RuntimeError(f"model {model} not in /v1/memory/status")


def scenario_7(c, rep, model, reps, files, keep_cache):
    rep.h(f"7. Model swap latency by start tier ({model})")
    cache = PageCache()
    files = [f for f in files if os.path.exists(f)]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready."}],
        "max_tokens": 8,
        "temperature": 0,
    }
    # One unmeasured request so the model is loadable and the prompt path is warm.
    print(f"-- scenario 7: warm-up request to {model} ...")
    c.stream_chat(body)
    runs = {tier: [] for tier, _ in SWAP_TIERS}
    for rep_i in range(1, reps + 1):
        for tier, _ in SWAP_TIERS:
            _, stage_ms, _ = c.post_json("/v1/memory/tier", {"model": model, "target_tier": tier})
            evicted = False
            if tier == "tier3_nvme_ssd" and not keep_cache:
                evicted = cache.evict(files)
            actual = model_tier(c, model)
            resident = cache.resident_fraction(files)
            r = c.stream_chat(body)
            tel = r["telemetry"]
            run = {
                "rep": rep_i,
                "staged": actual,
                "stage_ms": stage_ms,
                "evicted": evicted,
                "resident": resident,
                "ttft_ms": r["ttft_ms"],
                "switch_ms": tel.get("tier_switch_latency_ms"),
                "previous_tier": tel.get("previous_tier"),
                "engine_ttft_ms": eng(tel, "ttft_ms"),
                "total_ms": r["total_ms"],
            }
            runs[tier].append(run)
            print(
                f"     rep {rep_i} {tier}: staged={actual} cache={fmt_pct(resident)} "
                f"switch={fmt(run['switch_ms'])}ms ttft={fmt(run['ttft_ms'])}ms"
            )
            if actual != tier:
                print(f"     [WARN] asked for {tier}, server staged {actual}", file=sys.stderr)

    rows = []
    for tier, _ in SWAP_TIERS:
        for run in runs[tier]:
            rows.append([
                run["rep"], tier, run["staged"], run["previous_tier"] or "n/a", fmt(run["stage_ms"]),
                fmt_pct(run["resident"]), fmt(run["switch_ms"]), fmt(run["engine_ttft_ms"]), fmt(run["ttft_ms"]),
                fmt(run["total_ms"]),
            ])
    rep.table(
        ["rep", "start tier", "staged as", "telemetry previous tier", "staging call ms", "page cache resident",
         "tier switch ms", "engine TTFT ms", "TTFT ms", "total ms"],
        rows,
    )

    summary = []
    for tier, label in SWAP_TIERS:
        rs = runs[tier]
        ttfts = [r["ttft_ms"] for r in rs if r["ttft_ms"] is not None]
        summary.append({
            "tier": tier,
            "label": label,
            "switch_ms": median(r["switch_ms"] for r in rs),
            "ttft_ms": median(ttfts),
            "ttft_min": min(ttfts) if ttfts else None,
            "ttft_max": max(ttfts) if ttfts else None,
            "resident": median(r["resident"] for r in rs),
        })
    rep.table(
        ["start tier", "runs", "median tier switch ms", "median TTFT ms", "TTFT range ms", "median page cache resident"],
        [[s["label"], len(runs[s["tier"]]), fmt(s["switch_ms"]), fmt(s["ttft_ms"]),
          f"{fmt(s['ttft_min'])}-{fmt(s['ttft_max'])}", fmt_pct(s["resident"])] for s in summary],
    )
    if keep_cache:
        cache_note = "The OS page cache was left alone (`--swap-keep-page-cache`), so Tier 3 reads may be served from RAM."
    elif cache.supported and files:
        cache_note = (
            "Before each Tier 3 run the weight files were evicted from the OS page cache "
            f"({', '.join(os.path.basename(f) for f in files)}), so Tier 3 reads come from the SSD; "
            "`page cache resident` is measured with `mincore` just before the request."
        )
    elif not files:
        cache_note = "No `--swap-files` given, so the OS page cache was not dropped and Tier 3 reads may be served from RAM."
    else:
        cache_note = f"Per-file page cache eviction is not available on {sys.platform}, so Tier 3 reads may be served from the OS standby list (RAM)."
    rep.p(
        "Each run stages the model with `POST /v1/memory/tier`, then times one streamed chat request "
        "(8 output tokens). `tier switch ms` is the server's `hypnos_telemetry.tier_switch_latency_ms` "
        "(promotion inside `acquire_model`); TTFT is measured by this client and includes it. "
        + cache_note
    )
    return summary


def fmt_pct(v):
    return "n/a" if v is None else f"{v * 100:.0f}%"


def parse_swap_summary(path):
    """The scenario 7 summary of an earlier report: one entry per start tier with the median tier switch and TTFT."""
    labels = {label for _, label in SWAP_TIERS}

    def num(s):
        try:
            return float(s)
        except ValueError:
            return None

    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    found = {}
    in_7 = False
    for line in lines:
        if line.startswith("### "):
            in_7 = line.startswith("### 7.")
            continue
        if not in_7 or not line.startswith("| Tier "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 4 and cells[0] in labels:
            found[cells[0]] = {"label": cells[0], "switch_ms": num(cells[2]), "ttft_ms": num(cells[3])}
    if len(found) != len(SWAP_TIERS):
        raise SystemExit(f"{path}: no scenario 7 summary table (one row per start tier) found")
    return [found[label] for _, label in SWAP_TIERS]


def swap_chart_svg(series, title, subtitle):
    """Horizontal bars per start tier: tier switch + rest of TTFT, median runs. Light/dark aware.

    `series` is a list of (name, summary) pairs; with one series each bar is labelled with its start
    tier, with several the tiers become groups holding one bar per series, labelled with its name.
    """
    width, left, right, top, bar_h, gap = 720, 210, 90, 78, 30, 18
    grouped = len(series) > 1
    if grouped:
        bar_h, gap, head = 22, 6, 24
        # Room for the longest series name (about 6.4 px per character at 12 px).
        left = max(left, 24 + int(6.4 * max(len(name) for name, _ in series)))
    plot_w = width - left - right
    # Lay out the rows: (y, label, summary entry), plus a header line per tier when grouped.
    rows, headers, y = [], [], top
    for _, tier_label in SWAP_TIERS:
        if grouped:
            headers.append((y, tier_label))
            y += head
        for name, summary in series:
            entry = next((s for s in summary if s["label"] == tier_label), None)
            if entry is None:
                continue
            rows.append((y, name if grouped else tier_label, entry))
            y += bar_h + gap
        if grouped:
            y += 10
    axis_y = y - (10 if grouped else 0)
    height = axis_y + 40
    vmax = max((s["ttft_ms"] or 0) for _, summary in series for s in summary) or 1.0
    # Round the axis up to a 1/2/5 step.
    step = 10 ** max(0, len(str(int(vmax))) - 1)
    for m in (1, 2, 5, 10):
        if vmax <= step * m * 5:
            step *= m
            break
    axis_max = step * (int(vmax // step) + 1)

    def x(v):
        return left + plot_w * v / axis_max

    def esc(t):
        return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif" role="img" aria-label="{esc(title)}">',
        "<style>",
        ".bg{fill:#fcfcfb}.ink{fill:#0b0b0b}.ink2{fill:#52514e}.muted{fill:#898781}.grid{stroke:#e1e0d9}"
        ".s1{fill:#2a78d6}.s2{fill:#eb6834}.gap{stroke:#fcfcfb}",
        "@media (prefers-color-scheme: dark){.bg{fill:#1a1a19}.ink{fill:#f0efec}.ink2{fill:#c3c2b7}"
        ".grid{stroke:#2c2c2a}.s1{fill:#3987e5}.s2{fill:#d95926}.gap{stroke:#1a1a19}}",
        "</style>",
        f'<rect class="bg" width="{width}" height="{height}" rx="8"/>',
        f'<text class="ink" x="20" y="30" font-size="16" font-weight="600">{esc(title)}</text>',
        f'<text class="ink2" x="20" y="50" font-size="12">{esc(subtitle)}</text>',
    ]
    # Legend
    lx = left
    for cls, name in (("s2", "tier switch (load)"), ("s1", "rest of time to first token")):
        out.append(f'<rect class="{cls}" x="{lx}" y="58" width="10" height="10" rx="2"/>')
        out.append(f'<text class="ink2" x="{lx + 15}" y="67" font-size="11">{name}</text>')
        lx += 190
    # Grid + axis labels
    v = 0
    while v <= axis_max:
        gx = x(v)
        out.append(f'<line class="grid" x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" y2="{axis_y}" stroke-width="1"/>')
        label = f"{v / 1000:g} s" if axis_max >= 2000 else f"{v:g} ms"
        out.append(f'<text class="muted" x="{gx:.1f}" y="{axis_y + 16}" font-size="11" text-anchor="middle">{label}</text>')
        v += step
    for hy, tier_label in headers:
        out.append(f'<text class="ink" x="20" y="{hy + 14}" font-size="13" font-weight="600">{esc(tier_label)}</text>')
    for y, label, s in rows:
        ttft = s["ttft_ms"] or 0.0
        switch = min(s["switch_ms"] or 0.0, ttft)
        cls, size = ("ink2", 12) if grouped else ("ink", 13)
        out.append(f'<text class="{cls}" x="{left - 12}" y="{y + bar_h / 2 + 4}" font-size="{size}" text-anchor="end">{esc(label)}</text>')
        if switch > 0:
            out.append(f'<rect class="s2" x="{left}" y="{y}" width="{max(x(switch) - left, 1):.1f}" height="{bar_h}" rx="4"/>')
        if ttft > switch:
            out.append(
                f'<rect class="s1 gap" x="{x(switch):.1f}" y="{y}" width="{max(x(ttft) - x(switch), 1):.1f}" '
                f'height="{bar_h}" rx="4" stroke-width="{2 if switch > 0 else 0}"/>'
            )
        value = f"{ttft / 1000:.2f} s" if ttft >= 1000 else f"{ttft:.0f} ms"
        out.append(f'<text class="ink" x="{x(ttft) + 8:.1f}" y="{y + bar_h / 2 + 4}" font-size="12" font-weight="600">{value}</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------


def swap(c, rep, args):
    model = args.swap_model or args.chat_model
    files = [f for f in args.swap_files.split(",") if f.strip()]
    summary = scenario_7(c, rep, model, args.swap_reps, files, args.swap_keep_page_cache)
    if args.chart:
        series = [(args.chart_label or args.device or sys.platform, summary)]
        for spec in args.chart_compare or []:
            name, _, path = spec.partition("=")
            if not name or not path:
                raise SystemExit("--chart-compare takes LABEL=path/to/report.md")
            series.append((name, parse_swap_summary(path)))
        os.makedirs(os.path.dirname(os.path.abspath(args.chart)), exist_ok=True)
        subtitle = f"{model}, median of {args.swap_reps} runs, first streamed token"
        if len(series) == 1:
            subtitle += f", {series[0][0]}"
        with open(args.chart, "w", encoding="utf-8", newline="\n") as f:
            f.write(swap_chart_svg(series, "Time to first token by start tier", subtitle))
        print(f"chart written to {args.chart}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:9999")
    ap.add_argument("--out", help="write the Markdown report here (directories are created)")
    ap.add_argument("--scenarios", default="1,2,3,4,5,6")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--chat-model", default="gemma4:26b-q4")
    ap.add_argument("--stt-model", default="gemma4:e4b-q4")
    ap.add_argument("--tts-model", default="kokoro-82m-q4")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--label", default="", help="free text recorded in the report header (e.g. git sha, 'after B')")
    ap.add_argument("--swap-model", default=None, help="scenario 7 model (default: --chat-model)")
    ap.add_argument("--swap-reps", type=int, default=3, help="scenario 7 runs per start tier")
    ap.add_argument(
        "--swap-files",
        default=",".join(os.path.join(ROOT, "backend", "models", f) for f in DEFAULT_SWAP_FILES),
        help="comma-separated weight files of the scenario 7 model, evicted from the page cache before Tier 3 runs",
    )
    ap.add_argument("--swap-keep-page-cache", action="store_true", help="scenario 7: do not evict the weight files for Tier 3")
    ap.add_argument("--chart", help="scenario 7: write an SVG chart of the median runs here")
    ap.add_argument("--chart-label", help="scenario 7 chart: name of this run's bars (default: the server's device name)")
    ap.add_argument(
        "--chart-compare",
        action="append",
        metavar="LABEL=REPORT.md",
        help="scenario 7 chart: also draw the scenario 7 medians of an earlier report, one bar per report under each start tier (repeatable)",
    )
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
    args.device = (mem.get("capabilities") or {}).get("device_name")

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
        (7, lambda: swap(c, rep, args)),
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
