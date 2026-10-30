#!/usr/bin/env python3
"""Soak test: concurrent clients across every modality for a long time, against a running server.

    ./backend/scripts/soak.sh [--host URL] [--duration 60] [--clients 6] [--out docs/bench/soak-DATE.md]

Workers pick a request kind by weight (chat, streaming chat, speech, embeddings, entities,
document conversion with the pictures kept, and an image generation every --image-every seconds
to force evictions). A sampler reads /v1/memory/status and /metrics every --sample-every seconds.
The report says whether the process footprint and the VRAM bookings stayed flat after the
warm-up, how many requests failed and how, the latency per kind, and whether anything stalled.
Stdlib only, same Client as bench.py.
"""

import argparse
import datetime as dt
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402  (bench.py beside this file)

TIMEOUT = 600.0

PROMPTS = [
    "In two sentences, what does a lease-based GPU scheduler do for a multi-modal server?",
    "List three reasons to pin an embedding model in VRAM. Be brief.",
    "Explain KV cache prefix reuse to a new engineer in three sentences.",
    "Why would a scheduler evict an image model before a chat model? Two sentences.",
    "Give one sentence on what pinned host RAM is for.",
]
TTS_TEXTS = [
    "Hypnos keeps Kokoro pinned in GPU memory for instant speech.",
    "The scheduler moved the chat model back into VRAM in four seconds.",
    "Document conversion runs on the CPU while the GPU serves chat.",
]
NER_TEXTS = [
    "Tim Cook announced the iPhone 17 at the Steve Jobs Theater in Cupertino on September 9.",
    "Mario Rossi, who works in Milan for Ferrari, watched the stream with a cappuccino.",
    "Dr. Ada Lovelace met Charles Babbage in London in 1833 to discuss the Analytical Engine.",
]


def two_page_pdf():
    """Two pages, a paragraph and a small picture each (exercises the picture store)."""
    objs = []

    def add(b):
        objs.append(b)
        return len(objs)

    font = add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pages = add(b"")
    w, h = 32, 24
    img = b"".join(bytes(((x * 8) % 256, (y * 10) % 256, 90)) for y in range(h) for x in range(w))
    image = add(
        b"<< /Type /XObject /Subtype /Image /Width %d /Height %d /ColorSpace /DeviceRGB /BitsPerComponent 8 /Length %d >>\nstream\n"
        % (w, h, len(img)) + img + b"\nendstream"
    )
    ids = []
    for n in (1, 2):
        text = (
            f"BT /F1 16 Tf 72 740 Td (Soak test page {n}) Tj ET\n"
            f"BT /F1 11 Tf 72 712 Td (This page has one embedded picture and this paragraph of text.) Tj ET\n"
            f"q 160 0 0 120 72 540 cm /Im1 Do Q\n"
            f"BT /F1 10 Tf 72 520 Td (Figure {n}: a gradient) Tj ET\n"
        ).encode()
        content = add(b"<< /Length %d >>\nstream\n" % len(text) + text + b"\nendstream")
        ids.append(add(
            b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 612 792] /Contents %d 0 R "
            b"/Resources << /Font << /F1 %d 0 R >> /XObject << /Im1 %d 0 R >> >> >>" % (pages, content, font, image)
        ))
    objs[pages - 1] = b"<< /Type /Pages /Kids [" + b" ".join(b"%d 0 R" % p for p in ids) + b"] /Count 2 >>"
    catalog = add(b"<< /Type /Catalog /Pages %d 0 R >>" % pages)
    buf = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(buf))
        buf += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(buf)
    buf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offsets:
        buf += b"%010d 00000 n \n" % off
    buf += b"trailer\n<< /Size %d /Root %d 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, catalog, xref)
    return bytes(buf)


class Stats:
    """Per-kind counts and latencies, plus every error, from all workers."""

    def __init__(self):
        self.lock = threading.Lock()
        self.latency = {}
        self.ttft = {}
        self.errors = {}
        self.count = {}
        self.last_completion = time.time()
        self.tokens_out = 0
        self.docs_stored = 0
        self.docs_released = 0

    def ok(self, kind, ms, ttft=None, tokens=0):
        with self.lock:
            self.count[kind] = self.count.get(kind, 0) + 1
            self.latency.setdefault(kind, []).append(ms)
            if ttft is not None:
                self.ttft.setdefault(kind, []).append(ttft)
            self.tokens_out += tokens
            self.last_completion = time.time()

    def err(self, kind, message):
        key = message.split("\n")[0][:80]
        with self.lock:
            self.errors.setdefault(kind, {})
            self.errors[kind][key] = self.errors[kind].get(key, 0) + 1
            self.last_completion = time.time()


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def worker(idx, c, args, stats, stop, pdf, png_b64):
    rnd = random.Random(1000 + idx)
    kinds = [k for k, w in args.mix.items() for _ in range(w)]
    docs = []
    while not stop.is_set():
        kind = rnd.choice(kinds)
        try:
            if kind == "chat" or kind == "stream":
                prompt = rnd.choice(PROMPTS)
                body = {
                    "model": args.chat_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": rnd.choice([32, 48, 64]),
                    "temperature": 0.7,
                }
                if kind == "stream":
                    r = c.stream_chat(body)
                    stats.ok(kind, r["total_ms"], r["ttft_ms"], r["completion_tokens"] or 0)
                else:
                    resp, ms, _ = c.post_json("/v1/chat/completions", body)
                    stats.ok(kind, ms, tokens=(resp.get("usage") or {}).get("completion_tokens", 0))
            elif kind == "vision":
                body = {
                    "model": args.chat_model,
                    "messages": [{"role": "user", "content": [
                        {"type": "text", "text": "What shapes are in this image? One sentence."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + png_b64}},
                    ]}],
                    "max_tokens": 40,
                }
                resp, ms, _ = c.post_json("/v1/chat/completions", body)
                stats.ok(kind, ms, tokens=(resp.get("usage") or {}).get("completion_tokens", 0))
            elif kind == "tts":
                _, ttfb, total, _ = c.post_binary(
                    "/v1/audio/speech",
                    {"model": args.tts_model, "input": rnd.choice(TTS_TEXTS), "voice": args.voice, "response_format": "wav"},
                    skip=44,
                )
                stats.ok(kind, total, ttfb)
            elif kind == "embed":
                _, ms, _ = c.post_json("/v1/embeddings", {"model": args.embed_model, "input": rnd.sample(PROMPTS, 3)})
                stats.ok(kind, ms)
            elif kind == "entities":
                _, ms, _ = c.post_json(
                    "/v1/entities",
                    {"model": args.ner_model, "input": rnd.choice(NER_TEXTS), "labels": ["person", "company", "city", "date", "product"]},
                )
                stats.ok(kind, ms)
            elif kind == "documents":
                resp, ms = c.post_multipart(
                    "/v1/documents/convert",
                    {"model": args.doc_model, "return_pictures": "true"},
                    "file", "soak.pdf", pdf, "application/pdf",
                )
                stats.ok(kind, ms)
                if resp.get("id"):
                    docs.append(resp["id"])
                    with stats.lock:
                        stats.docs_stored += 1
                # Release every other document, as a client that fetched its pictures would.
                if len(docs) >= 2:
                    doc = docs.pop(0)
                    try:
                        c._open(c._req(f"/v1/documents/{doc}", method="DELETE")).read()
                        with stats.lock:
                            stats.docs_released += 1
                    except Exception as e:  # noqa: BLE001
                        stats.err("documents-delete", str(e))
            else:
                time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            stats.err(kind, str(e))
        time.sleep(rnd.uniform(0.0, 0.5))


def image_worker(c, args, stats, stop):
    """One image every --image-every seconds: the surest way to make models evict each other."""
    while not stop.is_set():
        try:
            _, ms, _ = c.post_json(
                "/v1/images/generations",
                {"model": args.image_model, "prompt": "a small red cube on a white table", "size": args.image_size, "n": 1},
            )
            stats.ok("image", ms)
        except Exception as e:  # noqa: BLE001
            stats.err("image", str(e))
        stop.wait(args.image_every)


def metric(lines, name, label=None):
    for l in lines:
        if label is None and l.startswith(name + " "):
            return float(l.split()[-1])
        if label is not None and l.startswith(name + "{") and label in l:
            return float(l.split()[-1])
    return None


def sampler(c, args, samples, stop, t0):
    while not stop.is_set():
        try:
            st = c.get_json("/v1/memory/status")
            lines = c.metrics(("hypnos_scheduler_", "hypnos_http_requests_total", "hypnos_gpu_"))
            logs = c.get_json("/v1/logs?limit=1")
            samples.append({
                "t": time.time() - t0,
                "footprint": st.get("process_footprint_bytes"),
                "vram_booked": st.get("tier1_vram_used_bytes"),
                "vram_real_free": st.get("tier1_vram_real_free_bytes"),
                "vram_unaccounted": st.get("tier1_vram_unaccounted_bytes"),
                "pinned": st.get("tier2_pinned_ram_used_bytes"),
                "gpu_temp": (st.get("gpu") or {}).get("temperature_celsius"),
                "promotions": metric(lines, "hypnos_scheduler_promotions_total"),
                "evictions": metric(lines, "hypnos_scheduler_evictions_total"),
                "load_failures": metric(lines, "hypnos_scheduler_load_failures_total"),
                "rejections": metric(lines, "hypnos_scheduler_admissions_rejected_total"),
                "http_5xx": sum(float(l.split()[-1]) for l in lines if l.startswith("hypnos_http_requests_total{") and 'class="5xx"' in l),
                "log_stored": logs.get("stored"),
                "tiers": {m["id"]: m["current_tier"] for m in st.get("models", [])},
                "evictions_recent": st.get("recent_evictions", []),
            })
        except Exception as e:  # noqa: BLE001
            samples.append({"t": time.time() - t0, "error": str(e)})
        stop.wait(args.sample_every)


def gb(v):
    return "n/a" if v is None else f"{v / 1e9:.2f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="http://127.0.0.1:8888")
    ap.add_argument("--duration", type=float, default=60.0, help="minutes")
    ap.add_argument("--warmup", type=float, default=5.0, help="minutes excluded from the growth verdict")
    ap.add_argument("--clients", type=int, default=6)
    ap.add_argument("--sample-every", type=float, default=15.0, help="seconds")
    ap.add_argument("--image-every", type=float, default=180.0, help="seconds between image generations (0 = none)")
    ap.add_argument("--image-size", default="512x512")
    ap.add_argument("--mix", default="chat:3,stream:3,vision:1,tts:2,embed:2,entities:2,documents:1",
                    help="kind:weight list")
    ap.add_argument("--chat-model", default="gemma4:26b-q4")
    ap.add_argument("--tts-model", default="kokoro-82m-q4")
    ap.add_argument("--embed-model", default="gemmaembedding2")
    ap.add_argument("--ner-model", default="gliner2.5-multi")
    ap.add_argument("--doc-model", default="docling")
    ap.add_argument("--image-model", default="moss-image")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--stall-minutes", type=float, default=5.0, help="no request completing for this long = stall")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", help="write the Markdown report here")
    args = ap.parse_args()
    args.mix = {k: int(w) for k, w in (kv.split(":") for kv in args.mix.split(","))}

    bench.TIMEOUT = TIMEOUT
    c = bench.Client(args.host)
    try:
        c.get_json("/ready")
    except Exception as e:  # noqa: BLE001
        print(f"server not ready at {args.host}: {e}", file=sys.stderr)
        return 2
    pdf = two_page_pdf()
    import base64
    png_b64 = base64.b64encode(bench.make_png()).decode()

    stats, samples, stop = Stats(), [], threading.Event()
    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i, bench.Client(args.host), args, stats, stop, pdf, png_b64), daemon=True)
               for i in range(args.clients)]
    threads.append(threading.Thread(target=sampler, args=(bench.Client(args.host), args, samples, stop, t0), daemon=True))
    if args.image_every > 0:
        threads.append(threading.Thread(target=image_worker, args=(bench.Client(args.host), args, stats, stop), daemon=True))
    for t in threads:
        t.start()

    end = t0 + args.duration * 60
    stalled = False
    try:
        while time.time() < end:
            time.sleep(5)
            n = sum(stats.count.values())
            e = sum(sum(v.values()) for v in stats.errors.values())
            s = samples[-1] if samples else {}
            print(f"[{(time.time() - t0) / 60:5.1f} min] ok {n} err {e} footprint {gb(s.get('footprint'))} GB "
                  f"vram booked {gb(s.get('vram_booked'))} unaccounted {gb(s.get('vram_unaccounted'))} "
                  f"evictions {s.get('evictions')} rejections {s.get('rejections')}", flush=True)
            if time.time() - stats.last_completion > args.stall_minutes * 60:
                print("STALL: no request completed for %.0f minutes" % args.stall_minutes, flush=True)
                stalled = True
                break
    except KeyboardInterrupt:
        print("interrupted", flush=True)
    stop.set()
    for t in threads:
        t.join(timeout=TIMEOUT + 5)

    # ---- verdicts --------------------------------------------------------------------------
    good = [s for s in samples if "error" not in s]
    after = [s for s in good if s["t"] >= args.warmup * 60]
    verdicts = []

    def growth(key, label, limit_pct, floor_bytes):
        """Growth between the first and the last quarter of the post-warm-up samples; a change
        under `floor_bytes` never fails (percentages of a near-zero value mean nothing)."""
        vals = [s[key] for s in after if s.get(key) is not None]
        if len(vals) < 4:
            verdicts.append((label, "n/a", "too few samples"))
            return
        first = statistics.median(vals[: max(1, len(vals) // 4)])
        last = statistics.median(vals[-max(1, len(vals) // 4):])
        pct_change = (last - first) / first * 100 if first else 0.0
        ok = (last - first) <= floor_bytes or pct_change <= limit_pct
        verdicts.append((label, "pass" if ok else "FAIL", f"{gb(first)} -> {gb(last)} GB ({pct_change:+.1f}% after warm-up, limit {limit_pct}% or {gb(floor_bytes)} GB)"))

    growth("footprint", "process footprint flat", 5.0, 512 * 1024 * 1024)
    growth("vram_unaccounted", "VRAM unaccounted flat", 25.0, 1024 * 1024 * 1024)
    total_ok = sum(stats.count.values())
    total_err = sum(sum(v.values()) for v in stats.errors.values())
    busy = sum(n for kind in stats.errors.values() for msg, n in kind.items() if "HTTP 429" in msg)
    capacity = sum(n for kind in stats.errors.values() for msg, n in kind.items() if "HTTP 507" in msg)
    server_err = sum(n for kind in stats.errors.values() for msg, n in kind.items() if "HTTP 5" in msg) - capacity
    verdicts.append(("no 5xx other than 507", "pass" if server_err == 0 else "FAIL", f"{server_err} server errors"))
    verdicts.append(("507 capacity refusals", "info", f"{capacity} (a model that did not fit while the others were leased: expected under load, not a fault)"))
    verdicts.append(("429 under 5% of requests", "pass" if busy <= 0.05 * max(1, total_ok + total_err) else "FAIL", f"{busy} of {total_ok + total_err}"))
    verdicts.append(("no stall", "FAIL" if stalled else "pass", f"longest gap between completions below {args.stall_minutes:.0f} min"))
    lf = [s["load_failures"] for s in good if s.get("load_failures") is not None]
    verdicts.append(("no load failures", "pass" if lf and lf[-1] == lf[0] else ("n/a" if not lf else "FAIL"), f"{lf[-1] - lf[0] if lf else 'n/a'} during the run"))
    unexplained = [e for s in good for e in s.get("evictions_recent", []) if not e.get("reason")]
    verdicts.append(("every eviction explained", "pass" if not unexplained else "FAIL", f"{len(unexplained)} without a reason"))

    # ---- report ----------------------------------------------------------------------------
    rep = bench.Report()
    rep.lines += [f"# Soak test {dt.date.today().isoformat()}", ""]
    rep.p(f"{args.label + ': ' if args.label else ''}{args.clients} clients for {(time.time() - t0) / 60:.0f} min against `{args.host}` "
          f"(mix `{','.join(f'{k}:{v}' for k, v in args.mix.items())}`, an image every {args.image_every:.0f} s, "
          f"warm-up {args.warmup:.0f} min excluded from the growth verdicts).")
    rep.h("Verdict")
    rep.table(["Check", "Result", "Detail"], [[a, b, c_] for a, b, c_ in verdicts])
    rep.h("Requests")
    rows = []
    for kind in sorted(set(list(stats.count) + list(stats.errors))):
        lat = stats.latency.get(kind, [])
        ttft = stats.ttft.get(kind, [])
        errs = sum(stats.errors.get(kind, {}).values())
        rows.append([kind, stats.count.get(kind, 0), errs, bench.fmt(pct(lat, 0.5)), bench.fmt(pct(lat, 0.95)),
                     bench.fmt(pct(ttft, 0.5)) if ttft else "", bench.fmt(pct(ttft, 0.95)) if ttft else ""])
    rep.table(["Kind", "OK", "Errors", "p50 ms", "p95 ms", "TTFT/TTFB p50", "p95"], rows)
    rep.p(f"Completion tokens generated: {stats.tokens_out}. Documents stored: {stats.docs_stored}, released by the client: {stats.docs_released}.")
    if stats.errors:
        rep.h("Errors")
        rep.table(["Kind", "Count", "Message"], [[k, n, f"`{m}`"] for k, d in stats.errors.items() for m, n in sorted(d.items(), key=lambda x: -x[1])])
    rep.h("Samples")
    step = max(1, len(good) // 24)
    rep.table(
        ["min", "footprint GB", "VRAM booked", "real free", "unaccounted", "pinned GB", "promotions", "evictions", "load fail", "rejected", "5xx", "log", "GPU C"],
        [[f"{s['t'] / 60:.1f}", gb(s.get("footprint")), gb(s.get("vram_booked")), gb(s.get("vram_real_free")), gb(s.get("vram_unaccounted")),
          gb(s.get("pinned")), bench.fmt(s.get("promotions")), bench.fmt(s.get("evictions")), bench.fmt(s.get("load_failures")),
          bench.fmt(s.get("rejections")), bench.fmt(s.get("http_5xx")), bench.fmt(s.get("log_stored")), bench.fmt(s.get("gpu_temp"))]
         for s in good[::step]],
    )
    bad_samples = [s for s in samples if "error" in s]
    if bad_samples:
        rep.p(f"{len(bad_samples)} sampler reads failed, first: `{bad_samples[0]['error']}`")
    text = "\n".join(rep.lines) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"report written to {args.out}")
    return 0 if all(v[1] != "FAIL" for v in verdicts) else 1


if __name__ == "__main__":
    sys.exit(main())
