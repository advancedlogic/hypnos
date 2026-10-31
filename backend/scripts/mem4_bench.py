import json, time, random, sys, urllib.request
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9999"
MODEL = "gemma4:26b-q4"
NUM_CTX = 20480
random.seed(7)
WORDS = ("memory tier cache model token layer prefill decode latency bandwidth server request context "
         "window slot engine weight vision audio image document stream scheduler lease placement budget "
         "eviction promotion transfer kernel batch sequence attention expert router gradient sample").split()

def document(target_tokens):
    out, n, i = [], 0, 0
    while n < target_tokens:
        i += 1
        sent = " ".join(random.choice(WORDS) for _ in range(random.randint(10, 22)))
        out.append(f"Section {i}. The {sent.capitalize()}.")
        n += len(out[-1]) / 4.2  # rough tokens per char
    return "\n".join(out)

def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=900)

def turn(messages):
    t0 = time.perf_counter()
    resp = post("/v1/chat/completions", {"model": MODEL, "messages": messages, "stream": True,
                                         "max_tokens": 40, "temperature": 0, "num_ctx": NUM_CTX}, True)
    ttft, text, last = None, "", None
    for line in resp:
        line = line.decode().strip()
        if not line.startswith("data: ") or line[6:] == "[DONE]":
            continue
        ev = json.loads(line[6:])
        delta = ev["choices"][0]["delta"].get("content") if ev.get("choices") else None
        if delta:
            ttft = ttft or (time.perf_counter() - t0) * 1000
            text += delta
        if ev.get("hypnos_telemetry"):
            last = ev
    tel = last["hypnos_telemetry"]; eng = tel["engine"]; usage = last["usage"]
    return {"ttft_client_ms": round(ttft or 0), "prompt_tokens": usage["prompt_tokens"],
            "cached": eng.get("cached_tokens", 0), "prefill_ms": round(eng.get("prefill_ms", 0)),
            "switch_ms": round(tel.get("tier_switch_latency_ms", 0)), "prev_tier": tel["previous_tier"], "text": text}

rows = []
for target in (2000, 8000, 16000):
    doc = document(target)
    msgs = [{"role": "system", "content": "Answer in one short sentence."},
            {"role": "user", "content": f"Here is a document:\n{doc}\n\nQuestion 1: what is section 3 about?"}]
    t1 = turn(msgs)
    msgs += [{"role": "assistant", "content": t1["text"]}, {"role": "user", "content": "Question 2: what is section 5 about?"}]
    t2 = turn(msgs)
    post("/v1/memory/tier", {"model": MODEL, "target_tier": "tier2_pinned_ram"}).read()
    msgs += [{"role": "assistant", "content": t2["text"]}, {"role": "user", "content": "Question 3: what is section 7 about?"}]
    t3 = turn(msgs)
    rows.append((target, t1, t2, t3))
    print(f"~{target}: first {t1['ttft_client_ms']} ms | warm {t2['ttft_client_ms']} ms (cached {t2['cached']}) | "
          f"after swap {t3['ttft_client_ms']} ms (switch {t3['switch_ms']}, prefill {t3['prefill_ms']}, cached {t3['cached']}, prompt {t3['prompt_tokens']})", flush=True)

post("/v1/memory/tier", {"model": MODEL, "target_tier": "tier1_gpu_vram", "context_len": 16384}).read()
print(json.dumps([{"target": r[0], "first": {k: v for k, v in r[1].items() if k != "text"},
                   "warm": {k: v for k, v in r[2].items() if k != "text"},
                   "after_swap": {k: v for k, v in r[3].items() if k != "text"}} for r in rows]))
