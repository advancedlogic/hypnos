#!/usr/bin/env bash
# End-to-end smoke test for a running Hypnos server: one real request per modality.
#
#   ./backend/scripts/smoke_test.sh [http://localhost:8080]
#
# Outputs (audio, images, raw JSON) are written to .run/smoke/ at the repository root.
set -uo pipefail

HOST="${1:-http://localhost:8080}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT_DIR/.run/smoke"
FIX="$ROOT_DIR/backend/tests/fixtures"
mkdir -p "$OUT"

PY=python
command -v python >/dev/null 2>&1 || PY=python3

pass=0; fail=0
ok()   { echo "  [PASS] $*"; pass=$((pass+1)); }
bad()  { echo "  [FAIL] $*"; fail=$((fail+1)); }
json() { "$PY" -I -X utf8 -c "$1" "${@:2}"; }

echo "== Hypnos smoke test against $HOST"

# 0. health ---------------------------------------------------------------------------------
if curl -sf -m 5 "$HOST/health" | grep -q '"ok"'; then ok "health"; else bad "health (is the backend running? ./run.sh status)"; exit 1; fi

# 1. models ---------------------------------------------------------------------------------
curl -s -m 10 "$HOST/v1/models" > "$OUT/models.json"
n=$(json 'import sys,json; print(len(json.load(open(sys.argv[1]))["data"]))' "$OUT/models.json" 2>/dev/null || echo 0)
[ "$n" -ge 5 ] && ok "models listed: $n" || bad "models listed: $n"

# 2. chat (text) ----------------------------------------------------------------------------
echo "-- chat/completions text (gemma4:26b-q4) ..."
t0=$(date +%s)
curl -s -m 900 "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:26b-q4","messages":[{"role":"user","content":"Reply with exactly: Hypnos online."}],"max_tokens":32,"temperature":0}' > "$OUT/chat_text.json"
txt=$(json 'import sys,json; d=json.load(open(sys.argv[1])); print(d.get("choices",[{}])[0].get("message",{}).get("content") or d.get("error",{}).get("message",""))' "$OUT/chat_text.json" 2>/dev/null)
echo "     -> $(echo "$txt" | head -c 200)  ($(( $(date +%s) - t0 ))s)"
echo "$txt" | grep -qi 'hypnos' && ok "chat text" || bad "chat text"

# 3. chat (vision) --------------------------------------------------------------------------
echo "-- chat/completions vision (gemma4:26b-q4, 1x1 red PNG) ..."
PNG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
curl -s -m 900 "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"gemma4:26b-q4\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What colour is this image? One word.\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$PNG\"}}]}],\"max_tokens\":16,\"temperature\":0}" > "$OUT/chat_vision.json"
txt=$(json 'import sys,json; d=json.load(open(sys.argv[1])); print(d.get("choices",[{}])[0].get("message",{}).get("content") or d.get("error",{}).get("message",""))' "$OUT/chat_vision.json" 2>/dev/null)
echo "     -> $(echo "$txt" | head -c 200)"
echo "$txt" | grep -qi 'red' && ok "chat vision" || bad "chat vision"

# 4. streaming ------------------------------------------------------------------------------
echo "-- chat/completions stream ..."
curl -s -N -m 900 "$HOST/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model":"gemma4:26b-q4","stream":true,"messages":[{"role":"user","content":"Count from one to five in words."}],"max_tokens":40,"temperature":0}' > "$OUT/chat_stream.sse"
chunks=$(grep -c '^data: {' "$OUT/chat_stream.sse")
grep -q '^data: \[DONE\]' "$OUT/chat_stream.sse" && [ "$chunks" -ge 3 ] && ok "chat stream ($chunks chunks)" || bad "chat stream ($chunks chunks)"

# 5. transcription --------------------------------------------------------------------------
echo "-- audio/transcriptions (gemma4:e4b-q4, jfk.wav) ..."
if [ -f "$FIX/jfk.wav" ]; then
  t0=$(date +%s)
  curl -s -m 900 "$HOST/v1/audio/transcriptions" -F model=gemma4:e4b-q4 -F "file=@$FIX/jfk.wav" > "$OUT/transcription.json"
  txt=$(json 'import sys,json; d=json.load(open(sys.argv[1])); print(d.get("text") or d.get("error",{}).get("message",""))' "$OUT/transcription.json" 2>/dev/null)
  echo "     -> $(echo "$txt" | head -c 200)  ($(( $(date +%s) - t0 ))s)"
  echo "$txt" | grep -qi 'country' && ok "transcription" || bad "transcription"
else
  bad "transcription (fixture $FIX/jfk.wav missing)"
fi

# 6. text-to-speech -------------------------------------------------------------------------
echo "-- audio/speech (kokoro-82m-q4) ..."
code=$(curl -s -m 300 -o "$OUT/speech.wav" -w '%{http_code}' "$HOST/v1/audio/speech" -H 'Content-Type: application/json' \
  -d '{"model":"kokoro-82m-q4","input":"Hypnos speaks with Kokoro, entirely in process.","voice":"af_heart"}')
size=$(wc -c < "$OUT/speech.wav" 2>/dev/null | tr -d " " || echo 0)  # wc: portable (BSD stat has no -c)
head -c 4 "$OUT/speech.wav" | grep -q RIFF && [ "$code" = 200 ] && [ "$size" -gt 20000 ] && ok "tts ($size bytes wav)" || bad "tts (http $code, $size bytes)"

# 7. embeddings -----------------------------------------------------------------------------
echo "-- embeddings (gemmaembedding2) ..."
curl -s -m 300 "$HOST/v1/embeddings" -H 'Content-Type: application/json' \
  -d '{"model":"gemmaembedding2","input":["a cat on a mat","a kitten on a rug","quarterly tax filing deadline"]}' > "$OUT/embeddings.json"
res=$(json '
import sys,json,math
d=json.load(open(sys.argv[1]))
if "data" not in d: print("ERR", d.get("error",{}).get("message")); sys.exit()
v=[x["embedding"] for x in d["data"]]
dot=lambda a,b: sum(i*j for i,j in zip(a,b))
print(f"dims={len(v[0])} sim(cat,kitten)={dot(v[0],v[1]):.3f} sim(cat,tax)={dot(v[0],v[2]):.3f}")
' "$OUT/embeddings.json" 2>/dev/null)
echo "     -> $res"
echo "$res" | grep -q 'dims=768' && ok "embeddings" || bad "embeddings"

# 8. image generation -----------------------------------------------------------------------
echo "-- images/generations (moss-image, 512x512, 8 steps) ..."
t0=$(date +%s)
curl -s -m 1800 "$HOST/v1/images/generations" -H 'Content-Type: application/json' \
  -d '{"model":"moss-image","prompt":"a small red apple on a white table, studio photo","size":"512x512","steps":8}' > "$OUT/image.json"
res=$(json '
import sys,json,base64
d=json.load(open(sys.argv[1]))
if "data" not in d: print("ERR", d.get("error",{}).get("message")); sys.exit()
b=base64.b64decode(d["data"][0]["b64_json"]); open(sys.argv[2],"wb").write(b)
print(f"png bytes={len(b)} magic_ok={b[:8]==bytes([137,80,78,71,13,10,26,10])}")
' "$OUT/image.json" "$OUT/image.png" 2>/dev/null)
echo "     -> $res  ($(( $(date +%s) - t0 ))s)"
echo "$res" | grep -q 'magic_ok=True' && ok "image generation" || bad "image generation"

# 9. document conversion -----------------------------------------------------------------------
echo "-- documents/convert (docling, a generated one-page PDF) ..."
json '
import sys
text = "Hypnos converts this sample page."
objs = ["<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
        None,
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
stream = f"BT /F1 18 Tf 72 700 Td ({text}) Tj ET"
objs[3] = f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream"
out, offsets = "%PDF-1.4\n", []
for i, o in enumerate(objs, 1):
    offsets.append(len(out)); out += f"{i} 0 obj\n{o}\nendobj\n"
xref = len(out)
out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
open(sys.argv[1], "w", encoding="latin-1").write(out)
' "$OUT/sample.pdf"
t0=$(date +%s)
curl -s -m 600 "$HOST/v1/documents/convert" -F model=docling -F "file=@$OUT/sample.pdf;type=application/pdf" > "$OUT/document.json"
res=$(json '
import sys,json
d=json.load(open(sys.argv[1]))
if "markdown" not in d: print("ERR", d.get("error",{}).get("message")); sys.exit()
pages, found = d["usage"]["pages"], "sample page" in d["markdown"]
print(f"pages={pages} found={found}")
' "$OUT/document.json" 2>/dev/null)
echo "     -> $res  ($(( $(date +%s) - t0 ))s)"
echo "$res" | grep -q 'found=True' && ok "document conversion" || bad "document conversion"

# 10. entity extraction ---------------------------------------------------------------------
echo "-- entities (gliner2.5-multi) ..."
curl -s -m 600 "$HOST/v1/entities" -H 'Content-Type: application/json' \
  -d '{"model":"gliner2.5-multi","input":"Mario Rossi works in Milan for Ferrari.","labels":["person","city","company"]}' > "$OUT/entities.json"
res=$(json '
import sys,json
d=json.load(open(sys.argv[1]))
if "data" not in d: print("ERR", d.get("error",{}).get("message")); sys.exit()
found = sorted((e["label"], e["text"]) for e in d["data"][0]["entities"])
print(" ".join(f"{label}={text}" for label, text in found))
' "$OUT/entities.json" 2>/dev/null)
echo "     -> $res"
echo "$res" | grep -q 'person=Mario Rossi' && ok "entity extraction" || bad "entity extraction"

# 11. memory status --------------------------------------------------------------------------
curl -s -m 10 "$HOST/v1/memory/status" > "$OUT/memory_status.json"
json 'import sys,json; d=json.load(open(sys.argv[1])); print("   VRAM used %.1f GiB / free %.1f GiB; models: %s" % (d["tier1_vram_used_bytes"]/2**30, d["tier1_vram_free_bytes"]/2**30, ", ".join(f"{m[\"id\"]}={m[\"current_tier\"]}" for m in d["models"])))' "$OUT/memory_status.json" 2>/dev/null

echo "== done: $pass passed, $fail failed  (artifacts in $OUT)"
[ "$fail" = 0 ]
