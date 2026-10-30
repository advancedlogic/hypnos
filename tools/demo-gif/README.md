# Demo GIF

Re-records `assets/demo.gif` (the README's playground demo) with headless Chromium.

```bash
./run.sh start            # backend on :9999 with the default lineup, playground on :5173
cd tools/demo-gif
npm install               # Playwright + Chromium, the GIF encoder, a PNG decoder
python make_pictures_pdf.py pictures.pdf     # a two-page PDF with a picture per page
# optional: start the 26B in pinned RAM so the first chat shows a promotion
curl -X POST localhost:9999/v1/memory/tier -H 'Content-Type: application/json' \
     -d '{"model":"gemma4:26b-q4","target_tier":"tier2_pinned_ram"}'
npm run record            # about two minutes of real time; writes ../../assets/demo.gif
```

The sequence is chat (Gemma 4 26B promoted from pinned RAM), document conversion with the pictures
kept (docling on the CPU), entity extraction (GLiNER), image generation (MOSS-Image evicts the
26B), speech (Kokoro, pinned) and chat again (the 26B swaps back). Frames are captured once a
second at 944 x 632; frames that differ from the last kept one by under 1% of pixels are dropped
unless 4 s passed, and kept frames play at most 1.5 s apart, so waits play faster than real time
while every visible change stays. Edit `record.js` to change the sequence, the prompts or the sizes.
