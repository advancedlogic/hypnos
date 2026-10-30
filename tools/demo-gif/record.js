// Records the Hypnos playground (http://localhost:5173) with headless Chromium while a scripted
// sequence of requests moves models between tiers, then encodes the frames as a GIF.
//   node record.js <out.gif> <pictures.pdf>
const { chromium } = require('playwright');
const { GIFEncoder, quantize, applyPalette } = require('gifenc');
const { PNG } = require('pngjs');
const fs = require('fs');

const [out, pdf] = process.argv.slice(2);
const URL = 'http://localhost:5173';
const W = 1180, H = 790;
const SCALE = 0.8;            // screenshot scale: 944 x 632
const FRAME_MS = 1000;        // capture 1 fps of real time

async function main() {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: SCALE });
  page.setDefaultTimeout(240000);
  await page.goto(URL);
  await page.waitForSelector('text=Registered Multi-Modal Models');

  // Frame capture loop, independent of the director.
  const frames = [];
  let recording = true;
  const capture = (async () => {
    const t0 = Date.now();
    while (recording) {
      const png = await page.screenshot({ type: 'png' });
      frames.push({ t: Date.now() - t0, png });
      const next = t0 + frames.length * FRAME_MS;
      await new Promise((r) => setTimeout(r, Math.max(0, next - Date.now())));
    }
  })();

  const idle = (re) => page.waitForFunction((src) => {
    const re = new RegExp(src);
    return [...document.querySelectorAll('button[type=submit]')].some((b) => re.test(b.textContent));
  }, re, { timeout: 240000 });
  const tab = async (re) => { await page.getByRole('button', { name: re }).first().click(); await page.waitForTimeout(1200); };
  const step = async (label, fn) => { console.log(new Date().toISOString().slice(11, 19), label); await fn(); };

  await page.waitForTimeout(2500);

  await step('chat: Gemma 4 26B promoted to VRAM', async () => {
    await tab(/^1\./);
    await page.fill('#chat-max-tokens', '60');
    await page.fill('#chat-prompt', 'In two sentences: what does a lease-based GPU scheduler do for a multi-modal server?');
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: /Send to \/v1\/chat\/completions/ }).click();
    await page.waitForTimeout(1500);
    await idle('Send to /v1/chat/completions');
    await page.waitForTimeout(2500);
  });

  await step('documents: docling converts a PDF, pictures kept', async () => {
    await tab(/^6\./);
    await page.getByText('Keep the extracted pictures').click();
    await page.setInputFiles('#doc-file', pdf);
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: /Convert via \/v1\/documents\/convert/ }).click();
    await page.waitForTimeout(1500);
    await idle('Convert via /v1/documents/convert');
    await page.waitForTimeout(3500);
  });

  await step('entities: GLiNER zero-shot extraction', async () => {
    await tab(/^7\./);
    await page.waitForTimeout(600);
    await page.getByRole('button', { name: /Extract via \/v1\/entities/ }).click();
    await page.waitForTimeout(1500);
    await idle('Extract via /v1/entities');
    await page.waitForTimeout(3000);
  });

  await step('image: MOSS-Image evicts the 26B', async () => {
    await tab(/^5\./);
    const sizes = await page.$$eval('#img-size option', (os) => os.map((o) => o.value));
    if (sizes.includes('512x512')) await page.selectOption('#img-size', '512x512');
    await page.fill('#img-prompt', 'A glowing crab arranging GPU memory blocks, cyberpunk poster');
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: /Generate Image via/ }).click();
    await page.waitForTimeout(1500);
    await idle('Generate Image via');
    await page.waitForTimeout(3500);
  });

  await step('speech: Kokoro, pinned', async () => {
    await tab(/^3\./);
    await page.waitForTimeout(600);
    await page.getByRole('button', { name: /Synthesize Speech via/ }).click();
    await page.waitForTimeout(1500);
    await idle('Synthesize Speech via');
    await page.waitForTimeout(2500);
  });

  await step('chat again: the 26B swaps back in', async () => {
    await tab(/^1\./);
    await page.fill('#chat-prompt', 'One sentence: why keep an embedding model pinned in VRAM?');
    await page.waitForTimeout(800);
    await page.getByRole('button', { name: /Send to \/v1\/chat\/completions/ }).click();
    await page.waitForTimeout(1500);
    await idle('Send to /v1/chat/completions');
    await page.waitForTimeout(3000);
  });

  recording = false;
  await capture;
  await browser.close();
  console.log('frames', frames.length, 'duration', ((frames.at(-1).t) / 1000).toFixed(1), 's');

  // Encode. Frames that differ from the last kept one by under 1% of pixels (the visualizer's
  // idle counters tick) are dropped unless 4 s of real time passed, so waits play faster than
  // real time while every visible change is kept. Kept frames play at min(real gap, 1.5 s).
  const decoded = frames.map((f) => ({ t: f.t, img: PNG.sync.read(f.png) }));
  const FW = decoded[0].img.width, FH = decoded[0].img.height;
  const differs = (a, b) => {
    let n = 0; const total = FW * FH;
    for (let i = 0; i < a.length; i += 4) {
      if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 24) n++;
    }
    return n / total;
  };
  const kept = [];
  for (const f of decoded) {
    const last = kept[kept.length - 1];
    if (last && differs(last.img.data, f.img.data) < 0.01 && f.t - last.t < 4000) continue;
    kept.push(f);
  }
  const gif = GIFEncoder();
  kept.forEach((f, i) => {
    const next = kept[i + 1];
    const gap = next ? next.t - f.t : 3000;
    const delay = Math.min(gap, 1500);
    const palette = quantize(f.img.data, 192, { format: 'rgb444' });
    const index = applyPalette(f.img.data, palette, 'rgb444');
    gif.writeFrame(index, FW, FH, { palette, delay, repeat: 0 });
  });
  gif.finish();
  fs.writeFileSync(out, gif.bytes());
  const playback = kept.reduce((a, f, i) => a + Math.min((kept[i + 1] ? kept[i + 1].t - f.t : 3000), 1500), 0) / 1000;
  console.log('wrote', out, (fs.statSync(out).size / 1048576).toFixed(2), 'MB', kept.length, 'frames,', playback.toFixed(1), 's playback');
}

main().catch((e) => { console.error(e); process.exit(1); });
