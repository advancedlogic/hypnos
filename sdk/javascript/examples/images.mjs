// Generates an image with moss-image and saves it as PNG.
//
//   node examples/images.mjs "a lighthouse at dawn, oil painting" lighthouse.png

import { writeFile } from "node:fs/promises";

import { Client } from "../dist/index.js";

const prompt = process.argv[2] ?? "a lighthouse at dawn, oil painting";
const out = process.argv[3] ?? "image.png";

const result = await new Client().images("moss-image", prompt, { size: "1024x1024", steps: 20, seed: 7 });
await writeFile(out, Buffer.from(result.data[0].b64_json, "base64"));
console.log(`wrote ${out} in ${((result.hypnos_telemetry?.inference_latency_ms ?? 0) / 1000).toFixed(1)} s`);
