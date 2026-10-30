// Synthesizes speech with kokoro-82m-q4: the whole file, then the same text streamed.
//
//   node examples/speech.mjs "Hello from Hypnos." out.wav

import { writeFile } from "node:fs/promises";

import { Client } from "../dist/index.js";

const text = process.argv[2] ?? "Hello from Hypnos.";
const out = process.argv[3] ?? "speech.wav";
const hypnos = new Client();

const speech = await hypnos.speech("kokoro-82m-q4", text, { voice: "nova" });
await writeFile(out, speech.audio);
console.log(`wrote ${out} (${speech.contentType}, real-time factor ${speech.telemetry.rtf})`);

// Streaming: raw 16-bit PCM at 24 kHz, playable as it arrives.
const stream = await hypnos.speechStream("kokoro-82m-q4", text, { response_format: "pcm" });
let received = 0;
for await (const chunk of stream) received += chunk.length;
console.log(`streamed ${(received / 48000).toFixed(2)} s of audio`);
