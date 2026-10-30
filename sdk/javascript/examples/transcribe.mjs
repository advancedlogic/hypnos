// Transcribes a recording with gemma4:e4b-q4 (WAV, 16-bit PCM is the reliable format).
//
//   node examples/transcribe.mjs clip.wav [language]

import { Client } from "../dist/index.js";

const [file, language] = process.argv.slice(2);
const result = await new Client().transcribe("gemma4:e4b-q4", file, { language });
console.log(result.text);
const engine = result.hypnos_telemetry?.engine ?? {};
if (engine.rtf !== undefined) {
  console.log(`\n${(engine.audio_secs ?? 0).toFixed(1)} s of audio, real-time factor ${engine.rtf.toFixed(2)}`);
}
