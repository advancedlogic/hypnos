// Streams a chat reply from gemma4:26b-q4, then prints what the request cost.
//
//   node examples/chat.mjs "Explain sliding-window attention in two sentences."

import { Client } from "../dist/index.js";

const prompt = process.argv.slice(2).join(" ") || "Explain sliding-window attention in two sentences.";
const hypnos = new Client();
await hypnos.waitUntilReady();

const stream = await hypnos.chatStream(
  "gemma4:26b-q4",
  [
    { role: "system", content: "You are terse." },
    { role: "user", content: prompt },
  ],
  { max_tokens: 300 },
);
for await (const piece of stream.text()) process.stdout.write(piece);
const t = stream.telemetry ?? {};
console.log(
  `\n\n${stream.usage?.completion_tokens ?? "?"} tokens, model was in ${t.previous_tier}, ` +
    `load ${Math.round(t.tier_switch_latency_ms ?? 0)} ms, ` +
    `decode ${Math.round(t.engine?.decode_tokens_per_s ?? 0)} tok/s`,
);
