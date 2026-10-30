// Finds people, places and companies with gliner2.5-multi (zero-shot: any labels work).
//
//   node examples/entities.mjs "Mario Rossi works in Milan for Ferrari."

import { Client, sliceByCodePoints } from "../dist/index.js";

const text = process.argv.slice(2).join(" ") || "Mario Rossi works in Milan for Ferrari.";
const result = await new Client().entities("gliner2.5-multi", text, ["person", "city", "company"]);
for (const e of result.data[0].entities) {
  // start/end are code point offsets: sliceByCodePoints, not text.slice.
  console.log(`${e.label.padStart(8)}  ${JSON.stringify(sliceByCodePoints(text, e.start, e.end))}  ${e.score.toFixed(2)}`);
}
