// Embeds a few sentences with gemmaembedding2 and ranks them against a query.
//
//   node examples/embeddings.mjs "how do I reset my password"

import { Client } from "../dist/index.js";

const docs = [
  "To change your password, open Settings and choose Security.",
  "Our office is closed on public holidays.",
  "Invoices are sent on the first day of each month.",
];
const query = process.argv.slice(2).join(" ") || "how do I reset my password";

const { data } = await new Client().embeddings("gemmaembedding2", [query, ...docs]);
const [q, ...rest] = data.map((d) => d.embedding);
// The vectors are L2-normalised: the dot product is the cosine similarity.
const scored = rest.map((v, i) => [v.reduce((s, x, j) => s + x * q[j], 0), docs[i]]);
for (const [score, doc] of scored.sort((a, b) => b[0] - a[0])) console.log(`${score.toFixed(3)}  ${doc}`);
