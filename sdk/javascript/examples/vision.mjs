// Asks gemma4:26b-q4 about an image.
//
//   node examples/vision.mjs photo.jpg "What is in this picture?"

import { readFile } from "node:fs/promises";
import { extname } from "node:path";

import { Client } from "../dist/index.js";

const [file, ...words] = process.argv.slice(2);
const question = words.join(" ") || "Describe this picture.";
const mime = { ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp" }[extname(file).toLowerCase()] ?? "image/png";
const dataUrl = `data:${mime};base64,${(await readFile(file)).toString("base64")}`;

const reply = await new Client().chat(
  "gemma4:26b-q4",
  [
    {
      role: "user",
      content: [
        { type: "text", text: question },
        { type: "image_url", image_url: { url: dataUrl } },
      ],
    },
  ],
  { max_tokens: 300 },
);
console.log(reply.choices[0].message.content);
