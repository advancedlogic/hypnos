// Converts a document to Markdown with docling, page by page, and saves its pictures.
//
//   node examples/documents.mjs report.pdf [pages]

import { writeFile } from "node:fs/promises";

import { Client } from "../dist/index.js";

const [file, pages] = process.argv.slice(2);
const hypnos = new Client();

const stream = await hypnos.convertDocumentStream("docling", { file, pages, return_pictures: true });
for await (const event of stream) {
  if (event.object === "document.started") {
    console.log(`${event.page_count} pages, converting ${JSON.stringify(event.pages)}`);
  } else if (event.object === "document.page") {
    console.log(`page ${event.page}: ${event.tables} tables, ${event.pictures} pictures`);
  }
}
const doc = await stream.final();
await writeFile(`${file}.md`, doc.markdown);
console.log(`wrote ${file}.md`);

if (doc.id) {
  for (const picture of (await hypnos.documentPictures(doc.id)).data) {
    const { data, contentType } = await hypnos.documentPictureBytes(doc.id, picture.index);
    const name = `${file}.picture${picture.index}.${contentType.split("/").pop()}`;
    await writeFile(name, data);
    console.log(`wrote ${name} (${picture.caption ?? "no caption"})`);
  }
  await hypnos.deleteDocument(doc.id);
}
