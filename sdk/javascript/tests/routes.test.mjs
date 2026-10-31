// Every route against the mock server: request shapes, response shapes, streaming.

import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { after, before, beforeEach, test } from "node:test";

import {
  APIError, BadRequestError, NotFoundError, StreamError, assistantMessage, parseArguments, sliceByCodePoints, toolMessage,
} from "../dist/index.js";
import {
  CHAT, DOCS, EMBED, IMAGE, MockServer, NER, STT, TTS, USER, b64decode, enc, startsWith,
} from "./helpers.mjs";

const PNG = [0x89, 0x50, 0x4e, 0x47];
let server;
let client;

before(async () => {
  server = await MockServer.start();
  client = server.client();
});
after(() => server.stop());
beforeEach(() => server.reset());

test("models catalog", async () => {
  const catalog = await client.modelsCatalog();
  assert.equal(catalog.object, "list");
  assert.ok(catalog.data.length >= 10);
  const entries = new Map(catalog.data.map((e) => [e.id, e]));
  assert.equal(entries.get(CHAT).configured, true);
  const lfm3 = entries.get("lfm3:1.7b");
  assert.equal(lfm3.configured, false);
  assert.ok(!("enabled" in lfm3));
  const q8 = lfm3.variants.find((v) => v.quant === "Q8_0");
  assert.equal(q8.default, true);
  assert.ok(lfm3.variants.some((v) => v.quant === "Q4_K_M"));
  assert.ok(q8.config_block.includes('weights_path = "LFM3-1.7B-Q8_0.gguf"'));
  assert.ok(catalog.data.every((e) => e.variants.every((v) => !v.installed)));
  const vision = new Set((await client.modelsCatalog({ modality: "vision" })).data.map((e) => e.id));
  for (const id of ["gemma4:26b-q4", "gemma4:e4b-q4", "lfm3-vl:2b"]) assert.ok(vision.has(id), id);
  assert.ok(!vision.has("lfm3:1.7b"));
});

test("advise", async () => {
  const GIB = 2 ** 30;
  const advice = await client.advise({ needs: ["chat", "voice"], vramBytes: 24 * GIB, ramBytes: 64 * GIB });
  assert.deepEqual(advice.machine, { vram_bytes: 24 * GIB, ram_bytes: 64 * GIB, unified_memory: false });
  assert.equal(advice.interactive_coresident, true);
  assert.ok(advice.specs.length > 0);
  assert.ok(advice.config.includes("[[models]]"));
  assert.ok(advice.models.some((m) => m.id === TTS));
  await assert.rejects(client.advise({ needs: ["images"], vramBytes: 2 * GIB, ramBytes: 8 * GIB }), BadRequestError);
  await assert.rejects(client.advise({ needs: [] }), BadRequestError);
});

test("health, ready, models, metrics", async () => {
  assert.equal((await client.health()).status, "ok");
  assert.ok(await client.isReady());
  await client.waitUntilReady({ timeoutMs: 1000 });
  const ids = new Set((await client.models()).data.map((m) => m.id));
  for (const id of [CHAT, STT, EMBED, TTS, IMAGE, DOCS, NER]) assert.ok(ids.has(id), id);
  assert.match(await client.metrics(), /hypnos_http_requests_total/);
});

test("chat", async () => {
  const reply = await client.chat(CHAT, USER, { max_tokens: 20, temperature: 0.2, seed: 1, num_ctx: 4096 });
  assert.equal(reply.choices[0].message.content, "echo: hello there");
  assert.equal(reply.choices[0].finish_reason, "stop");
  assert.equal(reply.usage.completion_tokens, 3);
  assert.equal(reply.hypnos_telemetry.placement.primary_tier, "tier1_gpu_vram");
  assert.ok(["tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd"].includes(reply.hypnos_telemetry.previous_tier));
});

test("chat with media parts", async () => {
  const png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
  const reply = await client.chat(CHAT, [
    {
      role: "user",
      content: [
        { type: "text", text: "what is this" },
        { type: "image_url", image_url: { url: `data:image/png;base64,${png}` } },
      ],
    },
  ]);
  assert.ok(reply.choices[0].message.content.startsWith("echo: what is this"));
});

test("chat stream", async () => {
  const stream = await client.chatStream(CHAT, [{ role: "user", content: "count" }], { max_tokens: 5 });
  const pieces = [];
  for await (const piece of stream.text()) pieces.push(piece);
  assert.equal(pieces.join(""), "1 2 3 4 5");
  assert.equal(stream.finishReason, "length");
  assert.equal(stream.usage.completion_tokens, 5);
  assert.ok(stream.telemetry);
  assert.match(stream.requestId, /^req-/);
  // Iterating a finished stream yields nothing; final() returns what was collected.
  assert.equal((await stream.final()).choices[0].message.content, "1 2 3 4 5");
});

test("chat reasoning", async () => {
  const hi = [{ role: "user", content: "hi" }];
  const reply = await client.chat(CHAT, hi, { reasoning_effort: "medium" });
  assert.equal(reply.choices[0].message.reasoning_content, "Mock thoughts (medium).");
  const stream = await client.chatStream(CHAT, hi, { reasoning_effort: "medium" });
  const completion = await stream.final();
  assert.equal(stream.reasoningSoFar, "Mock thoughts (medium).");
  assert.equal(completion.choices[0].message.reasoning_content, "Mock thoughts (medium).");
  assert.equal(completion.choices[0].message.content, "echo: hi");
  const off = await client.chat(CHAT, hi, { reasoning_effort: "none" });
  assert.equal(off.choices[0].message.reasoning_content, undefined);
});

test("chat stream final", async () => {
  const stream = await client.chatStream(CHAT, USER);
  const completion = await stream.final();
  assert.equal(completion.choices[0].message.content, "echo: hello there");
  assert.equal(completion.object, "chat.completion");
  assert.ok(completion.hypnos_telemetry);
});

test("closing a stream cancels it", async () => {
  const started = Date.now();
  const stream = await client.chatStream(CHAT, [{ role: "user", content: "count" }], { max_tokens: 2000 });
  let i = 0;
  for await (const _ of stream.text()) {
    if (i++ === 3) break;
  }
  // 2000 tokens at 10 ms each would take 20 s.
  assert.ok(Date.now() - started < 5000);
  assert.ok((await client.chat(CHAT, USER)).choices[0].message.content);
});

test("an abort signal cancels a stream", async () => {
  const controller = new AbortController();
  const stream = await client.chatStream(CHAT, [{ role: "user", content: "count" }], {
    max_tokens: 2000,
    signal: controller.signal,
  });
  const started = Date.now();
  let n = 0;
  await assert.rejects(async () => {
    for await (const _ of stream) {
      if (++n === 3) controller.abort();
    }
  });
  assert.ok(Date.now() - started < 5000);
});

test("a stream that fails throws", async () => {
  await server.fail(CHAT, "stream_error");
  const stream = await client.chatStream(CHAT, [{ role: "user", content: "count" }], { max_tokens: 10 });
  await assert.rejects(
    async () => {
      for await (const _ of stream) {
        // read until it fails
      }
    },
    (err) => {
      assert.ok(err instanceof StreamError);
      assert.equal(err.status, 200);
      assert.match(err.message, /stream_error/);
      return true;
    },
  );
  assert.equal(stream.textSoFar, "1 2");
});

test("embeddings", async () => {
  const one = await client.embeddings(EMBED, "a");
  assert.equal(one.data.length, 1);
  const many = await client.embeddings(EMBED, ["a", "b", "c"]);
  assert.deepEqual(many.data.map((d) => d.index), [0, 1, 2]);
  assert.notDeepEqual(many.data[0].embedding, many.data[1].embedding);
  assert.equal(many.data[0].embedding.length, 8);
});

test("transcribe", async () => {
  const clip = enc("RIFF\0\0\0\0WAVEdata");
  const reply = await client.transcribe(STT, clip, { filename: "clip.wav", language: "it", prompt: "names" });
  assert.equal(reply.text, `mock transcript of ${clip.length} bytes`);
  assert.equal(reply.language, "it");
  const dir = mkdtempSync(path.join(tmpdir(), "hypnos-js-"));
  const file = path.join(dir, "clip.wav");
  writeFileSync(file, clip);
  assert.equal((await client.transcribe(STT, file)).text, reply.text);
  assert.equal((await client.transcribe(STT, new Blob([clip]))).text, reply.text);
});

test("speech", async () => {
  const wav = await client.speech(TTS, "one two three", { voice: "alloy" });
  assert.equal(wav.contentType, "audio/wav");
  assert.equal(new TextDecoder().decode(wav.audio.slice(0, 4)), "RIFF");
  assert.ok(["tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd"].includes(wav.telemetry.previousTier));
  assert.notEqual(wav.telemetry.gpuLayers, undefined);
  const pcm = await client.speech(TTS, "one two three", { response_format: "pcm" });
  assert.equal(pcm.contentType, "audio/pcm");
  // 50 ms per word at 24 kHz, 16-bit: 3 * 1200 samples * 2 bytes.
  assert.equal(pcm.audio.length, 3 * 1200 * 2);
});

test("speech stream", async () => {
  const stream = await client.speechStream(TTS, "one two three four", { response_format: "pcm" });
  let total = 0;
  for await (const chunk of stream) total += chunk.length;
  assert.equal(total, 4 * 1200 * 2);
  assert.equal(stream.contentType, "audio/pcm");
  assert.ok(stream.telemetry.previousTier);
});

test("images", async () => {
  const reply = await client.images(IMAGE, "a lighthouse", { n: 2, size: "512x512", steps: 4, seed: 7 });
  assert.equal(reply.data.length, 2);
  assert.ok(startsWith(b64decode(reply.data[0].b64_json), [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]));
  assert.ok(reply.hypnos_telemetry);
});

test("entities", async () => {
  const text = "Mario Rossi works in Milan.";
  const reply = await client.entities(NER, [text, "nothing here"], ["person", "city"], { threshold: 0.4 });
  const found = reply.data[0].entities;
  assert.deepEqual(found.map((e) => e.text), ["Mario", "Rossi", "Milan"]);
  for (const e of found) assert.equal(sliceByCodePoints(text, e.start, e.end), e.text);
  assert.deepEqual(reply.data[1].entities, []);
  assert.equal(reply.usage.texts, 2);
});

test("entity offsets are code points", async () => {
  const text = "🙂 Mario lives in Milan";
  const reply = await client.entities(NER, text, ["person", "city"]);
  const mario = reply.data[0].entities.find((e) => e.text === "Mario");
  assert.equal(sliceByCodePoints(text, mario.start, mario.end), "Mario");
  // The UTF-16 slice is off by one after the emoji.
  assert.notEqual(text.slice(mario.start, mario.end), "Mario");
});

test("documents and pictures", async () => {
  const doc = await client.convertDocument(DOCS, {
    file: enc("%PDF-1.4 fake"),
    filename: "report.pdf",
    pages: "2-3",
    return_pictures: true,
  });
  assert.equal(doc.page_count, 3);
  assert.deepEqual(doc.pages.map((p) => p.page), [2, 3]);
  assert.match(doc.markdown, /## Page 2/);
  const id = doc.id;
  assert.deepEqual(doc.pages[0].picture_urls, [`/v1/documents/${id}/pictures/1`]);

  const listed = await client.documentPictures(id);
  assert.deepEqual(listed.data.map((p) => p.index), [1, 2]);
  assert.equal(listed.data[0].image, undefined);
  const picture = await client.documentPicture(id, 1);
  assert.equal(picture.caption, "Figure 2");
  assert.ok(startsWith(b64decode(picture.image), PNG));
  const raw = await client.documentPictureBytes(id, 2);
  assert.equal(raw.contentType, "image/png");
  assert.ok(startsWith(raw.data, PNG));

  assert.equal((await client.deleteDocument(id)).deleted, true);
  await assert.rejects(client.documentPictures(id), NotFoundError);
});

test("document stream", async () => {
  let stream = await client.convertDocumentStream(DOCS, {
    file: enc("%PDF fake"),
    filename: "a.pdf",
    output_format: "json",
  });
  const kinds = [];
  for await (const event of stream) kinds.push(event.object);
  assert.deepEqual(kinds, ["document.started", "document.page", "document.page", "document.page", "document"]);
  assert.deepEqual(stream.started.pages, [1, 2, 3]);
  // Iterating first and asking for the result afterwards works too.
  assert.equal((await stream.final()).pages.length, 3);

  stream = await client.convertDocumentStream(DOCS, { file: enc("%PDF fake"), filename: "a.pdf" });
  const doc = await stream.final();
  // The server leaves `pages` empty in its last event; the stream puts them back.
  assert.deepEqual(doc.pages.map((p) => p.page), [1, 2, 3]);
  assert.equal(doc.markdown.match(/## Page/g).length, 3);

  stream = await client.convertDocumentStream(DOCS, { file: enc("%PDF fake"), pages: "2" });
  const pages = [];
  for await (const page of stream.pages()) pages.push(page.page);
  assert.deepEqual(pages, [2]);
});

test("document needs a file or a url", async () => {
  await assert.rejects(client.convertDocument(DOCS, {}), TypeError);
});

test("memory and placement", async () => {
  await client.chat(CHAT, USER);
  const status = await client.memoryStatus();
  const model = status.models.find((m) => m.id === CHAT);
  assert.equal(model.current_tier, "tier1_gpu_vram");
  const moved = await client.setTier(CHAT, "tier2_pinned_ram");
  assert.equal(moved.placement.primary_tier, "tier2_pinned_ram");
  const back = await client.setTier(CHAT, "tier1_gpu_vram", { contextLen: 4096 });
  assert.equal(back.placement.primary_tier, "tier1_gpu_vram");
});

test("kv parking", async () => {
  const status = await client.kvPark();
  assert.equal(status.enabled, false);
  assert.equal(status.used_bytes, 0);
  try {
    const on = await client.setKvPark({ enabled: true, max_bytes: 1073741824 });
    assert.equal(on.enabled, true);
    assert.equal(on.max_bytes, 1073741824);
    assert.deepEqual(on.models, []);
    assert.equal((await client.memoryStatus()).kv_park.enabled, true);
    await assert.rejects(client.setKvPark({}), TypeError);
  } finally {
    const off = await client.setKvPark({ enabled: false });
    assert.equal(off.enabled, false);
  }
});

test("enable and disable", async () => {
  assert.equal((await client.disableModel(IMAGE)).enabled, false);
  assert.ok(!(await client.models()).data.some((m) => m.id === IMAGE));
  await assert.rejects(client.images(IMAGE, "x"), (err) => {
    assert.ok(err instanceof NotFoundError);
    assert.equal(err.type, "model_disabled");
    return true;
  });
  assert.equal((await client.enableModel(IMAGE)).enabled, true);
  assert.ok((await client.images(IMAGE, "x")).data.length);
});

test("model pool: add, replace, remove", async () => {
  const pooled = "lfm3:1.7b";
  const isNotFound = (err) => err instanceof NotFoundError;
  try {
    const added = await client.addModel({ catalog: pooled });
    assert.equal(added.model, pooled);
    assert.equal(added.action, "added");
    assert.equal(typeof added.persisted, "boolean");
    assert.ok(added.placement);
    assert.ok((await client.models()).data.some((m) => m.id === pooled));
    const reply = await client.chat(pooled, USER, { max_tokens: 10 });
    assert.equal(reply.choices[0].message.content, "echo: hello there");
    await assert.rejects(client.addModel({ catalog: pooled }), (err) => err instanceof BadRequestError);
    const replaced = await client.replaceModel(pooled, { catalog: "lfm3:1.7b:Q4_K_M", drainSecs: 5 });
    assert.equal(replaced.model, pooled);
    assert.equal(replaced.action, "replaced");
    const removed = await client.removeModel(pooled, { drainSecs: 5 });
    assert.equal(removed.model, pooled);
    assert.equal(removed.action, "removed");
    assert.equal(removed.placement, undefined);
    await assert.rejects(client.chat(pooled, USER), isNotFound);
    await assert.rejects(client.removeModel(pooled), isNotFound);
    await assert.rejects(client.replaceModel(pooled, { catalog: pooled }), isNotFound);
  } finally {
    await client.removeModel(pooled).catch((err) => {
      if (!isNotFound(err)) throw err;
    });
  }
  await assert.rejects(client.addModel({}), TypeError);
  await assert.rejects(client.addModel({ catalog: pooled, config: "[[models]]" }), TypeError);
});

test("moe offload", async () => {
  try {
    const on = await client.setMoeOffload(CHAT, "on");
    assert.equal(on.model, CHAT);
    assert.equal(on.moe_offload, "on");
    assert.ok(["now", "next_load"].includes(on.applies));
    assert.equal(on.placement.cpu_expert_layers ?? 0, 0);
    const status = (await client.memoryStatus()).models.find((m) => m.id === CHAT);
    assert.equal(status.moe_offload, "on");
    assert.ok(status.expert_bytes >= 0);
    await assert.rejects(client.setMoeOffload("no-such-model", "on"), NotFoundError);
    await assert.rejects(client.setMoeOffload(CHAT, "sometimes"), (err) => {
      assert.ok(err instanceof APIError);
      assert.equal(err.status, 422);
      return true;
    });
  } finally {
    assert.equal((await client.setMoeOffload(CHAT, "off")).moe_offload, "off");
  }
});

test("logs", async () => {
  assert.ok((await client.chat(CHAT, USER)).id);
  const logs = await client.logs({ route: "/v1/chat/completions", limit: 5 });
  assert.ok(logs.records.length);
  const record = await client.log(logs.records[0].id);
  assert.equal(record.route, "/v1/chat/completions");
  assert.equal((await client.setLogLevel("debug")).level, "debug");
  await client.setLogLevel("info");
  assert.equal((await client.clearLogs()).cleared, true);
});

// --- tool calling (docs/api.md section 7.5) ------------------------------------------------------

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Current weather for a city",
      parameters: { type: "object", properties: { city: { type: "string" } }, required: ["city"] },
    },
  },
  { type: "function", function: { name: "get_time" } },
];

test("chat tool call with auto", async () => {
  const text = "please get_weather for Paris";
  const reply = await client.chat(CHAT, [{ role: "user", content: text }], { tools: TOOLS, tool_choice: "auto" });
  const choice = reply.choices[0];
  assert.equal(choice.finish_reason, "tool_calls");
  assert.equal(choice.message.content, "");
  assert.equal(choice.message.tool_calls.length, 1);
  const call = choice.message.tool_calls[0];
  assert.equal(call.type, "function");
  assert.equal(call.function.name, "get_weather");
  assert.ok(call.id);
  assert.deepEqual(JSON.parse(call.function.arguments), { text });
  assert.deepEqual(parseArguments(call), { text });
});

test("chat without a tool named answers in text", async () => {
  const reply = await client.chat(CHAT, USER, { tools: TOOLS });
  assert.equal(reply.choices[0].finish_reason, "stop");
  assert.equal(reply.choices[0].message.content, "echo: hello there");
  assert.equal(reply.choices[0].message.tool_calls, undefined);
});

test("chat tool round trip", async () => {
  const messages = [{ role: "user", content: "what is get_time now" }];
  const first = await client.chat(CHAT, messages, { tools: TOOLS });
  const [call] = first.choices[0].message.tool_calls;
  assert.equal(call.function.name, "get_time");
  const assistant = assistantMessage(first);
  assert.equal(assistant.role, "assistant");
  assert.equal(assistant.content, null);
  assert.deepEqual(assistant.tool_calls, [call]);
  messages.push(assistant, toolMessage(call.id, '{"time": "12:00"}'));
  const second = await client.chat(CHAT, messages, { tools: TOOLS });
  assert.equal(second.choices[0].finish_reason, "stop");
  assert.equal(second.choices[0].message.content, 'tool said: {"time": "12:00"}');
});

test("chat stream tool call with required", async () => {
  const stream = await client.chatStream(CHAT, USER, { tools: TOOLS, tool_choice: "required" });
  const completion = await stream.final();
  assert.equal(stream.finishReason, "tool_calls");
  assert.equal(stream.toolCalls.length, 1);
  const call = stream.toolCalls[0];
  assert.equal(call.function.name, "get_weather");
  assert.ok(call.id);
  assert.deepEqual(JSON.parse(call.function.arguments), { text: "hello there" });
  assert.equal(completion.choices[0].finish_reason, "tool_calls");
  assert.deepEqual(completion.choices[0].message.tool_calls, [call]);
  assert.equal(assistantMessage(completion).content, null);
});

test("chat stream tool call with a named function", async () => {
  const stream = await client.chatStream(CHAT, USER, {
    tools: TOOLS,
    tool_choice: { type: "function", function: { name: "get_time" } },
  });
  const chunks = [];
  for await (const chunk of stream) chunks.push(chunk);
  assert.ok(chunks.some((c) => c.choices[0]?.delta?.tool_calls?.[0]?.index === 0));
  assert.deepEqual(stream.toolCalls.map((c) => c.function.name), ["get_time"]);
});

test("chat tool_choice naming an unknown tool", async () => {
  await assert.rejects(
    client.chat(CHAT, USER, { tools: TOOLS, tool_choice: { type: "function", function: { name: "nope" } } }),
    (err) => err instanceof BadRequestError && err.status === 400 && err.type === "invalid_request_error",
  );
});
