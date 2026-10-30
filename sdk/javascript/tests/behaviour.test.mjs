// Errors, retries, readiness, auth and the per-model bound.

import assert from "node:assert/strict";
import { after, before, beforeEach, test } from "node:test";

import {
  APIConnectionError,
  AuthenticationError,
  BadRequestError,
  Client,
  HypnosError,
  InsufficientMemoryError,
  ModelBusyError,
  NO_RETRY,
  NotFoundError,
  RateLimitError,
  ServerError,
  ServiceUnavailableError,
} from "../dist/index.js";
import { CHAT, EMBED, FAST_RETRY, MockServer } from "./helpers.mjs";

const USER = [{ role: "user", content: "hi" }];
let server;
let keyed;
let client;

before(async () => {
  server = await MockServer.start();
  keyed = await MockServer.start(["--api-key", "secret", "--limited-key", "limited:1"]);
  client = server.client();
});
after(() => {
  server.stop();
  keyed.stop();
});
beforeEach(() => server.reset());

test("an unknown model is not found", async () => {
  await assert.rejects(client.chat("no-such-model", USER), (err) => {
    assert.ok(err instanceof NotFoundError);
    assert.equal(err.status, 404);
    assert.equal(err.type, "model_not_found");
    assert.match(err.requestId, /^req-/);
    return true;
  });
});

test("bad request", async () => {
  await assert.rejects(client.chat(CHAT, []), BadRequestError);
});

test("busy is retried", async () => {
  await server.fail(EMBED, "busy", 2);
  assert.ok((await client.embeddings(EMBED, "a")).data.length);
});

test("busy without retries", async () => {
  await server.fail(EMBED, "busy");
  const c = server.client({ retry: NO_RETRY });
  await assert.rejects(c.embeddings(EMBED, "a"), (err) => {
    assert.ok(err instanceof ModelBusyError);
    assert.equal(err.status, 429);
    assert.equal(err.type, "model_busy");
    return true;
  });
});

test("memory errors are retried", async () => {
  await server.fail(EMBED, "insufficient_memory", 2);
  assert.ok((await client.embeddings(EMBED, "a")).data.length);
});

test("a memory error surfaces after the retries", async () => {
  await server.fail(EMBED, "insufficient_memory", 10);
  await assert.rejects(client.embeddings(EMBED, "a"), (err) => {
    assert.ok(err instanceof InsufficientMemoryError);
    assert.equal(err.status, 507);
    return true;
  });
});

test("a 500 is retried once", async () => {
  await server.fail(EMBED, "server_error", 1);
  assert.ok((await client.embeddings(EMBED, "a")).data.length);
  await server.fail(EMBED, "server_error", 2);
  await assert.rejects(client.embeddings(EMBED, "a"), ServerError);
});

test("not ready", async () => {
  await server.control("ready", { ready: false });
  const c = server.client({ retry: NO_RETRY });
  assert.equal(await c.isReady(), false);
  await assert.rejects(c.embeddings(EMBED, "a"), (err) => {
    assert.ok(err instanceof ServiceUnavailableError);
    assert.equal(err.retryAfter, 5);
    return true;
  });
  // Models stay open while staging.
  assert.ok((await c.models()).data.length);
  setTimeout(() => server.control("ready", { ready: true }), 300);
  await c.waitUntilReady({ timeoutMs: 5000, intervalMs: 50 });
  assert.ok((await c.embeddings(EMBED, "a")).data.length);
});

test("an unreachable server", async () => {
  const c = new Client({ baseUrl: "http://127.0.0.1:9", retry: NO_RETRY });
  await assert.rejects(c.health(), APIConnectionError);
  assert.equal(await c.isReady(), false);
  await assert.rejects(c.waitUntilReady({ timeoutMs: 200, intervalMs: 50 }), HypnosError);
});

test("api keys", async () => {
  const anonymous = new Client({ baseUrl: keyed.url, apiKey: "", retry: NO_RETRY });
  await assert.rejects(anonymous.models(), (err) => {
    assert.ok(err instanceof AuthenticationError);
    assert.equal(err.code, "invalid_api_key");
    return true;
  });
  assert.equal((await anonymous.health()).status, "ok");
  const c = new Client({ baseUrl: keyed.url, apiKey: "secret" });
  assert.ok((await c.models()).data.length);
});

test("rate limit", async () => {
  const c = new Client({ baseUrl: keyed.url, apiKey: "limited", retry: NO_RETRY });
  await c.models();
  await assert.rejects(c.models(), (err) => {
    assert.ok(err instanceof RateLimitError);
    assert.equal(err.code, "requests_per_minute");
    assert.ok(err.retryAfter > 0);
    return true;
  });
});

test("the base url accepts /v1", async () => {
  const c = new Client({ baseUrl: server.url + "/v1/" });
  assert.equal(c.baseUrl, server.url);
  assert.ok((await c.models()).data.length);
});

test("the per-model bound", async () => {
  const c = server.client({ maxConcurrencyPerModel: 1 });
  const stream = await c.chatStream(CHAT, [{ role: "user", content: "count" }], { max_tokens: 1000 });
  let done = false;
  const second = c.chat(CHAT, USER).then(() => {
    done = true;
  });
  // The second request waits for the open stream's permit.
  await new Promise((r) => setTimeout(r, 300));
  assert.equal(done, false);
  stream.close();
  await second;
  assert.equal(done, true);
  // Another model is not affected by the bound.
  assert.ok((await c.embeddings(EMBED, "a")).data.length);
});

test("an aborted request waiting for a permit gives up", async () => {
  const c = server.client({ maxConcurrencyPerModel: 1 });
  const stream = await c.chatStream(CHAT, [{ role: "user", content: "count" }], { max_tokens: 1000 });
  const controller = new AbortController();
  const waiting = c.chat(CHAT, USER, { signal: controller.signal });
  controller.abort();
  await assert.rejects(waiting);
  stream.close();
  // The permit was not lost.
  assert.ok((await c.chat(CHAT, USER)).choices.length);
});
