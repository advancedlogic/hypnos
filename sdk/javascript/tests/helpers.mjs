// Starts the mock Hypnos server (the real router with mock engines) for the tests: the binary is
// $HYPNOS_MOCK_SERVER when set, else backend/target/debug/examples/mock_server, built with cargo
// when it is missing.

import { spawn, spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";

import { Client, RetryPolicy } from "../dist/index.js";

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../../..");
const BACKEND = path.join(REPO, "backend");
const EXE = process.platform === "win32" ? ".exe" : "";

/** Retries without the real waits, so the retry tests take milliseconds. */
export const FAST_RETRY = new RetryPolicy({
  busyBackoff: 0.01,
  memoryBackoff: 0.01,
  backoff: 0.01,
  unavailableWait: 0.01,
});

function mockBinary() {
  if (process.env.HYPNOS_MOCK_SERVER) return process.env.HYPNOS_MOCK_SERVER;
  const binary = path.join(BACKEND, "target", "debug", "examples", `mock_server${EXE}`);
  if (!existsSync(binary)) {
    const args = ["build", "-p", "hypnos-server", "--example", "mock_server", "--no-default-features"];
    if (process.platform === "darwin" && process.arch === "arm64") args.push("--features", "metal");
    const built = spawnSync("cargo", args, { cwd: BACKEND, stdio: "inherit" });
    if (built.status !== 0) throw new Error("could not build the mock server");
  }
  return binary;
}

export class MockServer {
  static async start(args = []) {
    const binary = mockBinary();
    const debug = path.dirname(path.dirname(binary));
    const libs = [debug, path.join(debug, "deps")].join(path.delimiter);
    const env = { ...process.env };
    for (const name of ["LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "PATH"]) {
      env[name] = libs + path.delimiter + (env[name] ?? "");
    }
    const child = spawn(binary, ["--port", "0", ...args], { env, stdio: ["ignore", "pipe", "ignore"] });
    const line = await new Promise((resolve, reject) => {
      const rl = createInterface({ input: child.stdout });
      rl.once("line", (l) => resolve(l.trim()));
      child.once("exit", (code) => reject(new Error(`mock server exited with ${code}`)));
    });
    if (!line.startsWith("listening on ")) {
      child.kill();
      throw new Error(`mock server did not start: ${line}`);
    }
    const server = new MockServer();
    server.child = child;
    server.url = line.slice("listening on ".length);
    return server;
  }

  async control(route, body) {
    const res = await fetch(`${this.url}/mock/${route}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`/mock/${route}: ${res.status}`);
    await res.body?.cancel();
  }

  fail(model, error, times = 1) {
    return this.control("fail", { model, error, times });
  }

  reset() {
    return this.control("reset", {});
  }

  client(options = {}) {
    return new Client({ baseUrl: this.url, retry: FAST_RETRY, ...options });
  }

  stop() {
    this.child.kill();
  }
}

export const CHAT = "gemma4:26b-q4";
export const STT = "gemma4:e4b-q4";
export const EMBED = "gemmaembedding2";
export const TTS = "kokoro-82m-q4";
export const IMAGE = "moss-image";
export const DOCS = "docling";
export const NER = "gliner2.5-multi";
export const USER = [{ role: "user", content: "hello there" }];

export const enc = (s) => new TextEncoder().encode(s);
export const b64decode = (s) => Uint8Array.from(Buffer.from(s, "base64"));
export const startsWith = (bytes, prefix) => prefix.every((b, i) => bytes[i] === b);
