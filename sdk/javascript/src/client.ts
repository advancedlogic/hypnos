/** The Hypnos client: one method per route of docs/api.md. */

import {
  APIConnectionError,
  APIError,
  APITimeoutError,
  HypnosError,
  errorClass,
} from "./errors.js";
import { RetryPolicy } from "./retry.js";
import { ChatStream, DocumentStream, SpeechStream, speechTelemetry } from "./streams.js";
import type {
  ChatCompletion,
  ChatMessage,
  Document,
  EmbeddingResponse,
  EntityResponse,
  Health,
  ImageResponse,
  LogList,
  LogRecord,
  MemoryStatus,
  MemoryTier,
  ModelList,
  ModelSwitch,
  Picture,
  PictureList,
  Speech,
  TierMove,
  Transcription,
} from "./types.js";

export const VERSION = "0.2.0";
export const DEFAULT_BASE_URL = "http://127.0.0.1:9999";

/** A file to upload: bytes, a Blob/File, or (Node only) a path on disk. */
export type FileInput = Blob | Uint8Array | ArrayBuffer | string;

export interface ClientOptions {
  /** Default `$HYPNOS_BASE_URL` or `http://127.0.0.1:9999`; a trailing `/v1` is accepted. */
  baseUrl?: string;
  /** Default `$HYPNOS_API_KEY`; sent as `Authorization: Bearer <key>`. */
  apiKey?: string;
  /**
   * Milliseconds to wait for the response headers (default 900 000, the server's default
   * `request_timeout_secs`: a cold request pays a model load). Streams run as long as they take.
   */
  timeoutMs?: number;
  /** The retry policy of docs/api.md section 17. */
  retry?: RetryPolicy;
  /** At most this many requests in flight per model from this client (a stream until closed). */
  maxConcurrencyPerModel?: number;
  /** A fetch implementation; default the global one. */
  fetch?: typeof fetch;
}

/** Options every call accepts. */
export interface CallOptions {
  /** Aborts the request (and closes a stream it opened). */
  signal?: AbortSignal;
}

export interface ChatOptions extends CallOptions {
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  seed?: number;
  /** Hypnos extension: the context window to reserve. */
  num_ctx?: number;
  /** Hypnos extension: Gemma 4's thinking channel. */
  think?: boolean;
  /** Any other body field. */
  extra?: Record<string, unknown>;
}

export interface SpeechOptions extends CallOptions {
  voice?: string;
  response_format?: "wav" | "pcm" | string;
  speed?: number;
}

export interface ImageOptions extends CallOptions {
  n?: number;
  size?: string;
  negative_prompt?: string;
  steps?: number;
  seed?: number;
  cfg_scale?: number;
}

export interface TranscribeOptions extends CallOptions {
  filename?: string;
  language?: string;
  prompt?: string;
}

export interface EntityOptions extends CallOptions {
  threshold?: number;
  flat?: boolean;
}

export interface DocumentOptions extends CallOptions {
  /** The document to upload; pass this or `url`. */
  file?: FileInput;
  /** An http(s) URL the server fetches (needs `allow_remote_image_urls`). */
  url?: string;
  filename?: string;
  /** `"3"` or `"2-5"`. */
  pages?: string;
  ocr?: boolean;
  output_format?: "markdown" | "json";
  describe_pictures?: string;
  return_pictures?: boolean;
}

export interface LogQuery extends CallOptions {
  model?: string;
  route?: string;
  /** `ok`, `error`, `4xx` or `5xx`. */
  status?: string;
  key?: string;
  limit?: number;
}

interface SendOptions {
  model?: string;
  json?: unknown;
  form?: FormData;
  query?: Record<string, string | number | undefined>;
  accept?: string;
  retry?: boolean;
  signal?: AbortSignal;
}

interface Sent {
  response: Response;
  controller: AbortController;
  release: () => void;
}

function env(name: string): string | undefined {
  const proc = (globalThis as { process?: { env?: Record<string, string | undefined> } }).process;
  return proc?.env?.[name];
}

function baseUrlFrom(configured?: string): string {
  let url = (configured ?? env("HYPNOS_BASE_URL") ?? DEFAULT_BASE_URL).replace(/\/+$/, "");
  if (url.endsWith("/v1")) url = url.slice(0, -3);
  return url;
}

function dropUndefined<T extends Record<string, unknown>>(body: T): Partial<T> {
  return Object.fromEntries(Object.entries(body).filter(([, v]) => v !== undefined)) as Partial<T>;
}

function sleep(seconds: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(signal.reason);
    const timer = setTimeout(() => {
      signal?.removeEventListener("abort", onAbort);
      resolve();
    }, seconds * 1000);
    const onAbort = () => {
      clearTimeout(timer);
      reject(signal?.reason);
    };
    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function parseRetryAfter(value: string | null): number | undefined {
  if (value === null) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? Math.max(0, n) : undefined;
}

async function apiError(response: Response): Promise<APIError> {
  const text = await response.text().catch(() => "");
  let message = text.trim() || response.statusText;
  let type: string | undefined;
  let code: string | undefined;
  let body: unknown;
  try {
    const parsed = JSON.parse(text) as { error?: { message?: string; type?: string; code?: string } };
    if (parsed && typeof parsed.error === "object" && parsed.error) {
      body = parsed;
      message = String(parsed.error.message ?? message);
      type = parsed.error.type;
      code = parsed.error.code ?? undefined;
    }
  } catch {
    // plain-text body (408, 413)
  }
  const Cls = errorClass(response.status, type);
  return new Cls(message, {
    status: response.status,
    type,
    code,
    requestId: response.headers.get("x-request-id") ?? undefined,
    retryAfter: parseRetryAfter(response.headers.get("retry-after")),
    body,
  });
}

/** A counting semaphore: `acquire` waits for a free permit, `release` hands it on. */
class Semaphore {
  private waiters: (() => void)[] = [];
  constructor(private permits: number) {}

  async acquire(signal?: AbortSignal): Promise<void> {
    if (this.permits > 0) {
      this.permits--;
      return;
    }
    await new Promise<void>((resolve, reject) => {
      const wake = () => {
        signal?.removeEventListener("abort", onAbort);
        resolve();
      };
      const onAbort = () => {
        this.waiters = this.waiters.filter((w) => w !== wake);
        reject(signal?.reason);
      };
      this.waiters.push(wake);
      signal?.addEventListener("abort", onAbort, { once: true });
    });
  }

  release(): void {
    const next = this.waiters.shift();
    if (next) next();
    else this.permits++;
  }
}

async function toBlob(file: FileInput, filename?: string): Promise<{ blob: Blob; name: string }> {
  if (typeof file === "string") {
    // A path: Node only. The module name is built at run time so bundlers leave it alone.
    const fs = (await import(/* @vite-ignore */ ["node", "fs/promises"].join(":"))) as {
      readFile(path: string): Promise<Uint8Array>;
    };
    const bytes = await fs.readFile(file);
    return { blob: new Blob([bytes as BlobPart]), name: filename ?? file.split(/[\\/]/).pop()! };
  }
  if (file instanceof Blob) {
    return { blob: file, name: filename ?? (file as { name?: string }).name ?? "upload" };
  }
  return { blob: new Blob([file as BlobPart]), name: filename ?? "upload" };
}

function formValue(value: unknown): string {
  return typeof value === "boolean" ? (value ? "true" : "false") : String(value);
}

/**
 * A client for one Hypnos server.
 *
 * ```ts
 * const hypnos = new Client();
 * const reply = await hypnos.chat("gemma4:26b-q4", [{ role: "user", content: "Hello" }]);
 * ```
 */
export class Client {
  readonly baseUrl: string;
  readonly retry: RetryPolicy;
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly maxPerModel?: number;
  private readonly limits = new Map<string, Semaphore>();
  private readonly fetchImpl: typeof fetch;

  constructor(options: ClientOptions = {}) {
    this.baseUrl = baseUrlFrom(options.baseUrl);
    this.apiKey = options.apiKey ?? env("HYPNOS_API_KEY");
    this.timeoutMs = options.timeoutMs ?? 900_000;
    this.retry = options.retry ?? new RetryPolicy();
    this.maxPerModel = options.maxConcurrencyPerModel;
    this.fetchImpl = options.fetch ?? globalThis.fetch.bind(globalThis);
  }

  // --- plumbing ------------------------------------------------------------------------------

  private headers(accept?: string): Record<string, string> {
    const h: Record<string, string> = { Accept: accept ?? "application/json" };
    // Browsers refuse to set User-Agent; Node sends it.
    if (typeof window === "undefined") h["User-Agent"] = `hypnos-client-js/${VERSION}`;
    if (this.apiKey) h.Authorization = `Bearer ${this.apiKey}`;
    return h;
  }

  private limit(model?: string): Semaphore | undefined {
    if (!model || !this.maxPerModel) return undefined;
    let sem = this.limits.get(model);
    if (!sem) {
      sem = new Semaphore(this.maxPerModel);
      this.limits.set(model, sem);
    }
    return sem;
  }

  /**
   * Sends the request, retrying per the policy, and returns the open response with its abort
   * controller and a release function for the model's permit (call it when the body is done).
   */
  private async send(method: string, path: string, opts: SendOptions = {}): Promise<Sent> {
    const sem = this.limit(opts.model);
    await sem?.acquire(opts.signal);
    let released = false;
    const release = () => {
      if (!released) {
        released = true;
        sem?.release();
      }
    };
    const query = opts.query
      ? new URLSearchParams(
          Object.entries(opts.query)
            .filter(([, v]) => v !== undefined)
            .map(([k, v]) => [k, String(v)]),
        ).toString()
      : "";
    const url = this.baseUrl + path + (query ? `?${query}` : "");
    try {
      for (let attempt = 0; ; attempt++) {
        const controller = new AbortController();
        const onAbort = () => controller.abort(opts.signal?.reason);
        if (opts.signal?.aborted) throw opts.signal.reason;
        opts.signal?.addEventListener("abort", onAbort, { once: true });
        let timedOut = false;
        const timer = setTimeout(() => {
          timedOut = true;
          controller.abort();
        }, this.timeoutMs);
        const headers = this.headers(opts.accept);
        let body: BodyInit | undefined;
        if (opts.form) body = opts.form;
        else if (opts.json !== undefined) {
          headers["Content-Type"] = "application/json";
          body = JSON.stringify(opts.json);
        }
        let error: HypnosError;
        try {
          const response = await this.fetchImpl(url, {
            method,
            headers,
            body,
            signal: controller.signal,
          });
          clearTimeout(timer);
          if (response.ok) return { response, controller, release };
          opts.signal?.removeEventListener("abort", onAbort);
          error = await apiError(response);
        } catch (e) {
          clearTimeout(timer);
          opts.signal?.removeEventListener("abort", onAbort);
          if (opts.signal?.aborted) throw opts.signal.reason;
          error = timedOut
            ? new APITimeoutError(`no response within ${this.timeoutMs} ms`, { status: 408 })
            : new APIConnectionError(`${method} ${path}: ${String((e as Error)?.message ?? e)}`);
        }
        const wait = opts.retry === false ? undefined : this.retry.delay(error, attempt);
        if (wait === undefined) throw error;
        await sleep(wait, opts.signal);
      }
    } catch (e) {
      release();
      throw e;
    }
  }

  private async json<T>(method: string, path: string, opts: SendOptions = {}): Promise<T> {
    const { response, release } = await this.send(method, path, opts);
    try {
      return (await response.json()) as T;
    } finally {
      release();
    }
  }

  private async bytes(method: string, path: string, opts: SendOptions): Promise<Response> {
    const { response, release } = await this.send(method, path, opts);
    try {
      // Read now so the permit is held for the whole body.
      const buf = await response.arrayBuffer();
      return new Response(buf, { status: response.status, headers: response.headers });
    } finally {
      release();
    }
  }

  // --- server --------------------------------------------------------------------------------

  /** `GET /health`: the server is up (it may still be staging models). */
  health(opts: CallOptions = {}): Promise<Health> {
    return this.json("GET", "/health", { retry: false, signal: opts.signal });
  }

  /** `GET /ready`: true once the startup models are staged. */
  async isReady(opts: CallOptions = {}): Promise<boolean> {
    try {
      const response = await this.fetchImpl(this.baseUrl + "/ready", {
        headers: this.headers(),
        signal: opts.signal,
      });
      await response.body?.cancel();
      return response.status === 200;
    } catch (e) {
      if (opts.signal?.aborted) throw e;
      return false;
    }
  }

  /** Polls `/ready` until it answers 200; throws after `timeoutMs` (default 10 minutes). */
  async waitUntilReady(
    options: { timeoutMs?: number; intervalMs?: number; signal?: AbortSignal } = {},
  ): Promise<void> {
    const timeout = options.timeoutMs ?? 600_000;
    const deadline = Date.now() + timeout;
    while (!(await this.isReady({ signal: options.signal }))) {
      if (Date.now() >= deadline) {
        throw new HypnosError(`${this.baseUrl} was not ready within ${Math.round(timeout / 1000)} s`);
      }
      await sleep((options.intervalMs ?? 1000) / 1000, options.signal);
    }
  }

  /** `GET /metrics`: Prometheus text. */
  async metrics(opts: CallOptions = {}): Promise<string> {
    const response = await this.bytes("GET", "/metrics", { accept: "text/plain", signal: opts.signal });
    return response.text();
  }

  // --- models --------------------------------------------------------------------------------

  /** `GET /v1/models`: the models in service. */
  models(opts: CallOptions = {}): Promise<ModelList> {
    return this.json("GET", "/v1/models", { signal: opts.signal });
  }

  /** `POST /v1/models/{id}/enable`. */
  enableModel(model: string, opts: CallOptions = {}): Promise<ModelSwitch> {
    return this.json("POST", `/v1/models/${encodeURIComponent(model)}/enable`, { signal: opts.signal });
  }

  /** `POST /v1/models/{id}/disable`; refused with `ModelBusyError` while leased. */
  disableModel(model: string, opts: CallOptions = {}): Promise<ModelSwitch> {
    return this.json("POST", `/v1/models/${encodeURIComponent(model)}/disable`, { signal: opts.signal });
  }

  // --- chat ----------------------------------------------------------------------------------

  private chatBody(model: string, messages: ChatMessage[], stream: boolean, o: ChatOptions) {
    return {
      ...dropUndefined({
        model,
        messages,
        max_tokens: o.max_tokens,
        temperature: o.temperature,
        top_p: o.top_p,
        top_k: o.top_k,
        seed: o.seed,
        num_ctx: o.num_ctx,
        think: o.think,
        stream,
      }),
      ...(o.extra ?? {}),
    };
  }

  /** `POST /v1/chat/completions`. */
  chat(model: string, messages: ChatMessage[], options: ChatOptions = {}): Promise<ChatCompletion> {
    return this.json("POST", "/v1/chat/completions", {
      model,
      json: this.chatBody(model, messages, false, options),
      signal: options.signal,
    });
  }

  /** `POST /v1/chat/completions` with `stream: true`. `close()` the stream to cancel. */
  async chatStream(model: string, messages: ChatMessage[], options: ChatOptions = {}): Promise<ChatStream> {
    const { response, controller, release } = await this.send("POST", "/v1/chat/completions", {
      model,
      json: this.chatBody(model, messages, true, options),
      accept: "text/event-stream",
      signal: options.signal,
    });
    return new ChatStream(response, controller, release);
  }

  // --- embeddings, audio, images, entities ---------------------------------------------------

  /** `POST /v1/embeddings`; `data` is in input order. */
  embeddings(model: string, input: string | string[], opts: CallOptions = {}): Promise<EmbeddingResponse> {
    return this.json("POST", "/v1/embeddings", { model, json: { model, input }, signal: opts.signal });
  }

  /** `POST /v1/audio/transcriptions` (multipart). WAV 16-bit PCM is the reliable format. */
  async transcribe(model: string, file: FileInput, options: TranscribeOptions = {}): Promise<Transcription> {
    const { blob, name } = await toBlob(file, options.filename);
    const form = new FormData();
    form.append("model", model);
    if (options.language !== undefined) form.append("language", options.language);
    if (options.prompt !== undefined) form.append("prompt", options.prompt);
    form.append("file", blob, name);
    return this.json("POST", "/v1/audio/transcriptions", { model, form, signal: options.signal });
  }

  private speechBody(model: string, input: string, o: SpeechOptions, stream: boolean) {
    return dropUndefined({
      model,
      input,
      voice: o.voice,
      response_format: o.response_format,
      speed: o.speed,
      stream,
    });
  }

  /**
   * `POST /v1/audio/speech`: WAV (or `response_format: "pcm"`) bytes plus the `x-hypnos-*`
   * telemetry headers. Check `contentType`: unsupported formats fall back to WAV.
   */
  async speech(model: string, input: string, options: SpeechOptions = {}): Promise<Speech> {
    const response = await this.bytes("POST", "/v1/audio/speech", {
      model,
      json: this.speechBody(model, input, options, false),
      accept: "audio/*",
      signal: options.signal,
    });
    return {
      audio: new Uint8Array(await response.arrayBuffer()),
      contentType: response.headers.get("content-type") ?? "",
      telemetry: speechTelemetry(response.headers),
      requestId: response.headers.get("x-request-id") ?? undefined,
    };
  }

  /** `POST /v1/audio/speech` with `stream: true`: audio chunks as they are made. */
  async speechStream(model: string, input: string, options: SpeechOptions = {}): Promise<SpeechStream> {
    const { response, controller, release } = await this.send("POST", "/v1/audio/speech", {
      model,
      json: this.speechBody(model, input, options, true),
      accept: "audio/*",
      signal: options.signal,
    });
    return new SpeechStream(response, controller, release);
  }

  /** `POST /v1/images/generations`: base64 PNGs in `data[i].b64_json`. */
  images(model: string, prompt: string, options: ImageOptions = {}): Promise<ImageResponse> {
    const { signal, ...rest } = options;
    return this.json("POST", "/v1/images/generations", {
      model,
      json: dropUndefined({ model, prompt, ...rest }),
      signal,
    });
  }

  /**
   * `POST /v1/entities`: zero-shot entity extraction. `start`/`end` are Unicode code point
   * offsets: slice with `sliceByCodePoints`, not `String.prototype.slice`.
   */
  entities(
    model: string,
    input: string | string[],
    labels: string[],
    options: EntityOptions = {},
  ): Promise<EntityResponse> {
    return this.json("POST", "/v1/entities", {
      model,
      json: dropUndefined({ model, input, labels, threshold: options.threshold, flat: options.flat }),
      signal: options.signal,
    });
  }

  // --- documents -----------------------------------------------------------------------------

  private async documentRequest(model: string, o: DocumentOptions, stream: boolean): Promise<SendOptions> {
    if ((o.file === undefined) === (o.url === undefined)) {
      throw new TypeError("pass exactly one of `file` and `url`");
    }
    const fields = dropUndefined({
      model,
      pages: o.pages,
      ocr: o.ocr,
      output_format: o.output_format,
      describe_pictures: o.describe_pictures,
      return_pictures: o.return_pictures,
      stream,
    });
    if (o.file !== undefined) {
      const { blob, name } = await toBlob(o.file, o.filename);
      const form = new FormData();
      for (const [k, v] of Object.entries(fields)) form.append(k, formValue(v));
      form.append("filename", name);
      form.append("file", blob, name);
      return { model, form, signal: o.signal };
    }
    return {
      model,
      json: dropUndefined({ ...fields, url: o.url, filename: o.filename }),
      signal: o.signal,
    };
  }

  /** `POST /v1/documents/convert`: an upload (`file`) or a `url` the server fetches, to Markdown. */
  async convertDocument(model: string, options: DocumentOptions): Promise<Document> {
    return this.json("POST", "/v1/documents/convert", await this.documentRequest(model, options, false));
  }

  /** The streaming form of `convertDocument`: one event per page. */
  async convertDocumentStream(model: string, options: DocumentOptions): Promise<DocumentStream> {
    const req = await this.documentRequest(model, options, true);
    const { response, controller, release } = await this.send("POST", "/v1/documents/convert", {
      ...req,
      accept: "text/event-stream",
    });
    return new DocumentStream(response, controller, release);
  }

  /** `GET /v1/documents/{id}/pictures`: the records, without image data. */
  documentPictures(documentId: string, opts: CallOptions = {}): Promise<PictureList> {
    return this.json("GET", `/v1/documents/${encodeURIComponent(documentId)}/pictures`, {
      signal: opts.signal,
    });
  }

  /** `GET /v1/documents/{id}/pictures/{n}`: one record with `image` (base64). */
  documentPicture(documentId: string, index: number, opts: CallOptions = {}): Promise<Picture> {
    return this.json("GET", `/v1/documents/${encodeURIComponent(documentId)}/pictures/${index}`, {
      signal: opts.signal,
    });
  }

  /** The picture's image bytes and media type (`?format=image`). */
  async documentPictureBytes(
    documentId: string,
    index: number,
    opts: CallOptions = {},
  ): Promise<{ data: Uint8Array; contentType: string }> {
    const response = await this.bytes(
      "GET",
      `/v1/documents/${encodeURIComponent(documentId)}/pictures/${index}`,
      { query: { format: "image" }, accept: "*/*", signal: opts.signal },
    );
    return {
      data: new Uint8Array(await response.arrayBuffer()),
      contentType: response.headers.get("content-type") ?? "",
    };
  }

  /**
   * `DELETE /v1/documents/{id}`: releases the stored pictures. Not callable from a browser on
   * another origin (DELETE is not in the server's CORS allow-list).
   */
  deleteDocument(documentId: string, opts: CallOptions = {}): Promise<{ object: string; id: string; deleted: boolean }> {
    return this.json("DELETE", `/v1/documents/${encodeURIComponent(documentId)}`, { signal: opts.signal });
  }

  // --- memory --------------------------------------------------------------------------------

  /** `GET /v1/memory/status`: the three tiers and every model's placement. */
  memoryStatus(opts: CallOptions = {}): Promise<MemoryStatus> {
    return this.json("GET", "/v1/memory/status", { signal: opts.signal });
  }

  /** `POST /v1/memory/tier`: promote or demote a model by hand (blocks for the load). */
  setTier(
    model: string,
    targetTier: MemoryTier,
    options: CallOptions & { contextLen?: number } = {},
  ): Promise<TierMove> {
    return this.json("POST", "/v1/memory/tier", {
      json: dropUndefined({ model, target_tier: targetTier, context_len: options.contextLen }),
      signal: options.signal,
    });
  }

  // --- request log ---------------------------------------------------------------------------

  /** `GET /v1/logs`, newest first. */
  logs(query: LogQuery = {}): Promise<LogList> {
    const { signal, ...q } = query;
    return this.json("GET", "/v1/logs", { query: q, signal });
  }

  /** `GET /v1/logs/{id}`: one record by its `x-request-id`. */
  log(requestId: string, opts: CallOptions = {}): Promise<LogRecord> {
    return this.json("GET", `/v1/logs/${encodeURIComponent(requestId)}`, { signal: opts.signal });
  }

  /** `DELETE /v1/logs` (not callable cross-origin from a browser). */
  clearLogs(opts: CallOptions = {}): Promise<{ cleared: boolean }> {
    return this.json("DELETE", "/v1/logs", { signal: opts.signal });
  }

  /** `POST /v1/logs/level`: `off`, `error`, `warn`, `info`, `debug`, `trace`. */
  setLogLevel(level: string, opts: CallOptions = {}): Promise<{ level: string }> {
    return this.json("POST", "/v1/logs/level", { json: { level }, signal: opts.signal });
  }
}

/**
 * Slices `text` by Unicode code point offsets, as the server reports entity `start`/`end`.
 * `text.slice(start, end)` counts UTF-16 units and goes wrong after an astral character.
 */
export function sliceByCodePoints(text: string, start: number, end?: number): string {
  return Array.from(text).slice(start, end).join("");
}
