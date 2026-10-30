/**
 * Streaming responses: chat (SSE), document conversion (SSE) and speech (chunked audio).
 *
 * Each stream holds the HTTP response open; `close()` (or `break` out of a `for await`) aborts
 * the fetch, which is how the server is told to stop (docs/api.md section 17, "Cancellation").
 * A stream that fails after it started throws `StreamError` from the iteration. Iterating a
 * stream that is already finished yields nothing; `final()` then returns what was collected.
 */

import { APIConnectionError, StreamError } from "./errors.js";
import type {
  ChatCompletion,
  ChatCompletionChunk,
  Document,
  DocumentEvent,
  DocumentPage,
  DocumentStarted,
  SpeechTelemetry,
  Telemetry,
  Usage,
} from "./types.js";

/** Feeds lines of a `text/event-stream` body and returns each event's `data`. */
export class SSEDecoder {
  private data: string[] = [];

  feed(rawLine: string): string | undefined {
    const line = rawLine.endsWith("\r") ? rawLine.slice(0, -1) : rawLine;
    if (line === "") {
      if (this.data.length === 0) return undefined;
      const data = this.data.join("\n");
      this.data = [];
      return data;
    }
    if (line.startsWith(":")) return undefined;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "data") this.data.push(value);
    return undefined;
  }

  flush(): string | undefined {
    return this.feed("");
  }
}

async function* lines(chunks: AsyncIterable<Uint8Array>): AsyncGenerator<string> {
  const decoder = new TextDecoder();
  let buffer = "";
  for await (const chunk of chunks) {
    buffer += decoder.decode(chunk, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      yield buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
    }
  }
  buffer += decoder.decode();
  if (buffer) yield buffer;
}

export function speechTelemetry(headers: Headers): SpeechTelemetry {
  const num = (name: string): number | undefined => {
    const v = headers.get(name);
    if (v === null) return undefined;
    const n = Number(v);
    return Number.isFinite(n) ? n : undefined;
  };
  return {
    previousTier: headers.get("x-hypnos-previous-tier") ?? undefined,
    gpuLayers: num("x-hypnos-gpu-layers"),
    switchMs: num("x-hypnos-switch-ms"),
    inferenceMs: num("x-hypnos-inference-ms"),
    audioSecs: num("x-hypnos-audio-secs"),
    rtf: num("x-hypnos-rtf"),
  };
}

export function streamError(payload: Record<string, unknown>, requestId?: string): StreamError {
  const err = (payload.error ?? {}) as { message?: string; type?: string };
  return new StreamError(err.message ?? "the stream failed", {
    status: 200,
    type: err.type ?? "server_error",
    requestId,
    body: payload,
  });
}

/** What every stream shares: the open response, its abort controller and the model permit. */
abstract class BaseStream {
  readonly response: Response;
  readonly requestId?: string;
  protected closed = false;

  constructor(
    response: Response,
    private readonly controller: AbortController,
    private readonly release: () => void,
  ) {
    this.response = response;
    this.requestId = response.headers.get("x-request-id") ?? undefined;
  }

  /** Closes the connection; the server stops at the next token, page or chunk. */
  close(): void {
    if (this.closed) return;
    this.closed = true;
    this.controller.abort();
    this.release();
  }

  /** The body's chunks; a break, an abort or the end closes the stream. */
  protected async *chunks(): AsyncGenerator<Uint8Array> {
    if (this.closed || !this.response.body) return;
    const reader = this.response.body.getReader();
    try {
      for (;;) {
        let result: ReadableStreamReadResult<Uint8Array>;
        try {
          result = await reader.read();
        } catch (e) {
          if (this.closed) return;
          throw new APIConnectionError(`the stream broke: ${String(e)}`);
        }
        if (result.done) return;
        yield result.value;
      }
    } finally {
      reader.releaseLock();
      this.close();
    }
  }

  /** The `data` of each SSE event. */
  protected async *events(): AsyncGenerator<string> {
    if (this.closed) return;
    const decoder = new SSEDecoder();
    for await (const line of lines(this.chunks())) {
      const data = decoder.feed(line);
      if (data !== undefined) yield data;
    }
    const tail = decoder.flush();
    if (tail !== undefined) yield tail;
  }
}

/**
 * `POST /v1/chat/completions` with `stream: true`. Iterate it for chunks, or `text()` for the
 * content pieces; `textSoFar`, `usage`, `telemetry` and `finishReason` fill in as they arrive.
 */
export class ChatStream extends BaseStream implements AsyncIterable<ChatCompletionChunk> {
  textSoFar = "";
  usage?: Usage;
  telemetry?: Telemetry;
  finishReason?: string;
  private id?: string;
  private model?: string;
  private created?: number;

  async *[Symbol.asyncIterator](): AsyncGenerator<ChatCompletionChunk> {
    try {
      for await (const data of this.events()) {
        if (data === "[DONE]") break;
        const chunk = JSON.parse(data) as ChatCompletionChunk;
        this.id ??= chunk.id;
        this.model ??= chunk.model;
        this.created ??= chunk.created;
        for (const choice of chunk.choices ?? []) {
          this.textSoFar += choice.delta?.content ?? "";
          if (choice.finish_reason) this.finishReason = choice.finish_reason;
        }
        if (chunk.usage) this.usage = chunk.usage;
        if (chunk.hypnos_telemetry) this.telemetry = chunk.hypnos_telemetry;
        if (this.finishReason === "error" || chunk.error) {
          throw streamError(chunk as unknown as Record<string, unknown>, this.requestId);
        }
        yield chunk;
      }
    } finally {
      this.close();
    }
  }

  /** The `delta.content` pieces, as they arrive. */
  async *text(): AsyncGenerator<string> {
    for await (const chunk of this) {
      for (const choice of chunk.choices ?? []) {
        if (choice.delta?.content) yield choice.delta.content;
      }
    }
  }

  /** Reads the rest of the stream and returns it as a non-streaming completion. */
  async final(): Promise<ChatCompletion> {
    for await (const _ of this) {
      // drain
    }
    const completion: ChatCompletion = {
      id: this.id ?? "",
      object: "chat.completion",
      created: this.created ?? 0,
      model: this.model ?? "",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: this.textSoFar },
          finish_reason: this.finishReason ?? "stop",
        },
      ],
      usage: this.usage ?? { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };
    if (this.telemetry) completion.hypnos_telemetry = this.telemetry;
    return completion;
  }
}

/**
 * `POST /v1/documents/convert` with `stream: true`. Iterating yields every event
 * (`document.started`, `document.page`, `document`); `pages()` only the pages; `final()` the
 * whole document with its pages put back (the server's last event leaves them out).
 */
export class DocumentStream extends BaseStream implements AsyncIterable<DocumentEvent> {
  started?: DocumentStarted;
  readonly collectedPages: DocumentPage[] = [];
  private document?: Document;

  async *[Symbol.asyncIterator](): AsyncGenerator<DocumentEvent> {
    try {
      for await (const data of this.events()) {
        if (data === "[DONE]") break;
        const event = JSON.parse(data) as Record<string, unknown>;
        switch (event.object) {
          case "document.started":
            this.started = event as unknown as DocumentStarted;
            break;
          case "document.page": {
            const { object: _, ...page } = event;
            this.collectedPages.push(page as unknown as DocumentPage);
            break;
          }
          case "document": {
            const doc = { ...event } as unknown as Document;
            if (!doc.pages || doc.pages.length === 0) doc.pages = [...this.collectedPages];
            this.document = doc;
            break;
          }
          case "error":
            throw streamError(event, this.requestId);
        }
        yield event as unknown as DocumentEvent;
      }
    } finally {
      this.close();
    }
  }

  async *pages(): AsyncGenerator<DocumentPage> {
    for await (const event of this) {
      if (event.object === "document.page") {
        yield this.collectedPages[this.collectedPages.length - 1];
      }
    }
  }

  async final(): Promise<Document> {
    for await (const _ of this) {
      // drain
    }
    if (!this.document) {
      throw new StreamError("the document stream ended without its final event", {
        status: 200,
        requestId: this.requestId,
      });
    }
    return this.document;
  }
}

/**
 * `POST /v1/audio/speech` with `stream: true`: the audio as it is synthesized. For `wav` the
 * first chunk starts with a header that declares an open-ended length; for `pcm` the chunks are
 * raw 16-bit little-endian mono samples at 24 kHz.
 */
export class SpeechStream extends BaseStream implements AsyncIterable<Uint8Array> {
  readonly contentType: string;
  readonly telemetry: SpeechTelemetry;

  constructor(response: Response, controller: AbortController, release: () => void) {
    super(response, controller, release);
    this.contentType = response.headers.get("content-type") ?? "";
    this.telemetry = speechTelemetry(response.headers);
  }

  [Symbol.asyncIterator](): AsyncGenerator<Uint8Array> {
    return this.chunks();
  }

  /** The whole remaining audio. */
  async read(): Promise<Uint8Array> {
    const parts: Uint8Array[] = [];
    for await (const chunk of this) parts.push(chunk);
    const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
    let at = 0;
    for (const p of parts) {
      out.set(p, at);
      at += p.length;
    }
    return out;
  }
}
