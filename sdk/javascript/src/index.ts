/**
 * JavaScript/TypeScript client for the Hypnos inference server
 * (https://github.com/advancedlogic/hypnos). The API is described in docs/api.md.
 */

export { Client, DEFAULT_BASE_URL, VERSION, sliceByCodePoints } from "./client.js";
export type {
  CallOptions,
  CatalogQuery,
  ChatOptions,
  ClientOptions,
  DocumentOptions,
  EntityOptions,
  FileInput,
  ImageOptions,
  LogQuery,
  SpeechOptions,
  TranscribeOptions,
} from "./client.js";
export * from "./errors.js";
export { NO_RETRY, RetryPolicy } from "./retry.js";
export type { RetryOptions } from "./retry.js";
export { ChatStream, DocumentStream, SSEDecoder, SpeechStream } from "./streams.js";
export { assistantMessage, parseArguments, toolMessage } from "./tools.js";
export type * from "./types.js";
