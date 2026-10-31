/** Errors thrown by the client, one per error `type` of docs/api.md section 4. */

export class HypnosError extends Error {
  constructor(message: string) {
    super(message);
    this.name = new.target.name;
  }
}

/** The server could not be reached, or the connection broke before a response arrived. */
export class APIConnectionError extends HypnosError {}

export interface APIErrorInit {
  status: number;
  type?: string;
  code?: string;
  requestId?: string;
  retryAfter?: number;
  body?: unknown;
}

/** The server answered with an error status (or failed a stream it had started). */
export class APIError extends HypnosError {
  readonly status: number;
  readonly type?: string;
  readonly code?: string;
  readonly requestId?: string;
  /** Seconds, from the `Retry-After` header. */
  readonly retryAfter?: number;
  readonly body?: unknown;

  constructor(message: string, init: APIErrorInit) {
    const kind = init.type ? ` ${init.type}` : "";
    const rid = init.requestId ? ` (request ${init.requestId})` : "";
    super(`HTTP ${init.status}${kind}: ${message}${rid}`);
    this.status = init.status;
    this.type = init.type;
    this.code = init.code;
    this.requestId = init.requestId;
    this.retryAfter = init.retryAfter;
    this.body = init.body;
  }
}

/** 400 `invalid_request_error`, or 413 (body above `max_body_bytes`). */
export class BadRequestError extends APIError {}
/** 401: missing or unknown API key. */
export class AuthenticationError extends APIError {}
/** 403 `permission_denied`: changing the model pool with a key that is not an admin key. */
export class PermissionDeniedError extends APIError {}
/** 404 `model_not_found`, `model_disabled` or `not_found`. */
export class NotFoundError extends APIError {}
/** 429 `model_busy`: the model's queue is full, or a placement change met a leased model. */
export class ModelBusyError extends APIError {}
/** 429 `rate_limit_error`: a per-key limit; `code` says which, `retryAfter` when. */
export class RateLimitError extends APIError {}
/** 503 `model_not_available`: still staging the startup models, or a configuration problem. */
export class ServiceUnavailableError extends APIError {}
/** 507: the model does not fit next to the leased ones, or the engine ran out of memory. */
export class InsufficientMemoryError extends APIError {}
/** 408 (no response headers in time) or 504 `timeout`. */
export class APITimeoutError extends APIError {}
/** 500 `server_error` (an engine failure) or any other 5xx. */
export class ServerError extends APIError {}
/** A stream that had already started (HTTP 200) ended with an error event. */
export class StreamError extends APIError {}

export function errorClass(status: number, type?: string): typeof APIError {
  if (status === 400 || status === 413) return BadRequestError;
  if (status === 401) return AuthenticationError;
  if (status === 403) return PermissionDeniedError;
  if (status === 404) return NotFoundError;
  if (status === 429) return type === "rate_limit_error" ? RateLimitError : ModelBusyError;
  if (status === 503) return ServiceUnavailableError;
  if (status === 507) return InsufficientMemoryError;
  if (status === 408 || status === 504) return APITimeoutError;
  if (status >= 500) return ServerError;
  return APIError;
}
