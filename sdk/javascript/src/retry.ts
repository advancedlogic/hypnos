/** The retry policy of docs/api.md section 17. */

import {
  APIConnectionError,
  APITimeoutError,
  HypnosError,
  InsufficientMemoryError,
  ModelBusyError,
  RateLimitError,
  ServerError,
  ServiceUnavailableError,
} from "./errors.js";

export interface RetryOptions {
  /** Attempts after the first one; 0 turns retrying off. Default 4. */
  maxRetries?: number;
  /** First wait (seconds) after 429 `model_busy`; doubles each attempt. Default 0.25. */
  busyBackoff?: number;
  /** Wait (seconds) after 429/503 without `Retry-After`. Default 5. */
  unavailableWait?: number;
  /** First wait (seconds) after 507; doubles each attempt. Default 2. */
  memoryBackoff?: number;
  /** First wait (seconds) after 504 or a connection error; doubles each attempt. Default 1. */
  backoff?: number;
  /** Upper bound of any single wait (seconds). Default 60. */
  maxWait?: number;
}

/**
 * When and how long to wait before sending a request again. Only failures that arrive before
 * any response body are retried: a stream that produced output is never replayed.
 */
export class RetryPolicy {
  readonly maxRetries: number;
  readonly busyBackoff: number;
  readonly unavailableWait: number;
  readonly memoryBackoff: number;
  readonly backoff: number;
  readonly maxWait: number;

  constructor(options: RetryOptions = {}) {
    this.maxRetries = options.maxRetries ?? 4;
    this.busyBackoff = options.busyBackoff ?? 0.25;
    this.unavailableWait = options.unavailableWait ?? 5;
    this.memoryBackoff = options.memoryBackoff ?? 2;
    this.backoff = options.backoff ?? 1;
    this.maxWait = options.maxWait ?? 60;
  }

  /** Seconds to wait before attempt `attempt + 1` (`attempt` counts from 0), or undefined. */
  delay(error: HypnosError, attempt: number): number | undefined {
    if (attempt >= this.maxRetries) return undefined;
    const retryAfter = (error as { retryAfter?: number }).retryAfter;
    let wait: number;
    if (error instanceof ModelBusyError) {
      wait = this.busyBackoff * 2 ** attempt;
    } else if (error instanceof RateLimitError || error instanceof ServiceUnavailableError) {
      wait = retryAfter ?? this.unavailableWait;
    } else if (error instanceof InsufficientMemoryError) {
      wait = Math.max(retryAfter ?? 0, this.memoryBackoff * 2 ** attempt);
    } else if (error instanceof APITimeoutError && error.status === 504) {
      wait = this.backoff * 2 ** attempt;
    } else if (error instanceof APIConnectionError) {
      wait = this.backoff * 2 ** attempt;
    } else if (error instanceof ServerError && error.status === 500 && attempt === 0) {
      wait = this.backoff;
    } else {
      return undefined;
    }
    // Up to 10 % jitter so that many clients do not come back in lockstep.
    return Math.min(wait * (1 + Math.random() * 0.1), this.maxWait);
  }
}

export const NO_RETRY = new RetryPolicy({ maxRetries: 0 });
