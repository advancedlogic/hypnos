import {
  APIConnectionError,
  AuthenticationError,
  InsufficientMemoryError,
  ModelBusyError,
  NotFoundError,
  RateLimitError,
  ServiceUnavailableError,
  StreamError,
} from '@hypnos/client';

/** A sentence for the user instead of the raw `HTTP 429 ...` message. */
export function describeError(e) {
  const after = e?.retryAfter ? ` Try again in ${e.retryAfter} s.` : '';
  if (e instanceof ModelBusyError) return `The model's queue is full.${after || ' Try again in a moment.'}`;
  if (e instanceof RateLimitError) return `This API key hit its limit.${after}`;
  if (e instanceof ServiceUnavailableError) return 'Hypnos is still loading its models. Try again when the status turns Ready.';
  if (e instanceof InsufficientMemoryError) return 'Not enough memory to load this model right now (other models are in use).';
  if (e instanceof NotFoundError) return 'This model is not available (disabled or removed from the config). Pick another one.';
  if (e instanceof AuthenticationError) return 'The server refused the API key.';
  if (e instanceof APIConnectionError) return 'Lost the connection to Hypnos.';
  if (e instanceof StreamError) return `The model stopped with an error: ${e.message}`;
  return String(e?.message ?? e);
}
