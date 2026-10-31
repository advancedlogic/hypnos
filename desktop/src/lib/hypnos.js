import { AuthenticationError, Client } from '@hypnos/client';
import { fetch as tauriFetch } from '@tauri-apps/plugin-http';

export const DEFAULT_SERVER_URL = 'http://localhost:9999';

/** True inside the Tauri webview, false in a plain browser (`npm run dev`). */
export const inTauri = typeof window !== 'undefined' && '__TAURI_INTERNALS__' in window;

/**
 * A client for the Hypnos server. Inside Tauri the requests go through the Rust HTTP plugin:
 * the webview's origin (`tauri://localhost`) is not a loopback origin, so the server's default
 * CORS policy would refuse it. In a browser the global fetch is used.
 */
export function createClient({ baseUrl = DEFAULT_SERVER_URL, apiKey } = {}) {
  return new Client({ baseUrl, apiKey, fetch: inTauri ? tauriFetch : undefined });
}

/**
 * One look at the server: `unreachable`, `unauthorized` (the API key is missing or refused),
 * `starting` (up, models still loading) or `ready`. `/health` is open to everyone and
 * `GET /v1/models` checks the key even during startup, so each state is told apart.
 */
export async function probe(client, signal) {
  let health;
  try {
    health = await client.health({ signal });
  } catch (e) {
    if (signal?.aborted) throw e;
    return { state: 'unreachable', health: null, error: String(e?.message ?? e) };
  }
  try {
    await client.models({ signal });
  } catch (e) {
    if (signal?.aborted) throw e;
    if (e instanceof AuthenticationError) return { state: 'unauthorized', health };
  }
  const ready = await client.isReady({ signal });
  return { state: ready ? 'ready' : 'starting', health };
}
