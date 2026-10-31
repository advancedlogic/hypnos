import { invoke } from '@tauri-apps/api/core';
import { DEFAULT_SERVER_URL, inTauri } from './hypnos.js';
import { load, save } from './storage.js';

const URL_KEY = 'serverUrl';

// Outside Tauri there is no keychain: the key lives in memory for the session only.
let browserKey;

/** `http://host:port` without a trailing slash or `/v1`; throws on anything else. */
export function normalizeUrl(input) {
  const raw = input.trim();
  let url;
  try {
    url = new URL(raw);
  } catch {
    throw new Error('Enter a full URL, such as http://localhost:9999');
  }
  if (url.protocol !== 'http:' && url.protocol !== 'https:') throw new Error('The URL must start with http:// or https://');
  return raw.replace(/\/+$/, '').replace(/\/v1$/, '');
}

/** Lets the Tauri HTTP client reach this server (loopback is always allowed). */
export async function allowServer(url) {
  if (inTauri) await invoke('allow_server', { serverUrl: url });
}

/** The saved server URL and its API key from the OS keychain. */
export async function loadConnection() {
  return connectionFor(await load(URL_KEY, DEFAULT_SERVER_URL));
}

/** `{ url, apiKey }` for a server, with the key saved for it (if any). */
export async function connectionFor(url) {
  let apiKey;
  try {
    apiKey = inTauri ? await invoke('get_api_key', { serverUrl: url }) : browserKey;
  } catch (e) {
    console.warn('could not read the API key from the keychain', e);
  }
  await allowServer(url);
  return { url, apiKey: apiKey ?? '' };
}

export async function saveConnection({ url, apiKey }) {
  if (inTauri) await invoke('set_api_key', { serverUrl: url, apiKey: apiKey || null });
  else browserKey = apiKey;
  await allowServer(url);
  await save(URL_KEY, url);
}
