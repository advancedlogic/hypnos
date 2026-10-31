import { invoke } from '@tauri-apps/api/core';
import { DEFAULT_SERVER_URL, connect } from './hypnos.js';
import { load, save } from './storage.js';

const URL_KEY = 'serverUrl';

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

/** The saved server URL and its API key from the OS keychain. */
export async function loadConnection() {
  return connectionFor(await load(URL_KEY, DEFAULT_SERVER_URL));
}

/** `{ url, apiKey }` for a server, with the key saved for it (if any). */
export async function connectionFor(url) {
  let apiKey;
  try {
    apiKey = await invoke('get_api_key', { serverUrl: url });
  } catch (e) {
    console.warn('could not read the API key from the keychain', e);
  }
  return { url, apiKey: apiKey ?? '' };
}

/** Saves the URL and the key, then points the Rust client at them. */
export async function saveConnection({ url, apiKey }) {
  await invoke('set_api_key', { serverUrl: url, apiKey: apiKey || null });
  await save(URL_KEY, url);
  await connect(url);
}
