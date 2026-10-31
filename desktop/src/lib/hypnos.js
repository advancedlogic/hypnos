import { Channel, invoke } from '@tauri-apps/api/core';

// Every server call is a command of the Rust side (src-tauri/src/hypnos.rs, on the Rust SDK).
// A failed command rejects with `{ kind, message, detail, retryAfterSecs }`: `message` is a
// sentence for the user, `kind` what to branch on (`busy`, `starting`, `unauthorized`, ...).

export const DEFAULT_SERVER_URL = 'http://localhost:9999';

/** Uses this server from now on, with the API key saved for it in the keychain. */
export const connect = (url) => invoke('hypnos_connect', { url });

/** `{ state, health }`: `unreachable`, `unauthorized`, `starting` (models loading) or `ready`. */
export const probe = () => invoke('hypnos_probe');

/** Like `probe`, for a server and key not in use yet (the settings' Test connection). */
export const test = (url, apiKey) => invoke('hypnos_test', { url, apiKey: apiKey || null });

export const memoryStatus = () => invoke('hypnos_memory_status');

/** The enabled models with all their `modalities` and their tier. */
export const models = () => invoke('hypnos_models');

/** Modality -> model id chosen in the settings; absent modalities are routed automatically. */
export const setRoutes = (routes) => invoke('hypnos_set_routes', { routes });

/** The model that would serve `modality` now, preferring `preferred` when it can. */
export const route = (modality, preferred) => invoke('hypnos_route', { modality, preferred: preferred ?? null });

/**
 * Streams an answer: `onModel(id, reason)` when another model answers (images for a model that
 * cannot see them), `onDelta(text)` per piece, `onDone({ telemetry, usage, finishReason })` at
 * the end. Resolves to `{ cancelled }`; `cancelChat(requestId)` stops it.
 */
export function chat(requestId, model, messages, { onModel, onDelta, onDone }) {
  const onEvent = new Channel();
  onEvent.onmessage = (e) => {
    if (e.event === 'model') onModel?.(e.id, e.reason);
    else if (e.event === 'delta') onDelta(e.text);
    else if (e.event === 'done') onDone(e);
  };
  return invoke('hypnos_chat', { requestId, model, messages, onEvent });
}

export const cancelChat = (requestId) => invoke('hypnos_chat_cancel', { requestId });

/** The sentence to show for a rejected command (or any other error). */
export function errorMessage(e) {
  if (e?.detail) console.warn(e.detail);
  return e?.message ?? String(e);
}

/**
 * Prepares a file for a message (src-tauri/src/attachments.rs): `{ kind: 'image' | 'document' |
 * 'audio' | 'video', name, size, ... }`. Documents are converted and audio transcribed here, so
 * this can take a while (and load a model).
 */
export const attachFromPath = (path, chatModel) => invoke('attachment_from_path', { path, chatModel: chatModel ?? null });

/** The same for bytes pasted into the composer. */
export function attachFromBytes(name, bytes, chatModel) {
  return invoke('attachment_from_bytes', bytes, { headers: { 'x-name': name, ...(chatModel ? { 'x-chat-model': chatModel } : {}) } });
}

/** Fetches a URL on the Rust side and converts it: a `{ kind: 'web', name (title), url, text }` attachment. */
export const readUrl = (url) => invoke('url_extract', { url });

/** Writes text to a path chosen in the save dialog. */
export const saveText = (path, text) => invoke('save_text', { path, text });

/** Downloads the original file of a URL again into a path chosen in the save dialog. */
export const downloadUrl = (url, path) => invoke('url_download', { url, path });

/** Starts recording from the default microphone (src-tauri/src/mic.rs). */
export const micStart = () => invoke('mic_start');

/** Stops and transcribes: `{ text, seconds, model }`. */
export const micStop = (chatModel) => invoke('mic_stop', { chatModel: chatModel ?? null });

/** Stops and discards the recording. */
export const micCancel = () => invoke('mic_cancel');

/**
 * Reads an answer aloud (src-tauri/src/tts.rs): call with the text so far while it streams and
 * with `done` at the end; another `id` replaces the current answer. `speech-state` events carry
 * `{ id, playing, error }`.
 */
export const speak = (id, text, done, { voice, speed } = {}) =>
  invoke('speak', { id, text, done, voice: voice || null, speed: speed ?? null });

export const speakStop = () => invoke('speak_stop');

/** Voice and speed for reading aloud, kept on this machine. */
export function voicePrefs() {
  try {
    return { voice: localStorage.getItem('hypnos.desktop.voice') || '', speed: Number(localStorage.getItem('hypnos.desktop.speed')) || 1 };
  } catch {
    return { voice: '', speed: 1 };
  }
}

export function setVoicePrefs({ voice, speed }) {
  try {
    localStorage.setItem('hypnos.desktop.voice', voice ?? '');
    localStorage.setItem('hypnos.desktop.speed', String(speed ?? 1));
  } catch {}
}
