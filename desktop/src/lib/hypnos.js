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

/** Thinking levels for `chat` (`reasoning_effort`); `none` is off. */
export const THINKING_LEVELS = [
  { value: 'none', label: 'Off' },
  { value: 'low', label: 'Low' },
  { value: 'medium', label: 'Medium' },
  { value: 'high', label: 'High' },
];

/**
 * Streams an answer: `onModel(id, reason)` when another model answers (images for a model that
 * cannot see them), `onReasoning(text)` per piece of the model's thoughts (with `thinking` other
 * than `none`, before the answer), `onDelta(text)` per piece, `onDone({ telemetry, usage, finishReason })` at
 * the end. With `tools`, the model may ask for an image: `onToolCall({ prompt, size, negativePrompt,
 * n, steps, seed, reference })`, then `onImage({ dataUrl, prompt, size, model, seconds, seed,
 * negativePrompt, steps, reference, more: [{ dataUrl, seed }], note })`, then its comment as
 * deltas. With `assistant`, the other tools of the registry (DESK-23: web pages, reading aloud,
 * entities, describing an image) are offered too: each call arrives as `onStep({ id, tool, label,
 * status, model, seconds, summary })` (`running`, then `done` or `error` with the same id),
 * `onSpeak(text)` asks to read a text aloud, `onToolError(message)` reports a failed image
 * call. Resolves to
 * `{ cancelled }`; `cancelChat(requestId)` stops it.
 */
export function chat(
  requestId,
  model,
  messages,
  { onModel, onReasoning, onDelta, onDone, onToolCall, onImage, onStep, onSpeak, onToolError, tools = false, assistant = false, thinking = 'none' },
) {
  const onEvent = new Channel();
  onEvent.onmessage = (e) => {
    if (e.event === 'model') onModel?.(e.id, e.reason);
    else if (e.event === 'toolCall') onToolCall?.(e);
    else if (e.event === 'image') onImage?.(e);
    else if (e.event === 'step') onStep?.(e);
    else if (e.event === 'speak') onSpeak?.(e.text);
    else if (e.event === 'toolError') onToolError?.(e.message);
    else if (e.event === 'reasoning') onReasoning?.(e.text);
    else if (e.event === 'delta') onDelta(e.text);
    else if (e.event === 'done') onDone(e);
  };
  return invoke('hypnos_chat', { requestId, model, messages, tools, assistant, reasoningEffort: thinking, onEvent });
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

/** Generates an image without the chat model (the Image button, `/image`). */
/**
 * One image straight from the image model (the Image button, `/image`, Regenerate). `settings`:
 * `{ size, negativePrompt, n, steps, seed }`, each optional; resolves to what `onImage` gets.
 */
export const generateImage = (prompt, settings = {}) =>
  invoke('hypnos_generate_image', { call: { prompt, size: settings.size ?? '1024x1024', ...withoutEmpty(settings) } });

const withoutEmpty = (o) => Object.fromEntries(Object.entries(o).filter(([, v]) => v !== undefined && v !== null && v !== ''));

/** Writes a data URL (a generated image) to a path chosen in the save dialog. */
export const saveDataUrl = (path, dataUrl) => invoke('save_data_url', { path, dataUrl });

/** Whether the chat model may create images (on unless turned off in the settings). */
export function imageToolOn() {
  try {
    return localStorage.getItem('hypnos.desktop.imageTool') !== 'false';
  } catch {
    return true;
  }
}

export function setImageTool(on) {
  try {
    localStorage.setItem('hypnos.desktop.imageTool', String(on));
  } catch {}
}

/** Whether the chat model may use the other tools (DESK-23; on unless turned off in the settings). */
export function assistantToolsOn() {
  try {
    return localStorage.getItem('hypnos.desktop.assistantTools') !== 'false';
  } catch {
    return true;
  }
}

export function setAssistantTools(on) {
  try {
    localStorage.setItem('hypnos.desktop.assistantTools', String(on));
  } catch {}
}

/** The request log (DESK-13): `{ level, capacity, stored, records }`, newest first. */
export const logs = (filter) => invoke('hypnos_logs', { filter });

/** One record by its request id, with the captured text and payloads at debug / trace. */
export const log = (id) => invoke('hypnos_log', { id });

export const setLogLevel = (level) => invoke('hypnos_set_log_level', { level });

export const clearLogs = () => invoke('hypnos_clear_logs');

/** KV parking (MEM-4): `{ enabled, max_bytes, used_bytes, models }`; also `kv_park` in the memory status. */
export const setKvPark = ({ enabled, maxBytes } = {}) =>
  invoke('hypnos_set_kv_park', { enabled: enabled ?? null, maxBytes: maxBytes ?? null });

/** MoE expert offload (MEM-5): `mode` is `off`, `auto` or `on`; answers `{ applies: 'now' | 'next_load', placement }`. */
export const setMoeOffload = (model, mode) => invoke('hypnos_set_moe_offload', { model, mode });
