import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { load, save } from './storage.js';

const SETTINGS_KEY = 'localServer';
export const DEFAULT_PORT = 9999;

/**
 * `{ binary, config, port, autoStart }`, filled from detection the first time. The server
 * bundled in the macOS app (DESK-15) starts with the app by default, as Ollama's does.
 */
export async function loadLocalSettings() {
  const saved = await load(SETTINGS_KEY, null);
  if (saved?.binary) return saved;
  let found = {};
  try {
    found = await invoke('local_server_detect');
  } catch (e) {
    console.warn('local server detection failed', e);
  }
  return {
    binary: found.binary ?? '',
    config: found.config ?? '',
    port: DEFAULT_PORT,
    autoStart: found.source === 'bundled',
    ...saved,
  };
}

export function saveLocalSettings(settings) {
  return save(SETTINGS_KEY, settings);
}

export const detect = () => invoke('local_server_detect');
export const status = () => invoke('local_server_status');
export const logs = () => invoke('local_server_logs');
export const stop = () => invoke('local_server_stop');
export const probe = (port) => invoke('local_server_probe', { port });

export function start({ binary, config, port }) {
  return invoke('local_server_start', { options: { binary, config: config || null, port: Number(port) } });
}

/** Calls `onLine` for each output line and `onExit` when the process ends by itself. */
export async function subscribe(onLine, onExit) {
  const unlisten = await Promise.all([
    listen('local-server-log', (e) => onLine(e.payload)),
    listen('local-server-exit', (e) => onExit(e.payload)),
  ]);
  return () => unlisten.forEach((u) => u());
}

export function localUrl(port) {
  return `http://127.0.0.1:${port}`;
}
