import { Channel, invoke } from '@tauri-apps/api/core';

// The guided setup (DESK-18, src-tauri/src/setup.rs).

/** The uses the wizard offers, as `hypnos advise --needs` names them. */
export const NEEDS = [
  { id: 'chat', label: 'Chat', hint: 'Ask questions, write and summarize' },
  { id: 'vision', label: 'Vision', hint: 'Show it images in the chat' },
  { id: 'voice', label: 'Voice', hint: 'Talk to it and hear the answers' },
  { id: 'documents', label: 'Documents', hint: 'Read PDFs and office files' },
  { id: 'images', label: 'Image generation', hint: 'Make pictures from a description' },
  { id: 'embeddings', label: 'Search', hint: 'Find related text (embeddings)' },
  { id: 'entities', label: 'Entities', hint: 'Pick out names, places and dates' },
];

export const DEFAULT_NEEDS = ['chat', 'vision', 'voice'];

/**
 * The recommended lineup for `needs` on this machine: `{ ramBytes, budgets: { tier1_bytes,
 * tier2_bytes, unified_memory }, models: [{ id, name, quant, spec, role, serves, sizeBytes,
 * missingBytes, installed, keptLoaded, runsOnCpu, partial, smaller, loadSecs }], together,
 * warnings, totalBytes, downloadBytes, freeBytes, modelsDir, managed, specs, config }`.
 */
export const advise = (binary, config, needs) => invoke('setup_advise', { binary, config, needs });

/** Downloads the specs' missing files (events as `models.download`); resolves false when stopped. */
export function install(config, specs, onEvent) {
  const channel = new Channel();
  channel.onmessage = onEvent;
  return invoke('setup_install', { config, specs, onEvent: channel });
}

/** The app's own config gets the advised models: `{ backup, overlayBackup, models }`. */
export const applyConfig = (config, advisedConfig) => invoke('setup_apply_config', { config, advisedConfig });

/** Adds the specs a running server lacks: `[{ spec, id, action: 'added' | 'present' | 'failed', message }]`. */
export const applyLive = (config, specs) => invoke('setup_apply_live', { config, specs });

/**
 * Speaks `text` with the text-to-speech model and transcribes it back: `{ spoken, heard,
 * speechModel, listenModel, speakMs, listenMs, audioSecs, share, audio (a data URL) }`.
 */
export const voiceCheck = (text) => invoke('setup_voice_check', { text });

/** Memory in binary GB, as RAM is sold ("32 GB"). */
export function mem(bytes) {
  if (bytes == null) return '?';
  const v = bytes / 2 ** 30;
  return `${v >= 10 ? Math.round(v) : v.toFixed(1)} GB`;
}
