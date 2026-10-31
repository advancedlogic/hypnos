import { Channel, invoke } from '@tauri-apps/api/core';

/**
 * The lineup of `models.manifest` for a config (DESK-17): `{ modelsDir, dirExists, freeBytes,
 * managed, errors, models: [{ id, modality, inConfig, enabled, size, have, installed, files }] }`.
 */
export const catalog = (config) => invoke('models_catalog', { config });

/**
 * Downloads the missing files of `ids`. `onEvent` gets `{ event: 'start', files, totalBytes }`,
 * `file` (`model`, `local`, `size`, `from`), `progress` (`doneBytes`, `fileBytes`),
 * `verifying` (`local`), `done` and `cancelled`. Resolves when it stops; rejects on an error.
 */
export function download(config, ids, onEvent) {
  const channel = new Channel();
  channel.onmessage = onEvent;
  return invoke('models_download', { config, ids, onEvent: channel });
}

export const cancel = () => invoke('models_cancel');

/** Points the app's own config at `dir`, moving the old folder's files when `moveFiles`. */
export function setDir(config, dir, moveFiles, onEvent) {
  const channel = new Channel();
  channel.onmessage = onEvent;
  return invoke('models_set_dir', { config, dir, moveFiles, onEvent: channel });
}

/** Decimal GB, as disks and downloads are labelled. */
export function gb(bytes) {
  if (bytes == null) return '?';
  const v = bytes / 1e9;
  return v >= 10 ? `${v.toFixed(0)} GB` : v >= 0.1 ? `${v.toFixed(1)} GB` : `${Math.max(1, Math.round(bytes / 1e6))} MB`;
}
