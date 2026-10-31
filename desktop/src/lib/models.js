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

/**
 * The curated catalog's models (API-10, API-11): `{ live, offline, entries }`. `live`: the app is
 * connected to a server that uses this config's models folder, so `entries[].inPool` is its pool
 * and `poolPut` / `poolRemove` change it; else `offline` says why and the pool is read from the
 * config and its overlay. Entries: `{ id, name, publisher, license, modality,
 * additionalModalities, engine, capabilities, summary, lineup, inPool, enabled, currentQuant,
 * variants: [{ quant, default, note, size, have, installed }] }`.
 */
export const catalogEntries = (config) => invoke('models_catalog_entries', { config });

/** The `[[models]]` block of a catalog variant, from `hypnos --catalog-block` of `binary`. */
export const catalogBlock = (binary, id, quant) => invoke('models_catalog_block', { binary, id, quant });

/**
 * Without a live server: downloads a catalog variant (events as `download`) and appends its block
 * to the app's own config; the caller starts the server. Resolves true when added, false when
 * stopped.
 */
export function addFromCatalog(config, binary, id, quant, onEvent) {
  const channel = new Channel();
  channel.onmessage = onEvent;
  return invoke('models_catalog_add', { config, binary, id, quant, onEvent: channel });
}

/**
 * Adds a catalog variant to the running server's pool, or with `replace` puts it in place of the
 * pool's model with that id: downloads its missing files first (events as `download`). Resolves
 * to `{ model, action, persisted }`, or null when the download was stopped. Rejects with
 * `{ kind, message }` (`permission`, `busy`, `bad_request`, ...).
 */
export function poolPut(config, id, quant, replace, onEvent) {
  const channel = new Channel();
  channel.onmessage = onEvent;
  return invoke('models_pool_put', { config, id, quant, replace, onEvent: channel });
}

/** Takes a model out of the running server's pool; its files stay. */
export const poolRemove = (id) => invoke('models_pool_remove', { id });

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
