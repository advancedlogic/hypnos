import { LazyStore } from '@tauri-apps/plugin-store';
import { inTauri } from './hypnos.js';

// Inside Tauri the data lives in the app's data directory (plugin-store); in a plain browser
// (`npm run dev`) it falls back to localStorage.
const store = inTauri ? new LazyStore('hypnos-desktop.json', { autoSave: false }) : null;

export async function load(key, fallback) {
  try {
    if (store) return (await store.get(key)) ?? fallback;
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (e) {
    console.warn(`could not load ${key}`, e);
    return fallback;
  }
}

export async function save(key, value) {
  try {
    if (store) {
      await store.set(key, value);
      await store.save();
    } else {
      localStorage.setItem(key, JSON.stringify(value));
    }
  } catch (e) {
    console.warn(`could not save ${key}`, e);
  }
}
