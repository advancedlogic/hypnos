import { LazyStore } from '@tauri-apps/plugin-store';

// Chats and settings live in the app's data directory (tauri-plugin-store).
const store = new LazyStore('hypnos-desktop.json', { autoSave: false });

export async function load(key, fallback) {
  try {
    return (await store.get(key)) ?? fallback;
  } catch (e) {
    console.warn(`could not load ${key}`, e);
    return fallback;
  }
}

export async function save(key, value) {
  try {
    await store.set(key, value);
    await store.save();
  } catch (e) {
    console.warn(`could not save ${key}`, e);
  }
}
