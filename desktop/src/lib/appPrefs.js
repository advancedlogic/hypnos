import { invoke } from '@tauri-apps/api/core';
import { isPermissionGranted, requestPermission, sendNotification } from '@tauri-apps/plugin-notification';

// Window and notification preferences (DESK-12), kept on this machine.
const KEYS = { closeToTray: 'hypnos.desktop.closeToTray', hideDock: 'hypnos.desktop.hideDock', notify: 'hypnos.desktop.notify' };
const DEFAULTS = { closeToTray: true, hideDock: false, notify: true };

export const isMac = navigator.userAgent.includes('Mac');

export function appPrefs() {
  const prefs = { ...DEFAULTS };
  try {
    for (const [k, key] of Object.entries(KEYS)) {
      const v = localStorage.getItem(key);
      if (v !== null) prefs[k] = v === 'true';
    }
  } catch {}
  return prefs;
}

/** Saves the preferences and applies the window ones on the Rust side. */
export async function applyAppPrefs(prefs = appPrefs()) {
  try {
    for (const [k, key] of Object.entries(KEYS)) localStorage.setItem(key, String(prefs[k]));
  } catch {}
  await invoke('tray_settings', { closeToTray: prefs.closeToTray, hideDock: isMac && prefs.hideDock });
  if (prefs.notify && !(await isPermissionGranted())) await requestPermission();
}

/** A system notification, only while the window is hidden or in the background. */
export async function notifyIfAway(title, body) {
  if (!appPrefs().notify || (!document.hidden && document.hasFocus())) return;
  try {
    if (!(await isPermissionGranted()) && (await requestPermission()) !== 'granted') return;
    sendNotification({ title, body: body.length > 140 ? `${body.slice(0, 139)}…` : body });
  } catch (e) {
    console.warn('notification failed', e);
  }
}

/** Tells the tray icon the server state and what the server item should say. */
export function updateTray({ state, label, serverItem, serverItemEnabled }) {
  return invoke('tray_update', { state, label, serverItem, serverItemEnabled }).catch((e) => console.warn('tray', e));
}
