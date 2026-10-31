//! The menu bar / system tray icon (DESK-12): the logo with a dot in the server's colour, a menu
//! (status, open, local server, settings, quit), a left click that shows or hides the window, and
//! a window that hides into the tray on close. The webview owns the server state and the local
//! server logic; it reports the state with `tray_update` and handles the menu's
//! `tray-server` (start or stop the server) and `tray-open-settings` events.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use tauri::image::Image;
use tauri::menu::{Menu, MenuItem, PredefinedMenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIcon, TrayIconBuilder, TrayIconEvent};
use tauri::{AppHandle, Emitter, Manager, Runtime, State, WebviewWindow};

const ICON_PNG: &[u8] = include_bytes!("../icons/128x128.png");
const ICON_SIZE: u32 = 64;

pub struct Tray<R: Runtime> {
    icon: TrayIcon<R>,
    status: MenuItem<R>,
    server: MenuItem<R>,
    state: Mutex<String>,
}

/// Window behaviour chosen in the settings.
pub struct TraySettings {
    /// Closing the window hides it into the tray (default) instead of quitting.
    pub close_to_tray: AtomicBool,
}

impl Default for TraySettings {
    fn default() -> Self {
        Self {
            close_to_tray: AtomicBool::new(true),
        }
    }
}

/// RGBA of the dot for a server state.
fn colour(state: &str) -> [u8; 4] {
    match state {
        "ready" => [52, 211, 153, 255],                       // emerald-400
        "starting" => [251, 191, 36, 255],                    // amber-400
        "unreachable" | "unauthorized" => [244, 63, 94, 255], // rose-500
        _ => [148, 163, 184, 255],                            // slate-400: connecting
    }
}

/// The logo scaled to the tray size with a status dot (white ring) in the bottom-right corner.
pub fn icon_with_dot(state: &str) -> Image<'static> {
    let base = image::load_from_memory(ICON_PNG)
        .expect("bundled icon")
        .resize_exact(ICON_SIZE, ICON_SIZE, image::imageops::FilterType::Triangle)
        .to_rgba8();
    let mut px = base.into_raw();
    let fill = colour(state);
    let (cx, cy) = (ICON_SIZE as f32 - 14.0, ICON_SIZE as f32 - 14.0);
    for y in 0..ICON_SIZE {
        for x in 0..ICON_SIZE {
            let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            let rgba = if d <= 10.0 {
                fill
            } else if d <= 13.0 {
                [255, 255, 255, 255]
            } else {
                continue;
            };
            let i = ((y * ICON_SIZE + x) * 4) as usize;
            px[i..i + 4].copy_from_slice(&rgba);
        }
    }
    Image::new_owned(px, ICON_SIZE, ICON_SIZE)
}

pub fn show_window<R: Runtime>(app: &AppHandle<R>) {
    if let Some(w) = main_window(app) {
        let _ = w.unminimize();
        let _ = w.show();
        let _ = w.set_focus();
    }
}

fn main_window<R: Runtime>(app: &AppHandle<R>) -> Option<WebviewWindow<R>> {
    app.get_webview_window("main")
}

fn toggle_window<R: Runtime>(app: &AppHandle<R>) {
    let Some(w) = main_window(app) else { return };
    if w.is_visible().unwrap_or(false) && w.is_focused().unwrap_or(false) {
        let _ = w.hide();
    } else {
        show_window(app);
    }
}

/// Builds the tray; called once from `setup`.
pub fn create<R: Runtime>(app: &AppHandle<R>) -> tauri::Result<()> {
    let status = MenuItem::with_id(app, "status", "Hypnos: connecting…", false, None::<&str>)?;
    let open = MenuItem::with_id(app, "open", "Open Hypnos", true, None::<&str>)?;
    let server = MenuItem::with_id(app, "server", "Start the server", true, None::<&str>)?;
    let settings = MenuItem::with_id(app, "settings", "Connection settings…", true, None::<&str>)?;
    let quit = MenuItem::with_id(app, "quit", "Quit Hypnos", true, Some("CmdOrCtrl+Q"))?;
    let menu = Menu::with_items(
        app,
        &[
            &status,
            &PredefinedMenuItem::separator(app)?,
            &open,
            &server,
            &settings,
            &PredefinedMenuItem::separator(app)?,
            &quit,
        ],
    )?;
    let icon = TrayIconBuilder::with_id("hypnos")
        .icon(icon_with_dot("connecting"))
        .tooltip("Hypnos")
        .menu(&menu)
        // A left click toggles the window; the menu opens on a right click.
        .show_menu_on_left_click(false)
        .on_menu_event(|app, event| match event.id().as_ref() {
            "open" => show_window(app),
            "server" => {
                let _ = app.emit("tray-server", ());
            }
            "settings" => {
                show_window(app);
                let _ = app.emit("tray-open-settings", ());
            }
            "quit" => app.exit(0),
            _ => {}
        })
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                toggle_window(tray.app_handle());
            }
        })
        .build(app)?;
    app.manage(Tray {
        icon,
        status,
        server,
        state: Mutex::new("connecting".into()),
    });
    Ok(())
}

/// The webview reports the server state (`connecting`, `unreachable`, `unauthorized`,
/// `starting`, `ready`), the status line, and the server item ("Stop the server", "Start the
/// server", or a disabled note for a remote server).
#[tauri::command]
pub fn tray_update(
    tray: State<'_, Tray<tauri::Wry>>,
    state: String,
    label: String,
    server_item: String,
    server_item_enabled: bool,
) -> Result<(), String> {
    let mut current = tray.state.lock().unwrap();
    if *current != state {
        tray.icon
            .set_icon(Some(icon_with_dot(&state)))
            .map_err(|e| e.to_string())?;
        *current = state;
    }
    let _ = tray.icon.set_tooltip(Some(&label));
    tray.status.set_text(&label).map_err(|e| e.to_string())?;
    tray.server
        .set_text(&server_item)
        .map_err(|e| e.to_string())?;
    tray.server
        .set_enabled(server_item_enabled)
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// The settings' window behaviour: keep running in the tray on close, hide the Dock icon (macOS).
#[tauri::command]
pub fn tray_settings<R: Runtime>(
    app: AppHandle<R>,
    settings: State<'_, TraySettings>,
    close_to_tray: bool,
    hide_dock: bool,
) -> Result<(), String> {
    settings
        .close_to_tray
        .store(close_to_tray, Ordering::SeqCst);
    #[cfg(target_os = "macos")]
    app.set_dock_visibility(!hide_dock)
        .map_err(|e| e.to_string())?;
    #[cfg(not(target_os = "macos"))]
    let _ = (app, hide_dock);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dot_takes_the_state_colour() {
        let img = icon_with_dot("ready");
        assert_eq!((img.width(), img.height()), (ICON_SIZE, ICON_SIZE));
        let at = |x: u32, y: u32| {
            let i = ((y * ICON_SIZE + x) * 4) as usize;
            img.rgba()[i..i + 4].to_vec()
        };
        let c = ICON_SIZE - 14;
        assert_eq!(at(c, c), colour("ready").to_vec());
        assert_eq!(at(c + 11, c), vec![255, 255, 255, 255], "white ring");
        assert_eq!(
            icon_with_dot("unreachable").rgba()[((c * ICON_SIZE + c) * 4) as usize],
            244
        );
        assert_ne!(colour("starting"), colour("ready"));
    }
}
