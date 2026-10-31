use std::collections::HashSet;
use std::sync::Mutex;

use serde_json::json;
use tauri::ipc::CapabilityBuilder;
use tauri::{Manager, State};

/// The keychain service name; the account is the server's origin, so each server keeps its key.
const KEYCHAIN_SERVICE: &str = "dev.hypnos.desktop";

/// Origins the HTTP plugin may reach beyond the loopback ones of `capabilities/default.json`.
#[derive(Default)]
struct AllowedOrigins(Mutex<HashSet<String>>);

/// `http://host:port` of a server URL; only http and https are accepted.
fn origin_of(server_url: &str) -> Result<String, String> {
    let url = url::Url::parse(server_url).map_err(|e| format!("invalid server URL: {e}"))?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        return Err("the server URL must start with http:// or https://".into());
    }
    Ok(url.origin().ascii_serialization())
}

fn keychain_entry(server_url: &str) -> Result<keyring::Entry, String> {
    keyring::Entry::new(KEYCHAIN_SERVICE, &origin_of(server_url)?).map_err(|e| e.to_string())
}

/// Lets the main window's HTTP client reach `server_url`'s origin. Capabilities cannot be
/// removed at runtime, so an origin used earlier stays reachable until the app restarts.
#[tauri::command]
fn allow_server(
    app: tauri::AppHandle,
    allowed: State<AllowedOrigins>,
    server_url: String,
) -> Result<(), String> {
    let origin = origin_of(&server_url)?;
    let mut origins = allowed.0.lock().unwrap();
    if origins.contains(&origin) {
        return Ok(());
    }
    let capability = CapabilityBuilder::new(format!("server-{}", origins.len()))
        .window("main")
        .permission_scoped(
            "http:default",
            vec![json!({ "url": format!("{origin}/*") })],
            Vec::<serde_json::Value>::new(),
        );
    app.add_capability(capability).map_err(|e| e.to_string())?;
    origins.insert(origin);
    Ok(())
}

/// The API key saved for this server in the OS keychain, if any.
#[tauri::command]
fn get_api_key(server_url: String) -> Result<Option<String>, String> {
    match keychain_entry(&server_url)?.get_password() {
        Ok(key) => Ok(Some(key)),
        Err(keyring::Error::NoEntry) => Ok(None),
        Err(e) => Err(e.to_string()),
    }
}

/// Saves (or, with no key, forgets) the API key for this server in the OS keychain.
#[tauri::command]
fn set_api_key(server_url: String, api_key: Option<String>) -> Result<(), String> {
    let entry = keychain_entry(&server_url)?;
    match api_key.filter(|k| !k.is_empty()) {
        Some(key) => entry.set_password(&key).map_err(|e| e.to_string()),
        None => match entry.delete_credential() {
            Ok(()) | Err(keyring::Error::NoEntry) => Ok(()),
            Err(e) => Err(e.to_string()),
        },
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        .setup(|app| {
            app.manage(AllowedOrigins::default());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            allow_server,
            get_api_key,
            set_api_key
        ])
        .run(tauri::generate_context!())
        .expect("error while running the Hypnos desktop app");
}

#[cfg(test)]
mod tests {
    use super::origin_of;

    #[test]
    fn origin_keeps_scheme_host_and_port() {
        assert_eq!(
            origin_of("http://localhost:9999/v1").unwrap(),
            "http://localhost:9999"
        );
        assert_eq!(origin_of("https://gpu.lan/").unwrap(), "https://gpu.lan");
        assert_eq!(origin_of("https://gpu.lan:443").unwrap(), "https://gpu.lan");
    }

    #[test]
    fn origin_refuses_other_schemes() {
        assert!(origin_of("file:///etc/passwd").is_err());
        assert!(origin_of("localhost:9999").is_err());
        assert!(origin_of("not a url").is_err());
    }
}
