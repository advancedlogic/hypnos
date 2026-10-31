mod attachments;
mod hypnos;
mod local_server;

use tauri::Manager;

/// The keychain service name; the account is the server's origin, so each server keeps its key.
const KEYCHAIN_SERVICE: &str = "dev.hypnos.desktop";

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
        .plugin(tauri_plugin_store::Builder::default().build())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            app.manage(hypnos::Hypnos::default());
            app.manage(local_server::LocalServer::default());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_api_key,
            set_api_key,
            hypnos::hypnos_connect,
            hypnos::hypnos_probe,
            hypnos::hypnos_test,
            hypnos::hypnos_memory_status,
            hypnos::hypnos_models,
            hypnos::hypnos_set_routes,
            hypnos::hypnos_route,
            hypnos::hypnos_chat,
            hypnos::hypnos_chat_cancel,
            attachments::attachment_from_path,
            attachments::attachment_from_bytes,
            local_server::local_server_detect,
            local_server::local_server_start,
            local_server::local_server_stop,
            local_server::local_server_status,
            local_server::local_server_logs,
            local_server::local_server_probe,
        ])
        .build(tauri::generate_context!())
        .expect("error while building the Hypnos desktop app")
        .run(|app, event| {
            // A server the app started does not outlive it.
            if let tauri::RunEvent::Exit = event {
                app.state::<local_server::LocalServer>().shutdown_blocking();
            }
        });
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
