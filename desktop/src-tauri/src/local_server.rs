//! A Hypnos server started by the app (DESK-5). The app never bundles the server: it finds an
//! installed `hypnos` binary, runs it with a config and a port, keeps its output for the log
//! view, and stops it when asked or when the app quits. Health and readiness go through the
//! Rust SDK (`hypnos-client`).

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime, State};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const LOG_LINES: usize = 2000;
const STOP_GRACE: Duration = Duration::from_secs(15);

#[cfg(windows)]
const BINARY_NAME: &str = "hypnos.exe";
#[cfg(not(windows))]
const BINARY_NAME: &str = "hypnos";

/// What the user chose (or the app detected) to run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LaunchOptions {
    pub binary: String,
    /// Optional: without it the server reads `hypnos.toml` from its working directory.
    pub config: Option<String>,
    pub port: u16,
}

#[derive(Debug, Clone, Serialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Status {
    pub running: bool,
    pub pid: Option<u32>,
    pub options: Option<LaunchOptions>,
    /// Set once the process has exited on its own (a crash, a refused config).
    pub exit_code: Option<i32>,
    pub exited: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Detected {
    pub binary: Option<String>,
    pub config: Option<String>,
    /// Where the binary was found: `repo`, `beside-app` or `path`.
    pub source: Option<String>,
}

#[derive(Default)]
struct Inner {
    child: Option<Child>,
    status: Status,
    log: VecDeque<String>,
}

/// Shared by the commands and the tasks that read the process output.
#[derive(Default, Clone)]
pub struct LocalServer(Arc<Mutex<Inner>>);

impl LocalServer {
    /// Stops the process on app exit; blocking, since the event loop is going away.
    pub fn shutdown_blocking(&self) {
        let server = self.clone();
        tauri::async_runtime::block_on(async move {
            let _ = server.stop().await;
        });
    }

    async fn stop(&self) -> Result<(), String> {
        let child = self.0.lock().await.child.take();
        let Some(mut child) = child else {
            return Ok(());
        };
        terminate(&child);
        if tokio::time::timeout(STOP_GRACE, child.wait())
            .await
            .is_err()
        {
            child.kill().await.map_err(|e| e.to_string())?;
        }
        let mut inner = self.0.lock().await;
        inner.status.running = false;
        inner.status.pid = None;
        Ok(())
    }

    async fn push_line<R: Runtime>(&self, app: &AppHandle<R>, line: String) {
        let line = strip_ansi(&line);
        let mut inner = self.0.lock().await;
        if inner.log.len() == LOG_LINES {
            inner.log.pop_front();
        }
        inner.log.push_back(line.clone());
        drop(inner);
        let _ = app.emit("local-server-log", line);
    }
}

/// SIGTERM on Unix, which the server handles as a graceful shutdown (it releases the loaded
/// models); Windows has no equivalent for a console-less child, so it is killed outright.
#[cfg(unix)]
fn terminate(child: &Child) {
    if let Some(pid) = child.id() {
        // SAFETY: plain kill(2) on a pid this process spawned and still owns.
        unsafe {
            libc::kill(pid as libc::pid_t, libc::SIGTERM);
        }
    }
}

#[cfg(not(unix))]
fn terminate(child: &Child) {
    let _ = child;
}

/// Drops terminal colour sequences (`ESC [ ... letter`) from a log line.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\u{1b}' && chars.peek() == Some(&'[') {
            chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn existing(path: PathBuf) -> Option<PathBuf> {
    path.is_file().then_some(path)
}

/// The repo build next to this crate, for running from a checkout: `backend/target/release`.
fn repo_backend() -> Option<PathBuf> {
    let backend = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend");
    let backend = backend.canonicalize().ok()?;
    ["release", "debug"]
        .iter()
        .find_map(|profile| existing(backend.join("target").join(profile).join(BINARY_NAME)))
}

/// The example config the run scripts would pick for a repo build.
fn repo_config(binary: &Path) -> Option<PathBuf> {
    let backend = binary.parent()?.parent()?.parent()?;
    let mut names = vec!["hypnos.toml"];
    if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        names.push("hypnos.macos.example.toml");
    }
    names.push("hypnos.example.toml");
    names.iter().find_map(|n| existing(backend.join(n)))
}

fn on_path() -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).find_map(|dir| existing(dir.join(BINARY_NAME)))
}

fn beside_app() -> Option<PathBuf> {
    existing(std::env::current_exe().ok()?.parent()?.join(BINARY_NAME))
}

/// Finds an installed server: this checkout's build, a binary beside the app, then PATH.
#[tauri::command]
pub fn local_server_detect() -> Detected {
    let found = repo_backend()
        .map(|b| (b, "repo"))
        .or_else(|| beside_app().map(|b| (b, "beside-app")))
        .or_else(|| on_path().map(|b| (b, "path")));
    match found {
        Some((binary, source)) => Detected {
            config: if source == "repo" {
                repo_config(&binary)
            } else {
                None
            }
            .map(|c| c.display().to_string()),
            binary: Some(binary.display().to_string()),
            source: Some(source.to_string()),
        },
        None => Detected {
            binary: None,
            config: None,
            source: None,
        },
    }
}

fn sdk_client(port: u16) -> Result<hypnos_client::Client, String> {
    hypnos_client::Client::builder()
        .base_url(format!("http://127.0.0.1:{port}"))
        .no_api_key()
        .timeout(Duration::from_secs(3))
        .build()
        .map_err(|e| e.to_string())
}

/// Starts the server. Refuses when a Hypnos server already answers on the port (one started by
/// `run.sh`, say): the app then simply connects to it.
#[tauri::command]
pub async fn local_server_start<R: Runtime>(
    app: AppHandle<R>,
    server: State<'_, LocalServer>,
    options: LaunchOptions,
) -> Result<Status, String> {
    if server.0.lock().await.child.is_some() {
        return Err("the local server is already running".into());
    }
    let binary = PathBuf::from(&options.binary);
    if !binary.is_file() {
        return Err(format!("no server binary at {}", binary.display()));
    }
    if let Ok(health) = sdk_client(options.port)?.health().await {
        return Err(format!(
            "Hypnos {} is already running on port {}",
            health.version, options.port
        ));
    }
    let config = options.config.as_deref().filter(|c| !c.trim().is_empty());
    if let Some(c) = config {
        if !Path::new(c).is_file() {
            return Err(format!("no config file at {c}"));
        }
    }

    // Relative paths in a config (weights, models/) are relative to its folder; without a
    // config the server reads hypnos.toml from beside its binary, as in a release archive.
    let workdir = config
        .and_then(|c| Path::new(c).parent())
        .or_else(|| binary.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let bin_dir = binary.parent().map(Path::to_path_buf).unwrap_or_default();

    let mut cmd = Command::new(&binary);
    if let Some(c) = config {
        cmd.arg(c);
    }
    cmd.arg("--port")
        .arg(options.port.to_string())
        .current_dir(&workdir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    // The stable-diffusion shared library sits next to the binary (and in deps/ for a cargo
    // build); `run.sh` sets the same loader paths.
    #[cfg(target_os = "macos")]
    cmd.env(
        "DYLD_LIBRARY_PATH",
        join_paths(&bin_dir, "DYLD_LIBRARY_PATH"),
    );
    #[cfg(target_os = "linux")]
    cmd.env("LD_LIBRARY_PATH", join_paths(&bin_dir, "LD_LIBRARY_PATH"));
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
        let _ = &bin_dir;
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| format!("could not start {}: {e}", binary.display()))?;
    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let pid = child.id();

    {
        let mut inner = server.0.lock().await;
        inner.log.clear();
        inner.child = Some(child);
        inner.status = Status {
            running: true,
            pid,
            options: Some(options.clone()),
            exit_code: None,
            exited: false,
        };
    }
    let header = format!(
        "$ {} {}--port {} (in {})",
        binary.display(),
        config.map(|c| format!("{c} ")).unwrap_or_default(),
        options.port,
        workdir.display()
    );
    server.push_line(&app, header).await;

    if let Some(out) = stdout {
        tauri::async_runtime::spawn(forward(app.clone(), server.inner().clone(), out));
    }
    if let Some(err) = stderr {
        tauri::async_runtime::spawn(forward(app.clone(), server.inner().clone(), err));
    }
    tauri::async_runtime::spawn(watch_exit(app.clone(), server.inner().clone()));

    Ok(server.0.lock().await.status.clone())
}

#[cfg(unix)]
fn join_paths(first: &Path, var: &str) -> std::ffi::OsString {
    let mut paths = vec![first.to_path_buf(), first.join("deps")];
    if let Some(existing) = std::env::var_os(var) {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths).unwrap_or_default()
}

async fn forward<R: Runtime>(
    app: AppHandle<R>,
    server: LocalServer,
    stream: impl AsyncRead + Unpin,
) {
    let mut lines = BufReader::new(stream).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        server.push_line(&app, line).await;
    }
}

/// Polls the child until it exits; an exit nobody asked for is reported to the UI.
async fn watch_exit<R: Runtime>(app: AppHandle<R>, server: LocalServer) {
    loop {
        tokio::time::sleep(Duration::from_millis(500)).await;
        let mut inner = server.0.lock().await;
        let Some(child) = inner.child.as_mut() else {
            return; // stopped through local_server_stop
        };
        if let Ok(Some(exit)) = child.try_wait() {
            inner.child = None;
            inner.status.running = false;
            inner.status.pid = None;
            inner.status.exited = true;
            inner.status.exit_code = exit.code();
            let status = inner.status.clone();
            drop(inner);
            let code = exit
                .code()
                .map_or("a signal".to_string(), |c| format!("code {c}"));
            server
                .push_line(&app, format!("[hypnos exited with {code}]"))
                .await;
            let _ = app.emit("local-server-exit", status);
            return;
        }
    }
}

#[tauri::command]
pub async fn local_server_stop(server: State<'_, LocalServer>) -> Result<Status, String> {
    server.stop().await?;
    Ok(server.0.lock().await.status.clone())
}

#[tauri::command]
pub async fn local_server_status(server: State<'_, LocalServer>) -> Result<Status, String> {
    Ok(server.0.lock().await.status.clone())
}

#[tauri::command]
pub async fn local_server_logs(server: State<'_, LocalServer>) -> Result<Vec<String>, String> {
    Ok(server.0.lock().await.log.iter().cloned().collect())
}

/// Through the SDK: whether the server on this port is up and whether its models are staged.
#[tauri::command]
pub async fn local_server_probe(port: u16) -> Result<(bool, bool), String> {
    let client = sdk_client(port)?;
    let up = client.health().await.is_ok();
    let ready = up && client.is_ready().await;
    Ok((up, ready))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tauri::Manager;

    #[test]
    fn strips_colour_sequences() {
        assert_eq!(
            strip_ansi("\u{1b}[2m2026\u{1b}[0m \u{1b}[32m INFO\u{1b}[0m ready"),
            "2026  INFO ready"
        );
        assert_eq!(strip_ansi("plain"), "plain");
    }

    /// Runs a real server: `HYPNOS_TEST_BINARY=.../hypnos HYPNOS_TEST_CONFIG=small.toml
    /// cargo test -- --ignored` (a config with one small model keeps it quick).
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn starts_logs_and_stops_a_real_server() {
        let binary = std::env::var("HYPNOS_TEST_BINARY").expect("HYPNOS_TEST_BINARY");
        let config = std::env::var("HYPNOS_TEST_CONFIG").ok();
        let port = 9997;
        let app = tauri::test::mock_builder()
            .manage(LocalServer::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let handle = app.handle().clone();
        let options = LaunchOptions {
            binary,
            config,
            port,
        };

        let status = local_server_start(handle.clone(), app.state(), options.clone())
            .await
            .unwrap();
        assert!(status.running && status.pid.is_some());

        let mut ready = false;
        for _ in 0..120 {
            if local_server_probe(port).await.unwrap() == (true, true) {
                ready = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        assert!(ready, "the server never became ready");

        // A second start on the same port is refused while the first one answers.
        let again = local_server_start(handle.clone(), app.state(), options).await;
        assert!(again.unwrap_err().contains("already"));

        let lines = local_server_logs(app.state()).await.unwrap();
        assert!(lines[0].starts_with("$ "), "{lines:?}");
        assert!(lines.len() > 3 && lines.iter().all(|l| !l.contains('\u{1b}')));

        let status = local_server_stop(app.state()).await.unwrap();
        assert!(!status.running);
        assert_eq!(local_server_probe(port).await.unwrap(), (false, false));
    }
}
