//! A Hypnos server started by the app (DESK-5). It runs a `hypnos` binary with a config and a
//! port, keeps its output for the log view, and stops it when asked or when the app quits.
//! Health and readiness go through the Rust SDK (`hypnos-client`).
//!
//! The app can carry the server (DESK-16 on macOS, DESK-32 on Windows and Linux): `hypnos` as a
//! Tauri sidecar beside the app's own binary (`Contents/MacOS`, the install folder, `/usr/bin`),
//! its sd.cpp library in `Contents/Frameworks` / beside it / with the resources, and the data
//! (espeak, playground, catalog) with a config template in `server/` under the app's resources
//! (`Contents/Resources`, the install folder, `/usr/lib/Hypnos`). The bundle is read-only (an
//! installer or a signature covers it), so the bundled server runs with a config the app owns
//! in its data folder (`~/Library/Application Support/Hypnos/`, `%APPDATA%\Hypnos\`,
//! `~/.local/share/Hypnos/`: `hypnos.toml`, `models/`). Before each start the app disables the
//! models of that config whose files are missing (the server refuses to start otherwise) and
//! enables them again once the files are there; models the user disabled stay disabled.
//! Without a bundled server the app finds an installed binary: this checkout's build, one
//! beside the app, or PATH.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager, Runtime, State};
use tokio::io::{AsyncBufReadExt, AsyncRead, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;

const LOG_LINES: usize = 2000;
const STOP_GRACE: Duration = Duration::from_secs(15);

#[cfg(windows)]
const BINARY_NAME: &str = "hypnos.exe";
#[cfg(not(windows))]
const BINARY_NAME: &str = "hypnos";
/// Read by the server (`hypnos_core::config::beside_executable`): a folder holding its data
/// folders, which the bundled server on Windows and Linux needs since its data sits with the
/// app's resources rather than beside the binary.
const RESOURCE_DIR_VAR: &str = "HYPNOS_RESOURCE_DIR";

/// The app's resource folder, set once at startup: `Contents/Resources` on macOS, the install
/// folder on Windows, `/usr/lib/Hypnos` on Linux (`$APPDIR/usr/lib/Hypnos` in an AppImage),
/// the cargo target dir of a development build.
static RESOURCES: OnceLock<PathBuf> = OnceLock::new();

pub(crate) fn init_resources<R: Runtime>(app: &AppHandle<R>) {
    if let Ok(dir) = app.path().resource_dir() {
        let _ = RESOURCES.set(dir);
    }
}

fn resources() -> Option<&'static Path> {
    RESOURCES.get().map(PathBuf::as_path)
}

/// A file of the bundled server's data, `server/<name>` under the app's resources (the config
/// template, `models.manifest`, `models.catalog.toml`), when the app carries the server.
pub(crate) fn bundle_file(name: &str) -> Option<PathBuf> {
    existing(resources()?.join("server").join(name))
}

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

/// The server inside this app's bundle (DESK-16, DESK-32): the sidecar `hypnos` beside the
/// app's binary, with the config template the bundling build puts in `server/` under the
/// app's resources. A bare `hypnos` beside the app without the template is `beside-app`.
fn bundled() -> Option<PathBuf> {
    let binary = beside_app()?;
    bundled_template(resources()?)?;
    Some(binary)
}

fn bundled_template(resources: &Path) -> Option<PathBuf> {
    existing(resources.join("server").join("hypnos.toml"))
}

/// Whether `binary` is the server bundled in this app (and so needs the app's resources).
fn is_bundled(binary: &Path) -> bool {
    match (
        bundled().and_then(|b| b.canonicalize().ok()),
        binary.canonicalize(),
    ) {
        (Some(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// The folder the bundled server's config and models live in: `Hypnos` in the app data folder
/// (`~/Library/Application Support`, `%APPDATA%`, `~/.local/share`), outside the bundle, which
/// must stay unchanged for its signature or its installer's records.
fn managed_dir<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    Ok(app
        .path()
        .data_dir()
        .map_err(|e| format!("no application data folder: {e}"))?
        .join("Hypnos"))
}

/// The bundled server's config, written from the bundle's template the first time.
fn managed_config<R: Runtime>(app: &AppHandle<R>) -> Result<PathBuf, String> {
    let dir = managed_dir(app)?;
    let config = dir.join("hypnos.toml");
    std::fs::create_dir_all(dir.join("models"))
        .map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    if !config.exists() {
        let template = bundle_file("hypnos.toml").ok_or("the app bundle has no config template")?;
        std::fs::copy(&template, &config)
            .map_err(|e| format!("could not write {}: {e}", config.display()))?;
    }
    Ok(config)
}

/// Finds a server: the one bundled in this app, this checkout's build, a binary beside the
/// app, then PATH. The bundled one comes with the app's own config (created on first use).
#[tauri::command]
pub fn local_server_detect<R: Runtime>(app: AppHandle<R>) -> Result<Detected, String> {
    let found = bundled()
        .map(|b| (b, "bundled"))
        .or_else(|| repo_backend().map(|b| (b, "repo")))
        .or_else(|| beside_app().map(|b| (b, "beside-app")))
        .or_else(|| on_path().map(|b| (b, "path")));
    let Some((binary, source)) = found else {
        return Ok(Detected {
            binary: None,
            config: None,
            source: None,
        });
    };
    let config = match source {
        "bundled" => Some(managed_config(&app)?),
        "repo" => repo_config(&binary),
        _ => None,
    };
    Ok(Detected {
        config: config.map(|c| c.display().to_string()),
        binary: Some(binary.display().to_string()),
        source: Some(source.to_string()),
    })
}

const MODEL_PATH_KEYS: [&str; 4] = [
    "weights_path",
    "aux_weights_path",
    "text_encoder_path",
    "tokenizer_path",
];
/// Beside the managed config: the ids the app disabled because their files were missing.
const AUTO_DISABLED_FILE: &str = "auto-disabled.json";

/// What `sync_models` changed in the config.
#[derive(Debug, Default, PartialEq)]
struct ModelSync {
    disabled: Vec<String>,
    enabled: Vec<String>,
    /// The folder the model paths resolve against.
    models_dir: PathBuf,
    /// `models_dir` is set but not there (an unplugged drive): nothing was changed, and the
    /// server reports it as one error.
    missing_dir: bool,
}

/// Matches `enabled` in the app's own config to the model files on disk, as the server's
/// validation checks them: a model whose files are missing is disabled (the server would refuse
/// to start), one the app disabled that way is enabled again when they are all there. A model
/// the user disabled is not in the auto-disabled list and stays as it is.
fn sync_models(config: &Path) -> Result<ModelSync, String> {
    let dir = config.parent().unwrap_or(Path::new("."));
    let text = std::fs::read_to_string(config)
        .map_err(|e| format!("could not read {}: {e}", config.display()))?;
    let mut doc: toml_edit::DocumentMut = text
        .parse()
        .map_err(|e| format!("{}: {e}", config.display()))?;
    let state_path = dir.join(AUTO_DISABLED_FILE);
    let mut auto: std::collections::BTreeSet<String> = std::fs::read_to_string(&state_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_default();
    let before = auto.clone();

    // Model paths resolve against [server] models_dir (relative to the config's folder), as
    // the server resolves them (it runs in that folder).
    let base = match doc
        .get("server")
        .and_then(|s| s.get("models_dir"))
        .and_then(|v| v.as_str())
    {
        Some(m) => dir.join(m),
        None => dir.to_path_buf(),
    };
    let mut sync = ModelSync {
        models_dir: base.clone(),
        ..ModelSync::default()
    };
    if !base.is_dir() {
        // Disabling everything would only hide the real problem.
        sync.missing_dir = true;
        return Ok(sync);
    }
    let mut ids = std::collections::BTreeSet::new();
    if let Some(models) = doc
        .get_mut("models")
        .and_then(|m| m.as_array_of_tables_mut())
    {
        for model in models.iter_mut() {
            let Some(id) = model.get("id").and_then(|v| v.as_str()).map(String::from) else {
                continue;
            };
            ids.insert(id.clone());
            let present = MODEL_PATH_KEYS
                .iter()
                .filter_map(|k| model.get(k).and_then(|v| v.as_str()))
                .all(|p| base.join(p).is_file());
            let enabled = model
                .get("enabled")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            if enabled && !present {
                model["enabled"] = toml_edit::value(false);
                auto.insert(id.clone());
                sync.disabled.push(id);
            } else if !enabled && present && auto.remove(&id) {
                model["enabled"] = toml_edit::value(true);
                sync.enabled.push(id);
            }
        }
    }
    auto.retain(|id| ids.contains(id));

    if !sync.disabled.is_empty() || !sync.enabled.is_empty() {
        std::fs::write(config, doc.to_string())
            .map_err(|e| format!("could not write {}: {e}", config.display()))?;
    }
    if auto != before {
        let json = serde_json::to_string_pretty(&auto).unwrap_or_default();
        std::fs::write(&state_path, json)
            .map_err(|e| format!("could not write {}: {e}", state_path.display()))?;
    }
    Ok(sync)
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
    // The app's own config (the bundled server's): match `enabled` to the files on disk.
    let mut notes = Vec::new();
    if let Some(c) = config.filter(|c| is_managed(&app, Path::new(c))) {
        let sync = sync_models(Path::new(c))?;
        let models = &sync.models_dir;
        if sync.missing_dir {
            notes.push(format!(
                "[app] the models folder {} is not there (an unplugged drive?): connect it, or choose another in Models",
                models.display()
            ));
        }
        if !sync.disabled.is_empty() {
            notes.push(format!(
                "[app] disabled until their files are in {}: {}",
                models.display(),
                sync.disabled.join(", ")
            ));
        }
        if !sync.enabled.is_empty() {
            notes.push(format!(
                "[app] enabled again, their files are there now: {}",
                sync.enabled.join(", ")
            ));
        }
    }

    // Relative paths in a config (weights, models/) are relative to its folder; without a
    // config the server reads hypnos.toml from beside its binary, as in a release archive.
    let workdir = config
        .and_then(|c| Path::new(c).parent())
        .or_else(|| binary.parent())
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

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
    apply_server_env(&mut cmd, &binary);

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
    for note in notes {
        server.push_line(&app, note).await;
    }

    if let Some(out) = stdout {
        tauri::async_runtime::spawn(forward(app.clone(), server.inner().clone(), out));
    }
    if let Some(err) = stderr {
        tauri::async_runtime::spawn(forward(app.clone(), server.inner().clone(), err));
    }
    tauri::async_runtime::spawn(watch_exit(app.clone(), server.inner().clone()));

    Ok(server.0.lock().await.status.clone())
}

pub fn is_managed<R: Runtime>(app: &AppHandle<R>, config: &Path) -> bool {
    let Ok(dir) = managed_dir(app) else {
        return false;
    };
    let managed = dir.join("hypnos.toml");
    match (managed.canonicalize(), config.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    }
}

/// What a server process the app runs needs from its environment: the loader path for the
/// sd.cpp shared library (beside the binary and in `deps/` for a cargo build, as `run.sh` sets
/// it; with the app's resources for the bundled server on Linux), `HYPNOS_RESOURCE_DIR` so the
/// bundled server finds its data where the installer put it on Windows and Linux (on macOS it
/// looks in `../Resources` by itself), and no console window on Windows.
pub(crate) fn apply_server_env(cmd: &mut Command, binary: &Path) {
    let bundled = is_bundled(binary).then(resources).flatten();
    if let Some(res) = bundled {
        cmd.env(RESOURCE_DIR_VAR, res);
    }
    let bin_dir = binary.parent().map(Path::to_path_buf).unwrap_or_default();
    #[cfg(target_os = "macos")]
    cmd.env(
        "DYLD_LIBRARY_PATH",
        join_paths(&bin_dir, None, "DYLD_LIBRARY_PATH"),
    );
    #[cfg(target_os = "linux")]
    cmd.env(
        "LD_LIBRARY_PATH",
        join_paths(&bin_dir, bundled, "LD_LIBRARY_PATH"),
    );
    #[cfg(windows)]
    {
        // tokio's Command has creation_flags of its own; the DLLs sit beside the binary.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
        let _ = (&bin_dir, bundled);
    }
}

/// `first`, `first/deps`, `extra`, then the variable's current value.
#[cfg_attr(windows, allow(dead_code))]
pub(crate) fn join_paths(first: &Path, extra: Option<&Path>, var: &str) -> std::ffi::OsString {
    let mut paths = vec![first.to_path_buf(), first.join("deps")];
    paths.extend(extra.map(Path::to_path_buf));
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

/// The process listening on `port` on this machine, if its name says it is a Hypnos server (not
/// this app): `(pid, name)`.
fn listening_hypnos(port: u16) -> Option<(u32, String)> {
    let is_server = |name: &str| {
        let base = std::path::Path::new(name.trim())
            .file_name()
            .map(|n| n.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        base == "hypnos" || base == "hypnos.exe"
    };
    #[cfg(unix)]
    {
        let out = std::process::Command::new("lsof")
            .args(["-nP", &format!("-iTCP:{port}"), "-sTCP:LISTEN", "-t"])
            .output()
            .ok()?;
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter_map(|l| l.trim().parse::<u32>().ok())
            .find_map(|pid| {
                let ps = std::process::Command::new("ps")
                    .args(["-p", &pid.to_string(), "-o", "comm="])
                    .output()
                    .ok()?;
                let name = String::from_utf8_lossy(&ps.stdout).trim().to_string();
                is_server(&name).then_some((pid, name))
            })
    }
    #[cfg(windows)]
    {
        let out = std::process::Command::new("netstat")
            .args(["-ano", "-p", "TCP"])
            .output()
            .ok()?;
        let needle = format!(":{port}");
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| l.contains("LISTENING"))
            .filter_map(|l| {
                let cols: Vec<&str> = l.split_whitespace().collect();
                (cols.len() >= 5 && cols[1].ends_with(&needle))
                    .then(|| cols[4].parse::<u32>().ok())?
            })
            .find_map(|pid| {
                let out = std::process::Command::new("tasklist")
                    .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
                    .output()
                    .ok()?;
                let line = String::from_utf8_lossy(&out.stdout);
                let name = line.split(',').next()?.trim_matches('"').to_string();
                is_server(&name).then_some((pid, name))
            })
    }
}

/// Stops the Hypnos server listening on `port` even when this app did not start it (the tray's
/// "Stop the server"): SIGTERM on Linux/macOS, which the server handles as a graceful shutdown,
/// `taskkill` on Windows. Only a process named `hypnos` is touched. Waits up to 20 s for it to go.
#[tauri::command]
pub async fn local_server_stop_port(
    server: State<'_, LocalServer>,
    port: u16,
) -> Result<(), String> {
    let ours = server
        .0
        .lock()
        .await
        .status
        .options
        .as_ref()
        .map(|o| o.port)
        == Some(port)
        && server.0.lock().await.child.is_some();
    if ours {
        return server.stop().await;
    }
    let (pid, name) = tauri::async_runtime::spawn_blocking(move || listening_hypnos(port))
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("no Hypnos server process found listening on port {port}"))?;
    #[cfg(unix)]
    // SAFETY: kill(2) with SIGTERM on a pid that was just checked to be a `hypnos` process.
    let sent = unsafe { libc::kill(pid as libc::pid_t, libc::SIGTERM) } == 0;
    #[cfg(windows)]
    let sent = std::process::Command::new("taskkill")
        .args(["/PID", &pid.to_string(), "/T", "/F"])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !sent {
        return Err(format!("could not stop {name} (pid {pid})"));
    }
    for _ in 0..40 {
        tokio::time::sleep(Duration::from_millis(500)).await;
        if sdk_client(port)?.health().await.is_err() {
            return Ok(());
        }
    }
    Err(format!("{name} (pid {pid}) is still running after 20 s"))
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

    const CONFIG: &str = r#"# the user's comment survives
[server]
port = 9999
models_dir = "models"

[[models]]
id = "chat"
weights_path = "chat.gguf"
aux_weights_path = "mmproj.gguf"

[[models]]
id = "tts"
weights_path = "kokoro/model.onnx"

[[models]]
id = "off-by-user"
enabled = false
weights_path = "gone.gguf"
"#;

    fn scratch(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("hypnos-desktop-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("models/kokoro")).unwrap();
        std::fs::write(dir.join("hypnos.toml"), CONFIG).unwrap();
        dir
    }

    fn enabled(config: &Path, id: &str) -> bool {
        let doc: toml_edit::DocumentMut = std::fs::read_to_string(config).unwrap().parse().unwrap();
        let found = doc["models"]
            .as_array_of_tables()
            .unwrap()
            .iter()
            .find(|m| m["id"].as_str() == Some(id))
            .and_then(|m| m.get("enabled").and_then(|v| v.as_bool()));
        found.unwrap_or(true)
    }

    #[test]
    fn disables_models_without_files_and_enables_them_once_they_arrive() {
        let dir = scratch("sync");
        let config = dir.join("hypnos.toml");
        std::fs::write(dir.join("models/kokoro/model.onnx"), b"x").unwrap();
        std::fs::write(dir.join("models/chat.gguf"), b"x").unwrap();

        // The projector is missing: chat goes off, tts stays on, the user's choice is kept.
        let sync = sync_models(&config).unwrap();
        assert_eq!(sync.disabled, vec!["chat".to_string()]);
        assert!(sync.enabled.is_empty());
        assert!(!enabled(&config, "chat"));
        assert!(enabled(&config, "tts"));
        assert!(!enabled(&config, "off-by-user"));
        assert!(std::fs::read_to_string(&config)
            .unwrap()
            .starts_with("# the user's comment survives"));

        // Nothing changed on disk: nothing to do.
        let sync = sync_models(&config).unwrap();
        assert!(sync.disabled.is_empty() && sync.enabled.is_empty());
        assert_eq!(sync.models_dir, dir.join("models"));

        // The projector arrives: chat is back; off-by-user still off although never auto-disabled.
        std::fs::write(dir.join("models/mmproj.gguf"), b"x").unwrap();
        std::fs::write(dir.join("models/gone.gguf"), b"x").unwrap();
        let sync = sync_models(&config).unwrap();
        assert_eq!(sync.enabled, vec!["chat".to_string()]);
        assert!(enabled(&config, "chat"));
        assert!(!enabled(&config, "off-by-user"));
        let auto: Vec<String> =
            serde_json::from_str(&std::fs::read_to_string(dir.join(AUTO_DISABLED_FILE)).unwrap())
                .unwrap();
        assert!(auto.is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_missing_models_folder_changes_nothing() {
        let dir = scratch("unplugged");
        let config = dir.join("hypnos.toml");
        let unplugged = CONFIG.replace(
            "models_dir = \"models\"",
            "models_dir = \"/Volumes/hypnos-test-unplugged\"",
        );
        std::fs::write(&config, &unplugged).unwrap();
        let sync = sync_models(&config).unwrap();
        assert!(sync.missing_dir);
        assert!(sync.disabled.is_empty());
        assert_eq!(std::fs::read_to_string(&config).unwrap(), unplugged);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_bundled_server_needs_the_template_in_the_resources() {
        let root = scratch("bundle");
        // The app's resources without a staged server (a plain build): no bundled server.
        let resources = root.join("Hypnos.app/Contents/Resources");
        std::fs::create_dir_all(&resources).unwrap();
        assert_eq!(bundled_template(&resources), None);
        std::fs::create_dir_all(resources.join("server")).unwrap();
        std::fs::write(resources.join("server/hypnos.toml"), CONFIG).unwrap();
        assert_eq!(
            bundled_template(&resources),
            Some(resources.join("server").join("hypnos.toml"))
        );
        // Windows: the resources are the install folder itself.
        let install = root.join("Program Files/Hypnos");
        std::fs::create_dir_all(install.join("server")).unwrap();
        assert_eq!(bundled_template(&install), None);
        std::fs::write(install.join("server/hypnos.toml"), CONFIG).unwrap();
        assert_eq!(
            bundled_template(&install),
            Some(install.join("server").join("hypnos.toml"))
        );
        // Not in a bundle (no resource dir known, as in these tests): no bundled file either.
        assert_eq!(bundle_file("hypnos.toml"), None);
        let _ = std::fs::remove_dir_all(&root);
    }

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

    /// Stops a server this app did not start (same env as the test above), and leaves a
    /// non-Hypnos process alone: `cargo test stops_a_server -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn stops_a_server_it_did_not_start() {
        let binary = std::env::var("HYPNOS_TEST_BINARY").expect("HYPNOS_TEST_BINARY");
        let config = std::env::var("HYPNOS_TEST_CONFIG").expect("HYPNOS_TEST_CONFIG");
        let port = 9997;
        let dir = std::path::Path::new(&binary)
            .parent()
            .unwrap()
            .to_path_buf();
        let mut child = std::process::Command::new(&binary)
            .args([config.as_str(), "--port", &port.to_string()])
            .env(
                "DYLD_LIBRARY_PATH",
                join_paths(&dir, None, "DYLD_LIBRARY_PATH"),
            )
            .env("LD_LIBRARY_PATH", join_paths(&dir, None, "LD_LIBRARY_PATH"))
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        for _ in 0..120 {
            if local_server_probe(port).await.unwrap().0 {
                break;
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        let app = tauri::test::mock_builder()
            .manage(LocalServer::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        local_server_stop_port(app.state(), port).await.unwrap();
        assert_eq!(local_server_probe(port).await.unwrap(), (false, false));
        let exit = child.wait().unwrap();
        assert!(exit.code() == Some(0) || exit.code().is_none(), "{exit:?}");

        // Something else listening on a port is not touched.
        let other = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let other_port = other.local_addr().unwrap().port();
        let err = local_server_stop_port(app.state(), other_port)
            .await
            .unwrap_err();
        assert!(err.contains("no Hypnos server"), "{err}");
    }
}
