//! Model files for the local server (DESK-17): the lineup from `models.manifest`, downloads with
//! resume and verification, and the folder they live in.
//!
//! The manifest is the one `./run.sh models` reads (`backend/models.manifest`; the macOS app
//! bundles a copy in `Contents/Resources/server/`). Its `local` column is a path under the models
//! folder, which is the config's `[server] models_dir` (relative to the config's folder), or
//! `<config folder>/models` for a config without one. Sizes and SHA-256 sums come from the hosts:
//! the Hugging Face tree API (`lfs.oid` is the SHA-256 of an LFS file) and the GitHub release API
//! (`digest`); a file without a published sum is checked by size. Downloads go to `<file>.part`
//! first (the same name `fetch-models.sh` uses, so either resumes the other's) and resume with a
//! `Range` request. Enabling the downloaded models is left to `local_server::sync_models` on the
//! next start; the app restarts the server after a download.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::ipc::Channel;
use tauri::{AppHandle, Runtime, State};
use tokio::sync::Mutex;

const HF: &str = "https://huggingface.co";
const PROGRESS_EVERY: Duration = Duration::from_millis(250);
const USER_AGENT: &str = concat!("hypnos-desktop/", env!("CARGO_PKG_VERSION"));

/// One line of `models.manifest`.
#[derive(Debug, Clone, PartialEq)]
struct ManifestLine {
    model: String,
    repo: String,
    remote: String,
    local: String,
}

/// The manifest's lines for this machine (`platform` = all, or this one), in file order.
fn parse_manifest(text: &str, apple_silicon: bool) -> Result<Vec<ManifestLine>, String> {
    let mut lines = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if let Some(version) = line.strip_prefix("# manifest-version:") {
            let v = version.split_whitespace().next().unwrap_or("");
            if v != "1" {
                return Err(format!(
                    "models.manifest version {v} is newer than this app understands"
                ));
            }
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < 5 {
            continue;
        }
        let for_here = match cols[4] {
            "all" => true,
            "macos-arm64" => apple_silicon,
            "other" => !apple_silicon,
            _ => false,
        };
        if for_here {
            lines.push(ManifestLine {
                model: cols[0].into(),
                repo: cols[1].into(),
                remote: cols[2].into(),
                local: cols[3].into(),
            });
        }
    }
    Ok(lines)
}

/// A file to have under the models folder, with what the host says about it.
#[derive(Debug, Clone, PartialEq)]
struct RemoteFile {
    model: String,
    url: String,
    local: String,
    size: Option<u64>,
    sha256: Option<String>,
}

#[derive(Deserialize)]
struct HfEntry {
    #[serde(rename = "type")]
    kind: String,
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(default)]
    lfs: Option<HfLfs>,
}

#[derive(Deserialize)]
struct HfLfs {
    oid: String,
}

#[derive(Deserialize)]
struct GhRelease {
    assets: Vec<GhAsset>,
}

#[derive(Deserialize)]
struct GhAsset {
    name: String,
    size: u64,
    #[serde(default)]
    digest: Option<String>,
}

/// What a host knows about its files: path in the repo (or asset name) -> (size, sha256).
type Listing = BTreeMap<String, (u64, Option<String>)>;

fn file_url(repo: &str, remote: &str) -> String {
    if repo.starts_with("https://") {
        format!("{}/{remote}", repo.trim_end_matches('/'))
    } else {
        format!("{HF}/{repo}/resolve/main/{remote}")
    }
}

/// `https://github.com/<owner>/<repo>/releases/download/<tag>` -> the release API URL.
fn github_release_api(base: &str) -> Option<String> {
    let rest = base.strip_prefix("https://github.com/")?;
    let parts: Vec<&str> = rest.trim_end_matches('/').split('/').collect();
    match parts.as_slice() {
        [owner, repo, "releases", "download", tag] => Some(format!(
            "https://api.github.com/repos/{owner}/{repo}/releases/tags/{tag}"
        )),
        _ => None,
    }
}

async fn listing(client: &reqwest::Client, repo: &str) -> Result<Listing, String> {
    let get = |url: String| async move {
        client
            .get(&url)
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| format!("{url}: {e}"))
    };
    if !repo.starts_with("https://") {
        let entries: Vec<HfEntry> = get(format!("{HF}/api/models/{repo}/tree/main?recursive=true"))
            .await?
            .json()
            .await
            .map_err(|e| format!("{repo}: {e}"))?;
        return Ok(entries
            .into_iter()
            .filter(|e| e.kind == "file")
            .map(|e| (e.path, (e.size, e.lfs.map(|l| l.oid))))
            .collect());
    }
    if let Some(api) = github_release_api(repo) {
        let release: GhRelease = get(api)
            .await?
            .json()
            .await
            .map_err(|e| format!("{repo}: {e}"))?;
        return Ok(release
            .assets
            .into_iter()
            .map(|a| {
                let sha = a
                    .digest
                    .and_then(|d| d.strip_prefix("sha256:").map(str::to_string));
                (a.name, (a.size, sha))
            })
            .collect());
    }
    Ok(Listing::new())
}

/// The manifest's lines as files: folders (`remote` ending in `/`) become their files, sizes
/// and sums are filled from the hosts' listings. Without a listing (offline, an unknown host) a
/// plain file keeps unknown size and sum, and a folder cannot be expanded yet.
fn resolve(lines: &[ManifestLine], listings: &BTreeMap<String, Listing>) -> Vec<RemoteFile> {
    let mut files = Vec::new();
    for l in lines {
        let listing = listings.get(&l.repo);
        if let Some(dir) = l.remote.strip_suffix('/') {
            let Some(listing) = listing else { continue };
            let prefix = format!("{dir}/");
            for (path, (size, sha)) in listing.range(prefix.clone()..) {
                let Some(rest) = path.strip_prefix(&prefix) else {
                    break;
                };
                files.push(RemoteFile {
                    model: l.model.clone(),
                    url: file_url(&l.repo, path),
                    local: format!("{}{rest}", l.local),
                    size: Some(*size),
                    sha256: sha.clone(),
                });
            }
        } else {
            let meta = listing.and_then(|ls| ls.get(&l.remote));
            files.push(RemoteFile {
                model: l.model.clone(),
                url: file_url(&l.repo, &l.remote),
                local: l.local.clone(),
                size: meta.map(|m| m.0),
                sha256: meta.and_then(|m| m.1.clone()),
            });
        }
    }
    files
}

/// Downloads in progress and the host listings of this session.
#[derive(Default)]
pub struct ModelFiles {
    listings: Mutex<Option<BTreeMap<String, Listing>>>,
    cancel: Arc<AtomicBool>,
    busy: Arc<AtomicBool>,
}

fn http() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(Duration::from_secs(15))
        .read_timeout(Duration::from_secs(60))
        .build()
        .map_err(|e| e.to_string())
}

/// The listings for the manifest's hosts, fetched once per session (and again after a failure).
async fn listings(
    state: &ModelFiles,
    lines: &[ManifestLine],
) -> (BTreeMap<String, Listing>, Vec<String>) {
    let mut cached = state.listings.lock().await;
    if let Some(l) = cached.as_ref() {
        return (l.clone(), Vec::new());
    }
    let Ok(client) = http() else {
        return (BTreeMap::new(), vec!["no HTTP client".into()]);
    };
    let repos: BTreeSet<&str> = lines.iter().map(|l| l.repo.as_str()).collect();
    let fetched = futures::future::join_all(repos.iter().map(|r| listing(&client, r))).await;
    let mut out = BTreeMap::new();
    let mut errors = Vec::new();
    for (repo, result) in repos.into_iter().zip(fetched) {
        match result {
            Ok(l) => {
                out.insert(repo.to_string(), l);
            }
            Err(e) => errors.push(e),
        }
    }
    if errors.is_empty() {
        *cached = Some(out.clone());
    }
    (out, errors)
}

/// `models.manifest`: the bundled copy, else this checkout's.
fn manifest_path() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let bundled = exe_dir.parent()?.join("Resources/server/models.manifest");
    if bundled.is_file() {
        return Some(bundled);
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/models.manifest");
    repo.is_file().then_some(repo)
}

fn read_doc(config: &Path) -> Result<toml_edit::DocumentMut, String> {
    std::fs::read_to_string(config)
        .map_err(|e| format!("could not read {}: {e}", config.display()))?
        .parse()
        .map_err(|e| format!("{}: {e}", config.display()))
}

/// Where the manifest's files go for this config: `[server] models_dir` against the config's
/// folder, else `<config folder>/models` (the layout of a config from before `models_dir`, whose
/// paths start with `models/`).
pub fn models_root(config: &Path, doc: &toml_edit::DocumentMut) -> PathBuf {
    let dir = config.parent().unwrap_or(Path::new("."));
    match doc
        .get("server")
        .and_then(|s| s.get("models_dir"))
        .and_then(|v| v.as_str())
    {
        Some(m) => dir.join(m),
        None => dir.join("models"),
    }
}

/// Free bytes on the volume that holds `path` (or its nearest existing parent).
fn free_bytes(path: &Path) -> Option<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        let c = std::ffi::CString::new(p.as_os_str().as_bytes()).ok()?;
        // SAFETY: statvfs fills a zeroed struct for a NUL-terminated path we own.
        let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
            return None;
        }
        Some(st.f_bavail as u64 * st.f_frsize as u64)
    }
    #[cfg(not(unix))]
    {
        let _ = p;
        None
    }
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// A model's state on disk: bytes there (complete files plus `.part`s) and whether it is whole.
fn have(root: &Path, f: &RemoteFile) -> (u64, bool) {
    let dest = root.join(&f.local);
    if dest.is_file() {
        let len = file_len(&dest);
        return (len, f.size.is_none_or(|s| s == len));
    }
    (file_len(&part_path(&dest)), false)
}

fn part_path(dest: &Path) -> PathBuf {
    let mut name = dest.file_name().unwrap_or_default().to_os_string();
    name.push(".part");
    dest.with_file_name(name)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogModel {
    id: String,
    /// From the config's `[[models]]` block, when it has one for this id.
    modality: Option<String>,
    in_config: bool,
    enabled: Option<bool>,
    /// None while a size is unknown (offline).
    size: Option<u64>,
    have: u64,
    installed: bool,
    files: usize,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Catalog {
    models_dir: String,
    dir_exists: bool,
    free_bytes: Option<u64>,
    /// The app's own config: the folder can be changed from the app.
    managed: bool,
    models: Vec<CatalogModel>,
    /// Hosts that could not be asked (offline): sizes and folder contents are incomplete.
    errors: Vec<String>,
}

/// The lineup for a config: every model of the manifest with its size and what is on disk.
#[tauri::command]
pub async fn models_catalog<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, ModelFiles>,
    config: String,
) -> Result<Catalog, String> {
    let config = PathBuf::from(config);
    let doc = read_doc(&config)?;
    let root = models_root(&config, &doc);
    let manifest = manifest_path().ok_or("no models.manifest in the app or this checkout")?;
    let text = std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?;
    let lines = parse_manifest(
        &text,
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
    )?;
    let (listings, errors) = listings(&state, &lines).await;
    let files = resolve(&lines, &listings);

    let blocks: BTreeMap<String, (Option<String>, bool)> = doc
        .get("models")
        .and_then(|m| m.as_array_of_tables())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    let modality = m.get("modality").and_then(|v| v.as_str()).map(String::from);
                    let enabled = m.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true);
                    Some((id, (modality, enabled)))
                })
                .collect()
        })
        .unwrap_or_default();

    let mut ids: Vec<String> = Vec::new();
    for l in &lines {
        if !ids.contains(&l.model) {
            ids.push(l.model.clone());
        }
    }
    let models = ids
        .into_iter()
        // A manifest entry no [[models]] block uses (docling-gpu: the fp32 files for docling on
        // the GPU) does nothing for this config: listed only once some of it is on disk.
        .filter(|id| {
            blocks.contains_key(id)
                || files
                    .iter()
                    .filter(|f| &f.model == id)
                    .any(|f| have(&root, f).0 > 0)
        })
        .map(|id| {
            let mine: Vec<&RemoteFile> = files.iter().filter(|f| f.model == id).collect();
            // A folder that could not be listed leaves the model with fewer files than lines.
            let unexpanded = lines.iter().any(|l| {
                l.model == id && l.remote.ends_with('/') && !listings.contains_key(&l.repo)
            });
            let size = (!unexpanded)
                .then(|| mine.iter().map(|f| f.size).sum::<Option<u64>>())
                .flatten();
            let states: Vec<(u64, bool)> = mine.iter().map(|f| have(&root, f)).collect();
            let block = blocks.get(&id);
            CatalogModel {
                modality: block.and_then(|b| b.0.clone()),
                in_config: block.is_some(),
                enabled: block.map(|b| b.1),
                size,
                have: states.iter().map(|s| s.0).sum(),
                installed: !mine.is_empty() && !unexpanded && states.iter().all(|s| s.1),
                files: mine.len(),
                id,
            }
        })
        .collect();

    Ok(Catalog {
        models_dir: root.display().to_string(),
        dir_exists: root.is_dir(),
        free_bytes: free_bytes(&root),
        managed: crate::local_server::is_managed(&app, &config),
        models,
        errors,
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum DownloadEvent {
    /// What will be fetched: bytes still missing over all files (None if a size is unknown).
    #[serde(rename_all = "camelCase")]
    Start {
        files: usize,
        total_bytes: Option<u64>,
    },
    /// A file starts (or resumes at `from`).
    #[serde(rename_all = "camelCase")]
    File {
        model: String,
        local: String,
        size: Option<u64>,
        from: u64,
    },
    #[serde(rename_all = "camelCase")]
    Progress {
        done_bytes: u64,
        file_bytes: u64,
    },
    /// Checking the SHA-256 of a finished file.
    Verifying {
        local: String,
    },
    Done {
        models: Vec<String>,
    },
    /// Stopped by `models_cancel`; `.part` files stay for a resume.
    Cancelled,
}

/// Downloads the missing files of `ids` into the config's models folder. One download runs at a
/// time; `models_cancel` stops it at the next chunk.
#[tauri::command]
pub async fn models_download(
    state: State<'_, ModelFiles>,
    config: String,
    ids: Vec<String>,
    on_event: Channel<DownloadEvent>,
) -> Result<(), String> {
    if state.busy.swap(true, Ordering::SeqCst) {
        return Err("a download is already running".into());
    }
    state.cancel.store(false, Ordering::SeqCst);
    let result = download(&state, &config, &ids, &on_event).await;
    state.busy.store(false, Ordering::SeqCst);
    result
}

async fn download(
    state: &ModelFiles,
    config: &str,
    ids: &[String],
    on_event: &Channel<DownloadEvent>,
) -> Result<(), String> {
    let config = PathBuf::from(config);
    let root = models_root(&config, &read_doc(&config)?);
    std::fs::create_dir_all(&root)
        .map_err(|e| format!("could not create {}: {e}", root.display()))?;
    let manifest = manifest_path().ok_or("no models.manifest")?;
    let text = std::fs::read_to_string(&manifest).map_err(|e| e.to_string())?;
    let lines: Vec<ManifestLine> = parse_manifest(
        &text,
        cfg!(all(target_os = "macos", target_arch = "aarch64")),
    )?
    .into_iter()
    .filter(|l| ids.contains(&l.model))
    .collect();
    let (listings, errors) = listings(state, &lines).await;
    if let Some(l) = lines
        .iter()
        .find(|l| l.remote.ends_with('/') && !listings.contains_key(&l.repo))
    {
        return Err(format!(
            "cannot list {} for {} ({}); check the connection",
            l.remote,
            l.model,
            errors.join("; ")
        ));
    }
    let todo: Vec<RemoteFile> = resolve(&lines, &listings)
        .into_iter()
        .filter(|f| !have(&root, f).1)
        .collect();
    let total = todo
        .iter()
        .map(|f| f.size.map(|s| s.saturating_sub(have(&root, f).0)))
        .sum::<Option<u64>>();
    if let (Some(total), Some(free)) = (total, free_bytes(&root)) {
        if total > free {
            return Err(format!(
                "{:.1} GB to download but only {:.1} GB free in {}",
                total as f64 / 1e9,
                free as f64 / 1e9,
                root.display()
            ));
        }
    }
    let _ = on_event.send(DownloadEvent::Start {
        files: todo.len(),
        total_bytes: total,
    });

    let client = http()?;
    let mut done: u64 = 0;
    for f in &todo {
        match fetch(&client, &root, f, &state.cancel, &mut done, on_event).await? {
            Fetched::Complete => {}
            Fetched::Cancelled => {
                let _ = on_event.send(DownloadEvent::Cancelled);
                return Ok(());
            }
        }
    }
    let _ = on_event.send(DownloadEvent::Done {
        models: ids.to_vec(),
    });
    Ok(())
}

enum Fetched {
    Complete,
    Cancelled,
}

/// One file into `<dest>.part`, resumed from what is there, then checked and renamed.
async fn fetch(
    client: &reqwest::Client,
    root: &Path,
    f: &RemoteFile,
    cancel: &AtomicBool,
    done: &mut u64,
    on_event: &Channel<DownloadEvent>,
) -> Result<Fetched, String> {
    let dest = root.join(&f.local);
    let part = part_path(&dest);
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let mut from = file_len(&part);
    if f.size.is_some_and(|s| from > s) {
        from = 0;
    }
    let _ = on_event.send(DownloadEvent::File {
        model: f.model.clone(),
        local: f.local.clone(),
        size: f.size,
        from,
    });

    let complete_already = f.size.is_some_and(|s| s == from && s > 0);
    if !complete_already {
        let mut req = client.get(&f.url);
        if from > 0 {
            req = req.header(reqwest::header::RANGE, format!("bytes={from}-"));
        }
        let mut resp = req
            .send()
            .await
            .and_then(|r| r.error_for_status())
            .map_err(|e| format!("{}: {e}", f.url))?;
        // 206: the range is honoured; 200: the host sends the whole file again.
        let append = from > 0 && resp.status() == reqwest::StatusCode::PARTIAL_CONTENT;
        if !append {
            from = 0;
        }
        let mut out = std::fs::OpenOptions::new()
            .create(true)
            .write(true)
            .append(append)
            .truncate(!append)
            .open(&part)
            .map_err(|e| format!("{}: {e}", part.display()))?;
        let mut file_bytes = from;
        let mut last = Instant::now();
        while let Some(chunk) = resp
            .chunk()
            .await
            .map_err(|e| format!("{}: {e} (rerun to resume)", f.local))?
        {
            out.write_all(&chunk)
                .map_err(|e| format!("{}: {e}", part.display()))?;
            file_bytes += chunk.len() as u64;
            *done += chunk.len() as u64;
            if cancel.load(Ordering::SeqCst) {
                return Ok(Fetched::Cancelled);
            }
            if last.elapsed() >= PROGRESS_EVERY {
                last = Instant::now();
                let _ = on_event.send(DownloadEvent::Progress {
                    done_bytes: *done,
                    file_bytes,
                });
            }
        }
        out.flush().map_err(|e| e.to_string())?;
    }

    let len = file_len(&part);
    if let Some(size) = f.size {
        if len != size {
            return Err(format!(
                "{}: {len} bytes, expected {size} (rerun to resume)",
                f.local
            ));
        }
    }
    if let Some(expected) = &f.sha256 {
        let _ = on_event.send(DownloadEvent::Verifying {
            local: f.local.clone(),
        });
        let path = part.clone();
        let actual = tokio::task::spawn_blocking(move || sha256_file(&path))
            .await
            .map_err(|e| e.to_string())??;
        if !actual.eq_ignore_ascii_case(expected) {
            let _ = std::fs::remove_file(&part);
            return Err(format!(
                "{}: checksum mismatch (got {actual}, expected {expected}); the file was removed, rerun to download it again",
                f.local
            ));
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("{}: {e}", dest.display()))?;
    let _ = on_event.send(DownloadEvent::Progress {
        done_bytes: *done,
        file_bytes: len,
    });
    Ok(Fetched::Complete)
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = file.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Stops the running download at its next chunk.
#[tauri::command]
pub fn models_cancel(state: State<'_, ModelFiles>) {
    state.cancel.store(true, Ordering::SeqCst);
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum MoveEvent {
    #[serde(rename_all = "camelCase")]
    Start { total_bytes: u64 },
    #[serde(rename_all = "camelCase")]
    Progress { done_bytes: u64 },
}

/// Points the app's own config at another models folder (`[server] models_dir`, absolute),
/// optionally moving what the old folder holds. Only the app's own config is written; for any
/// other the folder is the config's business.
#[tauri::command]
pub async fn models_set_dir<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, ModelFiles>,
    config: String,
    dir: String,
    move_files: bool,
    on_event: Channel<MoveEvent>,
) -> Result<(), String> {
    let config = PathBuf::from(config);
    if !crate::local_server::is_managed(&app, &config) {
        return Err("the models folder can be changed for the app's own config only".into());
    }
    if state.busy.load(Ordering::SeqCst) {
        return Err("wait for the download to finish".into());
    }
    let mut doc = read_doc(&config)?;
    let old = models_root(&config, &doc);
    let new = PathBuf::from(dir.trim());
    if !new.is_absolute() {
        return Err("choose a folder (an absolute path)".into());
    }
    std::fs::create_dir_all(&new)
        .map_err(|e| format!("could not create {}: {e}", new.display()))?;
    let same = |a: &Path, b: &Path| match (a.canonicalize(), b.canonicalize()) {
        (Ok(a), Ok(b)) => a == b,
        _ => false,
    };
    if move_files && old.is_dir() && !same(&old, &new) {
        if new.starts_with(&old) {
            return Err("the new folder is inside the old one; choose another".into());
        }
        let (old2, new2) = (old.clone(), new.clone());
        tokio::task::spawn_blocking(move || move_tree(&old2, &new2, &on_event))
            .await
            .map_err(|e| e.to_string())??;
    }
    let server = doc
        .entry("server")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
    server["models_dir"] = toml_edit::value(new.display().to_string());
    std::fs::write(&config, doc.to_string())
        .map_err(|e| format!("could not write {}: {e}", config.display()))
}

/// Moves every entry of `from` into `to`: a rename on the same volume, else a copy (with
/// progress) and a removal. Entries that already exist in `to` are left in `from`.
fn move_tree(from: &Path, to: &Path, on_event: &Channel<MoveEvent>) -> Result<(), String> {
    let entries: Vec<PathBuf> = std::fs::read_dir(from)
        .map_err(|e| e.to_string())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| !to.join(p.file_name().unwrap_or_default()).exists())
        .collect();
    let mut pending = Vec::new();
    for src in entries {
        let dst = to.join(src.file_name().unwrap_or_default());
        if std::fs::rename(&src, &dst).is_err() {
            pending.push((src, dst)); // another volume
        }
    }
    if pending.is_empty() {
        return Ok(());
    }
    let total: u64 = pending.iter().map(|(s, _)| tree_size(s)).sum();
    let _ = on_event.send(MoveEvent::Start { total_bytes: total });
    let mut done = 0u64;
    let mut last = Instant::now();
    for (src, dst) in pending {
        copy_tree(&src, &dst, &mut |n| {
            done += n;
            if last.elapsed() >= PROGRESS_EVERY {
                last = Instant::now();
                let _ = on_event.send(MoveEvent::Progress { done_bytes: done });
            }
        })?;
        if src.is_dir() {
            std::fs::remove_dir_all(&src).map_err(|e| e.to_string())?;
        } else {
            std::fs::remove_file(&src).map_err(|e| e.to_string())?;
        }
    }
    let _ = on_event.send(MoveEvent::Progress { done_bytes: done });
    Ok(())
}

fn tree_size(p: &Path) -> u64 {
    if p.is_dir() {
        std::fs::read_dir(p)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .map(|e| tree_size(&e.path()))
                    .sum()
            })
            .unwrap_or(0)
    } else {
        file_len(p)
    }
}

fn copy_tree(src: &Path, dst: &Path, progress: &mut dyn FnMut(u64)) -> Result<(), String> {
    if src.is_dir() {
        std::fs::create_dir_all(dst).map_err(|e| e.to_string())?;
        for e in std::fs::read_dir(src).map_err(|e| e.to_string())? {
            let e = e.map_err(|e| e.to_string())?;
            copy_tree(&e.path(), &dst.join(e.file_name()), progress)?;
        }
        return Ok(());
    }
    let mut input = std::fs::File::open(src).map_err(|e| format!("{}: {e}", src.display()))?;
    let mut output = std::fs::File::create(dst).map_err(|e| format!("{}: {e}", dst.display()))?;
    let mut buf = vec![0u8; 4 << 20];
    loop {
        let n = input.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        output.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        progress(n as u64);
    }
    output.flush().map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MANIFEST: &str = "# manifest-version: 1\n\
        chat   org/chat-GGUF  chat.gguf      chat.gguf          all\n\
        tts    org/tts        onnx/fp16.onnx tts/model_fp16.onnx other\n\
        tts    org/tts        onnx/fp32.onnx tts/model.onnx      macos-arm64\n\
        tts    org/tts        voices/        tts/voices/         all\n\
        doc    https://github.com/o/r/releases/download/models-v1  layout.onnx  doc/layout.onnx  all\n";

    #[test]
    fn manifest_lines_for_this_platform() {
        let mac = parse_manifest(MANIFEST, true).unwrap();
        assert_eq!(mac.len(), 4);
        assert!(mac.iter().any(|l| l.local == "tts/model.onnx"));
        assert!(!mac.iter().any(|l| l.local == "tts/model_fp16.onnx"));
        let other = parse_manifest(MANIFEST, false).unwrap();
        assert!(other.iter().any(|l| l.local == "tts/model_fp16.onnx"));
        assert!(parse_manifest("# manifest-version: 2\n", true).is_err());
    }

    #[test]
    fn hosts_and_folders_resolve_to_files_with_sizes_and_sums() {
        assert_eq!(
            github_release_api("https://github.com/o/r/releases/download/models-v1").as_deref(),
            Some("https://api.github.com/repos/o/r/releases/tags/models-v1")
        );
        let lines = parse_manifest(MANIFEST, true).unwrap();
        let mut listings = BTreeMap::new();
        listings.insert(
            "org/tts".to_string(),
            Listing::from([
                ("onnx/fp32.onnx".to_string(), (300, Some("ab".to_string()))),
                ("voices/a.bin".to_string(), (10, None)),
                ("voices/b.bin".to_string(), (20, None)),
                ("voicesX/other.bin".to_string(), (1, None)),
            ]),
        );
        listings.insert(
            "https://github.com/o/r/releases/download/models-v1".to_string(),
            Listing::from([("layout.onnx".to_string(), (50, Some("cd".to_string())))]),
        );
        let files = resolve(&lines, &listings);
        let local: Vec<&str> = files.iter().map(|f| f.local.as_str()).collect();
        assert_eq!(
            local,
            [
                "chat.gguf",
                "tts/model.onnx",
                "tts/voices/a.bin",
                "tts/voices/b.bin",
                "doc/layout.onnx"
            ]
        );
        let chat = &files[0];
        assert_eq!(
            chat.url,
            "https://huggingface.co/org/chat-GGUF/resolve/main/chat.gguf"
        );
        assert_eq!(chat.size, None); // no listing for that repo (offline)
        assert_eq!(files[1].sha256.as_deref(), Some("ab"));
        assert_eq!(
            files[2].url,
            "https://huggingface.co/org/tts/resolve/main/voices/a.bin"
        );
        assert_eq!(
            files[4].url,
            "https://github.com/o/r/releases/download/models-v1/layout.onnx"
        );
    }

    /// Against the real hosts: `cargo test -- --ignored real_manifest`.
    #[tokio::test]
    #[ignore]
    async fn real_manifest_lists_every_file_and_resumes_a_verified_download() {
        let text = std::fs::read_to_string(manifest_path().unwrap()).unwrap();
        let lines = parse_manifest(&text, true).unwrap();
        let state = ModelFiles::default();
        let (listings, errors) = listings(&state, &lines).await;
        assert!(errors.is_empty(), "{errors:?}");
        let files = resolve(&lines, &listings);
        let no_size: Vec<&str> = files
            .iter()
            .filter(|f| f.size.is_none())
            .map(|f| f.local.as_str())
            .collect();
        assert!(no_size.is_empty(), "no size for {no_size:?}");
        let total: u64 = files.iter().filter_map(|f| f.size).sum();
        let with_sum = files.iter().filter(|f| f.sha256.is_some()).count();
        println!(
            "{} files, {:.1} GB, {with_sum} with a SHA-256",
            files.len(),
            total as f64 / 1e9
        );

        // The smallest file with a sum: half of it as a leftover .part, then a resumed fetch.
        let f = files
            .iter()
            .filter(|f| f.sha256.is_some() && f.size.unwrap_or(0) > 1000)
            .min_by_key(|f| f.size)
            .unwrap()
            .clone();
        let root = std::env::temp_dir().join(format!("hypnos-dl-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        let client = http().unwrap();
        let whole = client
            .get(&f.url)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        let part = part_path(&root.join(&f.local));
        std::fs::create_dir_all(part.parent().unwrap()).unwrap();
        std::fs::write(&part, &whole[..whole.len() / 2]).unwrap();
        let mut done = 0;
        let cancel = AtomicBool::new(false);
        let r = fetch(
            &client,
            &root,
            &f,
            &cancel,
            &mut done,
            &Channel::new(|_| Ok(())),
        )
        .await
        .unwrap();
        assert!(matches!(r, Fetched::Complete));
        assert_eq!(
            done,
            (whole.len() - whole.len() / 2) as u64,
            "resumed, not restarted"
        );
        assert_eq!(std::fs::read(root.join(&f.local)).unwrap(), whole);
        assert!(!part.exists());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn models_root_follows_models_dir_or_the_old_layout() {
        let config = Path::new("/home/u/Hypnos/hypnos.toml");
        let doc: toml_edit::DocumentMut = "[server]\nmodels_dir = \"models\"\n".parse().unwrap();
        assert_eq!(
            models_root(config, &doc),
            PathBuf::from("/home/u/Hypnos/models")
        );
        let doc: toml_edit::DocumentMut = "[server]\nmodels_dir = \"/Volumes/SSD/m\"\n"
            .parse()
            .unwrap();
        assert_eq!(models_root(config, &doc), PathBuf::from("/Volumes/SSD/m"));
        let doc: toml_edit::DocumentMut = "[server]\nport = 1\n".parse().unwrap();
        assert_eq!(
            models_root(config, &doc),
            PathBuf::from("/home/u/Hypnos/models")
        );
    }

    #[test]
    fn moves_a_tree_and_keeps_what_the_target_has() {
        let base = std::env::temp_dir().join(format!("hypnos-move-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (from, to) = (base.join("old"), base.join("new"));
        std::fs::create_dir_all(from.join("kokoro/voices")).unwrap();
        std::fs::create_dir_all(&to).unwrap();
        std::fs::write(from.join("chat.gguf"), b"weights").unwrap();
        std::fs::write(from.join("kokoro/voices/a.bin"), b"v").unwrap();
        std::fs::write(from.join("kept.gguf"), b"old").unwrap();
        std::fs::write(to.join("kept.gguf"), b"new").unwrap();
        let mut copied = 0;
        copy_tree(&from.join("kokoro"), &base.join("copy"), &mut |n| {
            copied += n
        })
        .unwrap();
        assert_eq!(copied, 1);
        assert!(base.join("copy/voices/a.bin").is_file());
        assert_eq!(tree_size(&from), 7 + 1 + 3);

        move_tree(&from, &to, &Channel::new(|_| Ok(()))).unwrap();
        assert!(to.join("chat.gguf").is_file());
        assert!(to.join("kokoro/voices/a.bin").is_file());
        assert!(!from.join("chat.gguf").exists());
        // An entry the target already has is not overwritten; the old copy stays behind.
        assert_eq!(std::fs::read(to.join("kept.gguf")).unwrap(), b"new");
        assert!(from.join("kept.gguf").is_file());
        let _ = std::fs::remove_dir_all(&base);
    }
}
