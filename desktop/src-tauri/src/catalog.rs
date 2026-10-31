//! The model catalog in the Models dialog (API-10, API-11): the curated models of
//! `models.catalog.toml` with their quantization variants, each marked as in the server's model
//! pool or not.
//!
//! With the app connected to a server that uses this config's models folder (the local one),
//! the pool is the server's (`GET /v1/models/catalog`, `configured`), which includes models added
//! while it ran, and changes go to it live: Add and Replace download the variant's missing files
//! with the lineup's downloader (`models.rs`) and call `POST /v1/models` / `PUT /v1/models/{id}`,
//! Remove calls `DELETE /v1/models/{id}`. The server keeps them in its overlay
//! (`<config stem>.models.toml`); nothing here writes the config. Without such a server the pool
//! is read from the config and that overlay, and Add on the app's own config falls back to
//! appending the block to the config (`models_catalog_add`), after which the caller starts the
//! server; another config gets the block to copy (`models_catalog_block`).
//!
//! The catalog is read here, not from the server, so the dialog works while the server is
//! stopped: the bundled copy (`Contents/Resources/server/models.catalog.toml`, staged by
//! `stage-server.sh`), else this checkout's `backend/models.catalog.toml`. Its files (repo,
//! remote, local, size, platform) come from this parse: the picker needs sizes and folder
//! entries for every variant up front, which `hypnos --catalog-files` does not print and which
//! would take a process per entry. The `[[models]]` block comes from the server binary
//! (`hypnos --catalog-block <id>:<quant>`), so its rendering stays in one place
//! (`hypnos_core::catalog::CatalogEntry::render_block`).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use hypnos_client::{Client, Error, ModelCatalog, ModelPoolChange, ModelSource};
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::{AppHandle, Runtime, State};

use crate::hypnos::{ErrorPayload, Hypnos};

use crate::models::{
    download_lines, file_len, models_root, part_path, read_doc, tree_size, DownloadEvent,
    ManifestLine, ModelFiles,
};

const FILE_NAME: &str = "models.catalog.toml";
/// The layout version this app reads; a newer catalog is refused.
const CATALOG_VERSION: u32 = 1;
const RENDER_TIMEOUT: Duration = Duration::from_secs(30);
/// How long the catalog request that tells whether the server is there may take.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
/// How long a replacement or removal waits for running requests before the server refuses it
/// (429 `model_busy`, nothing changed); the dialog waits that long at most.
const DRAIN_SECS: u64 = 30;

#[derive(Debug, Deserialize)]
struct Catalog {
    catalog_version: u32,
    #[serde(default)]
    models: Vec<Entry>,
}

/// One `[[models]]` of the catalog; `config` and `architecture` are the server's business.
#[derive(Debug, Deserialize)]
struct Entry {
    id: String,
    name: String,
    publisher: String,
    license: String,
    modality: String,
    #[serde(default)]
    additional_modalities: Vec<String>,
    engine: String,
    #[serde(default)]
    capabilities: Vec<String>,
    summary: String,
    #[serde(default)]
    lineup: bool,
    variants: Vec<Variant>,
}

#[derive(Debug, Deserialize)]
struct Variant {
    quant: String,
    #[serde(default)]
    default: bool,
    #[serde(default)]
    note: Option<String>,
    files: Vec<CatalogFile>,
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    /// `weights`, `aux`, ... (the config key the file fills).
    #[serde(default)]
    role: String,
    repo: String,
    /// A trailing `/` is a whole folder.
    remote: String,
    local: String,
    /// Bytes (a folder: all of its files).
    size: u64,
    /// `all`, `macos-arm64` or `other`, as in `models.manifest`.
    #[serde(default = "all")]
    platform: String,
}

fn all() -> String {
    "all".into()
}

fn parse_catalog(text: &str) -> Result<Catalog, String> {
    let catalog: Catalog = toml::from_str(text).map_err(|e| format!("{FILE_NAME}: {e}"))?;
    if catalog.catalog_version > CATALOG_VERSION {
        return Err(format!(
            "{FILE_NAME} version {} is newer than this app understands",
            catalog.catalog_version
        ));
    }
    Ok(catalog)
}

fn for_platform(platform: &str, apple_silicon: bool) -> bool {
    match platform {
        "all" => true,
        "macos-arm64" => apple_silicon,
        "other" => !apple_silicon,
        _ => false,
    }
}

const APPLE_SILICON: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

impl Variant {
    fn files_for(&self, apple_silicon: bool) -> impl Iterator<Item = &CatalogFile> {
        self.files
            .iter()
            .filter(move |f| for_platform(&f.platform, apple_silicon))
    }

    /// The files as manifest lines for the downloader.
    fn lines(&self, id: &str, apple_silicon: bool) -> Vec<ManifestLine> {
        self.files_for(apple_silicon)
            .map(|f| ManifestLine {
                model: id.to_string(),
                repo: f.repo.clone(),
                remote: f.remote.clone(),
                local: f.local.clone(),
            })
            .collect()
    }

    /// Bytes on disk under `root` (complete files and `.part`s) and whether every file is whole.
    fn state(&self, root: &Path, apple_silicon: bool) -> (u64, bool) {
        let mut have = 0;
        let mut whole = true;
        for f in self.files_for(apple_silicon) {
            let dest = root.join(&f.local);
            let (n, complete) = if f.remote.ends_with('/') {
                let n = tree_size(&dest);
                (n, dest.is_dir() && !has_part(&dest) && n >= f.size)
            } else if dest.is_file() {
                let n = file_len(&dest);
                (n, n == f.size)
            } else {
                (file_len(&part_path(&dest)), false)
            };
            have += n;
            whole &= complete;
        }
        (have, whole)
    }
}

fn has_part(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                let p = e.path();
                if p.is_dir() {
                    has_part(&p)
                } else {
                    p.extension().is_some_and(|x| x == "part")
                }
            })
        })
        .unwrap_or(false)
}

impl Entry {
    fn variant(&self, quant: &str) -> Option<&Variant> {
        self.variants
            .iter()
            .find(|v| v.quant.eq_ignore_ascii_case(quant))
    }

    /// The variant whose weights file a block's `weights_path` names.
    fn quant_of(&self, weights: &str, apple_silicon: bool) -> Option<&str> {
        let weights = Path::new(weights.trim_end_matches('/'));
        self.variants
            .iter()
            .find(|v| {
                v.files_for(apple_silicon).any(|f| {
                    f.role == "weights" && weights.ends_with(f.local.trim_end_matches('/'))
                })
            })
            .map(|v| v.quant.as_str())
    }
}

/// `models.catalog.toml`: the bundled copy, else this checkout's (as `models::manifest_path`).
fn catalog_path() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let bundled = exe_dir.parent()?.join("Resources/server").join(FILE_NAME);
    if bundled.is_file() {
        return Some(bundled);
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../backend")
        .join(FILE_NAME);
    repo.is_file().then_some(repo)
}

fn load_catalog() -> Result<(PathBuf, Catalog), String> {
    let path = catalog_path().ok_or(format!("no {FILE_NAME} in the app or this checkout"))?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok((path, parse_catalog(&text)?))
}

/// The ids of the config's `[[models]]` blocks.
fn config_ids(doc: &toml_edit::DocumentMut) -> BTreeSet<String> {
    doc.get("models")
        .and_then(|m| m.as_array_of_tables())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("id")?.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// A model of the pool: whether it is enabled, and the weights file its block names (when the
/// block is known here: in the config or its overlay).
#[derive(Debug, Clone, PartialEq)]
struct PoolModel {
    enabled: bool,
    weights: Option<String>,
}

fn pool_model(enabled: Option<bool>, weights: Option<&str>) -> PoolModel {
    PoolModel {
        enabled: enabled.unwrap_or(true),
        weights: weights.map(String::from),
    }
}

/// The server's overlay for a config: `<folder>/<config stem>.models.toml` (as the server's
/// `pool::overlay_path_for`).
fn overlay_path(config: &Path) -> PathBuf {
    let stem = config
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "hypnos".into());
    config.with_file_name(format!("{stem}.models.toml"))
}

/// The overlay the server writes: ids taken out of the config, blocks added or replacing one.
#[derive(Debug, Default, Deserialize)]
struct Overlay {
    #[serde(default)]
    removed: Vec<String>,
    #[serde(default)]
    models: Vec<toml::Table>,
}

/// The config's models, then the overlay applied over them as the server does at startup.
fn pool_of(doc: &toml_edit::DocumentMut, overlay: Option<&str>) -> BTreeMap<String, PoolModel> {
    let mut pool: BTreeMap<String, PoolModel> = doc
        .get("models")
        .and_then(|m| m.as_array_of_tables())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| {
                    let id = m.get("id")?.as_str()?.to_string();
                    let model = pool_model(
                        m.get("enabled").and_then(|v| v.as_bool()),
                        m.get("weights_path").and_then(|v| v.as_str()),
                    );
                    Some((id, model))
                })
                .collect()
        })
        .unwrap_or_default();
    let overlay = match overlay.map(toml::from_str::<Overlay>) {
        Some(Ok(o)) => o,
        Some(Err(e)) => {
            eprintln!("model pool overlay: {e}");
            return pool;
        }
        None => return pool,
    };
    for id in &overlay.removed {
        pool.remove(id);
    }
    for t in &overlay.models {
        if let Some(id) = t.get("id").and_then(|v| v.as_str()) {
            let model = pool_model(
                t.get("enabled").and_then(|v| v.as_bool()),
                t.get("weights_path").and_then(|v| v.as_str()),
            );
            pool.insert(id.to_string(), model);
        }
    }
    pool
}

/// The pool as the config and its overlay describe it (what the server starts with).
fn file_pool(config: &Path, doc: &toml_edit::DocumentMut) -> BTreeMap<String, PoolModel> {
    let overlay = std::fs::read_to_string(overlay_path(config)).ok();
    pool_of(doc, overlay.as_deref())
}

/// The running server's pool: the catalog entries it has configured, with the weights the files
/// know of for them.
fn live_pool(
    catalog: &ModelCatalog,
    files: &BTreeMap<String, PoolModel>,
) -> BTreeMap<String, PoolModel> {
    catalog
        .data
        .iter()
        .filter(|e| e.configured)
        .map(|e| {
            let weights = files.get(&e.id).and_then(|m| m.weights.as_deref());
            (e.id.clone(), pool_model(e.enabled, weights))
        })
        .collect()
}

fn same_dir(a: &Path, b: &Path) -> bool {
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    canon(a) == canon(b)
}

/// The connected server and its catalog when it serves the models under `root` (files the app
/// downloads there are the ones it checks), else why the dialog works from the files instead.
async fn live_server(hypnos: &Hypnos, root: &Path) -> Result<(Arc<Client>, ModelCatalog), String> {
    let client = hypnos.client().await.map_err(|e| e.message)?;
    let catalog = tokio::time::timeout(PROBE_TIMEOUT, client.models_catalog(None))
        .await
        .map_err(|_| "the server did not answer".to_string())?
        .map_err(|e| match e {
            Error::Connection(_) | Error::Timeout(_) => "the server is not running".to_string(),
            Error::NotFound(_) => "the server has no model catalog (an older server?)".into(),
            other => ErrorPayload::from(other).message,
        })?;
    if !same_dir(Path::new(&catalog.models_dir), root) {
        return Err(format!(
            "the connected server uses another models folder ({})",
            catalog.models_dir
        ));
    }
    Ok((client, catalog))
}

/// An error of a pool change as the dialog shows it.
fn pool_error(e: Error) -> ErrorPayload {
    let (kind, message) = match &e {
        Error::PermissionDenied(_) => (
            "permission",
            "This server needs an admin key for that. The app keeps one API key per server: \
             enter an admin key in Settings."
                .to_string(),
        ),
        Error::ModelBusy(_) => (
            "busy",
            "Requests were still running on the model; nothing changed. Try again when they \
             finish."
                .to_string(),
        ),
        Error::BadRequest(api) => ("bad_request", api.message.clone()),
        _ => return e.into(),
    };
    ErrorPayload {
        kind: kind.into(),
        message,
        detail: Some(e.to_string()),
        retry_after_secs: None,
    }
}

/// The offline Add appends to the config, so the model must be in neither the config nor the
/// overlay; one the overlay removed would be removed again at the next start.
fn offline_addable(
    pool: &BTreeMap<String, PoolModel>,
    config_ids: &BTreeSet<String>,
    id: &str,
) -> Result<(), String> {
    if pool.contains_key(id) {
        return Err(format!("\"{id}\" is already in the model pool"));
    }
    if config_ids.contains(id) {
        return Err(format!(
            "\"{id}\" was removed while the server ran; start the server and add it from there"
        ));
    }
    Ok(())
}

/// `text` with `block` (one `[[models]]` with `id`) appended; everything already there stays as
/// written. Refused when the config has a block with that id, or when the result would not hold
/// exactly one more model.
fn append_block(text: &str, block: &str, id: &str) -> Result<String, String> {
    let doc: toml_edit::DocumentMut = text.parse().map_err(|e| format!("the config: {e}"))?;
    let before = config_ids(&doc);
    if before.contains(id) {
        return Err(format!(
            "the config already has a [[models]] block with id \"{id}\""
        ));
    }
    let parsed: toml_edit::DocumentMut = block
        .parse()
        .map_err(|e| format!("the catalog block: {e}"))?;
    let block_ids: Vec<String> = config_ids(&parsed).into_iter().collect();
    if block_ids != [id] {
        return Err(format!(
            "the catalog block is not one [[models]] block for \"{id}\""
        ));
    }
    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(block.trim_end());
    out.push('\n');
    let after: toml_edit::DocumentMut = out
        .parse()
        .map_err(|e| format!("the config with the new block does not parse: {e}"))?;
    let count = after
        .get("models")
        .and_then(|m| m.as_array_of_tables())
        .map_or(0, |m| m.len());
    if count != before.len() + 1 || !config_ids(&after).contains(id) {
        return Err("the config's [[models]] are not an array of tables".into());
    }
    Ok(out)
}

/// The variant's `[[models]]` block from `hypnos --catalog-block <id>:<quant>`, run against the
/// catalog the app read.
async fn render_block(binary: &Path, catalog: &Path, spec: &str) -> Result<String, String> {
    if !binary.is_file() {
        return Err(format!(
            "no server binary at {}: the config block comes from `hypnos --catalog-block`",
            binary.display()
        ));
    }
    let mut cmd = tokio::process::Command::new(binary);
    cmd.arg("--catalog-block")
        .arg(spec)
        .env("HYPNOS_CATALOG", catalog)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    if let Some(dir) = catalog.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(target_os = "macos")]
    if let Some(dir) = binary.parent() {
        cmd.env(
            "DYLD_LIBRARY_PATH",
            crate::local_server::join_paths(dir, "DYLD_LIBRARY_PATH"),
        );
    }
    #[cfg(target_os = "linux")]
    if let Some(dir) = binary.parent() {
        cmd.env(
            "LD_LIBRARY_PATH",
            crate::local_server::join_paths(dir, "LD_LIBRARY_PATH"),
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = tokio::time::timeout(RENDER_TIMEOUT, cmd.output())
        .await
        .map_err(|_| format!("{} --catalog-block timed out", binary.display()))?
        .map_err(|e| format!("could not run {}: {e}", binary.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim();
        return Err(
            if err.contains("--catalog-block") || err.contains("usage") {
                format!("{err} (a server too old for the catalog?)")
            } else {
                format!("hypnos --catalog-block {spec}: {err}")
            },
        );
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VariantView {
    quant: String,
    default: bool,
    note: Option<String>,
    /// Bytes of this machine's files, as the catalog lists them.
    size: u64,
    have: u64,
    installed: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryView {
    id: String,
    name: String,
    publisher: String,
    license: String,
    modality: String,
    additional_modalities: Vec<String>,
    engine: String,
    capabilities: Vec<String>,
    summary: String,
    lineup: bool,
    variants: Vec<VariantView>,
    /// The server's pool has a model with this id.
    in_pool: bool,
    /// Set only when `in_pool`.
    enabled: Option<bool>,
    /// The variant the pool's block uses, when its weights file is one of the catalog's.
    current_quant: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CatalogView {
    /// The app is connected to a server that uses this config's models folder: the pool is its
    /// own, and Add, Replace and Remove change it live.
    live: bool,
    /// Why not, when not.
    offline: Option<String>,
    entries: Vec<EntryView>,
}

/// The catalog's models, each marked as in the pool or not, with each variant's size and what is
/// on disk under the config's models folder. The pool is the running server's when the app is
/// connected to one that uses this folder, else the config's with its overlay.
#[tauri::command]
pub async fn models_catalog_entries(
    hypnos: State<'_, Hypnos>,
    config: String,
) -> Result<CatalogView, String> {
    let config = PathBuf::from(config);
    let doc = read_doc(&config)?;
    let root = models_root(&config, &doc);
    let files = file_pool(&config, &doc);
    let (_, catalog) = load_catalog()?;
    let (pool, offline) = match live_server(&hypnos, &root).await {
        Ok((_, server)) => (live_pool(&server, &files), None),
        Err(why) => (files, Some(why)),
    };
    Ok(CatalogView {
        live: offline.is_none(),
        offline,
        entries: entries_view(catalog, &pool, &root, APPLE_SILICON),
    })
}

fn entries_view(
    catalog: Catalog,
    pool: &BTreeMap<String, PoolModel>,
    root: &Path,
    apple_silicon: bool,
) -> Vec<EntryView> {
    catalog
        .models
        .into_iter()
        .map(|e| {
            let variants = e
                .variants
                .iter()
                .map(|v| {
                    let (have, installed) = v.state(root, apple_silicon);
                    VariantView {
                        quant: v.quant.clone(),
                        default: v.default,
                        note: v.note.clone(),
                        size: v.files_for(apple_silicon).map(|f| f.size).sum(),
                        have,
                        installed,
                    }
                })
                .collect();
            let pooled = pool.get(&e.id);
            let current_quant = pooled
                .and_then(|m| m.weights.as_deref())
                .and_then(|w| e.quant_of(w, apple_silicon))
                .map(String::from);
            EntryView {
                in_pool: pooled.is_some(),
                enabled: pooled.map(|m| m.enabled),
                current_quant,
                id: e.id,
                name: e.name,
                publisher: e.publisher,
                license: e.license,
                modality: e.modality,
                additional_modalities: e.additional_modalities,
                engine: e.engine,
                capabilities: e.capabilities,
                summary: e.summary,
                lineup: e.lineup,
                variants,
            }
        })
        .collect()
}

fn resolve<'a>(
    catalog: &'a Catalog,
    id: &str,
    quant: &str,
) -> Result<(&'a Entry, &'a Variant), String> {
    let entry = catalog
        .models
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| format!("no model \"{id}\" in the catalog"))?;
    let variant = entry
        .variant(quant)
        .ok_or_else(|| format!("\"{id}\" has no variant \"{quant}\""))?;
    Ok((entry, variant))
}

/// The `[[models]]` block of a variant, for a config the app does not write (copy it there).
#[tauri::command]
pub async fn models_catalog_block(
    binary: Option<String>,
    id: String,
    quant: String,
) -> Result<String, String> {
    let (path, catalog) = load_catalog()?;
    let (entry, variant) = resolve(&catalog, &id, &quant)?;
    let binary = binary
        .filter(|b| !b.is_empty())
        .ok_or("no server binary chosen: the config block comes from `hypnos --catalog-block`")?;
    render_block(
        Path::new(&binary),
        &path,
        &format!("{}:{}", entry.id, variant.quant),
    )
    .await
}

/// The offline Add (no server serving this config's models folder): adds a catalog model to the
/// app's own config by downloading the variant's missing files, then appending its block.
/// `Ok(false)`: the download was stopped and nothing was added. The caller starts the server
/// afterwards, as after a lineup download. With a server, `models_pool_put` adds it live.
#[tauri::command]
pub async fn models_catalog_add<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, ModelFiles>,
    config: String,
    binary: Option<String>,
    id: String,
    quant: String,
    on_event: Channel<DownloadEvent>,
) -> Result<bool, String> {
    let config = PathBuf::from(config);
    if !crate::local_server::is_managed(&app, &config) {
        return Err(
            "models can be added to the app's own config only; copy the block into this one".into(),
        );
    }
    let doc = read_doc(&config)?;
    if doc
        .get("server")
        .and_then(|s| s.get("models_dir"))
        .is_none()
    {
        return Err("the config has no [server] models_dir, which the catalog's paths are relative to; choose a models folder first".into());
    }
    offline_addable(&file_pool(&config, &doc), &config_ids(&doc), &id)?;
    let root = models_root(&config, &doc);
    let (path, catalog) = load_catalog()?;
    let (entry, variant) = resolve(&catalog, &id, &quant)?;
    let binary = binary
        .filter(|b| !b.is_empty())
        .ok_or("no server binary chosen: the config block comes from `hypnos --catalog-block`")?;
    let block = render_block(
        Path::new(&binary),
        &path,
        &format!("{}:{}", entry.id, variant.quant),
    )
    .await?;
    // Everything that could refuse the block is checked before the download.
    let read = |p: &Path| {
        std::fs::read_to_string(p).map_err(|e| format!("could not read {}: {e}", p.display()))
    };
    append_block(&read(&config)?, &block, &entry.id)?;

    let _guard = state.exclusive()?;
    let lines = variant.lines(&entry.id, APPLE_SILICON);
    if !download_lines(
        &state,
        &root,
        &lines,
        std::slice::from_ref(&entry.id),
        &on_event,
    )
    .await?
    {
        return Ok(false);
    }
    let text = append_block(&read(&config)?, &block, &entry.id)?;
    std::fs::write(&config, text)
        .map_err(|e| format!("could not write {}: {e}", config.display()))?;
    Ok(true)
}

/// Adds a catalog variant to the running server's pool (`replace`: puts it in place of the
/// pool's model with that id, which keeps the id): downloads its missing files into the models
/// folder, then `POST /v1/models` or `PUT /v1/models/{id}`. `Ok(None)`: the download was stopped
/// and nothing changed. The server needs no restart; it loads the model on its first request.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn models_pool_put(
    state: State<'_, ModelFiles>,
    hypnos: State<'_, Hypnos>,
    config: String,
    id: String,
    quant: String,
    replace: bool,
    on_event: Channel<DownloadEvent>,
) -> Result<Option<ModelPoolChange>, ErrorPayload> {
    let config = PathBuf::from(config);
    let doc = read_doc(&config)?;
    let root = models_root(&config, &doc);
    let (_, catalog) = load_catalog()?;
    let (entry, variant) = resolve(&catalog, &id, &quant)?;
    let (client, _) = live_server(&hypnos, &root)
        .await
        .map_err(|why| format!("The model pool cannot be changed: {why}."))?;
    if !variant.state(&root, APPLE_SILICON).1 {
        let _guard = state.exclusive()?;
        let lines = variant.lines(&entry.id, APPLE_SILICON);
        let ids = std::slice::from_ref(&entry.id);
        if !download_lines(&state, &root, &lines, ids, &on_event).await? {
            return Ok(None);
        }
    }
    let source = ModelSource::catalog(format!("{}:{}", entry.id, variant.quant));
    let change = if replace {
        client
            .replace_model(&entry.id, &source, Some(DRAIN_SECS))
            .await
    } else {
        client.add_model(&source).await
    };
    change.map(Some).map_err(pool_error)
}

/// Takes a model out of the running server's pool (`DELETE /v1/models/{id}`); running requests
/// get [`DRAIN_SECS`] to finish, else nothing changes. Its files stay on disk.
#[tauri::command]
pub async fn models_pool_remove(
    hypnos: State<'_, Hypnos>,
    id: String,
) -> Result<ModelPoolChange, ErrorPayload> {
    hypnos
        .client()
        .await?
        .remove_model(&id, Some(DRAIN_SECS))
        .await
        .map_err(pool_error)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"
catalog_version = 1

[[models]]
id = "chat:1b"
name = "Chat 1B"
publisher = "Org"
license = "mit"
architecture = "lfm3"
modality = "text_generation"
additional_modalities = ["vision"]
engine = "ggml_llama"
capabilities = ["thinking", "tools"]
summary = "A chat model."

[models.config]
priority = "high"

[models.config.arch]
default_context_len = 4096

[[models.variants]]
quant = "Q8_0"
default = true
files = [
  { role = "weights", repo = "org/chat", remote = "chat-Q8_0.gguf", local = "chat-Q8_0.gguf", size = 10 },
  { role = "aux", repo = "org/chat", remote = "mmproj.gguf", local = "mmproj.gguf", size = 5 },
]

[[models.variants]]
quant = "Q4_K_M"
note = "Smaller."
files = [
  { role = "weights", repo = "org/chat", remote = "chat-Q4.gguf", local = "chat-Q4.gguf", size = 6 },
]

[[models]]
id = "tts"
name = "TTS"
publisher = "Org"
license = "apache-2.0"
architecture = "kokoro"
modality = "text_to_speech"
engine = "onnx_runtime"
summary = "Speech."
lineup = true

[[models.variants]]
quant = "fp32"
default = true
files = [
  { role = "weights", repo = "org/tts", remote = "onnx/model.onnx", local = "tts/model.onnx", size = 4, platform = "macos-arm64" },
  { role = "weights", repo = "org/tts", remote = "onnx/fp16.onnx", local = "tts/fp16.onnx", size = 2, platform = "other" },
  { role = "voices", repo = "org/tts", remote = "voices/", local = "tts/voices/", size = 3 },
]
"#;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hypnos-cat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_entries_and_refuses_a_newer_catalog() {
        let c = parse_catalog(CATALOG).unwrap();
        assert_eq!(c.models.len(), 2);
        let chat = &c.models[0];
        assert_eq!(chat.capabilities, ["thinking", "tools"]);
        assert_eq!(chat.additional_modalities, ["vision"]);
        assert_eq!(chat.variants.len(), 2);
        assert!(chat.variant("q4_k_m").is_some());
        assert_eq!(
            chat.variant("Q4_K_M").unwrap().note.as_deref(),
            Some("Smaller.")
        );
        assert!(c.models[1].lineup);
        assert!(parse_catalog("catalog_version = 2\n").is_err());
        assert!(resolve(&c, "chat:1b", "Q2").is_err());
        assert!(resolve(&c, "nope", "Q8_0").is_err());
    }

    #[test]
    fn the_real_catalog_parses_with_one_default_per_entry() {
        let (_, c) = load_catalog().unwrap();
        assert!(!c.models.is_empty());
        for e in &c.models {
            assert_eq!(
                e.variants.iter().filter(|v| v.default).count(),
                1,
                "{}",
                e.id
            );
            for v in &e.variants {
                assert!(v.files_for(true).count() > 0 && v.files_for(false).count() > 0);
            }
        }
    }

    #[test]
    fn picks_this_platforms_files() {
        let c = parse_catalog(CATALOG).unwrap();
        let tts = &c.models[1].variants[0];
        let mac: Vec<&str> = tts.files_for(true).map(|f| f.local.as_str()).collect();
        assert_eq!(mac, ["tts/model.onnx", "tts/voices/"]);
        let other: Vec<&str> = tts.files_for(false).map(|f| f.local.as_str()).collect();
        assert_eq!(other, ["tts/fp16.onnx", "tts/voices/"]);
        let lines = tts.lines("tts", true);
        assert_eq!(lines[1].remote, "voices/");
        assert!(lines.iter().all(|l| l.model == "tts"));
    }

    #[test]
    fn lists_the_catalog_with_the_pool_and_what_is_on_disk() {
        let root = scratch("state");
        let c = parse_catalog(CATALOG).unwrap();
        // Q8_0: weights complete, projector half there as a .part.
        std::fs::write(root.join("chat-Q8_0.gguf"), [0u8; 10]).unwrap();
        std::fs::write(root.join("mmproj.gguf.part"), [0u8; 2]).unwrap();
        // Q4_K_M: whole. tts: model and the voices folder whole.
        std::fs::write(root.join("chat-Q4.gguf"), [0u8; 6]).unwrap();
        std::fs::create_dir_all(root.join("tts/voices")).unwrap();
        std::fs::write(root.join("tts/model.onnx"), [0u8; 4]).unwrap();
        std::fs::write(root.join("tts/voices/a.bin"), [0u8; 3]).unwrap();

        let none = BTreeMap::new();
        let view = entries_view(c, &none, &root, true);
        let chat = &view[0];
        assert!(!chat.in_pool && chat.enabled.is_none());
        assert_eq!((chat.variants[0].size, chat.variants[0].have), (15, 12));
        assert!(!chat.variants[0].installed);
        assert!(chat.variants[1].installed);
        assert!(view[1].variants[0].installed);

        // A .part inside a folder: not whole yet.
        std::fs::write(root.join("tts/voices/b.bin.part"), [0u8; 1]).unwrap();
        let view = entries_view(parse_catalog(CATALOG).unwrap(), &none, &root, true);
        assert!(!view[1].variants[0].installed);

        // Models of the pool are listed too, with the variant their block uses.
        let pool = BTreeMap::from([(
            "chat:1b".to_string(),
            pool_model(Some(false), Some("chat-Q4.gguf")),
        )]);
        let view = entries_view(parse_catalog(CATALOG).unwrap(), &pool, &root, true);
        assert_eq!(view.len(), 2);
        assert!(view[0].in_pool);
        assert_eq!(view[0].enabled, Some(false));
        assert_eq!(view[0].current_quant.as_deref(), Some("Q4_K_M"));
        assert!(!view[1].in_pool);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_pool_is_the_config_with_its_overlay() {
        let doc: toml_edit::DocumentMut = "[[models]]\nid = \"a\"\nweights_path = \"a.gguf\"\n\n\
             [[models]]\nid = \"b\"\nenabled = false\nweights_path = \"b.gguf\"\n\n\
             [[models]]\nid = \"c\"\nweights_path = \"c.gguf\"\n"
            .parse()
            .unwrap();
        let ids = |p: &BTreeMap<String, PoolModel>| p.keys().cloned().collect::<Vec<_>>();
        assert_eq!(ids(&pool_of(&doc, None)), ["a", "b", "c"]);
        assert!(!pool_of(&doc, None)["b"].enabled);

        // As the server writes it: c removed, a replaced, d added.
        let overlay = "config_version = 1\nremoved = [\"c\"]\n\n\
             [[models]]\nid = \"a\"\nweights_path = \"a-Q4.gguf\"\n\n\
             [[models]]\nid = \"d\"\nweights_path = \"d.gguf\"\n\n[models.arch]\n";
        let pool = pool_of(&doc, Some(overlay));
        assert_eq!(ids(&pool), ["a", "b", "d"]);
        assert_eq!(pool["a"].weights.as_deref(), Some("a-Q4.gguf"));
        // A broken overlay leaves the config's pool.
        assert_eq!(ids(&pool_of(&doc, Some("removed = 3"))), ["a", "b", "c"]);

        // The offline Add refuses what is in the pool and what the overlay removed.
        let config_ids = config_ids(&doc);
        assert!(offline_addable(&pool, &config_ids, "e").is_ok());
        assert!(offline_addable(&pool, &config_ids, "d")
            .unwrap_err()
            .contains("already"));
        assert!(offline_addable(&pool, &config_ids, "c")
            .unwrap_err()
            .contains("removed"));

        assert_eq!(
            overlay_path(Path::new("/x/hypnos.lfm3.toml")),
            Path::new("/x/hypnos.lfm3.models.toml")
        );
    }

    #[test]
    fn pool_errors_read_as_sentences() {
        let api = |status: u16, kind: &str, message: &str| {
            Box::new(hypnos_client::ApiError {
                status,
                message: message.into(),
                kind: Some(kind.into()),
                code: None,
                request_id: None,
                retry_after: None,
                body: None,
            })
        };
        let e = pool_error(Error::PermissionDenied(api(403, "permission_denied", "no")));
        assert!(e.message.contains("admin key"), "{}", e.message);
        let e = pool_error(Error::ModelBusy(api(429, "model_busy", "busy")));
        assert!(e.message.contains("nothing changed"), "{}", e.message);
        let e = pool_error(Error::BadRequest(api(
            400,
            "invalid_request_error",
            "weights not found: x.gguf",
        )));
        assert_eq!(e.message, "weights not found: x.gguf");
        assert_eq!(e.kind, "bad_request");
    }

    const CONFIG: &str = "config_version = 1\n\n[server]\nmodels_dir = \"models\" # where they live\n\n[[models]]\nid = \"a\"\nweights_path = \"a.gguf\"\n\n[models.arch]\nnum_layers = 2\n";
    const BLOCK: &str = "# B (Q8_0). From models.catalog.toml.\n[[models]]\nid = \"b\"\nweights_path = \"b.gguf\"\n\n[models.arch]\ndefault_context_len = 4096\n";

    #[test]
    fn appends_a_block_and_keeps_the_rest_of_the_file() {
        let out = append_block(CONFIG, BLOCK, "b").unwrap();
        assert!(
            out.starts_with(CONFIG.trim_end()),
            "the old text is unchanged"
        );
        assert!(out.ends_with(BLOCK));
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        let models = doc["models"].as_array_of_tables().unwrap();
        assert_eq!(models.len(), 2);
        // `[models.arch]` of the new block belongs to the new model, not to `a`.
        assert_eq!(
            models.get(0).unwrap()["arch"]["num_layers"].as_integer(),
            Some(2)
        );
        assert_eq!(
            models.get(1).unwrap()["arch"]["default_context_len"].as_integer(),
            Some(4096)
        );
        assert!(models.get(0).unwrap()["arch"]
            .get("default_context_len")
            .is_none());
        // Into an empty config too.
        assert_eq!(append_block("", BLOCK, "b").unwrap(), BLOCK);
    }

    #[test]
    fn refuses_a_duplicate_id_and_a_wrong_block() {
        let err = append_block(CONFIG, &BLOCK.replace("\"b\"", "\"a\""), "a").unwrap_err();
        assert!(err.contains("already has"), "{err}");
        let twice = append_block(&append_block(CONFIG, BLOCK, "b").unwrap(), BLOCK, "b");
        assert!(twice.unwrap_err().contains("already has"));
        assert!(append_block(CONFIG, BLOCK, "c").is_err());
        assert!(append_block(CONFIG, "not toml [", "b").is_err());
        // An inline `models = [...]` cannot take a [[models]] table.
        assert!(append_block("models = []\n", BLOCK, "b").is_err());
    }

    /// With this checkout's release server: the block comes from `hypnos --catalog-block`.
    #[tokio::test]
    async fn renders_a_block_with_the_server_binary() {
        let binary =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/target/release/hypnos");
        if !binary.is_file() {
            eprintln!("skipped: no {}", binary.display());
            return;
        }
        let (path, _) = load_catalog().unwrap();
        let block = render_block(&binary, &path, "lfm3:1.7b:Q4_K_M")
            .await
            .unwrap();
        assert!(
            block.contains("weights_path = \"LFM3-1.7B-Q4_K_M.gguf\""),
            "{block}"
        );
        let out = append_block(CONFIG, &block, "lfm3:1.7b").unwrap();
        assert!(out.contains("id = \"lfm3:1.7b\""));
        assert!(render_block(&binary, &path, "nope:x").await.is_err());
        assert!(render_block(Path::new("/no/hypnos"), &path, "x")
            .await
            .is_err());
    }

    /// Removes, adds back and replaces a catalog model on a running server whose models folder
    /// is the config's (its files must be on disk; nothing is downloaded):
    /// `HYPNOS_TEST_URL=http://127.0.0.1:9993 HYPNOS_TEST_CONFIG=/tmp/x/hypnos.toml
    /// [HYPNOS_TEST_POOL_MODEL=lfm3-audio:2b] cargo test changes_the_pool -- --ignored`.
    /// The server writes its overlay next to its config: use a copy.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn changes_the_pool_of_a_running_server() {
        use tauri::Manager;
        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let config = std::env::var("HYPNOS_TEST_CONFIG").expect("HYPNOS_TEST_CONFIG");
        let id = std::env::var("HYPNOS_TEST_POOL_MODEL").unwrap_or("lfm3-audio:2b".into());
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .manage(ModelFiles::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        crate::hypnos::hypnos_connect(app.state(), url)
            .await
            .unwrap();
        let entry = |v: &CatalogView| {
            let e = v.entries.iter().find(|e| e.id == id).unwrap();
            (e.in_pool, e.current_quant.clone())
        };
        let in_status = || async {
            let status = crate::hypnos::hypnos_memory_status(app.state())
                .await
                .unwrap();
            status.models.iter().any(|m| m.id == id)
        };

        let view = models_catalog_entries(app.state(), config.clone())
            .await
            .unwrap();
        assert!(view.live, "{:?}", view.offline);
        let (in_pool, quant) = entry(&view);
        assert!(in_pool, "{id} is not in the pool");
        let quant = quant.expect("the config's variant");

        let removed = models_pool_remove(app.state(), id.clone()).await.unwrap();
        assert_eq!(removed.action, hypnos_client::ModelPoolAction::Removed);
        assert!(removed.persisted && removed.placement.is_none());
        assert!(!in_status().await);
        let view = models_catalog_entries(app.state(), config.clone())
            .await
            .unwrap();
        assert!(!entry(&view).0);
        let overlay = overlay_path(Path::new(&config));
        assert!(std::fs::read_to_string(&overlay).unwrap().contains(&id));

        let put = |replace: bool| {
            let (config, id, quant) = (config.clone(), id.clone(), quant.clone());
            let app = &app;
            async move {
                let channel = Channel::new(|_| Ok(()));
                models_pool_put(
                    app.state(),
                    app.state(),
                    config,
                    id,
                    quant,
                    replace,
                    channel,
                )
                .await
            }
        };
        let added = put(false).await.unwrap().unwrap();
        assert_eq!(added.action, hypnos_client::ModelPoolAction::Added);
        assert!(in_status().await);
        let view = models_catalog_entries(app.state(), config.clone())
            .await
            .unwrap();
        assert_eq!(entry(&view), (true, Some(quant.clone())));
        let err = put(false).await.unwrap_err();
        assert_eq!(err.kind, "bad_request", "{}", err.message);

        let replaced = put(true).await.unwrap().unwrap();
        assert_eq!(replaced.action, hypnos_client::ModelPoolAction::Replaced);
        assert!(in_status().await);
    }
}
