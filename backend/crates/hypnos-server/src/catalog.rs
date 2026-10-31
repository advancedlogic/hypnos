//! `GET /v1/models/catalog` (API-10): the curated model catalog (`models.catalog.toml`) with, per
//! variant, what is on disk under `models_dir`, whether the model is configured, and the
//! `[[models]]` block that runs it.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use axum::extract::{Query, State};
use axum::Json;
use hypnos_core::catalog::{CatalogFile, ModelCatalog, TESTED_GGUF_ARCHITECTURES};
use hypnos_core::config::Modality;
use hypnos_core::error::HypnosError;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::{ApiError, AppState};

pub const FILE_NAME: &str = "models.catalog.toml";

/// Where the catalog is: `$HYPNOS_CATALOG`, else `./models.catalog.toml` (a server started in
/// `backend/`), else beside the executable (release archives, `Contents/Resources[/server]` in
/// the macOS app), else this checkout's (tests, the mock server).
pub fn locate() -> Option<PathBuf> {
    if let Some(p) = std::env::var_os("HYPNOS_CATALOG").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(p));
    }
    let mut candidates = vec![PathBuf::from(FILE_NAME)];
    if let Some(dir) = std::env::current_exe()
        .ok()
        .and_then(|e| e.parent().map(Path::to_path_buf))
    {
        candidates.extend(hypnos_core::config::beside_executable(&dir, FILE_NAME));
        if let Some(contents) = dir.parent() {
            candidates.push(contents.join("Resources/server").join(FILE_NAME));
        }
    }
    candidates.push(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(FILE_NAME),
    );
    candidates.into_iter().find(|p| p.is_file())
}

/// The catalog, or `None` (logged) when there is none or it does not parse: the route then
/// answers 404 and everything else works as before.
pub fn load() -> Option<Arc<ModelCatalog>> {
    let path = locate()?;
    match ModelCatalog::load(&path) {
        Ok(c) => {
            info!(path = %path.display(), models = c.entries.len(), "model catalog");
            Some(Arc::new(c))
        }
        Err(e) => {
            warn!("model catalog not used: {e}");
            None
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct CatalogQuery {
    /// Only entries serving this modality (primary or additional).
    pub modality: Option<Modality>,
}

#[derive(Debug, Serialize)]
struct FileState<'a> {
    #[serde(flatten)]
    file: &'a CatalogFile,
    /// On disk: the file, or a folder that is not empty.
    present: bool,
    /// Present with another size than the catalog lists (an older or replaced upload).
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    outdated: bool,
}

#[derive(Debug, Serialize)]
struct VariantState<'a> {
    quant: &'a str,
    default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<&'a str>,
    /// Bytes of this machine's files.
    size_bytes: u64,
    files: Vec<FileState<'a>>,
    /// Every file is present.
    installed: bool,
    /// Some present file differs in size from the catalog: re-downloading it gets the current one.
    outdated: bool,
    /// Downloaded weights that are not what the entry expects (their GGUF architecture).
    #[serde(skip_serializing_if = "Option::is_none")]
    problem: Option<String>,
    config_block: String,
}

/// (present, outdated) for one file.
fn on_disk(root: &Path, f: &CatalogFile) -> (bool, bool) {
    let p = root.join(&f.local);
    if f.local.ends_with('/') {
        (
            std::fs::read_dir(&p).is_ok_and(|mut d| d.next().is_some()),
            false,
        )
    } else {
        match std::fs::metadata(&p) {
            Ok(m) if m.is_file() => (true, m.len() != f.size),
            _ => (false, false),
        }
    }
}

pub async fn list_catalog(
    State(state): State<AppState>,
    Query(q): Query<CatalogQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(catalog) = state.catalog.clone() else {
        return Err(HypnosError::NotFound(format!(
            "no model catalog: {FILE_NAME} was not found next to the server (or $HYPNOS_CATALOG)"
        ))
        .into());
    };
    let root = state.models_dir.clone();
    // Id, enabled, and the weights file it runs (to name the configured variant).
    let configured: Vec<(String, bool, PathBuf)> = state
        .scheduler
        .status()
        .models
        .into_iter()
        .map(|m| {
            let weights = state
                .scheduler
                .memory()
                .get_model_state(&m.id)
                .map(|(c, _)| c.weights_path)
                .unwrap_or_default();
            (m.id, m.enabled, weights)
        })
        .collect();
    let apple_silicon = cfg!(all(target_os = "macos", target_arch = "aarch64"));
    // Reading GGUF headers touches the disk: off the async runtime.
    let body = tokio::task::spawn_blocking(move || {
        let data: Vec<serde_json::Value> = catalog
            .entries
            .iter()
            .filter(|e| {
                q.modality
                    .is_none_or(|m| e.modality == m || e.additional_modalities.contains(&m))
            })
            .map(|e| {
                let variants: Vec<VariantState> = e
                    .variants
                    .iter()
                    .map(|v| {
                        let files: Vec<FileState> = v
                            .files_for(apple_silicon)
                            .map(|f| {
                                let (present, outdated) = on_disk(&root, f);
                                FileState {
                                    file: f,
                                    present,
                                    outdated,
                                }
                            })
                            .collect();
                        let installed = files.iter().all(|f| f.present);
                        let outdated = files.iter().any(|f| f.outdated);
                        VariantState {
                            quant: &v.quant,
                            default: v.default,
                            note: v.note.as_deref(),
                            size_bytes: v.size_for(apple_silicon),
                            installed,
                            outdated,
                            problem: e.check_weights(v, &root).err(),
                            config_block: e.render_block(v),
                            files,
                        }
                    })
                    .collect();
                let config = configured.iter().find(|(id, _, _)| *id == e.id);
                // The variant whose weights file the configured model runs.
                let configured_quant = config.and_then(|(_, _, weights)| {
                    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
                    e.variants.iter().find_map(|v| {
                        let w = v
                            .files
                            .iter()
                            .find(|f| f.role == hypnos_core::catalog::FileRole::Weights)?;
                        let local = root.join(w.local.trim_end_matches('/'));
                        let local = match &w.entry {
                            Some(entry) => local.join(entry),
                            None => local,
                        };
                        (canon(&local) == canon(weights)).then(|| v.quant.clone())
                    })
                });
                let mut entry = serde_json::to_value(e).unwrap_or_default();
                if let serde_json::Value::Object(map) = &mut entry {
                    map.remove("config");
                    map.insert("configured".into(), config.is_some().into());
                    if let Some((_, enabled, _)) = config {
                        map.insert("enabled".into(), (*enabled).into());
                    }
                    if let Some(q) = configured_quant {
                        map.insert("configured_quant".into(), q.into());
                    }
                    map.insert(
                        "variants".into(),
                        serde_json::to_value(variants).unwrap_or_default(),
                    );
                }
                entry
            })
            .collect();
        serde_json::json!({
            "object": "list",
            "catalog_version": catalog.catalog_version,
            "models_dir": root.display().to_string(),
            "tested_architectures": TESTED_GGUF_ARCHITECTURES,
            "data": data,
        })
    })
    .await
    .map_err(|e| HypnosError::BackendError(format!("catalog task failed: {e}")))?;
    Ok(Json(body))
}

/// `POST /v1/advisor` body: what is left out is this server's machine.
#[derive(Debug, Deserialize)]
pub struct AdvisorBody {
    pub needs: Vec<hypnos_memory::advisor::Need>,
    pub vram_bytes: Option<u64>,
    pub ram_bytes: Option<u64>,
    pub unified_memory: Option<bool>,
}

/// `POST /v1/advisor` (API-12): a lineup from the catalog for this machine or the one described,
/// with tiers, priorities, slots, warnings and a config file.
pub async fn advisor(
    State(state): State<AppState>,
    axum::Json(body): axum::Json<AdvisorBody>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let Some(catalog) = state.catalog.clone() else {
        return Err(HypnosError::NotFound(format!(
            "no model catalog: {FILE_NAME} was not found next to the server (or $HYPNOS_CATALOG)"
        ))
        .into());
    };
    let caps = state.scheduler.memory().capabilities();
    let here_unified = state.scheduler.memory().is_unified_memory();
    let unified = body
        .unified_memory
        .unwrap_or(here_unified && body.vram_bytes.is_none());
    let ram = body
        .ram_bytes
        .or(hypnos_memory::process::physical_memory_bytes())
        .unwrap_or(caps.physical_memory_bytes);
    let vram = match (body.vram_bytes, unified) {
        (Some(v), _) => v,
        (None, true) => 0,
        (None, false) => caps.total_vram_bytes,
    };
    let req = hypnos_memory::advisor::AdvisorRequest {
        vram_bytes: vram,
        ram_bytes: ram,
        unified_memory: unified,
        needs: body.needs,
    };
    let advice =
        hypnos_memory::advisor::advise(&catalog, &req).map_err(HypnosError::InvalidRequest)?;
    let mut v = serde_json::to_value(advice).unwrap_or_default();
    if let serde_json::Value::Object(map) = &mut v {
        map.insert(
            "machine".into(),
            serde_json::json!({
                "vram_bytes": req.vram_bytes,
                "ram_bytes": req.ram_bytes,
                "unified_memory": req.unified_memory,
            }),
        );
    }
    Ok(Json(v))
}
