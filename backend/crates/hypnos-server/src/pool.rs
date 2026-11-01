//! The runtime model pool (API-11): `POST /v1/models` adds a model, `PUT /v1/models/{id}`
//! replaces one under the same id, `DELETE /v1/models/{id}` removes one, while the server runs.
//!
//! A model comes as a catalog spec (`{"catalog": "lfm3-vl:2b:Q4_K_M"}`) or as a `[[models]]`
//! block (`{"config": "..."}`) and goes through the startup checks: parsed as the config is, its
//! paths resolved against `models_dir` (and kept inside it), `[models.arch]` filled from the GGUF,
//! `HypnosConfig::validate`, and for `ggml_llama` an architecture of
//! [`TESTED_GGUF_ARCHITECTURES`]. It is registered cold; its first request loads it.
//!
//! Changes are kept in an overlay next to the config (`<config>.models.toml`: `removed` ids and
//! the added or replaced `[[models]]` blocks), applied at startup by [`apply_overlay`], so the
//! hand-written config is never rewritten. With API keys set, these routes need an admin key.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use axum::extract::{Path as UrlPath, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Extension, Json};
use std::sync::Arc;

use hypnos_backends::{create_backend_for_model, EngineOptions};
use hypnos_core::catalog::TESTED_GGUF_ARCHITECTURES;
use hypnos_core::config::{EngineKind, HypnosConfig, ModelConfig};
use hypnos_core::error::HypnosError;
use hypnos_core::ModelBackend;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tracing::{info, warn};

use crate::auth::ApiKeyLabel;
use crate::{ApiError, AppState};

/// Makes the engine for a model the pool takes in.
pub type EngineFactory = Arc<dyn Fn(ModelConfig) -> Arc<dyn ModelBackend> + Send + Sync>;

/// How long a removal or replacement waits for running requests by default.
const DEFAULT_DRAIN_SECS: u64 = 120;

/// What the pool needs beyond the scheduler: the startup config without its models (server,
/// hardware, limits) to check new blocks against, the engine options, and the overlay file.
pub struct ModelPool {
    template: HypnosConfig,
    base_dir: PathBuf,
    factory: EngineFactory,
    /// Run the file, GGUF and budget checks (off only for the mock server's fake engines).
    strict: bool,
    /// Ids the config file itself defines (a removal of one is recorded as `removed`).
    config_ids: parking_lot::RwLock<BTreeSet<String>>,
    overlay: parking_lot::RwLock<Option<PathBuf>>,
    /// One change at a time.
    changing: tokio::sync::Mutex<()>,
}

impl ModelPool {
    pub fn new(config: &HypnosConfig, engine_opts: EngineOptions) -> Self {
        Self::with_factory(
            config,
            Arc::new(move |m| create_backend_for_model(m, &engine_opts)),
            true,
        )
    }

    /// A pool whose engines come from `factory`; `strict = false` skips the checks that need
    /// real files (the mock server).
    pub fn with_factory(config: &HypnosConfig, factory: EngineFactory, strict: bool) -> Self {
        let mut template = config.clone();
        template.models.clear();
        Self {
            template,
            base_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            factory,
            strict,
            config_ids: parking_lot::RwLock::new(
                config.models.iter().map(|m| m.id.clone()).collect(),
            ),
            overlay: parking_lot::RwLock::new(None),
            changing: tokio::sync::Mutex::new(()),
        }
    }

    /// Persists runtime changes to `path` from now on; `config_ids` are the ids the config file
    /// defines without the overlay.
    pub fn set_overlay(&self, path: PathBuf, config_ids: BTreeSet<String>) {
        *self.config_ids.write() = config_ids;
        *self.overlay.write() = Some(path);
    }

    pub fn overlay_path(&self) -> Option<PathBuf> {
        self.overlay.read().clone()
    }

    /// Parses and checks one model block; returns its config (paths resolved, arch detected) and
    /// the block as a table (for the overlay, paths as written).
    fn check_block(
        &self,
        block: &str,
        force_id: Option<&str>,
    ) -> Result<(ModelConfig, toml::Table), HypnosError> {
        let bad = |m: String| HypnosError::InvalidRequest(m);
        let mut raw: toml::Table =
            toml::from_str(block).map_err(|e| bad(format!("config: {e}")))?;
        let mut tables = match raw.remove("models") {
            Some(toml::Value::Array(a)) if a.len() == 1 => a,
            _ => return Err(bad("config must hold exactly one [[models]] block".into())),
        };
        if let Some(key) = raw.keys().next() {
            return Err(bad(format!(
                "config may only hold a [[models]] block, not '{key}'"
            )));
        }
        let Some(toml::Value::Table(mut table)) = tables.pop() else {
            return Err(bad("[[models]] must be a table".into()));
        };
        if let Some(id) = force_id {
            table.insert("id".into(), toml::Value::String(id.to_string()));
        }
        // The server's workload profile fills in what the block leaves out, as at startup.
        let stored = table.clone();
        if let Some(profile) = self.template.server.profile {
            profile.apply_to_model(&mut table);
        }
        let parsed = HypnosConfig::from_toml_str(&overlay_text(&[], &[table.clone()]))
            .map_err(|e| bad(e.to_string()))?;
        if !parsed.unknown_keys.is_empty() {
            return Err(bad(format!(
                "unknown config keys: {}",
                parsed.unknown_keys.join(", ")
            )));
        }
        let mut full = self.template.clone();
        full.models = parsed.models;
        if !self.strict {
            return Ok((full.models.remove(0), stored));
        }
        // The template's models_dir is already resolved (absolute) at startup.
        full.apply_models_dir(&self.base_dir, None);
        if let Some(dir) = &self.template.server.models_dir {
            let m = &full.models[0];
            for p in [
                Some(&m.weights_path),
                m.aux_weights_path.as_ref(),
                m.text_encoder_path.as_ref(),
                m.tokenizer_path.as_ref(),
                m.voices_path.as_ref(),
            ]
            .into_iter()
            .flatten()
            {
                if !inside(p, dir) {
                    return Err(bad(format!(
                        "{} is outside models_dir ({})",
                        p.display(),
                        dir.display()
                    )));
                }
            }
        }
        for note in full.detect_arch_from_gguf(&self.base_dir) {
            info!(config = %note);
        }
        let report = full.validate(&self.base_dir);
        if !report.errors.is_empty() {
            return Err(bad(report.errors.join("; ")));
        }
        let model = full.models.remove(0);
        if model.engine == EngineKind::GgmlLlama {
            let arch = hypnos_core::gguf::GgufArchMetadata::read(&model.weights_path)
                .ok()
                .flatten()
                .map(|m| m.architecture)
                .unwrap_or_default();
            if !TESTED_GGUF_ARCHITECTURES.contains(&arch.as_str()) {
                return Err(bad(format!(
                    "{} has architecture '{arch}', not one Hypnos has been tested with ({})",
                    model.weights_path.display(),
                    TESTED_GGUF_ARCHITECTURES.join(", ")
                )));
            }
        }
        Ok((model, stored))
    }

    /// The workload profile the server runs with (`[server] profile`).
    pub fn profile(&self) -> Option<hypnos_core::profile::WorkloadProfile> {
        self.template.server.profile
    }

    /// Writes one change to the overlay (when there is one); a failure is logged, not returned:
    /// the change is live either way.
    fn record(&self, id: &str, block: Option<toml::Table>) -> bool {
        let Some(path) = self.overlay_path() else {
            return false;
        };
        let mut overlay = read_overlay(&path).unwrap_or_default();
        overlay
            .models
            .retain(|t| t.get("id").and_then(|v| v.as_str()) != Some(id));
        overlay.removed.retain(|r| r != id);
        match block {
            Some(table) => overlay.models.push(table),
            None if self.config_ids.read().contains(id) => overlay.removed.push(id.to_string()),
            None => {}
        }
        match write_overlay(&path, &overlay) {
            Ok(()) => true,
            Err(e) => {
                warn!(path = %path.display(), "model pool overlay not written: {e}");
                false
            }
        }
    }
}

/// The `[[models]]` overlay: ids taken out of the config, and blocks added or replacing one.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Overlay {
    #[serde(default)]
    pub removed: Vec<String>,
    #[serde(default)]
    pub models: Vec<toml::Table>,
}

/// `<folder>/<config stem>.models.toml` for a config file.
pub fn overlay_path_for(config: &Path) -> PathBuf {
    let stem = config
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "hypnos".into());
    config.with_file_name(format!("{stem}.models.toml"))
}

fn overlay_text(removed: &[String], models: &[toml::Table]) -> String {
    #[derive(Serialize)]
    struct Doc<'a> {
        config_version: u32,
        #[serde(skip_serializing_if = "<[String]>::is_empty")]
        removed: &'a [String],
        models: &'a [toml::Table],
    }
    toml::to_string(&Doc {
        config_version: 1,
        removed,
        models,
    })
    .unwrap_or_default()
}

pub fn read_overlay(path: &Path) -> Result<Overlay, String> {
    let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut table: toml::Table = toml::from_str(&text).map_err(|e| e.to_string())?;
    table.remove("config_version");
    toml::Value::Table(table)
        .try_into()
        .map_err(|e: toml::de::Error| e.to_string())
}

fn write_overlay(path: &Path, overlay: &Overlay) -> std::io::Result<()> {
    let text = format!(
        "# Written by Hypnos: models added, replaced (by id) or removed while the server ran\n\
         # (POST /v1/models, PUT and DELETE /v1/models/{{id}}). Applied over the config at startup.\n\
         # Delete this file to go back to the config alone.\n{}",
        overlay_text(&overlay.removed, &overlay.models)
    );
    let tmp = path.with_extension("toml.tmp");
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path)
}

/// Applies the overlay at `path` (if it exists) to `config` before anything else reads the
/// models. Returns a line per change for the log.
pub fn apply_overlay(config: &mut HypnosConfig, path: &Path) -> Result<Vec<String>, String> {
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let overlay = read_overlay(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut notes = Vec::new();
    config.models.retain(|m| {
        let keep = !overlay.removed.contains(&m.id);
        if !keep {
            notes.push(format!("'{}' removed", m.id));
        }
        keep
    });
    // The overlay's blocks get the config's workload profile, as the config's own blocks did.
    let mut blocks = overlay.models.clone();
    if let Some(profile) = config.server.profile {
        for b in &mut blocks {
            profile.apply_to_model(b);
        }
    }
    let added = HypnosConfig::from_toml_str(&overlay_text(&[], &blocks))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    for model in added.models {
        match config.models.iter_mut().find(|m| m.id == model.id) {
            Some(slot) => {
                notes.push(format!("'{}' replaced", model.id));
                *slot = model;
            }
            None => {
                notes.push(format!("'{}' added", model.id));
                config.models.push(model);
            }
        }
    }
    Ok(notes)
}

fn inside(path: &Path, dir: &Path) -> bool {
    let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
    canon(path).starts_with(canon(dir))
}

/// `POST /v1/models` and `PUT /v1/models/{id}` body: one of the two.
#[derive(Debug, Deserialize)]
pub struct PoolRequest {
    /// A catalog entry, `<id>` or `<id>:<quant>`.
    pub catalog: Option<String>,
    /// A `[[models]]` block (with its `[models.arch]`), paths relative to `models_dir`.
    pub config: Option<String>,
    /// Seconds a replacement waits for running requests (default 120).
    pub drain_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct RemoveQuery {
    pub drain_secs: Option<u64>,
}

fn forbidden() -> Response {
    (
        StatusCode::FORBIDDEN,
        Json(json!({"error": {
            "message": "changing the model pool needs an admin key (a plain server.api_keys entry, or [[server.keys]] with admin = true)",
            "type": "permission_denied",
        }})),
    )
        .into_response()
}

fn block_for(state: &AppState, req: &PoolRequest) -> Result<String, HypnosError> {
    match (&req.catalog, &req.config) {
        (Some(spec), None) => {
            let catalog = state.catalog.as_ref().ok_or_else(|| {
                HypnosError::InvalidRequest("this server has no model catalog".into())
            })?;
            let (entry, variant) = catalog.resolve(spec).map_err(HypnosError::InvalidRequest)?;
            Ok(entry.render_block(variant))
        }
        (None, Some(block)) => Ok(block.clone()),
        _ => Err(HypnosError::InvalidRequest(
            "send either \"catalog\" (<id>[:<quant>]) or \"config\" (a [[models]] block)".into(),
        )),
    }
}

fn admin(state: &AppState, label: &Option<Extension<ApiKeyLabel>>) -> bool {
    state
        .auth
        .is_admin(label.as_ref().map(|Extension(l)| l.0.as_str()))
}

fn placement(state: &AppState, id: &str) -> serde_json::Value {
    state
        .scheduler
        .memory()
        .get_model_state(id)
        .map(|(_, p)| serde_json::to_value(p).unwrap_or_default())
        .unwrap_or_default()
}

/// `POST /v1/models`: adds a model to the pool.
pub async fn add_model(
    State(state): State<AppState>,
    label: Option<Extension<ApiKeyLabel>>,
    Json(req): Json<PoolRequest>,
) -> Result<Response, ApiError> {
    if !admin(&state, &label) {
        return Ok(forbidden());
    }
    let pool = state.pool.clone();
    let _one = pool.changing.lock().await;
    let block = block_for(&state, &req)?;
    let (model, table) = pool.check_block(&block, None)?;
    let id = model.id.clone();
    state.scheduler.add_model((pool.factory)(model)).await?;
    let persisted = pool.record(&id, Some(table));
    info!(model = %id, persisted, "Model added to the pool");
    Ok((
        StatusCode::CREATED,
        Json(
            json!({"model": id, "action": "added", "persisted": persisted,
                    "placement": placement(&state, &id)}),
        ),
    )
        .into_response())
}

/// `PUT /v1/models/{id}`: replaces the model under the same id. Requests that arrive meanwhile
/// wait and are served by the new model; running ones finish on the old one.
pub async fn replace_model(
    State(state): State<AppState>,
    label: Option<Extension<ApiKeyLabel>>,
    UrlPath(id): UrlPath<String>,
    Json(req): Json<PoolRequest>,
) -> Result<Response, ApiError> {
    if !admin(&state, &label) {
        return Ok(forbidden());
    }
    let pool = state.pool.clone();
    let _one = pool.changing.lock().await;
    if state.scheduler.backend(&id).is_none() {
        return Err(HypnosError::ModelNotFound(id).into());
    }
    let block = block_for(&state, &req)?;
    let (model, table) = pool.check_block(&block, Some(&id))?;
    let drain = Duration::from_secs(req.drain_secs.unwrap_or(DEFAULT_DRAIN_SECS));
    state
        .scheduler
        .replace_model((pool.factory)(model), drain)
        .await?;
    let persisted = pool.record(&id, Some(table));
    info!(model = %id, persisted, "Model replaced in the pool");
    Ok(Json(
        json!({"model": id, "action": "replaced", "persisted": persisted,
                   "placement": placement(&state, &id)}),
    )
    .into_response())
}

/// `DELETE /v1/models/{id}`: removes the model once its running requests have finished.
pub async fn remove_model(
    State(state): State<AppState>,
    label: Option<Extension<ApiKeyLabel>>,
    UrlPath(id): UrlPath<String>,
    Query(q): Query<RemoveQuery>,
) -> Result<Response, ApiError> {
    if !admin(&state, &label) {
        return Ok(forbidden());
    }
    let pool = state.pool.clone();
    let _one = pool.changing.lock().await;
    let drain = Duration::from_secs(q.drain_secs.unwrap_or(DEFAULT_DRAIN_SECS));
    state.scheduler.remove_model(&id, drain).await?;
    let persisted = pool.record(&id, None);
    info!(model = %id, persisted, "Model removed from the pool");
    Ok(Json(json!({"model": id, "action": "removed", "persisted": persisted})).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_overlay_removes_replaces_and_adds() {
        let dir = std::env::temp_dir().join(format!("hypnos-overlay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("hypnos.models.toml");
        let block = |id: &str, weights: &str| -> toml::Table {
            toml::from_str(&format!(
                "id = \"{id}\"\nmodality = \"text_generation\"\nengine = \"ggml_llama\"\nweights_path = \"{weights}\"\n[arch]\nnum_layers = 2\n"
            ))
            .unwrap()
        };
        write_overlay(
            &path,
            &Overlay {
                removed: vec!["gone".into()],
                models: vec![block("kept", "new.gguf"), block("fresh", "fresh.gguf")],
            },
        )
        .unwrap();
        let mut config = HypnosConfig::from_toml_str(&overlay_text(
            &[],
            &[block("gone", "a.gguf"), block("kept", "old.gguf")],
        ))
        .unwrap();
        let notes = apply_overlay(&mut config, &path).unwrap();
        let ids: Vec<&str> = config.models.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["kept", "fresh"]);
        assert_eq!(config.models[0].weights_path, PathBuf::from("new.gguf"));
        assert_eq!(notes.len(), 3, "{notes:?}");
        assert_eq!(
            overlay_path_for(Path::new("/etc/hypnos/hypnos.toml")),
            PathBuf::from("/etc/hypnos/hypnos.models.toml")
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
