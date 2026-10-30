//! Zero-shot named-entity extraction with GLiNER2.5 boundary models (`gliner` engine), on the CPU
//! through the vendored `gliner25-rs` (ported to ort rc.15, CPU-only).
//!
//! The labels are free text chosen per request. More than [`FAMILY_SIZE`] labels are split into
//! families that each get their own encoder pass, so they do not compete for one candidate pool;
//! texts longer than the model's largest bucket (512 words) are read in overlapping windows.
//!
//! The engine's sessions are not shared across threads: one worker thread owns it from load to
//! unload and serves requests over a channel.

use async_trait::async_trait;
use gliner25_rs::families::chunk_into_families;
use gliner25_rs::{
    BoundaryConfig, BoundaryEngine, BoundaryParams, Chunker, Mention, OverlapPolicy,
};
use hypnos_core::config::{MemoryTier, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    Entity, EntityExtractionRequest, EntityExtractionResponse, EntityResult, EntityUsage,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::time::Instant;
use tokio::sync::oneshot;
use tracing::info;

/// Labels per encoder pass; more compete for the same candidates and fragment the spans.
const FAMILY_SIZE: usize = 8;
/// Upper bounds for one request.
const MAX_TEXTS: usize = 256;
const MAX_LABELS: usize = 64;

struct Job {
    texts: Vec<String>,
    labels: Vec<String>,
    params: BoundaryParams,
    ctx: RequestContext,
    reply: oneshot::Sender<Result<Vec<Vec<Entity>>>>,
}

pub struct GlinerBackend {
    config: ModelConfig,
    /// The worker's queue while resident; dropping it ends the worker and frees the engine.
    worker: Mutex<Option<mpsc::Sender<Job>>>,
    resident: AtomicBool,
}

impl GlinerBackend {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            worker: Mutex::new(None),
            resident: AtomicBool::new(false),
        }
    }
}

/// Byte offset to character offset in `text`.
fn char_offset(text: &str, byte: usize) -> usize {
    text.get(..byte.min(text.len()))
        .map_or_else(|| text.chars().count(), |prefix| prefix.chars().count())
}

/// Runs every label family over one text and merges the mentions: the same span and label from
/// two passes keeps the higher score.
fn extract_one(
    engine: &mut BoundaryEngine,
    text: &str,
    labels: &[String],
    params: &BoundaryParams,
) -> Result<Vec<Entity>> {
    let mut merged: HashMap<(usize, usize, String), Mention> = HashMap::new();
    for family in chunk_into_families(labels, FAMILY_SIZE) {
        let out = engine
            .extract_long_with(text, &[family.task()], params, Chunker::default())
            .map_err(|e| HypnosError::BackendError(format!("gliner: {e:#}")))?;
        for m in out.mentions {
            let key = (m.char_start, m.char_end, m.field.clone());
            match merged.get(&key) {
                Some(kept) if kept.score >= m.score => {}
                _ => {
                    merged.insert(key, m);
                }
            }
        }
    }
    let mut entities: Vec<Entity> = merged
        .into_values()
        .map(|m| Entity {
            start: char_offset(text, m.char_start),
            end: char_offset(text, m.char_end),
            text: m.text,
            label: m.field,
            score: m.score,
        })
        .collect();
    entities.sort_by(|a, b| {
        a.start
            .cmp(&b.start)
            .then(a.end.cmp(&b.end))
            .then(b.score.total_cmp(&a.score))
    });
    Ok(entities)
}

/// Loads the engine on this thread, reports the outcome through `loaded`, then serves jobs until
/// the queue is dropped.
fn worker_main(
    dir: std::path::PathBuf,
    threads: usize,
    jobs: mpsc::Receiver<Job>,
    loaded: mpsc::SyncSender<Result<()>>,
) {
    let engine = (|| -> Result<BoundaryEngine> {
        gliner25_rs::init(crate::onnx::environment()?);
        BoundaryEngine::new(
            BoundaryConfig::new(&dir)
                .with_intra_threads(threads)
                .eager_heads(),
        )
        .map_err(|e| {
            HypnosError::BackendError(format!("gliner could not load {}: {e:#}", dir.display()))
        })
    })();
    let mut engine = match engine {
        Ok(engine) => {
            let _ = loaded.send(Ok(()));
            engine
        }
        Err(e) => {
            let _ = loaded.send(Err(e));
            return;
        }
    };
    while let Ok(job) = jobs.recv() {
        let mut results = Vec::with_capacity(job.texts.len());
        let mut outcome = Ok(());
        for text in &job.texts {
            if !job.ctx.should_continue() {
                outcome = Err(HypnosError::Cancelled);
                break;
            }
            match extract_one(&mut engine, text, &job.labels, &job.params) {
                Ok(entities) => results.push(entities),
                Err(e) => {
                    outcome = Err(e);
                    break;
                }
            }
        }
        let _ = job.reply.send(outcome.map(|()| results));
    }
}

#[async_trait]
impl ModelBackend for GlinerBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }

    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        _pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        if plan.primary_tier != MemoryTier::Tier1GpuVram {
            // The worker drops the engine once its queue is gone.
            if self.worker.lock().take().is_some() {
                info!(model = %self.config.id, "GLiNER engine released");
            }
            self.resident.store(false, Ordering::SeqCst);
            return Ok(());
        }
        if self.worker.lock().is_some() {
            return Ok(());
        }
        let dir = self
            .config
            .weights_path
            .parent()
            .filter(|d| !d.as_os_str().is_empty())
            .unwrap_or(Path::new("."))
            .to_path_buf();
        let threads = crate::default_threads().clamp(1, 8) as usize;
        let started = Instant::now();
        let (tx, rx) = mpsc::channel::<Job>();
        let (loaded_tx, loaded_rx) = mpsc::sync_channel(1);
        std::thread::Builder::new()
            .name(format!("gliner-{}", self.config.id))
            .spawn(move || worker_main(dir, threads, rx, loaded_tx))
            .map_err(|e| HypnosError::BackendError(format!("gliner worker thread: {e}")))?;
        tokio::task::spawn_blocking(move || loaded_rx.recv())
            .await
            .map_err(|e| HypnosError::BackendError(format!("gliner load task: {e}")))?
            .map_err(|_| {
                HypnosError::BackendError("gliner worker exited while loading".into())
            })??;
        info!(
            model = %self.config.id,
            threads,
            load_secs = format!("{:.2}", started.elapsed().as_secs_f32()),
            "GLiNER engine resident"
        );
        *self.worker.lock() = Some(tx);
        self.resident.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn extract_entities(
        &self,
        req: &EntityExtractionRequest,
        ctx: &RequestContext,
    ) -> Result<EntityExtractionResponse> {
        let texts = req.input.clone().into_vec();
        let labels: Vec<String> = req
            .labels
            .iter()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        if texts.is_empty() || texts.len() > MAX_TEXTS {
            return Err(HypnosError::InvalidRequest(format!(
                "input must hold 1 to {MAX_TEXTS} texts"
            )));
        }
        if labels.is_empty() || labels.len() > MAX_LABELS {
            return Err(HypnosError::InvalidRequest(format!(
                "labels must hold 1 to {MAX_LABELS} entity types"
            )));
        }
        let characters: u64 = texts.iter().map(|t| t.chars().count() as u64).sum();
        if characters > ctx.limits.max_prompt_chars as u64 {
            return Err(HypnosError::InvalidRequest(format!(
                "input is {characters} characters, above limits.max_prompt_chars ({})",
                ctx.limits.max_prompt_chars
            )));
        }
        let threshold = req.threshold.unwrap_or(0.5);
        if !(0.0..=1.0).contains(&threshold) {
            return Err(HypnosError::InvalidRequest(
                "threshold must be within 0..1".into(),
            ));
        }
        let params = BoundaryParams {
            threshold,
            overlap_policy: req.flat.map(|flat| {
                if flat {
                    OverlapPolicy::Flat
                } else {
                    OverlapPolicy::Nested
                }
            }),
            ..BoundaryParams::default()
        };
        let worker = self.worker.lock().clone().ok_or_else(|| {
            HypnosError::BackendError(format!("model '{}' is not loaded", self.config.id))
        })?;
        let (reply, result) = oneshot::channel();
        let n_texts = texts.len() as u32;
        worker
            .send(Job {
                texts,
                labels,
                params,
                ctx: ctx.clone(),
                reply,
            })
            .map_err(|_| HypnosError::BackendError("gliner worker is gone".into()))?;
        let per_text = result
            .await
            .map_err(|_| HypnosError::BackendError("gliner worker is gone".into()))??;
        Ok(EntityExtractionResponse {
            object: "list".into(),
            model: req.model.clone(),
            data: per_text
                .into_iter()
                .enumerate()
                .map(|(index, entities)| EntityResult {
                    object: "entities".into(),
                    index,
                    entities,
                })
                .collect(),
            usage: EntityUsage {
                texts: n_texts,
                characters,
            },
            hypnos_telemetry: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_offsets_become_character_offsets() {
        let text = "Café Müller in Zürich";
        let start = text.find("Zürich").unwrap();
        assert_eq!(char_offset(text, start), 15);
        assert_eq!(char_offset(text, text.len()), text.chars().count());
        assert_eq!(char_offset(text, 0), 0);
    }
}
