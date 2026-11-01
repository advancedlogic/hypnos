//! `hypnos measure-memory` (TEST-3): what each model really takes next to what the planner books.
//!
//! Memory budgets are planner estimates, not limits the OS enforces. This loads each configured
//! model alone (everything else evicted first), runs one small request of its modality, and
//! samples the process footprint (on Apple Silicon it includes the Metal allocations) and, where a
//! GPU probe reads it, device memory in use, every 50 ms. The booking is the placement the
//! scheduler committed: GPU weights + KV + workspace, or host RAM for a CPU model. The difference
//! is how far the estimate is off for that engine and model.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use hypnos_core::config::{MemoryTier, Modality};
use hypnos_core::request::RequestContext;
use serde::Serialize;
use serde_json::json;

use crate::AppState;

#[derive(Debug, Clone, Serialize)]
pub struct Measurement {
    pub model: String,
    pub modality: Modality,
    /// The request run after loading (`load only` when there is none for the modality).
    pub request: String,
    /// The planner's booking: on the GPU (weights + KV + workspace) and in host RAM.
    pub booked_device_bytes: u64,
    pub booked_host_bytes: u64,
    /// Process footprint before loading and its peak while loaded and serving.
    pub footprint_before_bytes: Option<u64>,
    pub footprint_peak_bytes: Option<u64>,
    /// Device memory in use (GPU probe) before and at its peak; absent without a probe.
    pub device_before_bytes: Option<u64>,
    pub device_peak_bytes: Option<u64>,
    pub load_ms: f64,
    pub request_ms: f64,
    pub error: Option<String>,
}

impl Measurement {
    pub fn booked_bytes(&self) -> u64 {
        self.booked_device_bytes + self.booked_host_bytes
    }

    /// Real growth: the device reading when there is one (discrete GPU memory is outside the
    /// process footprint), else the footprint.
    pub fn real_bytes(&self) -> Option<u64> {
        let device = match (self.device_before_bytes, self.device_peak_bytes) {
            (Some(b), Some(p)) => Some(p.saturating_sub(b)),
            _ => None,
        };
        let footprint = match (self.footprint_before_bytes, self.footprint_peak_bytes) {
            (Some(b), Some(p)) => Some(p.saturating_sub(b)),
            _ => None,
        };
        match (device, footprint) {
            (Some(d), Some(f)) => Some(d.max(f)),
            (d, f) => d.or(f),
        }
    }
}

fn device_used(state: &AppState) -> Option<u64> {
    state
        .scheduler
        .memory()
        .reconcile_vram()
        .map(|(free, total)| total.saturating_sub(free))
}

async fn evict_all(state: &AppState) {
    for id in state.scheduler.memory().model_ids() {
        let _ = state
            .scheduler
            .set_model_tier(&id, MemoryTier::Tier3NvmeSsd, None)
            .await;
    }
}

/// A checkout file next to the sources (the measurement is a development tool).
fn fixture(rel: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)).ok()
}

/// The requests run on a loaded model: its primary modality, then an image for a model that sees
/// and an audio clip for one that hears, so the encoders' working memory is in the peak.
async fn exercise_all(
    backend: &Arc<dyn hypnos_core::ModelBackend>,
    id: &str,
    cfg: &hypnos_core::config::ModelConfig,
    ctx: &RequestContext,
) -> (String, Result<(), String>) {
    // A dedicated speech recognizer (Qwen3-ASR) does not chat: only the transcription below.
    let transcribes_only =
        cfg.modality == Modality::SpeechToText && !cfg.supports_modality(Modality::TextGeneration);
    let (mut what, mut result) = if transcribes_only {
        (String::new(), Ok(()))
    } else {
        exercise(backend, id, cfg.modality, ctx).await
    };
    if cfg.supports_modality(Modality::Vision) && result.is_ok() {
        if let Some(png) = fixture("../../../assets/logo.png") {
            use base64::Engine as _;
            let url = format!(
                "data:image/png;base64,{}",
                base64::engine::general_purpose::STANDARD.encode(png)
            );
            let req = serde_json::from_value(json!({
                "model": id, "max_tokens": 32, "temperature": 0.0,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "Describe this image in one sentence."},
                    {"type": "image_url", "image_url": {"url": url}}]}]
            }))
            .expect("vision request");
            what += " + an image";
            result = backend
                .chat_completion(&req, ctx)
                .await
                .map(|_| ())
                .map_err(|e| e.to_string());
        }
    }
    if cfg.supports_modality(Modality::SpeechToText) && result.is_ok() {
        if let Some(wav) = fixture("../../tests/fixtures/jfk.wav") {
            let req = hypnos_core::openai::AudioTranscriptionRequest {
                model: id.to_string(),
                audio_bytes: wav,
                filename: Some("jfk.wav".into()),
                language: Some("en".into()),
                prompt: None,
            };
            what += if what.is_empty() {
                "an 11 s transcription"
            } else {
                " + an 11 s transcription"
            };
            result = backend
                .transcribe(&req, ctx)
                .await
                .map(|_| ())
                .map_err(|e| e.to_string());
        }
    }
    (what, result)
}

/// The request run on a loaded model of `modality`, by name, and its result.
async fn exercise(
    backend: &Arc<dyn hypnos_core::ModelBackend>,
    id: &str,
    modality: Modality,
    ctx: &RequestContext,
) -> (String, Result<(), String>) {
    let err = |e: hypnos_core::error::HypnosError| e.to_string();
    let filler =
        "Hypnos keeps several models in a memory hierarchy and swaps them on demand. ".repeat(40);
    match modality {
        Modality::TextGeneration | Modality::Vision | Modality::SpeechToText => {
            let req = serde_json::from_value(json!({
                "model": id, "max_tokens": 64, "temperature": 0.0,
                "messages": [{"role": "user", "content": format!("{filler}\nSummarize the text above in one sentence.")}]
            }))
            .expect("chat request");
            (
                "chat, ~600 prompt tokens + 64".into(),
                backend
                    .chat_completion(&req, ctx)
                    .await
                    .map(|_| ())
                    .map_err(err),
            )
        }
        Modality::Embeddings => {
            let req =
                serde_json::from_value(json!({"model": id, "input": vec![filler.clone(); 8]}))
                    .expect("embedding request");
            (
                "8 embeddings of ~600 tokens".into(),
                backend.embeddings(&req, ctx).await.map(|_| ()).map_err(err),
            )
        }
        Modality::TextToSpeech => {
            let req = serde_json::from_value(json!({
                "model": id, "input": "Hypnos keeps several models in memory and swaps them on demand.",
                "voice": "af_heart", "response_format": "wav", "speed": 1.0
            }))
            .expect("speech request");
            (
                "one spoken sentence".into(),
                backend
                    .synthesize_speech(&req, ctx)
                    .await
                    .map(|_| ())
                    .map_err(err),
            )
        }
        Modality::ImageGeneration => {
            let req = serde_json::from_value(json!({
                "model": id, "prompt": "a lighthouse at dusk", "n": 1, "size": "1024x1024",
                "response_format": "b64_json", "steps": 4
            }))
            .expect("image request");
            (
                "one 1024x1024 image, 4 steps".into(),
                backend
                    .generate_image(&req, ctx)
                    .await
                    .map(|_| ())
                    .map_err(err),
            )
        }
        Modality::EntityExtraction => {
            let req = serde_json::from_value(json!({
                "model": id, "input": filler, "labels": ["organization", "product", "concept"]
            }))
            .expect("entity request");
            (
                "entities over ~600 tokens".into(),
                backend
                    .extract_entities(&req, ctx)
                    .await
                    .map(|_| ())
                    .map_err(err),
            )
        }
        Modality::DocumentConversion => ("load only".into(), Ok(())),
    }
}

/// Measures every enabled model (or the `only` ones), one at a time.
pub async fn measure(state: &AppState, only: &[String]) -> Vec<Measurement> {
    let memory = state.scheduler.memory().clone();
    let mut ids = memory.model_ids();
    ids.sort();
    let mut rows = Vec::new();
    for id in ids {
        if (!only.is_empty() && !only.contains(&id)) || !memory.is_enabled(&id).unwrap_or(false) {
            continue;
        }
        let Ok((cfg, _)) = memory.get_model_state(&id) else {
            continue;
        };
        evict_all(state).await;
        tokio::time::sleep(Duration::from_millis(800)).await;
        let footprint_before = memory.process_footprint_bytes();
        let device_before = device_used(state);

        // Peaks, sampled while the model loads and serves.
        let peak_fp = Arc::new(AtomicU64::new(footprint_before.unwrap_or(0)));
        let peak_dev = Arc::new(AtomicU64::new(device_before.unwrap_or(0)));
        let stop = Arc::new(AtomicBool::new(false));
        let sampler = {
            let (st, fp, dev, stop) = (
                state.clone(),
                peak_fp.clone(),
                peak_dev.clone(),
                stop.clone(),
            );
            tokio::spawn(async move {
                while !stop.load(Ordering::SeqCst) {
                    if let Some(f) = st.scheduler.memory().process_footprint_bytes() {
                        fp.fetch_max(f, Ordering::SeqCst);
                    }
                    if let Some(d) = device_used(&st) {
                        dev.fetch_max(d, Ordering::SeqCst);
                    }
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            })
        };

        let ctx = RequestContext::unbounded();
        let started = Instant::now();
        let mut row = Measurement {
            model: id.clone(),
            modality: cfg.modality,
            request: "load only".into(),
            booked_device_bytes: 0,
            booked_host_bytes: 0,
            footprint_before_bytes: footprint_before,
            footprint_peak_bytes: None,
            device_before_bytes: device_before,
            device_peak_bytes: None,
            load_ms: 0.0,
            request_ms: 0.0,
            error: None,
        };
        match state.scheduler.acquire_model(&id, None, 1, &ctx).await {
            Ok(lease) => {
                row.load_ms = started.elapsed().as_secs_f64() * 1000.0;
                let p = &lease.placement;
                if cfg.runs_on_cpu() {
                    row.booked_host_bytes = p.pinned_ram_bytes;
                } else {
                    row.booked_device_bytes = p.vram_weight_bytes + p.vram_kv_and_workspace_bytes;
                }
                let t = Instant::now();
                let (what, result) = exercise_all(&lease.backend, &id, &cfg, &ctx).await;
                row.request = what;
                row.request_ms = t.elapsed().as_secs_f64() * 1000.0;
                row.error = result.err();
                drop(lease);
            }
            Err(e) => row.error = Some(format!("load failed: {e}")),
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
        stop.store(true, Ordering::SeqCst);
        let _ = sampler.await;
        row.footprint_peak_bytes = footprint_before.map(|_| peak_fp.load(Ordering::SeqCst));
        row.device_peak_bytes = device_before.map(|_| peak_dev.load(Ordering::SeqCst));
        rows.push(row);
    }
    evict_all(state).await;
    rows
}

/// The measurements as a Markdown table, with the drift per model.
pub fn markdown(rows: &[Measurement]) -> String {
    let gib = |b: u64| format!("{:.2}", b as f64 / (1u64 << 30) as f64);
    let mut out = String::from(
        "| Model | Modality | Request | Booked GiB | Real GiB | Real / booked | Load ms | Request ms | Note |\n\
         | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |\n",
    );
    for r in rows {
        let real = r.real_bytes();
        let ratio = match (real, r.booked_bytes()) {
            (Some(real), booked) if booked > 0 => format!("{:.2}", real as f64 / booked as f64),
            _ => "-".into(),
        };
        out += &format!(
            "| {} | {:?} | {} | {} | {} | {} | {:.0} | {:.0} | {} |\n",
            r.model,
            r.modality,
            r.request,
            gib(r.booked_bytes()),
            real.map(gib).unwrap_or_else(|| "-".into()),
            ratio,
            r.load_ms,
            r.request_ms,
            r.error.clone().unwrap_or_default()
        );
    }
    out
}
