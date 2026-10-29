//! llama.cpp / libmtmd engine, linked in-process through the vendored `llama-cpp-2` bindings.
//!
//! Serves three of the configured models:
//! * `gemma4:26b-q4`  — text + vision chat (`/v1/chat/completions`)
//! * `gemma4:e4b-q4`  — audio understanding / speech-to-text (`/v1/audio/transcriptions`) plus text + vision chat
//! * `gemmaembedding2` — dense embeddings (`/v1/embeddings`)
//!
//! Weights are loaded when the scheduler promotes the model to Tier 1 (with `n_gpu_layers` taken from
//! the placement plan) and released again when it is demoted, so VRAM accounting in `hypnos-memory`
//! mirrors what the GPU actually holds. While resident, the model is owned by an engine worker
//! thread ([`crate::engine`]) that keeps one `llama_context` alive and serves requests through
//! execution slots with KV prefix reuse.

use std::ffi::CString;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use hypnos_core::config::{MemoryTier, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    AudioTranscriptionRequest, AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, ChatStreamEvent, EmbeddingData,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ExecutionTelemetry, PromptTokensDetails,
    TokenUsage,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{mtmd_default_marker, MtmdContext, MtmdContextParams};
use parking_lot::RwLock;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::engine::{
    spawn_worker, ChatJob, EmbedJob, GenParams, GenStats, LlamaJob, PromptSpec, WorkerSpec,
};
use crate::media::load_media;
use crate::worker::{FailJob, WorkerHandle};
use crate::{default_threads, BackendContextState, EngineOptions};

// ----------------------------------------------------------------------------
// Process-global llama.cpp backend handle
// ----------------------------------------------------------------------------

pub(crate) fn llama_backend() -> Result<&'static LlamaBackend> {
    static BACKEND: OnceLock<std::result::Result<LlamaBackend, String>> = OnceLock::new();
    let state = BACKEND.get_or_init(|| {
        llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default());
        LlamaBackend::init().map_err(|e| format!("{e:?}"))
    });
    match state {
        Ok(b) => Ok(b),
        Err(e) => Err(HypnosError::HardwareError(format!(
            "llama.cpp backend initialization failed: {e}"
        ))),
    }
}

// ----------------------------------------------------------------------------
// Resident model state
// ----------------------------------------------------------------------------

/// A model that is currently resident (weights loaded, optionally with a multimodal projector).
/// Owned by the engine worker thread.
pub(crate) struct LoadedLlama {
    /// Declared before `model` so the projector (which points into the text model) is torn down first.
    pub mtmd: Option<MtmdContext>,
    pub model: LlamaModel,
    pub arch: String,
}

pub(crate) fn load_model(
    cfg: &ModelConfig,
    gpu_layers: u32,
    n_ctx: u32,
    n_threads: i32,
) -> Result<LoadedLlama> {
    let backend = llama_backend()?;
    if !cfg.weights_path.exists() {
        return Err(HypnosError::Config(format!(
            "weights file for '{}' not found: {}",
            cfg.id,
            cfg.weights_path.display()
        )));
    }
    let started = Instant::now();
    let params = LlamaModelParams::default()
        .with_n_gpu_layers(gpu_layers)
        .with_use_mmap(true);
    let model = LlamaModel::load_from_file(backend, &cfg.weights_path, &params).map_err(|e| {
        HypnosError::BackendError(format!(
            "llama.cpp could not load {}: {e:?}",
            cfg.weights_path.display()
        ))
    })?;
    let arch = model
        .meta_val_str("general.architecture")
        .unwrap_or_default();

    let mtmd = match &cfg.aux_weights_path {
        Some(aux) if aux.exists() => {
            let mut mp = MtmdContextParams::default();
            mp.use_gpu = gpu_layers > 0;
            mp.print_timings = false;
            mp.n_threads = n_threads;
            mp.media_marker = CString::new(mtmd_default_marker()).unwrap_or_default();
            let ctx =
                MtmdContext::init_from_file(&aux.to_string_lossy(), &model, &mp).map_err(|e| {
                    HypnosError::BackendError(format!(
                        "libmtmd could not load projector {}: {e:?}",
                        aux.display()
                    ))
                })?;
            info!(
                model = %cfg.id,
                projector = %aux.display(),
                vision = ctx.support_vision(),
                audio = ctx.support_audio(),
                "Loaded multimodal projector"
            );
            Some(ctx)
        }
        Some(aux) => {
            warn!(model = %cfg.id, path = %aux.display(), "Projector file missing; vision/audio disabled");
            None
        }
        None => None,
    };

    info!(
        model = %cfg.id,
        arch = %arch,
        gpu_layers,
        n_layer = model.n_layer(),
        n_ctx,
        load_secs = format!("{:.1}", started.elapsed().as_secs_f32()),
        "llama.cpp model resident"
    );
    Ok(LoadedLlama { mtmd, model, arch })
}

// ----------------------------------------------------------------------------
// Prompt construction
// ----------------------------------------------------------------------------

/// Builds a Gemma 4 canonical prompt by hand (llama.cpp's built-in minimal templates only know the
/// Gemma 1-3 `<start_of_turn>` format). Media parts are rendered as `marker`, which libmtmd replaces
/// with the proper image / audio token spans. `<bos>` is added by the tokenizer (`add_special`).
fn gemma4_prompt(messages: &[ChatMessage], marker: &str, think: bool) -> String {
    let mut out = String::new();
    let system_text: Vec<String> = messages
        .iter()
        .filter(|m| m.role == "system" || m.role == "developer")
        .map(|m| m.content.extract_text().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if think || !system_text.is_empty() {
        out.push_str("<|turn>system\n");
        if think {
            out.push_str("<|think|>\n");
        }
        out.push_str(&system_text.join("\n"));
        out.push_str("<turn|>\n");
    }
    for m in messages
        .iter()
        .filter(|m| m.role != "system" && m.role != "developer")
    {
        let role = match m.role.as_str() {
            "assistant" | "model" => "model",
            _ => "user",
        };
        out.push_str("<|turn>");
        out.push_str(role);
        out.push('\n');
        out.push_str(m.content.render_with_media_marker(marker).trim());
        out.push_str("<turn|>\n");
    }
    out.push_str("<|turn>model\n");
    out
}

/// Generic fallback for non-Gemma models: the GGUF's own template through llama.cpp, else ChatML.
pub(crate) fn build_prompt(
    model: &LlamaModel,
    arch: &str,
    messages: &[ChatMessage],
    marker: &str,
    think: bool,
) -> String {
    if arch.starts_with("gemma4") || arch.starts_with("gemma-embedding") {
        return gemma4_prompt(messages, marker, think);
    }
    if let Ok(tmpl) = model.chat_template(None) {
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .filter_map(|m| {
                LlamaChatMessage::new(m.role.clone(), m.content.render_with_media_marker(marker))
                    .ok()
            })
            .collect();
        if let Ok(s) = model.apply_chat_template(&tmpl, &chat, true) {
            return s;
        }
    }
    let mut out = String::new();
    for m in messages {
        out.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n",
            m.role,
            m.content.render_with_media_marker(marker).trim()
        ));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}

/// Best-effort duration of a RIFF/WAV payload, for the transcription response.
fn wav_duration_secs(bytes: &[u8]) -> Option<f32> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).ok()?;
    let spec = reader.spec();
    Some(reader.duration() as f32 / spec.sample_rate.max(1) as f32)
}

fn usage(stats: &GenStats) -> TokenUsage {
    TokenUsage {
        prompt_tokens: stats.prompt_tokens,
        completion_tokens: stats.completion_tokens,
        total_tokens: stats.prompt_tokens + stats.completion_tokens,
        prompt_tokens_details: (stats.cached_tokens > 0).then_some(PromptTokensDetails {
            cached_tokens: stats.cached_tokens,
        }),
    }
}

// ----------------------------------------------------------------------------
// Backend
// ----------------------------------------------------------------------------

/// The running engine worker plus the placement it was built for.
struct LlamaWorker {
    handle: WorkerHandle<LlamaJob>,
    gpu_layers: u32,
    /// Context window per slot.
    n_ctx: u32,
}

pub struct GgmlLlamaBackend {
    config: ModelConfig,
    state: RwLock<BackendContextState>,
    worker: Arc<StdMutex<Option<LlamaWorker>>>,
    n_threads: i32,
    opts: EngineOptions,
}

impl GgmlLlamaBackend {
    pub fn new(config: ModelConfig, opts: EngineOptions) -> Self {
        let state = RwLock::new(BackendContextState::new(&config));
        Self {
            config,
            state,
            worker: Arc::new(StdMutex::new(None)),
            n_threads: default_threads(),
            opts,
        }
    }

    fn require_modality(&self, m: Modality) -> Result<()> {
        if self.config.supports_modality(m) {
            Ok(())
        } else {
            Err(HypnosError::ModalityMismatch {
                model: self.config.id.clone(),
                expected: self.config.modality,
                actual: m,
            })
        }
    }

    /// Hands `job` to the resident worker. The lock is held only for the send.
    fn submit(&self, job: LlamaJob) -> Result<()> {
        let guard = self
            .worker
            .lock()
            .map_err(|_| HypnosError::BackendError("worker mutex poisoned".into()))?;
        match guard.as_ref() {
            Some(w) => w.handle.submit(job),
            None => {
                let err = HypnosError::BackendError(format!(
                    "model '{}' is not resident in VRAM (scheduler must promote it first)",
                    self.config.id
                ));
                job.fail(HypnosError::BackendError(err.to_string()));
                Err(err)
            }
        }
    }

    /// Shared implementation for streaming and non-streaming chat.
    async fn generate_chat(
        &self,
        req: &ChatCompletionRequest,
        rctx: &RequestContext,
        tx: Option<mpsc::Sender<ChatStreamEvent>>,
    ) -> Result<(String, GenStats)> {
        let total_chars: usize = req
            .messages
            .iter()
            .map(|m| m.content.extract_text().chars().count())
            .sum();
        if total_chars > rctx.limits.max_prompt_chars {
            return Err(HypnosError::InvalidRequest(format!(
                "messages contain {total_chars} characters, above limits.max_prompt_chars ({})",
                rctx.limits.max_prompt_chars
            )));
        }
        let has_images = req.messages.iter().any(|m| m.content.contains_images());
        let has_audio = req.messages.iter().any(|m| m.content.contains_audio());
        if has_images {
            self.require_modality(Modality::Vision)?;
        }
        if has_audio {
            self.require_modality(Modality::SpeechToText)?;
        }
        if !has_images && !has_audio {
            self.require_modality(Modality::TextGeneration)?;
        }

        // Resolve media payloads (data: URLs, http(s), base64 audio) before queueing the job.
        let mut media: Vec<Vec<u8>> = Vec::new();
        for m in &req.messages {
            for r in m.content.media() {
                media.push(load_media(&r, &rctx.limits)?);
            }
        }

        let params = GenParams {
            max_tokens: req
                .max_tokens
                .unwrap_or(512)
                .min(rctx.limits.max_tokens_cap.max(1)),
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(0.95),
            top_k: req.top_k.unwrap_or(64),
            seed: req.seed.map(|s| s as u32).unwrap_or_else(rand::random),
        };
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit(LlamaJob::Chat(ChatJob {
            prompt: PromptSpec::Messages {
                messages: req.messages.clone(),
                think: req.think.unwrap_or(false),
            },
            media,
            params,
            rctx: rctx.clone(),
            started: Instant::now(),
            stream: tx,
            reply: reply_tx,
        }))?;
        reply_rx
            .await
            .map_err(|_| HypnosError::BackendError("engine worker dropped the request".into()))?
    }
}

#[async_trait]
impl ModelBackend for GgmlLlamaBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.worker.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        {
            let mut guard = self.state.write();
            guard.active_plan = plan.clone();
            guard.pinned_host_ptr = pinned_host_ptr;
        }
        let want_resident = plan.primary_tier == MemoryTier::Tier1GpuVram;
        let cfg = self.config.clone();
        let plan = plan.clone();
        let worker = self.worker.clone();
        let n_threads = self.n_threads;
        let opts = self.opts.clone();

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut guard = worker
                .lock()
                .map_err(|_| HypnosError::BackendError("worker mutex poisoned".into()))?;
            if !want_resident {
                // Dropping the handle stops the worker, which frees the context and the weights.
                if guard.take().is_some() {
                    info!(model = %cfg.id, to_tier = ?plan.primary_tier, "Released llama.cpp model from VRAM");
                }
                return Ok(());
            }
            let n_ctx = if plan.allocated_context_len > 0 {
                plan.allocated_context_len
            } else {
                cfg.arch.default_context_len
            };
            match guard.as_mut() {
                Some(w) if w.gpu_layers == plan.gpu_layers => {
                    if w.n_ctx != n_ctx {
                        let (tx, rx) = oneshot::channel();
                        w.handle
                            .submit(LlamaJob::Reconfigure { n_ctx, reply: tx })?;
                        rx.blocking_recv().map_err(|_| {
                            HypnosError::BackendError("engine worker dropped the reconfigure".into())
                        })??;
                        w.n_ctx = n_ctx;
                        debug!(model = %cfg.id, n_ctx, "Context rebuilt for the new window; weights already resident");
                    }
                    Ok(())
                }
                _ => {
                    // Stop the previous residency first so VRAM is free before the new split loads.
                    *guard = None;
                    let handle = spawn_worker(WorkerSpec {
                        cfg: cfg.clone(),
                        gpu_layers: plan.gpu_layers,
                        n_ctx,
                        n_threads,
                        opts,
                    })?;
                    *guard = Some(LlamaWorker {
                        handle,
                        gpu_layers: plan.gpu_layers,
                        n_ctx,
                    });
                    Ok(())
                }
            }
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("load task panicked: {e}")))?
    }

    async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
        ctx: &RequestContext,
    ) -> Result<ChatCompletionResponse> {
        let (content, stats) = self.generate_chat(req, ctx, None).await?;
        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: self.config.id.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".to_string(),
                    content,
                },
                finish_reason: stats.finish_reason.to_string(),
            }],
            usage: usage(&stats),
            hypnos_telemetry: Some(ExecutionTelemetry::for_engine(stats.timing())),
        })
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatCompletionRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<ChatStreamEvent>,
    ) -> Result<()> {
        match self.generate_chat(req, ctx, Some(tx.clone())).await {
            Ok((_, stats)) => {
                let _ = tx
                    .send(ChatStreamEvent::Done {
                        usage: usage(&stats),
                        finish_reason: stats.finish_reason.to_string(),
                        timing: stats.timing(),
                    })
                    .await;
                Ok(())
            }
            Err(e) => {
                let _ = tx.send(ChatStreamEvent::Error(e.to_string())).await;
                Err(e)
            }
        }
    }

    async fn transcribe(
        &self,
        req: &AudioTranscriptionRequest,
        rctx: &RequestContext,
    ) -> Result<AudioTranscriptionResponse> {
        self.require_modality(Modality::SpeechToText)?;
        if req.audio_bytes.is_empty() {
            return Err(HypnosError::InvalidRequest(
                "no audio data in request (multipart 'file' field is empty)".to_string(),
            ));
        }
        if req.audio_bytes.len() as u64 > rctx.limits.max_audio_bytes {
            return Err(HypnosError::InvalidRequest(format!(
                "audio payload is {} bytes, above limits.max_audio_bytes ({})",
                req.audio_bytes.len(),
                rctx.limits.max_audio_bytes
            )));
        }
        let mut instruction = match req.prompt.as_deref().map(str::trim) {
            Some(p) if !p.is_empty() => p.to_string(),
            _ => "Transcribe this audio exactly. Output only the transcript.".to_string(),
        };
        if let Some(lang) = req
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty())
        {
            instruction.push_str(&format!(" The speech is in '{lang}'."));
        }
        let marker = mtmd_default_marker();
        let user_text = format!("{marker}\n{instruction}");
        let prompt = gemma4_prompt(
            &[ChatMessage {
                role: "user".to_string(),
                content: hypnos_core::openai::ChatMessageContent::Text(user_text),
            }],
            marker,
            false,
        );
        let duration = wav_duration_secs(&req.audio_bytes);
        let started = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit(LlamaJob::Chat(ChatJob {
            prompt: PromptSpec::Raw(prompt),
            media: vec![req.audio_bytes.clone()],
            params: GenParams {
                max_tokens: 1024,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                seed: 0,
            },
            rctx: rctx.clone(),
            started,
            stream: None,
            reply: reply_tx,
        }))?;
        let (text, stats) = reply_rx
            .await
            .map_err(|_| HypnosError::BackendError("engine worker dropped the request".into()))??;

        let mut timing = stats.timing();
        timing.audio_secs = duration.map(f64::from);
        timing.rtf = duration
            .filter(|d| *d > 0.0)
            .map(|d| started.elapsed().as_secs_f64() / f64::from(d));

        Ok(AudioTranscriptionResponse {
            text: text.trim().to_string(),
            language: req.language.clone(),
            duration_secs: duration,
            hypnos_telemetry: Some(ExecutionTelemetry::for_engine(timing)),
        })
    }

    async fn embeddings(
        &self,
        req: &EmbeddingRequest,
        rctx: &RequestContext,
    ) -> Result<EmbeddingResponse> {
        self.require_modality(Modality::Embeddings)?;
        let inputs = req.input.clone().into_vec();
        if inputs.is_empty() {
            return Err(HypnosError::InvalidRequest(
                "embeddings input is empty".to_string(),
            ));
        }
        let total_chars: usize = inputs.iter().map(|t| t.chars().count()).sum();
        if total_chars > rctx.limits.max_prompt_chars {
            return Err(HypnosError::InvalidRequest(format!(
                "embedding inputs contain {total_chars} characters, above limits.max_prompt_chars"
            )));
        }
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit(LlamaJob::Embed(EmbedJob {
            inputs,
            rctx: rctx.clone(),
            started: Instant::now(),
            reply: reply_tx,
        }))?;
        let (vectors, total_tokens, timing) = reply_rx
            .await
            .map_err(|_| HypnosError::BackendError("engine worker dropped the request".into()))??;

        let data = vectors
            .into_iter()
            .enumerate()
            .map(|(index, embedding)| EmbeddingData {
                object: "embedding".to_string(),
                index,
                embedding,
            })
            .collect();
        Ok(EmbeddingResponse {
            object: "list".to_string(),
            data,
            model: self.config.id.clone(),
            usage: EmbeddingUsage {
                prompt_tokens: total_tokens,
                total_tokens,
            },
            hypnos_telemetry: Some(ExecutionTelemetry::for_engine(timing)),
        })
    }
}
