//! llama.cpp / libmtmd engine, linked in-process through the vendored `llama-cpp-2` bindings.
//!
//! Serves three of the configured models:
//! * `gemma4:26b-q4`  — text + vision chat (`/v1/chat/completions`)
//! * `gemma4:e4b-q4`  — audio understanding / speech-to-text (`/v1/audio/transcriptions`) plus text + vision chat
//! * `gemmaembedding2` — dense embeddings (`/v1/embeddings`)
//!
//! Weights are loaded when the scheduler promotes the model to Tier 1 (with `n_gpu_layers` taken from
//! the placement plan) and released again when it is demoted, so VRAM accounting in `hypnos-memory`
//! mirrors what the GPU actually holds.

use std::ffi::CString;
use std::num::NonZeroU32;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};
use std::time::Instant;

use async_trait::async_trait;
use chrono::Utc;
use hypnos_core::config::{KvDataType, MemoryTier, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    AudioTranscriptionRequest, AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, ChatStreamEvent, EmbeddingData,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, TokenUsage,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdContextParams, MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use parking_lot::RwLock;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::media::load_media;
use crate::{default_threads, BackendContextState};

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

fn backend_err<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> HypnosError + '_ {
    move |e| HypnosError::BackendError(format!("{what}: {e:?}"))
}

// ----------------------------------------------------------------------------
// Resident model state
// ----------------------------------------------------------------------------

/// A model that is currently resident (weights loaded, optionally with a multimodal projector).
struct LoadedLlama {
    /// Declared before `model` so the projector (which points into the text model) is torn down first.
    mtmd: Option<MtmdContext>,
    model: LlamaModel,
    gpu_layers: u32,
    n_ctx: u32,
    arch: String,
}

fn load_model(
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
    Ok(LoadedLlama {
        mtmd,
        model,
        gpu_layers,
        n_ctx,
        arch,
    })
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
fn build_prompt(
    loaded: &LoadedLlama,
    messages: &[ChatMessage],
    marker: &str,
    think: bool,
) -> String {
    if loaded.arch.starts_with("gemma4") || loaded.arch.starts_with("gemma-embedding") {
        return gemma4_prompt(messages, marker, think);
    }
    if let Ok(tmpl) = loaded.model.chat_template(None) {
        let chat: Vec<LlamaChatMessage> = messages
            .iter()
            .filter_map(|m| {
                LlamaChatMessage::new(m.role.clone(), m.content.render_with_media_marker(marker))
                    .ok()
            })
            .collect();
        if let Ok(s) = loaded.model.apply_chat_template(&tmpl, &chat, true) {
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

// ----------------------------------------------------------------------------
// Generation core (runs on a blocking thread, model mutex held)
// ----------------------------------------------------------------------------

struct GenParams {
    max_tokens: u32,
    temperature: f32,
    top_p: f32,
    top_k: i32,
    seed: u32,
    n_ctx_override: Option<u32>,
}

struct GenStats {
    prompt_tokens: u32,
    completion_tokens: u32,
    finish_reason: &'static str,
}

fn kv_types(dtype: KvDataType) -> (KvCacheType, KvCacheType) {
    match dtype {
        KvDataType::F16 | KvDataType::Bf16 => (KvCacheType::F16, KvCacheType::F16),
        KvDataType::Q8_0 => (KvCacheType::Q8_0, KvCacheType::Q8_0),
        KvDataType::Q4_0 => (KvCacheType::Q4_0, KvCacheType::Q4_0),
    }
}

/// Prefill `prompt` (+ `media`) and sample up to `max_tokens` tokens, invoking `on_piece` for each
/// decoded text fragment. `on_piece` returns `false` to stop early (e.g. the client went away).
fn run_generation(
    loaded: &mut LoadedLlama,
    cfg: &ModelConfig,
    n_threads: i32,
    prompt: &str,
    media: &[Vec<u8>],
    gp: &GenParams,
    rctx: &RequestContext,
    mut on_piece: impl FnMut(&str) -> bool,
) -> Result<GenStats> {
    if !rctx.should_continue() {
        return Err(HypnosError::Cancelled);
    }
    let backend = llama_backend()?;
    let LoadedLlama {
        mtmd,
        model,
        gpu_layers,
        n_ctx: resident_ctx,
        ..
    } = loaded;

    let n_ctx = gp
        .n_ctx_override
        .unwrap_or(*resident_ctx)
        .clamp(512, model.n_ctx_train().max(512));
    let n_batch = n_ctx.min(2048);
    let (type_k, type_v) = kv_types(cfg.arch.kv_dtype);
    let cparams = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_batch(n_batch)
        .with_n_ubatch(n_batch.min(512))
        .with_n_threads(n_threads)
        .with_n_threads_batch(n_threads)
        .with_offload_kqv(*gpu_layers > 0)
        .with_swa_full(false)
        .with_type_k(type_k)
        .with_type_v(type_v);
    let mut ctx = model
        .new_context(backend, cparams)
        .map_err(backend_err("llama.cpp context creation failed"))?;
    let vocab = model.vocab();

    // ---- prefill -------------------------------------------------------------------------
    let (n_prompt, mut n_past): (usize, i32) = if !media.is_empty() {
        let m = mtmd.as_mut().ok_or_else(|| {
            HypnosError::InvalidRequest(format!(
                "model '{}' has no multimodal projector configured; it cannot take images or audio",
                cfg.id
            ))
        })?;
        let mut bitmaps = Vec::with_capacity(media.len());
        for bytes in media {
            let bm = MtmdBitmap::from_buffer(m, bytes, false).map_err(|e| {
                HypnosError::InvalidRequest(format!(
                    "could not decode media attachment ({} bytes): {e:?}. Supported audio containers are WAV, MP3 and FLAC (browser recordings in WebM/Opus or M4A must be converted first); images may be PNG, JPEG, BMP or GIF",
                    bytes.len()
                ))
            })?;
            if bm.is_audio() && !m.support_audio() {
                return Err(HypnosError::ModalityMismatch {
                    model: cfg.id.clone(),
                    expected: cfg.modality,
                    actual: Modality::SpeechToText,
                });
            }
            if !bm.is_audio() && !m.support_vision() {
                return Err(HypnosError::ModalityMismatch {
                    model: cfg.id.clone(),
                    expected: cfg.modality,
                    actual: Modality::Vision,
                });
            }
            bitmaps.push(bm);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = m
            .tokenize(
                MtmdInputText {
                    text: prompt.to_string(),
                    add_special: true,
                    parse_special: true,
                },
                &refs,
            )
            .map_err(backend_err("libmtmd tokenization failed"))?;
        let n_tokens = chunks.total_tokens();
        if n_tokens + 8 > n_ctx as usize {
            return Err(HypnosError::InvalidRequest(format!(
                "prompt + media need {n_tokens} tokens but the context window is {n_ctx}; raise num_ctx"
            )));
        }
        let n_past = chunks
            .eval_chunks(m, &mut ctx, 0, 0, n_batch as i32, true)
            .map_err(backend_err("libmtmd prefill failed"))?;
        (n_tokens, n_past)
    } else {
        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.len() + 8 > n_ctx as usize {
            return Err(HypnosError::InvalidRequest(format!(
                "prompt needs {} tokens but the context window is {n_ctx}; raise num_ctx",
                tokens.len()
            )));
        }
        let last = tokens.len().saturating_sub(1);
        let mut pos: i32 = 0;
        for chunk in tokens.chunks(n_batch as usize) {
            let mut batch = LlamaBatch::new(chunk.len(), 1);
            for (i, t) in chunk.iter().enumerate() {
                let abs = pos as usize + i;
                batch
                    .add(*t, pos + i as i32, &[0], abs == last)
                    .map_err(backend_err("batch add failed"))?;
            }
            ctx.decode(&mut batch)
                .map_err(backend_err("llama.cpp prefill decode failed"))?;
            pos += chunk.len() as i32;
        }
        (tokens.len(), pos)
    };

    // ---- sampling ------------------------------------------------------------------------
    let mut sampler = if gp.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(gp.top_k.max(1)),
            LlamaSampler::top_p(gp.top_p.clamp(0.0, 1.0), 1),
            LlamaSampler::temp(gp.temperature),
            LlamaSampler::dist(gp.seed),
        ])
    };

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut batch = LlamaBatch::new(1, 1);
    let mut completion: u32 = 0;
    let mut finish_reason = "length";
    let max_tokens = gp.max_tokens.max(1);
    // Gemma 4 wraps reasoning in `<|channel>thought ... <channel|>`; even with thinking disabled the
    // model may emit an empty channel. Everything inside it is swallowed, as are bare special tokens.
    let mut in_channel = false;
    let mut channel_header_pending = false;

    while completion < max_tokens {
        if !rctx.should_continue() {
            finish_reason = "cancelled";
            break;
        }
        let token = sampler.sample(&ctx, -1);
        sampler.accept(token);
        if vocab.is_eog(token) {
            finish_reason = "stop";
            break;
        }
        let piece = vocab.token_to_piece(token, true, None);
        if piece.as_slice() == b"<turn|>" || piece.as_slice() == b"<end_of_turn>" {
            finish_reason = "stop";
            break;
        }
        completion += 1;
        let is_marker = is_special_marker(&piece);
        if piece.as_slice() == b"<|channel>" {
            in_channel = true;
            channel_header_pending = true;
        } else if piece.as_slice() == b"<channel|>" {
            in_channel = false;
        } else if !vocab.is_control(token) && !is_marker && !in_channel {
            let mut text = String::with_capacity(
                decoder
                    .max_utf8_buffer_length(piece.len())
                    .unwrap_or(piece.len() * 3),
            );
            let _ = decoder.decode_to_string(&piece, &mut text, false);
            // Right after a channel closes the model emits the leading newline(s) of the answer.
            if channel_header_pending {
                channel_header_pending = false;
                let trimmed = text.trim_start_matches(['\n', '\r']).to_string();
                if trimmed.is_empty() {
                    text.clear();
                } else {
                    text = trimmed;
                }
            }
            if !text.is_empty() && !on_piece(&text) {
                finish_reason = "stop";
                break;
            }
        }
        if (n_past as u32).saturating_add(1) >= n_ctx {
            finish_reason = "length";
            break;
        }
        batch.clear();
        batch
            .add(token, n_past, &[0], true)
            .map_err(backend_err("batch add failed"))?;
        ctx.decode(&mut batch)
            .map_err(backend_err("llama.cpp decode failed"))?;
        n_past += 1;
    }

    Ok(GenStats {
        prompt_tokens: n_prompt as u32,
        completion_tokens: completion,
        finish_reason,
    })
}

/// Single-token special markers such as `<|turn>`, `<|think|>`, `<|image|>`, `<channel|>`.
fn is_special_marker(piece: &[u8]) -> bool {
    let is_open = piece.starts_with(b"<|") && piece.ends_with(b">");
    let is_close = piece.starts_with(b"<") && piece.ends_with(b"|>");
    (is_open || is_close) && piece.len() <= 24 && !piece.contains(&b' ')
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

/// Best-effort duration of a RIFF/WAV payload, for the transcription response.
fn wav_duration_secs(bytes: &[u8]) -> Option<f32> {
    let reader = hound::WavReader::new(std::io::Cursor::new(bytes)).ok()?;
    let spec = reader.spec();
    Some(reader.duration() as f32 / spec.sample_rate.max(1) as f32)
}

// ----------------------------------------------------------------------------
// Backend
// ----------------------------------------------------------------------------

pub struct GgmlLlamaBackend {
    config: ModelConfig,
    state: RwLock<BackendContextState>,
    loaded: Arc<StdMutex<Option<LoadedLlama>>>,
    n_threads: i32,
}

impl GgmlLlamaBackend {
    pub fn new(config: ModelConfig) -> Self {
        let state = RwLock::new(BackendContextState::new(&config));
        Self {
            config,
            state,
            loaded: Arc::new(StdMutex::new(None)),
            n_threads: default_threads(),
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

        // Resolve media payloads (data: URLs, http(s), base64 audio) before taking the model lock.
        let mut media: Vec<Vec<u8>> = Vec::new();
        for m in &req.messages {
            for r in m.content.media() {
                media.push(load_media(&r, &rctx.limits)?);
            }
        }

        let gp = GenParams {
            max_tokens: req
                .max_tokens
                .unwrap_or(512)
                .min(rctx.limits.max_tokens_cap.max(1)),
            temperature: req.temperature.unwrap_or(1.0),
            top_p: req.top_p.unwrap_or(0.95),
            top_k: req.top_k.unwrap_or(64),
            seed: req.seed.map(|s| s as u32).unwrap_or_else(rand::random),
            n_ctx_override: req.num_ctx,
        };
        let think = req.think.unwrap_or(false);
        let messages = req.messages.clone();
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;
        let rctx = rctx.clone();

        tokio::task::spawn_blocking(move || -> Result<(String, GenStats)> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("model mutex poisoned".into()))?;
            let resident = guard.as_mut().ok_or_else(|| {
                HypnosError::BackendError(format!(
                    "model '{}' is not resident in VRAM (scheduler must promote it first)",
                    cfg.id
                ))
            })?;
            let prompt = build_prompt(resident, &messages, mtmd_default_marker(), think);
            debug!(model = %cfg.id, prompt_chars = prompt.len(), media = media.len(), "chat prefill");
            let mut text = String::new();
            let stats = run_generation(resident, &cfg, n_threads, &prompt, &media, &gp, &rctx, |piece| {
                text.push_str(piece);
                match &tx {
                    Some(tx) => tx.blocking_send(ChatStreamEvent::Delta(piece.to_string())).is_ok(),
                    None => true,
                }
            })?;
            Ok((text, stats))
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("inference task panicked: {e}")))?
    }
}

#[async_trait]
impl ModelBackend for GgmlLlamaBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.loaded.lock().map(|g| g.is_some()).unwrap_or(false)
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
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;

        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("model mutex poisoned".into()))?;
            if !want_resident {
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
                Some(resident) if resident.gpu_layers == plan.gpu_layers => {
                    resident.n_ctx = n_ctx;
                    debug!(model = %cfg.id, n_ctx, "Context budget updated; weights already resident");
                    Ok(())
                }
                _ => {
                    // Drop the previous residency first so VRAM is free before the new split loads.
                    *guard = None;
                    *guard = Some(load_model(&cfg, plan.gpu_layers, n_ctx, n_threads)?);
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
            usage: TokenUsage {
                prompt_tokens: stats.prompt_tokens,
                completion_tokens: stats.completion_tokens,
                total_tokens: stats.prompt_tokens + stats.completion_tokens,
            },
            hypnos_telemetry: None,
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
                        usage: TokenUsage {
                            prompt_tokens: stats.prompt_tokens,
                            completion_tokens: stats.completion_tokens,
                            total_tokens: stats.prompt_tokens + stats.completion_tokens,
                        },
                        finish_reason: stats.finish_reason.to_string(),
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
        let media = vec![req.audio_bytes.clone()];
        let duration = wav_duration_secs(&req.audio_bytes);
        let gp = GenParams {
            max_tokens: 1024,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
            seed: 0,
            n_ctx_override: None,
        };
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;
        let language = req.language.clone();
        let rctx = rctx.clone();

        let text = tokio::task::spawn_blocking(move || -> Result<String> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("model mutex poisoned".into()))?;
            let resident = guard.as_mut().ok_or_else(|| {
                HypnosError::BackendError(format!("model '{}' is not resident in VRAM", cfg.id))
            })?;
            let mut text = String::new();
            run_generation(
                resident,
                &cfg,
                n_threads,
                &prompt,
                &media,
                &gp,
                &rctx,
                |piece| {
                    text.push_str(piece);
                    true
                },
            )?;
            Ok(text)
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("transcription task panicked: {e}")))??;

        Ok(AudioTranscriptionResponse {
            text: text.trim().to_string(),
            language,
            duration_secs: duration,
            hypnos_telemetry: None,
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
        let rctx = rctx.clone();
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;

        let (vectors, total_tokens) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<Vec<f32>>, u32)> {
                let backend = llama_backend()?;
                let mut guard = loaded
                    .lock()
                    .map_err(|_| HypnosError::BackendError("model mutex poisoned".into()))?;
                let resident = guard.as_mut().ok_or_else(|| {
                    HypnosError::BackendError(format!("model '{}' is not resident in VRAM", cfg.id))
                })?;
                let model = &resident.model;
                let n_ctx = resident
                    .n_ctx
                    .clamp(512, cfg.arch.max_context_len.max(512))
                    .min(model.n_ctx_train().max(512));
                // Non-causal encoders need the whole sequence inside one micro-batch.
                let cparams = LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(n_ctx))
                    .with_n_batch(n_ctx)
                    .with_n_ubatch(n_ctx)
                    .with_n_threads(n_threads)
                    .with_n_threads_batch(n_threads)
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Unspecified);
                let mut ctx = model
                    .new_context(backend, cparams)
                    .map_err(backend_err("llama.cpp embedding context creation failed"))?;
                let vocab = model.vocab();

                let mut out = Vec::with_capacity(inputs.len());
                let mut total: u32 = 0;
                for text in &inputs {
                    if !rctx.should_continue() {
                        return Err(HypnosError::Cancelled);
                    }
                    let mut tokens = vocab.tokenize(text.as_bytes(), true, true);
                    if tokens.is_empty() {
                        tokens = vocab.tokenize(b" ", true, true);
                    }
                    tokens.truncate(n_ctx as usize);
                    total += tokens.len() as u32;
                    let mut batch = LlamaBatch::new(tokens.len(), 1);
                    batch
                        .add_sequence(&tokens, 0, true)
                        .map_err(backend_err("batch add failed"))?;
                    ctx.clear_kv_cache();
                    ctx.decode(&mut batch)
                        .map_err(backend_err("llama.cpp embedding decode failed"))?;
                    let emb: Vec<f32> = match ctx.embeddings_seq_ith(0) {
                        Ok(e) => e.to_vec(),
                        Err(_) => ctx
                            .embeddings_ith(tokens.len() as i32 - 1)
                            .map_err(backend_err("no embeddings produced"))?
                            .to_vec(),
                    };
                    out.push(l2_normalize(&emb));
                }
                Ok((out, total))
            })
            .await
            .map_err(|e| HypnosError::BackendError(format!("embedding task panicked: {e}")))??;

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
            hypnos_telemetry: None,
        })
    }
}
