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

use crate::tools::{CallParser, Parsed, CALL_OPEN};
use async_trait::async_trait;
use chrono::Utc;
use hypnos_core::config::{MemoryTier, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    AudioTranscriptionRequest, AudioTranscriptionResponse, ChatChoice, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatResponseMessage, ChatStreamEvent, EmbeddingData,
    EmbeddingRequest, EmbeddingResponse, EmbeddingUsage, ExecutionTelemetry, LoadPhases,
    PromptTokensDetails, TokenUsage, ToolCall, ToolChoice, ToolSpec,
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
    spawn_worker, ChatJob, ChatReply, EmbedJob, GenParams, GenStats, LlamaJob, PromptSpec,
    WorkerSpec,
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
    /// Weights and projector phases; the worker adds the context and the total.
    pub phases: LoadPhases,
}

/// Whether to map the weights file or read it through pinned staging buffers: mapping is zero
/// copy on Metal and fine for CPU placements, but a CUDA upload from a mapping goes tensor by
/// tensor through pageable memory, while llama.cpp's read path streams through pinned buffers
/// with asynchronous uploads (PERF-6: 7.3 s -> see docs/bench/swap-cuda-2026-10-30.md).
pub(crate) fn default_use_mmap(gpu_layers: u32) -> bool {
    !(cfg!(feature = "cuda") && gpu_layers > 0)
}

/// The `tensor_buft_overrides` pattern that keeps the Mixture-of-Experts expert tensors of the
/// last `cpu_expert_layers` layers in a CPU buffer (MEM-5). It covers the fused `ffn_gate_up_exps`
/// of Gemma 4, which the bindings' own `add_cpu_moe_override` pattern does not match.
pub(crate) fn cpu_expert_pattern(cpu_expert_layers: u32, num_layers: u32) -> Option<String> {
    if cpu_expert_layers == 0 {
        return None;
    }
    let tensors = r"\.ffn_(up|down|gate|gate_up)_(ch|)exps";
    if cpu_expert_layers >= num_layers {
        return Some(tensors.to_string());
    }
    let layers: Vec<String> = (num_layers - cpu_expert_layers..num_layers)
        .map(|l| l.to_string())
        .collect();
    Some(format!(
        r"^blk\.({})\.ffn_(up|down|gate|gate_up)_(ch|)exps",
        layers.join("|")
    ))
}

pub(crate) fn load_model(
    cfg: &ModelConfig,
    gpu_layers: u32,
    cpu_expert_layers: u32,
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
    // llama.cpp counts the output layer as layer n_layer + 1: a full placement must ask for more
    // than `num_layers`, or the vocabulary projection stays on the CPU while the plan books it on GPU.
    let n_gpu_layers = if gpu_layers >= cfg.arch.num_layers {
        u32::MAX
    } else {
        gpu_layers
    };
    let use_mmap = cfg.use_mmap.unwrap_or_else(|| default_use_mmap(gpu_layers));
    let mut params = Box::pin(
        LlamaModelParams::default()
            .with_n_gpu_layers(n_gpu_layers)
            .with_use_mmap(use_mmap),
    );
    // The pattern must outlive the load: llama.cpp keeps a pointer to it.
    let expert_pattern = cpu_expert_pattern(cpu_expert_layers, cfg.arch.num_layers.max(1))
        .map(|p| CString::new(p).expect("pattern has no NUL"));
    if let Some(pattern) = &expert_pattern {
        params.as_mut().add_cpu_buft_override(pattern);
        info!(model = %cfg.id, cpu_expert_layers, pattern = %pattern.to_string_lossy(), "Mixture-of-Experts tensors stay in host memory");
    }
    let model = LlamaModel::load_from_file(backend, &cfg.weights_path, &params).map_err(|e| {
        HypnosError::BackendError(format!(
            "llama.cpp could not load {}: {e:?}",
            cfg.weights_path.display()
        ))
    })?;
    drop(expert_pattern);
    let weights_ms = started.elapsed().as_secs_f64() * 1000.0;
    let projector_started = Instant::now();
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

    let projector_ms = mtmd
        .is_some()
        .then(|| projector_started.elapsed().as_secs_f64() * 1000.0);
    info!(
        model = %cfg.id,
        arch = %arch,
        gpu_layers,
        n_layer = model.n_layer(),
        n_ctx,
        load_mode = if use_mmap { "mmap" } else { "read" },
        weights_secs = format!("{:.1}", weights_ms / 1000.0),
        projector_secs = projector_ms.map(|ms| format!("{:.1}", ms / 1000.0)),
        "llama.cpp model resident"
    );
    Ok(LoadedLlama {
        mtmd,
        model,
        arch,
        phases: LoadPhases {
            load_mode: if use_mmap { "mmap" } else { "read" }.to_string(),
            weights_ms,
            projector_ms,
            context_ms: 0.0,
            total_ms: 0.0,
            ..LoadPhases::default()
        },
    })
}

// ----------------------------------------------------------------------------
// Prompt construction
// ----------------------------------------------------------------------------

/// Gemma 4's turn format, as its GGUF chat template renders it (the template itself is not run:
/// llama.cpp's basic template call cannot evaluate it), with the proper image / audio token
/// spans. `<bos>` is added by the tokenizer (`add_special`). With `tools`, the declarations close
/// the system turn and past calls and their results sit in the model's turn (see `tools.rs`);
/// `prefill` is appended after the generation prompt (a forced tool call's opening).
fn gemma4_prompt(
    messages: &[ChatMessage],
    marker: &str,
    think: bool,
    tools: &[ToolSpec],
    prefill: &str,
) -> String {
    let mut out = String::new();
    let system_text: Vec<String> = messages
        .iter()
        .filter(|m| m.role == "system" || m.role == "developer")
        .map(|m| m.content.extract_text().trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if think || !system_text.is_empty() || !tools.is_empty() {
        out.push_str("<|turn>system\n");
        if think {
            out.push_str("<|think|>\n");
        }
        out.push_str(&system_text.join("\n"));
        for tool in tools {
            out.push_str(&crate::tools::declaration(tool));
        }
        out.push_str("<turn|>\n");
    }
    let turns: Vec<&ChatMessage> = messages
        .iter()
        .filter(|m| m.role != "system" && m.role != "developer")
        .collect();
    let mut prev_role: Option<&str> = None;
    // Whether the rendering stopped right after tool results (the model continues there).
    let mut ended_on_results = false;
    for (i, m) in turns.iter().enumerate() {
        if m.role == "tool" {
            continue; // rendered with the assistant message whose call it answers
        }
        let role = match m.role.as_str() {
            "assistant" | "model" => "model",
            _ => "user",
        };
        if !(role == "model" && prev_role == Some("assistant")) {
            out.push_str("<|turn>");
            out.push_str(role);
            out.push('\n');
        }
        let calls = m.tool_calls.as_deref().unwrap_or(&[]);
        for call in calls {
            out.push_str(&crate::tools::call_text(call));
        }
        let mut answered = false;
        if !calls.is_empty() {
            for follow in turns[i + 1..].iter().take_while(|f| f.role == "tool") {
                let name = follow
                    .tool_call_id
                    .as_deref()
                    .and_then(|id| calls.iter().find(|c| c.id == id))
                    .map(|c| c.function.name.as_str())
                    .or(follow.name.as_deref())
                    .unwrap_or("unknown");
                out.push_str(&crate::tools::response_text(
                    name,
                    follow.content.extract_text().trim(),
                ));
                answered = true;
            }
        }
        let content = m.content.render_with_media_marker(marker);
        let content = content.trim();
        out.push_str(content);
        let next_role = turns[i + 1..]
            .iter()
            .find(|f| f.role != "tool")
            .map(|f| f.role.as_str());
        let continues =
            role == "model" && next_role == Some("assistant") && (calls.is_empty() || answered);
        ended_on_results = false;
        if !calls.is_empty() && !answered {
            out.push_str(crate::tools::RESPONSE_OPEN);
        } else if continues {
        } else if answered && content.is_empty() && next_role.is_none() {
            ended_on_results = true;
        } else {
            out.push_str("<turn|>\n");
        }
        prev_role = Some(m.role.as_str());
    }
    if !ended_on_results {
        out.push_str("<|turn>model\n");
    }
    out.push_str(prefill);
    out
}

/// LFM3's ChatML turns, as its GGUF chat template renders them (the Jinja template itself is not
/// run: llama.cpp's basic template call cannot evaluate it). The GGUF does not add a BOS token, so
/// `<|startoftext|>` is written here. `think` is `None` for a model without a thinking block
/// (LFM3-Audio); otherwise the generation prompt opens one (`<think>`, the engine keeps the
/// thoughts out of the answer) or closes it empty, and earlier answers carry an empty block with
/// any thoughts stripped, as the template does. Images and audio are the media marker alone:
/// libmtmd adds `<|vision_start|>` / `<|vision_end|>` around them.
fn lfm3_prompt(messages: &[ChatMessage], marker: &str, think: Option<bool>) -> String {
    let mut out = String::from("<|startoftext|>");
    let system_text: Vec<String> = messages
        .iter()
        .filter(|m| m.role == "system" || m.role == "developer")
        .map(|m| {
            m.content
                .render_with_media_marker(marker)
                .trim()
                .to_string()
        })
        .filter(|s| !s.is_empty())
        .collect();
    if !system_text.is_empty() {
        out.push_str("<|im_start|>system\n");
        out.push_str(&system_text.join("\n"));
        out.push_str("<|im_end|>\n");
    }
    for m in messages
        .iter()
        .filter(|m| m.role != "system" && m.role != "developer")
    {
        let content = m.content.render_with_media_marker(marker);
        let content = content.trim();
        match m.role.as_str() {
            "assistant" | "model" => {
                out.push_str("<|im_start|>assistant\n");
                let answer = content.rsplit("</think>").next().unwrap_or(content).trim();
                if think.is_some() {
                    out.push_str("<think></think>");
                }
                out.push_str(answer);
            }
            "tool" => {
                out.push_str("<|im_start|>tool\n<tool_response>\n");
                out.push_str(content);
                out.push_str("\n</tool_response>");
            }
            _ => {
                out.push_str("<|im_start|>user\n");
                out.push_str(content);
            }
        }
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n");
    match think {
        Some(true) => out.push_str("<think>"),
        Some(false) => out.push_str("<think></think>"),
        None => {}
    }
    out
}

/// Whether the model's native tool format is implemented (Gemma 4, `tools.rs`).
pub(crate) fn supports_tools(arch: &str) -> bool {
    arch.starts_with("gemma4")
}

/// Gemma 4 and LFM3 have their own renderers; other models get the GGUF's own template through
/// llama.cpp, else ChatML.
pub(crate) fn build_prompt(
    model: &LlamaModel,
    arch: &str,
    messages: &[ChatMessage],
    marker: &str,
    think: bool,
    tools: &[ToolSpec],
    prefill: &str,
) -> String {
    if supports_tools(arch) || arch.starts_with("gemma-embedding") {
        return gemma4_prompt(messages, marker, think, tools, prefill);
    }
    if arch == "lfm3" {
        // The chat models' template has a thinking switch; the audio model's has none.
        let thinks = model
            .chat_template(None)
            .ok()
            .and_then(|t| t.to_str().ok().map(|t| t.contains("enable_thinking")))
            .unwrap_or(false);
        return lfm3_prompt(messages, marker, thinks.then_some(think));
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

/// The single-turn transcription prompt, as each model family was trained for it. LFM audio
/// models take the task as a system prompt (`Perform ASR in <language>.`) and the audio alone as
/// the user turn; the others (Gemma 4) get the audio followed by the instruction.
pub(crate) fn transcription_prompt(
    model: &LlamaModel,
    arch: &str,
    prompt: Option<&str>,
    language: Option<&str>,
) -> String {
    let marker = mtmd_default_marker();
    build_prompt(
        model,
        arch,
        &transcription_messages(arch, marker, prompt, language),
        marker,
        false,
        &[],
        "",
    )
}

fn transcription_messages(
    arch: &str,
    marker: &str,
    prompt: Option<&str>,
    language: Option<&str>,
) -> Vec<ChatMessage> {
    let text = |role: &str, t: String| ChatMessage {
        role: role.to_string(),
        content: hypnos_core::openai::ChatMessageContent::Text(t),
        ..ChatMessage::default()
    };
    let prompt = prompt.map(str::trim).filter(|p| !p.is_empty());
    let language = language.map(str::trim).filter(|l| !l.is_empty());
    if arch.starts_with("lfm") {
        let system = match (prompt, language) {
            (Some(p), _) => p.to_string(),
            (None, Some(l)) => format!("Perform ASR in {}.", language_name(l)),
            (None, None) => "Perform ASR.".to_string(),
        };
        return vec![text("system", system), text("user", marker.to_string())];
    }
    let mut instruction = prompt
        .unwrap_or("Transcribe this audio exactly. Output only the transcript.")
        .to_string();
    if let Some(lang) = language {
        instruction.push_str(&format!(" The speech is in '{lang}'."));
    }
    vec![text("user", format!("{marker}\n{instruction}"))]
}

/// English name of an ISO 639-1 code (the API's `language`), for prompts that name the language;
/// anything else is passed through as given.
fn language_name(code: &str) -> &str {
    match code.to_ascii_lowercase().as_str() {
        "ar" => "Arabic",
        "da" => "Danish",
        "de" => "German",
        "el" => "Greek",
        "en" => "English",
        "es" => "Spanish",
        "fi" => "Finnish",
        "fr" => "French",
        "he" => "Hebrew",
        "hi" => "Hindi",
        "it" => "Italian",
        "ja" => "Japanese",
        "ko" => "Korean",
        "nl" => "Dutch",
        "pl" => "Polish",
        "pt" => "Portuguese",
        "ru" => "Russian",
        "sv" => "Swedish",
        "tr" => "Turkish",
        "zh" => "Chinese",
        _ => code,
    }
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
    cpu_expert_layers: u32,
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

    /// Shared implementation for streaming and non-streaming chat. With tools the engine's raw
    /// output goes through a [`CallParser`]: text and calls reach the stream as they are parsed,
    /// and the result carries both.
    async fn generate_chat(
        &self,
        req: &ChatCompletionRequest,
        rctx: &RequestContext,
        tx: Option<mpsc::Sender<ChatStreamEvent>>,
    ) -> Result<ChatOutput> {
        let (tools, choice) = req.active_tools().map_err(HypnosError::InvalidRequest)?;
        // No grammar constraint here: a required or named call is forced by writing its opening.
        let prefill = match &choice {
            ToolChoice::Required => format!("{CALL_OPEN}call:"),
            ToolChoice::Function(name) => format!("{CALL_OPEN}call:{name}{{"),
            ToolChoice::Auto | ToolChoice::None => String::new(),
        };
        let with_tools = !tools.is_empty();
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
        let id_prefix = Uuid::new_v4().simple().to_string()[..12].to_string();
        // With tools, the engine streams into a parser that forwards text and calls.
        let (engine_tx, forward) = match (tx, with_tools) {
            (Some(tx), true) => {
                let (itx, mut irx) = mpsc::channel::<ChatStreamEvent>(128);
                let mut parser = CallParser::new(&prefill, &id_prefix);
                let forward = tokio::spawn(async move {
                    let mut calls = Vec::new();
                    while let Some(event) = irx.recv().await {
                        let parsed = match event {
                            ChatStreamEvent::Delta(text) => parser.feed(&text),
                            other => {
                                let _ = tx.send(other).await;
                                continue;
                            }
                        };
                        forward_parsed(&tx, parsed, &mut calls).await;
                    }
                    forward_parsed(&tx, parser.finish(), &mut calls).await;
                    calls
                });
                (Some(itx), Some(forward))
            }
            (tx, _) => (tx, None),
        };
        let (think, think_budget) = req.thinking();
        // Thinking off: a block the model opens anyway (small LFM3 models do) is closed at once.
        let think_budget = if think { think_budget } else { Some(0) };
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit(LlamaJob::Chat(ChatJob {
            think_budget,
            prompt: PromptSpec::Messages {
                messages: req.messages.clone(),
                think,
                tools: tools.to_vec(),
                prefill: prefill.clone(),
            },
            tool_markers: with_tools,
            media,
            params,
            rctx: rctx.clone(),
            started: Instant::now(),
            stream: engine_tx,
            reply: reply_tx,
        }))?;
        let ChatReply {
            text: raw,
            reasoning,
            mut stats,
        } = reply_rx
            .await
            .map_err(|_| HypnosError::BackendError("engine worker dropped the request".into()))??;
        let (content, tool_calls) = match forward {
            // The engine dropped its sender with the request; the parser has flushed.
            Some(forward) => (
                String::new(),
                forward.await.map_err(|e| {
                    HypnosError::BackendError(format!("tool-call parser failed: {e}"))
                })?,
            ),
            None if with_tools => {
                let mut parser = CallParser::new(&prefill, &id_prefix);
                let mut parsed = parser.feed(&raw);
                parsed.extend(parser.finish());
                let mut text = String::new();
                let mut calls = Vec::new();
                for p in parsed {
                    match p {
                        Parsed::Text(t) => text.push_str(&t),
                        Parsed::Call(c) => calls.push(c),
                    }
                }
                (text.trim().to_string(), calls)
            }
            None => (raw, Vec::new()),
        };
        if !tool_calls.is_empty() {
            stats.finish_reason = "tool_calls";
        }
        Ok(ChatOutput {
            content,
            reasoning: (!reasoning.is_empty()).then_some(reasoning),
            tool_calls,
            stats,
        })
    }
}

/// Asks a worker that is about to be dropped to save its slots into the park pool (MEM-4).
/// Blocking; runs on the placement task, never on the async runtime.
fn park_slots(cfg: &ModelConfig, opts: &EngineOptions, worker: &LlamaWorker) {
    if !opts.kv_park.enabled() {
        return;
    }
    let (tx, rx) = oneshot::channel();
    if worker.handle.submit(LlamaJob::Park { reply: tx }).is_err() {
        return;
    }
    if let Ok(Some(parked)) = rx.blocking_recv() {
        opts.kv_park.put(&cfg.id, parked);
    }
}

/// A chat answer: its text, the thoughts before it, the calls it made (with tools) and the
/// engine's numbers.
struct ChatOutput {
    content: String,
    reasoning: Option<String>,
    tool_calls: Vec<ToolCall>,
    stats: GenStats,
}

async fn forward_parsed(
    tx: &mpsc::Sender<ChatStreamEvent>,
    parsed: Vec<Parsed>,
    calls: &mut Vec<ToolCall>,
) {
    for p in parsed {
        let event = match p {
            Parsed::Text(t) => ChatStreamEvent::Delta(t),
            Parsed::Call(c) => {
                calls.push(c.clone());
                ChatStreamEvent::ToolCall(c)
            }
        };
        let _ = tx.send(event).await;
    }
}

#[async_trait]
impl ModelBackend for GgmlLlamaBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn last_load_phases(&self) -> Option<LoadPhases> {
        self.state.read().last_load_phases.clone()
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

        let phases = tokio::task::spawn_blocking(move || -> Result<Option<LoadPhases>> {
            let mut guard = worker
                .lock()
                .map_err(|_| HypnosError::BackendError("worker mutex poisoned".into()))?;
            if !want_resident {
                // Dropping the handle stops the worker, which frees the context and the weights.
                if let Some(w) = guard.take() {
                    park_slots(&cfg, &opts, &w);
                    info!(model = %cfg.id, to_tier = ?plan.primary_tier, "Released llama.cpp model from VRAM");
                }
                return Ok(None);
            }
            let n_ctx = if plan.allocated_context_len > 0 {
                plan.allocated_context_len
            } else {
                cfg.arch.default_context_len
            };
            match guard.as_mut() {
                Some(w)
                    if w.gpu_layers == plan.gpu_layers
                        && w.cpu_expert_layers == plan.cpu_expert_layers =>
                {
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
                    Ok(None)
                }
                _ => {
                    // Stop the previous residency first so VRAM is free before the new split loads.
                    if let Some(w) = guard.take() {
                        park_slots(&cfg, &opts, &w);
                    }
                    let parked = opts.kv_park.take(&cfg.id);
                    let (handle, phases) = spawn_worker(WorkerSpec {
                        cfg: cfg.clone(),
                        gpu_layers: plan.gpu_layers,
                        cpu_expert_layers: plan.cpu_expert_layers,
                        n_ctx,
                        n_threads,
                        opts,
                        parked,
                    })?;
                    *guard = Some(LlamaWorker {
                        handle,
                        gpu_layers: plan.gpu_layers,
                        cpu_expert_layers: plan.cpu_expert_layers,
                        n_ctx,
                    });
                    Ok(Some(phases))
                }
            }
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("load task panicked: {e}")))??;
        if let Some(phases) = phases {
            info!(
                model = %self.config.id,
                load_mode = %phases.load_mode,
                weights_ms = phases.weights_ms as u64,
                projector_ms = phases.projector_ms.map(|v| v as u64),
                context_ms = phases.context_ms as u64,
                total_ms = phases.total_ms as u64,
                "Tier 1 load phases"
            );
            self.state.write().last_load_phases = Some(phases);
        }
        Ok(())
    }

    async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
        ctx: &RequestContext,
    ) -> Result<ChatCompletionResponse> {
        let ChatOutput {
            content,
            reasoning,
            tool_calls,
            stats,
        } = self.generate_chat(req, ctx, None).await?;
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
                    reasoning_content: reasoning,
                    tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
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
            Ok(ChatOutput { stats, .. }) => {
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
        let duration = wav_duration_secs(&req.audio_bytes);
        let started = Instant::now();
        let (reply_tx, reply_rx) = oneshot::channel();
        self.submit(LlamaJob::Chat(ChatJob {
            think_budget: None,
            prompt: PromptSpec::Transcription {
                prompt: req.prompt.clone(),
                language: req.language.clone(),
            },
            tool_markers: false,
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
        let ChatReply { text, stats, .. } = reply_rx
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

#[cfg(test)]
mod tests {
    use super::{cpu_expert_pattern, gemma4_prompt, lfm3_prompt, transcription_messages};

    #[test]
    fn lfm3_renders_like_its_chat_template() {
        let conversation = [
            msg("system", "Be brief."),
            msg("user", "Hi <__media__>"),
            msg("assistant", "<think>greet back</think>Hello!"),
            msg("user", "Bye"),
        ];
        let expected_turns = "<|startoftext|><|im_start|>system\nBe brief.<|im_end|>\n\
             <|im_start|>user\nHi <__media__><|im_end|>\n\
             <|im_start|>assistant\n<think></think>Hello!<|im_end|>\n\
             <|im_start|>user\nBye<|im_end|>\n<|im_start|>assistant\n";
        assert_eq!(
            lfm3_prompt(&conversation, "<__media__>", Some(false)),
            format!("{expected_turns}<think></think>")
        );
        assert_eq!(
            lfm3_prompt(&conversation, "<__media__>", Some(true)),
            format!("{expected_turns}<think>")
        );
        // LFM3-Audio: no thinking block anywhere.
        assert_eq!(
            lfm3_prompt(&[msg("user", "Hi"), msg("assistant", "Yo")], "<m>", None),
            "<|startoftext|><|im_start|>user\nHi<|im_end|>\n\
             <|im_start|>assistant\nYo<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn transcription_prompts_follow_the_model_family() {
        let as_json = |m: Vec<ChatMessage>| serde_json::to_value(m).unwrap();
        assert_eq!(
            as_json(transcription_messages("lfm3", "<m>", None, Some("de"))),
            as_json(vec![
                msg("system", "Perform ASR in German."),
                msg("user", "<m>")
            ])
        );
        assert_eq!(
            as_json(transcription_messages("lfm3", "<m>", Some("  "), None))[0],
            as_json(vec![msg("system", "Perform ASR.")])[0]
        );
        assert_eq!(
            as_json(transcription_messages("lfm3", "<m>", None, Some("tlh")))[0],
            as_json(vec![msg("system", "Perform ASR in tlh.")])[0]
        );
        assert_eq!(
            as_json(transcription_messages("gemma4", "<m>", None, Some("de"))),
            as_json(vec![msg(
                "user",
                "<m>\nTranscribe this audio exactly. Output only the transcript. The speech is in 'de'."
            )])
        );
    }

    #[test]
    fn the_expert_pattern_covers_all_or_the_last_layers() {
        assert_eq!(cpu_expert_pattern(0, 30), None);
        assert_eq!(
            cpu_expert_pattern(30, 30).as_deref(),
            Some(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")
        );
        assert_eq!(
            cpu_expert_pattern(3, 30).as_deref(),
            Some(r"^blk\.(27|28|29)\.ffn_(up|down|gate|gate_up)_(ch|)exps")
        );
    }
    use hypnos_core::openai::{ChatMessage, ChatMessageContent, FunctionCall, ToolCall, ToolSpec};
    use serde_json::json;

    fn msg(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.into(),
            content: ChatMessageContent::Text(text.into()),
            ..ChatMessage::default()
        }
    }

    /// The same conversation through the GGUF's Jinja template gives exactly this (minus <bos>).
    #[test]
    fn a_tool_round_renders_like_the_chat_template() {
        let tool: ToolSpec = serde_json::from_value(json!({
            "type": "function",
            "function": {"name": "get_weather", "description": "Weather",
                         "parameters": {"type": "object",
                                        "properties": {"city": {"type": "string"}},
                                        "required": ["city"]}}
        }))
        .unwrap();
        let mut call = msg("assistant", "");
        call.tool_calls = Some(vec![ToolCall {
            id: "c1".into(),
            kind: "function".into(),
            function: FunctionCall {
                name: "get_weather".into(),
                arguments: r#"{"city":"Paris"}"#.into(),
            },
        }]);
        let mut result = msg("tool", "18 C");
        result.tool_call_id = Some("c1".into());
        let messages = vec![
            msg("system", "Be brief."),
            msg("user", "Weather in Paris?"),
            call,
            result,
        ];
        let prompt = gemma4_prompt(
            &messages,
            "<__media__>",
            false,
            std::slice::from_ref(&tool),
            "",
        );
        assert_eq!(
            prompt,
            "<|turn>system\nBe brief.<|tool>declaration:get_weather{description:<|\"|>Weather<|\"|>,parameters:{properties:{city:{type:<|\"|>STRING<|\"|>}},required:[<|\"|>city<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|><turn|>\n\
             <|turn>user\nWeather in Paris?<turn|>\n\
             <|turn>model\n<|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|><|tool_response>response:get_weather{value:<|\"|>18 C<|\"|>}<tool_response|>"
        );

        // The answer that followed continues the same model turn; a new question opens one.
        let mut later = messages.clone();
        later.push(msg("assistant", "It is 18 C."));
        later.push(msg("user", "Thanks"));
        let prompt = gemma4_prompt(&later, "<__media__>", false, &[tool], "<|tool_call>call:");
        assert!(prompt.contains("<tool_response|>It is 18 C.<turn|>\n<|turn>user\nThanks<turn|>\n<|turn>model\n<|tool_call>call:"));
        // Without tools nothing changes for an ordinary chat.
        let plain = gemma4_prompt(&[msg("user", "Hi")], "<__media__>", false, &[], "");
        assert_eq!(plain, "<|turn>user\nHi<turn|>\n<|turn>model\n");
    }
}
