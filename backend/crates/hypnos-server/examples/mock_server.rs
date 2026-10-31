//! The real Hypnos router, scheduler, auth and request log in front of mock engines: every route
//! answers as the server does, with deterministic content and no model weights. The SDKs in
//! `sdk/` run their test suites against it (API-7).
//!
//! ```text
//! cargo run -p hypnos-server --no-default-features --example mock_server -- \
//!     [--port N] [--api-key KEY]... [--limited-key KEY:REQUESTS_PER_MINUTE]... [--not-ready]
//! ```
//!
//! `--port 0` (the default) takes a free port. The first line on stdout is
//! `listening on http://127.0.0.1:<port>`; tests wait for it.
//!
//! The models are the default lineup (`gemma4:26b-q4`, `gemma4:e4b-q4`, `gemmaembedding2`,
//! `kokoro-82m-q4`, `moss-image`, `docling`, `gliner2.5-multi`) with their real modalities:
//! - chat answers `echo: <last user text>`, or `1 2 3 … max_tokens` when that text starts with
//!   `count`; streams send one word per chunk, 10 ms apart. With `tools` it calls one
//!   (`finish_reason: "tool_calls"`, arguments `{"text": <last user text>}`): the named one for a
//!   function `tool_choice`, the first for `"required"`, and for `"auto"` the first whose name
//!   appears in the last user text. After a `tool` message it answers `tool said: <its content>`;
//! - embeddings are 8-dimensional unit vectors derived from the text;
//! - transcription answers `mock transcript of <n> bytes`;
//! - speech is 50 ms of silence per input word (24 kHz, 16-bit mono WAV or PCM);
//! - images are a 1x1 PNG each;
//! - documents have 3 pages (or the requested `pages`), each with one picture;
//! - entities are the words that start with a capital letter, labelled in turn with the labels.
//!
//! Control routes (outside auth, never in the request log):
//! - `POST /mock/fail` `{"model": "...", "error": "<kind>", "times": 1}` makes the model's next
//!   `times` engine calls fail. Kinds: `busy` (429 `model_busy`), `device_out_of_memory`,
//!   `insufficient_memory` (507), `server_error` (500), `timeout` (504), `not_available` (503),
//!   `stream_error` (a chat stream fails after its first words). The server retries a
//!   `device_out_of_memory` once by itself when it can evict an idle model.
//! - `POST /mock/ready` `{"ready": false}` flips readiness (503 with `Retry-After` while false).
//! - `POST /mock/reset` clears the injected failures and makes the server ready.

use async_trait::async_trait;
use axum::{extract::State, routing::post, Json, Router};
use base64::Engine;
use hypnos_backends::media::{encode_pcm16_raw, encode_wav_pcm16};
use hypnos_core::config::{
    AcceleratorKind, ApiKeyLimits, ApiKeyProfile, HardwareConfig, MemoryTier, ModelConfig,
};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::*;
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use hypnos_memory::{DeviceMemoryCapabilities, TieredMemoryManager};
use hypnos_scheduler::ModelScheduler;
use hypnos_server::request_log::{LogLevel, RequestLog};
use hypnos_server::{auth, build_router, metrics::Metrics, pictures, quota, AppState};
use serde::Deserialize;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;

const GB: u64 = 1024 * 1024 * 1024;
const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
const SAMPLE_RATE: u32 = 24_000;

/// Failures injected through `POST /mock/fail`, per model, consumed one per engine call.
#[derive(Default)]
struct Faults(Mutex<HashMap<String, VecDeque<String>>>);

impl Faults {
    fn take(&self, model: &str) -> Option<String> {
        self.0.lock().unwrap().get_mut(model)?.pop_front()
    }

    /// The injected failure for this call as an engine error; `stream_error` is left to the
    /// streaming chat, which fails after it started.
    fn check(&self, model: &str) -> Result<Option<String>> {
        let Some(kind) = self.take(model) else {
            return Ok(None);
        };
        let model = model.to_string();
        Err(match kind.as_str() {
            "busy" => HypnosError::Busy {
                model,
                in_flight: 8,
                limit: 8,
            },
            "device_out_of_memory" => HypnosError::DeviceOutOfMemory {
                model,
                detail: "mock: compute buffer allocation failed".into(),
            },
            "insufficient_memory" => HypnosError::InsufficientMemory {
                model,
                required_bytes: 16 * GB,
                available_bytes: GB,
            },
            "timeout" => HypnosError::Timeout(1),
            "not_available" => {
                HypnosError::Config(format!("mock: model '{model}' is not available"))
            }
            "stream_error" => return Ok(Some(kind)),
            _ => HypnosError::BackendError(format!("mock: injected {kind}")),
        })
    }
}

struct MockEngine {
    config: ModelConfig,
    resident: AtomicBool,
    faults: Arc<Faults>,
}

fn last_user_text(req: &ChatCompletionRequest) -> String {
    req.messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .map(|m| m.content.extract_text())
        .unwrap_or_default()
}

/// The call a request with tools makes (see the module docs), if any.
fn mock_tool_call(req: &ChatCompletionRequest) -> Result<Option<ToolCall>> {
    let (tools, choice) = req.active_tools().map_err(HypnosError::InvalidRequest)?;
    if tools.is_empty() || req.messages.last().is_some_and(|m| m.role == "tool") {
        return Ok(None);
    }
    let text = last_user_text(req);
    let tool = match &choice {
        ToolChoice::Function(name) => tools.iter().find(|t| &t.function.name == name),
        ToolChoice::Required => tools.first(),
        _ => tools.iter().find(|t| text.contains(&t.function.name)),
    };
    Ok(tool.map(|t| ToolCall {
        id: format!("call_{}", uuid::Uuid::new_v4().simple()),
        kind: "function".into(),
        function: FunctionCall {
            name: t.function.name.clone(),
            arguments: serde_json::json!({ "text": text }).to_string(),
        },
    }))
}

/// With thinking on (`reasoning_effort` other than `none`, or `think`), the thoughts the mock
/// "had": `Mock thoughts (<level>).`, where the level is `unbounded` for `think` alone.
fn mock_reasoning(req: &ChatCompletionRequest) -> Option<String> {
    let (think, _) = req.thinking();
    think.then(|| {
        let level = match req.reasoning_effort {
            Some(effort) => serde_json::to_value(effort)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default(),
            None => "unbounded".to_string(),
        };
        format!("Mock thoughts ({level}).")
    })
}

/// The words a chat request produces (see the module docs).
fn reply_words(req: &ChatCompletionRequest) -> Vec<String> {
    if let Some(result) = req.messages.last().filter(|m| m.role == "tool") {
        return std::iter::once("tool said:".to_string())
            .chain(
                result
                    .content
                    .extract_text()
                    .split_whitespace()
                    .map(str::to_string),
            )
            .collect();
    }
    let text = last_user_text(req);
    let max = req.max_tokens.unwrap_or(512).clamp(1, 8192) as usize;
    let mut words: Vec<String> = if text.trim_start().starts_with("count") {
        (1..=max).map(|n| n.to_string()).collect()
    } else {
        std::iter::once("echo:".to_string())
            .chain(text.split_whitespace().map(str::to_string))
            .collect()
    };
    words.truncate(max);
    words
}

fn usage(req: &ChatCompletionRequest, completion: usize) -> TokenUsage {
    let prompt: usize = req
        .messages
        .iter()
        .map(|m| m.content.extract_text().split_whitespace().count())
        .sum();
    TokenUsage {
        prompt_tokens: prompt as u32,
        completion_tokens: completion as u32,
        total_tokens: (prompt + completion) as u32,
        prompt_tokens_details: None,
    }
}

fn embedding(text: &str) -> Vec<f32> {
    let mut v = [0f32; 8];
    for (i, b) in text.bytes().enumerate() {
        v[(i * 31 + usize::from(b)) % 8] += 1.0;
    }
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / norm).collect()
}

/// 1-based pages from `"3"` or `"2-5"`, within `1..=count`.
fn page_numbers(spec: Option<&str>, count: u32) -> Result<Vec<u32>> {
    let Some(spec) = spec.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok((1..=count).collect());
    };
    let bad = || HypnosError::InvalidRequest(format!("pages: cannot read '{spec}'"));
    let (a, b) = match spec.split_once('-') {
        Some((a, b)) => (
            a.trim().parse().map_err(|_| bad())?,
            b.trim().parse().map_err(|_| bad())?,
        ),
        None => {
            let n = spec.parse().map_err(|_| bad())?;
            (n, n)
        }
    };
    if a == 0 || a > b || b > count {
        return Err(HypnosError::InvalidRequest(format!(
            "pages '{spec}' is outside 1-{count}"
        )));
    }
    Ok((a..=b).collect())
}

#[async_trait]
impl ModelBackend for MockEngine {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }

    async fn apply_placement(&self, plan: &LayerPlacementPlan, _ptr: Option<usize>) -> Result<()> {
        self.resident.store(
            plan.primary_tier == MemoryTier::Tier1GpuVram,
            Ordering::SeqCst,
        );
        Ok(())
    }

    async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
        _ctx: &RequestContext,
    ) -> Result<ChatCompletionResponse> {
        self.faults.check(&req.model)?;
        if let Some(call) = mock_tool_call(req)? {
            return Ok(ChatCompletionResponse {
                id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
                object: "chat.completion".into(),
                created: chrono::Utc::now().timestamp(),
                model: req.model.clone(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatResponseMessage {
                        role: "assistant".into(),
                        content: String::new(),
                        reasoning_content: None,
                        tool_calls: Some(vec![call]),
                    },
                    finish_reason: "tool_calls".into(),
                }],
                usage: usage(req, 1),
                hypnos_telemetry: None,
            });
        }
        let words = reply_words(req);
        let finish = if words.len() as u32 >= req.max_tokens.unwrap_or(512) {
            "length"
        } else {
            "stop"
        };
        Ok(ChatCompletionResponse {
            id: format!("chatcmpl-{}", uuid::Uuid::new_v4().simple()),
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".into(),
                    content: words.join(" "),
                    reasoning_content: mock_reasoning(req),
                    tool_calls: None,
                },
                finish_reason: finish.into(),
            }],
            usage: usage(req, words.len()),
            hypnos_telemetry: None,
        })
    }

    async fn chat_completion_stream(
        &self,
        req: &ChatCompletionRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<ChatStreamEvent>,
    ) -> Result<()> {
        let fail_midway = self.faults.check(&req.model)?.is_some();
        if let Some(call) = mock_tool_call(req)? {
            let _ = tx.send(ChatStreamEvent::ToolCall(call)).await;
            let _ = tx
                .send(ChatStreamEvent::Done {
                    usage: usage(req, 1),
                    finish_reason: "tool_calls".into(),
                    timing: EngineTiming::default(),
                })
                .await;
            return Ok(());
        }
        if let Some(reasoning) = mock_reasoning(req) {
            for (i, word) in reasoning.split(' ').enumerate() {
                let piece = if i == 0 {
                    word.to_string()
                } else {
                    format!(" {word}")
                };
                if tx
                    .send(ChatStreamEvent::ReasoningDelta(piece))
                    .await
                    .is_err()
                {
                    return Err(HypnosError::Cancelled);
                }
            }
        }
        let words = reply_words(req);
        for (i, word) in words.iter().enumerate() {
            if !ctx.should_continue() {
                return Err(HypnosError::Cancelled);
            }
            if fail_midway && i == 2 {
                let _ = tx
                    .send(ChatStreamEvent::Error("mock: injected stream_error".into()))
                    .await;
                return Ok(());
            }
            let piece = if i == 0 {
                word.clone()
            } else {
                format!(" {word}")
            };
            if tx.send(ChatStreamEvent::Delta(piece)).await.is_err() {
                return Err(HypnosError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let finish = if words.len() as u32 >= req.max_tokens.unwrap_or(512) {
            "length"
        } else {
            "stop"
        };
        let _ = tx
            .send(ChatStreamEvent::Done {
                usage: usage(req, words.len()),
                finish_reason: finish.into(),
                timing: EngineTiming::default(),
            })
            .await;
        Ok(())
    }

    async fn embeddings(
        &self,
        req: &EmbeddingRequest,
        _ctx: &RequestContext,
    ) -> Result<EmbeddingResponse> {
        self.faults.check(&req.model)?;
        let inputs = req.input.clone().into_vec();
        let tokens: usize = inputs.iter().map(|t| t.split_whitespace().count()).sum();
        Ok(EmbeddingResponse {
            object: "list".into(),
            data: inputs
                .iter()
                .enumerate()
                .map(|(index, text)| EmbeddingData {
                    object: "embedding".into(),
                    index,
                    embedding: embedding(text),
                })
                .collect(),
            model: req.model.clone(),
            usage: EmbeddingUsage {
                prompt_tokens: tokens as u32,
                total_tokens: tokens as u32,
            },
            hypnos_telemetry: None,
        })
    }

    async fn transcribe(
        &self,
        req: &AudioTranscriptionRequest,
        _ctx: &RequestContext,
    ) -> Result<AudioTranscriptionResponse> {
        self.faults.check(&req.model)?;
        Ok(AudioTranscriptionResponse {
            text: format!("mock transcript of {} bytes", req.audio_bytes.len()),
            language: req.language.clone(),
            duration_secs: None,
            hypnos_telemetry: None,
        })
    }

    async fn synthesize_speech(
        &self,
        req: &AudioSpeechRequest,
        _ctx: &RequestContext,
    ) -> Result<AudioSpeechOutput> {
        self.faults.check(&req.model)?;
        let words = req.input.split_whitespace().count().max(1);
        let samples = vec![0f32; words * SAMPLE_RATE as usize / 20];
        let (audio_bytes, content_type) = if req.response_format.eq_ignore_ascii_case("pcm") {
            (encode_pcm16_raw(&samples), "audio/pcm")
        } else {
            (encode_wav_pcm16(&samples, SAMPLE_RATE)?, "audio/wav")
        };
        Ok(AudioSpeechOutput {
            audio_bytes,
            content_type: content_type.into(),
            telemetry: ExecutionTelemetry::for_engine(EngineTiming::default()),
        })
    }

    async fn synthesize_speech_stream(
        &self,
        req: &AudioSpeechRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<AudioStreamEvent>,
    ) -> Result<()> {
        self.faults.check(&req.model)?;
        for _ in req.input.split_whitespace() {
            if !ctx.should_continue() {
                return Err(HypnosError::Cancelled);
            }
            let chunk = AudioChunk {
                pcm: vec![0f32; SAMPLE_RATE as usize / 20],
                sample_rate: SAMPLE_RATE,
            };
            if tx.send(AudioStreamEvent::Chunk(chunk)).await.is_err() {
                return Err(HypnosError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let _ = tx
            .send(AudioStreamEvent::Done {
                timing: EngineTiming::default(),
            })
            .await;
        Ok(())
    }

    async fn convert_document(
        &self,
        req: &DocumentConversionRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<DocumentStreamEvent>,
    ) -> Result<()> {
        self.faults.check(&req.model)?;
        if req.file_bytes.is_empty() {
            return Err(HypnosError::InvalidRequest("mock: empty document".into()));
        }
        let page_count = 3;
        let pages = page_numbers(req.pages.as_deref(), page_count)?;
        let name = req.filename.clone().unwrap_or_else(|| "document".into());
        let _ = tx
            .send(DocumentStreamEvent::Started {
                page_count,
                pages: pages.clone(),
            })
            .await;
        for page in pages {
            if !ctx.should_continue() {
                return Err(HypnosError::Cancelled);
            }
            let page = DocumentPage {
                page,
                markdown: format!(
                    "## Page {page}\n\n{name}, {} bytes.\n\n<!-- image -->",
                    req.file_bytes.len()
                ),
                json: req
                    .wants_json()
                    .then(|| serde_json::json!({ "page_no": page })),
                pictures: 1,
                tables: 0,
                ms: 1.0,
                picture_urls: Vec::new(),
                picture_images: vec![ExtractedPicture {
                    data_uri: format!("data:image/png;base64,{PNG_1X1}"),
                    mime: "image/png".into(),
                    width: Some(1),
                    height: Some(1),
                    dpi: Some(72),
                    bbox: Some(PictureBox {
                        l: 10.0,
                        t: 20.0,
                        r: 11.0,
                        b: 21.0,
                        coord_origin: "TOPLEFT".into(),
                    }),
                    page_size: Some(PageSize {
                        width: 612.0,
                        height: 792.0,
                    }),
                    caption: Some(format!("Figure {page}")),
                    self_ref: Some(format!("#/pictures/{}", page - 1)),
                }],
            };
            if tx.send(DocumentStreamEvent::Page(page)).await.is_err() {
                return Err(HypnosError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let _ = tx
            .send(DocumentStreamEvent::Done {
                timing: EngineTiming::default(),
            })
            .await;
        Ok(())
    }

    async fn extract_entities(
        &self,
        req: &EntityExtractionRequest,
        _ctx: &RequestContext,
    ) -> Result<EntityExtractionResponse> {
        self.faults.check(&req.model)?;
        if req.labels.is_empty() {
            return Err(HypnosError::InvalidRequest(
                "labels must not be empty".into(),
            ));
        }
        let texts = req.input.clone().into_vec();
        let mut characters = 0u64;
        let data = texts
            .iter()
            .enumerate()
            .map(|(index, text)| {
                characters += text.chars().count() as u64;
                let mut entities = Vec::new();
                let mut start = None;
                let chars: Vec<char> = text.chars().collect();
                for i in 0..=chars.len() {
                    let in_word = i < chars.len() && chars[i].is_alphanumeric();
                    match (start, in_word) {
                        (None, true) => start = Some(i),
                        (Some(s), false) => {
                            if chars[s].is_uppercase() {
                                entities.push(Entity {
                                    text: chars[s..i].iter().collect(),
                                    label: req.labels[entities.len() % req.labels.len()].clone(),
                                    start: s,
                                    end: i,
                                    score: 0.9,
                                });
                            }
                            start = None;
                        }
                        _ => {}
                    }
                }
                EntityResult {
                    object: "entities".into(),
                    index,
                    entities,
                }
            })
            .collect();
        Ok(EntityExtractionResponse {
            object: "list".into(),
            model: req.model.clone(),
            data,
            usage: EntityUsage {
                texts: texts.len() as u32,
                characters,
            },
            hypnos_telemetry: None,
        })
    }

    async fn generate_image(
        &self,
        req: &ImageGenerationRequest,
        _ctx: &RequestContext,
    ) -> Result<ImageGenerationResponse> {
        self.faults.check(&req.model)?;
        let png = base64::engine::general_purpose::STANDARD
            .decode(PNG_1X1)
            .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(png);
        Ok(ImageGenerationResponse {
            created: chrono::Utc::now().timestamp(),
            data: (0..req.n.max(1))
                .map(|_| GeneratedImage {
                    b64_json: Some(b64.clone()),
                    revised_prompt: None,
                })
                .collect(),
            hypnos_telemetry: None,
        })
    }
}

struct Args {
    port: u16,
    keys: Vec<String>,
    limited: Vec<(String, u32)>,
    ready: bool,
}

fn parse_args() -> anyhow::Result<Args> {
    let mut args = Args {
        port: 0,
        keys: Vec::new(),
        limited: Vec::new(),
        ready: true,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || {
            it.next()
                .ok_or_else(|| anyhow::anyhow!("{arg} needs a value"))
        };
        match arg.as_str() {
            "--port" => args.port = value()?.parse()?,
            "--api-key" => args.keys.push(value()?),
            "--limited-key" => {
                let v = value()?;
                let (key, rpm) = v.rsplit_once(':').ok_or_else(|| {
                    anyhow::anyhow!("--limited-key wants KEY:REQUESTS_PER_MINUTE")
                })?;
                args.limited.push((key.to_string(), rpm.parse()?));
            }
            "--not-ready" => args.ready = false,
            other => anyhow::bail!("unknown argument {other}"),
        }
    }
    Ok(args)
}

/// The default lineup with small weights and placeholder files, so every model fits next to the
/// others and nothing is read from disk.
fn mock_models() -> anyhow::Result<Vec<ModelConfig>> {
    let dir = std::env::temp_dir().join("hypnos-mock-server");
    std::fs::create_dir_all(&dir)?;
    let mut models = hypnos_server::default_profile::default_six_model_config().models;
    for m in &mut models {
        let path = dir.join(format!("{}.bin", m.id.replace([':', '/'], "_")));
        if !path.exists() {
            std::fs::write(&path, b"GGUF")?;
        }
        m.weights_path = path;
        m.aux_weights_path = None;
        m.text_encoder_path = None;
        m.tokenizer_path = None;
        m.voices_path = None;
        m.total_weight_bytes = 256 * 1024 * 1024;
        m.initial_tier = MemoryTier::Tier3NvmeSsd;
        m.arch.default_context_len = m.arch.default_context_len.min(4096);
        m.arch.max_context_len = m.arch.max_context_len.min(16384);
        m.arch.activation_workspace_bytes = 64 * 1024 * 1024;
        m.arch.non_layer_overhead_bytes = 64 * 1024 * 1024;
    }
    Ok(models)
}

#[derive(Clone)]
struct Control {
    faults: Arc<Faults>,
    ready: Arc<AtomicBool>,
}

#[derive(Deserialize)]
struct FailRequest {
    model: String,
    error: String,
    #[serde(default = "one")]
    times: usize,
}

fn one() -> usize {
    1
}

async fn mock_fail(
    State(c): State<Control>,
    Json(req): Json<FailRequest>,
) -> Json<serde_json::Value> {
    let mut faults = c.faults.0.lock().unwrap();
    let queue = faults.entry(req.model.clone()).or_default();
    queue.extend(std::iter::repeat_n(req.error.clone(), req.times));
    Json(serde_json::json!({ "model": req.model, "pending": queue.len() }))
}

#[derive(Deserialize)]
struct ReadyRequest {
    ready: bool,
}

async fn mock_ready(
    State(c): State<Control>,
    Json(req): Json<ReadyRequest>,
) -> Json<serde_json::Value> {
    c.ready.store(req.ready, Ordering::SeqCst);
    Json(serde_json::json!({ "ready": req.ready }))
}

async fn mock_reset(State(c): State<Control>) -> Json<serde_json::Value> {
    c.faults.0.lock().unwrap().clear();
    c.ready.store(true, Ordering::SeqCst);
    Json(serde_json::json!({ "reset": true }))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "warn".into()),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = parse_args()?;
    let models = mock_models()?;
    let hw = HardwareConfig {
        accelerator: AcceleratorKind::Vulkan, // no probing of this machine's GPU
        device_id: 0,
        vram_budget_bytes: 48 * GB,
        vram_safety_margin_bytes: 256 * 1024 * 1024,
        pinned_ram_budget_bytes: 32 * GB,
        lock_host_memory: Some(false),
        simulate_physical_memory_bytes: 0,
        max_process_memory_bytes: 0,
        kv_park: false,
        kv_park_bytes: 0,
    };
    let mut caps = DeviceMemoryCapabilities::detect(&hw);
    caps.physical_memory_bytes = 0;
    caps.installed_memory_bytes = 0;
    caps.pinned_ram_budget_bytes = hw.pinned_ram_budget_bytes;
    let memory = Arc::new(TieredMemoryManager::with_capabilities(
        &hw,
        caps,
        models.clone(),
    )?);

    let faults = Arc::new(Faults::default());
    let engines: Vec<Arc<dyn ModelBackend>> = models
        .into_iter()
        .map(|config| {
            Arc::new(MockEngine {
                config,
                resident: AtomicBool::new(false),
                faults: faults.clone(),
            }) as Arc<dyn ModelBackend>
        })
        .collect();
    let limits = hypnos_core::config::LimitsConfig::default();
    let scheduler = Arc::new(ModelScheduler::new(memory, engines, &limits));

    let keyed: Vec<ApiKeyProfile> = args
        .limited
        .iter()
        .map(|(key, rpm)| ApiKeyProfile {
            label: format!("limited-{rpm}"),
            key: key.clone(),
            requests_per_minute: Some(*rpm),
            max_concurrent: None,
            tokens_per_day: None,
        })
        .collect();
    let auth = Arc::new(auth::ApiAuth::new(
        &args.keys,
        &keyed,
        &ApiKeyLimits::default(),
        false,
    ));
    let ready = Arc::new(AtomicBool::new(args.ready));
    let state = AppState {
        scheduler,
        limits,
        metrics: Arc::new(Metrics::default()),
        request_log: Arc::new(RequestLog::new(LogLevel::Info, 200)),
        quota: Arc::new(quota::KeyLimiter::new(auth.key_limits())),
        auth,
        pictures: Arc::new(pictures::PictureStore::open(
            None,
            Duration::from_secs(3600),
            0,
        )?),
        // Mock engines have no KV state; the switch and its status still answer as the server's.
        kv_park: Arc::new(hypnos_backends::kv_park::KvParkPool::disabled()),
        ready: ready.clone(),
        started_at: Instant::now(),
        version: env!("CARGO_PKG_VERSION"),
    };

    let control = Router::new()
        .route("/mock/fail", post(mock_fail))
        .route("/mock/ready", post(mock_ready))
        .route("/mock/reset", post(mock_reset))
        .with_state(Control { faults, ready });
    let app = build_router(state, &[]).merge(control);

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", args.port)).await?;
    println!("listening on http://{}", listener.local_addr()?);
    axum::serve(listener, app).await?;
    Ok(())
}
