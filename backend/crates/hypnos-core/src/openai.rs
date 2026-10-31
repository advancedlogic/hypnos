use crate::config::{MemoryTier, Modality};
use crate::kv_cache::LayerPlacementPlan;
use serde::{Deserialize, Serialize};

// ============================================================================
// 1. Chat, Vision & Audio-in (/v1/chat/completions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ChatMessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Default for ChatMessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

/// `content: null` (an assistant message that only calls tools) reads as empty text.
fn content_or_empty<'de, D>(d: D) -> Result<ChatMessageContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    Ok(Option::<ChatMessageContent>::deserialize(d)?.unwrap_or_default())
}

/// A media attachment extracted from a chat message, in the order it appeared.
#[derive(Debug, Clone)]
pub enum MediaRef {
    /// `image_url` part: a `data:` URL or an http(s) URL.
    ImageUrl(String),
    /// `input_audio` part: base64 payload plus a format hint such as `wav` or `mp3`.
    Audio { data_b64: String, format: String },
}

impl ChatMessageContent {
    pub fn extract_text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }

    pub fn contains_images(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::ImageUrl { .. })),
        }
    }

    pub fn contains_audio(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Parts(parts) => parts
                .iter()
                .any(|p| matches!(p, ContentPart::InputAudio { .. })),
        }
    }

    pub fn image_urls(&self) -> Vec<String> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => Some(image_url.url.clone()),
                    _ => None,
                })
                .collect(),
        }
    }

    /// All media attachments (images and audio clips) in message order.
    pub fn media(&self) -> Vec<MediaRef> {
        match self {
            Self::Text(_) => Vec::new(),
            Self::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::ImageUrl { image_url } => {
                        Some(MediaRef::ImageUrl(image_url.url.clone()))
                    }
                    ContentPart::InputAudio { input_audio } => Some(MediaRef::Audio {
                        data_b64: input_audio.data.clone(),
                        format: input_audio.format.clone(),
                    }),
                    ContentPart::Text { .. } => None,
                })
                .collect(),
        }
    }

    /// Renders the content as a single text string where every media part is replaced by
    /// `marker`, so an engine can later splice the encoded media back in at the right spot.
    pub fn render_with_media_marker(&self, marker: &str) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Parts(parts) => {
                let mut out = String::new();
                for p in parts {
                    match p {
                        ContentPart::Text { text } => out.push_str(text),
                        ContentPart::ImageUrl { .. } | ContentPart::InputAudio { .. } => {
                            if !out.is_empty() && !out.ends_with('\n') {
                                out.push('\n');
                            }
                            out.push_str(marker);
                            out.push('\n');
                        }
                    }
                }
                out
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrlSpec },
    InputAudio { input_audio: InputAudioSpec },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrlSpec {
    pub url: String,
    #[serde(default)]
    pub detail: Option<String>,
}

/// OpenAI-style `input_audio` part: base64-encoded audio bytes plus a container/format hint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputAudioSpec {
    pub data: String,
    #[serde(default = "default_audio_format")]
    pub format: String,
}

fn default_audio_format() -> String {
    "wav".to_string()
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, deserialize_with = "content_or_empty")]
    pub content: ChatMessageContent,
    /// An assistant message's calls (OpenAI tool calling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    /// A `tool` message: the call it answers.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// A `tool` message: the function's name (optional; resolved from `tool_call_id`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// A tool the model may call (`tools[]`); only `type: "function"` exists.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema of the arguments.
    #[serde(default)]
    pub parameters: Option<serde_json::Value>,
}

/// A call the model made; `arguments` is a JSON object serialized as a string (as in OpenAI).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

fn function_type() -> String {
    "function".to_string()
}

/// `tool_choice`: `"none"`, `"auto"` (default with tools), `"required"`, or one function.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolChoice {
    None,
    Auto,
    Required,
    Function(String),
}

impl ToolChoice {
    /// Reads the OpenAI forms: a string, or `{"type": "function", "function": {"name": ...}}`.
    pub fn parse(value: Option<&serde_json::Value>) -> Result<Self, String> {
        match value {
            None | Some(serde_json::Value::Null) => Ok(Self::Auto),
            Some(serde_json::Value::String(s)) => match s.as_str() {
                "none" => Ok(Self::None),
                "auto" => Ok(Self::Auto),
                "required" | "any" => Ok(Self::Required),
                other => Err(format!("tool_choice {other:?} is not none, auto or required")),
            },
            Some(v) => v
                .pointer("/function/name")
                .and_then(|n| n.as_str())
                .map(|n| Self::Function(n.to_string()))
                .ok_or_else(|| "tool_choice must be a string or {\"type\": \"function\", \"function\": {\"name\": ...}}".to_string()),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub top_k: Option<i32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stream: Option<bool>,
    /// Optional Hypnos extension: explicit context window reservation override
    #[serde(default)]
    pub num_ctx: Option<u32>,
    /// Optional Hypnos extension: enable the model's thinking (Gemma 4, LFM3) without a budget.
    /// Default off; `reasoning_effort` takes precedence.
    #[serde(default)]
    pub think: Option<bool>,
    /// How much the model may think before it answers (OpenAI's `reasoning_effort`): `none`
    /// turns thinking off, the other levels turn it on with a budget of thinking tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Functions the model may call (OpenAI tool calling).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<serde_json::Value>,
}

/// Thinking level of a chat request. `minimal` (an OpenAI value) is read as `low`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningEffort {
    None,
    #[serde(alias = "minimal")]
    Low,
    Medium,
    High,
}

impl ReasoningEffort {
    /// Thinking tokens allowed before the engine closes the thinking block; `None` = unbounded.
    pub fn budget(self) -> Option<u32> {
        match self {
            ReasoningEffort::None => Some(0),
            ReasoningEffort::Low => Some(512),
            ReasoningEffort::Medium => Some(2048),
            ReasoningEffort::High => Some(6144),
        }
    }
}

impl ChatCompletionRequest {
    /// Whether the model thinks, and its budget of thinking tokens (`None` = unbounded):
    /// `reasoning_effort` first, else `think`.
    pub fn thinking(&self) -> (bool, Option<u32>) {
        match self.reasoning_effort {
            Some(ReasoningEffort::None) => (false, None),
            Some(effort) => (true, effort.budget()),
            None => (self.think.unwrap_or(false), None),
        }
    }

    /// The tools in effect: none when absent, empty or `tool_choice: "none"`.
    pub fn active_tools(&self) -> Result<(&[ToolSpec], ToolChoice), String> {
        let choice = ToolChoice::parse(self.tool_choice.as_ref())?;
        let tools = self.tools.as_deref().unwrap_or(&[]);
        if tools.is_empty() || choice == ToolChoice::None {
            return Ok((&[], ToolChoice::None));
        }
        if let ToolChoice::Function(name) = &choice {
            if !tools.iter().any(|t| &t.function.name == name) {
                return Err(format!("tool_choice names {name:?}, which is not in tools"));
            }
        }
        Ok((tools, choice))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatChoice>,
    pub usage: TokenUsage,
    /// Hypnos telemetry extension showing memory switch time and tier placement
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponseMessage {
    pub role: String,
    /// The answer's text; empty when the model only called tools.
    pub content: String,
    /// The model's thoughts before the answer (with thinking on and a model that thinks).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    /// OpenAI-style breakdown; present when part of the prompt was served from the KV cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    /// Prompt positions reused from a slot's KV cache instead of being prefilled again.
    pub cached_tokens: u32,
}

/// Incremental events emitted by a streaming chat completion.
// `Done` (with the engine timings) is sent once per stream; boxing it would only add indirection.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    /// A new piece of generated text.
    Delta(String),
    /// A new piece of the model's thoughts (`delta.reasoning_content`).
    ReasoningDelta(String),
    /// A complete tool call (sent as one `tool_calls` delta, arguments whole).
    ToolCall(ToolCall),
    /// Generation finished; carries final usage, finish reason and engine timings.
    Done {
        usage: TokenUsage,
        finish_reason: String,
        timing: EngineTiming,
    },
    /// Generation failed after the stream started.
    Error(String),
}

// ============================================================================
// 2. Embeddings (/v1/embeddings)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            Self::Single(s) => vec![s],
            Self::Batch(v) => v,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: EmbeddingInput,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// 3. Speech-to-Text (/v1/audio/transcriptions)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptionRequest {
    pub model: String,
    /// Raw audio bytes (populated from multipart upload or base64 JSON payload)
    #[serde(default)]
    pub audio_bytes: Vec<u8>,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioTranscriptionResponse {
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration_secs: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

// ============================================================================
// 4. Text-to-Speech (/v1/audio/speech)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioSpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(default = "default_voice")]
    pub voice: String,
    #[serde(default = "default_speech_format")]
    pub response_format: String,
    #[serde(default = "default_speed")]
    pub speed: f32,
    /// Hypnos extension: stream audio as it is synthesized (raw PCM, or a WAV whose header declares
    /// an open-ended length) instead of returning the whole file at the end.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// A piece of synthesized audio: mono f32 samples in `[-1, 1]`.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
}

/// Incremental events emitted by streaming speech synthesis.
#[derive(Debug, Clone)]
pub enum AudioStreamEvent {
    Chunk(AudioChunk),
    /// Synthesis finished; carries the engine timings (time to first audio, real-time factor).
    Done {
        timing: EngineTiming,
    },
    /// Synthesis failed after the stream started.
    Error(String),
}

fn default_voice() -> String {
    "alloy".to_string()
}
fn default_speech_format() -> String {
    "wav".to_string()
}
fn default_speed() -> f32 {
    1.0
}

#[derive(Debug, Clone)]
pub struct AudioSpeechOutput {
    pub audio_bytes: Vec<u8>,
    pub content_type: String,
    pub telemetry: ExecutionTelemetry,
}

// ============================================================================
// 5. Image Generation (/v1/images/generations)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub negative_prompt: Option<String>,
    #[serde(default = "default_image_count")]
    pub n: u32,
    #[serde(default = "default_image_size")]
    pub size: String,
    #[serde(default = "default_response_format")]
    pub response_format: String,
    #[serde(default)]
    pub steps: Option<u32>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub cfg_scale: Option<f32>,
}

fn default_image_count() -> u32 {
    1
}
fn default_image_size() -> String {
    "1024x1024".to_string()
}
fn default_response_format() -> String {
    "b64_json".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerationResponse {
    pub created: i64,
    pub data: Vec<GeneratedImage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedImage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_json: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub revised_prompt: Option<String>,
}

// ============================================================================
// 6. Telemetry & Model Status (/v1/models, /v1/memory/status)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionTelemetry {
    /// Tier the model was in BEFORE this request arrived
    pub previous_tier: MemoryTier,
    /// Active layer split across GPU VRAM and Pinned Host RAM
    pub placement: LayerPlacementPlan,
    /// Time spent promoting/switching weights into GPU VRAM in milliseconds
    pub tier_switch_latency_ms: f64,
    /// Time spent running inference in milliseconds
    pub inference_latency_ms: f64,
    /// Models evicted/compacted to make room for this request
    pub evicted_models: Vec<String>,
    /// Engine-side timings for this request (prefill/decode split, time to first token, real-time
    /// factor for audio). Filled by the backend; absent fields were not measured.
    #[serde(default)]
    pub engine: EngineTiming,
    /// Where the tier switch spent its time when this request loaded the model into Tier 1
    /// (llama.cpp models); absent on the fast path and for the other engines.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tier_switch_phases: Option<LoadPhases>,
}

/// Phases of a llama.cpp load into Tier 1 (`hypnos_telemetry.tier_switch_phases`), milliseconds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LoadPhases {
    /// `mmap` (the file is mapped and copied tensor by tensor) or `read` (read into pinned
    /// staging buffers and uploaded asynchronously); see `models.use_mmap`.
    pub load_mode: String,
    /// Weights read and placed (GPU upload included).
    pub weights_ms: f64,
    /// The multimodal projector (`aux_weights_path`), when there is one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub projector_ms: Option<f64>,
    /// Context creation: KV cache for `slots x allocated_context_len` and compute buffers.
    pub context_ms: f64,
    /// Everything the worker did from start to ready.
    pub total_ms: f64,
    /// Restoring the slots parked when the model last left Tier 1 (`hardware.kv_park`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_restore_ms: Option<f64>,
    /// Positions restored that way (a conversation that will not be prefilled again).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kv_restored_tokens: Option<u64>,
}

/// What happened between the lease being granted and the response being produced. Every field is
/// optional because each modality measures a different subset. Durations are milliseconds.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EngineTiming {
    /// Waiting for the model's execution lock (today) or worker slot (with the engine worker).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub queue_ms: Option<f64>,
    /// Creating the llama.cpp context (KV cache + compute buffers); 0 once contexts persist.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_ms: Option<f64>,
    /// Prompt (and media) evaluation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_ms: Option<f64>,
    /// Token-by-token generation after prefill.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_ms: Option<f64>,
    /// Lease granted to the first visible text piece (includes queueing, context and prefill).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefill_tokens_per_s: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decode_tokens_per_s: Option<f64>,
    /// Duration of the audio consumed (speech-to-text) or produced (text-to-speech), in seconds.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub audio_secs: Option<f64>,
    /// Real-time factor: processing seconds / audio seconds (lower is faster).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rtf: Option<f64>,
    /// Text-to-speech: time to the first audio chunk (streaming synthesis only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttfa_ms: Option<f64>,
    /// Chat: prompt positions served from the slot's KV cache (prefix reuse); 0 on a miss.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cached_tokens: Option<u32>,
    /// Chat: the most execution slots that were active at once while this request ran
    /// (1 means it never shared a decode step).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub concurrent_slots: Option<u32>,
    /// Document conversion: pages converted.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pages: Option<u32>,
    /// Document conversion: mean conversion time per page (models already loaded excluded).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms_per_page: Option<f64>,
}

impl ExecutionTelemetry {
    /// Telemetry carrying only engine timings; the scheduler fields are placeholders until the
    /// server merges in the lease (`with_engine`).
    pub fn for_engine(engine: EngineTiming) -> Self {
        Self {
            previous_tier: MemoryTier::Tier3NvmeSsd,
            placement: LayerPlacementPlan {
                primary_tier: MemoryTier::Tier3NvmeSsd,
                gpu_layers: 0,
                pinned_ram_layers: 0,
                ssd_layers: 0,
                vram_weight_bytes: 0,
                vram_kv_and_workspace_bytes: 0,
                pinned_ram_bytes: 0,
                allocated_context_len: 0,
                cpu_expert_layers: 0,
            },
            tier_switch_latency_ms: 0.0,
            inference_latency_ms: 0.0,
            evicted_models: Vec::new(),
            engine,
            tier_switch_phases: None,
        }
    }

    pub fn with_engine(mut self, engine: EngineTiming) -> Self {
        self.engine = engine;
        self
    }
}

// ============================================================================
// Document conversion (Hypnos extension: POST /v1/documents/convert)
// ============================================================================

/// Converts a document (PDF, image, or an office / web format docling reads) to Markdown. JSON
/// bodies carry the file as base64 (or a `data:` URI) in `file`, or a `url`; multipart bodies send
/// the same fields as form parts with the document in `file`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentConversionRequest {
    pub model: String,
    /// The document, base64-encoded or as a `data:` URI (JSON bodies only).
    #[serde(default, skip_serializing)]
    pub file: Option<String>,
    /// The document's bytes, filled by the server from `file`, `url` or the multipart part.
    #[serde(skip)]
    pub file_bytes: Vec<u8>,
    /// Fetch the document from this http(s) URL (needs `limits.allow_remote_image_urls`).
    #[serde(default)]
    pub url: Option<String>,
    /// Original file name; its extension helps detect the format.
    #[serde(default)]
    pub filename: Option<String>,
    /// 1-based page or inclusive range, e.g. `"3"` or `"2-5"`; every page when absent.
    #[serde(default)]
    pub pages: Option<String>,
    /// Read pages that have no text layer (scans, images) with OCR. Default true; turning it off
    /// rebuilds the pipeline without the OCR models.
    #[serde(default = "default_true")]
    pub ocr: bool,
    /// `markdown` (default) or `json`: `json` adds each page's docling document as `json`.
    #[serde(default = "default_document_format")]
    pub output_format: String,
    /// Send each page as a server-sent event as soon as it is converted.
    #[serde(default)]
    pub stream: bool,
    /// A vision model (e.g. `gemma4:e4b-q4`) that describes every picture; the description
    /// replaces the picture's `<!-- image -->` placeholder in the Markdown.
    #[serde(default)]
    pub describe_pictures: Option<String>,
    /// Keep the pictures docling cut out of the pages in the server's picture store, so they can
    /// be fetched afterwards from `GET /v1/documents/{id}/pictures[/{n}]`. The response then
    /// carries the document `id`, each page its `picture_urls`.
    #[serde(default)]
    pub return_pictures: bool,
}

fn default_true() -> bool {
    true
}
fn default_document_format() -> String {
    "markdown".to_string()
}

impl DocumentConversionRequest {
    pub fn wants_json(&self) -> bool {
        self.output_format.eq_ignore_ascii_case("json")
    }
}

/// One converted page (or the whole document, for formats without pages).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentPage {
    /// 1-based page number; 1 for formats without pages.
    pub page: u32,
    pub markdown: String,
    /// The page's docling document (`output_format = "json"` only), picture images removed.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json: Option<serde_json::Value>,
    pub pictures: u32,
    pub tables: u32,
    /// Conversion time of this page in milliseconds.
    pub ms: f64,
    /// Where each of the page's pictures can be fetched (`return_pictures`), in Markdown order:
    /// `/v1/documents/{id}/pictures/{n}`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub picture_urls: Vec<String>,
    /// The page's pictures with their image data, in Markdown order (`describe_pictures`,
    /// `return_pictures`). Never serialized: the store and the vision model consume them.
    #[serde(skip)]
    pub picture_images: Vec<ExtractedPicture>,
}

/// A picture docling cut out of a page, with where it was and how it was rendered.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExtractedPicture {
    /// `data:<mime>;base64,...` as docling exports it.
    #[serde(skip)]
    pub data_uri: String,
    /// Image media type (docling renders pictures as `image/png`).
    pub mime: String,
    /// Pixel size of the rendered image.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
    /// Where the picture sits on its page, in the page's units (PDF points for PDFs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<PictureBox>,
    /// The page's size in the same units as `bbox`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_size: Option<PageSize>,
    /// The caption docling attached to the picture, when it found one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// The picture's reference inside the docling document (`#/pictures/0`).
    #[serde(rename = "ref", skip_serializing_if = "Option::is_none")]
    pub self_ref: Option<String>,
}

/// A docling bounding box: `l`, `t`, `r`, `b` with the origin `coord_origin` (`TOPLEFT` or
/// `BOTTOMLEFT`) says which way `t` and `b` count.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PictureBox {
    pub l: f64,
    pub t: f64,
    pub r: f64,
    pub b: f64,
    pub coord_origin: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DocumentUsage {
    pub pages: u32,
    pub pictures: u32,
    pub tables: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pictures_described: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentConversionResponse {
    /// Always `"document"`.
    pub object: String,
    /// The document's id in the picture store (`return_pictures`): list its pictures at
    /// `GET /v1/documents/{id}/pictures`, release them with `DELETE /v1/documents/{id}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// When the stored pictures are dropped (RFC 3339), unless deleted earlier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pictures_expire_at: Option<String>,
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filename: Option<String>,
    /// Pages in the source document (not only the converted ones).
    pub page_count: u32,
    pub pages: Vec<DocumentPage>,
    /// The converted pages joined into one document, with tables, numbered lists and paragraphs
    /// that run across a page break put back together.
    pub markdown: String,
    pub usage: DocumentUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

/// Incremental events emitted by a streaming document conversion.
#[derive(Debug, Clone)]
pub enum DocumentStreamEvent {
    /// The document was read: how many pages it has and which ones will be converted.
    Started {
        page_count: u32,
        pages: Vec<u32>,
    },
    Page(DocumentPage),
    /// Every requested page is converted.
    Done {
        timing: EngineTiming,
    },
    Error(String),
}

// ============================================================================
// Entity extraction (Hypnos extension: POST /v1/entities)
// ============================================================================

/// Finds spans of the given entity types in one text or a list of texts (zero-shot: the labels are
/// free text, e.g. `"person"`, `"medication"`, `"programming language"`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionRequest {
    pub model: String,
    pub input: EmbeddingInput,
    pub labels: Vec<String>,
    /// Minimum score for a span (default 0.5).
    #[serde(default)]
    pub threshold: Option<f32>,
    /// `true` keeps only non-overlapping spans (the highest-scoring wins); `false` also returns
    /// nested ones. Unset uses what the model was trained for.
    #[serde(default)]
    pub flat: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub text: String,
    pub label: String,
    /// Character (Unicode scalar) offsets `[start, end)` in the input text.
    pub start: usize,
    pub end: usize,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityResult {
    /// Always `"entities"`.
    pub object: String,
    pub index: usize,
    pub entities: Vec<Entity>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntityUsage {
    pub texts: u32,
    pub characters: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityExtractionResponse {
    /// Always `"list"`.
    pub object: String,
    pub model: String,
    pub data: Vec<EntityResult>,
    pub usage: EntityUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hypnos_telemetry: Option<ExecutionTelemetry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiModelList {
    pub object: String,
    pub data: Vec<OpenAiModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiModelInfo {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub modality: Modality,
    pub current_tier: MemoryTier,
    pub placement: LayerPlacementPlan,
    pub active_leases: usize,
}

#[cfg(test)]
mod reasoning_tests {
    use super::{ChatCompletionRequest, ReasoningEffort};

    fn req(json: &str) -> ChatCompletionRequest {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn reasoning_effort_takes_precedence_over_think() {
        let base = r#""model":"m","messages":[]"#;
        assert_eq!(req(&format!("{{{base}}}")).thinking(), (false, None));
        assert_eq!(
            req(&format!(r#"{{{base},"think":true}}"#)).thinking(),
            (true, None)
        );
        assert_eq!(
            req(&format!(
                r#"{{{base},"think":true,"reasoning_effort":"none"}}"#
            ))
            .thinking(),
            (false, None)
        );
        assert_eq!(
            req(&format!(r#"{{{base},"reasoning_effort":"minimal"}}"#)).reasoning_effort,
            Some(ReasoningEffort::Low)
        );
        assert_eq!(
            req(&format!(r#"{{{base},"reasoning_effort":"medium"}}"#)).thinking(),
            (true, Some(2048))
        );
        assert!(serde_json::from_str::<ChatCompletionRequest>(&format!(
            r#"{{{base},"reasoning_effort":"max"}}"#
        ))
        .is_err());
    }
}
