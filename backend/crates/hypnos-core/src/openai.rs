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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: ChatMessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// Optional Hypnos extension: enable the model's thinking channel (Gemma 4). Default off.
    #[serde(default)]
    pub think: Option<bool>,
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
    pub content: String,
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
            },
            tier_switch_latency_ms: 0.0,
            inference_latency_ms: 0.0,
            evicted_models: Vec::new(),
            engine,
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
    /// PNG data URIs of the page's pictures, in Markdown order (for `describe_pictures`).
    #[serde(skip)]
    pub picture_images: Vec<String>,
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
