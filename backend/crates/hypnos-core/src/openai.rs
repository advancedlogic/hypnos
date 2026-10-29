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
}

/// Incremental events emitted by a streaming chat completion.
#[derive(Debug, Clone)]
pub enum ChatStreamEvent {
    /// A new piece of generated text.
    Delta(String),
    /// Generation finished; carries final usage and finish reason.
    Done {
        usage: TokenUsage,
        finish_reason: String,
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
