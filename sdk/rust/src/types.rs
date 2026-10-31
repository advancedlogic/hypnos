//! Request and response shapes of the Hypnos HTTP API (`docs/api.md`).
//!
//! The server only ever adds fields (`docs/api.md` section 17, "Versioning"): every response
//! ignores what it does not know, and the top-level responses keep it in `extra`.

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::error::{Error, Result};

/// Unknown fields of a response, kept as JSON.
pub type Extra = Map<String, Value>;

/// `tier1_gpu_vram`, `tier2_pinned_ram` or `tier3_nvme_ssd`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    Tier1GpuVram,
    Tier2PinnedRam,
    Tier3NvmeSsd,
}

impl MemoryTier {
    /// The wire spelling (`tier2_pinned_ram`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Tier1GpuVram => "tier1_gpu_vram",
            Self::Tier2PinnedRam => "tier2_pinned_ram",
            Self::Tier3NvmeSsd => "tier3_nvme_ssd",
        }
    }
}

// --- shared ---------------------------------------------------------------------------------

/// Where a model lives (`LayerPlacementPlan`, appendix of `docs/api.md`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Placement {
    pub primary_tier: MemoryTier,
    #[serde(default)]
    pub gpu_layers: u32,
    #[serde(default)]
    pub pinned_ram_layers: u32,
    #[serde(default)]
    pub ssd_layers: u32,
    #[serde(default)]
    pub vram_weight_bytes: u64,
    #[serde(default)]
    pub vram_kv_and_workspace_bytes: u64,
    #[serde(default)]
    pub pinned_ram_bytes: u64,
    #[serde(default)]
    pub allocated_context_len: u32,
}

/// Engine timings; each field is present only when the modality measures it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineTiming {
    pub queue_ms: Option<f64>,
    pub context_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub ttft_ms: Option<f64>,
    pub prefill_tokens_per_s: Option<f64>,
    pub decode_tokens_per_s: Option<f64>,
    pub cached_tokens: Option<u32>,
    pub concurrent_slots: Option<u32>,
    pub audio_secs: Option<f64>,
    pub rtf: Option<f64>,
    pub ttfa_ms: Option<f64>,
    pub pages: Option<u32>,
    pub ms_per_page: Option<f64>,
}

/// Phases of a llama.cpp load into Tier 1.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LoadPhases {
    pub load_mode: Option<String>,
    pub weights_ms: Option<f64>,
    pub projector_ms: Option<f64>,
    pub context_ms: Option<f64>,
    pub total_ms: Option<f64>,
    /// With KV parking: time spent restoring the parked conversations.
    pub kv_restore_ms: Option<f64>,
    /// With KV parking: the tokens restored from the parked conversations.
    pub kv_restored_tokens: Option<u64>,
}

/// `hypnos_telemetry`: where the model was, what loading it cost, engine timings.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Telemetry {
    pub previous_tier: MemoryTier,
    pub placement: Placement,
    #[serde(default)]
    pub tier_switch_latency_ms: f64,
    #[serde(default)]
    pub inference_latency_ms: f64,
    #[serde(default)]
    pub evicted_models: Vec<String>,
    #[serde(default)]
    pub engine: EngineTiming,
    #[serde(default)]
    pub tier_switch_phases: Option<LoadPhases>,
}

// --- chat -----------------------------------------------------------------------------------

/// A part of a message's `content` array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
    InputAudio { input_audio: InputAudio },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// An image as a `data:` URL or an `http(s)` URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrl {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Base64 audio (`format` defaults to `wav` on the server).
    pub fn input_audio(data: impl Into<String>, format: Option<&str>) -> Self {
        Self::InputAudio {
            input_audio: InputAudio {
                data: data.into(),
                format: format.map(str::to_string),
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InputAudio {
    pub data: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

/// A message's `content`: a string, an array of parts, or `null` (an assistant message that
/// only calls tools, see [`ChatMessage::assistant_with_calls`]).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
    /// Serialized as `null`.
    Null,
}

fn content_null() -> Content {
    Content::Null
}

/// One chat message (`system`, `user`, `assistant` or `tool`).
///
/// `tool_calls` is set on an assistant message that called tools, `tool_call_id` (and optionally
/// `name`) on a `tool` message carrying a result (`docs/api.md` section 7.5). All three are left
/// out of the JSON when `None`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default = "content_null")]
    pub content: Content,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

impl ChatMessage {
    pub fn system(text: impl Into<String>) -> Self {
        Self::new("system", Content::Text(text.into()))
    }
    pub fn user(text: impl Into<String>) -> Self {
        Self::new("user", Content::Text(text.into()))
    }
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::new("assistant", Content::Text(text.into()))
    }
    /// A user message made of parts (text, images, audio).
    pub fn user_parts(parts: Vec<ContentPart>) -> Self {
        Self::new("user", Content::Parts(parts))
    }
    /// An assistant message that called tools, to put back in the history before the `tool`
    /// results. An empty `content` is sent as `null`.
    pub fn assistant_with_calls(content: impl Into<String>, calls: Vec<ToolCall>) -> Self {
        let content: String = content.into();
        let content = if content.is_empty() {
            Content::Null
        } else {
            Content::Text(content)
        };
        Self {
            tool_calls: Some(calls),
            ..Self::new("assistant", content)
        }
    }
    /// A `tool` message: the result of the call `tool_call_id` (send JSON as a string).
    pub fn tool(tool_call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            tool_call_id: Some(tool_call_id.into()),
            ..Self::new("tool", Content::Text(content.into()))
        }
    }
    /// Sets `name` (optional on a `tool` message: the server finds the function from
    /// `tool_call_id`).
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
    pub fn new(role: impl Into<String>, content: Content) -> Self {
        Self {
            role: role.into(),
            content,
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }
}

// --- tools ----------------------------------------------------------------------------------

/// A function the model may call (`docs/api.md` section 7.5).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionSpec {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// A JSON Schema for the arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
}

/// An entry of `tools`: `{"type": "function", "function": {...}}`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolSpec {
    /// Always `function`.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: FunctionSpec,
}

impl ToolSpec {
    /// A function tool; `parameters` is the JSON Schema of its arguments.
    pub fn function(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Value,
    ) -> Self {
        Self {
            kind: "function".into(),
            function: FunctionSpec {
                name: name.into(),
                description: Some(description.into()),
                parameters: Some(parameters),
            },
        }
    }
}

/// `tool_choice`: `"none"`, `"auto"` (the server's default), `"required"` or a named function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoice {
    /// The tools are not shown to the model.
    None,
    /// The model decides whether to call.
    Auto,
    /// The model must call one of the tools.
    Required,
    /// The model must call this function (400 if it is not in `tools`).
    Function(String),
}

impl ToolChoice {
    pub fn function(name: impl Into<String>) -> Self {
        Self::Function(name.into())
    }
}

impl Serialize for ToolChoice {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        match self {
            Self::None => s.serialize_str("none"),
            Self::Auto => s.serialize_str("auto"),
            Self::Required => s.serialize_str("required"),
            Self::Function(name) => serde_json::json!({
                "type": "function",
                "function": { "name": name },
            })
            .serialize(s),
        }
    }
}

/// The function a [`ToolCall`] names; `arguments` is a JSON object serialized as a string.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

impl FunctionCall {
    /// Parses `arguments` into `T`. The model follows the schema as well as it manages:
    /// validate before running the tool.
    pub fn parse_arguments<T: DeserializeOwned>(&self) -> Result<T> {
        serde_json::from_str(&self.arguments).map_err(|e| {
            Error::Decode(format!(
                "tool call arguments of {}: {e}: {}",
                self.name, self.arguments
            ))
        })
    }
}

/// A call the model asked for (`message.tool_calls`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    #[serde(default)]
    pub id: String,
    /// Always `function`.
    #[serde(rename = "type", default = "function_kind")]
    pub kind: String,
    #[serde(default)]
    pub function: FunctionCall,
}

fn function_kind() -> String {
    "function".into()
}

impl ToolCall {
    pub fn new(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            kind: function_kind(),
            function: FunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
    /// The function's name.
    pub fn name(&self) -> &str {
        &self.function.name
    }
    /// [`FunctionCall::parse_arguments`].
    pub fn parse_arguments<T: DeserializeOwned>(&self) -> Result<T> {
        self.function.parse_arguments()
    }
    /// The arguments as a JSON value.
    pub fn arguments_json(&self) -> Result<Value> {
        self.function.parse_arguments()
    }
}

/// A fragment of a function call in a stream chunk.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FunctionCallDelta {
    pub name: Option<String>,
    pub arguments: Option<String>,
}

/// An element of `delta.tool_calls`. Hypnos sends each call whole, but [`crate::ChatStream`]
/// accumulates fragments by `index` as OpenAI clients do.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ToolCallDelta {
    pub index: u32,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    pub function: Option<FunctionCallDelta>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PromptTokensDetails {
    #[serde(default)]
    pub cached_tokens: u32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub completion_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
    #[serde(default)]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ResponseMessage {
    #[serde(default)]
    pub role: String,
    #[serde(default)]
    pub content: String,
    /// The calls the model asked for (`finish_reason` `tool_calls`); absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ResponseMessage {
    /// This message as a history entry: an assistant message with its `tool_calls` (content
    /// `null` when empty and there are calls).
    pub fn to_chat_message(&self) -> ChatMessage {
        match &self.tool_calls {
            Some(calls) if !calls.is_empty() => {
                ChatMessage::assistant_with_calls(self.content.clone(), calls.clone())
            }
            _ => ChatMessage::assistant(self.content.clone()),
        }
    }
}

impl From<&ResponseMessage> for ChatMessage {
    fn from(m: &ResponseMessage) -> Self {
        m.to_chat_message()
    }
}

impl From<ResponseMessage> for ChatMessage {
    fn from(m: ResponseMessage) -> Self {
        m.to_chat_message()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub message: ResponseMessage,
    #[serde(default)]
    pub finish_reason: String,
}

/// `POST /v1/chat/completions`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletion {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default)]
    pub usage: Usage,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

impl ChatCompletion {
    /// The first choice's text.
    pub fn text(&self) -> &str {
        self.choices
            .first()
            .map(|c| c.message.content.as_str())
            .unwrap_or("")
    }

    /// The first choice's tool calls (empty when the model answered in text).
    pub fn tool_calls(&self) -> &[ToolCall] {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_deref())
            .unwrap_or(&[])
    }

    /// The first choice's `finish_reason` (`stop`, `length` or `tool_calls`).
    pub fn finish_reason(&self) -> &str {
        self.choices
            .first()
            .map(|c| c.finish_reason.as_str())
            .unwrap_or("")
    }

    /// The first choice's message as a history entry, tool calls included: append it, then one
    /// [`ChatMessage::tool`] per call, and send the conversation again.
    pub fn to_message(&self) -> ChatMessage {
        self.choices
            .first()
            .map(|c| c.message.to_chat_message())
            .unwrap_or_else(|| ChatMessage::assistant(""))
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChunkChoice {
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub delta: ChunkDelta,
    #[serde(default)]
    pub finish_reason: Option<String>,
}

/// One event of a chat stream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ChatCompletionChunk {
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChunkChoice>,
    #[serde(default)]
    pub usage: Option<Usage>,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(default)]
    pub error: Option<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

// --- embeddings, audio, images ----------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Embedding {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub embedding: Vec<f32>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingUsage {
    #[serde(default)]
    pub prompt_tokens: u32,
    #[serde(default)]
    pub total_tokens: u32,
}

/// `POST /v1/embeddings`; `data` is in input order.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub data: Vec<Embedding>,
    #[serde(default)]
    pub usage: EmbeddingUsage,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `POST /v1/audio/transcriptions`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcription {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub duration_secs: Option<f64>,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GeneratedImage {
    #[serde(default)]
    pub b64_json: String,
    #[serde(default)]
    pub revised_prompt: Option<String>,
}

/// `POST /v1/images/generations`: base64 PNGs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub data: Vec<GeneratedImage>,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

// --- documents ------------------------------------------------------------------------------

/// One converted page.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentPage {
    #[serde(default)]
    pub page: u32,
    #[serde(default)]
    pub markdown: String,
    #[serde(default)]
    pub pictures: u32,
    #[serde(default)]
    pub tables: u32,
    #[serde(default)]
    pub ms: f64,
    /// The page's docling document (`output_format = "json"` only).
    #[serde(default)]
    pub json: Option<Value>,
    #[serde(default)]
    pub picture_urls: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentUsage {
    #[serde(default)]
    pub pages: u32,
    #[serde(default)]
    pub pictures: u32,
    #[serde(default)]
    pub tables: u32,
    #[serde(default)]
    pub pictures_described: Option<u32>,
}

/// `POST /v1/documents/convert`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Document {
    #[serde(default)]
    pub object: String,
    /// The document's id in the picture store (`return_pictures` only).
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub pictures_expire_at: Option<String>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub filename: Option<String>,
    #[serde(default)]
    pub page_count: u32,
    #[serde(default)]
    pub pages: Vec<DocumentPage>,
    #[serde(default)]
    pub markdown: String,
    #[serde(default)]
    pub usage: DocumentUsage,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// The first event of a document stream.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DocumentStarted {
    #[serde(default)]
    pub page_count: u32,
    #[serde(default)]
    pub pages: Vec<u32>,
    #[serde(default)]
    pub id: Option<String>,
}

/// An event of a document stream, by its `object`.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentEvent {
    /// `document.started`.
    Started(DocumentStarted),
    /// `document.page`.
    Page(DocumentPage),
    /// `document`: the last event, with its pages put back.
    Done(Box<Document>),
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PictureBox {
    pub l: f64,
    pub t: f64,
    pub r: f64,
    pub b: f64,
    pub coord_origin: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PageSize {
    pub width: f64,
    pub height: f64,
}

/// A stored picture of a converted document.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Picture {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub document_id: String,
    #[serde(default)]
    pub index: u32,
    #[serde(default)]
    pub page: u32,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub mime: String,
    #[serde(default)]
    pub bytes: u64,
    #[serde(default)]
    pub width: Option<u32>,
    #[serde(default)]
    pub height: Option<u32>,
    #[serde(default)]
    pub dpi: Option<u32>,
    #[serde(default)]
    pub bbox: Option<PictureBox>,
    #[serde(default)]
    pub page_size: Option<PageSize>,
    #[serde(default)]
    pub caption: Option<String>,
    #[serde(default, rename = "ref")]
    pub reference: Option<String>,
    /// Base64 image data (single-picture route only).
    #[serde(default)]
    pub image: Option<String>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `GET /v1/documents/{id}/pictures`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PictureList {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub document_id: String,
    #[serde(default)]
    pub expires_at: String,
    #[serde(default)]
    pub data: Vec<Picture>,
    #[serde(flatten)]
    pub extra: Extra,
}

// --- entities -------------------------------------------------------------------------------

/// A span. `start` and `end` are Unicode scalar (char) offsets: use [`crate::slice_chars`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Entity {
    #[serde(default)]
    pub text: String,
    #[serde(default)]
    pub label: String,
    #[serde(default)]
    pub start: usize,
    #[serde(default)]
    pub end: usize,
    #[serde(default)]
    pub score: f32,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityResult {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub index: usize,
    #[serde(default)]
    pub entities: Vec<Entity>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityUsage {
    #[serde(default)]
    pub texts: u32,
    #[serde(default)]
    pub characters: u64,
}

/// `POST /v1/entities`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EntityResponse {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub data: Vec<EntityResult>,
    #[serde(default)]
    pub usage: EntityUsage,
    #[serde(default)]
    pub hypnos_telemetry: Option<Telemetry>,
    #[serde(flatten)]
    pub extra: Extra,
}

// --- models, memory, logs -------------------------------------------------------------------

/// An entry of `GET /v1/models`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: i64,
    #[serde(default)]
    pub owned_by: String,
    #[serde(default)]
    pub modality: String,
    pub current_tier: Option<MemoryTier>,
    pub placement: Option<Placement>,
    #[serde(default)]
    pub active_leases: u32,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `GET /v1/models`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelList {
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub data: Vec<ModelInfo>,
}

/// `POST /v1/models/{id}/enable|disable`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelSwitch {
    pub model: String,
    pub enabled: bool,
    pub placement: Placement,
}

/// `POST /v1/memory/tier`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TierMove {
    pub model: String,
    pub placement: Placement,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Eviction {
    pub model: String,
    pub for_model: String,
    pub reason: String,
    pub to_tier: MemoryTier,
    #[serde(default)]
    pub unix_ms: u64,
}

/// One model of `GET /v1/memory/status`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelStatus {
    pub id: String,
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub modality: String,
    #[serde(default)]
    pub modalities: Vec<String>,
    #[serde(default)]
    pub engine: String,
    #[serde(default)]
    pub total_weight_bytes: u64,
    #[serde(default)]
    pub priority: String,
    pub current_tier: MemoryTier,
    pub placement: Placement,
    #[serde(default)]
    pub active_leases: u32,
    #[serde(default)]
    pub idle_seconds: f64,
    #[serde(default)]
    pub total_requests_served: u64,
    #[serde(flatten)]
    pub extra: Extra,
}

/// One model's parked KV state (`kv_park.models`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ParkedModel {
    pub model: String,
    pub slots: u32,
    pub tokens: u64,
    pub bytes: u64,
    pub parked_secs_ago: f64,
}

/// `GET`/`POST /v1/memory/kv-park`, and `kv_park` in `GET /v1/memory/status`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct KvParkStatus {
    pub enabled: bool,
    pub max_bytes: u64,
    pub used_bytes: u64,
    pub models: Vec<ParkedModel>,
}

/// `GET /v1/memory/status`; the fields not typed here are in `extra` (docs/api.md section 14).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct MemoryStatus {
    #[serde(default)]
    pub capabilities: Value,
    #[serde(default)]
    pub tier1_vram_used_bytes: u64,
    #[serde(default)]
    pub tier1_vram_free_bytes: u64,
    #[serde(default)]
    pub tier2_pinned_ram_used_bytes: u64,
    #[serde(default)]
    pub tier2_pinned_ram_free_bytes: u64,
    #[serde(default)]
    pub models: Vec<ModelStatus>,
    #[serde(default)]
    pub recent_evictions: Vec<Eviction>,
    #[serde(default)]
    pub gpu: Option<Value>,
    #[serde(default)]
    pub kv_park: Option<KvParkStatus>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// A request-log record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LogRecord {
    pub id: String,
    pub route: String,
    pub method: String,
    pub status: u16,
    pub key: Option<String>,
    pub started_at: Option<String>,
    pub duration_ms: Option<f64>,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub tokens_in: Option<u64>,
    pub tokens_out: Option<u64>,
    pub cached_tokens: Option<u64>,
    pub tokens_per_s: Option<f64>,
    pub inference_ms: Option<f64>,
    pub tier_switch_ms: Option<f64>,
    pub previous_tier: Option<String>,
    pub error: Option<String>,
    pub level: Option<String>,
    pub truncated: Option<bool>,
    pub input: Option<String>,
    pub output: Option<String>,
    pub request_payload: Option<Value>,
    pub response_payload: Option<Value>,
    #[serde(flatten)]
    pub extra: Extra,
}

/// `GET /v1/logs`, newest first.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LogList {
    #[serde(default)]
    pub level: String,
    #[serde(default)]
    pub capacity: u32,
    #[serde(default)]
    pub stored: u32,
    #[serde(default)]
    pub records: Vec<LogRecord>,
}

/// `GET /health`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Health {
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub service: String,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub uptime_secs: u64,
}

/// The `x-hypnos-*` headers of a speech response (docs/api.md section 5).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SpeechTelemetry {
    pub previous_tier: Option<String>,
    pub gpu_layers: Option<u32>,
    pub switch_ms: Option<f64>,
    pub inference_ms: Option<f64>,
    pub audio_secs: Option<f64>,
    pub rtf: Option<f64>,
}

impl SpeechTelemetry {
    pub(crate) fn from_headers(h: &reqwest::header::HeaderMap) -> Self {
        fn get<T: std::str::FromStr>(h: &reqwest::header::HeaderMap, name: &str) -> Option<T> {
            h.get(name)?.to_str().ok()?.parse().ok()
        }
        Self {
            previous_tier: get(h, "x-hypnos-previous-tier"),
            gpu_layers: get(h, "x-hypnos-gpu-layers"),
            switch_ms: get(h, "x-hypnos-switch-ms"),
            inference_ms: get(h, "x-hypnos-inference-ms"),
            audio_secs: get(h, "x-hypnos-audio-secs"),
            rtf: get(h, "x-hypnos-rtf"),
        }
    }
}

/// `POST /v1/audio/speech`: the audio, its media type and the telemetry headers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Speech {
    pub audio: bytes::Bytes,
    /// `audio/wav` or `audio/pcm`; unsupported formats fall back to WAV, so check it.
    pub content_type: String,
    pub telemetry: SpeechTelemetry,
    pub request_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_wire_forms() {
        assert_eq!(
            serde_json::to_value(ToolChoice::None).unwrap(),
            json!("none")
        );
        assert_eq!(
            serde_json::to_value(ToolChoice::Auto).unwrap(),
            json!("auto")
        );
        assert_eq!(
            serde_json::to_value(ToolChoice::Required).unwrap(),
            json!("required")
        );
        assert_eq!(
            serde_json::to_value(ToolChoice::function("x")).unwrap(),
            json!({"type": "function", "function": {"name": "x"}})
        );
        let call = ToolCall::new("call_1", "get_weather", r#"{"city":"Paris"}"#);
        assert_eq!(
            serde_json::to_value(ChatMessage::assistant_with_calls("", vec![call.clone()]))
                .unwrap(),
            json!({"role": "assistant", "content": null, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]})
        );
        assert_eq!(
            serde_json::to_value(ChatMessage::tool("call_1", "18C")).unwrap(),
            json!({"role": "tool", "content": "18C", "tool_call_id": "call_1"})
        );
        assert_eq!(
            serde_json::to_value(ChatMessage::user("hi")).unwrap(),
            json!({"role": "user", "content": "hi"})
        );
        let back: ChatMessage = serde_json::from_value(
            json!({"role": "assistant", "content": null, "tool_calls": [call]}),
        )
        .unwrap();
        assert_eq!(back.content, Content::Null);
        assert_eq!(
            back.tool_calls.unwrap()[0].arguments_json().unwrap(),
            json!({"city": "Paris"})
        );
    }
}
