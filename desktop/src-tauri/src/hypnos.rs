//! Every call to the Hypnos server goes through here, on the Rust SDK (`hypnos-client`). The
//! webview never talks HTTP: it calls these commands, and chat tokens come back over a
//! [`Channel`]. The connection (URL + the key from the keychain) lives in [`Hypnos`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use hypnos_client::{
    ChatMessage, ChatRequest, ChatStream, Client, Content, ContentPart, Error, KvParkStatus,
    LogList, LogQuery, LogRecord, MemoryStatus, MemoryTier, ModelStatus, MoeOffload,
    MoeOffloadSwitch, ReasoningEffort, Telemetry, ToolChoice, Usage,
};

use crate::attachments::Attachment;
use crate::images::{
    generate, image_tool_spec, Generated, ToolCall as ImageCall, ToolGate, IMAGE_TOOL,
    TOOL_INSTRUCTION,
};
use serde::{Deserialize, Serialize};
use tauri::async_runtime::JoinHandle;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, Runtime, State};
use tokio::sync::{Mutex, RwLock};

use futures::StreamExt;

#[derive(Default)]
pub struct Hypnos {
    client: RwLock<Option<Arc<Client>>>,
    /// Modality -> model id chosen in the settings (absent = automatic).
    routes: RwLock<HashMap<String, String>>,
    /// Request id -> the task streaming that answer, so it can be cancelled.
    streams: Mutex<HashMap<String, JoinHandle<()>>>,
}

impl Hypnos {
    pub(crate) async fn client(&self) -> Result<Arc<Client>, ErrorPayload> {
        self.client
            .read()
            .await
            .clone()
            .ok_or_else(|| ErrorPayload {
                kind: "not_connected".into(),
                message: "Not connected to a Hypnos server yet.".into(),
                detail: None,
                retry_after_secs: None,
            })
    }

    /// The model for a job ([`route`]), or an error saying that no enabled model does it.
    pub(crate) async fn pick(
        &self,
        modality: &str,
        preferred: Option<&str>,
    ) -> Result<String, ErrorPayload> {
        let models = self.client().await?.memory_status().await?.models;
        let routes = self.routes.read().await;
        route(
            &models,
            modality,
            preferred,
            routes.get(modality).map(String::as_str),
        )
        .map(str::to_string)
        .ok_or_else(|| no_model_for(modality))
    }
}

fn no_model_for(modality: &str) -> ErrorPayload {
    let job = match modality {
        "vision" => "can see images",
        "speech_to_text" => "transcribes speech",
        "text_to_speech" => "speaks text",
        "image_generation" => "generates images",
        "document_conversion" => "converts documents",
        other => other,
    };
    ErrorPayload {
        kind: "no_model".into(),
        message: format!("No enabled model on this server {job}."),
        detail: None,
        retry_after_secs: None,
    }
}

/// An SDK error as the UI shows it: a `kind` to branch on and a sentence for the user.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorPayload {
    pub kind: String,
    pub message: String,
    /// The raw error (status, type, request id), for the console.
    pub detail: Option<String>,
    pub retry_after_secs: Option<u64>,
}

impl From<Error> for ErrorPayload {
    fn from(e: Error) -> Self {
        let retry = e.retry_after().map(|d| d.as_secs());
        let after = retry
            .map(|s| format!(" Try again in {s} s."))
            .unwrap_or_default();
        let (kind, message) = match &e {
            Error::ModelBusy(_) => (
                "busy",
                if after.is_empty() {
                    "The model's queue is full. Try again in a moment.".to_string()
                } else {
                    format!("The model's queue is full.{after}")
                },
            ),
            Error::RateLimit(_) => ("rate_limit", format!("This API key hit its limit.{after}")),
            Error::ServiceUnavailable(_) => (
                "starting",
                "Hypnos is still loading its models. Try again when the status turns Ready."
                    .into(),
            ),
            Error::InsufficientMemory(_) => (
                "memory",
                "Not enough memory to load this model right now (other models are in use).".into(),
            ),
            Error::NotFound(_) => (
                "not_found",
                "This model is not available (disabled or removed from the config). Pick another one."
                    .into(),
            ),
            Error::Authentication(_) => ("unauthorized", "The server refused the API key.".into()),
            Error::Connection(_) => ("connection", "Lost the connection to Hypnos.".into()),
            Error::Timeout(_) => ("timeout", "Hypnos did not answer in time.".into()),
            Error::Stream(api) => (
                "stream",
                format!("The model stopped with an error: {}", api.message),
            ),
            Error::BadRequest(api) => ("bad_request", format!("The server refused the request: {}", api.message)),
            other => ("error", other.to_string()),
        };
        ErrorPayload {
            kind: kind.into(),
            message,
            detail: Some(e.to_string()),
            retry_after_secs: retry,
        }
    }
}

type CmdResult<T> = Result<T, ErrorPayload>;

fn build_client(url: &str, api_key: Option<String>) -> CmdResult<Client> {
    let mut builder = Client::builder().base_url(url);
    builder = match api_key.filter(|k| !k.is_empty()) {
        Some(key) => builder.api_key(key),
        None => builder.no_api_key(),
    };
    builder.build().map_err(ErrorPayload::from)
}

/// Uses this server from now on, with the key saved for it in the keychain.
#[tauri::command]
pub async fn hypnos_connect(hypnos: State<'_, Hypnos>, url: String) -> CmdResult<()> {
    let key = crate::get_api_key(url.clone()).unwrap_or_else(|e| {
        eprintln!("could not read the API key from the keychain: {e}");
        None
    });
    let client = build_client(&url, key)?;
    *hypnos.client.write().await = Some(Arc::new(client));
    Ok(())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Probe {
    /// `unreachable`, `unauthorized`, `starting` or `ready`.
    pub state: &'static str,
    pub health: Option<hypnos_client::Health>,
}

/// `/health` is open to everyone and `GET /v1/models` checks the key even during startup, so
/// the four states are told apart.
async fn probe_with(client: &Client) -> Probe {
    let Ok(health) = client.health().await else {
        return Probe {
            state: "unreachable",
            health: None,
        };
    };
    if let Err(Error::Authentication(_)) = client.models().await {
        return Probe {
            state: "unauthorized",
            health: Some(health),
        };
    }
    let state = if client.is_ready().await {
        "ready"
    } else {
        "starting"
    };
    Probe {
        state,
        health: Some(health),
    }
}

#[tauri::command]
pub async fn hypnos_probe(hypnos: State<'_, Hypnos>) -> CmdResult<Probe> {
    let client = hypnos.client().await?;
    Ok(probe_with(&client).await)
}

/// Checks a server and key without switching to them (the settings' "Test connection").
#[tauri::command]
pub async fn hypnos_test(url: String, api_key: Option<String>) -> CmdResult<Probe> {
    let client = Client::builder()
        .base_url(&url)
        .timeout(Duration::from_secs(10));
    let client = match api_key.filter(|k| !k.is_empty()) {
        Some(key) => client.api_key(key),
        None => client.no_api_key(),
    }
    .build()?;
    Ok(probe_with(&client).await)
}

#[tauri::command]
pub async fn hypnos_memory_status(hypnos: State<'_, Hypnos>) -> CmdResult<MemoryStatus> {
    Ok(hypnos.client().await?.memory_status().await?)
}

/// The models in service with every modality they serve and their tier, from
/// `/v1/memory/status` (`/v1/models` lists only the primary modality).
#[tauri::command]
pub async fn hypnos_models(hypnos: State<'_, Hypnos>) -> CmdResult<Vec<ModelStatus>> {
    let status = hypnos.client().await?.memory_status().await?;
    Ok(status.models.into_iter().filter(|m| m.enabled).collect())
}

/// Sets the per-modality choices of the settings (modality -> model id).
#[tauri::command]
pub async fn hypnos_set_routes(
    hypnos: State<'_, Hypnos>,
    routes: HashMap<String, String>,
) -> CmdResult<()> {
    *hypnos.routes.write().await = routes;
    Ok(())
}

/// The model that would serve `modality` now (see [`route`]), or none.
#[tauri::command]
pub async fn hypnos_route(
    hypnos: State<'_, Hypnos>,
    modality: String,
    preferred: Option<String>,
) -> CmdResult<Option<String>> {
    let models = hypnos.client().await?.memory_status().await?.models;
    let routes = hypnos.routes.read().await;
    Ok(route(
        &models,
        &modality,
        preferred.as_deref(),
        routes.get(&modality).map(String::as_str),
    )
    .map(str::to_string))
}

fn serves(m: &ModelStatus, modality: &str) -> bool {
    m.enabled
        && (m.modalities.iter().any(|x| x == modality)
            || (m.modalities.is_empty() && m.modality == modality))
}

/// Picks the model for a job: the preferred one (the chat's model) when it serves the
/// modality, else the one chosen in the settings, else an enabled model that serves it,
/// preferring one already in Tier 1 so no swap is needed. Never a hard-coded id.
pub fn route<'a>(
    models: &'a [ModelStatus],
    modality: &str,
    preferred: Option<&str>,
    chosen: Option<&str>,
) -> Option<&'a str> {
    let find = |id: &str| models.iter().find(|m| m.id == id && serves(m, modality));
    preferred
        .and_then(find)
        .or_else(|| chosen.and_then(find))
        .or_else(|| {
            let mut candidates: Vec<&ModelStatus> =
                models.iter().filter(|m| serves(m, modality)).collect();
            candidates.sort_by_key(|m| m.current_tier != MemoryTier::Tier1GpuVram);
            candidates.into_iter().next()
        })
        .map(|m| m.id.as_str())
}

/// What the webview sends for a message.
#[derive(Debug, Clone, Deserialize)]
pub struct InMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub attachments: Vec<Attachment>,
    /// Set on an answer that was a generated image (DESK-11): the prompt it was made from.
    #[serde(default)]
    pub image_prompt: Option<String>,
}

impl InMessage {
    fn has_images(&self) -> bool {
        self.attachments.iter().any(|a| match a {
            Attachment::Image { .. } => true,
            Attachment::Video { frames, .. } => !frames.is_empty(),
            _ => false,
        })
    }

    /// The message as the API takes it: documents and transcripts as tagged text after the
    /// user's words, images (and video frames) as `image_url` parts when the model can see them.
    fn to_chat_message(&self, vision: bool) -> ChatMessage {
        let mut text = self.content.clone();
        if let Some(prompt) = &self.image_prompt {
            text = format!("[An image was generated from the prompt: \"{prompt}\"]\n\n{text}");
        }
        let mut images = Vec::new();
        for a in &self.attachments {
            match a {
                Attachment::Document {
                    name,
                    text: body,
                    truncated,
                    ..
                } => {
                    let cut = if *truncated {
                        " truncated=\"true\""
                    } else {
                        ""
                    };
                    text.push_str(&format!(
                        "\n\n<document name=\"{name}\"{cut}>\n{body}\n</document>"
                    ));
                }
                Attachment::Web {
                    name,
                    url,
                    text: body,
                    truncated,
                    ..
                } => {
                    let cut = if *truncated {
                        " truncated=\"true\""
                    } else {
                        ""
                    };
                    text.push_str(&format!(
                        "\n\n<web_page url=\"{url}\" title=\"{name}\"{cut}>\n{body}\n</web_page>"
                    ));
                }
                Attachment::Audio {
                    name, transcript, ..
                } => {
                    text.push_str(&format!(
                        "\n\n<audio_transcript name=\"{name}\">\n{transcript}\n</audio_transcript>"
                    ));
                }
                Attachment::Video {
                    name,
                    transcript,
                    frames,
                    ..
                } => {
                    if let Some(t) = transcript {
                        text.push_str(&format!(
                            "\n\n<video_transcript name=\"{name}\">\n{t}\n</video_transcript>"
                        ));
                    }
                    if vision && !frames.is_empty() {
                        text.push_str(&format!(
                            "\n\n[{} frames of the video {name} follow, in order]",
                            frames.len()
                        ));
                        images.extend(frames.iter().cloned());
                    } else if !frames.is_empty() {
                        text.push_str(&format!(
                            "\n\n[video {name}: frames not shown to this model]"
                        ));
                    }
                }
                Attachment::Image { name, data_url, .. } => {
                    if vision {
                        images.push(data_url.clone());
                    } else {
                        text.push_str(&format!("\n\n[image {name}: not shown to this model]"));
                    }
                }
            }
        }
        let content = if images.is_empty() {
            Content::Text(text)
        } else {
            let mut parts = vec![ContentPart::text(text)];
            parts.extend(images.into_iter().map(ContentPart::image_url));
            Content::Parts(parts)
        };
        ChatMessage::new(self.role.clone(), content)
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "event", rename_all = "camelCase")]
pub enum ChatEvent {
    /// Another model answers than the one asked for, and why (e.g. the message has images).
    Model { id: String, reason: String },
    /// A piece of the model's thoughts (thinking on); they all come before the first `Delta`.
    Reasoning { text: String },
    /// A piece of the answer.
    Delta { text: String },
    /// The chat model asked for an image; it is being generated.
    ToolCall { prompt: String, size: String },
    /// The generated image.
    Image(Box<Generated>),
    /// The answer is complete.
    #[serde(rename_all = "camelCase")]
    Done {
        telemetry: Option<Box<Telemetry>>,
        usage: Option<Usage>,
        finish_reason: Option<String>,
        /// The answer's `x-request-id`: its record in the request log (DESK-13).
        request_id: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ChatOutcome {
    pub cancelled: bool,
}

/// Streams an answer over `on_event`. Resolves when the answer is complete, failed, or was
/// cancelled with [`hypnos_chat_cancel`] (dropping the stream closes the connection, which is
/// how the server is told to stop).
///
/// With `tools`, the chat model may ask for an image (see `images.rs`): the request never shows
/// as text, the image is generated and sent as an `image` event, and the model then comments on
/// it in a short second round.
///
/// `reasoning_effort` is the thinking level (absent or `none` = off); the thoughts stream as
/// `reasoning` events before the answer.
#[tauri::command]
#[allow(clippy::too_many_arguments)]
pub async fn hypnos_chat<R: Runtime>(
    app: AppHandle<R>,
    hypnos: State<'_, Hypnos>,
    request_id: String,
    model: String,
    messages: Vec<InMessage>,
    tools: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    on_event: Channel<ChatEvent>,
) -> CmdResult<ChatOutcome> {
    let thinking = Thinking(reasoning_effort.unwrap_or(ReasoningEffort::None));
    let client = hypnos.client().await?;
    let models = client.memory_status().await?.models;
    let can_see = |id: &str| models.iter().any(|m| m.id == id && serves(m, "vision"));
    let mut model = model;
    // Images in the new message need a model that sees; older images are described as text
    // to a model that does not.
    let images_now = messages
        .iter()
        .rev()
        .find(|m| m.role == "user")
        .is_some_and(InMessage::has_images);
    if images_now && !can_see(&model) {
        let routes = hypnos.routes.read().await;
        let vision = route(
            &models,
            "vision",
            None,
            routes.get("vision").map(String::as_str),
        )
        .ok_or_else(|| no_model_for("vision"))?
        .to_string();
        let reason = format!("{model} cannot see images");
        let _ = on_event.send(ChatEvent::Model {
            id: vision.clone(),
            reason,
        });
        model = vision;
    }
    let vision = can_see(&model);
    let messages: Vec<ChatMessage> = messages.iter().map(|m| m.to_chat_message(vision)).collect();
    let image_tool = tools.unwrap_or(false) && models.iter().any(|m| serves(m, "image_generation"));
    let (tx, rx) = tokio::sync::oneshot::channel();
    let task = tauri::async_runtime::spawn(async move {
        let result = if image_tool {
            answer_with_image_tool(&app, &client, &model, messages, thinking, &on_event).await
        } else {
            let request = thinking.apply(ChatRequest::new(model, messages));
            stream_answer(&client, &request, &on_event).await
        };
        let _ = tx.send(result);
    });
    hypnos.streams.lock().await.insert(request_id.clone(), task);
    let result = rx.await;
    hypnos.streams.lock().await.remove(&request_id);
    match result {
        Ok(outcome) => outcome.map(|()| ChatOutcome { cancelled: false }),
        // The task was aborted before it could answer.
        Err(_) => Ok(ChatOutcome { cancelled: true }),
    }
}

/// Room left for the answer when the model thinks: the thoughts count toward `max_tokens`.
const ANSWER_TOKENS: u32 = 2048;

/// The thinking level of a chat. The server budgets 512 / 2048 / 6144 thinking tokens for
/// low / medium / high, so `max_tokens` becomes that budget plus [`ANSWER_TOKENS`] (2560 / 4096
/// / 8192, the last one the server's default `max_tokens_cap`); off, the server's default
/// `max_tokens` applies as before and the field is left out (the server's default is off).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Thinking(ReasoningEffort);

impl Thinking {
    const OFF: Self = Self(ReasoningEffort::None);

    fn budget(self) -> u32 {
        match self.0 {
            ReasoningEffort::None => 0,
            ReasoningEffort::Low => 512,
            ReasoningEffort::Medium => 2048,
            ReasoningEffort::High => 6144,
        }
    }

    fn apply(self, request: ChatRequest) -> ChatRequest {
        match self.budget() {
            0 => request,
            budget => request
                .reasoning_effort(self.0)
                .max_tokens(budget + ANSWER_TOKENS),
        }
    }
}

/// Reads a stream to its end: thoughts go out as `Reasoning` events, answer pieces through
/// `text` (which says what to send as a `Delta`, if anything). False when the window is gone.
async fn relay(
    stream: &mut ChatStream,
    on_event: &Channel<ChatEvent>,
    mut text: impl FnMut(&str) -> Option<String>,
) -> CmdResult<bool> {
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        let mut thought = String::new();
        let mut piece = String::new();
        for choice in &chunk.choices {
            thought.push_str(choice.delta.reasoning_content.as_deref().unwrap_or(""));
            piece.push_str(choice.delta.content.as_deref().unwrap_or(""));
        }
        if !thought.is_empty()
            && on_event
                .send(ChatEvent::Reasoning { text: thought })
                .is_err()
        {
            return Ok(false);
        }
        if let Some(out) = (!piece.is_empty()).then(|| text(&piece)).flatten() {
            if on_event.send(ChatEvent::Delta { text: out }).is_err() {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

/// Streams one answer and ends it with `Done`.
async fn stream_answer(
    client: &Client,
    request: &ChatRequest,
    on_event: &Channel<ChatEvent>,
) -> CmdResult<()> {
    let mut stream = client.chat_stream(request).await?;
    if relay(&mut stream, on_event, |p| Some(p.to_string())).await? {
        send_done(&stream, on_event);
    }
    Ok(())
}

fn send_done(stream: &ChatStream, on_event: &Channel<ChatEvent>) {
    let _ = on_event.send(ChatEvent::Done {
        telemetry: stream.telemetry().cloned().map(Box::new),
        usage: stream.usage().cloned(),
        finish_reason: stream.finish_reason().map(str::to_string),
        request_id: stream.request_id().map(str::to_string),
    });
}

/// The image tool: as a native tool call when the server can run tools for the model (API-9),
/// else through the prompt convention.
async fn answer_with_image_tool<R: Runtime>(
    app: &AppHandle<R>,
    client: &Client,
    model: &str,
    messages: Vec<ChatMessage>,
    thinking: Thinking,
    on_event: &Channel<ChatEvent>,
) -> CmdResult<()> {
    match answer_with_native_tool(app, client, model, messages.clone(), thinking, on_event).await {
        Err(e)
            if e.kind == "bad_request"
                && e.detail.as_deref().is_some_and(|d| d.contains("tool")) =>
        {
            answer_with_prompt_tool(app, client, model, messages, thinking, on_event).await
        }
        other => other,
    }
}

/// `generate_image` as a real tool: the answer streams as usual, a call becomes an image and a
/// `tool` result, and the model comments on it in a second round (tools hidden, so no new call).
///
/// The first round thinks at the chat's level (it decides whether to draw and what); the second
/// one never does ([`Thinking::OFF`]): it only says in one sentence that the image is ready.
async fn answer_with_native_tool<R: Runtime>(
    app: &AppHandle<R>,
    client: &Client,
    model: &str,
    messages: Vec<ChatMessage>,
    thinking: Thinking,
    on_event: &Channel<ChatEvent>,
) -> CmdResult<()> {
    let tools = vec![image_tool_spec()];
    let request = thinking.apply(
        ChatRequest::new(model, messages.clone())
            .tools(tools.clone())
            .tool_choice(ToolChoice::Auto),
    );
    let mut stream = client.chat_stream(&request).await?;
    if !relay(&mut stream, on_event, |p| Some(p.to_string())).await? {
        return Ok(());
    }
    let calls = stream.tool_calls().to_vec();
    let Some(call) = calls.iter().find(|c| c.function.name == IMAGE_TOOL) else {
        send_done(&stream, on_event);
        return Ok(());
    };
    let args: ImageCall = call.function.parse_arguments().map_err(|e| ErrorPayload {
        kind: "image".into(),
        message: "The model asked for an image without a usable prompt.".into(),
        detail: Some(e.to_string()),
        retry_after_secs: None,
    })?;
    let _ = on_event.send(ChatEvent::ToolCall {
        prompt: args.prompt.clone(),
        size: args.size.clone(),
    });
    let image = generate(&app.state::<Hypnos>(), &args.prompt, Some(&args.size)).await?;
    let _ = on_event.send(ChatEvent::Image(Box::new(image)));

    let mut follow_up = messages;
    follow_up.push(ChatMessage::assistant_with_calls("", calls.clone()));
    for c in &calls {
        let result = if c.id == call.id {
            "The image was generated and is now shown to the user. Tell them in one short sentence that it is ready and what it shows; do not claim to see it and do not ask a question."
        } else {
            "Unknown tool."
        };
        follow_up.push(ChatMessage::tool(c.id.clone(), result));
    }
    let request = Thinking::OFF.apply(
        ChatRequest::new(model, follow_up)
            .tools(tools)
            .tool_choice(ToolChoice::None),
    );
    stream_answer(client, &request, on_event).await
}

/// The prompt convention for models without a native tool format: the first round runs with the
/// tool instruction and its start held back by a [`ToolGate`]; a call becomes an image and a
/// second, instruction-free round for the model's comment. Thinking as in
/// [`answer_with_native_tool`]: the first round at the chat's level (the thoughts arrive apart
/// from the text the gate reads), the comment without.
async fn answer_with_prompt_tool<R: Runtime>(
    app: &AppHandle<R>,
    client: &Client,
    model: &str,
    messages: Vec<ChatMessage>,
    thinking: Thinking,
    on_event: &Channel<ChatEvent>,
) -> CmdResult<()> {
    let mut with_tool = messages.clone();
    match with_tool.first_mut() {
        Some(ChatMessage {
            role,
            content: Content::Text(system),
            ..
        }) if role == "system" => {
            system.push_str("\n\n");
            system.push_str(TOOL_INSTRUCTION);
        }
        _ => with_tool.insert(0, ChatMessage::system(TOOL_INSTRUCTION)),
    }
    let mut stream = client
        .chat_stream(&thinking.apply(ChatRequest::new(model, with_tool)))
        .await?;
    let mut gate = ToolGate::default();
    if !relay(&mut stream, on_event, |p| gate.feed(p)).await? {
        return Ok(());
    }
    let call = match gate.finish() {
        Ok(call) => call,
        Err(held) => {
            if !held.is_empty() {
                let _ = on_event.send(ChatEvent::Delta { text: held });
            }
            send_done(&stream, on_event);
            return Ok(());
        }
    };
    let _ = on_event.send(ChatEvent::ToolCall {
        prompt: call.prompt.clone(),
        size: call.size.clone(),
    });
    let image = generate(&app.state::<Hypnos>(), &call.prompt, Some(&call.size)).await?;
    let _ = on_event.send(ChatEvent::Image(Box::new(image)));

    let mut follow_up = messages;
    follow_up.push(ChatMessage::assistant(format!(
        "[I generated an image from the prompt: \"{}\"]",
        call.prompt
    )));
    follow_up.push(ChatMessage::user(
        "[The image is now shown to the user. In one short sentence, tell them it is ready and what it shows; do not claim to see it and do not ask a question.]",
    ));
    let request = Thinking::OFF.apply(ChatRequest::new(model, follow_up));
    stream_answer(client, &request, on_event).await
}

/// KV parking (MEM-4): whether conversations are kept across a model swap, and what is held.
#[tauri::command]
pub async fn hypnos_kv_park(hypnos: State<'_, Hypnos>) -> CmdResult<KvParkStatus> {
    Ok(hypnos.client().await?.kv_park().await?)
}

/// Turns KV parking on or off and / or sets its cap (bytes) on the server.
#[tauri::command]
pub async fn hypnos_set_kv_park(
    hypnos: State<'_, Hypnos>,
    enabled: Option<bool>,
    max_bytes: Option<u64>,
) -> CmdResult<KvParkStatus> {
    Ok(hypnos
        .client()
        .await?
        .set_kv_park(enabled, max_bytes)
        .await?)
}

/// Sets a model's MoE expert offload (`off`, `auto`, `on`; MEM-5).
#[tauri::command]
pub async fn hypnos_set_moe_offload(
    hypnos: State<'_, Hypnos>,
    model: String,
    mode: MoeOffload,
) -> CmdResult<MoeOffloadSwitch> {
    Ok(hypnos.client().await?.set_moe_offload(&model, mode).await?)
}

/// Filters of the request log view; `status` is `ok`, `error`, `4xx` or `5xx`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LogFilter {
    pub model: Option<String>,
    pub route: Option<String>,
    pub status: Option<String>,
    pub limit: Option<u32>,
}

/// `GET /v1/logs`, newest first (DESK-13).
#[tauri::command]
pub async fn hypnos_logs(hypnos: State<'_, Hypnos>, filter: LogFilter) -> CmdResult<LogList> {
    let nonempty = |s: Option<String>| s.filter(|v| !v.trim().is_empty());
    let query = LogQuery {
        model: nonempty(filter.model),
        route: nonempty(filter.route),
        status: nonempty(filter.status),
        key: None,
        limit: filter.limit,
    };
    Ok(hypnos.client().await?.logs(&query).await?)
}

/// `GET /v1/logs/{id}`: one record, with the captured text and payloads at debug / trace.
#[tauri::command]
pub async fn hypnos_log(hypnos: State<'_, Hypnos>, id: String) -> CmdResult<LogRecord> {
    Ok(hypnos.client().await?.log(&id).await?)
}

/// `POST /v1/logs/level`: `off`, `error`, `warn`, `info`, `debug` or `trace`.
#[tauri::command]
pub async fn hypnos_set_log_level(hypnos: State<'_, Hypnos>, level: String) -> CmdResult<()> {
    hypnos.client().await?.set_log_level(&level).await?;
    Ok(())
}

/// `DELETE /v1/logs`.
#[tauri::command]
pub async fn hypnos_clear_logs(hypnos: State<'_, Hypnos>) -> CmdResult<()> {
    hypnos.client().await?.clear_logs().await?;
    Ok(())
}

#[tauri::command]
pub async fn hypnos_chat_cancel(hypnos: State<'_, Hypnos>, request_id: String) -> CmdResult<()> {
    if let Some(task) = hypnos.streams.lock().await.remove(&request_id) {
        task.abort();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn model(id: &str, modalities: &[&str], tier: &str, enabled: bool) -> ModelStatus {
        serde_json::from_value(json!({
            "id": id, "enabled": enabled, "modality": modalities[0], "modalities": modalities,
            "current_tier": tier,
            "placement": { "primary_tier": tier },
        }))
        .unwrap()
    }

    fn lineup() -> Vec<ModelStatus> {
        vec![
            model(
                "big",
                &["text_generation", "vision"],
                "tier2_pinned_ram",
                true,
            ),
            model(
                "small",
                &["speech_to_text", "text_generation", "vision"],
                "tier1_gpu_vram",
                true,
            ),
            model("tts", &["text_to_speech"], "tier1_gpu_vram", true),
            model("off", &["image_generation"], "tier3_nvme_ssd", false),
        ]
    }

    #[test]
    fn the_preferred_model_wins_when_it_serves_the_modality() {
        let m = lineup();
        assert_eq!(route(&m, "vision", Some("big"), None), Some("big"));
        // `big` does not transcribe: the router falls back to a model that does.
        assert_eq!(
            route(&m, "speech_to_text", Some("big"), None),
            Some("small")
        );
    }

    #[test]
    fn the_settings_choice_comes_next_then_a_resident_model() {
        let m = lineup();
        assert_eq!(route(&m, "text_generation", None, Some("big")), Some("big"));
        assert_eq!(route(&m, "text_generation", None, None), Some("small"));
        assert_eq!(
            route(&m, "text_generation", None, Some("tts")),
            Some("small")
        );
    }

    #[test]
    fn disabled_or_missing_models_are_never_picked() {
        let m = lineup();
        assert_eq!(route(&m, "image_generation", None, Some("off")), None);
        assert_eq!(route(&m, "embeddings", None, None), None);
    }

    #[test]
    fn thinking_leaves_room_for_the_answer() {
        let request = |t: Thinking| t.apply(ChatRequest::new("m", vec![]));
        let off = request(Thinking::OFF);
        assert_eq!((off.reasoning_effort, off.max_tokens), (None, None));
        let low = request(Thinking(ReasoningEffort::Low));
        assert_eq!(low.reasoning_effort, Some(ReasoningEffort::Low));
        assert_eq!(low.max_tokens, Some(512 + ANSWER_TOKENS));
        let high = request(Thinking(ReasoningEffort::High));
        assert_eq!(high.max_tokens, Some(8192));
        let wire = serde_json::to_value(request(Thinking(ReasoningEffort::Medium))).unwrap();
        assert_eq!(wire["reasoning_effort"], "medium");
        assert_eq!(wire["max_tokens"], 4096);
    }

    /// The thoughts of a thinking model (Gemma 4, LFM3) stream as `reasoning` events before the
    /// answer: `HYPNOS_TEST_URL=... cargo test thinks_before_answering -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn thinks_before_answering() {
        use std::sync::Mutex as StdMutex;
        use tauri::Manager;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();
        let model = hypnos_route(app.state(), "text_generation".into(), None)
            .await
            .unwrap()
            .expect("no chat model");
        let events = Arc::new(StdMutex::new(Vec::<String>::new()));
        let e = events.clone();
        let channel = Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                let event: serde_json::Value = serde_json::from_str(&s).unwrap();
                e.lock()
                    .unwrap()
                    .push(event["event"].as_str().unwrap().to_string());
            }
            Ok(())
        });
        let message = vec![InMessage {
            role: "user".into(),
            content: "What is 17 * 23? Answer with the number.".into(),
            attachments: vec![],
            image_prompt: None,
        }];
        hypnos_chat(
            app.handle().clone(),
            app.state(),
            "think1".into(),
            model,
            message,
            None,
            Some(ReasoningEffort::Low),
            channel,
        )
        .await
        .unwrap();
        let events = events.lock().unwrap();
        let first_delta = events.iter().position(|e| e == "delta").expect("no answer");
        let last_thought = events
            .iter()
            .rposition(|e| e == "reasoning")
            .expect("no thoughts");
        assert!(last_thought < first_delta, "{events:?}");
        assert_eq!(events.last().map(String::as_str), Some("done"));
    }

    #[test]
    fn errors_read_as_sentences() {
        let e: ErrorPayload = Error::Connection("refused".into()).into();
        assert_eq!(e.kind, "connection");
        assert_eq!(e.message, "Lost the connection to Hypnos.");
    }

    #[test]
    fn attachments_become_text_and_image_parts() {
        let msg = InMessage {
            role: "user".into(),
            content: "Compare these".into(),
            attachments: vec![
                Attachment::Document {
                    name: "a.pdf".into(),
                    size: 1,
                    text: "Alpha".into(),
                    pages: 1,
                    truncated: true,
                },
                Attachment::Image {
                    name: "b.png".into(),
                    size: 1,
                    data_url: "data:image/jpeg;base64,AAAA".into(),
                    width: 1,
                    height: 1,
                },
            ],
            image_prompt: None,
        };
        let seen = serde_json::to_value(msg.to_chat_message(true)).unwrap();
        let text = seen["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("Compare these"));
        assert!(text.contains("<document name=\"a.pdf\" truncated=\"true\">\nAlpha\n</document>"));
        assert_eq!(
            seen["content"][1]["image_url"]["url"],
            "data:image/jpeg;base64,AAAA"
        );

        let blind = serde_json::to_value(msg.to_chat_message(false)).unwrap();
        let text = blind["content"].as_str().unwrap();
        assert!(text.contains("[image b.png: not shown to this model]"));
    }

    /// Against a running server: `HYPNOS_TEST_URL=http://127.0.0.1:9999 cargo test -- --ignored`.
    /// It sends two short chats to the model the router picks for text generation.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn chats_and_cancels_through_the_sdk() {
        use std::sync::Mutex as StdMutex;
        use tauri::Manager;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();

        hypnos_connect(app.state(), url).await.unwrap();
        assert_eq!(hypnos_probe(app.state()).await.unwrap().state, "ready");
        let models = hypnos_models(app.state()).await.unwrap();
        assert!(models.iter().all(|m| m.enabled));
        let model = hypnos_route(app.state(), "text_generation".into(), None)
            .await
            .unwrap()
            .expect("no chat model");

        let collect = || {
            let text = Arc::new(StdMutex::new(String::new()));
            let done = Arc::new(StdMutex::new(false));
            let (t, d) = (text.clone(), done.clone());
            let channel = Channel::new(move |body| {
                let event: serde_json::Value = match body {
                    tauri::ipc::InvokeResponseBody::Json(s) => serde_json::from_str(&s).unwrap(),
                    _ => panic!("expected JSON"),
                };
                match event["event"].as_str() {
                    Some("delta") => t.lock().unwrap().push_str(event["text"].as_str().unwrap()),
                    Some("reasoning") => {}
                    Some("done") => {
                        assert!(event["telemetry"]["engine"].is_object(), "{event}");
                        *d.lock().unwrap() = true;
                    }
                    _ => panic!("unexpected event {event}"),
                }
                Ok(())
            });
            (channel, text, done)
        };

        let (channel, text, done) = collect();
        let message = |s: &str| {
            vec![InMessage {
                role: "user".into(),
                content: s.into(),
                attachments: vec![],
                image_prompt: None,
            }]
        };
        let outcome = hypnos_chat(
            app.handle().clone(),
            app.state(),
            "r1".into(),
            model.clone(),
            message("Reply with exactly: hello from rust"),
            None,
            None,
            channel,
        )
        .await
        .unwrap();
        assert!(!outcome.cancelled);
        assert!(*done.lock().unwrap());
        assert!(text
            .lock()
            .unwrap()
            .to_lowercase()
            .contains("hello from rust"));

        // A long answer, cancelled after a second.
        let (channel, text, done) = collect();
        let state = app.state::<Hypnos>();
        let chat = hypnos_chat(
            app.handle().clone(),
            state.clone(),
            "r2".into(),
            model,
            message("Count from 1 to 500, one number per line."),
            None,
            None,
            channel,
        );
        let cancel = async {
            tokio::time::sleep(Duration::from_millis(1500)).await;
            hypnos_chat_cancel(app.state(), "r2".into()).await.unwrap();
        };
        let (outcome, ()) = tokio::join!(chat, cancel);
        assert!(outcome.unwrap().cancelled);
        assert!(!*done.lock().unwrap());
        assert!(!text.lock().unwrap().is_empty());

        // Errors come back as sentences.
        let (channel, _, _) = collect();
        let err = hypnos_chat(
            app.handle().clone(),
            app.state(),
            "r3".into(),
            "no-such-model".into(),
            message("hi"),
            None,
            None,
            channel,
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, "not_found", "{err:?}");
    }

    /// Reads the request log of a running server without changing it (no level change, no
    /// clear): `HYPNOS_TEST_URL=... cargo test reads_the_request_log -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn reads_the_request_log() {
        use std::sync::Mutex as StdMutex;
        use tauri::Manager;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();
        let model = app
            .state::<Hypnos>()
            .pick("text_generation", None)
            .await
            .unwrap();

        let request_id = std::sync::Arc::new(StdMutex::new(None::<String>));
        let r = request_id.clone();
        let channel = Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                let v: serde_json::Value = serde_json::from_str(&s).unwrap();
                if v["event"] == "done" {
                    *r.lock().unwrap() = v["requestId"].as_str().map(str::to_string);
                }
            }
            Ok(())
        });
        let message = vec![InMessage {
            role: "user".into(),
            content: "Reply with one word: ok".into(),
            attachments: vec![],
            image_prompt: None,
        }];
        hypnos_chat(
            app.handle().clone(),
            app.state(),
            "log1".into(),
            model.clone(),
            message,
            None,
            None,
            channel,
        )
        .await
        .unwrap();
        let id = request_id
            .lock()
            .unwrap()
            .clone()
            .expect("the done event carries the request id");

        let filter = LogFilter {
            model: Some(model.clone()),
            ..LogFilter::default()
        };
        let list = hypnos_logs(app.state(), filter).await.unwrap();
        assert!(list.capacity > 0);
        let row = list
            .records
            .iter()
            .find(|r| r.id == id)
            .expect("the chat is in the log");
        assert_eq!(row.route, "/v1/chat/completions");
        assert!(list
            .records
            .iter()
            .all(|r| r.model.as_deref() == Some(model.as_str())));

        let record = hypnos_log(app.state(), id.clone()).await.unwrap();
        assert_eq!(record.id, id);
        assert_eq!(record.status, 200);

        let errors = hypnos_logs(
            app.state(),
            LogFilter {
                status: Some("error".into()),
                ..LogFilter::default()
            },
        )
        .await
        .unwrap();
        assert!(errors
            .records
            .iter()
            .all(|r| r.status >= 400 || r.error.is_some()));
    }

    /// Switches KV parking and a model's MoE offload on a running server built with MEM-4/5:
    /// `HYPNOS_TEST_URL=... HYPNOS_TEST_MOE_MODEL=gemma4:26b-q4 cargo test switches -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn switches_kv_parking_and_moe_offload() {
        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let model = std::env::var("HYPNOS_TEST_MOE_MODEL").expect("HYPNOS_TEST_MOE_MODEL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();

        let before = hypnos_kv_park(app.state()).await.unwrap();
        let on = hypnos_set_kv_park(app.state(), Some(true), Some(2 << 30))
            .await
            .unwrap();
        assert!(on.enabled);
        assert_eq!(on.max_bytes, 2 << 30);
        let status = hypnos_memory_status(app.state()).await.unwrap();
        assert!(status.kv_park.as_ref().unwrap().enabled);
        hypnos_set_kv_park(app.state(), Some(before.enabled), Some(before.max_bytes))
            .await
            .unwrap();

        let m = status.models.iter().find(|m| m.id == model).unwrap();
        assert!(m.expert_bytes > 0, "{model} has no experts");
        let original = m.moe_offload.unwrap_or(MoeOffload::Off);
        let other = if original == MoeOffload::On {
            MoeOffload::Off
        } else {
            MoeOffload::On
        };
        let r = hypnos_set_moe_offload(app.state(), model.clone(), other)
            .await
            .unwrap();
        assert_eq!(r.moe_offload, other);
        let status = hypnos_memory_status(app.state()).await.unwrap();
        assert_eq!(
            status
                .models
                .iter()
                .find(|m| m.id == model)
                .unwrap()
                .moe_offload,
            Some(other)
        );
        hypnos_set_moe_offload(app.state(), model, original)
            .await
            .unwrap();
    }
}
