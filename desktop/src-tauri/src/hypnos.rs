//! Every call to the Hypnos server goes through here, on the Rust SDK (`hypnos-client`). The
//! webview never talks HTTP: it calls these commands, and chat tokens come back over a
//! [`Channel`]. The connection (URL + the key from the keychain) lives in [`Hypnos`].

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use hypnos_client::{
    ChatMessage, ChatRequest, Client, Content, ContentPart, Error, MemoryStatus, MemoryTier,
    ModelStatus, Telemetry, Usage,
};

use crate::attachments::Attachment;
use serde::{Deserialize, Serialize};
use tauri::async_runtime::JoinHandle;
use tauri::ipc::Channel;
use tauri::State;
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
    /// A piece of the answer.
    Delta { text: String },
    /// The answer is complete.
    #[serde(rename_all = "camelCase")]
    Done {
        telemetry: Option<Box<Telemetry>>,
        usage: Option<Usage>,
        finish_reason: Option<String>,
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
#[tauri::command]
pub async fn hypnos_chat(
    hypnos: State<'_, Hypnos>,
    request_id: String,
    model: String,
    messages: Vec<InMessage>,
    on_event: Channel<ChatEvent>,
) -> CmdResult<ChatOutcome> {
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
    let messages = messages.iter().map(|m| m.to_chat_message(vision)).collect();
    let request = ChatRequest::new(model, messages);
    let (tx, rx) = tokio::sync::oneshot::channel();
    let task = tauri::async_runtime::spawn(async move {
        let _ = tx.send(stream_answer(&client, &request, &on_event).await);
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

async fn stream_answer(
    client: &Client,
    request: &ChatRequest,
    on_event: &Channel<ChatEvent>,
) -> CmdResult<()> {
    let mut stream = client.chat_stream(request).await?;
    {
        let mut pieces = stream.text();
        while let Some(piece) = pieces.next().await {
            let text = piece?;
            if on_event.send(ChatEvent::Delta { text }).is_err() {
                return Ok(()); // the window is gone
            }
        }
    }
    let _ = on_event.send(ChatEvent::Done {
        telemetry: stream.telemetry().cloned().map(Box::new),
        usage: stream.usage().cloned(),
        finish_reason: stream.finish_reason().map(str::to_string),
    });
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
            }]
        };
        let outcome = hypnos_chat(
            app.state(),
            "r1".into(),
            model.clone(),
            message("Reply with exactly: hello from rust"),
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
            state.clone(),
            "r2".into(),
            model,
            message("Count from 1 to 500, one number per line."),
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
            app.state(),
            "r3".into(),
            "no-such-model".into(),
            message("hi"),
            channel,
        )
        .await
        .unwrap_err();
        assert_eq!(err.kind, "not_found", "{err:?}");
    }
}
