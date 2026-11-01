//! The tool registry (DESK-23): the chat model orchestrates the specialist models.
//!
//! Each [`Tool`] wraps a job another model does (or a fetch the app does) and is offered to the
//! chat model only when a running model can do that job, as the image tool always was. The chat
//! model may call several in one round; they run in parallel ([`run_all`]), each shown in the chat
//! as a [`ChatEvent::Step`], and their results go back for the next round (`hypnos.rs`
//! `answer_with_tools`). Tool requests carry `x-hypnos-next-model` naming the chat model, so the
//! server brings it back while a specialist works when that one had to evict it, and every request
//! of a turn carries the turn's `x-hypnos-workflow`.
//!
//! Attachments are prepared when they are attached (documents, audio and video become text), so
//! reading them needs no tool; `fetch_url` covers pages and files the user only names.

use std::time::Instant;

use futures::future::join_all;
use hypnos_client::{
    ChatMessage, ChatRequest, Client, ContentPart, EntitiesRequest, ModelStatus, ToolCall as Call,
    ToolSpec,
};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tauri::ipc::Channel;
use tauri::{AppHandle, Manager, Runtime};

use crate::attachments::Attachment;
use crate::hypnos::{serves, ChatEvent, ErrorPayload, Hypnos, References};
use crate::images::{self, Reference, IMAGE_TOOL};

/// Characters of a fetched page the chat model gets back (about 4k tokens).
const PAGE_CHARS: usize = 16_000;
/// Entities listed back to the chat model.
const MAX_ENTITIES: usize = 60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tool {
    /// `generate_image` (DESK-11, DESK-22): create or edit an image.
    Image,
    /// `fetch_url`: a web page or file as markdown (docling for HTML, PDF, Office).
    FetchUrl,
    /// `speak`: read a text aloud with the speech model, in the app.
    Speak,
    /// `extract_entities`: GLiNER's named entities for the labels the chat model picks.
    Entities,
    /// `describe_image`: a vision model looks at an image the chat model cannot see.
    DescribeImage,
}

pub const ALL: [Tool; 5] = [
    Tool::Image,
    Tool::FetchUrl,
    Tool::Speak,
    Tool::Entities,
    Tool::DescribeImage,
];

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Tool::Image => IMAGE_TOOL,
            Tool::FetchUrl => "fetch_url",
            Tool::Speak => "speak",
            Tool::Entities => "extract_entities",
            Tool::DescribeImage => "describe_image",
        }
    }

    pub fn by_name(name: &str) -> Option<Tool> {
        ALL.into_iter().find(|t| t.name() == name)
    }

    /// The modality a model must serve for the tool to be offered.
    pub fn job(self) -> &'static str {
        match self {
            Tool::Image => "image_generation",
            Tool::FetchUrl => "document_conversion",
            Tool::Speak => "text_to_speech",
            Tool::Entities => "entity_extraction",
            Tool::DescribeImage => "vision",
        }
    }

    pub fn spec(self) -> ToolSpec {
        match self {
            Tool::Image => images::image_tool_spec(),
            Tool::FetchUrl => ToolSpec::function(
                self.name(),
                "Read a web page or an online file (HTML, PDF, Word, slides) as text. Call it when the user gives a URL or asks about a page they name by address; then answer from its text and say where it came from.",
                json!({
                    "type": "object",
                    "properties": {
                        "url": { "type": "string", "description": "The full http:// or https:// address" }
                    },
                    "required": ["url"]
                }),
            ),
            Tool::Speak => ToolSpec::function(
                self.name(),
                "Read a text aloud to the user with the speech model. Call it only when the user asks to hear something (\"read this out\", \"say it\", \"pronounce\"); pass exactly the words to speak.",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "The words to say, in plain sentences (no markdown)" }
                    },
                    "required": ["text"]
                }),
            ),
            Tool::Entities => ToolSpec::function(
                self.name(),
                "Find named entities in a text with a dedicated extraction model: people, organizations, places, dates, amounts, products, or any labels you choose. Call it when the user asks to list, extract or tag such things in a long text or document.",
                json!({
                    "type": "object",
                    "properties": {
                        "text": { "type": "string", "description": "The text to search, copied in full" },
                        "labels": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "What to find, in plain words: [\"person\", \"company\", \"date\"]"
                        }
                    },
                    "required": ["text", "labels"]
                }),
            ),
            Tool::DescribeImage => ToolSpec::function(
                self.name(),
                "Ask a vision model about an image you cannot see: the one the user attached, or the image you generated last. Call it before answering a question about that image.",
                json!({
                    "type": "object",
                    "properties": {
                        "question": { "type": "string", "description": "What to find out, e.g. \"Describe the image in detail\" or \"What is written on the sign?\"" },
                        "image": { "type": "string", "enum": ["attached", "previous"], "description": "attached = the user's image, previous = the one you generated" }
                    },
                    "required": ["question"]
                }),
            ),
        }
    }

    /// What the step says while it runs and when it is done (`Step::label`).
    fn label(self, args: &serde_json::Value) -> String {
        let s = |k: &str| args.get(k).and_then(|v| v.as_str()).unwrap_or("").trim();
        match self {
            Tool::Image => match args.get("reference").and_then(|v| v.as_str()) {
                Some("attached" | "previous") => "Editing an image".into(),
                _ => "Creating an image".into(),
            },
            Tool::FetchUrl => format!("Reading {}", s("url")),
            Tool::Speak => "Reading aloud".into(),
            Tool::Entities => {
                let labels: Vec<&str> = args
                    .get("labels")
                    .and_then(|v| v.as_array())
                    .map(|a| a.iter().filter_map(|l| l.as_str()).collect())
                    .unwrap_or_default();
                format!("Finding {}", labels.join(", "))
            }
            Tool::DescribeImage => format!("Looking at the image: {}", s("question")),
        }
    }
}

/// Which tools are on for a chat: the image tool when the settings allow it, the others when
/// `assistant` is on; each only when an enabled model does its job. `describe_image` only
/// helps a chat model that cannot see, and only when there is an image to look at.
pub fn available(
    models: &[ModelStatus],
    chat_model: &str,
    image: bool,
    assistant: bool,
    references: &References,
) -> Vec<Tool> {
    let any = |job: &str| models.iter().any(|m| serves(m, job));
    let chat_sees = models
        .iter()
        .any(|m| m.id == chat_model && serves(m, "vision"));
    ALL.into_iter()
        .filter(|t| match t {
            Tool::Image => image,
            Tool::DescribeImage => {
                assistant && !chat_sees && references.pick(Reference::Previous).is_some()
            }
            _ => assistant,
        })
        .filter(|t| any(t.job()))
        .collect()
}

/// One tool call as the chat shows it.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Step {
    /// The call's id: a `done` / `error` step replaces the `running` one with the same id.
    pub id: String,
    pub tool: String,
    pub label: String,
    /// `running`, `done` or `error`.
    pub status: &'static str,
    pub model: Option<String>,
    pub seconds: Option<f64>,
    /// One line about the result (a page's title, how many entities), or the error.
    pub summary: Option<String>,
    /// The call's arguments, as the chat model sent them.
    pub input: serde_json::Value,
    /// What went back to the chat model (cut at [`STEP_OUTPUT_CHARS`] for the panel); set when
    /// the call is done or failed.
    pub output: Option<String>,
}

/// Characters of a tool's output the chat's tool panel shows.
pub const STEP_OUTPUT_CHARS: usize = 8_000;

/// `text` cut for the tool panel.
pub fn panel_text(text: &str) -> String {
    let total = text.chars().count();
    if total <= STEP_OUTPUT_CHARS {
        return text.to_string();
    }
    let head: String = text.chars().take(STEP_OUTPUT_CHARS).collect();
    format!("{head}\n… ({} more characters)", total - STEP_OUTPUT_CHARS)
}

/// What a tool run gives back: the text for the chat model, plus what the step shows.
struct Outcome {
    result: String,
    model: Option<String>,
    summary: Option<String>,
}

/// The context every tool runs in.
pub struct Ctx<'a, R: Runtime> {
    pub app: &'a AppHandle<R>,
    /// With the turn's workflow and the chat model as the next model.
    pub client: &'a Client,
    pub references: &'a References,
    pub on_event: &'a Channel<ChatEvent>,
}

/// Runs a round's calls in parallel; their results, in the calls' order, for `tool` messages.
pub async fn run_all<R: Runtime>(ctx: &Ctx<'_, R>, calls: &[Call]) -> Vec<String> {
    join_all(calls.iter().map(|c| run(ctx, c))).await
}

async fn run<R: Runtime>(ctx: &Ctx<'_, R>, call: &Call) -> String {
    let Some(tool) = Tool::by_name(&call.function.name) else {
        return format!("Unknown tool {}.", call.function.name);
    };
    let args: serde_json::Value =
        serde_json::from_str(&call.function.arguments).unwrap_or(json!({}));
    let label = tool.label(&args);
    let step = |status, model, seconds, summary, output: Option<&str>| Step {
        id: call.id.clone(),
        tool: tool.name().into(),
        label: label.clone(),
        status,
        model,
        seconds,
        summary,
        input: args.clone(),
        output: output.map(panel_text),
    };
    let _ = ctx
        .on_event
        .send(ChatEvent::Step(step("running", None, None, None, None)));
    let started = Instant::now();
    let outcome = match tool {
        Tool::Image => image(ctx, call).await,
        Tool::FetchUrl => fetch_url(ctx, &args).await,
        Tool::Speak => speak(ctx, &args),
        Tool::Entities => entities(ctx, &args).await,
        Tool::DescribeImage => describe(ctx, &args).await,
    };
    let seconds = Some(started.elapsed().as_secs_f64());
    match outcome {
        Ok(o) => {
            let _ = ctx.on_event.send(ChatEvent::Step(step(
                "done",
                o.model,
                seconds,
                o.summary,
                Some(&o.result),
            )));
            o.result
        }
        Err(e) => {
            let result = format!(
                "The tool failed: {}. Tell the user briefly; do not call it again for this.",
                e.message
            );
            let _ = ctx.on_event.send(ChatEvent::Step(step(
                "error",
                None,
                seconds,
                Some(e.message.clone()),
                Some(&result),
            )));
            if tool == Tool::Image {
                // The image's own progress line goes away with a note.
                let _ = ctx.on_event.send(ChatEvent::ToolError {
                    message: e.message.clone(),
                });
            }
            result
        }
    }
}

fn hypnos<'a, R: Runtime>(ctx: &Ctx<'a, R>) -> tauri::State<'a, Hypnos> {
    ctx.app.state::<Hypnos>()
}

fn bad_args(tool: Tool, e: impl std::fmt::Display) -> ErrorPayload {
    ErrorPayload {
        kind: "tool".into(),
        message: format!("{} was called without usable arguments", tool.name()),
        detail: Some(e.to_string()),
        retry_after_secs: None,
    }
}

async fn image<R: Runtime>(ctx: &Ctx<'_, R>, call: &Call) -> Result<Outcome, ErrorPayload> {
    let args: images::ToolCall = call
        .function
        .parse_arguments()
        .map_err(|e| bad_args(Tool::Image, e))?;
    let args = args.normalized();
    if args.prompt.is_empty() {
        return Err(bad_args(Tool::Image, "an empty prompt"));
    }
    let _ = ctx.on_event.send(ChatEvent::ToolCall(args.clone()));
    let reference = ctx.references.pick(args.reference);
    let image = images::generate_with(&hypnos(ctx), ctx.client, &args, reference).await?;
    let result = crate::hypnos::tool_result(&image);
    let model = Some(image.model.clone());
    let summary = Some(match (&image.seed, image.more.len()) {
        (Some(seed), 0) => format!("{}, seed {seed}", image.size),
        (_, more) if more > 0 => format!("{} images, {}", more + 1, image.size),
        _ => image.size.clone(),
    });
    let _ = ctx.on_event.send(ChatEvent::Image(Box::new(image)));
    Ok(Outcome {
        result,
        model,
        summary,
    })
}

#[derive(Deserialize)]
struct UrlArgs {
    url: String,
}

async fn fetch_url<R: Runtime>(
    ctx: &Ctx<'_, R>,
    args: &serde_json::Value,
) -> Result<Outcome, ErrorPayload> {
    let a: UrlArgs =
        serde_json::from_value(args.clone()).map_err(|e| bad_args(Tool::FetchUrl, e))?;
    let page = crate::web::extract(&hypnos(ctx), ctx.client, a.url.trim()).await?;
    let Attachment::Web {
        name, url, text, ..
    } = page
    else {
        return Err(bad_args(Tool::FetchUrl, "not a web page"));
    };
    let total = text.chars().count();
    let cut = total > PAGE_CHARS;
    let body: String = text.chars().take(PAGE_CHARS).collect();
    let more = if cut {
        format!(" truncated=\"true\" chars=\"{total}\"")
    } else {
        String::new()
    };
    Ok(Outcome {
        result: format!("<page title=\"{name}\" url=\"{url}\"{more}>\n{body}\n</page>"),
        model: None,
        summary: Some(format!("{name} ({} k characters)", total.div_ceil(1000))),
    })
}

#[derive(Deserialize)]
struct SpeakArgs {
    text: String,
}

/// The webview reads it aloud with the user's voice and speed (`speak` event), as it reads
/// answers.
fn speak<R: Runtime>(ctx: &Ctx<'_, R>, args: &serde_json::Value) -> Result<Outcome, ErrorPayload> {
    let a: SpeakArgs =
        serde_json::from_value(args.clone()).map_err(|e| bad_args(Tool::Speak, e))?;
    let text = a.text.trim().to_string();
    if text.is_empty() {
        return Err(bad_args(Tool::Speak, "no text"));
    }
    let words = text.split_whitespace().count();
    let _ = ctx.on_event.send(ChatEvent::Speak { text });
    Ok(Outcome {
        result: "The text is being read aloud to the user now. Do not repeat it; at most say in a few words that you are reading it.".into(),
        model: None,
        summary: Some(format!("{words} words")),
    })
}

#[derive(Deserialize)]
struct EntitiesArgs {
    text: String,
    labels: Vec<String>,
}

async fn entities<R: Runtime>(
    ctx: &Ctx<'_, R>,
    args: &serde_json::Value,
) -> Result<Outcome, ErrorPayload> {
    let a: EntitiesArgs =
        serde_json::from_value(args.clone()).map_err(|e| bad_args(Tool::Entities, e))?;
    let labels: Vec<&str> = a
        .labels
        .iter()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    if labels.is_empty() || a.text.trim().is_empty() {
        return Err(bad_args(Tool::Entities, "no text or no labels"));
    }
    let model = hypnos(ctx).pick("entity_extraction", None).await?;
    let response = ctx
        .client
        .entities(&EntitiesRequest::new(
            model.clone(),
            a.text.as_str(),
            &labels,
        ))
        .await?;
    let found: Vec<_> = response
        .data
        .iter()
        .flat_map(|d| d.entities.iter())
        .collect();
    let mut lines: Vec<String> = found
        .iter()
        .take(MAX_ENTITIES)
        .map(|e| format!("{}: {} ({:.2})", e.label, e.text, e.score))
        .collect();
    if found.len() > MAX_ENTITIES {
        lines.push(format!("… and {} more", found.len() - MAX_ENTITIES));
    }
    let result = if lines.is_empty() {
        "No entities of these kinds were found.".to_string()
    } else {
        format!(
            "Entities found (label: text (score)):\n{}",
            lines.join("\n")
        )
    };
    Ok(Outcome {
        result,
        model: Some(model),
        summary: Some(format!("{} found", found.len())),
    })
}

#[derive(Deserialize)]
struct DescribeArgs {
    question: String,
    #[serde(default)]
    image: Option<String>,
}

async fn describe<R: Runtime>(
    ctx: &Ctx<'_, R>,
    args: &serde_json::Value,
) -> Result<Outcome, ErrorPayload> {
    let a: DescribeArgs =
        serde_json::from_value(args.clone()).map_err(|e| bad_args(Tool::DescribeImage, e))?;
    let which = match a.image.as_deref() {
        Some("previous") => Reference::Previous,
        _ => Reference::Attached,
    };
    let data_url = ctx
        .references
        .pick(which)
        .ok_or_else(|| bad_args(Tool::DescribeImage, "there is no image in this chat"))?;
    let model = hypnos(ctx).pick("vision", None).await?;
    let question = if a.question.trim().is_empty() {
        "Describe the image in detail."
    } else {
        a.question.trim()
    };
    let request = ChatRequest::new(
        model.clone(),
        vec![ChatMessage::user_parts(vec![
            ContentPart::text(question),
            ContentPart::image_url(data_url),
        ])],
    );
    let answer = ctx.client.chat(&request).await?;
    let text = answer.text().trim().to_string();
    Ok(Outcome {
        summary: Some(format!("{} words", text.split_whitespace().count())),
        result: format!("{model} looked at the image and says:\n{text}"),
        model: Some(model),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn status(id: &str, modalities: &[&str]) -> ModelStatus {
        serde_json::from_value(json!({
            "id": id, "enabled": true, "modality": modalities[0], "modalities": modalities,
            "current_tier": "tier1_gpu_vram", "placement": { "primary_tier": "tier1_gpu_vram" },
        }))
        .unwrap()
    }

    #[test]
    fn tools_follow_the_models_that_do_their_job() {
        let models = vec![
            status("chat", &["text_generation"]),
            status("img", &["image_generation"]),
            status("docs", &["document_conversion"]),
            status("seer", &["vision", "text_generation"]),
        ];
        let none = References::default();
        let names = |t: Vec<Tool>| t.into_iter().map(Tool::name).collect::<Vec<_>>();
        assert_eq!(
            names(available(&models, "chat", true, true, &none)),
            ["generate_image", "fetch_url"]
        );
        assert_eq!(
            names(available(&models, "chat", false, false, &none)),
            Vec::<&str>::new()
        );
        // With an image in the chat, a chat model that cannot see gets describe_image.
        let with_image = References::with_previous("data:x");
        assert!(available(&models, "chat", false, true, &with_image).contains(&Tool::DescribeImage));
        assert!(
            !available(&models, "seer", false, true, &with_image).contains(&Tool::DescribeImage)
        );
    }

    #[test]
    fn the_panel_shows_long_outputs_cut() {
        assert_eq!(panel_text("short"), "short");
        let long = "x".repeat(STEP_OUTPUT_CHARS + 5);
        let cut = panel_text(&long);
        assert!(
            cut.ends_with("… (5 more characters)"),
            "{}",
            &cut[cut.len() - 40..]
        );
        assert_eq!(
            Tool::Image.label(&json!({"prompt": "a", "reference": "previous"})),
            "Editing an image"
        );
    }

    #[test]
    fn every_tool_has_a_valid_spec_and_a_label() {
        for tool in ALL {
            let spec = serde_json::to_value(tool.spec()).unwrap();
            assert_eq!(spec["function"]["name"], tool.name());
            assert_eq!(Tool::by_name(tool.name()), Some(tool));
            assert!(spec["function"]["parameters"]["required"].is_array());
        }
        assert_eq!(
            Tool::Entities.label(&json!({"labels": ["person", "date"]})),
            "Finding person, date"
        );
        assert_eq!(
            Tool::FetchUrl.label(&json!({"url": "https://example.org"})),
            "Reading https://example.org"
        );
    }

    /// On a running server with a tool-calling chat model, docling and GLiNER (the lineup):
    /// one turn that needs two tools, a web page and entities, run in the same round or two.
    /// `HYPNOS_TEST_URL=... cargo test two_tools_in_one_turn -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn two_tools_in_one_turn() {
        use crate::hypnos::{hypnos_chat, hypnos_connect, InMessage};
        use std::sync::{Arc, Mutex};

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
        let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let e = events.clone();
        let channel = tauri::ipc::Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                e.lock().unwrap().push(serde_json::from_str(&s).unwrap());
            }
            Ok(())
        });
        let message = InMessage {
            role: "user".into(),
            content: "Two things. First, read https://example.com and tell me in one sentence what that page is for. Second, use the entity extractor to list the organizations and places in this sentence: \"Apple and Microsoft met the European Commission in Brussels before flying to Tokyo.\"".into(),
            attachments: vec![],
            image_prompt: None,
            image: None,
        };
        hypnos_chat(
            app.handle().clone(),
            app.state(),
            "tools1".into(),
            model,
            vec![message],
            Some(false),
            None,
            Some(true),
            channel,
        )
        .await
        .unwrap();
        let events = events.lock().unwrap();
        let steps: Vec<&serde_json::Value> =
            events.iter().filter(|e| e["event"] == "step").collect();
        for s in &steps {
            eprintln!("step: {s}");
        }
        let answer: String = events
            .iter()
            .filter(|e| e["event"] == "delta")
            .filter_map(|e| e["text"].as_str())
            .collect();
        eprintln!("answer: {answer}");
        let done = |tool: &str| {
            steps
                .iter()
                .any(|s| s["tool"] == tool && s["status"] == "done")
        };
        assert!(done("fetch_url"), "no fetch_url step");
        assert!(done("extract_entities"), "no extract_entities step");
        assert!(
            answer.contains("Brussels") || answer.contains("Microsoft"),
            "{answer}"
        );
        assert_eq!(
            events
                .iter()
                .filter_map(|e| e["event"].as_str())
                .next_back(),
            Some("done")
        );
    }

    /// `speak` on a running server with a speech model: the call becomes a `speak` event.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn the_speak_tool_reads_aloud() {
        use crate::hypnos::{hypnos_chat, hypnos_connect, InMessage};
        use std::sync::{Arc, Mutex};

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
        let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
        let e = events.clone();
        let channel = tauri::ipc::Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                e.lock().unwrap().push(serde_json::from_str(&s).unwrap());
            }
            Ok(())
        });
        let message = InMessage {
            role: "user".into(),
            content: "Say \"good morning, Hypnos\" out loud for me.".into(),
            attachments: vec![],
            image_prompt: None,
            image: None,
        };
        hypnos_chat(
            app.handle().clone(),
            app.state(),
            "speak1".into(),
            model,
            vec![message],
            Some(false),
            None,
            Some(true),
            channel,
        )
        .await
        .unwrap();
        let events = events.lock().unwrap();
        let spoken = events
            .iter()
            .find(|e| e["event"] == "speak")
            .expect("a speak event");
        eprintln!("spoken: {spoken}");
        assert!(spoken["text"]
            .as_str()
            .unwrap()
            .to_lowercase()
            .contains("good morning"));
    }
}
