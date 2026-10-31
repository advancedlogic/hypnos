//! The image tool (DESK-11). With a model the server can run tools for (API-9), `generate_image`
//! is sent as a real OpenAI tool ([`image_tool_spec`]). For a model without a native tool format
//! the app falls back to a prompt convention: a system instruction tells the chat model how to ask
//! for an image and [`ToolGate`] holds back the start of each answer until it is clear whether it
//! is such a request. Either way the request runs on the `image_generation` model. The image button
//! and `/image` call [`hypnos_generate_image`] directly.

use std::time::Instant;

use base64::Engine;
use hypnos_client::{ImageRequest, Telemetry};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::attachments::downscale_jpeg;
use crate::hypnos::{ErrorPayload, Hypnos};

pub const IMAGE_TOOL: &str = "generate_image";

/// `generate_image` as an OpenAI tool.
pub fn image_tool_spec() -> hypnos_client::ToolSpec {
    hypnos_client::ToolSpec::function(
        IMAGE_TOOL,
        "Create an image with the image generator. Call it when the user asks you to draw, paint, generate, create or show an image or picture.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "A detailed visual description in English: subject, setting, style, lighting, composition"
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1344x768", "768x1344"],
                    "description": "1344x768 for wide scenes, 768x1344 for tall ones"
                }
            },
            "required": ["prompt"]
        }),
    )
}

/// The fence the chat model opens to call the tool (prompt-convention fallback).
pub const TOOL_FENCE: &str = "```generate_image";
pub const DEFAULT_SIZE: &str = "1024x1024";

/// Added to the system prompt when an enabled model generates images.
pub const TOOL_INSTRUCTION: &str = "You can create images with an image generator. When the user asks you to draw, paint, generate, create or show an image or picture, reply with only this block and nothing before or after it:\n```generate_image\n{\"prompt\": \"<a detailed visual description in English: subject, setting, style, lighting, composition>\", \"size\": \"1024x1024\"}\n```\nUse \"1344x768\" for wide scenes and \"768x1344\" for tall ones. For anything else, answer normally and never mention this block.";

/// What the chat model asked for.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct ToolCall {
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
}

fn default_size() -> String {
    DEFAULT_SIZE.into()
}

/// Watches the start of a streamed answer. Text that cannot be a tool call is passed on as soon
/// as that is clear; a tool call is kept whole and never shown.
#[derive(Debug, Default)]
pub struct ToolGate {
    held: String,
    decided: Option<bool>,
}

impl ToolGate {
    /// Takes a piece of the answer; returns what may be shown now.
    pub fn feed(&mut self, piece: &str) -> Option<String> {
        match self.decided {
            Some(false) => return Some(piece.to_string()),
            Some(true) => {
                self.held.push_str(piece);
                return None;
            }
            None => self.held.push_str(piece),
        }
        let start = self.held.trim_start();
        if start.len() >= TOOL_FENCE.len() {
            let is_call = start.starts_with(TOOL_FENCE);
            self.decided = Some(is_call);
            if !is_call {
                return Some(std::mem::take(&mut self.held));
            }
        } else if !TOOL_FENCE.starts_with(start) {
            self.decided = Some(false);
            return Some(std::mem::take(&mut self.held));
        }
        None
    }

    /// At the end of the answer: the tool call if there was one, else the text still held.
    pub fn finish(self) -> Result<ToolCall, String> {
        if self.decided == Some(true) {
            parse_call(&self.held).ok_or(self.held)
        } else {
            Err(self.held)
        }
    }
}

/// The JSON inside a ```generate_image block (lenient about what surrounds it).
pub fn parse_call(text: &str) -> Option<ToolCall> {
    let body = text.trim_start().strip_prefix(TOOL_FENCE)?;
    let body = body.split("```").next()?.trim();
    let call: ToolCall = serde_json::from_str(body).ok().or_else(|| {
        let start = body.find('{')?;
        let end = body.rfind('}')?;
        serde_json::from_str(&body[start..=end]).ok()
    })?;
    (!call.prompt.trim().is_empty()).then_some(call)
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Generated {
    /// A JPEG data URL no larger than 1024 px (it is kept with the chat).
    pub data_url: String,
    pub prompt: String,
    pub size: String,
    pub model: String,
    pub seconds: f64,
    pub telemetry: Option<Box<Telemetry>>,
}

/// Generates one image on the model the router picks for `image_generation`.
pub async fn generate(
    hypnos: &Hypnos,
    prompt: &str,
    size: Option<&str>,
) -> Result<Generated, ErrorPayload> {
    let model = hypnos.pick("image_generation", None).await?;
    let client = hypnos.client().await?;
    let size = size
        .filter(|s| !s.trim().is_empty())
        .unwrap_or(DEFAULT_SIZE)
        .to_string();
    let started = Instant::now();
    let response = client
        .images(&ImageRequest::new(model.clone(), prompt).size(size.clone()))
        .await?;
    let seconds = started.elapsed().as_secs_f64();
    let b64 = response
        .data
        .first()
        .map(|d| d.b64_json.clone())
        .filter(|b| !b.is_empty())
        .ok_or_else(|| fail("The image model returned no image."))?;
    let png = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| fail(format!("The image could not be read: {e}")))?;
    let (data_url, _, _) = tauri::async_runtime::spawn_blocking(move || downscale_jpeg(&png))
        .await
        .map_err(|e| fail(e.to_string()))?
        .map_err(fail)?;
    Ok(Generated {
        data_url,
        prompt: prompt.to_string(),
        size,
        model,
        seconds,
        telemetry: response.hypnos_telemetry.map(Box::new),
    })
}

fn fail(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload {
        kind: "image".into(),
        message: message.into(),
        detail: None,
        retry_after_secs: None,
    }
}

/// The image button and `/image <prompt>`: no chat model involved.
#[tauri::command]
pub async fn hypnos_generate_image(
    hypnos: State<'_, Hypnos>,
    prompt: String,
    size: Option<String>,
) -> Result<Generated, ErrorPayload> {
    generate(&hypnos, &prompt, size.as_deref()).await
}

/// Writes a `data:` URL's bytes to the path chosen in the save dialog.
#[tauri::command]
pub async fn save_data_url(path: String, data_url: String) -> Result<u64, ErrorPayload> {
    let b64 = data_url
        .split_once(";base64,")
        .map(|(_, b)| b)
        .ok_or_else(|| fail("Not a base64 data URL."))?;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .map_err(|e| fail(e.to_string()))?;
    tokio::fs::write(&path, &bytes)
        .await
        .map_err(|e| fail(format!("Could not save {path}: {e}")))?;
    Ok(bytes.len() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(pieces: &[&str]) -> (String, ToolGate) {
        let mut gate = ToolGate::default();
        let shown: String = pieces.iter().filter_map(|p| gate.feed(p)).collect();
        (shown, gate)
    }

    #[test]
    fn ordinary_answers_pass_through_almost_at_once() {
        let (shown, gate) = run(&["Sure", "! Here is", " the answer."]);
        assert_eq!(shown, "Sure! Here is the answer.");
        assert_eq!(gate.finish().unwrap_err(), "");
        // A code block of another language is not a tool call.
        let (shown, _) = run(&["```py", "thon\nprint(1)\n```"]);
        assert_eq!(shown, "```python\nprint(1)\n```");
    }

    #[test]
    fn a_tool_call_is_held_and_parsed() {
        let (shown, gate) = run(&[
            "\n``",
            "`generate_im",
            "age\n{\"prompt\": \"a red fox in snow, watercolor\",",
            " \"size\": \"1344x768\"}\n```",
        ]);
        assert_eq!(shown, "");
        let call = gate.finish().unwrap();
        assert_eq!(call.prompt, "a red fox in snow, watercolor");
        assert_eq!(call.size, "1344x768");
    }

    #[test]
    fn calls_tolerate_missing_size_and_chatter() {
        let call = parse_call("```generate_image\nHere: {\"prompt\": \"a cat\"}\n```").unwrap();
        assert_eq!(call.size, DEFAULT_SIZE);
        assert!(parse_call("```generate_image\n{\"prompt\": \"\"}\n```").is_none());
        assert!(parse_call("```generate_image\nnot json\n```").is_none());
    }

    #[test]
    fn a_short_answer_that_looks_like_the_fence_start_is_kept_until_the_end() {
        let (shown, gate) = run(&["``"]);
        assert_eq!(shown, "");
        assert_eq!(gate.finish().unwrap_err(), "``");
    }

    /// The whole loop on a running server (it loads the image model; a minute or more):
    /// `HYPNOS_TEST_URL=... cargo test images -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn the_chat_model_calls_the_image_tool() {
        use crate::hypnos::{hypnos_chat, hypnos_connect, InMessage};
        use std::sync::{Arc, Mutex};
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
            content: "Please draw a picture of a single red apple on a white table.".into(),
            attachments: vec![],
            image_prompt: None,
        };
        let outcome = hypnos_chat(
            app.handle().clone(),
            app.state(),
            "img1".into(),
            model,
            vec![message],
            Some(true),
            None,
            channel,
        )
        .await
        .unwrap();
        assert!(!outcome.cancelled);

        let events = events.lock().unwrap();
        let kinds: Vec<&str> = events.iter().filter_map(|e| e["event"].as_str()).collect();
        let call = events
            .iter()
            .find(|e| e["event"] == "toolCall")
            .unwrap_or_else(|| panic!("{kinds:?}"));
        assert!(
            call["prompt"]
                .as_str()
                .unwrap()
                .to_lowercase()
                .contains("apple"),
            "{call}"
        );
        let image = events.iter().find(|e| e["event"] == "image").unwrap();
        assert!(image["dataUrl"]
            .as_str()
            .unwrap()
            .starts_with("data:image/jpeg;base64,"));
        let comment: String = events
            .iter()
            .filter(|e| e["event"] == "delta")
            .filter_map(|e| e["text"].as_str())
            .collect();
        assert!(!comment.contains("generate_image"), "{comment}");
        assert!(!comment.trim().is_empty());
        assert_eq!(kinds.last(), Some(&"done"));
        eprintln!(
            "prompt: {}\ncomment: {comment}\ntook {} s",
            call["prompt"], image["seconds"]
        );
    }
}
