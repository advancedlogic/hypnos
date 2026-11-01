//! The image tool (DESK-11). With a model the server can run tools for (API-9), `generate_image`
//! is sent as a real OpenAI tool ([`image_tool_spec`]). For a model without a native tool format
//! the app falls back to a prompt convention: a system instruction tells the chat model how to ask
//! for an image and [`ToolGate`] holds back the start of each answer until it is clear whether it
//! is such a request. Either way the request runs on the `image_generation` model. The image button
//! and `/image` call [`hypnos_generate_image`] directly.
//!
//! DESK-22: the chat model fills the request from the conversation ([`ToolCall`]: size, negative
//! prompt, count, steps, seed, and a reference image to edit: one attached to the message or the
//! image made just before). Each image keeps the settings it was made with ([`Generated`]); they
//! go back to the chat model as the tool result and with the history, so a follow-up can change
//! one and keep the rest (the server returns each image's seed).

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
        "Create or edit an image with the image generator. Call it when the user asks you to draw, paint, generate, create, show or change an image or picture. Take the settings from the conversation; leave out what the user did not ask for.",
        serde_json::json!({
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "In English. A new image: a detailed visual description (subject, setting, style, lighting, composition); styles and looks the user asks for (black and white, watercolor, vintage film, pixel art) and what to leave out (\"no text anywhere\") go here too. An edit (reference set): the change as an instruction, then what to keep (\"Make it night under a full moon with stars; keep the lighthouse, the cliff and the framing\")."
                },
                "size": {
                    "type": "string",
                    "description": "WIDTHxHEIGHT in pixels, each 256 to 2048: 1024x1024 square (default), 1344x768 wide or landscape, 768x1344 tall, portrait or a phone wallpaper, 1536x640 a banner; the size the user names"
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "What must not appear, when the user says so (\"text, people\"); also say it in the prompt, since fast models ignore this field"
                },
                "n": {
                    "type": "integer",
                    "description": "How many images, 1 to 4, when the user asks for several options"
                },
                "steps": {
                    "type": "integer",
                    "description": "Only when the user asks for speed or quality: fewer (4-8) for a quick sketch, more (30-50) for the most detail"
                },
                "seed": {
                    "type": "integer",
                    "description": "The seed of an earlier image, to keep its composition when the user asks for the same image with a change"
                },
                "reference": {
                    "type": "string",
                    "enum": ["none", "attached", "previous"],
                    "description": "Edit an image instead of starting fresh: attached = the image in the user's message, previous = the image you made last (\"make it darker\", \"add a hat\")"
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
pub const TOOL_INSTRUCTION: &str = "You can create and edit images with an image generator. When the user asks you to draw, paint, generate, create, show or change an image or picture, reply with only this block and nothing before or after it:\n```generate_image\n{\"prompt\": \"<a detailed visual description in English: subject, setting, style, lighting, composition>\", \"size\": \"1024x1024\"}\n```\nSize is WIDTHxHEIGHT (256 to 2048): 1344x768 for wide scenes, 768x1344 for tall ones or phone wallpapers. Add only what the user asks for: \"negative_prompt\" (what must not appear), \"n\" (1 to 4 images), \"steps\" (4-8 quick, 30-50 detailed), \"seed\" (an earlier image's, to keep its composition), \"reference\": \"attached\" or \"previous\" to edit the user's image or the one you made last (then the prompt is the change as an instruction, plus what to keep). Styles (black and white, watercolor, vintage film) and what to leave out go into the prompt. For anything else, answer normally and never mention this block.";

/// Where an edit's reference image comes from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Reference {
    #[default]
    None,
    /// The image attached to the user's latest message.
    Attached,
    /// The image generated last in this chat.
    Previous,
}

/// What the chat model (or the image button) asks for. Lenient: numbers may come as strings,
/// unknown values fall back to the defaults; [`ToolCall::normalized`] clamps them.
#[derive(Debug, Clone, Default, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolCall {
    pub prompt: String,
    #[serde(default = "default_size")]
    pub size: String,
    #[serde(
        default,
        alias = "negative_prompt",
        skip_serializing_if = "Option::is_none"
    )]
    pub negative_prompt: Option<String>,
    #[serde(
        default,
        deserialize_with = "lenient_number",
        skip_serializing_if = "Option::is_none"
    )]
    pub n: Option<u32>,
    #[serde(
        default,
        deserialize_with = "lenient_number",
        skip_serializing_if = "Option::is_none"
    )]
    pub steps: Option<u32>,
    #[serde(
        default,
        deserialize_with = "lenient_number",
        skip_serializing_if = "Option::is_none"
    )]
    pub seed: Option<u64>,
    #[serde(default, deserialize_with = "lenient_reference")]
    pub reference: Reference,
}

fn default_size() -> String {
    DEFAULT_SIZE.into()
}

/// A number, a numeric string, or nothing (null, "", anything else).
fn lenient_number<'de, D, T>(d: D) -> Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: std::str::FromStr,
{
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v {
        Some(serde_json::Value::Number(n)) => n.to_string().parse().ok(),
        Some(serde_json::Value::String(s)) => s.trim().parse().ok(),
        _ => None,
    })
}

fn lenient_reference<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Reference, D::Error> {
    let v = Option::<serde_json::Value>::deserialize(d)?;
    Ok(match v.as_ref().and_then(|v| v.as_str()).map(str::trim) {
        Some(s) if s.eq_ignore_ascii_case("attached") => Reference::Attached,
        Some(s) if s.eq_ignore_ascii_case("previous") || s.eq_ignore_ascii_case("last") => {
            Reference::Previous
        }
        _ => Reference::None,
    })
}

/// `WIDTHxHEIGHT` as the server takes it (256..2048, multiples of 16), or None.
fn parse_size(size: &str) -> Option<(u32, u32)> {
    let lower = size.to_ascii_lowercase();
    let (w, h) = lower.split_once(['x', '×'])?;
    let (w, h): (u32, u32) = (w.trim().parse().ok()?, h.trim().parse().ok()?);
    let snap = |v: u32| (v.clamp(256, 2048) / 16) * 16;
    Some((snap(w), snap(h)))
}

impl ToolCall {
    /// The call as it is sent: a valid size (else the default), 1..=4 images, 1..=100 steps, no
    /// empty negative prompt.
    pub fn normalized(mut self) -> Self {
        self.prompt = self.prompt.trim().to_string();
        self.size = parse_size(&self.size)
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(default_size);
        self.negative_prompt = self
            .negative_prompt
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty());
        self.n = self.n.map(|n| n.clamp(1, 4)).filter(|&n| n > 1);
        self.steps = self.steps.map(|s| s.clamp(1, 100));
        self
    }
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

/// One more image of a call that asked for several (`n`).
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Picture {
    pub data_url: String,
    pub seed: Option<u64>,
}

/// A generated image with the settings it was made with (DESK-22).
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
    /// The seed the server made the first image with: reuse it to keep the composition.
    pub seed: Option<u64>,
    pub negative_prompt: Option<String>,
    /// As asked for; None = the model's default.
    pub steps: Option<u32>,
    /// What the image was edited from (`none` when it was generated from the prompt alone).
    pub reference: Reference,
    /// The other images when the call asked for several.
    pub more: Vec<Picture>,
    /// Why the result differs from the request (an edit the model cannot do).
    pub note: Option<String>,
}

impl Generated {
    /// The settings as the chat model reads them (tool result and history), without the pixels.
    pub fn settings_text(&self) -> String {
        let mut parts = vec![
            format!("prompt \"{}\"", self.prompt),
            format!("size {}", self.size),
        ];
        if let Some(seed) = self.seed {
            parts.push(format!("seed {seed}"));
        }
        if let Some(steps) = self.steps {
            parts.push(format!("steps {steps}"));
        }
        if let Some(neg) = &self.negative_prompt {
            parts.push(format!("negative prompt \"{neg}\""));
        }
        match self.reference {
            Reference::Attached => parts.push("edited from the user's image".into()),
            Reference::Previous => parts.push("edited from the previous image".into()),
            Reference::None => {}
        }
        if !self.more.is_empty() {
            parts.push(format!("{} images", self.more.len() + 1));
        }
        parts.push(format!("model {}", self.model));
        parts.join(", ")
    }
}

/// The bytes of a `data:` URL.
pub(crate) fn data_url_bytes(data_url: &str) -> Option<Vec<u8>> {
    let b64 = data_url.split_once(";base64,")?.1;
    base64::engine::general_purpose::STANDARD.decode(b64).ok()
}

/// Generates the images of `call` on the model the router picks for `image_generation`; with
/// `reference` (a data URL), edits it when the model can (`/v1/images/edits`), else makes a new
/// image from the prompt and says so in `note`.
pub async fn generate(
    hypnos: &Hypnos,
    call: &ToolCall,
    reference: Option<&str>,
) -> Result<Generated, ErrorPayload> {
    let call = call.clone().normalized();
    let model = hypnos.pick("image_generation", None).await?;
    let client = hypnos.client().await?;
    let mut request = ImageRequest::new(model.clone(), call.prompt.clone()).size(call.size.clone());
    if let Some(n) = call.n {
        request = request.n(n);
    }
    if let Some(neg) = &call.negative_prompt {
        request = request.negative_prompt(neg.clone());
    }
    if let Some(steps) = call.steps {
        request = request.steps(steps);
    }
    if let Some(seed) = call.seed {
        request = request.seed(seed);
    }
    let started = Instant::now();
    let mut note = None;
    let mut used = Reference::None;
    let response = match reference.and_then(data_url_bytes) {
        Some(bytes) if call.reference != Reference::None => {
            let images = [("reference.jpg".to_string(), bytes)];
            match client.edit_image(&request, &images).await {
                Ok(r) => {
                    used = call.reference;
                    r
                }
                Err(e) => {
                    let e = ErrorPayload::from(e);
                    // A model without `[models.image] edit = true`: "... does not edit images".
                    let cannot_edit = e.kind == "bad_request"
                        && e.detail
                            .as_deref()
                            .unwrap_or(&e.message)
                            .contains("does not edit");
                    if !cannot_edit {
                        return Err(e);
                    }
                    note = Some(format!(
                        "{model} cannot edit images: this is a new image from the prompt"
                    ));
                    client.images(&request).await?
                }
            }
        }
        _ => {
            if call.reference != Reference::None {
                note =
                    Some("No image to edit was found: this is a new image from the prompt".into());
            }
            client.images(&request).await?
        }
    };
    let seconds = started.elapsed().as_secs_f64();
    let mut pictures = Vec::new();
    for d in &response.data {
        if d.b64_json.is_empty() {
            continue;
        }
        let png = base64::engine::general_purpose::STANDARD
            .decode(&d.b64_json)
            .map_err(|e| fail(format!("The image could not be read: {e}")))?;
        let (data_url, _, _) = tauri::async_runtime::spawn_blocking(move || downscale_jpeg(&png))
            .await
            .map_err(|e| fail(e.to_string()))?
            .map_err(fail)?;
        pictures.push(Picture {
            data_url,
            seed: d.seed,
        });
    }
    if pictures.is_empty() {
        return Err(fail("The image model returned no image."));
    }
    let first = pictures.remove(0);
    Ok(Generated {
        data_url: first.data_url,
        seed: first.seed.or(call.seed),
        prompt: call.prompt,
        size: call.size,
        negative_prompt: call.negative_prompt,
        steps: call.steps,
        reference: used,
        more: pictures,
        note,
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
    call: ToolCall,
) -> Result<Generated, ErrorPayload> {
    generate(&hypnos, &call, None).await
}

/// Writes a `data:` URL's bytes to the path chosen in the save dialog.
#[tauri::command]
pub async fn save_data_url(path: String, data_url: String) -> Result<u64, ErrorPayload> {
    let bytes = data_url_bytes(&data_url).ok_or_else(|| fail("Not a base64 data URL."))?;
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
    fn calls_take_settings_leniently_and_are_clamped() {
        let call: ToolCall = serde_json::from_str(
            r#"{"prompt": " a fox ", "size": "1000X700", "negative_prompt": " ", "n": "9",
                "steps": 200, "seed": "42", "reference": "Previous"}"#,
        )
        .unwrap();
        let call = call.normalized();
        assert_eq!(call.prompt, "a fox");
        assert_eq!(call.size, "992x688");
        assert_eq!(call.negative_prompt, None);
        assert_eq!(call.n, Some(4));
        assert_eq!(call.steps, Some(100));
        assert_eq!(call.seed, Some(42));
        assert_eq!(call.reference, Reference::Previous);

        let call: ToolCall = serde_json::from_str(
            r#"{"prompt": "a cat", "size": "wide", "n": 1, "seed": null, "reference": "both"}"#,
        )
        .unwrap();
        let call = call.normalized();
        assert_eq!(call.size, DEFAULT_SIZE);
        assert_eq!(
            (call.n, call.seed, call.reference),
            (None, None, Reference::None)
        );
        // The webview sends camelCase.
        let call: ToolCall =
            serde_json::from_str(r#"{"prompt": "x", "negativePrompt": "no text"}"#).unwrap();
        assert_eq!(call.negative_prompt.as_deref(), Some("no text"));
    }

    #[test]
    fn the_prompt_convention_carries_the_new_settings() {
        let call = parse_call(
            "```generate_image\n{\"prompt\": \"the same fox at night\", \"size\": \"768x1344\", \"seed\": 7, \"reference\": \"previous\"}\n```",
        )
        .unwrap();
        assert_eq!(call.seed, Some(7));
        assert_eq!(call.reference, Reference::Previous);
        assert!(TOOL_INSTRUCTION.contains("\"seed\""));
        let spec = serde_json::to_value(image_tool_spec()).unwrap();
        let props = &spec["function"]["parameters"]["properties"];
        for key in [
            "prompt",
            "size",
            "negative_prompt",
            "n",
            "steps",
            "seed",
            "reference",
        ] {
            assert!(props.get(key).is_some(), "{key}");
        }
    }

    #[test]
    fn settings_go_back_to_the_chat_model() {
        let image = Generated {
            data_url: String::new(),
            prompt: "a fox".into(),
            size: "1344x768".into(),
            model: "flux3-image-klein".into(),
            seconds: 1.0,
            telemetry: None,
            seed: Some(7),
            negative_prompt: Some("no text".into()),
            steps: None,
            reference: Reference::Previous,
            more: vec![Picture {
                data_url: String::new(),
                seed: Some(8),
            }],
            note: None,
        };
        assert_eq!(
            image.settings_text(),
            "prompt \"a fox\", size 1344x768, seed 7, negative prompt \"no text\", edited from the previous image, 2 images, model flux3-image-klein"
        );
    }

    #[test]
    fn a_short_answer_that_looks_like_the_fence_start_is_kept_until_the_end() {
        let (shown, gate) = run(&["``"]);
        assert_eq!(shown, "");
        assert_eq!(gate.finish().unwrap_err(), "``");
    }

    /// DESK-22 on a running server with a tool-calling chat model and an image model that edits
    /// (FLUX.2 klein / FLUX.3): the settings come from the words, the seed comes back, and "the
    /// same at night" edits the previous image. Several minutes:
    /// `HYPNOS_TEST_URL=... cargo test settings_from_the_conversation -- --ignored --nocapture`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn settings_from_the_conversation_and_an_edit_of_the_previous_image() {
        use crate::hypnos::{hypnos_chat, hypnos_connect, ImageNote, InMessage};
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
        let turn = |history: Vec<InMessage>, id: &'static str| {
            let app = app.handle().clone();
            let model = model.clone();
            async move {
                let events = Arc::new(Mutex::new(Vec::<serde_json::Value>::new()));
                let e = events.clone();
                let channel = tauri::ipc::Channel::new(move |body| {
                    if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                        e.lock().unwrap().push(serde_json::from_str(&s).unwrap());
                    }
                    Ok(())
                });
                hypnos_chat(
                    app.clone(),
                    app.state(),
                    id.into(),
                    model,
                    history,
                    Some(true),
                    None,
                    channel,
                )
                .await
                .unwrap();
                let events = events.lock().unwrap().clone();
                events
            }
        };
        let user = |text: &str| InMessage {
            role: "user".into(),
            content: text.into(),
            attachments: vec![],
            image_prompt: None,
            image: None,
        };

        let first = user("Make me a wide landscape banner of a lighthouse on a cliff at sunset, with no text in it.");
        let events = turn(vec![first.clone()], "desk22a").await;
        let call = events
            .iter()
            .find(|e| e["event"] == "toolCall")
            .expect("a tool call");
        let image = events
            .iter()
            .find(|e| e["event"] == "image")
            .expect("an image");
        eprintln!(
            "call 1: {call}\nsettings 1: {} / seed {} / {}",
            image["size"], image["seed"], image["negativePrompt"]
        );
        let (w, h) = parse_size(image["size"].as_str().unwrap()).unwrap();
        assert!(w > h, "a wide banner: {}", image["size"]);
        let seed = image["seed"].as_u64().expect("the server returns the seed");

        let made = InMessage {
            role: "assistant".into(),
            content: String::new(),
            attachments: vec![],
            image_prompt: Some(image["prompt"].as_str().unwrap().into()),
            image: Some(ImageNote {
                prompt: image["prompt"].as_str().unwrap().into(),
                size: image["size"].as_str().map(String::from),
                seed: Some(seed),
                model: image["model"].as_str().map(String::from),
                data_url: image["dataUrl"].as_str().map(String::from),
                ..ImageNote::default()
            }),
        };
        let events = turn(
            vec![
                first,
                made,
                user("Now the same image, but at night under a full moon."),
            ],
            "desk22b",
        )
        .await;
        let call = events
            .iter()
            .find(|e| e["event"] == "toolCall")
            .expect("a second tool call");
        let image = events
            .iter()
            .find(|e| e["event"] == "image")
            .expect("a second image");
        eprintln!(
            "call 2: {call}\nsettings 2: size {} / seed {} / reference {} / note {}",
            image["size"], image["seed"], image["reference"], image["note"]
        );
        assert!(
            call["reference"] == "previous" || call["seed"].as_u64() == Some(seed),
            "the follow-up keeps the image: {call}"
        );
        let dir = std::env::var("DESK22_OUT").ok();
        if let Some(dir) = dir {
            for (n, e) in events.iter().filter(|e| e["event"] == "image").enumerate() {
                let bytes = data_url_bytes(e["dataUrl"].as_str().unwrap()).unwrap();
                std::fs::write(format!("{dir}/desk22-{n}.jpg"), bytes).unwrap();
            }
        }
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
            image: None,
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
