//! `POST /v1/documents/convert` (Hypnos extension): documents to Markdown with a `docling` model.
//!
//! The document comes as a multipart upload (`file` part) or a JSON body (`file` as base64 / data
//! URI, or `url`). Pages are converted one at a time; with `stream: true` each one is a
//! server-sent event as soon as it is ready, then a final `document` event carries the joined
//! Markdown and the telemetry. `describe_pictures` names a vision model that writes a description
//! under every picture.

use super::{finish_telemetry, with_log, ApiError, AppState};
use crate::request_log::{LogDetails, RequestId};
use axum::{
    body::Body,
    extract::{Multipart, State},
    http::{header, HeaderMap},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Extension, Json,
};
use futures::stream;
use hypnos_backends::document::join_pages;
use hypnos_backends::media::load_document;
use hypnos_core::{
    config::Modality,
    error::HypnosError,
    openai::{
        ChatCompletionRequest, ChatMessage, ChatMessageContent, ContentPart,
        DocumentConversionRequest, DocumentConversionResponse, DocumentPage, DocumentStreamEvent,
        DocumentUsage, EngineTiming, ExecutionTelemetry, ImageUrlSpec,
    },
    request::{CancelOnDrop, RequestContext},
};
use hypnos_scheduler::ActiveModelLease;
use serde_json::json;
use std::{collections::VecDeque, convert::Infallible, time::Instant};
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::warn;

const PICTURE_PROMPT: &str = "This picture was extracted from a document. Describe what it shows: the kind of visual (photo, chart, diagram, map, logo, ...), its subject, any visible text, labels or numbers, and the key takeaway. Be factual and concise (2-4 sentences); do not speculate beyond what is visible.";
const PLACEHOLDER: &str = "<!-- image -->";

pub(crate) async fn convert_document(
    State(state): State<AppState>,
    Extension(rid): Extension<RequestId>,
    headers: HeaderMap,
    body: Body,
) -> Response {
    let mut details = LogDetails::default();
    match convert_document_inner(&state, &rid.0, &headers, body, &mut details).await {
        Ok(resp) => with_log(resp, details),
        Err(e) => {
            let message = e.0.to_string();
            with_log(e.into_response(), details.with_error(message))
        }
    }
}

fn bad(e: impl std::fmt::Display) -> HypnosError {
    HypnosError::InvalidRequest(e.to_string())
}

fn parse_bool(field: &str, value: &str) -> Result<bool, HypnosError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        other => Err(bad(format!("{field} must be true or false, got {other:?}"))),
    }
}

/// Reads a multipart form or a JSON body into a request with `file_bytes` filled.
async fn parse_request(
    state: &AppState,
    headers: &HeaderMap,
    body: Body,
) -> Result<DocumentConversionRequest, HypnosError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let max = state.limits.max_document_bytes;
    let mut req = if content_type.starts_with("multipart/form-data") {
        let request = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .map_err(bad)?;
        use axum::extract::FromRequest;
        let mut multipart = Multipart::from_request(request, state).await.map_err(bad)?;
        let mut req = DocumentConversionRequest {
            ocr: true,
            output_format: "markdown".into(),
            ..Default::default()
        };
        while let Some(field) = multipart.next_field().await.map_err(bad)? {
            let name = field.name().unwrap_or("").to_string();
            if name == "file" {
                if req.filename.is_none() {
                    req.filename = field.file_name().map(str::to_string);
                }
                let bytes = field.bytes().await.map_err(bad)?;
                if bytes.len() as u64 > max {
                    return Err(bad(format!(
                        "document is {} bytes, above limits.max_document_bytes ({max})",
                        bytes.len()
                    )));
                }
                req.file_bytes = bytes.to_vec();
                continue;
            }
            let value = field.text().await.map_err(bad)?;
            match name.as_str() {
                "model" => req.model = value,
                "url" => req.url = Some(value),
                "filename" => req.filename = Some(value),
                "pages" => req.pages = Some(value),
                "ocr" => req.ocr = parse_bool("ocr", &value)?,
                "output_format" => req.output_format = value,
                "stream" => req.stream = parse_bool("stream", &value)?,
                "describe_pictures" => {
                    req.describe_pictures = Some(value).filter(|v| !v.trim().is_empty())
                }
                _ => {}
            }
        }
        req
    } else {
        let limit = usize::try_from(max.saturating_mul(2).max(1024 * 1024)).unwrap_or(usize::MAX);
        let bytes = axum::body::to_bytes(body, limit).await.map_err(bad)?;
        serde_json::from_slice::<DocumentConversionRequest>(&bytes).map_err(bad)?
    };
    if req.model.is_empty() {
        return Err(bad("missing 'model'"));
    }
    if !matches!(
        req.output_format.to_ascii_lowercase().as_str(),
        "markdown" | "json"
    ) {
        return Err(bad("output_format must be \"markdown\" or \"json\""));
    }
    if req.file_bytes.is_empty() {
        let (file, url, limits) = (req.file.take(), req.url.clone(), state.limits.clone());
        req.file_bytes = tokio::task::spawn_blocking(move || {
            load_document(file.as_deref(), url.as_deref(), &limits)
        })
        .await
        .map_err(|e| HypnosError::BackendError(e.to_string()))??;
        if req.filename.is_none() {
            // The URL's last path segment usually carries the extension.
            req.filename = req.url.as_deref().and_then(|u| {
                let path = u.split(['?', '#']).next().unwrap_or(u);
                path.rsplit('/')
                    .next()
                    .filter(|s| s.contains('.'))
                    .map(str::to_string)
            });
        }
    }
    Ok(req)
}

/// Asks the vision model for a description of each of the page's pictures and puts it under the
/// picture's placeholder. A failed description is logged and skipped: the page still converts.
async fn describe_pictures(
    state: &AppState,
    model: &str,
    page: &mut DocumentPage,
    ctx: &RequestContext,
) -> u32 {
    let images = std::mem::take(&mut page.picture_images);
    if images.is_empty() {
        return 0;
    }
    let lease = match state.scheduler.acquire_model(model, None, 1, ctx).await {
        Ok(lease) => lease,
        Err(e) => {
            warn!(model, page = page.page, error = %e, "picture descriptions skipped");
            return 0;
        }
    };
    let mut descriptions = Vec::with_capacity(images.len());
    for uri in images {
        let chat = ChatCompletionRequest {
            model: model.to_string(),
            messages: vec![ChatMessage {
                role: "user".into(),
                content: ChatMessageContent::Parts(vec![
                    ContentPart::ImageUrl {
                        image_url: ImageUrlSpec {
                            url: uri,
                            detail: None,
                        },
                    },
                    ContentPart::Text {
                        text: PICTURE_PROMPT.into(),
                    },
                ]),
            }],
            max_tokens: Some(300),
            temperature: Some(0.2),
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            num_ctx: None,
            think: Some(false),
        };
        match lease.backend.chat_completion(&chat, ctx).await {
            Ok(resp) => descriptions.push(
                resp.choices
                    .first()
                    .map(|c| c.message.content.trim().replace('\n', " "))
                    .unwrap_or_default(),
            ),
            Err(e) => {
                warn!(model, page = page.page, error = %e, "picture description failed");
                descriptions.push(String::new());
            }
        }
    }
    drop(lease);
    let (markdown, described) = insert_descriptions(&page.markdown, &descriptions);
    page.markdown = markdown;
    described
}

/// Puts `> **Picture:** ...` under the n-th `<!-- image -->` placeholder for the n-th
/// non-empty description. Returns the new Markdown and how many descriptions went in.
fn insert_descriptions(markdown: &str, descriptions: &[String]) -> (String, u32) {
    let mut out = String::with_capacity(markdown.len() + descriptions.len() * 200);
    let mut rest = markdown;
    let mut described = 0;
    for description in descriptions {
        let Some(at) = rest.find(PLACEHOLDER) else {
            break;
        };
        let end = at + PLACEHOLDER.len();
        out.push_str(&rest[..end]);
        if !description.is_empty() {
            out.push_str("\n\n> **Picture:** ");
            out.push_str(description);
            described += 1;
        }
        rest = &rest[end..];
    }
    out.push_str(rest);
    (out, described)
}

/// What a conversion has produced so far; shared by the JSON and the streaming responses.
struct Progress {
    model: String,
    filename: Option<String>,
    describe: Option<String>,
    page_count: u32,
    pages: Vec<DocumentPage>,
    pictures_described: u32,
}

impl Progress {
    fn new(req: &DocumentConversionRequest) -> Self {
        Self {
            model: req.model.clone(),
            filename: req.filename.clone(),
            describe: req.describe_pictures.clone(),
            page_count: 0,
            pages: Vec::new(),
            pictures_described: 0,
        }
    }

    async fn page(
        &mut self,
        state: &AppState,
        ctx: &RequestContext,
        mut page: DocumentPage,
    ) -> DocumentPage {
        if let Some(model) = self.describe.clone() {
            self.pictures_described += describe_pictures(state, &model, &mut page, ctx).await;
        }
        self.pages.push(page.clone());
        page
    }

    fn usage(&self) -> DocumentUsage {
        DocumentUsage {
            pages: self.pages.len() as u32,
            pictures: self.pages.iter().map(|p| p.pictures).sum(),
            tables: self.pages.iter().map(|p| p.tables).sum(),
            pictures_described: self.describe.as_ref().map(|_| self.pictures_described),
        }
    }

    fn finish(self, telemetry: ExecutionTelemetry) -> DocumentConversionResponse {
        DocumentConversionResponse {
            object: "document".into(),
            markdown: join_pages(&self.pages),
            usage: self.usage(),
            model: self.model,
            filename: self.filename,
            page_count: self.page_count,
            pages: self.pages,
            hypnos_telemetry: Some(telemetry),
        }
    }
}

async fn convert_document_inner(
    state: &AppState,
    request_id: &str,
    headers: &HeaderMap,
    body: Body,
    details: &mut LogDetails,
) -> Result<Response, ApiError> {
    let req = parse_request(state, headers, body).await?;
    details.model = Some(req.model.clone());
    details.stream = req.stream;
    if state.request_log.capture_text() {
        details.input = Some(format!(
            "{} ({} bytes){}{}",
            req.filename.as_deref().unwrap_or("document"),
            req.file_bytes.len(),
            req.pages
                .as_deref()
                .map(|p| format!(", pages {p}"))
                .unwrap_or_default(),
            req.describe_pictures
                .as_deref()
                .map(|m| format!(", pictures described by {m}"))
                .unwrap_or_default()
        ));
    }
    if state.request_log.capture_payloads() {
        details.request_payload = Some(json!({
            "model": req.model,
            "filename": req.filename,
            "bytes": req.file_bytes.len(),
            "url": req.url,
            "pages": req.pages,
            "ocr": req.ocr,
            "output_format": req.output_format,
            "stream": req.stream,
            "describe_pictures": req.describe_pictures,
        }));
    }
    if let Some(vision) = &req.describe_pictures {
        let backend = state
            .scheduler
            .backend(vision)
            .ok_or_else(|| HypnosError::ModelNotFound(vision.clone()))?;
        if !backend.config().supports_modality(Modality::Vision) {
            return Err(bad(format!(
                "describe_pictures needs a vision model; '{vision}' is not one"
            ))
            .into());
        }
    }

    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, 1, &ctx)
        .await?;
    let infer_start = Instant::now();

    let (tx, mut rx) = mpsc::channel::<DocumentStreamEvent>(8);
    // The task's own result carries the error with its type (a bad page range is a 400); the
    // channel closes when it ends.
    let mut task = {
        let backend = lease.backend.clone();
        let req = req.clone();
        let ctx = ctx.clone();
        Some(tokio::spawn(async move {
            backend.convert_document(&req, &ctx, tx).await
        }))
    };
    let progress = Progress::new(&req);

    if req.stream {
        // Wait for the document to be read before answering, so a request the backend refuses
        // gets its HTTP status rather than an error event in a 200 stream.
        let first = match rx.recv().await {
            Some(ev) => ev,
            None => return Err(task_error(task.take()).await.into()),
        };
        let initial = DocumentSse {
            rx,
            pending: Some(first),
            task,
            state: state.clone(),
            ctx,
            lease,
            _cancel_guard: guard,
            request_id: request_id.to_string(),
            infer_start,
            progress: Some(progress),
            queued: VecDeque::new(),
            finished: false,
        };
        let events = stream::unfold(initial, |mut st| async move {
            if let Some(ev) = st.queued.pop_front() {
                return Some((Ok::<Event, Infallible>(ev), st));
            }
            if st.finished {
                return None;
            }
            let ev = st.next_event().await;
            Some((Ok(ev), st))
        });
        return Ok(Sse::new(events)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let mut progress = progress;
    let timing = loop {
        match rx.recv().await {
            Some(DocumentStreamEvent::Started { page_count, .. }) => {
                progress.page_count = page_count
            }
            Some(DocumentStreamEvent::Page(page)) => {
                progress.page(state, &ctx, page).await;
            }
            Some(DocumentStreamEvent::Done { timing }) => break timing,
            Some(DocumentStreamEvent::Error(message)) => {
                return Err(HypnosError::BackendError(message).into())
            }
            None => return Err(task_error(task.take()).await.into()),
        }
    };
    let telemetry = finish_telemetry(&lease, infer_start, None).with_engine(timing);
    state.metrics.record_document(&req.model, &telemetry.engine);
    guard.disarm();
    let resp = progress.finish(telemetry);
    details.telemetry = resp.hypnos_telemetry.clone();
    if state.request_log.capture_text() {
        details.output = Some(resp.markdown.clone());
    }
    if state.request_log.capture_payloads() {
        details.response_payload = Some(json!({
            "page_count": resp.page_count,
            "usage": resp.usage,
            "hypnos_telemetry": resp.hypnos_telemetry,
        }));
    }
    Ok(Json(resp).into_response())
}

/// Why the conversion task stopped before `Done`.
async fn task_error(task: Option<JoinHandle<hypnos_core::error::Result<()>>>) -> HypnosError {
    match task {
        Some(task) => match task.await {
            Ok(Err(e)) => e,
            Ok(Ok(())) => HypnosError::BackendError("conversion ended early".into()),
            Err(e) => HypnosError::BackendError(format!("conversion task: {e}")),
        },
        None => HypnosError::BackendError("conversion ended early".into()),
    }
}

/// Streaming state: owning the lease keeps the model loaded until the final event, and dropping
/// the guard (client gone) cancels the conversion at the next page.
struct DocumentSse {
    rx: mpsc::Receiver<DocumentStreamEvent>,
    /// The first event, received before the response was committed.
    pending: Option<DocumentStreamEvent>,
    task: Option<JoinHandle<hypnos_core::error::Result<()>>>,
    state: AppState,
    ctx: RequestContext,
    lease: ActiveModelLease,
    _cancel_guard: CancelOnDrop,
    request_id: String,
    infer_start: Instant,
    progress: Option<Progress>,
    queued: VecDeque<Event>,
    finished: bool,
}

impl DocumentSse {
    fn fail(&mut self, error: HypnosError) -> Event {
        self.finished = true;
        let message = error.to_string();
        let kind = match error {
            HypnosError::InvalidRequest(_) => "invalid_request_error",
            HypnosError::Cancelled | HypnosError::Timeout(_) => "cancelled",
            _ => "server_error",
        };
        self.state
            .request_log
            .update(&self.request_id, |r, _| r.error = Some(message.clone()));
        self.queued.push_back(Event::default().data("[DONE]"));
        Event::default().data(
            json!({ "object": "error", "error": { "message": message, "type": kind } }).to_string(),
        )
    }

    fn done(&mut self, timing: EngineTiming) -> Event {
        self.finished = true;
        let model = self.lease.model_id().to_string();
        let infer_ms = self.infer_start.elapsed().as_secs_f64() * 1000.0;
        let telemetry = self.lease.build_telemetry(infer_ms).with_engine(timing);
        self.state
            .metrics
            .record_document(&model, &telemetry.engine);
        let mut resp = self
            .progress
            .take()
            .expect("progress is taken once")
            .finish(telemetry);
        let capture_text = self.state.request_log.capture_text();
        let log = LogDetails {
            telemetry: resp.hypnos_telemetry.clone(),
            output: capture_text.then(|| resp.markdown.clone()),
            ..Default::default()
        };
        self.state
            .request_log
            .update(&self.request_id, |r, level| r.apply_details(log, level));
        // The pages already went out one by one.
        resp.pages.clear();
        self.queued.push_back(Event::default().data("[DONE]"));
        Event::default().data(serde_json::to_string(&resp).unwrap_or_default())
    }

    async fn next_event(&mut self) -> Event {
        let event = match self.pending.take() {
            Some(ev) => Some(ev),
            None => self.rx.recv().await,
        };
        match event {
            Some(DocumentStreamEvent::Started { page_count, pages }) => {
                if let Some(p) = self.progress.as_mut() {
                    p.page_count = page_count;
                }
                Event::default().data(
                    json!({ "object": "document.started", "page_count": page_count, "pages": pages })
                        .to_string(),
                )
            }
            Some(DocumentStreamEvent::Page(page)) => {
                let progress = self.progress.as_mut().expect("progress until done");
                let page = progress.page(&self.state, &self.ctx, page).await;
                let mut body = serde_json::Map::new();
                body.insert("object".into(), json!("document.page"));
                if let Ok(serde_json::Value::Object(fields)) = serde_json::to_value(&page) {
                    body.extend(fields);
                }
                Event::default().data(serde_json::Value::Object(body).to_string())
            }
            Some(DocumentStreamEvent::Done { timing }) => self.done(timing),
            Some(DocumentStreamEvent::Error(message)) => {
                self.fail(HypnosError::BackendError(message))
            }
            None => {
                let error = task_error(self.task.take()).await;
                self.fail(error)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptions_go_under_their_placeholders() {
        let md = "Intro\n\n<!-- image -->\n\nText\n\n<!-- image -->\n\nEnd";
        let (out, n) = insert_descriptions(md, &["A chart.".into(), "".into()]);
        assert_eq!(n, 1);
        assert_eq!(
            out,
            "Intro\n\n<!-- image -->\n\n> **Picture:** A chart.\n\nText\n\n<!-- image -->\n\nEnd"
        );
        // More descriptions than placeholders: the extra ones are dropped.
        let (out, n) = insert_descriptions("none here", &["x".into()]);
        assert_eq!((out.as_str(), n), ("none here", 0));
    }
}
