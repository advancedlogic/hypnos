//! Streaming responses: chat (SSE), document conversion (SSE) and speech (chunked audio).
//!
//! Each stream owns the HTTP response. Dropping it closes the connection, which is how the server
//! is told to stop at the next token, page or audio chunk (`docs/api.md` section 17,
//! "Cancellation"); it also returns the per-model concurrency permit. A stream that fails after it
//! started yields [`Error::Stream`].

use std::collections::VecDeque;
use std::pin::Pin;
use std::task::{Context, Poll};

use bytes::Bytes;
use futures::{future, Stream, StreamExt};
use serde_json::Value;
use tokio::sync::OwnedSemaphorePermit;

use crate::error::{ApiError, Error, Result};
use crate::types::{
    ChatChoice, ChatCompletion, ChatCompletionChunk, Document, DocumentEvent, DocumentPage,
    DocumentStarted, ResponseMessage, SpeechTelemetry, Telemetry, Usage,
};

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// The open body of a response plus the permit that bounds the model's concurrency.
pub(crate) struct Body {
    bytes: Option<ByteStream>,
    _permit: Option<OwnedSemaphorePermit>,
}

impl Body {
    pub(crate) fn new(response: reqwest::Response, permit: Option<OwnedSemaphorePermit>) -> Self {
        Self {
            bytes: Some(Box::pin(response.bytes_stream())),
            _permit: permit,
        }
    }

    /// Drops the connection and the permit.
    fn close(&mut self) {
        self.bytes = None;
        self._permit = None;
    }
}

/// Decodes a `text/event-stream` body into each event's `data` (lines joined with `\n`).
/// `:` comments and other fields are ignored; `[DONE]` comes back as is.
#[derive(Default)]
pub(crate) struct SseDecoder {
    buffer: Vec<u8>,
    data: Vec<String>,
    events: VecDeque<String>,
}

impl SseDecoder {
    pub(crate) fn push(&mut self, chunk: &[u8]) {
        self.buffer.extend_from_slice(chunk);
        while let Some(pos) = self.buffer.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buffer.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line[..line.len() - 1]).into_owned();
            self.line(line.trim_end_matches('\r'));
        }
    }

    pub(crate) fn finish(&mut self) {
        if !self.buffer.is_empty() {
            let rest = String::from_utf8_lossy(&std::mem::take(&mut self.buffer)).into_owned();
            self.line(rest.trim_end_matches('\r'));
        }
        self.line("");
    }

    fn line(&mut self, line: &str) {
        if line.is_empty() {
            if !self.data.is_empty() {
                self.events.push_back(self.data.join("\n"));
                self.data.clear();
            }
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = line.split_once(':').unwrap_or((line, ""));
        if field == "data" {
            self.data
                .push(value.strip_prefix(' ').unwrap_or(value).to_string());
        }
    }

    pub(crate) fn next(&mut self) -> Option<String> {
        self.events.pop_front()
    }
}

/// Reads SSE events from a body.
struct SseReader {
    body: Body,
    decoder: SseDecoder,
    ended: bool,
}

impl SseReader {
    fn poll_event(&mut self, cx: &mut Context<'_>) -> Poll<Option<Result<String>>> {
        loop {
            if let Some(event) = self.decoder.next() {
                return Poll::Ready(Some(Ok(event)));
            }
            if self.ended {
                return Poll::Ready(None);
            }
            let Some(bytes) = self.body.bytes.as_mut() else {
                self.ended = true;
                continue;
            };
            match bytes.poll_next_unpin(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(chunk))) => self.decoder.push(&chunk),
                Poll::Ready(Some(Err(e))) => {
                    self.ended = true;
                    self.body.close();
                    return Poll::Ready(Some(Err(Error::Connection(format!(
                        "the stream broke: {e}"
                    )))));
                }
                Poll::Ready(None) => {
                    self.ended = true;
                    self.decoder.finish();
                }
            }
        }
    }
}

fn stream_error(payload: &Value, request_id: Option<&str>) -> Error {
    let err = payload.get("error").cloned().unwrap_or(Value::Null);
    Error::Stream(Box::new(ApiError {
        status: 200,
        message: err
            .get("message")
            .and_then(Value::as_str)
            .unwrap_or("the stream failed")
            .to_string(),
        kind: Some(
            err.get("type")
                .and_then(Value::as_str)
                .unwrap_or("server_error")
                .to_string(),
        ),
        code: None,
        request_id: request_id.map(str::to_string),
        retry_after: None,
        body: Some(payload.clone()),
    }))
}

fn decode<T: serde::de::DeserializeOwned>(data: &str) -> Result<T> {
    serde_json::from_str(data).map_err(|e| Error::Decode(format!("{e}: {data}")))
}

// --- chat ----------------------------------------------------------------------------------

/// `POST /v1/chat/completions` with `stream: true`.
///
/// A [`Stream`] of [`ChatCompletionChunk`]s; [`ChatStream::text`] yields only the content
/// pieces. `text_so_far`, `usage`, `telemetry` and `finish_reason` fill in as chunks arrive.
pub struct ChatStream {
    reader: SseReader,
    request_id: Option<String>,
    id: String,
    model: String,
    created: i64,
    text: String,
    finish_reason: Option<String>,
    usage: Option<Usage>,
    telemetry: Option<Telemetry>,
    done: bool,
}

impl ChatStream {
    pub(crate) fn new(body: Body, request_id: Option<String>) -> Self {
        Self {
            reader: SseReader {
                body,
                decoder: SseDecoder::default(),
                ended: false,
            },
            request_id,
            id: String::new(),
            model: String::new(),
            created: 0,
            text: String::new(),
            finish_reason: None,
            usage: None,
            telemetry: None,
            done: false,
        }
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }
    pub fn text_so_far(&self) -> &str {
        &self.text
    }
    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }
    pub fn telemetry(&self) -> Option<&Telemetry> {
        self.telemetry.as_ref()
    }
    pub fn finish_reason(&self) -> Option<&str> {
        self.finish_reason.as_deref()
    }

    /// The `delta.content` pieces, as they arrive.
    pub fn text(&mut self) -> impl Stream<Item = Result<String>> + Unpin + '_ {
        self.filter_map(|chunk| {
            future::ready(match chunk {
                Err(e) => Some(Err(e)),
                Ok(chunk) => {
                    let piece: String = chunk
                        .choices
                        .iter()
                        .filter_map(|c| c.delta.content.as_deref())
                        .collect();
                    (!piece.is_empty()).then_some(Ok(piece))
                }
            })
        })
    }

    /// Reads the rest of the stream and returns it as a non-streaming completion.
    pub async fn final_completion(&mut self) -> Result<ChatCompletion> {
        while let Some(chunk) = self.next().await {
            chunk?;
        }
        Ok(ChatCompletion {
            id: self.id.clone(),
            object: "chat.completion".into(),
            created: self.created,
            model: self.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ResponseMessage {
                    role: "assistant".into(),
                    content: self.text.clone(),
                },
                finish_reason: self.finish_reason.clone().unwrap_or_else(|| "stop".into()),
            }],
            usage: self.usage.clone().unwrap_or_default(),
            hypnos_telemetry: self.telemetry.clone(),
            extra: Default::default(),
        })
    }

    fn add(&mut self, data: &str) -> Result<Option<ChatCompletionChunk>> {
        if data == "[DONE]" {
            self.done = true;
            return Ok(None);
        }
        let raw: Value = decode(data)?;
        let chunk: ChatCompletionChunk = decode(data)?;
        if !chunk.id.is_empty() {
            self.id = chunk.id.clone();
        }
        if !chunk.model.is_empty() {
            self.model = chunk.model.clone();
        }
        if chunk.created != 0 {
            self.created = chunk.created;
        }
        for choice in &chunk.choices {
            if let Some(piece) = &choice.delta.content {
                self.text.push_str(piece);
            }
            if let Some(reason) = &choice.finish_reason {
                self.finish_reason = Some(reason.clone());
            }
        }
        if let Some(usage) = &chunk.usage {
            self.usage = Some(usage.clone());
        }
        if let Some(telemetry) = &chunk.hypnos_telemetry {
            self.telemetry = Some(telemetry.clone());
        }
        if self.finish_reason.as_deref() == Some("error") || chunk.error.is_some() {
            self.done = true;
            return Err(stream_error(&raw, self.request_id.as_deref()));
        }
        Ok(Some(chunk))
    }
}

impl Stream for ChatStream {
    type Item = Result<ChatCompletionChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.done {
                this.reader.body.close();
                return Poll::Ready(None);
            }
            match this.reader.poll_event(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => this.done = true,
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(data))) => match this.add(&data) {
                    Ok(Some(chunk)) => return Poll::Ready(Some(Ok(chunk))),
                    Ok(None) => {}
                    Err(e) => {
                        this.reader.body.close();
                        return Poll::Ready(Some(Err(e)));
                    }
                },
            }
        }
    }
}

// --- documents -----------------------------------------------------------------------------

/// `POST /v1/documents/convert` with `stream: true`.
///
/// A [`Stream`] of [`DocumentEvent`]s; the last one ([`DocumentEvent::Done`]) carries the whole
/// document with its pages put back (the server sends them only as page events).
pub struct DocumentStream {
    reader: SseReader,
    request_id: Option<String>,
    started: Option<DocumentStarted>,
    pages: Vec<DocumentPage>,
    document: Option<Document>,
    done: bool,
}

impl DocumentStream {
    pub(crate) fn new(body: Body, request_id: Option<String>) -> Self {
        Self {
            reader: SseReader {
                body,
                decoder: SseDecoder::default(),
                ended: false,
            },
            request_id,
            started: None,
            pages: Vec::new(),
            document: None,
            done: false,
        }
    }

    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// The `document.started` event, once it arrived.
    pub fn started(&self) -> Option<&DocumentStarted> {
        self.started.as_ref()
    }

    /// Only the pages, as they are converted.
    pub fn pages(&mut self) -> impl Stream<Item = Result<DocumentPage>> + Unpin + '_ {
        self.filter_map(|event| {
            future::ready(match event {
                Err(e) => Some(Err(e)),
                Ok(DocumentEvent::Page(page)) => Some(Ok(page)),
                Ok(_) => None,
            })
        })
    }

    /// Reads the rest of the stream and returns the document with all its pages.
    pub async fn final_document(&mut self) -> Result<Document> {
        while let Some(event) = self.next().await {
            event?;
        }
        self.document.clone().ok_or_else(|| {
            Error::Stream(Box::new(ApiError {
                status: 200,
                message: "the document stream ended without its final event".into(),
                request_id: self.request_id.clone(),
                ..Default::default()
            }))
        })
    }

    fn add(&mut self, data: &str) -> Result<Option<DocumentEvent>> {
        if data == "[DONE]" {
            self.done = true;
            return Ok(None);
        }
        let event: Value = decode(data)?;
        match event.get("object").and_then(Value::as_str) {
            Some("document.started") => {
                let started: DocumentStarted = decode(data)?;
                self.started = Some(started.clone());
                Ok(Some(DocumentEvent::Started(started)))
            }
            Some("document.page") => {
                let page: DocumentPage = decode(data)?;
                self.pages.push(page.clone());
                Ok(Some(DocumentEvent::Page(page)))
            }
            Some("document") => {
                let mut doc: Document = decode(data)?;
                if doc.pages.is_empty() {
                    doc.pages = self.pages.clone();
                }
                self.document = Some(doc.clone());
                Ok(Some(DocumentEvent::Done(Box::new(doc))))
            }
            Some("error") => {
                self.done = true;
                Err(stream_error(&event, self.request_id.as_deref()))
            }
            _ => Ok(None),
        }
    }
}

impl Stream for DocumentStream {
    type Item = Result<DocumentEvent>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            if this.done {
                this.reader.body.close();
                return Poll::Ready(None);
            }
            match this.reader.poll_event(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => this.done = true,
                Poll::Ready(Some(Err(e))) => {
                    this.done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(Some(Ok(data))) => match this.add(&data) {
                    Ok(Some(event)) => return Poll::Ready(Some(Ok(event))),
                    Ok(None) => {}
                    Err(e) => {
                        this.reader.body.close();
                        return Poll::Ready(Some(Err(e)));
                    }
                },
            }
        }
    }
}

// --- speech --------------------------------------------------------------------------------

/// `POST /v1/audio/speech` with `stream: true`: the audio as it is synthesized.
///
/// For `wav` the first chunk starts with a header that declares an open-ended length; for `pcm`
/// the chunks are raw 16-bit little-endian mono samples at 24 kHz.
pub struct SpeechStream {
    body: Body,
    content_type: String,
    telemetry: SpeechTelemetry,
    request_id: Option<String>,
}

impl SpeechStream {
    pub(crate) fn new(response: reqwest::Response, permit: Option<OwnedSemaphorePermit>) -> Self {
        let headers = response.headers();
        let content_type = headers
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or_default()
            .to_string();
        let telemetry = SpeechTelemetry::from_headers(headers);
        let request_id = headers
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        Self {
            body: Body::new(response, permit),
            content_type,
            telemetry,
            request_id,
        }
    }

    pub fn content_type(&self) -> &str {
        &self.content_type
    }
    pub fn telemetry(&self) -> &SpeechTelemetry {
        &self.telemetry
    }
    pub fn request_id(&self) -> Option<&str> {
        self.request_id.as_deref()
    }

    /// Reads the rest of the audio.
    pub async fn read(&mut self) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        while let Some(chunk) = self.next().await {
            out.extend_from_slice(&chunk?);
        }
        Ok(out)
    }
}

impl Stream for SpeechStream {
    type Item = Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(bytes) = this.body.bytes.as_mut() else {
            return Poll::Ready(None);
        };
        match bytes.poll_next_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Some(Ok(chunk))) => Poll::Ready(Some(Ok(chunk))),
            Poll::Ready(Some(Err(e))) => {
                this.body.close();
                Poll::Ready(Some(Err(Error::Connection(format!(
                    "the stream broke: {e}"
                )))))
            }
            Poll::Ready(None) => {
                this.body.close();
                Poll::Ready(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SseDecoder;

    #[test]
    fn decodes_events_across_chunks() {
        let mut d = SseDecoder::default();
        d.push(b": keep-alive\n\ndata: {\"a\"");
        assert_eq!(d.next(), None);
        d.push(b":1}\r\n\r\ndata: one\ndata: two\n\nevent: x\ndata: [DONE]\n");
        assert_eq!(d.next().as_deref(), Some("{\"a\":1}"));
        assert_eq!(d.next().as_deref(), Some("one\ntwo"));
        assert_eq!(d.next(), None);
        d.finish();
        assert_eq!(d.next().as_deref(), Some("[DONE]"));
    }
}
