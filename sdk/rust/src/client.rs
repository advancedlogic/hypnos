//! The client: one method per route of `docs/api.md`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, AUTHORIZATION, USER_AGENT};
use reqwest::{Method, RequestBuilder, Response};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Map, Value};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::error::{ApiError, Error, Result};
use crate::retry::RetryPolicy;
use crate::streams::{Body, ChatStream, DocumentStream, SpeechStream};
use crate::types::*;

/// `http://127.0.0.1:9999`.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:9999";
/// Chat, images and documents may pay a model load: keep at or above the server's
/// `request_timeout_secs` (900 s by default).
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(900);

// --- requests -------------------------------------------------------------------------------

/// One text or a batch (`input` of embeddings and entities).
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum Input {
    One(String),
    Many(Vec<String>),
}

impl From<&str> for Input {
    fn from(s: &str) -> Self {
        Self::One(s.to_string())
    }
}
impl From<String> for Input {
    fn from(s: String) -> Self {
        Self::One(s)
    }
}
impl From<Vec<String>> for Input {
    fn from(v: Vec<String>) -> Self {
        Self::Many(v)
    }
}
impl From<Vec<&str>> for Input {
    fn from(v: Vec<&str>) -> Self {
        Self::Many(v.into_iter().map(str::to_string).collect())
    }
}
impl From<&[&str]> for Input {
    fn from(v: &[&str]) -> Self {
        Self::Many(v.iter().map(|s| s.to_string()).collect())
    }
}

/// `POST /v1/chat/completions`. `num_ctx` and `think` are Hypnos extensions; `extra` adds any
/// other body field.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_ctx: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub think: Option<bool>,
    /// Functions the model may call (`docs/api.md` section 7.5).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolSpec>>,
    /// `auto` on the server when unset.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatRequest {
    pub fn new(model: impl Into<String>, messages: Vec<ChatMessage>) -> Self {
        Self {
            model: model.into(),
            messages,
            ..Self::default()
        }
    }
    pub fn max_tokens(mut self, n: u32) -> Self {
        self.max_tokens = Some(n);
        self
    }
    pub fn temperature(mut self, t: f32) -> Self {
        self.temperature = Some(t);
        self
    }
    pub fn top_p(mut self, p: f32) -> Self {
        self.top_p = Some(p);
        self
    }
    pub fn top_k(mut self, k: u32) -> Self {
        self.top_k = Some(k);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn num_ctx(mut self, n: u32) -> Self {
        self.num_ctx = Some(n);
        self
    }
    pub fn think(mut self, on: bool) -> Self {
        self.think = Some(on);
        self
    }
    /// The functions the model may call; send the same tools again with the results.
    pub fn tools(mut self, tools: Vec<ToolSpec>) -> Self {
        self.tools = Some(tools);
        self
    }
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.tool_choice = Some(choice);
        self
    }
    pub fn extra(mut self, key: impl Into<String>, value: Value) -> Self {
        self.extra.insert(key.into(), value);
        self
    }
}

/// `POST /v1/audio/transcriptions` (sent as multipart).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptionRequest {
    pub model: String,
    pub audio: Vec<u8>,
    pub filename: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
}

impl TranscriptionRequest {
    /// WAV, 16-bit PCM, is the reliable format.
    pub fn new(model: impl Into<String>, filename: impl Into<String>, audio: Vec<u8>) -> Self {
        Self {
            model: model.into(),
            audio,
            filename: filename.into(),
            ..Self::default()
        }
    }
    /// Reads the file at `path`.
    pub fn from_path(model: impl Into<String>, path: impl AsRef<std::path::Path>) -> Result<Self> {
        let (name, bytes) = read_path(path.as_ref())?;
        Ok(Self::new(model, name, bytes))
    }
    pub fn language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }
}

/// `POST /v1/audio/speech`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct SpeechRequest {
    pub model: String,
    pub input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice: Option<String>,
    /// `wav` (default) or `pcm`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub speed: Option<f32>,
}

impl SpeechRequest {
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            ..Self::default()
        }
    }
    pub fn voice(mut self, voice: impl Into<String>) -> Self {
        self.voice = Some(voice.into());
        self
    }
    pub fn response_format(mut self, format: impl Into<String>) -> Self {
        self.response_format = Some(format.into());
        self
    }
    pub fn speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed);
        self
    }
}

/// `POST /v1/images/generations`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct ImageRequest {
    pub model: String,
    pub prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    /// `WIDTHxHEIGHT`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negative_prompt: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f32>,
}

impl ImageRequest {
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            ..Self::default()
        }
    }
    pub fn n(mut self, n: u32) -> Self {
        self.n = Some(n);
        self
    }
    pub fn size(mut self, size: impl Into<String>) -> Self {
        self.size = Some(size.into());
        self
    }
    pub fn negative_prompt(mut self, text: impl Into<String>) -> Self {
        self.negative_prompt = Some(text.into());
        self
    }
    pub fn steps(mut self, steps: u32) -> Self {
        self.steps = Some(steps);
        self
    }
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    pub fn cfg_scale(mut self, scale: f32) -> Self {
        self.cfg_scale = Some(scale);
        self
    }
}

/// `POST /v1/entities`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct EntitiesRequest {
    pub model: String,
    pub input: Input,
    pub labels: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub flat: Option<bool>,
}

impl EntitiesRequest {
    pub fn new(model: impl Into<String>, input: impl Into<Input>, labels: &[&str]) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            labels: labels.iter().map(|s| s.to_string()).collect(),
            threshold: None,
            flat: None,
        }
    }
    pub fn threshold(mut self, t: f32) -> Self {
        self.threshold = Some(t);
        self
    }
    pub fn flat(mut self, flat: bool) -> Self {
        self.flat = Some(flat);
        self
    }
}

/// Where a document comes from.
#[derive(Debug, Clone, PartialEq)]
pub enum DocumentSource {
    /// Uploaded as multipart.
    File { filename: String, bytes: Vec<u8> },
    /// Fetched by the server (needs `allow_remote_image_urls`).
    Url(String),
}

/// `POST /v1/documents/convert`.
#[derive(Debug, Clone, PartialEq)]
pub struct DocumentRequest {
    pub model: String,
    pub source: DocumentSource,
    /// Overrides the file name (its extension helps detect the format).
    pub filename: Option<String>,
    /// `"3"` or `"2-5"`.
    pub pages: Option<String>,
    pub ocr: Option<bool>,
    /// `markdown` (default) or `json`.
    pub output_format: Option<String>,
    /// A vision model id that describes every picture.
    pub describe_pictures: Option<String>,
    pub return_pictures: Option<bool>,
}

impl DocumentRequest {
    pub fn file(model: impl Into<String>, filename: impl Into<String>, bytes: Vec<u8>) -> Self {
        Self::with_source(
            model,
            DocumentSource::File {
                filename: filename.into(),
                bytes,
            },
        )
    }
    pub fn from_path(model: impl Into<String>, path: impl AsRef<std::path::Path>) -> Result<Self> {
        let (filename, bytes) = read_path(path.as_ref())?;
        Ok(Self::file(model, filename, bytes))
    }
    pub fn url(model: impl Into<String>, url: impl Into<String>) -> Self {
        Self::with_source(model, DocumentSource::Url(url.into()))
    }
    fn with_source(model: impl Into<String>, source: DocumentSource) -> Self {
        Self {
            model: model.into(),
            source,
            filename: None,
            pages: None,
            ocr: None,
            output_format: None,
            describe_pictures: None,
            return_pictures: None,
        }
    }
    pub fn filename(mut self, name: impl Into<String>) -> Self {
        self.filename = Some(name.into());
        self
    }
    pub fn pages(mut self, pages: impl Into<String>) -> Self {
        self.pages = Some(pages.into());
        self
    }
    pub fn ocr(mut self, on: bool) -> Self {
        self.ocr = Some(on);
        self
    }
    pub fn output_format(mut self, format: impl Into<String>) -> Self {
        self.output_format = Some(format.into());
        self
    }
    pub fn describe_pictures(mut self, vision_model: impl Into<String>) -> Self {
        self.describe_pictures = Some(vision_model.into());
        self
    }
    pub fn return_pictures(mut self, on: bool) -> Self {
        self.return_pictures = Some(on);
        self
    }

    fn fields(&self, stream: bool) -> Vec<(&'static str, String)> {
        let mut fields = vec![("model", self.model.clone())];
        let flag = |b: bool| if b { "true" } else { "false" }.to_string();
        if let Some(v) = &self.pages {
            fields.push(("pages", v.clone()));
        }
        if let Some(v) = self.ocr {
            fields.push(("ocr", flag(v)));
        }
        if let Some(v) = &self.output_format {
            fields.push(("output_format", v.clone()));
        }
        if let Some(v) = &self.describe_pictures {
            fields.push(("describe_pictures", v.clone()));
        }
        if let Some(v) = self.return_pictures {
            fields.push(("return_pictures", flag(v)));
        }
        fields.push(("stream", flag(stream)));
        fields
    }
}

/// Filters of `GET /v1/logs`.
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LogQuery {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route: Option<String>,
    /// `ok`, `error`, `4xx` or `5xx`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub limit: Option<u32>,
}

fn read_path(path: &std::path::Path) -> Result<(String, Vec<u8>)> {
    let bytes = std::fs::read(path)
        .map_err(|e| Error::InvalidArgument(format!("{}: {e}", path.display())))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "upload".into());
    Ok((name, bytes))
}

// --- client ---------------------------------------------------------------------------------

/// Configures a [`Client`].
#[derive(Debug, Clone, Default)]
pub struct ClientBuilder {
    base_url: Option<String>,
    api_key: Option<String>,
    timeout: Option<Duration>,
    retry: Option<RetryPolicy>,
    max_concurrency_per_model: Option<usize>,
    http: Option<reqwest::Client>,
}

impl ClientBuilder {
    /// Defaults to `$HYPNOS_BASE_URL` or [`DEFAULT_BASE_URL`]; a trailing `/v1` is accepted.
    pub fn base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }
    /// Defaults to `$HYPNOS_API_KEY`.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
    /// Sends no key even when `$HYPNOS_API_KEY` is set.
    pub fn no_api_key(mut self) -> Self {
        self.api_key = Some(String::new());
        self
    }
    /// Per request, headers included ([`DEFAULT_TIMEOUT`] by default).
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }
    pub fn retry(mut self, policy: RetryPolicy) -> Self {
        self.retry = Some(policy);
        self
    }
    /// At most `n` requests per model in flight from this client (a stream counts until it is
    /// dropped). Beyond a model's slots the server only queues them, beyond its queue it
    /// answers 429.
    pub fn max_concurrency_per_model(mut self, n: usize) -> Self {
        self.max_concurrency_per_model = Some(n.max(1));
        self
    }
    /// Uses this `reqwest::Client` (its own timeout applies).
    pub fn http_client(mut self, http: reqwest::Client) -> Self {
        self.http = Some(http);
        self
    }

    pub fn build(self) -> Result<Client> {
        let base = self
            .base_url
            .or_else(|| std::env::var("HYPNOS_BASE_URL").ok())
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let base = base.trim_end_matches('/');
        let base = base.strip_suffix("/v1").unwrap_or(base).to_string();

        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static(concat!("hypnos-client-rust/", env!("CARGO_PKG_VERSION"))),
        );
        let key = self
            .api_key
            .or_else(|| std::env::var("HYPNOS_API_KEY").ok())
            .filter(|k| !k.is_empty());
        if let Some(key) = key {
            let value = HeaderValue::from_str(&format!("Bearer {key}"))
                .map_err(|e| Error::InvalidArgument(format!("api key: {e}")))?;
            headers.insert(AUTHORIZATION, value);
        }
        let http = match self.http {
            Some(http) => http,
            None => reqwest::Client::builder()
                .timeout(self.timeout.unwrap_or(DEFAULT_TIMEOUT))
                .connect_timeout(Duration::from_secs(10))
                .build()
                .map_err(|e| Error::InvalidArgument(format!("http client: {e}")))?,
        };
        Ok(Client {
            inner: Arc::new(Inner {
                base_url: base,
                headers,
                http,
                retry: self.retry.unwrap_or_default(),
                max_per_model: self.max_concurrency_per_model,
                limits: Mutex::new(HashMap::new()),
            }),
        })
    }
}

struct Inner {
    base_url: String,
    headers: HeaderMap,
    http: reqwest::Client,
    retry: RetryPolicy,
    max_per_model: Option<usize>,
    limits: Mutex<HashMap<String, Arc<Semaphore>>>,
}

/// A client for one Hypnos server. Cheap to clone; clones share the connection pool and the
/// per-model bound.
#[derive(Clone)]
pub struct Client {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for Client {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Client")
            .field("base_url", &self.inner.base_url)
            .finish_non_exhaustive()
    }
}

fn header(response: &Response, name: &str) -> Option<String> {
    response
        .headers()
        .get(name)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

fn api_error(status: u16, headers: &HeaderMap, text: &str) -> Error {
    let get = |name: &str| headers.get(name).and_then(|v| v.to_str().ok());
    let mut e = ApiError {
        status,
        message: if text.trim().is_empty() {
            format!("HTTP {status}")
        } else {
            text.trim().to_string()
        },
        request_id: get("x-request-id").map(str::to_string),
        retry_after: get("retry-after")
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| s.is_finite() && *s >= 0.0)
            .map(Duration::from_secs_f64),
        ..ApiError::default()
    };
    if let Ok(body) = serde_json::from_str::<Value>(text) {
        if let Some(err) = body.get("error").filter(|e| e.is_object()) {
            if let Some(m) = err.get("message").and_then(Value::as_str) {
                e.message = m.to_string();
            }
            e.kind = err.get("type").and_then(Value::as_str).map(str::to_string);
            e.code = err.get("code").and_then(Value::as_str).map(str::to_string);
            e.body = Some(body);
        }
    }
    Error::from_api(e)
}

fn transport_error(e: reqwest::Error, what: &str) -> Error {
    if e.is_timeout() {
        Error::Timeout(Box::new(ApiError {
            status: 408,
            message: format!("no response in time: {e}"),
            ..ApiError::default()
        }))
    } else {
        Error::Connection(format!("{what}: {e}"))
    }
}

impl Client {
    /// A client configured from the environment (`$HYPNOS_BASE_URL`, `$HYPNOS_API_KEY`).
    pub fn new() -> Result<Self> {
        Self::builder().build()
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::default()
    }

    /// The server root, without `/v1`.
    pub fn base_url(&self) -> &str {
        &self.inner.base_url
    }

    fn request(&self, method: Method, path: &str) -> RequestBuilder {
        self.inner
            .http
            .request(method, format!("{}{}", self.inner.base_url, path))
            .headers(self.inner.headers.clone())
    }

    async fn permit(&self, model: Option<&str>) -> Option<OwnedSemaphorePermit> {
        let (model, max) = (model?, self.inner.max_per_model?);
        let sem = {
            let mut limits = self.inner.limits.lock().expect("limits lock");
            limits
                .entry(model.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(max)))
                .clone()
        };
        sem.acquire_owned().await.ok()
    }

    /// Sends the request built by `build`, retrying per the policy, and returns the successful
    /// response with the model's permit (held until the body is done).
    async fn send(
        &self,
        model: Option<&str>,
        retry: bool,
        build: impl Fn() -> RequestBuilder,
    ) -> Result<(Response, Option<OwnedSemaphorePermit>)> {
        let permit = self.permit(model).await;
        let mut attempt = 0;
        loop {
            let error = match build().send().await {
                Err(e) => transport_error(e, "request failed"),
                Ok(response) if response.status().is_success() => {
                    return Ok((response, permit));
                }
                Ok(response) => {
                    let status = response.status().as_u16();
                    let headers = response.headers().clone();
                    let text = response.text().await.unwrap_or_default();
                    api_error(status, &headers, &text)
                }
            };
            let wait = if retry {
                self.inner.retry.delay(&error, attempt)
            } else {
                None
            };
            match wait {
                None => return Err(error),
                Some(wait) => {
                    attempt += 1;
                    tokio::time::sleep(wait).await;
                }
            }
        }
    }

    async fn json<T: DeserializeOwned>(
        &self,
        model: Option<&str>,
        build: impl Fn() -> RequestBuilder,
    ) -> Result<T> {
        let (response, _permit) = self.send(model, true, build).await?;
        let bytes = response
            .bytes()
            .await
            .map_err(|e| transport_error(e, "reading the response"))?;
        serde_json::from_slice(&bytes)
            .map_err(|e| Error::Decode(format!("{e}: {}", String::from_utf8_lossy(&bytes))))
    }

    fn post_json<'a>(&'a self, path: &'a str, body: &'a Value) -> impl Fn() -> RequestBuilder + 'a {
        move || self.request(Method::POST, path).json(body)
    }

    // --- server --------------------------------------------------------------------------

    /// `GET /health`: the server is up (it may still be staging models).
    pub async fn health(&self) -> Result<Health> {
        let (response, _) = self
            .send(None, false, || self.request(Method::GET, "/health"))
            .await?;
        response
            .json()
            .await
            .map_err(|e| Error::Decode(e.to_string()))
    }

    /// `GET /ready`: true once the startup models are staged.
    pub async fn is_ready(&self) -> bool {
        matches!(
            self.request(Method::GET, "/ready").send().await,
            Ok(r) if r.status().as_u16() == 200
        )
    }

    /// Polls `/ready` every `interval` until it answers 200; [`Error::NotReady`] after `timeout`.
    pub async fn wait_until_ready(&self, timeout: Duration, interval: Duration) -> Result<()> {
        let deadline = Instant::now() + timeout;
        while !self.is_ready().await {
            if Instant::now() >= deadline {
                return Err(Error::NotReady(timeout));
            }
            tokio::time::sleep(interval).await;
        }
        Ok(())
    }

    /// `GET /metrics`: Prometheus text.
    pub async fn metrics(&self) -> Result<String> {
        let (response, _) = self
            .send(None, true, || {
                self.request(Method::GET, "/metrics")
                    .header(ACCEPT, "text/plain")
            })
            .await?;
        response
            .text()
            .await
            .map_err(|e| transport_error(e, "reading the response"))
    }

    // --- models --------------------------------------------------------------------------

    /// `GET /v1/models`: the models in service.
    pub async fn models(&self) -> Result<ModelList> {
        self.json(None, || self.request(Method::GET, "/v1/models"))
            .await
    }

    /// `POST /v1/models/{id}/enable`.
    pub async fn enable_model(&self, model: &str) -> Result<ModelSwitch> {
        let path = format!("/v1/models/{model}/enable");
        self.json(None, || self.request(Method::POST, &path)).await
    }

    /// `POST /v1/models/{id}/disable`; [`Error::ModelBusy`] while the model is leased.
    pub async fn disable_model(&self, model: &str) -> Result<ModelSwitch> {
        let path = format!("/v1/models/{model}/disable");
        self.json(None, || self.request(Method::POST, &path)).await
    }

    // --- chat ----------------------------------------------------------------------------

    fn chat_body(req: &ChatRequest, stream: bool) -> Result<Value> {
        let mut body =
            serde_json::to_value(req).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        body["stream"] = Value::Bool(stream);
        Ok(body)
    }

    /// `POST /v1/chat/completions`.
    pub async fn chat(&self, req: &ChatRequest) -> Result<ChatCompletion> {
        let body = Self::chat_body(req, false)?;
        self.json(
            Some(&req.model),
            self.post_json("/v1/chat/completions", &body),
        )
        .await
    }

    /// `POST /v1/chat/completions` with `stream: true`. Drop the stream to cancel.
    pub async fn chat_stream(&self, req: &ChatRequest) -> Result<ChatStream> {
        let body = Self::chat_body(req, true)?;
        let (response, permit) = self
            .send(Some(&req.model), true, || {
                self.request(Method::POST, "/v1/chat/completions")
                    .header(ACCEPT, "text/event-stream")
                    .json(&body)
            })
            .await?;
        let request_id = header(&response, "x-request-id");
        Ok(ChatStream::new(Body::new(response, permit), request_id))
    }

    // --- embeddings, audio, images, entities -----------------------------------------------

    /// `POST /v1/embeddings`; `data` is in input order.
    pub async fn embeddings(
        &self,
        model: &str,
        input: impl Into<Input>,
    ) -> Result<EmbeddingResponse> {
        let body = json!({ "model": model, "input": input.into() });
        self.json(Some(model), self.post_json("/v1/embeddings", &body))
            .await
    }

    /// `POST /v1/audio/transcriptions` (multipart).
    pub async fn transcribe(&self, req: &TranscriptionRequest) -> Result<Transcription> {
        self.json(Some(&req.model), || {
            let mut form = reqwest::multipart::Form::new()
                .text("model", req.model.clone())
                .part(
                    "file",
                    reqwest::multipart::Part::bytes(req.audio.clone())
                        .file_name(req.filename.clone()),
                );
            if let Some(v) = &req.language {
                form = form.text("language", v.clone());
            }
            if let Some(v) = &req.prompt {
                form = form.text("prompt", v.clone());
            }
            self.request(Method::POST, "/v1/audio/transcriptions")
                .multipart(form)
        })
        .await
    }

    fn speech_body(req: &SpeechRequest, stream: bool) -> Result<Value> {
        let mut body =
            serde_json::to_value(req).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        body["stream"] = Value::Bool(stream);
        Ok(body)
    }

    /// `POST /v1/audio/speech`: WAV (or `pcm`) bytes plus the `x-hypnos-*` telemetry headers.
    pub async fn speech(&self, req: &SpeechRequest) -> Result<Speech> {
        let body = Self::speech_body(req, false)?;
        let (response, _permit) = self
            .send(
                Some(&req.model),
                true,
                self.post_json("/v1/audio/speech", &body),
            )
            .await?;
        let content_type = header(&response, "content-type").unwrap_or_default();
        let telemetry = SpeechTelemetry::from_headers(response.headers());
        let request_id = header(&response, "x-request-id");
        let audio = response
            .bytes()
            .await
            .map_err(|e| transport_error(e, "reading the audio"))?;
        Ok(Speech {
            audio,
            content_type,
            telemetry,
            request_id,
        })
    }

    /// `POST /v1/audio/speech` with `stream: true`: audio chunks as they are made.
    pub async fn speech_stream(&self, req: &SpeechRequest) -> Result<SpeechStream> {
        let body = Self::speech_body(req, true)?;
        let (response, permit) = self
            .send(
                Some(&req.model),
                true,
                self.post_json("/v1/audio/speech", &body),
            )
            .await?;
        Ok(SpeechStream::new(response, permit))
    }

    /// `POST /v1/images/generations`: base64 PNGs in `data[i].b64_json`.
    pub async fn images(&self, req: &ImageRequest) -> Result<ImageResponse> {
        let body = serde_json::to_value(req).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        self.json(
            Some(&req.model),
            self.post_json("/v1/images/generations", &body),
        )
        .await
    }

    /// `POST /v1/entities`: zero-shot entity extraction.
    pub async fn entities(&self, req: &EntitiesRequest) -> Result<EntityResponse> {
        let body = serde_json::to_value(req).map_err(|e| Error::InvalidArgument(e.to_string()))?;
        self.json(Some(&req.model), self.post_json("/v1/entities", &body))
            .await
    }

    // --- documents -----------------------------------------------------------------------

    fn document_request(&self, req: &DocumentRequest, stream: bool) -> RequestBuilder {
        let builder = self.request(Method::POST, "/v1/documents/convert");
        let builder = if stream {
            builder.header(ACCEPT, "text/event-stream")
        } else {
            builder
        };
        match &req.source {
            DocumentSource::File { filename, bytes } => {
                let name = req.filename.clone().unwrap_or_else(|| filename.clone());
                let mut form = reqwest::multipart::Form::new().text("filename", name.clone());
                for (k, v) in req.fields(stream) {
                    form = form.text(k, v);
                }
                form = form.part(
                    "file",
                    reqwest::multipart::Part::bytes(bytes.clone()).file_name(name),
                );
                builder.multipart(form)
            }
            DocumentSource::Url(url) => {
                let mut body = Map::new();
                body.insert("url".into(), Value::String(url.clone()));
                if let Some(name) = &req.filename {
                    body.insert("filename".into(), Value::String(name.clone()));
                }
                for (k, v) in req.fields(stream) {
                    let value = match v.as_str() {
                        "true" => Value::Bool(true),
                        "false" => Value::Bool(false),
                        _ => Value::String(v),
                    };
                    body.insert(k.into(), value);
                }
                builder.json(&body)
            }
        }
    }

    /// `POST /v1/documents/convert`: a file or a URL to Markdown.
    pub async fn convert_document(&self, req: &DocumentRequest) -> Result<Document> {
        self.json(Some(&req.model), || self.document_request(req, false))
            .await
    }

    /// The streaming form of [`Client::convert_document`]: one event per page.
    pub async fn convert_document_stream(&self, req: &DocumentRequest) -> Result<DocumentStream> {
        let (response, permit) = self
            .send(Some(&req.model), true, || self.document_request(req, true))
            .await?;
        let request_id = header(&response, "x-request-id");
        Ok(DocumentStream::new(Body::new(response, permit), request_id))
    }

    /// `GET /v1/documents/{id}/pictures`: the records, without image data.
    pub async fn document_pictures(&self, document_id: &str) -> Result<PictureList> {
        let path = format!("/v1/documents/{document_id}/pictures");
        self.json(None, || self.request(Method::GET, &path)).await
    }

    /// `GET /v1/documents/{id}/pictures/{n}`: one record with `image` (base64).
    pub async fn document_picture(&self, document_id: &str, index: u32) -> Result<Picture> {
        let path = format!("/v1/documents/{document_id}/pictures/{index}");
        self.json(None, || self.request(Method::GET, &path)).await
    }

    /// The picture's image bytes and media type (`?format=image`).
    pub async fn document_picture_bytes(
        &self,
        document_id: &str,
        index: u32,
    ) -> Result<(bytes::Bytes, String)> {
        let path = format!("/v1/documents/{document_id}/pictures/{index}");
        let (response, _) = self
            .send(None, true, || {
                self.request(Method::GET, &path)
                    .query(&[("format", "image")])
                    .header(ACCEPT, "*/*")
            })
            .await?;
        let mime = header(&response, "content-type").unwrap_or_default();
        let bytes = response
            .bytes()
            .await
            .map_err(|e| transport_error(e, "reading the picture"))?;
        Ok((bytes, mime))
    }

    /// `DELETE /v1/documents/{id}`: releases the stored pictures.
    pub async fn delete_document(&self, document_id: &str) -> Result<Value> {
        let path = format!("/v1/documents/{document_id}");
        self.json(None, || self.request(Method::DELETE, &path))
            .await
    }

    // --- memory --------------------------------------------------------------------------

    /// `GET /v1/memory/status`: the three tiers and every model's placement.
    pub async fn memory_status(&self) -> Result<MemoryStatus> {
        self.json(None, || self.request(Method::GET, "/v1/memory/status"))
            .await
    }

    /// `POST /v1/memory/tier`: promote or demote a model by hand (blocks for the load).
    pub async fn set_tier(
        &self,
        model: &str,
        target_tier: MemoryTier,
        context_len: Option<u32>,
    ) -> Result<TierMove> {
        let mut body = json!({ "model": model, "target_tier": target_tier });
        if let Some(n) = context_len {
            body["context_len"] = json!(n);
        }
        self.json(None, self.post_json("/v1/memory/tier", &body))
            .await
    }

    // --- request log ---------------------------------------------------------------------

    /// `GET /v1/logs`, newest first.
    pub async fn logs(&self, query: &LogQuery) -> Result<LogList> {
        self.json(None, || self.request(Method::GET, "/v1/logs").query(query))
            .await
    }

    /// `GET /v1/logs/{id}`: one record by its `x-request-id`.
    pub async fn log(&self, request_id: &str) -> Result<LogRecord> {
        let path = format!("/v1/logs/{request_id}");
        self.json(None, || self.request(Method::GET, &path)).await
    }

    /// `DELETE /v1/logs`.
    pub async fn clear_logs(&self) -> Result<Value> {
        self.json(None, || self.request(Method::DELETE, "/v1/logs"))
            .await
    }

    /// `POST /v1/logs/level`: `off`, `error`, `warn`, `info`, `debug` or `trace`.
    pub async fn set_log_level(&self, level: &str) -> Result<Value> {
        let body = json!({ "level": level });
        self.json(None, self.post_json("/v1/logs/level", &body))
            .await
    }
}
