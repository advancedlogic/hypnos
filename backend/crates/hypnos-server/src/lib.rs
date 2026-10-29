pub mod default_profile;
pub mod metrics;

use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Multipart, Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::{self, Next},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use chrono::Utc;
use futures::stream;
use hypnos_backends::media::{encode_pcm16_raw, wav_stream_header};
use hypnos_backends::{create_backend_for_model, EngineOptions};
use hypnos_core::{
    config::{HypnosConfig, LimitsConfig, MemoryTier},
    error::HypnosError,
    openai::{
        AudioSpeechRequest, AudioStreamEvent, AudioTranscriptionRequest, ChatCompletionRequest,
        ChatStreamEvent, EmbeddingRequest, ExecutionTelemetry, ImageGenerationRequest,
        OpenAiModelInfo, OpenAiModelList,
    },
    request::{CancelOnDrop, RequestContext},
};
use hypnos_memory::TieredMemoryManager;
use hypnos_scheduler::{ActiveModelLease, ModelScheduler};
use metrics::Metrics;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::{
    collections::VecDeque,
    convert::Infallible,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::mpsc;
use tower_http::{
    cors::{AllowOrigin, CorsLayer},
    timeout::TimeoutLayer,
};
use tracing::{info, warn};
use uuid::Uuid;

#[derive(Clone)]
pub struct AppState {
    pub scheduler: Arc<ModelScheduler>,
    pub limits: LimitsConfig,
    pub metrics: Arc<Metrics>,
    /// Flipped once startup tier initialization has finished (`/ready`).
    pub ready: Arc<AtomicBool>,
    pub started_at: Instant,
    pub version: &'static str,
}

impl AppState {
    pub fn from_config(config: HypnosConfig) -> hypnos_core::error::Result<Self> {
        let engine_opts = EngineOptions {
            kv_idle_timeout: Duration::from_secs(config.server.kv_idle_timeout_secs.max(1)),
        };
        let backends = config
            .models
            .iter()
            .cloned()
            .map(|m| create_backend_for_model(m, &engine_opts))
            .collect();
        let memory = Arc::new(TieredMemoryManager::new(&config.hardware, config.models)?);
        let scheduler = Arc::new(ModelScheduler::new(memory, backends, &config.limits));
        Ok(Self {
            scheduler,
            limits: config.limits,
            metrics: Arc::new(Metrics::default()),
            ready: Arc::new(AtomicBool::new(false)),
            started_at: Instant::now(),
            version: env!("CARGO_PKG_VERSION"),
        })
    }

    /// A fresh request context bound to the configured timeout and limits.
    pub fn new_request(&self) -> RequestContext {
        RequestContext::new(
            self.limits.clone(),
            Some(Duration::from_secs(self.limits.request_timeout_secs)),
            format!("req-{}", Uuid::new_v4().simple()),
        )
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_secs(self.limits.request_timeout_secs)
    }
}

/// Builds the CORS layer: exact origins from the config, or any loopback origin when empty.
pub fn cors_layer(allowed_origins: &[String]) -> CorsLayer {
    let allow = if allowed_origins.is_empty() {
        AllowOrigin::predicate(|origin: &HeaderValue, _| {
            origin.to_str().map(is_loopback_origin).unwrap_or(false)
        })
    } else {
        let patterns: Vec<String> = allowed_origins.to_vec();
        AllowOrigin::predicate(move |origin: &HeaderValue, _| {
            let Ok(o) = origin.to_str() else { return false };
            patterns.iter().any(|p| origin_matches(p, o))
        })
    };
    CorsLayer::new()
        .allow_origin(allow)
        .allow_methods([
            axum::http::Method::GET,
            axum::http::Method::POST,
            axum::http::Method::OPTIONS,
        ])
        .allow_headers([header::CONTENT_TYPE, header::AUTHORIZATION, header::ACCEPT])
        .expose_headers(
            [
                HeaderValue::from_static("x-request-id"),
                HeaderValue::from_static("x-hypnos-previous-tier"),
                HeaderValue::from_static("x-hypnos-gpu-layers"),
                HeaderValue::from_static("x-hypnos-switch-ms"),
                HeaderValue::from_static("x-hypnos-inference-ms"),
                HeaderValue::from_static("x-hypnos-audio-secs"),
                HeaderValue::from_static("x-hypnos-rtf"),
            ]
            .into_iter()
            .map(|v| header::HeaderName::from_bytes(v.as_bytes()).expect("static header name"))
            .collect::<Vec<_>>(),
        )
}

fn is_loopback_origin(origin: &str) -> bool {
    let rest = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"));
    let Some(rest) = rest else { return false };
    let host = if let Some(v6) = rest.strip_prefix('[') {
        // bracketed IPv6 literal, e.g. [::1]:3000
        match v6.split_once(']') {
            Some((h, _)) => format!("[{h}]"),
            None => return false,
        }
    } else {
        rest.split(':').next().unwrap_or("").to_string()
    };
    matches!(host.as_str(), "localhost" | "127.0.0.1" | "[::1]")
}

/// `http://localhost:*` style wildcards or exact matches.
fn origin_matches(pattern: &str, origin: &str) -> bool {
    if let Some(prefix) = pattern.strip_suffix(":*") {
        return origin
            .strip_prefix(prefix)
            .map(|rest| rest.starts_with(':') && rest[1..].chars().all(|c| c.is_ascii_digit()))
            .unwrap_or(false);
    }
    pattern == origin
}

pub fn build_router(state: AppState, allowed_origins: &[String]) -> Router {
    let body_limit = usize::try_from(state.limits.max_body_bytes).unwrap_or(usize::MAX);
    let timeout = state.request_timeout();
    Router::new()
        // OpenAI Standard Endpoints across all 6 modalities
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/audio/transcriptions", post(audio_transcriptions))
        .route("/v1/audio/speech", post(audio_speech))
        .route("/v1/images/generations", post(image_generations))
        // Hypnos 3-Tier Memory Observability & Control Endpoints
        .route("/v1/memory/status", get(memory_status))
        .route("/v1/memory/tier", post(set_model_tier))
        .route("/health", get(health_check))
        .route("/ready", get(ready_check))
        .route("/metrics", get(metrics_endpoint))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            observe_request,
        ))
        .layer(TimeoutLayer::new(timeout))
        .layer(DefaultBodyLimit::max(body_limit))
        .layer(cors_layer(allowed_origins))
        .with_state(state)
}

/// Adds a request id, records latency and status per route.
async fn observe_request(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let route = req.uri().path().to_string();
    let start = Instant::now();
    let request_id = format!("req-{}", Uuid::new_v4().simple());
    let mut resp = next.run(req).await;
    let status = resp.status();
    state
        .metrics
        .record(&route, status.as_u16(), start.elapsed());
    if let Ok(v) = HeaderValue::from_str(&request_id) {
        resp.headers_mut().insert("x-request-id", v);
    }
    if status.is_server_error() {
        warn!(route = %route, status = status.as_u16(), elapsed_ms = start.elapsed().as_millis() as u64, request_id = %request_id, "request failed");
    } else {
        tracing::debug!(route = %route, status = status.as_u16(), elapsed_ms = start.elapsed().as_millis() as u64, request_id = %request_id, "request");
    }
    resp
}

async fn health_check(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "service": "hypnos",
        "version": state.version,
        "uptime_secs": state.started_at.elapsed().as_secs(),
    }))
}

async fn ready_check(State(state): State<AppState>) -> Response {
    if state.ready.load(Ordering::SeqCst) {
        Json(json!({ "ready": true })).into_response()
    } else {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({ "ready": false, "reason": "startup tier initialization in progress" })),
        )
            .into_response()
    }
}

async fn metrics_endpoint(State(state): State<AppState>) -> Response {
    let status = state.scheduler.status();
    let body = state
        .metrics
        .render(&status, state.scheduler.stats(), state.started_at.elapsed());
    (
        [(
            header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        body,
    )
        .into_response()
}

async fn list_models(State(state): State<AppState>) -> Json<OpenAiModelList> {
    let snapshot = state.scheduler.status();
    let created = Utc::now().timestamp();
    let data = snapshot
        .models
        .into_iter()
        .map(|m| OpenAiModelInfo {
            id: m.id,
            object: "model".to_string(),
            created,
            owned_by: "hypnos".to_string(),
            modality: m.modality,
            current_tier: m.current_tier,
            placement: m.placement,
            active_leases: m.active_leases,
        })
        .collect();

    Json(OpenAiModelList {
        object: "list".to_string(),
        data,
    })
}

async fn memory_status(
    State(state): State<AppState>,
) -> Json<hypnos_memory::MemoryHierarchyStatus> {
    state.scheduler.memory().reconcile_vram();
    Json(state.scheduler.status())
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SetTierRequest {
    pub model: String,
    pub target_tier: MemoryTier,
    #[serde(default)]
    pub context_len: Option<u32>,
}

async fn set_model_tier(
    State(state): State<AppState>,
    Json(req): Json<SetTierRequest>,
) -> std::result::Result<Json<serde_json::Value>, ApiError> {
    let plan = state
        .scheduler
        .set_model_tier(&req.model, req.target_tier, req.context_len)
        .await?;
    Ok(Json(json!({
        "model": req.model,
        "placement": plan,
    })))
}

/// Lease telemetry (tiers, switch time) plus whatever engine timings the backend attached to its
/// response. The server owns the overall inference clock; the backend owns the phase split.
fn finish_telemetry(
    lease: &ActiveModelLease,
    infer_start: Instant,
    from_backend: Option<ExecutionTelemetry>,
) -> ExecutionTelemetry {
    let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    let engine = from_backend.map(|t| t.engine).unwrap_or_default();
    lease.build_telemetry(infer_ms).with_engine(engine)
}

/// State carried through the SSE stream; owning the lease keeps the model pinned in VRAM until the
/// client has received the final chunk, and dropping the guard cancels generation on disconnect.
struct ChatSseState {
    rx: mpsc::Receiver<ChatStreamEvent>,
    lease: ActiveModelLease,
    metrics: Arc<Metrics>,
    _cancel_guard: CancelOnDrop,
    id: String,
    model: String,
    created: i64,
    infer_start: Instant,
    queued: VecDeque<Event>,
    finished: bool,
}

fn sse_chunk(
    st: &ChatSseState,
    delta: serde_json::Value,
    finish_reason: serde_json::Value,
    extra: serde_json::Map<String, serde_json::Value>,
) -> Event {
    let mut body = json!({
        "id": st.id,
        "object": "chat.completion.chunk",
        "created": st.created,
        "model": st.model,
        "choices": [{ "index": 0, "delta": delta, "finish_reason": finish_reason }],
    });
    if let serde_json::Value::Object(ref mut map) = body {
        map.extend(extra);
    }
    Event::default().data(body.to_string())
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> std::result::Result<Response, ApiError> {
    if req.messages.is_empty() {
        return Err(HypnosError::InvalidRequest("messages must not be empty".into()).into());
    }
    // Estimate required KV cache context window from prompt + max_tokens + vision image patches
    let estimated_prompt_tokens: u32 = req
        .messages
        .iter()
        .map(|m| {
            let text_tokens = (m.content.extract_text().len() as u32 / 3).max(8);
            let vision_tokens = (m.content.image_urls().len() as u32) * 576;
            let audio_tokens = if m.content.contains_audio() { 1024 } else { 0 };
            text_tokens + vision_tokens + audio_tokens
        })
        .sum();
    let max_new_tokens = req
        .max_tokens
        .unwrap_or(512)
        .min(state.limits.max_tokens_cap.max(1));
    let requested_ctx = req
        .num_ctx
        .or(Some((estimated_prompt_tokens + max_new_tokens).max(2048)));

    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, requested_ctx, 1, &ctx)
        .await?;

    let infer_start = Instant::now();

    if req.stream.unwrap_or(false) {
        let (tx, rx) = mpsc::channel::<ChatStreamEvent>(128);
        let backend = lease.backend.clone();
        let req_for_task = req.clone();
        let ctx_for_task = ctx.clone();
        tokio::spawn(async move {
            if let Err(e) = backend
                .chat_completion_stream(&req_for_task, &ctx_for_task, tx.clone())
                .await
            {
                let _ = tx.send(ChatStreamEvent::Error(e.to_string())).await;
            }
        });

        let mut initial = ChatSseState {
            rx,
            model: req.model.clone(),
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            created: Utc::now().timestamp(),
            infer_start,
            lease,
            metrics: state.metrics.clone(),
            _cancel_guard: guard,
            queued: VecDeque::new(),
            finished: false,
        };
        // First chunk announces the assistant role (OpenAI convention).
        let first = sse_chunk(
            &initial,
            json!({"role": "assistant", "content": ""}),
            serde_json::Value::Null,
            Default::default(),
        );
        initial.queued.push_back(first);

        let event_stream = stream::unfold(initial, |mut st| async move {
            if let Some(ev) = st.queued.pop_front() {
                return Some((Ok::<Event, Infallible>(ev), st));
            }
            if st.finished {
                return None;
            }
            match st.rx.recv().await {
                Some(ChatStreamEvent::Delta(text)) => {
                    let ev = sse_chunk(
                        &st,
                        json!({"content": text}),
                        serde_json::Value::Null,
                        Default::default(),
                    );
                    Some((Ok(ev), st))
                }
                Some(ChatStreamEvent::Done {
                    usage,
                    finish_reason,
                    timing,
                }) => {
                    st.finished = true;
                    st.metrics.record_chat(&st.model, &timing);
                    let infer_ms = st.infer_start.elapsed().as_secs_f64() * 1000.0;
                    let telemetry = st.lease.build_telemetry(infer_ms).with_engine(timing);
                    let mut extra = serde_json::Map::new();
                    extra.insert(
                        "usage".into(),
                        serde_json::to_value(&usage).unwrap_or(serde_json::Value::Null),
                    );
                    extra.insert(
                        "hypnos_telemetry".into(),
                        serde_json::to_value(telemetry).unwrap_or(serde_json::Value::Null),
                    );
                    let ev = sse_chunk(&st, json!({}), json!(finish_reason), extra);
                    st.queued.push_back(Event::default().data("[DONE]"));
                    Some((Ok(ev), st))
                }
                Some(ChatStreamEvent::Error(message)) => {
                    st.finished = true;
                    let mut extra = serde_json::Map::new();
                    extra.insert(
                        "error".into(),
                        json!({ "message": message, "type": "server_error" }),
                    );
                    let ev = sse_chunk(&st, json!({}), json!("error"), extra);
                    st.queued.push_back(Event::default().data("[DONE]"));
                    Some((Ok(ev), st))
                }
                None => {
                    st.finished = true;
                    Some((Ok(Event::default().data("[DONE]")), st))
                }
            }
        });
        return Ok(Sse::new(event_stream)
            .keep_alive(KeepAlive::default())
            .into_response());
    }

    let mut resp = lease.backend.chat_completion(&req, &ctx).await?;
    let telemetry = finish_telemetry(&lease, infer_start, resp.hypnos_telemetry.take());
    state.metrics.record_chat(&req.model, &telemetry.engine);
    resp.hypnos_telemetry = Some(telemetry);
    guard.disarm();
    Ok(Json(resp).into_response())
}

async fn embeddings(
    State(state): State<AppState>,
    Json(req): Json<EmbeddingRequest>,
) -> std::result::Result<Json<hypnos_core::openai::EmbeddingResponse>, ApiError> {
    let batch_size = match &req.input {
        hypnos_core::openai::EmbeddingInput::Single(_) => 1,
        hypnos_core::openai::EmbeddingInput::Batch(v) => v.len().max(1) as u32,
    };

    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, batch_size, &ctx)
        .await?;

    let infer_start = Instant::now();
    let mut resp = lease.backend.embeddings(&req, &ctx).await?;
    resp.hypnos_telemetry = Some(finish_telemetry(
        &lease,
        infer_start,
        resp.hypnos_telemetry.take(),
    ));
    guard.disarm();
    Ok(Json(resp))
}

async fn audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Body,
) -> std::result::Result<Json<hypnos_core::openai::AudioTranscriptionResponse>, ApiError> {
    let content_type = headers
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let max_audio = usize::try_from(state.limits.max_audio_bytes).unwrap_or(usize::MAX);

    let req = if content_type.starts_with("multipart/form-data") {
        // Parse OpenAI-style multipart form (`model`, `file`, `language`, `prompt`)
        let req_reconstruct = axum::http::Request::builder()
            .header(header::CONTENT_TYPE, content_type)
            .body(body)
            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?;
        use axum::extract::FromRequest;
        let mut multipart = Multipart::from_request(req_reconstruct, &state)
            .await
            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?;

        let mut model = String::new();
        let mut audio_bytes = Vec::new();
        let mut filename = None;
        let mut language = None;
        let mut prompt = None;

        while let Some(field) = multipart
            .next_field()
            .await
            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?
        {
            let name = field.name().unwrap_or("").to_string();
            match name.as_str() {
                "model" => {
                    model = field
                        .text()
                        .await
                        .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?;
                }
                "file" => {
                    filename = field.file_name().map(|s| s.to_string());
                    let bytes = field
                        .bytes()
                        .await
                        .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?;
                    if bytes.len() > max_audio {
                        return Err(HypnosError::InvalidRequest(format!(
                            "audio file is {} bytes, above limits.max_audio_bytes ({max_audio})",
                            bytes.len()
                        ))
                        .into());
                    }
                    audio_bytes = bytes.to_vec();
                }
                "language" => {
                    language = Some(
                        field
                            .text()
                            .await
                            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?,
                    );
                }
                "prompt" => {
                    prompt = Some(
                        field
                            .text()
                            .await
                            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?,
                    );
                }
                _ => {}
            }
        }

        if model.is_empty() {
            return Err(HypnosError::InvalidRequest(
                "Missing 'model' field in multipart/form-data request".to_string(),
            )
            .into());
        }

        AudioTranscriptionRequest {
            model,
            audio_bytes,
            filename,
            language,
            prompt,
        }
    } else {
        let bytes = axum::body::to_bytes(body, max_audio.saturating_mul(2).max(1024))
            .await
            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?;
        serde_json::from_slice::<AudioTranscriptionRequest>(&bytes)
            .map_err(|e| HypnosError::InvalidRequest(e.to_string()))?
    };

    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, 1, &ctx)
        .await?;

    let infer_start = Instant::now();
    let mut resp = lease.backend.transcribe(&req, &ctx).await?;
    let telemetry = finish_telemetry(&lease, infer_start, resp.hypnos_telemetry.take());
    state.metrics.record_stt(&req.model, &telemetry.engine);
    resp.hypnos_telemetry = Some(telemetry);
    guard.disarm();
    Ok(Json(resp))
}

/// Streaming speech: the lease is held until the last chunk is sent, dropping the guard cancels
/// synthesis when the client disconnects, and the first chunk is prefixed with a live WAV header
/// when the client asked for `wav`.
struct AudioStreamState {
    rx: mpsc::Receiver<AudioStreamEvent>,
    lease: ActiveModelLease,
    metrics: Arc<Metrics>,
    _cancel_guard: CancelOnDrop,
    model: String,
    /// A live WAV header still has to go out in front of the first chunk.
    header_pending: bool,
    infer_start: Instant,
    first_audio_ms: Option<f64>,
}

fn speech_telemetry_headers(headers: &mut HeaderMap, telemetry: &ExecutionTelemetry) {
    if let Ok(v) = HeaderValue::from_str(&format!("{:?}", telemetry.previous_tier)) {
        headers.insert("x-hypnos-previous-tier", v);
    }
    if let Ok(v) = HeaderValue::from_str(&telemetry.placement.gpu_layers.to_string()) {
        headers.insert("x-hypnos-gpu-layers", v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("{:.3}", telemetry.tier_switch_latency_ms)) {
        headers.insert("x-hypnos-switch-ms", v);
    }
}

async fn audio_speech(
    State(state): State<AppState>,
    Json(req): Json<AudioSpeechRequest>,
) -> std::result::Result<Response, ApiError> {
    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, 1, &ctx)
        .await?;

    let infer_start = Instant::now();

    if req.stream.unwrap_or(false) {
        let wav = !req.response_format.eq_ignore_ascii_case("pcm");
        let (tx, rx) = mpsc::channel::<AudioStreamEvent>(64);
        let backend = lease.backend.clone();
        let req_for_task = req.clone();
        let ctx_for_task = ctx.clone();
        tokio::spawn(async move {
            if let Err(e) = backend
                .synthesize_speech_stream(&req_for_task, &ctx_for_task, tx.clone())
                .await
            {
                let _ = tx.send(AudioStreamEvent::Error(e.to_string())).await;
            }
        });
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static(if wav { "audio/wav" } else { "audio/pcm" }),
        );
        speech_telemetry_headers(&mut headers, &lease.build_telemetry(0.0));
        let initial = AudioStreamState {
            rx,
            model: req.model.clone(),
            lease,
            metrics: state.metrics.clone(),
            _cancel_guard: guard,
            header_pending: wav,
            infer_start,
            first_audio_ms: None,
        };
        let body = stream::unfold(initial, |mut st| async move {
            loop {
                match st.rx.recv().await {
                    Some(AudioStreamEvent::Chunk(chunk)) => {
                        st.first_audio_ms
                            .get_or_insert_with(|| st.infer_start.elapsed().as_secs_f64() * 1000.0);
                        let mut bytes = Vec::with_capacity(chunk.pcm.len() * 2 + 44);
                        if st.header_pending {
                            st.header_pending = false;
                            bytes.extend_from_slice(&wav_stream_header(chunk.sample_rate));
                        }
                        bytes.extend_from_slice(&encode_pcm16_raw(&chunk.pcm));
                        return Some((Ok::<Bytes, Infallible>(Bytes::from(bytes)), st));
                    }
                    Some(AudioStreamEvent::Done { mut timing }) => {
                        // The server's clock sees the lease wait and channel hop as well.
                        timing.ttfa_ms = st.first_audio_ms.or(timing.ttfa_ms);
                        st.metrics.record_tts(&st.model, &timing);
                        let infer_ms = st.infer_start.elapsed().as_secs_f64() * 1000.0;
                        tracing::debug!(model = %st.model, infer_ms = format!("{infer_ms:.0}"), ttfa_ms = timing.ttfa_ms.map(|v| format!("{v:.0}")), lease_tier = ?st.lease.previous_tier, "speech stream complete");
                        return None;
                    }
                    Some(AudioStreamEvent::Error(message)) => {
                        warn!(model = %st.model, error = %message, "speech stream failed after it started");
                        return None;
                    }
                    None => return None,
                }
            }
        });
        return Ok((headers, Body::from_stream(body)).into_response());
    }

    let out = lease.backend.synthesize_speech(&req, &ctx).await?;
    let telemetry = finish_telemetry(&lease, infer_start, Some(out.telemetry.clone()));
    state.metrics.record_tts(&req.model, &telemetry.engine);
    guard.disarm();

    let mut headers = HeaderMap::new();
    headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_str(&out.content_type)
            .unwrap_or_else(|_| HeaderValue::from_static("audio/wav")),
    );
    if let Ok(v) = HeaderValue::from_str(&format!("{:?}", telemetry.previous_tier)) {
        headers.insert("x-hypnos-previous-tier", v);
    }
    if let Ok(v) = HeaderValue::from_str(&telemetry.placement.gpu_layers.to_string()) {
        headers.insert("x-hypnos-gpu-layers", v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("{:.3}", telemetry.tier_switch_latency_ms)) {
        headers.insert("x-hypnos-switch-ms", v);
    }
    if let Ok(v) = HeaderValue::from_str(&format!("{:.1}", telemetry.inference_latency_ms)) {
        headers.insert("x-hypnos-inference-ms", v);
    }
    if let Some(v) = telemetry
        .engine
        .audio_secs
        .and_then(|s| HeaderValue::from_str(&format!("{s:.3}")).ok())
    {
        headers.insert("x-hypnos-audio-secs", v);
    }
    if let Some(v) = telemetry
        .engine
        .rtf
        .and_then(|r| HeaderValue::from_str(&format!("{r:.4}")).ok())
    {
        headers.insert("x-hypnos-rtf", v);
    }

    Ok((headers, out.audio_bytes).into_response())
}

async fn image_generations(
    State(state): State<AppState>,
    Json(req): Json<ImageGenerationRequest>,
) -> std::result::Result<Json<hypnos_core::openai::ImageGenerationResponse>, ApiError> {
    if req.n > state.limits.max_images_per_request {
        return Err(HypnosError::InvalidRequest(format!(
            "n = {} exceeds limits.max_images_per_request ({})",
            req.n, state.limits.max_images_per_request
        ))
        .into());
    }
    let ctx = state.new_request();
    let guard = ctx.cancel_on_drop();
    let lease = state
        .scheduler
        .acquire_model(&req.model, None, req.n.max(1), &ctx)
        .await?;

    let infer_start = Instant::now();
    let mut resp = lease.backend.generate_image(&req, &ctx).await?;
    state
        .metrics
        .record_image(&req.model, infer_start.elapsed());
    resp.hypnos_telemetry = Some(finish_telemetry(
        &lease,
        infer_start,
        resp.hypnos_telemetry.take(),
    ));
    guard.disarm();
    Ok(Json(resp))
}

// ============================================================================
// OpenAI-Compatible Error Wrapper
// ============================================================================

pub struct ApiError(pub HypnosError);

impl From<HypnosError> for ApiError {
    fn from(err: HypnosError) -> Self {
        Self(err)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, error_type) = match &self.0 {
            HypnosError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "model_not_found"),
            HypnosError::ModalityMismatch { .. } | HypnosError::InvalidRequest(_) => {
                (StatusCode::BAD_REQUEST, "invalid_request_error")
            }
            HypnosError::InsufficientMemory { .. } => {
                (StatusCode::INSUFFICIENT_STORAGE, "insufficient_vram_or_ram")
            }
            HypnosError::Busy { .. } => (StatusCode::TOO_MANY_REQUESTS, "model_busy"),
            HypnosError::Timeout(_) => (StatusCode::GATEWAY_TIMEOUT, "timeout"),
            HypnosError::Cancelled => (
                StatusCode::from_u16(499).unwrap_or(StatusCode::BAD_REQUEST),
                "cancelled",
            ),
            HypnosError::Config(_) => (StatusCode::SERVICE_UNAVAILABLE, "model_not_available"),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, "server_error"),
        };

        if status.is_server_error() {
            warn!(error = %self.0, "request error");
        } else {
            info!(error = %self.0, status = status.as_u16(), "request rejected");
        }

        let payload = json!({
            "error": {
                "message": self.0.to_string(),
                "type": error_type,
            }
        });

        (status, Json(payload)).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_origins() {
        assert!(is_loopback_origin("http://localhost:5173"));
        assert!(is_loopback_origin("http://127.0.0.1:8080"));
        assert!(is_loopback_origin("http://[::1]:3000"));
        assert!(!is_loopback_origin("http://evil.example"));
        assert!(!is_loopback_origin("http://localhost.evil.example"));
    }

    #[test]
    fn origin_patterns() {
        assert!(origin_matches(
            "http://localhost:*",
            "http://localhost:5173"
        ));
        assert!(!origin_matches(
            "http://localhost:*",
            "http://localhost:5173x"
        ));
        assert!(origin_matches("https://app.example", "https://app.example"));
        assert!(!origin_matches(
            "https://app.example",
            "https://app.example.evil"
        ));
    }
}
