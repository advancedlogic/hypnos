//! Engine failure paths through the real router, with a mock engine instead of llama.cpp
//! (TEST-2): a load the engine refuses answers 500 and is counted per model, a failure in the
//! middle of a generation releases the lease and ends the stream with an error event, and the
//! model keeps serving afterwards, and an engine that runs out of device memory is retried once
//! after idle models are evicted, or answered with 507.

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use hypnos_core::config::{
    AcceleratorKind, ApiKeyLimits, EngineKind, HardwareConfig, KvDataType, LimitsConfig,
    MemoryTier, Modality, ModelArchitectureParams, ModelConfig, ModelPriority,
};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatResponseMessage,
    ChatStreamEvent, EngineTiming, TokenUsage,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use hypnos_memory::{DeviceMemoryCapabilities, TieredMemoryManager};
use hypnos_scheduler::ModelScheduler;
use hypnos_server::request_log::{LogLevel, RequestLog};
use hypnos_server::{auth, build_router, metrics::Metrics, pictures, quota, AppState};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc;
use tower::ServiceExt;

const GB: u64 = 1024 * 1024 * 1024;

struct MockEngine {
    config: ModelConfig,
    resident: AtomicBool,
    fail_loads: AtomicBool,
    /// Fail the next generations with a mid-stream error.
    fail_generation: AtomicBool,
    /// Fail this many upcoming generations with `DeviceOutOfMemory`.
    device_oom: AtomicUsize,
    generations: AtomicUsize,
    placements: Mutex<Vec<MemoryTier>>,
    /// Stress mode (TEST-4): loads take this long, and streams send twenty pieces this far
    /// apart, checking for cancellation before each (as the engines do).
    delay_ms: std::sync::atomic::AtomicU64,
}

#[async_trait]
impl ModelBackend for MockEngine {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }
    /// A model whose id starts with `no-tools` has no tool-calling format (like LFM3).
    fn supports_tool_calls(&self) -> bool {
        !self.config.id.starts_with("no-tools")
    }
    async fn apply_placement(&self, plan: &LayerPlacementPlan, _ptr: Option<usize>) -> Result<()> {
        self.placements.lock().unwrap().push(plan.primary_tier);
        let delay = self.delay_ms.load(Ordering::SeqCst);
        if delay > 0 && plan.primary_tier == MemoryTier::Tier1GpuVram {
            tokio::time::sleep(Duration::from_millis(delay)).await;
        }
        if plan.primary_tier == MemoryTier::Tier1GpuVram {
            if self.fail_loads.load(Ordering::SeqCst) {
                return Err(HypnosError::BackendError(
                    "simulated: weights file changed on disk".into(),
                ));
            }
            self.resident.store(true, Ordering::SeqCst);
        } else {
            self.resident.store(false, Ordering::SeqCst);
        }
        Ok(())
    }
    async fn chat_completion(
        &self,
        req: &ChatCompletionRequest,
        _ctx: &RequestContext,
    ) -> Result<ChatCompletionResponse> {
        self.generations.fetch_add(1, Ordering::SeqCst);
        let delay = self.delay_ms.load(Ordering::SeqCst);
        for _ in 0..(if delay > 0 { 4 } else { 0 }) {
            if !_ctx.should_continue() {
                return Err(HypnosError::Cancelled);
            }
            tokio::time::sleep(Duration::from_millis(delay / 4)).await;
        }
        if self.fail_generation.load(Ordering::SeqCst) {
            return Err(HypnosError::BackendError(
                "simulated CUDA out of memory during decode".into(),
            ));
        }
        if self
            .device_oom
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok()
        {
            return Err(HypnosError::DeviceOutOfMemory {
                model: req.model.clone(),
                detail: "simulated: compute buffer allocation failed".into(),
            });
        }
        Ok(ChatCompletionResponse {
            id: "chatcmpl-mock".into(),
            object: "chat.completion".into(),
            created: 0,
            model: req.model.clone(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatResponseMessage {
                    role: "assistant".into(),
                    content: "mock reply".into(),
                    reasoning_content: None,
                    tool_calls: None,
                },
                finish_reason: "stop".into(),
            }],
            usage: TokenUsage {
                prompt_tokens: 3,
                completion_tokens: 2,
                total_tokens: 5,
                prompt_tokens_details: None,
            },
            hypnos_telemetry: None,
        })
    }
    async fn chat_completion_stream(
        &self,
        _req: &ChatCompletionRequest,
        _ctx: &RequestContext,
        tx: mpsc::Sender<ChatStreamEvent>,
    ) -> Result<()> {
        self.generations.fetch_add(1, Ordering::SeqCst);
        let delay = self.delay_ms.load(Ordering::SeqCst);
        if delay > 0 {
            for _ in 0..20 {
                if !_ctx.should_continue() {
                    return Err(HypnosError::Cancelled);
                }
                if tx
                    .send(ChatStreamEvent::Delta("tok ".into()))
                    .await
                    .is_err()
                {
                    return Err(HypnosError::Cancelled);
                }
                tokio::time::sleep(Duration::from_millis(delay / 4)).await;
            }
        }
        let _ = tx.send(ChatStreamEvent::Delta("mock ".into())).await;
        if self.fail_generation.load(Ordering::SeqCst) {
            let _ = tx
                .send(ChatStreamEvent::Error(
                    "simulated CUDA out of memory mid-generation".into(),
                ))
                .await;
            return Ok(());
        }
        let _ = tx.send(ChatStreamEvent::Delta("reply".into())).await;
        let _ = tx
            .send(ChatStreamEvent::Done {
                usage: TokenUsage {
                    prompt_tokens: 3,
                    completion_tokens: 2,
                    total_tokens: 5,
                    prompt_tokens_details: None,
                },
                finish_reason: "stop".into(),
                timing: EngineTiming::default(),
            })
            .await;
        Ok(())
    }
}

fn model(id: &str) -> ModelConfig {
    let dir = std::env::temp_dir().join("hypnos-fault-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{id}.bin"));
    if !path.exists() {
        std::fs::write(&path, b"GGUF").unwrap();
    }
    ModelConfig {
        id: id.to_string(),
        enabled: true,
        use_mmap: None,
        moe_offload: Default::default(),
        modality: Modality::TextGeneration,
        additional_modalities: vec![],
        engine: EngineKind::GgmlLlama,
        weights_path: path,
        aux_weights_path: None,
        text_encoder_path: None,
        tokenizer_path: None,
        voices_path: None,
        total_weight_bytes: 2 * GB,
        min_gpu_layer_ratio: 0.25,
        initial_tier: MemoryTier::Tier3NvmeSsd,
        keep_warm_in_ram: true,
        priority: ModelPriority::Normal,
        slots: 1,
        prefix_cache: true,
        device: Default::default(),
        arch: ModelArchitectureParams {
            num_layers: 10,
            num_kv_heads: 0,
            head_dim: 128,
            default_context_len: 1024,
            max_context_len: 4096,
            sliding_window_len: None,
            global_attn_every_n_layers: None,
            kv_dtype: KvDataType::F16,
            non_layer_overhead_bytes: 64 * 1024 * 1024,
            activation_workspace_bytes: 64 * 1024 * 1024,
            expert_bytes: 0,
            global_kv_layers: None,
            global_num_kv_heads: None,
            swa_kv_layers: None,
            swa_num_kv_heads: None,
            swa_head_dim: None,
            configured_keys: None,
        },
    }
}

/// The real server state with one mock engine behind the real scheduler and memory manager.
fn state(engine: Arc<MockEngine>) -> AppState {
    state_with(vec![engine])
}

fn state_with(engines: Vec<Arc<MockEngine>>) -> AppState {
    let hw = HardwareConfig {
        accelerator: AcceleratorKind::Vulkan, // no probing of this machine's GPU
        device_id: 0,
        vram_budget_bytes: 12 * GB,
        vram_safety_margin_bytes: 256 * 1024 * 1024,
        pinned_ram_budget_bytes: 16 * GB,
        lock_host_memory: Some(false),
        simulate_physical_memory_bytes: 0,
        max_process_memory_bytes: 0,
        ..HardwareConfig::default()
    };
    let mut caps = DeviceMemoryCapabilities::detect(&hw);
    caps.physical_memory_bytes = 0;
    caps.installed_memory_bytes = 0;
    caps.pinned_ram_budget_bytes = hw.pinned_ram_budget_bytes;
    let memory = Arc::new(
        TieredMemoryManager::with_capabilities(
            &hw,
            caps,
            engines.iter().map(|e| e.config.clone()).collect(),
        )
        .unwrap(),
    );
    let limits = LimitsConfig::default();
    let scheduler = Arc::new(ModelScheduler::new(
        memory,
        engines
            .into_iter()
            .map(|e| e as Arc<dyn ModelBackend>)
            .collect(),
        &limits,
    ));
    let pictures = pictures::PictureStore::open(
        Some(
            &std::env::temp_dir().join(format!("hypnos-faults-{}", uuid::Uuid::new_v4().simple())),
        ),
        Duration::from_secs(60),
        0,
    )
    .unwrap();
    AppState {
        scheduler,
        limits,
        metrics: Arc::new(Metrics::default()),
        request_log: Arc::new(RequestLog::new(LogLevel::Info, 50)),
        auth: Arc::new(auth::ApiAuth::new(
            &[],
            &[],
            &ApiKeyLimits::default(),
            false,
        )),
        quota: Arc::new(quota::KeyLimiter::new([])),
        pictures: Arc::new(pictures),
        kv_park: Arc::new(hypnos_backends::kv_park::KvParkPool::disabled()),
        ready: Arc::new(AtomicBool::new(true)),
        started_at: Instant::now(),
        version: "test",
        catalog: hypnos_server::catalog::load(),
        models_dir: std::env::temp_dir(),
        pool: Arc::new(hypnos_server::pool::ModelPool::with_factory(
            &hypnos_core::config::HypnosConfig::from_toml_str("").unwrap(),
            Arc::new(|config| {
                Arc::new(MockEngine {
                    config,
                    resident: AtomicBool::new(false),
                    fail_loads: AtomicBool::new(false),
                    fail_generation: AtomicBool::new(false),
                    device_oom: AtomicUsize::new(0),
                    generations: AtomicUsize::new(0),
                    placements: Mutex::new(Vec::new()),
                    delay_ms: std::sync::atomic::AtomicU64::new(0),
                }) as Arc<dyn ModelBackend>
            }),
            false,
        )),
    }
}

async fn post(app: &axum::Router, path: &str, body: &str) -> axum::response::Response {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(req).await.expect("response")
}

async fn get(app: &axum::Router, path: &str) -> axum::response::Response {
    let req = Request::builder().uri(path).body(Body::empty()).unwrap();
    app.clone().oneshot(req).await.expect("response")
}

async fn text(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

async fn leases(app: &axum::Router, model: &str) -> (usize, String) {
    let status: serde_json::Value =
        serde_json::from_str(&text(get(app, "/v1/memory/status").await).await).unwrap();
    let m = status["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == model)
        .unwrap();
    (
        m["active_leases"].as_u64().unwrap() as usize,
        m["current_tier"].as_str().unwrap().to_string(),
    )
}

const CHAT: &str = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
const CHAT_STREAM: &str =
    r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"stream":true}"#;

#[tokio::test]
async fn a_refused_load_answers_500_and_is_counted_per_model() {
    let engine = Arc::new(MockEngine {
        config: model("m"),
        resident: AtomicBool::new(false),
        fail_loads: AtomicBool::new(true),
        fail_generation: AtomicBool::new(false),
        device_oom: AtomicUsize::new(0),
        generations: AtomicUsize::new(0),
        placements: Mutex::new(Vec::new()),
        delay_ms: std::sync::atomic::AtomicU64::new(0),
    });
    let app = build_router(state(engine.clone()), &[]);

    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert_eq!(body["error"]["type"], "server_error");
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("weights file changed"));
    // Rolled back: nothing leased, nothing booked in Tier 1, the engine was asked to unload.
    assert_eq!(leases(&app, "m").await, (0, "tier2_pinned_ram".to_string()));
    assert_eq!(engine.generations.load(Ordering::SeqCst), 0);
    let metrics = text(get(&app, "/metrics").await).await;
    assert!(
        metrics.contains("hypnos_model_load_failures_total{model=\"m\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("hypnos_process_start_time_seconds "),
        "{metrics}"
    );

    // The file is back: the next request loads and serves.
    engine.fail_loads.store(false, Ordering::SeqCst);
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));
}

#[tokio::test]
async fn a_failure_mid_generation_releases_the_lease_and_keeps_the_model() {
    let engine = Arc::new(MockEngine {
        config: model("m"),
        resident: AtomicBool::new(false),
        fail_loads: AtomicBool::new(false),
        fail_generation: AtomicBool::new(true),
        device_oom: AtomicUsize::new(0),
        generations: AtomicUsize::new(0),
        placements: Mutex::new(Vec::new()),
        delay_ms: std::sync::atomic::AtomicU64::new(0),
    });
    let app = build_router(state(engine.clone()), &[]);

    // Plain request: 500 in the OpenAI shape, lease released, model still resident.
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert!(body["error"]["message"]
        .as_str()
        .unwrap()
        .contains("out of memory"));
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));

    // Stream: headers already went out, so the failure is an error event, then [DONE].
    let resp = post(&app, "/v1/chat/completions", CHAT_STREAM).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let sse = text(resp).await;
    assert!(sse.contains("mock "), "{sse}");
    assert!(sse.contains("\"error\""), "{sse}");
    assert!(sse.contains("mid-generation"), "{sse}");
    assert!(sse.trim_end().ends_with("data: [DONE]"), "{sse}");
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));

    // The engine recovers: the same model serves the next request without a reload.
    engine.fail_generation.store(false, Ordering::SeqCst);
    let placements_before = engine.placements.lock().unwrap().len();
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "mock reply");
    assert_eq!(
        engine.placements.lock().unwrap().len(),
        placements_before,
        "no reload"
    );
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));
}

fn engine(id: &str) -> Arc<MockEngine> {
    Arc::new(MockEngine {
        config: model(id),
        resident: AtomicBool::new(false),
        fail_loads: AtomicBool::new(false),
        fail_generation: AtomicBool::new(false),
        device_oom: AtomicUsize::new(0),
        generations: AtomicUsize::new(0),
        placements: Mutex::new(Vec::new()),
        delay_ms: std::sync::atomic::AtomicU64::new(0),
    })
}

#[tokio::test]
async fn a_device_out_of_memory_evicts_idle_models_and_retries_once_then_answers_507() {
    let m = engine("m");
    let idle = engine("idle");
    let app = build_router(state_with(vec![m.clone(), idle.clone()]), &[]);
    let resp = post(
        &app,
        "/v1/chat/completions",
        &CHAT.replace(r#""model":"m""#, r#""model":"idle""#),
    )
    .await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        leases(&app, "idle").await,
        (0, "tier1_gpu_vram".to_string())
    );

    // The bookings fit but the engine finds the device full: the idle model goes, the retry serves.
    m.device_oom.store(1, Ordering::SeqCst);
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(m.generations.load(Ordering::SeqCst), 2);
    assert_eq!(
        leases(&app, "idle").await,
        (0, "tier2_pinned_ram".to_string())
    );
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));
    let status: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/status").await).await).unwrap();
    let eviction = status["recent_evictions"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["model"] == "idle")
        .expect("eviction recorded");
    assert_eq!(eviction["reason"], "device_out_of_memory");
    assert_eq!(eviction["for_model"], "m");

    // Nothing left to evict: no second attempt, 507 in the OpenAI shape, the model keeps serving.
    m.device_oom.store(1, Ordering::SeqCst);
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::INSUFFICIENT_STORAGE);
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert_eq!(body["error"]["type"], "device_out_of_memory");
    assert_eq!(m.generations.load(Ordering::SeqCst), 3);
    assert_eq!(leases(&app, "m").await, (0, "tier1_gpu_vram".to_string()));
    let resp = post(&app, "/v1/chat/completions", CHAT).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn kv_parking_is_off_by_default_and_switched_by_the_api() {
    let engine = Arc::new(MockEngine {
        config: model("m"),
        resident: AtomicBool::new(false),
        fail_loads: AtomicBool::new(false),
        fail_generation: AtomicBool::new(false),
        device_oom: AtomicUsize::new(0),
        generations: AtomicUsize::new(0),
        placements: Mutex::new(Vec::new()),
        delay_ms: std::sync::atomic::AtomicU64::new(0),
    });
    let app = build_router(state(engine), &[]);
    let status: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/kv-park").await).await).unwrap();
    assert_eq!(status["enabled"], false);
    assert_eq!(status["used_bytes"], 0);

    let on = post(
        &app,
        "/v1/memory/kv-park",
        r#"{"enabled": true, "max_bytes": 1073741824}"#,
    )
    .await;
    assert_eq!(on.status(), 200);
    let on: serde_json::Value = serde_json::from_str(&text(on).await).unwrap();
    assert_eq!(on["enabled"], true);
    assert_eq!(on["max_bytes"], 1073741824u64);

    let memory: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/status").await).await).unwrap();
    assert_eq!(memory["kv_park"]["enabled"], true);
    assert!(memory["kv_park"]["models"].as_array().unwrap().is_empty());

    let empty = post(&app, "/v1/memory/kv-park", "{}").await;
    assert_eq!(empty.status(), 400);
    let off: serde_json::Value = serde_json::from_str(
        &text(post(&app, "/v1/memory/kv-park", r#"{"enabled": false}"#).await).await,
    )
    .unwrap();
    assert_eq!(off["enabled"], false);
}

#[tokio::test]
async fn moe_offload_is_switched_per_model_by_the_api() {
    let engine = Arc::new(MockEngine {
        config: model("m"),
        resident: AtomicBool::new(false),
        fail_loads: AtomicBool::new(false),
        fail_generation: AtomicBool::new(false),
        device_oom: AtomicUsize::new(0),
        generations: AtomicUsize::new(0),
        placements: Mutex::new(Vec::new()),
        delay_ms: std::sync::atomic::AtomicU64::new(0),
    });
    let app = build_router(state(engine), &[]);
    let resp = post(&app, "/v1/models/m/moe-offload", r#"{"mode": "on"}"#).await;
    assert_eq!(resp.status(), 200);
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert_eq!(body["moe_offload"], "on");
    assert_eq!(
        body["applies"], "now",
        "not resident: the next load uses it"
    );
    let memory: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/status").await).await).unwrap();
    let m = memory["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "m")
        .unwrap()
        .clone();
    assert_eq!(m["moe_offload"], "on");
    assert_eq!(
        m["placement"]["cpu_expert_layers"], 0,
        "a dense model ignores the mode"
    );

    assert_eq!(
        post(&app, "/v1/models/m/moe-offload", r#"{"mode": "sideways"}"#)
            .await
            .status(),
        422
    );
    assert_eq!(
        post(&app, "/v1/models/nope/moe-offload", r#"{"mode": "on"}"#)
            .await
            .status(),
        404
    );
}

async fn send(
    app: &axum::Router,
    method: Method,
    path: &str,
    body: &str,
    key: Option<&str>,
) -> axum::response::Response {
    let mut req = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json");
    if let Some(k) = key {
        req = req.header(header::AUTHORIZATION, format!("Bearer {k}"));
    }
    app.clone()
        .oneshot(req.body(Body::from(body.to_string())).unwrap())
        .await
        .expect("response")
}

/// API-11: a model added, used, replaced and removed through the routes, no restart.
#[tokio::test]
async fn models_are_added_replaced_and_removed_through_the_api() {
    let app = build_router(state(engine("base")), &[]);
    let chat = |m: &str| {
        format!(r#"{{"model":"{m}","messages":[{{"role":"user","content":"hi"}}],"max_tokens":4}}"#)
    };
    let block = r#"[[models]]
id = "extra"
modality = "text_generation"
engine = "ggml_llama"
weights_path = "extra.gguf"
total_weight_bytes = 1073741824
[models.arch]
num_layers = 4
"#;
    let body = serde_json::json!({ "config": block }).to_string();
    let resp = send(&app, Method::POST, "/v1/models", &body, None).await;
    assert_eq!(resp.status(), 201, "{}", text(resp).await);
    let resp = send(&app, Method::POST, "/v1/models", &body, None).await;
    assert_eq!(resp.status(), 400, "the id is taken");

    let resp = post(&app, "/v1/chat/completions", &chat("extra")).await;
    assert_eq!(resp.status(), 200, "{}", text(resp).await);
    let models: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/models").await).await).unwrap();
    assert!(models["data"]
        .as_array()
        .unwrap()
        .iter()
        .any(|m| m["id"] == "extra"));

    // Replaced from the catalog under the same id.
    let resp = send(
        &app,
        Method::PUT,
        "/v1/models/extra",
        r#"{"catalog": "lfm3:1.7b:Q4_K_M"}"#,
        None,
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", text(resp).await);
    let status: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/status").await).await).unwrap();
    let extra = status["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["id"] == "extra")
        .unwrap()
        .clone();
    assert_eq!(extra["total_weight_bytes"], 1_099_627_808u64, "{extra}");
    assert_eq!(
        post(&app, "/v1/chat/completions", &chat("extra"))
            .await
            .status(),
        200
    );

    let resp = send(&app, Method::DELETE, "/v1/models/extra", "", None).await;
    assert_eq!(resp.status(), 200, "{}", text(resp).await);
    assert_eq!(
        post(&app, "/v1/chat/completions", &chat("extra"))
            .await
            .status(),
        404
    );
    assert_eq!(
        send(&app, Method::DELETE, "/v1/models/extra", "", None)
            .await
            .status(),
        404
    );
    let resp = send(
        &app,
        Method::POST,
        "/v1/models",
        r#"{"catalog": "nope"}"#,
        None,
    )
    .await;
    assert_eq!(resp.status(), 400);
}

#[tokio::test]
async fn changing_the_pool_needs_an_admin_key() {
    let mut st = state(engine("base"));
    st.auth = Arc::new(auth::ApiAuth::new(
        &["operator".to_string()],
        &[hypnos_core::config::ApiKeyProfile {
            label: "app".into(),
            key: "user-key".into(),
            requests_per_minute: None,
            max_concurrent: None,
            tokens_per_day: None,
            admin: false,
        }],
        &ApiKeyLimits::default(),
        false,
    ));
    let app = build_router(st, &[]);
    let resp = send(
        &app,
        Method::DELETE,
        "/v1/models/base",
        "",
        Some("user-key"),
    )
    .await;
    assert_eq!(resp.status(), 403);
    assert!(text(resp).await.contains("permission_denied"));
    let resp = send(
        &app,
        Method::DELETE,
        "/v1/models/base",
        "",
        Some("operator"),
    )
    .await;
    assert_eq!(resp.status(), 200, "{}", text(resp).await);
}

/// MEM-6: `x-hypnos-next-model` loads the next stage while the current request runs.
#[tokio::test]
async fn a_declared_next_model_is_loaded_in_the_background() {
    let (a, b) = (engine("stt"), engine("chat"));
    let app = build_router(state_with(vec![a.clone(), b.clone()]), &[]);
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-hypnos-next-model", "chat")
        .body(Body::from(
            r#"{"model":"stt","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#,
        ))
        .unwrap();
    assert_eq!(app.clone().oneshot(req).await.unwrap().status(), 200);
    for _ in 0..50 {
        if b.resident.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    assert!(
        b.resident.load(Ordering::SeqCst),
        "the next model was loaded"
    );
    assert!(a.resident.load(Ordering::SeqCst), "the current one too");
}

/// PERF-10: a deadline the estimated wait cannot meet is refused at once with Retry-After; the
/// estimate comes back on admitted requests; workflows get end-to-end percentiles.
#[tokio::test]
async fn deadlines_are_checked_at_admission_and_workflows_are_timed() {
    let st = state(engine("m"));
    let scheduler = st.scheduler.clone();
    let app = build_router(st, &[]);
    let chat = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let send_with = |headers: Vec<(&'static str, &'static str)>| {
        let app = app.clone();
        async move {
            let mut req = Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json");
            for (k, v) in headers {
                req = req.header(k, v);
            }
            app.oneshot(req.body(Body::from(chat)).unwrap())
                .await
                .unwrap()
        }
    };
    // Load the model, then pretend a request takes 500 ms and one is running.
    assert_eq!(send_with(vec![]).await.status(), 200);
    scheduler.memory().record_service("m", 500.0);
    for _ in 0..10 {
        scheduler.memory().record_service("m", 500.0);
    }
    let ctx = hypnos_core::request::RequestContext::unbounded();
    let running = scheduler.acquire_model("m", None, 1, &ctx).await.unwrap();

    let resp = send_with(vec![("x-hypnos-deadline-ms", "100")]).await;
    assert_eq!(resp.status(), 503);
    assert!(resp.headers().contains_key(header::RETRY_AFTER));
    assert!(text(resp).await.contains("deadline_unmeetable"));

    let resp = send_with(vec![("x-hypnos-deadline-ms", "5000")]).await;
    assert_eq!(resp.status(), 200);
    let wait: u64 = resp.headers()["x-hypnos-estimated-wait-ms"]
        .to_str()
        .unwrap()
        .parse()
        .unwrap();
    assert!(wait >= 400, "one request ahead of ~500 ms: {wait}");
    let _ = text(resp).await;
    drop(running);

    // A two-request workflow, closed by its last request.
    let r = send_with(vec![("x-hypnos-workflow", "voice_turn/abc")]).await;
    let _ = text(r).await;
    let r = send_with(vec![
        ("x-hypnos-workflow", "voice_turn/abc"),
        ("x-hypnos-workflow-end", "true"),
    ])
    .await;
    let _ = text(r).await;
    let status: serde_json::Value =
        serde_json::from_str(&text(get(&app, "/v1/memory/status").await).await).unwrap();
    assert_eq!(
        status["latency"]["workflows"]["voice_turn"]["count"], 1,
        "{}",
        status["latency"]
    );
    assert!(
        status["latency"]["routes"]["/v1/chat/completions"]["count"]
            .as_u64()
            .unwrap()
            >= 4
    );
    let metrics = text(get(&app, "/metrics").await).await;
    assert!(metrics.contains("hypnos_workflow_e2e_ms_count{workflow=\"voice_turn\"} 1"));
    assert!(metrics.contains("hypnos_request_e2e_ms_bucket{route=\"/v1/chat/completions\""));
}

/// TEST-4: hundreds of concurrent clients over three models that do not all fit, streaming and
/// not, cancelled at random moments (queued, loading, generating, mid-stream). Afterwards no
/// lease, admission slot or booking is left behind and every model still serves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_clients_and_cancellations_leave_nothing_behind() {
    use std::sync::atomic::AtomicU64;
    let engines: Vec<Arc<MockEngine>> = ["a", "b", "c"]
        .iter()
        .map(|id| {
            let mut config = model(id);
            config.total_weight_bytes = 5 * GB; // 12 GB holds two of the three
            config.slots = 2;
            config.min_gpu_layer_ratio = 1.0; // all or nothing: a third model must evict
            Arc::new(MockEngine {
                config,
                resident: AtomicBool::new(false),
                fail_loads: AtomicBool::new(false),
                fail_generation: AtomicBool::new(false),
                device_oom: AtomicUsize::new(0),
                generations: AtomicUsize::new(0),
                placements: Mutex::new(Vec::new()),
                delay_ms: AtomicU64::new(8),
            })
        })
        .collect();
    let st = state_with(engines.clone());
    let scheduler = st.scheduler.clone();
    let app = build_router(st, &[]);

    let mut tasks = Vec::new();
    for i in 0..300u64 {
        let app = app.clone();
        tasks.push(tokio::spawn(async move {
            // A cheap deterministic spread of models, modes and cancellation points.
            let id = ["a", "b", "c"][(i * 7 % 3) as usize];
            let stream = i % 2 == 0;
            let cancel_after_ms = match i % 5 {
                0 => None,          // runs to the end
                1 => Some(1),       // queued or loading
                2 => Some(10),      // loading or early generation
                3 => Some(25),      // mid-generation
                _ => Some(45),      // late, often mid-stream
            };
            let body = format!(
                r#"{{"model":"{id}","stream":{stream},"messages":[{{"role":"user","content":"hi"}}],"max_tokens":8}}"#
            );
            let req = Request::builder()
                .method(Method::POST)
                .uri("/v1/chat/completions")
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap();
            // Arrivals spread over ~120 ms, so models go idle and evict each other under load.
            tokio::time::sleep(Duration::from_millis(i * 7 % 120)).await;
            let call = async move {
                let resp = app.oneshot(req).await.unwrap();
                let status = resp.status();
                let _ = axum::body::to_bytes(resp.into_body(), 1 << 20).await;
                status
            };
            match cancel_after_ms {
                None => Some(call.await),
                Some(ms) => tokio::time::timeout(Duration::from_millis(ms), call)
                    .await
                    .ok(),
            }
        }));
    }
    let mut finished = 0;
    for t in tasks {
        if let Some(status) = t.await.expect("no panic") {
            // 429: the model's queue is full; 507: the other two models are both busy, so
            // neither can be evicted for the third (documented, retry later).
            assert!(
                status == StatusCode::OK
                    || status == StatusCode::TOO_MANY_REQUESTS
                    || status == StatusCode::INSUFFICIENT_STORAGE,
                "unexpected {status}"
            );
            finished += 1;
        }
    }
    assert!(
        finished >= 60,
        "the uncancelled fifth completes: {finished}"
    );

    // Cancelled requests unwind asynchronously; give them a moment.
    for _ in 0..100 {
        if ["a", "b", "c"]
            .iter()
            .all(|id| scheduler.memory().active_leases(id) == 0 && scheduler.in_flight(id) == 0)
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let status = scheduler.status();
    for m in &status.models {
        assert_eq!(m.active_leases, 0, "{} kept a lease", m.id);
        assert_eq!(
            scheduler.in_flight(&m.id),
            0,
            "{} kept an admission slot",
            m.id
        );
    }
    // What is booked is exactly what is resident.
    let resident: u64 = status
        .models
        .iter()
        .map(|m| m.placement.total_vram_bytes())
        .sum();
    assert_eq!(status.tier1_vram_used_bytes, resident);
    assert!(resident <= 12 * GB);
    let generated: usize = engines
        .iter()
        .map(|e| e.generations.load(Ordering::SeqCst))
        .sum();
    assert!(generated >= 60, "generations ran: {generated}");
    // Every model still serves.
    for id in ["a", "b", "c"] {
        let body = format!(
            r#"{{"model":"{id}","messages":[{{"role":"user","content":"hi"}}],"max_tokens":4}}"#
        );
        assert_eq!(
            post(&app, "/v1/chat/completions", &body).await.status(),
            200,
            "{id}"
        );
    }
    // Under sustained load on two models the third can starve (every request 507s while
    // neither is idle; see TODO TEST-5); served one after another, each evicts another.
    assert!(
        scheduler
            .stats()
            .evictions
            .load(std::sync::atomic::Ordering::Relaxed)
            > 0,
        "three models in room for two: serving each in turn evicted"
    );
}

/// A streamed chat with `tools` for a model without a tool-calling format is a 400 before the
/// stream starts (not an error event after a 200), so clients can fall back.
#[tokio::test]
async fn tools_for_a_model_without_a_tool_format_are_refused_before_streaming() {
    let app = build_router(state(engine("no-tools")), &[]);
    let body = r#"{"model":"no-tools","stream":true,"messages":[{"role":"user","content":"draw"}],
        "tools":[{"type":"function","function":{"name":"generate_image","parameters":{"type":"object"}}}]}"#;
    let resp = post(&app, "/v1/chat/completions", body).await;
    assert_eq!(resp.status(), 400);
    assert!(text(resp).await.contains("no tool-calling format"));
    let plain = r#"{"model":"no-tools","stream":true,"messages":[{"role":"user","content":"hi"}]}"#;
    assert_eq!(
        post(&app, "/v1/chat/completions", plain).await.status(),
        200
    );
}
