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
}

#[async_trait]
impl ModelBackend for MockEngine {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }
    async fn apply_placement(&self, plan: &LayerPlacementPlan, _ptr: Option<usize>) -> Result<()> {
        self.placements.lock().unwrap().push(plan.primary_tier);
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
        ready: Arc::new(AtomicBool::new(true)),
        started_at: Instant::now(),
        version: "test",
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
