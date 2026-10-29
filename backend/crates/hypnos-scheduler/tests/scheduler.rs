//! Engine-free tests of the scheduler: a mock backend records the placements it receives and can be
//! told to fail loads, so promotion, eviction order, rollback and admission control are checked
//! without any weights or GPU.

use async_trait::async_trait;
use hypnos_core::config::{
    AcceleratorKind, EngineKind, HardwareConfig, KvDataType, LimitsConfig, MemoryTier, Modality,
    ModelArchitectureParams, ModelConfig, ModelPriority,
};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use hypnos_memory::{EvictionReason, TieredMemoryManager};
use hypnos_scheduler::ModelScheduler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const GB: u64 = 1024 * 1024 * 1024;

struct MockBackend {
    config: ModelConfig,
    placements: Mutex<Vec<LayerPlacementPlan>>,
    fail_loads: AtomicBool,
    resident: AtomicBool,
}

impl MockBackend {
    fn new(config: ModelConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            placements: Mutex::new(Vec::new()),
            fail_loads: AtomicBool::new(false),
            resident: AtomicBool::new(false),
        })
    }
    fn tiers(&self) -> Vec<MemoryTier> {
        self.placements
            .lock()
            .unwrap()
            .iter()
            .map(|p| p.primary_tier)
            .collect()
    }
}

#[async_trait]
impl ModelBackend for MockBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }
    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }
    async fn apply_placement(&self, plan: &LayerPlacementPlan, _ptr: Option<usize>) -> Result<()> {
        self.placements.lock().unwrap().push(plan.clone());
        if plan.primary_tier == MemoryTier::Tier1GpuVram {
            if self.fail_loads.load(Ordering::SeqCst) {
                return Err(HypnosError::BackendError("simulated CUDA OOM".into()));
            }
            self.resident.store(true, Ordering::SeqCst);
        } else {
            self.resident.store(false, Ordering::SeqCst);
        }
        Ok(())
    }
}

fn model(id: &str, weight_gb: u64, priority: ModelPriority, initial: MemoryTier) -> ModelConfig {
    let dir = std::env::temp_dir().join("hypnos-sched-tests");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join(format!("{id}.bin"));
    if !path.exists() {
        std::fs::write(&path, b"GGUF").unwrap();
    }
    ModelConfig {
        id: id.to_string(),
        modality: Modality::TextGeneration,
        additional_modalities: vec![],
        engine: EngineKind::GgmlLlama,
        weights_path: path,
        aux_weights_path: None,
        text_encoder_path: None,
        tokenizer_path: None,
        voices_path: None,
        total_weight_bytes: weight_gb * GB,
        min_gpu_layer_ratio: 0.25,
        initial_tier: initial,
        keep_warm_in_ram: true,
        priority,
        slots: 1,
        prefix_cache: true,
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
        },
    }
}

fn hardware(vram_gb: u64) -> HardwareConfig {
    HardwareConfig {
        accelerator: AcceleratorKind::Vulkan, // no probing of this machine's GPU
        device_id: 0,
        vram_budget_bytes: vram_gb * GB,
        vram_safety_margin_bytes: 256 * 1024 * 1024,
        pinned_ram_budget_bytes: 64 * GB,
        lock_host_memory: Some(false),
        simulate_physical_memory_bytes: 0,
        max_process_memory_bytes: 0,
    }
}

fn build(
    vram_gb: u64,
    models: Vec<ModelConfig>,
    queue: usize,
) -> (Arc<ModelScheduler>, Vec<Arc<MockBackend>>) {
    let mocks: Vec<Arc<MockBackend>> = models.iter().cloned().map(MockBackend::new).collect();
    let backends: Vec<Arc<dyn ModelBackend>> = mocks
        .iter()
        .map(|m| m.clone() as Arc<dyn ModelBackend>)
        .collect();
    let memory = Arc::new(TieredMemoryManager::new(&hardware(vram_gb), models).unwrap());
    let limits = LimitsConfig {
        max_queue_per_model: queue,
        ..LimitsConfig::default()
    };
    (
        Arc::new(ModelScheduler::new(memory, backends, &limits)),
        mocks,
    )
}

#[tokio::test]
async fn promotion_loads_engine_and_lease_pins_it() {
    let (sched, mocks) = build(
        12,
        vec![model(
            "a",
            4,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        4,
    );
    let ctx = RequestContext::unbounded();
    let lease = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    assert_eq!(lease.previous_tier, MemoryTier::Tier3NvmeSsd);
    assert_eq!(lease.placement.primary_tier, MemoryTier::Tier1GpuVram);
    assert_eq!(lease.placement.gpu_layers, 10);
    assert!(mocks[0].is_resident());
    assert_eq!(sched.memory().active_leases("a"), 1);
    drop(lease);
    assert_eq!(sched.memory().active_leases("a"), 0);
    // Second acquisition is the fast path: no new placement callback.
    let n_before = mocks[0].placements.lock().unwrap().len();
    let lease2 = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    assert_eq!(lease2.tier_switch_latency_ms, 0.0);
    assert_eq!(mocks[0].placements.lock().unwrap().len(), n_before);
}

#[tokio::test]
async fn evicts_lowest_priority_idle_model_first() {
    let (sched, mocks) = build(
        12,
        vec![
            model("low", 4, ModelPriority::Low, MemoryTier::Tier3NvmeSsd),
            model("high", 4, ModelPriority::High, MemoryTier::Tier3NvmeSsd),
            model("big", 7, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
        ],
        4,
    );
    let ctx = RequestContext::unbounded();
    drop(sched.acquire_model("high", None, 1, &ctx).await.unwrap());
    drop(sched.acquire_model("low", None, 1, &ctx).await.unwrap());
    // ~8.1 GB booked of 11.75 usable; "big" needs ~7.1 GB -> evicting the low-priority model suffices.
    let lease = sched.acquire_model("big", None, 1, &ctx).await.unwrap();
    assert_eq!(lease.evicted_models, vec!["low".to_string()]);
    assert!(!mocks[0].is_resident(), "low should have been released");
    assert!(mocks[1].is_resident(), "high priority should stay resident");
    assert_eq!(
        mocks[0].tiers().last().copied(),
        Some(MemoryTier::Tier2PinnedRam)
    );

    // The eviction is listed in /v1/memory/status with the model it made room for and why.
    let events = sched.status().recent_evictions;
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].model, "low");
    assert_eq!(events[0].for_model, "big");
    assert_eq!(events[0].to_tier, MemoryTier::Tier2PinnedRam);
    // "big" would fit with a partial GPU split; the eviction buys it a full one.
    assert_eq!(events[0].reason, EvictionReason::FullGpuPlacement);
}

#[tokio::test]
async fn pinned_models_are_never_evicted_and_insufficient_memory_is_reported() {
    let (sched, _mocks) = build(
        8,
        vec![
            model(
                "pinned",
                4,
                ModelPriority::RealtimePinned,
                MemoryTier::Tier3NvmeSsd,
            ),
            model("huge", 20, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
        ],
        4,
    );
    let ctx = RequestContext::unbounded();
    drop(sched.acquire_model("pinned", None, 1, &ctx).await.unwrap());
    let err = sched
        .acquire_model("huge", None, 1, &ctx)
        .await
        .err()
        .unwrap();
    assert!(
        matches!(err, HypnosError::InsufficientMemory { .. }),
        "{err}"
    );
    assert_eq!(
        sched
            .status()
            .models
            .iter()
            .find(|m| m.id == "pinned")
            .unwrap()
            .current_tier,
        MemoryTier::Tier1GpuVram
    );
}

#[tokio::test]
async fn failed_engine_load_rolls_back_accounting() {
    let (sched, mocks) = build(
        12,
        vec![model(
            "flaky",
            4,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        4,
    );
    mocks[0].fail_loads.store(true, Ordering::SeqCst);
    let ctx = RequestContext::unbounded();
    let err = sched
        .acquire_model("flaky", None, 1, &ctx)
        .await
        .err()
        .unwrap();
    assert!(matches!(err, HypnosError::BackendError(_)), "{err}");
    let st = sched.status();
    assert_eq!(
        st.tier1_vram_used_bytes, 0,
        "VRAM booking must be rolled back"
    );
    assert_eq!(st.models[0].current_tier, MemoryTier::Tier2PinnedRam);
    assert_eq!(sched.stats().load_failures.load(Ordering::Relaxed), 1);
    // Recovers once the engine works again.
    mocks[0].fail_loads.store(false, Ordering::SeqCst);
    assert!(sched.acquire_model("flaky", None, 1, &ctx).await.is_ok());
}

#[tokio::test]
async fn admission_is_bounded_per_model() {
    let (sched, _mocks) = build(
        12,
        vec![model(
            "a",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        2,
    );
    let ctx = RequestContext::unbounded();
    let l1 = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    let l2 = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    let err = sched.acquire_model("a", None, 1, &ctx).await.err().unwrap();
    assert!(matches!(err, HypnosError::Busy { .. }), "{err}");
    assert_eq!(sched.in_flight("a"), 2);
    drop(l1);
    assert!(sched.acquire_model("a", None, 1, &ctx).await.is_ok());
    drop(l2);
}

#[tokio::test]
async fn cancelled_request_is_refused_before_loading() {
    let (sched, mocks) = build(
        12,
        vec![model(
            "a",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        2,
    );
    let ctx = RequestContext::unbounded();
    ctx.cancel();
    let err = sched.acquire_model("a", None, 1, &ctx).await.err().unwrap();
    assert!(matches!(err, HypnosError::Cancelled));
    assert!(mocks[0].placements.lock().unwrap().is_empty());
}

#[tokio::test]
async fn startup_tiers_and_shutdown_release_engines() {
    let (sched, mocks) = build(
        12,
        vec![
            model(
                "hot",
                2,
                ModelPriority::RealtimePinned,
                MemoryTier::Tier1GpuVram,
            ),
            model("warm", 2, ModelPriority::Normal, MemoryTier::Tier2PinnedRam),
        ],
        2,
    );
    sched.initialize_startup_tiers().await.unwrap();
    assert!(mocks[0].is_resident());
    assert_eq!(mocks[1].tiers(), vec![MemoryTier::Tier2PinnedRam]);
    sched.shutdown().await;
    assert!(!mocks[0].is_resident());
    let st = sched.status();
    assert_eq!(st.tier1_vram_used_bytes, 0);
    assert_eq!(st.tier2_pinned_ram_used_bytes, 0);
}

#[tokio::test]
async fn manual_demotion_is_refused_while_leased() {
    let (sched, _mocks) = build(
        12,
        vec![model(
            "a",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        2,
    );
    let ctx = RequestContext::unbounded();
    let lease = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    let err = sched
        .set_model_tier("a", MemoryTier::Tier2PinnedRam, None)
        .await
        .err()
        .unwrap();
    assert!(matches!(err, HypnosError::Busy { .. }));
    drop(lease);
    assert!(sched
        .set_model_tier("a", MemoryTier::Tier2PinnedRam, None)
        .await
        .is_ok());
}
