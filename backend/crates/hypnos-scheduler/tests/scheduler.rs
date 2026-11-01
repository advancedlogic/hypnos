//! Engine-free tests of the scheduler: a mock backend records the placements it receives and can be
//! told to fail loads, so promotion, eviction order, rollback and admission control are checked
//! without any weights or GPU.

use async_trait::async_trait;
use hypnos_core::config::{
    AcceleratorKind, EngineKind, ExecutionDevice, HardwareConfig, KvDataType, LimitsConfig,
    MemoryTier, Modality, ModelArchitectureParams, ModelConfig, ModelPriority,
};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use hypnos_memory::{DeviceMemoryCapabilities, EvictionReason, TieredMemoryManager};
use hypnos_scheduler::ModelScheduler;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

const GB: u64 = 1024 * 1024 * 1024;

struct MockBackend {
    config: ModelConfig,
    placements: Mutex<Vec<LayerPlacementPlan>>,
    fail_loads: AtomicBool,
    resident: AtomicBool,
    /// When set, a Tier 1 load waits for `gate` before it completes (a slow engine load).
    block_loads: AtomicBool,
    gate: tokio::sync::Notify,
}

impl MockBackend {
    fn new(config: ModelConfig) -> Arc<Self> {
        Arc::new(Self {
            config,
            placements: Mutex::new(Vec::new()),
            fail_loads: AtomicBool::new(false),
            resident: AtomicBool::new(false),
            block_loads: AtomicBool::new(false),
            gate: tokio::sync::Notify::new(),
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
            if self.block_loads.load(Ordering::SeqCst) {
                self.gate.notified().await;
            }
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
        total_weight_bytes: weight_gb * GB,
        min_gpu_layer_ratio: 0.25,
        initial_tier: initial,
        keep_warm_in_ram: true,
        priority,
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

fn hardware(vram_gb: u64, ram_gb: u64) -> HardwareConfig {
    HardwareConfig {
        accelerator: AcceleratorKind::Vulkan, // no probing of this machine's GPU
        device_id: 0,
        vram_budget_bytes: vram_gb * GB,
        vram_safety_margin_bytes: 256 * 1024 * 1024,
        pinned_ram_budget_bytes: ram_gb * GB,
        lock_host_memory: Some(false),
        simulate_physical_memory_bytes: 0,
        max_process_memory_bytes: 0,
        kv_park: false,
        kv_park_bytes: 0,
    }
}

fn build(
    vram_gb: u64,
    models: Vec<ModelConfig>,
    queue: usize,
) -> (Arc<ModelScheduler>, Vec<Arc<MockBackend>>) {
    build_with_ram(vram_gb, 64, models, queue)
}

fn build_with_ram(
    vram_gb: u64,
    ram_gb: u64,
    models: Vec<ModelConfig>,
    queue: usize,
) -> (Arc<ModelScheduler>, Vec<Arc<MockBackend>>) {
    let mocks: Vec<Arc<MockBackend>> = models.iter().cloned().map(MockBackend::new).collect();
    let backends: Vec<Arc<dyn ModelBackend>> = mocks
        .iter()
        .map(|m| m.clone() as Arc<dyn ModelBackend>)
        .collect();
    // Physical RAM "unknown": Tier 2 keeps its configured budget instead of being capped by the
    // RAM of this machine (a 7 GB CI runner leaves too little for the evictions tested here).
    let hw = hardware(vram_gb, ram_gb);
    let mut caps = DeviceMemoryCapabilities::detect(&hw);
    caps.physical_memory_bytes = 0;
    caps.installed_memory_bytes = 0;
    caps.pinned_ram_budget_bytes = hw.pinned_ram_budget_bytes;
    let memory = Arc::new(TieredMemoryManager::with_capabilities(&hw, caps, models).unwrap());
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
async fn cancelled_while_another_load_holds_the_transition_lock() {
    // TEST-2: a request that gives up while it waits for the transition lock must not load its
    // model once the lock frees up; the one holding the lock finishes normally.
    let (sched, mocks) = build(
        12,
        vec![
            model("slow", 2, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
            model("other", 2, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
        ],
        2,
    );
    mocks[0].block_loads.store(true, Ordering::SeqCst);
    let slow = {
        let sched = sched.clone();
        tokio::spawn(async move {
            let ctx = RequestContext::unbounded();
            sched
                .acquire_model("slow", None, 1, &ctx)
                .await
                .map(|l| l.placement.primary_tier)
        })
    };
    // Let the slow load take the lock, then queue a request that is cancelled while it waits.
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    let ctx = RequestContext::unbounded();
    let waiting = {
        let (sched, ctx) = (sched.clone(), ctx.clone());
        tokio::spawn(async move { sched.acquire_model("other", None, 1, &ctx).await.err() })
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    ctx.cancel();
    mocks[0].gate.notify_one();
    assert_eq!(slow.await.unwrap().unwrap(), MemoryTier::Tier1GpuVram);
    assert!(matches!(
        waiting.await.unwrap(),
        Some(HypnosError::Cancelled)
    ));
    assert!(
        mocks[1].placements.lock().unwrap().is_empty(),
        "the cancelled request loaded nothing"
    );
    assert_eq!(sched.memory().active_leases("other"), 0);
}

#[tokio::test]
async fn no_fast_path_lease_while_the_engine_is_still_loading() {
    // BUG-2: the booking says Tier 1 as soon as the load starts; a second request must wait for
    // the engine to be resident instead of running on it (GLiNER answered "not loaded").
    let (sched, mocks) = build(
        12,
        vec![model(
            "a",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        4,
    );
    mocks[0].block_loads.store(true, Ordering::SeqCst);
    let first = {
        let sched = sched.clone();
        tokio::spawn(async move {
            let ctx = RequestContext::unbounded();
            sched.acquire_model("a", None, 1, &ctx).await.map(|_| ())
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    // The booking already says Tier 1 (what the fast path used to trust)...
    let booked = sched.memory().get_model_state("a").unwrap().1;
    assert_eq!(booked.primary_tier, MemoryTier::Tier1GpuVram);
    assert!(!mocks[0].is_resident());
    // ...but a second request does not get a lease until the engine is resident.
    let second = {
        let sched = sched.clone();
        tokio::spawn(async move {
            let ctx = RequestContext::unbounded();
            let lease = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
            lease.backend.is_resident()
        })
    };
    tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    assert!(
        !second.is_finished(),
        "the second request waited for the load"
    );
    mocks[0].gate.notify_one();
    first.await.unwrap().unwrap();
    assert!(
        second.await.unwrap(),
        "served only once the engine was resident"
    );
    assert_eq!(sched.memory().active_leases("a"), 0);
}

#[tokio::test]
async fn load_failures_are_counted_per_model() {
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
    mocks[0].fail_loads.store(true, Ordering::SeqCst);
    let ctx = RequestContext::unbounded();
    assert!(sched.acquire_model("a", None, 1, &ctx).await.is_err());
    assert!(sched.acquire_model("a", None, 1, &ctx).await.is_err());
    assert_eq!(
        sched.stats().load_failures_by_model(),
        vec![("a".to_string(), 2)]
    );
    assert_eq!(sched.stats().load_failures.load(Ordering::Relaxed), 2);
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
async fn disabled_models_are_skipped_refused_and_evicted() {
    // `enabled = false` in the config: not pre-staged, refused, invisible to tier moves.
    let mut off = model("off", 2, ModelPriority::High, MemoryTier::Tier1GpuVram);
    off.enabled = false;
    let (sched, mocks) = build(
        12,
        vec![
            off,
            model("on", 2, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
        ],
        2,
    );
    sched.initialize_startup_tiers().await.unwrap();
    assert!(
        !mocks[0].is_resident(),
        "a disabled model is not pre-loaded"
    );
    assert!(mocks[0].tiers().is_empty());
    let ctx = RequestContext::unbounded();
    let err = sched
        .acquire_model("off", None, 1, &ctx)
        .await
        .err()
        .unwrap();
    assert!(matches!(err, HypnosError::ModelDisabled(ref id) if id == "off"));
    let err = sched
        .set_model_tier("off", MemoryTier::Tier2PinnedRam, None)
        .await
        .err()
        .unwrap();
    assert!(matches!(err, HypnosError::ModelDisabled(_)));
    let st = sched.status();
    let snap = |id: &str| st.models.iter().find(|m| m.id == id).unwrap().clone();
    assert!(!snap("off").enabled);
    assert!(snap("on").enabled);

    // Enabling at runtime makes it loadable again.
    sched.set_model_enabled("off", true).await.unwrap();
    assert!(sched.is_enabled("off").unwrap());
    let lease = sched.acquire_model("off", None, 1, &ctx).await.unwrap();
    assert!(mocks[0].is_resident());

    // Disabling is refused while leased, then evicts to Tier 3 and frees the bookings.
    let err = sched.set_model_enabled("off", false).await.err().unwrap();
    assert!(matches!(err, HypnosError::Busy { .. }));
    drop(lease);
    let plan = sched.set_model_enabled("off", false).await.unwrap();
    assert_eq!(plan.primary_tier, MemoryTier::Tier3NvmeSsd);
    assert!(!mocks[0].is_resident());
    assert_eq!(mocks[0].tiers().last(), Some(&MemoryTier::Tier3NvmeSsd));
    assert_eq!(sched.status().tier1_vram_used_bytes, 0);
    assert_eq!(sched.status().tier2_pinned_ram_used_bytes, 0);
    assert!(matches!(
        sched
            .acquire_model("off", None, 1, &ctx)
            .await
            .err()
            .unwrap(),
        HypnosError::ModelDisabled(_)
    ));
    // Unknown ids are still "not found", enabled or not.
    assert!(matches!(
        sched.set_model_enabled("nope", true).await.err().unwrap(),
        HypnosError::ModelNotFound(_)
    ));
}

#[tokio::test]
async fn eviction_skips_a_model_that_was_leased_meanwhile() {
    // BUG-1: the fast-path lease and the eviction decide under one lock.
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
    let memory = sched.memory();
    // Not resident: no fast-path lease.
    assert!(memory
        .increment_lease_if_resident("a", 1024)
        .unwrap()
        .is_none());
    let lease = sched.acquire_model("a", None, 1, &ctx).await.unwrap();
    assert!(mocks[0].is_resident());
    // Leased: an eviction decided now must skip it, and the engine stays loaded.
    assert!(memory
        .demote_if_idle("a", MemoryTier::Tier3NvmeSsd)
        .unwrap()
        .is_none());
    assert_eq!(memory.active_leases("a"), 1);
    // Resident with a large enough window: the fast path counts a second lease atomically.
    let plan = memory
        .increment_lease_if_resident("a", 512)
        .unwrap()
        .expect("resident");
    assert_eq!(plan.primary_tier, MemoryTier::Tier1GpuVram);
    assert_eq!(memory.active_leases("a"), 2);
    memory.decrement_lease("a");
    // A window larger than what is allocated takes the slow path.
    assert!(memory
        .increment_lease_if_resident("a", 1 << 20)
        .unwrap()
        .is_none());
    drop(lease);
    // Idle: the demotion goes through.
    let demoted = memory
        .demote_if_idle("a", MemoryTier::Tier3NvmeSsd)
        .unwrap()
        .expect("idle");
    assert_eq!(demoted.primary_tier, MemoryTier::Tier3NvmeSsd);
    assert_eq!(sched.status().tier1_vram_used_bytes, 0);
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

#[tokio::test]
async fn cpu_model_books_ram_and_frees_it_from_idle_warm_models() {
    let mut cpu = model("cpu", 1, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd);
    cpu.device = ExecutionDevice::Cpu;
    cpu.arch.activation_workspace_bytes = 4 * GB;
    // 10 GB of RAM, 8 of them held by two idle warm models: the CPU model (1 + 4 GB) needs one gone.
    let (sched, mocks) = build_with_ram(
        12,
        10,
        vec![
            model(
                "warm-low",
                4,
                ModelPriority::Low,
                MemoryTier::Tier2PinnedRam,
            ),
            model(
                "warm-high",
                4,
                ModelPriority::High,
                MemoryTier::Tier2PinnedRam,
            ),
            cpu,
        ],
        4,
    );
    sched.initialize_startup_tiers().await.unwrap();
    assert_eq!(sched.memory().used_pinned_ram_bytes(), 8 * GB);

    let ctx = RequestContext::unbounded();
    let lease = sched.acquire_model("cpu", None, 1, &ctx).await.unwrap();
    assert_eq!(lease.placement.primary_tier, MemoryTier::Tier1GpuVram);
    assert_eq!(
        lease.placement.total_vram_bytes(),
        0,
        "a CPU model books no VRAM"
    );
    assert_eq!(lease.placement.gpu_layers, 0);
    assert_eq!(lease.placement.pinned_ram_bytes, 5 * GB);
    assert_eq!(lease.evicted_models, vec!["warm-low".to_string()]);
    assert_eq!(
        mocks[0].tiers().last().copied(),
        Some(MemoryTier::Tier3NvmeSsd)
    );
    assert!(mocks[2].is_resident());
    assert_eq!(sched.memory().used_vram_bytes(), 0);
    drop(lease);

    // Resident CPU models take the fast path like GPU ones.
    let lease = sched.acquire_model("cpu", None, 1, &ctx).await.unwrap();
    assert_eq!(lease.tier_switch_latency_ms, 0.0);
}

// ---- runtime model pool (API-11) ------------------------------------------------------------

#[tokio::test]
async fn models_are_added_served_and_removed_at_runtime() {
    let (sched, _) = build(
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
    let b = MockBackend::new(model(
        "b",
        2,
        ModelPriority::Normal,
        MemoryTier::Tier3NvmeSsd,
    ));
    sched.add_model(b.clone()).await.unwrap();
    assert!(sched.memory().model_ids().contains(&"b".to_string()));
    let lease = sched.acquire_model("b", None, 1, &ctx).await.unwrap();
    assert!(b.is_resident(), "the first request loads an added model");
    drop(lease);

    let dup = MockBackend::new(model(
        "a",
        1,
        ModelPriority::Normal,
        MemoryTier::Tier3NvmeSsd,
    ));
    assert!(matches!(
        sched.add_model(dup).await,
        Err(HypnosError::InvalidRequest(_))
    ));

    sched
        .remove_model("b", std::time::Duration::from_secs(1))
        .await
        .unwrap();
    assert!(!b.is_resident(), "removal unloads the engine");
    assert_eq!(b.tiers().last(), Some(&MemoryTier::Tier3NvmeSsd));
    assert!(sched.backend("b").is_none());
    assert!(matches!(
        sched.acquire_model("b", None, 1, &ctx).await,
        Err(HypnosError::ModelNotFound(_))
    ));
    // The VRAM it booked is free again.
    assert_eq!(sched.status().tier1_vram_used_bytes, 0);
}

#[tokio::test]
async fn removal_waits_for_running_requests_or_gives_up() {
    let (sched, _) = build(
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

    // Too short a drain: refused as busy, the model stays.
    assert!(matches!(
        sched
            .remove_model("a", std::time::Duration::from_millis(100))
            .await,
        Err(HypnosError::Busy { .. })
    ));
    assert!(sched.backend("a").is_some());

    let s = sched.clone();
    let removal =
        tokio::spawn(async move { s.remove_model("a", std::time::Duration::from_secs(5)).await });
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;
    assert!(!removal.is_finished(), "waits for the running request");
    drop(lease);
    removal.await.unwrap().unwrap();
    assert!(sched.backend("a").is_none());
}

#[tokio::test]
async fn a_replacement_serves_waiting_requests_with_the_new_engine() {
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
    let old_lease = sched.acquire_model("a", None, 1, &ctx).await.unwrap();

    let new = MockBackend::new(model(
        "a",
        2,
        ModelPriority::Normal,
        MemoryTier::Tier3NvmeSsd,
    ));
    let s = sched.clone();
    let n = new.clone();
    let replacement =
        tokio::spawn(async move { s.replace_model(n, std::time::Duration::from_secs(5)).await });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    // A request arriving during the replacement waits for it rather than failing.
    let s = sched.clone();
    let waiting = tokio::spawn(async move {
        let ctx = RequestContext::unbounded();
        s.acquire_model("a", None, 1, &ctx)
            .await
            .map(|l| l.backend.config().total_weight_bytes)
    });
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    assert!(!replacement.is_finished() && !waiting.is_finished());

    drop(old_lease);
    replacement.await.unwrap().unwrap();
    assert_eq!(
        waiting.await.unwrap().unwrap(),
        2 * GB,
        "served by the new engine"
    );
    assert!(!mocks[0].is_resident(), "the old engine was unloaded");
    assert!(new.is_resident());
}

// ---- workload-aware scheduling (MEM-6) -----------------------------------------------------

#[tokio::test]
async fn eviction_keeps_the_model_that_is_asked_for_more_often() {
    let (sched, mocks) = build(
        12,
        vec![
            model("busy", 4, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
            model("rare", 4, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
            model("new", 6, ModelPriority::Normal, MemoryTier::Tier3NvmeSsd),
        ],
        4,
    );
    let ctx = RequestContext::unbounded();
    for _ in 0..5 {
        drop(sched.acquire_model("busy", None, 1, &ctx).await.unwrap());
    }
    // `rare` is used last: least-recently-used alone would evict `busy`.
    drop(sched.acquire_model("rare", None, 1, &ctx).await.unwrap());
    let lease = sched.acquire_model("new", None, 1, &ctx).await.unwrap();
    assert_eq!(lease.evicted_models, vec!["rare".to_string()]);
    assert!(mocks[0].is_resident(), "the busy model stays");
    let status = sched.status();
    let busy = status.models.iter().find(|m| m.id == "busy").unwrap();
    assert!(busy.requests_per_minute > 0.0 && busy.reload_estimate_ms > 0.0);
}

#[tokio::test]
async fn a_declared_next_stage_is_loaded_ahead() {
    let (sched, mocks) = build(
        12,
        vec![model(
            "next",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        4,
    );
    assert!(sched.prestage("next").await.unwrap());
    assert!(mocks[0].is_resident());
    assert_eq!(sched.memory().active_leases("next"), 0, "released at once");
    assert!(!sched.prestage("next").await.unwrap(), "already resident");
    let ctx = RequestContext::unbounded();
    let lease = sched.acquire_model("next", None, 1, &ctx).await.unwrap();
    assert_eq!(
        lease.tier_switch_latency_ms, 0.0,
        "the request takes the fast path"
    );
    assert!(sched.prestage("unknown").await.is_err());
}

// ---- deadline-aware admission (PERF-10) ----------------------------------------------------

#[tokio::test]
async fn a_request_that_cannot_start_in_time_is_refused_at_once() {
    let (sched, _) = build(
        12,
        vec![model(
            "m",
            2,
            ModelPriority::Normal,
            MemoryTier::Tier3NvmeSsd,
        )],
        4,
    );
    let ctx = RequestContext::unbounded();
    assert_eq!(
        sched.estimated_wait_ms("m").map(|w| w > 0.0),
        Some(true),
        "cold: the load is the wait"
    );
    // One request held for ~100 ms measures the service time.
    let lease = sched.acquire_model("m", None, 1, &ctx).await.unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    drop(lease);
    assert_eq!(sched.estimated_wait_ms("m"), Some(0.0), "resident and free");

    let busy = sched.acquire_model("m", None, 1, &ctx).await.unwrap();
    let wait = sched.estimated_wait_ms("m").unwrap();
    assert!(wait >= 90.0, "one request ahead on one slot: {wait}");

    let mut hurried = RequestContext::unbounded();
    hurried.start_deadline_ms = Some(10);
    match sched.acquire_model("m", None, 1, &hurried).await {
        Err(HypnosError::DeadlineUnmeetable {
            estimated_ms,
            deadline_ms,
            ..
        }) => assert!(estimated_ms >= 90 && deadline_ms == 10),
        Err(e) => panic!("unexpected error: {e}"),
        Ok(_) => panic!("admitted past its deadline"),
    }
    assert!(
        hurried
            .estimated_wait_ms
            .load(std::sync::atomic::Ordering::Relaxed)
            >= 90,
        "the estimate is handed back to the caller"
    );
    let mut patient = RequestContext::unbounded();
    patient.start_deadline_ms = Some(5000);
    assert!(sched.acquire_model("m", None, 1, &patient).await.is_ok());
    drop(busy);
}
