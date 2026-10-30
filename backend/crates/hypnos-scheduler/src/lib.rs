//! KV-cache-aware multi-modal model scheduler.
//!
//! Promotes models from Tier 3 (SSD) or Tier 2 (pinned RAM) into Tier 1 (GPU VRAM), co-locates
//! several models when VRAM permits, splits layers between GPU and host RAM, evicts idle
//! lower-priority models, and bounds admission per model. Every promotion is reconciled against
//! the device's real free memory when a [`GpuMemoryProbe`](hypnos_core::GpuMemoryProbe) is set.

use hypnos_core::config::{LimitsConfig, MemoryTier};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::{KvCacheEstimator, LayerPlacementPlan};
use hypnos_core::openai::{EngineTiming, ExecutionTelemetry, LoadPhases};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use hypnos_memory::{EvictionReason, MemoryHierarchyStatus, TieredMemoryManager};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore};
use tracing::{info, warn};

/// RAII lease guard ensuring an executing model is never evicted from Tier 1 VRAM
/// while a request (or streaming response) is in flight. Also holds the model's admission slot.
pub struct ActiveModelLease {
    model_id: String,
    memory: Arc<TieredMemoryManager>,
    pub backend: Arc<dyn ModelBackend>,
    pub previous_tier: MemoryTier,
    pub placement: LayerPlacementPlan,
    pub tier_switch_latency_ms: f64,
    pub evicted_models: Vec<String>,
    /// Where the switch spent its time when this lease loaded the model (llama.cpp engines).
    pub load_phases: Option<LoadPhases>,
    _admission: OwnedSemaphorePermit,
}

impl ActiveModelLease {
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn build_telemetry(&self, inference_latency_ms: f64) -> ExecutionTelemetry {
        ExecutionTelemetry {
            previous_tier: self.previous_tier,
            placement: self.placement.clone(),
            tier_switch_latency_ms: self.tier_switch_latency_ms,
            inference_latency_ms,
            evicted_models: self.evicted_models.clone(),
            engine: EngineTiming::default(),
            tier_switch_phases: self.load_phases.clone(),
        }
    }
}

impl Drop for ActiveModelLease {
    fn drop(&mut self) {
        self.memory.decrement_lease(&self.model_id);
    }
}

/// Plans a Tier 1 placement that serves `needed_ctx`, preferring the model's `default_context_len`
/// whenever that fits with the same GPU split. The engines keep one context per residency, so a
/// window sized for the default serves a growing conversation without being rebuilt (and losing
/// its prefix cache) every time a request's estimate exceeds the previous one.
fn plan_placement(
    cfg: &hypnos_core::config::ModelConfig,
    needed_ctx: u32,
    batch_size: u32,
    avail_vram: u64,
    avail_ram: u64,
    is_uma: bool,
) -> Option<LayerPlacementPlan> {
    let plan = |ctx: u32| {
        KvCacheEstimator::compute_optimal_placement(
            cfg,
            Some(ctx),
            batch_size,
            avail_vram,
            avail_ram,
            is_uma,
        )
    };
    let preferred_ctx = needed_ctx
        .max(cfg.arch.default_context_len)
        .min(cfg.arch.max_context_len);
    if preferred_ctx == needed_ctx {
        return plan(needed_ctx);
    }
    match (plan(preferred_ctx), plan(needed_ctx)) {
        (Some(preferred), Some(needed)) if preferred.gpu_layers >= needed.gpu_layers => {
            Some(preferred)
        }
        (preferred, None) => preferred,
        (_, needed) => needed,
    }
}

/// Counters exposed on `/metrics`.
#[derive(Debug, Default)]
pub struct SchedulerStats {
    pub promotions: AtomicU64,
    pub evictions: AtomicU64,
    pub load_failures: AtomicU64,
    pub admissions_rejected: AtomicU64,
    pub tier_switch_ms_total: AtomicU64,
    /// Load failures per model (`hypnos_model_load_failures_total{model}`): an engine that refused
    /// its weights (file gone or changed, CUDA out of memory) shows up here, not only in the log.
    pub load_failures_by_model: std::sync::Mutex<HashMap<String, u64>>,
}

impl SchedulerStats {
    pub fn load_failures_by_model(&self) -> Vec<(String, u64)> {
        let mut v: Vec<(String, u64)> = self
            .load_failures_by_model
            .lock()
            .map(|m| m.iter().map(|(k, n)| (k.clone(), *n)).collect())
            .unwrap_or_default();
        v.sort();
        v
    }
}

/// KV-Cache-Aware Multi-Modal Model Scheduler.
pub struct ModelScheduler {
    memory: Arc<TieredMemoryManager>,
    backends: HashMap<String, Arc<dyn ModelBackend>>,
    /// Per-model admission: bounds queued + running requests (HTTP 429 beyond the limit).
    admission: HashMap<String, Arc<Semaphore>>,
    max_queue_per_model: usize,
    /// Serializes tier promotion/eviction transitions while allowing concurrent execution
    /// on already-resident models.
    transition_lock: Mutex<()>,
    stats: SchedulerStats,
}

impl ModelScheduler {
    pub fn new(
        memory: Arc<TieredMemoryManager>,
        backends: Vec<Arc<dyn ModelBackend>>,
        limits: &LimitsConfig,
    ) -> Self {
        let max_queue_per_model = limits.max_queue_per_model.max(1);
        let mut backend_map = HashMap::new();
        let mut admission = HashMap::new();
        for b in backends {
            let id = b.config().id.clone();
            admission.insert(id.clone(), Arc::new(Semaphore::new(max_queue_per_model)));
            backend_map.insert(id, b);
        }
        Self {
            memory,
            backends: backend_map,
            admission,
            max_queue_per_model,
            transition_lock: Mutex::new(()),
            stats: SchedulerStats::default(),
        }
    }

    pub fn memory(&self) -> &Arc<TieredMemoryManager> {
        &self.memory
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    pub fn status(&self) -> MemoryHierarchyStatus {
        self.memory.status_snapshot()
    }

    pub fn backend(&self, model_id: &str) -> Option<Arc<dyn ModelBackend>> {
        self.backends.get(model_id).cloned()
    }

    /// Number of requests currently queued or running for `model_id`.
    pub fn in_flight(&self, model_id: &str) -> usize {
        self.admission
            .get(model_id)
            .map(|s| self.max_queue_per_model - s.available_permits())
            .unwrap_or(0)
    }

    /// Pre-stages models whose `initial_tier` is configured as `Tier1GpuVram` or `Tier2PinnedRam`.
    /// A model that fails to load is logged and left cold; it does not abort startup.
    pub async fn initialize_startup_tiers(&self) -> Result<()> {
        let ctx = RequestContext::unbounded();
        for id in self.memory.model_ids() {
            let (cfg, _) = self.memory.get_model_state(&id)?;
            if !self.memory.is_enabled(&id)? {
                if cfg.initial_tier != MemoryTier::Tier3NvmeSsd {
                    info!(model = %cfg.id, initial_tier = ?cfg.initial_tier, "Model is disabled; not pre-staged");
                }
                continue;
            }
            match cfg.initial_tier {
                MemoryTier::Tier1GpuVram => {
                    info!(model = %cfg.id, "Pre-loading model into Tier 1 GPU VRAM on startup");
                    if let Err(e) = self.acquire_model(&cfg.id, None, 1, &ctx).await {
                        warn!(model = %cfg.id, error = %e, "Startup pre-load failed; model stays cold until requested");
                    }
                }
                MemoryTier::Tier2PinnedRam => {
                    info!(model = %cfg.id, "Pre-staging model weights into Tier 2 Pinned Host RAM on startup");
                    let plan = self
                        .memory
                        .demote_model_from_vram(&cfg.id, MemoryTier::Tier2PinnedRam)?;
                    if let Some(backend) = self.backends.get(&cfg.id) {
                        if let Err(e) = backend.apply_placement(&plan, None).await {
                            warn!(model = %cfg.id, error = %e, "Tier 2 pre-stage callback failed");
                        }
                    }
                }
                MemoryTier::Tier3NvmeSsd => {}
            }
        }
        self.memory.reconcile_vram();
        Ok(())
    }

    /// Acquires an execution lease for `model_id`, lazily loading or promoting it to Tier 1 VRAM
    /// (with 100% or partial GPU layers + reserved KV cache for `requested_context_len`).
    pub async fn acquire_model(
        &self,
        model_id: &str,
        requested_context_len: Option<u32>,
        batch_size: u32,
        ctx: &RequestContext,
    ) -> Result<ActiveModelLease> {
        let backend = self
            .backends
            .get(model_id)
            .ok_or_else(|| HypnosError::ModelNotFound(model_id.to_string()))?
            .clone();
        if !self.memory.is_enabled(model_id)? {
            return Err(HypnosError::ModelDisabled(model_id.to_string()));
        }

        // Admission control: bounded queue per model.
        let sem = self
            .admission
            .get(model_id)
            .cloned()
            .ok_or_else(|| HypnosError::ModelNotFound(model_id.to_string()))?;
        let admission = match sem.clone().try_acquire_owned() {
            Ok(p) => p,
            Err(_) => {
                self.stats
                    .admissions_rejected
                    .fetch_add(1, Ordering::Relaxed);
                return Err(HypnosError::Busy {
                    model: model_id.to_string(),
                    in_flight: self.max_queue_per_model,
                    limit: self.max_queue_per_model,
                });
            }
        };
        if !ctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }

        // Fast path: already resident with a sufficient context window. The check and the lease
        // count happen under one lock, so an eviction cannot unload the model in between.
        {
            let (cfg, _) = self.memory.get_model_state(model_id)?;
            let needed_ctx = requested_context_len
                .unwrap_or(cfg.arch.default_context_len)
                .min(cfg.arch.max_context_len);
            // The booking is written before the engine finishes loading (BUG-2): only an engine
            // that reports itself resident serves from the fast path; otherwise the request waits
            // for the transition lock, which the load in progress holds.
            if backend.is_resident() {
                if let Some(current_plan) = self
                    .memory
                    .increment_lease_if_resident(model_id, needed_ctx)?
                {
                    return Ok(ActiveModelLease {
                        model_id: model_id.to_string(),
                        memory: self.memory.clone(),
                        backend,
                        previous_tier: MemoryTier::Tier1GpuVram,
                        placement: current_plan,
                        tier_switch_latency_ms: 0.0,
                        evicted_models: Vec::new(),
                        load_phases: None,
                        _admission: admission,
                    });
                }
            }
        }

        // Slow path: coordinate tier promotion, KV cache expansion, or eviction of idle models.
        let _guard = self.transition_lock.lock().await;
        if !ctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }
        // Re-checked under the lock: a disable that ran while this request waited wins.
        if !self.memory.is_enabled(model_id)? {
            return Err(HypnosError::ModelDisabled(model_id.to_string()));
        }
        let switch_start = Instant::now();

        let (cfg, current_plan) = self.memory.get_model_state(model_id)?;
        let needed_ctx = requested_context_len
            .unwrap_or(cfg.arch.default_context_len)
            .min(cfg.arch.max_context_len);

        // Double-check after acquiring transition_lock (a load that held it has finished or
        // rolled back by now, so the engine's residency is settled).
        if backend.is_resident() {
            if let Some(current_plan) = self
                .memory
                .increment_lease_if_resident(model_id, needed_ctx)?
            {
                return Ok(ActiveModelLease {
                    model_id: model_id.to_string(),
                    memory: self.memory.clone(),
                    backend,
                    previous_tier: MemoryTier::Tier1GpuVram,
                    placement: current_plan,
                    tier_switch_latency_ms: switch_start.elapsed().as_secs_f64() * 1000.0,
                    evicted_models: Vec::new(),
                    load_phases: None,
                    _admission: admission,
                });
            }
        }

        let is_uma = self.memory.is_unified_memory();
        let mut evicted_models = Vec::new();

        // Ground the budget view in the device's real free memory before planning.
        self.memory.reconcile_vram();

        let (mut avail_vram, mut avail_ram) = self.available_for(&current_plan);

        let mut target_plan =
            plan_placement(&cfg, needed_ctx, batch_size, avail_vram, avail_ram, is_uma);

        // `Some((growth, headroom))` when loading `plan` would push this process past
        // `hardware.max_process_memory_bytes` (measured footprint, not bookings).
        let over_process_cap = |plan: &LayerPlacementPlan| -> Option<(u64, u64)> {
            let headroom = self.memory.process_memory_headroom()?;
            let growth = self.memory.process_growth_bytes(&current_plan, plan);
            (growth > headroom).then_some((growth, headroom))
        };

        // Evict idle lower-priority models from Tier 1 while the model does not fit, only fits
        // with a partial GPU split, or would cross the process memory cap.
        let candidates = self.memory.idle_vram_eviction_candidates(model_id);
        for candidate_id in candidates {
            let reason = match &target_plan {
                None => EvictionReason::MemoryBudget,
                Some(plan) if !cfg.runs_on_cpu() && plan.gpu_layers < cfg.arch.num_layers => {
                    EvictionReason::FullGpuPlacement
                }
                Some(plan) if over_process_cap(plan).is_some() => EvictionReason::ProcessMemoryCap,
                Some(_) => break,
            };

            let (cand_cfg, _) = self.memory.get_model_state(&candidate_id)?;
            let target_demote_tier = if cand_cfg.keep_warm_in_ram {
                MemoryTier::Tier2PinnedRam
            } else {
                MemoryTier::Tier3NvmeSsd
            };

            info!(
                evicting = %candidate_id,
                for_model = %model_id,
                to_tier = ?target_demote_tier,
                reason = ?reason,
                "Evicting idle model from Tier 1 VRAM to free space"
            );

            let Some(demoted_plan) = self
                .memory
                .demote_if_idle(&candidate_id, target_demote_tier)?
            else {
                // A request took a lease on it since the candidates were listed.
                continue;
            };
            if let Some(cand_backend) = self.backends.get(&candidate_id) {
                cand_backend.apply_placement(&demoted_plan, None).await?;
            }
            self.stats.evictions.fetch_add(1, Ordering::Relaxed);
            // The demotion falls back to Tier 3 when Tier 2 has no room: record where it landed.
            self.memory
                .record_eviction(&candidate_id, model_id, reason, demoted_plan.primary_tier);
            evicted_models.push(candidate_id);

            self.memory.reconcile_vram();
            (avail_vram, avail_ram) = self.available_for(&current_plan);

            target_plan =
                plan_placement(&cfg, needed_ctx, batch_size, avail_vram, avail_ram, is_uma);
        }

        // A CPU model needs host RAM, which evicting GPU models does not free: send idle models
        // that only hold RAM (warm Tier 2 weights, other CPU models) to Tier 3 until it fits.
        if target_plan.is_none() && cfg.runs_on_cpu() {
            for candidate_id in self.memory.idle_ram_eviction_candidates(model_id) {
                info!(
                    evicting = %candidate_id,
                    for_model = %model_id,
                    "Moving idle model to Tier 3 to free host RAM"
                );
                let Some(demoted_plan) = self
                    .memory
                    .demote_if_idle(&candidate_id, MemoryTier::Tier3NvmeSsd)?
                else {
                    continue;
                };
                if let Some(cand_backend) = self.backends.get(&candidate_id) {
                    cand_backend.apply_placement(&demoted_plan, None).await?;
                }
                self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                self.memory.record_eviction(
                    &candidate_id,
                    model_id,
                    EvictionReason::MemoryBudget,
                    demoted_plan.primary_tier,
                );
                evicted_models.push(candidate_id);
                (avail_vram, avail_ram) = self.available_for(&current_plan);
                target_plan =
                    plan_placement(&cfg, needed_ctx, batch_size, avail_vram, avail_ram, is_uma);
                if target_plan.is_some() {
                    break;
                }
            }
        }

        let final_plan = target_plan.ok_or_else(|| {
            warn!(
                model = %model_id,
                avail_vram_mb = avail_vram / (1024 * 1024),
                avail_ram_mb = avail_ram / (1024 * 1024),
                unaccounted_vram_mb = self.memory.unaccounted_vram_bytes() / (1024 * 1024),
                "Insufficient VRAM/RAM even after evicting idle models"
            );
            let (required_bytes, available_bytes) = if cfg.runs_on_cpu() {
                (
                    cfg.total_weight_bytes
                        .saturating_add(cfg.arch.activation_workspace_bytes),
                    avail_ram,
                )
            } else {
                (cfg.total_weight_bytes, avail_vram)
            };
            HypnosError::InsufficientMemory {
                model: model_id.to_string(),
                required_bytes,
                available_bytes,
            }
        })?;

        if let Some((growth, headroom)) = over_process_cap(&final_plan) {
            warn!(
                model = %model_id,
                needed_mb = growth / (1024 * 1024),
                headroom_mb = headroom / (1024 * 1024),
                footprint_mb = self.memory.process_footprint_bytes().unwrap_or(0) / (1024 * 1024),
                "Loading would cross max_process_memory_bytes even after evicting idle models"
            );
            return Err(HypnosError::ProcessMemoryCap {
                model: model_id.to_string(),
                required_bytes: growth,
                available_bytes: headroom,
                cap_bytes: self.memory.process_memory_cap_bytes(),
            });
        }

        if !ctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }

        let (previous_tier, host_ptr) = self
            .memory
            .apply_model_placement(model_id, final_plan.clone())?;
        if let Err(load_err) = backend.apply_placement(&final_plan, host_ptr).await {
            // The engine refused the weights (missing file, CUDA OOM, ...): undo the Tier 1 booking so
            // the accounting keeps matching what the GPU really holds.
            self.stats.load_failures.fetch_add(1, Ordering::Relaxed);
            if let Ok(mut by_model) = self.stats.load_failures_by_model.lock() {
                *by_model.entry(model_id.to_string()).or_insert(0) += 1;
            }
            let fallback_tier = if cfg.keep_warm_in_ram {
                MemoryTier::Tier2PinnedRam
            } else {
                MemoryTier::Tier3NvmeSsd
            };
            warn!(model = %model_id, error = %load_err, to_tier = ?fallback_tier, "Engine load failed; rolling back Tier 1 placement");
            if let Ok(rollback_plan) = self.memory.demote_model_from_vram(model_id, fallback_tier) {
                let _ = backend.apply_placement(&rollback_plan, None).await;
            }
            self.memory.reconcile_vram();
            return Err(load_err);
        }
        self.memory.increment_lease(model_id)?;
        let load_phases = backend.last_load_phases();

        // Compare the engine's real footprint with the booking; a large gap means the estimates in
        // the config are off and a later promotion could overcommit the device.
        if let Some((real_free, _)) = self.memory.reconcile_vram() {
            let unaccounted = self.memory.unaccounted_vram_bytes();
            if unaccounted > final_plan.total_vram_bytes() / 2 && unaccounted > 1024 * 1024 * 1024 {
                warn!(
                    model = %model_id,
                    unaccounted_mb = unaccounted / (1024 * 1024),
                    real_free_mb = real_free / (1024 * 1024),
                    "Device VRAM usage exceeds the planner's bookings by a wide margin; check total_weight_bytes / overhead settings"
                );
            }
        }

        let switch_ms = switch_start.elapsed().as_secs_f64() * 1000.0;
        self.stats.promotions.fetch_add(1, Ordering::Relaxed);
        self.stats
            .tier_switch_ms_total
            .fetch_add(switch_ms as u64, Ordering::Relaxed);
        Ok(ActiveModelLease {
            model_id: model_id.to_string(),
            memory: self.memory.clone(),
            backend,
            previous_tier,
            placement: final_plan,
            tier_switch_latency_ms: switch_ms,
            evicted_models,
            load_phases,
            _admission: admission,
        })
    }

    /// `(vram, pinned_ram)` a new placement for a model currently at `current` may use, counting what
    /// that model already holds.
    fn available_for(&self, current: &LayerPlacementPlan) -> (u64, u64) {
        let mut vram = self
            .memory
            .effective_free_vram_bytes()
            .saturating_add(current.total_vram_bytes());
        let ram = self
            .memory
            .free_pinned_ram_bytes()
            .saturating_add(current.pinned_ram_bytes);
        if let Some(pool_free) = self.memory.physical_pool_free_bytes() {
            // Unified memory: Tier 1 and Tier 2 are the same RAM, so the model's Tier 2 pages are
            // released when it becomes resident, and other models' Tier 2 bookings limit Tier 1.
            vram = vram.min(
                pool_free
                    .saturating_add(current.total_vram_bytes())
                    .saturating_add(current.pinned_ram_bytes),
            );
        }
        (vram, ram)
    }

    /// Whether the model is in service.
    pub fn is_enabled(&self, model_id: &str) -> Result<bool> {
        self.memory.is_enabled(model_id)
    }

    /// Takes a model out of service or puts it back (`POST /v1/models/{id}/disable` / `enable`).
    /// Disabling evicts it to Tier 3 first and, like a manual tier move, is refused while the
    /// model is leased. Enabling only flips the switch: the model loads on its next request.
    /// Returns the model's placement afterwards.
    pub async fn set_model_enabled(
        &self,
        model_id: &str,
        enabled: bool,
    ) -> Result<LayerPlacementPlan> {
        let _guard = self.transition_lock.lock().await;
        let (_, plan) = self.memory.get_model_state(model_id)?;
        if enabled {
            if !self.memory.is_enabled(model_id)? {
                info!(model = %model_id, "Model enabled");
            }
            self.memory.set_enabled(model_id, true)?;
            return Ok(plan);
        }
        let plan = if plan.primary_tier != MemoryTier::Tier3NvmeSsd || plan.pinned_ram_bytes > 0 {
            let Some(demoted) = self
                .memory
                .demote_if_idle(model_id, MemoryTier::Tier3NvmeSsd)?
            else {
                return Err(HypnosError::Busy {
                    model: model_id.to_string(),
                    in_flight: self.memory.active_leases(model_id).max(1),
                    limit: 0,
                });
            };
            if let Some(backend) = self.backends.get(model_id) {
                backend.apply_placement(&demoted, None).await?;
            }
            self.memory.reconcile_vram();
            demoted
        } else {
            if self.memory.active_leases(model_id) > 0 {
                return Err(HypnosError::Busy {
                    model: model_id.to_string(),
                    in_flight: self.memory.active_leases(model_id),
                    limit: 0,
                });
            }
            plan
        };
        if self.memory.is_enabled(model_id)? {
            info!(model = %model_id, "Model disabled; evicted to Tier 3");
        }
        self.memory.set_enabled(model_id, false)?;
        Ok(plan)
    }

    /// Manually promotes or demotes a model to a specific tier (`/v1/memory/tier`).
    pub async fn set_model_tier(
        &self,
        model_id: &str,
        target_tier: MemoryTier,
        context_len: Option<u32>,
    ) -> Result<LayerPlacementPlan> {
        if !self.memory.is_enabled(model_id)? {
            return Err(HypnosError::ModelDisabled(model_id.to_string()));
        }
        match target_tier {
            MemoryTier::Tier1GpuVram => {
                let ctx = RequestContext::unbounded();
                let lease = self.acquire_model(model_id, context_len, 1, &ctx).await?;
                Ok(lease.placement.clone())
            }
            MemoryTier::Tier2PinnedRam | MemoryTier::Tier3NvmeSsd => {
                let _guard = self.transition_lock.lock().await;
                let Some(plan) = self.memory.demote_if_idle(model_id, target_tier)? else {
                    return Err(HypnosError::Busy {
                        model: model_id.to_string(),
                        in_flight: self.memory.active_leases(model_id).max(1),
                        limit: 0,
                    });
                };
                if let Some(backend) = self.backends.get(model_id) {
                    backend.apply_placement(&plan, None).await?;
                }
                self.memory.reconcile_vram();
                Ok(plan)
            }
        }
    }

    /// Releases every engine (frees VRAM and host allocations). Used on graceful shutdown.
    pub async fn shutdown(&self) {
        let _guard = self.transition_lock.lock().await;
        for id in self.memory.model_ids() {
            let Ok((_, plan)) = self.memory.get_model_state(&id) else {
                continue;
            };
            if plan.primary_tier == MemoryTier::Tier1GpuVram || plan.pinned_ram_bytes > 0 {
                match self
                    .memory
                    .demote_model_from_vram(&id, MemoryTier::Tier3NvmeSsd)
                {
                    Ok(p) => {
                        if let Some(b) = self.backends.get(&id) {
                            if let Err(e) = b.apply_placement(&p, None).await {
                                warn!(model = %id, error = %e, "Engine release failed during shutdown");
                            }
                        }
                    }
                    Err(e) => warn!(model = %id, error = %e, "Demotion failed during shutdown"),
                }
            }
        }
        info!("All engines released");
    }
}
