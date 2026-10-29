use crate::hardware::{DeviceMemoryCapabilities, OS_RESERVE_BYTES};
use crate::mmap_cache::MmapWeightBuffer;
use hypnos_core::config::{HardwareConfig, MemoryTier, MemoryTopology, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::GpuMemoryProbe;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, info, warn};

/// Live runtime state of a registered model inside the 3-tier memory manager.
pub struct ManagedModelMemory {
    pub config: ModelConfig,
    pub placement: LayerPlacementPlan,
    pub host_buffer: MmapWeightBuffer,
    pub last_used: Instant,
    pub active_leases: usize,
    pub total_requests_served: u64,
}

/// Real-time snapshot of the 3-tier memory hierarchy for observability (`/v1/memory/status`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryHierarchyStatus {
    pub capabilities: DeviceMemoryCapabilities,
    /// VRAM the planner has booked for resident models (weights + KV + workspace).
    pub tier1_vram_used_bytes: u64,
    /// Budget minus booked VRAM.
    pub tier1_vram_free_bytes: u64,
    /// Last measured free VRAM on the device (None when no probe is available).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier1_vram_real_free_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tier1_vram_real_total_bytes: Option<u64>,
    /// VRAM in use on the device that the planner did not book (driver, display, other processes,
    /// engine overhead beyond the estimates). Subtracted from what the planner may hand out.
    pub tier1_vram_unaccounted_bytes: u64,
    /// What the planner will actually allow a new promotion to use right now.
    pub tier1_vram_effective_free_bytes: u64,
    pub tier2_pinned_ram_used_bytes: u64,
    pub tier2_pinned_ram_free_bytes: u64,
    /// Unified memory only: physical RAM not booked by either tier (Tier 1 and Tier 2 share it).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub physical_pool_free_bytes: Option<u64>,
    /// This process's memory footprint as the OS reports it (see `process.rs`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_footprint_bytes: Option<u64>,
    /// Memory this process holds, as compared against the cap: the footprint plus, on macOS
    /// unified memory, the model weights llama.cpp maps from disk into Metal buffers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_memory_bytes: Option<u64>,
    /// `hardware.max_process_memory_bytes`, when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process_memory_cap_bytes: Option<u64>,
    pub models: Vec<ModelTierSnapshot>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelTierSnapshot {
    pub id: String,
    pub modality: hypnos_core::config::Modality,
    pub modalities: Vec<hypnos_core::config::Modality>,
    pub engine: hypnos_core::config::EngineKind,
    pub total_weight_bytes: u64,
    pub priority: hypnos_core::config::ModelPriority,
    pub current_tier: MemoryTier,
    pub placement: LayerPlacementPlan,
    pub active_leases: usize,
    pub idle_seconds: f64,
    pub total_requests_served: u64,
}

/// Coordinates memory placement across Tier 1 (GPU VRAM / Wired UMA),
/// Tier 2 (Pinned Host RAM / Warm UMA), and Tier 3 (NVMe SSD).
pub struct TieredMemoryManager {
    /// Updated once the engine's probe reports the device: on unified memory the GPU's own
    /// working-set figure replaces the startup estimate, and a budget above the device is clamped.
    capabilities: RwLock<DeviceMemoryCapabilities>,
    /// `hardware.vram_budget_bytes` (0 = derive from the device).
    configured_vram_budget_bytes: u64,
    configured_pinned_ram_budget_bytes: u64,
    /// `hardware.simulate_physical_memory_bytes` is set: keep the simulated budgets.
    simulating: bool,
    /// `hardware.max_process_memory_bytes` (0 = no cap).
    max_process_memory_bytes: u64,
    lock_host_memory: bool,
    models: RwLock<HashMap<String, ManagedModelMemory>>,
    vram_used_bytes: AtomicU64,
    pinned_ram_used_bytes: AtomicU64,
    /// Optional live reading of device memory (set once the engines are initialized).
    gpu_probe: RwLock<Option<Arc<dyn GpuMemoryProbe>>>,
    vram_real_free_bytes: AtomicU64,
    vram_real_total_bytes: AtomicU64,
    vram_unaccounted_bytes: AtomicU64,
    has_real_reading: std::sync::atomic::AtomicBool,
}

impl TieredMemoryManager {
    pub fn new(hw_config: &HardwareConfig, model_configs: Vec<ModelConfig>) -> Result<Self> {
        let capabilities = DeviceMemoryCapabilities::detect(hw_config);
        let is_uma = capabilities.topology == MemoryTopology::UnifiedMemory;
        let lock_host_memory = hw_config.lock_host_memory_for(capabilities.topology);
        if is_uma {
            if lock_host_memory {
                warn!("lock_host_memory = true on unified memory: mlock and the Metal working set share one wired-memory limit");
            }
            if let Some(w) =
                hw_config.unified_memory_oversubscription(capabilities.physical_memory_bytes)
            {
                warn!(config_warning = %w);
            }
        }
        if hw_config.max_process_memory_bytes > 0 {
            match crate::process::process_footprint_bytes() {
                Some(now) => info!(
                    cap_mb = hw_config.max_process_memory_bytes / (1024 * 1024),
                    footprint_now_mb = now / (1024 * 1024),
                    "Process memory cap active: model loads that would cross it evict or are refused"
                ),
                None => warn!("max_process_memory_bytes is set but this OS does not report the process footprint; the cap is not enforced"),
            }
        }
        if lock_host_memory {
            crate::mmap_cache::raise_working_set_limit(
                hw_config
                    .pinned_ram_budget_bytes
                    .saturating_add(1024 * 1024 * 1024),
            );
        }
        let mut models = HashMap::new();

        for mut cfg in model_configs {
            let host_buffer = MmapWeightBuffer::open_or_virtual(
                &cfg.weights_path,
                cfg.total_weight_bytes,
                !is_uma,
            )?;
            if cfg.total_weight_bytes == 0 {
                cfg.total_weight_bytes = host_buffer.len_bytes().max(512 * 1024 * 1024);
            }

            let initial_placement = LayerPlacementPlan {
                primary_tier: MemoryTier::Tier3NvmeSsd,
                gpu_layers: 0,
                pinned_ram_layers: 0,
                ssd_layers: cfg.arch.num_layers,
                vram_weight_bytes: 0,
                vram_kv_and_workspace_bytes: 0,
                pinned_ram_bytes: 0,
                allocated_context_len: 0,
            };

            models.insert(
                cfg.id.clone(),
                ManagedModelMemory {
                    config: cfg,
                    placement: initial_placement,
                    host_buffer,
                    last_used: Instant::now(),
                    active_leases: 0,
                    total_requests_served: 0,
                },
            );
        }

        Ok(Self {
            capabilities: RwLock::new(capabilities),
            configured_vram_budget_bytes: hw_config.vram_budget_bytes,
            configured_pinned_ram_budget_bytes: hw_config.pinned_ram_budget_bytes,
            simulating: hw_config.simulate_physical_memory_bytes > 0,
            max_process_memory_bytes: hw_config.max_process_memory_bytes,
            lock_host_memory,
            models: RwLock::new(models),
            vram_used_bytes: AtomicU64::new(0),
            pinned_ram_used_bytes: AtomicU64::new(0),
            gpu_probe: RwLock::new(None),
            vram_real_free_bytes: AtomicU64::new(0),
            vram_real_total_bytes: AtomicU64::new(0),
            vram_unaccounted_bytes: AtomicU64::new(0),
            has_real_reading: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub fn capabilities(&self) -> DeviceMemoryCapabilities {
        self.capabilities.read().clone()
    }

    pub fn is_unified_memory(&self) -> bool {
        self.capabilities.read().topology == MemoryTopology::UnifiedMemory
    }

    /// Installs a live device-memory probe. Called by the server once an engine is available.
    ///
    /// On unified memory with an auto-detected budget (and no simulation), the device's total
    /// (Metal's `recommendedMaxWorkingSetSize`) becomes the Tier 1 ceiling, replacing the estimate.
    /// Otherwise a Tier 1 budget above what the device reports is clamped to it.
    pub fn set_gpu_probe(&self, probe: Arc<dyn GpuMemoryProbe>) {
        *self.gpu_probe.write() = Some(probe);
        let Some((_, device_total)) = self.reconcile_vram() else {
            return;
        };
        if device_total == 0 {
            return;
        }
        let mut caps = self.capabilities.write();
        if caps.total_vram_bytes == device_total {
            return;
        }
        let before_mb = caps.total_vram_bytes / (1024 * 1024);
        let adopt = caps.topology == MemoryTopology::UnifiedMemory
            && self.configured_vram_budget_bytes == 0
            && !self.simulating;
        if adopt {
            caps.set_vram_total(device_total, self.configured_pinned_ram_budget_bytes);
            info!(
                estimated_mb = before_mb,
                device_working_set_mb = device_total / (1024 * 1024),
                usable_vram_mb = caps.usable_vram_budget_bytes / (1024 * 1024),
                pinned_ram_mb = caps.pinned_ram_budget_bytes / (1024 * 1024),
                "Unified memory: Tier 1 budget set from the GPU's recommended working set"
            );
        } else if caps.total_vram_bytes > device_total {
            caps.set_vram_total(device_total, self.configured_pinned_ram_budget_bytes);
            warn!(
                budget_mb = before_mb,
                device_mb = device_total / (1024 * 1024),
                "Tier 1 budget exceeds the memory the GPU reports; clamped"
            );
        }
    }

    /// This process's memory footprint as the OS reports it
    /// (see [`crate::process::process_footprint_bytes`]).
    pub fn process_footprint_bytes(&self) -> Option<u64> {
        crate::process::process_footprint_bytes()
    }

    /// Memory this process holds. On macOS with unified memory, llama.cpp maps model weights from
    /// disk straight into Metal buffers: those pages are file-backed, so `phys_footprint` leaves
    /// them out, yet they cannot be reclaimed while the model is resident. Their size is the
    /// booked weight bytes of the resident llama models (buffers Metal allocates itself, such as
    /// stable-diffusion.cpp's, are already part of the footprint).
    pub fn process_memory_bytes(&self) -> Option<u64> {
        let footprint = self.process_footprint_bytes()?;
        if !(cfg!(target_os = "macos") && self.is_unified_memory()) {
            return Some(footprint);
        }
        let mapped_weights: u64 = self
            .models
            .read()
            .values()
            .filter(|m| {
                m.config.engine == hypnos_core::config::EngineKind::GgmlLlama
                    && m.placement.primary_tier == MemoryTier::Tier1GpuVram
            })
            .map(|m| m.placement.vram_weight_bytes)
            .sum();
        Some(footprint.saturating_add(mapped_weights))
    }

    /// `hardware.max_process_memory_bytes` (0 = no cap).
    pub fn process_memory_cap_bytes(&self) -> u64 {
        self.max_process_memory_bytes
    }

    /// How much a model load may still add before the process crosses
    /// `hardware.max_process_memory_bytes`. `None` when no cap is set or the OS gives no reading.
    pub fn process_memory_headroom(&self) -> Option<u64> {
        if self.max_process_memory_bytes == 0 {
            return None;
        }
        let used = self.process_memory_bytes()?;
        Some(self.max_process_memory_bytes.saturating_sub(used))
    }

    /// The part of a placement change that adds to [`Self::process_memory_bytes`]. On unified
    /// memory that is the Tier 1 booking: Tier 2 pages there are reclaimable page cache, which the
    /// process measure leaves out, so a promotion from Tier 2 still adds the full model. Next to a
    /// discrete GPU only the host-RAM side counts (resident set / working set include it).
    pub fn process_growth_bytes(&self, from: &LayerPlacementPlan, to: &LayerPlacementPlan) -> u64 {
        if self.is_unified_memory() {
            to.total_vram_bytes()
                .saturating_sub(from.total_vram_bytes())
        } else {
            to.pinned_ram_bytes.saturating_sub(from.pinned_ram_bytes)
        }
    }

    /// Re-reads device memory and derives how much VRAM is in use beyond what the planner booked.
    /// Returns `(free, total)` when a reading was possible.
    pub fn reconcile_vram(&self) -> Option<(u64, u64)> {
        let probe = self.gpu_probe.read().clone()?;
        let (free, total) = probe.probe()?;
        let used_real = total.saturating_sub(free);
        let booked = self.used_vram_bytes();
        let unaccounted = used_real.saturating_sub(booked);
        self.vram_real_free_bytes.store(free, Ordering::SeqCst);
        self.vram_real_total_bytes.store(total, Ordering::SeqCst);
        self.vram_unaccounted_bytes
            .store(unaccounted, Ordering::SeqCst);
        self.has_real_reading.store(true, Ordering::SeqCst);
        debug!(
            real_free_mb = free / (1024 * 1024),
            real_total_mb = total / (1024 * 1024),
            booked_mb = booked / (1024 * 1024),
            unaccounted_mb = unaccounted / (1024 * 1024),
            "Reconciled VRAM bookkeeping with device"
        );
        Some((free, total))
    }

    pub fn used_vram_bytes(&self) -> u64 {
        self.vram_used_bytes.load(Ordering::SeqCst)
    }

    /// Budget-based free VRAM (what the planner has not booked).
    pub fn free_vram_bytes(&self) -> u64 {
        self.capabilities
            .read()
            .usable_vram_budget_bytes
            .saturating_sub(self.used_vram_bytes())
    }

    pub fn real_free_vram_bytes(&self) -> Option<u64> {
        if self.has_real_reading.load(Ordering::SeqCst) {
            Some(self.vram_real_free_bytes.load(Ordering::SeqCst))
        } else {
            None
        }
    }

    pub fn unaccounted_vram_bytes(&self) -> u64 {
        self.vram_unaccounted_bytes.load(Ordering::SeqCst)
    }

    /// Free VRAM a new promotion may rely on: the budget view, capped by the last real reading
    /// minus the safety margin so that driver/display/other-process usage is never handed out.
    pub fn effective_free_vram_bytes(&self) -> u64 {
        let budget_free = self.free_vram_bytes();
        match self.real_free_vram_bytes() {
            Some(real_free) => {
                let caps = self.capabilities.read();
                let margin = caps
                    .total_vram_bytes
                    .saturating_sub(caps.usable_vram_budget_bytes);
                budget_free.min(real_free.saturating_sub(margin))
            }
            None => budget_free,
        }
    }

    pub fn used_pinned_ram_bytes(&self) -> u64 {
        self.pinned_ram_used_bytes.load(Ordering::SeqCst)
    }

    pub fn free_pinned_ram_bytes(&self) -> u64 {
        self.capabilities
            .read()
            .pinned_ram_budget_bytes
            .saturating_sub(self.used_pinned_ram_bytes())
    }

    /// Unified memory only: physical RAM (minus the OS reserve) that neither Tier 1 nor Tier 2 has
    /// booked. `None` on discrete GPUs, or when physical memory could not be read.
    pub fn physical_pool_free_bytes(&self) -> Option<u64> {
        let caps = self.capabilities.read();
        if caps.topology != MemoryTopology::UnifiedMemory || caps.physical_memory_bytes == 0 {
            return None;
        }
        Some(
            caps.physical_memory_bytes
                .saturating_sub(OS_RESERVE_BYTES)
                .saturating_sub(self.used_vram_bytes())
                .saturating_sub(self.used_pinned_ram_bytes()),
        )
    }

    /// Returns a clone of the model's static configuration and current placement.
    pub fn get_model_state(&self, model_id: &str) -> Result<(ModelConfig, LayerPlacementPlan)> {
        let guard = self.models.read();
        let entry = guard
            .get(model_id)
            .ok_or_else(|| HypnosError::ModelNotFound(model_id.to_string()))?;
        Ok((entry.config.clone(), entry.placement.clone()))
    }

    pub fn model_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.models.read().keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Commits a new `LayerPlacementPlan` for `model_id`, updating Tier 1 VRAM and Tier 2 Pinned RAM
    /// accounting and pinning/unpinning OS host pages as needed.
    ///
    /// Returns `(previous_tier, host_ptr)`.
    pub fn apply_model_placement(
        &self,
        model_id: &str,
        new_plan: LayerPlacementPlan,
    ) -> Result<(MemoryTier, Option<usize>)> {
        let mut guard = self.models.write();
        let entry = guard
            .get_mut(model_id)
            .ok_or_else(|| HypnosError::ModelNotFound(model_id.to_string()))?;

        let prev_plan = entry.placement.clone();
        let prev_tier = prev_plan.primary_tier;

        let old_vram = prev_plan.total_vram_bytes();
        let new_vram = new_plan.total_vram_bytes();

        // Unified memory: both tiers are carved from the same RAM, so the combined growth must fit.
        if let Some(pool_free) = self.physical_pool_free_bytes() {
            let old_total = old_vram.saturating_add(prev_plan.pinned_ram_bytes);
            let new_total = new_vram.saturating_add(new_plan.pinned_ram_bytes);
            if new_total > old_total && new_total - old_total > pool_free {
                return Err(HypnosError::InsufficientMemory {
                    model: model_id.to_string(),
                    required_bytes: new_total - old_total,
                    available_bytes: pool_free,
                });
            }
        }

        // Reconcile Tier 1 VRAM delta
        if new_vram > old_vram {
            let delta = new_vram - old_vram;
            let free = self.free_vram_bytes();
            if delta > free {
                return Err(HypnosError::InsufficientMemory {
                    model: model_id.to_string(),
                    required_bytes: delta,
                    available_bytes: free,
                });
            }
            self.vram_used_bytes.fetch_add(delta, Ordering::SeqCst);
        } else if old_vram > new_vram {
            self.vram_used_bytes
                .fetch_sub(old_vram - new_vram, Ordering::SeqCst);
        }

        // Reconcile Tier 2 Pinned Host RAM delta
        let old_ram = prev_plan.pinned_ram_bytes;
        let new_ram = new_plan.pinned_ram_bytes;
        if new_ram > old_ram {
            let delta = new_ram - old_ram;
            let free = self.free_pinned_ram_bytes();
            if delta > free {
                // Rollback VRAM change before returning error
                if new_vram > old_vram {
                    self.vram_used_bytes
                        .fetch_sub(new_vram - old_vram, Ordering::SeqCst);
                } else if old_vram > new_vram {
                    self.vram_used_bytes
                        .fetch_add(old_vram - new_vram, Ordering::SeqCst);
                }
                return Err(HypnosError::InsufficientMemory {
                    model: model_id.to_string(),
                    required_bytes: delta,
                    available_bytes: free,
                });
            }
            self.pinned_ram_used_bytes
                .fetch_add(delta, Ordering::SeqCst);
        } else if old_ram > new_ram {
            self.pinned_ram_used_bytes
                .fetch_sub(old_ram - new_ram, Ordering::SeqCst);
        }

        // Manage physical OS page pinning (`VirtualLock` / `mlock`)
        if new_plan.pinned_ram_bytes > 0 || new_plan.primary_tier == MemoryTier::Tier2PinnedRam {
            entry.host_buffer.pin_in_host_ram(self.lock_host_memory)?;
        } else if new_plan.primary_tier == MemoryTier::Tier3NvmeSsd {
            entry.host_buffer.unpin_from_host_ram()?;
        }

        let host_ptr = entry.host_buffer.host_ptr();
        info!(
            model = %model_id,
            from_tier = ?prev_tier,
            to_tier = ?new_plan.primary_tier,
            gpu_layers = new_plan.gpu_layers,
            ram_layers = new_plan.pinned_ram_layers,
            vram_mb = new_plan.total_vram_bytes() / (1024 * 1024),
            pinned_ram_mb = new_plan.pinned_ram_bytes / (1024 * 1024),
            "Updated model tier & layer placement"
        );

        entry.placement = new_plan;
        entry.last_used = Instant::now();
        Ok((prev_tier, host_ptr))
    }

    /// Demotes a model from Tier 1 (GPU VRAM) down to Tier 2 (Pinned Host RAM) or Tier 3 (NVMe SSD).
    /// Frees 100% of its GPU VRAM (weights + KV cache) while optionally keeping weights warm in Pinned RAM.
    pub fn demote_model_from_vram(
        &self,
        model_id: &str,
        target_tier: MemoryTier,
    ) -> Result<LayerPlacementPlan> {
        let (cfg, _) = self.get_model_state(model_id)?;
        let total_layers = cfg.arch.num_layers;

        let new_plan = match target_tier {
            MemoryTier::Tier2PinnedRam => {
                // Keep weights pinned in Tier 2 RAM if budget allows, otherwise fallback to Tier 3 SSD
                let needed_ram = cfg.total_weight_bytes;
                let current_ram = self.get_model_state(model_id)?.1.pinned_ram_bytes;
                let additional_ram = needed_ram.saturating_sub(current_ram);

                if additional_ram <= self.free_pinned_ram_bytes() {
                    LayerPlacementPlan {
                        primary_tier: MemoryTier::Tier2PinnedRam,
                        gpu_layers: 0,
                        pinned_ram_layers: total_layers,
                        ssd_layers: 0,
                        vram_weight_bytes: 0,
                        vram_kv_and_workspace_bytes: 0,
                        pinned_ram_bytes: needed_ram,
                        allocated_context_len: 0,
                    }
                } else {
                    warn!(model = %model_id, "Tier 2 pinned RAM budget exhausted; demoting to Tier 3 instead");
                    LayerPlacementPlan {
                        primary_tier: MemoryTier::Tier3NvmeSsd,
                        gpu_layers: 0,
                        pinned_ram_layers: 0,
                        ssd_layers: total_layers,
                        vram_weight_bytes: 0,
                        vram_kv_and_workspace_bytes: 0,
                        pinned_ram_bytes: 0,
                        allocated_context_len: 0,
                    }
                }
            }
            _ => LayerPlacementPlan {
                primary_tier: MemoryTier::Tier3NvmeSsd,
                gpu_layers: 0,
                pinned_ram_layers: 0,
                ssd_layers: total_layers,
                vram_weight_bytes: 0,
                vram_kv_and_workspace_bytes: 0,
                pinned_ram_bytes: 0,
                allocated_context_len: 0,
            },
        };

        self.apply_model_placement(model_id, new_plan.clone())?;
        Ok(new_plan)
    }

    pub fn increment_lease(&self, model_id: &str) -> Result<()> {
        let mut guard = self.models.write();
        let entry = guard
            .get_mut(model_id)
            .ok_or_else(|| HypnosError::ModelNotFound(model_id.to_string()))?;
        entry.active_leases += 1;
        entry.total_requests_served += 1;
        entry.last_used = Instant::now();
        Ok(())
    }

    pub fn decrement_lease(&self, model_id: &str) {
        let mut guard = self.models.write();
        if let Some(entry) = guard.get_mut(model_id) {
            entry.active_leases = entry.active_leases.saturating_sub(1);
            entry.last_used = Instant::now();
        }
    }

    pub fn active_leases(&self, model_id: &str) -> usize {
        self.models
            .read()
            .get(model_id)
            .map(|m| m.active_leases)
            .unwrap_or(0)
    }

    /// Returns candidate models currently occupying Tier 1 VRAM that have `active_leases == 0`,
    /// ordered from best eviction candidate (lowest priority, oldest `last_used`) to worst.
    pub fn idle_vram_eviction_candidates(&self, exclude_model_id: &str) -> Vec<String> {
        let guard = self.models.read();
        let mut candidates: Vec<(&String, &ManagedModelMemory)> = guard
            .iter()
            .filter(|(id, m)| {
                id.as_str() != exclude_model_id
                    && m.active_leases == 0
                    && m.placement.total_vram_bytes() > 0
                    && m.config.priority != hypnos_core::config::ModelPriority::RealtimePinned
            })
            .collect();

        // Sort by:
        // 1. Lowest priority first
        // 2. Oldest `last_used` timestamp first (LRU within priority class)
        candidates.sort_by(|(_, a), (_, b)| {
            a.config
                .priority
                .cmp(&b.config.priority)
                .then_with(|| a.last_used.cmp(&b.last_used))
        });

        candidates.into_iter().map(|(id, _)| id.clone()).collect()
    }

    /// Returns a full snapshot of the 3-tier hierarchy.
    pub fn status_snapshot(&self) -> MemoryHierarchyStatus {
        // Read before taking the models lock: `process_memory_bytes` takes it too, and a nested
        // read can deadlock behind a waiting writer.
        let process_footprint_bytes = self.process_footprint_bytes();
        let process_memory_bytes = self.process_memory_bytes();
        let guard = self.models.read();
        let now = Instant::now();
        let mut models: Vec<ModelTierSnapshot> = guard
            .values()
            .map(|m| ModelTierSnapshot {
                id: m.config.id.clone(),
                modality: m.config.modality,
                modalities: m.config.all_modalities(),
                engine: m.config.engine,
                total_weight_bytes: m.config.total_weight_bytes,
                priority: m.config.priority,
                current_tier: m.placement.primary_tier,
                placement: m.placement.clone(),
                active_leases: m.active_leases,
                idle_seconds: now.duration_since(m.last_used).as_secs_f64(),
                total_requests_served: m.total_requests_served,
            })
            .collect();

        models.sort_by(|a, b| a.id.cmp(&b.id));

        let has_real = self.has_real_reading.load(Ordering::SeqCst);
        MemoryHierarchyStatus {
            capabilities: self.capabilities(),
            tier1_vram_used_bytes: self.used_vram_bytes(),
            tier1_vram_free_bytes: self.free_vram_bytes(),
            tier1_vram_real_free_bytes: has_real
                .then(|| self.vram_real_free_bytes.load(Ordering::SeqCst)),
            tier1_vram_real_total_bytes: has_real
                .then(|| self.vram_real_total_bytes.load(Ordering::SeqCst)),
            tier1_vram_unaccounted_bytes: self.unaccounted_vram_bytes(),
            tier1_vram_effective_free_bytes: self.effective_free_vram_bytes(),
            tier2_pinned_ram_used_bytes: self.used_pinned_ram_bytes(),
            tier2_pinned_ram_free_bytes: self.free_pinned_ram_bytes(),
            physical_pool_free_bytes: self.physical_pool_free_bytes(),
            process_footprint_bytes,
            process_memory_bytes,
            process_memory_cap_bytes: (self.max_process_memory_bytes > 0)
                .then_some(self.max_process_memory_bytes),
            models,
        }
    }
}
