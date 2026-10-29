use hypnos_core::config::{AcceleratorKind, HardwareConfig, MemoryTopology};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, info};

const GIB: u64 = 1024 * 1024 * 1024;

/// Unified memory left to the OS, the window server and other GPU clients when Tier 1 and Tier 2
/// are carved out of the same physical RAM.
pub const UMA_OS_RESERVE_BYTES: u64 = 4 * GIB;

/// Tier 2 budget on unified memory: what physical RAM leaves after the GPU working set and the OS
/// reserve, never more than configured. Both tiers are the same RAM, so independent budgets would
/// let the planner admit models that cannot exist.
pub fn uma_pinned_ram_budget(configured: u64, physical: u64, working_set: u64) -> u64 {
    configured.min(
        physical
            .saturating_sub(working_set)
            .saturating_sub(UMA_OS_RESERVE_BYTES),
    )
}

/// Detected hardware capabilities and memory budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemoryCapabilities {
    pub accelerator: AcceleratorKind,
    pub topology: MemoryTopology,
    pub device_name: String,
    pub total_vram_bytes: u64,
    pub usable_vram_budget_bytes: u64,
    pub pinned_ram_budget_bytes: u64,
    /// Physical RAM shared by CPU and GPU on unified memory (0 on discrete GPUs or when unknown).
    #[serde(default)]
    pub physical_memory_bytes: u64,
    /// Estimated Tier 2 -> Tier 1 transfer bandwidth in GB/s
    pub tier2_to_tier1_bandwidth_gbps: f64,
    /// Estimated Tier 3 -> Tier 2 NVMe read bandwidth in GB/s
    pub tier3_to_tier2_bandwidth_gbps: f64,
}

impl DeviceMemoryCapabilities {
    /// Probes the current system or uses the explicit overrides in `HardwareConfig`.
    pub fn detect(config: &HardwareConfig) -> Self {
        let resolved_accel = match config.accelerator {
            AcceleratorKind::Auto => Self::auto_detect_accelerator(),
            explicit => explicit,
        };

        let mut physical_memory_bytes = 0;
        let (topology, device_name, detected_vram, t2_to_t1_bw) = match resolved_accel {
            AcceleratorKind::NvidiaCuda | AcceleratorKind::Auto => {
                if let Some((name, vram)) = Self::probe_nvidia_smi(config.device_id) {
                    (MemoryTopology::DiscretePcie, name, vram, 28.0)
                } else {
                    (
                        MemoryTopology::DiscretePcie,
                        "NVIDIA CUDA Device (Configured Budget)".to_string(),
                        config.vram_budget_bytes.max(8 * 1024 * 1024 * 1024),
                        28.0,
                    )
                }
            }
            AcceleratorKind::AmdRocm => {
                if let Some((name, vram, is_apu)) = Self::probe_rocm_smi(config.device_id) {
                    let topo = if is_apu {
                        MemoryTopology::UnifiedMemory
                    } else {
                        MemoryTopology::DiscretePcie
                    };
                    let bw = if is_apu { 256.0 } else { 26.0 };
                    (topo, name, vram, bw)
                } else {
                    (
                        MemoryTopology::DiscretePcie,
                        "AMD ROCm HIP Device (Configured Budget)".to_string(),
                        config.vram_budget_bytes.max(16 * 1024 * 1024 * 1024),
                        26.0,
                    )
                }
            }
            AcceleratorKind::AppleMetalUma => {
                let total_uma = match Self::probe_macos_unified_memory() {
                    Some((physical, working_set)) => {
                        physical_memory_bytes = physical;
                        working_set
                    }
                    None => config.vram_budget_bytes.max(16 * GIB),
                };
                (
                    MemoryTopology::UnifiedMemory,
                    "Apple Silicon Metal (Unified Memory)".to_string(),
                    total_uma,
                    400.0, // Zero-copy page wiring in unified memory
                )
            }
            AcceleratorKind::Vulkan => (
                MemoryTopology::DiscretePcie,
                "Vulkan Compute Device".to_string(),
                config.vram_budget_bytes.max(8 * 1024 * 1024 * 1024),
                22.0,
            ),
        };

        let total_vram_bytes = if config.vram_budget_bytes > 0 {
            config.vram_budget_bytes
        } else {
            detected_vram
        };

        let usable_vram_budget_bytes = total_vram_bytes
            .saturating_sub(config.vram_safety_margin_bytes)
            .max(512 * 1024 * 1024);

        let pinned_ram_budget_bytes =
            if topology == MemoryTopology::UnifiedMemory && physical_memory_bytes > 0 {
                let derived = uma_pinned_ram_budget(
                    config.pinned_ram_budget_bytes,
                    physical_memory_bytes,
                    total_vram_bytes,
                );
                if derived < config.pinned_ram_budget_bytes {
                    info!(
                    configured_mb = config.pinned_ram_budget_bytes / (1024 * 1024),
                    derived_mb = derived / (1024 * 1024),
                    physical_mb = physical_memory_bytes / (1024 * 1024),
                    "Unified memory: Tier 2 budget limited to RAM the GPU working set leaves free"
                );
                }
                derived
            } else {
                config.pinned_ram_budget_bytes
            };

        let caps = Self {
            accelerator: resolved_accel,
            topology,
            device_name,
            total_vram_bytes,
            usable_vram_budget_bytes,
            pinned_ram_budget_bytes,
            physical_memory_bytes,
            tier2_to_tier1_bandwidth_gbps: t2_to_t1_bw,
            tier3_to_tier2_bandwidth_gbps: 5.5, // Typical PCIe 4.0 NVMe SSD sequential read
        };

        info!(
            accelerator = ?caps.accelerator,
            topology = ?caps.topology,
            device = %caps.device_name,
            usable_vram_mb = caps.usable_vram_budget_bytes / (1024 * 1024),
            pinned_ram_mb = caps.pinned_ram_budget_bytes / (1024 * 1024),
            physical_mb = caps.physical_memory_bytes / (1024 * 1024),
            "Initialized hardware memory topology"
        );

        caps
    }

    /// Unified memory: replaces the Tier 1 ceiling with the working set the GPU driver reports
    /// (Metal's `recommendedMaxWorkingSetSize`), keeping the safety margin, and re-derives Tier 2.
    pub fn apply_uma_working_set(&mut self, working_set: u64, configured_pinned_ram: u64) {
        let margin = self
            .total_vram_bytes
            .saturating_sub(self.usable_vram_budget_bytes);
        self.total_vram_bytes = working_set;
        self.usable_vram_budget_bytes = working_set.saturating_sub(margin).max(512 * 1024 * 1024);
        if self.physical_memory_bytes > 0 {
            self.pinned_ram_budget_bytes = uma_pinned_ram_budget(
                configured_pinned_ram,
                self.physical_memory_bytes,
                working_set,
            );
        }
    }

    fn auto_detect_accelerator() -> AcceleratorKind {
        if cfg!(target_os = "macos") && cfg!(target_arch = "aarch64") {
            return AcceleratorKind::AppleMetalUma;
        }
        if Self::probe_nvidia_smi(0).is_some() {
            return AcceleratorKind::NvidiaCuda;
        }
        if Self::probe_rocm_smi(0).is_some() {
            return AcceleratorKind::AmdRocm;
        }
        AcceleratorKind::NvidiaCuda
    }

    fn probe_nvidia_smi(device_id: usize) -> Option<(String, u64)> {
        let output = Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
                &format!("--id={device_id}"),
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let line = String::from_utf8_lossy(&output.stdout);
        let first_line = line.lines().next()?.trim();
        let mut parts = first_line.split(',');
        let name = parts.next()?.trim().to_string();
        let mem_mib: u64 = parts.next()?.trim().parse().ok()?;
        debug!(device = %name, vram_mib = mem_mib, "Detected NVIDIA GPU via nvidia-smi");
        Some((name, mem_mib * 1024 * 1024))
    }

    fn probe_rocm_smi(_device_id: usize) -> Option<(String, u64, bool)> {
        let output = Command::new("rocm-smi")
            .args(["--showmeminfo", "vram", "--csv"])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        for line in stdout.lines().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            if cols.len() >= 2 {
                if let Ok(vram_bytes) = cols[1].trim().parse::<u64>() {
                    return Some(("AMD ROCm GPU".to_string(), vram_bytes, false));
                }
            }
        }
        None
    }

    /// Returns `(physical_bytes, gpu_working_set_bytes)`. The working set is a first estimate;
    /// once llama.cpp is initialised the Metal device's own figure replaces it (see
    /// `TieredMemoryManager::set_gpu_probe`).
    fn probe_macos_unified_memory() -> Option<(u64, u64)> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        let physical = Self::sysctl_u64("hw.memsize")?;
        // A user-raised wired limit (`sudo sysctl iogpu.wired_limit_mb=...`, macOS 14+; older
        // systems: `debug.iogpu.wired_limit` in bytes) is what Metal will actually grant.
        let working_set = match Self::sysctl_u64("iogpu.wired_limit_mb") {
            Some(mib) if mib > 0 => mib * 1024 * 1024,
            _ => match Self::sysctl_u64("debug.iogpu.wired_limit") {
                Some(bytes) if bytes > 0 => bytes,
                // Metal's default recommendedMaxWorkingSetSize is roughly 75% of RAM.
                _ => (physical as f64 * 0.75) as u64,
            },
        };
        Some((physical, working_set.min(physical)))
    }

    fn sysctl_u64(name: &str) -> Option<u64> {
        let output = Command::new("sysctl").args(["-n", name]).output().ok()?;
        if !output.status.success() {
            return None;
        }
        String::from_utf8_lossy(&output.stdout).trim().parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uma_tier2_budget_never_exceeds_physical_memory() {
        // 32 GB Mac, ~24 GB Metal working set, 32 GB configured Tier 2.
        let physical = 32 * GIB;
        let working_set = 24 * GIB;
        let tier2 = uma_pinned_ram_budget(32 * GIB, physical, working_set);
        assert_eq!(tier2, physical - working_set - UMA_OS_RESERVE_BYTES);
        assert!(working_set + tier2 <= physical);
        // A configured budget below the leftover is kept.
        assert_eq!(uma_pinned_ram_budget(GIB, physical, working_set), GIB);
        // A raised wired limit that leaves less than the OS reserve yields no Tier 2 at all.
        assert_eq!(uma_pinned_ram_budget(32 * GIB, physical, 30 * GIB), 0);
    }

    #[test]
    fn device_working_set_rederives_both_budgets() {
        let mut caps = DeviceMemoryCapabilities {
            accelerator: AcceleratorKind::AppleMetalUma,
            topology: MemoryTopology::UnifiedMemory,
            device_name: "test".into(),
            total_vram_bytes: 48 * GIB,
            usable_vram_budget_bytes: 46 * GIB,
            pinned_ram_budget_bytes: 12 * GIB,
            physical_memory_bytes: 64 * GIB,
            tier2_to_tier1_bandwidth_gbps: 400.0,
            tier3_to_tier2_bandwidth_gbps: 5.5,
        };
        caps.apply_uma_working_set(52 * GIB, 32 * GIB);
        assert_eq!(caps.total_vram_bytes, 52 * GIB);
        assert_eq!(caps.usable_vram_budget_bytes, 50 * GIB);
        assert_eq!(caps.pinned_ram_budget_bytes, 8 * GIB);
    }
}
