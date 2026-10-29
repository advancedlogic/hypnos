use hypnos_core::config::{AcceleratorKind, HardwareConfig, MemoryTopology};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, info, warn};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

/// Physical RAM left to the OS and other applications: the Tier 2 budget never claims it, and on
/// unified memory it is also kept away from the window server and other GPU clients.
pub const OS_RESERVE_BYTES: u64 = 4 * GIB;

/// Tier 2 budget on unified memory: what physical RAM leaves after the GPU working set and the OS
/// reserve, never more than configured. Both tiers are the same RAM, so independent budgets would
/// let the planner admit models that cannot exist.
pub fn uma_pinned_ram_budget(configured: u64, physical: u64, working_set: u64) -> u64 {
    configured.min(
        physical
            .saturating_sub(working_set)
            .saturating_sub(OS_RESERVE_BYTES),
    )
}

/// Estimate of Metal's default `recommendedMaxWorkingSetSize` for a Mac with `physical` bytes of
/// RAM. Apple does not document it; the commonly reported values are about two thirds of RAM up to
/// 36 GB and three quarters above. Newer systems can grant more (a 128 GB M5 Max on macOS 27
/// reports 84%), so this errs low. Outside a simulation the device's own figure replaces it.
pub fn metal_default_working_set(physical: u64) -> u64 {
    if physical <= 36 * GIB {
        physical / 3 * 2
    } else {
        physical / 4 * 3
    }
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
    /// Physical RAM (shared by CPU and GPU on unified memory), or the simulated amount when
    /// `simulate_physical_memory_bytes` is set; 0 when unknown.
    #[serde(default)]
    pub physical_memory_bytes: u64,
    /// True when `physical_memory_bytes` and the budgets derived from it are a simulation.
    #[serde(default)]
    pub simulated: bool,
    /// RAM actually installed (equal to `physical_memory_bytes` unless simulating; 0 if unknown).
    #[serde(default)]
    pub installed_memory_bytes: u64,
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

        let real_physical = crate::process::physical_memory_bytes().unwrap_or(0);
        let simulated = config.simulate_physical_memory_bytes > 0;
        let physical_memory_bytes = match (simulated, real_physical) {
            (false, real) => real,
            (true, 0) => config.simulate_physical_memory_bytes,
            (true, real) => config.simulate_physical_memory_bytes.min(real),
        };
        if simulated {
            if real_physical > 0 && config.simulate_physical_memory_bytes > real_physical {
                warn!(
                    requested_mb = config.simulate_physical_memory_bytes / MIB,
                    real_mb = real_physical / MIB,
                    "simulate_physical_memory_bytes is above the installed RAM; using the real amount"
                );
            }
            info!(
                simulated_mb = physical_memory_bytes / MIB,
                real_mb = real_physical / MIB,
                "Simulating a machine with this much physical RAM; budgets are derived from it"
            );
        }

        // `vram_measured`: the device reported its memory, so a configured budget can be checked
        // against it. Estimates are refined later by the engine's probe (`set_gpu_probe`).
        let (topology, device_name, detected_vram, vram_measured, t2_to_t1_bw) =
            match resolved_accel {
                AcceleratorKind::NvidiaCuda | AcceleratorKind::Auto => {
                    if let Some((name, vram)) = Self::probe_nvidia_smi(config.device_id) {
                        (MemoryTopology::DiscretePcie, name, vram, true, 28.0)
                    } else {
                        (
                            MemoryTopology::DiscretePcie,
                            "NVIDIA CUDA Device (Configured Budget)".to_string(),
                            config.vram_budget_bytes.max(8 * 1024 * 1024 * 1024),
                            false,
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
                        (topo, name, vram, true, bw)
                    } else {
                        (
                            MemoryTopology::DiscretePcie,
                            "AMD ROCm HIP Device (Configured Budget)".to_string(),
                            config.vram_budget_bytes.max(16 * 1024 * 1024 * 1024),
                            false,
                            26.0,
                        )
                    }
                }
                AcceleratorKind::AppleMetalUma => {
                    // A user-raised wired limit is what Metal will grant on this machine; a simulated
                    // machine gets Metal's default for its size instead.
                    let wired_limit = if simulated {
                        None
                    } else {
                        Self::macos_wired_limit_bytes()
                    };
                    let working_set = match (wired_limit, physical_memory_bytes) {
                        (Some(limit), physical) if physical > 0 => limit.min(physical),
                        (Some(limit), _) => limit,
                        (None, 0) => config.vram_budget_bytes.max(16 * GIB),
                        (None, physical) => metal_default_working_set(physical),
                    };
                    (
                        MemoryTopology::UnifiedMemory,
                        "Apple Silicon Metal (Unified Memory)".to_string(),
                        working_set,
                        false,
                        400.0, // Zero-copy page wiring in unified memory
                    )
                }
                AcceleratorKind::Vulkan => (
                    MemoryTopology::DiscretePcie,
                    "Vulkan Compute Device".to_string(),
                    config.vram_budget_bytes.max(8 * 1024 * 1024 * 1024),
                    false,
                    22.0,
                ),
            };

        let mut total_vram_bytes = if config.vram_budget_bytes > 0 {
            config.vram_budget_bytes
        } else {
            detected_vram
        };
        // A configured budget is a cap, never a promise of memory the hardware does not have.
        let vram_ceiling = if vram_measured {
            detected_vram
        } else if topology == MemoryTopology::UnifiedMemory && physical_memory_bytes > 0 {
            physical_memory_bytes.saturating_sub(OS_RESERVE_BYTES)
        } else {
            u64::MAX
        };
        if total_vram_bytes > vram_ceiling {
            warn!(
                configured_mb = total_vram_bytes / MIB,
                available_mb = vram_ceiling / MIB,
                "vram_budget_bytes exceeds what this machine has; clamped"
            );
            total_vram_bytes = vram_ceiling;
        }

        let usable_vram_budget_bytes = total_vram_bytes
            .saturating_sub(config.vram_safety_margin_bytes)
            .max(512 * 1024 * 1024);

        let pinned_ram_budget_bytes = if physical_memory_bytes == 0 {
            config.pinned_ram_budget_bytes
        } else {
            let derived = if topology == MemoryTopology::UnifiedMemory {
                uma_pinned_ram_budget(
                    config.pinned_ram_budget_bytes,
                    physical_memory_bytes,
                    total_vram_bytes,
                )
            } else {
                config
                    .pinned_ram_budget_bytes
                    .min(physical_memory_bytes.saturating_sub(OS_RESERVE_BYTES))
            };
            if derived < config.pinned_ram_budget_bytes {
                info!(
                    configured_mb = config.pinned_ram_budget_bytes / MIB,
                    derived_mb = derived / MIB,
                    physical_mb = physical_memory_bytes / MIB,
                    "Tier 2 budget limited to the RAM this machine leaves free"
                );
            }
            derived
        };

        let caps = Self {
            accelerator: resolved_accel,
            topology,
            device_name,
            total_vram_bytes,
            usable_vram_budget_bytes,
            pinned_ram_budget_bytes,
            physical_memory_bytes,
            simulated,
            installed_memory_bytes: real_physical,
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

    /// Replaces the Tier 1 ceiling (e.g. with the memory the GPU driver reports, Metal's
    /// `recommendedMaxWorkingSetSize` on Apple Silicon), keeping the safety margin. On unified
    /// memory Tier 2 is re-derived from it.
    pub fn set_vram_total(&mut self, total: u64, configured_pinned_ram: u64) {
        let margin = self
            .total_vram_bytes
            .saturating_sub(self.usable_vram_budget_bytes);
        self.total_vram_bytes = total;
        self.usable_vram_budget_bytes = total.saturating_sub(margin).max(512 * 1024 * 1024);
        if self.topology == MemoryTopology::UnifiedMemory && self.physical_memory_bytes > 0 {
            self.pinned_ram_budget_bytes =
                uma_pinned_ram_budget(configured_pinned_ram, self.physical_memory_bytes, total);
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

    /// A GPU wired-memory limit the user raised (`sudo sysctl iogpu.wired_limit_mb=...` on
    /// macOS 14+, `debug.iogpu.wired_limit` in bytes before). `None` when unset (0) or not macOS.
    /// Once llama.cpp is initialised the Metal device's own figure replaces any estimate (see
    /// `TieredMemoryManager::set_gpu_probe`).
    fn macos_wired_limit_bytes() -> Option<u64> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        match Self::sysctl_u64("iogpu.wired_limit_mb") {
            Some(mib) if mib > 0 => Some(mib * MIB),
            _ => Self::sysctl_u64("debug.iogpu.wired_limit").filter(|&b| b > 0),
        }
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
        assert_eq!(tier2, physical - working_set - OS_RESERVE_BYTES);
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
            simulated: false,
            installed_memory_bytes: 64 * GIB,
            tier2_to_tier1_bandwidth_gbps: 400.0,
            tier3_to_tier2_bandwidth_gbps: 5.5,
        };
        caps.set_vram_total(52 * GIB, 32 * GIB);
        assert_eq!(caps.total_vram_bytes, 52 * GIB);
        assert_eq!(caps.usable_vram_budget_bytes, 50 * GIB);
        assert_eq!(caps.pinned_ram_budget_bytes, 8 * GIB);
    }

    fn has_at_least_32gb() -> bool {
        crate::process::physical_memory_bytes().unwrap_or(0) >= 32 * GIB
    }

    fn uma_config(simulate_gib: u64, vram_budget_gib: u64) -> HardwareConfig {
        HardwareConfig {
            accelerator: AcceleratorKind::AppleMetalUma,
            vram_budget_bytes: vram_budget_gib * GIB,
            vram_safety_margin_bytes: 2 * GIB,
            pinned_ram_budget_bytes: 32 * GIB,
            simulate_physical_memory_bytes: simulate_gib * GIB,
            ..HardwareConfig::default()
        }
    }

    #[test]
    fn simulated_32gb_mac_gets_its_own_budgets() {
        if !has_at_least_32gb() {
            return; // a simulation never exceeds the installed RAM
        }
        let caps = DeviceMemoryCapabilities::detect(&uma_config(32, 0));
        assert!(caps.simulated);
        assert_eq!(caps.physical_memory_bytes, 32 * GIB);
        // Metal's default for a 32 GB machine: two thirds of RAM.
        assert_eq!(caps.total_vram_bytes, 32 * GIB / 3 * 2);
        assert_eq!(
            caps.usable_vram_budget_bytes,
            caps.total_vram_bytes - 2 * GIB
        );
        let tier2 = 32 * GIB - caps.total_vram_bytes - OS_RESERVE_BYTES;
        assert_eq!(caps.pinned_ram_budget_bytes, tier2);
        assert!(caps.total_vram_bytes + caps.pinned_ram_budget_bytes <= 32 * GIB);
    }

    #[test]
    fn configured_vram_budget_is_clamped_to_physical_memory() {
        if !has_at_least_32gb() {
            return; // a simulation never exceeds the installed RAM
        }
        // 100 GiB of "VRAM" on a simulated 32 GB machine is clamped to what the machine has.
        let caps = DeviceMemoryCapabilities::detect(&uma_config(32, 100));
        assert_eq!(caps.total_vram_bytes, 32 * GIB - OS_RESERVE_BYTES);
        assert_eq!(caps.pinned_ram_budget_bytes, 0);
    }

    #[test]
    fn metal_working_set_estimate() {
        assert_eq!(metal_default_working_set(16 * GIB), 16 * GIB / 3 * 2);
        assert_eq!(metal_default_working_set(64 * GIB), 48 * GIB);
    }
}
