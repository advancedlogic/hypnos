use hypnos_core::config::{AcceleratorKind, HardwareConfig, MemoryTopology};
use serde::{Deserialize, Serialize};
use std::process::Command;
use tracing::{debug, info};

/// Detected hardware capabilities and memory budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemoryCapabilities {
    pub accelerator: AcceleratorKind,
    pub topology: MemoryTopology,
    pub device_name: String,
    pub total_vram_bytes: u64,
    pub usable_vram_budget_bytes: u64,
    pub pinned_ram_budget_bytes: u64,
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
                let total_uma = Self::probe_macos_unified_memory()
                    .unwrap_or_else(|| config.vram_budget_bytes.max(16 * 1024 * 1024 * 1024));
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

        let caps = Self {
            accelerator: resolved_accel,
            topology,
            device_name,
            total_vram_bytes,
            usable_vram_budget_bytes,
            pinned_ram_budget_bytes: config.pinned_ram_budget_bytes,
            tier2_to_tier1_bandwidth_gbps: t2_to_t1_bw,
            tier3_to_tier2_bandwidth_gbps: 5.5, // Typical PCIe 4.0 NVMe SSD sequential read
        };

        info!(
            accelerator = ?caps.accelerator,
            topology = ?caps.topology,
            device = %caps.device_name,
            usable_vram_mb = caps.usable_vram_budget_bytes / (1024 * 1024),
            pinned_ram_mb = caps.pinned_ram_budget_bytes / (1024 * 1024),
            "Initialized hardware memory topology"
        );

        caps
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

    fn probe_macos_unified_memory() -> Option<u64> {
        if !cfg!(target_os = "macos") {
            return None;
        }
        let output = Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let mem_bytes: u64 = String::from_utf8_lossy(&output.stdout)
            .trim()
            .parse()
            .ok()?;
        // macOS allows ~75% of Unified Memory for wired Metal working set by default
        Some((mem_bytes as f64 * 0.75) as u64)
    }
}
