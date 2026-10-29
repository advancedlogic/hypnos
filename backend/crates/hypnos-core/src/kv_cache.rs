use crate::config::{MemoryTier, ModelConfig};
use serde::{Deserialize, Serialize};

/// Describes how a model's layers and buffers are distributed across the 3 memory tiers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LayerPlacementPlan {
    /// Current highest active tier of the model
    pub primary_tier: MemoryTier,
    /// Number of layers placed in Tier 1 GPU VRAM (or wired Metal UMA)
    pub gpu_layers: u32,
    /// Number of layers placed in Tier 2 Pinned Host RAM
    pub pinned_ram_layers: u32,
    /// Number of layers remaining on Tier 3 NVMe SSD
    pub ssd_layers: u32,
    /// Total bytes occupied in Tier 1 GPU VRAM by model weights
    pub vram_weight_bytes: u64,
    /// Total bytes occupied in Tier 1 GPU VRAM by reserved KV cache + workspace
    pub vram_kv_and_workspace_bytes: u64,
    /// Total bytes occupied in Tier 2 Pinned Host RAM by model weights (and CPU-side KV if split)
    pub pinned_ram_bytes: u64,
    /// Active context window length funded by this placement
    pub allocated_context_len: u32,
}

impl LayerPlacementPlan {
    /// Total Tier 1 GPU VRAM footprint (weights + KV cache + workspace).
    pub fn total_vram_bytes(&self) -> u64 {
        self.vram_weight_bytes
            .saturating_add(self.vram_kv_and_workspace_bytes)
    }

    /// Fraction of layers currently residing on the GPU [0.0 ..= 1.0].
    pub fn gpu_layer_ratio(&self) -> f32 {
        let total = self.gpu_layers + self.pinned_ram_layers + self.ssd_layers;
        if total == 0 {
            1.0
        } else {
            self.gpu_layers as f32 / total as f32
        }
    }
}

/// Computes exact KV cache and partial-layer memory footprints for any model,
/// including Gemma 4 hybrid Sliding Window Attention (SWA) + Global Attention models.
pub struct KvCacheEstimator;

impl KvCacheEstimator {
    /// Calculates the KV cache size in bytes for a given number of layers, sequence length, and batch size.
    ///
    /// Supports both standard full-context attention and Gemma 4's hybrid Sliding Window Attention
    /// (`sliding_window_len` + `global_attn_every_n_layers`, e.g., 5 SWA layers per 1 Global layer).
    pub fn kv_bytes_for_layers(
        model: &ModelConfig,
        layers: u32,
        context_len: u32,
        batch_size: u32,
    ) -> u64 {
        if model.arch.num_kv_heads == 0 || layers == 0 || context_len == 0 {
            return 0;
        }

        let elements_per_token_per_layer =
            2u64 * (model.arch.num_kv_heads as u64) * (model.arch.head_dim as u64);

        let total_layer_tokens: u64 = match (
            model.arch.sliding_window_len,
            model.arch.global_attn_every_n_layers,
        ) {
            (Some(swa_len), Some(interval)) if interval > 1 && context_len > swa_len => {
                // In Gemma 4 (interval = 6), 1 out of 6 layers is Global (full `context_len`),
                // while 5 out of 6 layers are capped at `swa_len` tokens.
                let global_layers = layers.div_ceil(interval);
                let local_swa_layers = layers.saturating_sub(global_layers);
                (global_layers as u64)
                    .saturating_mul(context_len as u64)
                    .saturating_add((local_swa_layers as u64).saturating_mul(swa_len as u64))
            }
            (Some(swa_len), None) => {
                (layers as u64).saturating_mul(context_len.min(swa_len) as u64)
            }
            _ => (layers as u64).saturating_mul(context_len as u64),
        };

        let total_elements = elements_per_token_per_layer
            .saturating_mul(total_layer_tokens)
            .saturating_mul(batch_size.max(1) as u64);

        ((total_elements as f64) * model.arch.kv_dtype.bytes_per_element()).ceil() as u64
    }

    /// Computes total KV cache + transient activation workspace needed on GPU for `gpu_layers`.
    pub fn vram_runtime_overhead(
        model: &ModelConfig,
        gpu_layers: u32,
        context_len: u32,
        batch_size: u32,
    ) -> u64 {
        let kv_bytes = if model.arch.num_kv_heads > 0 {
            Self::kv_bytes_for_layers(model, gpu_layers, context_len, batch_size)
        } else {
            0
        };

        kv_bytes.saturating_add(model.arch.activation_workspace_bytes)
    }

    /// Computes the weight size of a single splittable layer in bytes.
    pub fn bytes_per_layer(model: &ModelConfig) -> u64 {
        let total_layers = model.arch.num_layers.max(1) as u64;
        let splittable_bytes = model
            .total_weight_bytes
            .saturating_sub(model.arch.non_layer_overhead_bytes);
        splittable_bytes.div_ceil(total_layers)
    }

    /// Plans the optimal layer split across Tier 1 (GPU VRAM) and Tier 2 (Pinned Host RAM)
    /// given `available_vram_bytes` and `available_pinned_ram_bytes`.
    pub fn compute_optimal_placement(
        model: &ModelConfig,
        requested_context_len: Option<u32>,
        batch_size: u32,
        available_vram_bytes: u64,
        available_pinned_ram_bytes: u64,
        is_unified_memory: bool,
    ) -> Option<LayerPlacementPlan> {
        let context_len = requested_context_len
            .unwrap_or(model.arch.default_context_len)
            .min(model.arch.max_context_len);

        let total_layers = model.arch.num_layers.max(1);
        let min_gpu_layers =
            ((total_layers as f32) * model.min_gpu_layer_ratio.clamp(0.0, 1.0)).ceil() as u32;
        let min_gpu_layers = min_gpu_layers.max(1).min(total_layers);

        let fixed_vram_weights = model
            .arch
            .non_layer_overhead_bytes
            .min(model.total_weight_bytes);
        let per_layer_weight = Self::bytes_per_layer(model);

        for candidate_gpu_layers in (min_gpu_layers..=total_layers).rev() {
            let splittable_vram = if candidate_gpu_layers == total_layers {
                model.total_weight_bytes.saturating_sub(fixed_vram_weights)
            } else {
                per_layer_weight.saturating_mul(candidate_gpu_layers as u64)
            };

            let vram_weight_bytes = fixed_vram_weights.saturating_add(splittable_vram);
            let vram_kv_and_workspace_bytes =
                Self::vram_runtime_overhead(model, candidate_gpu_layers, context_len, batch_size);
            let total_vram_needed = vram_weight_bytes.saturating_add(vram_kv_and_workspace_bytes);

            if total_vram_needed <= available_vram_bytes {
                let cpu_layers = total_layers - candidate_gpu_layers;
                let cpu_weight_bytes = model.total_weight_bytes.saturating_sub(vram_weight_bytes);
                let cpu_kv_bytes =
                    Self::kv_bytes_for_layers(model, cpu_layers, context_len, batch_size);

                let required_pinned_ram = if is_unified_memory {
                    cpu_weight_bytes.saturating_add(cpu_kv_bytes)
                } else if model.keep_warm_in_ram
                    && model.total_weight_bytes.saturating_add(cpu_kv_bytes)
                        <= available_pinned_ram_bytes
                {
                    model.total_weight_bytes.saturating_add(cpu_kv_bytes)
                } else {
                    cpu_weight_bytes.saturating_add(cpu_kv_bytes)
                };

                if required_pinned_ram <= available_pinned_ram_bytes {
                    return Some(LayerPlacementPlan {
                        primary_tier: MemoryTier::Tier1GpuVram,
                        gpu_layers: candidate_gpu_layers,
                        pinned_ram_layers: cpu_layers,
                        ssd_layers: 0,
                        vram_weight_bytes,
                        vram_kv_and_workspace_bytes,
                        pinned_ram_bytes: required_pinned_ram,
                        allocated_context_len: context_len,
                    });
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{EngineKind, KvDataType, Modality, ModelArchitectureParams, ModelPriority};
    use std::path::PathBuf;

    fn sample_gemma4_26b_config() -> ModelConfig {
        ModelConfig {
            id: "gemma4:26b-q4".to_string(),
            modality: Modality::TextGeneration,
            additional_modalities: vec![Modality::Vision],
            engine: EngineKind::GgmlLlama,
            weights_path: PathBuf::from("models/gemma-4-26b-it-q4_k_m.gguf"),
            aux_weights_path: Some(PathBuf::from("models/gemma-4-26b-mmproj-f16.gguf")),
            text_encoder_path: None,
            tokenizer_path: None,
            voices_path: None,
            total_weight_bytes: 15_500 * 1024 * 1024, // ~15.5 GB
            min_gpu_layer_ratio: 0.25,
            initial_tier: MemoryTier::Tier2PinnedRam,
            keep_warm_in_ram: true,
            priority: ModelPriority::High,
            arch: ModelArchitectureParams {
                num_layers: 46,
                num_kv_heads: 8,
                head_dim: 256,
                default_context_len: 16384,
                max_context_len: 131072,
                sliding_window_len: Some(1024),
                global_attn_every_n_layers: Some(6),
                kv_dtype: KvDataType::Q8_0,
                non_layer_overhead_bytes: 1200 * 1024 * 1024,
                activation_workspace_bytes: 512 * 1024 * 1024,
            },
        }
    }

    #[test]
    fn test_gemma4_hybrid_swa_kv_cache_savings() {
        let mut model = sample_gemma4_26b_config();
        let swa_kv_bytes = KvCacheEstimator::kv_bytes_for_layers(&model, 46, 16384, 1);

        // Compare against naive full-context KV cache without SWA
        model.arch.sliding_window_len = None;
        model.arch.global_attn_every_n_layers = None;
        let full_kv_bytes = KvCacheEstimator::kv_bytes_for_layers(&model, 46, 16384, 1);

        // Hybrid 5:1 SWA at 16k context saves >70% KV cache VRAM!
        assert!(swa_kv_bytes < full_kv_bytes / 3);
    }

    #[test]
    fn test_partial_layer_offloading_for_gemma4_26b_on_16gb_vram() {
        let model = sample_gemma4_26b_config();
        // On a 16 GB GPU (15.5 GB usable VRAM), Gemma 4 26B Q4 (15.5 GB weights + KV + workspace)
        // automatically splits layers between GPU VRAM and Pinned Host RAM!
        let avail_vram = 15_500 * 1024 * 1024;
        let avail_ram = 32 * 1024 * 1024 * 1024;

        let plan = KvCacheEstimator::compute_optimal_placement(
            &model,
            Some(16384),
            1,
            avail_vram,
            avail_ram,
            false,
        )
        .expect("Should find partial GPU+RAM offload plan for Gemma 4 26B on 16GB VRAM");

        assert!(plan.gpu_layers >= 40 && plan.gpu_layers < 46);
        assert_eq!(plan.gpu_layers + plan.pinned_ram_layers, 46);
        assert!(plan.total_vram_bytes() <= avail_vram);
    }
}
