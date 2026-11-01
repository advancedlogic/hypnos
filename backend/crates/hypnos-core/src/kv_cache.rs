use crate::config::{MemoryTier, ModelConfig, MoeOffload};
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
    /// Layers (counted from the last) whose Mixture-of-Experts expert tensors run on the CPU from
    /// host memory while the rest of the layer stays on the GPU (`moe_offload`, MEM-5). Their
    /// bytes are booked in `pinned_ram_bytes`, not in `vram_weight_bytes`.
    #[serde(default)]
    pub cpu_expert_layers: u32,
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

        // The exact layout (TEST-3): full-context layers, and sliding-window layers whose cache
        // llama.cpp sizes as the window plus one micro-batch, padded to 256 cells (2048-token
        // micro-batches with a vision / audio projector, 512 otherwise; see engine.rs).
        if let (Some(global), Some(swa)) = (model.arch.global_kv_layers, model.arch.swa_kv_layers) {
            let total = model.arch.num_layers.max(1) as u64;
            let share =
                |n: u32| (n as u64 * layers.min(model.arch.num_layers) as u64).div_ceil(total);
            let ubatch: u64 = if model.aux_weights_path.is_some() {
                2048
            } else {
                512
            };
            let swa_tokens = model
                .arch
                .sliding_window_len
                .map(|w| (w as u64 + ubatch).div_ceil(256) * 256)
                .unwrap_or(context_len as u64)
                .min(context_len as u64);
            let swa_per_token = 2u64
                * model
                    .arch
                    .swa_num_kv_heads
                    .unwrap_or(model.arch.num_kv_heads) as u64
                * model.arch.swa_head_dim.unwrap_or(model.arch.head_dim) as u64;
            let global_per_token = 2u64
                * model
                    .arch
                    .global_num_kv_heads
                    .unwrap_or(model.arch.num_kv_heads) as u64
                * model.arch.head_dim as u64;
            let elements = global_per_token
                .saturating_mul(share(global))
                .saturating_mul(context_len as u64)
                .saturating_add(
                    swa_per_token
                        .saturating_mul(share(swa))
                        .saturating_mul(swa_tokens),
                )
                .saturating_mul(batch_size.max(1) as u64)
                .saturating_mul(model.slots.max(1) as u64);
            return ((elements as f64) * model.arch.kv_dtype.bytes_per_element()).ceil() as u64;
        }

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
            .saturating_mul(batch_size.max(1) as u64)
            // Every execution slot keeps its own KV cache of `context_len` positions.
            .saturating_mul(model.slots.max(1) as u64);

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
    ///
    /// On unified memory only a full GPU placement is considered: CPU and GPU read the same RAM at
    /// the same bandwidth, so a partial split saves no memory and only adds CPU compute. If the full
    /// model does not fit, `None` lets the scheduler evict or reject instead.
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
        if model.runs_on_cpu() {
            // Loaded means everything in host RAM: the weights plus the engine's working memory.
            // On unified memory that RAM is the same pool, so the Tier 1 view still applies.
            let ram = model
                .total_weight_bytes
                .saturating_add(model.arch.activation_workspace_bytes);
            return (ram <= available_pinned_ram_bytes).then_some(LayerPlacementPlan {
                primary_tier: MemoryTier::Tier1GpuVram,
                gpu_layers: 0,
                pinned_ram_layers: total_layers,
                ssd_layers: 0,
                vram_weight_bytes: 0,
                vram_kv_and_workspace_bytes: 0,
                pinned_ram_bytes: ram,
                allocated_context_len: context_len,
                cpu_expert_layers: 0,
            });
        }
        let min_gpu_layers =
            ((total_layers as f32) * model.min_gpu_layer_ratio.clamp(0.0, 1.0)).ceil() as u32;
        let min_gpu_layers = if is_unified_memory {
            total_layers
        } else {
            min_gpu_layers.max(1).min(total_layers)
        };

        let fixed_vram_weights = model
            .arch
            .non_layer_overhead_bytes
            .min(model.total_weight_bytes);
        let per_layer_weight = Self::bytes_per_layer(model);

        // Mixture-of-Experts offload (MEM-5): the whole model on the GPU except the expert tensors
        // of the last `n` layers, which stay in host memory and run on the CPU. `on` offloads
        // every layer's experts; `auto` takes the smallest `n` that fits (0 = no offload) before
        // any layer split, on discrete GPUs only (unified memory shares one pool: nothing saved).
        let expert_total = model.arch.expert_bytes.min(model.total_weight_bytes);
        let expert_steps: Vec<u32> = match (model.moe_offload, expert_total > 0, is_unified_memory)
        {
            (MoeOffload::On, true, _) => vec![total_layers],
            (MoeOffload::Auto, true, false) => (0..=total_layers).collect(),
            _ => Vec::new(),
        };
        for n in expert_steps {
            let cpu_experts = expert_total.saturating_mul(n as u64) / total_layers as u64;
            let vram_weight_bytes = model.total_weight_bytes.saturating_sub(cpu_experts);
            let vram_kv_and_workspace_bytes =
                Self::vram_runtime_overhead(model, total_layers, context_len, batch_size);
            if vram_weight_bytes.saturating_add(vram_kv_and_workspace_bytes) > available_vram_bytes
            {
                continue;
            }
            let warm = if is_unified_memory {
                cpu_experts
            } else if model.keep_warm_in_ram
                && model.total_weight_bytes <= available_pinned_ram_bytes
            {
                model.total_weight_bytes
            } else {
                cpu_experts
            };
            if warm <= available_pinned_ram_bytes {
                return Some(LayerPlacementPlan {
                    primary_tier: MemoryTier::Tier1GpuVram,
                    gpu_layers: total_layers,
                    pinned_ram_layers: 0,
                    ssd_layers: 0,
                    vram_weight_bytes,
                    vram_kv_and_workspace_bytes,
                    pinned_ram_bytes: warm,
                    allocated_context_len: context_len,
                    cpu_expert_layers: n,
                });
            }
        }

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
                        cpu_expert_layers: 0,
                    });
                }
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn the_exact_layout_books_only_the_layers_that_hold_a_cache() {
        // LFM3 1.7B: 24 layers, 6 full-attention + 5 sliding-window (window 256) with 4 KV heads
        // x 128, the other 13 convolution layers without a KV cache.
        let mut m = ModelConfig {
            slots: 1,
            ..sample_gemma4_26b_config()
        };
        m.aux_weights_path = None;
        m.arch.num_layers = 24;
        m.arch.num_kv_heads = 4;
        m.arch.head_dim = 128;
        m.arch.sliding_window_len = Some(256);
        m.arch.global_attn_every_n_layers = Some(1);
        m.arch.kv_dtype = crate::config::KvDataType::F16;
        let all_layers = KvCacheEstimator::kv_bytes_for_layers(&m, 24, 16384, 1);
        assert_eq!(
            all_layers,
            2 * 4 * 128 * 24 * 16384 * 2,
            "old: every layer, full context"
        );
        m.arch.global_kv_layers = Some(6);
        m.arch.global_num_kv_heads = Some(4);
        m.arch.swa_kv_layers = Some(5);
        m.arch.swa_num_kv_heads = Some(4);
        m.arch.swa_head_dim = Some(128);
        // Text-only: the window plus a 512-token micro-batch, padded to 256 cells = 768.
        let exact = KvCacheEstimator::kv_bytes_for_layers(&m, 24, 16384, 1);
        assert_eq!(exact, 2 * 4 * 128 * (6 * 16384 + 5 * 768) * 2);
        assert!(exact * 3 < all_layers);
        // With a projector the micro-batch is 2048 tokens: 2304 cells per sliding layer.
        m.aux_weights_path = Some("mmproj.gguf".into());
        let mm = KvCacheEstimator::kv_bytes_for_layers(&m, 24, 16384, 1);
        assert_eq!(mm, 2 * 4 * 128 * (6 * 16384 + 5 * 2304) * 2);
        // Half the layers on the GPU: half of each kind (rounded up).
        let half = KvCacheEstimator::kv_bytes_for_layers(&m, 12, 16384, 1);
        assert_eq!(half, 2 * 4 * 128 * (3 * 16384 + 3 * 2304) * 2);
    }

    use super::*;
    use crate::config::{EngineKind, KvDataType, Modality, ModelArchitectureParams, ModelPriority};
    use std::path::PathBuf;

    fn sample_gemma4_26b_config() -> ModelConfig {
        ModelConfig {
            id: "gemma4:26b-q4".to_string(),
            enabled: true,
            use_mmap: None,
            moe_offload: Default::default(),
            image: Default::default(),
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
            slots: 1,
            prefix_cache: true,
            device: Default::default(),
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

    #[test]
    fn unified_memory_never_returns_a_partial_split() {
        let model = sample_gemma4_26b_config();
        let avail_ram = 32 * 1024 * 1024 * 1024;

        // The same budget that yields a partial split on a discrete GPU fails cleanly on UMA.
        let tight = 15_500 * 1024 * 1024;
        assert!(KvCacheEstimator::compute_optimal_placement(
            &model,
            Some(16384),
            1,
            tight,
            avail_ram,
            true
        )
        .is_none());

        // With room for the whole model every layer lands on the GPU and nothing is booked in Tier 2.
        let roomy = 24 * 1024 * 1024 * 1024;
        let plan = KvCacheEstimator::compute_optimal_placement(
            &model,
            Some(16384),
            1,
            roomy,
            avail_ram,
            true,
        )
        .expect("fits on a 24 GB working set");
        assert_eq!(plan.gpu_layers, 46);
        assert_eq!(plan.pinned_ram_layers, 0);
        assert_eq!(plan.pinned_ram_bytes, 0);
    }

    fn moe_26b(mode: MoeOffload) -> ModelConfig {
        let mut m = sample_gemma4_26b_config();
        m.arch.expert_bytes = 12 * 1024 * 1024 * 1024;
        m.moe_offload = mode;
        m
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    fn place(m: &ModelConfig, vram: u64, uma: bool) -> Option<LayerPlacementPlan> {
        KvCacheEstimator::compute_optimal_placement(m, Some(8192), 1, vram, 64 * GIB, uma)
    }

    #[test]
    fn moe_offload_on_keeps_only_the_dense_part_on_the_gpu() {
        let full = place(&moe_26b(MoeOffload::Off), 64 * GIB, false).unwrap();
        let on = place(&moe_26b(MoeOffload::On), 64 * GIB, false).unwrap();
        let layers = on.gpu_layers;
        assert_eq!(on.cpu_expert_layers, layers);
        assert_eq!(on.pinned_ram_layers, 0);
        assert_eq!(full.vram_weight_bytes - on.vram_weight_bytes, 12 * GIB);
        assert!(
            on.pinned_ram_bytes >= 12 * GIB,
            "the experts are booked in host memory"
        );
        assert_eq!(full.cpu_expert_layers, 0);
    }

    #[test]
    fn moe_offload_auto_moves_as_few_experts_as_needed_before_splitting_layers() {
        let full = place(&moe_26b(MoeOffload::Auto), 64 * GIB, false).unwrap();
        assert_eq!(full.cpu_expert_layers, 0, "it fits whole: nothing moves");
        // 3 GiB short of the whole model: a few layers' experts (0.4 GiB each) go to the CPU.
        let tight = full.total_vram_bytes() - 3 * GIB;
        let auto = place(&moe_26b(MoeOffload::Auto), tight, false).unwrap();
        let per_layer = 12 * GIB / auto.gpu_layers as u64;
        assert_eq!(auto.gpu_layers, full.gpu_layers, "no layer split");
        assert!(auto.cpu_expert_layers > 0);
        assert!(auto.total_vram_bytes() <= tight);
        assert!(
            full.total_vram_bytes() - per_layer * (auto.cpu_expert_layers as u64 - 1) > tight,
            "the smallest n"
        );
        // Without offload the same budget means a layer split.
        let off = place(&moe_26b(MoeOffload::Off), tight, false).unwrap();
        assert!(off.gpu_layers < full.gpu_layers);
        assert_eq!(off.cpu_expert_layers, 0);
    }

    #[test]
    fn moe_offload_auto_does_nothing_on_unified_memory_or_for_dense_models() {
        let full = place(&moe_26b(MoeOffload::Off), 64 * GIB, true).unwrap();
        let tight = full.total_vram_bytes() - 3 * GIB;
        assert!(place(&moe_26b(MoeOffload::Auto), tight, true).is_none());
        let mut dense = moe_26b(MoeOffload::On);
        dense.arch.expert_bytes = 0;
        assert_eq!(place(&dense, 64 * GIB, false).unwrap().cpu_expert_layers, 0);
    }
}
