use hypnos_core::config::{
    AcceleratorKind, EngineKind, ExecutionDevice, HardwareConfig, HypnosConfig, KvDataType,
    LimitsConfig, MemoryTier, Modality, ModelArchitectureParams, ModelConfig, ModelPriority,
    ServerConfig,
};
use std::path::PathBuf;

/// Built-in profile used when no `hypnos.toml` is given. Mirrors `hypnos.example.toml`:
/// 1. `gemma4:26b-q4`   — Text Generation + Vision (llama.cpp + libmtmd)
/// 2. `gemma4:e4b-q4`   — Audio / Speech-to-Text + chat (llama.cpp + libmtmd audio encoder)
/// 3. `kokoro-82m-q4`   — Text-to-Speech (ONNX Runtime + espeak-ng)
/// 4. `gemmaembedding2` — Embeddings (llama.cpp, EmbeddingGemma 2)
/// 5. `moss-image`      — Image Generation (stable-diffusion.cpp)
/// 6. `docling`         — Document Conversion (docling.rs on ONNX Runtime, CPU)
/// 7. `gliner2.5-multi` — Entity Extraction (gliner25-rs on ONNX Runtime, CPU)
pub fn default_six_model_config() -> HypnosConfig {
    let gb: u64 = 1024 * 1024 * 1024;
    let mb: u64 = 1024 * 1024;

    HypnosConfig {
        config_version: hypnos_core::config::CONFIG_VERSION,
        unknown_keys: Vec::new(),
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 8080,
            kv_idle_timeout_secs: 300,
            request_log_level: "info".to_string(),
            request_log_capacity: 200,
            allowed_origins: Vec::new(),
            log_format: "pretty".to_string(),
            api_keys: Vec::new(), // HYPNOS_API_KEYS
            metrics_require_auth: false,
            keys: Vec::new(),
            api_key_defaults: Default::default(),
            request_log_redact: Vec::new(),
            frontend_dir: None,
            document_pictures_dir: None,
            document_pictures_ttl_secs: 3600,
            document_pictures_max_bytes: gb,
            tls: None,
        },
        limits: LimitsConfig::default(),
        hardware: HardwareConfig {
            accelerator: AcceleratorKind::Auto,
            device_id: 0,
            vram_budget_bytes: 0, // auto-detect (nvidia-smi / rocm-smi / sysctl)
            vram_safety_margin_bytes: gb,
            pinned_ram_budget_bytes: 32 * gb,
            lock_host_memory: None, // on for discrete GPUs, off on unified memory
            simulate_physical_memory_bytes: 0,
            max_process_memory_bytes: 0,
        },
        models: vec![
            // 1. Gemma 4 26B-A4B Q4_0 — Text Generation + Vision
            ModelConfig {
                id: "gemma4:26b-q4".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::TextGeneration,
                additional_modalities: vec![Modality::Vision],
                engine: EngineKind::GgmlLlama,
                weights_path: PathBuf::from("models/gemma-4-26B-A4B-it-Q4_0.gguf"),
                aux_weights_path: Some(PathBuf::from("models/mmproj-gemma-4-26B-A4B-it-BF16.gguf")),
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: None,
                total_weight_bytes: 15_814_641_056,
                min_gpu_layer_ratio: 0.25,
                initial_tier: MemoryTier::Tier2PinnedRam,
                keep_warm_in_ram: true,
                priority: ModelPriority::High,
                slots: 1,
                prefix_cache: true,
                device: Default::default(),
                arch: ModelArchitectureParams {
                    num_layers: 30,
                    num_kv_heads: 8,
                    head_dim: 512,
                    default_context_len: 16384,
                    max_context_len: 131072,
                    sliding_window_len: Some(1024),
                    global_attn_every_n_layers: Some(6),
                    kv_dtype: KvDataType::F16,
                    non_layer_overhead_bytes: 2304 * mb,
                    activation_workspace_bytes: 2 * gb,
                    configured_keys: None,
                },
            },
            // 2. Gemma 4 E4B Q4_0 — Audio / Speech-to-Text (+ text + vision chat)
            ModelConfig {
                id: "gemma4:e4b-q4".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::SpeechToText,
                additional_modalities: vec![Modality::TextGeneration, Modality::Vision],
                engine: EngineKind::GgmlLlama,
                weights_path: PathBuf::from("models/gemma-4-E4B-it-Q4_0.gguf"),
                aux_weights_path: Some(PathBuf::from("models/mmproj-gemma-4-E4B-it-BF16.gguf")),
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: None,
                total_weight_bytes: 5_582_359_648,
                min_gpu_layer_ratio: 0.25,
                initial_tier: MemoryTier::Tier2PinnedRam,
                keep_warm_in_ram: true,
                priority: ModelPriority::High,
                slots: 1,
                prefix_cache: true,
                device: Default::default(),
                arch: ModelArchitectureParams {
                    num_layers: 42,
                    num_kv_heads: 2,
                    head_dim: 512,
                    default_context_len: 8192,
                    max_context_len: 131072,
                    sliding_window_len: Some(512),
                    global_attn_every_n_layers: Some(6),
                    kv_dtype: KvDataType::Q8_0,
                    non_layer_overhead_bytes: 1536 * mb,
                    activation_workspace_bytes: 512 * mb,
                    configured_keys: None,
                },
            },
            // 3. Kokoro 82M (fp16 ONNX) — Text-to-Speech on CPU via ONNX Runtime.
            // The quantized q8f16/q4 graphs crash the static ORT 1.31 CPU provider at session creation.
            ModelConfig {
                id: "kokoro-82m-q4".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::TextToSpeech,
                additional_modalities: vec![],
                engine: EngineKind::OnnxRuntime,
                weights_path: PathBuf::from("models/kokoro/model_fp16.onnx"),
                aux_weights_path: None,
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: Some(PathBuf::from("models/kokoro/voices")),
                total_weight_bytes: 0,
                min_gpu_layer_ratio: 1.0,
                initial_tier: MemoryTier::Tier1GpuVram,
                keep_warm_in_ram: true,
                priority: ModelPriority::RealtimePinned,
                slots: 1,
                prefix_cache: true,
                device: Default::default(),
                arch: ModelArchitectureParams {
                    num_layers: 12,
                    num_kv_heads: 0,
                    head_dim: 64,
                    default_context_len: 512,
                    max_context_len: 2048,
                    sliding_window_len: None,
                    global_attn_every_n_layers: None,
                    kv_dtype: KvDataType::F16,
                    non_layer_overhead_bytes: 15 * mb,
                    activation_workspace_bytes: 64 * mb,
                    configured_keys: None,
                },
            },
            // 4. EmbeddingGemma 2 Q8_0 — Dense Vector Embeddings (768-dim, mean pooling)
            ModelConfig {
                id: "gemmaembedding2".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::Embeddings,
                additional_modalities: vec![],
                engine: EngineKind::GgmlLlama,
                weights_path: PathBuf::from("models/embeddinggemma-2-Q8_0.gguf"),
                aux_weights_path: None,
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: None,
                total_weight_bytes: 0,
                min_gpu_layer_ratio: 1.0,
                initial_tier: MemoryTier::Tier1GpuVram,
                keep_warm_in_ram: true,
                priority: ModelPriority::RealtimePinned,
                slots: 1,
                prefix_cache: true,
                device: Default::default(),
                arch: ModelArchitectureParams {
                    num_layers: 24,
                    num_kv_heads: 2,
                    head_dim: 512,
                    default_context_len: 2048,
                    max_context_len: 8192,
                    sliding_window_len: Some(1024),
                    global_attn_every_n_layers: Some(6),
                    kv_dtype: KvDataType::F16,
                    non_layer_overhead_bytes: 100 * mb,
                    activation_workspace_bytes: 128 * mb,
                    configured_keys: None,
                },
            },
            // 5. MOSS-Image SFT Q4_K — Image Generation (stable-diffusion.cpp)
            ModelConfig {
                id: "moss-image".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::ImageGeneration,
                additional_modalities: vec![],
                engine: EngineKind::GgmlDiffusion,
                weights_path: PathBuf::from("models/MOSS-Image-SFT-Q4_K.gguf"),
                aux_weights_path: Some(PathBuf::from("models/flux2-vae.safetensors")),
                text_encoder_path: Some(PathBuf::from("models/Qwen3.5-9B-Q4_K_M.gguf")),
                tokenizer_path: Some(PathBuf::from("models/moss_tokenizer.json")),
                voices_path: None,
                total_weight_bytes: 9_344_134_556,
                min_gpu_layer_ratio: 0.25,
                initial_tier: MemoryTier::Tier3NvmeSsd,
                keep_warm_in_ram: false, // sd.cpp reloads in ~1 s; pinning 9 GiB starves RAM
                priority: ModelPriority::Low,
                slots: 1,
                prefix_cache: true,
                device: Default::default(),
                arch: ModelArchitectureParams {
                    num_layers: 48,
                    num_kv_heads: 0,
                    head_dim: 128,
                    default_context_len: 512,
                    max_context_len: 1024,
                    sliding_window_len: None,
                    global_attn_every_n_layers: None,
                    kv_dtype: KvDataType::F16,
                    non_layer_overhead_bytes: 6 * gb,
                    activation_workspace_bytes: 3 * gb,
                    configured_keys: None,
                },
            },
            // 6. docling.rs — Document Conversion on the CPU (Tier 1 books host RAM only)
            ModelConfig {
                id: "docling".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::DocumentConversion,
                additional_modalities: vec![],
                engine: EngineKind::Docling,
                weights_path: PathBuf::from("models/docling/layout_heron_int8.onnx"),
                aux_weights_path: None,
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: None,
                total_weight_bytes: 320 * mb,
                min_gpu_layer_ratio: 0.25,
                initial_tier: MemoryTier::Tier3NvmeSsd,
                keep_warm_in_ram: false,
                priority: ModelPriority::Normal,
                slots: 1,
                prefix_cache: true,
                device: ExecutionDevice::Cpu,
                arch: ModelArchitectureParams {
                    num_layers: 1,
                    activation_workspace_bytes: 1536 * mb,
                    ..ModelArchitectureParams::default()
                },
            },
            // 7. GLiNER2.5 multilingual — Entity Extraction on the CPU
            ModelConfig {
                id: "gliner2.5-multi".to_string(),
                enabled: true,
                use_mmap: None,
                modality: Modality::EntityExtraction,
                additional_modalities: vec![],
                engine: EngineKind::Gliner,
                weights_path: PathBuf::from("models/gliner2.5-multi/encoder_fp32.onnx"),
                aux_weights_path: None,
                text_encoder_path: None,
                tokenizer_path: None,
                voices_path: None,
                total_weight_bytes: 1_228_931_072,
                min_gpu_layer_ratio: 0.25,
                initial_tier: MemoryTier::Tier3NvmeSsd,
                keep_warm_in_ram: false,
                priority: ModelPriority::Normal,
                slots: 1,
                prefix_cache: true,
                device: ExecutionDevice::Cpu,
                arch: ModelArchitectureParams {
                    num_layers: 1,
                    activation_workspace_bytes: 2 * gb,
                    ..ModelArchitectureParams::default()
                },
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use hypnos_core::config::{AcceleratorKind, HypnosConfig};

    fn load(name: &str) -> HypnosConfig {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(name);
        let raw = std::fs::read_to_string(&path).expect("example config readable");
        HypnosConfig::from_toml_str(&raw).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    #[test]
    fn example_profiles_parse_with_the_same_lineup() {
        let cuda = load("hypnos.example.toml");
        let macos = load("hypnos.macos.example.toml");
        let ids = |c: &HypnosConfig| c.models.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&cuda), ids(&macos));

        assert_eq!(ids(&super::default_six_model_config()), ids(&cuda));

        assert_eq!(macos.hardware.accelerator, AcceleratorKind::AppleMetalUma);
        assert_eq!(macos.hardware.lock_host_memory, Some(false));
        assert!(macos
            .models
            .iter()
            .filter(|m| !m.runs_on_cpu())
            .all(|m| m.min_gpu_layer_ratio == 1.0));
    }
}
