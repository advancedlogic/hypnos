use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::PathBuf;

/// Where a data folder that ships with the binary (`espeak-ng-data`, `frontend`) may sit, in
/// order: beside the executable (release archives, Docker, the cargo target dir), then
/// `../Resources` when the executable is in an app bundle's `Contents/MacOS` (the server bundled
/// in the macOS desktop app, DESK-16).
pub fn beside_executable(exe_dir: &std::path::Path, name: &str) -> Vec<PathBuf> {
    let mut dirs = vec![exe_dir.join(name)];
    if exe_dir.file_name().is_some_and(|n| n == "MacOS") {
        if let Some(contents) = exe_dir.parent() {
            dirs.push(contents.join("Resources").join(name));
        }
    }
    dirs
}

/// Supported GPU compute & memory architectures.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AcceleratorKind {
    /// Automatically detect CUDA -> ROCm -> Metal -> Vulkan
    #[default]
    Auto,
    /// NVIDIA CUDA (Discrete PCIe / NVLink VRAM + Pinned Host RAM via cudaHostAlloc)
    NvidiaCuda,
    /// AMD ROCm / HIP (Discrete PCIe VRAM + Pinned Host RAM via hipHostMalloc, or APU UMA)
    AmdRocm,
    /// Apple Silicon Metal (Unified Memory Architecture — zero-copy shared residency)
    AppleMetalUma,
    /// Cross-platform Vulkan compute fallback
    Vulkan,
}

/// Physical memory topology of the target accelerator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTopology {
    /// Separate physical Device VRAM (Tier 1) and Pinned System RAM (Tier 2) connected over PCIe.
    DiscretePcie,
    /// Unified Memory Architecture (Apple Silicon M-series / AMD Strix Halo APU) where
    /// CPU and GPU share physical RAM; Tier 1 = wired/active Metal working set, Tier 2 = warm page cache.
    UnifiedMemory,
}

/// The 3 storage tiers managed by Hypnos.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryTier {
    /// Cold storage on NVMe SSD (zero-copy mmap ready).
    Tier3NvmeSsd = 0,
    /// Warm storage in Page-Locked (Pinned) Host RAM (or unwired UMA pages).
    Tier2PinnedRam = 1,
    /// Hot storage in GPU VRAM (or wired Metal GPU working set), including KV cache.
    Tier1GpuVram = 2,
}

/// Primary model modalities supported by the server.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    /// Autoregressive text generation (/v1/chat/completions, /v1/completions)
    TextGeneration,
    /// Vision-Language multimodal generation (/v1/chat/completions with image_url)
    Vision,
    /// Speech-to-Text / Audio understanding (/v1/audio/transcriptions)
    SpeechToText,
    /// Text-to-Speech audio synthesis (/v1/audio/speech)
    TextToSpeech,
    /// Dense vector embeddings (/v1/embeddings)
    Embeddings,
    /// Diffusion / DiT / Autoregressive image generation (/v1/images/generations)
    ImageGeneration,
    /// PDF / image / office documents to Markdown or JSON (/v1/documents/convert)
    DocumentConversion,
    /// Zero-shot named-entity extraction with caller-chosen labels (/v1/entities)
    EntityExtraction,
}

/// Underlying inference engine runtime used for a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    /// llama.cpp / ggml / libmtmd backend (Gemma 4 Text+Vision, Gemma 4 E4B Audio, GGUF Embeddings)
    GgmlLlama,
    /// whisper.cpp / ggml backend (STT)
    GgmlWhisper,
    /// stable-diffusion.cpp / ggml diffusion/flow backend (MOSS-Image, Flux, SDXL)
    GgmlDiffusion,
    /// ONNX Runtime (CUDA / ROCm / CoreML Execution Providers — Kokoro TTS, Embeddings)
    OnnxRuntime,
    /// docling.rs on ONNX Runtime: layout, TableFormer and OCR models (document conversion)
    Docling,
    /// GLiNER2.5 boundary models on ONNX Runtime (entity extraction)
    Gliner,
}

/// Where a model computes. GPU models are booked in Tier 1 VRAM; a CPU model's Tier 1 placement
/// ("loaded and ready") books host RAM only: its weights plus `arch.activation_workspace_bytes`.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDevice {
    #[default]
    Gpu,
    Cpu,
}

/// KV Cache quantization data type for autoregressive models.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KvDataType {
    /// 16-bit float (2.0 bytes per element)
    #[default]
    F16,
    /// 16-bit brain float (2.0 bytes per element)
    Bf16,
    /// 8-bit quantized KV cache (~1.0625 bytes per element)
    Q8_0,
    /// 4-bit quantized KV cache (~0.5625 bytes per element)
    Q4_0,
}

impl KvDataType {
    /// Effective bytes per scalar element in the KV cache (including block scales).
    pub fn bytes_per_element(self) -> f64 {
        match self {
            Self::F16 | Self::Bf16 => 2.0,
            Self::Q8_0 => 34.0 / 32.0,
            Self::Q4_0 => 18.0 / 32.0,
        }
    }
}

/// Mixture-of-Experts offload (MEM-5, `docs/moe.md`): keep a model's expert tensors in host
/// memory and run them on the CPU while attention, the dense layers and the KV cache stay on the
/// GPU. No effect on a dense model (`arch.expert_bytes` = 0).
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MoeOffload {
    /// Experts follow their layer (the default).
    #[default]
    Off,
    /// When the model does not fit the GPU whole, move the experts of as few layers as needed to
    /// the CPU before splitting layers off (discrete GPUs; no effect on unified memory).
    Auto,
    /// Always run every layer's experts on the CPU.
    On,
}

/// Eviction/retention priority tier for models.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelPriority {
    /// Evict first when VRAM is needed (e.g., heavy batch image generator)
    Low = 1,
    /// Standard LRU/cost-aware eviction
    #[default]
    Normal = 2,
    /// Prefer keeping in VRAM or Tier 2 Pinned RAM
    High = 3,
    /// Pin permanently in Tier 1 VRAM (e.g., lightweight embedding or TTS model)
    RealtimePinned = 4,
}

/// Architectural hyper-parameters used to compute exact KV cache and partial layer budgets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitectureParams {
    /// Total number of transformer / DiT blocks that can be split between GPU and CPU.
    #[serde(default = "default_num_layers")]
    pub num_layers: u32,
    /// Number of Key/Value attention heads per layer (GQA/MQA aware). Zero for non-KV models.
    #[serde(default)]
    pub num_kv_heads: u32,
    /// Dimension of each attention head (e.g. 128 or 256 for Gemma).
    #[serde(default = "default_head_dim")]
    pub head_dim: u32,
    /// Default context window length to reserve when loading into VRAM.
    #[serde(default = "default_context_len")]
    pub default_context_len: u32,
    /// Maximum supported context window length.
    #[serde(default = "default_max_context_len")]
    pub max_context_len: u32,
    /// Optional Sliding Window Attention (SWA) cap for local attention layers (e.g., 1024 or 4096 in Gemma 4).
    #[serde(default)]
    pub sliding_window_len: Option<u32>,
    /// If using hybrid SWA + Global Attention (e.g., Gemma 4 uses 1 global layer every 6 layers),
    /// specifies the interval N where 1 out of N layers uses full context and (N-1) use `sliding_window_len`.
    #[serde(default)]
    pub global_attn_every_n_layers: Option<u32>,
    /// KV cache data type (f16, q8_0, q4_0).
    #[serde(default)]
    pub kv_dtype: KvDataType,
    /// Fixed non-splittable overhead in bytes (token embeddings, LM head, vision/audio mmproj, VAE).
    #[serde(default)]
    pub non_layer_overhead_bytes: u64,
    /// Peak transient workspace / scratch activation buffer in bytes during forward pass.
    #[serde(default = "default_activation_workspace")]
    pub activation_workspace_bytes: u64,
    /// Bytes of the Mixture-of-Experts expert tensors (filled from the GGUF; 0 = dense model).
    #[serde(default)]
    pub expert_bytes: u64,
    /// Exact KV layout (TEST-3), from the GGUF's per-layer KV heads and sliding-window pattern:
    /// layers with a full-context KV cache, layers with a sliding-window one (a hybrid model's
    /// convolution layers have none), and the sliding-window layers' KV heads and head size.
    /// Unset: every layer is booked as `num_kv_heads` x `head_dim`, as before.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_kv_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub global_num_kv_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swa_kv_layers: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swa_num_kv_heads: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub swa_head_dim: Option<u32>,
    /// Keys written in the config's `[models.arch]` table, recorded by
    /// [`HypnosConfig::from_toml_str`]. `None` when the params were built in code; then every
    /// field counts as explicit and [`HypnosConfig::detect_arch_from_gguf`] leaves them alone.
    #[serde(skip)]
    pub configured_keys: Option<BTreeSet<String>>,
}

impl ModelArchitectureParams {
    /// Whether `key` was set explicitly (and must not be replaced by detected values).
    pub fn is_configured(&self, key: &str) -> bool {
        self.configured_keys
            .as_ref()
            .is_none_or(|keys| keys.contains(key))
    }
}

fn default_num_layers() -> u32 {
    32
}
fn default_head_dim() -> u32 {
    128
}
fn default_context_len() -> u32 {
    4096
}
fn default_max_context_len() -> u32 {
    32768
}
fn default_activation_workspace() -> u64 {
    256 * 1024 * 1024 // 256 MB default scratch buffer
}

impl Default for ModelArchitectureParams {
    fn default() -> Self {
        Self {
            num_layers: 32,
            num_kv_heads: 8,
            head_dim: 128,
            default_context_len: 4096,
            max_context_len: 32768,
            sliding_window_len: None,
            global_attn_every_n_layers: None,
            kv_dtype: KvDataType::F16,
            non_layer_overhead_bytes: 512 * 1024 * 1024,
            activation_workspace_bytes: default_activation_workspace(),
            expert_bytes: 0,
            global_kv_layers: None,
            global_num_kv_heads: None,
            swa_kv_layers: None,
            swa_num_kv_heads: None,
            swa_head_dim: None,
            configured_keys: None,
        }
    }
}

/// Configuration for an individual registered model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// OpenAI model identifier (e.g., "gemma4:26b-q4", "gemma4:e4b-q4", "kokoro-82m-q4")
    pub id: String,
    /// Primary modality of the model
    pub modality: Modality,
    /// Additional modalities supported by unified multimodal models (e.g., `gemma4:26b-q4` supports both TextGeneration and Vision)
    #[serde(default)]
    pub additional_modalities: Vec<Modality>,
    /// Inference engine runtime
    pub engine: EngineKind,
    /// Path to the primary weights file (.gguf, .safetensors, .onnx) on NVMe SSD
    pub weights_path: PathBuf,
    /// Optional auxiliary projector/encoder path (e.g. Gemma 4 vision/audio mmproj.gguf or VAE)
    #[serde(default)]
    pub aux_weights_path: Option<PathBuf>,
    /// Optional LLM text-encoder weights for diffusion models (e.g. Qwen3.5-9B GGUF for MOSS-Image)
    #[serde(default)]
    pub text_encoder_path: Option<PathBuf>,
    /// Optional tokenizer.json override (required by MOSS-Image, whose vocabulary differs from Qwen's)
    #[serde(default)]
    pub tokenizer_path: Option<PathBuf>,
    /// Optional directory of speaker style vectors for TTS models (Kokoro `voices/*.bin`)
    #[serde(default)]
    pub voices_path: Option<PathBuf>,
    /// Total model weight size in bytes (auto-detected from file if 0 or omitted, or explicit)
    #[serde(default)]
    pub total_weight_bytes: u64,
    /// Minimum fraction of layers [0.0 ..= 1.0] that MUST fit in Tier 1 VRAM to execute.
    #[serde(default = "default_min_gpu_layer_ratio")]
    pub min_gpu_layer_ratio: f32,
    /// Initial tier on server startup (defaults to Tier3NvmeSsd for lazy loading, or Tier2PinnedRam)
    #[serde(default = "default_initial_tier")]
    pub initial_tier: MemoryTier,
    /// Whether to keep weights in Tier 2 Pinned RAM after eviction from Tier 1 VRAM
    #[serde(default = "default_keep_warm_in_ram")]
    pub keep_warm_in_ram: bool,
    /// Retention priority when competing for VRAM
    #[serde(default)]
    pub priority: ModelPriority,
    /// `false` keeps the block but takes the model out of service: it is not pre-staged at
    /// startup, left out of `/v1/models`, holds no Tier 1 / Tier 2 memory, and a request for it
    /// answers 404. Validation tolerates missing weights for a disabled model.
    /// `POST /v1/models/{id}/enable` and `/disable` change it at runtime (not persisted).
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Execution slots for llama.cpp models: sequences that share one persistent context. Each
    /// slot owns a KV cache of `allocated_context_len` positions, so the KV booking scales with
    /// `slots`. 1 (the default) serialises requests exactly as before while still keeping the
    /// context and its prefix cache alive between them.
    #[serde(default = "default_slots")]
    pub slots: u32,
    /// Reuse a slot's KV cache when the new prompt shares a prefix with what the slot last
    /// processed: multi-turn chats prefill only the new turn and unchanged images or audio clips
    /// are not re-encoded. Off means every request prefills from scratch.
    #[serde(default = "default_prefix_cache")]
    pub prefix_cache: bool,
    /// `cpu` for engines that run on the host (docling): Tier 1 then books RAM, not VRAM.
    #[serde(default)]
    pub device: ExecutionDevice,
    /// llama.cpp models: map the weights file (`true`) or read it through pinned staging buffers
    /// with asynchronous uploads (`false`). Unset = automatic: `false` when layers go to a CUDA
    /// device, `true` on Metal (unified memory, zero copy) and for CPU-only placements.
    #[serde(default)]
    pub use_mmap: Option<bool>,
    /// Mixture-of-Experts offload: `off` (default), `auto`, `on` (see [`MoeOffload`]);
    /// `POST /v1/models/{id}/moe-offload` changes it at runtime.
    #[serde(default)]
    pub moe_offload: MoeOffload,
    /// Architecture parameters for KV cache & layer-split math
    #[serde(default)]
    pub arch: ModelArchitectureParams,
}

impl ModelConfig {
    /// Checks whether this model supports a given modality (either as its primary or additional modality).
    pub fn supports_modality(&self, target: Modality) -> bool {
        self.modality == target || self.additional_modalities.contains(&target)
    }

    /// `true` when the model computes on the host CPU (see [`ExecutionDevice`]).
    pub fn runs_on_cpu(&self) -> bool {
        self.device == ExecutionDevice::Cpu
    }

    /// Returns all supported modalities for this model.
    pub fn all_modalities(&self) -> Vec<Modality> {
        let mut list = vec![self.modality];
        for m in &self.additional_modalities {
            if !list.contains(m) {
                list.push(*m);
            }
        }
        list
    }
}

fn default_min_gpu_layer_ratio() -> f32 {
    0.25
}
fn default_initial_tier() -> MemoryTier {
    MemoryTier::Tier3NvmeSsd
}
fn default_keep_warm_in_ram() -> bool {
    true
}
fn default_slots() -> u32 {
    1
}
fn default_prefix_cache() -> bool {
    true
}

/// Hardware & memory pool limits configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Accelerator backend selection
    #[serde(default)]
    pub accelerator: AcceleratorKind,
    /// Device index (for multi-GPU systems)
    #[serde(default)]
    pub device_id: usize,
    /// Maximum Tier 1 GPU VRAM budget in bytes (if 0, queries hardware and reserves 92%)
    #[serde(default)]
    pub vram_budget_bytes: u64,
    /// Safety margin in VRAM (bytes) always left untouched for OS/driver overhead
    #[serde(default = "default_vram_safety_margin")]
    pub vram_safety_margin_bytes: u64,
    /// Maximum Tier 2 Pinned Host RAM budget in bytes (e.g., 32 GB)
    #[serde(default = "default_pinned_ram_budget")]
    pub pinned_ram_budget_bytes: u64,
    /// Whether to use OS memory locking (`VirtualLock` / `mlock`) for Tier 2 Pinned RAM.
    /// Unset = on for discrete GPUs, off on unified memory, where `mlock` and the Metal working set
    /// count against the same wired-memory limit (see [`HardwareConfig::lock_host_memory_for`]).
    #[serde(default)]
    pub lock_host_memory: Option<bool>,
    /// Pretend the machine has this much physical RAM (0 = use the real amount; never above it).
    /// All budgets are derived as on that machine; on unified memory this includes the GPU working
    /// set. Meant for trying out smaller configurations, e.g. a 32 GB Mac on a 128 GB one.
    #[serde(default)]
    pub simulate_physical_memory_bytes: u64,
    /// Soft cap on this process's real memory footprint as the OS reports it (0 = no cap). Checked
    /// before every model load: idle models are evicted, or the request is refused, rather than
    /// letting the load cross the cap. Memory a resident model grows into later is not prevented.
    #[serde(default)]
    pub max_process_memory_bytes: u64,
    /// Keep each llama.cpp slot's KV state in host memory when its model leaves Tier 1, and
    /// restore it when the model comes back with the same context layout, so the next turn of a
    /// conversation is not prefilled again (MEM-4). Off by default; `POST /v1/memory/kv-park`
    /// switches it at runtime.
    #[serde(default)]
    pub kv_park: bool,
    /// Host memory the parked states may hold in total; the least recently parked go first.
    #[serde(default = "default_kv_park_bytes")]
    pub kv_park_bytes: u64,
}

fn default_kv_park_bytes() -> u64 {
    4 * 1024 * 1024 * 1024
}

impl HardwareConfig {
    /// Resolves `lock_host_memory` for the detected topology.
    pub fn lock_host_memory_for(&self, topology: MemoryTopology) -> bool {
        self.lock_host_memory
            .unwrap_or(topology == MemoryTopology::DiscretePcie)
    }

    /// On unified memory Tier 1 and Tier 2 are the same physical RAM. Returns a warning when the
    /// explicitly configured budgets add up to more than `physical_memory_bytes`.
    pub fn unified_memory_oversubscription(&self, physical_memory_bytes: u64) -> Option<String> {
        if self.vram_budget_bytes == 0 || physical_memory_bytes == 0 {
            return None;
        }
        let claimed = self
            .vram_budget_bytes
            .saturating_add(self.pinned_ram_budget_bytes);
        (claimed > physical_memory_bytes).then(|| {
            format!(
                "hardware.vram_budget_bytes + pinned_ram_budget_bytes ({} MiB) exceed the {} MiB of unified memory; Tier 2 is clamped to what Tier 1 leaves free",
                claimed / (1024 * 1024),
                physical_memory_bytes / (1024 * 1024)
            )
        })
    }
}

fn default_vram_safety_margin() -> u64 {
    512 * 1024 * 1024 // 512 MB safety margin
}
fn default_pinned_ram_budget() -> u64 {
    32 * 1024 * 1024 * 1024 // 32 GB pinned host RAM pool
}
fn default_true() -> bool {
    true
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            accelerator: AcceleratorKind::Auto,
            device_id: 0,
            vram_budget_bytes: 0, // auto-detect, as when [hardware] omits the key
            vram_safety_margin_bytes: default_vram_safety_margin(),
            pinned_ram_budget_bytes: default_pinned_ram_budget(),
            lock_host_memory: None,
            simulate_physical_memory_bytes: 0,
            max_process_memory_bytes: 0,
            kv_park: false,
            kv_park_bytes: default_kv_park_bytes(),
        }
    }
}

/// HTTP Server configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    /// Bind address. Defaults to the loopback interface. Before binding "0.0.0.0", set `api_keys`
    /// and `tls` (or put an authenticating, TLS-terminating proxy in front).
    #[serde(default = "default_host")]
    pub host: String,
    #[serde(default = "default_port")]
    pub port: u16,
    /// Idle timeout in seconds before an unused Tier 1 model's KV cache is compacted
    #[serde(default = "default_idle_timeout")]
    pub kv_idle_timeout_secs: u64,
    /// Origins allowed by CORS. Entries are exact origins (`http://localhost:5173`) or the
    /// wildcard forms `http://localhost:*` / `http://127.0.0.1:*`. Empty = any loopback origin.
    #[serde(default)]
    pub allowed_origins: Vec<String>,
    /// Log output format: "pretty" (human, ANSI) or "json" (one JSON object per line).
    #[serde(default = "default_log_format")]
    pub log_format: String,
    /// In-memory request log served on `GET /v1/logs`: "off", "error" (failed calls only),
    /// "warn" (4xx and 5xx), "info" (metadata of every call), "debug" (adds prompt and completion
    /// text) or "trace" (adds request and response payloads, media replaced by placeholders).
    /// Changeable at runtime through `POST /v1/logs/level`.
    #[serde(default = "default_request_log_level")]
    pub request_log_level: String,
    /// How many request records the ring keeps (oldest dropped first).
    #[serde(default = "default_request_log_capacity")]
    pub request_log_capacity: usize,
    /// Keys accepted as `Authorization: Bearer <key>` on every `/v1/*` route. Empty = no
    /// authentication. The `HYPNOS_API_KEYS` environment variable (comma-separated) replaces the
    /// list, so keys can stay out of the config file.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Also require a key on `/metrics`. `/health` and `/ready` always stay open for probes.
    #[serde(default)]
    pub metrics_require_auth: bool,
    /// `[[server.keys]]`: labelled keys with their own limits, next to the plain `api_keys`.
    #[serde(default)]
    pub keys: Vec<ApiKeyProfile>,
    /// `[server.api_key_defaults]`: limits for every key that does not set its own (0 = none).
    #[serde(default)]
    pub api_key_defaults: ApiKeyLimits,
    /// Redaction of what the request log captures at `debug` and `trace`: regular expressions
    /// whose matches in prompts, completions and payload strings become `[redacted]`, or the
    /// single entry `"all"` to keep no text or payloads at any level.
    #[serde(default)]
    pub request_log_redact: Vec<String>,
    /// Directory with the built playground (`frontend/` after `npm run build`, i.e. `dist/`) to
    /// serve at `/`. Unset: a `frontend/` directory next to the executable is served if it has an
    /// `index.html` (release archives and the Docker image ship it there), otherwise nothing.
    #[serde(default)]
    pub frontend_dir: Option<PathBuf>,
    /// The folder that relative model paths (`weights_path`, `aux_weights_path`,
    /// `text_encoder_path`, `tokenizer_path`, `voices_path`) resolve against, itself relative to
    /// the working directory. `HYPNOS_MODELS_DIR` overrides it. Unset: the working directory,
    /// as before this key existed. The example configs set `"models"` with bare file names, the
    /// layout `models.manifest` downloads; the desktop app points it at its models folder.
    #[serde(default)]
    pub models_dir: Option<PathBuf>,
    /// Where `/v1/documents/convert` keeps the pictures a `return_pictures` conversion cut out
    /// (one JSON file per picture: base64 image plus its metadata). Unset: `hypnos-pictures`
    /// under the OS temporary directory. Documents left from an earlier run are removed at
    /// startup.
    #[serde(default)]
    pub document_pictures_dir: Option<PathBuf>,
    /// How long a document's pictures stay after its last page was converted; `DELETE
    /// /v1/documents/{id}` releases them earlier.
    #[serde(default = "default_document_pictures_ttl_secs")]
    pub document_pictures_ttl_secs: u64,
    /// Cap on the store's total size; the oldest documents go first when it is crossed
    /// (0 = no cap).
    #[serde(default = "default_document_pictures_max_bytes")]
    pub document_pictures_max_bytes: u64,
    /// HTTPS: serve TLS in-process with this PEM certificate chain and private key.
    /// Plain HTTP when absent.
    #[serde(default)]
    pub tls: Option<TlsConfig>,
}

fn default_document_pictures_ttl_secs() -> u64 {
    3600
}
fn default_document_pictures_max_bytes() -> u64 {
    1024 * 1024 * 1024
}

/// Per-key limits (`[server.api_key_defaults]`, and the fields of `[[server.keys]]`): 0 = none.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ApiKeyLimits {
    /// Requests admitted per key in a wall-clock minute; beyond it 429 with `Retry-After`.
    #[serde(default)]
    pub requests_per_minute: u32,
    /// Requests of the key in flight at once (a streaming response counts until it ends).
    #[serde(default)]
    pub max_concurrent: u32,
    /// Prompt + completion tokens per UTC day; once spent, requests get 429 until midnight.
    #[serde(default)]
    pub tokens_per_day: u64,
}

/// `[[server.keys]]`: a labelled API key. The label names it in the request log, on `/metrics`
/// and in audit lines; a limit left unset takes `[server.api_key_defaults]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyProfile {
    pub label: String,
    /// The bearer token itself.
    pub key: String,
    #[serde(default)]
    pub requests_per_minute: Option<u32>,
    #[serde(default)]
    pub max_concurrent: Option<u32>,
    #[serde(default)]
    pub tokens_per_day: Option<u64>,
    /// May change the model pool (`POST /v1/models`, `PUT` / `DELETE /v1/models/{id}`). Plain
    /// `api_keys` always may.
    #[serde(default)]
    pub admin: bool,
}

impl ApiKeyProfile {
    /// This key's limits, with `defaults` for what it leaves unset.
    pub fn limits(&self, defaults: &ApiKeyLimits) -> ApiKeyLimits {
        ApiKeyLimits {
            requests_per_minute: self
                .requests_per_minute
                .unwrap_or(defaults.requests_per_minute),
            max_concurrent: self.max_concurrent.unwrap_or(defaults.max_concurrent),
            tokens_per_day: self.tokens_per_day.unwrap_or(defaults.tokens_per_day),
        }
    }
}

/// `[server.tls]`: PEM files, relative paths resolved from the working directory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate chain (leaf first), PEM.
    pub cert_path: PathBuf,
    /// Private key (PKCS#8, PKCS#1 or SEC1), PEM.
    pub key_path: PathBuf,
}

impl ServerConfig {
    /// Applies `HYPNOS_API_KEYS` (comma-separated; blanks ignored) over `api_keys` when it is set.
    pub fn apply_env_api_keys(&mut self, env_value: Option<&str>) {
        if let Some(value) = env_value {
            self.api_keys = value
                .split(',')
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(str::to_string)
                .collect();
        }
    }

    fn is_loopback_host(&self) -> bool {
        matches!(self.host.as_str(), "127.0.0.1" | "localhost" | "::1")
    }
}

fn default_host() -> String {
    "127.0.0.1".to_string()
}
fn default_port() -> u16 {
    9999
}
fn default_idle_timeout() -> u64 {
    300
}
fn default_log_format() -> String {
    "pretty".to_string()
}
fn default_request_log_level() -> String {
    "info".to_string()
}
fn default_request_log_capacity() -> usize {
    200
}

/// Accepted values of `server.request_log_level`, least to most verbose.
pub const REQUEST_LOG_LEVELS: &[&str] = &["off", "error", "warn", "info", "debug", "trace"];

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            host: default_host(),
            port: default_port(),
            kv_idle_timeout_secs: default_idle_timeout(),
            allowed_origins: Vec::new(),
            log_format: default_log_format(),
            request_log_level: default_request_log_level(),
            request_log_capacity: default_request_log_capacity(),
            api_keys: Vec::new(),
            metrics_require_auth: false,
            keys: Vec::new(),
            api_key_defaults: ApiKeyLimits::default(),
            request_log_redact: Vec::new(),
            frontend_dir: None,
            models_dir: None,
            document_pictures_dir: None,
            document_pictures_ttl_secs: default_document_pictures_ttl_secs(),
            document_pictures_max_bytes: default_document_pictures_max_bytes(),
            tls: None,
        }
    }
}

/// Request admission and input-size limits (`[limits]` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimitsConfig {
    /// Wall-clock budget for one non-streaming request (headers for streaming ones).
    #[serde(default = "default_request_timeout_secs")]
    pub request_timeout_secs: u64,
    /// Maximum requests queued or running per model; further ones get HTTP 429.
    #[serde(default = "default_max_queue_per_model")]
    pub max_queue_per_model: usize,
    /// Largest accepted HTTP body (multipart uploads included).
    #[serde(default = "default_max_body_bytes")]
    pub max_body_bytes: u64,
    /// Largest accepted audio clip (upload or base64 `input_audio`).
    #[serde(default = "default_max_audio_bytes")]
    pub max_audio_bytes: u64,
    /// Largest accepted image (data URL or remote download).
    #[serde(default = "default_max_image_bytes")]
    pub max_image_bytes: u64,
    /// Largest accepted document for `/v1/documents/convert` (upload, base64 or remote download).
    #[serde(default = "default_max_document_bytes")]
    pub max_document_bytes: u64,
    /// Maximum total characters across chat messages / embedding inputs / TTS input.
    #[serde(default = "default_max_prompt_chars")]
    pub max_prompt_chars: usize,
    /// Upper bound applied to `max_tokens`.
    #[serde(default = "default_max_tokens_cap")]
    pub max_tokens_cap: u32,
    /// Maximum images per `/v1/images/generations` request (`n`).
    #[serde(default = "default_max_images_per_request")]
    pub max_images_per_request: u32,
    /// Whether `image_url` and a document's `url` may reference http(s) URLs (outbound fetch), or
    /// only data: URLs and uploads.
    #[serde(default = "default_true")]
    pub allow_remote_image_urls: bool,
}

fn default_request_timeout_secs() -> u64 {
    900
}
fn default_max_queue_per_model() -> usize {
    8
}
fn default_max_body_bytes() -> u64 {
    256 * 1024 * 1024
}
fn default_max_audio_bytes() -> u64 {
    64 * 1024 * 1024
}
fn default_max_image_bytes() -> u64 {
    32 * 1024 * 1024
}
fn default_max_document_bytes() -> u64 {
    100 * 1024 * 1024
}
fn default_max_prompt_chars() -> usize {
    1_000_000
}
fn default_max_tokens_cap() -> u32 {
    8192
}
fn default_max_images_per_request() -> u32 {
    4
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            request_timeout_secs: default_request_timeout_secs(),
            max_queue_per_model: default_max_queue_per_model(),
            max_body_bytes: default_max_body_bytes(),
            max_audio_bytes: default_max_audio_bytes(),
            max_image_bytes: default_max_image_bytes(),
            max_document_bytes: default_max_document_bytes(),
            max_prompt_chars: default_max_prompt_chars(),
            max_tokens_cap: default_max_tokens_cap(),
            max_images_per_request: default_max_images_per_request(),
            allow_remote_image_urls: true,
        }
    }
}

/// Top-level configuration file (`hypnos.toml`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypnosConfig {
    /// Schema version of the file (`config_version`); absent means 1. A file written for a newer
    /// Hypnos is refused; an older one is migrated in memory and the steps are reported.
    #[serde(default = "default_config_version")]
    pub config_version: u32,
    /// Keys the file set that no field consumes (`server.foo`, `models[1].bar`): typos or keys a
    /// later version removed. Filled by [`HypnosConfig::from_toml_str`], reported by `validate`.
    #[serde(skip)]
    pub unknown_keys: Vec<String>,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub hardware: HardwareConfig,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub models: Vec<ModelConfig>,
}

/// The config schema this build writes and reads. Bump it when a key is renamed, removed or
/// changes meaning, and add the migration to [`HypnosConfig::migrate`].
pub const CONFIG_VERSION: u32 = 1;

fn default_config_version() -> u32 {
    1
}

/// Keys in `raw` (what the file says) that `known` (the parsed config, serialized back) has no
/// field for, as dotted paths. Arrays of tables are indexed (`models[0].arch.foo`).
fn unknown_keys(raw: &toml::Value, known: &toml::Value, prefix: &str, out: &mut Vec<String>) {
    match (raw, known) {
        (toml::Value::Table(r), toml::Value::Table(k)) => {
            for (key, value) in r {
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}.{key}")
                };
                match k.get(key) {
                    None => out.push(path),
                    Some(kv) => unknown_keys(value, kv, &path, out),
                }
            }
        }
        (toml::Value::Array(r), toml::Value::Array(k)) => {
            for (i, value) in r.iter().enumerate() {
                if let Some(kv) = k.get(i) {
                    unknown_keys(value, kv, &format!("{prefix}[{i}]"), out);
                }
            }
        }
        _ => {}
    }
}

/// Outcome of [`HypnosConfig::validate`]: hard errors abort startup, warnings are logged.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ValidationReport {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl ValidationReport {
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }
}

impl HypnosConfig {
    pub fn from_toml_str(content: &str) -> crate::error::Result<Self> {
        let err = |e: toml::de::Error| crate::error::HypnosError::Config(e.to_string());
        let mut config: Self = toml::from_str(content).map_err(err)?;
        if config.config_version > CONFIG_VERSION {
            return Err(crate::error::HypnosError::Config(format!(
                "config_version = {} was written for a newer Hypnos; this build reads version {CONFIG_VERSION} (see README, \"Config versions\")",
                config.config_version
            )));
        }
        if config.config_version == 0 {
            return Err(crate::error::HypnosError::Config(
                "config_version must be 1 or higher".into(),
            ));
        }
        // Serde fills defaults silently; remember which arch keys the file actually wrote so GGUF
        // detection can fill the rest without overriding them.
        let raw: toml::Table = toml::from_str(content).map_err(err)?;
        // Keys no field consumed: serde ignores them, which hides typos and removed keys.
        if let Ok(known) = toml::Value::try_from(&config) {
            let mut unknown = Vec::new();
            unknown_keys(&toml::Value::Table(raw.clone()), &known, "", &mut unknown);
            config.unknown_keys = unknown;
        }
        let raw_models = raw.get("models").and_then(|m| m.as_array());
        for (i, model) in config.models.iter_mut().enumerate() {
            let keys = raw_models
                .and_then(|m| m.get(i))
                .and_then(|m| m.get("arch"))
                .and_then(|a| a.as_table())
                .map(|t| t.keys().cloned().collect())
                .unwrap_or_default();
            model.arch.configured_keys = Some(keys);
        }
        config.migrate();
        Ok(config)
    }

    /// Brings a config written for an older `config_version` up to [`CONFIG_VERSION`], one step
    /// per version, and returns a note per step taken (none so far: version 1 is the first).
    pub fn migrate(&mut self) -> Vec<String> {
        let notes = Vec::new();
        while self.config_version < CONFIG_VERSION {
            // Future steps go here as `match self.config_version { 1 => { ...; notes.push(...) } }`.
            self.config_version += 1;
        }
        notes
    }

    /// Applies `server.models_dir` (or `env_override`, `HYPNOS_MODELS_DIR`): the folder becomes
    /// absolute against `base_dir` and every relative model path is joined onto it, so the
    /// engines, the GGUF detection and `validate` all see the final paths. Without a folder the
    /// paths stay relative to `base_dir`, as before. Returns the folder that was applied.
    pub fn apply_models_dir(
        &mut self,
        base_dir: &std::path::Path,
        env_override: Option<&str>,
    ) -> Option<PathBuf> {
        let configured = env_override
            .map(str::trim)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
            .or_else(|| self.server.models_dir.clone())?;
        let dir = if configured.is_absolute() {
            configured
        } else {
            base_dir.join(configured)
        };
        let rebase = |p: &mut PathBuf| {
            if p.is_relative() {
                *p = dir.join(&*p);
            }
        };
        for m in &mut self.models {
            rebase(&mut m.weights_path);
            for p in [
                &mut m.aux_weights_path,
                &mut m.text_encoder_path,
                &mut m.tokenizer_path,
                &mut m.voices_path,
            ]
            .into_iter()
            .flatten()
            {
                rebase(p);
            }
        }
        self.server.models_dir = Some(dir.clone());
        Some(dir)
    }

    /// Fills `[models.arch]` fields the config left unset from each model's GGUF metadata
    /// (`block_count`, `head_count_kv`, `key_length`, `context_length`, sliding window). Explicit
    /// values are kept. Returns one line per model describing what was detected, for the log.
    pub fn detect_arch_from_gguf(&mut self, base_dir: &std::path::Path) -> Vec<String> {
        let mut notes = Vec::new();
        for m in &mut self.models {
            let path = if m.weights_path.is_absolute() {
                m.weights_path.clone()
            } else {
                base_dir.join(&m.weights_path)
            };
            let is_gguf = path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
            if !is_gguf || !path.is_file() || m.arch.configured_keys.is_none() {
                continue;
            }
            let meta = match crate::gguf::GgufArchMetadata::read(&path) {
                Ok(Some(meta)) => meta,
                Ok(None) => continue,
                Err(e) => {
                    notes.push(format!("'{}': could not read GGUF metadata: {e}", m.id));
                    continue;
                }
            };
            let arch = &mut m.arch;
            let mut applied = Vec::new();
            let mut kept = Vec::new();
            let mut fill = |key: &str, detected: Option<u32>, field: &mut u32, configured: bool| {
                let Some(v) = detected else { return };
                if !configured {
                    *field = v;
                    applied.push(format!("{key}={v}"));
                } else if *field != v {
                    kept.push(format!("{key}={} (GGUF says {v})", *field));
                }
            };
            let c = |k: &str| arch.is_configured(k);
            let (c_layers, c_kv, c_dim, c_ctx) = (
                c("num_layers"),
                c("num_kv_heads"),
                c("head_dim"),
                c("max_context_len"),
            );
            let (c_swa, c_global) = (c("sliding_window_len"), c("global_attn_every_n_layers"));
            fill(
                "num_layers",
                meta.block_count,
                &mut arch.num_layers,
                c_layers,
            );
            fill(
                "num_kv_heads",
                meta.head_count_kv,
                &mut arch.num_kv_heads,
                c_kv,
            );
            fill("head_dim", meta.head_dim, &mut arch.head_dim, c_dim);
            fill(
                "max_context_len",
                meta.context_length,
                &mut arch.max_context_len,
                c_ctx,
            );
            // Sliding-window fields are optional; fill them as a pair so a config that sets only
            // one of them is not half-overridden.
            if !c_swa && !c_global {
                if let (Some(w), Some(n)) = (meta.sliding_window, meta.global_attn_every_n_layers) {
                    arch.sliding_window_len = Some(w);
                    arch.global_attn_every_n_layers = Some(n);
                    applied.push(format!(
                        "sliding_window_len={w}, global_attn_every_n_layers={n}"
                    ));
                }
            }
            if let Some((global, global_heads, swa, swa_heads, swa_dim)) = meta.kv_layout {
                let keys = [
                    "global_kv_layers",
                    "global_num_kv_heads",
                    "swa_kv_layers",
                    "swa_num_kv_heads",
                    "swa_head_dim",
                ];
                if keys.iter().all(|k| !arch.is_configured(k)) {
                    arch.global_kv_layers = Some(global);
                    arch.global_num_kv_heads = Some(global_heads);
                    arch.swa_kv_layers = Some(swa);
                    arch.swa_num_kv_heads = Some(swa_heads);
                    arch.swa_head_dim = Some(swa_dim);
                    applied.push(format!(
                        "KV layout: {global} full ({global_heads} KV heads) + {swa} sliding-window layers ({swa_heads} x {swa_dim})"
                    ));
                }
            }
            if !arch.is_configured("expert_bytes") {
                if let Some(e) = meta.expert_bytes {
                    arch.expert_bytes = e;
                    applied.push(format!("expert_bytes={e}"));
                }
            }
            // A detected training context below the configured default would fail validation;
            // the default is a reservation, so shrink it to what the model supports.
            if !arch.is_configured("default_context_len")
                && arch.default_context_len > arch.max_context_len
            {
                arch.default_context_len = arch.max_context_len;
                applied.push(format!("default_context_len={}", arch.max_context_len));
            }
            let mut line = format!(
                "'{}': [models.arch] from GGUF ({})",
                m.id, meta.architecture
            );
            line += &if applied.is_empty() {
                ": nothing to fill".to_string()
            } else {
                format!(": {}", applied.join(", "))
            };
            if !kept.is_empty() {
                line += &format!("; kept from config: {}", kept.join(", "));
            }
            notes.push(line);
        }
        notes
    }

    /// Checks the configuration against the filesystem and basic invariants. `base_dir` is the
    /// directory relative paths are resolved from (normally the working directory).
    pub fn validate(&self, base_dir: &std::path::Path) -> ValidationReport {
        let mut r = ValidationReport::default();
        let resolve = |p: &std::path::Path| {
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                base_dir.join(p)
            }
        };

        if self.models.is_empty() {
            r.errors.push("no [[models]] configured".into());
        }
        for key in &self.unknown_keys {
            r.warnings.push(format!(
                "unknown key `{key}` is ignored (a typo, or a key this version no longer has; see README \"Config versions\")"
            ));
        }
        // A missing models_dir (an unplugged drive, a moved folder) is one problem, reported once
        // below, not one "not found" per model file.
        let missing_models_dir = self
            .server
            .models_dir
            .as_deref()
            .map(resolve)
            .filter(|d| !d.is_dir());
        let in_missing_dir = |full: &std::path::Path| {
            missing_models_dir
                .as_deref()
                .is_some_and(|d| full.starts_with(d))
        };
        let mut stranded: Vec<String> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for m in &self.models {
            let id = &m.id;
            if id.trim().is_empty() {
                r.errors.push("a model has an empty id".into());
                continue;
            }
            if !seen.insert(id.clone()) {
                r.errors.push(format!("duplicate model id '{id}'"));
            }
            let mut required: Vec<(&str, Option<&std::path::PathBuf>)> =
                vec![("weights_path", Some(&m.weights_path))];
            if let Some(p) = &m.aux_weights_path {
                required.push(("aux_weights_path", Some(p)));
            }
            if let Some(p) = &m.text_encoder_path {
                required.push(("text_encoder_path", Some(p)));
            }
            if let Some(p) = &m.tokenizer_path {
                required.push(("tokenizer_path", Some(p)));
            }
            // A disabled model keeps its block without its files having to be there.
            for (label, path) in required {
                if let Some(p) = path {
                    let full = resolve(p);
                    if !full.is_file() {
                        if in_missing_dir(&full) {
                            if m.enabled && !stranded.contains(id) {
                                stranded.push(id.clone());
                            }
                        } else if m.enabled {
                            r.errors
                                .push(format!("'{id}': {label} not found: {}", full.display()));
                        } else {
                            r.warnings.push(format!(
                                "'{id}': disabled; {label} not found: {}",
                                full.display()
                            ));
                        }
                    }
                }
            }
            if let Some(v) = &m.voices_path {
                let full = resolve(v);
                if !full.is_dir() {
                    if in_missing_dir(&full) {
                        if m.enabled && !stranded.contains(id) {
                            stranded.push(id.clone());
                        }
                    } else if m.enabled {
                        r.errors.push(format!(
                            "'{id}': voices_path is not a directory: {}",
                            full.display()
                        ));
                    } else {
                        r.warnings.push(format!(
                            "'{id}': disabled; voices_path is not a directory: {}",
                            full.display()
                        ));
                    }
                }
            }
            match m.engine {
                EngineKind::GgmlDiffusion => {
                    if m.text_encoder_path.is_none() {
                        r.errors.push(format!("'{id}': ggml_diffusion needs text_encoder_path (LLM text encoder GGUF)"));
                    }
                    if m.aux_weights_path.is_none() {
                        r.errors.push(format!(
                            "'{id}': ggml_diffusion needs aux_weights_path (VAE)"
                        ));
                    }
                }
                EngineKind::OnnxRuntime => {
                    if m.modality != Modality::TextToSpeech {
                        r.warnings.push(format!(
                            "'{id}': onnx_runtime engine only implements text_to_speech"
                        ));
                    }
                }
                EngineKind::Docling => {
                    if m.modality != Modality::DocumentConversion {
                        r.errors.push(format!(
                            "'{id}': docling engine only implements document_conversion"
                        ));
                    }
                    // `device = "gpu"` depends on how the server was built; it checks that itself.
                }
                EngineKind::GgmlWhisper => {
                    r.errors.push(format!("'{id}': ggml_whisper is not compiled into this build; use ggml_llama with a Gemma 4 model for speech_to_text"));
                }
                EngineKind::GgmlLlama | EngineKind::Gliner => {}
            }
            if m.engine == EngineKind::Docling
                && m.arch.activation_workspace_bytes < 1024 * 1024 * 1024
            {
                r.warnings.push(format!(
                    "'{id}': arch.activation_workspace_bytes is under 1 GiB; docling needs about 1.5 GiB while converting on the CPU (about 5 GiB with device = \"gpu\"), which this booking does not cover"
                ));
            }
            if m.engine == EngineKind::Gliner {
                if m.modality != Modality::EntityExtraction {
                    r.errors.push(format!(
                        "'{id}': gliner engine only implements entity_extraction"
                    ));
                }
                if m.device != ExecutionDevice::Cpu {
                    r.errors.push(format!(
                        "'{id}': gliner runs on the CPU in this build; set device = \"cpu\""
                    ));
                }
            }
            if m.modality == Modality::EntityExtraction && m.engine != EngineKind::Gliner {
                r.errors.push(format!(
                    "'{id}': entity_extraction needs engine = \"gliner\""
                ));
            }
            if m.modality == Modality::DocumentConversion && m.engine != EngineKind::Docling {
                r.errors.push(format!(
                    "'{id}': document_conversion needs engine = \"docling\""
                ));
            }
            if m.arch.num_layers == 0 {
                r.errors
                    .push(format!("'{id}': arch.num_layers must be > 0"));
            }
            if m.arch.default_context_len == 0
                || m.arch.default_context_len > m.arch.max_context_len
            {
                r.errors.push(format!(
                    "'{id}': arch.default_context_len ({}) must be between 1 and max_context_len ({})",
                    m.arch.default_context_len, m.arch.max_context_len
                ));
            }
            if !(0.0..=1.0).contains(&m.min_gpu_layer_ratio) {
                r.errors.push(format!(
                    "'{id}': min_gpu_layer_ratio must be within 0.0..=1.0"
                ));
            }
            if m.slots == 0 {
                r.errors.push(format!("'{id}': slots must be >= 1"));
            }
            if m.slots > 1 && m.min_gpu_layer_ratio < 1.0 {
                r.warnings.push(format!(
                    "'{id}': slots = {} with min_gpu_layer_ratio < 1.0; a partial GPU split decodes each slot on the CPU side too, so batching gains little",
                    m.slots
                ));
            }
            if m.total_weight_bytes > 0 {
                let full = resolve(&m.weights_path);
                if let Ok(meta) = std::fs::metadata(&full) {
                    let on_disk = meta.len();
                    if on_disk > 0 && m.total_weight_bytes < on_disk {
                        r.warnings.push(format!(
                            "'{id}': total_weight_bytes ({}) is smaller than the weights file ({} bytes); planning will under-book VRAM",
                            m.total_weight_bytes, on_disk
                        ));
                    }
                }
            }
        }

        if self
            .models
            .iter()
            .filter(|m| m.engine == EngineKind::Docling)
            .count()
            > 1
        {
            r.errors.push(
                "only one docling model per server: docling.rs reads its models directory from the process environment".into(),
            );
        }
        if self.hardware.vram_budget_bytes > 0
            && self.hardware.vram_safety_margin_bytes >= self.hardware.vram_budget_bytes
        {
            r.errors.push(
                "hardware.vram_safety_margin_bytes must be smaller than vram_budget_bytes".into(),
            );
        }
        if self.hardware.pinned_ram_budget_bytes < 256 * 1024 * 1024 {
            r.warnings.push(
                "hardware.pinned_ram_budget_bytes is below 256 MiB; Tier 2 will be unusable".into(),
            );
        }
        if self.server.api_keys.iter().any(|k| k.trim().is_empty()) {
            r.errors
                .push("server.api_keys contains an empty key".into());
        }
        if self.server.api_keys.iter().any(|k| k.trim().len() < 16) {
            r.warnings.push(
                "server.api_keys has a key shorter than 16 characters; use long random keys (e.g. `openssl rand -hex 32`)".into(),
            );
        }
        let mut labels = std::collections::HashSet::new();
        for k in &self.server.keys {
            let label = k.label.trim();
            if label.is_empty() {
                r.errors
                    .push("a [[server.keys]] entry has an empty label".into());
            } else if !labels.insert(label.to_string()) {
                r.errors
                    .push(format!("duplicate [[server.keys]] label '{label}'"));
            }
            if k.key.trim().is_empty() {
                r.errors
                    .push(format!("[[server.keys]] '{label}' has an empty key"));
            } else if k.key.trim().len() < 16 {
                r.warnings.push(format!(
                    "[[server.keys]] '{label}' is shorter than 16 characters; use long random keys"
                ));
            }
        }
        if self.server.request_log_redact.len() > 1
            && self
                .server
                .request_log_redact
                .iter()
                .any(|p| p.trim() == "all")
        {
            r.warnings.push(
                "server.request_log_redact contains \"all\" next to patterns; \"all\" wins".into(),
            );
        }
        if let Some(dir) = &self.server.frontend_dir {
            let index = resolve(dir).join("index.html");
            if !index.is_file() {
                r.errors.push(format!(
                    "server.frontend_dir has no index.html: {}",
                    index.display()
                ));
            }
        }
        if let Some(tls) = &self.server.tls {
            for (label, path) in [("cert_path", &tls.cert_path), ("key_path", &tls.key_path)] {
                let full = resolve(path);
                if !full.is_file() {
                    r.errors
                        .push(format!("server.tls.{label} not found: {}", full.display()));
                }
            }
        }
        if !self.server.is_loopback_host() {
            if self.server.api_keys.is_empty() {
                r.warnings.push(format!(
                    "server.host = \"{}\" exposes an unauthenticated API beyond this machine; set server.api_keys (or HYPNOS_API_KEYS) and server.tls",
                    self.server.host
                ));
            } else if self.server.tls.is_none() {
                r.warnings.push(format!(
                    "server.host = \"{}\" without server.tls: API keys and data cross the network unencrypted",
                    self.server.host
                ));
            }
        }
        if !matches!(self.server.log_format.as_str(), "pretty" | "json") {
            r.errors.push(format!(
                "server.log_format must be \"pretty\" or \"json\", got \"{}\"",
                self.server.log_format
            ));
        }
        if !REQUEST_LOG_LEVELS
            .contains(&self.server.request_log_level.to_ascii_lowercase().as_str())
        {
            r.errors.push(format!(
                "server.request_log_level must be one of {}, got \"{}\"",
                REQUEST_LOG_LEVELS.join(", "),
                self.server.request_log_level
            ));
        }
        if self.server.request_log_capacity == 0 {
            r.warnings
                .push("server.request_log_capacity is 0; the request log keeps nothing".into());
        }
        if self.limits.max_queue_per_model == 0 {
            r.errors
                .push("limits.max_queue_per_model must be >= 1".into());
        }
        if self.limits.request_timeout_secs == 0 {
            r.errors
                .push("limits.request_timeout_secs must be >= 1".into());
        }
        if let Some(dir) = &missing_models_dir {
            if stranded.is_empty() {
                r.warnings.push(format!(
                    "models_dir not found: {} (an unplugged drive or a moved folder?)",
                    dir.display()
                ));
            } else {
                r.errors.push(format!(
                    "models_dir not found: {} (an unplugged drive or a moved folder?); enabled models with their files there: {}",
                    dir.display(),
                    stranded.join(", ")
                ));
            }
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_folders_beside_the_binary_or_in_the_app_bundle() {
        use std::path::Path;
        assert_eq!(
            beside_executable(Path::new("/opt/hypnos"), "frontend"),
            vec![PathBuf::from("/opt/hypnos/frontend")]
        );
        assert_eq!(
            beside_executable(
                Path::new("/Applications/Hypnos.app/Contents/MacOS"),
                "espeak-ng-data"
            ),
            vec![
                PathBuf::from("/Applications/Hypnos.app/Contents/MacOS/espeak-ng-data"),
                PathBuf::from("/Applications/Hypnos.app/Contents/Resources/espeak-ng-data"),
            ]
        );
    }

    fn minimal_toml(weights: &str) -> String {
        format!(
            r#"
[[models]]
id = "m"
modality = "text_generation"
engine = "ggml_llama"
weights_path = "{weights}"
[models.arch]
num_layers = 4
"#
        )
    }

    #[test]
    fn unknown_keys_and_versions_are_reported() {
        let toml = format!(
            "[server]
prot = 8080
[limits]
max_queue = 4
{}",
            minimal_toml("w.gguf")
        );
        let cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        assert_eq!(cfg.config_version, 1, "absent means 1");
        assert_eq!(cfg.unknown_keys, vec!["limits.max_queue", "server.prot"]); // table order
        let report = cfg.validate(std::path::Path::new("."));
        assert!(report
            .warnings
            .iter()
            .any(|w| w.contains("unknown key `server.prot`")));
        // A key inside a model block is reported with its index.
        let cfg = HypnosConfig::from_toml_str(&minimal_toml("w.gguf").replace(
            "[models.arch]\n",
            "weihgts = 1\n[models.arch]\nnum_layerz = 4\n",
        ))
        .unwrap();
        assert_eq!(
            cfg.unknown_keys,
            vec!["models[0].arch.num_layerz", "models[0].weihgts"]
        );
        // Versions: newer than this build is refused, 0 is refused, current passes.
        let newer = format!(
            "config_version = {}\n{}",
            CONFIG_VERSION + 1,
            minimal_toml("w.gguf")
        );
        assert!(HypnosConfig::from_toml_str(&newer).is_err());
        assert!(HypnosConfig::from_toml_str("config_version = 0\n").is_err());
        let same = format!(
            "config_version = {CONFIG_VERSION}\n{}",
            minimal_toml("w.gguf")
        );
        assert_eq!(
            HypnosConfig::from_toml_str(&same).unwrap().config_version,
            CONFIG_VERSION
        );
    }

    #[test]
    fn defaults_bind_loopback() {
        let cfg = HypnosConfig::from_toml_str("").unwrap();
        assert_eq!(cfg.server.host, "127.0.0.1");
        assert_eq!(cfg.limits.max_queue_per_model, 8);
    }

    #[test]
    fn validate_reports_missing_files_and_duplicates() {
        let dir = std::env::temp_dir().join(format!("hypnos-cfg-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut toml = minimal_toml("missing.gguf");
        toml.push_str(&minimal_toml("missing.gguf"));
        let cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        let report = cfg.validate(&dir);
        assert!(report.errors.iter().any(|e| e.contains("not found")));
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("duplicate model id")));
    }

    #[test]
    fn disabled_models_tolerate_missing_weights() {
        let dir = std::env::temp_dir().join(format!("hypnos-cfg-disabled-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let toml = minimal_toml("missing.gguf").replace(
            "engine = \"ggml_llama\"",
            "engine = \"ggml_llama\"\nenabled = false",
        );
        let cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        assert!(!cfg.models[0].enabled);
        let report = cfg.validate(&dir);
        assert!(report.is_ok(), "{:?}", report.errors);
        assert!(report.warnings.iter().any(|w| w.contains("disabled")));
        // The default is enabled, and then the file must exist.
        let cfg = HypnosConfig::from_toml_str(&minimal_toml("missing.gguf")).unwrap();
        assert!(cfg.models[0].enabled);
        assert!(!cfg.validate(&dir).is_ok());
    }

    #[test]
    fn validate_accepts_existing_weights() {
        let dir = std::env::temp_dir().join(format!("hypnos-cfg-ok-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.gguf"), b"GGUF").unwrap();
        let cfg = HypnosConfig::from_toml_str(&minimal_toml("w.gguf")).unwrap();
        let report = cfg.validate(&dir);
        assert!(report.is_ok(), "{:?}", report.errors);
    }

    #[test]
    fn models_dir_rebases_relative_model_paths() {
        let base = std::env::temp_dir().join(format!("hypnos-cfg-md-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        std::fs::create_dir_all(base.join("store")).unwrap();
        std::fs::write(base.join("store/w.gguf"), b"GGUF").unwrap();
        // Absolute on every OS ("/abs" is not on Windows: it has no drive). TOML literal strings
        // keep Windows backslashes as they are.
        let abs_aux = base.join("abs").join("mmproj.gguf");
        let elsewhere = base.join("elsewhere");
        let toml = format!(
            "[server]\nmodels_dir = \"store\"\n{}",
            minimal_toml("w.gguf").replace(
                "[models.arch]",
                &format!("aux_weights_path = '{}'\n[models.arch]", abs_aux.display())
            )
        );
        let mut cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        assert_eq!(cfg.apply_models_dir(&base, None), Some(base.join("store")));
        assert_eq!(cfg.models[0].weights_path, base.join("store/w.gguf"));
        // Absolute paths are left alone.
        assert_eq!(
            cfg.models[0].aux_weights_path.as_deref(),
            Some(abs_aux.as_path())
        );

        // HYPNOS_MODELS_DIR wins over the file; an absolute folder is taken as is.
        let mut cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        cfg.apply_models_dir(&base, Some(&elsewhere.display().to_string()));
        assert_eq!(cfg.models[0].weights_path, elsewhere.join("w.gguf"));

        // No folder: paths stay relative to the working directory, as before the key existed.
        let mut cfg = HypnosConfig::from_toml_str(&minimal_toml("w.gguf")).unwrap();
        assert_eq!(cfg.apply_models_dir(&base, Some("  ")), None);
        assert_eq!(cfg.models[0].weights_path, PathBuf::from("w.gguf"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn a_missing_models_dir_is_one_error_not_one_per_file() {
        let base = std::env::temp_dir().join(format!("hypnos-cfg-gone-{}", std::process::id()));
        std::fs::create_dir_all(&base).unwrap();
        let toml = format!(
            "[server]\nmodels_dir = \"/Volumes/unplugged-{}\"\n{}",
            std::process::id(),
            minimal_toml("w.gguf")
        );
        let mut cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        cfg.apply_models_dir(&base, None);
        let report = cfg.validate(&base);
        assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
        assert!(report.errors[0].starts_with("models_dir not found"));
        assert!(report.errors[0].contains("unplugged"));
        // With the model disabled nothing needs the folder: a warning, and the server starts.
        cfg.models[0].enabled = false;
        let report = cfg.validate(&base);
        assert!(report.is_ok(), "{:?}", report.errors);
        assert!(report
            .warnings
            .iter()
            .any(|w| w.starts_with("models_dir not found")));
    }

    #[test]
    fn unified_memory_oversubscription_is_reported() {
        let gb = 1024 * 1024 * 1024;
        let mut hw = HardwareConfig {
            vram_budget_bytes: 24 * gb,
            pinned_ram_budget_bytes: 32 * gb,
            ..HardwareConfig::default()
        };
        assert!(hw.unified_memory_oversubscription(32 * gb).is_some());
        assert!(hw.unified_memory_oversubscription(64 * gb).is_none());
        // Auto-detected Tier 1 is derived from the device and clamped, so nothing to report.
        hw.vram_budget_bytes = 0;
        assert!(hw.unified_memory_oversubscription(32 * gb).is_none());
    }

    #[test]
    fn api_keys_env_override_replaces_the_list() {
        let mut server = ServerConfig {
            api_keys: vec!["from-file".into()],
            ..ServerConfig::default()
        };
        server.apply_env_api_keys(None);
        assert_eq!(server.api_keys, vec!["from-file"]);
        server.apply_env_api_keys(Some(" key-a , ,key-b "));
        assert_eq!(server.api_keys, vec!["key-a", "key-b"]);
    }

    #[test]
    fn validate_checks_auth_and_tls_settings() {
        let dir = std::env::temp_dir().join(format!("hypnos-tls-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("w.gguf"), b"GGUF").unwrap();
        let mut cfg = HypnosConfig::from_toml_str(&minimal_toml("w.gguf")).unwrap();

        // Public bind without keys: warned, not refused.
        cfg.server.host = "0.0.0.0".into();
        let r = cfg.validate(&dir);
        assert!(r.is_ok());
        assert!(r.warnings.iter().any(|w| w.contains("unauthenticated")));

        // Keys but no TLS: a different warning.
        cfg.server.api_keys = vec!["0123456789abcdef0123".into()];
        let r = cfg.validate(&dir);
        assert!(r.warnings.iter().any(|w| w.contains("unencrypted")));
        assert!(!r.warnings.iter().any(|w| w.contains("unauthenticated")));

        // Missing PEM files are errors; short and empty keys are flagged.
        cfg.server.tls = Some(TlsConfig {
            cert_path: "missing-cert.pem".into(),
            key_path: "missing-key.pem".into(),
        });
        cfg.server.api_keys = vec!["short".into(), " ".into()];
        let r = cfg.validate(&dir);
        assert!(r.errors.iter().any(|e| e.contains("server.tls.cert_path")));
        assert!(r.errors.iter().any(|e| e.contains("server.tls.key_path")));
        assert!(r.errors.iter().any(|e| e.contains("empty key")));
        assert!(r.warnings.iter().any(|w| w.contains("shorter than 16")));
    }

    #[test]
    fn lock_host_memory_defaults_by_topology() {
        let mut hw = HardwareConfig::default();
        assert!(hw.lock_host_memory_for(MemoryTopology::DiscretePcie));
        assert!(!hw.lock_host_memory_for(MemoryTopology::UnifiedMemory));
        hw.lock_host_memory = Some(true);
        assert!(hw.lock_host_memory_for(MemoryTopology::UnifiedMemory));
    }

    #[test]
    fn validate_flags_diffusion_without_companions() {
        let dir = std::env::temp_dir();
        let toml = r#"
[[models]]
id = "img"
modality = "image_generation"
engine = "ggml_diffusion"
weights_path = "x.gguf"
[models.arch]
num_layers = 4
"#;
        let cfg = HypnosConfig::from_toml_str(toml).unwrap();
        let report = cfg.validate(&dir);
        assert!(report
            .errors
            .iter()
            .any(|e| e.contains("text_encoder_path")));
        assert!(report.errors.iter().any(|e| e.contains("aux_weights_path")));
    }

    #[test]
    fn validate_checks_docling_models() {
        let dir = std::env::temp_dir().join(format!("hypnos-docling-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("layout.onnx"), b"onnx").unwrap();
        let docling = |id: &str, extra: &str| {
            format!(
                r#"
[[models]]
id = "{id}"
modality = "document_conversion"
engine = "docling"
weights_path = "layout.onnx"
{extra}
[models.arch]
activation_workspace_bytes = 3758096384
"#
            )
        };
        let cfg = HypnosConfig::from_toml_str(&docling("doc", "device = \"cpu\"")).unwrap();
        assert!(cfg.models[0].runs_on_cpu());
        let r = cfg.validate(&dir);
        assert!(r.is_ok(), "{:?}", r.errors);

        // Two of them, or the wrong engine: refused. Whether `device = "gpu"` works depends on the
        // build, which the server checks at startup.
        let two = docling("a", "device = \"cpu\"") + &docling("b", "device = \"cpu\"");
        let cfg = HypnosConfig::from_toml_str(&two).unwrap();
        assert!(cfg
            .validate(&dir)
            .errors
            .iter()
            .any(|e| e.contains("only one docling")));
        let wrong = docling("doc", "device = \"cpu\"")
            .replace("engine = \"docling\"", "engine = \"ggml_llama\"");
        let cfg = HypnosConfig::from_toml_str(&wrong).unwrap();
        assert!(cfg
            .validate(&dir)
            .errors
            .iter()
            .any(|e| e.contains("document_conversion needs engine")));
    }

    #[test]
    fn validate_checks_gliner_models() {
        let dir = std::env::temp_dir().join(format!("hypnos-gliner-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("encoder.onnx"), b"onnx").unwrap();
        let toml = |modality: &str, device: &str| {
            format!(
                r#"
[[models]]
id = "ner"
modality = "{modality}"
engine = "gliner"
device = "{device}"
weights_path = "encoder.onnx"
"#
            )
        };
        let ok = HypnosConfig::from_toml_str(&toml("entity_extraction", "cpu")).unwrap();
        assert!(ok.validate(&dir).is_ok(), "{:?}", ok.validate(&dir).errors);
        let gpu = HypnosConfig::from_toml_str(&toml("entity_extraction", "gpu")).unwrap();
        assert!(gpu
            .validate(&dir)
            .errors
            .iter()
            .any(|e| e.contains("gliner runs on the CPU")));
        let wrong = HypnosConfig::from_toml_str(&toml("embeddings", "cpu")).unwrap();
        assert!(wrong
            .validate(&dir)
            .errors
            .iter()
            .any(|e| e.contains("only implements entity_extraction")));
    }
}
