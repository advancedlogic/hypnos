//! Native inference engines for Hypnos. Everything here runs inside the server process:
//!
//! | engine          | library                         | models                                   |
//! |-----------------|---------------------------------|------------------------------------------|
//! | `ggml_llama`    | llama.cpp + libmtmd (CUDA)      | gemma4:26b-q4, gemma4:e4b-q4, gemmaembedding2 |
//! | `ggml_diffusion`| stable-diffusion.cpp (CUDA)     | moss-image                               |
//! | `onnx_runtime`  | ONNX Runtime + espeak-ng        | kokoro-82m-q4                            |
//! | `docling`       | docling.rs on ONNX Runtime (CPU)| docling                                  |
//! | `gliner`        | gliner25-rs on ONNX Runtime (CPU)| gliner2.5-multi                         |
//!
//! No external runtime (Ollama, LM Studio, Python sidecars) is involved.

pub mod diffusion;
pub mod document;
mod engine;
pub mod entities;
pub mod gguf_inspector;
pub mod kokoro;
pub mod kv_park;
pub mod llama;
pub mod media;
mod onnx;
mod tools;
mod worker;

use async_trait::async_trait;
use gguf_inspector::GgufHeaderSummary;
use hypnos_core::config::{EngineKind, MemoryTier, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::{GpuMemoryProbe, ModelBackend};
use std::sync::Arc;
use tracing::info;

/// Reads free/total memory of the primary GPU through ggml's backend-device API (the same device
/// llama.cpp allocates on), so the scheduler can reconcile its bookkeeping with reality.
pub struct LlamaGpuProbe;

impl GpuMemoryProbe for LlamaGpuProbe {
    fn probe(&self) -> Option<(u64, u64)> {
        llama::llama_backend().ok()?;
        use llama_cpp_2::LlamaBackendDeviceType as T;
        let devices = llama_cpp_2::list_llama_ggml_backend_devices();
        let gpu = devices
            .iter()
            .find(|d| matches!(d.device_type, T::Gpu) && d.memory_total > 0)
            .or_else(|| {
                devices.iter().find(|d| {
                    matches!(d.device_type, T::IntegratedGpu | T::Accelerator) && d.memory_total > 0
                })
            })?;
        Some((gpu.memory_free as u64, gpu.memory_total as u64))
    }
}

/// Tracks the active placement plan a backend was last given by the scheduler.
#[derive(Debug, Clone)]
pub struct BackendContextState {
    pub active_plan: LayerPlacementPlan,
    pub pinned_host_ptr: Option<usize>,
    pub gguf_header: Option<GgufHeaderSummary>,
    /// Phases of the last Tier 1 load (llama.cpp engines).
    pub last_load_phases: Option<hypnos_core::openai::LoadPhases>,
}

impl BackendContextState {
    fn new(config: &ModelConfig) -> Self {
        let gguf_header = GgufHeaderSummary::inspect(&config.weights_path);
        if let Some(ref hdr) = gguf_header {
            info!(
                model = %config.id,
                gguf_version = hdr.version,
                tensors = hdr.tensor_count,
                kv_pairs = hdr.metadata_kv_count,
                "Detected valid GGUF header on disk"
            );
        }
        Self {
            active_plan: LayerPlacementPlan {
                primary_tier: MemoryTier::Tier3NvmeSsd,
                gpu_layers: 0,
                pinned_ram_layers: 0,
                ssd_layers: config.arch.num_layers,
                vram_weight_bytes: 0,
                vram_kv_and_workspace_bytes: 0,
                pinned_ram_bytes: 0,
                allocated_context_len: 0,
            },
            pinned_host_ptr: None,
            gguf_header,
            last_load_phases: None,
        }
    }
}

/// Number of CPU threads handed to the native engines (capped; GPU-resident models need few).
/// On Apple Silicon only the performance cores count: llama.cpp slows down when work is spread
/// onto efficiency cores.
pub(crate) fn default_threads() -> i32 {
    static THREADS: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *THREADS.get_or_init(|| {
        apple_performance_cores()
            .or_else(|| std::thread::available_parallelism().map(|n| n.get()).ok())
            .unwrap_or(4)
            .clamp(1, 16) as i32
    })
}

fn apple_performance_cores() -> Option<usize> {
    if !cfg!(target_os = "macos") {
        return None;
    }
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse()
        .ok()
        .filter(|&n| n > 0)
}

/// The ONNX Runtime provider a docling model with `device = "gpu"` runs on in this build: CUDA with
/// the `docling-cuda` feature, CoreML in the Metal build, none otherwise.
pub fn docling_gpu_provider() -> Option<&'static str> {
    if cfg!(feature = "docling-cuda") {
        Some("cuda")
    } else if cfg!(all(target_os = "macos", feature = "metal")) {
        Some("coreml")
    } else {
        None
    }
}

/// Process setup that must happen before any thread starts. On Linux with the `docling-cuda`
/// feature, ONNX Runtime finds its CUDA provider library relative to `argv[0]`, so a binary started
/// by bare name re-executes itself under its absolute path; a no-op everywhere else.
pub fn prepare_process() {
    docling_onnx::reexec_for_provider_libraries();
}

/// Server-wide settings the engines need beyond their own [`ModelConfig`].
#[derive(Debug, Clone)]
pub struct EngineOptions {
    /// How long an execution slot keeps its KV cache after its last request before it is cleared
    /// (`server.kv_idle_timeout_secs`).
    pub kv_idle_timeout: std::time::Duration,
    /// Where llama.cpp workers park their slots' KV state across a demotion (MEM-4); shared by
    /// every model, off unless `hardware.kv_park`.
    pub kv_park: Arc<kv_park::KvParkPool>,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            kv_idle_timeout: std::time::Duration::from_secs(300),
            kv_park: Arc::new(kv_park::KvParkPool::disabled()),
        }
    }
}

/// Factory creating the appropriate native engine for a configured model.
pub fn create_backend_for_model(
    config: ModelConfig,
    opts: &EngineOptions,
) -> Arc<dyn ModelBackend> {
    match config.engine {
        EngineKind::GgmlLlama => Arc::new(llama::GgmlLlamaBackend::new(config, opts.clone())),
        EngineKind::GgmlDiffusion => Arc::new(diffusion::GgmlDiffusionBackend::new(config)),
        EngineKind::OnnxRuntime => Arc::new(kokoro::OnnxRuntimeBackend::new(config)),
        EngineKind::Docling => Arc::new(document::DoclingBackend::new(config)),
        EngineKind::Gliner => Arc::new(entities::GlinerBackend::new(config)),
        EngineKind::GgmlWhisper => Arc::new(UnavailableEngineBackend {
            reason: "whisper.cpp is not compiled into this build; use a gemma4 model (ggml_llama) for speech_to_text",
            config,
        }),
    }
}

/// Placeholder for engine kinds that are declared in the config schema but not compiled in.
/// Every inference call fails loudly instead of returning fabricated output.
pub struct UnavailableEngineBackend {
    config: ModelConfig,
    reason: &'static str,
}

#[async_trait]
impl ModelBackend for UnavailableEngineBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    async fn apply_placement(
        &self,
        _plan: &LayerPlacementPlan,
        _pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        Err(HypnosError::Config(format!(
            "model '{}': {}",
            self.config.id, self.reason
        )))
    }
}
