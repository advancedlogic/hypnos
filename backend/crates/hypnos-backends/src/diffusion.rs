//! stable-diffusion.cpp engine, linked in-process through the vendored `diffusion-rs-sys` FFI
//! bindings (sd.cpp master-1024).
//!
//! The models it serves are diffusion transformers with an LLM text encoder and a VAE (MOSS-Image,
//! Z-Image, FLUX.2 klein, ERNIE-Image, ...): `weights_path` (the diffusion model GGUF),
//! `text_encoder_path` (the LLM GGUF), `aux_weights_path` (the VAE) and, where the model needs
//! one, `tokenizer_path`. sd.cpp recognizes the family from the weights. `[models.image]` holds
//! the steps, guidance, sampler and flow shift its model card recommends.

use std::ffi::{c_char, c_void, CStr, CString};
use std::io::Cursor;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Arc, Mutex as StdMutex, Once};
use std::time::Instant;

use async_trait::async_trait;
use base64::Engine;
use chrono::Utc;
use diffusion_rs_sys as sd;
use hypnos_core::config::{MemoryTier, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{GeneratedImage, ImageGenerationRequest, ImageGenerationResponse};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::media::parse_image_size;
use crate::{default_threads, BackendContextState};

const DEFAULT_STEPS: u32 = 20;
const DEFAULT_CFG_SCALE: f32 = 3.0;

unsafe extern "C" fn sd_log_bridge(
    level: sd::sd_log_level_t,
    text: *const c_char,
    _data: *mut c_void,
) {
    if text.is_null() {
        return;
    }
    let msg = CStr::from_ptr(text).to_string_lossy();
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    if matches!(
        level,
        sd::sd_log_level_t::SD_LOG_ERROR | sd::sd_log_level_t::SD_LOG_WARN
    ) && is_out_of_memory(msg)
    {
        if let Ok(mut first) = OOM_MESSAGE.lock() {
            first.get_or_insert_with(|| msg.to_string());
        }
    }
    match level {
        sd::sd_log_level_t::SD_LOG_ERROR => tracing::error!(target: "sd.cpp", "{msg}"),
        sd::sd_log_level_t::SD_LOG_WARN => tracing::warn!(target: "sd.cpp", "{msg}"),
        sd::sd_log_level_t::SD_LOG_INFO => tracing::info!(target: "sd.cpp", "{msg}"),
        _ => tracing::debug!(target: "sd.cpp", "{msg}"),
    }
}

/// The context of the generation currently running (only one runs at a time per process, since
/// each context is behind a mutex and sd.cpp callbacks are global). Lets the progress callback
/// ask sd.cpp to abort when the request was cancelled.
static ACTIVE_CTX: AtomicPtr<sd::sd_ctx_t> = AtomicPtr::new(std::ptr::null_mut());
static ACTIVE_REQUEST: StdMutex<Option<RequestContext>> = StdMutex::new(None);

/// The first device-allocation failure sd.cpp (or its ggml) logged during the running job. sd.cpp
/// only returns false from `generate_image`, so this is how a full device becomes
/// [`HypnosError::DeviceOutOfMemory`] (507, retried by the server after evicting idle models)
/// instead of a generic backend error.
static OOM_MESSAGE: StdMutex<Option<String>> = StdMutex::new(None);

/// sd.cpp's model manager, ggml's buffer and graph allocators, and the CUDA and Metal backends.
fn is_out_of_memory(msg: &str) -> bool {
    const MARKERS: [&str; 5] = [
        "cannot make enough memory available",
        "is available under current VRAM limits",
        "failed to allocate",
        "cudaMalloc failed",
        "out of memory",
    ];
    MARKERS.iter().any(|m| msg.contains(m))
}

unsafe extern "C" fn sd_progress_bridge(step: i32, steps: i32, time: f32, _data: *mut c_void) {
    debug!(target: "sd.cpp", step, steps, secs = time, "diffusion step");
    let cancelled = ACTIVE_REQUEST
        .lock()
        .map(|g| g.as_ref().map(|c| !c.should_continue()).unwrap_or(false))
        .unwrap_or(false);
    if cancelled {
        let ctx = ACTIVE_CTX.load(Ordering::SeqCst);
        if !ctx.is_null() {
            warn!(target: "sd.cpp", step, steps, "request cancelled; aborting generation");
            sd::sd_cancel_generation(ctx, sd::sd_cancel_mode_t::SD_CANCEL_ALL);
        }
    }
}

fn install_sd_callbacks() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| unsafe {
        sd::sd_set_log_callback(Some(sd_log_bridge), std::ptr::null_mut());
        sd::sd_set_progress_callback(Some(sd_progress_bridge), std::ptr::null_mut());
    });
}

/// A resident sd.cpp context. The raw pointer is only touched while the owning mutex is held.
struct LoadedSd {
    ctx: *mut sd::sd_ctx_t,
    gpu_layers: u32,
}

unsafe impl Send for LoadedSd {}

impl Drop for LoadedSd {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { sd::free_sd_ctx(self.ctx) };
            self.ctx = std::ptr::null_mut();
        }
    }
}

fn cstring(path: &std::path::Path) -> Result<CString> {
    CString::new(path.to_string_lossy().as_bytes())
        .map_err(|_| HypnosError::Config(format!("path contains NUL byte: {}", path.display())))
}

fn load_sd(cfg: &ModelConfig, gpu_layers: u32, n_threads: i32) -> Result<LoadedSd> {
    install_sd_callbacks();
    for (label, p) in [
        ("weights_path (diffusion model)", Some(&cfg.weights_path)),
        (
            "text_encoder_path (LLM text encoder)",
            cfg.text_encoder_path.as_ref(),
        ),
        ("aux_weights_path (VAE)", cfg.aux_weights_path.as_ref()),
    ] {
        match p {
            None if label.starts_with("aux") && cfg.image.pixel_space => {}
            Some(p) if p.exists() => {}
            Some(p) => {
                return Err(HypnosError::Config(format!(
                    "'{}': {label} not found: {}",
                    cfg.id,
                    p.display()
                )))
            }
            None => {
                return Err(HypnosError::Config(format!(
                    "'{}': {label} must be configured for the ggml_diffusion engine",
                    cfg.id
                )))
            }
        }
    }

    let diffusion = cstring(&cfg.weights_path)?;
    let llm = cstring(cfg.text_encoder_path.as_ref().expect("checked above"))?;
    // A pixel-space model (Iris) has no VAE.
    let vae = cfg
        .aux_weights_path
        .as_ref()
        .map(|p| cstring(p))
        .transpose()?;
    let tokenizer = match &cfg.tokenizer_path {
        Some(p) if p.exists() => Some(cstring(p)?),
        Some(p) => {
            warn!(model = %cfg.id, path = %p.display(), "tokenizer_path missing; using the text encoder's embedded tokenizer");
            None
        }
        None => None,
    };
    let offload = CString::new("*=cpu").expect("static");
    let partial = gpu_layers < cfg.arch.num_layers.max(1);

    let started = Instant::now();
    let ctx = unsafe {
        let mut p: sd::sd_ctx_params_t = std::mem::zeroed();
        sd::sd_ctx_params_init(&mut p);
        p.diffusion_model_path = diffusion.as_ptr();
        p.llm_path = llm.as_ptr();
        if let Some(v) = &vae {
            p.vae_path = v.as_ptr();
        }
        if let Some(t) = &tokenizer {
            p.tokenizer = t.as_ptr();
        }
        p.n_threads = n_threads;
        p.wtype = sd::sd_type_t::SD_TYPE_COUNT; // keep the on-disk quantization
        p.diffusion_flash_attn = true;
        if partial {
            // Weights stay in host RAM and stream to the GPU per block (sd-cli `--offload-to-cpu`).
            p.params_backend = offload.as_ptr();
        }
        sd::new_sd_ctx(&p)
    };
    if ctx.is_null() {
        return Err(HypnosError::BackendError(format!(
            "stable-diffusion.cpp failed to create a context for '{}' (see sd.cpp log lines above)",
            cfg.id
        )));
    }
    info!(
        model = %cfg.id,
        gpu_layers,
        offload_to_cpu = partial,
        load_secs = format!("{:.1}", started.elapsed().as_secs_f32()),
        "stable-diffusion.cpp context resident"
    );
    Ok(LoadedSd { ctx, gpu_layers })
}

struct ImageJob {
    prompt: String,
    negative_prompt: String,
    width: u32,
    height: u32,
    steps: u32,
    cfg_scale: f32,
    /// sd.cpp's sampler name (`[models.image] sample_method`); `None` = the model's default.
    sample_method: Option<String>,
    flow_shift: Option<f32>,
    /// Reference images, decoded to RGB (`/v1/images/edits`).
    references: Vec<image::RgbImage>,
    /// A distilled model's own noise schedule (`[models.image] sigmas`).
    sigmas: Option<Vec<f32>>,
    seed: i64,
    count: u32,
}

fn encode_png(width: u32, height: u32, channels: u32, pixels: &[u8]) -> Result<Vec<u8>> {
    let dynimg = match channels {
        3 => image::RgbImage::from_raw(width, height, pixels.to_vec())
            .map(image::DynamicImage::ImageRgb8),
        4 => image::RgbaImage::from_raw(width, height, pixels.to_vec())
            .map(image::DynamicImage::ImageRgba8),
        1 => image::GrayImage::from_raw(width, height, pixels.to_vec())
            .map(image::DynamicImage::ImageLuma8),
        _ => None,
    }
    .ok_or_else(|| {
        HypnosError::BackendError(format!(
            "unexpected image buffer ({width}x{height}x{channels}, {} bytes)",
            pixels.len()
        ))
    })?;
    let mut out = Cursor::new(Vec::new());
    dynimg
        .write_to(&mut out, image::ImageFormat::Png)
        .map_err(|e| HypnosError::BackendError(format!("png encode failed: {e}")))?;
    Ok(out.into_inner())
}

fn run_image_job(
    model_id: &str,
    loaded: &LoadedSd,
    job: &ImageJob,
    ctx: &RequestContext,
) -> Result<Vec<Vec<u8>>> {
    if !ctx.should_continue() {
        return Err(HypnosError::Cancelled);
    }
    ACTIVE_CTX.store(loaded.ctx, Ordering::SeqCst);
    if let Ok(mut g) = ACTIVE_REQUEST.lock() {
        *g = Some(ctx.clone());
    }
    if let Ok(mut oom) = OOM_MESSAGE.lock() {
        *oom = None;
    }
    // Reset the cancel flag inside sd.cpp from any previous aborted job.
    unsafe { sd::sd_cancel_generation(loaded.ctx, sd::sd_cancel_mode_t::SD_CANCEL_RESET) };
    let result = run_image_job_inner(loaded, job);
    if let Ok(mut g) = ACTIVE_REQUEST.lock() {
        *g = None;
    }
    ACTIVE_CTX.store(std::ptr::null_mut(), Ordering::SeqCst);
    if !ctx.should_continue() {
        return Err(HypnosError::Cancelled);
    }
    let oom = OOM_MESSAGE.lock().ok().and_then(|mut m| m.take());
    match (result, oom) {
        (Err(HypnosError::BackendError(_)), Some(detail)) => Err(HypnosError::DeviceOutOfMemory {
            model: model_id.to_string(),
            detail,
        }),
        (result, _) => result,
    }
}

fn run_image_job_inner(loaded: &LoadedSd, job: &ImageJob) -> Result<Vec<Vec<u8>>> {
    let prompt = CString::new(job.prompt.as_bytes())
        .map_err(|_| HypnosError::InvalidRequest("prompt contains NUL byte".into()))?;
    let negative = CString::new(job.negative_prompt.as_bytes())
        .map_err(|_| HypnosError::InvalidRequest("negative_prompt contains NUL byte".into()))?;

    let mut pngs = Vec::with_capacity(job.count as usize);
    unsafe {
        let mut g: sd::sd_img_gen_params_t = std::mem::zeroed();
        sd::sd_img_gen_params_init(&mut g);
        g.prompt = prompt.as_ptr();
        g.negative_prompt = negative.as_ptr();
        g.width = job.width as i32;
        g.height = job.height as i32;
        g.seed = job.seed;
        g.batch_count = job.count as i32;
        g.sample_params.sample_steps = job.steps as i32;
        g.sample_params.guidance.txt_cfg = job.cfg_scale;
        if let Some(name) = job.sample_method.as_deref() {
            let c = CString::new(name)
                .map_err(|_| HypnosError::Config("sample_method contains a NUL byte".into()))?;
            let method = sd::str_to_sample_method(c.as_ptr());
            if method == sd::sample_method_t::SAMPLE_METHOD_COUNT {
                return Err(HypnosError::Config(format!(
                    "[models.image] sample_method {name:?} is not a stable-diffusion.cpp sampler"
                )));
            }
            g.sample_params.sample_method = method;
        }
        if let Some(shift) = job.flow_shift {
            g.sample_params.flow_shift = shift;
        }
        // sd.cpp reads the schedule during the call; `job` owns it until it returns.
        let mut sigmas = job.sigmas.clone().unwrap_or_default();
        if !sigmas.is_empty() {
            g.sample_params.custom_sigmas = sigmas.as_mut_ptr();
            g.sample_params.custom_sigmas_count = sigmas.len() as i32;
            g.sample_params.sample_steps = sigmas.len().saturating_sub(1).max(1) as i32;
        }
        // sd.cpp reads the pixels during the call; `job` owns them until it returns.
        let mut refs: Vec<sd::sd_image_t> = job
            .references
            .iter()
            .map(|img| sd::sd_image_t {
                width: img.width(),
                height: img.height(),
                channel: 3,
                data: img.as_raw().as_ptr() as *mut u8,
            })
            .collect();
        if !refs.is_empty() {
            g.ref_images = refs.as_mut_ptr();
            g.ref_images_count = refs.len() as i32;
        }

        let mut result: *mut sd::sd_image_generation_result_t = std::ptr::null_mut();
        let ok = sd::generate_image(loaded.ctx, &g, &mut result);
        if !ok || result.is_null() {
            return Err(HypnosError::BackendError(
                "stable-diffusion.cpp generate_image failed (see sd.cpp log lines above)".into(),
            ));
        }
        let res = &*result;
        for i in 0..res.count.max(0) as usize {
            let item = &*res.items.add(i);
            let img = &item.image;
            if img.data.is_null() || img.width == 0 || img.height == 0 {
                continue;
            }
            let len = (img.width * img.height * img.channel) as usize;
            let pixels = std::slice::from_raw_parts(img.data, len);
            match encode_png(img.width, img.height, img.channel, pixels) {
                Ok(png) => pngs.push(png),
                Err(e) => {
                    sd::free_sd_image_generation_result(result);
                    return Err(e);
                }
            }
        }
        sd::free_sd_image_generation_result(result);
    }
    if pngs.is_empty() {
        return Err(HypnosError::BackendError(
            "stable-diffusion.cpp returned no images".into(),
        ));
    }
    Ok(pngs)
}

pub struct GgmlDiffusionBackend {
    config: ModelConfig,
    state: RwLock<BackendContextState>,
    loaded: Arc<StdMutex<Option<LoadedSd>>>,
    n_threads: i32,
}

impl GgmlDiffusionBackend {
    pub fn new(config: ModelConfig) -> Self {
        let state = RwLock::new(BackendContextState::new(&config));
        Self {
            config,
            state,
            loaded: Arc::new(StdMutex::new(None)),
            n_threads: default_threads(),
        }
    }
}

#[async_trait]
impl ModelBackend for GgmlDiffusionBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.loaded.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        {
            let mut guard = self.state.write();
            guard.active_plan = plan.clone();
            guard.pinned_host_ptr = pinned_host_ptr;
        }
        let want_resident = plan.primary_tier == MemoryTier::Tier1GpuVram;
        let cfg = self.config.clone();
        let plan = plan.clone();
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("sd mutex poisoned".into()))?;
            if !want_resident {
                if guard.take().is_some() {
                    info!(model = %cfg.id, to_tier = ?plan.primary_tier, "Released stable-diffusion.cpp context");
                }
                return Ok(());
            }
            match guard.as_ref() {
                Some(r) if r.gpu_layers == plan.gpu_layers => Ok(()),
                _ => {
                    *guard = None;
                    *guard = Some(load_sd(&cfg, plan.gpu_layers, n_threads)?);
                    Ok(())
                }
            }
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("load task panicked: {e}")))?
    }

    async fn generate_image(
        &self,
        req: &ImageGenerationRequest,
        ctx: &RequestContext,
    ) -> Result<ImageGenerationResponse> {
        if req.prompt.trim().is_empty() {
            return Err(HypnosError::InvalidRequest("prompt is empty".into()));
        }
        if req.prompt.chars().count() > ctx.limits.max_prompt_chars {
            return Err(HypnosError::InvalidRequest(
                "prompt exceeds limits.max_prompt_chars".into(),
            ));
        }
        let max_n = ctx.limits.max_images_per_request.max(1);
        let mut references = Vec::with_capacity(req.reference_images.len());
        if !req.reference_images.is_empty() {
            if !self.config.image.edit {
                return Err(HypnosError::InvalidRequest(format!(
                    "'{}' does not edit images; use a model with [models.image] edit = true (FLUX.2 klein)",
                    self.config.id
                )));
            }
            if req.mask.is_some() {
                return Err(HypnosError::InvalidRequest(format!(
                    "'{}' edits from the prompt and reference images; masks are not supported",
                    self.config.id
                )));
            }
            for (i, bytes) in req.reference_images.iter().enumerate() {
                let img = image::load_from_memory(bytes).map_err(|e| {
                    HypnosError::InvalidRequest(format!(
                        "image {} is not a PNG, JPEG or WebP: {e}",
                        i + 1
                    ))
                })?;
                references.push(img.to_rgb8());
            }
        }
        let (width, height) = parse_image_size(&req.size)?;
        let seed = req
            .seed
            .map(|s| (s % (i64::MAX as u64)) as i64)
            .unwrap_or_else(|| rand::random::<u32>() as i64);
        let job = ImageJob {
            prompt: req.prompt.clone(),
            negative_prompt: req.negative_prompt.clone().unwrap_or_default(),
            width,
            height,
            steps: req
                .steps
                .or(self.config.image.steps)
                .unwrap_or(DEFAULT_STEPS)
                .clamp(1, 100),
            cfg_scale: req
                .cfg_scale
                .or(self.config.image.cfg_scale)
                .unwrap_or(DEFAULT_CFG_SCALE),
            sample_method: self.config.image.sample_method.clone(),
            flow_shift: self.config.image.flow_shift,
            references,
            sigmas: self.config.image.sigmas.clone(),
            seed,
            count: req.n.clamp(1, max_n),
        };
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let ctx = ctx.clone();
        let pngs = tokio::task::spawn_blocking(move || -> Result<Vec<Vec<u8>>> {
            let guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("sd mutex poisoned".into()))?;
            let resident = guard.as_ref().ok_or_else(|| {
                HypnosError::BackendError(format!("model '{}' is not resident in VRAM", cfg.id))
            })?;
            let started = Instant::now();
            let out = run_image_job(&cfg.id, resident, &job, &ctx)?;
            info!(model = %cfg.id, images = out.len(), width = job.width, height = job.height, steps = job.steps, seed = job.seed, secs = format!("{:.1}", started.elapsed().as_secs_f32()), "image generation complete");
            Ok(out)
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("image task panicked: {e}")))??;

        // sd.cpp seeds image `i` of a batch with `seed + i` (`seed_for_batch`).
        let data = pngs
            .into_iter()
            .enumerate()
            .map(|(i, png)| GeneratedImage {
                b64_json: Some(base64::engine::general_purpose::STANDARD.encode(png)),
                revised_prompt: None,
                seed: Some((seed as u64).wrapping_add(i as u64) & i64::MAX as u64),
            })
            .collect();
        Ok(ImageGenerationResponse {
            created: Utc::now().timestamp(),
            data,
            hypnos_telemetry: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::is_out_of_memory;

    #[test]
    fn recognises_device_allocation_failures() {
        assert!(is_out_of_memory("model manager cannot make enough memory available on CUDA0: need 518.00 MB device / 518.00 MB budget, available 0.00 MB device / unlimited budget"));
        assert!(is_out_of_memory(
            "ggml_gallocr_reserve_n: failed to allocate CUDA0 buffer of size 543162368"
        ));
        assert!(is_out_of_memory("alloc_tensor_range: allocating 512.00 MiB on device 0: cudaMalloc failed: out of memory"));
        assert!(is_out_of_memory("ggml_backend_metal_buffer_type_alloc_buffer: failed to allocate Metal buffer of 536870912 bytes (out of memory)"));
        assert!(!is_out_of_memory(
            "model manager memory on CUDA0: reported free 0.00 MB / total 24463.00 MB"
        ));
        assert!(!is_out_of_memory("sampling completed, taking 12.31s"));
    }
}
