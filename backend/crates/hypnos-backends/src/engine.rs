//! llama.cpp engine worker: one thread per resident model that owns a persistent `LlamaContext`
//! and serves chat, transcription and embedding jobs through execution slots.
//!
//! A slot is a llama.cpp sequence id plus a description of what its KV cache holds
//! ([`CacheKey`]s: text tokens, and media chunks by id). When a new prompt arrives the worker
//! picks the slot whose cache shares the longest prefix with it, drops the divergent tail
//! (`llama_memory_seq_rm`) and prefills only the remainder. A multi-turn chat therefore pays for
//! the new turn only, and an image that is already in the cache is never run through the vision
//! tower again. With `slots = 1` jobs are strictly serialised, as before; the slot table is
//! written so that work item C can keep several slots active per decode step.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use hypnos_core::config::{KvDataType, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::openai::{ChatMessage, ChatStreamEvent, EngineTiming};
use hypnos_core::request::RequestContext;
use llama_cpp_2::context::params::{KvCacheType, LlamaContextParams, LlamaPoolingType};
use llama_cpp_2::context::session::{LlamaStateSeqFlags, SeqState};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::mtmd::{
    mtmd_default_marker, MtmdBitmap, MtmdContext, MtmdInputChunkType, MtmdInputChunks,
    MtmdInputText,
};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::llama::{build_prompt, llama_backend, load_model};
use crate::worker::{FailJob, WorkerHandle, WorkerInbox, WorkerMsg};
use crate::EngineOptions;

// ----------------------------------------------------------------------------
// Jobs
// ----------------------------------------------------------------------------

pub(crate) struct GenParams {
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub seed: u32,
}

pub(crate) struct GenStats {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    /// Prompt positions served from the slot's KV cache.
    pub cached_tokens: u32,
    pub finish_reason: &'static str,
    /// Waiting in the worker queue, measured from `started`.
    pub queue_ms: f64,
    /// Context (re)creation attributed to this request; 0 while the resident context is reused.
    pub context_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// `started` to the first visible text piece; `None` when nothing visible was produced.
    pub ttft_ms: Option<f64>,
}

impl GenStats {
    pub(crate) fn timing(&self) -> EngineTiming {
        let per_s =
            |tokens: u32, ms: f64| (tokens > 0 && ms > 0.0).then(|| tokens as f64 / (ms / 1000.0));
        EngineTiming {
            queue_ms: Some(self.queue_ms),
            context_ms: Some(self.context_ms),
            prefill_ms: Some(self.prefill_ms),
            decode_ms: Some(self.decode_ms),
            ttft_ms: self.ttft_ms,
            // Only the positions that were actually evaluated count towards prefill throughput.
            prefill_tokens_per_s: per_s(
                self.prompt_tokens.saturating_sub(self.cached_tokens),
                self.prefill_ms,
            ),
            // The first token is sampled from the prefill logits; only the rest cost a decode step.
            decode_tokens_per_s: per_s(self.completion_tokens.saturating_sub(1), self.decode_ms),
            cached_tokens: Some(self.cached_tokens),
            ..EngineTiming::default()
        }
    }
}

pub(crate) enum PromptSpec {
    /// Rendered by the worker with the model's template (needs the model for non-Gemma GGUFs).
    Messages {
        messages: Vec<ChatMessage>,
        think: bool,
    },
    /// Already rendered (transcription builds its own single-turn prompt).
    Raw(String),
}

pub(crate) struct ChatJob {
    pub prompt: PromptSpec,
    pub media: Vec<Vec<u8>>,
    pub params: GenParams,
    pub rctx: RequestContext,
    pub started: Instant,
    pub stream: Option<mpsc::Sender<ChatStreamEvent>>,
    pub reply: oneshot::Sender<Result<(String, GenStats)>>,
}

pub(crate) struct EmbedJob {
    pub inputs: Vec<String>,
    pub rctx: RequestContext,
    pub started: Instant,
    pub reply: oneshot::Sender<Result<(Vec<Vec<f32>>, u32, EngineTiming)>>,
}

pub(crate) enum LlamaJob {
    Chat(ChatJob),
    Embed(EmbedJob),
    /// Rebuild the context for a new per-slot window (the scheduler re-planned the residency).
    Reconfigure {
        n_ctx: u32,
        reply: oneshot::Sender<Result<()>>,
    },
}

impl FailJob for LlamaJob {
    fn fail(self, error: HypnosError) {
        match self {
            LlamaJob::Chat(job) => {
                let _ = job.reply.send(Err(error));
            }
            LlamaJob::Embed(job) => {
                let _ = job.reply.send(Err(error));
            }
            LlamaJob::Reconfigure { reply, .. } => {
                let _ = reply.send(Err(error));
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Slots and prefix reuse (pure, unit-tested)
// ----------------------------------------------------------------------------

/// One unit of what a slot's KV cache holds, in sequence order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CacheKey {
    Token(LlamaToken),
    /// An image or audio chunk, identified by the hash of its bytes (see [`media_id`]).
    Media {
        id: String,
        n_pos: i32,
    },
}

impl CacheKey {
    fn n_pos(&self) -> i32 {
        match self {
            CacheKey::Token(_) => 1,
            CacheKey::Media { n_pos, .. } => *n_pos,
        }
    }
}

/// Where a slot's sliding-window state was snapshotted: at the end of the last prompt, before
/// generation started. Restoring it lets a later prompt that diverges inside the generated reply
/// (re-tokenisation differences, trimmed whitespace, a swallowed thinking channel) continue from
/// the prompt end instead of starting over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CheckpointAt {
    pub keys: usize,
    pub positions: i32,
}

#[derive(Debug)]
pub(crate) struct Slot {
    pub seq_id: i32,
    /// What the KV cache of `seq_id` currently holds.
    pub keys: Vec<CacheKey>,
    /// Positions consumed in `seq_id` (equals the key count unless a media chunk spans several).
    pub n_past: i32,
    pub last_used: Instant,
    /// Position of the saved sliding-window snapshot, if any (metadata only, for planning).
    pub checkpoint: Option<CheckpointAt>,
    /// The snapshot itself (sliding-window layers only; the full-attention layers keep every
    /// position and need no restore).
    pub checkpoint_state: Option<SeqState>,
}

impl Slot {
    /// Forgets everything the slot holds (the KV cache itself is cleared by the caller).
    pub(crate) fn forget(&mut self) {
        self.keys.clear();
        self.n_past = 0;
        self.checkpoint = None;
        self.checkpoint_state = None;
    }
}

/// A prefix of a prompt that a slot already holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Reuse {
    pub keys: usize,
    pub positions: i32,
    /// The sliding-window layers must be restored from the slot's checkpoint first.
    pub restore_checkpoint: bool,
}

/// How a slot's KV cache is prepared for a new prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CachePlan {
    /// Drop everything the slot holds and evaluate the whole prompt from position 0.
    Clear,
    /// Keep the first `keys` / `positions`, drop whatever follows, evaluate the remainder.
    Keep(Reuse),
}

/// Longest common prefix of two key sequences, in keys and in positions.
pub(crate) fn common_prefix(a: &[CacheKey], b: &[CacheKey]) -> Reuse {
    let mut keys = 0;
    let mut positions = 0;
    for (x, y) in a.iter().zip(b) {
        if x != y {
            break;
        }
        keys += 1;
        positions += x.n_pos();
    }
    Reuse {
        keys,
        positions,
        restore_checkpoint: false,
    }
}

/// Decides what of `slot` survives for `prompt`.
///
/// * The logits of the prompt's last token are needed for sampling, so at least one token is
///   always re-evaluated.
/// * The remainder after the reused prefix must be plain text: media chunks are evaluated by
///   libmtmd as a whole prompt, so a prompt whose new part contains an image starts over.
/// * With sliding-window attention (`swa`) the local layers only hold the last window of the
///   sequence, so a state at an earlier position cannot be reconstructed by trimming. A pure
///   extension of what the slot holds needs no trimming; a divergence at or after the slot's
///   checkpoint (the end of the previous prompt) is served by restoring that checkpoint; anything
///   earlier is a miss.
pub(crate) fn plan_cache(
    slot: &Slot,
    prompt: &[CacheKey],
    prefix_cache: bool,
    swa: Option<u32>,
) -> CachePlan {
    if slot.keys.is_empty() || !prefix_cache || prompt.is_empty() {
        return CachePlan::Clear;
    }
    let mut reuse = common_prefix(&slot.keys, prompt);
    if reuse.keys >= prompt.len() {
        reuse = Reuse {
            keys: prompt.len() - 1,
            positions: prompt[..prompt.len() - 1].iter().map(CacheKey::n_pos).sum(),
            restore_checkpoint: false,
        };
    }
    if reuse.keys == 0 {
        return CachePlan::Clear;
    }
    if swa.is_some() && reuse.positions < slot.n_past {
        match slot.checkpoint {
            Some(ck)
                if ck.keys >= 1 && ck.keys <= reuse.keys && ck.positions <= reuse.positions =>
            {
                reuse = Reuse {
                    keys: ck.keys,
                    positions: ck.positions,
                    restore_checkpoint: true,
                };
            }
            _ => return CachePlan::Clear,
        }
    }
    if prompt[reuse.keys..]
        .iter()
        .any(|k| matches!(k, CacheKey::Media { .. }))
    {
        return CachePlan::Clear;
    }
    CachePlan::Keep(reuse)
}

#[derive(Debug)]
pub(crate) struct SlotTable {
    pub slots: Vec<Slot>,
    prefix_cache: bool,
    swa: Option<u32>,
}

impl SlotTable {
    pub(crate) fn new(n: u32, prefix_cache: bool, swa: Option<u32>) -> Self {
        let now = Instant::now();
        Self {
            slots: (0..n.max(1) as i32)
                .map(|seq_id| Slot {
                    seq_id,
                    keys: Vec::new(),
                    n_past: 0,
                    last_used: now,
                    checkpoint: None,
                    checkpoint_state: None,
                })
                .collect(),
            prefix_cache,
            swa,
        }
    }

    /// Best slot for `prompt`: the longest reusable prefix, then the least recently used.
    /// `allow_reuse = false` (M-RoPE models, where text positions are not scalar) always clears.
    pub(crate) fn select(&self, prompt: &[CacheKey], allow_reuse: bool) -> (usize, CachePlan) {
        let mut best: Option<(usize, CachePlan, i32, Instant)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            let plan = plan_cache(slot, prompt, self.prefix_cache && allow_reuse, self.swa);
            let reused = match plan {
                CachePlan::Keep(r) => r.positions,
                CachePlan::Clear => 0,
            };
            let better = match &best {
                None => true,
                Some((_, _, best_reused, best_used)) => {
                    reused > *best_reused || (reused == *best_reused && slot.last_used < *best_used)
                }
            };
            if better {
                best = Some((i, plan, reused, slot.last_used));
            }
        }
        let (i, plan, _, _) = best.expect("slot table is never empty");
        (i, plan)
    }

    /// Forgets every cache (after the context was rebuilt).
    pub(crate) fn reset(&mut self) {
        for s in &mut self.slots {
            s.forget();
        }
    }

    /// Slots holding a cache that has been idle for longer than `timeout`.
    pub(crate) fn expired(&self, timeout: Duration, now: Instant) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| !s.keys.is_empty() && now.duration_since(s.last_used) >= timeout)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Stable id for a media payload: the hash of its bytes plus the projector it is encoded with,
/// so a cached image chunk is only matched by the same picture through the same encoder.
pub(crate) fn media_id(cfg: &ModelConfig, bytes: &[u8]) -> String {
    let mut h = DefaultHasher::new();
    cfg.aux_weights_path.hash(&mut h);
    bytes.hash(&mut h);
    format!("{:016x}", h.finish())
}

// ----------------------------------------------------------------------------
// Worker
// ----------------------------------------------------------------------------

pub(crate) struct WorkerSpec {
    pub cfg: ModelConfig,
    pub gpu_layers: u32,
    /// Context window per slot.
    pub n_ctx: u32,
    pub n_threads: i32,
    pub opts: EngineOptions,
}

/// Starts the worker thread, loads the weights on it and waits until the context exists.
pub(crate) fn spawn_worker(spec: WorkerSpec) -> Result<WorkerHandle<LlamaJob>> {
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let name = format!("llama-{}", spec.cfg.id);
    let handle = WorkerHandle::spawn(&name, move |inbox| run_worker(spec, inbox, ready_tx))?;
    match ready_rx.recv() {
        Ok(Ok(())) => Ok(handle),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(HypnosError::BackendError(
            "engine worker exited before it was ready".into(),
        )),
    }
}

fn run_worker(spec: WorkerSpec, inbox: WorkerInbox<LlamaJob>, ready: SyncSender<Result<()>>) {
    let WorkerSpec {
        cfg,
        gpu_layers,
        n_ctx,
        n_threads,
        opts,
    } = spec;
    let mut loaded = match load_model(&cfg, gpu_layers, n_ctx, n_threads) {
        Ok(l) => l,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let backend = match llama_backend() {
        Ok(b) => b,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let arch = loaded.arch.clone();
    let model: &LlamaModel = &loaded.model;
    let mtmd: &mut Option<MtmdContext> = &mut loaded.mtmd;
    let mut engine = match Engine::new(
        backend,
        model,
        arch,
        mtmd.is_some(),
        cfg,
        gpu_layers,
        n_ctx,
        n_threads,
    ) {
        Ok(e) => e,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(()));

    let tick = opts
        .kv_idle_timeout
        .min(Duration::from_secs(5))
        .max(Duration::from_millis(100));
    loop {
        match inbox.next(tick) {
            None => {
                let now = Instant::now();
                for i in engine.slots.expired(opts.kv_idle_timeout, now) {
                    debug!(model = %engine.cfg.id, slot = i, "clearing idle slot KV cache");
                    engine.clear_slot(i);
                }
            }
            Some(WorkerMsg::Shutdown) => break,
            Some(WorkerMsg::Job(LlamaJob::Reconfigure { n_ctx, reply })) => {
                let _ = reply.send(engine.reconfigure(n_ctx));
            }
            Some(WorkerMsg::Job(LlamaJob::Chat(job))) => engine.chat(mtmd, job),
            Some(WorkerMsg::Job(LlamaJob::Embed(job))) => engine.embed(job),
        }
    }
    inbox.drain_pending("model was released from VRAM while the request was queued");
    // `engine` (the context) drops before `loaded` (projector, then weights).
}

struct Tokenized {
    keys: Vec<CacheKey>,
    n_tokens: usize,
    /// Present for multimodal prompts; evaluated by libmtmd when the whole prompt is prefilled.
    chunks: Option<(MtmdInputChunks, Vec<MtmdBitmap>)>,
    /// M-RoPE models position text in several dimensions; the scalar prefix bookkeeping does
    /// not apply, so reuse is disabled for them.
    mrope: bool,
}

struct Engine<'m> {
    backend: &'static LlamaBackend,
    model: &'m LlamaModel,
    arch: String,
    cfg: ModelConfig,
    gpu_layers: u32,
    n_threads: i32,
    has_mtmd: bool,
    embeddings_mode: bool,
    ctx: LlamaContext<'m>,
    batch: LlamaBatch<'static>,
    /// Context window per slot.
    n_ctx: u32,
    n_batch: u32,
    slots: SlotTable,
}

fn kv_types(dtype: KvDataType) -> (KvCacheType, KvCacheType) {
    match dtype {
        KvDataType::F16 | KvDataType::Bf16 => (KvCacheType::F16, KvCacheType::F16),
        KvDataType::Q8_0 => (KvCacheType::Q8_0, KvCacheType::Q8_0),
        KvDataType::Q4_0 => (KvCacheType::Q4_0, KvCacheType::Q4_0),
    }
}

fn backend_err<E: std::fmt::Debug>(what: &str) -> impl FnOnce(E) -> HypnosError + '_ {
    move |e| HypnosError::BackendError(format!("{what}: {e:?}"))
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// Single-token special markers such as `<|turn>`, `<|think|>`, `<|image|>`, `<channel|>`.
fn is_special_marker(piece: &[u8]) -> bool {
    let is_open = piece.starts_with(b"<|") && piece.ends_with(b">");
    let is_close = piece.starts_with(b"<") && piece.ends_with(b"|>");
    (is_open || is_close) && piece.len() <= 24 && !piece.contains(&b' ')
}

fn l2_normalize(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter().map(|x| x / norm).collect()
}

/// Decodes `tokens` into `seq` starting at `pos`, `n_batch` at a time, requesting logits for the
/// last token only. Returns the next position.
fn decode_tokens(
    ctx: &mut LlamaContext,
    batch: &mut LlamaBatch,
    tokens: &[LlamaToken],
    mut pos: i32,
    seq: i32,
    n_batch: usize,
) -> Result<i32> {
    let last = tokens.len().saturating_sub(1);
    for (ci, chunk) in tokens.chunks(n_batch.max(1)).enumerate() {
        batch.clear();
        for (i, t) in chunk.iter().enumerate() {
            batch
                .add(*t, pos + i as i32, &[seq], ci * n_batch + i == last)
                .map_err(backend_err("batch add failed"))?;
        }
        ctx.decode(batch)
            .map_err(backend_err("llama.cpp prefill decode failed"))?;
        pos += chunk.len() as i32;
    }
    Ok(pos)
}

impl<'m> Engine<'m> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        backend: &'static LlamaBackend,
        model: &'m LlamaModel,
        arch: String,
        has_mtmd: bool,
        cfg: ModelConfig,
        gpu_layers: u32,
        n_ctx: u32,
        n_threads: i32,
    ) -> Result<Self> {
        let embeddings_mode = cfg.modality == Modality::Embeddings;
        let slots = if embeddings_mode { 1 } else { cfg.slots.max(1) };
        let (ctx, n_ctx, n_batch) = Self::create_context(
            backend,
            model,
            &cfg,
            has_mtmd,
            gpu_layers,
            n_ctx,
            n_threads,
            embeddings_mode,
            slots,
        )?;
        let batch = LlamaBatch::new(n_batch as usize, 1);
        let swa = if embeddings_mode {
            None
        } else {
            cfg.arch.sliding_window_len
        };
        info!(
            model = %cfg.id,
            slots,
            n_ctx_per_slot = n_ctx,
            n_batch,
            prefix_cache = cfg.prefix_cache && !embeddings_mode,
            "llama.cpp engine worker ready"
        );
        Ok(Self {
            backend,
            model,
            arch,
            slots: SlotTable::new(slots, cfg.prefix_cache, swa),
            cfg,
            gpu_layers,
            n_threads,
            has_mtmd,
            embeddings_mode,
            ctx,
            batch,
            n_ctx,
            n_batch,
        })
    }

    /// Builds the context. Returns it with the per-slot window and batch size actually used.
    #[allow(clippy::too_many_arguments)]
    fn create_context(
        backend: &'static LlamaBackend,
        model: &'m LlamaModel,
        cfg: &ModelConfig,
        has_mtmd: bool,
        gpu_layers: u32,
        n_ctx: u32,
        n_threads: i32,
        embeddings_mode: bool,
        slots: u32,
    ) -> Result<(LlamaContext<'m>, u32, u32)> {
        let n_ctx = n_ctx
            .clamp(512, cfg.arch.max_context_len.max(512))
            .min(model.n_ctx_train().max(512));
        let (type_k, type_v) = kv_types(cfg.arch.kv_dtype);
        let (params, n_batch) = if embeddings_mode {
            // Non-causal encoders need the whole sequence inside one micro-batch.
            (
                LlamaContextParams::default()
                    .with_n_ctx(NonZeroU32::new(n_ctx))
                    .with_n_batch(n_ctx)
                    .with_n_ubatch(n_ctx)
                    .with_n_threads(n_threads)
                    .with_n_threads_batch(n_threads)
                    .with_embeddings(true)
                    .with_pooling_type(LlamaPoolingType::Unspecified),
                n_ctx,
            )
        } else {
            let n_batch = n_ctx.min(2048);
            // Image embeddings are decoded by mtmd as ONE non-causal batch (676 tokens per Gemma 4
            // image slice) and llama.cpp asserts `n_ubatch >= n_tokens` for non-causal attention,
            // so a multimodal context gets the whole batch as its micro-batch. Text-only models
            // keep the smaller micro-batch (smaller compute buffer).
            let n_ubatch = if has_mtmd { n_batch } else { n_batch.min(512) };
            (
                LlamaContextParams::default()
                    // The KV cache is split into `slots` streams of `n_ctx` positions each.
                    .with_n_ctx(NonZeroU32::new(n_ctx.saturating_mul(slots)))
                    .with_n_seq_max(slots)
                    .with_n_batch(n_batch)
                    .with_n_ubatch(n_ubatch)
                    .with_n_threads(n_threads)
                    .with_n_threads_batch(n_threads)
                    .with_offload_kqv(gpu_layers > 0)
                    .with_swa_full(false)
                    .with_type_k(type_k)
                    .with_type_v(type_v),
                n_batch,
            )
        };
        let ctx = model
            .new_context(backend, params)
            .map_err(backend_err("llama.cpp context creation failed"))?;
        Ok((ctx, n_ctx, n_batch))
    }

    fn reconfigure(&mut self, n_ctx: u32) -> Result<()> {
        let started = Instant::now();
        let slots = self.slots.slots.len() as u32;
        let (ctx, n_ctx, n_batch) = Self::create_context(
            self.backend,
            self.model,
            &self.cfg,
            self.has_mtmd,
            self.gpu_layers,
            n_ctx,
            self.n_threads,
            self.embeddings_mode,
            slots,
        )?;
        self.ctx = ctx;
        self.n_ctx = n_ctx;
        self.n_batch = n_batch;
        self.batch = LlamaBatch::new(n_batch as usize, 1);
        self.slots.reset();
        info!(
            model = %self.cfg.id,
            n_ctx_per_slot = n_ctx,
            ms = format!("{:.0}", ms_since(started)),
            "llama.cpp context rebuilt for a new window; prefix caches dropped"
        );
        Ok(())
    }

    fn clear_slot(&mut self, i: usize) {
        let slot = &mut self.slots.slots[i];
        if !slot.keys.is_empty() {
            if let Err(e) = self
                .ctx
                .clear_kv_cache_seq(Some(slot.seq_id as u32), None, None)
            {
                warn!(model = %self.cfg.id, slot = i, error = ?e, "could not clear slot KV cache");
            }
        }
        slot.forget();
    }

    // ---- chat ---------------------------------------------------------------------------

    fn chat(&mut self, mtmd: &mut Option<MtmdContext>, job: ChatJob) {
        let ChatJob {
            prompt,
            media,
            params,
            rctx,
            started,
            stream,
            reply,
        } = job;
        let outcome = self.run_chat(
            mtmd,
            prompt,
            &media,
            &params,
            &rctx,
            started,
            stream.as_ref(),
        );
        let _ = reply.send(outcome);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chat(
        &mut self,
        mtmd: &mut Option<MtmdContext>,
        prompt: PromptSpec,
        media: &[Vec<u8>],
        params: &GenParams,
        rctx: &RequestContext,
        started: Instant,
        stream: Option<&mpsc::Sender<ChatStreamEvent>>,
    ) -> Result<(String, GenStats)> {
        if !rctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }
        if self.embeddings_mode {
            return Err(HypnosError::ModalityMismatch {
                model: self.cfg.id.clone(),
                expected: self.cfg.modality,
                actual: Modality::TextGeneration,
            });
        }
        let queue_ms = ms_since(started);
        let phase = Instant::now();
        let text = match prompt {
            PromptSpec::Messages { messages, think } => build_prompt(
                self.model,
                &self.arch,
                &messages,
                mtmd_default_marker(),
                think,
            ),
            PromptSpec::Raw(s) => s,
        };
        debug!(model = %self.cfg.id, prompt_chars = text.len(), media = media.len(), "chat prefill");
        let tokenized = self.tokenize(mtmd, &text, media)?;
        if tokenized.n_tokens + 8 > self.n_ctx as usize {
            return Err(HypnosError::InvalidRequest(format!(
                "prompt{} needs {} tokens but the context window is {}; raise num_ctx",
                if media.is_empty() { "" } else { " + media" },
                tokenized.n_tokens,
                self.n_ctx
            )));
        }
        let (slot_idx, plan) = self.slots.select(&tokenized.keys, !tokenized.mrope);
        {
            let slot = &self.slots.slots[slot_idx];
            let common = common_prefix(&slot.keys, &tokenized.keys);
            debug!(
                model = %self.cfg.id,
                slot = slot_idx,
                cached_keys = slot.keys.len(),
                cached_positions = slot.n_past,
                prompt_keys = tokenized.keys.len(),
                common_keys = common.keys,
                common_positions = common.positions,
                plan = ?plan,
                "prefix cache decision"
            );
        }
        let result = self.generate_in_slot(
            mtmd, slot_idx, plan, tokenized, params, rctx, started, phase, queue_ms, stream,
        );
        if result.is_err() {
            // A failed prefill or decode leaves the sequence in an unknown state.
            self.clear_slot(slot_idx);
        }
        self.slots.slots[slot_idx].last_used = Instant::now();
        result
    }

    fn tokenize(
        &self,
        mtmd: &mut Option<MtmdContext>,
        prompt: &str,
        media: &[Vec<u8>],
    ) -> Result<Tokenized> {
        if media.is_empty() {
            let tokens = self.model.vocab().tokenize(prompt.as_bytes(), true, true);
            return Ok(Tokenized {
                n_tokens: tokens.len(),
                keys: tokens.into_iter().map(CacheKey::Token).collect(),
                chunks: None,
                mrope: false,
            });
        }
        let m = mtmd.as_mut().ok_or_else(|| {
            HypnosError::InvalidRequest(format!(
                "model '{}' has no multimodal projector configured; it cannot take images or audio",
                self.cfg.id
            ))
        })?;
        let mut bitmaps = Vec::with_capacity(media.len());
        for bytes in media {
            let mut bm = MtmdBitmap::from_buffer(m, bytes, false).map_err(|e| {
                HypnosError::InvalidRequest(format!(
                    "could not decode media attachment ({} bytes): {e:?}. Supported audio containers are WAV, MP3 and FLAC (browser recordings in WebM/Opus or M4A must be converted first); images may be PNG, JPEG, BMP or GIF",
                    bytes.len()
                ))
            })?;
            if bm.is_audio() && !m.support_audio() {
                return Err(HypnosError::ModalityMismatch {
                    model: self.cfg.id.clone(),
                    expected: self.cfg.modality,
                    actual: Modality::SpeechToText,
                });
            }
            if !bm.is_audio() && !m.support_vision() {
                return Err(HypnosError::ModalityMismatch {
                    model: self.cfg.id.clone(),
                    expected: self.cfg.modality,
                    actual: Modality::Vision,
                });
            }
            let _ = bm.set_id(&media_id(&self.cfg, bytes));
            bitmaps.push(bm);
        }
        let refs: Vec<&MtmdBitmap> = bitmaps.iter().collect();
        let chunks = m
            .tokenize(
                MtmdInputText {
                    text: prompt.to_string(),
                    add_special: true,
                    parse_special: true,
                },
                &refs,
            )
            .map_err(backend_err("libmtmd tokenization failed"))?;
        let mut keys = Vec::new();
        for i in 0..chunks.len() {
            let Some(chunk) = chunks.get(i) else { break };
            match chunk.chunk_type() {
                MtmdInputChunkType::Text => {
                    keys.extend(
                        chunk
                            .text_tokens()
                            .unwrap_or(&[])
                            .iter()
                            .map(|t| CacheKey::Token(*t)),
                    );
                }
                MtmdInputChunkType::Image | MtmdInputChunkType::Audio => {
                    keys.push(CacheKey::Media {
                        id: chunk.id().unwrap_or_else(|| format!("chunk-{i}")),
                        n_pos: chunk.n_positions(),
                    })
                }
            }
        }
        Ok(Tokenized {
            keys,
            n_tokens: chunks.total_tokens(),
            mrope: m.decode_use_mrope(),
            chunks: Some((chunks, bitmaps)),
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_in_slot(
        &mut self,
        mtmd: &mut Option<MtmdContext>,
        slot_idx: usize,
        plan: CachePlan,
        tokenized: Tokenized,
        params: &GenParams,
        rctx: &RequestContext,
        started: Instant,
        phase: Instant,
        queue_ms: f64,
        stream: Option<&mpsc::Sender<ChatStreamEvent>>,
    ) -> Result<(String, GenStats)> {
        let Engine {
            ctx,
            batch,
            slots,
            model,
            n_batch,
            n_ctx,
            cfg,
            ..
        } = self;
        let n_batch = *n_batch as usize;
        let n_ctx = *n_ctx;
        let slot = &mut slots.slots[slot_idx];
        let seq = slot.seq_id;
        let vocab = model.vocab();

        // ---- prepare the slot's cache -------------------------------------------------------
        match plan {
            CachePlan::Clear => {
                if !slot.keys.is_empty() {
                    ctx.clear_kv_cache_seq(Some(seq as u32), None, None)
                        .map_err(backend_err("KV cache clear failed"))?;
                }
                slot.forget();
            }
            CachePlan::Keep(reuse) => {
                if reuse.restore_checkpoint {
                    let state = slot.checkpoint_state.as_ref().ok_or_else(|| {
                        HypnosError::BackendError("slot checkpoint missing".into())
                    })?;
                    ctx.state_seq_set(state, seq)
                        .map_err(backend_err("KV checkpoint restore failed"))?;
                }
                if reuse.positions < slot.n_past {
                    ctx.kv_cache_seq_rm(seq, Some(reuse.positions as u32), None)
                        .map_err(backend_err("KV cache trim failed"))?;
                }
                slot.keys.truncate(reuse.keys);
                slot.n_past = reuse.positions;
                if slot
                    .checkpoint
                    .is_some_and(|ck| ck.positions > reuse.positions)
                {
                    slot.checkpoint = None;
                    slot.checkpoint_state = None;
                }
            }
        }
        let cached_tokens = slot.n_past.max(0) as u32;

        // ---- prefill -------------------------------------------------------------------------
        let Tokenized {
            keys,
            n_tokens,
            chunks,
            ..
        } = tokenized;
        let mut n_past = match plan {
            CachePlan::Clear => match &chunks {
                Some((chunks, _bitmaps)) => {
                    let m = mtmd
                        .as_mut()
                        .expect("multimodal prompt tokenized without a projector");
                    chunks
                        .eval_chunks(m, ctx, 0, seq, n_batch as i32, true)
                        .map_err(backend_err("libmtmd prefill failed"))?
                }
                None => {
                    let tokens: Vec<LlamaToken> = keys
                        .iter()
                        .filter_map(|k| match k {
                            CacheKey::Token(t) => Some(*t),
                            CacheKey::Media { .. } => None,
                        })
                        .collect();
                    decode_tokens(ctx, batch, &tokens, 0, seq, n_batch)?
                }
            },
            CachePlan::Keep(reuse) => {
                // `plan_cache` guarantees the remainder is text.
                let suffix: Vec<LlamaToken> = keys[reuse.keys..]
                    .iter()
                    .filter_map(|k| match k {
                        CacheKey::Token(t) => Some(*t),
                        CacheKey::Media { .. } => None,
                    })
                    .collect();
                decode_tokens(ctx, batch, &suffix, slot.n_past, seq, n_batch)?
            }
        };
        drop(chunks);
        slot.keys = keys;
        slot.n_past = n_past;
        // Sliding-window layers forget old positions as generation proceeds; snapshot them at the
        // prompt end so the next turn can resume from here even if it diverges inside the reply.
        if cfg.prefix_cache && cfg.arch.sliding_window_len.is_some() {
            match ctx.state_seq_get(seq, LlamaStateSeqFlags::PARTIAL_ONLY) {
                Ok(state) => {
                    slot.checkpoint = Some(CheckpointAt {
                        keys: slot.keys.len(),
                        positions: n_past,
                    });
                    slot.checkpoint_state = Some(state);
                }
                Err(e) => {
                    warn!(model = %cfg.id, slot = slot_idx, error = ?e, "could not checkpoint slot state");
                    slot.checkpoint = None;
                    slot.checkpoint_state = None;
                }
            }
        }
        let prefill_ms = ms_since(phase);
        let phase = Instant::now();
        let mut ttft_ms: Option<f64> = None;

        // ---- sampling ------------------------------------------------------------------------
        let mut sampler = if params.temperature <= 0.0 {
            LlamaSampler::greedy()
        } else {
            LlamaSampler::chain_simple([
                LlamaSampler::top_k(params.top_k.max(1)),
                LlamaSampler::top_p(params.top_p.clamp(0.0, 1.0), 1),
                LlamaSampler::temp(params.temperature),
                LlamaSampler::dist(params.seed),
            ])
        };
        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut text = String::new();
        let mut completion: u32 = 0;
        let mut finish_reason = "length";
        let max_tokens = params.max_tokens.max(1);
        // Gemma 4 wraps reasoning in `<|channel>thought ... <channel|>`; even with thinking
        // disabled the model may emit an empty channel. Everything inside it is swallowed, as are
        // bare special tokens.
        let mut in_channel = false;
        let mut channel_header_pending = false;

        while completion < max_tokens {
            if !rctx.should_continue() {
                finish_reason = "cancelled";
                break;
            }
            let token = sampler.sample(ctx, -1);
            sampler.accept(token);
            if vocab.is_eog(token) {
                finish_reason = "stop";
                break;
            }
            let piece = vocab.token_to_piece(token, true, None);
            if piece.as_slice() == b"<turn|>" || piece.as_slice() == b"<end_of_turn>" {
                finish_reason = "stop";
                break;
            }
            completion += 1;
            let is_marker = is_special_marker(&piece);
            if piece.as_slice() == b"<|channel>" {
                in_channel = true;
                channel_header_pending = true;
            } else if piece.as_slice() == b"<channel|>" {
                in_channel = false;
            } else if !vocab.is_control(token) && !is_marker && !in_channel {
                let mut out = String::with_capacity(
                    decoder
                        .max_utf8_buffer_length(piece.len())
                        .unwrap_or(piece.len() * 3),
                );
                let _ = decoder.decode_to_string(&piece, &mut out, false);
                // Right after a channel closes the model emits the leading newline(s) of the answer.
                if channel_header_pending {
                    channel_header_pending = false;
                    let trimmed = out.trim_start_matches(['\n', '\r']).to_string();
                    if trimmed.is_empty() {
                        out.clear();
                    } else {
                        out = trimmed;
                    }
                }
                if !out.is_empty() {
                    ttft_ms.get_or_insert_with(|| ms_since(started));
                    text.push_str(&out);
                    if let Some(tx) = stream {
                        if tx.blocking_send(ChatStreamEvent::Delta(out)).is_err() {
                            finish_reason = "stop";
                            break;
                        }
                    }
                }
            }
            if (n_past as u32).saturating_add(1) >= n_ctx {
                finish_reason = "length";
                break;
            }
            batch.clear();
            batch
                .add(token, n_past, &[seq], true)
                .map_err(backend_err("batch add failed"))?;
            ctx.decode(batch)
                .map_err(backend_err("llama.cpp decode failed"))?;
            n_past += 1;
            slot.keys.push(CacheKey::Token(token));
            slot.n_past = n_past;
        }

        let decode_ms = ms_since(phase);
        debug!(
            model = %cfg.id,
            slot = slot_idx,
            prompt_tokens = n_tokens,
            cached_tokens,
            completion_tokens = completion,
            queue_ms = format!("{queue_ms:.0}"),
            prefill_ms = format!("{prefill_ms:.0}"),
            decode_ms = format!("{decode_ms:.0}"),
            ttft_ms = ttft_ms.map(|v| format!("{v:.0}")),
            "generation timings"
        );
        Ok((
            text,
            GenStats {
                prompt_tokens: n_tokens as u32,
                completion_tokens: completion,
                cached_tokens,
                finish_reason,
                queue_ms,
                context_ms: 0.0,
                prefill_ms,
                decode_ms,
                ttft_ms,
            },
        ))
    }

    // ---- embeddings ---------------------------------------------------------------------

    fn embed(&mut self, job: EmbedJob) {
        let EmbedJob {
            inputs,
            rctx,
            started,
            reply,
        } = job;
        let _ = reply.send(self.run_embed(&inputs, &rctx, started));
    }

    fn run_embed(
        &mut self,
        inputs: &[String],
        rctx: &RequestContext,
        started: Instant,
    ) -> Result<(Vec<Vec<f32>>, u32, EngineTiming)> {
        if !self.embeddings_mode {
            return Err(HypnosError::ModalityMismatch {
                model: self.cfg.id.clone(),
                expected: self.cfg.modality,
                actual: Modality::Embeddings,
            });
        }
        let queue_ms = ms_since(started);
        let phase = Instant::now();
        let vocab = self.model.vocab();
        let n_ctx = self.n_ctx as usize;
        let mut out = Vec::with_capacity(inputs.len());
        let mut total: u32 = 0;
        for text in inputs {
            if !rctx.should_continue() {
                return Err(HypnosError::Cancelled);
            }
            let mut tokens = vocab.tokenize(text.as_bytes(), true, true);
            if tokens.is_empty() {
                tokens = vocab.tokenize(b" ", true, true);
            }
            tokens.truncate(n_ctx);
            total += tokens.len() as u32;
            self.batch.clear();
            self.batch
                .add_sequence(&tokens, 0, true)
                .map_err(backend_err("batch add failed"))?;
            self.ctx.clear_kv_cache();
            self.ctx
                .decode(&mut self.batch)
                .map_err(backend_err("llama.cpp embedding decode failed"))?;
            let emb: Vec<f32> = match self.ctx.embeddings_seq_ith(0) {
                Ok(e) => e.to_vec(),
                Err(_) => self
                    .ctx
                    .embeddings_ith(tokens.len() as i32 - 1)
                    .map_err(backend_err("no embeddings produced"))?
                    .to_vec(),
            };
            out.push(l2_normalize(&emb));
        }
        let prefill_ms = ms_since(phase);
        let timing = EngineTiming {
            queue_ms: Some(queue_ms),
            context_ms: Some(0.0),
            prefill_ms: Some(prefill_ms),
            prefill_tokens_per_s: (total > 0 && prefill_ms > 0.0)
                .then(|| total as f64 / (prefill_ms / 1000.0)),
            ..EngineTiming::default()
        };
        Ok((out, total, timing))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(ids: &[i32]) -> Vec<CacheKey> {
        ids.iter()
            .map(|i| CacheKey::Token(LlamaToken(*i)))
            .collect()
    }

    fn media(id: &str, n_pos: i32) -> CacheKey {
        CacheKey::Media {
            id: id.to_string(),
            n_pos,
        }
    }

    fn slot_with(keys: Vec<CacheKey>) -> Slot {
        let n_past = keys.iter().map(CacheKey::n_pos).sum();
        Slot {
            seq_id: 0,
            keys,
            n_past,
            last_used: Instant::now(),
            checkpoint: None,
            checkpoint_state: None,
        }
    }

    fn reuse(keys: usize, positions: i32) -> Reuse {
        Reuse {
            keys,
            positions,
            restore_checkpoint: false,
        }
    }

    #[test]
    fn common_prefix_counts_keys_and_positions() {
        let a = vec![
            CacheKey::Token(LlamaToken(1)),
            media("img", 676),
            CacheKey::Token(LlamaToken(2)),
        ];
        let mut b = a.clone();
        b.push(CacheKey::Token(LlamaToken(3)));
        assert_eq!(common_prefix(&a, &b), reuse(3, 678));
        let c = vec![CacheKey::Token(LlamaToken(1)), media("other", 676)];
        assert_eq!(common_prefix(&a, &c), reuse(1, 1));
    }

    #[test]
    fn extension_of_the_cached_sequence_is_reused() {
        // Cache: prompt + the reply the model generated. New prompt: the same plus a new turn.
        let slot = slot_with(tok(&[1, 2, 3, 4, 5]));
        let prompt = tok(&[1, 2, 3, 4, 5, 6, 7]);
        let plan = plan_cache(&slot, &prompt, true, Some(1024));
        assert_eq!(plan, CachePlan::Keep(reuse(5, 5)));
    }

    #[test]
    fn identical_prompt_re_evaluates_its_last_token() {
        let slot = slot_with(tok(&[1, 2, 3]));
        let prompt = tok(&[1, 2, 3]);
        assert_eq!(
            plan_cache(&slot, &prompt, true, None),
            CachePlan::Keep(reuse(2, 2))
        );
        // The same request on a sliding-window model without a checkpoint is a miss.
        assert_eq!(
            plan_cache(&slot, &prompt, true, Some(512)),
            CachePlan::Clear
        );
    }

    #[test]
    fn divergence_in_the_middle_trims_without_swa_and_clears_with_swa() {
        let slot = slot_with(tok(&[1, 2, 3, 4]));
        let prompt = tok(&[1, 2, 9, 9]);
        assert_eq!(
            plan_cache(&slot, &prompt, true, None),
            CachePlan::Keep(reuse(2, 2))
        );
        assert_eq!(
            plan_cache(&slot, &prompt, true, Some(1024)),
            CachePlan::Clear
        );
    }

    #[test]
    fn divergence_inside_the_reply_restores_the_checkpoint_with_swa() {
        // Prompt 1..=4 was checkpointed before the model generated 5, 6.
        let mut slot = slot_with(tok(&[1, 2, 3, 4, 5, 6]));
        slot.checkpoint = Some(CheckpointAt {
            keys: 4,
            positions: 4,
        });
        // The client re-sends the reply tokenised differently (9, 9) plus a new turn (7).
        let diverging = tok(&[1, 2, 3, 4, 9, 9, 7]);
        assert_eq!(
            plan_cache(&slot, &diverging, true, Some(1024)),
            CachePlan::Keep(Reuse {
                keys: 4,
                positions: 4,
                restore_checkpoint: true
            })
        );
        // Divergence after the checkpoint still resumes from the checkpoint, not from the match.
        let later = tok(&[1, 2, 3, 4, 5, 9, 7]);
        assert_eq!(
            plan_cache(&slot, &later, true, Some(1024)),
            CachePlan::Keep(Reuse {
                keys: 4,
                positions: 4,
                restore_checkpoint: true
            })
        );
        // Divergence before the checkpoint cannot be served.
        let early = tok(&[1, 2, 8, 8]);
        assert_eq!(
            plan_cache(&slot, &early, true, Some(1024)),
            CachePlan::Clear
        );
        // A pure extension never needs the checkpoint.
        let extension = tok(&[1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(
            plan_cache(&slot, &extension, true, Some(1024)),
            CachePlan::Keep(reuse(6, 6))
        );
    }

    #[test]
    fn cached_image_is_reused_but_a_new_image_starts_over() {
        let slot = slot_with(vec![
            CacheKey::Token(LlamaToken(1)),
            media("img", 676),
            CacheKey::Token(LlamaToken(2)),
            CacheKey::Token(LlamaToken(3)),
        ]);
        let mut next_turn = slot.keys.clone();
        next_turn.extend(tok(&[4, 5]));
        assert_eq!(
            plan_cache(&slot, &next_turn, true, Some(1024)),
            CachePlan::Keep(reuse(4, 679))
        );
        let mut with_new_image = slot.keys.clone();
        with_new_image.push(media("img2", 676));
        with_new_image.push(CacheKey::Token(LlamaToken(4)));
        assert_eq!(
            plan_cache(&slot, &with_new_image, true, None),
            CachePlan::Clear
        );
    }

    #[test]
    fn disabled_cache_empty_slot_and_no_overlap_clear() {
        let slot = slot_with(tok(&[1, 2, 3]));
        assert_eq!(
            plan_cache(&slot, &tok(&[1, 2, 3, 4]), false, None),
            CachePlan::Clear
        );
        assert_eq!(
            plan_cache(&slot_with(vec![]), &tok(&[1]), true, None),
            CachePlan::Clear
        );
        assert_eq!(
            plan_cache(&slot, &tok(&[7, 8]), true, None),
            CachePlan::Clear
        );
    }

    #[test]
    fn select_prefers_longest_prefix_then_least_recently_used() {
        let mut table = SlotTable::new(3, true, None);
        table.slots[0].keys = tok(&[1, 2]);
        table.slots[0].n_past = 2;
        table.slots[1].keys = tok(&[1, 2, 3, 4]);
        table.slots[1].n_past = 4;
        let (i, plan) = table.select(&tok(&[1, 2, 3, 4, 5]), true);
        assert_eq!(i, 1);
        assert_eq!(plan, CachePlan::Keep(reuse(4, 4)));
        // No overlap anywhere: the least recently used slot wins (slot 2 was never used, but all
        // three share the creation instant, so make slot 0 clearly older).
        table.slots[0].last_used = Instant::now() - Duration::from_secs(60);
        let (i, plan) = table.select(&tok(&[9]), true);
        assert_eq!((i, plan), (0, CachePlan::Clear));
        // M-RoPE models never reuse.
        let (_, plan) = table.select(&tok(&[1, 2, 3, 4, 5]), false);
        assert_eq!(plan, CachePlan::Clear);
    }

    #[test]
    fn idle_slots_expire() {
        let mut table = SlotTable::new(2, true, None);
        table.slots[0].keys = tok(&[1]);
        table.slots[0].last_used = Instant::now() - Duration::from_secs(10);
        table.slots[1].keys = tok(&[1]);
        let expired = table.expired(Duration::from_secs(5), Instant::now());
        assert_eq!(expired, vec![0]);
        table.reset();
        assert!(table
            .slots
            .iter()
            .all(|s| s.keys.is_empty() && s.n_past == 0));
    }

    #[test]
    fn media_ids_depend_on_bytes_and_projector() {
        let mut cfg = hypnos_core::config::ModelConfig {
            id: "m".into(),
            modality: Modality::TextGeneration,
            additional_modalities: vec![],
            engine: hypnos_core::config::EngineKind::GgmlLlama,
            weights_path: "w.gguf".into(),
            aux_weights_path: Some("proj-a.gguf".into()),
            text_encoder_path: None,
            tokenizer_path: None,
            voices_path: None,
            total_weight_bytes: 0,
            min_gpu_layer_ratio: 1.0,
            initial_tier: hypnos_core::config::MemoryTier::Tier3NvmeSsd,
            keep_warm_in_ram: false,
            priority: Default::default(),
            slots: 1,
            prefix_cache: true,
            arch: Default::default(),
        };
        let a = media_id(&cfg, b"png-bytes");
        assert_eq!(a, media_id(&cfg, b"png-bytes"));
        assert_ne!(a, media_id(&cfg, b"other-bytes"));
        cfg.aux_weights_path = Some("proj-b.gguf".into());
        assert_ne!(a, media_id(&cfg, b"png-bytes"));
    }
}
