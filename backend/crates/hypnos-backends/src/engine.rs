//! llama.cpp engine worker: one thread per resident model that owns a persistent `LlamaContext`
//! and serves chat, transcription and embedding jobs through execution slots.
//!
//! A slot is a llama.cpp sequence id plus a description of what its KV cache holds
//! ([`CacheKey`]s: text tokens, and media chunks by id). When a new prompt arrives the worker
//! picks the free slot whose cache shares the longest prefix with it, drops the divergent tail
//! (`llama_memory_seq_rm`) and prefills only the remainder. A multi-turn chat therefore pays for
//! the new turn only, and an image that is already in the cache is never run through the vision
//! tower again.
//!
//! Requests in different slots run together: every step the worker builds one `llama_batch`
//! holding the next token of each decoding slot plus up to the remaining batch budget of prompt
//! tokens from slots still prefilling, decodes it once, then samples each slot from its own
//! logits. A long prompt is therefore chunked across steps and never stalls the slots that are
//! already generating. With `slots = 1` this degenerates to one request at a time.

use std::collections::hash_map::DefaultHasher;
use std::collections::VecDeque;
use std::hash::{Hash, Hasher};
use std::num::NonZeroU32;
use std::sync::mpsc::SyncSender;
use std::time::{Duration, Instant};

use hypnos_core::config::{KvDataType, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::openai::{ChatMessage, ChatStreamEvent, EngineTiming, LoadPhases, ToolSpec};
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
use llama_cpp_2::DecodeError;
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

use crate::llama::{build_prompt, llama_backend, load_model, supports_tools};
use crate::tools::{PASS_THROUGH, RESPONSE_OPEN};
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
    /// Waiting in the worker queue (and for a free slot), measured from `started`.
    pub queue_ms: f64,
    /// Context (re)creation attributed to this request; 0 while the resident context is reused.
    pub context_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
    /// `started` to the first visible text piece; `None` when nothing visible was produced.
    pub ttft_ms: Option<f64>,
    /// Most slots active at once while this request ran.
    pub concurrent_slots: u32,
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
            concurrent_slots: Some(self.concurrent_slots),
            ..EngineTiming::default()
        }
    }
}

pub(crate) enum PromptSpec {
    /// Rendered by the worker with the model's template (needs the model for non-Gemma GGUFs).
    Messages {
        messages: Vec<ChatMessage>,
        think: bool,
        /// Declared in the prompt (Gemma 4 only; see `tools.rs`).
        tools: Vec<ToolSpec>,
        /// Appended after the generation prompt (a forced tool call's opening).
        prefill: String,
    },
    /// Already rendered (transcription builds its own single-turn prompt).
    Raw(String),
}

pub(crate) struct ChatJob {
    pub prompt: PromptSpec,
    /// Pass the tool-call markers through as text and stop at `<|tool_response>`.
    pub tool_markers: bool,
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

/// A slot's sequence state saved when its model left Tier 1 (MEM-4, `kv_park.rs`).
pub(crate) struct ParkedSlot {
    pub index: usize,
    pub keys: Vec<CacheKey>,
    pub n_past: i32,
    /// `None` only in tests (a state cannot be made without a context).
    pub state: Option<SeqState>,
    pub bytes: u64,
    pub checkpoint: Option<CheckpointAt>,
    pub checkpoint_state: Option<SeqState>,
}

#[cfg(test)]
impl ParkedSlot {
    pub(crate) fn for_test(bytes: usize, tokens: i32) -> Self {
        Self {
            index: 0,
            keys: Vec::new(),
            n_past: tokens,
            state: None,
            bytes: bytes as u64,
            checkpoint: None,
            checkpoint_state: None,
        }
    }
}

/// What a context must match for parked states to be restored into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ParkSignature {
    pub weights: String,
    pub n_ctx: u32,
    pub slots: usize,
    pub kv: String,
}

pub(crate) struct ParkedContext {
    pub signature: ParkSignature,
    pub slots: Vec<ParkedSlot>,
}

impl ParkedContext {
    pub(crate) fn bytes(&self) -> u64 {
        self.slots.iter().map(|s| s.bytes).sum()
    }

    pub(crate) fn tokens(&self) -> u64 {
        self.slots.iter().map(|s| s.n_past.max(0) as u64).sum()
    }
}

pub(crate) enum LlamaJob {
    Chat(ChatJob),
    Embed(EmbedJob),
    /// Save every used, idle slot's state before the worker is dropped (MEM-4).
    Park {
        reply: oneshot::Sender<Option<ParkedContext>>,
    },
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
            LlamaJob::Park { reply } => {
                let _ = reply.send(None);
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
    /// A request is running in this slot.
    pub in_use: bool,
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
                    in_use: false,
                    checkpoint: None,
                    checkpoint_state: None,
                })
                .collect(),
            prefix_cache,
            swa,
        }
    }

    /// Best free slot for `prompt`: the longest reusable prefix, then the least recently used.
    /// `allow_reuse = false` (M-RoPE models, where text positions are not scalar) always clears.
    /// `None` when every slot is busy.
    pub(crate) fn select(
        &self,
        prompt: &[CacheKey],
        allow_reuse: bool,
    ) -> Option<(usize, CachePlan)> {
        let mut best: Option<(usize, CachePlan, i32, Instant)> = None;
        for (i, slot) in self.slots.iter().enumerate() {
            if slot.in_use {
                continue;
            }
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
        best.map(|(i, plan, _, _)| (i, plan))
    }

    pub(crate) fn has_free(&self) -> bool {
        self.slots.iter().any(|s| !s.in_use)
    }

    /// Forgets every cache (after the context was rebuilt).
    pub(crate) fn reset(&mut self) {
        for s in &mut self.slots {
            s.forget();
        }
    }

    /// Free slots holding a cache that has been idle for longer than `timeout`.
    pub(crate) fn expired(&self, timeout: Duration, now: Instant) -> Vec<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                !s.in_use && !s.keys.is_empty() && now.duration_since(s.last_used) >= timeout
            })
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
    /// Layers whose expert tensors stay in host memory (MEM-5).
    pub cpu_expert_layers: u32,
    /// Context window per slot.
    pub n_ctx: u32,
    pub n_threads: i32,
    pub opts: EngineOptions,
    /// Slot states to restore once the context exists (MEM-4).
    pub parked: Option<ParkedContext>,
}

/// Starts the worker thread, loads the weights on it and waits until the context exists.
/// Returns the handle and where the load spent its time.
pub(crate) fn spawn_worker(spec: WorkerSpec) -> Result<(WorkerHandle<LlamaJob>, LoadPhases)> {
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel(1);
    let name = format!("llama-{}", spec.cfg.id);
    let handle = WorkerHandle::spawn(&name, move |inbox| run_worker(spec, inbox, ready_tx))?;
    match ready_rx.recv() {
        Ok(Ok(phases)) => Ok((handle, phases)),
        Ok(Err(e)) => Err(e),
        Err(_) => Err(HypnosError::BackendError(
            "engine worker exited before it was ready".into(),
        )),
    }
}

fn run_worker(
    spec: WorkerSpec,
    inbox: WorkerInbox<LlamaJob>,
    ready: SyncSender<Result<LoadPhases>>,
) {
    let WorkerSpec {
        cfg,
        gpu_layers,
        cpu_expert_layers,
        n_ctx,
        n_threads,
        opts,
        parked,
    } = spec;
    let load_started = Instant::now();
    let mut loaded = match load_model(&cfg, gpu_layers, cpu_expert_layers, n_ctx, n_threads) {
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
    let mut phases = loaded.phases.clone();
    let context_started = Instant::now();
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
    phases.context_ms = context_started.elapsed().as_secs_f64() * 1000.0;
    if let Some(parked) = parked {
        let (ms, tokens) = engine.restore(parked);
        if tokens > 0 {
            phases.kv_restore_ms = Some(ms);
            phases.kv_restored_tokens = Some(tokens);
        }
    }
    phases.total_ms = load_started.elapsed().as_secs_f64() * 1000.0;
    let _ = ready.send(Ok(phases));

    let tick = opts
        .kv_idle_timeout
        .min(Duration::from_secs(5))
        .max(Duration::from_millis(100));
    'run: loop {
        if engine.active.is_empty() {
            match inbox.next(tick) {
                None => {
                    let now = Instant::now();
                    for i in engine.slots.expired(opts.kv_idle_timeout, now) {
                        debug!(model = %engine.cfg.id, slot = i, "clearing idle slot KV cache");
                        engine.clear_slot(i);
                    }
                }
                Some(WorkerMsg::Shutdown) => break 'run,
                Some(WorkerMsg::Job(job)) => engine.accept(mtmd, job),
            }
        } else {
            // Work in progress: pick up whatever arrived, then run one decode step.
            while let Some(msg) = inbox.try_next() {
                match msg {
                    WorkerMsg::Shutdown => break 'run,
                    WorkerMsg::Job(job) => engine.accept(mtmd, job),
                }
            }
            engine.step(mtmd);
        }
    }
    engine.abort_all(|| {
        HypnosError::BackendError(
            "model was released from VRAM while the request was running".into(),
        )
    });
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

/// A chat request executing in a slot.
struct Active {
    slot: usize,
    params: GenParams,
    rctx: RequestContext,
    started: Instant,
    admitted: Instant,
    stream: Option<mpsc::Sender<ChatStreamEvent>>,
    reply: Option<oneshot::Sender<Result<(String, GenStats)>>>,
    n_prompt: usize,
    cached_tokens: u32,
    /// Prompt tokens still to evaluate and how many of them have been decoded so far.
    pending: Vec<LlamaToken>,
    sent: usize,
    /// How many prompt tokens / which decode token went into the current batch.
    added: usize,
    decoding: Option<LlamaToken>,
    /// Set once the last prompt token has been decoded.
    prefill_done: Option<Instant>,
    /// The token to decode next (sampled in the previous step); `None` while prefilling.
    next_token: Option<LlamaToken>,
    /// Batch index whose logits belong to this request after the current decode.
    logits_idx: Option<i32>,
    sampler: LlamaSampler,
    decoder: encoding_rs::Decoder,
    text: String,
    completion: u32,
    in_channel: bool,
    channel_header_pending: bool,
    /// Tools are declared: tool-call markers are output, `<|tool_response>` ends the answer.
    tool_markers: bool,
    /// The SWA checkpoint was taken one position before the prompt end (see `step`).
    checkpointed: bool,
    ttft_ms: Option<f64>,
    /// Text the client has not accepted yet (its channel was full).
    unsent: String,
    finish: Option<&'static str>,
    concurrent_slots: u32,
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
    active: Vec<Active>,
    /// Chat jobs waiting for a free slot, in arrival order.
    waiting: VecDeque<ChatJob>,
    /// Embedding and reconfigure jobs that need an idle engine.
    deferred: VecDeque<LlamaJob>,
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

/// `llama_decode` returns -2 when ggml could not allocate the compute graph or the batch's
/// outputs: the device is full, which the server answers with 507 and a retry after evicting idle
/// models rather than a generic 500.
fn decode_err(model: &str, what: &str, e: &DecodeError) -> HypnosError {
    match e {
        DecodeError::Unknown(-2) => HypnosError::DeviceOutOfMemory {
            model: model.to_string(),
            detail: format!("{what}: could not allocate compute buffers ({e})"),
        },
        e => HypnosError::BackendError(format!("{what}: {e:?}")),
    }
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

fn make_sampler(params: &GenParams) -> LlamaSampler {
    if params.temperature <= 0.0 {
        LlamaSampler::greedy()
    } else {
        LlamaSampler::chain_simple([
            LlamaSampler::top_k(params.top_k.max(1)),
            LlamaSampler::top_p(params.top_p.clamp(0.0, 1.0), 1),
            LlamaSampler::temp(params.temperature),
            LlamaSampler::dist(params.seed),
        ])
    }
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
            active: Vec::new(),
            waiting: VecDeque::new(),
            deferred: VecDeque::new(),
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

    // ---- parked KV state (MEM-4) --------------------------------------------------------

    fn park_signature(&self) -> ParkSignature {
        ParkSignature {
            weights: self.cfg.weights_path.display().to_string(),
            n_ctx: self.n_ctx,
            slots: self.slots.slots.len(),
            kv: format!("{:?}", self.cfg.arch.kv_dtype),
        }
    }

    /// Copies out every slot that holds a conversation and is not running a request.
    fn park(&mut self) -> Option<ParkedContext> {
        if self.embeddings_mode || !self.cfg.prefix_cache {
            return None;
        }
        let started = Instant::now();
        let mut slots = Vec::new();
        for (index, slot) in self.slots.slots.iter().enumerate() {
            if slot.in_use || slot.n_past <= 0 || slot.keys.is_empty() {
                continue;
            }
            match self
                .ctx
                .state_seq_get(slot.seq_id, LlamaStateSeqFlags::empty())
            {
                Ok(state) => {
                    let bytes = state.byte_len() as u64
                        + slot
                            .checkpoint_state
                            .as_ref()
                            .map_or(0, |c| c.byte_len() as u64);
                    slots.push(ParkedSlot {
                        index,
                        keys: slot.keys.clone(),
                        n_past: slot.n_past,
                        state: Some(state),
                        bytes,
                        checkpoint: slot.checkpoint,
                        checkpoint_state: slot.checkpoint_state.clone(),
                    });
                }
                Err(e) => {
                    warn!(model = %self.cfg.id, slot = index, error = ?e, "could not save slot state")
                }
            }
        }
        if slots.is_empty() {
            return None;
        }
        let parked = ParkedContext {
            signature: self.park_signature(),
            slots,
        };
        info!(
            model = %self.cfg.id,
            slots = parked.slots.len(),
            tokens = parked.tokens(),
            bytes = parked.bytes(),
            ms = ms_since(started) as u64,
            "saved slot KV state for parking"
        );
        Some(parked)
    }

    /// Restores parked slots into this (new) context; returns the time taken and the positions
    /// restored. A different layout (window, slots, KV type, weights) restores nothing.
    fn restore(&mut self, parked: ParkedContext) -> (f64, u64) {
        let signature = self.park_signature();
        if parked.signature != signature {
            info!(model = %self.cfg.id, parked = ?parked.signature, now = ?signature, "parked KV state does not fit the new context; dropped");
            return (0.0, 0);
        }
        let started = Instant::now();
        let mut tokens = 0u64;
        for p in parked.slots {
            let Some(state) = p.state else { continue };
            let Some(seq_id) = self.slots.slots.get(p.index).map(|s| s.seq_id) else {
                continue;
            };
            match self.ctx.state_seq_set(&state, seq_id) {
                Ok(()) => {
                    let slot = &mut self.slots.slots[p.index];
                    slot.keys = p.keys;
                    slot.n_past = p.n_past;
                    slot.checkpoint = p.checkpoint;
                    slot.checkpoint_state = p.checkpoint_state;
                    slot.last_used = Instant::now();
                    tokens += p.n_past.max(0) as u64;
                }
                Err(e) => {
                    warn!(model = %self.cfg.id, slot = p.index, error = ?e, "could not restore slot state; it will prefill");
                    let _ = self.ctx.clear_kv_cache_seq(Some(seq_id as u32), None, None);
                    self.slots.slots[p.index].forget();
                }
            }
        }
        let ms = ms_since(started);
        if tokens > 0 {
            info!(model = %self.cfg.id, tokens, ms = ms as u64, "restored parked slot KV state");
        }
        (ms, tokens)
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

    // ---- job intake ---------------------------------------------------------------------

    fn accept(&mut self, mtmd: &mut Option<MtmdContext>, job: LlamaJob) {
        match job {
            LlamaJob::Chat(job) => {
                if self.embeddings_mode {
                    let _ = job.reply.send(Err(HypnosError::ModalityMismatch {
                        model: self.cfg.id.clone(),
                        expected: self.cfg.modality,
                        actual: Modality::TextGeneration,
                    }));
                } else if self.slots.has_free() {
                    self.admit(mtmd, job);
                } else {
                    self.waiting.push_back(job);
                }
            }
            other => {
                if self.active.is_empty() {
                    self.run_idle_job(other);
                } else {
                    self.deferred.push_back(other);
                }
            }
        }
    }

    /// Embedding and reconfigure jobs touch the whole context; they run between batches.
    fn run_idle_job(&mut self, job: LlamaJob) {
        match job {
            LlamaJob::Embed(job) => self.embed(job),
            LlamaJob::Reconfigure { n_ctx, reply } => {
                let _ = reply.send(self.reconfigure(n_ctx));
            }
            LlamaJob::Park { reply } => {
                let _ = reply.send(self.park());
            }
            LlamaJob::Chat(job) => self.waiting.push_back(job),
        }
    }

    /// Fills free slots from the waiting queue and runs deferred jobs once the engine is idle.
    fn drain_queues(&mut self, mtmd: &mut Option<MtmdContext>) {
        while self.slots.has_free() {
            let Some(job) = self.waiting.pop_front() else {
                break;
            };
            self.admit(mtmd, job);
        }
        if self.active.is_empty() {
            while let Some(job) = self.deferred.pop_front() {
                self.run_idle_job(job);
            }
        }
    }

    /// Fails everything in flight or queued (shutdown, a failed batch decode) with `err()`.
    fn abort_all(&mut self, err: impl Fn() -> HypnosError) {
        for mut a in std::mem::take(&mut self.active) {
            if let Some(reply) = a.reply.take() {
                let _ = reply.send(Err(err()));
            }
            self.slots.slots[a.slot].in_use = false;
            self.clear_slot(a.slot);
        }
        for job in self.waiting.drain(..) {
            LlamaJob::Chat(job).fail(err());
        }
        for job in self.deferred.drain(..) {
            job.fail(err());
        }
    }

    // ---- admission --------------------------------------------------------------------

    /// Tokenizes the prompt, prepares a slot's cache for it and starts the request. Multimodal
    /// prompts that cannot reuse the cache are evaluated here by libmtmd (one non-causal batch
    /// per image), everything else is prefilled through the shared batch in [`step`](Self::step).
    fn admit(&mut self, mtmd: &mut Option<MtmdContext>, job: ChatJob) {
        let ChatJob {
            prompt,
            tool_markers,
            media,
            params,
            rctx,
            started,
            stream,
            reply,
        } = job;
        if !rctx.should_continue() {
            let _ = reply.send(Err(HypnosError::Cancelled));
            return;
        }
        let admitted = Instant::now();
        let text = match prompt {
            PromptSpec::Messages {
                messages,
                think,
                tools,
                prefill,
            } => {
                if !tools.is_empty() && !supports_tools(&self.arch) {
                    let _ = reply.send(Err(HypnosError::InvalidRequest(format!(
                        "{} ({}) has no tool-calling format in Hypnos; tools work with Gemma 4 models",
                        self.cfg.id, self.arch
                    ))));
                    return;
                }
                build_prompt(
                    self.model,
                    &self.arch,
                    &messages,
                    mtmd_default_marker(),
                    think,
                    &tools,
                    &prefill,
                )
            }
            PromptSpec::Raw(s) => s,
        };
        debug!(model = %self.cfg.id, prompt_chars = text.len(), media = media.len(), "chat prefill");
        let tokenized = match self.tokenize(mtmd, &text, &media) {
            Ok(t) => t,
            Err(e) => {
                let _ = reply.send(Err(e));
                return;
            }
        };
        if tokenized.n_tokens + 8 > self.n_ctx as usize {
            let _ = reply.send(Err(HypnosError::InvalidRequest(format!(
                "prompt{} needs {} tokens but the context window is {}; raise num_ctx",
                if media.is_empty() { "" } else { " + media" },
                tokenized.n_tokens,
                self.n_ctx
            ))));
            return;
        }
        let Some((slot_idx, plan)) = self.slots.select(&tokenized.keys, !tokenized.mrope) else {
            // Raced with another admission; wait for the next free slot.
            self.waiting.push_front(ChatJob {
                prompt: PromptSpec::Raw(text),
                tool_markers,
                media,
                params,
                rctx,
                started,
                stream,
                reply,
            });
            return;
        };
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
        let mut active = Active {
            slot: slot_idx,
            sampler: make_sampler(&params),
            params,
            rctx,
            started,
            admitted,
            stream,
            reply: Some(reply),
            n_prompt: tokenized.n_tokens,
            cached_tokens: 0,
            pending: Vec::new(),
            sent: 0,
            added: 0,
            decoding: None,
            prefill_done: None,
            next_token: None,
            logits_idx: None,
            decoder: encoding_rs::UTF_8.new_decoder(),
            text: String::new(),
            completion: 0,
            in_channel: false,
            channel_header_pending: false,
            tool_markers,
            checkpointed: false,
            ttft_ms: None,
            unsent: String::new(),
            finish: None,
            concurrent_slots: 1,
        };
        self.slots.slots[slot_idx].in_use = true;
        match self.prepare_slot(mtmd, &mut active, plan, tokenized) {
            Ok(()) => {
                self.active.push(active);
            }
            Err(e) => {
                self.slots.slots[slot_idx].in_use = false;
                self.clear_slot(slot_idx);
                if let Some(reply) = active.reply.take() {
                    let _ = reply.send(Err(e));
                }
            }
        }
    }

    /// Applies the cache plan to the slot and either queues the prompt's text for the shared
    /// batch or (multimodal, from scratch) evaluates it right away.
    fn prepare_slot(
        &mut self,
        mtmd: &mut Option<MtmdContext>,
        a: &mut Active,
        plan: CachePlan,
        tokenized: Tokenized,
    ) -> Result<()> {
        let slot = &mut self.slots.slots[a.slot];
        let seq = slot.seq_id;
        match plan {
            CachePlan::Clear => {
                if !slot.keys.is_empty() {
                    self.ctx
                        .clear_kv_cache_seq(Some(seq as u32), None, None)
                        .map_err(backend_err("KV cache clear failed"))?;
                }
                slot.forget();
            }
            CachePlan::Keep(reuse) => {
                if reuse.restore_checkpoint {
                    let state = slot.checkpoint_state.as_ref().ok_or_else(|| {
                        HypnosError::BackendError("slot checkpoint missing".into())
                    })?;
                    self.ctx
                        .state_seq_set(state, seq)
                        .map_err(backend_err("KV checkpoint restore failed"))?;
                }
                if reuse.positions < slot.n_past {
                    self.ctx
                        .kv_cache_seq_rm(seq, Some(reuse.positions as u32), None)
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
        a.cached_tokens = slot.n_past.max(0) as u32;

        let Tokenized { keys, chunks, .. } = tokenized;
        let text_tokens = |ks: &[CacheKey]| -> Vec<LlamaToken> {
            ks.iter()
                .filter_map(|k| match k {
                    CacheKey::Token(t) => Some(*t),
                    CacheKey::Media { .. } => None,
                })
                .collect()
        };
        match (plan, &chunks) {
            (CachePlan::Clear, Some((chunks, _bitmaps))) => {
                // libmtmd runs its own decodes (non-causal image batches); do it now, alone.
                let m = mtmd
                    .as_mut()
                    .expect("multimodal prompt tokenized without a projector");
                let n_past = chunks
                    .eval_chunks(m, &mut self.ctx, 0, seq, self.n_batch as i32, true)
                    .map_err(backend_err("libmtmd prefill failed"))?;
                slot.keys = keys;
                slot.n_past = n_past;
                self.finish_prefill(a);
                // The logits of the last prompt token are only valid until the next decode.
                let token = a.sampler.sample(&self.ctx, -1);
                self.take_token(a, token);
            }
            (CachePlan::Clear, None) => {
                a.pending = text_tokens(&keys);
                slot.keys = keys;
            }
            (CachePlan::Keep(reuse), _) => {
                // `plan_cache` guarantees the remainder is text.
                a.pending = text_tokens(&keys[reuse.keys..]);
                slot.keys = keys;
            }
        }
        Ok(())
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

    // ---- one decode step over every active slot ---------------------------------------

    fn step(&mut self, mtmd: &mut Option<MtmdContext>) {
        let n_active = self.active.len() as u32;
        let n_ctx = self.n_ctx;
        let mut budget = self.n_batch as usize;
        self.batch.clear();

        // 1. Cancellations and length limits are checked before any work is queued.
        for a in &mut self.active {
            a.concurrent_slots = a.concurrent_slots.max(n_active);
            a.added = 0;
            a.decoding = None;
            a.logits_idx = None;
            if a.finish.is_none() && !a.rctx.should_continue() {
                a.finish = Some("cancelled");
            }
            if a.finish.is_none() {
                let n_past = self.slots.slots[a.slot].n_past as u32;
                if n_past.saturating_add(1) >= n_ctx {
                    a.finish = Some("length");
                }
            }
        }

        // 2. Decoding slots first (one token each), then prompt tokens up to the budget.
        for a in &mut self.active {
            if a.finish.is_some() {
                continue;
            }
            if let Some(token) = a.next_token.take() {
                let slot = &self.slots.slots[a.slot];
                if self
                    .batch
                    .add(token, slot.n_past, &[slot.seq_id], true)
                    .is_ok()
                {
                    a.logits_idx = Some(self.batch.n_tokens() - 1);
                    a.decoding = Some(token);
                    budget = budget.saturating_sub(1);
                } else {
                    a.next_token = Some(token);
                }
            }
        }
        // With sliding-window attention the checkpoint is taken one position before the prompt
        // end: a repeated prompt (Regenerate) must re-evaluate its last token for the logits, so
        // that is where a later request can resume. The prompt stops short of its last token
        // here, the checkpoint is taken, and the last token goes into the next step.
        let checkpoint_before_last =
            self.cfg.prefix_cache && self.cfg.arch.sliding_window_len.is_some();
        if checkpoint_before_last {
            for i in 0..self.active.len() {
                let a = &self.active[i];
                if a.finish.is_none()
                    && a.decoding.is_none()
                    && !a.checkpointed
                    && !a.pending.is_empty()
                    && a.sent + 1 == a.pending.len()
                {
                    let slot = a.slot;
                    self.checkpoint_slot(slot, 1);
                    self.active[i].checkpointed = true;
                }
            }
        }
        for a in &mut self.active {
            if a.finish.is_some() || a.decoding.is_some() || budget == 0 {
                continue;
            }
            let remaining = a.pending.len().saturating_sub(a.sent);
            if remaining == 0 {
                continue;
            }
            let mut take = remaining.min(budget);
            if checkpoint_before_last && !a.checkpointed && remaining > 1 {
                take = take.min(remaining - 1);
            }
            let slot = &self.slots.slots[a.slot];
            for i in 0..take {
                let last = a.sent + i + 1 == a.pending.len();
                if self
                    .batch
                    .add(
                        a.pending[a.sent + i],
                        slot.n_past + i as i32,
                        &[slot.seq_id],
                        last,
                    )
                    .is_err()
                {
                    break;
                }
                a.added += 1;
                if last {
                    a.logits_idx = Some(self.batch.n_tokens() - 1);
                }
            }
            budget -= a.added;
        }

        // 3. One decode for everyone.
        if self.batch.n_tokens() > 0 {
            if let Err(e) = self.ctx.decode(&mut self.batch) {
                let model = self.cfg.id.clone();
                warn!(model = %model, error = ?e, "batch decode failed; failing every active request");
                self.abort_all(|| decode_err(&model, "llama.cpp decode failed", &e));
                return;
            }
        }

        // 4. Bookkeeping, sampling and streaming per slot.
        let mut done = Vec::new();
        for i in 0..self.active.len() {
            let mut a = std::mem::replace(&mut self.active[i], Active::placeholder());
            let slot = &mut self.slots.slots[a.slot];
            if let Some(token) = a.decoding.take() {
                slot.n_past += 1;
                slot.keys.push(CacheKey::Token(token));
            } else if a.added > 0 {
                slot.n_past += a.added as i32;
                a.sent += a.added;
                if a.sent == a.pending.len() {
                    self.finish_prefill(&mut a);
                }
            }
            if let Some(idx) = a.logits_idx.take() {
                let token = a.sampler.sample(&self.ctx, idx);
                self.take_token(&mut a, token);
            }
            self.flush_stream(&mut a);
            if a.finish.is_some() {
                done.push(i);
            }
            self.active[i] = a;
        }
        for i in done.into_iter().rev() {
            let a = self.active.swap_remove(i);
            self.complete(a);
        }
        self.drain_queues(mtmd);
    }

    /// The prompt is fully in the KV cache: start timing the decode phase. A text prompt was
    /// checkpointed one position before its end (`step`); a multimodal one, evaluated by libmtmd
    /// in one go, is checkpointed here at its end.
    fn finish_prefill(&mut self, a: &mut Active) {
        if !a.checkpointed && self.cfg.prefix_cache && self.cfg.arch.sliding_window_len.is_some() {
            self.checkpoint_slot(a.slot, 0);
            a.checkpointed = true;
        }
        a.prefill_done = Some(Instant::now());
    }

    /// Snapshots the sliding-window state of slot `i`, whose KV cache holds its keys but the last
    /// `keys_ahead` (not evaluated yet). Sliding-window layers forget old positions as generation
    /// proceeds; the snapshot lets a later request resume from here even if it diverges inside
    /// the reply.
    fn checkpoint_slot(&mut self, i: usize, keys_ahead: usize) {
        let slot = &mut self.slots.slots[i];
        match self
            .ctx
            .state_seq_get(slot.seq_id, LlamaStateSeqFlags::PARTIAL_ONLY)
        {
            Ok(state) => {
                slot.checkpoint = Some(CheckpointAt {
                    keys: slot.keys.len().saturating_sub(keys_ahead),
                    positions: slot.n_past,
                });
                slot.checkpoint_state = Some(state);
            }
            Err(e) => {
                warn!(model = %self.cfg.id, slot = i, error = ?e, "could not checkpoint slot state");
                slot.checkpoint = None;
                slot.checkpoint_state = None;
            }
        }
    }

    /// Handles one sampled token: stop conditions, the Gemma thinking channel, text decoding.
    /// Sets `next_token` when generation continues.
    fn take_token(&mut self, a: &mut Active, token: LlamaToken) {
        let vocab = self.model.vocab();
        a.sampler.accept(token);
        if vocab.is_eog(token) {
            a.finish = Some("stop");
            return;
        }
        let piece = vocab.token_to_piece(token, true, None);
        if piece.as_slice() == b"<turn|>" || piece.as_slice() == b"<end_of_turn>" {
            a.finish = Some("stop");
            return;
        }
        a.completion += 1;
        if a.tool_markers && !a.in_channel {
            // The model waits for the tool's result here; the client runs it.
            if piece.as_slice() == RESPONSE_OPEN.as_bytes() {
                a.finish = Some("tool_calls");
                return;
            }
            if PASS_THROUGH
                .iter()
                .any(|m| piece.as_slice() == m.as_bytes())
            {
                let marker = String::from_utf8_lossy(&piece).into_owned();
                a.text.push_str(&marker);
                if a.stream.is_some() {
                    a.unsent.push_str(&marker);
                }
                if a.completion >= a.params.max_tokens.max(1) {
                    a.finish = Some("length");
                } else {
                    a.next_token = Some(token);
                }
                return;
            }
        }
        let is_marker = is_special_marker(&piece);
        if piece.as_slice() == b"<|channel>" {
            a.in_channel = true;
            a.channel_header_pending = true;
        } else if piece.as_slice() == b"<channel|>" {
            a.in_channel = false;
        } else if !vocab.is_control(token) && !is_marker && !a.in_channel {
            let mut out = String::with_capacity(
                a.decoder
                    .max_utf8_buffer_length(piece.len())
                    .unwrap_or(piece.len() * 3),
            );
            let _ = a.decoder.decode_to_string(&piece, &mut out, false);
            // Right after a channel closes the model emits the leading newline(s) of the answer.
            if a.channel_header_pending {
                a.channel_header_pending = false;
                let trimmed = out.trim_start_matches(['\n', '\r']).to_string();
                if trimmed.is_empty() {
                    out.clear();
                } else {
                    out = trimmed;
                }
            }
            if !out.is_empty() {
                a.ttft_ms.get_or_insert_with(|| ms_since(a.started));
                a.text.push_str(&out);
                if a.stream.is_some() {
                    a.unsent.push_str(&out);
                }
            }
        }
        if a.completion >= a.params.max_tokens.max(1) {
            a.finish = Some("length");
            return;
        }
        a.next_token = Some(token);
    }

    /// Pushes buffered text to the client without blocking the step: a full channel keeps the
    /// text for the next step, a closed one ends the request.
    fn flush_stream(&mut self, a: &mut Active) {
        let Some(tx) = &a.stream else {
            return;
        };
        if a.unsent.is_empty() {
            return;
        }
        let delta = std::mem::take(&mut a.unsent);
        match tx.try_send(ChatStreamEvent::Delta(delta)) {
            Ok(()) => {}
            Err(TrySendError::Full(ChatStreamEvent::Delta(delta))) => a.unsent = delta,
            Err(TrySendError::Full(_)) => {}
            Err(TrySendError::Closed(_)) => {
                a.unsent.clear();
                if a.finish.is_none() {
                    a.finish = Some("stop");
                }
            }
        }
    }

    /// Answers the request and frees its slot.
    fn complete(&mut self, mut a: Active) {
        if let (Some(tx), false) = (&a.stream, a.unsent.is_empty()) {
            // The client was slow; wait for it now that nothing else depends on this slot.
            let _ = tx.blocking_send(ChatStreamEvent::Delta(std::mem::take(&mut a.unsent)));
        }
        let slot = &mut self.slots.slots[a.slot];
        slot.in_use = false;
        slot.last_used = Instant::now();
        let prefill_done = a.prefill_done.unwrap_or_else(Instant::now);
        let stats = GenStats {
            prompt_tokens: a.n_prompt as u32,
            completion_tokens: a.completion,
            cached_tokens: a.cached_tokens,
            finish_reason: a.finish.unwrap_or("stop"),
            queue_ms: a.admitted.duration_since(a.started).as_secs_f64() * 1000.0,
            context_ms: 0.0,
            prefill_ms: prefill_done.duration_since(a.admitted).as_secs_f64() * 1000.0,
            decode_ms: ms_since(prefill_done),
            ttft_ms: a.ttft_ms,
            concurrent_slots: a.concurrent_slots,
        };
        debug!(
            model = %self.cfg.id,
            slot = a.slot,
            prompt_tokens = stats.prompt_tokens,
            cached_tokens = stats.cached_tokens,
            completion_tokens = stats.completion_tokens,
            concurrent_slots = stats.concurrent_slots,
            queue_ms = format!("{:.0}", stats.queue_ms),
            prefill_ms = format!("{:.0}", stats.prefill_ms),
            decode_ms = format!("{:.0}", stats.decode_ms),
            ttft_ms = stats.ttft_ms.map(|v| format!("{v:.0}")),
            "generation timings"
        );
        if let Some(reply) = a.reply.take() {
            let _ = reply.send(Ok((std::mem::take(&mut a.text), stats)));
        }
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
                .map_err(|e| decode_err(&self.cfg.id, "llama.cpp embedding decode failed", &e))?;
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

impl Active {
    /// A stand-in used while an entry is taken out of the active list for processing.
    fn placeholder() -> Self {
        Active {
            slot: usize::MAX,
            params: GenParams {
                max_tokens: 1,
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
                seed: 0,
            },
            rctx: RequestContext::unbounded(),
            started: Instant::now(),
            admitted: Instant::now(),
            stream: None,
            reply: None,
            n_prompt: 0,
            cached_tokens: 0,
            pending: Vec::new(),
            sent: 0,
            added: 0,
            decoding: None,
            prefill_done: None,
            next_token: None,
            logits_idx: None,
            sampler: LlamaSampler::greedy(),
            decoder: encoding_rs::UTF_8.new_decoder(),
            text: String::new(),
            completion: 0,
            in_channel: false,
            channel_header_pending: false,
            tool_markers: false,
            checkpointed: false,
            ttft_ms: None,
            unsent: String::new(),
            finish: Some("stop"),
            concurrent_slots: 0,
        }
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
            in_use: false,
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
    fn a_repeated_prompt_resumes_from_the_checkpoint_before_its_last_token() {
        // Prompt 1..=4 was checkpointed after 3 positions (before its last token), then the model
        // generated 5, 6. Regenerate sends 1..=4 again: only its last token is evaluated.
        let mut slot = slot_with(tok(&[1, 2, 3, 4, 5, 6]));
        slot.checkpoint = Some(CheckpointAt {
            keys: 3,
            positions: 3,
        });
        assert_eq!(
            plan_cache(&slot, &tok(&[1, 2, 3, 4]), true, Some(1024)),
            CachePlan::Keep(Reuse {
                keys: 3,
                positions: 3,
                restore_checkpoint: true
            })
        );
        // The next turn (the reply re-tokenised differently) resumes there too.
        assert_eq!(
            plan_cache(&slot, &tok(&[1, 2, 3, 4, 9, 7]), true, Some(1024)),
            CachePlan::Keep(Reuse {
                keys: 3,
                positions: 3,
                restore_checkpoint: true
            })
        );
        // A checkpoint at the prompt end (the old placement) could not serve the repeat.
        slot.checkpoint = Some(CheckpointAt {
            keys: 4,
            positions: 4,
        });
        assert_eq!(
            plan_cache(&slot, &tok(&[1, 2, 3, 4]), true, Some(1024)),
            CachePlan::Clear
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
    fn select_prefers_longest_prefix_then_least_recently_used_and_skips_busy_slots() {
        let mut table = SlotTable::new(3, true, None);
        table.slots[0].keys = tok(&[1, 2]);
        table.slots[0].n_past = 2;
        table.slots[1].keys = tok(&[1, 2, 3, 4]);
        table.slots[1].n_past = 4;
        let (i, plan) = table.select(&tok(&[1, 2, 3, 4, 5]), true).unwrap();
        assert_eq!(i, 1);
        assert_eq!(plan, CachePlan::Keep(reuse(4, 4)));
        // The best slot is busy: the next best wins.
        table.slots[1].in_use = true;
        let (i, plan) = table.select(&tok(&[1, 2, 3, 4, 5]), true).unwrap();
        assert_eq!((i, plan), (0, CachePlan::Keep(reuse(2, 2))));
        table.slots[1].in_use = false;
        // No overlap anywhere: the least recently used slot wins.
        table.slots[0].last_used = Instant::now() - Duration::from_secs(60);
        let (i, plan) = table.select(&tok(&[9]), true).unwrap();
        assert_eq!((i, plan), (0, CachePlan::Clear));
        // M-RoPE models never reuse.
        let (_, plan) = table.select(&tok(&[1, 2, 3, 4, 5]), false).unwrap();
        assert_eq!(plan, CachePlan::Clear);
        // Everything busy: nothing to select.
        for s in &mut table.slots {
            s.in_use = true;
        }
        assert!(table.select(&tok(&[1]), true).is_none());
        assert!(!table.has_free());
    }

    #[test]
    fn idle_slots_expire_unless_busy() {
        let mut table = SlotTable::new(3, true, None);
        table.slots[0].keys = tok(&[1]);
        table.slots[0].last_used = Instant::now() - Duration::from_secs(10);
        table.slots[1].keys = tok(&[1]);
        table.slots[2].keys = tok(&[1]);
        table.slots[2].last_used = Instant::now() - Duration::from_secs(10);
        table.slots[2].in_use = true;
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
            enabled: true,
            use_mmap: None,
            moe_offload: Default::default(),
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
            device: Default::default(),
            arch: Default::default(),
        };
        let a = media_id(&cfg, b"png-bytes");
        assert_eq!(a, media_id(&cfg, b"png-bytes"));
        assert_ne!(a, media_id(&cfg, b"other-bytes"));
        cfg.aux_weights_path = Some("proj-b.gguf".into());
        assert_ne!(a, media_id(&cfg, b"png-bytes"));
    }
}
