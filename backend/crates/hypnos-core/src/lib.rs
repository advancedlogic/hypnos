pub mod catalog;
pub mod config;
pub mod error;
pub mod gguf;
pub mod kv_cache;
pub mod openai;
pub mod request;

use crate::config::{Modality, ModelConfig};
use crate::error::Result;
use crate::kv_cache::LayerPlacementPlan;
use crate::openai::{
    AudioSpeechOutput, AudioSpeechRequest, AudioStreamEvent, AudioTranscriptionRequest,
    AudioTranscriptionResponse, ChatCompletionRequest, ChatCompletionResponse, ChatStreamEvent,
    DocumentConversionRequest, DocumentStreamEvent, EmbeddingRequest, EmbeddingResponse,
    EntityExtractionRequest, EntityExtractionResponse, ImageGenerationRequest,
    ImageGenerationResponse,
};
use crate::request::RequestContext;
use async_trait::async_trait;
use tokio::sync::mpsc;

/// Pluggable execution engine trait for any model modality.
///
/// Implementations own their native runtime context (`llama_model`/`llama_context` + `mtmd`,
/// `sd_ctx`, or an ONNX Runtime session) and load / unload real weights when the
/// `hypnos-scheduler` changes the model's tier placement.
///
/// Every inference method receives a [`RequestContext`]; engines must stop promptly once
/// `ctx.should_continue()` turns false and must honour `ctx.limits`.
#[async_trait]
pub trait ModelBackend: Send + Sync {
    /// Returns the static model configuration.
    fn config(&self) -> &ModelConfig;

    /// Returns the model's primary modality.
    fn modality(&self) -> Modality {
        self.config().modality
    }

    /// Reconfigures the model's active memory placement (Tier 1 GPU layers vs Tier 2 Pinned RAM)
    /// and initializes or resizes the KV cache context.
    ///
    /// Called by the scheduler whenever a model is promoted, partially compacted, or evicted.
    /// A Tier 1 plan loads the weights into the engine with `plan.gpu_layers` offloaded to the
    /// GPU; a Tier 2 / Tier 3 plan releases the engine's VRAM and host allocations.
    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        pinned_host_ptr: Option<usize>,
    ) -> Result<()>;

    /// `true` when weights are currently loaded in the engine (Tier 1).
    /// Phases of the most recent Tier 1 load (llama.cpp engines); `None` for the others.
    fn last_load_phases(&self) -> Option<crate::openai::LoadPhases> {
        None
    }

    fn is_resident(&self) -> bool {
        false
    }

    /// Executes Text, Vision or Audio-in chat completion (`/v1/chat/completions`).
    async fn chat_completion(
        &self,
        _req: &ChatCompletionRequest,
        _ctx: &RequestContext,
    ) -> Result<ChatCompletionResponse> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::TextGeneration,
        })
    }

    /// Streaming variant of [`chat_completion`](Self::chat_completion). Implementations push
    /// [`ChatStreamEvent`]s into `tx` as tokens are produced and finish with `Done` or `Error`.
    ///
    /// The default implementation runs the non-streaming path and emits the whole reply as a
    /// single delta, so every backend is usable with `stream: true`.
    async fn chat_completion_stream(
        &self,
        req: &ChatCompletionRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<ChatStreamEvent>,
    ) -> Result<()> {
        let resp = self.chat_completion(req, ctx).await?;
        let (content, finish_reason) = resp
            .choices
            .first()
            .map(|c| (c.message.content.clone(), c.finish_reason.clone()))
            .unwrap_or_default();
        let _ = tx.send(ChatStreamEvent::Delta(content)).await;
        let _ = tx
            .send(ChatStreamEvent::Done {
                usage: resp.usage,
                finish_reason,
                timing: resp.hypnos_telemetry.map(|t| t.engine).unwrap_or_default(),
            })
            .await;
        Ok(())
    }

    /// Executes vector embedding generation (`/v1/embeddings`).
    async fn embeddings(
        &self,
        _req: &EmbeddingRequest,
        _ctx: &RequestContext,
    ) -> Result<EmbeddingResponse> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::Embeddings,
        })
    }

    /// Executes Speech-to-Text transcription (`/v1/audio/transcriptions`).
    async fn transcribe(
        &self,
        _req: &AudioTranscriptionRequest,
        _ctx: &RequestContext,
    ) -> Result<AudioTranscriptionResponse> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::SpeechToText,
        })
    }

    /// Executes Text-to-Speech synthesis (`/v1/audio/speech`).
    async fn synthesize_speech(
        &self,
        _req: &AudioSpeechRequest,
        _ctx: &RequestContext,
    ) -> Result<AudioSpeechOutput> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::TextToSpeech,
        })
    }

    /// Streaming variant of [`synthesize_speech`](Self::synthesize_speech): pushes
    /// [`AudioStreamEvent`]s into `tx` as audio is produced and finishes with `Done` or `Error`.
    async fn synthesize_speech_stream(
        &self,
        _req: &AudioSpeechRequest,
        _ctx: &RequestContext,
        _tx: mpsc::Sender<AudioStreamEvent>,
    ) -> Result<()> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::TextToSpeech,
        })
    }

    /// Converts a document (`/v1/documents/convert`), pushing a `Started` event, one `Page` per
    /// converted page, then `Done` (or `Error`) into `tx`. Must stop between pages once
    /// `ctx.should_continue()` is false.
    async fn convert_document(
        &self,
        _req: &DocumentConversionRequest,
        _ctx: &RequestContext,
        _tx: mpsc::Sender<DocumentStreamEvent>,
    ) -> Result<()> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::DocumentConversion,
        })
    }

    /// Extracts entities of the requested labels (`/v1/entities`).
    async fn extract_entities(
        &self,
        _req: &EntityExtractionRequest,
        _ctx: &RequestContext,
    ) -> Result<EntityExtractionResponse> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::EntityExtraction,
        })
    }

    /// Executes Diffusion / DiT Image Generation (`/v1/images/generations`).
    async fn generate_image(
        &self,
        _req: &ImageGenerationRequest,
        _ctx: &RequestContext,
    ) -> Result<ImageGenerationResponse> {
        Err(crate::error::HypnosError::ModalityMismatch {
            model: self.config().id.clone(),
            expected: self.modality(),
            actual: Modality::ImageGeneration,
        })
    }
}

/// Real device-memory readings, used to reconcile the planner's bookkeeping with the GPU.
pub trait GpuMemoryProbe: Send + Sync {
    /// Returns `(free_bytes, total_bytes)` of the primary accelerator, if it can be measured.
    fn probe(&self) -> Option<(u64, u64)>;
}
