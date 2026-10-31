use hypnos_core::request::RequestContext;
use hypnos_core::{
    config::MemoryTier,
    openai::{ChatCompletionRequest, ChatMessage, ChatMessageContent, ContentPart, ImageUrlSpec},
};
use hypnos_server::{default_profile::default_six_model_config, AppState};

#[tokio::test]
#[ignore = "loads the real Gemma 4 / Kokoro / MOSS-Image weights from backend/models and needs a CUDA GPU; run with `cargo test -- --ignored`"]
async fn test_gemma4_kokoro_moss_multi_modal_lifecycle_24gb_vram() {
    let mut cfg = default_six_model_config();
    // models_dir = "models" against backend/, where `./run.sh models` puts the weights (cargo
    // runs tests in the crate's folder).
    cfg.apply_models_dir(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."),
        None,
    );
    cfg.hardware.vram_budget_bytes = 24 * 1024 * 1024 * 1024;
    cfg.hardware.vram_safety_margin_bytes = 512 * 1024 * 1024; // 23.5 GB usable VRAM
    cfg.hardware.pinned_ram_budget_bytes = 32 * 1024 * 1024 * 1024;

    let state = AppState::from_config(cfg).expect("Failed to initialize AppState");
    let ctx = RequestContext::unbounded();
    state
        .scheduler
        .initialize_startup_tiers()
        .await
        .expect("Startup tier init failed");

    // 1. Verify both `gemmaembedding2` (~600 MB) and `kokoro-82m-q4` (~85 MB) are pinned in Tier 1 VRAM on boot
    for pinned_id in ["gemmaembedding2", "kokoro-82m-q4"] {
        let (_, plan) = state.scheduler.memory().get_model_state(pinned_id).unwrap();
        assert_eq!(plan.primary_tier, MemoryTier::Tier1GpuVram);
    }

    // 2. Load `gemma4:26b-q4` (~15.5 GB weights + SWA KV cache + 512 MB workspace = ~16.5 GB)
    // AND `gemma4:e4b-q4` (~2.8 GB weights + SWA KV cache + 256 MB workspace = ~3.2 GB)
    // Total VRAM = ~0.72 GB (Embed) + ~0.15 GB (Kokoro) + ~16.5 GB (Gemma 4 26B) + ~3.2 GB (Gemma 4 E4B) = ~20.6 GB <= 23.5 GB!
    // All 4 models (covering 5 modalities: Text, Vision, Audio/STT, TTS, Embeddings) fit in 24 GB VRAM simultaneously!
    {
        let lease_26b = state
            .scheduler
            .acquire_model("gemma4:26b-q4", Some(16384), 1, &ctx)
            .await
            .unwrap();
        assert_eq!(lease_26b.previous_tier, MemoryTier::Tier2PinnedRam);
        assert_eq!(lease_26b.placement.gpu_layers, 46);

        // Verify `gemma4:26b-q4` serves BOTH Text Generation and Vision requests!
        let text_req = ChatCompletionRequest {
            model: "gemma4:26b-q4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: ChatMessageContent::Text("Hello Gemma 4!".to_string()),
                ..ChatMessage::default()
            }],
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            num_ctx: None,
            think: None,
            ..ChatCompletionRequest::default()
        };
        let text_resp = lease_26b
            .backend
            .chat_completion(&text_req, &ctx)
            .await
            .unwrap();
        assert!(!text_resp.choices[0].message.content.trim().is_empty());

        let vision_req = ChatCompletionRequest {
            model: "gemma4:26b-q4".to_string(),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: ChatMessageContent::Parts(vec![
                    ContentPart::Text {
                        text: "Describe this image".to_string(),
                    },
                    ContentPart::ImageUrl {
                        image_url: ImageUrlSpec {
                            url: "data:image/png;base64,iVBORw0KGgo=".to_string(),
                            detail: None,
                        },
                    },
                ]),
                ..ChatMessage::default()
            }],
            max_tokens: Some(32),
            temperature: None,
            top_p: None,
            top_k: None,
            seed: None,
            stream: None,
            num_ctx: None,
            think: None,
            ..ChatCompletionRequest::default()
        };
        let vision_resp = lease_26b
            .backend
            .chat_completion(&vision_req, &ctx)
            .await
            .unwrap();
        assert!(!vision_resp.choices[0].message.content.trim().is_empty());
    }

    {
        let lease_e4b = state
            .scheduler
            .acquire_model("gemma4:e4b-q4", Some(8192), 1, &ctx)
            .await
            .unwrap();
        assert_eq!(lease_e4b.placement.gpu_layers, 35);
    }

    // Confirm all 4 models are simultaneously hot in Tier 1 VRAM
    for id in [
        "gemmaembedding2",
        "kokoro-82m-q4",
        "gemma4:26b-q4",
        "gemma4:e4b-q4",
    ] {
        let (_, plan) = state.scheduler.memory().get_model_state(id).unwrap();
        assert_eq!(
            plan.primary_tier,
            MemoryTier::Tier1GpuVram,
            "Model {id} should be co-resident in 24GB Tier 1 VRAM"
        );
    }

    // 3. Now request `moss-image` (~8.5 GB weights + 1.5 GB workspace = 10 GB).
    // The scheduler evicts idle Gemma 4 models to Tier 2 Pinned Host RAM while keeping
    // `gemmaembedding2` and `kokoro-82m-q4` (`RealtimePinned`) in Tier 1 VRAM!
    {
        let moss_lease = state
            .scheduler
            .acquire_model("moss-image", None, 1, &ctx)
            .await
            .unwrap();
        assert_eq!(moss_lease.placement.primary_tier, MemoryTier::Tier1GpuVram);
        assert_eq!(moss_lease.placement.gpu_layers, 48);
        assert!(!moss_lease.evicted_models.is_empty());
        assert!(!moss_lease
            .evicted_models
            .contains(&"gemmaembedding2".to_string()));
        assert!(!moss_lease
            .evicted_models
            .contains(&"kokoro-82m-q4".to_string()));
    }

    // Verify evicted `gemma4:26b-q4` stayed warm in Tier 2 Pinned Host RAM (NOT cold SSD)
    let (_, gemma26b_after) = state
        .scheduler
        .memory()
        .get_model_state("gemma4:26b-q4")
        .unwrap();
    assert_eq!(gemma26b_after.primary_tier, MemoryTier::Tier2PinnedRam);
    assert!(gemma26b_after.pinned_ram_bytes > 0);
}
