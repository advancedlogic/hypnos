//! Every route against the mock server: request shapes, response shapes, streaming.

mod common;

use std::time::{Duration, Instant};

use base64::Engine;
use common::*;
use futures::StreamExt;
use hypnos_client::*;

const B64: base64::engine::GeneralPurpose = base64::engine::general_purpose::STANDARD;

fn user(text: &str) -> Vec<ChatMessage> {
    vec![ChatMessage::user(text)]
}

#[tokio::test]
async fn health_ready_models() {
    let server = MockServer::start();
    let c = server.client();
    assert_eq!(c.health().await.unwrap().status, "ok");
    assert!(c.is_ready().await);
    c.wait_until_ready(Duration::from_secs(1), Duration::from_millis(50))
        .await
        .unwrap();
    let ids: Vec<String> = c
        .models()
        .await
        .unwrap()
        .data
        .into_iter()
        .map(|m| m.id)
        .collect();
    for id in [CHAT, STT, EMBED, TTS, IMAGE, DOCS, NER] {
        assert!(ids.iter().any(|i| i == id), "{id} missing from {ids:?}");
    }
    assert!(c
        .metrics()
        .await
        .unwrap()
        .contains("hypnos_http_requests_total"));
}

#[tokio::test]
async fn chat() {
    let server = MockServer::start();
    let reply = server
        .client()
        .chat(
            &ChatRequest::new(CHAT, user("hello there"))
                .max_tokens(20)
                .temperature(0.2)
                .seed(1)
                .num_ctx(4096),
        )
        .await
        .unwrap();
    assert_eq!(reply.text(), "echo: hello there");
    assert_eq!(reply.choices[0].finish_reason, "stop");
    assert_eq!(reply.usage.completion_tokens, 3);
    let telemetry = reply.hypnos_telemetry.expect("telemetry");
    assert_eq!(telemetry.placement.primary_tier, MemoryTier::Tier1GpuVram);
}

#[tokio::test]
async fn chat_with_media_parts() {
    let server = MockServer::start();
    let png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";
    let message = ChatMessage::user_parts(vec![
        ContentPart::text("what is this"),
        ContentPart::image_url(format!("data:image/png;base64,{png}")),
    ]);
    let reply = server
        .client()
        .chat(&ChatRequest::new(CHAT, vec![message]))
        .await
        .unwrap();
    assert!(reply.text().starts_with("echo: what is this"));
}

#[tokio::test]
async fn chat_stream() {
    let server = MockServer::start();
    let mut stream = server
        .client()
        .chat_stream(&ChatRequest::new(CHAT, user("count")).max_tokens(5))
        .await
        .unwrap();
    let pieces: Vec<String> = stream.text().map(|p| p.unwrap()).collect().await;
    assert_eq!(pieces.concat(), "1 2 3 4 5");
    assert_eq!(stream.finish_reason(), Some("length"));
    assert_eq!(stream.usage().unwrap().completion_tokens, 5);
    assert!(stream.telemetry().is_some());
    assert!(stream.request_id().unwrap().starts_with("req-"));
    assert_eq!(stream.final_completion().await.unwrap().text(), "1 2 3 4 5");
}

#[tokio::test]
async fn chat_reasoning() {
    let server = MockServer::start();
    let c = server.client();
    let req = ChatRequest::new(CHAT, user("hi")).reasoning_effort(ReasoningEffort::Medium);
    let completion = c.chat(&req).await.unwrap();
    assert_eq!(
        completion.choices[0].message.reasoning_content.as_deref(),
        Some("Mock thoughts (medium).")
    );
    let mut stream = c.chat_stream(&req).await.unwrap();
    let completion = stream.final_completion().await.unwrap();
    assert_eq!(stream.reasoning_so_far(), "Mock thoughts (medium).");
    assert_eq!(completion.text(), "echo: hi");
    let off = ChatRequest::new(CHAT, user("hi")).reasoning_effort(ReasoningEffort::None);
    let completion = c.chat(&off).await.unwrap();
    assert_eq!(completion.choices[0].message.reasoning_content, None);
}

#[tokio::test]
async fn chat_stream_final() {
    let server = MockServer::start();
    let mut stream = server
        .client()
        .chat_stream(&ChatRequest::new(CHAT, user("hello there")))
        .await
        .unwrap();
    let completion = stream.final_completion().await.unwrap();
    assert_eq!(completion.text(), "echo: hello there");
    assert_eq!(completion.object, "chat.completion");
    assert!(completion.hypnos_telemetry.is_some());
}

#[tokio::test]
async fn dropping_a_stream_cancels_it() {
    let server = MockServer::start();
    let c = server.client();
    let started = Instant::now();
    {
        let mut stream = c
            .chat_stream(&ChatRequest::new(CHAT, user("count")).max_tokens(2000))
            .await
            .unwrap();
        let mut text = stream.text();
        for _ in 0..4 {
            text.next().await.unwrap().unwrap();
        }
    }
    // 2000 tokens at 10 ms each would take 20 s.
    assert!(started.elapsed() < Duration::from_secs(5));
    // The model is free again.
    let reply = c.chat(&ChatRequest::new(CHAT, user("hi"))).await.unwrap();
    assert!(!reply.text().is_empty());
}

#[tokio::test]
async fn a_stream_that_fails_yields_an_error() {
    let server = MockServer::start();
    server.fail(CHAT, "stream_error", 1).await;
    let mut stream = server
        .client()
        .chat_stream(&ChatRequest::new(CHAT, user("count")).max_tokens(10))
        .await
        .unwrap();
    let mut error = None;
    while let Some(item) = stream.next().await {
        if let Err(e) = item {
            error = Some(e);
        }
    }
    let error = error.expect("a stream error");
    assert!(matches!(error, Error::Stream(_)), "{error:?}");
    assert_eq!(error.status(), Some(200));
    assert!(error.to_string().contains("stream_error"));
    assert_eq!(stream.text_so_far(), "1 2");
}

#[tokio::test]
async fn embeddings() {
    let server = MockServer::start();
    let c = server.client();
    assert_eq!(c.embeddings(EMBED, "a").await.unwrap().data.len(), 1);
    let many = c.embeddings(EMBED, vec!["a", "b", "c"]).await.unwrap();
    let indexes: Vec<usize> = many.data.iter().map(|d| d.index).collect();
    assert_eq!(indexes, [0, 1, 2]);
    assert_ne!(many.data[0].embedding, many.data[1].embedding);
    assert_eq!(many.data[0].embedding.len(), 8);
}

#[tokio::test]
async fn transcribe() {
    let server = MockServer::start();
    let c = server.client();
    let clip = b"RIFF\x00\x00\x00\x00WAVEdata".to_vec();
    let reply = c
        .transcribe(
            &TranscriptionRequest::new(STT, "clip.wav", clip.clone())
                .language("it")
                .prompt("names"),
        )
        .await
        .unwrap();
    assert_eq!(
        reply.text,
        format!("mock transcript of {} bytes", clip.len())
    );
    assert_eq!(reply.language.as_deref(), Some("it"));
    let path = std::env::temp_dir().join(format!("hypnos-rust-clip-{}.wav", std::process::id()));
    std::fs::write(&path, &clip).unwrap();
    let from_path = c
        .transcribe(&TranscriptionRequest::from_path(STT, &path).unwrap())
        .await
        .unwrap();
    assert_eq!(from_path.text, reply.text);
    let _ = std::fs::remove_file(path);
}

#[tokio::test]
async fn speech() {
    let server = MockServer::start();
    let c = server.client();
    let wav = c
        .speech(&SpeechRequest::new(TTS, "one two three").voice("alloy"))
        .await
        .unwrap();
    assert_eq!(wav.content_type, "audio/wav");
    assert_eq!(&wav.audio[..4], b"RIFF");
    let tier = wav.telemetry.previous_tier.clone().expect("tier header");
    assert!(
        ["tier1_gpu_vram", "tier2_pinned_ram", "tier3_nvme_ssd"].contains(&tier.as_str()),
        "{tier}"
    );
    assert!(wav.telemetry.gpu_layers.is_some());
    let pcm = c
        .speech(&SpeechRequest::new(TTS, "one two three").response_format("pcm"))
        .await
        .unwrap();
    assert_eq!(pcm.content_type, "audio/pcm");
    // 50 ms per word at 24 kHz, 16-bit: 3 * 1200 samples * 2 bytes.
    assert_eq!(pcm.audio.len(), 3 * 1200 * 2);
}

#[tokio::test]
async fn speech_stream() {
    let server = MockServer::start();
    let mut stream = server
        .client()
        .speech_stream(&SpeechRequest::new(TTS, "one two three four").response_format("pcm"))
        .await
        .unwrap();
    assert_eq!(stream.content_type(), "audio/pcm");
    assert!(stream.telemetry().previous_tier.is_some());
    assert_eq!(stream.read().await.unwrap().len(), 4 * 1200 * 2);
}

#[tokio::test]
async fn images() {
    let server = MockServer::start();
    let reply = server
        .client()
        .images(
            &ImageRequest::new(IMAGE, "a lighthouse")
                .n(2)
                .size("512x512")
                .steps(4)
                .seed(7),
        )
        .await
        .unwrap();
    assert_eq!(reply.data.len(), 2);
    let png = B64.decode(&reply.data[0].b64_json).unwrap();
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    assert!(reply.hypnos_telemetry.is_some());
}

#[tokio::test]
async fn entities() {
    let server = MockServer::start();
    let text = "Mario Rossi works in Milan.";
    let reply = server
        .client()
        .entities(
            &EntitiesRequest::new(NER, vec![text, "nothing here"], &["person", "city"])
                .threshold(0.4),
        )
        .await
        .unwrap();
    let found = &reply.data[0].entities;
    let texts: Vec<&str> = found.iter().map(|e| e.text.as_str()).collect();
    assert_eq!(texts, ["Mario", "Rossi", "Milan"]);
    for e in found {
        assert_eq!(slice_chars(text, e.start, e.end), e.text);
    }
    assert!(reply.data[1].entities.is_empty());
    assert_eq!(reply.usage.texts, 2);
}

#[tokio::test]
async fn documents_and_pictures() {
    let server = MockServer::start();
    let c = server.client();
    let doc = c
        .convert_document(
            &DocumentRequest::file(DOCS, "report.pdf", b"%PDF-1.4 fake".to_vec())
                .pages("2-3")
                .return_pictures(true),
        )
        .await
        .unwrap();
    assert_eq!(doc.page_count, 3);
    let pages: Vec<u32> = doc.pages.iter().map(|p| p.page).collect();
    assert_eq!(pages, [2, 3]);
    assert!(doc.markdown.contains("## Page 2"));
    let id = doc.id.clone().expect("document id");
    assert_eq!(
        doc.pages[0].picture_urls,
        [format!("/v1/documents/{id}/pictures/1")]
    );

    let listed = c.document_pictures(&id).await.unwrap();
    let indexes: Vec<u32> = listed.data.iter().map(|p| p.index).collect();
    assert_eq!(indexes, [1, 2]);
    assert!(listed.data[0].image.is_none());
    let picture = c.document_picture(&id, 1).await.unwrap();
    assert_eq!(picture.caption.as_deref(), Some("Figure 2"));
    assert_eq!(
        &B64.decode(picture.image.unwrap()).unwrap()[..4],
        b"\x89PNG"
    );
    let (raw, mime) = c.document_picture_bytes(&id, 2).await.unwrap();
    assert_eq!(mime, "image/png");
    assert_eq!(&raw[..4], b"\x89PNG");

    assert_eq!(c.delete_document(&id).await.unwrap()["deleted"], true);
    assert!(matches!(
        c.document_pictures(&id).await,
        Err(Error::NotFound(_))
    ));
}

#[tokio::test]
async fn document_stream() {
    let server = MockServer::start();
    let c = server.client();
    let req = DocumentRequest::file(DOCS, "a.pdf", b"%PDF fake".to_vec());

    let mut stream = c
        .convert_document_stream(&req.clone().output_format("json"))
        .await
        .unwrap();
    let mut kinds = Vec::new();
    while let Some(event) = stream.next().await {
        kinds.push(match event.unwrap() {
            DocumentEvent::Started(_) => "document.started",
            DocumentEvent::Page(_) => "document.page",
            DocumentEvent::Done(_) => "document",
        });
    }
    assert_eq!(
        kinds,
        [
            "document.started",
            "document.page",
            "document.page",
            "document.page",
            "document"
        ]
    );
    assert_eq!(stream.started().unwrap().pages, [1, 2, 3]);
    // Iterating first and asking for the result afterwards works too.
    assert_eq!(stream.final_document().await.unwrap().pages.len(), 3);

    let mut stream = c.convert_document_stream(&req).await.unwrap();
    let doc = stream.final_document().await.unwrap();
    // The server leaves `pages` empty in its last event; the stream puts them back.
    let pages: Vec<u32> = doc.pages.iter().map(|p| p.page).collect();
    assert_eq!(pages, [1, 2, 3]);
    assert_eq!(doc.markdown.matches("## Page").count(), 3);

    let mut stream = c
        .convert_document_stream(&req.clone().pages("2"))
        .await
        .unwrap();
    let pages: Vec<u32> = stream.pages().map(|p| p.unwrap().page).collect().await;
    assert_eq!(pages, [2]);
}

#[tokio::test]
async fn kv_parking() {
    let server = MockServer::start();
    let c = server.client();
    let status = c.kv_park().await.unwrap();
    assert!(!status.enabled);
    assert_eq!(status.used_bytes, 0);
    let on = c.set_kv_park(Some(true), Some(1 << 30)).await.unwrap();
    assert!(on.enabled);
    assert_eq!(on.max_bytes, 1_073_741_824);
    assert!(on.models.is_empty());
    let memory = c.memory_status().await.unwrap();
    assert!(memory.kv_park.unwrap().enabled);
    let err = c.set_kv_park(None, None).await.unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
    let off = c.set_kv_park(Some(false), None).await.unwrap();
    assert!(!off.enabled);
}

#[tokio::test]
async fn memory_and_placement() {
    let server = MockServer::start();
    let c = server.client();
    c.chat(&ChatRequest::new(CHAT, user("hi"))).await.unwrap();
    let status = c.memory_status().await.unwrap();
    let model = status.models.iter().find(|m| m.id == CHAT).unwrap();
    assert_eq!(model.current_tier, MemoryTier::Tier1GpuVram);
    let moved = c
        .set_tier(CHAT, MemoryTier::Tier2PinnedRam, None)
        .await
        .unwrap();
    assert_eq!(moved.placement.primary_tier, MemoryTier::Tier2PinnedRam);
    let back = c
        .set_tier(CHAT, MemoryTier::Tier1GpuVram, Some(4096))
        .await
        .unwrap();
    assert_eq!(back.placement.primary_tier, MemoryTier::Tier1GpuVram);
}

#[tokio::test]
async fn moe_offload() {
    let server = MockServer::start();
    let c = server.client();
    let on = c.set_moe_offload(CHAT, MoeOffload::On).await.unwrap();
    assert_eq!(on.model, CHAT);
    assert_eq!(on.moe_offload, MoeOffload::On);
    assert!(
        on.applies == "now" || on.applies == "next_load",
        "{}",
        on.applies
    );
    assert_eq!(on.placement.cpu_expert_layers, 0);
    let status = c.memory_status().await.unwrap();
    let chat = status.models.iter().find(|m| m.id == CHAT).unwrap();
    assert_eq!(chat.moe_offload, Some(MoeOffload::On));
    match c.set_moe_offload("no-such-model", MoeOffload::On).await {
        Err(Error::NotFound(_)) => {}
        other => panic!("expected not found, got {other:?}"),
    }
    let off = c.set_moe_offload(CHAT, MoeOffload::Off).await.unwrap();
    assert_eq!(off.moe_offload, MoeOffload::Off);
}

#[tokio::test]
async fn enable_and_disable() {
    let server = MockServer::start();
    let c = server.client();
    assert!(!c.disable_model(IMAGE).await.unwrap().enabled);
    let ids: Vec<String> = c
        .models()
        .await
        .unwrap()
        .data
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert!(!ids.iter().any(|i| i == IMAGE));
    match c.images(&ImageRequest::new(IMAGE, "x")).await {
        Err(Error::NotFound(e)) => assert_eq!(e.kind.as_deref(), Some("model_disabled")),
        other => panic!("expected model_disabled, got {other:?}"),
    }
    assert!(c.enable_model(IMAGE).await.unwrap().enabled);
    assert!(!c
        .images(&ImageRequest::new(IMAGE, "x"))
        .await
        .unwrap()
        .data
        .is_empty());
}

#[tokio::test]
async fn model_pool() {
    let server = MockServer::start();
    let c = server.client();
    let pooled = "lfm3:1.7b";
    let added = c.add_model(&ModelSource::catalog(pooled)).await.unwrap();
    assert_eq!(added.model, pooled);
    assert_eq!(added.action, ModelPoolAction::Added);
    assert!(added.placement.is_some());
    let ids: Vec<String> = c
        .models()
        .await
        .unwrap()
        .data
        .into_iter()
        .map(|m| m.id)
        .collect();
    assert!(ids.iter().any(|i| i == pooled));
    let reply = c
        .chat(&ChatRequest::new(pooled, user("hello there")))
        .await
        .unwrap();
    assert_eq!(reply.text(), "echo: hello there");
    let err = c
        .add_model(&ModelSource::catalog(pooled))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::BadRequest(_)), "{err:?}");
    let replaced = c
        .replace_model(pooled, &ModelSource::catalog("lfm3:1.7b:Q4_K_M"), Some(5))
        .await
        .unwrap();
    assert_eq!(replaced.model, pooled);
    assert_eq!(replaced.action, ModelPoolAction::Replaced);
    let removed = c.remove_model(pooled, Some(5)).await.unwrap();
    assert_eq!(removed.model, pooled);
    assert_eq!(removed.action, ModelPoolAction::Removed);
    assert!(removed.placement.is_none());
    let err = c
        .chat(&ChatRequest::new(pooled, user("hi")))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    let err = c.remove_model(pooled, None).await.unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
    let err = c
        .replace_model(pooled, &ModelSource::catalog(pooled), None)
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotFound(_)), "{err:?}");
}

#[test]
fn model_source_wire_shape() {
    assert_eq!(
        serde_json::to_value(ModelSource::catalog("lfm3:1.7b")).unwrap(),
        serde_json::json!({ "catalog": "lfm3:1.7b" })
    );
    assert_eq!(
        serde_json::to_value(ModelSource::config("[[models]]")).unwrap(),
        serde_json::json!({ "config": "[[models]]" })
    );
}

#[tokio::test]
async fn logs() {
    let server = MockServer::start();
    let c = server.client();
    c.chat(&ChatRequest::new(CHAT, user("hi"))).await.unwrap();
    let logs = c
        .logs(&LogQuery {
            route: Some("/v1/chat/completions".into()),
            limit: Some(5),
            ..LogQuery::default()
        })
        .await
        .unwrap();
    assert!(!logs.records.is_empty());
    let record = c.log(&logs.records[0].id).await.unwrap();
    assert_eq!(record.route, "/v1/chat/completions");
    assert_eq!(c.set_log_level("debug").await.unwrap()["level"], "debug");
    c.set_log_level("info").await.unwrap();
    assert_eq!(c.clear_logs().await.unwrap()["cleared"], true);
}

#[tokio::test]
async fn models_catalog() {
    let server = MockServer::start();
    let c = server.client();
    let catalog = c.models_catalog(None).await.unwrap();
    assert_eq!(catalog.object, "list");
    assert!(catalog.data.len() >= 10, "{} entries", catalog.data.len());
    let entry = |id: &str| catalog.data.iter().find(|e| e.id == id).unwrap();
    assert!(entry(CHAT).configured);
    let lfm3 = entry("lfm3:1.7b");
    assert!(!lfm3.configured);
    assert_eq!(lfm3.enabled, None);
    let q8 = lfm3.variants.iter().find(|v| v.quant == "Q8_0").unwrap();
    assert!(q8.default);
    assert!(lfm3.variants.iter().any(|v| v.quant == "Q4_K_M"));
    assert!(q8
        .config_block
        .contains(r#"weights_path = "LFM3-1.7B-Q8_0.gguf""#));
    assert!(catalog
        .data
        .iter()
        .all(|e| e.variants.iter().all(|v| !v.installed)));

    let vision: Vec<String> = c
        .models_catalog(Some("vision"))
        .await
        .unwrap()
        .data
        .into_iter()
        .map(|e| e.id)
        .collect();
    for id in ["gemma4:26b-q4", "gemma4:e4b-q4", "lfm3-vl:2b"] {
        assert!(
            vision.iter().any(|i| i == id),
            "{id} missing from {vision:?}"
        );
    }
    assert!(!vision.iter().any(|i| i == "lfm3:1.7b"));
}
