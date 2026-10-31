//! `GET /v1/models/catalog` through the real router: the shipped catalog over a models folder
//! with one model on disk and one file of the wrong architecture. No engine is loaded.

use axum::body::Body;
use axum::http::{Request, StatusCode};
use hypnos_core::config::{AcceleratorKind, HypnosConfig};
use hypnos_server::{build_router, AppState};
use tower::ServiceExt;

/// A GGUF header that only says `general.architecture = <arch>`.
fn gguf_header(arch: &str) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend(0u64.to_le_bytes());
    b.extend(1u64.to_le_bytes());
    let key = b"general.architecture";
    b.extend((key.len() as u64).to_le_bytes());
    b.extend(key);
    b.extend(8u32.to_le_bytes());
    b.extend((arch.len() as u64).to_le_bytes());
    b.extend(arch.as_bytes());
    b
}

async fn get(app: &axum::Router, path: &str) -> (StatusCode, serde_json::Value) {
    let req = Request::builder().uri(path).body(Body::empty()).unwrap();
    let resp = app.clone().oneshot(req).await.unwrap();
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 4 << 20)
        .await
        .unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

#[tokio::test]
async fn the_catalog_reports_what_is_on_disk() {
    let dir =
        std::env::temp_dir().join(format!("hypnos-catalog-{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(dir.join("gliner2.5-multi")).unwrap();
    std::fs::write(dir.join("gliner2.5-multi/encoder_fp32.onnx"), b"x").unwrap();
    std::fs::write(dir.join("LFM3-1.7B-Q8_0.gguf"), gguf_header("qwen3")).unwrap();

    let mut cfg = HypnosConfig::from_toml_str("").unwrap();
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.document_pictures_dir = Some(dir.join("pictures"));
    cfg.server.models_dir = Some(dir.clone());
    let state = AppState::from_config(cfg).unwrap();
    assert!(state.catalog.is_some(), "the checkout's catalog is found");
    // Not ready yet: the catalog needs no model and answers anyway.
    let app = build_router(state, &[]);

    let (status, body) = get(&app, "/v1/models/catalog").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let entry = |id: &str| {
        body["data"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["id"] == id)
            .unwrap_or_else(|| panic!("{id} listed"))
            .clone()
    };
    let gliner = entry("gliner2.5-multi");
    assert_eq!(gliner["variants"][0]["installed"], true);
    assert_eq!(gliner["variants"][0]["outdated"], false);
    assert_eq!(gliner["lineup"], true);
    assert!(
        gliner.get("config").is_none(),
        "settings travel as config_block"
    );

    let lfm = entry("lfm3:1.7b");
    let q8 = &lfm["variants"][0];
    assert_eq!(q8["quant"], "Q8_0");
    assert_eq!(q8["default"], true);
    assert_eq!(q8["installed"], true);
    assert_eq!(q8["outdated"], true, "a header is not the listed size");
    assert_eq!(lfm["variants"][1]["installed"], false);
    assert!(q8["problem"].as_str().unwrap().contains("'qwen3'"), "{q8}");
    assert!(q8["config_block"]
        .as_str()
        .unwrap()
        .contains("weights_path = \"LFM3-1.7B-Q8_0.gguf\""));
    assert_eq!(lfm["capabilities"][0], "thinking");
    assert_eq!(lfm["configured"], false);

    let (_, vision) = get(&app, "/v1/models/catalog?modality=vision").await;
    let ids: Vec<&str> = vision["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["id"].as_str().unwrap())
        .collect();
    assert!(
        ids.contains(&"lfm3-vl:2b") && ids.contains(&"gemma4:26b-q4"),
        "{ids:?}"
    );
    assert!(!ids.contains(&"lfm3:1.7b"), "{ids:?}");

    std::fs::remove_dir_all(&dir).ok();
}

#[tokio::test]
async fn the_advisor_recommends_a_lineup() {
    let mut cfg = HypnosConfig::from_toml_str("").unwrap();
    cfg.hardware.accelerator = AcceleratorKind::Vulkan;
    cfg.server.document_pictures_dir = Some(
        std::env::temp_dir().join(format!("hypnos-advisor-{}", uuid::Uuid::new_v4().simple())),
    );
    let app = build_router(AppState::from_config(cfg).unwrap(), &[]);
    let post = |body: &'static str| {
        let app = app.clone();
        async move {
            let req = Request::builder()
                .method("POST")
                .uri("/v1/advisor")
                .header("content-type", "application/json")
                .body(Body::from(body))
                .unwrap();
            let resp = app.oneshot(req).await.unwrap();
            let status = resp.status();
            let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
                .await
                .unwrap();
            (
                status,
                serde_json::from_slice::<serde_json::Value>(&bytes).unwrap_or_default(),
            )
        }
    };
    let (status, body) = post(
        r#"{"needs": ["chat", "voice"], "vram_bytes": 25769803776, "ram_bytes": 68719476736}"#,
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["machine"]["unified_memory"], false);
    assert_eq!(body["interactive_coresident"], true);
    let ids: Vec<&str> = body["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert!(ids.contains(&"kokoro-82m-q4"), "{ids:?}");
    assert!(body["config"].as_str().unwrap().contains("[[models]]"));
    // This machine when nothing is said about it.
    let (status, body) = post(r#"{"needs": ["documents"]}"#).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["machine"]["ram_bytes"].as_u64().unwrap() > 0);
    // Impossible: an image model on a tiny GPU.
    let (status, _) =
        post(r#"{"needs": ["images"], "vram_bytes": 2147483648, "ram_bytes": 8589934592}"#).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}
