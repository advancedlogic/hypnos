//! The readiness gate: while startup pre-staging runs, observation routes answer and the
//! inference routes say 503, through the real router. No engine is loaded.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use hypnos_core::config::{AcceleratorKind, HypnosConfig};
use hypnos_server::{build_router, AppState};
use std::sync::atomic::Ordering;
use tower::ServiceExt;

fn state() -> AppState {
    let mut cfg = HypnosConfig::from_toml_str("").expect("empty config");
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.document_pictures_dir =
        Some(std::env::temp_dir().join(format!("hypnos-ready-{}", uuid::Uuid::new_v4().simple())));
    AppState::from_config(cfg).expect("state")
}

async fn send(
    app: &axum::Router,
    method: Method,
    path: &str,
    body: &str,
) -> axum::response::Response {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(req).await.expect("response")
}

async fn json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn inference_waits_for_readiness_while_observation_does_not() {
    let state = state();
    let ready = state.ready.clone();
    let app = build_router(state, &[]);
    let chat = r#"{"model":"nope","messages":[{"role":"user","content":"hi"}]}"#;

    // Not ready: probes and observation answer, inference is refused with 503 and Retry-After.
    assert_eq!(
        send(&app, Method::GET, "/health", "").await.status(),
        StatusCode::OK
    );
    let resp = send(&app, Method::GET, "/ready", "").await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(json(resp).await["ready"], false);
    assert_eq!(
        send(&app, Method::GET, "/v1/memory/status", "")
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, Method::GET, "/v1/models", "").await.status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, Method::GET, "/v1/logs", "").await.status(),
        StatusCode::OK
    );
    let resp = send(&app, Method::POST, "/v1/chat/completions", chat).await;
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(resp.headers()[header::RETRY_AFTER], "5");
    assert_eq!(json(resp).await["error"]["type"], "model_not_available");
    let tier = r#"{"model":"nope","target_tier":"tier3_nvme_ssd"}"#;
    assert_eq!(
        send(&app, Method::POST, "/v1/memory/tier", tier)
            .await
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    assert_eq!(
        send(&app, Method::POST, "/v1/models/nope/disable", "")
            .await
            .status(),
        StatusCode::SERVICE_UNAVAILABLE
    );

    // Ready: the same request reaches the handler (404: no such model in this empty config).
    ready.store(true, Ordering::SeqCst);
    assert_eq!(
        send(&app, Method::GET, "/ready", "").await.status(),
        StatusCode::OK
    );
    let resp = send(&app, Method::POST, "/v1/chat/completions", chat).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(json(resp).await["error"]["type"], "model_not_found");
}
