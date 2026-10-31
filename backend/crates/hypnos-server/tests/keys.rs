//! Per-key limits through the real router: a labelled key with `requests_per_minute = 1` is
//! refused with 429 on its second call, plain keys stay unlimited, the key's label reaches the
//! request log and `/metrics`. No engine is loaded.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use hypnos_core::config::{AcceleratorKind, ApiKeyProfile, HypnosConfig};
use hypnos_server::{build_router, AppState};
use std::sync::atomic::Ordering;
use tower::ServiceExt;

const TEAM: &str = "secret-team-a-0123456789";
const PLAIN: &str = "plain-key-0123456789abcdef";

fn router() -> axum::Router {
    let mut cfg = HypnosConfig::from_toml_str("").expect("empty config");
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.api_keys = vec![PLAIN.into()];
    cfg.server.keys = vec![ApiKeyProfile {
        label: "team-a".into(),
        key: TEAM.into(),
        requests_per_minute: Some(1),
        max_concurrent: None,
        tokens_per_day: None,
        admin: false,
    }];
    cfg.server.document_pictures_dir =
        Some(std::env::temp_dir().join(format!("hypnos-keys-{}", uuid::Uuid::new_v4().simple())));
    let state = AppState::from_config(cfg).expect("state");
    state.ready.store(true, Ordering::SeqCst);
    build_router(state, &[])
}

async fn send(
    app: &axum::Router,
    method: Method,
    path: &str,
    key: &str,
    body: &str,
) -> axum::response::Response {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .header(header::AUTHORIZATION, format!("Bearer {key}"))
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    app.clone().oneshot(req).await.expect("response")
}

async fn text(resp: axum::response::Response) -> String {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[tokio::test]
async fn labelled_key_is_rate_limited_and_named_in_logs_and_metrics() {
    let app = router();
    let chat = r#"{"model":"nope","messages":[{"role":"user","content":"hi"}]}"#;

    // First call of the minute goes through (404: no such model in this empty config).
    let resp = send(&app, Method::POST, "/v1/chat/completions", TEAM, chat).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    // Second call: over the key's requests_per_minute.
    let resp = send(&app, Method::GET, "/v1/models", TEAM, "").await;
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);
    assert!(resp.headers().contains_key(header::RETRY_AFTER));
    let body: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    assert_eq!(body["error"]["type"], "rate_limit_error");
    assert_eq!(body["error"]["code"], "requests_per_minute");

    // The plain key has no limits.
    for _ in 0..3 {
        let resp = send(&app, Method::GET, "/v1/models", PLAIN, "").await;
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // The request log names the key; the plain key gets a digest label.
    let resp = send(&app, Method::GET, "/v1/logs?key=team-a", PLAIN, "").await;
    let logs: serde_json::Value = serde_json::from_str(&text(resp).await).unwrap();
    let records = logs["records"].as_array().unwrap();
    assert!(!records.is_empty());
    assert!(records.iter().all(|r| r["key"] == "team-a"), "{records:?}");
    assert_eq!(records[0]["route"], "/v1/chat/completions");

    // /metrics carries the per-key counters.
    let resp = send(&app, Method::GET, "/metrics", PLAIN, "").await;
    let metrics = text(resp).await;
    assert!(
        metrics.contains("hypnos_api_key_requests_total{key=\"team-a\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains(
            "hypnos_api_key_rejections_total{key=\"team-a\",reason=\"requests_per_minute\"} 1"
        ),
        "{metrics}"
    );
    assert!(
        metrics.contains("hypnos_api_key_tokens_today{key=\"team-a\"} 0"),
        "{metrics}"
    );
}
