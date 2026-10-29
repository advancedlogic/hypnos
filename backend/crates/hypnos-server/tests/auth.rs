//! API-key authentication through the real router (middleware order, CORS, error shape).
//! No engine is loaded: the config has no models.

use axum::body::Body;
use axum::http::{header, Request, StatusCode};
use hypnos_core::config::{AcceleratorKind, HypnosConfig};
use hypnos_server::{build_router, AppState};
use tower::ServiceExt;

const KEY: &str = "test-key-0123456789abcdef";

fn router(keys: &[&str], metrics_require_auth: bool) -> axum::Router {
    let mut cfg = HypnosConfig::from_toml_str("").expect("empty config");
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.api_keys = keys.iter().map(|k| k.to_string()).collect();
    cfg.server.metrics_require_auth = metrics_require_auth;
    let state = AppState::from_config(cfg).expect("state");
    build_router(state, &[])
}

async fn send(app: &axum::Router, req: Request<Body>) -> axum::response::Response {
    app.clone().oneshot(req).await.expect("response")
}

fn get(path: &str, auth: Option<&str>) -> Request<Body> {
    let mut b = Request::get(path).header(header::ORIGIN, "http://localhost:5173");
    if let Some(a) = auth {
        b = b.header(header::AUTHORIZATION, a);
    }
    b.body(Body::empty()).unwrap()
}

async fn json(resp: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

#[tokio::test]
async fn v1_requires_a_valid_key_and_answers_in_openai_shape() {
    let app = router(&[KEY], false);

    let resp = send(&app, get("/v1/models", None)).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        resp.headers()[header::WWW_AUTHENTICATE],
        "Bearer realm=\"hypnos\""
    );
    // The browser playground must be able to read the 401: CORS headers are still present.
    assert_eq!(
        resp.headers()[header::ACCESS_CONTROL_ALLOW_ORIGIN],
        "http://localhost:5173"
    );
    assert!(resp.headers().contains_key("x-request-id"));
    let body = json(resp).await;
    assert_eq!(body["error"]["code"], "invalid_api_key");
    assert_eq!(body["error"]["type"], "invalid_request_error");

    let resp = send(&app, get("/v1/models", Some("Bearer wrong-key"))).await;
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    assert_eq!(
        json(resp).await["error"]["message"],
        "Incorrect API key provided."
    );

    let resp = send(&app, get("/v1/models", Some(&format!("Bearer {KEY}")))).await;
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn probes_and_preflight_stay_open() {
    let app = router(&[KEY], false);
    for path in ["/health", "/metrics"] {
        assert_eq!(
            send(&app, get(path, None)).await.status(),
            StatusCode::OK,
            "{path}"
        );
    }
    // /ready answers 503 before startup finishes, but never 401.
    assert_ne!(
        send(&app, get("/ready", None)).await.status(),
        StatusCode::UNAUTHORIZED
    );

    // CORS preflight carries no Authorization header and must not be rejected.
    let preflight = Request::builder()
        .method("OPTIONS")
        .uri("/v1/chat/completions")
        .header(header::ORIGIN, "http://localhost:5173")
        .header(header::ACCESS_CONTROL_REQUEST_METHOD, "POST")
        .header(
            header::ACCESS_CONTROL_REQUEST_HEADERS,
            "authorization,content-type",
        )
        .body(Body::empty())
        .unwrap();
    let resp = send(&app, preflight).await;
    assert!(
        resp.status().is_success(),
        "preflight got {}",
        resp.status()
    );
    let allowed = resp.headers()[header::ACCESS_CONTROL_ALLOW_HEADERS]
        .to_str()
        .unwrap()
        .to_ascii_lowercase();
    assert!(allowed.contains("authorization"));
}

#[tokio::test]
async fn metrics_can_be_protected_and_no_keys_means_open() {
    let app = router(&[KEY], true);
    assert_eq!(
        send(&app, get("/metrics", None)).await.status(),
        StatusCode::UNAUTHORIZED
    );
    assert_eq!(
        send(&app, get("/metrics", Some(&format!("Bearer {KEY}"))))
            .await
            .status(),
        StatusCode::OK
    );
    assert_eq!(
        send(&app, get("/health", None)).await.status(),
        StatusCode::OK
    );

    let open = router(&[], false);
    assert_eq!(
        send(&open, get("/v1/models", None)).await.status(),
        StatusCode::OK
    );
}
