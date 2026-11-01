//! Uploads at real sizes against the limits (TEST-6). BUG-8 (multipart uploads above 2 MB
//! refused) lived because every test sent tiny files: these send megabytes to each upload route,
//! just under and just over the configured limit, as multipart and as JSON, and check the answer
//! is the documented one (the request reaches the handler; the limit's 400 names the limit; a body
//! over `max_body_bytes` gets 413).
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use base64::Engine;
use hypnos_core::config::{AcceleratorKind, HypnosConfig};
use hypnos_server::{build_router, AppState};
use tower::ServiceExt;

const MIB: usize = 1 << 20;

/// A router with small limits, so the edges are megabytes and the test stays quick.
fn app() -> axum::Router {
    let mut cfg = HypnosConfig::from_toml_str(
        r#"
        [limits]
        max_body_bytes = 8388608      # 8 MiB
        max_document_bytes = 5242880  # 5 MiB
        max_audio_bytes = 5242880     # 5 MiB
        max_image_bytes = 2097152     # 2 MiB
        "#,
    )
    .expect("config");
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.document_pictures_dir =
        Some(std::env::temp_dir().join(format!("hypnos-limits-{}", uuid::Uuid::new_v4().simple())));
    let state = AppState::from_config(cfg).expect("state");
    state.ready.store(true, std::sync::atomic::Ordering::SeqCst);
    build_router(state, &[])
}

/// `multipart/form-data` with text fields and one file part of `size` bytes.
fn multipart(fields: &[(&str, &str)], file_field: &str, size: usize) -> (String, Vec<u8>) {
    let boundary = "hypnos-limits-boundary";
    let mut body = Vec::new();
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"upload.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        )
        .as_bytes(),
    );
    body.resize(body.len() + size, b'x');
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={boundary}"), body)
}

async fn send(
    app: &axum::Router,
    path: &str,
    content_type: &str,
    body: Vec<u8>,
) -> (StatusCode, String) {
    let req = Request::builder()
        .method(Method::POST)
        .uri(path)
        .header(header::CONTENT_TYPE, content_type)
        .body(Body::from(body))
        .unwrap();
    let resp = app.clone().oneshot(req).await.expect("response");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap();
    (status, String::from_utf8_lossy(&bytes).into_owned())
}

/// Under the limit the upload is read in full and the request goes on to the model, which this
/// test server does not have: never a size or parsing error.
fn reached_handler(path: &str, (status, text): &(StatusCode, String)) {
    assert!(
        !text.contains("above limits")
            && !text.contains("multipart")
            && *status != StatusCode::PAYLOAD_TOO_LARGE,
        "{path}: {status} {text}"
    );
}

fn refused_by(limit: &str, path: &str, (status, text): &(StatusCode, String)) {
    assert_eq!(*status, StatusCode::BAD_REQUEST, "{path}: {text}");
    assert!(text.contains(limit), "{path}: expected {limit} in {text}");
}

#[tokio::test]
async fn multipart_uploads_at_the_limits() {
    let app = app();
    for (path, field, limit, under, over) in [
        (
            "/v1/documents/convert",
            "file",
            "max_document_bytes",
            4 * MIB,
            6 * MIB,
        ),
        (
            "/v1/audio/transcriptions",
            "file",
            "max_audio_bytes",
            4 * MIB,
            6 * MIB,
        ),
        (
            "/v1/images/edits",
            "image",
            "max_image_bytes",
            MIB + MIB / 2,
            3 * MIB,
        ),
    ] {
        let fields = [("model", "no-such-model"), ("prompt", "make it darker")];
        let (ct, body) = multipart(&fields, field, under);
        reached_handler(path, &send(&app, path, &ct, body).await);
        let (ct, body) = multipart(&fields, field, over);
        refused_by(limit, path, &send(&app, path, &ct, body).await);
    }
}

#[tokio::test]
async fn bodies_over_max_body_bytes_get_413() {
    let app = app();
    for (path, field) in [
        ("/v1/documents/convert", "file"),
        ("/v1/audio/transcriptions", "file"),
        ("/v1/images/edits", "image"),
    ] {
        let (ct, body) = multipart(&[("model", "no-such-model")], field, 9 * MIB);
        let (status, text) = send(&app, path, &ct, body).await;
        assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{path}: {text}");
        assert!(
            text.contains("max_body_bytes") && text.contains("payload_too_large"),
            "{path}: {text}"
        );
    }
    let big = serde_json::json!({ "model": "no-such-model", "input": "x".repeat(9 * MIB) });
    let (status, text) = send(
        &app,
        "/v1/embeddings",
        "application/json",
        serde_json::to_vec(&big).unwrap(),
    )
    .await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE, "{text}");
}

#[tokio::test]
async fn json_document_uploads_at_the_limits() {
    let app = app();
    let path = "/v1/documents/convert";
    let b64 = |size: usize| base64::engine::general_purpose::STANDARD.encode(vec![b'x'; size]);
    // 4 MiB of file is about 5.3 MiB of JSON: under both limits.
    let under =
        serde_json::json!({ "model": "no-such-model", "file": b64(4 * MIB), "filename": "a.pdf" });
    reached_handler(
        path,
        &send(
            &app,
            path,
            "application/json",
            serde_json::to_vec(&under).unwrap(),
        )
        .await,
    );
    // 5.5 MiB of file is about 7.3 MiB of JSON: under the body limit, over the document limit.
    let over = serde_json::json!({ "model": "no-such-model", "file": b64(5 * MIB + MIB / 2), "filename": "a.pdf" });
    refused_by(
        "max_document_bytes",
        path,
        &send(
            &app,
            path,
            "application/json",
            serde_json::to_vec(&over).unwrap(),
        )
        .await,
    );
}
