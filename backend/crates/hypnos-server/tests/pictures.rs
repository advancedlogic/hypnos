//! The picture endpoints of `return_pictures` document conversions through the real router.
//! No engine is loaded: the pictures are put into the store directly, as a conversion would.

use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use hypnos_core::config::{AcceleratorKind, HypnosConfig};
use hypnos_core::openai::{ExtractedPicture, PictureBox};
use hypnos_server::{build_router, AppState};
use tower::ServiceExt;

const PNG_1X1: &str =
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

fn state() -> AppState {
    let mut cfg = HypnosConfig::from_toml_str("").expect("empty config");
    cfg.hardware.accelerator = AcceleratorKind::Vulkan; // no probing of this machine's GPU
    cfg.server.document_pictures_dir = Some(std::env::temp_dir().join(format!(
        "hypnos-pictures-router-{}",
        uuid::Uuid::new_v4().simple()
    )));
    AppState::from_config(cfg).expect("state")
}

async fn send(app: &axum::Router, method: Method, path: &str) -> axum::response::Response {
    let req = Request::builder()
        .method(method)
        .uri(path)
        .body(Body::empty())
        .unwrap();
    app.clone().oneshot(req).await.expect("response")
}

async fn body(resp: axum::response::Response) -> Vec<u8> {
    axum::body::to_bytes(resp.into_body(), 1 << 20)
        .await
        .unwrap()
        .to_vec()
}

async fn json(resp: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&body(resp).await).unwrap()
}

#[tokio::test]
async fn pictures_are_listed_fetched_and_released() {
    let state = state();
    let store = state.pictures.clone();
    let app = build_router(state, &[]);

    let id = store.new_document();
    let picture = ExtractedPicture {
        data_uri: format!("data:image/png;base64,{PNG_1X1}"),
        mime: "image/png".into(),
        width: Some(1),
        height: Some(1),
        dpi: Some(72),
        bbox: Some(PictureBox {
            l: 72.0,
            t: 100.0,
            r: 300.0,
            b: 400.0,
            coord_origin: "TOPLEFT".into(),
        }),
        page_size: None,
        caption: Some("Figure 1".into()),
        self_ref: Some("#/pictures/0".into()),
    };
    let urls = store.add_page(&id, 4, &[picture.clone(), picture]).unwrap();
    assert_eq!(urls[1], format!("/v1/documents/{id}/pictures/2"));

    // The list: metadata, no image.
    let resp = send(&app, Method::GET, &format!("/v1/documents/{id}/pictures")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let list = json(resp).await;
    assert_eq!(list["object"], "list");
    assert_eq!(list["document_id"], id);
    assert_eq!(list["data"].as_array().unwrap().len(), 2);
    assert_eq!(list["data"][1]["index"], 2);
    assert_eq!(list["data"][1]["page"], 4);
    assert_eq!(list["data"][1]["mime"], "image/png");
    assert_eq!(list["data"][1]["bbox"]["r"], 300.0);
    assert_eq!(list["data"][1]["caption"], "Figure 1");
    assert_eq!(list["data"][1]["ref"], "#/pictures/0");
    assert!(list["data"][1].get("image").is_none());
    assert!(list["expires_at"].as_str().unwrap().ends_with('Z'));

    // One picture as JSON carries the base64 image.
    let resp = send(&app, Method::GET, &urls[1]).await;
    assert_eq!(resp.status(), StatusCode::OK);
    let pic = json(resp).await;
    assert_eq!(pic["object"], "document.picture");
    assert_eq!(pic["index"], 2);
    assert_eq!(pic["image"], PNG_1X1);

    // ... or as the image itself.
    let resp = send(&app, Method::GET, &format!("{}?format=image", urls[0])).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()[header::CONTENT_TYPE], "image/png");
    let bytes = body(resp).await;
    assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");

    let resp = send(&app, Method::GET, &format!("{}?format=bmp", urls[0])).await;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    // Unknown pictures and documents are 404s in the OpenAI error shape.
    let resp = send(&app, Method::GET, &format!("/v1/documents/{id}/pictures/3")).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert_eq!(json(resp).await["error"]["type"], "not_found");
    let resp = send(&app, Method::GET, "/v1/documents/doc-nope/pictures").await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);

    // DELETE releases them; a second DELETE finds nothing.
    let resp = send(&app, Method::DELETE, &format!("/v1/documents/{id}")).await;
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(json(resp).await["deleted"], true);
    let resp = send(&app, Method::GET, &urls[0]).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    let resp = send(&app, Method::DELETE, &format!("/v1/documents/{id}")).await;
    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    assert!(!store.dir().join(&id).exists());
    let _ = std::fs::remove_dir_all(store.dir());
}

/// Multipart uploads above axum's 2 MB default reach the handlers: the documents, transcription
/// and image-edit routes rebuild the request to read the form, and the rebuilt request must keep
/// the router's `DefaultBodyLimit` (`limits.max_body_bytes`). Before the fix all three answered
/// "Error parsing `multipart/form-data` request" for a 3 MB file.
#[tokio::test]
async fn multipart_uploads_above_two_megabytes_are_read() {
    let state = state();
    state.ready.store(true, std::sync::atomic::Ordering::SeqCst);
    let app = build_router(state, &[]);
    let boundary = "hypnos-boundary";
    let big = vec![b'x'; 3 << 20];
    for (path, part) in [
        ("/v1/documents/convert", "file"),
        ("/v1/audio/transcriptions", "file"),
        ("/v1/images/edits", "image"),
    ] {
        let mut form = Vec::new();
        for (name, value) in [("model", "no-such-model"), ("prompt", "x")] {
            form.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
                    .as_bytes(),
            );
        }
        form.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{part}\"; filename=\"big.bin\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes(),
        );
        form.extend_from_slice(&big);
        form.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        let req = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(
                header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(Body::from(form))
            .unwrap();
        let resp = app.clone().oneshot(req).await.expect("response");
        let text = String::from_utf8_lossy(&body(resp).await).to_string();
        assert!(
            !text.contains("Error parsing `multipart/form-data` request"),
            "{path}: {text}"
        );
    }
}
