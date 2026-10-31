//! Errors, retries, readiness, auth and the per-model bound.

mod common;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::*;
use futures::StreamExt;
use hypnos_client::*;

fn hi() -> ChatRequest {
    ChatRequest::new(CHAT, vec![ChatMessage::user("hi")])
}

fn no_retry(server: &MockServer) -> Client {
    Client::builder()
        .base_url(&server.url)
        .no_api_key()
        .retry(RetryPolicy::none())
        .build()
        .unwrap()
}

#[tokio::test]
async fn unknown_model_is_not_found() {
    let server = MockServer::start();
    let err = server
        .client()
        .chat(&ChatRequest::new(
            "no-such-model",
            vec![ChatMessage::user("hi")],
        ))
        .await
        .unwrap_err();
    let Error::NotFound(e) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(e.status, 404);
    assert_eq!(e.kind.as_deref(), Some("model_not_found"));
    assert!(err.request_id().unwrap().starts_with("req-"));
}

#[tokio::test]
async fn bad_request() {
    let server = MockServer::start();
    let err = server
        .client()
        .chat(&ChatRequest::new(CHAT, vec![]))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::BadRequest(_)), "{err:?}");
}

#[tokio::test]
async fn busy_is_retried() {
    let server = MockServer::start();
    server.fail(EMBED, "busy", 2).await;
    assert!(!server
        .client()
        .embeddings(EMBED, "a")
        .await
        .unwrap()
        .data
        .is_empty());
}

#[tokio::test]
async fn busy_without_retries() {
    let server = MockServer::start();
    server.fail(EMBED, "busy", 1).await;
    let err = no_retry(&server).embeddings(EMBED, "a").await.unwrap_err();
    let Error::ModelBusy(e) = err else {
        panic!("{err:?}")
    };
    assert_eq!((e.status, e.kind.as_deref()), (429, Some("model_busy")));
}

#[tokio::test]
async fn memory_errors_are_retried() {
    let server = MockServer::start();
    server.fail(EMBED, "insufficient_memory", 2).await;
    assert!(!server
        .client()
        .embeddings(EMBED, "a")
        .await
        .unwrap()
        .data
        .is_empty());
}

#[tokio::test]
async fn memory_error_surfaces_after_the_retries() {
    let server = MockServer::start();
    server.fail(EMBED, "insufficient_memory", 10).await;
    let err = server.client().embeddings(EMBED, "a").await.unwrap_err();
    assert!(matches!(err, Error::InsufficientMemory(_)), "{err:?}");
    assert_eq!(err.status(), Some(507));
}

#[tokio::test]
async fn a_500_is_retried_once() {
    let server = MockServer::start();
    let c = server.client();
    server.fail(EMBED, "server_error", 1).await;
    assert!(!c.embeddings(EMBED, "a").await.unwrap().data.is_empty());
    server.fail(EMBED, "server_error", 2).await;
    let err = c.embeddings(EMBED, "a").await.unwrap_err();
    assert!(matches!(err, Error::Server(_)), "{err:?}");
}

#[tokio::test]
async fn not_ready() {
    let server = MockServer::start();
    server.set_ready(false).await;
    let c = no_retry(&server);
    assert!(!c.is_ready().await);
    let err = c.embeddings(EMBED, "a").await.unwrap_err();
    assert!(matches!(err, Error::ServiceUnavailable(_)), "{err:?}");
    assert_eq!(err.retry_after(), Some(Duration::from_secs(5)));
    // Models stay open while staging.
    assert!(!c.models().await.unwrap().data.is_empty());
    let flip = {
        let url = server.url.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(300)).await;
            reqwest::Client::new()
                .post(format!("{url}/mock/ready"))
                .json(&serde_json::json!({ "ready": true }))
                .send()
                .await
                .unwrap();
        })
    };
    c.wait_until_ready(Duration::from_secs(5), Duration::from_millis(50))
        .await
        .unwrap();
    flip.await.unwrap();
    assert!(!c.embeddings(EMBED, "a").await.unwrap().data.is_empty());
}

#[tokio::test]
async fn unreachable_server() {
    let c = Client::builder()
        .base_url("http://127.0.0.1:9")
        .retry(RetryPolicy::none())
        .build()
        .unwrap();
    let err = c.health().await.unwrap_err();
    assert!(matches!(err, Error::Connection(_)), "{err:?}");
    assert!(!c.is_ready().await);
    let err = c
        .wait_until_ready(Duration::from_millis(200), Duration::from_millis(50))
        .await
        .unwrap_err();
    assert!(matches!(err, Error::NotReady(_)), "{err:?}");
}

#[tokio::test]
async fn api_keys() {
    let server = MockServer::keyed();
    let anonymous = no_retry(&server);
    let err = anonymous.models().await.unwrap_err();
    let Error::Authentication(e) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(e.code.as_deref(), Some("invalid_api_key"));
    assert_eq!(anonymous.health().await.unwrap().status, "ok");
    let keyed = Client::builder()
        .base_url(&server.url)
        .api_key("secret")
        .build()
        .unwrap();
    assert!(!keyed.models().await.unwrap().data.is_empty());
}

#[tokio::test]
async fn model_pool_needs_an_admin_key() {
    let server = MockServer::with_args(&["--api-key", "secret", "--limited-key", "member:1000"]);
    let c = Client::builder()
        .base_url(&server.url)
        .api_key("member")
        .build()
        .unwrap();
    let err = c
        .add_model(&ModelSource::catalog("lfm3:1.7b"))
        .await
        .unwrap_err();
    let Error::PermissionDenied(e) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(e.status, 403);
    assert_eq!(e.kind.as_deref(), Some("permission_denied"));
    let err = c.remove_model(CHAT, None).await.unwrap_err();
    assert!(matches!(err, Error::PermissionDenied(_)), "{err:?}");
    assert!(!c.models().await.unwrap().data.is_empty());
}

#[tokio::test]
async fn rate_limit() {
    let server = MockServer::keyed();
    let c = Client::builder()
        .base_url(&server.url)
        .api_key("limited")
        .retry(RetryPolicy::none())
        .build()
        .unwrap();
    c.models().await.unwrap();
    let err = c.models().await.unwrap_err();
    let Error::RateLimit(e) = &err else {
        panic!("{err:?}")
    };
    assert_eq!(e.code.as_deref(), Some("requests_per_minute"));
    assert!(e.retry_after.is_some_and(|d| d > Duration::ZERO));
}

#[tokio::test]
async fn base_url_accepts_v1() {
    let server = MockServer::start();
    let c = Client::builder()
        .base_url(format!("{}/v1/", server.url))
        .build()
        .unwrap();
    assert_eq!(c.base_url(), server.url);
    assert!(!c.models().await.unwrap().data.is_empty());
}

#[tokio::test]
async fn per_model_bound() {
    let server = MockServer::start();
    let c = Client::builder()
        .base_url(&server.url)
        .no_api_key()
        .max_concurrency_per_model(1)
        .build()
        .unwrap();
    let stream = c
        .chat_stream(&ChatRequest::new(CHAT, vec![ChatMessage::user("count")]).max_tokens(1000))
        .await
        .unwrap();
    let done = Arc::new(AtomicBool::new(false));
    let second = {
        let (c, done) = (c.clone(), done.clone());
        tokio::spawn(async move {
            c.chat(&hi()).await.unwrap();
            done.store(true, Ordering::SeqCst);
        })
    };
    // The second request waits for the open stream's permit.
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(!done.load(Ordering::SeqCst));
    drop(stream);
    tokio::time::timeout(Duration::from_secs(5), second)
        .await
        .expect("the second request ran")
        .unwrap();
    // Another model is not affected by the bound.
    assert!(!c.embeddings(EMBED, "a").await.unwrap().data.is_empty());
}

#[tokio::test]
async fn cancel_returns_the_permit() {
    let server = MockServer::start();
    let c = Client::builder()
        .base_url(&server.url)
        .no_api_key()
        .max_concurrency_per_model(1)
        .build()
        .unwrap();
    let started = Instant::now();
    let mut stream = c
        .chat_stream(&ChatRequest::new(CHAT, vec![ChatMessage::user("count")]).max_tokens(2000))
        .await
        .unwrap();
    for _ in 0..3 {
        stream.next().await.unwrap().unwrap();
    }
    drop(stream);
    assert!(started.elapsed() < Duration::from_secs(5));
    assert!(!c.chat(&hi()).await.unwrap().choices.is_empty());
}

#[tokio::test]
async fn document_needs_a_readable_path() {
    let err = DocumentRequest::from_path(DOCS, "/no/such/file.pdf").unwrap_err();
    assert!(matches!(err, Error::InvalidArgument(_)), "{err:?}");
}
