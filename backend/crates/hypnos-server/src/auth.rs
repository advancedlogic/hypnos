//! Bearer-key authentication for the `/v1/*` API (`server.api_keys`, `HYPNOS_API_KEYS`).
//!
//! Keys are kept only as SHA-256 digests. A presented key is hashed and compared against every
//! digest in constant time, so neither the comparison nor the key length shows up in timing, and
//! no plaintext key stays in memory after startup or reaches a log line.

use axum::{
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;
use subtle::{Choice, ConstantTimeEq};
use tracing::warn;

use crate::request_log::RequestId;
use crate::AppState;

/// Why a request was turned away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthFailure {
    /// No `Authorization` header.
    Missing,
    /// An `Authorization` header that is not `Bearer <key>`.
    Malformed,
    /// A well-formed key that matches none of the configured ones.
    Invalid,
}

impl AuthFailure {
    fn reason(self) -> &'static str {
        match self {
            Self::Missing => "missing_key",
            Self::Malformed => "malformed_header",
            Self::Invalid => "invalid_key",
        }
    }

    fn message(self) -> &'static str {
        match self {
            Self::Missing => {
                "Missing API key. Send it as `Authorization: Bearer <key>` (server.api_keys)."
            }
            Self::Malformed => "Malformed Authorization header; expected `Bearer <key>`.",
            Self::Invalid => "Incorrect API key provided.",
        }
    }
}

/// The configured keys (as digests) and which routes they guard.
pub struct ApiAuth {
    key_digests: Vec<[u8; 32]>,
    metrics_require_auth: bool,
}

impl ApiAuth {
    pub fn new(keys: &[String], metrics_require_auth: bool) -> Self {
        Self {
            key_digests: keys
                .iter()
                .map(|k| k.trim())
                .filter(|k| !k.is_empty())
                .map(|k| sha256(k.as_bytes()))
                .collect(),
            metrics_require_auth,
        }
    }

    /// True when at least one key is configured.
    pub fn enabled(&self) -> bool {
        !self.key_digests.is_empty()
    }

    pub fn key_count(&self) -> usize {
        self.key_digests.len()
    }

    /// `/v1/*` always; `/metrics` with `metrics_require_auth`; `/health` and `/ready` never.
    fn protects(&self, path: &str) -> bool {
        path.starts_with("/v1/") || (self.metrics_require_auth && path == "/metrics")
    }

    /// Checks a request's `Authorization` header for `path`.
    pub fn check(&self, path: &str, headers: &HeaderMap) -> Result<(), AuthFailure> {
        if !self.enabled() || !self.protects(path) {
            return Ok(());
        }
        let value = headers
            .get(header::AUTHORIZATION)
            .ok_or(AuthFailure::Missing)?;
        let token = value
            .to_str()
            .ok()
            .and_then(bearer_token)
            .ok_or(AuthFailure::Malformed)?;
        let presented = sha256(token.as_bytes());
        // Visit every key: the time taken does not depend on which one (if any) matched.
        let matched = self
            .key_digests
            .iter()
            .fold(Choice::from(0), |acc, d| acc | d.ct_eq(&presented));
        if bool::from(matched) {
            Ok(())
        } else {
            Err(AuthFailure::Invalid)
        }
    }
}

/// `Bearer <token>` with a case-insensitive scheme, as clients (OpenAI SDKs included) send it.
fn bearer_token(value: &str) -> Option<&str> {
    let (scheme, token) = value.trim().split_once(' ')?;
    let token = token.trim();
    (scheme.eq_ignore_ascii_case("bearer") && !token.is_empty()).then_some(token)
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    let digest = ring::digest::digest(&ring::digest::SHA256, bytes);
    let mut out = [0u8; 32];
    out.copy_from_slice(digest.as_ref());
    out
}

/// 401 in the OpenAI error shape, so OpenAI SDKs raise their usual authentication error.
fn unauthorized(failure: AuthFailure) -> Response {
    let body = json!({
        "error": {
            "message": failure.message(),
            "type": "invalid_request_error",
            "param": null,
            "code": "invalid_api_key",
        }
    });
    let mut resp = (StatusCode::UNAUTHORIZED, Json(body)).into_response();
    resp.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"hypnos\""),
    );
    resp
}

/// Middleware: lets the request through or answers 401. Rejections are logged with the route and
/// the reason, never with the presented key.
pub async fn require_api_key(State(state): State<AppState>, req: Request, next: Next) -> Response {
    match state.auth.check(req.uri().path(), req.headers()) {
        Ok(()) => next.run(req).await,
        Err(failure) => {
            let request_id = req
                .extensions()
                .get::<RequestId>()
                .map(|r| r.0.clone())
                .unwrap_or_default();
            warn!(
                route = %req.uri().path(),
                method = %req.method(),
                reason = failure.reason(),
                request_id = %request_id,
                "API request rejected: authentication failed"
            );
            unauthorized(failure)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(auth: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(v) = auth {
            h.insert(header::AUTHORIZATION, HeaderValue::from_str(v).unwrap());
        }
        h
    }

    #[test]
    fn no_keys_means_open() {
        let auth = ApiAuth::new(&[], false);
        assert!(!auth.enabled());
        assert_eq!(auth.check("/v1/models", &headers(None)), Ok(()));
    }

    #[test]
    fn v1_routes_need_a_matching_bearer_key() {
        let auth = ApiAuth::new(&["key-one".into(), "key-two".into()], false);
        assert_eq!(auth.key_count(), 2);
        let p = "/v1/chat/completions";
        assert_eq!(auth.check(p, &headers(None)), Err(AuthFailure::Missing));
        assert_eq!(
            auth.check(p, &headers(Some("Basic abc"))),
            Err(AuthFailure::Malformed)
        );
        assert_eq!(
            auth.check(p, &headers(Some("Bearer"))),
            Err(AuthFailure::Malformed)
        );
        assert_eq!(
            auth.check(p, &headers(Some("Bearer key-three"))),
            Err(AuthFailure::Invalid)
        );
        assert_eq!(auth.check(p, &headers(Some("Bearer key-one"))), Ok(()));
        assert_eq!(auth.check(p, &headers(Some("bearer  key-two "))), Ok(()));
        // A prefix or extension of a valid key is not valid.
        assert_eq!(
            auth.check(p, &headers(Some("Bearer key-on"))),
            Err(AuthFailure::Invalid)
        );
        assert_eq!(
            auth.check(p, &headers(Some("Bearer key-one1"))),
            Err(AuthFailure::Invalid)
        );
    }

    #[test]
    fn probes_stay_open_and_metrics_is_opt_in() {
        let open_metrics = ApiAuth::new(&["k".into()], false);
        for path in ["/health", "/ready", "/metrics"] {
            assert_eq!(open_metrics.check(path, &headers(None)), Ok(()), "{path}");
        }
        let closed_metrics = ApiAuth::new(&["k".into()], true);
        assert_eq!(
            closed_metrics.check("/metrics", &headers(None)),
            Err(AuthFailure::Missing)
        );
        assert_eq!(closed_metrics.check("/health", &headers(None)), Ok(()));
    }
}
