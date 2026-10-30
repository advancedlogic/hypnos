//! Bearer-key authentication for the `/v1/*` API (`server.api_keys`, `[[server.keys]]`,
//! `HYPNOS_API_KEYS`) and the per-key limits behind it (`quota.rs`).
//!
//! Keys are kept only as SHA-256 digests. A presented key is hashed and compared against every
//! digest in constant time, so neither the comparison nor the key length shows up in timing, and
//! no plaintext key stays in memory after startup or reaches a log line. Each key has a label
//! (`[[server.keys]].label`, or `key-<8 hex of the digest>` for a plain `api_keys` entry) that
//! names it in the request log, on `/metrics` and in audit lines.

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use futures::StreamExt;
use hypnos_core::config::{ApiKeyLimits, ApiKeyProfile};
use serde_json::json;
use subtle::{Choice, ConstantTimeEq};
use tokio::sync::OwnedSemaphorePermit;
use tracing::warn;

use crate::quota::Rejection;
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

/// The label of the key a request presented; put on the request (for handlers) and on the
/// response (for `observe_request`) by [`require_api_key`].
#[derive(Debug, Clone)]
pub struct ApiKeyLabel(pub String);

struct KeyEntry {
    digest: [u8; 32],
    label: String,
    limits: ApiKeyLimits,
}

/// The configured keys (as digests), their labels and limits, and which routes they guard.
pub struct ApiAuth {
    entries: Vec<KeyEntry>,
    metrics_require_auth: bool,
}

impl ApiAuth {
    /// `keys` are plain `server.api_keys` (labelled by digest, with the default limits);
    /// `profiles` are `[[server.keys]]` entries with their own labels and limits.
    pub fn new(
        keys: &[String],
        profiles: &[ApiKeyProfile],
        defaults: &ApiKeyLimits,
        metrics_require_auth: bool,
    ) -> Self {
        let mut entries: Vec<KeyEntry> = keys
            .iter()
            .map(|k| k.trim())
            .filter(|k| !k.is_empty())
            .map(|k| {
                let digest = sha256(k.as_bytes());
                KeyEntry {
                    digest,
                    label: format!("key-{}", hex8(&digest)),
                    limits: defaults.clone(),
                }
            })
            .collect();
        entries.extend(
            profiles
                .iter()
                .filter(|p| !p.key.trim().is_empty())
                .map(|p| KeyEntry {
                    digest: sha256(p.key.trim().as_bytes()),
                    label: p.label.trim().to_string(),
                    limits: p.limits(defaults),
                }),
        );
        Self {
            entries,
            metrics_require_auth,
        }
    }

    /// True when at least one key is configured.
    pub fn enabled(&self) -> bool {
        !self.entries.is_empty()
    }

    pub fn key_count(&self) -> usize {
        self.entries.len()
    }

    /// `(label, limits)` of every key, for the limiter.
    pub fn key_limits(&self) -> Vec<(String, ApiKeyLimits)> {
        self.entries
            .iter()
            .map(|e| (e.label.clone(), e.limits.clone()))
            .collect()
    }

    /// `/v1/*` always; `/metrics` with `metrics_require_auth`; `/health` and `/ready` never.
    fn protects(&self, path: &str) -> bool {
        path.starts_with("/v1/") || (self.metrics_require_auth && path == "/metrics")
    }

    /// Checks a request's `Authorization` header for `path`. `Ok(None)` when the path is open or
    /// no keys are configured, `Ok(Some(label))` for the key that matched.
    pub fn check(&self, path: &str, headers: &HeaderMap) -> Result<Option<String>, AuthFailure> {
        if !self.enabled() || !self.protects(path) {
            return Ok(None);
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
        let matches: Vec<Choice> = self
            .entries
            .iter()
            .map(|e| e.digest.ct_eq(&presented))
            .collect();
        let any = matches.iter().fold(Choice::from(0), |acc, m| acc | *m);
        if !bool::from(any) {
            return Err(AuthFailure::Invalid);
        }
        let index = matches
            .iter()
            .position(|m| bool::from(*m))
            .expect("a key matched");
        Ok(Some(self.entries[index].label.clone()))
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

fn hex8(digest: &[u8; 32]) -> String {
    digest[..4].iter().map(|b| format!("{b:02x}")).collect()
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

/// 429 in the OpenAI error shape (`rate_limit_error`), with `Retry-After`.
fn too_many(rejection: &Rejection) -> Response {
    let body = json!({
        "error": {
            "message": rejection.message(),
            "type": "rate_limit_error",
            "param": null,
            "code": rejection.reason(),
        }
    });
    let mut resp = (StatusCode::TOO_MANY_REQUESTS, Json(body)).into_response();
    if let Ok(v) = HeaderValue::from_str(&rejection.retry_after_secs().to_string()) {
        resp.headers_mut().insert(header::RETRY_AFTER, v);
    }
    resp
}

/// Keeps a concurrency permit alive for as long as the response body streams.
fn hold_permit(resp: Response, permit: OwnedSemaphorePermit) -> Response {
    let (parts, body) = resp.into_parts();
    let guarded = futures::stream::unfold(
        (body.into_data_stream(), permit),
        |(mut stream, permit)| async move { stream.next().await.map(|item| (item, (stream, permit))) },
    );
    Response::from_parts(parts, Body::from_stream(guarded))
}

/// Middleware: lets the request through or answers 401, then applies the key's limits (429 with
/// `Retry-After`). Rejections are logged with the route, the key's label and the reason, never
/// with the presented key.
pub async fn require_api_key(
    State(state): State<AppState>,
    mut req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path().to_string();
    let request_id = req
        .extensions()
        .get::<RequestId>()
        .map(|r| r.0.clone())
        .unwrap_or_default();
    let label = match state.auth.check(&path, req.headers()) {
        Ok(label) => label,
        Err(failure) => {
            warn!(
                route = %path,
                method = %req.method(),
                reason = failure.reason(),
                request_id = %request_id,
                "API request rejected: authentication failed"
            );
            return unauthorized(failure);
        }
    };
    let Some(label) = label else {
        return next.run(req).await;
    };
    let permit = if path.starts_with("/v1/") {
        match state.quota.admit(&label) {
            Ok(permit) => permit,
            Err(rejection) => {
                warn!(
                    route = %path,
                    method = %req.method(),
                    key = %label,
                    reason = rejection.reason(),
                    request_id = %request_id,
                    "API request rejected: key limit"
                );
                let mut resp = too_many(&rejection);
                resp.extensions_mut().insert(ApiKeyLabel(label));
                return resp;
            }
        }
    } else {
        None
    };
    req.extensions_mut().insert(ApiKeyLabel(label.clone()));
    let mut resp = next.run(req).await;
    resp.extensions_mut().insert(ApiKeyLabel(label));
    match permit {
        Some(permit) => hold_permit(resp, permit),
        None => resp,
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

    fn plain(keys: &[&str], metrics: bool) -> ApiAuth {
        let keys: Vec<String> = keys.iter().map(|k| k.to_string()).collect();
        ApiAuth::new(&keys, &[], &ApiKeyLimits::default(), metrics)
    }

    #[test]
    fn no_keys_means_open() {
        let auth = plain(&[], false);
        assert!(!auth.enabled());
        assert_eq!(auth.check("/v1/models", &headers(None)), Ok(None));
    }

    #[test]
    fn v1_routes_need_a_matching_bearer_key() {
        let auth = plain(&["key-one", "key-two"], false);
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
        let one = auth
            .check(p, &headers(Some("Bearer key-one")))
            .unwrap()
            .unwrap();
        let two = auth
            .check(p, &headers(Some("bearer  key-two ")))
            .unwrap()
            .unwrap();
        assert!(one.starts_with("key-") && one.len() == 12, "{one}");
        assert_ne!(one, two);
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
    fn labelled_keys_carry_their_limits() {
        let defaults = ApiKeyLimits {
            requests_per_minute: 100,
            max_concurrent: 0,
            tokens_per_day: 0,
        };
        let profiles = vec![ApiKeyProfile {
            label: "team-a".into(),
            key: "secret-team-a".into(),
            requests_per_minute: Some(5),
            max_concurrent: Some(2),
            tokens_per_day: None,
        }];
        let auth = ApiAuth::new(&["plain-key".into()], &profiles, &defaults, false);
        assert_eq!(auth.key_count(), 2);
        assert_eq!(
            auth.check("/v1/models", &headers(Some("Bearer secret-team-a"))),
            Ok(Some("team-a".into()))
        );
        let limits = auth.key_limits();
        let team = limits.iter().find(|(l, _)| l == "team-a").unwrap();
        assert_eq!(
            (
                team.1.requests_per_minute,
                team.1.max_concurrent,
                team.1.tokens_per_day
            ),
            (5, 2, 0)
        );
        let plain = limits.iter().find(|(l, _)| l.starts_with("key-")).unwrap();
        assert_eq!(
            plain.1.requests_per_minute, 100,
            "plain keys take the defaults"
        );
    }

    #[test]
    fn probes_stay_open_and_metrics_is_opt_in() {
        let open_metrics = plain(&["k"], false);
        for path in ["/health", "/ready", "/metrics"] {
            assert_eq!(open_metrics.check(path, &headers(None)), Ok(None), "{path}");
        }
        let closed_metrics = plain(&["k"], true);
        assert_eq!(
            closed_metrics.check("/metrics", &headers(None)),
            Err(AuthFailure::Missing)
        );
        assert_eq!(closed_metrics.check("/health", &headers(None)), Ok(None));
    }
}
