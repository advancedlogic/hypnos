//! Errors, one kind per error `type` of `docs/api.md` section 4.

use std::fmt;
use std::time::Duration;

/// What an error response said, plus where it came from.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    /// `error.type`, e.g. `model_busy`.
    pub kind: Option<String>,
    /// `error.code`, e.g. `requests_per_minute`, `invalid_api_key`.
    pub code: Option<String>,
    /// `x-request-id`; quote it when reporting a problem.
    pub request_id: Option<String>,
    /// `Retry-After`.
    pub retry_after: Option<Duration>,
    /// The full JSON body, when there was one.
    pub body: Option<serde_json::Value>,
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP {}", self.status)?;
        if let Some(kind) = &self.kind {
            write!(f, " {kind}")?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(id) = &self.request_id {
            write!(f, " (request {id})")?;
        }
        Ok(())
    }
}

/// Everything the client returns as an error. The server's details are boxed to keep
/// `Result` small; read them through [`Error::api`] or by matching (`Error::NotFound(e)`).
#[derive(Debug)]
pub enum Error {
    /// 400 `invalid_request_error`, or 413.
    BadRequest(Box<ApiError>),
    /// 401: missing or unknown API key.
    Authentication(Box<ApiError>),
    /// 403 `permission_denied`: changing the model pool with a key that is not an admin key.
    PermissionDenied(Box<ApiError>),
    /// 404 `model_not_found`, `model_disabled` or `not_found`.
    NotFound(Box<ApiError>),
    /// 429 `model_busy`.
    ModelBusy(Box<ApiError>),
    /// 429 `rate_limit_error` (a per-key limit; `code` says which).
    RateLimit(Box<ApiError>),
    /// 503 `model_not_available`: still staging, or a configuration problem.
    ServiceUnavailable(Box<ApiError>),
    /// 507: the model does not fit, or the engine ran out of device memory.
    InsufficientMemory(Box<ApiError>),
    /// 408, 504, or no response within the client's timeout (status 408).
    Timeout(Box<ApiError>),
    /// 500 `server_error` or another 5xx.
    Server(Box<ApiError>),
    /// Any other status.
    Api(Box<ApiError>),
    /// A stream that had started (HTTP 200) ended with an error event.
    Stream(Box<ApiError>),
    /// The server could not be reached, or the connection broke.
    Connection(String),
    /// A response body that is not what the API documents.
    Decode(String),
    /// A request the client refuses to send (e.g. neither `file` nor `url`).
    InvalidArgument(String),
    /// `wait_until_ready` gave up.
    NotReady(Duration),
}

impl Error {
    /// The response details, for the variants that come from the server.
    pub fn api(&self) -> Option<&ApiError> {
        match self {
            Self::BadRequest(e)
            | Self::Authentication(e)
            | Self::PermissionDenied(e)
            | Self::NotFound(e)
            | Self::ModelBusy(e)
            | Self::RateLimit(e)
            | Self::ServiceUnavailable(e)
            | Self::InsufficientMemory(e)
            | Self::Timeout(e)
            | Self::Server(e)
            | Self::Api(e)
            | Self::Stream(e) => Some(e),
            _ => None,
        }
    }

    pub fn status(&self) -> Option<u16> {
        self.api().map(|e| e.status)
    }

    pub fn request_id(&self) -> Option<&str> {
        self.api().and_then(|e| e.request_id.as_deref())
    }

    pub fn retry_after(&self) -> Option<Duration> {
        self.api().and_then(|e| e.retry_after)
    }

    pub(crate) fn from_api(e: ApiError) -> Self {
        let e = Box::new(e);
        match e.status {
            400 | 413 => Self::BadRequest(e),
            401 => Self::Authentication(e),
            403 => Self::PermissionDenied(e),
            404 => Self::NotFound(e),
            429 if e.kind.as_deref() == Some("rate_limit_error") => Self::RateLimit(e),
            429 => Self::ModelBusy(e),
            503 => Self::ServiceUnavailable(e),
            507 => Self::InsufficientMemory(e),
            408 | 504 => Self::Timeout(e),
            s if s >= 500 => Self::Server(e),
            _ => Self::Api(e),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stream(e) => write!(f, "stream failed: {e}"),
            Self::Connection(m) => write!(f, "connection error: {m}"),
            Self::Decode(m) => write!(f, "unexpected response: {m}"),
            Self::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Self::NotReady(t) => write!(f, "the server was not ready within {} s", t.as_secs()),
            other => match other.api() {
                Some(e) => e.fmt(f),
                None => Ok(()),
            },
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T, E = Error> = std::result::Result<T, E>;
