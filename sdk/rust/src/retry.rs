//! The retry policy of `docs/api.md` section 17.

use std::time::Duration;

use crate::error::Error;

/// When and how long to wait before sending a request again.
///
/// Only failures that arrive before any response body are retried: a stream that produced
/// output is never replayed. `max_retries = 0` ([`RetryPolicy::none`]) turns retrying off.
#[derive(Debug, Clone, PartialEq)]
pub struct RetryPolicy {
    pub max_retries: u32,
    /// First wait after 429 `model_busy` (no `Retry-After`); doubles each attempt.
    pub busy_backoff: Duration,
    /// 503 without `Retry-After`.
    pub unavailable_wait: Duration,
    /// First wait after 507 (another model holds the memory); doubles each attempt.
    pub memory_backoff: Duration,
    /// First wait after 504 or a connection error; doubles each attempt.
    pub backoff: Duration,
    /// Upper bound of any single wait.
    pub max_wait: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 4,
            busy_backoff: Duration::from_millis(250),
            unavailable_wait: Duration::from_secs(5),
            memory_backoff: Duration::from_secs(2),
            backoff: Duration::from_secs(1),
            max_wait: Duration::from_secs(60),
        }
    }
}

impl RetryPolicy {
    /// No retries.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Self::default()
        }
    }

    /// How long to wait before attempt `attempt + 1` (`attempt` counts from 0), or `None`.
    pub fn delay(&self, error: &Error, attempt: u32) -> Option<Duration> {
        if attempt >= self.max_retries {
            return None;
        }
        let doubled = |base: Duration| base.saturating_mul(1u32 << attempt.min(16));
        let wait = match error {
            Error::ModelBusy(_) => doubled(self.busy_backoff),
            Error::RateLimit(e) | Error::ServiceUnavailable(e) => {
                e.retry_after.unwrap_or(self.unavailable_wait)
            }
            Error::InsufficientMemory(e) => e
                .retry_after
                .unwrap_or_default()
                .max(doubled(self.memory_backoff)),
            Error::Timeout(e) if e.status == 504 => doubled(self.backoff),
            Error::Connection(_) => doubled(self.backoff),
            Error::Server(e) if e.status == 500 && attempt == 0 => self.backoff,
            _ => return None,
        };
        // Up to 10 % jitter so that many clients do not come back in lockstep.
        let jitter = 1.0 + rand::random::<f64>() * 0.1;
        Some(wait.mul_f64(jitter).min(self.max_wait))
    }
}
