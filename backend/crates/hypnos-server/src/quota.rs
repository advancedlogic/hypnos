//! Per-key limits for the `/v1/*` API (`[[server.keys]]`, `[server.api_key_defaults]`): requests
//! per minute, concurrent requests and tokens per UTC day. A key without a limit set is unlimited
//! there. `require_api_key` admits a request through [`KeyLimiter::admit`] and holds the
//! concurrency permit for the life of the response body; token usage is added by the handlers
//! (`observe_request` for plain responses, the chat stream when it ends).

use hypnos_core::config::ApiKeyLimits;
use parking_lot::Mutex;
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

const DAY_SECS: u64 = 86_400;

/// Why a key's request was turned away.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// `requests_per_minute` reached; `retry_after` seconds until the window turns.
    RequestsPerMinute { limit: u32, retry_after: u64 },
    /// `max_concurrent` requests are in flight.
    Concurrent { limit: u32 },
    /// `tokens_per_day` spent; `retry_after` seconds until the UTC day turns.
    TokensPerDay {
        limit: u64,
        used: u64,
        retry_after: u64,
    },
}

impl Rejection {
    pub fn reason(&self) -> &'static str {
        match self {
            Self::RequestsPerMinute { .. } => "requests_per_minute",
            Self::Concurrent { .. } => "max_concurrent",
            Self::TokensPerDay { .. } => "tokens_per_day",
        }
    }

    pub fn retry_after_secs(&self) -> u64 {
        match self {
            Self::RequestsPerMinute { retry_after, .. } => (*retry_after).max(1),
            Self::Concurrent { .. } => 1,
            Self::TokensPerDay { retry_after, .. } => (*retry_after).max(1),
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::RequestsPerMinute { limit, retry_after } => format!(
                "Rate limit reached for this API key: {limit} requests per minute. Retry in {retry_after} s."
            ),
            Self::Concurrent { limit } => format!(
                "Too many concurrent requests for this API key (limit {limit}). Retry when one finishes."
            ),
            Self::TokensPerDay {
                limit,
                used,
                retry_after,
            } => format!(
                "Daily token quota reached for this API key: {used} of {limit} tokens used. Resets in {retry_after} s (UTC midnight)."
            ),
        }
    }
}

struct KeyState {
    limits: ApiKeyLimits,
    /// `(minute index, requests admitted in it)`.
    minute: Mutex<(u64, u32)>,
    /// `(UTC day index, tokens spent in it)`.
    day: Mutex<(u64, u64)>,
    concurrent: Option<Arc<Semaphore>>,
    admitted: AtomicU64,
    rejected_rpm: AtomicU64,
    rejected_concurrent: AtomicU64,
    rejected_tokens: AtomicU64,
    tokens_total: AtomicU64,
}

/// One key's counters, for `/metrics`.
#[derive(Debug, Clone)]
pub struct KeyStats {
    pub label: String,
    pub admitted: u64,
    pub rejected_requests_per_minute: u64,
    pub rejected_max_concurrent: u64,
    pub rejected_tokens_per_day: u64,
    pub tokens_total: u64,
    pub tokens_today: u64,
    pub limits: ApiKeyLimits,
}

/// The limiter for every configured key, by label.
#[derive(Default)]
pub struct KeyLimiter {
    keys: HashMap<String, KeyState>,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl KeyLimiter {
    /// `(label, limits)` for every key; a label that appears twice keeps its first limits.
    pub fn new(keys: impl IntoIterator<Item = (String, ApiKeyLimits)>) -> Self {
        let mut map = HashMap::new();
        for (label, limits) in keys {
            map.entry(label).or_insert_with(|| KeyState {
                concurrent: (limits.max_concurrent > 0)
                    .then(|| Arc::new(Semaphore::new(limits.max_concurrent as usize))),
                limits,
                minute: Mutex::new((0, 0)),
                day: Mutex::new((0, 0)),
                admitted: AtomicU64::new(0),
                rejected_rpm: AtomicU64::new(0),
                rejected_concurrent: AtomicU64::new(0),
                rejected_tokens: AtomicU64::new(0),
                tokens_total: AtomicU64::new(0),
            });
        }
        Self { keys: map }
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Admits one request for `label` now. `Ok(None)` when the key is unknown or has no limits.
    pub fn admit(&self, label: &str) -> Result<Option<OwnedSemaphorePermit>, Rejection> {
        self.admit_at(label, now_secs())
    }

    /// [`admit`](Self::admit) at a given Unix time (tests).
    pub fn admit_at(
        &self,
        label: &str,
        now: u64,
    ) -> Result<Option<OwnedSemaphorePermit>, Rejection> {
        let Some(k) = self.keys.get(label) else {
            return Ok(None);
        };
        if k.limits.tokens_per_day > 0 {
            let day = now / DAY_SECS;
            let used = {
                let mut d = k.day.lock();
                if d.0 != day {
                    *d = (day, 0);
                }
                d.1
            };
            if used >= k.limits.tokens_per_day {
                k.rejected_tokens.fetch_add(1, Ordering::Relaxed);
                return Err(Rejection::TokensPerDay {
                    limit: k.limits.tokens_per_day,
                    used,
                    retry_after: (day + 1) * DAY_SECS - now,
                });
            }
        }
        if k.limits.requests_per_minute > 0 {
            let minute = now / 60;
            let mut m = k.minute.lock();
            if m.0 != minute {
                *m = (minute, 0);
            }
            if m.1 >= k.limits.requests_per_minute {
                k.rejected_rpm.fetch_add(1, Ordering::Relaxed);
                return Err(Rejection::RequestsPerMinute {
                    limit: k.limits.requests_per_minute,
                    retry_after: (minute + 1) * 60 - now,
                });
            }
            m.1 += 1;
        }
        let permit = match &k.concurrent {
            Some(sem) => match sem.clone().try_acquire_owned() {
                Ok(p) => Some(p),
                Err(_) => {
                    k.rejected_concurrent.fetch_add(1, Ordering::Relaxed);
                    return Err(Rejection::Concurrent {
                        limit: k.limits.max_concurrent,
                    });
                }
            },
            None => None,
        };
        k.admitted.fetch_add(1, Ordering::Relaxed);
        Ok(permit)
    }

    /// Charges `tokens` (prompt + completion) to `label`'s day.
    pub fn add_tokens(&self, label: &str, tokens: u64) {
        self.add_tokens_at(label, tokens, now_secs());
    }

    pub fn add_tokens_at(&self, label: &str, tokens: u64, now: u64) {
        let Some(k) = self.keys.get(label) else {
            return;
        };
        if tokens == 0 {
            return;
        }
        k.tokens_total.fetch_add(tokens, Ordering::Relaxed);
        let day = now / DAY_SECS;
        let mut d = k.day.lock();
        if d.0 != day {
            *d = (day, 0);
        }
        d.1 += tokens;
    }

    pub fn stats(&self) -> Vec<KeyStats> {
        self.stats_at(now_secs())
    }

    pub fn stats_at(&self, now: u64) -> Vec<KeyStats> {
        let day = now / DAY_SECS;
        let mut out: Vec<KeyStats> = self
            .keys
            .iter()
            .map(|(label, k)| {
                let d = k.day.lock();
                KeyStats {
                    label: label.clone(),
                    admitted: k.admitted.load(Ordering::Relaxed),
                    rejected_requests_per_minute: k.rejected_rpm.load(Ordering::Relaxed),
                    rejected_max_concurrent: k.rejected_concurrent.load(Ordering::Relaxed),
                    rejected_tokens_per_day: k.rejected_tokens.load(Ordering::Relaxed),
                    tokens_total: k.tokens_total.load(Ordering::Relaxed),
                    tokens_today: if d.0 == day { d.1 } else { 0 },
                    limits: k.limits.clone(),
                }
            })
            .collect();
        out.sort_by(|a, b| a.label.cmp(&b.label));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limiter(rpm: u32, conc: u32, tpd: u64) -> KeyLimiter {
        KeyLimiter::new([(
            "k".to_string(),
            ApiKeyLimits {
                requests_per_minute: rpm,
                max_concurrent: conc,
                tokens_per_day: tpd,
            },
        )])
    }

    #[test]
    fn requests_per_minute_window_turns_with_the_clock() {
        let l = limiter(2, 0, 0);
        let t = 1_000_000; // some minute boundary + 40 s
        let t = t - t % 60 + 40;
        assert!(l.admit_at("k", t).is_ok());
        assert!(l.admit_at("k", t + 1).is_ok());
        let err = l.admit_at("k", t + 2).unwrap_err();
        assert_eq!(
            err,
            Rejection::RequestsPerMinute {
                limit: 2,
                retry_after: 18
            }
        );
        assert_eq!(err.retry_after_secs(), 18);
        assert!(l.admit_at("k", t + 20).is_ok(), "next minute");
        let s = &l.stats_at(t + 20)[0];
        assert_eq!((s.admitted, s.rejected_requests_per_minute), (3, 1));
        // Unknown labels and keys without limits are never limited.
        assert!(matches!(l.admit_at("other", t), Ok(None)));
        assert!(matches!(limiter(0, 0, 0).admit_at("k", t), Ok(None)));
    }

    #[test]
    fn concurrency_permit_is_held_until_dropped() {
        let l = limiter(0, 1, 0);
        let permit = l.admit("k").unwrap();
        assert!(permit.is_some());
        assert_eq!(
            l.admit("k").unwrap_err(),
            Rejection::Concurrent { limit: 1 }
        );
        drop(permit);
        assert!(l.admit("k").is_ok());
        assert_eq!(l.stats()[0].rejected_max_concurrent, 1);
    }

    #[test]
    fn daily_token_quota_resets_at_utc_midnight() {
        let l = limiter(0, 0, 100);
        let day_start = 20_000 * DAY_SECS;
        let t = day_start + 3600;
        assert!(l.admit_at("k", t).is_ok());
        l.add_tokens_at("k", 60, t);
        l.add_tokens_at("k", 40, t);
        let err = l.admit_at("k", t + 1).unwrap_err();
        assert_eq!(
            err,
            Rejection::TokensPerDay {
                limit: 100,
                used: 100,
                retry_after: DAY_SECS - 3601
            }
        );
        assert!(err.message().contains("100 of 100"));
        let s = &l.stats_at(t)[0];
        assert_eq!(
            (s.tokens_today, s.tokens_total, s.rejected_tokens_per_day),
            (100, 100, 1)
        );
        // The next day starts clean; the lifetime total keeps counting.
        assert!(l.admit_at("k", day_start + DAY_SECS + 1).is_ok());
        assert_eq!(l.stats_at(day_start + DAY_SECS + 1)[0].tokens_today, 0);
    }
}
