//! Per-request context shared between the HTTP layer, the scheduler and the engines:
//! cooperative cancellation, a deadline, and the input limits the request must respect.

use crate::config::LimitsConfig;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cancellation flag plus deadline. Cheap to clone; all clones observe the same flag.
#[derive(Debug, Clone)]
pub struct RequestContext {
    cancelled: Arc<AtomicBool>,
    deadline: Option<Instant>,
    pub limits: LimitsConfig,
    /// Correlation id for logs (e.g. `req-3f9a…`).
    pub request_id: String,
    /// The longest wait for execution to start the client accepts (`x-hypnos-deadline-ms`, or the
    /// profile's objective with `limits.overload = "shed"`); beyond it the request is refused at
    /// admission instead of queued (PERF-10).
    pub start_deadline_ms: Option<u64>,
    /// The scheduler's estimate of this request's wait, written at admission (`u64::MAX` = none);
    /// shared with the HTTP layer for `x-hypnos-estimated-wait-ms`.
    pub estimated_wait_ms: Arc<AtomicU64>,
}

impl RequestContext {
    pub fn new(limits: LimitsConfig, timeout: Option<Duration>, request_id: String) -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
            deadline: timeout.map(|t| Instant::now() + t),
            limits,
            request_id,
            start_deadline_ms: None,
            estimated_wait_ms: Arc::new(AtomicU64::new(u64::MAX)),
        }
    }

    /// Context with default limits and no deadline (tests, internal calls).
    pub fn unbounded() -> Self {
        Self::new(LimitsConfig::default(), None, "internal".to_string())
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    pub fn deadline_passed(&self) -> bool {
        self.deadline.map(|d| Instant::now() >= d).unwrap_or(false)
    }

    /// `true` while the request should keep running.
    pub fn should_continue(&self) -> bool {
        !self.is_cancelled() && !self.deadline_passed()
    }

    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    /// Returns a guard that cancels this context when dropped (unless disarmed). Hold it in an
    /// HTTP handler so a client disconnect, which drops the handler future, stops the engine.
    pub fn cancel_on_drop(&self) -> CancelOnDrop {
        CancelOnDrop {
            ctx: self.clone(),
            armed: true,
        }
    }
}

/// See [`RequestContext::cancel_on_drop`].
#[derive(Debug)]
pub struct CancelOnDrop {
    ctx: RequestContext,
    armed: bool,
}

impl CancelOnDrop {
    /// Keep the context alive past this guard (the request completed normally).
    pub fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        if self.armed {
            self.ctx.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancel_on_drop_sets_flag_unless_disarmed() {
        let ctx = RequestContext::unbounded();
        {
            let _g = ctx.cancel_on_drop();
        }
        assert!(ctx.is_cancelled());

        let ctx2 = RequestContext::unbounded();
        let g = ctx2.cancel_on_drop();
        g.disarm();
        assert!(!ctx2.is_cancelled());
    }

    #[test]
    fn deadline_is_observed() {
        let ctx = RequestContext::new(
            LimitsConfig::default(),
            Some(Duration::from_millis(0)),
            "t".into(),
        );
        assert!(ctx.deadline_passed());
        assert!(!ctx.should_continue());
    }
}
