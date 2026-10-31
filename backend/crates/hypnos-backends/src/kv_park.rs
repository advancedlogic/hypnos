//! Parked KV state (MEM-4, `docs/pagination.md`). When a llama.cpp model leaves Tier 1, its
//! worker saves each used slot's sequence state here instead of losing it with the context; when
//! the model comes back with the same context layout, the new worker restores the slots before it
//! takes requests, and prefix reuse carries on as if the model had never left. Off unless
//! `hardware.kv_park` (or `POST /v1/memory/kv-park`) turns it on; `max_bytes` caps the host memory
//! it may hold, the least recently parked models going first.

use std::sync::Mutex;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::engine::ParkedContext;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ParkedModelStatus {
    pub model: String,
    pub slots: usize,
    /// Positions held across the parked slots.
    pub tokens: u64,
    pub bytes: u64,
    pub parked_secs_ago: f64,
}

/// `kv_park` in `/v1/memory/status` and the answer of `POST /v1/memory/kv-park`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KvParkStatus {
    pub enabled: bool,
    pub max_bytes: u64,
    pub used_bytes: u64,
    pub models: Vec<ParkedModelStatus>,
}

struct Entry {
    model: String,
    parked: ParkedContext,
    at: Instant,
}

struct Inner {
    enabled: bool,
    max_bytes: u64,
    /// Oldest first.
    entries: Vec<Entry>,
}

impl Inner {
    fn used(&self) -> u64 {
        self.entries.iter().map(|e| e.parked.bytes()).sum()
    }

    fn shrink_to(&mut self, limit: u64) {
        while self.used() > limit && !self.entries.is_empty() {
            let gone = self.entries.remove(0);
            tracing::info!(model = %gone.model, bytes = gone.parked.bytes(), "dropped parked KV state (kv_park_bytes)");
        }
    }
}

pub struct KvParkPool {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for KvParkPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.status();
        f.debug_struct("KvParkPool")
            .field("enabled", &s.enabled)
            .field("max_bytes", &s.max_bytes)
            .field("used_bytes", &s.used_bytes)
            .finish()
    }
}

impl KvParkPool {
    pub fn new(enabled: bool, max_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(Inner {
                enabled,
                max_bytes,
                entries: Vec::new(),
            }),
        }
    }

    /// Off, nothing parked (what `EngineOptions::default` uses).
    pub fn disabled() -> Self {
        Self::new(false, 0)
    }

    pub fn enabled(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.enabled && inner.max_bytes > 0
    }

    /// Turns parking on or off and changes the cap. Turning it off drops every parked state;
    /// a lower cap drops the oldest until the rest fits.
    pub fn configure(&self, enabled: Option<bool>, max_bytes: Option<u64>) -> KvParkStatus {
        {
            let mut inner = self.inner.lock().unwrap();
            if let Some(e) = enabled {
                inner.enabled = e;
            }
            if let Some(m) = max_bytes {
                inner.max_bytes = m;
            }
            if !inner.enabled {
                inner.entries.clear();
            } else {
                let limit = inner.max_bytes;
                inner.shrink_to(limit);
            }
        }
        self.status()
    }

    /// Drops what is parked for `model` (it was disabled, or its context will not match).
    pub fn forget(&self, model: &str) {
        self.inner
            .lock()
            .unwrap()
            .entries
            .retain(|e| e.model != model);
    }

    pub fn status(&self) -> KvParkStatus {
        let inner = self.inner.lock().unwrap();
        KvParkStatus {
            enabled: inner.enabled,
            max_bytes: inner.max_bytes,
            used_bytes: inner.used(),
            models: inner
                .entries
                .iter()
                .map(|e| ParkedModelStatus {
                    model: e.model.clone(),
                    slots: e.parked.slots.len(),
                    tokens: e.parked.tokens(),
                    bytes: e.parked.bytes(),
                    parked_secs_ago: e.at.elapsed().as_secs_f64(),
                })
                .collect(),
        }
    }

    /// Keeps `parked` for `model`, replacing what it had, evicting the oldest others to fit.
    /// Dropped when parking is off or the state alone is larger than the cap.
    pub(crate) fn put(&self, model: &str, parked: ParkedContext) {
        let mut inner = self.inner.lock().unwrap();
        inner.entries.retain(|e| e.model != model);
        let bytes = parked.bytes();
        if !inner.enabled || bytes > inner.max_bytes {
            if inner.enabled {
                tracing::info!(
                    model,
                    bytes,
                    max = inner.max_bytes,
                    "KV state larger than kv_park_bytes; not parked"
                );
            }
            return;
        }
        let limit = inner.max_bytes - bytes;
        inner.shrink_to(limit);
        tracing::info!(
            model,
            bytes,
            slots = parked.slots.len(),
            tokens = parked.tokens(),
            "parked KV state"
        );
        inner.entries.push(Entry {
            model: model.to_string(),
            parked,
            at: Instant::now(),
        });
    }

    /// Hands back what was parked for `model` (it leaves the pool either way).
    pub(crate) fn take(&self, model: &str) -> Option<ParkedContext> {
        let mut inner = self.inner.lock().unwrap();
        if !inner.enabled {
            return None;
        }
        let i = inner.entries.iter().position(|e| e.model == model)?;
        Some(inner.entries.remove(i).parked)
    }
}

impl Default for KvParkPool {
    fn default() -> Self {
        Self::disabled()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{ParkSignature, ParkedSlot};

    fn parked(bytes: usize, tokens: i32) -> ParkedContext {
        ParkedContext {
            signature: ParkSignature {
                weights: "w.gguf".into(),
                n_ctx: 4096,
                slots: 1,
                kv: "F16".into(),
            },
            slots: vec![ParkedSlot::for_test(bytes, tokens)],
        }
    }

    #[test]
    fn off_by_default_and_parks_nothing() {
        let pool = KvParkPool::disabled();
        pool.put("m", parked(10, 5));
        assert!(pool.take("m").is_none());
        assert_eq!(pool.status().used_bytes, 0);
    }

    #[test]
    fn the_cap_evicts_the_oldest_and_take_removes() {
        let pool = KvParkPool::new(true, 100);
        pool.put("a", parked(60, 6));
        pool.put("b", parked(30, 3));
        pool.put("c", parked(50, 5)); // 60 + 30 + 50 > 100: "a" goes
        let s = pool.status();
        assert_eq!(
            s.models
                .iter()
                .map(|m| m.model.as_str())
                .collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        assert_eq!(s.used_bytes, 80);
        assert!(pool.take("b").is_some());
        assert!(pool.take("b").is_none());
        pool.put("big", parked(101, 1)); // larger than the cap alone
        assert!(pool.status().models.iter().all(|m| m.model != "big"));
    }

    #[test]
    fn turning_it_off_or_lowering_the_cap_drops_states() {
        let pool = KvParkPool::new(true, 100);
        pool.put("a", parked(40, 4));
        pool.put("b", parked(40, 4));
        let s = pool.configure(None, Some(50));
        assert_eq!(s.models.len(), 1);
        assert_eq!(s.models[0].model, "b");
        let s = pool.configure(Some(false), None);
        assert!(!s.enabled && s.models.is_empty());
        pool.put("c", parked(10, 1));
        assert!(pool.take("c").is_none());
        pool.configure(Some(true), None);
        pool.put("c", parked(10, 1));
        pool.forget("c");
        assert!(pool.take("c").is_none());
    }
}
