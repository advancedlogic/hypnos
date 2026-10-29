//! Minimal Prometheus text exposition (no external metrics crate): per-route request counters and
//! latency sums, scheduler counters, memory-tier gauges, and per-model execution histograms
//! (time to first token, prefill / decode throughput, audio real-time factors, image job time).

use hypnos_core::openai::EngineTiming;
use hypnos_memory::MemoryHierarchyStatus;
use hypnos_scheduler::SchedulerStats;
use parking_lot::Mutex;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::sync::atomic::Ordering;
use std::time::Duration;

#[derive(Debug, Default, Clone)]
struct RouteStats {
    requests: u64,
    errors_4xx: u64,
    errors_5xx: u64,
    latency_ms_sum: f64,
    latency_ms_max: f64,
}

/// Fixed-bucket cumulative histogram plus the last observed value (handy for a single bench run).
#[derive(Debug, Clone)]
struct Histogram {
    bounds: &'static [f64],
    counts: Vec<u64>,
    sum: f64,
    count: u64,
    last: f64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            counts: vec![0; bounds.len()],
            sum: 0.0,
            count: 0,
            last: 0.0,
        }
    }

    fn observe(&mut self, v: f64) {
        if !v.is_finite() {
            return;
        }
        for (i, b) in self.bounds.iter().enumerate() {
            if v <= *b {
                self.counts[i] += 1;
            }
        }
        self.sum += v;
        self.count += 1;
        self.last = v;
    }

    fn render(&self, out: &mut String, name: &str, model: &str) {
        for (i, b) in self.bounds.iter().enumerate() {
            let _ = writeln!(
                out,
                "{name}_bucket{{model=\"{model}\",le=\"{b}\"}} {}",
                self.counts[i]
            );
        }
        let _ = writeln!(
            out,
            "{name}_bucket{{model=\"{model}\",le=\"+Inf\"}} {}",
            self.count
        );
        let _ = writeln!(out, "{name}_sum{{model=\"{model}\"}} {:.3}", self.sum);
        let _ = writeln!(out, "{name}_count{{model=\"{model}\"}} {}", self.count);
        let _ = writeln!(out, "{name}_last{{model=\"{model}\"}} {:.3}", self.last);
    }
}

struct EngineMetric {
    name: &'static str,
    help: &'static str,
    bounds: &'static [f64],
}

const LATENCY_MS: &[f64] = &[
    50.0, 100.0, 200.0, 300.0, 500.0, 1000.0, 2000.0, 3000.0, 5000.0, 10000.0, 30000.0,
];
const TOKENS_PER_S: &[f64] = &[
    1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0,
];
const RTF: &[f64] = &[0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0];
const SECONDS: &[f64] = &[1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0, 600.0];

/// Every execution histogram, in exposition order. Each is keyed by model id.
const ENGINE_METRICS: &[EngineMetric] = &[
    EngineMetric {
        name: "hypnos_chat_ttft_ms",
        help: "Chat: lease granted to first visible token (ms).",
        bounds: LATENCY_MS,
    },
    EngineMetric {
        name: "hypnos_chat_queue_ms",
        help: "Chat: time waiting for the model's execution lock or slot (ms).",
        bounds: LATENCY_MS,
    },
    EngineMetric {
        name: "hypnos_chat_context_ms",
        help: "Chat: llama.cpp context creation per request (ms); 0 once contexts persist.",
        bounds: LATENCY_MS,
    },
    EngineMetric {
        name: "hypnos_chat_prefill_tokens_per_s",
        help: "Chat: prompt tokens / prefill wall time.",
        bounds: TOKENS_PER_S,
    },
    EngineMetric {
        name: "hypnos_chat_decode_tokens_per_s",
        help: "Chat: completion tokens / decode wall time.",
        bounds: TOKENS_PER_S,
    },
    EngineMetric {
        name: "hypnos_tts_rtf",
        help: "Text-to-speech: synthesis seconds / audio seconds.",
        bounds: RTF,
    },
    EngineMetric {
        name: "hypnos_tts_ttfa_ms",
        help: "Text-to-speech: time to first audio chunk (ms, streaming only).",
        bounds: LATENCY_MS,
    },
    EngineMetric {
        name: "hypnos_stt_rtf",
        help: "Speech-to-text: transcription seconds / audio seconds.",
        bounds: RTF,
    },
    EngineMetric {
        name: "hypnos_image_jct_s",
        help: "Image generation: request to last image (s).",
        bounds: SECONDS,
    },
];

/// Per-model counters, in exposition order.
const ENGINE_COUNTERS: &[(&str, &str)] = &[
    (
        "hypnos_prefix_cache_hits_total",
        "Chat requests whose prompt was partly served from a slot's KV cache.",
    ),
    (
        "hypnos_prefix_cache_misses_total",
        "Chat requests that prefilled their whole prompt.",
    ),
    (
        "hypnos_prefix_cache_tokens_reused_total",
        "Prompt positions served from the KV cache instead of being prefilled.",
    ),
];

#[derive(Debug, Default)]
pub struct Metrics {
    routes: Mutex<BTreeMap<String, RouteStats>>,
    /// `(metric name, model id)` -> histogram.
    engine: Mutex<BTreeMap<(&'static str, String), Histogram>>,
    /// `(metric name, model id)` -> counter.
    counters: Mutex<BTreeMap<(&'static str, String), u64>>,
}

impl Metrics {
    pub fn record(&self, route: &str, status: u16, elapsed: Duration) {
        let mut routes = self.routes.lock();
        let entry = routes.entry(route.to_string()).or_default();
        entry.requests += 1;
        match status {
            400..=499 => entry.errors_4xx += 1,
            500..=599 => entry.errors_5xx += 1,
            _ => {}
        }
        let ms = elapsed.as_secs_f64() * 1000.0;
        entry.latency_ms_sum += ms;
        if ms > entry.latency_ms_max {
            entry.latency_ms_max = ms;
        }
    }

    fn observe(&self, name: &'static str, model: &str, value: Option<f64>) {
        let Some(v) = value else { return };
        let bounds = ENGINE_METRICS
            .iter()
            .find(|m| m.name == name)
            .map(|m| m.bounds)
            .unwrap_or(LATENCY_MS);
        self.engine
            .lock()
            .entry((name, model.to_string()))
            .or_insert_with(|| Histogram::new(bounds))
            .observe(v);
    }

    fn count(&self, name: &'static str, model: &str, by: u64) {
        *self
            .counters
            .lock()
            .entry((name, model.to_string()))
            .or_insert(0) += by;
    }

    pub fn record_chat(&self, model: &str, t: &EngineTiming) {
        if let Some(cached) = t.cached_tokens {
            if cached > 0 {
                self.count("hypnos_prefix_cache_hits_total", model, 1);
                self.count(
                    "hypnos_prefix_cache_tokens_reused_total",
                    model,
                    u64::from(cached),
                );
            } else {
                self.count("hypnos_prefix_cache_misses_total", model, 1);
            }
        }
        self.observe("hypnos_chat_ttft_ms", model, t.ttft_ms);
        self.observe("hypnos_chat_queue_ms", model, t.queue_ms);
        self.observe("hypnos_chat_context_ms", model, t.context_ms);
        self.observe(
            "hypnos_chat_prefill_tokens_per_s",
            model,
            t.prefill_tokens_per_s,
        );
        self.observe(
            "hypnos_chat_decode_tokens_per_s",
            model,
            t.decode_tokens_per_s,
        );
    }

    pub fn record_tts(&self, model: &str, t: &EngineTiming) {
        self.observe("hypnos_tts_rtf", model, t.rtf);
        self.observe("hypnos_tts_ttfa_ms", model, t.ttfa_ms);
    }

    pub fn record_stt(&self, model: &str, t: &EngineTiming) {
        self.observe("hypnos_stt_rtf", model, t.rtf);
    }

    pub fn record_image(&self, model: &str, jct: Duration) {
        self.observe("hypnos_image_jct_s", model, Some(jct.as_secs_f64()));
    }

    pub fn render(
        &self,
        status: &MemoryHierarchyStatus,
        sched: &SchedulerStats,
        uptime: Duration,
    ) -> String {
        let mut out = String::with_capacity(8192);
        let _ = writeln!(
            out,
            "# HELP hypnos_uptime_seconds Seconds since the server started."
        );
        let _ = writeln!(out, "# TYPE hypnos_uptime_seconds gauge");
        let _ = writeln!(out, "hypnos_uptime_seconds {}", uptime.as_secs());

        let _ = writeln!(out, "# HELP hypnos_http_requests_total Requests per route.");
        let _ = writeln!(out, "# TYPE hypnos_http_requests_total counter");
        let _ = writeln!(
            out,
            "# HELP hypnos_http_request_latency_ms_sum Summed latency per route (ms)."
        );
        let _ = writeln!(out, "# TYPE hypnos_http_request_latency_ms_sum counter");
        let _ = writeln!(
            out,
            "# HELP hypnos_http_request_latency_ms_max Slowest request per route since start (ms)."
        );
        let _ = writeln!(out, "# TYPE hypnos_http_request_latency_ms_max gauge");
        for (route, s) in self.routes.lock().iter() {
            let r = route.replace('"', "");
            let _ = writeln!(
                out,
                "hypnos_http_requests_total{{route=\"{r}\",class=\"ok\"}} {}",
                s.requests - s.errors_4xx - s.errors_5xx
            );
            let _ = writeln!(
                out,
                "hypnos_http_requests_total{{route=\"{r}\",class=\"4xx\"}} {}",
                s.errors_4xx
            );
            let _ = writeln!(
                out,
                "hypnos_http_requests_total{{route=\"{r}\",class=\"5xx\"}} {}",
                s.errors_5xx
            );
            let _ = writeln!(
                out,
                "hypnos_http_request_latency_ms_sum{{route=\"{r}\"}} {:.3}",
                s.latency_ms_sum
            );
            let _ = writeln!(
                out,
                "hypnos_http_request_latency_ms_max{{route=\"{r}\"}} {:.3}",
                s.latency_ms_max
            );
        }

        let _ = writeln!(
            out,
            "# HELP hypnos_scheduler_promotions_total Models promoted into Tier 1 VRAM."
        );
        let _ = writeln!(out, "# TYPE hypnos_scheduler_promotions_total counter");
        let _ = writeln!(
            out,
            "hypnos_scheduler_promotions_total {}",
            sched.promotions.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE hypnos_scheduler_evictions_total counter");
        let _ = writeln!(
            out,
            "hypnos_scheduler_evictions_total {}",
            sched.evictions.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE hypnos_scheduler_load_failures_total counter");
        let _ = writeln!(
            out,
            "hypnos_scheduler_load_failures_total {}",
            sched.load_failures.load(Ordering::Relaxed)
        );
        let _ = writeln!(
            out,
            "# TYPE hypnos_scheduler_admissions_rejected_total counter"
        );
        let _ = writeln!(
            out,
            "hypnos_scheduler_admissions_rejected_total {}",
            sched.admissions_rejected.load(Ordering::Relaxed)
        );
        let _ = writeln!(out, "# TYPE hypnos_scheduler_tier_switch_ms_total counter");
        let _ = writeln!(
            out,
            "hypnos_scheduler_tier_switch_ms_total {}",
            sched.tier_switch_ms_total.load(Ordering::Relaxed)
        );

        let _ = writeln!(out, "# HELP hypnos_vram_bytes Tier 1 VRAM accounting (booked / budget-free / effective-free / real-free / unaccounted).");
        let _ = writeln!(out, "# TYPE hypnos_vram_bytes gauge");
        let _ = writeln!(
            out,
            "hypnos_vram_bytes{{kind=\"booked\"}} {}",
            status.tier1_vram_used_bytes
        );
        let _ = writeln!(
            out,
            "hypnos_vram_bytes{{kind=\"budget_free\"}} {}",
            status.tier1_vram_free_bytes
        );
        let _ = writeln!(
            out,
            "hypnos_vram_bytes{{kind=\"effective_free\"}} {}",
            status.tier1_vram_effective_free_bytes
        );
        if let Some(v) = status.tier1_vram_real_free_bytes {
            let _ = writeln!(out, "hypnos_vram_bytes{{kind=\"real_free\"}} {v}");
        }
        if let Some(v) = status.tier1_vram_real_total_bytes {
            let _ = writeln!(out, "hypnos_vram_bytes{{kind=\"real_total\"}} {v}");
        }
        let _ = writeln!(
            out,
            "hypnos_vram_bytes{{kind=\"unaccounted\"}} {}",
            status.tier1_vram_unaccounted_bytes
        );
        let _ = writeln!(out, "# TYPE hypnos_pinned_ram_bytes gauge");
        let _ = writeln!(
            out,
            "hypnos_pinned_ram_bytes{{kind=\"used\"}} {}",
            status.tier2_pinned_ram_used_bytes
        );
        let _ = writeln!(
            out,
            "hypnos_pinned_ram_bytes{{kind=\"free\"}} {}",
            status.tier2_pinned_ram_free_bytes
        );

        let _ = writeln!(
            out,
            "# HELP hypnos_model_tier Current tier per model (0 = SSD, 1 = pinned RAM, 2 = VRAM)."
        );
        let _ = writeln!(out, "# TYPE hypnos_model_tier gauge");
        let _ = writeln!(out, "# TYPE hypnos_model_gpu_layers gauge");
        let _ = writeln!(out, "# TYPE hypnos_model_active_leases gauge");
        let _ = writeln!(out, "# TYPE hypnos_model_requests_total counter");
        for m in &status.models {
            let id = m.id.replace('"', "");
            let _ = writeln!(
                out,
                "hypnos_model_tier{{model=\"{id}\"}} {}",
                m.current_tier as i32
            );
            let _ = writeln!(
                out,
                "hypnos_model_gpu_layers{{model=\"{id}\"}} {}",
                m.placement.gpu_layers
            );
            let _ = writeln!(
                out,
                "hypnos_model_active_leases{{model=\"{id}\"}} {}",
                m.active_leases
            );
            let _ = writeln!(
                out,
                "hypnos_model_requests_total{{model=\"{id}\"}} {}",
                m.total_requests_served
            );
        }

        let counters = self.counters.lock();
        for (name, help) in ENGINE_COUNTERS {
            let series: Vec<(&String, &u64)> = counters
                .iter()
                .filter(|((n, _), _)| n == name)
                .map(|((_, model), v)| (model, v))
                .collect();
            if series.is_empty() {
                continue;
            }
            let _ = writeln!(out, "# HELP {name} {help}");
            let _ = writeln!(out, "# TYPE {name} counter");
            for (model, v) in series {
                let _ = writeln!(out, "{name}{{model=\"{}\"}} {v}", model.replace('"', ""));
            }
        }
        drop(counters);

        let engine = self.engine.lock();
        for metric in ENGINE_METRICS {
            let series: Vec<(&String, &Histogram)> = engine
                .iter()
                .filter(|((name, _), _)| *name == metric.name)
                .map(|((_, model), h)| (model, h))
                .collect();
            if series.is_empty() {
                continue;
            }
            let _ = writeln!(out, "# HELP {} {}", metric.name, metric.help);
            let _ = writeln!(out, "# TYPE {} histogram", metric.name);
            let _ = writeln!(out, "# TYPE {}_last gauge", metric.name);
            for (model, h) in series {
                h.render(&mut out, metric.name, &model.replace('"', ""));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_buckets_are_cumulative() {
        let mut h = Histogram::new(&[1.0, 10.0, 100.0]);
        h.observe(0.5);
        h.observe(5.0);
        h.observe(500.0);
        h.observe(f64::NAN);
        assert_eq!(h.counts, vec![1, 2, 2]);
        assert_eq!(h.count, 3);
        assert_eq!(h.last, 500.0);
        let mut out = String::new();
        h.render(&mut out, "m", "x");
        assert!(out.contains("m_bucket{model=\"x\",le=\"+Inf\"} 3"));
        assert!(out.contains("m_last{model=\"x\"} 500.000"));
    }

    #[test]
    fn chat_timing_lands_in_per_model_series() {
        let m = Metrics::default();
        m.record_chat(
            "gemma4:26b-q4",
            &EngineTiming {
                ttft_ms: Some(2300.0),
                decode_tokens_per_s: Some(110.0),
                ..EngineTiming::default()
            },
        );
        let engine = m.engine.lock();
        assert_eq!(
            engine[&("hypnos_chat_ttft_ms", "gemma4:26b-q4".to_string())].count,
            1
        );
        assert!(engine
            .get(&(
                "hypnos_chat_prefill_tokens_per_s",
                "gemma4:26b-q4".to_string()
            ))
            .is_none());
    }
}
