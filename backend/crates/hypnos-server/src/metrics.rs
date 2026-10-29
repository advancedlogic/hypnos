//! Minimal Prometheus text exposition (no external metrics crate): per-route request counters and
//! latency sums, scheduler counters, and memory-tier gauges.

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

#[derive(Debug, Default)]
pub struct Metrics {
    routes: Mutex<BTreeMap<String, RouteStats>>,
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

    pub fn render(
        &self,
        status: &MemoryHierarchyStatus,
        sched: &SchedulerStats,
        uptime: Duration,
    ) -> String {
        let mut out = String::with_capacity(4096);
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
        out
    }
}
