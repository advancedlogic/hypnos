use anyhow::{Context, Result};
use hypnos_core::config::HypnosConfig;
use hypnos_server::{build_router, default_profile::default_six_model_config, AppState};
use std::path::Path;
use std::sync::atomic::Ordering;
use tokio::net::TcpListener;
use tracing::{error, info, warn};

fn init_logging(format: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,hypnos=debug,ort=warn".into());
    if format == "json" {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .with_current_span(false)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        use tokio::signal::unix::{signal, SignalKind};
        if let Ok(mut s) = signal(SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("Shutdown signal received; finishing in-flight requests");
}

#[tokio::main]
async fn main() -> Result<()> {
    let config_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "hypnos.toml".to_string());

    // Logging must exist before we can report configuration problems, so read the format first.
    let (config, source) = if Path::new(&config_path).exists() {
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {config_path}"))?;
        (
            HypnosConfig::from_toml_str(&raw).with_context(|| format!("parsing {config_path}"))?,
            config_path.clone(),
        )
    } else {
        (
            default_six_model_config(),
            "<built-in 5-model profile>".to_string(),
        )
    };
    let log_format =
        std::env::var("HYPNOS_LOG_FORMAT").unwrap_or_else(|_| config.server.log_format.clone());
    init_logging(&log_format);
    info!(config = %source, version = env!("CARGO_PKG_VERSION"), "Hypnos starting");

    // Fail fast on a broken configuration instead of on the first request.
    let base_dir = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
    let report = config.validate(&base_dir);
    for w in &report.warnings {
        warn!(config_warning = %w);
    }
    if !report.is_ok() {
        for e in &report.errors {
            error!(config_error = %e);
        }
        anyhow::bail!(
            "configuration invalid: {} error(s); fix {} and restart",
            report.errors.len(),
            source
        );
    }

    let bind_addr = format!("{}:{}", config.server.host, config.server.port);
    let allowed_origins = config.server.allowed_origins.clone();
    let state = AppState::from_config(config)?;

    // Ground the planner in real device memory (no-op when no GPU backend is compiled in).
    state
        .scheduler
        .memory()
        .set_gpu_probe(std::sync::Arc::new(hypnos_backends::LlamaGpuProbe));

    // Pre-stage models configured with `initial_tier = tier1_gpu_vram` / `tier2_pinned_ram`.
    state.scheduler.initialize_startup_tiers().await?;
    state.ready.store(true, Ordering::SeqCst);

    let status = state.scheduler.status();
    info!(
        accelerator = ?status.capabilities.accelerator,
        topology = ?status.capabilities.topology,
        vram_booked_mb = status.tier1_vram_used_bytes / (1024 * 1024),
        vram_effective_free_mb = status.tier1_vram_effective_free_bytes / (1024 * 1024),
        vram_real_free_mb = status.tier1_vram_real_free_bytes.map(|v| v / (1024 * 1024)),
        pinned_ram_used_mb = status.tier2_pinned_ram_used_bytes / (1024 * 1024),
        models_registered = status.models.len(),
        "Hypnos 3-Tier Memory Coordinator ready"
    );

    let app = build_router(state.clone(), &allowed_origins);
    let listener = TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("binding {bind_addr}"))?;
    info!(address = %bind_addr, "OpenAI-compatible multi-modal server listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    state.ready.store(false, Ordering::SeqCst);
    state.scheduler.shutdown().await;
    info!("Hypnos stopped");
    Ok(())
}
