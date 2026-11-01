use anyhow::{Context, Result};
use axum_server::tls_rustls::RustlsConfig;
use hypnos_core::config::HypnosConfig;
use hypnos_server::{
    build_router, default_profile::default_six_model_config, playground_dir, with_playground,
    AppState,
};
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

fn main() -> Result<()> {
    // Before any thread exists (the async runtime starts several).
    hypnos_backends::prepare_process();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("starting the async runtime")?
        .block_on(run())
}

async fn run() -> Result<()> {
    // `hypnos [config.toml] [--port N]`, `hypnos --check-config [config.toml]`, `hypnos --version`.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    // `--port N` / `--port=N` anywhere on the line overrides `server.port` (and `HYPNOS_PORT`).
    let mut port_arg: Option<u16> = None;
    let mut i = 0;
    while i < args.len() {
        let (is_port, value) = match args[i].as_str() {
            "--port" | "-p" => (true, args.get(i + 1).cloned()),
            a if a.starts_with("--port=") => (true, Some(a["--port=".len()..].to_string())),
            _ => (false, None),
        };
        if !is_port {
            i += 1;
            continue;
        }
        let value =
            value.ok_or_else(|| anyhow::anyhow!("--port needs a number, e.g. --port 9999"))?;
        port_arg = Some(
            value
                .parse::<u16>()
                .ok()
                .filter(|p| *p > 0)
                .ok_or_else(|| anyhow::anyhow!("--port: {value:?} is not a port (1-65535)"))?,
        );
        let consumed = if args[i].starts_with("--port=") { 1 } else { 2 };
        args.drain(i..i + consumed);
    }
    match args.first().map(String::as_str) {
        Some("--version" | "-V") => {
            println!("hypnos {}", env!("CARGO_PKG_VERSION"));
            return Ok(());
        }
        Some("advise") => {
            return advise_cli(&args[1..]);
        }
        Some("--catalog-files") => {
            // `hypnos --catalog-files <id>[:<quant>]`: the variant's files as models.manifest
            // lines (model repo remote local platform), for the fetch scripts.
            let Some(spec) = args.get(1) else {
                anyhow::bail!("usage: hypnos --catalog-files <id>[:<quant>]");
            };
            let path = hypnos_server::catalog::locate()
                .with_context(|| format!("{} not found", hypnos_server::catalog::FILE_NAME))?;
            let catalog =
                hypnos_core::catalog::ModelCatalog::load(&path).map_err(anyhow::Error::msg)?;
            let (entry, variant) = catalog.resolve(spec).map_err(anyhow::Error::msg)?;
            for f in &variant.files {
                let platform = serde_json::to_value(f.platform)?;
                println!(
                    "{} {} {} {} {}",
                    entry.id,
                    f.repo,
                    f.remote,
                    f.local,
                    platform.as_str().unwrap_or("all")
                );
            }
            return Ok(());
        }
        Some("--catalog-block") => {
            // `hypnos --catalog-block <id>[:<quant>]`: the [[models]] block for a catalog entry.
            let Some(spec) = args.get(1) else {
                anyhow::bail!("usage: hypnos --catalog-block <id>[:<quant>]");
            };
            let path = hypnos_server::catalog::locate()
                .with_context(|| format!("{} not found", hypnos_server::catalog::FILE_NAME))?;
            let catalog =
                hypnos_core::catalog::ModelCatalog::load(&path).map_err(anyhow::Error::msg)?;
            let (entry, variant) = catalog.resolve(spec).map_err(anyhow::Error::msg)?;
            print!("{}", entry.render_block(variant));
            return Ok(());
        }
        Some("--help" | "-h") => {
            println!(
                "usage: hypnos [config.toml] [--port N]   start the server (config: $HYPNOS_CONFIG, then ./hypnos.toml;\n                                         port: --port, then $HYPNOS_PORT, then server.port, default 9999)\n       hypnos --check-config [config.toml]   validate the config and exit (1 on errors)\n       hypnos --catalog-block <id>[:<quant>]   print the [[models]] block of a models.catalog.toml entry\n       hypnos --catalog-files <id>[:<quant>]   print its files as models.manifest lines\n       hypnos advise --needs chat,voice,documents [--vram 24G] [--ram 64G] [--unified] [--json|--config]\n                                         recommend a lineup from the catalog for this (or that) machine\n       hypnos measure-memory [config.toml] [--models a,b] [--json] [--out file]   booked vs real memory per model\n       hypnos --version"
            );
            return Ok(());
        }
        _ => {}
    }
    // `hypnos measure-memory [config] [--models a,b] [--json]` (TEST-3): booked vs real per model.
    let measure_mode = args.first().map(String::as_str) == Some("measure-memory");
    let mut measure_models: Vec<String> = Vec::new();
    let mut measure_json = false;
    let mut measure_out: Option<String> = None;
    if measure_mode {
        args.remove(0);
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--models" => {
                    let list = args
                        .get(i + 1)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--models needs a list"))?;
                    measure_models = list.split(',').map(|m| m.trim().to_string()).collect();
                    args.drain(i..i + 2);
                }
                "--json" => {
                    measure_json = true;
                    args.remove(i);
                }
                "--out" => {
                    measure_out = Some(
                        args.get(i + 1)
                            .cloned()
                            .ok_or_else(|| anyhow::anyhow!("--out needs a file"))?,
                    );
                    args.drain(i..i + 2);
                }
                _ => i += 1,
            }
        }
    }
    let check_only = args.first().map(String::as_str) == Some("--check-config");
    let config_arg = if check_only {
        args.get(1).cloned()
    } else {
        args.first().cloned()
    };
    // The argument, else HYPNOS_CONFIG (the Docker image), else ./hypnos.toml.
    let config_path = config_arg
        .or_else(|| {
            std::env::var("HYPNOS_CONFIG")
                .ok()
                .filter(|p| !p.is_empty())
        })
        .unwrap_or_else(|| "hypnos.toml".to_string());

    // Logging must exist before we can report configuration problems, so read the format first.
    let (mut config, source) = if Path::new(&config_path).exists() {
        let raw = std::fs::read_to_string(&config_path)
            .with_context(|| format!("reading {config_path}"))?;
        (
            HypnosConfig::from_toml_str(&raw).with_context(|| format!("parsing {config_path}"))?,
            config_path.clone(),
        )
    } else {
        (
            default_six_model_config(),
            "<built-in default profile>".to_string(),
        )
    };
    let log_format =
        std::env::var("HYPNOS_LOG_FORMAT").unwrap_or_else(|_| config.server.log_format.clone());
    init_logging(&log_format);
    for note in &config.profile_notes {
        info!(profile = ?config.server.profile, "set by the workload profile: {note}");
    }
    // Models added, replaced or removed at runtime (API-11) live in an overlay next to the config.
    let overlay_path = hypnos_server::pool::overlay_path_for(Path::new(&config_path));
    let config_ids: std::collections::BTreeSet<String> =
        config.models.iter().map(|m| m.id.clone()).collect();
    for note in hypnos_server::pool::apply_overlay(&mut config, &overlay_path)
        .map_err(anyhow::Error::msg)?
    {
        info!(overlay = %overlay_path.display(), "{note}");
    }
    if check_only {
        info!(config = %source, version = env!("CARGO_PKG_VERSION"), "Hypnos checking the configuration");
    } else {
        info!(config = %source, version = env!("CARGO_PKG_VERSION"), "Hypnos starting");
    }

    // Keys from the environment replace the config list, so they can stay out of the file.
    config
        .server
        .apply_env_api_keys(std::env::var("HYPNOS_API_KEYS").ok().as_deref());
    // The port: --port on the command line, then HYPNOS_PORT, then the file.
    let env_port = std::env::var("HYPNOS_PORT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| {
            v.trim()
                .parse::<u16>()
                .ok()
                .filter(|p| *p > 0)
                .ok_or_else(|| anyhow::anyhow!("HYPNOS_PORT={v:?} is not a port (1-65535)"))
        })
        .transpose()?;
    if let Some(port) = port_arg.or(env_port) {
        config.server.port = port;
    }

    let base_dir = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
    // Relative model paths resolve against models_dir (HYPNOS_MODELS_DIR, then the file).
    if let Some(dir) = config.apply_models_dir(
        &base_dir,
        std::env::var("HYPNOS_MODELS_DIR").ok().as_deref(),
    ) {
        info!(models_dir = %dir.display(), "Model files");
    }

    // Fill unset [models.arch] fields from the GGUF headers before validating them.
    for note in config.detect_arch_from_gguf(&base_dir) {
        info!(config = %note);
    }

    // Fail fast on a broken configuration instead of on the first request.
    let mut report = config.validate(&base_dir);
    if hypnos_backends::docling_gpu_provider().is_none() {
        for m in config
            .models
            .iter()
            .filter(|m| m.engine == hypnos_core::config::EngineKind::Docling && !m.runs_on_cpu())
        {
            report.errors.push(format!(
                "'{}': device = \"gpu\" needs a build with a docling GPU provider (Metal, or the docling-cuda feature); use device = \"cpu\"",
                m.id
            ));
        }
    }
    for w in &report.warnings {
        warn!(config_warning = %w);
    }
    if check_only {
        // What `validate` and the GGUF detection would do, without starting the server.
        for e in &report.errors {
            error!(config_error = %e);
        }
        println!(
            "{source}: config_version {}, {} model(s) ({} enabled), {} warning(s), {} error(s)",
            config.config_version,
            config.models.len(),
            config.models.iter().filter(|m| m.enabled).count(),
            report.warnings.len(),
            report.errors.len()
        );
        if !report.is_ok() {
            std::process::exit(1);
        }
        return Ok(());
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
    let playground = playground_dir(config.server.frontend_dir.as_deref());
    // Load the certificate now, so a bad PEM fails before the (slow) model startup.
    let tls = match &config.server.tls {
        Some(t) => {
            // rustls needs a process-wide crypto provider; ring is the one already compiled in.
            let _ = rustls::crypto::ring::default_provider().install_default();
            let cfg = RustlsConfig::from_pem_file(&t.cert_path, &t.key_path)
                .await
                .with_context(|| {
                    format!(
                        "loading server.tls ({} / {})",
                        t.cert_path.display(),
                        t.key_path.display()
                    )
                })?;
            Some(cfg)
        }
        None => None,
    };
    let state = AppState::from_config(config)?;
    state.pool.set_overlay(overlay_path, config_ids);
    state.pictures.spawn_sweeper();
    info!(dir = %state.pictures.dir().display(), "document picture store");
    if state.auth.enabled() {
        info!(
            keys = state.auth.key_count(),
            "API key authentication enabled for /v1/*"
        );
    }

    // Ground the planner in real device memory (no-op when no GPU backend is compiled in).
    state
        .scheduler
        .memory()
        .set_gpu_probe(std::sync::Arc::new(hypnos_backends::LlamaGpuProbe));

    if measure_mode {
        let rows = hypnos_server::measure::measure(&state, &measure_models).await;
        let text = if measure_json {
            serde_json::to_string_pretty(&rows)?
        } else {
            hypnos_server::measure::markdown(&rows)
        };
        // The log goes to stdout as well: --out keeps the result apart.
        match &measure_out {
            Some(path) => std::fs::write(path, &text).with_context(|| format!("writing {path}"))?,
            None => println!("{text}"),
        }
        state.scheduler.shutdown().await;
        return Ok(());
    }

    // Listen before the startup pre-staging (`initial_tier = tier1_gpu_vram` / `tier2_pinned_ram`),
    // which takes minutes with a large model: `/health` answers at once, `/ready` and the
    // inference routes say 503 until it is done, and `/v1/memory/status` shows its progress.
    let staging = {
        let state = state.clone();
        tokio::spawn(async move {
            if let Err(e) = state.scheduler.initialize_startup_tiers().await {
                // Models not pre-staged load on their first request; the server still serves.
                error!(error = %e, "Startup tier initialization failed; models load on demand");
            }
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
        })
    };

    let mut app = build_router(state.clone(), &allowed_origins);
    if let Some(dir) = &playground {
        info!(dir = %dir.display(), "Serving the playground at /");
        app = with_playground(app, dir);
    }
    let listener = TcpListener::bind(&bind_addr)
        .await
        .with_context(|| format!("binding {bind_addr}"))?;
    let scheme = if tls.is_some() { "https" } else { "http" };
    info!(address = %bind_addr, scheme, "OpenAI-compatible multi-modal server listening");

    match tls {
        None => {
            axum::serve(listener, app)
                .with_graceful_shutdown(shutdown_signal())
                .await?
        }
        Some(tls) => {
            let handle = axum_server::Handle::new();
            let on_signal = handle.clone();
            tokio::spawn(async move {
                shutdown_signal().await;
                on_signal.graceful_shutdown(None);
            });
            axum_server::from_tcp_rustls(listener.into_std()?, tls)
                .handle(handle)
                .serve(app.into_make_service())
                .await?
        }
    }

    state.ready.store(false, Ordering::SeqCst);
    // A shutdown during pre-staging: stop staging further models (the load in progress finishes,
    // `shutdown` waits for the transition lock) and release what was loaded.
    staging.abort();
    let _ = staging.await;
    state.scheduler.shutdown().await;
    info!("Hypnos stopped");
    Ok(())
}

/// `24G`, `24GB`, `24GiB`, `512M` or plain bytes (GiB / MiB either way).
fn parse_size(s: &str) -> anyhow::Result<u64> {
    let t = s.trim().to_ascii_lowercase();
    let t = t.trim_end_matches('b').trim_end_matches('i');
    let (num, mult) = match t.chars().last() {
        Some('g') => (&t[..t.len() - 1], 1u64 << 30),
        Some('m') => (&t[..t.len() - 1], 1u64 << 20),
        Some('t') => (&t[..t.len() - 1], 1u64 << 40),
        _ => (t, 1),
    };
    let n: f64 = num
        .trim()
        .parse()
        .map_err(|_| anyhow::anyhow!("{s:?} is not a size (e.g. 24G)"))?;
    Ok((n * mult as f64) as u64)
}

/// `hypnos advise`: a lineup from the catalog for this machine, or the one described.
fn advise_cli(args: &[String]) -> Result<()> {
    use hypnos_memory::advisor::{advise, AdvisorRequest, Need};
    let mut needs = Vec::new();
    let (mut vram, mut ram, mut unified) = (None, None, None);
    let (mut json, mut config_only) = (false, false);
    let mut i = 0;
    while i < args.len() {
        let value = || {
            args.get(i + 1)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{} needs a value", args[i]))
        };
        match args[i].as_str() {
            "--needs" => {
                for n in value()?.split(',').filter(|n| !n.trim().is_empty()) {
                    needs.push(Need::parse(n).ok_or_else(|| {
                        anyhow::anyhow!(
                            "unknown need {n:?} (chat, vision, voice, transcription, speech, embeddings, images, documents, entities)"
                        )
                    })?);
                }
                i += 1;
            }
            "--vram" => {
                vram = Some(parse_size(&value()?)?);
                i += 1;
            }
            "--ram" => {
                ram = Some(parse_size(&value()?)?);
                i += 1;
            }
            "--unified" => unified = Some(true),
            "--discrete" => unified = Some(false),
            "--json" => json = true,
            "--config" => config_only = true,
            other => anyhow::bail!("unknown argument {other:?} (see hypnos --help)"),
        }
        i += 1;
    }
    if needs.is_empty() {
        anyhow::bail!("--needs is required, e.g. --needs chat,voice,documents");
    }
    // This machine unless described: its RAM, Apple Silicon = unified, else the GPU's memory.
    let unified = unified
        .unwrap_or(cfg!(all(target_os = "macos", target_arch = "aarch64")) && vram.is_none());
    let ram = match ram {
        Some(r) => r,
        None => hypnos_memory::process::physical_memory_bytes()
            .ok_or_else(|| anyhow::anyhow!("could not read this machine's RAM: pass --ram"))?,
    };
    let vram = match (vram, unified) {
        (Some(v), _) => v,
        (None, true) => 0,
        (None, false) => {
            let caps = hypnos_memory::DeviceMemoryCapabilities::detect(
                &hypnos_core::config::HardwareConfig::default(),
            );
            caps.total_vram_bytes
        }
    };
    let path = hypnos_server::catalog::locate()
        .with_context(|| format!("{} not found", hypnos_server::catalog::FILE_NAME))?;
    let catalog = hypnos_core::catalog::ModelCatalog::load(&path).map_err(anyhow::Error::msg)?;
    let advice = advise(
        &catalog,
        &AdvisorRequest {
            vram_bytes: vram,
            ram_bytes: ram,
            unified_memory: unified,
            needs,
        },
    )
    .map_err(anyhow::Error::msg)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&advice)?);
        return Ok(());
    }
    if config_only {
        print!("{}", advice.config);
        return Ok(());
    }
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    println!(
        "Budgets: GPU (Tier 1) {:.1} GiB, warm RAM (Tier 2) {:.1} GiB{}",
        gib(advice.budgets.tier1_bytes),
        gib(advice.budgets.tier2_bytes),
        if unified { ", unified memory" } else { "" }
    );
    for m in &advice.models {
        let serves: Vec<String> = m
            .serves
            .iter()
            .map(|n| format!("{n:?}").to_lowercase())
            .collect();
        println!(
            "  {:<18} {:<7} {:>5.1} GiB resident{}  {:?}, {:?}  serves {}",
            m.id,
            m.quant,
            gib(m.resident_bytes),
            if m.runs_on_cpu { " (RAM)" } else { "" },
            m.initial_tier,
            m.priority,
            serves.join(", ")
        );
    }
    println!(
        "Interactive models together: {:.1} GiB ({})",
        gib(advice.interactive_bytes),
        if advice.interactive_coresident {
            "they stay resident together"
        } else {
            "they swap"
        }
    );
    for w in &advice.warnings {
        println!("  ! {w}");
    }
    println!(
        "Download: ./run.sh models <spec> for {}",
        advice.specs.join(" ")
    );
    println!("Config:   hypnos advise ... --config > hypnos.toml (or --json for everything)");
    Ok(())
}
