//! The guided setup (DESK-18): what the user wants to do, what this machine can hold, the lineup
//! the advisor recommends for both, its download, and a short demonstration that it works. No
//! knowledge of pinned memory, quantization or KV caches is needed: the wizard speaks of "kept in
//! GPU memory" and "a smaller, faster version".
//!
//! The recommendation comes from the server binary (`hypnos advise --needs ... --json`, API-12),
//! run against the catalog the app reads (as `--catalog-block` in `catalog.rs`), so it works
//! while no server runs. The download is the Models dialog's (`models::download_lines`).
//!
//! Applying it depends on the config. The app's own config (`local_server::is_managed`) gets the
//! advised `[[models]]` blocks in place of its own, the advised `[hardware]
//! pinned_ram_budget_bytes`, and keeps everything else; the previous file stays next to it as
//! `<config>.bak`, and the server's model-pool overlay (models added or removed while it ran,
//! which would otherwise be applied over the new lineup at the next start) is moved aside as
//! `<overlay>.bak`. The caller then (re)starts the server. Another config is never rewritten: a
//! running server that uses its models folder gets the models it lacks live (`POST /v1/models`),
//! or the wizard shows the advised config to copy.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

use base64::Engine;
use hypnos_client::{ModelSource, SpeechRequest, TranscriptionRequest};
use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::{AppHandle, Runtime, State};

use crate::catalog::{self, Catalog, APPLE_SILICON};
use crate::hypnos::{ErrorPayload, Hypnos};
use crate::models::{download_lines, free_bytes, models_root, read_doc, DownloadEvent, ModelFiles};

/// The needs the wizard offers, as `hypnos advise --needs` takes them.
const NEEDS: [&str; 9] = [
    "chat",
    "vision",
    "voice",
    "transcription",
    "speech",
    "embeddings",
    "images",
    "documents",
    "entities",
];

/// What the advisor's machine budgets are (`budgets` of the advice).
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(default)]
pub(crate) struct Budgets {
    pub tier1_bytes: u64,
    pub tier2_bytes: u64,
    pub unified_memory: bool,
}

/// One model of the advice (`models[]`); fields the wizard does not use are ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct AdvisedModel {
    pub id: String,
    pub name: String,
    pub quant: String,
    pub serves: Vec<String>,
    pub weights_bytes: u64,
    pub resident_bytes: u64,
    pub runs_on_cpu: bool,
    pub gpu_layers: u32,
    pub num_layers: u32,
    pub initial_tier: String,
    pub keep_warm_in_ram: bool,
    pub load_secs_warm: f64,
}

/// `hypnos advise --json` (the body of `POST /v1/advisor` without `machine`).
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub(crate) struct Advice {
    pub budgets: Budgets,
    pub models: Vec<AdvisedModel>,
    pub interactive_coresident: bool,
    pub interactive_bytes: u64,
    pub warnings: Vec<String>,
    pub specs: Vec<String>,
    pub config: String,
}

pub(crate) fn parse_advice(json: &str) -> Result<Advice, String> {
    let advice: Advice =
        serde_json::from_str(json).map_err(|e| format!("could not read the advice: {e}"))?;
    if advice.models.is_empty() || advice.specs.len() != advice.models.len() {
        return Err("the advice lists no models".into());
    }
    if !advice.config.contains("[[models]]") {
        return Err("the advice has no config".into());
    }
    Ok(advice)
}

/// `<id>:<quant>`; ids hold colons themselves (`gemma4:26b-q4:Q4_0`), the quant is the last part.
pub(crate) fn split_spec(spec: &str) -> Result<(&str, &str), String> {
    spec.rsplit_once(':')
        .filter(|(id, quant)| !id.is_empty() && !quant.is_empty())
        .ok_or_else(|| format!("\"{spec}\" is not <id>:<quant>"))
}

/// What a model does for the user, in words: "Chat and seeing images", "Speaking".
fn role(serves: &[String], modality: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for need in serves {
        let part = match need.as_str() {
            "chat" => "chat",
            "vision" => "seeing images",
            // Voice is speech in, chat and speech out: the chat model's part is the chat.
            "voice" | "transcription" | "speech" => match modality {
                "text_to_speech" => "speaking answers aloud",
                "speech_to_text" => "listening (speech to text)",
                _ if need == "voice" => "voice conversations",
                _ if need == "speech" => "speaking answers aloud",
                _ => "listening (speech to text)",
            },
            "embeddings" => "search (finding related text)",
            "images" => "making images",
            "documents" => "reading documents",
            "entities" => "finding names, places and dates",
            other => other,
        };
        if !parts.contains(&part) {
            parts.push(part);
        }
    }
    // The chat model also serves voice: its chat part says it.
    if parts.contains(&"chat") {
        parts.retain(|p| *p != "voice conversations");
    }
    let mut text = match parts.as_slice() {
        [] => "Helper".to_string(),
        [one] => one.to_string(),
        [rest @ .., last] => format!("{} and {last}", rest.join(", ")),
    };
    if let Some(first) = text.get(..1) {
        text = first.to_uppercase() + &text[1..];
    }
    text
}

/// An advisor warning without the memory-tier jargon.
pub(crate) fn plain_warning(w: &str) -> String {
    if let Some((name, _)) = w.split_once(" does not fit the warm-memory budget") {
        return format!(
            "{name} cannot also be kept ready in memory next to the others: when it has been set \
             aside, it loads from disk again, which takes a few seconds."
        );
    }
    if let Some((name, _)) = w.split_once(" only partly fits the GPU") {
        return format!(
            "{name} only partly fits in GPU memory: it works, but answers more slowly."
        );
    }
    if w.starts_with("the interactive models need") {
        return "The models you talk to do not all fit in GPU memory at once: they take turns, \
                and each switch takes a few seconds."
            .into();
    }
    if let Some(rest) = w.strip_prefix("generating an image (") {
        let name = rest.split(')').next().unwrap_or("the image model");
        return format!(
            "Making an image ({name}) sets the chat models aside; the next chat message loads \
             them again."
        );
    }
    let mut s = w.to_string();
    if let Some(first) = s.get(..1) {
        s = first.to_uppercase() + &s[1..];
    }
    s
}

/// An advisor refusal as a sentence for the user.
fn plain_advise_error(stderr: &str) -> String {
    let err = stderr.trim();
    let err = err.strip_prefix("Error: ").unwrap_or(err);
    if err.contains("no lineup fits") {
        return "This computer does not have enough memory for one of the chosen uses, even on \
                its own. Untick the most demanding one (making images needs the most) and try \
                again."
            .into();
    }
    if err.contains("unknown argument") || err.contains("usage") {
        return format!("{err} (a server too old for the guided setup?)");
    }
    format!("The recommendation failed: {err}")
}

/// Physical memory of this machine.
fn total_ram() -> Option<u64> {
    #[cfg(unix)]
    {
        // SAFETY: sysconf only reads system constants.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        (pages > 0 && size > 0).then(|| pages as u64 * size as u64)
    }
    #[cfg(not(unix))]
    {
        None
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelView {
    id: String,
    name: String,
    quant: String,
    spec: String,
    /// What it does, in words.
    role: String,
    serves: Vec<String>,
    /// This machine's files, as the catalog lists them.
    size_bytes: u64,
    /// Still to download.
    missing_bytes: u64,
    installed: bool,
    /// Loaded into GPU memory when the server starts.
    kept_loaded: bool,
    runs_on_cpu: bool,
    /// Only some of its layers fit the GPU.
    partial: bool,
    /// Not the catalog's default variant: a smaller, faster version.
    smaller: bool,
    load_secs: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AdviceView {
    ram_bytes: Option<u64>,
    budgets: Budgets,
    models: Vec<ModelView>,
    /// The models the user talks to stay loaded together.
    together: bool,
    warnings: Vec<String>,
    total_bytes: u64,
    download_bytes: u64,
    free_bytes: Option<u64>,
    models_dir: String,
    /// The app's own config: Install rewrites its models.
    managed: bool,
    specs: Vec<String>,
    /// The advised config file (to copy into a config the app does not write).
    config: String,
}

fn advice_view(
    advice: Advice,
    catalog: &Catalog,
    root: &Path,
    apple_silicon: bool,
) -> Result<Vec<ModelView>, String> {
    advice
        .models
        .into_iter()
        .map(|m| {
            let spec = format!("{}:{}", m.id, m.quant);
            let (entry, variant) = catalog::resolve(catalog, &m.id, &m.quant)?;
            let size_bytes: u64 = variant.files_for(apple_silicon).map(|f| f.size).sum();
            let (have, installed) = variant.state(root, apple_silicon);
            Ok(ModelView {
                role: role(&m.serves, &entry.modality),
                missing_bytes: if installed {
                    0
                } else {
                    size_bytes.saturating_sub(have)
                },
                installed,
                kept_loaded: m.initial_tier == "tier1_gpu_vram",
                partial: !m.runs_on_cpu && m.num_layers > 0 && m.gpu_layers < m.num_layers,
                smaller: !variant.default,
                runs_on_cpu: m.runs_on_cpu,
                load_secs: m.load_secs_warm,
                size_bytes,
                spec,
                id: m.id,
                name: m.name,
                quant: m.quant,
                serves: m.serves,
            })
        })
        .collect()
}

fn binary_path(binary: Option<String>) -> Result<PathBuf, String> {
    let binary = binary
        .filter(|b| !b.trim().is_empty())
        .ok_or("no server binary chosen: the recommendation comes from `hypnos advise`")?;
    let binary = PathBuf::from(binary.trim());
    if !binary.is_file() {
        return Err(format!("no server binary at {}", binary.display()));
    }
    Ok(binary)
}

/// Runs `hypnos advise --needs <needs> --json` for this machine.
async fn run_advise(binary: &Path, catalog: &Path, needs: &[String]) -> Result<Advice, String> {
    let mut cmd = catalog::server_command(binary, catalog);
    cmd.arg("advise")
        .arg("--needs")
        .arg(needs.join(","))
        .arg("--json");
    let out = tokio::time::timeout(catalog::RENDER_TIMEOUT, cmd.output())
        .await
        .map_err(|_| format!("{} advise timed out", binary.display()))?
        .map_err(|e| format!("could not run {}: {e}", binary.display()))?;
    if !out.status.success() {
        return Err(plain_advise_error(&String::from_utf8_lossy(&out.stderr)));
    }
    parse_advice(&String::from_utf8_lossy(&out.stdout))
}

/// The recommended lineup for `needs` on this machine, with what is already downloaded into the
/// config's models folder and the free space there.
#[tauri::command]
pub async fn setup_advise<R: Runtime>(
    app: AppHandle<R>,
    binary: Option<String>,
    config: String,
    needs: Vec<String>,
) -> Result<AdviceView, String> {
    if needs.is_empty() {
        return Err("Choose at least one thing to use Hypnos for.".into());
    }
    if let Some(n) = needs.iter().find(|n| !NEEDS.contains(&n.as_str())) {
        return Err(format!("unknown need \"{n}\""));
    }
    let binary = binary_path(binary)?;
    let config = PathBuf::from(config);
    let doc = read_doc(&config)?;
    let root = models_root(&config, &doc);
    let (catalog_path, catalog) = catalog::load_catalog()?;
    let advice = run_advise(&binary, &catalog_path, &needs).await?;
    let warnings = advice.warnings.iter().map(|w| plain_warning(w)).collect();
    let (budgets, together, specs, text) = (
        advice.budgets.clone(),
        advice.interactive_coresident,
        advice.specs.clone(),
        advice.config.clone(),
    );
    let models = advice_view(advice, &catalog, &root, APPLE_SILICON)?;
    Ok(AdviceView {
        ram_bytes: total_ram(),
        budgets,
        together,
        warnings,
        total_bytes: models.iter().map(|m| m.size_bytes).sum(),
        download_bytes: models.iter().map(|m| m.missing_bytes).sum(),
        free_bytes: free_bytes(&root),
        models_dir: root.display().to_string(),
        managed: crate::local_server::is_managed(&app, &config),
        models,
        specs,
        config: text,
    })
}

/// Downloads the missing files of every spec into the config's models folder (events as the
/// Models dialog's download). `Ok(false)`: stopped; what arrived is kept for a resume.
#[tauri::command]
pub async fn setup_install(
    state: State<'_, ModelFiles>,
    config: String,
    specs: Vec<String>,
    on_event: Channel<DownloadEvent>,
) -> Result<bool, String> {
    let config = PathBuf::from(config);
    let root = models_root(&config, &read_doc(&config)?);
    let (_, catalog) = catalog::load_catalog()?;
    let mut lines = Vec::new();
    let mut ids = Vec::new();
    for spec in &specs {
        let (id, quant) = split_spec(spec)?;
        let (entry, variant) = catalog::resolve(&catalog, id, quant)?;
        if !variant.state(&root, APPLE_SILICON).1 {
            lines.extend(variant.lines(&entry.id, APPLE_SILICON));
            ids.push(entry.id.clone());
        }
    }
    if lines.is_empty() {
        let _ = on_event.send(DownloadEvent::Done { models: ids });
        return Ok(true);
    }
    let _guard = state.exclusive()?;
    download_lines(&state, &root, &lines, &ids, &on_event).await
}

/// The ids of a config's `[[models]]`, in file order.
fn model_ids(doc: &toml_edit::DocumentMut) -> Vec<String> {
    doc.get("models")
        .and_then(|m| m.as_array_of_tables())
        .map(|models| {
            models
                .iter()
                .filter_map(|m| m.get("id")?.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

/// The `[[models]]` part of the advised config: from the comment lines right above its first
/// `[[models]]` to the end. Refused when anything but models is in it.
fn models_section(advised: &str) -> Result<String, String> {
    let lines: Vec<&str> = advised.lines().collect();
    let first = lines
        .iter()
        .position(|l| l.trim_start().starts_with("[[models]]"))
        .ok_or("the advised config has no [[models]]")?;
    let mut start = first;
    while start > 0 && lines[start - 1].trim_start().starts_with('#') {
        start -= 1;
    }
    let section = lines[start..].join("\n") + "\n";
    let doc: toml_edit::DocumentMut = section
        .parse()
        .map_err(|e| format!("the advised models: {e}"))?;
    if doc.iter().any(|(k, _)| k != "models") {
        return Err("the advised config has other tables after its [[models]]".into());
    }
    Ok(section)
}

/// `current` with its `[[models]]` (and their `[models.arch]`) replaced by the advised config's,
/// and `[hardware] pinned_ram_budget_bytes` set to the advised one; every other table, key and
/// comment stays as written. `[server] models_dir` is added as `"models"` when missing (where the
/// app downloads for a config without it, and what the advised paths are relative to). Refused
/// unless the result holds exactly the advised models.
pub(crate) fn replace_models(current: &str, advised: &str) -> Result<String, String> {
    let mut doc: toml_edit::DocumentMut =
        current.parse().map_err(|e| format!("the config: {e}"))?;
    let adv: toml_edit::DocumentMut = advised
        .parse()
        .map_err(|e| format!("the advised config: {e}"))?;
    let want = model_ids(&adv);
    if want.is_empty() {
        return Err("the advised config has no models".into());
    }
    if doc
        .get("models")
        .is_some_and(|m| m.as_array_of_tables().is_none())
    {
        return Err("the config's models are not [[models]] tables".into());
    }
    doc.remove("models");

    let server = doc
        .entry("server")
        .or_insert(toml_edit::Item::Table(toml_edit::Table::new()));
    let server = server
        .as_table_like_mut()
        .ok_or("the config's [server] is not a table")?;
    if server.get("models_dir").is_none() {
        server.insert("models_dir", toml_edit::value("models"));
    }

    let pinned = adv
        .get("hardware")
        .and_then(|h| h.get("pinned_ram_budget_bytes"))
        .and_then(|v| v.as_integer());
    if let Some(pinned) = pinned {
        let hardware = doc
            .entry("hardware")
            .or_insert(toml_edit::Item::Table(toml_edit::Table::new()))
            .as_table_like_mut()
            .ok_or("the config's [hardware] is not a table")?;
        match hardware
            .get_mut("pinned_ram_budget_bytes")
            .and_then(|i| i.as_value_mut())
        {
            Some(v) => {
                // Keeps the key's comment.
                let decor = v.decor().clone();
                *v = pinned.into();
                *v.decor_mut() = decor;
            }
            None => {
                hardware.insert("pinned_ram_budget_bytes", toml_edit::value(pinned));
            }
        }
    }

    let mut out = doc.to_string().trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(models_section(advised)?.trim_end());
    out.push('\n');
    let check: toml_edit::DocumentMut = out
        .parse()
        .map_err(|e| format!("the new config does not parse: {e}"))?;
    if model_ids(&check) != want {
        return Err("the new config does not hold exactly the advised models".into());
    }
    Ok(out)
}

/// `<file>.bak` beside `path`.
fn bak_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".bak");
    path.with_file_name(name)
}

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Applied {
    /// The previous config.
    backup: String,
    /// The server's model-pool overlay, moved aside so it does not change the new lineup.
    overlay_backup: Option<String>,
    models: Vec<String>,
}

/// Writes the advised lineup into `config` (see [`replace_models`]), the previous file kept as
/// `<config>.bak` and the pool overlay moved to `<overlay>.bak`.
fn apply_to_file(config: &Path, advised: &str) -> Result<Applied, String> {
    let current = std::fs::read_to_string(config)
        .map_err(|e| format!("could not read {}: {e}", config.display()))?;
    let next = replace_models(&current, advised)?;
    let backup = bak_path(config);
    std::fs::write(&backup, &current)
        .map_err(|e| format!("could not write {}: {e}", backup.display()))?;
    let tmp = config.with_extension("toml.tmp");
    std::fs::write(&tmp, &next).map_err(|e| format!("could not write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, config)
        .map_err(|e| format!("could not write {}: {e}", config.display()))?;
    let overlay = catalog::overlay_path(config);
    let overlay_backup = if overlay.is_file() {
        let to = bak_path(&overlay);
        std::fs::rename(&overlay, &to)
            .map_err(|e| format!("could not move {}: {e}", overlay.display()))?;
        Some(to.display().to_string())
    } else {
        None
    };
    let doc: toml_edit::DocumentMut = next.parse().map_err(|e| format!("{e}"))?;
    Ok(Applied {
        backup: backup.display().to_string(),
        overlay_backup,
        models: model_ids(&doc),
    })
}

/// Puts the advised lineup into the app's own config. The caller stops the server before and
/// starts it after (the overlay must not change while this runs).
#[tauri::command]
pub async fn setup_apply_config<R: Runtime>(
    app: AppHandle<R>,
    config: String,
    advised_config: String,
) -> Result<Applied, String> {
    let config = PathBuf::from(config);
    if !crate::local_server::is_managed(&app, &config) {
        return Err(
            "only the app's own config is rewritten; add the models to the running server or \
             copy the config"
                .into(),
        );
    }
    apply_to_file(&config, &advised_config)
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveStep {
    spec: String,
    id: String,
    /// `added`, `present` (the server already has a model with this id) or `failed`.
    action: &'static str,
    message: Option<String>,
}

/// The live path for a config the app does not write: adds each spec the running server (which
/// must use this config's models folder) does not have yet, `POST /v1/models`. The files must be
/// on disk already (`setup_install`).
#[tauri::command]
pub async fn setup_apply_live(
    hypnos: State<'_, Hypnos>,
    config: String,
    specs: Vec<String>,
) -> Result<Vec<LiveStep>, ErrorPayload> {
    let config = PathBuf::from(config);
    let root = models_root(&config, &read_doc(&config)?);
    let (client, server) = catalog::live_server(&hypnos, &root)
        .await
        .map_err(|why| format!("The models cannot be added live: {why}."))?;
    let pool: BTreeSet<&str> = server
        .data
        .iter()
        .filter(|e| e.configured)
        .map(|e| e.id.as_str())
        .collect();
    let mut steps = Vec::new();
    for spec in specs {
        let id = split_spec(&spec)?.0.to_string();
        if pool.contains(id.as_str()) {
            steps.push(LiveStep {
                spec,
                id,
                action: "present",
                message: None,
            });
            continue;
        }
        let step = match client.add_model(&ModelSource::catalog(spec.clone())).await {
            Ok(_) => LiveStep {
                spec,
                id,
                action: "added",
                message: None,
            },
            Err(e) => LiveStep {
                spec,
                id,
                action: "failed",
                message: Some(catalog::pool_error(e).message),
            },
        };
        steps.push(step);
    }
    Ok(steps)
}

/// Lowercase words of a text, without punctuation.
fn words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// The share of the spoken words heard back (0..=1), each heard word counted once.
pub(crate) fn heard_share(spoken: &str, heard: &str) -> f64 {
    let spoken = words(spoken);
    if spoken.is_empty() {
        return 0.0;
    }
    let mut heard = words(heard);
    let mut hits = 0;
    for w in &spoken {
        if let Some(i) = heard.iter().position(|h| h == w) {
            heard.swap_remove(i);
            hits += 1;
        }
    }
    hits as f64 / spoken.len() as f64
}

/// At most the first two sentences of an answer, as it should be heard, and at most 300 chars.
fn speakable(text: &str) -> String {
    let said = crate::tts::speech_text(text);
    let mut out = crate::tts::sentences(&said, true)
        .into_iter()
        .take(2)
        .collect::<Vec<_>>()
        .join(" ");
    if out.chars().count() > 300 {
        out = out.chars().take(300).collect();
    }
    out.trim().to_string()
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VoiceCheck {
    spoken: String,
    heard: String,
    speech_model: String,
    listen_model: String,
    speak_ms: u64,
    listen_ms: u64,
    audio_secs: f64,
    /// Share of the spoken words heard back.
    share: f64,
    /// The synthesized audio, to play in the wizard.
    audio: String,
}

/// The voice round trip of the demo: `text` spoken by the text-to-speech model, then
/// transcribed by the speech-to-text model.
#[tauri::command]
pub async fn setup_voice_check(
    hypnos: State<'_, Hypnos>,
    text: String,
) -> Result<VoiceCheck, ErrorPayload> {
    let spoken = speakable(&text);
    if spoken.is_empty() {
        return Err("There is no answer to speak.".to_string().into());
    }
    let client = hypnos.client().await?;
    let speech_model = hypnos.pick("text_to_speech", None).await?;
    let started = Instant::now();
    let speech = client
        .speech(&SpeechRequest::new(speech_model.clone(), spoken.clone()).response_format("wav"))
        .await?;
    let speak_ms = started.elapsed().as_millis() as u64;
    let audio = speech.audio.to_vec();
    let data_url = format!(
        "data:audio/wav;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(&audio)
    );
    let (wav, audio_secs) = tauri::async_runtime::spawn_blocking(move || {
        crate::attachments::decode_to_wav(audio, Some("wav"))
    })
    .await
    .map_err(|e| e.to_string())??;
    let listen_model = hypnos.pick("speech_to_text", None).await?;
    let started = Instant::now();
    let heard = client
        .transcribe(&TranscriptionRequest::new(
            listen_model.clone(),
            "speech.wav",
            wav,
        ))
        .await?
        .text
        .trim()
        .to_string();
    let listen_ms = started.elapsed().as_millis() as u64;
    Ok(VoiceCheck {
        share: heard_share(&spoken, &heard),
        spoken,
        heard,
        speech_model,
        listen_model,
        speak_ms,
        listen_ms,
        audio_secs,
        audio: data_url,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADVICE: &str = r#"{
      "budgets": { "tier1_bytes": 100931731456, "tier2_bytes": 30064771072, "unified_memory": true },
      "models": [
        { "id": "gemma4:26b-q4", "name": "Gemma 4 26B-A4B", "quant": "Q4_0",
          "serves": ["chat", "vision", "voice"], "weights_bytes": 15812973984,
          "resident_bytes": 23245280672, "runs_on_cpu": false, "gpu_layers": 30, "num_layers": 30,
          "cpu_expert_layers": 0, "context_len": 16384, "slots": 3,
          "initial_tier": "tier1_gpu_vram", "priority": "high", "keep_warm_in_ram": true,
          "moe_offload": "off", "load_secs_cold": 5.3, "load_secs_warm": 5.3 },
        { "id": "kokoro-82m-q4", "name": "Kokoro 82M", "quant": "fp32", "serves": ["voice"],
          "weights_bytes": 354257480, "resident_bytes": 421366344, "initial_tier": "tier1_gpu_vram",
          "priority": "realtime_pinned", "a_field_from_a_newer_server": 1 }
      ],
      "interactive_coresident": true, "interactive_bytes": 30440975432,
      "warnings": ["Gemma 4 26B-A4B only partly fits the GPU (20 of 30 layers): it runs, slower"],
      "specs": ["gemma4:26b-q4:Q4_0", "kokoro-82m-q4:fp32"],
      "config": "config_version = 1\n\n[server]\nmodels_dir = \"models\"\n\n[hardware]\naccelerator = \"apple_metal_uma\"\npinned_ram_budget_bytes = 30064771072\n\n# Gemma 4 26B-A4B (Q4_0), Google, license gemma. From models.catalog.toml.\n[[models]]\nid = \"gemma4:26b-q4\"\nmodality = \"text_generation\"\nweights_path = \"gemma-4-26B-A4B-it-Q4_0.gguf\"\n\n[models.arch]\ndefault_context_len = 16384\n\n# Kokoro 82M (fp32). From models.catalog.toml.\n[[models]]\nid = \"kokoro-82m-q4\"\nmodality = \"text_to_speech\"\nweights_path = \"kokoro/model.onnx\"\n\n[models.arch]\nnum_layers = 12\n"
    }"#;

    #[test]
    fn parses_the_advice_and_ignores_unknown_fields() {
        let a = parse_advice(ADVICE).unwrap();
        assert!(a.budgets.unified_memory);
        assert_eq!(a.budgets.tier2_bytes, 30064771072);
        assert_eq!(a.models.len(), 2);
        assert_eq!(a.models[0].serves, ["chat", "vision", "voice"]);
        assert_eq!(a.models[0].initial_tier, "tier1_gpu_vram");
        assert_eq!(a.models[1].num_layers, 0, "absent fields default");
        assert!(a.interactive_coresident);
        assert_eq!(a.specs[1], "kokoro-82m-q4:fp32");
        assert!(a.config.contains("[[models]]"));
        assert!(parse_advice("{}").is_err());
        assert!(parse_advice("not json").is_err());
    }

    #[test]
    fn specs_split_at_the_last_colon() {
        assert_eq!(
            split_spec("gemma4:26b-q4:Q4_0").unwrap(),
            ("gemma4:26b-q4", "Q4_0")
        );
        assert_eq!(split_spec("docling:int8").unwrap(), ("docling", "int8"));
        assert!(split_spec("docling").is_err());
        assert!(split_spec("docling:").is_err());
    }

    #[test]
    fn roles_and_warnings_are_plain_words() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(
            role(&s(&["chat", "vision", "voice"]), "text_generation"),
            "Chat and seeing images"
        );
        assert_eq!(
            role(&s(&["voice"]), "text_to_speech"),
            "Speaking answers aloud"
        );
        assert_eq!(
            role(&s(&["voice"]), "speech_to_text"),
            "Listening (speech to text)"
        );
        assert_eq!(
            role(&s(&["documents"]), "document_conversion"),
            "Reading documents"
        );
        let w = plain_warning(
            "Gemma 4 26B-A4B only partly fits the GPU (20 of 30 layers): it runs, slower",
        );
        assert!(
            w.starts_with("Gemma 4 26B-A4B only partly fits in GPU memory"),
            "{w}"
        );
        let w = plain_warning("the interactive models need 30.0 GiB of GPU memory together and 20.0 GiB is available: they swap, each switch costing a load (see load_secs_warm)");
        assert!(!w.contains("load_secs") && w.contains("take turns"), "{w}");
        let w = plain_warning("Kokoro does not fit the warm-memory budget next to the others: it reloads from disk when evicted");
        assert!(w.starts_with("Kokoro cannot also be kept ready"), "{w}");
        let w = plain_warning(
            "generating an image (MOSS-Image) evicts the chat models; the next chat reloads them",
        );
        assert!(w.contains("(MOSS-Image)") && !w.contains("evicts"), "{w}");
        assert_eq!(plain_warning("something new"), "Something new");
        assert!(plain_advise_error("Error: no lineup fits: x").contains("Untick"));
    }

    const CURRENT: &str = r#"# The app's config. This comment stays.
config_version = 1

[server]
port = 9999 # the port
models_dir = "/Volumes/SSD/models"

[limits]
max_queue_per_model = 8

[hardware]
accelerator = "apple_metal_uma"
pinned_ram_budget_bytes = 1 # warm memory

# -----------------------------------------------------------------------------
# 1. old chat model
# -----------------------------------------------------------------------------
[[models]]
id = "old-chat"
weights_path = "old.gguf"

[models.arch]
num_layers = 2

[[models]]
id = "old-tts"
weights_path = "tts.onnx"

[server.tls]
cert = "c.pem"
"#;

    fn advised() -> String {
        parse_advice(ADVICE).unwrap().config
    }

    #[test]
    fn replaces_the_models_and_keeps_the_rest() {
        let out = replace_models(CURRENT, &advised()).unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(model_ids(&doc), ["gemma4:26b-q4", "kokoro-82m-q4"]);
        // The old blocks are gone with their comments and [models.arch].
        assert!(!out.contains("old-chat") && !out.contains("1. old chat model"));
        assert_eq!(
            doc["models"].as_array_of_tables().unwrap().get(1).unwrap()["arch"]["num_layers"]
                .as_integer(),
            Some(12)
        );
        // The rest stays as written, comments included.
        assert!(out.starts_with("# The app's config. This comment stays.\nconfig_version = 1\n"));
        assert!(out.contains("port = 9999 # the port"));
        assert!(out.contains("models_dir = \"/Volumes/SSD/models\""));
        assert_eq!(doc["limits"]["max_queue_per_model"].as_integer(), Some(8));
        assert_eq!(doc["server"]["tls"]["cert"].as_str(), Some("c.pem"));
        assert_eq!(
            doc["hardware"]["accelerator"].as_str(),
            Some("apple_metal_uma")
        );
        // The advised warm memory, with the key's comment.
        assert!(
            out.contains("pinned_ram_budget_bytes = 30064771072 # warm memory"),
            "{out}"
        );
        // The advised blocks keep their comment lines.
        assert!(out.contains("# Gemma 4 26B-A4B (Q4_0), Google, license gemma."));
        // Only the advisor's models were taken, not its [server] or [hardware].
        assert_eq!(out.matches("[server]").count(), 1);
        assert_eq!(out.matches("[hardware]").count(), 1);
        // Twice gives the same file.
        assert_eq!(replace_models(&out, &advised()).unwrap(), out);
    }

    #[test]
    fn a_config_without_models_dir_or_hardware_gets_them() {
        let out = replace_models("[server]\nport = 1\n", &advised()).unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(doc["server"]["models_dir"].as_str(), Some("models"));
        assert_eq!(
            doc["hardware"]["pinned_ram_budget_bytes"].as_integer(),
            Some(30064771072)
        );
        assert_eq!(model_ids(&doc).len(), 2);
        let out = replace_models("", &advised()).unwrap();
        assert_eq!(model_ids(&out.parse().unwrap()).len(), 2);
    }

    #[test]
    fn refuses_what_it_cannot_rewrite_safely() {
        assert!(replace_models("models = []\n", &advised()).is_err());
        assert!(replace_models("not toml [", &advised()).is_err());
        assert!(replace_models(CURRENT, "[server]\nport = 1\n").is_err());
        let trailing = format!("{}\n[limits]\nmax_queue_per_model = 1\n", advised());
        assert!(replace_models(CURRENT, &trailing).is_err());
    }

    #[test]
    fn the_real_template_keeps_its_settings() {
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/hypnos.macos.example.toml");
        let Ok(text) = std::fs::read_to_string(&path) else {
            eprintln!("skipped: no {}", path.display());
            return;
        };
        let out = replace_models(&text, &advised()).unwrap();
        let (before, after): (toml::Table, toml::Table) = (
            toml::from_str(&text).unwrap(),
            toml::from_str(&out).unwrap(),
        );
        for (key, value) in &before {
            if key != "models" && key != "hardware" {
                assert_eq!(after.get(key), Some(value), "{key}");
            }
        }
        let mut hw = before["hardware"].as_table().unwrap().clone();
        hw.insert("pinned_ram_budget_bytes".into(), 30064771072i64.into());
        assert_eq!(after["hardware"].as_table(), Some(&hw));
        assert_eq!(after["models"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn applying_writes_a_backup_and_moves_the_overlay_aside() {
        let dir = std::env::temp_dir().join(format!("hypnos-setup-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let config = dir.join("hypnos.toml");
        std::fs::write(&config, CURRENT).unwrap();
        let overlay = dir.join("hypnos.models.toml");
        std::fs::write(&overlay, "removed = [\"gemma4:26b-q4\"]\n").unwrap();

        let applied = apply_to_file(&config, &advised()).unwrap();
        assert_eq!(applied.models, ["gemma4:26b-q4", "kokoro-82m-q4"]);
        assert_eq!(
            std::fs::read_to_string(dir.join("hypnos.toml.bak")).unwrap(),
            CURRENT
        );
        assert_eq!(
            applied.backup,
            dir.join("hypnos.toml.bak").display().to_string()
        );
        assert!(!overlay.exists());
        assert!(dir.join("hypnos.models.toml.bak").is_file());
        assert!(applied.overlay_backup.is_some());
        assert!(std::fs::read_to_string(&config)
            .unwrap()
            .contains("id = \"kokoro-82m-q4\""));
        assert!(!dir.join("hypnos.toml.tmp").exists());

        // A refused rewrite leaves the config (and the backup) as they were.
        let before = std::fs::read_to_string(&config).unwrap();
        assert!(apply_to_file(&config, "[server]\n").is_err());
        assert_eq!(std::fs::read_to_string(&config).unwrap(), before);
        assert_eq!(
            std::fs::read_to_string(dir.join("hypnos.toml.bak")).unwrap(),
            CURRENT
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_advice_view_reads_sizes_from_the_catalog() {
        let (_, catalog) = catalog::load_catalog().unwrap();
        let root = std::env::temp_dir().join(format!("hypnos-setup-view-{}", std::process::id()));
        let models = advice_view(parse_advice(ADVICE).unwrap(), &catalog, &root, true).unwrap();
        assert_eq!(models.len(), 2);
        assert_eq!(models[0].spec, "gemma4:26b-q4:Q4_0");
        assert!(models[0].size_bytes > 10_000_000_000);
        assert_eq!(models[0].missing_bytes, models[0].size_bytes);
        assert!(!models[0].installed && models[0].kept_loaded && !models[0].partial);
        assert_eq!(models[1].role, "Speaking answers aloud");
        let mut unknown = parse_advice(ADVICE).unwrap();
        unknown.models[0].id = "nope".into();
        assert!(advice_view(unknown, &catalog, &root, true).is_err());
    }

    #[test]
    fn the_voice_check_scores_words_heard_back() {
        assert_eq!(heard_share("Hello there, world.", "hello there world"), 1.0);
        assert!((heard_share("one two three four", "one two") - 0.5).abs() < 1e-9);
        assert_eq!(heard_share("", "x"), 0.0);
        assert_eq!(heard_share("the the", "the"), 0.5);
        let s = speakable("**Hypnos** runs models. It keeps them [ready](http://x). A third one.");
        assert_eq!(s, "Hypnos runs models. It keeps them ready.");
    }

    /// With this checkout's release server: the advice for this machine, as the wizard runs it.
    #[tokio::test]
    async fn advises_with_the_server_binary() {
        let binary =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/target/release/hypnos");
        if !binary.is_file() {
            eprintln!("skipped: no {}", binary.display());
            return;
        }
        let (path, catalog) = catalog::load_catalog().unwrap();
        let advice = match run_advise(&binary, &path, &["chat".into(), "voice".into()]).await {
            Ok(a) => a,
            // A machine too small for any chat model.
            Err(e) if e.contains("enough memory") => return,
            Err(e) => panic!("{e}"),
        };
        assert!(advice
            .models
            .iter()
            .any(|m| m.serves.contains(&"chat".into())));
        for spec in &advice.specs {
            let (id, quant) = split_spec(spec).unwrap();
            catalog::resolve(&catalog, id, quant).unwrap();
        }
        let out = replace_models(CURRENT, &advice.config).unwrap();
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        assert_eq!(model_ids(&doc).len(), advice.specs.len());
    }
}
