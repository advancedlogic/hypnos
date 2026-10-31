//! The model catalog in the Models dialog (API-10): the curated models of `models.catalog.toml`
//! that the config has no `[[models]]` block for yet, with their quantization variants. Adding
//! one downloads the variant's files with the lineup's downloader (`models.rs`) and appends its
//! block to the app's own config.
//!
//! The catalog is read here, not from the server, so the dialog works while the server is
//! stopped: the bundled copy (`Contents/Resources/server/models.catalog.toml`, staged by
//! `stage-server.sh`), else this checkout's `backend/models.catalog.toml`. Its files (repo,
//! remote, local, size, platform) come from this parse: the picker needs sizes and folder
//! entries for every variant up front, which `hypnos --catalog-files` does not print and which
//! would take a process per entry. The `[[models]]` block comes from the server binary
//! (`hypnos --catalog-block <id>:<quant>`), so its rendering stays in one place
//! (`hypnos_core::catalog::CatalogEntry::render_block`).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::ipc::Channel;
use tauri::{AppHandle, Runtime, State};

use crate::models::{
    download_lines, file_len, models_root, part_path, read_doc, tree_size, DownloadEvent,
    ManifestLine, ModelFiles,
};

const FILE_NAME: &str = "models.catalog.toml";
/// The layout version this app reads; a newer catalog is refused.
const CATALOG_VERSION: u32 = 1;
const RENDER_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Deserialize)]
struct Catalog {
    catalog_version: u32,
    #[serde(default)]
    models: Vec<Entry>,
}

/// One `[[models]]` of the catalog; `config` and `architecture` are the server's business.
#[derive(Debug, Deserialize)]
struct Entry {
    id: String,
    name: String,
    publisher: String,
    license: String,
    modality: String,
    #[serde(default)]
    additional_modalities: Vec<String>,
    engine: String,
    #[serde(default)]
    capabilities: Vec<String>,
    summary: String,
    #[serde(default)]
    lineup: bool,
    variants: Vec<Variant>,
}

#[derive(Debug, Deserialize)]
struct Variant {
    quant: String,
    #[serde(default)]
    default: bool,
    #[serde(default)]
    note: Option<String>,
    files: Vec<CatalogFile>,
}

#[derive(Debug, Deserialize)]
struct CatalogFile {
    repo: String,
    /// A trailing `/` is a whole folder.
    remote: String,
    local: String,
    /// Bytes (a folder: all of its files).
    size: u64,
    /// `all`, `macos-arm64` or `other`, as in `models.manifest`.
    #[serde(default = "all")]
    platform: String,
}

fn all() -> String {
    "all".into()
}

fn parse_catalog(text: &str) -> Result<Catalog, String> {
    let catalog: Catalog = toml::from_str(text).map_err(|e| format!("{FILE_NAME}: {e}"))?;
    if catalog.catalog_version > CATALOG_VERSION {
        return Err(format!(
            "{FILE_NAME} version {} is newer than this app understands",
            catalog.catalog_version
        ));
    }
    Ok(catalog)
}

fn for_platform(platform: &str, apple_silicon: bool) -> bool {
    match platform {
        "all" => true,
        "macos-arm64" => apple_silicon,
        "other" => !apple_silicon,
        _ => false,
    }
}

const APPLE_SILICON: bool = cfg!(all(target_os = "macos", target_arch = "aarch64"));

impl Variant {
    fn files_for(&self, apple_silicon: bool) -> impl Iterator<Item = &CatalogFile> {
        self.files
            .iter()
            .filter(move |f| for_platform(&f.platform, apple_silicon))
    }

    /// The files as manifest lines for the downloader.
    fn lines(&self, id: &str, apple_silicon: bool) -> Vec<ManifestLine> {
        self.files_for(apple_silicon)
            .map(|f| ManifestLine {
                model: id.to_string(),
                repo: f.repo.clone(),
                remote: f.remote.clone(),
                local: f.local.clone(),
            })
            .collect()
    }

    /// Bytes on disk under `root` (complete files and `.part`s) and whether every file is whole.
    fn state(&self, root: &Path, apple_silicon: bool) -> (u64, bool) {
        let mut have = 0;
        let mut whole = true;
        for f in self.files_for(apple_silicon) {
            let dest = root.join(&f.local);
            let (n, complete) = if f.remote.ends_with('/') {
                let n = tree_size(&dest);
                (n, dest.is_dir() && !has_part(&dest) && n >= f.size)
            } else if dest.is_file() {
                let n = file_len(&dest);
                (n, n == f.size)
            } else {
                (file_len(&part_path(&dest)), false)
            };
            have += n;
            whole &= complete;
        }
        (have, whole)
    }
}

fn has_part(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok()).any(|e| {
                let p = e.path();
                if p.is_dir() {
                    has_part(&p)
                } else {
                    p.extension().is_some_and(|x| x == "part")
                }
            })
        })
        .unwrap_or(false)
}

impl Entry {
    fn variant(&self, quant: &str) -> Option<&Variant> {
        self.variants
            .iter()
            .find(|v| v.quant.eq_ignore_ascii_case(quant))
    }
}

/// `models.catalog.toml`: the bundled copy, else this checkout's (as `models::manifest_path`).
fn catalog_path() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok()?.parent()?.to_path_buf();
    let bundled = exe_dir.parent()?.join("Resources/server").join(FILE_NAME);
    if bundled.is_file() {
        return Some(bundled);
    }
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../backend")
        .join(FILE_NAME);
    repo.is_file().then_some(repo)
}

fn load_catalog() -> Result<(PathBuf, Catalog), String> {
    let path = catalog_path().ok_or(format!("no {FILE_NAME} in the app or this checkout"))?;
    let text = std::fs::read_to_string(&path)
        .map_err(|e| format!("could not read {}: {e}", path.display()))?;
    Ok((path, parse_catalog(&text)?))
}

/// The ids of the config's `[[models]]` blocks.
fn config_ids(doc: &toml_edit::DocumentMut) -> BTreeSet<String> {
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

/// `text` with `block` (one `[[models]]` with `id`) appended; everything already there stays as
/// written. Refused when the config has a block with that id, or when the result would not hold
/// exactly one more model.
fn append_block(text: &str, block: &str, id: &str) -> Result<String, String> {
    let doc: toml_edit::DocumentMut = text.parse().map_err(|e| format!("the config: {e}"))?;
    let before = config_ids(&doc);
    if before.contains(id) {
        return Err(format!(
            "the config already has a [[models]] block with id \"{id}\""
        ));
    }
    let parsed: toml_edit::DocumentMut = block
        .parse()
        .map_err(|e| format!("the catalog block: {e}"))?;
    let block_ids: Vec<String> = config_ids(&parsed).into_iter().collect();
    if block_ids != [id] {
        return Err(format!(
            "the catalog block is not one [[models]] block for \"{id}\""
        ));
    }
    let mut out = text.trim_end().to_string();
    if !out.is_empty() {
        out.push_str("\n\n");
    }
    out.push_str(block.trim_end());
    out.push('\n');
    let after: toml_edit::DocumentMut = out
        .parse()
        .map_err(|e| format!("the config with the new block does not parse: {e}"))?;
    let count = after
        .get("models")
        .and_then(|m| m.as_array_of_tables())
        .map_or(0, |m| m.len());
    if count != before.len() + 1 || !config_ids(&after).contains(id) {
        return Err("the config's [[models]] are not an array of tables".into());
    }
    Ok(out)
}

/// The variant's `[[models]]` block from `hypnos --catalog-block <id>:<quant>`, run against the
/// catalog the app read.
async fn render_block(binary: &Path, catalog: &Path, spec: &str) -> Result<String, String> {
    if !binary.is_file() {
        return Err(format!(
            "no server binary at {}: the config block comes from `hypnos --catalog-block`",
            binary.display()
        ));
    }
    let mut cmd = tokio::process::Command::new(binary);
    cmd.arg("--catalog-block")
        .arg(spec)
        .env("HYPNOS_CATALOG", catalog)
        .stdin(std::process::Stdio::null())
        .kill_on_drop(true);
    if let Some(dir) = catalog.parent() {
        cmd.current_dir(dir);
    }
    #[cfg(target_os = "macos")]
    if let Some(dir) = binary.parent() {
        cmd.env(
            "DYLD_LIBRARY_PATH",
            crate::local_server::join_paths(dir, "DYLD_LIBRARY_PATH"),
        );
    }
    #[cfg(target_os = "linux")]
    if let Some(dir) = binary.parent() {
        cmd.env(
            "LD_LIBRARY_PATH",
            crate::local_server::join_paths(dir, "LD_LIBRARY_PATH"),
        );
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let out = tokio::time::timeout(RENDER_TIMEOUT, cmd.output())
        .await
        .map_err(|_| format!("{} --catalog-block timed out", binary.display()))?
        .map_err(|e| format!("could not run {}: {e}", binary.display()))?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let err = err.trim();
        return Err(
            if err.contains("--catalog-block") || err.contains("usage") {
                format!("{err} (a server too old for the catalog?)")
            } else {
                format!("hypnos --catalog-block {spec}: {err}")
            },
        );
    }
    String::from_utf8(out.stdout).map_err(|e| e.to_string())
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct VariantView {
    quant: String,
    default: bool,
    note: Option<String>,
    /// Bytes of this machine's files, as the catalog lists them.
    size: u64,
    have: u64,
    installed: bool,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EntryView {
    id: String,
    name: String,
    publisher: String,
    license: String,
    modality: String,
    additional_modalities: Vec<String>,
    engine: String,
    capabilities: Vec<String>,
    summary: String,
    lineup: bool,
    variants: Vec<VariantView>,
}

/// The catalog's models the config has no block for, with each variant's size and what is on
/// disk under the config's models folder.
#[tauri::command]
pub async fn models_catalog_entries(config: String) -> Result<Vec<EntryView>, String> {
    let config = PathBuf::from(config);
    let doc = read_doc(&config)?;
    let root = models_root(&config, &doc);
    let ids = config_ids(&doc);
    let (_, catalog) = load_catalog()?;
    Ok(entries_view(catalog, &ids, &root, APPLE_SILICON))
}

fn entries_view(
    catalog: Catalog,
    ids: &BTreeSet<String>,
    root: &Path,
    apple_silicon: bool,
) -> Vec<EntryView> {
    catalog
        .models
        .into_iter()
        .filter(|e| !ids.contains(&e.id))
        .map(|e| {
            let variants = e
                .variants
                .iter()
                .map(|v| {
                    let (have, installed) = v.state(root, apple_silicon);
                    VariantView {
                        quant: v.quant.clone(),
                        default: v.default,
                        note: v.note.clone(),
                        size: v.files_for(apple_silicon).map(|f| f.size).sum(),
                        have,
                        installed,
                    }
                })
                .collect();
            EntryView {
                id: e.id,
                name: e.name,
                publisher: e.publisher,
                license: e.license,
                modality: e.modality,
                additional_modalities: e.additional_modalities,
                engine: e.engine,
                capabilities: e.capabilities,
                summary: e.summary,
                lineup: e.lineup,
                variants,
            }
        })
        .collect()
}

fn resolve<'a>(
    catalog: &'a Catalog,
    id: &str,
    quant: &str,
) -> Result<(&'a Entry, &'a Variant), String> {
    let entry = catalog
        .models
        .iter()
        .find(|e| e.id == id)
        .ok_or_else(|| format!("no model \"{id}\" in the catalog"))?;
    let variant = entry
        .variant(quant)
        .ok_or_else(|| format!("\"{id}\" has no variant \"{quant}\""))?;
    Ok((entry, variant))
}

/// The `[[models]]` block of a variant, for a config the app does not write (copy it there).
#[tauri::command]
pub async fn models_catalog_block(
    binary: Option<String>,
    id: String,
    quant: String,
) -> Result<String, String> {
    let (path, catalog) = load_catalog()?;
    let (entry, variant) = resolve(&catalog, &id, &quant)?;
    let binary = binary
        .filter(|b| !b.is_empty())
        .ok_or("no server binary chosen: the config block comes from `hypnos --catalog-block`")?;
    render_block(
        Path::new(&binary),
        &path,
        &format!("{}:{}", entry.id, variant.quant),
    )
    .await
}

/// Adds a catalog model to the app's own config: downloads the variant's missing files, then
/// appends its block. `Ok(false)`: the download was stopped and nothing was added. The caller
/// restarts the server afterwards, as after a lineup download.
#[tauri::command]
pub async fn models_catalog_add<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, ModelFiles>,
    config: String,
    binary: Option<String>,
    id: String,
    quant: String,
    on_event: Channel<DownloadEvent>,
) -> Result<bool, String> {
    let config = PathBuf::from(config);
    if !crate::local_server::is_managed(&app, &config) {
        return Err(
            "models can be added to the app's own config only; copy the block into this one".into(),
        );
    }
    let doc = read_doc(&config)?;
    if doc
        .get("server")
        .and_then(|s| s.get("models_dir"))
        .is_none()
    {
        return Err("the config has no [server] models_dir, which the catalog's paths are relative to; choose a models folder first".into());
    }
    let root = models_root(&config, &doc);
    let (path, catalog) = load_catalog()?;
    let (entry, variant) = resolve(&catalog, &id, &quant)?;
    let binary = binary
        .filter(|b| !b.is_empty())
        .ok_or("no server binary chosen: the config block comes from `hypnos --catalog-block`")?;
    let block = render_block(
        Path::new(&binary),
        &path,
        &format!("{}:{}", entry.id, variant.quant),
    )
    .await?;
    // Everything that could refuse the block is checked before the download.
    let read = |p: &Path| {
        std::fs::read_to_string(p).map_err(|e| format!("could not read {}: {e}", p.display()))
    };
    append_block(&read(&config)?, &block, &entry.id)?;

    let _guard = state.exclusive()?;
    let lines = variant.lines(&entry.id, APPLE_SILICON);
    if !download_lines(
        &state,
        &root,
        &lines,
        std::slice::from_ref(&entry.id),
        &on_event,
    )
    .await?
    {
        return Ok(false);
    }
    let text = append_block(&read(&config)?, &block, &entry.id)?;
    std::fs::write(&config, text)
        .map_err(|e| format!("could not write {}: {e}", config.display()))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    const CATALOG: &str = r#"
catalog_version = 1

[[models]]
id = "chat:1b"
name = "Chat 1B"
publisher = "Org"
license = "mit"
architecture = "lfm3"
modality = "text_generation"
additional_modalities = ["vision"]
engine = "ggml_llama"
capabilities = ["thinking", "tools"]
summary = "A chat model."

[models.config]
priority = "high"

[models.config.arch]
default_context_len = 4096

[[models.variants]]
quant = "Q8_0"
default = true
files = [
  { role = "weights", repo = "org/chat", remote = "chat-Q8_0.gguf", local = "chat-Q8_0.gguf", size = 10 },
  { role = "aux", repo = "org/chat", remote = "mmproj.gguf", local = "mmproj.gguf", size = 5 },
]

[[models.variants]]
quant = "Q4_K_M"
note = "Smaller."
files = [
  { role = "weights", repo = "org/chat", remote = "chat-Q4.gguf", local = "chat-Q4.gguf", size = 6 },
]

[[models]]
id = "tts"
name = "TTS"
publisher = "Org"
license = "apache-2.0"
architecture = "kokoro"
modality = "text_to_speech"
engine = "onnx_runtime"
summary = "Speech."
lineup = true

[[models.variants]]
quant = "fp32"
default = true
files = [
  { role = "weights", repo = "org/tts", remote = "onnx/model.onnx", local = "tts/model.onnx", size = 4, platform = "macos-arm64" },
  { role = "weights", repo = "org/tts", remote = "onnx/fp16.onnx", local = "tts/fp16.onnx", size = 2, platform = "other" },
  { role = "voices", repo = "org/tts", remote = "voices/", local = "tts/voices/", size = 3 },
]
"#;

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("hypnos-cat-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parses_entries_and_refuses_a_newer_catalog() {
        let c = parse_catalog(CATALOG).unwrap();
        assert_eq!(c.models.len(), 2);
        let chat = &c.models[0];
        assert_eq!(chat.capabilities, ["thinking", "tools"]);
        assert_eq!(chat.additional_modalities, ["vision"]);
        assert_eq!(chat.variants.len(), 2);
        assert!(chat.variant("q4_k_m").is_some());
        assert_eq!(
            chat.variant("Q4_K_M").unwrap().note.as_deref(),
            Some("Smaller.")
        );
        assert!(c.models[1].lineup);
        assert!(parse_catalog("catalog_version = 2\n").is_err());
        assert!(resolve(&c, "chat:1b", "Q2").is_err());
        assert!(resolve(&c, "nope", "Q8_0").is_err());
    }

    #[test]
    fn the_real_catalog_parses_with_one_default_per_entry() {
        let (_, c) = load_catalog().unwrap();
        assert!(!c.models.is_empty());
        for e in &c.models {
            assert_eq!(
                e.variants.iter().filter(|v| v.default).count(),
                1,
                "{}",
                e.id
            );
            for v in &e.variants {
                assert!(v.files_for(true).count() > 0 && v.files_for(false).count() > 0);
            }
        }
    }

    #[test]
    fn picks_this_platforms_files() {
        let c = parse_catalog(CATALOG).unwrap();
        let tts = &c.models[1].variants[0];
        let mac: Vec<&str> = tts.files_for(true).map(|f| f.local.as_str()).collect();
        assert_eq!(mac, ["tts/model.onnx", "tts/voices/"]);
        let other: Vec<&str> = tts.files_for(false).map(|f| f.local.as_str()).collect();
        assert_eq!(other, ["tts/fp16.onnx", "tts/voices/"]);
        let lines = tts.lines("tts", true);
        assert_eq!(lines[1].remote, "voices/");
        assert!(lines.iter().all(|l| l.model == "tts"));
    }

    #[test]
    fn lists_models_not_in_the_config_with_what_is_on_disk() {
        let root = scratch("state");
        let c = parse_catalog(CATALOG).unwrap();
        // Q8_0: weights complete, projector half there as a .part.
        std::fs::write(root.join("chat-Q8_0.gguf"), [0u8; 10]).unwrap();
        std::fs::write(root.join("mmproj.gguf.part"), [0u8; 2]).unwrap();
        // Q4_K_M: whole. tts: model and the voices folder whole.
        std::fs::write(root.join("chat-Q4.gguf"), [0u8; 6]).unwrap();
        std::fs::create_dir_all(root.join("tts/voices")).unwrap();
        std::fs::write(root.join("tts/model.onnx"), [0u8; 4]).unwrap();
        std::fs::write(root.join("tts/voices/a.bin"), [0u8; 3]).unwrap();

        let view = entries_view(c, &BTreeSet::new(), &root, true);
        let chat = &view[0];
        assert_eq!((chat.variants[0].size, chat.variants[0].have), (15, 12));
        assert!(!chat.variants[0].installed);
        assert!(chat.variants[1].installed);
        assert!(view[1].variants[0].installed);

        // A .part inside a folder: not whole yet.
        std::fs::write(root.join("tts/voices/b.bin.part"), [0u8; 1]).unwrap();
        let view = entries_view(
            parse_catalog(CATALOG).unwrap(),
            &BTreeSet::new(),
            &root,
            true,
        );
        assert!(!view[1].variants[0].installed);

        // Models the config already has are not offered.
        let ids = BTreeSet::from(["chat:1b".to_string()]);
        let view = entries_view(parse_catalog(CATALOG).unwrap(), &ids, &root, true);
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].id, "tts");
        let _ = std::fs::remove_dir_all(&root);
    }

    const CONFIG: &str = "config_version = 1\n\n[server]\nmodels_dir = \"models\" # where they live\n\n[[models]]\nid = \"a\"\nweights_path = \"a.gguf\"\n\n[models.arch]\nnum_layers = 2\n";
    const BLOCK: &str = "# B (Q8_0). From models.catalog.toml.\n[[models]]\nid = \"b\"\nweights_path = \"b.gguf\"\n\n[models.arch]\ndefault_context_len = 4096\n";

    #[test]
    fn appends_a_block_and_keeps_the_rest_of_the_file() {
        let out = append_block(CONFIG, BLOCK, "b").unwrap();
        assert!(
            out.starts_with(CONFIG.trim_end()),
            "the old text is unchanged"
        );
        assert!(out.ends_with(BLOCK));
        let doc: toml_edit::DocumentMut = out.parse().unwrap();
        let models = doc["models"].as_array_of_tables().unwrap();
        assert_eq!(models.len(), 2);
        // `[models.arch]` of the new block belongs to the new model, not to `a`.
        assert_eq!(
            models.get(0).unwrap()["arch"]["num_layers"].as_integer(),
            Some(2)
        );
        assert_eq!(
            models.get(1).unwrap()["arch"]["default_context_len"].as_integer(),
            Some(4096)
        );
        assert!(models.get(0).unwrap()["arch"]
            .get("default_context_len")
            .is_none());
        // Into an empty config too.
        assert_eq!(append_block("", BLOCK, "b").unwrap(), BLOCK);
    }

    #[test]
    fn refuses_a_duplicate_id_and_a_wrong_block() {
        let err = append_block(CONFIG, &BLOCK.replace("\"b\"", "\"a\""), "a").unwrap_err();
        assert!(err.contains("already has"), "{err}");
        let twice = append_block(&append_block(CONFIG, BLOCK, "b").unwrap(), BLOCK, "b");
        assert!(twice.unwrap_err().contains("already has"));
        assert!(append_block(CONFIG, BLOCK, "c").is_err());
        assert!(append_block(CONFIG, "not toml [", "b").is_err());
        // An inline `models = [...]` cannot take a [[models]] table.
        assert!(append_block("models = []\n", BLOCK, "b").is_err());
    }

    /// With this checkout's release server: the block comes from `hypnos --catalog-block`.
    #[tokio::test]
    async fn renders_a_block_with_the_server_binary() {
        let binary =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../backend/target/release/hypnos");
        if !binary.is_file() {
            eprintln!("skipped: no {}", binary.display());
            return;
        }
        let (path, _) = load_catalog().unwrap();
        let block = render_block(&binary, &path, "lfm3:1.7b:Q4_K_M")
            .await
            .unwrap();
        assert!(
            block.contains("weights_path = \"LFM3-1.7B-Q4_K_M.gguf\""),
            "{block}"
        );
        let out = append_block(CONFIG, &block, "lfm3:1.7b").unwrap();
        assert!(out.contains("id = \"lfm3:1.7b\""));
        assert!(render_block(&binary, &path, "nope:x").await.is_err());
        assert!(render_block(Path::new("/no/hypnos"), &path, "x")
            .await
            .is_err());
    }
}
