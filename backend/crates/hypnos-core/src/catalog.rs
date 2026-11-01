//! The model catalog (API-10): `backend/models.catalog.toml`, a curated list of models Hypnos has
//! been tested with, each with its quantization variants, the files they need and the
//! `[[models]]` block that runs them.
//!
//! The catalog is curated rather than searched live: a model that loads is not yet a model that
//! works (LFM3 needed its own prompt renderer, transcription prompt and checkpointing before it
//! answered properly). A `ggml_llama` entry must name an architecture of
//! [`TESTED_GGUF_ARCHITECTURES`], and when its weights are on disk their GGUF must report it
//! ([`CatalogEntry::check_weights`]). `models.manifest` stays the default lineup the fetch
//! scripts and the desktop app download; the catalog repeats those entries with `lineup = true`
//! (a test keeps the two in step) and adds the others.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::{EngineKind, HypnosConfig, Modality};

/// Version of the catalog's layout; a newer file is refused.
pub const CATALOG_VERSION: u32 = 1;

/// `general.architecture` values of GGUF models Hypnos runs properly through llama.cpp: prompt
/// format, media markers, thinking and memory handling are implemented and checked for them.
pub const TESTED_GGUF_ARCHITECTURES: &[&str] = &[
    "gemma4",
    "gemma-embedding2",
    "lfm3",
    "qwen35",
    "qwen35moe",
    "qwen3vl",
    "qwen3tts",
    "qwen3moe",
    "lfm2moe",
    "deepseek2",
    "nemotron_h_moe",
];

/// Keys the rendered block writes itself; an entry's `config` may not set them.
const GENERATED_KEYS: &[&str] = &[
    "id",
    "modality",
    "additional_modalities",
    "engine",
    "weights_path",
    "aux_weights_path",
    "text_encoder_path",
    "tokenizer_path",
    "voices_path",
    "total_weight_bytes",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCatalog {
    pub catalog_version: u32,
    #[serde(rename = "models")]
    pub entries: Vec<CatalogEntry>,
}

/// What a model can do beyond its modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    /// Has a thinking block (`reasoning_effort`).
    Thinking,
    /// Has a native tool-calling format in Hypnos.
    Tools,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogEntry {
    /// The model id the rendered block uses (`[[models]] id`).
    pub id: String,
    pub name: String,
    pub publisher: String,
    pub license: String,
    /// The GGUF's `general.architecture` for `ggml_llama`; a free label for other engines.
    pub architecture: String,
    pub modality: Modality,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub additional_modalities: Vec<Modality>,
    pub engine: EngineKind,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub capabilities: Vec<Capability>,
    /// One or two sentences for a picker.
    pub summary: String,
    /// Part of the default lineup (`models.manifest`, the example configs).
    #[serde(default)]
    pub lineup: bool,
    /// Further `[[models]]` keys, `arch` as the `[models.arch]` table.
    #[serde(default)]
    pub config: toml::Table,
    /// `[models.arch]` values the server reads from the GGUF at startup (layers, KV heads, head
    /// size, context, sliding window, expert bytes), for sizing the model without its files (the
    /// advisor, API-12). Never written into the block: the GGUF stays the source at runtime.
    #[serde(default, skip_serializing_if = "toml::Table::is_empty")]
    pub estimate: toml::Table,
    pub variants: Vec<CatalogVariant>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogVariant {
    /// The quantization, e.g. `Q4_K_M`; selects the variant (`<id>:<quant>` in the fetch scripts).
    pub quant: String,
    /// The variant a picker preselects; exactly one per entry.
    #[serde(default)]
    pub default: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    pub files: Vec<CatalogFile>,
}

/// Which `[[models]]` path a file fills.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileRole {
    Weights,
    /// `aux_weights_path`: a projector (mmproj) or a VAE.
    Aux,
    TextEncoder,
    Tokenizer,
    Voices,
    /// Needed beside the weights but named by no key (docling's other models).
    Data,
}

/// Which machines a file is for, as in `models.manifest`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Platform {
    #[default]
    All,
    MacosArm64,
    Other,
}

impl Platform {
    pub fn includes(self, apple_silicon: bool) -> bool {
        match self {
            Platform::All => true,
            Platform::MacosArm64 => apple_silicon,
            Platform::Other => !apple_silicon,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatalogFile {
    pub role: FileRole,
    /// A Hugging Face repository, or an https:// base URL the remote path is appended to.
    pub repo: String,
    /// Path inside the repository; a trailing `/` is a whole folder.
    pub remote: String,
    /// Path under the models folder (a trailing `/` for folders).
    pub local: String,
    /// For a folder: the file inside it that the config path names (GLiNER's encoder).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub entry: Option<String>,
    /// Bytes (a folder: all of its files), for display before the host is asked.
    pub size: u64,
    #[serde(default)]
    pub platform: Platform,
    /// A gated Hugging Face repository: the download needs a token whose account accepted the
    /// model's license (`HF_TOKEN` for the fetch scripts, Settings in the desktop app).
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub gated: bool,
}

impl ModelCatalog {
    /// Parses and checks a catalog; every problem is reported, one per line.
    pub fn from_toml_str(text: &str) -> Result<Self, String> {
        let catalog: Self = toml::from_str(text).map_err(|e| e.to_string())?;
        if catalog.catalog_version > CATALOG_VERSION {
            return Err(format!(
                "catalog_version {} is newer than this build reads ({CATALOG_VERSION})",
                catalog.catalog_version
            ));
        }
        let errors = catalog.validate();
        if errors.is_empty() {
            Ok(catalog)
        } else {
            Err(errors.join("\n"))
        }
    }

    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("could not read {}: {e}", path.display()))?;
        Self::from_toml_str(&text).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn entry(&self, id: &str) -> Option<&CatalogEntry> {
        self.entries.iter().find(|e| e.id == id)
    }

    /// `<id>` or `<id>:<quant>` (ids contain `:` themselves, so the whole spec is tried as an id
    /// first): the entry and the chosen (else default) variant.
    pub fn resolve(&self, spec: &str) -> Result<(&CatalogEntry, &CatalogVariant), String> {
        let (entry, quant) = match self.entry(spec) {
            Some(e) => (e, None),
            None => {
                let (id, quant) = spec
                    .rsplit_once(':')
                    .ok_or_else(|| format!("no model '{spec}' in the catalog"))?;
                let e = self
                    .entry(id)
                    .ok_or_else(|| format!("no model '{spec}' in the catalog"))?;
                (e, Some(quant))
            }
        };
        let variant = entry.variant(quant).ok_or_else(|| {
            let quants: Vec<&str> = entry.variants.iter().map(|v| v.quant.as_str()).collect();
            format!(
                "'{}' has no variant '{}' (variants: {})",
                entry.id,
                quant.unwrap_or_default(),
                quants.join(", ")
            )
        })?;
        Ok((entry, variant))
    }

    fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let at = format!("model '{}'", e.id);
            if self.entries[..i].iter().any(|o| o.id == e.id) {
                errors.push(format!("{at}: duplicate id"));
            }
            if e.engine == EngineKind::GgmlLlama
                && !TESTED_GGUF_ARCHITECTURES.contains(&e.architecture.as_str())
            {
                errors.push(format!(
                    "{at}: architecture '{}' is not one Hypnos has been tested with ({})",
                    e.architecture,
                    TESTED_GGUF_ARCHITECTURES.join(", ")
                ));
            }
            for key in GENERATED_KEYS {
                if e.config.contains_key(*key) {
                    errors.push(format!(
                        "{at}: config.{key} is written from the entry itself"
                    ));
                }
            }
            if e.variants.iter().filter(|v| v.default).count() != 1 {
                errors.push(format!("{at}: exactly one variant must be the default"));
            }
            for (j, v) in e.variants.iter().enumerate() {
                let at = format!("{at} variant '{}'", v.quant);
                if e.variants[..j].iter().any(|o| o.quant == v.quant) {
                    errors.push(format!("{at}: duplicate quant"));
                }
                for role in [
                    FileRole::Weights,
                    FileRole::Aux,
                    FileRole::TextEncoder,
                    FileRole::Tokenizer,
                    FileRole::Voices,
                ] {
                    let n = v.files.iter().filter(|f| f.role == role).count();
                    if n > 1 || (role == FileRole::Weights && n == 0) {
                        errors.push(format!("{at}: needs one {role:?} file, has {n}"));
                    }
                }
                // The block must be one the server reads without a warning.
                match HypnosConfig::from_toml_str(&format!(
                    "config_version = 1\n{}",
                    e.render_block(v)
                )) {
                    Ok(c) if !c.unknown_keys.is_empty() => errors.push(format!(
                        "{at}: unknown config keys {}",
                        c.unknown_keys.join(", ")
                    )),
                    Ok(_) => {}
                    Err(err) => {
                        errors.push(format!("{at}: the rendered block does not parse: {err}"))
                    }
                }
            }
        }
        errors
    }
}

impl CatalogEntry {
    /// The variant with `quant` (case-insensitive), or the default one.
    pub fn variant(&self, quant: Option<&str>) -> Option<&CatalogVariant> {
        match quant {
            Some(q) => self
                .variants
                .iter()
                .find(|v| v.quant.eq_ignore_ascii_case(q)),
            None => self.variants.iter().find(|v| v.default),
        }
    }

    /// The `[[models]]` block that runs `variant`, with paths relative to `models_dir` (the
    /// manifest's `local` column). Folders (`voices/`) are written without their slash, or as
    /// the file `entry` names inside them.
    pub fn render_block(&self, variant: &CatalogVariant) -> String {
        let q = |s: &str| toml::Value::String(s.to_string()).to_string();
        let path = |role: FileRole| {
            variant
                .files
                .iter()
                .find(|f| f.role == role)
                .map(|f| match &f.entry {
                    Some(entry) => format!("{}{entry}", f.local),
                    None => f.local.trim_end_matches('/').to_string(),
                })
        };
        let total: u64 = variant.files.iter().map(|f| f.size).sum();
        let mut out = format!(
            "# {} ({}), {}, license {}. From models.catalog.toml.\n[[models]]\nid = {}\nmodality = {}\n",
            self.name,
            variant.quant,
            self.publisher,
            self.license,
            q(&self.id),
            q(modality_name(self.modality)),
        );
        if !self.additional_modalities.is_empty() {
            let mods: Vec<String> = self
                .additional_modalities
                .iter()
                .map(|m| q(modality_name(*m)))
                .collect();
            out += &format!("additional_modalities = [{}]\n", mods.join(", "));
        }
        out += &format!("engine = {}\n", q(engine_name(self.engine)));
        for (key, role) in [
            ("weights_path", FileRole::Weights),
            ("aux_weights_path", FileRole::Aux),
            ("text_encoder_path", FileRole::TextEncoder),
            ("tokenizer_path", FileRole::Tokenizer),
            ("voices_path", FileRole::Voices),
        ] {
            if let Some(p) = path(role) {
                out += &format!("{key} = {}\n", q(&p));
            }
        }
        out += &format!("total_weight_bytes = {total}\n");
        let mut arch = None;
        for (key, value) in &self.config {
            match value {
                toml::Value::Table(t) if key == "arch" => arch = Some(t),
                _ => out += &format!("{key} = {value}\n"),
            }
        }
        out += "\n[models.arch]\n";
        if let Some(arch) = arch {
            for (key, value) in arch {
                out += &format!("{key} = {value}\n");
            }
        }
        out
    }

    /// The `ModelConfig` of `variant` as the server would run it, with the `estimate` values for
    /// the `[models.arch]` keys the block leaves to the GGUF (paths stay relative).
    pub fn model_config(
        &self,
        variant: &CatalogVariant,
    ) -> Result<crate::config::ModelConfig, String> {
        let mut doc: toml::Table = toml::from_str(&format!(
            "config_version = 1\n{}",
            self.render_block(variant)
        ))
        .map_err(|e| e.to_string())?;
        if let Some(toml::Value::Table(model)) = doc
            .get_mut("models")
            .and_then(|m| m.as_array_mut())
            .and_then(|a| a.first_mut())
        {
            let arch = model
                .entry("arch")
                .or_insert_with(|| toml::Value::Table(toml::Table::new()));
            if let toml::Value::Table(arch) = arch {
                for (k, v) in &self.estimate {
                    arch.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
        }
        let text = toml::to_string(&doc).map_err(|e| e.to_string())?;
        let mut config = HypnosConfig::from_toml_str(&text).map_err(|e| e.to_string())?;
        config
            .models
            .pop()
            .ok_or_else(|| "no [[models]] block".to_string())
    }

    /// With the weights of `variant` under `models_dir`: an error when their GGUF reports an
    /// architecture other than the entry's (a wrong file, or a renamed one). Missing or non-GGUF
    /// weights pass.
    pub fn check_weights(&self, variant: &CatalogVariant, models_dir: &Path) -> Result<(), String> {
        if self.engine != EngineKind::GgmlLlama {
            return Ok(());
        }
        let Some(weights) = variant.files.iter().find(|f| f.role == FileRole::Weights) else {
            return Ok(());
        };
        let path = models_dir.join(&weights.local);
        match crate::gguf::GgufArchMetadata::read(&path) {
            Ok(Some(meta)) if meta.architecture != self.architecture => Err(format!(
                "{} reports architecture '{}', the catalog expects '{}'",
                path.display(),
                meta.architecture,
                self.architecture
            )),
            _ => Ok(()),
        }
    }
}

impl CatalogVariant {
    /// The files for this machine.
    pub fn files_for(&self, apple_silicon: bool) -> impl Iterator<Item = &CatalogFile> {
        self.files
            .iter()
            .filter(move |f| f.platform.includes(apple_silicon))
    }

    pub fn size_for(&self, apple_silicon: bool) -> u64 {
        self.files_for(apple_silicon).map(|f| f.size).sum()
    }
}

fn modality_name(m: Modality) -> &'static str {
    match m {
        Modality::TextGeneration => "text_generation",
        Modality::Vision => "vision",
        Modality::SpeechToText => "speech_to_text",
        Modality::TextToSpeech => "text_to_speech",
        Modality::Embeddings => "embeddings",
        Modality::ImageGeneration => "image_generation",
        Modality::DocumentConversion => "document_conversion",
        Modality::EntityExtraction => "entity_extraction",
    }
}

fn engine_name(e: EngineKind) -> &'static str {
    match e {
        EngineKind::GgmlLlama => "ggml_llama",
        EngineKind::GgmlWhisper => "ggml_whisper",
        EngineKind::GgmlDiffusion => "ggml_diffusion",
        EngineKind::OnnxRuntime => "onnx_runtime",
        EngineKind::Docling => "docling",
        EngineKind::Gliner => "gliner",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn backend() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
    }

    fn catalog() -> ModelCatalog {
        ModelCatalog::load(&backend().join("models.catalog.toml")).unwrap()
    }

    #[test]
    fn the_shipped_catalog_is_valid() {
        let c = catalog();
        assert!(c.entries.len() >= 10);
        let lfm = c.entry("lfm3-vl:2b").unwrap();
        let block = lfm.render_block(lfm.variant(None).unwrap());
        assert!(block.contains("aux_weights_path = \"mmproj-LFM3-VL-2B-BF16.gguf\""));
        let parsed = HypnosConfig::from_toml_str(&format!("config_version = 1\n{block}")).unwrap();
        assert_eq!(parsed.models[0].id, "lfm3-vl:2b");
        assert_eq!(parsed.models[0].total_weight_bytes, 2_225_234_528);
        assert_eq!(
            lfm.variant(Some("q4_k_m")).unwrap().quant,
            "Q4_K_M",
            "quant lookup ignores case"
        );
    }

    /// The default variant of every lineup entry downloads exactly the manifest's files.
    #[test]
    fn lineup_entries_match_the_manifest() {
        let manifest = std::fs::read_to_string(backend().join("models.manifest")).unwrap();
        let mut lines: Vec<(String, String, String, String)> = manifest
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .map(|l| {
                let c: Vec<&str> = l.split_whitespace().collect();
                (c[0].into(), c[1].into(), c[2].into(), c[3].into())
            })
            // Manifest-only extras that no [[models]] block of their own uses.
            .filter(|l| l.0 != "docling-gpu")
            .collect();
        lines.sort();
        let mut from_catalog: Vec<(String, String, String, String)> = catalog()
            .entries
            .iter()
            .filter(|e| e.lineup)
            .flat_map(|e| {
                e.variant(None).unwrap().files.iter().map(|f| {
                    (
                        e.id.clone(),
                        f.repo.clone(),
                        f.remote.clone(),
                        f.local.clone(),
                    )
                })
            })
            .collect();
        from_catalog.sort();
        assert_eq!(from_catalog, lines);
    }

    #[test]
    fn model_configs_take_the_estimate_where_the_gguf_would_fill_in() {
        let c = catalog();
        let e = c.entry("lfm3:1.7b").unwrap();
        let m = e.model_config(e.variant(None).unwrap()).unwrap();
        assert_eq!((m.arch.num_layers, m.arch.num_kv_heads), (24, 4));
        assert_eq!(
            m.arch.default_context_len, 16384,
            "the block's own value wins"
        );
        assert!(
            !e.render_block(e.variant(None).unwrap())
                .contains("num_layers"),
            "the estimate never reaches the block"
        );
        // Every ggml_llama entry can be sized without its files.
        for e in c
            .entries
            .iter()
            .filter(|e| e.engine == EngineKind::GgmlLlama)
        {
            let m = e.model_config(e.variant(None).unwrap()).unwrap();
            assert!(m.arch.num_layers > 1 && m.arch.head_dim > 0, "{}", e.id);
        }
    }

    #[test]
    fn specs_name_an_entry_and_maybe_a_quant() {
        let c = catalog();
        let (e, v) = c.resolve("lfm3:1.7b").unwrap();
        assert_eq!((e.id.as_str(), v.quant.as_str()), ("lfm3:1.7b", "Q8_0"));
        let (e, v) = c.resolve("lfm3:1.7b:q4_k_m").unwrap();
        assert_eq!((e.id.as_str(), v.quant.as_str()), ("lfm3:1.7b", "Q4_K_M"));
        assert!(c
            .resolve("lfm3:1.7b:Q2")
            .unwrap_err()
            .contains("variants: Q8_0, Q4_K_M"));
        assert!(c.resolve("nope").is_err());
    }

    #[test]
    fn untested_architectures_and_overridden_keys_are_refused() {
        let text = r#"
catalog_version = 1
[[models]]
id = "x"
name = "X"
publisher = "P"
license = "mit"
architecture = "qwen3"
modality = "text_generation"
engine = "ggml_llama"
summary = "s"
config = { weights_path = "other.gguf" }
[[models.variants]]
quant = "Q4_0"
files = [{ role = "weights", repo = "r", remote = "x.gguf", local = "x.gguf", size = 1 }]
"#;
        let err = ModelCatalog::from_toml_str(text).unwrap_err();
        assert!(err.contains("architecture 'qwen3'"), "{err}");
        assert!(err.contains("config.weights_path"), "{err}");
        assert!(err.contains("exactly one variant"), "{err}");
    }

    #[test]
    fn weights_of_another_architecture_are_reported() {
        let c = catalog();
        let e = c.entry("lfm3:1.7b").unwrap();
        let v = e.variant(None).unwrap();
        let dir = std::env::temp_dir().join(format!("hypnos-catalog-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert!(e.check_weights(v, &dir).is_ok(), "missing weights pass");
        // A GGUF header that says "qwen3".
        let mut gguf = b"GGUF".to_vec();
        gguf.extend(3u32.to_le_bytes());
        gguf.extend(0u64.to_le_bytes());
        gguf.extend(1u64.to_le_bytes());
        let key = b"general.architecture";
        gguf.extend((key.len() as u64).to_le_bytes());
        gguf.extend(key);
        gguf.extend(8u32.to_le_bytes());
        gguf.extend(5u64.to_le_bytes());
        gguf.extend(b"qwen3");
        std::fs::write(dir.join(&v.files[0].local), gguf).unwrap();
        let err = e.check_weights(v, &dir).unwrap_err();
        assert!(err.contains("'qwen3'"), "{err}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
