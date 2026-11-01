//! Workload profiles (MEM-7): `[server] profile = "voice_assistant"` turns the knobs into an
//! outcome. A profile is a preset, not a layer of behaviour: before the config is deserialized it
//! fills in the `[[models]]` keys a block leaves out (priority, starting tier, warm memory, slots,
//! context) according to each model's role, and `[limits] max_queue_per_model`. Keys a block or
//! `[limits]` sets explicitly always win, and `hypnos --check-config` lists what the profile set.
//! Models added at runtime (`POST /v1/models`) and from the overlay get the same treatment.
//! The latency objective it names is reported in `/v1/memory/status` for the admission work of
//! PERF-10.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkloadProfile {
    /// People chatting: the chat models loaded and quick, search kept, the rest on demand.
    InteractiveChat,
    /// A voice assistant: speech in, chat, speech out, all loaded and never evicted.
    VoiceAssistant,
    /// Documents through conversion, entities and embeddings: those loaded, deep queues.
    DocumentPipeline,
    /// Throughput over latency: more slots, deep queues, models loaded on demand.
    BatchProcessing,
}

/// What a profile promises, for clients and for admission (PERF-10).
#[derive(Debug, Clone, Serialize)]
pub struct LatencyObjective {
    /// What is measured: `first_token`, `first_audio`, or `throughput` (no latency target).
    pub metric: &'static str,
    /// Target in milliseconds; absent for throughput profiles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target_ms: Option<u64>,
}

/// One model's settings under a profile; `None` = the profile says nothing.
#[derive(Debug, Clone, Copy, Default)]
struct Preset {
    priority: Option<&'static str>,
    initial_tier: Option<&'static str>,
    keep_warm_in_ram: Option<bool>,
    slots: Option<i64>,
    default_context_len: Option<i64>,
}

const PRIORITY_ORDER: [&str; 4] = ["low", "normal", "high", "realtime_pinned"];
const TIER_ORDER: [&str; 3] = ["tier3_nvme_ssd", "tier2_pinned_ram", "tier1_gpu_vram"];

impl Preset {
    /// Several roles in one model: the highest priority, the earliest load, warm if any is,
    /// the fewest slots and the shortest context any role asks for.
    fn merge(self, o: Preset) -> Preset {
        let max_by = |a: Option<&'static str>, b: Option<&'static str>, order: &[&str]| match (a, b)
        {
            (Some(x), Some(y)) => {
                let rank = |v| order.iter().position(|o| *o == v).unwrap_or(0);
                Some(if rank(y) > rank(x) { y } else { x })
            }
            (x, y) => x.or(y),
        };
        let min = |a: Option<i64>, b: Option<i64>| match (a, b) {
            (Some(x), Some(y)) => Some(x.min(y)),
            (x, y) => x.or(y),
        };
        Preset {
            priority: max_by(self.priority, o.priority, &PRIORITY_ORDER),
            initial_tier: max_by(self.initial_tier, o.initial_tier, &TIER_ORDER),
            keep_warm_in_ram: match (self.keep_warm_in_ram, o.keep_warm_in_ram) {
                (Some(a), Some(b)) => Some(a || b),
                (a, b) => a.or(b),
            },
            slots: min(self.slots, o.slots),
            default_context_len: min(self.default_context_len, o.default_context_len),
        }
    }
}

fn preset(priority: &'static str, tier: &'static str, warm: bool) -> Preset {
    Preset {
        priority: Some(priority),
        initial_tier: Some(tier),
        keep_warm_in_ram: Some(warm),
        ..Preset::default()
    }
}

const T1: &str = "tier1_gpu_vram";
const T3: &str = "tier3_nvme_ssd";

impl WorkloadProfile {
    pub fn objective(self) -> LatencyObjective {
        match self {
            WorkloadProfile::InteractiveChat => LatencyObjective {
                metric: "first_token",
                target_ms: Some(1000),
            },
            WorkloadProfile::VoiceAssistant => LatencyObjective {
                metric: "first_audio",
                target_ms: Some(1500),
            },
            WorkloadProfile::DocumentPipeline | WorkloadProfile::BatchProcessing => {
                LatencyObjective {
                    metric: "throughput",
                    target_ms: None,
                }
            }
        }
    }

    fn max_queue_per_model(self) -> i64 {
        match self {
            WorkloadProfile::InteractiveChat => 8,
            WorkloadProfile::VoiceAssistant => 4,
            WorkloadProfile::DocumentPipeline => 32,
            WorkloadProfile::BatchProcessing => 64,
        }
    }

    /// The settings for a model serving `modality` under this profile.
    fn for_role(self, modality: &str) -> Preset {
        use WorkloadProfile::*;
        let chat = matches!(modality, "text_generation" | "vision");
        match (self, modality) {
            (InteractiveChat, _) if chat => Preset {
                slots: Some(2),
                ..preset("high", T1, true)
            },
            (InteractiveChat, "embeddings") => preset("realtime_pinned", T1, true),
            (InteractiveChat, "image_generation") => preset("low", T3, false),
            (InteractiveChat, _) => preset("normal", T3, false),

            (VoiceAssistant, _) if chat => Preset {
                slots: Some(1),
                default_context_len: Some(8192),
                ..preset("high", T1, true)
            },
            (VoiceAssistant, "speech_to_text" | "text_to_speech") => {
                preset("realtime_pinned", T1, true)
            }
            (VoiceAssistant, "embeddings") => preset("normal", T3, true),
            (VoiceAssistant, _) => preset("low", T3, false),

            (DocumentPipeline, "document_conversion" | "entity_extraction" | "embeddings") => {
                preset("high", T1, true)
            }
            (DocumentPipeline, _) if chat => Preset {
                slots: Some(2),
                ..preset("normal", T3, true)
            },
            (DocumentPipeline, _) => preset("low", T3, false),

            (BatchProcessing, "document_conversion" | "entity_extraction" | "embeddings") => {
                preset("high", T1, true)
            }
            (BatchProcessing, _) if chat => Preset {
                slots: Some(4),
                ..preset("normal", T3, true)
            },
            (BatchProcessing, _) => preset("normal", T3, false),
        }
    }

    /// Fills in what a `[[models]]` table leaves out; returns the keys it set.
    pub fn apply_to_model(self, model: &mut toml::Table) -> Vec<String> {
        let mut modalities: Vec<String> = Vec::new();
        if let Some(m) = model.get("modality").and_then(|v| v.as_str()) {
            modalities.push(m.to_string());
        }
        if let Some(extra) = model
            .get("additional_modalities")
            .and_then(|v| v.as_array())
        {
            modalities.extend(extra.iter().filter_map(|v| v.as_str()).map(str::to_string));
        }
        let Some(p) = modalities
            .iter()
            .map(|m| self.for_role(m))
            .reduce(Preset::merge)
        else {
            return Vec::new();
        };
        let mut set = Vec::new();
        let mut put = |t: &mut toml::Table, k: &str, v: toml::Value| {
            if !t.contains_key(k) {
                set.push(format!("{k} = {v}"));
                t.insert(k.to_string(), v);
            }
        };
        let s = |v: &str| toml::Value::String(v.to_string());
        if let Some(v) = p.priority {
            put(model, "priority", s(v));
        }
        if let Some(v) = p.initial_tier {
            put(model, "initial_tier", s(v));
        }
        if let Some(v) = p.keep_warm_in_ram {
            put(model, "keep_warm_in_ram", toml::Value::Boolean(v));
        }
        let llama = model.get("engine").and_then(|v| v.as_str()) == Some("ggml_llama");
        if llama {
            if let Some(v) = p.slots {
                put(model, "slots", toml::Value::Integer(v));
            }
            if let Some(v) = p.default_context_len {
                let arch = model
                    .entry("arch")
                    .or_insert_with(|| toml::Value::Table(toml::Table::new()));
                if let toml::Value::Table(arch) = arch {
                    if !arch.contains_key("default_context_len") {
                        set.push(format!("arch.default_context_len = {v}"));
                        arch.insert("default_context_len".into(), toml::Value::Integer(v));
                    }
                }
            }
        }
        set
    }

    /// Applies the profile named in `doc`'s `[server] profile` to its models and `[limits]`.
    /// Returns one note per model (and for the limits) listing what it set.
    pub fn apply_to_document(doc: &mut toml::Table) -> Result<Vec<String>, String> {
        let Some(name) = doc
            .get("server")
            .and_then(|s| s.get("profile"))
            .and_then(|v| v.as_str())
        else {
            return Ok(Vec::new());
        };
        let profile: WorkloadProfile = toml::Value::String(name.to_string())
            .try_into()
            .map_err(|_| {
                format!(
                    "server.profile = {name:?} is not a profile (interactive_chat, voice_assistant, document_pipeline, batch_processing)"
                )
            })?;
        let mut notes = Vec::new();
        if let Some(toml::Value::Array(models)) = doc.get_mut("models") {
            for m in models.iter_mut() {
                if let toml::Value::Table(t) = m {
                    let id = t
                        .get("id")
                        .and_then(|v| v.as_str())
                        .unwrap_or("?")
                        .to_string();
                    let set = profile.apply_to_model(t);
                    if !set.is_empty() {
                        notes.push(format!("'{id}': {}", set.join(", ")));
                    }
                }
            }
        }
        let limits = doc
            .entry("limits")
            .or_insert_with(|| toml::Value::Table(toml::Table::new()));
        if let toml::Value::Table(l) = limits {
            if !l.contains_key("max_queue_per_model") {
                let q = profile.max_queue_per_model();
                l.insert("max_queue_per_model".into(), toml::Value::Integer(q));
                notes.push(format!("[limits] max_queue_per_model = {q}"));
            }
        }
        Ok(notes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(s: &str) -> toml::Table {
        toml::from_str(s).unwrap()
    }

    #[test]
    fn a_profile_fills_only_what_the_config_leaves_out() {
        let mut doc = table(
            r#"
[server]
profile = "voice_assistant"
[limits]
max_queue_per_model = 2
[[models]]
id = "e4b"
modality = "speech_to_text"
additional_modalities = ["text_generation"]
engine = "ggml_llama"
priority = "low"
[[models]]
id = "moss"
modality = "image_generation"
engine = "ggml_diffusion"
"#,
        );
        let notes = WorkloadProfile::apply_to_document(&mut doc).unwrap();
        let models = doc["models"].as_array().unwrap();
        let e4b = models[0].as_table().unwrap();
        assert_eq!(e4b["priority"].as_str(), Some("low"), "explicit keys win");
        assert_eq!(e4b["initial_tier"].as_str(), Some("tier1_gpu_vram"));
        assert_eq!(e4b["slots"].as_integer(), Some(1), "chat role: one slot");
        assert_eq!(e4b["arch"]["default_context_len"].as_integer(), Some(8192));
        let moss = models[1].as_table().unwrap();
        assert_eq!(moss["priority"].as_str(), Some("low"));
        assert!(
            !moss.contains_key("slots"),
            "slots only for llama.cpp models"
        );
        assert_eq!(doc["limits"]["max_queue_per_model"].as_integer(), Some(2));
        assert_eq!(notes.len(), 2, "{notes:?}");
    }

    #[test]
    fn roles_merge_to_the_most_demanding() {
        let mut m = table(
            "id = \"x\"\nmodality = \"speech_to_text\"\nadditional_modalities = [\"text_generation\"]\nengine = \"ggml_llama\"",
        );
        WorkloadProfile::VoiceAssistant.apply_to_model(&mut m);
        assert_eq!(m["priority"].as_str(), Some("realtime_pinned"));
        assert!(
            WorkloadProfile::apply_to_document(&mut table("[server]\nprofile = \"fast\""))
                .unwrap_err()
                .contains("not a profile")
        );
    }
}
