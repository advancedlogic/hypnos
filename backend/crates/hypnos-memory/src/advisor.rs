//! Hardware and model advisor (API-12): "I have 24 GB of VRAM and 64 GB of RAM and need chat,
//! voice and documents" -> a lineup from the model catalog, with variants, context, slots, tiers
//! and priorities, and a config that runs it.
//!
//! It reuses the server's own sizing instead of a second model of memory: budgets are derived as
//! `DeviceMemoryCapabilities::detect` derives them (Metal's default working set and the OS
//! reserve on unified memory, VRAM minus the safety margin on a discrete GPU), and every candidate
//! model is placed with [`KvCacheEstimator::compute_optimal_placement`], the planner the scheduler
//! runs. Catalog entries carry `estimate` values for what the server reads from the GGUF, so no
//! file is needed.
//!
//! The search is exhaustive over the catalog's entries and variants that serve the needs (a few
//! thousand combinations). A lineup scores for the size of the model serving each need (bigger
//! is better within the tested set), for keeping the interactive models (chat, vision, speech,
//! embeddings) resident together, and against models that only partly fit the GPU and against
//! extra models. Load-time figures are rough estimates from bandwidth, labelled as such.

use std::collections::BTreeSet;

use hypnos_core::catalog::{CatalogEntry, CatalogVariant, ModelCatalog};
use hypnos_core::config::{MemoryTier, Modality, ModelConfig, ModelPriority, MoeOffload};
use hypnos_core::kv_cache::{KvCacheEstimator, LayerPlacementPlan};
use serde::{Deserialize, Serialize};

use crate::hardware::{metal_default_working_set, uma_pinned_ram_budget, OS_RESERVE_BYTES};

const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;
/// Context the chat models fall back to when the interactive set does not fit together.
const TIGHT_CONTEXT: u32 = 8192;
/// Rough load bandwidths for the estimates (GB/s): NVMe into memory, and a warm (Tier 2 / page
/// cache) model into the GPU (measured: the 26B, 13.6 GiB, in 4.6 s on Apple Silicon).
const COLD_GBPS: f64 = 3.0;
const WARM_GBPS: f64 = 3.0;

/// What the user wants to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Need {
    Chat,
    Vision,
    /// A voice assistant: speech in, chat, speech out.
    Voice,
    Transcription,
    Speech,
    Embeddings,
    Images,
    Documents,
    Entities,
}

impl Need {
    pub const ALL: [Need; 9] = [
        Need::Chat,
        Need::Vision,
        Need::Voice,
        Need::Transcription,
        Need::Speech,
        Need::Embeddings,
        Need::Images,
        Need::Documents,
        Need::Entities,
    ];

    pub fn modalities(self) -> &'static [Modality] {
        match self {
            Need::Chat => &[Modality::TextGeneration],
            Need::Vision => &[Modality::Vision],
            Need::Voice => &[
                Modality::SpeechToText,
                Modality::TextGeneration,
                Modality::TextToSpeech,
            ],
            Need::Transcription => &[Modality::SpeechToText],
            Need::Speech => &[Modality::TextToSpeech],
            Need::Embeddings => &[Modality::Embeddings],
            Need::Images => &[Modality::ImageGeneration],
            Need::Documents => &[Modality::DocumentConversion],
            Need::Entities => &[Modality::EntityExtraction],
        }
    }

    pub fn parse(s: &str) -> Option<Need> {
        let s = s.trim().to_ascii_lowercase();
        Need::ALL.into_iter().find(|n| {
            serde_json::to_value(n)
                .ok()
                .and_then(|v| v.as_str().map(|v| v == s))
                .unwrap_or(false)
        })
    }
}

/// A modality served interactively: someone waits for it, so its model should stay resident.
fn interactive(m: Modality) -> bool {
    matches!(
        m,
        Modality::TextGeneration
            | Modality::Vision
            | Modality::SpeechToText
            | Modality::TextToSpeech
            | Modality::Embeddings
    )
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvisorRequest {
    /// GPU memory; ignored on unified memory, where the GPU shares `ram_bytes`.
    #[serde(default)]
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    /// Apple Silicon (or an APU): one pool for CPU and GPU.
    #[serde(default)]
    pub unified_memory: bool,
    pub needs: Vec<Need>,
}

/// The tiers' budgets for the described machine.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Budgets {
    pub tier1_bytes: u64,
    pub tier2_bytes: u64,
    /// What a CPU model (docling, GLiNER) may use: RAM after the OS reserve. The server places
    /// it in free RAM, pushing idle RAM-only models out, not inside the warm-memory budget.
    pub host_bytes: u64,
    pub unified_memory: bool,
}

/// Budgets as the server derives them on such a machine with the example configs' margins.
pub fn budgets(req: &AdvisorRequest) -> Budgets {
    if req.unified_memory {
        let working_set = metal_default_working_set(req.ram_bytes);
        Budgets {
            tier1_bytes: working_set.saturating_sub(2 * GIB),
            tier2_bytes: uma_pinned_ram_budget(u64::MAX, req.ram_bytes, working_set),
            host_bytes: req.ram_bytes.saturating_sub(OS_RESERVE_BYTES),
            unified_memory: true,
        }
    } else {
        Budgets {
            tier1_bytes: req.vram_bytes.saturating_sub(512 * MIB),
            // The server warns above half of physical RAM: pinned memory is gone for the OS.
            tier2_bytes: (req.ram_bytes / 2).min(req.ram_bytes.saturating_sub(OS_RESERVE_BYTES)),
            host_bytes: req.ram_bytes.saturating_sub(OS_RESERVE_BYTES),
            unified_memory: false,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AdvisedModel {
    pub id: String,
    pub name: String,
    pub quant: String,
    pub serves: Vec<Need>,
    pub weights_bytes: u64,
    /// What the model books while resident: VRAM (weights on the GPU + KV + workspace), or host
    /// RAM for a CPU model.
    pub resident_bytes: u64,
    pub runs_on_cpu: bool,
    pub gpu_layers: u32,
    pub num_layers: u32,
    pub cpu_expert_layers: u32,
    pub context_len: u32,
    pub slots: u32,
    pub initial_tier: MemoryTier,
    pub priority: ModelPriority,
    pub keep_warm_in_ram: bool,
    pub moe_offload: MoeOffload,
    /// Rough: from disk, and from Tier 2 (warm).
    pub load_secs_cold: f64,
    pub load_secs_warm: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Advice {
    pub budgets: Budgets,
    pub models: Vec<AdvisedModel>,
    /// The interactive models fit the GPU (Tier 1) together: no swaps between them.
    pub interactive_coresident: bool,
    pub interactive_bytes: u64,
    pub warnings: Vec<String>,
    /// The catalog specs to download (`./run.sh models <spec>`) or add (`POST /v1/models`).
    pub specs: Vec<String>,
    /// A complete config file for the lineup.
    pub config: String,
}

/// One candidate model with its sizing.
#[derive(Clone)]
struct Pick<'a> {
    entry: &'a CatalogEntry,
    variant: &'a CatalogVariant,
    cfg: ModelConfig,
    plan: LayerPlacementPlan,
}

impl Pick<'_> {
    fn modalities(&self) -> Vec<Modality> {
        let mut m = vec![self.entry.modality];
        m.extend(self.entry.additional_modalities.iter().copied());
        m
    }
    fn resident_bytes(&self) -> u64 {
        if self.cfg.runs_on_cpu() {
            self.plan.pinned_ram_bytes
        } else {
            self.plan.vram_weight_bytes + self.plan.vram_kv_and_workspace_bytes
        }
    }
    fn is_interactive(&self) -> bool {
        self.modalities().into_iter().any(interactive)
    }
    fn partial(&self) -> bool {
        !self.cfg.runs_on_cpu()
            && (self.plan.gpu_layers < self.cfg.arch.num_layers.max(1)
                || self.plan.cpu_expert_layers > 0)
    }
}

/// Sizes `variant` of `entry` on these budgets; `tight` = one slot and a shorter context for a
/// chat model. `None` when it cannot run at all.
fn size<'a>(
    entry: &'a CatalogEntry,
    variant: &'a CatalogVariant,
    b: &Budgets,
    tight: bool,
) -> Option<Pick<'a>> {
    let mut cfg = entry.model_config(variant).ok()?;
    let chat = entry.modality == Modality::TextGeneration
        || entry
            .additional_modalities
            .contains(&Modality::TextGeneration);
    if tight && chat {
        cfg.slots = 1;
        cfg.arch.default_context_len = cfg.arch.default_context_len.min(TIGHT_CONTEXT);
    }
    let place = |cfg: &ModelConfig| {
        KvCacheEstimator::compute_optimal_placement(
            cfg,
            None,
            1,
            b.tier1_bytes,
            if cfg.runs_on_cpu() {
                b.host_bytes
            } else {
                b.tier2_bytes
            },
            b.unified_memory,
        )
    };
    let mut plan = place(&cfg)?;
    // A Mixture-of-Experts model that does not fit whole: experts to the CPU before a layer
    // split (MEM-5; discrete GPUs only).
    if !b.unified_memory && cfg.arch.expert_bytes > 0 && plan.gpu_layers < cfg.arch.num_layers {
        let mut moe = cfg.clone();
        moe.moe_offload = MoeOffload::Auto;
        if let Some(p) = place(&moe) {
            cfg = moe;
            plan = p;
        }
    }
    Some(Pick {
        entry,
        variant,
        cfg,
        plan,
    })
}

struct Scored<'a> {
    picks: Vec<Pick<'a>>,
    score: f64,
    coresident: bool,
    interactive_bytes: u64,
}

fn score<'a>(picks: Vec<Pick<'a>>, needed: &BTreeSet<ModalityKey>, b: &Budgets) -> Scored<'a> {
    let interactive_bytes: u64 = picks
        .iter()
        .filter(|p| p.is_interactive() && !p.cfg.runs_on_cpu())
        .map(Pick::resident_bytes)
        .sum();
    let coresident = interactive_bytes <= b.tier1_bytes;
    let gb = |bytes: u64| bytes as f64 / GIB as f64;
    // Each needed modality counts the largest model serving it.
    let quality: f64 = needed
        .iter()
        .map(|m| {
            picks
                .iter()
                .filter(|p| p.modalities().iter().any(|x| ModalityKey(*x) == *m))
                .map(|p| (gb(p.cfg.total_weight_bytes) + 1.0).ln() * 10.0)
                .fold(0.0, f64::max)
        })
        .sum();
    let any_interactive = picks.iter().any(Pick::is_interactive);
    let score = quality
        + if coresident && any_interactive {
            30.0
        } else {
            0.0
        }
        - 15.0 * picks.iter().filter(|p| p.partial()).count() as f64
        - 2.0 * picks.len() as f64;
    Scored {
        picks,
        score,
        coresident,
        interactive_bytes,
    }
}

/// `Modality` with an order, for sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModalityKey(Modality);
impl PartialOrd for ModalityKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ModalityKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.0 as u8).cmp(&(other.0 as u8))
    }
}

/// The best lineup from `catalog` for `req`, or why there is none.
pub fn advise(catalog: &ModelCatalog, req: &AdvisorRequest) -> Result<Advice, String> {
    if req.needs.is_empty() {
        return Err("name at least one need".into());
    }
    if req.ram_bytes == 0 || (!req.unified_memory && req.vram_bytes == 0) {
        return Err("give the machine's RAM, and its VRAM unless it has unified memory".into());
    }
    let b = budgets(req);
    let needed: BTreeSet<ModalityKey> = req
        .needs
        .iter()
        .flat_map(|n| n.modalities().iter().map(|m| ModalityKey(*m)))
        .collect();
    let serves = |e: &CatalogEntry, m: ModalityKey| {
        e.modality == m.0 || e.additional_modalities.contains(&m.0)
    };
    let candidates: Vec<&CatalogEntry> = catalog
        .entries
        .iter()
        .filter(|e| needed.iter().any(|m| serves(e, *m)))
        .collect();
    let unservable: Vec<String> = needed
        .iter()
        .filter(|m| !candidates.iter().any(|e| serves(e, **m)))
        .map(|m| format!("{:?}", m.0))
        .collect();
    if !unservable.is_empty() {
        return Err(format!(
            "the catalog has no model for {}",
            unservable.join(", ")
        ));
    }

    let mut best: Option<Scored> = None;
    for tight in [false, true] {
        // Every choice of (entry -> one of its variants, or none).
        let mut choice = vec![0usize; candidates.len()];
        loop {
            let chosen: Vec<(&CatalogEntry, &CatalogVariant)> = candidates
                .iter()
                .zip(&choice)
                .filter(|(_, c)| **c > 0)
                .map(|(e, c)| (*e, &e.variants[*c - 1]))
                .collect();
            let covers = needed
                .iter()
                .all(|m| chosen.iter().any(|(e, _)| serves(e, *m)));
            // Every model must be the lineup's largest for at least one need (a big chat model
            // next to a small multimodal one is fine; a second, smaller model for the same needs
            // is not).
            let weight = |v: &CatalogVariant| v.files.iter().map(|f| f.size).sum::<u64>();
            let minimal = covers
                && chosen.iter().enumerate().all(|(i, (e, v))| {
                    needed.iter().any(|m| {
                        serves(e, *m)
                            && chosen.iter().enumerate().all(|(j, (o, ov))| {
                                j == i || !serves(o, *m) || weight(ov) < weight(v)
                            })
                    })
                });
            if minimal {
                let picks: Option<Vec<Pick>> =
                    chosen.iter().map(|(e, v)| size(e, v, &b, tight)).collect();
                if let Some(picks) = picks {
                    let s = score(picks, &needed, &b);
                    if best.as_ref().is_none_or(|x| s.score > x.score + 1e-9) {
                        best = Some(s);
                    }
                }
            }
            // Next combination (mixed radix).
            let mut i = 0;
            loop {
                if i == choice.len() {
                    break;
                }
                choice[i] += 1;
                if choice[i] <= candidates[i].variants.len() {
                    break;
                }
                choice[i] = 0;
                i += 1;
            }
            if i == choice.len() {
                break;
            }
        }
    }
    let best = best.ok_or_else(|| {
        "no lineup fits: a model needed for these needs does not fit this machine's memory even alone"
            .to_string()
    })?;
    Ok(finish(best, req, &b))
}

/// Tiers, priorities and warm memory for the chosen lineup, its warnings and its config.
fn finish(best: Scored, req: &AdvisorRequest, b: &Budgets) -> Advice {
    let mut warnings = Vec::new();
    let mut picks = best.picks;
    // Largest first: the order the tiers are handed out in.
    picks.sort_by_key(|p| std::cmp::Reverse(p.cfg.total_weight_bytes));
    let mut warm_left = b.tier2_bytes;
    let mut models = Vec::new();
    // The model that does each modality: the lineup's largest serving it.
    let doer = |m: Modality| {
        picks
            .iter()
            .filter(|p| p.modalities().contains(&m))
            .max_by_key(|p| p.cfg.total_weight_bytes)
            .map(|p| p.entry.id.as_str())
    };
    // Something that can push a resident interactive model out: an image model, or swaps.
    let evictions_possible = !best.coresident
        || picks
            .iter()
            .any(|p| p.modalities().contains(&Modality::ImageGeneration));
    for p in &picks {
        let mods = p.modalities();
        let serves: Vec<Need> = req
            .needs
            .iter()
            .copied()
            .filter(|n| {
                n.modalities()
                    .iter()
                    .any(|m| mods.contains(m) && doer(*m) == Some(p.entry.id.as_str()))
            })
            .collect();
        let small = p.cfg.total_weight_bytes < GIB;
        let (initial_tier, priority, mut keep_warm) = if mods.contains(&Modality::ImageGeneration) {
            (MemoryTier::Tier3NvmeSsd, ModelPriority::Low, false)
        } else if p.cfg.runs_on_cpu() {
            (MemoryTier::Tier3NvmeSsd, ModelPriority::Normal, false)
        } else if p.is_interactive() && small {
            (
                MemoryTier::Tier1GpuVram,
                ModelPriority::RealtimePinned,
                true,
            )
        } else if p.is_interactive() && best.coresident {
            (MemoryTier::Tier1GpuVram, ModelPriority::High, true)
        } else if p.is_interactive() {
            (MemoryTier::Tier2PinnedRam, ModelPriority::High, true)
        } else {
            (MemoryTier::Tier3NvmeSsd, ModelPriority::Normal, false)
        };
        let mut initial_tier = initial_tier;
        if keep_warm && !small {
            if p.cfg.total_weight_bytes <= warm_left {
                warm_left -= p.cfg.total_weight_bytes;
            } else {
                keep_warm = false;
                if initial_tier == MemoryTier::Tier2PinnedRam {
                    initial_tier = MemoryTier::Tier3NvmeSsd;
                }
            }
            if !keep_warm && evictions_possible {
                warnings.push(format!(
                    "{} does not fit the warm-memory budget next to the others: it reloads from disk when evicted",
                    p.entry.name
                ));
            }
        }
        if p.partial() {
            warnings.push(format!(
                "{} only partly fits the GPU ({} of {} layers{}): it runs, slower",
                p.entry.name,
                p.plan.gpu_layers,
                p.cfg.arch.num_layers,
                if p.plan.cpu_expert_layers > 0 {
                    format!(
                        ", experts of {} layers on the CPU",
                        p.plan.cpu_expert_layers
                    )
                } else {
                    String::new()
                }
            ));
        }
        let gb = p.cfg.total_weight_bytes as f64 / 1e9;
        models.push(AdvisedModel {
            id: p.entry.id.clone(),
            name: p.entry.name.clone(),
            quant: p.variant.quant.clone(),
            serves,
            weights_bytes: p.cfg.total_weight_bytes,
            resident_bytes: p.resident_bytes(),
            runs_on_cpu: p.cfg.runs_on_cpu(),
            gpu_layers: p.plan.gpu_layers,
            num_layers: p.cfg.arch.num_layers,
            cpu_expert_layers: p.plan.cpu_expert_layers,
            context_len: p.plan.allocated_context_len,
            slots: p.cfg.slots,
            initial_tier,
            priority,
            keep_warm_in_ram: keep_warm,
            moe_offload: p.cfg.moe_offload,
            load_secs_cold: (gb / COLD_GBPS * 10.0).round() / 10.0,
            load_secs_warm: (gb / WARM_GBPS * 10.0).round() / 10.0,
        });
    }
    if !best.coresident {
        warnings.push(format!(
            "the interactive models need {:.1} GiB of GPU memory together and {:.1} GiB is available: they swap, each switch costing a load (see load_secs_warm)",
            best.interactive_bytes as f64 / GIB as f64,
            b.tier1_bytes as f64 / GIB as f64
        ));
    }
    let interactive_gpu = best.interactive_bytes;
    for m in models.iter().filter(|m| m.serves.contains(&Need::Images)) {
        if m.resident_bytes + interactive_gpu > b.tier1_bytes {
            warnings.push(format!(
                "generating an image ({}) evicts the chat models; the next chat reloads them",
                m.name
            ));
        }
    }
    let specs = models
        .iter()
        .map(|m| format!("{}:{}", m.id, m.quant))
        .collect();
    let config = render_config(&picks, &models, req, b);
    Advice {
        budgets: *b,
        models,
        interactive_coresident: best.coresident,
        interactive_bytes: best.interactive_bytes,
        warnings,
        specs,
        config,
    }
}

fn render_config(
    picks: &[Pick],
    models: &[AdvisedModel],
    req: &AdvisorRequest,
    b: &Budgets,
) -> String {
    let needs: Vec<String> = req
        .needs
        .iter()
        .map(|n| {
            serde_json::to_value(n)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default()
        })
        .collect();
    let mut out = format!(
        "# Generated by the Hypnos advisor (API-12) for {} with {} GiB of RAM{}; needs: {}.\n\
         # Download the files first: ./run.sh models <id>:<quant> for each model below.\n\
         config_version = 1\n\n[server]\nmodels_dir = \"models\"\n\n[hardware]\n",
        if req.unified_memory {
            "unified memory"
        } else {
            "a discrete GPU"
        },
        req.ram_bytes / GIB,
        if req.unified_memory {
            String::new()
        } else {
            format!(" and {} GiB of VRAM", req.vram_bytes / GIB)
        },
        needs.join(", ")
    );
    if req.unified_memory {
        out += "accelerator = \"apple_metal_uma\"\nvram_budget_bytes = 0\nvram_safety_margin_bytes = 2147483648\nlock_host_memory = false\n";
    } else {
        out += &format!(
            "accelerator = \"auto\"\nvram_budget_bytes = {}\nvram_safety_margin_bytes = 536870912\n",
            req.vram_bytes
        );
    }
    out += &format!("pinned_ram_budget_bytes = {}\n", b.tier2_bytes);
    for (p, m) in picks.iter().zip(models) {
        let mut entry = p.entry.clone();
        let set = |t: &mut toml::Table, k: &str, v: toml::Value| {
            t.insert(k.to_string(), v);
        };
        let tier = serde_json::to_value(m.initial_tier).unwrap_or_default();
        let prio = serde_json::to_value(m.priority).unwrap_or_default();
        set(
            &mut entry.config,
            "initial_tier",
            toml::Value::String(tier.as_str().unwrap_or_default().into()),
        );
        set(
            &mut entry.config,
            "priority",
            toml::Value::String(prio.as_str().unwrap_or_default().into()),
        );
        if !m.runs_on_cpu {
            set(
                &mut entry.config,
                "keep_warm_in_ram",
                toml::Value::Boolean(m.keep_warm_in_ram),
            );
        }
        if p.cfg.slots != 1 || entry.config.contains_key("slots") {
            set(
                &mut entry.config,
                "slots",
                toml::Value::Integer(i64::from(p.cfg.slots)),
            );
        }
        if m.moe_offload == MoeOffload::Auto {
            set(
                &mut entry.config,
                "moe_offload",
                toml::Value::String("auto".into()),
            );
        }
        if let Some(toml::Value::Table(arch)) = entry.config.get_mut("arch") {
            if arch.contains_key("default_context_len") {
                arch.insert(
                    "default_context_len".into(),
                    toml::Value::Integer(i64::from(p.cfg.arch.default_context_len)),
                );
            }
        }
        out += "\n";
        out += &entry.render_block(p.variant);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use hypnos_core::config::HypnosConfig;
    use std::path::Path;

    fn catalog() -> ModelCatalog {
        ModelCatalog::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models.catalog.toml"))
            .unwrap()
    }

    fn ask(vram_gb: u64, ram_gb: u64, unified: bool, needs: &[Need]) -> Result<Advice, String> {
        advise(
            &catalog(),
            &AdvisorRequest {
                vram_bytes: vram_gb * GIB,
                ram_bytes: ram_gb * GIB,
                unified_memory: unified,
                needs: needs.to_vec(),
            },
        )
    }

    fn covers(a: &Advice, needs: &[Need]) {
        for n in needs {
            assert!(
                a.models.iter().any(|m| m.serves.contains(n)),
                "{n:?} not served: {:?}",
                a.models.iter().map(|m| &m.id).collect::<Vec<_>>()
            );
        }
        // The generated config is one the server reads without unknown keys.
        let c = HypnosConfig::from_toml_str(&a.config).unwrap();
        assert!(c.unknown_keys.is_empty(), "{:?}", c.unknown_keys);
        assert_eq!(c.models.len(), a.models.len());
    }

    #[test]
    fn a_24gb_gpu_keeps_chat_and_voice_resident_together() {
        let needs = [Need::Chat, Need::Voice, Need::Documents];
        let a = ask(24, 64, false, &needs).unwrap();
        covers(&a, &needs);
        assert!(a.interactive_coresident, "{:#?}", a.warnings);
        assert!(a.interactive_bytes <= a.budgets.tier1_bytes);
        // The big chat model fits next to a small speech model with one slot and 8k context.
        let chat = a
            .models
            .iter()
            .find(|m| m.serves.contains(&Need::Chat))
            .unwrap();
        assert_eq!(chat.id, "gemma4:26b-q4");
        assert_eq!(chat.slots, 1);
        assert_eq!(
            a.models
                .iter()
                .filter(|m| m.serves.contains(&Need::Chat))
                .count(),
            1,
            "one model chats"
        );
        let docling = a
            .models
            .iter()
            .find(|m| m.serves.contains(&Need::Documents))
            .unwrap();
        assert!(docling.runs_on_cpu && docling.initial_tier == MemoryTier::Tier3NvmeSsd);
    }

    #[test]
    fn a_16gb_mac_gets_small_models_that_fit_together() {
        let needs = [Need::Chat, Need::Vision, Need::Transcription];
        let a = ask(0, 16, true, &needs).unwrap();
        covers(&a, &needs);
        assert!(a.interactive_coresident, "{:#?}", a.warnings);
        assert!(
            a.models.iter().all(|m| m.weights_bytes < 8 * GIB),
            "{:?}",
            a.models.iter().map(|m| &m.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cpu_models_use_free_ram_not_the_warm_budget() {
        // A 16 GB Mac has 1.3 GiB of warm memory; docling (1.8 GiB) runs in free RAM.
        let needs = [Need::Chat, Need::Vision, Need::Voice, Need::Documents];
        let a = ask(0, 16, true, &needs).unwrap();
        covers(&a, &needs);
    }

    #[test]
    fn a_large_machine_gets_the_largest_chat_model() {
        let a = ask(0, 128, true, &[Need::Chat, Need::Images]).unwrap();
        covers(&a, &[Need::Chat, Need::Images]);
        assert!(a.models.iter().any(|m| m.id == "gemma4:26b-q4"));
        let image = a
            .models
            .iter()
            .find(|m| m.serves.contains(&Need::Images))
            .unwrap();
        assert_eq!(image.priority, ModelPriority::Low);
    }

    #[test]
    fn impossible_requests_say_why() {
        assert!(ask(2, 8, false, &[Need::Images])
            .unwrap_err()
            .contains("does not fit"));
        assert!(ask(24, 64, false, &[]).is_err());
        assert_eq!(Need::parse("Voice"), Some(Need::Voice));
        assert_eq!(Need::parse("nope"), None);
    }
}
