//! Kokoro-82M text-to-speech through ONNX Runtime, with espeak-ng (statically linked) for
//! grapheme-to-phoneme conversion. Both run inside the server process.
//!
//! Model inputs (onnx-community/Kokoro-82M-v1.0-ONNX): `input_ids` int64 `[1, T]` (0-padded on both
//! ends), `style` f32 `[1, 256]` (row `T-1` of the speaker's `[510, 256]` style table), `speed` f32 `[1]`.
//! Output: `waveform` f32 `[1, N]` at 24 kHz.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Instant;

use async_trait::async_trait;
use hypnos_core::config::{MemoryTier, Modality, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{AudioSpeechOutput, AudioSpeechRequest, ExecutionTelemetry};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use ort::environment::Environment;
use ort::session::builder::AutoDevicePolicy;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::RwLock;
use tracing::{debug, info, warn};

use crate::media::{encode_pcm16_raw, encode_wav_pcm16};
use crate::{default_threads, BackendContextState};

pub const KOKORO_SAMPLE_RATE: u32 = 24_000;
const STYLE_DIM: usize = 256;
const MAX_PHONEMES: usize = 510;
const INTER_CHUNK_SILENCE_SECS: f32 = 0.18;

/// Kokoro v1.0 phoneme vocabulary (hexgrad/Kokoro-82M `config.json`).
const VOCAB: &[(char, i64)] = &[
    (';', 1),
    (':', 2),
    (',', 3),
    ('.', 4),
    ('!', 5),
    ('?', 6),
    ('—', 9),
    ('…', 10),
    ('"', 11),
    ('(', 12),
    (')', 13),
    ('“', 14),
    ('”', 15),
    (' ', 16),
    ('\u{0303}', 17),
    ('ʣ', 18),
    ('ʥ', 19),
    ('ʦ', 20),
    ('ʨ', 21),
    ('ᵝ', 22),
    ('\u{AB67}', 23),
    ('A', 24),
    ('I', 25),
    ('O', 31),
    ('Q', 33),
    ('S', 35),
    ('T', 36),
    ('W', 39),
    ('Y', 41),
    ('ᵊ', 42),
    ('a', 43),
    ('b', 44),
    ('c', 45),
    ('d', 46),
    ('e', 47),
    ('f', 48),
    ('h', 50),
    ('i', 51),
    ('j', 52),
    ('k', 53),
    ('l', 54),
    ('m', 55),
    ('n', 56),
    ('o', 57),
    ('p', 58),
    ('q', 59),
    ('r', 60),
    ('s', 61),
    ('t', 62),
    ('u', 63),
    ('v', 64),
    ('w', 65),
    ('x', 66),
    ('y', 67),
    ('z', 68),
    ('ɑ', 69),
    ('ɐ', 70),
    ('ɒ', 71),
    ('æ', 72),
    ('β', 75),
    ('ɔ', 76),
    ('ɕ', 77),
    ('ç', 78),
    ('ɖ', 80),
    ('ð', 81),
    ('ʤ', 82),
    ('ə', 83),
    ('ɚ', 85),
    ('ɛ', 86),
    ('ɜ', 87),
    ('ɟ', 90),
    ('ɡ', 92),
    ('ɥ', 99),
    ('ɨ', 101),
    ('ɪ', 102),
    ('ʝ', 103),
    ('ɯ', 110),
    ('ɰ', 111),
    ('ŋ', 112),
    ('ɳ', 113),
    ('ɲ', 114),
    ('ɴ', 115),
    ('ø', 116),
    ('ɸ', 118),
    ('θ', 119),
    ('œ', 120),
    ('ɹ', 123),
    ('ɾ', 125),
    ('ɻ', 126),
    ('ʁ', 128),
    ('ɽ', 129),
    ('ʂ', 130),
    ('ʃ', 131),
    ('ʈ', 132),
    ('ʧ', 133),
    ('ʊ', 135),
    ('ʋ', 136),
    ('ʌ', 138),
    ('ɣ', 139),
    ('ɤ', 140),
    ('χ', 142),
    ('ʎ', 143),
    ('ʒ', 147),
    ('ʔ', 148),
    ('ˈ', 156),
    ('ˌ', 157),
    ('ː', 158),
    ('ʰ', 162),
    ('ʲ', 164),
    ('↓', 169),
    ('→', 171),
    ('↗', 172),
    ('↘', 173),
    ('ᵻ', 177),
];

/// OpenAI voice names mapped onto Kokoro speaker ids; raw Kokoro ids pass through untouched.
fn resolve_voice<'a>(requested: &str, voices: &'a HashMap<String, Vec<f32>>) -> Option<&'a str> {
    let key = requested.trim().to_ascii_lowercase();
    if let Some((k, _)) = voices.get_key_value(key.as_str()) {
        return Some(k.as_str());
    }
    let mapped = match key.as_str() {
        "alloy" => "af_alloy",
        "echo" => "am_echo",
        "fable" => "bm_fable",
        "onyx" => "am_onyx",
        "nova" => "af_nova",
        "shimmer" | "coral" => "af_bella",
        "ash" => "am_adam",
        "sage" => "af_sarah",
        "ballad" => "bm_george",
        "verse" => "am_michael",
        _ => "af_heart",
    };
    voices
        .get_key_value(mapped)
        .or_else(|| voices.get_key_value("af_heart"))
        .or_else(|| voices.iter().next())
        .map(|(k, _)| k.as_str())
}

struct LoadedKokoro {
    /// Kept alive for the session's lifetime.
    _env: Environment,
    session: Session,
    voices: HashMap<String, Vec<f32>>,
    tokens_input: String,
}

fn load_voices(dir: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let mut voices = HashMap::new();
    let entries = std::fs::read_dir(dir).map_err(|e| {
        HypnosError::Config(format!(
            "cannot read voices directory {}: {e}",
            dir.display()
        ))
    })?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("bin") {
            continue;
        }
        let bytes = std::fs::read(&path).map_err(|e| HypnosError::Io(e))?;
        if bytes.len() % (4 * STYLE_DIM) != 0 || bytes.len() < 4 * STYLE_DIM {
            warn!(path = %path.display(), len = bytes.len(), "skipping voice file with unexpected size");
            continue;
        }
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        voices.insert(name, floats);
    }
    if voices.is_empty() {
        return Err(HypnosError::Config(format!(
            "no Kokoro voice style files (*.bin) found in {}",
            dir.display()
        )));
    }
    Ok(voices)
}

fn load_kokoro(cfg: &ModelConfig, n_threads: i32) -> Result<LoadedKokoro> {
    if !cfg.weights_path.exists() {
        return Err(HypnosError::Config(format!(
            "ONNX model for '{}' not found: {}",
            cfg.id,
            cfg.weights_path.display()
        )));
    }
    let voices_dir: PathBuf = cfg.voices_path.clone().unwrap_or_else(|| {
        cfg.weights_path
            .parent()
            .map(|p| p.join("voices"))
            .unwrap_or_else(|| PathBuf::from("voices"))
    });
    // Let espeak-rs find the phoneme data copied next to the binaries by our build script.
    if std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY").is_none() {
        if let Some(parent) = option_env!("HYPNOS_ESPEAK_DATA_PARENT") {
            std::env::set_var("PIPER_ESPEAKNG_DATA_DIRECTORY", parent);
        }
    }

    let started = Instant::now();
    let env = ort::init()
        .with_name("hypnos-kokoro")
        .build()
        .map_err(|e| HypnosError::BackendError(format!("ONNX Runtime init failed: {e}")))?;
    let session = Session::builder(&env)
        .map_err(|e| HypnosError::BackendError(format!("ONNX session builder: {e}")))?
        .with_intra_threads(n_threads.clamp(1, 8) as usize)
        .map_err(|e| HypnosError::BackendError(format!("ONNX session options: {e}")))?
        // CPU only: ort's default policy registers CoreML on macOS, which splits the graph into 24
        // partitions. The CPU provider is the fastest measured on Apple Silicon with the fp32 graph.
        .with_auto_device(AutoDevicePolicy::PreferCPU)
        .map_err(|e| HypnosError::BackendError(format!("ONNX session options: {e}")))?
        .commit_from_file(&cfg.weights_path)
        .map_err(|e| {
            HypnosError::BackendError(format!(
                "ONNX Runtime could not load {}: {e}",
                cfg.weights_path.display()
            ))
        })?;
    let input_names: Vec<String> = session
        .inputs()
        .iter()
        .map(|i| i.name().to_string())
        .collect();
    let tokens_input = if input_names.iter().any(|n| n == "input_ids") {
        "input_ids".to_string()
    } else {
        "tokens".to_string()
    };
    let voices = load_voices(&voices_dir)?;
    info!(
        model = %cfg.id,
        inputs = ?input_names,
        voices = voices.len(),
        load_secs = format!("{:.2}", started.elapsed().as_secs_f32()),
        "Kokoro ONNX session resident"
    );
    Ok(LoadedKokoro {
        _env: env,
        session,
        voices,
        tokens_input,
    })
}

/// Text -> IPA phoneme chunks of at most `MAX_PHONEMES` vocabulary symbols each.
fn phonemize_chunks(text: &str, vocab: &HashMap<char, i64>) -> Result<Vec<Vec<i64>>> {
    let sentences = espeak_rs::text_to_phonemes(text, "en-us", None)
        .map_err(|e| HypnosError::BackendError(format!("espeak-ng phonemization failed: {e}")))?;
    let mut chunks: Vec<Vec<i64>> = Vec::new();
    let mut current: Vec<i64> = Vec::new();
    for sentence in sentences {
        let ids: Vec<i64> = sentence
            .chars()
            .filter_map(|c| vocab.get(&c).copied())
            .collect();
        if ids.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + 1 + ids.len() > MAX_PHONEMES {
            chunks.push(std::mem::take(&mut current));
        }
        if ids.len() > MAX_PHONEMES {
            for piece in ids.chunks(MAX_PHONEMES) {
                chunks.push(piece.to_vec());
            }
            continue;
        }
        if !current.is_empty() {
            current.push(16); // space between sentences
        }
        current.extend(ids);
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    Ok(chunks)
}

fn synthesize(
    loaded: &mut LoadedKokoro,
    vocab: &HashMap<char, i64>,
    text: &str,
    voice: &str,
    speed: f32,
    ctx: &RequestContext,
) -> Result<Vec<f32>> {
    let style_table = loaded
        .voices
        .get(voice)
        .ok_or_else(|| HypnosError::InvalidRequest(format!("unknown voice '{voice}'")))?
        .clone();
    let rows = style_table.len() / STYLE_DIM;
    let chunks = phonemize_chunks(text, vocab)?;
    if chunks.is_empty() {
        return Err(HypnosError::InvalidRequest(
            "input produced no speakable phonemes".to_string(),
        ));
    }
    let silence = vec![0.0f32; (KOKORO_SAMPLE_RATE as f32 * INTER_CHUNK_SILENCE_SECS) as usize];
    let mut audio: Vec<f32> = Vec::new();
    for (i, tokens) in chunks.iter().enumerate() {
        if !ctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }
        let n = tokens.len();
        let mut ids: Vec<i64> = Vec::with_capacity(n + 2);
        ids.push(0);
        ids.extend_from_slice(tokens);
        ids.push(0);
        let row = n.min(rows).saturating_sub(1);
        let style = style_table[row * STYLE_DIM..(row + 1) * STYLE_DIM].to_vec();
        let ids_len = ids.len() as i64;

        let input_ids = Tensor::from_array((vec![1i64, ids_len], ids))
            .map_err(|e| HypnosError::BackendError(format!("input tensor: {e}")))?;
        let style_t = Tensor::from_array((vec![1i64, STYLE_DIM as i64], style))
            .map_err(|e| HypnosError::BackendError(format!("style tensor: {e}")))?;
        let speed_t = Tensor::from_array((vec![1i64], vec![speed]))
            .map_err(|e| HypnosError::BackendError(format!("speed tensor: {e}")))?;
        let outputs = loaded
            .session
            .run(ort::inputs![
                loaded.tokens_input.as_str() => input_ids,
                "style" => style_t,
                "speed" => speed_t,
            ])
            .map_err(|e| {
                HypnosError::BackendError(format!("ONNX Runtime inference failed: {e}"))
            })?;
        let (_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| HypnosError::BackendError(format!("waveform output: {e}")))?;
        // NaN would be written as zero samples: a valid WAV of pure silence with HTTP 200.
        let non_finite = data.iter().filter(|x| !x.is_finite()).count();
        if non_finite > 0 {
            return Err(HypnosError::BackendError(format!(
                "Kokoro returned {non_finite} non-finite samples of {} for voice '{voice}'; \
                 the fp16 graph overflows on some inputs, use the fp32 model.onnx",
                data.len()
            )));
        }
        if i > 0 {
            audio.extend_from_slice(&silence);
        }
        audio.extend_from_slice(data);
        debug!(
            chunk = i,
            phonemes = n,
            samples = data.len(),
            "kokoro chunk synthesized"
        );
    }
    Ok(audio)
}

pub struct OnnxRuntimeBackend {
    config: ModelConfig,
    state: RwLock<BackendContextState>,
    loaded: Arc<StdMutex<Option<LoadedKokoro>>>,
    vocab: Arc<HashMap<char, i64>>,
    n_threads: i32,
}

impl OnnxRuntimeBackend {
    pub fn new(config: ModelConfig) -> Self {
        let state = RwLock::new(BackendContextState::new(&config));
        Self {
            config,
            state,
            loaded: Arc::new(StdMutex::new(None)),
            vocab: Arc::new(VOCAB.iter().copied().collect()),
            n_threads: default_threads(),
        }
    }
}

#[async_trait]
impl ModelBackend for OnnxRuntimeBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.loaded.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        {
            let mut guard = self.state.write();
            guard.active_plan = plan.clone();
            guard.pinned_host_ptr = pinned_host_ptr;
        }
        let want_resident = plan.primary_tier == MemoryTier::Tier1GpuVram;
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let n_threads = self.n_threads;
        tokio::task::spawn_blocking(move || -> Result<()> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("kokoro mutex poisoned".into()))?;
            if !want_resident {
                if guard.take().is_some() {
                    info!(model = %cfg.id, "Released Kokoro ONNX session");
                }
                return Ok(());
            }
            if guard.is_none() {
                *guard = Some(load_kokoro(&cfg, n_threads)?);
            }
            Ok(())
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("load task panicked: {e}")))?
    }

    async fn synthesize_speech(
        &self,
        req: &AudioSpeechRequest,
        ctx: &RequestContext,
    ) -> Result<AudioSpeechOutput> {
        if !self.config.supports_modality(Modality::TextToSpeech) {
            return Err(HypnosError::ModalityMismatch {
                model: self.config.id.clone(),
                expected: self.config.modality,
                actual: Modality::TextToSpeech,
            });
        }
        if req.input.trim().is_empty() {
            return Err(HypnosError::InvalidRequest(
                "input text is empty".to_string(),
            ));
        }
        if req.input.chars().count() > ctx.limits.max_prompt_chars {
            return Err(HypnosError::InvalidRequest(format!(
                "input text exceeds limits.max_prompt_chars ({})",
                ctx.limits.max_prompt_chars
            )));
        }
        let ctx = ctx.clone();
        let speed = if req.speed.is_finite() {
            req.speed.clamp(0.5, 2.0)
        } else {
            1.0
        };
        let text = req.input.clone();
        let requested_voice = req.voice.clone();
        let format = req.response_format.to_ascii_lowercase();
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let vocab = self.vocab.clone();

        let (samples, voice_used) = tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, String)> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("kokoro mutex poisoned".into()))?;
            let resident = guard.as_mut().ok_or_else(|| {
                HypnosError::BackendError(format!("model '{}' is not loaded", cfg.id))
            })?;
            let voice = resolve_voice(&requested_voice, &resident.voices)
                .ok_or_else(|| HypnosError::Config("no Kokoro voices available".into()))?
                .to_string();
            let started = Instant::now();
            let samples = synthesize(resident, &vocab, &text, &voice, speed, &ctx)?;
            info!(model = %cfg.id, voice = %voice, chars = text.len(), audio_secs = format!("{:.2}", samples.len() as f32 / KOKORO_SAMPLE_RATE as f32), synth_secs = format!("{:.2}", started.elapsed().as_secs_f32()), "speech synthesized");
            Ok((samples, voice))
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("tts task panicked: {e}")))??;

        let (audio_bytes, content_type) = match format.as_str() {
            "pcm" => (encode_pcm16_raw(&samples), "audio/pcm".to_string()),
            "wav" | "" => (
                encode_wav_pcm16(&samples, KOKORO_SAMPLE_RATE)?,
                "audio/wav".to_string(),
            ),
            other => {
                debug!(
                    requested = other,
                    "unsupported response_format; returning wav"
                );
                (
                    encode_wav_pcm16(&samples, KOKORO_SAMPLE_RATE)?,
                    "audio/wav".to_string(),
                )
            }
        };
        let plan = self.state.read().active_plan.clone();
        let _ = voice_used;
        Ok(AudioSpeechOutput {
            audio_bytes,
            content_type,
            telemetry: ExecutionTelemetry {
                previous_tier: plan.primary_tier,
                placement: plan,
                tier_switch_latency_ms: 0.0,
                inference_latency_ms: 0.0,
                evicted_models: Vec::new(),
            },
        })
    }
}
