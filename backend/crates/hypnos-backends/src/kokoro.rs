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
use hypnos_core::openai::{
    AudioChunk, AudioSpeechOutput, AudioSpeechRequest, AudioStreamEvent, EngineTiming,
    ExecutionTelemetry,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use ort::environment::Environment;
use ort::session::builder::AutoDevicePolicy;
use ort::session::Session;
use ort::value::Tensor;
use parking_lot::RwLock;
use tokio::sync::mpsc;
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
    // Let espeak-rs find the phoneme data: next to the executable (release archives, Docker),
    // else where our build script copied it on the build machine.
    if std::env::var_os("PIPER_ESPEAKNG_DATA_DIRECTORY").is_none() {
        let beside_exe = std::env::current_exe()
            .ok()
            .and_then(|exe| exe.parent().map(PathBuf::from))
            .filter(|dir| dir.join("espeak-ng-data").is_dir());
        let parent =
            beside_exe.or_else(|| option_env!("HYPNOS_ESPEAK_DATA_PARENT").map(PathBuf::from));
        if let Some(parent) = parent {
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

/// Streaming: the first piece is cut after this many characters (at a clause boundary when one
/// follows soon, else at the next word boundary) so the first audio arrives after one very short
/// synthesis. Kokoro costs about 8 ms per phoneme on a CPU, so ~35 phonemes keep it under 300 ms.
const FIRST_PIECE_MIN_CHARS: usize = 28;
/// First sentences up to this length are left whole.
const FIRST_PIECE_MAX_CHARS: usize = 40;
/// A clause boundary is preferred over a word boundary when it lies within this many characters.
const FIRST_PIECE_CLAUSE_MAX_CHARS: usize = 60;
/// Phoneme budgets of successive chunks when streaming. Synthesis runs several times faster than
/// real time, so each chunk may be a few times longer than everything already playing without
/// the player running dry; after that the normal `MAX_PHONEMES` applies.
const STREAM_CHUNK_BUDGETS: &[usize] = &[48, 192];

/// A unit of text to synthesize and whether a pause belongs after it (sentence or clause end, as
/// opposed to a cut in the middle of a phrase).
#[derive(Debug, Clone, PartialEq, Eq)]
struct Piece {
    text: String,
    pause_after: bool,
}

/// Phoneme ids of one synthesis call.
struct PhonemeChunk {
    ids: Vec<i64>,
    pause_after: bool,
}

/// Splits text into sentences at `. ! ? …` followed by whitespace (or the end), keeping the
/// punctuation with its sentence. espeak-ng's phoneme output carries no sentence or punctuation
/// markers, so chunk boundaries have to be decided on the text.
fn split_sentences(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        current.push(c);
        if matches!(c, '.' | '!' | '?' | '…') && chars.peek().is_none_or(|n| n.is_whitespace()) {
            let s = current.trim();
            if !s.is_empty() {
                out.push(s.to_string());
            }
            current.clear();
        }
    }
    let s = current.trim();
    if !s.is_empty() {
        out.push(s.to_string());
    }
    out
}

/// Sentences as pieces; with `short_first` the first sentence is cut so the opening piece is a
/// few words long (see [`FIRST_PIECE_MIN_CHARS`]).
fn split_pieces(text: &str, short_first: bool) -> Vec<Piece> {
    let mut pieces: Vec<Piece> = split_sentences(text)
        .into_iter()
        .map(|text| Piece {
            text,
            pause_after: true,
        })
        .collect();
    if short_first {
        shorten_first_piece(&mut pieces);
    }
    pieces
}

fn shorten_first_piece(pieces: &mut Vec<Piece>) {
    let Some(first) = pieces.first() else {
        return;
    };
    let text = first.text.clone();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    if chars.len() <= FIRST_PIECE_MAX_CHARS {
        return;
    }
    let min_byte = chars[FIRST_PIECE_MIN_CHARS.min(chars.len() - 1)].0;
    let clause_max_byte = chars[FIRST_PIECE_CLAUSE_MAX_CHARS.min(chars.len() - 1)].0;
    let clause = chars
        .iter()
        .find(|(i, c)| *i >= min_byte && *i <= clause_max_byte && matches!(c, ',' | ';' | ':'))
        .map(|(i, c)| (i + c.len_utf8(), true));
    let word = chars
        .iter()
        .find(|(i, c)| *i >= min_byte && c.is_whitespace())
        .map(|(i, _)| (*i, false));
    let Some((cut, pause_after)) = clause.or(word) else {
        return;
    };
    let head = text[..cut].trim().to_string();
    let rest = text[cut..].trim().to_string();
    if head.is_empty() || rest.is_empty() {
        return;
    }
    pieces[0] = Piece {
        text: head,
        pause_after,
    };
    pieces.insert(
        1,
        Piece {
            text: rest,
            pause_after: true,
        },
    );
}

/// Text -> phoneme chunks of at most `MAX_PHONEMES` vocabulary symbols each, merging whole
/// sentences. With `short_first` the chunks follow [`STREAM_CHUNK_BUDGETS`] (a few words first,
/// then a sentence or two, then full chunks) so streaming playback starts almost at once.
fn phonemize_chunks(
    text: &str,
    vocab: &HashMap<char, i64>,
    short_first: bool,
) -> Result<Vec<PhonemeChunk>> {
    let pieces = split_pieces(text, short_first);
    let budget = |k: usize| {
        if short_first {
            STREAM_CHUNK_BUDGETS.get(k).copied().unwrap_or(MAX_PHONEMES)
        } else {
            MAX_PHONEMES
        }
    };
    let mut chunks: Vec<PhonemeChunk> = Vec::new();
    let mut current: Vec<i64> = Vec::new();
    let mut current_pause = true;
    for piece in &pieces {
        let phonemes = espeak_rs::text_to_phonemes(&piece.text, "en-us", None).map_err(|e| {
            HypnosError::BackendError(format!("espeak-ng phonemization failed: {e}"))
        })?;
        let ids: Vec<i64> = phonemes
            .iter()
            .flat_map(|s| s.chars())
            .filter_map(|c| vocab.get(&c).copied())
            .collect();
        if ids.is_empty() {
            continue;
        }
        if !current.is_empty() && current.len() + 1 + ids.len() > budget(chunks.len()) {
            chunks.push(PhonemeChunk {
                ids: std::mem::take(&mut current),
                pause_after: current_pause,
            });
        }
        if ids.len() > MAX_PHONEMES {
            let n = ids.len().div_ceil(MAX_PHONEMES);
            for (i, part) in ids.chunks(MAX_PHONEMES).enumerate() {
                chunks.push(PhonemeChunk {
                    ids: part.to_vec(),
                    pause_after: i + 1 == n && piece.pause_after,
                });
            }
            continue;
        }
        if !current.is_empty() {
            current.push(16); // space between sentences
        }
        current.extend(ids);
        current_pause = piece.pause_after;
    }
    if !current.is_empty() {
        chunks.push(PhonemeChunk {
            ids: current,
            pause_after: current_pause,
        });
    }
    Ok(chunks)
}

/// Synthesizes `text` chunk by chunk, handing each chunk's samples (and the silence that
/// precedes it) to `on_chunk` as soon as ONNX Runtime returns them. `on_chunk` returns `false`
/// to stop (the client went away). Returns the number of samples produced.
#[allow(clippy::too_many_arguments)]
fn synthesize(
    loaded: &mut LoadedKokoro,
    vocab: &HashMap<char, i64>,
    text: &str,
    voice: &str,
    speed: f32,
    short_first: bool,
    ctx: &RequestContext,
    on_chunk: &mut dyn FnMut(&[f32]) -> bool,
) -> Result<usize> {
    let style_table = loaded
        .voices
        .get(voice)
        .ok_or_else(|| HypnosError::InvalidRequest(format!("unknown voice '{voice}'")))?
        .clone();
    let rows = style_table.len() / STYLE_DIM;
    let chunks = phonemize_chunks(text, vocab, short_first)?;
    if chunks.is_empty() {
        return Err(HypnosError::InvalidRequest(
            "input produced no speakable phonemes".to_string(),
        ));
    }
    let silence = vec![0.0f32; (KOKORO_SAMPLE_RATE as f32 * INTER_CHUNK_SILENCE_SECS) as usize];
    let mut total = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        if !ctx.should_continue() {
            return Err(HypnosError::Cancelled);
        }
        let tokens = &chunk.ids;
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
        if i > 0 && chunks[i - 1].pause_after {
            if !on_chunk(&silence) {
                return Err(HypnosError::Cancelled);
            }
            total += silence.len();
        }
        if !on_chunk(data) {
            return Err(HypnosError::Cancelled);
        }
        total += data.len();
        debug!(
            chunk = i,
            phonemes = n,
            samples = data.len(),
            "kokoro chunk synthesized"
        );
    }
    Ok(total)
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
        let (text, speed) = self.check_request(req, ctx)?;
        let ctx = ctx.clone();
        let requested_voice = req.voice.clone();
        let format = req.response_format.to_ascii_lowercase();
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let vocab = self.vocab.clone();

        let requested_at = Instant::now();
        let (samples, voice_used, queue_ms, synth_ms) =
            tokio::task::spawn_blocking(move || -> Result<(Vec<f32>, String, f64, f64)> {
                let mut guard = loaded
                    .lock()
                    .map_err(|_| HypnosError::BackendError("kokoro mutex poisoned".into()))?;
                let queue_ms = requested_at.elapsed().as_secs_f64() * 1000.0;
                let resident = guard.as_mut().ok_or_else(|| {
                    HypnosError::BackendError(format!("model '{}' is not loaded", cfg.id))
                })?;
                let voice = resolve_voice(&requested_voice, &resident.voices)
                    .ok_or_else(|| HypnosError::Config("no Kokoro voices available".into()))?
                    .to_string();
                let started = Instant::now();
                let mut samples: Vec<f32> = Vec::new();
                synthesize(resident, &vocab, &text, &voice, speed, false, &ctx, &mut |c| {
                    samples.extend_from_slice(c);
                    true
                })?;
                let synth_ms = started.elapsed().as_secs_f64() * 1000.0;
                info!(model = %cfg.id, voice = %voice, chars = text.len(), audio_secs = format!("{:.2}", samples.len() as f32 / KOKORO_SAMPLE_RATE as f32), synth_secs = format!("{:.2}", synth_ms / 1000.0), "speech synthesized");
                Ok((samples, voice, queue_ms, synth_ms))
            })
            .await
            .map_err(|e| HypnosError::BackendError(format!("tts task panicked: {e}")))??;
        let audio_secs = samples.len() as f64 / f64::from(KOKORO_SAMPLE_RATE);
        let engine = EngineTiming {
            queue_ms: Some(queue_ms),
            decode_ms: Some(synth_ms),
            audio_secs: Some(audio_secs),
            rtf: (audio_secs > 0.0).then(|| synth_ms / 1000.0 / audio_secs),
            ..EngineTiming::default()
        };

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
                inference_latency_ms: synth_ms,
                evicted_models: Vec::new(),
                engine,
            },
        })
    }

    async fn synthesize_speech_stream(
        &self,
        req: &AudioSpeechRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<AudioStreamEvent>,
    ) -> Result<()> {
        let (text, speed) = self.check_request(req, ctx)?;
        let ctx = ctx.clone();
        let requested_voice = req.voice.clone();
        let cfg = self.config.clone();
        let loaded = self.loaded.clone();
        let vocab = self.vocab.clone();
        let requested_at = Instant::now();
        let chunk_tx = tx.clone();

        let outcome = tokio::task::spawn_blocking(move || -> Result<EngineTiming> {
            let mut guard = loaded
                .lock()
                .map_err(|_| HypnosError::BackendError("kokoro mutex poisoned".into()))?;
            let queue_ms = requested_at.elapsed().as_secs_f64() * 1000.0;
            let resident = guard.as_mut().ok_or_else(|| {
                HypnosError::BackendError(format!("model '{}' is not loaded", cfg.id))
            })?;
            let voice = resolve_voice(&requested_voice, &resident.voices)
                .ok_or_else(|| HypnosError::Config("no Kokoro voices available".into()))?
                .to_string();
            let started = Instant::now();
            let mut ttfa_ms: Option<f64> = None;
            let total = synthesize(resident, &vocab, &text, &voice, speed, true, &ctx, &mut |c| {
                ttfa_ms.get_or_insert_with(|| requested_at.elapsed().as_secs_f64() * 1000.0);
                chunk_tx
                    .blocking_send(AudioStreamEvent::Chunk(AudioChunk {
                        pcm: c.to_vec(),
                        sample_rate: KOKORO_SAMPLE_RATE,
                    }))
                    .is_ok()
            })?;
            let synth_ms = started.elapsed().as_secs_f64() * 1000.0;
            let audio_secs = total as f64 / f64::from(KOKORO_SAMPLE_RATE);
            info!(model = %cfg.id, voice = %voice, chars = text.len(), audio_secs = format!("{audio_secs:.2}"), synth_secs = format!("{:.2}", synth_ms / 1000.0), ttfa_ms = ttfa_ms.map(|v| format!("{v:.0}")), "speech streamed");
            Ok(EngineTiming {
                queue_ms: Some(queue_ms),
                decode_ms: Some(synth_ms),
                ttfa_ms,
                audio_secs: Some(audio_secs),
                rtf: (audio_secs > 0.0).then(|| synth_ms / 1000.0 / audio_secs),
                ..EngineTiming::default()
            })
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("tts task panicked: {e}")))?;

        match outcome {
            Ok(timing) => {
                let _ = tx.send(AudioStreamEvent::Done { timing }).await;
                Ok(())
            }
            Err(e) => {
                let _ = tx.send(AudioStreamEvent::Error(e.to_string())).await;
                Err(e)
            }
        }
    }
}

impl OnnxRuntimeBackend {
    /// Shared request validation: returns the text to speak and the clamped speed.
    fn check_request(
        &self,
        req: &AudioSpeechRequest,
        ctx: &RequestContext,
    ) -> Result<(String, f32)> {
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
        let speed = if req.speed.is_finite() {
            req.speed.clamp(0.5, 2.0)
        } else {
            1.0
        };
        Ok((req.input.clone(), speed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentences_split_at_terminal_punctuation() {
        let s = split_sentences("Hello there. Is it working? Yes! Trailing text without a stop");
        assert_eq!(
            s,
            vec![
                "Hello there.",
                "Is it working?",
                "Yes!",
                "Trailing text without a stop"
            ]
        );
        assert_eq!(
            split_sentences("Version 2.5 is out. Good."),
            vec!["Version 2.5 is out.", "Good."]
        );
    }

    #[test]
    fn first_piece_is_cut_at_a_clause_or_word_boundary() {
        // A clause boundary soon after the minimum: cut there, with a pause.
        let clause =
            "Hypnos keeps every model in one process, and moves weights between tiers as needed.";
        let p = split_pieces(clause, true);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].text, "Hypnos keeps every model in one process,");
        assert!(p[0].pause_after);
        assert_eq!(p[1].text, "and moves weights between tiers as needed.");
        // No clause boundary in reach: cut at the next word boundary, without a pause.
        let words = "The lighthouse keeper logged the weather twice a day in a ledger bound in green cloth.";
        let p = split_pieces(words, true);
        assert_eq!(p.len(), 2);
        assert_eq!(p[0].text, "The lighthouse keeper logged");
        assert!(!p[0].pause_after);
        assert!(p[1].pause_after);
        // Short first sentences stay whole; nothing changes without `short_first`.
        assert_eq!(
            split_pieces("Short one, with a comma. Next.", true).len(),
            2
        );
        assert_eq!(split_pieces(words, false).len(), 1);
    }

    #[test]
    fn streaming_starts_with_a_few_words_then_grows() {
        let vocab: HashMap<char, i64> = VOCAB.iter().copied().collect();
        let text = "Hypnos keeps every model inside one process. The scheduler decides what stays \
                    resident, and the execution path decides how fast a resident model answers. \
                    This third sentence only makes the paragraph longer.";
        let merged = phonemize_chunks(text, &vocab, false).unwrap();
        let streamed = phonemize_chunks(text, &vocab, true).unwrap();
        assert_eq!(merged.len(), 1, "short paragraphs merge into one chunk");
        assert!(streamed.len() >= 2, "{}", streamed.len());
        assert!(streamed[0].ids.len() <= STREAM_CHUNK_BUDGETS[0]);
        assert!(streamed[1].ids.len() <= STREAM_CHUNK_BUDGETS[1]);
        assert!(streamed.last().unwrap().pause_after);
    }
}
