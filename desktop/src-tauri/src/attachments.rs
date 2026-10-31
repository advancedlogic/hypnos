//! Files attached to a message (DESK-7), prepared when they are added so the message only carries
//! what the model needs: images as a downscaled JPEG data URL, documents as markdown (docling, or
//! read directly for plain text), audio as a transcript, video as its transcript plus frames the
//! webview samples (it can decode video; Rust only reads the soundtrack).

use std::io::Cursor;
use std::path::Path;

use base64::Engine;
use hypnos_client::{DocumentRequest, TranscriptionRequest};
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Manager, Runtime, State};

use crate::hypnos::{ErrorPayload, Hypnos};

/// Longest side of an image sent to a vision model (and kept with the chat).
const IMAGE_MAX_SIDE: u32 = 1024;
const JPEG_QUALITY: u8 = 85;
/// About 15k tokens; a longer document is cut with a note (the context window is finite).
const DOCUMENT_MAX_CHARS: usize = 60_000;
/// The server's default upload limits (docs/api.md section 1).
const MAX_IMAGE_BYTES: u64 = 32 << 20;
const MAX_AUDIO_BYTES: u64 = 64 << 20;
const MAX_DOCUMENT_BYTES: u64 = 100 << 20;
/// Video files are not uploaded (only frames and the soundtrack are), so the cap is generous.
const MAX_VIDEO_BYTES: u64 = 2 << 30;
const TRANSCRIPTION_RATE: u32 = 16_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "camelCase")]
pub enum Attachment {
    #[serde(rename_all = "camelCase")]
    Image {
        name: String,
        size: u64,
        data_url: String,
        width: u32,
        height: u32,
    },
    #[serde(rename_all = "camelCase")]
    Document {
        name: String,
        size: u64,
        text: String,
        pages: u32,
        truncated: bool,
    },
    #[serde(rename_all = "camelCase")]
    Audio {
        name: String,
        size: u64,
        transcript: String,
        duration_secs: f64,
    },
    /// `frames` are filled by the webview; `path` lets it open the file through the asset protocol.
    #[serde(rename_all = "camelCase")]
    Video {
        name: String,
        size: u64,
        path: Option<String>,
        transcript: Option<String>,
        #[serde(default)]
        frames: Vec<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Image,
    Text,
    Document,
    Audio,
    Video,
}

fn kind_of(name: &str) -> Option<Kind> {
    let ext = Path::new(name).extension()?.to_str()?.to_ascii_lowercase();
    Some(match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "gif" | "bmp" | "tif" | "tiff" => Kind::Image,
        "txt" | "md" | "markdown" | "csv" | "tsv" | "json" | "yaml" | "yml" | "toml" | "xml"
        | "log" | "rs" | "py" | "js" | "ts" | "go" | "java" | "c" | "h" | "cpp" | "sh" => {
            Kind::Text
        }
        "pdf" | "docx" | "pptx" | "xlsx" | "html" | "htm" | "xhtml" | "asciidoc" | "adoc" => {
            Kind::Document
        }
        "wav" | "mp3" | "m4a" | "aac" | "flac" | "ogg" | "oga" => Kind::Audio,
        "mp4" | "mov" | "m4v" | "webm" | "mkv" => Kind::Video,
        _ => return None,
    })
}

fn invalid(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload {
        kind: "attachment".into(),
        message: message.into(),
        detail: None,
        retry_after_secs: None,
    }
}

fn mib(bytes: u64) -> String {
    format!("{} MiB", bytes >> 20)
}

/// Prepares a file picked or dropped by the user.
#[tauri::command]
pub async fn attachment_from_path<R: Runtime>(
    app: AppHandle<R>,
    hypnos: State<'_, Hypnos>,
    path: String,
    chat_model: Option<String>,
) -> Result<Attachment, ErrorPayload> {
    let p = Path::new(&path);
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.clone());
    let kind = kind_of(&name).ok_or_else(|| unsupported(&name))?;
    let size = std::fs::metadata(p)
        .map_err(|e| invalid(format!("Cannot read {name}: {e}")))?
        .len();
    if kind == Kind::Video {
        check_size(&name, size, MAX_VIDEO_BYTES)?;
        // The webview reads the frames through the asset protocol, for this file only.
        app.asset_protocol_scope()
            .allow_file(p)
            .map_err(|e| invalid(format!("Cannot open {name}: {e}")))?;
        let transcript = match std::fs::read(p) {
            Ok(bytes) => soundtrack_transcript(&hypnos, &name, bytes, chat_model.as_deref()).await,
            Err(_) => None,
        };
        return Ok(Attachment::Video {
            name,
            size,
            path: Some(path),
            transcript,
            frames: vec![],
        });
    }
    let bytes = tokio::fs::read(p)
        .await
        .map_err(|e| invalid(format!("Cannot read {name}: {e}")))?;
    prepare(&hypnos, name, bytes, kind, chat_model.as_deref()).await
}

/// Prepares bytes pasted into the composer (the raw request body; the name in a header).
#[tauri::command]
pub async fn attachment_from_bytes(
    hypnos: State<'_, Hypnos>,
    request: tauri::ipc::Request<'_>,
) -> Result<Attachment, ErrorPayload> {
    let tauri::ipc::InvokeBody::Raw(bytes) = request.body() else {
        return Err(invalid("Expected the file's bytes."));
    };
    let header = |h: &str| {
        request
            .headers()
            .get(h)
            .and_then(|v| v.to_str().ok())
            .map(str::to_string)
    };
    let name = header("x-name").unwrap_or_else(|| "pasted.png".into());
    let chat_model = header("x-chat-model");
    let kind = kind_of(&name).ok_or_else(|| unsupported(&name))?;
    if kind == Kind::Video {
        return Err(invalid(
            "Paste a video as a file (drag it in or use the paperclip).",
        ));
    }
    prepare(&hypnos, name, bytes.clone(), kind, chat_model.as_deref()).await
}

fn unsupported(name: &str) -> ErrorPayload {
    invalid(format!(
        "{name}: unsupported file type (images, PDF and office documents, text, audio and video work)."
    ))
}

fn check_size(name: &str, size: u64, max: u64) -> Result<(), ErrorPayload> {
    if size > max {
        return Err(invalid(format!(
            "{name} is larger than the server accepts ({}).",
            mib(max)
        )));
    }
    Ok(())
}

async fn prepare(
    hypnos: &Hypnos,
    name: String,
    bytes: Vec<u8>,
    kind: Kind,
    chat_model: Option<&str>,
) -> Result<Attachment, ErrorPayload> {
    let size = bytes.len() as u64;
    match kind {
        Kind::Image => {
            check_size(&name, size, MAX_IMAGE_BYTES)?;
            let (data_url, width, height) =
                tauri::async_runtime::spawn_blocking(move || downscale_jpeg(&bytes))
                    .await
                    .map_err(|e| invalid(e.to_string()))?
                    .map_err(|e| invalid(format!("{name}: {e}")))?;
            Ok(Attachment::Image {
                name,
                size,
                data_url,
                width,
                height,
            })
        }
        Kind::Text => {
            check_size(&name, size, MAX_DOCUMENT_BYTES)?;
            let (text, truncated) = cap(String::from_utf8_lossy(&bytes).into_owned());
            Ok(Attachment::Document {
                name,
                size,
                text,
                pages: 1,
                truncated,
            })
        }
        Kind::Document => {
            check_size(&name, size, MAX_DOCUMENT_BYTES)?;
            let model = hypnos.pick("document_conversion", None).await?;
            let client = hypnos.client().await?;
            let doc = client
                .convert_document(&DocumentRequest::file(model, name.clone(), bytes))
                .await?;
            let (text, truncated) = cap(doc.markdown);
            Ok(Attachment::Document {
                name,
                size,
                text,
                pages: doc.page_count,
                truncated,
            })
        }
        Kind::Audio => {
            check_size(&name, size, MAX_AUDIO_BYTES)?;
            let (wav, duration_secs) = to_wav(bytes, &name).await?;
            let transcript = transcribe(hypnos, &name, wav, chat_model).await?;
            Ok(Attachment::Audio {
                name,
                size,
                transcript,
                duration_secs,
            })
        }
        Kind::Video => unreachable!("videos are prepared from their path"),
    }
}

fn cap(text: String) -> (String, bool) {
    if text.chars().count() <= DOCUMENT_MAX_CHARS {
        return (text, false);
    }
    (text.chars().take(DOCUMENT_MAX_CHARS).collect(), true)
}

/// Decodes any image the `image` crate reads and re-encodes it as a JPEG no larger than
/// [`IMAGE_MAX_SIDE`]; returns the data URL and the new size.
pub fn downscale_jpeg(bytes: &[u8]) -> Result<(String, u32, u32), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("not a readable image ({e})"))?;
    let img = if img.width().max(img.height()) > IMAGE_MAX_SIDE {
        img.resize(
            IMAGE_MAX_SIDE,
            IMAGE_MAX_SIDE,
            image::imageops::FilterType::Triangle,
        )
    } else {
        img
    };
    let rgb = img.to_rgb8();
    let mut out = Cursor::new(Vec::new());
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, JPEG_QUALITY)
        .encode_image(&rgb)
        .map_err(|e| e.to_string())?;
    let b64 = base64::engine::general_purpose::STANDARD.encode(out.into_inner());
    Ok((
        format!("data:image/jpeg;base64,{b64}"),
        rgb.width(),
        rgb.height(),
    ))
}

async fn to_wav(bytes: Vec<u8>, name: &str) -> Result<(Vec<u8>, f64), ErrorPayload> {
    let hint = Path::new(name)
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_string);
    let name = name.to_string();
    tauri::async_runtime::spawn_blocking(move || decode_to_wav(bytes, hint.as_deref()))
        .await
        .map_err(|e| invalid(e.to_string()))?
        .map_err(|e| invalid(format!("{name}: {e}")))
}

async fn transcribe(
    hypnos: &Hypnos,
    name: &str,
    wav: Vec<u8>,
    chat_model: Option<&str>,
) -> Result<String, ErrorPayload> {
    let model = hypnos.pick("speech_to_text", chat_model).await?;
    let client = hypnos.client().await?;
    let wav_name = format!(
        "{}.wav",
        Path::new(name)
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let result = client
        .transcribe(&TranscriptionRequest::new(model, wav_name, wav))
        .await?;
    Ok(result.text.trim().to_string())
}

/// The soundtrack of a video, transcribed; `None` when it has none or its codec is unsupported.
async fn soundtrack_transcript(
    hypnos: &Hypnos,
    name: &str,
    bytes: Vec<u8>,
    chat_model: Option<&str>,
) -> Option<String> {
    let (wav, duration) = to_wav(bytes, name).await.ok()?;
    if duration < 0.5 {
        return None;
    }
    transcribe(hypnos, name, wav, chat_model)
        .await
        .ok()
        .filter(|t| !t.is_empty())
}

/// Decodes the first audio track of `bytes` (WAV, MP3, AAC/M4A, FLAC, Vorbis, and the audio of
/// MP4/MOV/MKV) to 16 kHz mono 16-bit WAV; returns it with its duration in seconds.
pub fn decode_to_wav(bytes: Vec<u8>, ext: Option<&str>) -> Result<(Vec<u8>, f64), String> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = ext {
        hint.with_extension(ext);
    }
    let probed = symphonia::default::get_probe()
        .format(
            &hint,
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("cannot read the audio ({e})"))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("no audio track")?;
    let track_id = track.id;
    let source_rate = track
        .codec_params
        .sample_rate
        .ok_or("unknown sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| format!("unsupported audio codec ({e})"))?;

    let mut mono: Vec<f32> = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(symphonia::core::errors::Error::IoError(_)) => break, // end of stream
            Err(symphonia::core::errors::Error::ResetRequired) => break,
            Err(e) => return Err(e.to_string()),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(symphonia::core::errors::Error::DecodeError(_)) => continue,
            Err(e) => return Err(e.to_string()),
        };
        let channels = decoded.spec().channels.count().max(1);
        let mut buf = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        buf.copy_interleaved_ref(decoded);
        mono.extend(
            buf.samples()
                .chunks(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
    }
    if mono.is_empty() {
        return Err("the audio track is empty".into());
    }
    let duration = mono.len() as f64 / source_rate as f64;
    let samples = resample(&mono, source_rate, TRANSCRIPTION_RATE);
    Ok((wav_bytes(&samples, TRANSCRIPTION_RATE), duration))
}

/// Linear-interpolation resampling; plenty for speech recognition.
pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }
    let ratio = from as f64 / to as f64;
    let len = (input.len() as f64 / ratio).floor() as usize;
    (0..len)
        .map(|i| {
            let pos = i as f64 * ratio;
            let j = pos as usize;
            let frac = (pos - j as f64) as f32;
            let a = input[j];
            let b = *input.get(j + 1).unwrap_or(&a);
            a + (b - a) * frac
        })
        .collect()
}

/// A mono 16-bit PCM WAV file.
pub fn wav_bytes(samples: &[f32], rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_follow_the_extension() {
        assert_eq!(kind_of("a.PNG"), Some(Kind::Image));
        assert_eq!(kind_of("report.pdf"), Some(Kind::Document));
        assert_eq!(kind_of("notes.md"), Some(Kind::Text));
        assert_eq!(kind_of("talk.m4a"), Some(Kind::Audio));
        assert_eq!(kind_of("clip.mov"), Some(Kind::Video));
        assert_eq!(kind_of("archive.zip"), None);
        assert_eq!(kind_of("noextension"), None);
    }

    #[test]
    fn a_wav_round_trips_through_the_decoder_at_16_khz() {
        // One second of a 440 Hz tone at 44.1 kHz.
        let tone: Vec<f32> = (0..44_100)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 44_100.0).sin() * 0.5)
            .collect();
        let (wav, duration) = decode_to_wav(wav_bytes(&tone, 44_100), Some("wav")).unwrap();
        assert!((duration - 1.0).abs() < 0.01, "{duration}");
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        let samples = (wav.len() - 44) / 2;
        assert!((15_900..=16_000).contains(&samples), "{samples}");
    }

    #[test]
    fn large_images_are_downscaled_to_a_jpeg() {
        let img = image::RgbImage::from_pixel(3000, 1500, image::Rgb([200, 100, 50]));
        let mut png = Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let (url, w, h) = downscale_jpeg(png.get_ref()).unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));
        assert_eq!((w, h), (1024, 512));
    }

    #[test]
    fn long_documents_are_cut() {
        let (text, truncated) = cap("x".repeat(DOCUMENT_MAX_CHARS + 10));
        assert!(truncated);
        assert_eq!(text.len(), DOCUMENT_MAX_CHARS);
    }

    /// Against a running server, with `HYPNOS_TEST_FILES` holding `jfk.m4a` (speech, AAC) and
    /// `sample.pdf` (a page containing "pelican"): `cargo test attachments -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn prepares_audio_documents_and_images_on_a_live_server() {
        use crate::hypnos::{hypnos_chat, hypnos_connect, InMessage};
        use tauri::Manager;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let dir = std::path::PathBuf::from(
            std::env::var("HYPNOS_TEST_FILES").expect("HYPNOS_TEST_FILES"),
        );
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();
        let path = |f: &str| dir.join(f).display().to_string();

        let audio = attachment_from_path(app.handle().clone(), app.state(), path("jfk.m4a"), None)
            .await
            .unwrap();
        let Attachment::Audio {
            transcript,
            duration_secs,
            ..
        } = &audio
        else {
            panic!("{audio:?}")
        };
        assert!(
            transcript.to_lowercase().contains("country"),
            "{transcript}"
        );
        assert!((10.0..12.5).contains(duration_secs), "{duration_secs}");

        let doc = attachment_from_path(app.handle().clone(), app.state(), path("sample.pdf"), None)
            .await
            .unwrap();
        let Attachment::Document { text, pages, .. } = &doc else {
            panic!("{doc:?}")
        };
        assert!(text.contains("pelican"), "{text}");
        assert_eq!(*pages, 1);

        // A pure red image, then a chat that has to look at it.
        let img = image::RgbImage::from_pixel(64, 64, image::Rgb([220, 20, 20]));
        let mut png = Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png).unwrap();
        let hypnos = app.state::<Hypnos>();
        let image = prepare(
            &hypnos,
            "red.png".into(),
            png.into_inner(),
            Kind::Image,
            None,
        )
        .await
        .unwrap();
        let model = hypnos.pick("vision", None).await.unwrap();
        let text = std::sync::Arc::new(std::sync::Mutex::new(String::new()));
        let t = text.clone();
        let channel = tauri::ipc::Channel::new(move |body| {
            if let tauri::ipc::InvokeResponseBody::Json(s) = body {
                let v: serde_json::Value = serde_json::from_str(&s).unwrap();
                if v["event"] == "delta" {
                    t.lock().unwrap().push_str(v["text"].as_str().unwrap());
                }
            }
            Ok(())
        });
        let message = InMessage {
            role: "user".into(),
            content: "Answer with one word: what colour fills this image? Then say the secret word from the document.".into(),
            attachments: vec![image, doc],
        };
        hypnos_chat(app.state(), "a1".into(), model, vec![message], channel)
            .await
            .unwrap();
        let answer = text.lock().unwrap().to_lowercase();
        assert!(answer.contains("red"), "{answer}");
        assert!(answer.contains("pelican"), "{answer}");
    }
}
