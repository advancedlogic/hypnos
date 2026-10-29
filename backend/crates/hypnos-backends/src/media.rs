//! Media helpers shared by the engines: fetching image / audio payloads referenced by chat
//! requests (bounded by the request's limits) and encoding PCM audio into RIFF WAV containers.

use base64::Engine;
use hypnos_core::config::LimitsConfig;
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::openai::MediaRef;
use std::io::{Cursor, Read};

/// Resolves a chat media reference into raw bytes, enforcing the size limits in `limits`.
///
/// * `data:<mime>;base64,<payload>` URLs are decoded in place.
/// * `http(s)://` URLs are fetched when `limits.allow_remote_image_urls` is set.
/// * `input_audio` parts are base64-decoded.
pub fn load_media(media: &MediaRef, limits: &LimitsConfig) -> Result<Vec<u8>> {
    match media {
        MediaRef::ImageUrl(url) => {
            let bytes = load_url(url, limits)?;
            check_size(bytes.len(), limits.max_image_bytes, "image")?;
            Ok(bytes)
        }
        MediaRef::Audio { data_b64, .. } => {
            // base64 inflates by 4/3: reject before decoding when the payload is obviously too big.
            check_size(data_b64.len() / 4 * 3, limits.max_audio_bytes, "audio")?;
            let bytes = decode_base64(data_b64)?;
            check_size(bytes.len(), limits.max_audio_bytes, "audio")?;
            Ok(bytes)
        }
    }
}

fn check_size(actual: usize, limit: u64, what: &str) -> Result<()> {
    if actual as u64 > limit {
        return Err(HypnosError::InvalidRequest(format!(
            "{what} payload is {actual} bytes, above the configured limit of {limit} bytes"
        )));
    }
    Ok(())
}

fn load_url(url: &str, limits: &LimitsConfig) -> Result<Vec<u8>> {
    if let Some(rest) = url.strip_prefix("data:") {
        let (_meta, payload) = rest.split_once(',').ok_or_else(|| {
            HypnosError::InvalidRequest("malformed data: URL (missing ',')".to_string())
        })?;
        check_size(payload.len() / 4 * 3, limits.max_image_bytes, "image")?;
        return decode_base64(payload);
    }
    if url.starts_with("http://") || url.starts_with("https://") {
        if !limits.allow_remote_image_urls {
            return Err(HypnosError::InvalidRequest(
                "remote image URLs are disabled (limits.allow_remote_image_urls = false); send a data: URL".to_string(),
            ));
        }
        let resp = ureq::get(url)
            .timeout(std::time::Duration::from_secs(30))
            .call()
            .map_err(|e| HypnosError::InvalidRequest(format!("failed to fetch {url}: {e}")))?;
        let mut buf = Vec::new();
        resp.into_reader()
            .take(limits.max_image_bytes.saturating_add(1))
            .read_to_end(&mut buf)
            .map_err(|e| HypnosError::InvalidRequest(format!("failed to read {url}: {e}")))?;
        return Ok(buf);
    }
    Err(HypnosError::InvalidRequest(format!(
        "unsupported media URL scheme (expected data: or http(s)://): {}",
        url.chars().take(64).collect::<String>()
    )))
}

pub fn decode_base64(payload: &str) -> Result<Vec<u8>> {
    let cleaned: String = payload.chars().filter(|c| !c.is_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(cleaned.as_bytes())
        .or_else(|_| base64::engine::general_purpose::URL_SAFE.decode(cleaned.as_bytes()))
        .map_err(|e| HypnosError::InvalidRequest(format!("invalid base64 media payload: {e}")))
}

/// Encodes mono f32 PCM samples in `[-1, 1]` as a 16-bit PCM RIFF WAV byte buffer.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)
            .map_err(|e| HypnosError::BackendError(format!("wav writer: {e}")))?;
        for &s in samples {
            let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
            writer
                .write_sample(v)
                .map_err(|e| HypnosError::BackendError(format!("wav write: {e}")))?;
        }
        writer
            .finalize()
            .map_err(|e| HypnosError::BackendError(format!("wav finalize: {e}")))?;
    }
    Ok(cursor.into_inner())
}

/// Encodes mono f32 PCM samples as raw little-endian 16-bit PCM (no container).
pub fn encode_pcm16_raw(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Parses an OpenAI-style `WIDTHxHEIGHT` size string, rounding each side down to a multiple of 16
/// (diffusion latents require it) and clamping to a sane range.
pub fn parse_image_size(size: &str) -> Result<(u32, u32)> {
    let (w, h) = size
        .to_ascii_lowercase()
        .split_once('x')
        .and_then(|(w, h)| Some((w.trim().parse::<u32>().ok()?, h.trim().parse::<u32>().ok()?)))
        .ok_or_else(|| {
            HypnosError::InvalidRequest(format!("invalid size '{size}', expected WIDTHxHEIGHT"))
        })?;
    let snap = |v: u32| (v.clamp(256, 2048) / 16) * 16;
    Ok((snap(w), snap(h)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_size_snaps_and_clamps() {
        assert_eq!(parse_image_size("1024x1024").unwrap(), (1024, 1024));
        assert_eq!(parse_image_size("1000x700").unwrap(), (992, 688));
        assert_eq!(parse_image_size("16x4096").unwrap(), (256, 2048));
        assert!(parse_image_size("square").is_err());
    }

    #[test]
    fn wav_roundtrip() {
        let samples: Vec<f32> = (0..2400).map(|i| ((i as f32) * 0.05).sin() * 0.5).collect();
        let wav = encode_wav_pcm16(&samples, 24_000).unwrap();
        let reader = hound::WavReader::new(Cursor::new(&wav)).unwrap();
        assert_eq!(reader.spec().sample_rate, 24_000);
        assert_eq!(reader.spec().channels, 1);
        assert_eq!(reader.duration(), 2400);
    }

    #[test]
    fn media_limits_are_enforced() {
        let mut limits = LimitsConfig::default();
        limits.max_image_bytes = 8;
        let big = MediaRef::ImageUrl(format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode([0u8; 64])
        ));
        assert!(load_media(&big, &limits).is_err());
        limits.allow_remote_image_urls = false;
        let remote = MediaRef::ImageUrl("https://example.com/a.png".into());
        assert!(load_media(&remote, &limits).is_err());
        let small = MediaRef::Audio {
            data_b64: base64::engine::general_purpose::STANDARD.encode([1u8; 4]),
            format: "wav".into(),
        };
        assert_eq!(load_media(&small, &limits).unwrap(), vec![1u8; 4]);
    }
}
