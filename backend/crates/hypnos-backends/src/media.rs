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
            "{what} payload is {actual} bytes, above limits.max_{what}_bytes ({limit})"
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
        return fetch_remote(url, limits.max_image_bytes, "image", limits);
    }
    Err(HypnosError::InvalidRequest(format!(
        "unsupported media URL scheme (expected data: or http(s)://): {}",
        url.chars().take(64).collect::<String>()
    )))
}

/// Whether `ip` is a public internet address: not loopback, private, link-local, carrier-grade
/// NAT, documentation, benchmarking, multicast, broadcast, reserved or unspecified.
pub(crate) fn is_public(ip: std::net::IpAddr) -> bool {
    use std::net::IpAddr;
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, ..] = v4.octets();
            !(v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_broadcast()
                || v4.is_documentation()
                || v4.is_unspecified()
                || v4.is_multicast()
                || a == 0
                || (a == 100 && (64..128).contains(&b)) // carrier-grade NAT
                || (a == 198 && (b == 18 || b == 19)) // benchmarking
                || a >= 240) // reserved
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_public(IpAddr::V4(v4));
            }
            let first = v6.segments()[0];
            !(v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                || (first & 0xfe00) == 0xfc00 // unique local
                || (first & 0xffc0) == 0xfe80 // link-local
                || first == 0x2001 && v6.segments()[1] == 0x0db8) // documentation
        }
    }
}

/// Resolves only to public addresses: a remote image or document URL must not reach the LAN,
/// `localhost` or a cloud metadata endpoint unless `limits.allow_private_urls` says so.
struct PublicOnly;

impl ureq::Resolver for PublicOnly {
    fn resolve(&self, netloc: &str) -> std::io::Result<Vec<std::net::SocketAddr>> {
        use std::net::ToSocketAddrs;
        let public: Vec<_> = netloc
            .to_socket_addrs()?
            .filter(|a| is_public(a.ip()))
            .collect();
        if public.is_empty() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                format!("{netloc} is not a public address (limits.allow_private_urls = false)"),
            ));
        }
        Ok(public)
    }
}

/// Downloads `url` (at most `limit` + 1 bytes, so the caller's size check can reject it) when
/// `limits.allow_remote_image_urls` permits outbound fetches.
fn fetch_remote(url: &str, limit: u64, what: &str, limits: &LimitsConfig) -> Result<Vec<u8>> {
    if !limits.allow_remote_image_urls {
        return Err(HypnosError::InvalidRequest(format!(
            "remote {what} URLs are disabled (limits.allow_remote_image_urls = false); send the {what} itself"
        )));
    }
    let mut agent = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(30));
    if !limits.allow_private_urls {
        // Checked at every connection, redirects included, on the addresses actually used.
        agent = agent.resolver(PublicOnly);
    }
    let resp = agent
        .build()
        .get(url)
        .call()
        .map_err(|e| HypnosError::InvalidRequest(format!("failed to fetch {url}: {e}")))?;
    let mut buf = Vec::new();
    resp.into_reader()
        .take(limit.saturating_add(1))
        .read_to_end(&mut buf)
        .map_err(|e| HypnosError::InvalidRequest(format!("failed to read {url}: {e}")))?;
    Ok(buf)
}

/// Resolves a document sent to `/v1/documents/convert` as base64 or a `data:` URI (`file`) or as
/// an http(s) `url` (blocking), enforcing `limits.max_document_bytes`.
pub fn load_document(
    file: Option<&str>,
    url: Option<&str>,
    limits: &LimitsConfig,
) -> Result<Vec<u8>> {
    let limit = limits.max_document_bytes;
    let bytes = match (
        file.filter(|f| !f.is_empty()),
        url.filter(|u| !u.is_empty()),
    ) {
        (Some(file), None) => {
            let payload = file
                .strip_prefix("data:")
                .and_then(|rest| rest.split_once(','))
                .map_or(file, |(_, payload)| payload);
            check_size(payload.len() / 4 * 3, limit, "document")?;
            decode_base64(payload)?
        }
        (None, Some(url)) if url.starts_with("http://") || url.starts_with("https://") => {
            fetch_remote(url, limit, "document", limits)?
        }
        (None, Some(_)) => {
            return Err(HypnosError::InvalidRequest(
                "url must be an http(s) URL".to_string(),
            ))
        }
        _ => {
            return Err(HypnosError::InvalidRequest(
                "send the document as `file` (base64 or a data: URI) or as `url`, not both"
                    .to_string(),
            ))
        }
    };
    check_size(bytes.len(), limit, "document")?;
    Ok(bytes)
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

/// RIFF/WAV header for a live mono 16-bit PCM stream whose length is not known in advance: both
/// size fields are `0xFFFFFFFF`, which browsers and mainstream players accept and read to EOF.
pub fn wav_stream_header(sample_rate: u32) -> [u8; 44] {
    let mut h = [0u8; 44];
    h[0..4].copy_from_slice(b"RIFF");
    h[4..8].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    h[8..12].copy_from_slice(b"WAVE");
    h[12..16].copy_from_slice(b"fmt ");
    h[16..20].copy_from_slice(&16u32.to_le_bytes());
    h[20..22].copy_from_slice(&1u16.to_le_bytes()); // PCM
    h[22..24].copy_from_slice(&1u16.to_le_bytes()); // mono
    h[24..28].copy_from_slice(&sample_rate.to_le_bytes());
    h[28..32].copy_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    h[32..34].copy_from_slice(&2u16.to_le_bytes()); // block align
    h[34..36].copy_from_slice(&16u16.to_le_bytes()); // bits per sample
    h[36..40].copy_from_slice(b"data");
    h[40..44].copy_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
    h
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
    fn only_public_addresses_count_as_public() {
        use std::net::IpAddr;
        for private in [
            "127.0.0.1",
            "10.1.2.3",
            "172.16.0.1",
            "192.168.1.1",
            "169.254.169.254",
            "100.64.0.1",
            "0.0.0.0",
            "::1",
            "fc00::1",
            "fe80::1",
            "::ffff:127.0.0.1",
        ] {
            assert!(
                !super::is_public(private.parse::<IpAddr>().unwrap()),
                "{private}"
            );
        }
        for public in ["93.184.216.34", "1.1.1.1", "2606:4700:4700::1111"] {
            assert!(
                super::is_public(public.parse::<IpAddr>().unwrap()),
                "{public}"
            );
        }
    }

    #[test]
    fn remote_fetches_do_not_reach_private_addresses() {
        let limits = LimitsConfig::default();
        let err = load_media(
            &MediaRef::ImageUrl("http://127.0.0.1:9/secret.png".into()),
            &limits,
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("not a public address"), "{err}");
    }

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
