//! Synthesizes speech with kokoro-82m-q4: the whole file, then the same text streamed.
//!
//!     cargo run --example speech -- "Hello from Hypnos." out.wav

use futures::StreamExt;
use hypnos_client::{Client, SpeechRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let text = args.next().unwrap_or_else(|| "Hello from Hypnos.".into());
    let out = args.next().unwrap_or_else(|| "speech.wav".into());
    let hypnos = Client::new()?;

    let speech = hypnos
        .speech(&SpeechRequest::new("kokoro-82m-q4", &text).voice("nova"))
        .await?;
    std::fs::write(&out, &speech.audio)?;
    println!(
        "wrote {out} ({}, real-time factor {:?})",
        speech.content_type, speech.telemetry.rtf
    );

    // Streaming: raw 16-bit PCM at 24 kHz, playable as it arrives.
    let mut stream = hypnos
        .speech_stream(&SpeechRequest::new("kokoro-82m-q4", &text).response_format("pcm"))
        .await?;
    let mut received = 0usize;
    while let Some(chunk) = stream.next().await {
        received += chunk?.len();
    }
    println!("streamed {:.2} s of audio", received as f64 / 48_000.0);
    Ok(())
}
