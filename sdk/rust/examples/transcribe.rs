//! Transcribes a recording with gemma4:e4b-q4 (WAV, 16-bit PCM is the reliable format).
//!
//!     cargo run --example transcribe -- clip.wav [language]

use hypnos_client::{Client, TranscriptionRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: transcribe <audio> [language]")?;
    let mut request = TranscriptionRequest::from_path("gemma4:e4b-q4", &path)?;
    if let Some(language) = args.next() {
        request = request.language(language);
    }
    let result = Client::new()?.transcribe(&request).await?;
    println!("{}", result.text);
    if let Some(engine) = result.hypnos_telemetry.map(|t| t.engine) {
        if let Some(rtf) = engine.rtf {
            println!(
                "\n{:.1} s of audio, real-time factor {rtf:.2}",
                engine.audio_secs.unwrap_or(0.0)
            );
        }
    }
    Ok(())
}
