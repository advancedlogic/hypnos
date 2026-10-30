//! Generates an image with moss-image and saves it as PNG.
//!
//!     cargo run --example images -- "a lighthouse at dawn, oil painting" lighthouse.png

use base64::Engine;
use hypnos_client::{Client, ImageRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let prompt = args
        .next()
        .unwrap_or_else(|| "a lighthouse at dawn, oil painting".into());
    let out = args.next().unwrap_or_else(|| "image.png".into());
    let result = Client::new()?
        .images(
            &ImageRequest::new("moss-image", prompt)
                .size("1024x1024")
                .steps(20)
                .seed(7),
        )
        .await?;
    let png = base64::engine::general_purpose::STANDARD.decode(&result.data[0].b64_json)?;
    std::fs::write(&out, png)?;
    let secs = result
        .hypnos_telemetry
        .map(|t| t.inference_latency_ms / 1000.0)
        .unwrap_or(0.0);
    println!("wrote {out} in {secs:.1} s");
    Ok(())
}
