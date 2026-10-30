//! Streams a chat reply from gemma4:26b-q4, then prints what the request cost.
//!
//!     cargo run --example chat -- "Explain sliding-window attention in two sentences."

use std::io::Write;
use std::time::Duration;

use futures::StreamExt;
use hypnos_client::{ChatMessage, ChatRequest, Client};

#[tokio::main]
async fn main() -> hypnos_client::Result<()> {
    let prompt = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let prompt = if prompt.is_empty() {
        "Explain sliding-window attention in two sentences.".to_string()
    } else {
        prompt
    };
    let hypnos = Client::new()?;
    hypnos
        .wait_until_ready(Duration::from_secs(600), Duration::from_secs(1))
        .await?;
    let request = ChatRequest::new(
        "gemma4:26b-q4",
        vec![
            ChatMessage::system("You are terse."),
            ChatMessage::user(prompt),
        ],
    )
    .max_tokens(300);
    let mut stream = hypnos.chat_stream(&request).await?;
    {
        let mut text = stream.text();
        while let Some(piece) = text.next().await {
            print!("{}", piece?);
            std::io::stdout().flush().ok();
        }
    }
    println!();
    if let (Some(usage), Some(t)) = (stream.usage(), stream.telemetry()) {
        println!(
            "\n{} tokens, model was in {}, load {:.0} ms, decode {:.0} tok/s",
            usage.completion_tokens,
            t.previous_tier.as_str(),
            t.tier_switch_latency_ms,
            t.engine.decode_tokens_per_s.unwrap_or(0.0)
        );
    }
    Ok(())
}
