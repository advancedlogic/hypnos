//! Asks gemma4:26b-q4 about an image.
//!
//!     cargo run --example vision -- photo.jpg "What is in this picture?"

use base64::Engine;
use hypnos_client::{ChatMessage, ChatRequest, Client, ContentPart};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: vision <image> [question]")?;
    let question = args.collect::<Vec<_>>().join(" ");
    let question = if question.is_empty() {
        "Describe this picture.".into()
    } else {
        question
    };
    let mime = match path.rsplit('.').next().map(str::to_lowercase).as_deref() {
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("webp") => "image/webp",
        _ => "image/png",
    };
    let data = base64::engine::general_purpose::STANDARD.encode(std::fs::read(&path)?);

    let hypnos = Client::new()?;
    let message = ChatMessage::user_parts(vec![
        ContentPart::text(question),
        ContentPart::image_url(format!("data:{mime};base64,{data}")),
    ]);
    let reply = hypnos
        .chat(&ChatRequest::new("gemma4:26b-q4", vec![message]).max_tokens(300))
        .await?;
    println!("{}", reply.text());
    Ok(())
}
