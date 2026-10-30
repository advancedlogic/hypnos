//! Finds people, places and companies with gliner2.5-multi (zero-shot: any labels work).
//!
//!     cargo run --example entities -- "Mario Rossi works in Milan for Ferrari."

use hypnos_client::{slice_chars, Client, EntitiesRequest};

#[tokio::main]
async fn main() -> hypnos_client::Result<()> {
    let text = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let text = if text.is_empty() {
        "Mario Rossi works in Milan for Ferrari.".to_string()
    } else {
        text
    };
    let result = Client::new()?
        .entities(&EntitiesRequest::new(
            "gliner2.5-multi",
            text.as_str(),
            &["person", "city", "company"],
        ))
        .await?;
    for e in &result.data[0].entities {
        // start/end are character offsets, not byte offsets.
        println!(
            "{:>8}  {:?}  {:.2}",
            e.label,
            slice_chars(&text, e.start, e.end),
            e.score
        );
    }
    Ok(())
}
