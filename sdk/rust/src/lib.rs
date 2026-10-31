//! Async Rust client for the [Hypnos](https://github.com/advancedlogic/hypnos) multi-modal
//! inference server: every route of the HTTP API (`docs/api.md`), streaming with cancellation,
//! the retry policy the API recommends, readiness polling and a per-model concurrency bound.
//!
//! ```no_run
//! use hypnos_client::{ChatMessage, ChatRequest, Client};
//! use std::time::Duration;
//!
//! # async fn run() -> hypnos_client::Result<()> {
//! let hypnos = Client::new()?; // http://127.0.0.1:9999, or $HYPNOS_BASE_URL / $HYPNOS_API_KEY
//! hypnos.wait_until_ready(Duration::from_secs(600), Duration::from_secs(1)).await?;
//! let reply = hypnos
//!     .chat(&ChatRequest::new("gemma4:26b-q4", vec![ChatMessage::user("Hello")]))
//!     .await?;
//! println!("{}", reply.text());
//! # Ok(()) }
//! ```

mod client;
mod error;
mod retry;
mod streams;
pub mod types;

pub use client::{
    AdvisorRequest, ChatRequest, Client, ClientBuilder, DocumentRequest, DocumentSource,
    EntitiesRequest, ImageRequest, Input, LogQuery, SpeechRequest, TranscriptionRequest,
    DEFAULT_BASE_URL, DEFAULT_TIMEOUT,
};
pub use error::{ApiError, Error, Result};
pub use retry::RetryPolicy;
pub use streams::{ChatStream, DocumentStream, SpeechStream};
pub use types::*;

/// Slices `text` by the character offsets entities use (`[start, end)`, Unicode scalar values,
/// not bytes). Out-of-range offsets are clamped.
pub fn slice_chars(text: &str, start: usize, end: usize) -> &str {
    let byte = |n: usize| {
        text.char_indices()
            .nth(n)
            .map(|(i, _)| i)
            .unwrap_or(text.len())
    };
    let (a, b) = (byte(start), byte(end.max(start)));
    &text[a..b]
}

#[cfg(test)]
mod tests {
    use super::slice_chars;

    #[test]
    fn slices_by_chars() {
        let text = "Zoë met 🦀 Ferris";
        assert_eq!(slice_chars(text, 0, 3), "Zoë");
        assert_eq!(slice_chars(text, 8, 9), "🦀");
        assert_eq!(slice_chars(text, 10, 99), "Ferris");
    }
}
