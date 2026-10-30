//! Embeds a few sentences with gemmaembedding2 and ranks them against a query.
//!
//!     cargo run --example embeddings -- "how do I reset my password"

use hypnos_client::Client;

#[tokio::main]
async fn main() -> hypnos_client::Result<()> {
    let docs = [
        "To change your password, open Settings and choose Security.",
        "Our office is closed on public holidays.",
        "Invoices are sent on the first day of each month.",
    ];
    let query = std::env::args().skip(1).collect::<Vec<_>>().join(" ");
    let query = if query.is_empty() {
        "how do I reset my password".into()
    } else {
        query
    };

    let mut input = vec![query.as_str()];
    input.extend(docs);
    let result = Client::new()?.embeddings("gemmaembedding2", input).await?;
    let q = &result.data[0].embedding;
    // The vectors are L2-normalised: the dot product is the cosine similarity.
    let mut ranked: Vec<(f32, &str)> = result.data[1..]
        .iter()
        .zip(docs)
        .map(|(d, doc)| (q.iter().zip(&d.embedding).map(|(a, b)| a * b).sum(), doc))
        .collect();
    ranked.sort_by(|a, b| b.0.total_cmp(&a.0));
    for (score, doc) in ranked {
        println!("{score:.3}  {doc}");
    }
    Ok(())
}
