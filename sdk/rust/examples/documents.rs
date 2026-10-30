//! Converts a document to Markdown with docling, page by page, and saves its pictures.
//!
//!     cargo run --example documents -- report.pdf [pages]

use futures::StreamExt;
use hypnos_client::{Client, DocumentEvent, DocumentRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args.next().ok_or("usage: documents <file> [pages]")?;
    let mut request = DocumentRequest::from_path("docling", &path)?.return_pictures(true);
    if let Some(pages) = args.next() {
        request = request.pages(pages);
    }
    let hypnos = Client::new()?;

    let mut stream = hypnos.convert_document_stream(&request).await?;
    while let Some(event) = stream.next().await {
        match event? {
            DocumentEvent::Started(s) => {
                println!("{} pages, converting {:?}", s.page_count, s.pages)
            }
            DocumentEvent::Page(p) => {
                println!(
                    "page {}: {} tables, {} pictures",
                    p.page, p.tables, p.pictures
                )
            }
            DocumentEvent::Done(_) => {}
        }
    }
    let doc = stream.final_document().await?;
    std::fs::write(format!("{path}.md"), &doc.markdown)?;
    println!("wrote {path}.md");

    if let Some(id) = &doc.id {
        for picture in hypnos.document_pictures(id).await?.data {
            let (data, mime) = hypnos.document_picture_bytes(id, picture.index).await?;
            let ext = mime.rsplit('/').next().unwrap_or("bin");
            let name = format!("{path}.picture{}.{ext}", picture.index);
            std::fs::write(&name, data)?;
            println!(
                "wrote {name} ({})",
                picture.caption.as_deref().unwrap_or("no caption")
            );
        }
        hypnos.delete_document(id).await?;
    }
    Ok(())
}
