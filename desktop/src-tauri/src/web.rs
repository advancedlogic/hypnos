//! Reading a URL into the chat (DESK-8). The Rust side fetches the page (so it works even when the
//! server sets `allow_remote_image_urls = false`, and from the user's network rather than the
//! server's), docling turns HTML, PDF and office files into markdown, and the user can save the
//! markdown or the original file.

use std::path::Path;
use std::time::Duration;

use hypnos_client::DocumentRequest;
use serde::Serialize;
use tauri::State;

use crate::attachments::{cap, Attachment};
use crate::hypnos::{ErrorPayload, Hypnos};

const MAX_BYTES: usize = 50 << 20;
const TIMEOUT: Duration = Duration::from_secs(30);
const USER_AGENT: &str = concat!("HypnosDesktop/", env!("CARGO_PKG_VERSION"));

fn fail(message: impl Into<String>) -> ErrorPayload {
    ErrorPayload {
        kind: "url".into(),
        message: message.into(),
        detail: None,
        retry_after_secs: None,
    }
}

/// What was downloaded: the bytes, their content type and a file name with a fitting extension.
pub struct Fetched {
    pub bytes: Vec<u8>,
    pub content_type: String,
    pub file_name: String,
    pub final_url: String,
}

fn parse(url: &str) -> Result<url::Url, ErrorPayload> {
    let parsed = url::Url::parse(url.trim()).map_err(|_| fail(format!("{url} is not a URL.")))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(fail("Only http:// and https:// links can be read."));
    }
    Ok(parsed)
}

pub async fn fetch(url: &str) -> Result<Fetched, ErrorPayload> {
    let parsed = parse(url)?;
    let http = reqwest::Client::builder()
        .timeout(TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(5))
        .user_agent(USER_AGENT)
        .build()
        .map_err(|e| fail(e.to_string()))?;
    let mut response = http
        .get(parsed.clone())
        .send()
        .await
        .map_err(|e| fail(format!("Could not fetch {parsed}: {e}")))?;
    if !response.status().is_success() {
        return Err(fail(format!("{parsed} answered {}.", response.status())));
    }
    if response
        .content_length()
        .is_some_and(|n| n as usize > MAX_BYTES)
    {
        return Err(fail(format!(
            "{parsed} is larger than {} MiB.",
            MAX_BYTES >> 20
        )));
    }
    let content_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| {
            v.split(';')
                .next()
                .unwrap_or("")
                .trim()
                .to_ascii_lowercase()
        })
        .unwrap_or_default();
    let final_url = response.url().to_string();
    let mut bytes = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .map_err(|e| fail(format!("The download broke: {e}")))?
    {
        bytes.extend_from_slice(&chunk);
        if bytes.len() > MAX_BYTES {
            return Err(fail(format!(
                "{parsed} is larger than {} MiB.",
                MAX_BYTES >> 20
            )));
        }
    }
    let file_name = file_name_for(&final_url, &content_type);
    Ok(Fetched {
        bytes,
        content_type,
        file_name,
        final_url,
    })
}

/// A file name for the download: the URL's last segment when it has an extension, else one
/// derived from the content type (`index.html` for a page).
pub fn file_name_for(url: &str, content_type: &str) -> String {
    let segment = url::Url::parse(url)
        .ok()
        .and_then(|u| {
            u.path_segments()?
                .rev()
                .find(|s| !s.is_empty())
                .map(str::to_string)
        })
        .map(|s| percent_decode(&s))
        .unwrap_or_default();
    let from_type = match content_type {
        "text/html" | "application/xhtml+xml" => Some("html"),
        "application/pdf" => Some("pdf"),
        "text/plain" => Some("txt"),
        "text/markdown" => Some("md"),
        "application/json" => Some("json"),
        "text/csv" => Some("csv"),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => Some("docx"),
        "application/vnd.openxmlformats-officedocument.presentationml.presentation" => Some("pptx"),
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" => Some("xlsx"),
        _ => None,
    };
    let has_ext = Path::new(&segment).extension().is_some();
    match (has_ext, from_type) {
        // A page served from /docs/intro.php is still HTML.
        (true, Some("html")) if !segment.ends_with(".html") && !segment.ends_with(".htm") => {
            format!(
                "{}.html",
                Path::new(&segment)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            )
        }
        (true, _) => segment,
        (false, Some(ext)) if !segment.is_empty() => format!("{segment}.{ext}"),
        (false, Some(ext)) => format!("index.{ext}"),
        (false, None) if !segment.is_empty() => segment,
        (false, None) => "download".into(),
    }
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok();
            if let Some(v) = hex.and_then(|h| u8::from_str_radix(h, 16).ok()) {
                out.push(v);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The `<title>` of an HTML page, with the common entities decoded.
pub fn html_title(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let start = lower.find("<title")?;
    let open_end = lower[start..].find('>')? + start + 1;
    let close = lower[open_end..].find("</title>")? + open_end;
    let title = html[open_end..close]
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ");
    let title = title.split_whitespace().collect::<Vec<_>>().join(" ");
    (!title.is_empty()).then_some(title)
}

fn first_heading(markdown: &str) -> Option<String> {
    markdown
        .lines()
        .find_map(|l| l.trim_start().strip_prefix('#'))
        .map(|h| h.trim_start_matches('#').trim().to_string())
        .filter(|h| !h.is_empty())
}

fn is_plain_text(content_type: &str, file_name: &str) -> bool {
    content_type.starts_with("text/") && content_type != "text/html"
        || matches!(
            Path::new(file_name).extension().and_then(|e| e.to_str()),
            Some("txt" | "md" | "json" | "csv")
        )
}

/// Fetches `url` and turns it into markdown for the chat.
#[tauri::command]
pub async fn url_extract(
    hypnos: State<'_, Hypnos>,
    url: String,
) -> Result<Attachment, ErrorPayload> {
    let fetched = fetch(&url).await?;
    let size = fetched.bytes.len() as u64;
    let html_title_text = (fetched.content_type == "text/html")
        .then(|| html_title(&String::from_utf8_lossy(&fetched.bytes)))
        .flatten();
    let markdown = if is_plain_text(&fetched.content_type, &fetched.file_name) {
        String::from_utf8_lossy(&fetched.bytes).into_owned()
    } else {
        let model = hypnos.pick("document_conversion", None).await?;
        let client = hypnos.client().await?;
        client
            .convert_document(&DocumentRequest::file(
                model,
                fetched.file_name.clone(),
                fetched.bytes,
            ))
            .await?
            .markdown
    };
    if markdown.trim().is_empty() {
        return Err(fail(format!(
            "No text could be extracted from {}.",
            fetched.final_url
        )));
    }
    let title = html_title_text
        .or_else(|| first_heading(&markdown))
        .unwrap_or_else(|| fetched.file_name.clone());
    let (text, truncated) = cap(markdown);
    Ok(Attachment::Web {
        name: title,
        url: fetched.final_url,
        size,
        text,
        truncated,
        content_type: fetched.content_type,
        file_name: fetched.file_name,
    })
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Saved {
    pub path: String,
    pub bytes: u64,
}

/// Writes extracted markdown to the file the user picked in the save dialog.
#[tauri::command]
pub async fn save_text(path: String, text: String) -> Result<Saved, ErrorPayload> {
    tokio::fs::write(&path, text.as_bytes())
        .await
        .map_err(|e| fail(format!("Could not save {path}: {e}")))?;
    Ok(Saved {
        path,
        bytes: text.len() as u64,
    })
}

/// Downloads the original file again and writes it where the user picked.
#[tauri::command]
pub async fn url_download(url: String, path: String) -> Result<Saved, ErrorPayload> {
    let fetched = fetch(&url).await?;
    tokio::fs::write(&path, &fetched.bytes)
        .await
        .map_err(|e| fail(format!("Could not save {path}: {e}")))?;
    Ok(Saved {
        path,
        bytes: fetched.bytes.len() as u64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn file_names_follow_the_url_then_the_content_type() {
        assert_eq!(
            file_name_for("https://a.org/papers/x.pdf", "application/pdf"),
            "x.pdf"
        );
        assert_eq!(file_name_for("https://a.org/", "text/html"), "index.html");
        assert_eq!(
            file_name_for("https://a.org/blog/post", "text/html"),
            "post.html"
        );
        assert_eq!(
            file_name_for("https://a.org/intro.php?x=1", "text/html"),
            "intro.html"
        );
        assert_eq!(
            file_name_for("https://a.org/my%20file.txt", "text/plain"),
            "my file.txt"
        );
        assert_eq!(
            file_name_for("https://a.org/", "application/octet-stream"),
            "download"
        );
    }

    #[test]
    fn titles_come_from_the_html_or_the_first_heading() {
        let html = "<html><head><TITLE>\n  Rust &amp; Tauri\n</TITLE></head></html>";
        assert_eq!(html_title(html).as_deref(), Some("Rust & Tauri"));
        assert_eq!(html_title("<p>none</p>"), None);
        assert_eq!(
            first_heading("intro\n## Getting started\n").as_deref(),
            Some("Getting started")
        );
    }

    #[test]
    fn only_http_links_are_read() {
        assert!(parse("file:///etc/passwd").is_err());
        assert!(parse("ftp://a.org/x").is_err());
        assert!(parse("https://a.org").is_ok());
    }

    /// Needs the network and a running server: `HYPNOS_TEST_URL=... cargo test web -- --ignored`.
    #[tokio::test(flavor = "multi_thread")]
    #[ignore]
    async fn reads_a_page_and_a_pdf_and_saves_them() {
        use crate::hypnos::hypnos_connect;
        use tauri::Manager;

        let url = std::env::var("HYPNOS_TEST_URL").expect("HYPNOS_TEST_URL");
        let app = tauri::test::mock_builder()
            .manage(Hypnos::default())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        hypnos_connect(app.state(), url).await.unwrap();

        let page = url_extract(app.state(), "https://example.com".into())
            .await
            .unwrap();
        let Attachment::Web {
            name,
            text,
            file_name,
            ..
        } = &page
        else {
            panic!("{page:?}")
        };
        assert_eq!(name, "Example Domain");
        assert_eq!(file_name, "index.html");
        assert!(text.contains("documentation examples"), "{text}");

        let pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf";
        let pdf = url_extract(app.state(), pdf_url.into()).await.unwrap();
        let Attachment::Web {
            text,
            file_name,
            content_type,
            ..
        } = &pdf
        else {
            panic!("{pdf:?}")
        };
        assert_eq!(file_name, "dummy.pdf");
        assert_eq!(content_type, "application/pdf");
        assert!(text.contains("Dummy PDF"), "{text}");

        let dir = std::env::temp_dir().join(format!("hypnos-web-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let original = dir.join("dummy.pdf").display().to_string();
        let saved = url_download(pdf_url.into(), original.clone())
            .await
            .unwrap();
        assert!(std::fs::read(&original).unwrap().starts_with(b"%PDF"));
        assert_eq!(saved.bytes, std::fs::metadata(&original).unwrap().len());
        let md = dir.join("page.md").display().to_string();
        save_text(md.clone(), text.clone()).await.unwrap();
        assert_eq!(std::fs::read_to_string(&md).unwrap(), *text);
        std::fs::remove_dir_all(dir).unwrap();

        let err = url_extract(
            app.state(),
            "https://example.com/surely-missing-page-404".into(),
        )
        .await
        .unwrap_err();
        assert!(err.message.contains("404"), "{err:?}");
    }
}
