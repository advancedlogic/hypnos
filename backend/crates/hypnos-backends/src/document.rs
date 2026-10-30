//! Document conversion with docling.rs (`docling` engine): PDFs, images and office / web formats
//! to Markdown, on the CPU. The layout, TableFormer and OCR models run on ONNX Runtime; PDFs are
//! parsed and rendered in pure Rust.
//!
//! A PDF is converted one page per pass through a warm pipeline (models loaded once per
//! residency): converting a whole document in one call keeps every page's intermediate data until
//! the end (a 6-page PDF peaked at 12.7 GB that way, 3.8 GB page by page), so a fixed booking of
//! the model's working memory holds for documents of any length. [`join_pages`] puts back
//! together what the page split cuts: tables, numbered lists and paragraphs across a page break.

use async_trait::async_trait;
use docling::{DocumentConverter, InputFormat, SourceDocument, WarmPipeline};
use hypnos_core::config::{MemoryTier, ModelConfig};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::kv_cache::LayerPlacementPlan;
use hypnos_core::openai::{
    DocumentConversionRequest, DocumentPage, DocumentStreamEvent, EngineTiming, ExtractedPicture,
    PageSize,
};
use hypnos_core::request::RequestContext;
use hypnos_core::ModelBackend;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use tokio::sync::mpsc;
use tracing::{info, warn};

pub struct DoclingBackend {
    config: ModelConfig,
    /// The loaded models while resident (`WarmPipeline` is a shared handle: conversions take
    /// turns on it).
    pipeline: Mutex<Option<WarmPipeline>>,
    resident: AtomicBool,
}

impl DoclingBackend {
    pub fn new(config: ModelConfig) -> Self {
        Self {
            config,
            pipeline: Mutex::new(None),
            resident: AtomicBool::new(false),
        }
    }
}

/// Sets `name` unless the operator already did (every docling knob stays overridable).
fn set_default_env(name: &str, value: &str) {
    if std::env::var_os(name).is_none() {
        std::env::set_var(name, value);
    }
}

/// Points docling.rs at its models and execution provider, which it reads from the process
/// environment: `$DOCLING_RS_MODELS_DIR` (the release layout: `layout_heron_int8.onnx`,
/// `tableformer/`, `ocr_*.onnx`) is `weights_path`'s folder, and `DOCLING_RS_EP` follows
/// `device`. Set before the first model loads, while no conversion runs; docling resolves the
/// provider once per process, which is why validation allows one docling model per server.
fn configure_docling(cfg: &ModelConfig) -> Result<PathBuf> {
    let dir = cfg
        .weights_path
        .parent()
        .filter(|d| !d.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let dir = dir.canonicalize().map_err(|e| {
        HypnosError::BackendError(format!("docling models directory {}: {e}", dir.display()))
    })?;
    if std::env::var_os("DOCLING_RS_MODELS_DIR").as_deref() != Some(dir.as_os_str()) {
        std::env::set_var("DOCLING_RS_MODELS_DIR", &dir);
    }
    // ONNX Runtime's CPU arena keeps a slab for every tensor shape it has seen and never gives it
    // back, so a server's memory would ratchet up with the largest page ever converted.
    set_default_env("DOCLING_RS_NO_ARENA", "1");
    if cfg.runs_on_cpu() {
        set_default_env("DOCLING_RS_EP", "cpu");
        return Ok(dir);
    }
    let provider = crate::docling_gpu_provider().ok_or_else(|| {
        HypnosError::Config(format!(
            "'{}': device = \"gpu\" needs a build with a docling GPU provider (Metal, or the docling-cuda feature)",
            cfg.id
        ))
    })?;
    // The GPU providers run the fp32 graphs (the int8 ones are calibrated for CPU kernels).
    if !dir.join("layout_heron.onnx").is_file() {
        return Err(HypnosError::Config(format!(
            "'{}': device = \"gpu\" needs the fp32 layout model {} (./run.sh models docling-gpu)",
            cfg.id,
            dir.join("layout_heron.onnx").display()
        )));
    }
    set_default_env("DOCLING_RS_EP", provider);
    if provider == "cuda" {
        // About ten sessions share the booked workspace: cap each arena so the device stays
        // within the booking. The arena keeps ONNX Runtime's power-of-two growth: growing by
        // exactly what is asked made the TableFormer decoder, whose KV cache grows every step,
        // allocate on every step (1270 instead of 1070 ms per two-table page on an RTX 5090
        // laptop, see docs/bench/docling-cuda-2026-10-30.md).
        let per_session_mb = (cfg.arch.activation_workspace_bytes / 4 / (1024 * 1024)).max(256);
        set_default_env("DOCLING_RS_CUDA_MEM_LIMIT_MB", &per_session_mb.to_string());
        set_default_env("DOCLING_RS_CUDA_ARENA_EXTEND", "next_power_of_two");
    }
    Ok(dir)
}

/// Builds the pipeline by converting a small blank image, so the layout and OCR models load with
/// the placement (where the tier switch is timed and a missing file is reported) rather than on
/// the first request. TableFormer still loads on the first table.
fn warm_up(pipeline: &WarmPipeline) -> Result<()> {
    let mut png = Vec::new();
    image::RgbImage::from_pixel(256, 256, image::Rgb([255, 255, 255]))
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .map_err(|e| HypnosError::BackendError(format!("warm-up image: {e}")))?;
    DocumentConverter::new()
        .warm_pipeline(pipeline)
        .convert(SourceDocument::from_bytes(
            "warm-up.png",
            InputFormat::Image,
            png,
        ))
        .map(|_| ())
        .map_err(|e| HypnosError::BackendError(format!("docling could not load its models: {e}")))
}

/// The document's format, from the file name's extension or else the content.
fn detect_format(filename: Option<&str>, bytes: &[u8]) -> Option<InputFormat> {
    filename
        .and_then(|n| Path::new(n).extension())
        .and_then(|e| InputFormat::from_extension(&e.to_string_lossy()))
        .or_else(|| InputFormat::detect(bytes))
}

/// Parses `"3"`, `"2-5"` or `"2-"` (1-based, inclusive) against a document of `count` pages.
fn parse_pages(spec: Option<&str>, count: u32) -> Result<Vec<u32>> {
    let Some(spec) = spec.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok((1..=count).collect());
    };
    let bad = || {
        HypnosError::InvalidRequest(format!(
            "pages must be a page or a range like \"2-5\" within 1-{count}, got {spec:?}"
        ))
    };
    let (first, last) = match spec.split_once('-') {
        Some((a, b)) => {
            let first: u32 = a.trim().parse().map_err(|_| bad())?;
            let last = if b.trim().is_empty() {
                count
            } else {
                b.trim().parse().map_err(|_| bad())?
            };
            (first, last)
        }
        None => {
            let p: u32 = spec.parse().map_err(|_| bad())?;
            (p, p)
        }
    };
    if first == 0 || first > last || last > count {
        return Err(bad());
    }
    Ok((first..=last).collect())
}

/// ZapfDingbats bullets docling cannot map to Unicode come out as their glyph name at the start of
/// a list item (`- /a136 Pair-weighted error`); the list marker already says it is a bullet.
fn strip_glyph_names(markdown: &str) -> String {
    let clean = |line: &str| -> String {
        let indent = line.len() - line.trim_start().len();
        let rest = &line[indent..];
        let marker_len =
            if rest.starts_with("- ") || rest.starts_with("* ") || rest.starts_with("+ ") {
                2
            } else {
                let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
                if digits > 0 && rest[digits..].starts_with(". ") {
                    digits + 2
                } else {
                    0
                }
            };
        let body = &rest[marker_len..];
        let glyph = body.strip_prefix("/a").and_then(|g| {
            let digits = g.bytes().take_while(u8::is_ascii_digit).count();
            (1..=3)
                .contains(&digits)
                .then(|| &g[digits..])
                .and_then(|t| t.strip_prefix(' ').or_else(|| t.strip_prefix('\t')))
        });
        match glyph {
            Some(text) => format!("{}{}", &line[..indent + marker_len], text),
            None => line.to_string(),
        }
    };
    markdown.lines().map(clean).collect::<Vec<_>>().join("\n")
}

/// Removes the embedded page and picture images from a page's docling JSON (they are megabytes of
/// base64), keeping each picture's data URI and metadata (size, position, caption) aside when
/// `keep` asks for them.
fn split_off_images(json: &mut serde_json::Value, keep: bool) -> Vec<ExtractedPicture> {
    let mut images = Vec::new();
    let page_sizes: HashMap<u64, PageSize> = json
        .get("pages")
        .and_then(|p| p.as_object())
        .map(|pages| {
            pages
                .values()
                .filter_map(|page| {
                    let no = page.get("page_no")?.as_u64()?;
                    let size = page.get("size")?;
                    Some((
                        no,
                        PageSize {
                            width: size.get("width")?.as_f64()?,
                            height: size.get("height")?.as_f64()?,
                        },
                    ))
                })
                .collect()
        })
        .unwrap_or_default();
    let caption_of = |pic: &serde_json::Value| -> Option<String> {
        let reference = pic
            .get("captions")?
            .as_array()?
            .first()?
            .get("$ref")?
            .as_str()?;
        let index: usize = reference.strip_prefix("#/texts/")?.parse().ok()?;
        let text = json
            .get("texts")?
            .as_array()?
            .get(index)?
            .get("text")?
            .as_str()?;
        Some(text.trim().to_string()).filter(|t| !t.is_empty())
    };
    let pictures: Vec<ExtractedPicture> = json
        .get("pictures")
        .and_then(|p| p.as_array())
        .map(|pictures| {
            pictures
                .iter()
                .filter_map(|pic| {
                    let image = pic.get("image")?;
                    let prov = pic.get("prov").and_then(|p| p.as_array()?.first());
                    let page_no = prov.and_then(|p| p.get("page_no")?.as_u64());
                    Some(ExtractedPicture {
                        data_uri: image.get("uri")?.as_str()?.to_string(),
                        mime: image
                            .get("mimetype")
                            .and_then(|m| m.as_str())
                            .unwrap_or("image/png")
                            .to_string(),
                        width: image
                            .pointer("/size/width")
                            .and_then(|v| v.as_f64())
                            .map(|v| v as u32),
                        height: image
                            .pointer("/size/height")
                            .and_then(|v| v.as_f64())
                            .map(|v| v as u32),
                        dpi: image.get("dpi").and_then(|v| v.as_u64()).map(|v| v as u32),
                        bbox: prov
                            .and_then(|p| p.get("bbox"))
                            .and_then(|b| serde_json::from_value(b.clone()).ok()),
                        page_size: page_no.and_then(|n| page_sizes.get(&n).cloned()),
                        caption: caption_of(pic),
                        self_ref: pic
                            .get("self_ref")
                            .and_then(|r| r.as_str())
                            .map(str::to_string),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    if keep {
        images = pictures;
    }
    if let Some(pages) = json.get_mut("pages").and_then(|p| p.as_object_mut()) {
        for page in pages.values_mut() {
            if let Some(p) = page.as_object_mut() {
                p.remove("image");
            }
        }
    }
    if let Some(pictures) = json.get_mut("pictures").and_then(|p| p.as_array_mut()) {
        for pic in pictures {
            if let Some(p) = pic.as_object_mut() {
                p.remove("image");
            }
        }
    }
    images
}

fn count(json: &serde_json::Value, key: &str) -> u32 {
    json.get(key)
        .and_then(|v| v.as_array())
        .map_or(0, |a| a.len() as u32)
}

struct PageJob<'a> {
    pipeline: &'a WarmPipeline,
    format: InputFormat,
    name: &'a str,
    bytes: &'a [u8],
    ocr: bool,
    keep_pictures: bool,
    keep_json: bool,
}

/// Converts one page (`Some(p)`) of a paginated document, or a whole document without pages.
fn convert_page(job: &PageJob<'_>, page: Option<u32>) -> Result<DocumentPage> {
    let started = Instant::now();
    let mut converter = DocumentConverter::new()
        .warm_pipeline(job.pipeline)
        .no_ocr(!job.ocr)
        .keep_picture_images(job.keep_pictures);
    if let Some(p) = page {
        converter = converter.page_range(p as usize, p as usize);
    }
    let source = SourceDocument::from_bytes(job.name, job.format, job.bytes.to_vec());
    let result = converter
        .convert(source)
        .map_err(|e| HypnosError::BackendError(format!("docling: {e}")))?;
    for e in &result.errors {
        warn!(page = ?page, module = %e.module_name, error = %e.error_message, "docling reported a problem");
    }
    if result.status == docling::ConversionStatus::Failure {
        return Err(HypnosError::BackendError(format!(
            "docling could not convert {}{}",
            page.map(|p| format!("page {p} of ")).unwrap_or_default(),
            job.name
        )));
    }
    let markdown = strip_glyph_names(result.document.export_to_markdown().trim());
    let mut json = result.document.export_to_json_value();
    let picture_images = split_off_images(&mut json, job.keep_pictures);
    Ok(DocumentPage {
        page: page.unwrap_or(1),
        markdown,
        pictures: count(&json, "pictures"),
        tables: count(&json, "tables"),
        json: job.keep_json.then_some(json),
        ms: started.elapsed().as_secs_f64() * 1000.0,
        picture_urls: Vec::new(),
        picture_images,
    })
}

#[async_trait]
impl ModelBackend for DoclingBackend {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn is_resident(&self) -> bool {
        self.resident.load(Ordering::SeqCst)
    }

    async fn apply_placement(
        &self,
        plan: &LayerPlacementPlan,
        _pinned_host_ptr: Option<usize>,
    ) -> Result<()> {
        let want_resident = plan.primary_tier == MemoryTier::Tier1GpuVram;
        if !want_resident {
            // Dropping the last handle frees the models.
            if self.pipeline.lock().take().is_some() {
                info!(model = %self.config.id, "docling models released");
            }
            self.resident.store(false, Ordering::SeqCst);
            return Ok(());
        }
        if self.pipeline.lock().is_some() {
            return Ok(());
        }
        let cfg = self.config.clone();
        let id = self.config.id.clone();
        let pipeline = tokio::task::spawn_blocking(move || -> Result<WarmPipeline> {
            let started = Instant::now();
            let dir = configure_docling(&cfg)?;
            let pipeline = WarmPipeline::new();
            warm_up(&pipeline)?;
            info!(
                model = %id,
                models_dir = %dir.display(),
                provider = %std::env::var("DOCLING_RS_EP").unwrap_or_default(),
                load_secs = format!("{:.2}", started.elapsed().as_secs_f32()),
                "docling models resident"
            );
            Ok(pipeline)
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("docling load task: {e}")))??;
        *self.pipeline.lock() = Some(pipeline);
        self.resident.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn convert_document(
        &self,
        req: &DocumentConversionRequest,
        ctx: &RequestContext,
        tx: mpsc::Sender<DocumentStreamEvent>,
    ) -> Result<()> {
        let pipeline = self.pipeline.lock().clone().ok_or_else(|| {
            HypnosError::BackendError(format!("model '{}' is not loaded", self.config.id))
        })?;
        if req.file_bytes.is_empty() {
            return Err(HypnosError::InvalidRequest("the document is empty".into()));
        }
        let format = detect_format(req.filename.as_deref(), &req.file_bytes).ok_or_else(|| {
            HypnosError::InvalidRequest(
                "unrecognised document format (PDF, images, DOCX, PPTX, XLSX, HTML, Markdown, ... are supported)"
                    .into(),
            )
        })?;
        let req = req.clone();
        let ctx = ctx.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            let name = req
                .filename
                .clone()
                .unwrap_or_else(|| "document".to_string());
            let job = PageJob {
                pipeline: &pipeline,
                format,
                name: &name,
                bytes: &req.file_bytes,
                ocr: req.ocr,
                keep_pictures: req.describe_pictures.is_some() || req.return_pictures,
                keep_json: req.wants_json(),
            };
            let paginated = format == InputFormat::Pdf;
            let page_count = if paginated {
                docling_pdf::page_count(&req.file_bytes, None)
                    .map_err(|e| HypnosError::InvalidRequest(format!("unreadable PDF: {e}")))?
                    as u32
            } else {
                1
            };
            let pages = if paginated {
                parse_pages(req.pages.as_deref(), page_count)?
            } else {
                vec![1]
            };
            let _ = tx.blocking_send(DocumentStreamEvent::Started {
                page_count,
                pages: pages.clone(),
            });
            let started = Instant::now();
            for &p in &pages {
                if !ctx.should_continue() {
                    return Err(HypnosError::Cancelled);
                }
                let page = convert_page(&job, paginated.then_some(p))?;
                if tx.blocking_send(DocumentStreamEvent::Page(page)).is_err() {
                    return Err(HypnosError::Cancelled);
                }
            }
            let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
            let _ = tx.blocking_send(DocumentStreamEvent::Done {
                timing: EngineTiming {
                    pages: Some(pages.len() as u32),
                    ms_per_page: Some(elapsed_ms / pages.len().max(1) as f64),
                    ..Default::default()
                },
            });
            Ok(())
        })
        .await
        .map_err(|e| HypnosError::BackendError(format!("docling conversion task: {e}")))?
    }
}

// ----------------------------------------------------------------------------------------------
// Joining pages
// ----------------------------------------------------------------------------------------------

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.ends_with('|') && t.len() > 1
}

fn is_table_separator(line: &str) -> bool {
    is_table_row(line)
        && line
            .trim()
            .chars()
            .all(|c| matches!(c, '|' | '-' | ':' | ' '))
}

fn table_columns(line: &str) -> usize {
    line.trim().trim_matches('|').split('|').count()
}

fn table_cells(line: &str) -> Vec<String> {
    line.trim()
        .trim_matches('|')
        .split('|')
        .map(|c| c.trim().to_string())
        .collect()
}

/// `Some(n)` for an ordered list item `n. text`.
fn list_number(line: &str) -> Option<u32> {
    let t = line.trim_start();
    let digits = t.bytes().take_while(u8::is_ascii_digit).count();
    if digits == 0 || !t[digits..].starts_with(". ") {
        return None;
    }
    t[..digits].parse().ok()
}

/// A line of running text: not a heading, list item, table row, quote, code fence or comment.
fn is_prose(line: &str) -> bool {
    let t = line.trim_start();
    !t.is_empty()
        && !t.starts_with('#')
        && !t.starts_with('|')
        && !t.starts_with('>')
        && !t.starts_with("```")
        && !t.starts_with("<!--")
        && !t.starts_with("- ")
        && !t.starts_with("* ")
        && !t.starts_with('$')
        && list_number(t).is_none()
}

/// Joins two consecutive pages' Markdown, mending what the page break cut.
fn join_two(prev: &str, next: &str) -> String {
    let (prev, next) = (prev.trim_end(), next.trim_start());
    if prev.is_empty() {
        return next.to_string();
    }
    if next.is_empty() {
        return prev.to_string();
    }
    let last = prev.lines().last().unwrap_or("");
    let mut next_lines: Vec<&str> = next.lines().collect();
    let first = next_lines[0];

    // A table that runs on: same column count. Docling makes the continuation's first row a
    // header; drop it when it repeats the original header, and drop its separator either way.
    if is_table_row(last) && is_table_row(first) && table_columns(last) == table_columns(first) {
        let header = prev
            .lines()
            .rev()
            .take_while(|l| is_table_row(l))
            .last()
            .unwrap_or(last);
        let mut skip = 0;
        if next_lines.len() > 1 && is_table_separator(next_lines[1]) {
            if table_cells(first) == table_cells(header) {
                skip = 2;
            } else {
                next_lines.remove(1);
            }
        }
        return format!("{prev}\n{}", next_lines[skip..].join("\n"));
    }

    // A numbered list that runs on but restarts at 1 on the new page.
    if let (Some(n), Some(1)) = (list_number(last), list_number(first)) {
        let mut expected = 1;
        let mut renumbered = Vec::with_capacity(next_lines.len());
        let mut in_list = true;
        for line in &next_lines {
            match list_number(line) {
                Some(k) if in_list && k == expected && !line.starts_with(' ') => {
                    let rest = &line.trim_start()[k.to_string().len()..];
                    renumbered.push(format!("{}{rest}", n + k));
                    expected += 1;
                }
                _ => {
                    // Continuation lines of an item stay; a blank line ends the list.
                    if line.trim().is_empty() {
                        in_list = false;
                    }
                    renumbered.push(line.to_string());
                }
            }
        }
        return format!("{prev}\n{}", renumbered.join("\n"));
    }

    // A sentence cut by the page break: the page ends mid-sentence and the next starts lowercase.
    let ends_sentence = last
        .trim_end()
        .ends_with(['.', '!', '?', ':', ';', ')', '"', '\u{201d}', '\u{2019}']);
    if is_prose(last)
        && is_prose(first)
        && !ends_sentence
        && first.trim_start().starts_with(|c: char| c.is_lowercase())
    {
        return format!("{prev} {}", next.trim_start());
    }
    format!("{prev}\n\n{next}")
}

/// The converted pages as one Markdown document, in page order. Pages that do not follow each
/// other (a range with gaps never happens today, but a failed page could) are joined plainly.
pub fn join_pages(pages: &[DocumentPage]) -> String {
    let mut out = String::new();
    let mut prev_page: Option<u32> = None;
    for page in pages {
        out = match prev_page {
            Some(p) if p + 1 == page.page => join_two(&out, &page.markdown),
            Some(_) => format!("{}\n\n{}", out.trim_end(), page.markdown.trim_start()),
            None => page.markdown.clone(),
        };
        prev_page = Some(page.page);
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_off_images_keeps_the_pictures_with_their_metadata() {
        let mut json = serde_json::json!({
            "pages": { "1": { "page_no": 1, "size": { "width": 612.0, "height": 792.0 }, "image": { "uri": "data:image/png;base64,PAGE" } } },
            "texts": [ { "text": "Figure 1: a chart " } ],
            "pictures": [ {
                "self_ref": "#/pictures/0",
                "prov": [ { "page_no": 1, "bbox": { "l": 10.0, "t": 20.0, "r": 110.0, "b": 220.0, "coord_origin": "TOPLEFT" }, "charspan": [0, 0] } ],
                "captions": [ { "$ref": "#/texts/0" } ],
                "image": { "mimetype": "image/png", "dpi": 144, "size": { "width": 200.0, "height": 400.0 }, "uri": "data:image/png;base64,PIC" }
            } ]
        });
        let kept = split_off_images(&mut json.clone(), true);
        assert_eq!(kept.len(), 1);
        let pic = &kept[0];
        assert_eq!(pic.data_uri, "data:image/png;base64,PIC");
        assert_eq!(pic.mime, "image/png");
        assert_eq!(
            (pic.width, pic.height, pic.dpi),
            (Some(200), Some(400), Some(144))
        );
        assert_eq!(
            pic.bbox
                .as_ref()
                .map(|b| (b.l, b.b, b.coord_origin.as_str())),
            Some((10.0, 220.0, "TOPLEFT"))
        );
        assert_eq!(
            pic.page_size,
            Some(PageSize {
                width: 612.0,
                height: 792.0
            })
        );
        assert_eq!(pic.caption.as_deref(), Some("Figure 1: a chart"));
        assert_eq!(pic.self_ref.as_deref(), Some("#/pictures/0"));

        // Dropped or kept, the images leave the JSON.
        assert!(split_off_images(&mut json, false).is_empty());
        assert!(json["pages"]["1"].get("image").is_none());
        assert!(json["pictures"][0].get("image").is_none());
        assert_eq!(json["pictures"][0]["self_ref"], "#/pictures/0");
    }

    fn page(n: u32, md: &str) -> DocumentPage {
        DocumentPage {
            page: n,
            markdown: md.to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn page_ranges() {
        assert_eq!(parse_pages(None, 3).unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_pages(Some("2"), 3).unwrap(), vec![2]);
        assert_eq!(parse_pages(Some(" 2-3 "), 5).unwrap(), vec![2, 3]);
        assert_eq!(parse_pages(Some("4-"), 5).unwrap(), vec![4, 5]);
        assert!(parse_pages(Some("0"), 3).is_err());
        assert!(parse_pages(Some("3-2"), 3).is_err());
        assert!(parse_pages(Some("2-9"), 3).is_err());
        assert!(parse_pages(Some("x"), 3).is_err());
    }

    #[test]
    fn glyph_names_are_dropped_from_list_items() {
        let md = "- /a136 Pair-Weighted Error\n1. /a12 first\nkeep /a136 inside text";
        assert_eq!(
            strip_glyph_names(md),
            "- Pair-Weighted Error\n1. first\nkeep /a136 inside text"
        );
    }

    #[test]
    fn continued_table_drops_the_repeated_header() {
        let p1 = "Intro\n\n| A | B |\n|---|---|\n| 1 | 2 |";
        let p2 = "| A | B |\n|---|---|\n| 3 | 4 |\n\nAfter";
        assert_eq!(
            join_pages(&[page(1, p1), page(2, p2)]),
            "Intro\n\n| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n\nAfter"
        );
    }

    #[test]
    fn continued_table_keeps_a_data_row_taken_for_a_header() {
        let p1 = "| A | B |\n|---|---|\n| 1 | 2 |";
        let p2 = "| 3 | 4 |\n|---|---|\n| 5 | 6 |";
        assert_eq!(
            join_pages(&[page(1, p1), page(2, p2)]),
            "| A | B |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |\n| 5 | 6 |"
        );
    }

    #[test]
    fn tables_with_other_columns_stay_apart() {
        let p1 = "| A | B |\n|---|---|\n| 1 | 2 |";
        let p2 = "| X | Y | Z |\n|---|---|---|\n| 1 | 2 | 3 |";
        assert_eq!(
            join_pages(&[page(1, p1), page(2, p2)]),
            format!("{p1}\n\n{p2}")
        );
    }

    #[test]
    fn numbered_list_continues_across_the_break() {
        let p1 = "Steps\n\n1. one\n2. two";
        let p2 = "1. three\n2. four\n\nDone";
        assert_eq!(
            join_pages(&[page(1, p1), page(2, p2)]),
            "Steps\n\n1. one\n2. two\n3. three\n4. four\n\nDone"
        );
    }

    #[test]
    fn sentence_cut_by_the_page_break_is_rejoined() {
        let p1 = "The system predicts a discrete origin count and";
        let p2 = "the error is not zero.\n\n## Next";
        assert_eq!(
            join_pages(&[page(1, p1), page(2, p2)]),
            "The system predicts a discrete origin count and the error is not zero.\n\n## Next"
        );
        // A finished sentence or a capitalised start is a new paragraph.
        let done = "It ends here.";
        assert_eq!(
            join_pages(&[page(1, done), page(2, "the next one")]),
            "It ends here.\n\nthe next one"
        );
        assert_eq!(
            join_pages(&[page(1, "no stop"), page(2, "New paragraph")]),
            "no stop\n\nNew paragraph"
        );
    }

    #[test]
    fn non_consecutive_pages_are_joined_plainly() {
        assert_eq!(
            join_pages(&[page(1, "cut and"), page(3, "more")]),
            "cut and\n\nmore"
        );
    }
}
