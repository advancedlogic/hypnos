//! The picture store behind `return_pictures` on `POST /v1/documents/convert`.
//!
//! A conversion that asks for its pictures gets a document id; every picture docling cut out of
//! its pages is written as one JSON file (`<n>.json`: the base64 image plus mime type, pixel
//! size, page, bounding box and caption) under `<dir>/<id>/`. Clients fetch them afterwards, one
//! by one or as a list, from `GET /v1/documents/{id}/pictures[/{n}]` and release them with
//! `DELETE /v1/documents/{id}`. What is not deleted expires `ttl` after the document's last page,
//! and the oldest documents go first when the store crosses `max_bytes`. The index lives in
//! memory, so files from an earlier run are orphans: they are removed at startup.

use base64::Engine;
use chrono::{DateTime, Utc};
use hypnos_core::error::{HypnosError, Result};
use hypnos_core::openai::{ExtractedPicture, PageSize, PictureBox};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, SystemTime},
};
use tracing::{info, warn};
use uuid::Uuid;

/// One stored picture: its metadata, and the base64 image when read back from its file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredPicture {
    /// Always `"document.picture"`.
    pub object: String,
    pub document_id: String,
    /// 1-based position in the document, in Markdown order across pages.
    pub index: u32,
    /// 1-based page the picture was cut from.
    pub page: u32,
    /// `/v1/documents/{id}/pictures/{index}`.
    pub url: String,
    pub mime: String,
    /// Size of the encoded image.
    pub bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub width: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub height: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
    /// Where it sits on the page (page units, PDF points for PDFs).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bbox: Option<PictureBox>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub page_size: Option<PageSize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caption: Option<String>,
    /// The picture's reference in the docling document (`#/pictures/0`).
    #[serde(rename = "ref", skip_serializing_if = "Option::is_none")]
    pub self_ref: Option<String>,
    /// The image, base64 (standard alphabet); absent in listings.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
}

/// `GET /v1/documents/{id}/pictures`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PictureList {
    /// Always `"list"`.
    pub object: String,
    pub document_id: String,
    pub expires_at: String,
    pub data: Vec<StoredPicture>,
}

struct StoredDocument {
    created: SystemTime,
    expires: SystemTime,
    bytes: u64,
    pictures: Vec<StoredPicture>,
}

#[derive(Default)]
struct Inner {
    docs: HashMap<String, StoredDocument>,
    total_bytes: u64,
}

pub struct PictureStore {
    dir: PathBuf,
    ttl: Duration,
    max_bytes: u64,
    inner: Mutex<Inner>,
}

fn is_document_id(id: &str) -> bool {
    id.strip_prefix("doc-")
        .map(|rest| rest.len() == 32 && rest.bytes().all(|b| b.is_ascii_hexdigit()))
        .unwrap_or(false)
}

fn rfc3339(t: SystemTime) -> String {
    DateTime::<Utc>::from(t).to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
}

/// Splits `data:<mime>;base64,<payload>` into its media type and decoded bytes. A bare base64
/// string decodes with `fallback_mime`.
fn decode_data_uri(uri: &str, fallback_mime: &str) -> Result<(String, Vec<u8>)> {
    let (mime, payload) = match uri.strip_prefix("data:") {
        Some(rest) => {
            let (head, payload) = rest.split_once(',').ok_or_else(|| {
                HypnosError::BackendError("picture data URI has no payload".into())
            })?;
            let mime = head.split(';').next().unwrap_or("").trim();
            let mime = if mime.is_empty() { fallback_mime } else { mime };
            (mime.to_string(), payload)
        }
        None => (fallback_mime.to_string(), uri),
    };
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|e| HypnosError::BackendError(format!("picture base64: {e}")))?;
    Ok((mime, bytes))
}

impl PictureStore {
    /// Opens the store in `dir` (default: `hypnos-pictures` under the OS temporary directory),
    /// creating it and removing documents left behind by an earlier run.
    pub fn open(dir: Option<&Path>, ttl: Duration, max_bytes: u64) -> std::io::Result<Self> {
        let dir = dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::temp_dir().join("hypnos-pictures"));
        std::fs::create_dir_all(&dir)?;
        let mut stale = 0;
        for entry in std::fs::read_dir(&dir)?.flatten() {
            let name = entry.file_name();
            if is_document_id(&name.to_string_lossy()) && entry.path().is_dir() {
                if std::fs::remove_dir_all(entry.path()).is_ok() {
                    stale += 1;
                }
            }
        }
        if stale > 0 {
            info!(dir = %dir.display(), documents = stale, "removed stale picture documents");
        }
        Ok(Self {
            dir,
            ttl,
            max_bytes,
            inner: Mutex::new(Inner::default()),
        })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Registers a new, empty document and returns its id.
    pub fn new_document(&self) -> String {
        let id = format!("doc-{}", Uuid::new_v4().simple());
        let now = SystemTime::now();
        self.inner.lock().docs.insert(
            id.clone(),
            StoredDocument {
                created: now,
                expires: now + self.ttl,
                bytes: 0,
                pictures: Vec::new(),
            },
        );
        id
    }

    /// Writes a page's pictures and returns their URLs, in the order given. Resets the
    /// document's expiry and, when the store is over its cap, drops the oldest other documents.
    pub fn add_page(
        &self,
        id: &str,
        page: u32,
        pictures: &[ExtractedPicture],
    ) -> Result<Vec<String>> {
        if pictures.is_empty() {
            return Ok(Vec::new());
        }
        let doc_dir = self.dir.join(id);
        std::fs::create_dir_all(&doc_dir)
            .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
        let first_index = {
            let inner = self.inner.lock();
            let doc = inner
                .docs
                .get(id)
                .ok_or_else(|| HypnosError::BackendError(format!("unknown document {id}")))?;
            doc.pictures.len() as u32 + 1
        };
        let mut written = Vec::with_capacity(pictures.len());
        let mut added_bytes = 0;
        for (i, pic) in pictures.iter().enumerate() {
            let index = first_index + i as u32;
            let (mime, bytes) = decode_data_uri(&pic.data_uri, &pic.mime)?;
            let record = StoredPicture {
                object: "document.picture".into(),
                document_id: id.to_string(),
                index,
                page,
                url: format!("/v1/documents/{id}/pictures/{index}"),
                mime,
                bytes: bytes.len() as u64,
                width: pic.width,
                height: pic.height,
                dpi: pic.dpi,
                bbox: pic.bbox.clone(),
                page_size: pic.page_size.clone(),
                caption: pic.caption.clone(),
                self_ref: pic.self_ref.clone(),
                image: Some(base64::engine::general_purpose::STANDARD.encode(&bytes)),
            };
            let body = serde_json::to_vec(&record)
                .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
            std::fs::write(doc_dir.join(format!("{index}.json")), &body)
                .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
            added_bytes += body.len() as u64;
            written.push(StoredPicture {
                image: None,
                ..record
            });
        }
        let urls = written.iter().map(|p| p.url.clone()).collect();
        let evict = {
            let mut inner = self.inner.lock();
            let Some(doc) = inner.docs.get_mut(id) else {
                // Deleted while the page was being written: do not leave the files behind.
                drop(inner);
                let _ = std::fs::remove_dir_all(&doc_dir);
                return Err(HypnosError::BackendError(format!(
                    "document {id} was deleted"
                )));
            };
            doc.pictures.extend(written);
            doc.bytes += added_bytes;
            doc.expires = SystemTime::now() + self.ttl;
            inner.total_bytes += added_bytes;
            self.over_cap(&mut inner, Some(id))
        };
        self.remove_dirs(&evict, "over document_pictures_max_bytes");
        Ok(urls)
    }

    /// Ids of the oldest documents (never `keep`) that must go for the store to fit its cap.
    fn over_cap(&self, inner: &mut Inner, keep: Option<&str>) -> Vec<String> {
        let mut evicted = Vec::new();
        if self.max_bytes == 0 {
            return evicted;
        }
        while inner.total_bytes > self.max_bytes {
            let oldest = inner
                .docs
                .iter()
                .filter(|(id, _)| Some(id.as_str()) != keep)
                .min_by_key(|(_, d)| d.created)
                .map(|(id, _)| id.clone());
            let Some(id) = oldest else { break };
            if let Some(doc) = inner.docs.remove(&id) {
                inner.total_bytes -= doc.bytes;
            }
            evicted.push(id);
        }
        evicted
    }

    fn remove_dirs(&self, ids: &[String], reason: &str) {
        for id in ids {
            match std::fs::remove_dir_all(self.dir.join(id)) {
                Ok(()) => info!(document = %id, reason, "pictures released"),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => warn!(document = %id, error = %e, "could not remove pictures"),
            }
        }
    }

    pub fn expires_at(&self, id: &str) -> Option<String> {
        self.inner.lock().docs.get(id).map(|d| rfc3339(d.expires))
    }

    /// The document's pictures, metadata only.
    pub fn list(&self, id: &str) -> Option<PictureList> {
        let inner = self.inner.lock();
        let doc = inner.docs.get(id)?;
        Some(PictureList {
            object: "list".into(),
            document_id: id.to_string(),
            expires_at: rfc3339(doc.expires),
            data: doc.pictures.clone(),
        })
    }

    /// One picture with its image, read back from its file.
    pub fn get(&self, id: &str, index: u32) -> Result<Option<StoredPicture>> {
        let known = {
            let inner = self.inner.lock();
            inner
                .docs
                .get(id)
                .map(|d| d.pictures.iter().any(|p| p.index == index))
                .unwrap_or(false)
        };
        if !known {
            return Ok(None);
        }
        let path = self.dir.join(id).join(format!("{index}.json"));
        let body = std::fs::read(&path)
            .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
        let record: StoredPicture = serde_json::from_slice(&body)
            .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
        Ok(Some(record))
    }

    /// The picture's decoded bytes with their media type.
    pub fn get_bytes(&self, id: &str, index: u32) -> Result<Option<(String, Vec<u8>)>> {
        let Some(record) = self.get(id, index)? else {
            return Ok(None);
        };
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(record.image.unwrap_or_default())
            .map_err(|e| HypnosError::BackendError(format!("picture store: {e}")))?;
        Ok(Some((record.mime, bytes)))
    }

    /// Drops the document and its files. `false` when it was not there.
    pub fn delete(&self, id: &str) -> bool {
        let removed = {
            let mut inner = self.inner.lock();
            let doc = inner.docs.remove(id);
            if let Some(doc) = &doc {
                inner.total_bytes -= doc.bytes;
            }
            doc.is_some()
        };
        if removed {
            self.remove_dirs(&[id.to_string()], "deleted");
        }
        removed
    }

    /// Removes the documents whose time is up. Returns how many went.
    pub fn sweep(&self) -> usize {
        let expired: Vec<String> = {
            let mut inner = self.inner.lock();
            let now = SystemTime::now();
            let expired: Vec<String> = inner
                .docs
                .iter()
                .filter(|(_, d)| d.expires <= now)
                .map(|(id, _)| id.clone())
                .collect();
            for id in &expired {
                if let Some(doc) = inner.docs.remove(id) {
                    inner.total_bytes -= doc.bytes;
                }
            }
            expired
        };
        self.remove_dirs(&expired, "expired");
        expired.len()
    }

    /// Documents held and bytes on disk.
    pub fn stats(&self) -> (usize, u64) {
        let inner = self.inner.lock();
        (inner.docs.len(), inner.total_bytes)
    }

    /// Runs `sweep` every minute for as long as the runtime lives.
    pub fn spawn_sweeper(self: &Arc<Self>) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(Duration::from_secs(60));
            tick.tick().await;
            loop {
                tick.tick().await;
                store.sweep();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PNG_1X1: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==";

    fn picture(caption: &str) -> ExtractedPicture {
        ExtractedPicture {
            data_uri: format!("data:image/png;base64,{PNG_1X1}"),
            mime: "image/png".into(),
            width: Some(1),
            height: Some(1),
            dpi: Some(72),
            bbox: Some(PictureBox {
                l: 1.0,
                t: 2.0,
                r: 3.0,
                b: 4.0,
                coord_origin: "TOPLEFT".into(),
            }),
            page_size: Some(PageSize {
                width: 612.0,
                height: 792.0,
            }),
            caption: Some(caption.into()),
            self_ref: Some("#/pictures/0".into()),
        }
    }

    fn temp_store(ttl: Duration, max_bytes: u64) -> PictureStore {
        let dir =
            std::env::temp_dir().join(format!("hypnos-pictures-test-{}", Uuid::new_v4().simple()));
        PictureStore::open(Some(&dir), ttl, max_bytes).expect("store")
    }

    #[test]
    fn pictures_round_trip_through_json_files() {
        let store = temp_store(Duration::from_secs(60), 0);
        let id = store.new_document();
        assert!(is_document_id(&id));
        let urls = store
            .add_page(&id, 2, &[picture("Figure 1"), picture("Figure 2")])
            .unwrap();
        assert_eq!(
            urls,
            vec![
                format!("/v1/documents/{id}/pictures/1"),
                format!("/v1/documents/{id}/pictures/2")
            ]
        );
        // A later page continues the numbering.
        let urls = store.add_page(&id, 3, &[picture("Figure 3")]).unwrap();
        assert_eq!(urls, vec![format!("/v1/documents/{id}/pictures/3")]);

        let list = store.list(&id).unwrap();
        assert_eq!(list.data.len(), 3);
        assert_eq!(list.data[2].page, 3);
        assert_eq!(list.data[0].caption.as_deref(), Some("Figure 1"));
        assert!(list.data.iter().all(|p| p.image.is_none()));
        assert!(list.data[0].bytes > 0);

        let pic = store.get(&id, 2).unwrap().unwrap();
        assert_eq!(pic.index, 2);
        assert_eq!(pic.image.as_deref(), Some(PNG_1X1));
        assert_eq!(pic.bbox.as_ref().map(|b| b.b), Some(4.0));
        let (mime, bytes) = store.get_bytes(&id, 2).unwrap().unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");
        assert!(store.get(&id, 4).unwrap().is_none());
        assert!(store.get("doc-nope", 1).unwrap().is_none());

        assert!(store.delete(&id));
        assert!(!store.delete(&id));
        assert!(store.list(&id).is_none());
        assert!(!store.dir().join(&id).exists());
        assert_eq!(store.stats(), (0, 0));
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn expired_and_oversized_documents_are_dropped() {
        let store = temp_store(Duration::ZERO, 0);
        let id = store.new_document();
        store.add_page(&id, 1, &[picture("x")]).unwrap();
        assert_eq!(store.sweep(), 1);
        assert!(store.list(&id).is_none());
        assert!(!store.dir().join(&id).exists());
        let _ = std::fs::remove_dir_all(store.dir());

        // Each record is a few hundred bytes: a cap of 500 keeps one document at a time.
        let store = temp_store(Duration::from_secs(60), 500);
        let first = store.new_document();
        store.add_page(&first, 1, &[picture("a")]).unwrap();
        let second = store.new_document();
        store.add_page(&second, 1, &[picture("b")]).unwrap();
        assert!(store.list(&first).is_none(), "the oldest document goes");
        assert!(store.list(&second).is_some(), "the one being written stays");
        assert!(!store.dir().join(&first).exists());
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn stale_documents_are_removed_at_open() {
        let dir =
            std::env::temp_dir().join(format!("hypnos-pictures-test-{}", Uuid::new_v4().simple()));
        let stale = dir.join("doc-0123456789abcdef0123456789abcdef");
        std::fs::create_dir_all(&stale).unwrap();
        std::fs::write(stale.join("1.json"), b"{}").unwrap();
        let other = dir.join("keep-me");
        std::fs::create_dir_all(&other).unwrap();
        let store = PictureStore::open(Some(&dir), Duration::from_secs(60), 0).unwrap();
        assert!(!stale.exists());
        assert!(other.exists(), "only document directories are touched");
        let _ = std::fs::remove_dir_all(store.dir());
    }

    #[test]
    fn data_uris_decode_with_their_mime() {
        let (mime, bytes) = decode_data_uri("data:image/jpeg;base64,aGk=", "image/png").unwrap();
        assert_eq!(
            (mime.as_str(), bytes.as_slice()),
            ("image/jpeg", b"hi".as_slice())
        );
        let (mime, bytes) = decode_data_uri("aGk=", "image/png").unwrap();
        assert_eq!(
            (mime.as_str(), bytes.as_slice()),
            ("image/png", b"hi".as_slice())
        );
        assert!(decode_data_uri("data:image/png;base64,***", "image/png").is_err());
    }
}
