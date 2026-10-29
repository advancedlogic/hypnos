use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Basic metadata extracted directly from a `.gguf` file header on disk.
#[derive(Debug, Clone)]
pub struct GgufHeaderSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub file_size_bytes: u64,
}

impl GgufHeaderSummary {
    /// Reads the 24-byte GGUF v2/v3 header if the file exists and starts with `GGUF`.
    pub fn inspect(path: &Path) -> Option<Self> {
        let mut file = File::open(path).ok()?;
        let file_size_bytes = file.metadata().ok()?.len();
        let mut buf = [0u8; 24];
        file.read_exact(&mut buf).ok()?;

        if &buf[0..4] != b"GGUF" {
            return None;
        }

        let version = u32::from_le_bytes(buf[4..8].try_into().ok()?);
        let tensor_count = u64::from_le_bytes(buf[8..16].try_into().ok()?);
        let metadata_kv_count = u64::from_le_bytes(buf[16..24].try_into().ok()?);

        Some(Self {
            version,
            tensor_count,
            metadata_kv_count,
            file_size_bytes,
        })
    }
}
