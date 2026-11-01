//! Minimal GGUF metadata reader (std only) for the architecture values the KV planner needs.
//!
//! Only the metadata key/value section is read; tensor data is never touched, and large arrays
//! (tokenizer vocabularies) are skipped with a seek.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek};
use std::path::Path;

/// Architecture values found in a GGUF header. `None` when the file does not carry the key.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GgufArchMetadata {
    /// `general.architecture` (e.g. "gemma4", "qwen3").
    pub architecture: String,
    /// `<arch>.block_count`.
    pub block_count: Option<u32>,
    /// `<arch>.attention.head_count_kv`; the largest per-layer value when stored as an array.
    pub head_count_kv: Option<u32>,
    /// Largest of `<arch>.attention.{key,value}_length[_swa]`, else embedding_length / head_count.
    pub head_dim: Option<u32>,
    /// `<arch>.context_length` (training context).
    pub context_length: Option<u32>,
    /// `<arch>.attention.sliding_window`.
    pub sliding_window: Option<u32>,
    /// From `<arch>.attention.sliding_window_pattern`: one global layer every N layers.
    pub global_attn_every_n_layers: Option<u32>,
    /// Bytes of the Mixture-of-Experts expert tensors (`*_exps`), from the tensor table; `None`
    /// for a dense model or when the table could not be read.
    pub expert_bytes: Option<u64>,
    /// The KV layout when it is not uniform (per-layer KV heads, or a sliding-window pattern):
    /// `(global_kv_layers, global_num_kv_heads, swa_kv_layers, swa_num_kv_heads, swa_head_dim)`.
    /// Layers without KV heads (a hybrid model's convolution layers) count in neither.
    pub kv_layout: Option<(u32, u32, u32, u32, u32)>,
}

#[derive(Debug, Clone)]
enum Value {
    Int(u64),
    Str(String),
    IntArray(Vec<u64>),
    BoolArray(Vec<bool>),
    Other,
}

// GGUF value type ids.
const T_U8: u32 = 0;
const T_I8: u32 = 1;
const T_U16: u32 = 2;
const T_I16: u32 = 3;
const T_U32: u32 = 4;
const T_I32: u32 = 5;
const T_F32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_U64: u32 = 10;
const T_I64: u32 = 11;
const T_F64: u32 = 12;

/// Arrays longer than this are skipped instead of decoded (per-layer arrays are far shorter).
const MAX_DECODED_ARRAY: u64 = 4096;
/// Sanity cap for key and string lengths, so a corrupt header cannot request huge allocations.
const MAX_STRING: u64 = 1 << 20;

struct Reader<R> {
    r: R,
}

impl<R: Read + Seek> Reader<R> {
    fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut b = [0u8; N];
        self.r.read_exact(&mut b)?;
        Ok(b)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes::<4>()?))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes::<8>()?))
    }
    fn skip(&mut self, n: u64) -> io::Result<()> {
        let n = i64::try_from(n).map_err(|_| invalid("skip length overflows"))?;
        self.r.seek_relative(n)
    }
    fn string(&mut self) -> io::Result<String> {
        let len = self.u64()?;
        if len > MAX_STRING {
            return Err(invalid("string longer than 1 MiB"));
        }
        let mut buf = vec![0u8; len as usize];
        self.r.read_exact(&mut buf)?;
        Ok(String::from_utf8_lossy(&buf).into_owned())
    }
    fn skip_string(&mut self) -> io::Result<()> {
        let len = self.u64()?;
        self.skip(len)
    }

    /// Scalar integer of any width; negative values are clamped to 0, floats are not integers.
    fn int(&mut self, ty: u32) -> io::Result<Option<u64>> {
        Ok(match ty {
            T_U8 | T_BOOL => Some(self.bytes::<1>()?[0] as u64),
            T_I8 => Some((self.bytes::<1>()?[0] as i8).max(0) as u64),
            T_U16 => Some(u16::from_le_bytes(self.bytes::<2>()?) as u64),
            T_I16 => Some(i16::from_le_bytes(self.bytes::<2>()?).max(0) as u64),
            T_U32 => Some(self.u32()? as u64),
            T_I32 => Some((self.u32()? as i32).max(0) as u64),
            T_U64 => Some(self.u64()?),
            T_I64 => Some((self.u64()? as i64).max(0) as u64),
            T_F32 => {
                self.skip(4)?;
                None
            }
            T_F64 => {
                self.skip(8)?;
                None
            }
            _ => return Err(invalid("unknown scalar type")),
        })
    }

    fn scalar_size(ty: u32) -> Option<u64> {
        match ty {
            T_U8 | T_I8 | T_BOOL => Some(1),
            T_U16 | T_I16 => Some(2),
            T_U32 | T_I32 | T_F32 => Some(4),
            T_U64 | T_I64 | T_F64 => Some(8),
            _ => None,
        }
    }

    fn value(&mut self, ty: u32, decode: bool) -> io::Result<Value> {
        match ty {
            T_STRING if decode => Ok(Value::Str(self.string()?)),
            T_STRING => self.skip_string().map(|_| Value::Other),
            T_ARRAY => {
                let elem = self.u32()?;
                let n = self.u64()?;
                if decode && n <= MAX_DECODED_ARRAY && elem != T_STRING && elem != T_ARRAY {
                    if elem == T_BOOL {
                        let mut v = Vec::with_capacity(n as usize);
                        for _ in 0..n {
                            v.push(self.bytes::<1>()?[0] != 0);
                        }
                        return Ok(Value::BoolArray(v));
                    }
                    let mut v = Vec::with_capacity(n as usize);
                    for _ in 0..n {
                        if let Some(x) = self.int(elem)? {
                            v.push(x);
                        }
                    }
                    return Ok(Value::IntArray(v));
                }
                match Self::scalar_size(elem) {
                    Some(size) => self.skip(n.saturating_mul(size))?,
                    None => {
                        for _ in 0..n {
                            self.value(elem, false)?;
                        }
                    }
                }
                Ok(Value::Other)
            }
            _ => Ok(self.int(ty)?.map(Value::Int).unwrap_or(Value::Other)),
        }
    }
}

fn invalid(msg: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, format!("GGUF: {msg}"))
}

/// Reads every metadata key except the tokenizer's, which can be megabytes of strings, and then
/// the tensor table for the size of the expert tensors (`*_exps`).
/// The metadata keys and the expert tensor bytes (`None` for a dense model).
type Metadata = (HashMap<String, Value>, Option<u64>);

fn read_metadata(path: &Path) -> io::Result<Option<Metadata>> {
    let file = File::open(path)?;
    let file_len = file.metadata()?.len();
    let mut r = Reader {
        r: BufReader::new(file),
    };
    if &r.bytes::<4>()? != b"GGUF" {
        return Ok(None);
    }
    let version = r.u32()?;
    if version < 2 {
        return Err(invalid("GGUF v1 is not supported"));
    }
    let tensor_count = r.u64()?;
    let kv_count = r.u64()?;
    let mut out = HashMap::new();
    for _ in 0..kv_count {
        let key = r.string()?;
        let ty = r.u32()?;
        let decode = !key.starts_with("tokenizer.");
        let value = r.value(ty, decode)?;
        if !matches!(value, Value::Other) {
            out.insert(key, value);
        }
    }
    let alignment = match out.get("general.alignment") {
        Some(Value::Int(a)) if *a > 0 => *a,
        _ => 32,
    };
    // The table is optional for the caller: a failure here only loses the expert size.
    let experts = expert_bytes(&mut r, tensor_count, alignment, file_len).unwrap_or(None);
    Ok(Some((out, experts)))
}

/// Sums the sizes of the tensors named `*_exps`. A tensor's size is the distance from its data
/// offset to the next one (the last runs to the end of the file), so no table of quantization
/// formats is needed; alignment padding adds at most `alignment - 1` bytes per tensor.
fn expert_bytes<R: Read + Seek>(
    r: &mut Reader<R>,
    tensor_count: u64,
    alignment: u64,
    file_len: u64,
) -> io::Result<Option<u64>> {
    if tensor_count == 0 || tensor_count > 1_000_000 {
        return Ok(None);
    }
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()?;
        if n_dims > 8 {
            return Err(invalid("tensor with more than 8 dimensions"));
        }
        r.skip(u64::from(n_dims) * 8)?;
        let _ty = r.u32()?;
        let offset = r.u64()?;
        tensors.push((offset, name.contains("_exps")));
    }
    if !tensors.iter().any(|(_, e)| *e) {
        return Ok(None);
    }
    let table_end = r.r.stream_position()?;
    let data_start = table_end.div_ceil(alignment) * alignment;
    let data_len = file_len.saturating_sub(data_start);
    tensors.sort_by_key(|(o, _)| *o);
    let mut total = 0u64;
    for (i, (offset, is_expert)) in tensors.iter().enumerate() {
        let end = tensors.get(i + 1).map_or(data_len, |(next, _)| *next);
        if *is_expert {
            total += end.saturating_sub(*offset);
        }
    }
    Ok((total > 0).then_some(total))
}

impl GgufArchMetadata {
    /// Reads the architecture values from a GGUF file. `Ok(None)` when the file is not GGUF or
    /// has no `general.architecture` (e.g. a diffusion checkpoint converted without metadata).
    pub fn read(path: &Path) -> io::Result<Option<Self>> {
        let Some((kv, experts)) = read_metadata(path)? else {
            return Ok(None);
        };
        Ok(Self::from_metadata(&kv).map(|m| Self {
            expert_bytes: experts,
            ..m
        }))
    }

    fn from_metadata(kv: &HashMap<String, Value>) -> Option<Self> {
        let Some(Value::Str(arch)) = kv.get("general.architecture") else {
            return None;
        };
        let key = |k: &str| kv.get(&format!("{arch}.{k}"));
        let int = |k: &str| match key(k) {
            Some(Value::Int(v)) => u32::try_from(*v).ok(),
            _ => None,
        };
        // Scalar, or the largest per-layer value: the planner books every layer at that size.
        let int_or_max = |k: &str| match key(k) {
            Some(Value::Int(v)) => u32::try_from(*v).ok(),
            Some(Value::IntArray(v)) => v.iter().max().and_then(|m| u32::try_from(*m).ok()),
            _ => None,
        };

        let head_dim = [
            "attention.key_length",
            "attention.value_length",
            "attention.key_length_swa",
            "attention.value_length_swa",
        ]
        .iter()
        .filter_map(|k| int(k))
        .max()
        .or_else(|| {
            let heads = int_or_max("attention.head_count")?;
            (heads > 0).then(|| int("embedding_length").map(|e| e / heads))?
        });

        let sliding_window = int("attention.sliding_window").filter(|w| *w > 0);
        let global_attn_every_n_layers = match key("attention.sliding_window_pattern") {
            // One flag per layer, true = sliding; N = layers per global layer.
            Some(Value::BoolArray(v)) => {
                let global = v.iter().filter(|s| !**s).count();
                (global > 0 && global < v.len()).then(|| ((v.len() + global / 2) / global) as u32)
            }
            Some(Value::Int(n)) if *n > 1 => u32::try_from(*n).ok(),
            _ => None,
        }
        .filter(|_| sliding_window.is_some());

        // Per layer: KV heads (an array, or the scalar everywhere) and sliding or not.
        let blocks = int("block_count").unwrap_or(0) as usize;
        let mut kv_per_layer: Option<Vec<u64>> = match key("attention.head_count_kv") {
            Some(Value::IntArray(v)) => Some(v.clone()),
            Some(Value::Int(n)) => Some(vec![*n; blocks]),
            _ => None,
        };
        // Hybrid models whose recurrent layers keep no KV (Qwen 3.5+: linear attention with a
        // full-attention layer every N): one flag per layer, else every N-th layer has KV.
        let recurrent: Option<Vec<bool>> = match (
            key("attention.recurrent_layers"),
            int("full_attention_interval"),
        ) {
            (Some(Value::BoolArray(v)), _) => Some(v.clone()),
            (_, Some(n)) if n > 1 => Some((0..blocks).map(|i| (i + 1) % n as usize != 0).collect()),
            _ => None,
        };
        if let (Some(kv), Some(rec)) = (kv_per_layer.as_mut(), recurrent.as_ref()) {
            for (heads, is_recurrent) in kv.iter_mut().zip(rec) {
                if *is_recurrent {
                    *heads = 0;
                }
            }
        }
        let sliding: Option<Vec<bool>> = match key("attention.sliding_window_pattern") {
            Some(Value::BoolArray(v)) => Some(v.clone()),
            _ => None,
        };
        let non_uniform = matches!(key("attention.head_count_kv"), Some(Value::IntArray(_)))
            || sliding.is_some()
            || recurrent.is_some();
        let kv_layout = match (kv_per_layer, non_uniform) {
            (Some(kv), true) => {
                let is_swa = |i: usize| sliding.as_ref().is_some_and(|s| s.get(i) == Some(&true));
                let (mut global, mut swa, mut swa_heads, mut global_heads) =
                    (0u32, 0u32, 0u64, 0u64);
                for (i, heads) in kv.iter().enumerate() {
                    if *heads == 0 {
                        continue;
                    }
                    if is_swa(i) && sliding_window.is_some() {
                        swa += 1;
                        swa_heads = swa_heads.max(*heads);
                    } else {
                        global += 1;
                        global_heads = global_heads.max(*heads);
                    }
                }
                let swa_dim = ["attention.key_length_swa", "attention.value_length_swa"]
                    .iter()
                    .filter_map(|k| int(k))
                    .max()
                    .or(head_dim)
                    .unwrap_or(0);
                Some((global, global_heads as u32, swa, swa_heads as u32, swa_dim))
            }
            _ => None,
        };

        Some(Self {
            kv_layout,
            architecture: arch.clone(),
            block_count: int("block_count").filter(|n| *n > 0),
            head_count_kv: int_or_max("attention.head_count_kv"),
            head_dim,
            context_length: int("context_length").filter(|n| *n > 0),
            sliding_window,
            global_attn_every_n_layers,
            expert_bytes: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Writes a GGUF v3 header with the given metadata and no tensors.
    fn gguf(entries: &[(&str, u32, Vec<u8>)]) -> tempfile_path::TempPath {
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&0u64.to_le_bytes());
        b.extend_from_slice(&(entries.len() as u64).to_le_bytes());
        for (k, ty, payload) in entries {
            b.extend_from_slice(&(k.len() as u64).to_le_bytes());
            b.extend_from_slice(k.as_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(payload);
        }
        tempfile_path::TempPath::with_bytes(&b)
    }

    fn s(v: &str) -> Vec<u8> {
        let mut b = (v.len() as u64).to_le_bytes().to_vec();
        b.extend_from_slice(v.as_bytes());
        b
    }
    fn u32v(v: u32) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }
    fn arr(elem: u32, items: &[Vec<u8>]) -> Vec<u8> {
        let mut b = elem.to_le_bytes().to_vec();
        b.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for i in items {
            b.extend_from_slice(i);
        }
        b
    }

    /// Tiny self-deleting temp file (hypnos-core has no tempfile dependency).
    mod tempfile_path {
        use super::*;
        pub struct TempPath(pub std::path::PathBuf);
        impl TempPath {
            pub fn with_bytes(b: &[u8]) -> Self {
                static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let p = std::env::temp_dir().join(format!(
                    "hypnos-gguf-test-{}-{}.gguf",
                    std::process::id(),
                    N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                ));
                std::fs::File::create(&p).unwrap().write_all(b).unwrap();
                Self(p)
            }
        }
        impl Drop for TempPath {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }
    }

    #[test]
    fn reads_gemma4_style_per_layer_metadata() {
        // 6 layers, one global (false) every 6th, per-layer KV heads, a skipped tokenizer array.
        let pattern: Vec<Vec<u8>> = [1u8, 1, 1, 1, 1, 0].iter().map(|b| vec![*b]).collect();
        let kv_heads: Vec<Vec<u8>> = [8i32, 8, 8, 8, 8, 2]
            .iter()
            .map(|v| v.to_le_bytes().to_vec())
            .collect();
        let tokens: Vec<Vec<u8>> = (0..10).map(|i| s(&format!("tok{i}"))).collect();
        let f = gguf(&[
            ("general.architecture", T_STRING, s("gemma4")),
            ("tokenizer.ggml.tokens", T_ARRAY, arr(T_STRING, &tokens)),
            ("gemma4.block_count", T_U32, u32v(6)),
            ("gemma4.context_length", T_U32, u32v(262144)),
            (
                "gemma4.rope.freq_base",
                T_F32,
                1.0f32.to_le_bytes().to_vec(),
            ),
            (
                "gemma4.attention.head_count_kv",
                T_ARRAY,
                arr(T_I32, &kv_heads),
            ),
            ("gemma4.attention.key_length", T_U32, u32v(512)),
            ("gemma4.attention.key_length_swa", T_U32, u32v(256)),
            ("gemma4.attention.sliding_window", T_U32, u32v(1024)),
            (
                "gemma4.attention.sliding_window_pattern",
                T_ARRAY,
                arr(T_BOOL, &pattern),
            ),
        ]);
        let m = GgufArchMetadata::read(&f.0).unwrap().unwrap();
        assert_eq!(
            m,
            GgufArchMetadata {
                architecture: "gemma4".into(),
                block_count: Some(6),
                head_count_kv: Some(8),
                head_dim: Some(512),
                context_length: Some(262144),
                sliding_window: Some(1024),
                global_attn_every_n_layers: Some(6),
                expert_bytes: None,
                kv_layout: Some((1, 2, 5, 8, 256)),
            }
        );
    }

    #[test]
    fn expert_tensor_sizes_come_from_the_data_offsets() {
        // One metadata key, three tensors: two experts (64 bytes each) and an attention matrix.
        let mut b = Vec::new();
        b.extend_from_slice(b"GGUF");
        b.extend_from_slice(&3u32.to_le_bytes());
        b.extend_from_slice(&3u64.to_le_bytes()); // tensors
        b.extend_from_slice(&1u64.to_le_bytes()); // metadata keys
        let key = "general.architecture";
        b.extend_from_slice(&(key.len() as u64).to_le_bytes());
        b.extend_from_slice(key.as_bytes());
        b.extend_from_slice(&T_STRING.to_le_bytes());
        b.extend_from_slice(&s("gemma4"));
        for (name, offset) in [
            ("blk.0.ffn_gate_up_exps.weight", 0u64),
            ("blk.0.attn_q.weight", 64),
            ("blk.0.ffn_down_exps.weight", 96),
        ] {
            b.extend_from_slice(&(name.len() as u64).to_le_bytes());
            b.extend_from_slice(name.as_bytes());
            b.extend_from_slice(&1u32.to_le_bytes()); // one dimension
            b.extend_from_slice(&8u64.to_le_bytes());
            b.extend_from_slice(&0u32.to_le_bytes()); // f32
            b.extend_from_slice(&offset.to_le_bytes());
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend_from_slice(&[0u8; 160]); // data: 64 + 32 + 64
        let f = tempfile_path::TempPath::with_bytes(&b);
        let meta = GgufArchMetadata::read(&f.0).unwrap().unwrap();
        assert_eq!(meta.expert_bytes, Some(128));
    }

    #[test]
    fn falls_back_to_embedding_over_heads_and_ignores_missing_keys() {
        let f = gguf(&[
            ("general.architecture", T_STRING, s("llama")),
            ("llama.block_count", T_U32, u32v(32)),
            ("llama.embedding_length", T_U32, u32v(4096)),
            ("llama.attention.head_count", T_U32, u32v(32)),
            ("llama.attention.head_count_kv", T_U32, u32v(8)),
        ]);
        let m = GgufArchMetadata::read(&f.0).unwrap().unwrap();
        assert_eq!(m.head_dim, Some(128));
        assert_eq!(m.head_count_kv, Some(8));
        assert_eq!(m.context_length, None);
        assert_eq!(m.sliding_window, None);
        assert_eq!(m.global_attn_every_n_layers, None);
    }

    /// Qwen 3.5+: only the full-attention layers keep KV, named per layer or by an interval.
    #[test]
    fn hybrid_recurrent_layers_keep_no_kv() {
        let flags: Vec<Vec<u8>> = [1u8, 1, 1, 0, 1, 1, 1, 0]
            .iter()
            .map(|b| vec![*b])
            .collect();
        let per_layer = gguf(&[
            ("general.architecture", T_STRING, s("qwen35")),
            ("qwen35.block_count", T_U32, u32v(8)),
            ("qwen35.attention.head_count_kv", T_U32, u32v(4)),
            ("qwen35.attention.key_length", T_U32, u32v(256)),
            (
                "qwen35.attention.recurrent_layers",
                T_ARRAY,
                arr(T_BOOL, &flags),
            ),
        ]);
        let m = GgufArchMetadata::read(&per_layer.0).unwrap().unwrap();
        assert_eq!(m.kv_layout, Some((2, 4, 0, 0, 256)));
        let interval = gguf(&[
            ("general.architecture", T_STRING, s("qwen35moe")),
            ("qwen35moe.block_count", T_U32, u32v(40)),
            ("qwen35moe.attention.head_count_kv", T_U32, u32v(2)),
            ("qwen35moe.attention.key_length", T_U32, u32v(256)),
            ("qwen35moe.full_attention_interval", T_U32, u32v(4)),
        ]);
        let m = GgufArchMetadata::read(&interval.0).unwrap().unwrap();
        assert_eq!(m.kv_layout, Some((10, 2, 0, 0, 256)));
    }

    #[test]
    fn non_gguf_and_metadata_free_files_yield_none() {
        let f = tempfile_path::TempPath::with_bytes(b"not a gguf file at all");
        assert_eq!(GgufArchMetadata::read(&f.0).unwrap(), None);
        let f = gguf(&[]);
        assert_eq!(GgufArchMetadata::read(&f.0).unwrap(), None);
    }

    #[test]
    fn config_detection_fills_unset_arch_fields_and_keeps_explicit_ones() {
        use crate::config::HypnosConfig;
        let pattern: Vec<Vec<u8>> = [1u8, 1, 0].iter().map(|b| vec![*b]).collect();
        let f = gguf(&[
            ("general.architecture", T_STRING, s("toy")),
            ("toy.block_count", T_U32, u32v(12)),
            ("toy.context_length", T_U32, u32v(2048)),
            ("toy.attention.head_count_kv", T_U32, u32v(4)),
            ("toy.attention.key_length", T_U32, u32v(64)),
            ("toy.attention.sliding_window", T_U32, u32v(256)),
            (
                "toy.attention.sliding_window_pattern",
                T_ARRAY,
                arr(T_BOOL, &pattern),
            ),
        ]);
        let toml = format!(
            r#"
[[models]]
id = "detected"
modality = "text_generation"
engine = "ggml_llama"
weights_path = '{p}'

[[models]]
id = "explicit"
modality = "text_generation"
engine = "ggml_llama"
weights_path = '{p}'
[models.arch]
num_layers = 10
head_dim = 128
"#,
            p = f.0.display()
        );
        let mut cfg = HypnosConfig::from_toml_str(&toml).unwrap();
        let notes = cfg.detect_arch_from_gguf(std::path::Path::new("."));
        assert_eq!(notes.len(), 2, "{notes:?}");

        let a = &cfg.models[0].arch;
        assert_eq!((a.num_layers, a.num_kv_heads, a.head_dim), (12, 4, 64));
        assert_eq!(a.max_context_len, 2048);
        // The 4096 default reservation is clamped to the detected training context.
        assert_eq!(a.default_context_len, 2048);
        assert_eq!(
            (a.sliding_window_len, a.global_attn_every_n_layers),
            (Some(256), Some(3))
        );

        let b = &cfg.models[1].arch;
        assert_eq!((b.num_layers, b.head_dim), (10, 128), "explicit values win");
        assert_eq!(b.num_kv_heads, 4, "unset fields are still filled");
        assert!(
            notes[1].contains("kept from config: num_layers=10 (GGUF says 12)"),
            "{}",
            notes[1]
        );

        // Code-built params (the built-in profile) are never touched.
        cfg.models[0].arch.configured_keys = None;
        cfg.models[0].arch.num_layers = 99;
        cfg.detect_arch_from_gguf(std::path::Path::new("."));
        assert_eq!(cfg.models[0].arch.num_layers, 99);
    }
}
