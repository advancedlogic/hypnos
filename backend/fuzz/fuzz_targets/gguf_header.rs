//! The GGUF header reader (`hypnos_core::gguf`): every model file the server detects, at startup
//! and when a block is added to the pool, goes through it.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = hypnos_core::gguf::GgufArchMetadata::from_reader(std::io::Cursor::new(data), data.len() as u64);
});
