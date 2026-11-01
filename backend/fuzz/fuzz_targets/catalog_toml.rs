//! The model catalog (`models.catalog.toml`, bundled and read by the server and the app).
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = hypnos_core::catalog::ModelCatalog::from_toml_str(text);
    }
});
