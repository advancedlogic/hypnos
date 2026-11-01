//! The config file and the `[[models]]` blocks of `POST /v1/models` (`HypnosConfig::from_toml_str`:
//! parse, profile, migrations), then validation of what parsed.
#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        if let Ok(config) = hypnos_core::config::HypnosConfig::from_toml_str(text) {
            // Paths resolve against a folder that does not exist: only the checks themselves run.
            let _ = config.validate(std::path::Path::new("/nonexistent-hypnos-fuzz"));
        }
    }
});
