//! Upgrade compatibility (TEST-4): the example configs of every release still load without
//! errors or unknown keys, and files written for a newer Hypnos are refused with a clear message
//! instead of being misread. Add the new release's examples to `fixtures/` when tagging.

use std::path::Path;

use hypnos_core::catalog::ModelCatalog;
use hypnos_core::config::HypnosConfig;

#[test]
fn every_released_example_config_still_loads() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|e| e.to_str()) != Some("toml") {
            continue;
        }
        let text = std::fs::read_to_string(&path).unwrap();
        let config = HypnosConfig::from_toml_str(&text)
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert!(
            config.unknown_keys.is_empty(),
            "{}: keys this version no longer reads: {:?}",
            path.display(),
            config.unknown_keys
        );
        assert!(!config.models.is_empty(), "{}", path.display());
        seen += 1;
    }
    assert!(seen >= 6, "fixtures for v0.2.0 .. v0.4.0: {seen}");
}

#[test]
fn files_written_for_a_newer_hypnos_are_refused() {
    let err = HypnosConfig::from_toml_str("config_version = 99\n")
        .unwrap_err()
        .to_string();
    assert!(err.contains("newer Hypnos"), "{err}");
    let err = ModelCatalog::from_toml_str("catalog_version = 99\nmodels = []\n").unwrap_err();
    assert!(err.contains("newer than this build reads"), "{err}");
}
