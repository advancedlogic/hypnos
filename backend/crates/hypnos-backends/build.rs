//! Makes the espeak-ng phoneme data (compiled by the `espeak-rs-sys` build script deep inside
//! `target/<profile>/build/espeak-rs-sys-*/out/`) discoverable at runtime by copying it next to the
//! final binaries as `target/<profile>/espeak-ng-data`. `espeak-rs` looks for that directory beside
//! the executable (or via `PIPER_ESPEAKNG_DATA_DIRECTORY`), so the server works out of the box.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // macOS: dyld does not search the executable's directory, and libstable-diffusion.dylib is
    // installed as @rpath/... and copied next to the test binaries (target/<profile>/deps).
    // Link args only apply to this package's own targets; hypnos-server emits the same for its binary.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path"),
        // Linux: find libstable-diffusion.so beside the binary without LD_LIBRARY_PATH.
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    // OUT_DIR = <target>/<profile>/build/hypnos-backends-<hash>/out
    let profile_dir = match out_dir.ancestors().nth(3) {
        Some(p) => p.to_path_buf(),
        None => return,
    };
    let build_dir = profile_dir.join("build");
    let pattern = format!(
        "{}/espeak-rs-sys-*/out/build/espeak-ng-data",
        build_dir.display()
    );
    let Ok(matches) = glob::glob(&pattern) else {
        return;
    };
    // Pick the most recently modified compiled data directory.
    let mut candidates: Vec<PathBuf> = matches.flatten().collect();
    candidates.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    let Some(src) = candidates.pop() else {
        println!("cargo:warning=espeak-ng-data not found yet under {}; Kokoro TTS will need PIPER_ESPEAKNG_DATA_DIRECTORY", build_dir.display());
        return;
    };
    let dst = profile_dir.join("espeak-ng-data");
    if let Err(e) = copy_dir(&src, &dst) {
        println!("cargo:warning=failed to copy espeak-ng-data: {e}");
    } else {
        println!(
            "cargo:rustc-env=HYPNOS_ESPEAK_DATA_PARENT={}",
            profile_dir.display()
        );
    }
}

fn copy_dir(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let target = dst.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            let needs_copy = match (std::fs::metadata(&entry.path()), std::fs::metadata(&target)) {
                (Ok(a), Ok(b)) => a.len() != b.len() || a.modified().ok() > b.modified().ok(),
                _ => true,
            };
            if needs_copy {
                std::fs::copy(entry.path(), &target)?;
            }
        }
    }
    Ok(())
}
