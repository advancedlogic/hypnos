//! macOS: lets the `hypnos` binary (and this crate's tests) find libstable-diffusion.dylib, which the
//! diffusion-rs build script copies into target/<profile>/ and target/<profile>/deps. dyld only
//! resolves its `@rpath/` install name through rpath entries, and a dependency's build script cannot
//! add link arguments to this crate's targets.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    }
}
