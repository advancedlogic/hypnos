//! Lets the `hypnos` binary (and this crate's tests) find the sd.cpp shared library, which the
//! diffusion-rs build script copies into target/<profile>/ and target/<profile>/deps and release
//! archives ship beside the binary. macOS: dyld only resolves its `@rpath/` install name through
//! rpath entries. Linux: `$ORIGIN` replaces LD_LIBRARY_PATH. A dependency's build script cannot add
//! link arguments to this crate's targets.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path"),
        // Linux: find libstable-diffusion.so beside the binary without LD_LIBRARY_PATH.
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN"),
        _ => {}
    }
}
