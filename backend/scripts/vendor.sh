#!/usr/bin/env bash
# Reproduces backend/vendor/ from pinned upstream commits plus the patches in backend/vendor/patches/.
# The vendor directory is git-ignored (over 1 GB of engine sources); this script is the source of truth.
#
#   ./backend/scripts/vendor.sh          # (re)create everything that is missing
#   ./backend/scripts/vendor.sh --force  # delete and recreate every vendored crate
#
# Pins (see backend/vendor/VENDOR.md):
LLAMA_CPP_RS_TAG="0.1.160"                        # utilityai/llama-cpp-rs (bindings)
LLAMA_CPP_TAG="b11982"                            # ggml-org/llama.cpp (gemma4 audio + gemma-embedding2)
DIFFUSION_RS_COMMIT="89dcf933f754a173c31b2db30ca3df14ec899dac"   # newfla/diffusion-rs (bindings)
SD_CPP_TAG="master-1024-77b72a2"                  # leejet/stable-diffusion.cpp (MOSS-Image)
ESPEAK_RS_SYS_VERSION="0.2.0"                     # crates.io (vendored to drop the debug-CRT link)
GLINER25_RS_VERSION="0.5.6"                       # crates.io (vendored: ort rc.15, CPU-only ort features)
DOCLING_ONNX_VERSION="2.55.4"                     # crates.io (vendored: CPU sessions really on the CPU)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENDOR="$ROOT/backend/vendor"
PATCHES="$VENDOR/patches"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1
mkdir -p "$VENDOR"
export GIT_CONFIG_PARAMETERS="'core.longpaths=true' 'advice.detachedHead=false'"

clone() { # <url> <ref> <dest> [extra git clone args]
    local url="$1" ref="$2" dest="$3"; shift 3
    if [ -d "$dest/.git" ] && [ "$FORCE" = 0 ]; then
        echo "[keep] $dest"
        return
    fi
    rm -rf "$dest"
    echo "[clone] $url @ $ref -> ${dest#$ROOT/}"
    if git clone -q --depth 1 --branch "$ref" "$@" "$url" "$dest" 2>/dev/null; then
        return
    fi
    # Not a tag/branch: fetch the commit explicitly.
    git init -q "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch -q --depth 1 origin "$ref"
    git -C "$dest" checkout -q FETCH_HEAD
    if [ "$*" = "--recurse-submodules --shallow-submodules" ]; then
        git -C "$dest" submodule update --init --depth 1 --recursive
    fi
}

apply_patch() { # <dir> <patch>
    local dir="$1" patch="$2"
    [ -f "$patch" ] || { echo "[skip] no patch $patch"; return; }
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
        git -C "$dir" apply "$patch"
        echo "[patch] applied $(basename "$patch")"
    elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "[patch] already applied $(basename "$patch")"
    else
        echo "[error] $(basename "$patch") does not apply to $dir" >&2
        exit 1
    fi
}

# 1. llama-cpp-rs bindings with upstream llama.cpp swapped in ---------------------------------
clone https://github.com/utilityai/llama-cpp-rs "$LLAMA_CPP_RS_TAG" "$VENDOR/llama-cpp-rs"
if [ ! -d "$VENDOR/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/.git" ] || [ "$FORCE" = 1 ]; then
    rm -rf "$VENDOR/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
    clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_TAG" "$VENDOR/llama-cpp-rs/llama-cpp-sys-2/llama.cpp"
fi
apply_patch "$VENDOR/llama-cpp-rs" "$PATCHES/llama-cpp-rs.patch"

# 2. diffusion-rs bindings with upstream stable-diffusion.cpp swapped in ------------------------
clone https://github.com/newfla/diffusion-rs "$DIFFUSION_RS_COMMIT" "$VENDOR/diffusion-rs"
if [ ! -d "$VENDOR/diffusion-rs/sys/stable-diffusion.cpp/.git" ] || [ "$FORCE" = 1 ]; then
    rm -rf "$VENDOR/diffusion-rs/sys/stable-diffusion.cpp"
    clone https://github.com/leejet/stable-diffusion.cpp "$SD_CPP_TAG" "$VENDOR/diffusion-rs/sys/stable-diffusion.cpp" --recurse-submodules --shallow-submodules
fi
apply_patch "$VENDOR/diffusion-rs" "$PATCHES/diffusion-rs.patch"

# 3. espeak-rs-sys from crates.io, minus the debug-CRT link directive ---------------------------
if [ ! -f "$VENDOR/espeak-rs-sys/Cargo.toml" ] || [ "$FORCE" = 1 ]; then
    rm -rf "$VENDOR/espeak-rs-sys"
    tmp="$(mktemp -d)"
    echo "[fetch] espeak-rs-sys $ESPEAK_RS_SYS_VERSION from crates.io"
    curl -sSL -A "hypnos-vendor" "https://crates.io/api/v1/crates/espeak-rs-sys/$ESPEAK_RS_SYS_VERSION/download" | tar xz -C "$tmp"
    mv "$tmp/espeak-rs-sys-$ESPEAK_RS_SYS_VERSION" "$VENDOR/espeak-rs-sys"
    rm -f "$VENDOR/espeak-rs-sys/.cargo_vcs_info.json" "$VENDOR/espeak-rs-sys/Cargo.toml.orig"
    (cd "$VENDOR/espeak-rs-sys" && patch -p1 --forward < "$PATCHES/espeak-rs-sys.patch")
    echo "[patch] applied espeak-rs-sys.patch"
else
    echo "[keep] $VENDOR/espeak-rs-sys"
fi

# 4. gliner25-rs from crates.io, ported to ort rc.15 and trimmed to CPU ort features --------------
if [ ! -f "$VENDOR/gliner25-rs/Cargo.toml" ] || [ "$FORCE" = 1 ]; then
    rm -rf "$VENDOR/gliner25-rs"
    tmp="$(mktemp -d)"
    echo "[fetch] gliner25-rs $GLINER25_RS_VERSION from crates.io"
    curl -sSL -A "hypnos-vendor" "https://crates.io/api/v1/crates/gliner25-rs/$GLINER25_RS_VERSION/download" | tar xz -C "$tmp"
    mv "$tmp/gliner25-rs-$GLINER25_RS_VERSION" "$VENDOR/gliner25-rs"
    rm -f "$VENDOR/gliner25-rs/.cargo_vcs_info.json" "$VENDOR/gliner25-rs/Cargo.toml.orig"
    (cd "$VENDOR/gliner25-rs" && patch -p1 --forward < "$PATCHES/gliner25-rs.patch")
    echo "[patch] applied gliner25-rs.patch"
else
    echo "[keep] $VENDOR/gliner25-rs"
fi

# 5. docling-onnx from crates.io: PreferCPU, so ort's default policy does not put CoreML under the
#    CPU choice on macOS ------------------------------------------------------------------------
if [ ! -f "$VENDOR/docling-onnx/Cargo.toml" ] || [ "$FORCE" = 1 ]; then
    rm -rf "$VENDOR/docling-onnx"
    tmp="$(mktemp -d)"
    echo "[fetch] docling-onnx $DOCLING_ONNX_VERSION from crates.io"
    curl -sSL -A "hypnos-vendor" "https://crates.io/api/v1/crates/docling-onnx/$DOCLING_ONNX_VERSION/download" | tar xz -C "$tmp"
    mv "$tmp/docling-onnx-$DOCLING_ONNX_VERSION" "$VENDOR/docling-onnx"
    rm -f "$VENDOR/docling-onnx/.cargo_vcs_info.json" "$VENDOR/docling-onnx/Cargo.toml.orig"
    (cd "$VENDOR/docling-onnx" && patch -p1 --forward < "$PATCHES/docling-onnx.patch")
    echo "[patch] applied docling-onnx.patch"
else
    echo "[keep] $VENDOR/docling-onnx"
fi

echo "[done] vendor tree ready under ${VENDOR#$ROOT/}"
