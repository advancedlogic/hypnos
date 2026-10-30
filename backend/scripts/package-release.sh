#!/usr/bin/env bash
# Assembles a release archive from an existing release build (see .github/workflows/release.yml).
#
#   ./backend/scripts/package-release.sh <platform> [--version v0.1.0] [--out dist]
#
#   platform: linux-x64-cuda | windows-x64-cuda | macos-arm64-metal
#
# Expects `cargo build --release -p hypnos-server ...` in backend/ and `npm run build` in
# frontend/ to have run. Produces <out>/hypnos-<version>-<platform>.tar.gz (.zip on Windows):
#
#   hypnos[.exe]                  server binary
#   (lib)stable-diffusion.*       sd.cpp shared library, found beside the binary
#   (lib)onnxruntime_providers_*  CUDA archives: ONNX Runtime's CUDA provider for docling's
#                                 device = "gpu" (needs the CUDA 13 runtime and cuDNN 9 at run time)
#   espeak-ng-data/               phoneme data for Kokoro TTS
#   frontend/                     the built playground, served by hypnos at /
#   hypnos.toml                   the platform's example config (paths relative to this folder)
#   models.manifest, scripts/     model downloader (fetch-models.sh / .ps1) into models/
#   LICENSE-*, licenses/, THIRD-PARTY-NOTICES.md, README.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLATFORM="${1:-}"
shift || true
VERSION="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo dev)"
OUT="$ROOT/dist"
while [ $# -gt 0 ]; do
    case "$1" in
        --version) VERSION="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

TARGET="$ROOT/backend/target/release"
case "$PLATFORM" in
    linux-x64-cuda) EXE=hypnos; LIB=libstable-diffusion.so; CONFIG=hypnos.example.toml ;;
    windows-x64-cuda) EXE=hypnos.exe; LIB=stable-diffusion.dll; CONFIG=hypnos.example.toml ;;
    macos-arm64-metal) EXE=hypnos; LIB=libstable-diffusion.dylib; CONFIG=hypnos.macos.example.toml ;;
    *) echo "usage: $0 <linux-x64-cuda|windows-x64-cuda|macos-arm64-metal> [--version V] [--out DIR]" >&2; exit 2 ;;
esac

NAME="hypnos-$VERSION-$PLATFORM"
STAGE="$OUT/$NAME"
REPO_URL="https://github.com/${GITHUB_REPOSITORY:-advancedlogic/hypnos}"

for f in "$TARGET/$EXE" "$TARGET/$LIB" "$TARGET/espeak-ng-data" "$ROOT/frontend/dist/index.html"; do
    [ -e "$f" ] || { echo "missing $f (build the backend in release mode and the frontend first)" >&2; exit 1; }
done

rm -rf "$STAGE"
mkdir -p "$STAGE/scripts" "$STAGE/licenses" "$STAGE/models"
cp "$TARGET/$EXE" "$TARGET/$LIB" "$STAGE/"
# ONNX Runtime's CUDA provider (the `docling-cuda` feature; ort leaves symlinks into its cache next
# to the binary): loaded only by a docling model with device = "gpu", so the archive works without
# the CUDA 13 runtime it needs. The TensorRT providers are not used.
DOCLING_CUDA=0
case "$PLATFORM" in
    linux-x64-cuda|windows-x64-cuda)
        for lib in "$TARGET"/*onnxruntime_providers_cuda.* "$TARGET"/*onnxruntime_providers_shared.*; do
            [ -e "$lib" ] || continue
            cp -L "$lib" "$STAGE/"
            DOCLING_CUDA=1
        done
        [ "$DOCLING_CUDA" = 1 ] || echo "note: no ONNX Runtime CUDA provider in $TARGET (built without --features docling-cuda); docling stays CPU-only" >&2
        ;;
esac
cp -R "$TARGET/espeak-ng-data" "$STAGE/espeak-ng-data"
cp -R "$ROOT/frontend/dist" "$STAGE/frontend"
cp "$ROOT/backend/$CONFIG" "$STAGE/hypnos.toml"
cp "$ROOT/backend/models.manifest" "$STAGE/"
cp "$ROOT/backend/scripts/fetch-models.sh" "$ROOT/backend/scripts/fetch-models.ps1" "$STAGE/scripts/"
cp "$ROOT/LICENSE-MIT" "$ROOT/LICENSE-APACHE" "$ROOT/THIRD-PARTY-NOTICES.md" "$STAGE/"
cp "$ROOT/licenses/GPL-3.0.txt" "$STAGE/licenses/"

RUN="./$EXE"
FETCH="./scripts/fetch-models.sh"
if [ "$PLATFORM" = windows-x64-cuda ]; then
    RUN=".\\$EXE"
    FETCH="powershell -ExecutionPolicy Bypass -File scripts\\fetch-models.ps1"
fi

{
    echo "Hypnos $VERSION ($PLATFORM)"
    echo "One OpenAI-compatible endpoint for every modality on one GPU."
    echo
    echo "Quick start (run everything from this folder; config paths are relative to it):"
    echo
    printf '  1. %-36s # ~31 GB into models/, or pass one model id\n' "$FETCH"
    printf '  2. %-36s # reads hypnos.toml\n' "$RUN"
    echo "  3. open http://127.0.0.1:8080 for the playground; the API is at http://127.0.0.1:8080/v1"
    echo
    echo "hypnos.toml lists the default lineup; remove [[models]] you did not download, or write"
    echo "your own (see the README's \"Bring your own models\"). Startup prints one error per problem."
    echo
    echo "Requirements:"
    case "$PLATFORM" in
        linux-x64-cuda)
            echo "  - x86-64 Linux with glibc 2.39 or newer (Ubuntu 24.04+, Debian 13+, RHEL 10+)."
            echo "  - NVIDIA driver 570 or newer (CUDA 12.8)."
            echo "  - CUDA 12 runtime libraries on the loader path: libcudart.so.12, libcublas.so.12,"
            echo "    libcublasLt.so.12 (Ubuntu: apt install cuda-cudart-12-8 libcublas-12-8 from NVIDIA's"
            echo "    repository, or the CUDA toolkit), and libgomp1."
            echo "  - Or use the Docker image: ghcr.io/${GITHUB_REPOSITORY:-advancedlogic/hypnos}"
            if [ "$DOCLING_CUDA" = 1 ]; then
            echo "  - Optional, only for docling with device = \"gpu\" (document conversion on the GPU;"
            echo "    the default device = \"cpu\" needs none of this): NVIDIA driver 580 or newer, the"
            echo "    CUDA 13 runtime (libcudart.so.13, libcublas.so.13, libcublasLt.so.13) and cuDNN 9 for"
            echo "    CUDA 13 (libcudnn.so.9) on the loader path (Ubuntu: apt install cuda-cudart-13-0"
            echo "    libcublas-13-0 libcudnn9-cuda-13 from NVIDIA's repository), plus the fp32 models"
            echo "    (scripts/fetch-models.sh docling-gpu). libonnxruntime_providers_cuda.so in this"
            echo "    folder is ONNX Runtime's CUDA provider; it is loaded only for that device."
            fi
            ;;
        windows-x64-cuda)
            echo "  - Windows 10/11 x64, NVIDIA driver 570 or newer (CUDA 12.8)."
            echo "  - CUDA 12 runtime DLLs (cudart64_12.dll, cublas64_12.dll, cublasLt64_12.dll) next to"
            echo "    hypnos.exe or on PATH: unpack the matching hypnos-$VERSION-windows-x64-cuda12-runtime.zip"
            echo "    from the same release here, or install the CUDA Toolkit 12.x."
            echo "  - Microsoft Visual C++ Redistributable 2015-2022 (x64)."
            if [ "$DOCLING_CUDA" = 1 ]; then
            echo "  - Optional, only for docling with device = \"gpu\" (document conversion on the GPU;"
            echo "    the default device = \"cpu\" needs none of this): NVIDIA driver 580 or newer, the"
            echo "    CUDA 13 runtime DLLs (cudart64_13.dll, cublas64_13.dll, cublasLt64_13.dll) and cuDNN 9"
            echo "    for CUDA 13 (cudnn64_9.dll and its cudnn_*64_9.dll) on PATH or next to hypnos.exe:"
            echo "    the CUDA Toolkit 13.x plus the cuDNN installer, or the pip wheels nvidia-cuda-runtime-cu13,"
            echo "    nvidia-cublas-cu13 and nvidia-cudnn-cu13 with their bin folders on PATH. Also the fp32"
            echo "    models (scripts\\fetch-models.ps1 docling-gpu). onnxruntime_providers_cuda.dll in this"
            echo "    folder is ONNX Runtime's CUDA provider; it is loaded only for that device."
            fi
            ;;
        macos-arm64-metal)
            echo "  - Apple Silicon Mac, macOS 14 or newer. Uses Metal and unified memory."
            echo "  - The binaries are not notarized: after downloading with a browser, run"
            echo "    xattr -dr com.apple.quarantine .   in this folder once."
            ;;
    esac
    echo
    echo "License: Hypnos's own code is MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE). This binary"
    echo "statically links espeak-ng (GPL-3.0-or-later), so the archive as a whole is distributed under"
    echo "the GNU GPL v3 or later (licenses/GPL-3.0.txt). Third-party components: THIRD-PARTY-NOTICES.md."
    echo "Corresponding source: $REPO_URL/tree/$VERSION (engine pins and patches:"
    echo "backend/scripts/vendor.sh and backend/vendor/patches/ in that tree)."
} > "$STAGE/README.txt"

cd "$OUT"
if [ "$PLATFORM" = windows-x64-cuda ]; then
    ARCHIVE="$NAME.zip"
    rm -f "$ARCHIVE"
    if command -v 7z >/dev/null 2>&1; then 7z a -tzip -mx=7 "$ARCHIVE" "$NAME" >/dev/null; else zip -qr "$ARCHIVE" "$NAME"; fi
else
    ARCHIVE="$NAME.tar.gz"
    tar -czf "$ARCHIVE" "$NAME"
fi
echo "$OUT/$ARCHIVE"
