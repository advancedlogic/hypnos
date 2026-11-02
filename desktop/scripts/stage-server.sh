#!/usr/bin/env bash
# Stages the server for the desktop app's bundle (DESK-16 on macOS, DESK-32 on Windows and
# Linux) in desktop/src-tauri/server/ (git-ignored), where the platform's
# tauri.server-<os>.conf.json picks it up:
#
#   ./desktop/scripts/stage-server.sh                          # from this checkout's release build
#   ./desktop/scripts/stage-server.sh --archive hypnos-v0.4.0-macos-arm64-metal.tar.gz
#   ./desktop/scripts/stage-server.sh --archive hypnos-v0.4.0-windows-x64-cuda.zip \
#                                     --cuda-runtime hypnos-v0.4.0-windows-x64-cuda12-runtime.zip
#   cd desktop && npx tauri build --bundles dmg --config src-tauri/tauri.server-macos.conf.json
#
# Without --archive it takes backend/target/release (cargo build --release -p hypnos-server, with
# --no-default-features --features metal on a Mac) and frontend/dist (npm run build). With it,
# the release archive's files, as release.yml does. --cuda-runtime (Windows) adds the CUDA 12
# runtime DLLs of the release's cudart zip; without it they are taken from $CUDA_PATH/bin when
# that toolkit is installed, else the installed app needs the CUDA toolkit on PATH.
#
# Layout in the bundle (Tauri puts the sidecar beside the app's binary and the resources in
# Contents/Resources on macOS, the install folder on Windows, /usr/lib/Hypnos on Linux):
#   hypnos[.exe]                 the server (Tauri sidecar, staged as hypnos-<target triple>)
#   libstable-diffusion.dylib    sd.cpp on macOS: Contents/Frameworks (rpath @executable_path/../Frameworks)
#   stable-diffusion.dll / libstable-diffusion.so, onnxruntime_providers_*, cudart64_12.dll...
#                                sd.cpp, ONNX Runtime's CUDA provider and the CUDA runtime, with
#                                the resources (beside hypnos.exe on Windows; on Linux the app
#                                puts the resources on LD_LIBRARY_PATH)
#   espeak-ng-data/, frontend/   found by the server through ../Resources (macOS) or
#                                HYPNOS_RESOURCE_DIR, which the app sets (Windows, Linux)
#   server/hypnos.toml           the template of the app's own config
#   server/models.manifest       the lineup the app downloads (Models dialog)
#   server/models.catalog.toml   the model catalog (API-10), when the source has it
#   server/LICENSE-*, licenses/, THIRD-PARTY-NOTICES.md, SOURCE.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE=""
CUDA_RUNTIME=""
while [ $# -gt 0 ]; do
    case "$1" in
        --archive) ARCHIVE="$2"; shift 2 ;;
        --cuda-runtime) CUDA_RUNTIME="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Tauri looks for externalBin "server/hypnos" as server/hypnos-<target triple>[.exe].
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
case "$TRIPLE" in
    aarch64-apple-darwin)
        PLATFORM=macos-arm64-metal; EXE=hypnos; LIB=libstable-diffusion.dylib
        CONFIG=hypnos.macos.example.toml ;;
    x86_64-unknown-linux-gnu)
        PLATFORM=linux-x64-cuda; EXE=hypnos; LIB=libstable-diffusion.so
        CONFIG=hypnos.example.toml ;;
    x86_64-pc-windows-msvc)
        PLATFORM=windows-x64-cuda; EXE=hypnos.exe; LIB=stable-diffusion.dll
        CONFIG=hypnos.example.toml ;;
    *) echo "no bundled server for this host ($TRIPLE): Apple Silicon, x86-64 Linux and Windows only" >&2; exit 1 ;;
esac

STAGE="$ROOT/desktop/src-tauri/server"
rm -rf "$STAGE"
mkdir -p "$STAGE/resources/server/licenses"

extract() { # extract <archive> <dir>
    case "$1" in
        *.tar.gz) tar -xzf "$1" -C "$2" ;;
        *.zip)
            if command -v 7z >/dev/null 2>&1; then 7z x -o"$2" "$1" >/dev/null
            else unzip -q "$1" -d "$2"; fi ;;
        *) echo "unknown archive format: $1" >&2; exit 2 ;;
    esac
}

if [ -n "$ARCHIVE" ]; then
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    extract "$ARCHIVE" "$work"
    SRC="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
    BIN="$SRC/$EXE"; LIBFILE="$SRC/$LIB"; ESPEAK="$SRC/espeak-ng-data"
    FRONTEND="$SRC/frontend"; CONFIGFILE="$SRC/hypnos.toml"; LEGAL="$SRC"; MANIFEST="$SRC/models.manifest"
    PROVIDERS="$SRC"
    VERSION="$(basename "$SRC" | sed -E "s/^hypnos-(.*)-$PLATFORM\$/\\1/")"
else
    T="$ROOT/backend/target/release"
    BIN="$T/$EXE"; LIBFILE="$T/$LIB"; ESPEAK="$T/espeak-ng-data"
    FRONTEND="$ROOT/frontend/dist"; CONFIGFILE="$ROOT/backend/$CONFIG"; LEGAL="$ROOT"
    MANIFEST="$ROOT/backend/models.manifest"
    PROVIDERS="$T"
    # --match: the `engines` prerelease tag (BUILD-13) is not a version.
    VERSION="$(git -C "$ROOT" describe --tags --always --match 'v*' 2>/dev/null || echo dev)"
fi
for f in "$BIN" "$LIBFILE" "$ESPEAK" "$FRONTEND/index.html" "$CONFIGFILE" "$MANIFEST"; do
    [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done

cp "$BIN" "$STAGE/hypnos-$TRIPLE${EXE#hypnos}"
case "$PLATFORM" in
    macos-arm64-metal)
        # The binary must find the dylib in Contents/Frameworks (hypnos-server/build.rs adds the
        # rpath); no Mach-O files among the resources, which codesign treats as data.
        otool -l "$BIN" | grep -q '@executable_path/../Frameworks' \
            || { echo "$BIN has no @executable_path/../Frameworks rpath: rebuild it" >&2; exit 1; }
        cp "$LIBFILE" "$STAGE/"
        ;;
    *)
        # The shared libraries go with the resources: beside hypnos.exe in the install folder on
        # Windows, on LD_LIBRARY_PATH (set by the app) on Linux. ONNX Runtime's CUDA provider
        # (the docling-cuda feature; ort leaves symlinks into its cache next to a local build) is
        # loaded only by a docling model with device = "gpu".
        cp -L "$LIBFILE" "$STAGE/resources/"
        for lib in "$PROVIDERS"/*onnxruntime_providers_cuda.* "$PROVIDERS"/*onnxruntime_providers_shared.*; do
            [ -e "$lib" ] && cp -L "$lib" "$STAGE/resources/"
        done
        ;;
esac
if [ "$PLATFORM" = windows-x64-cuda ]; then
    # The CUDA 12 runtime (cudart, cuBLAS), which the archive ships as a separate zip: the
    # installed app carries it, so a machine without the CUDA toolkit can run the server.
    n=0
    if [ -n "$CUDA_RUNTIME" ]; then
        rt="$(mktemp -d)"
        extract "$CUDA_RUNTIME" "$rt"
        for dll in "$rt"/*.dll; do cp "$dll" "$STAGE/resources/"; n=$((n + 1)); done
        rm -rf "$rt"
    elif [ -n "${CUDA_PATH:-}" ]; then
        bin="$CUDA_PATH/bin"
        command -v cygpath >/dev/null 2>&1 && bin="$(cygpath -u "$CUDA_PATH")/bin"
        # CUDA 13 keeps the Windows DLLs in bin/x64, CUDA 12 in bin.
        for dll in "$bin"/cudart64_1?.dll "$bin"/cublas64_1?.dll "$bin"/cublasLt64_1?.dll                    "$bin"/x64/cudart64_1?.dll "$bin"/x64/cublas64_1?.dll "$bin"/x64/cublasLt64_1?.dll; do
            [ -e "$dll" ] || continue
            cp "$dll" "$STAGE/resources/"; n=$((n + 1))
        done
    fi
    [ "$n" -gt 0 ] || echo "note: no CUDA runtime DLLs staged (--cuda-runtime <zip> or CUDA_PATH): the installed app needs the CUDA toolkit on PATH" >&2
fi

cp -R "$ESPEAK" "$STAGE/resources/espeak-ng-data"
cp -R "$FRONTEND" "$STAGE/resources/frontend"
cp "$CONFIGFILE" "$STAGE/resources/server/hypnos.toml"
cp "$MANIFEST" "$STAGE/resources/server/models.manifest"   # the app's model downloader (DESK-17)
# The catalog sits next to the manifest (archives before API-10 have none).
CATALOG="$(dirname "$MANIFEST")/models.catalog.toml"
[ -f "$CATALOG" ] && cp "$CATALOG" "$STAGE/resources/server/models.catalog.toml"
cp "$LEGAL/LICENSE-MIT" "$LEGAL/LICENSE-APACHE" "$LEGAL/THIRD-PARTY-NOTICES.md" "$STAGE/resources/server/"
cp "$LEGAL/licenses/GPL-3.0.txt" "$STAGE/resources/server/licenses/"

REPO_URL="https://github.com/${GITHUB_REPOSITORY:-advancedlogic/hypnos}"
case "$PLATFORM" in
    macos-arm64-metal) WHERE="(Contents/MacOS/hypnos, Contents/Frameworks/libstable-diffusion.dylib)" ;;
    windows-x64-cuda) WHERE="(hypnos.exe and its DLLs in the app's install folder)" ;;
    linux-x64-cuda) WHERE="(/usr/bin/hypnos, libstable-diffusion.so in /usr/lib/Hypnos)" ;;
esac
cat > "$STAGE/resources/server/SOURCE.txt" <<EOF
Hypnos server $VERSION ($PLATFORM), bundled with the desktop app as a separate program
$WHERE.

Hypnos's own code is MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE). The server binary
statically links espeak-ng (GPL-3.0-or-later), so the server as a whole is distributed under the
GNU GPL v3 or later (licenses/GPL-3.0.txt). Third-party components: THIRD-PARTY-NOTICES.md.

Corresponding source: $REPO_URL/tree/$VERSION
(engine pins and patches: backend/scripts/vendor.sh and backend/vendor/patches/ in that tree).
EOF

echo "staged the server ($VERSION, $PLATFORM) in $STAGE"
