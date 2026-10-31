#!/usr/bin/env bash
# Stages the Metal server for the macOS app bundle (DESK-16) in desktop/src-tauri/server/
# (git-ignored), where tauri.server-macos.conf.json picks it up:
#
#   ./desktop/scripts/stage-server.sh                          # from this checkout's release build
#   ./desktop/scripts/stage-server.sh --archive hypnos-v0.4.0-macos-arm64-metal.tar.gz
#   cd desktop && npx tauri build --bundles dmg --config src-tauri/tauri.server-macos.conf.json
#
# Without --archive it takes backend/target/release (cargo build --release -p hypnos-server
# --no-default-features --features metal) and frontend/dist (npm run build). With it, the
# release archive's files, as release.yml does.
#
# Layout in the bundle:
#   Contents/MacOS/hypnos                        the server (Tauri sidecar)
#   Contents/Frameworks/libstable-diffusion.dylib  sd.cpp (rpath @executable_path/../Frameworks)
#   Contents/Resources/espeak-ng-data/, frontend/  found by the server through ../Resources
#   Contents/Resources/server/hypnos.toml        the template of the app's own config
#   Contents/Resources/server/models.manifest    the lineup the app downloads (Models dialog)
#   Contents/Resources/server/LICENSE-*, licenses/, THIRD-PARTY-NOTICES.md, SOURCE.txt
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARCHIVE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --archive) ARCHIVE="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# Tauri looks for externalBin "server/hypnos" as server/hypnos-<target triple>.
TRIPLE="$(rustc -vV | sed -n 's/^host: //p')"
[ "$TRIPLE" = aarch64-apple-darwin ] || { echo "the bundled server is Apple Silicon only (host: $TRIPLE)" >&2; exit 1; }

STAGE="$ROOT/desktop/src-tauri/server"
rm -rf "$STAGE"
mkdir -p "$STAGE/resources/server/licenses"

if [ -n "$ARCHIVE" ]; then
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    tar -xzf "$ARCHIVE" -C "$work"
    SRC="$(find "$work" -mindepth 1 -maxdepth 1 -type d | head -1)"
    BIN="$SRC/hypnos"; LIB="$SRC/libstable-diffusion.dylib"; ESPEAK="$SRC/espeak-ng-data"
    FRONTEND="$SRC/frontend"; CONFIG="$SRC/hypnos.toml"; LEGAL="$SRC"; MANIFEST="$SRC/models.manifest"
    VERSION="$(basename "$SRC" | sed -E 's/^hypnos-(.*)-macos-arm64-metal$/\1/')"
else
    T="$ROOT/backend/target/release"
    BIN="$T/hypnos"; LIB="$T/libstable-diffusion.dylib"; ESPEAK="$T/espeak-ng-data"
    FRONTEND="$ROOT/frontend/dist"; CONFIG="$ROOT/backend/hypnos.macos.example.toml"; LEGAL="$ROOT"
    MANIFEST="$ROOT/backend/models.manifest"
    VERSION="$(git -C "$ROOT" describe --tags --always 2>/dev/null || echo dev)"
fi
for f in "$BIN" "$LIB" "$ESPEAK" "$FRONTEND/index.html" "$CONFIG" "$MANIFEST"; do
    [ -e "$f" ] || { echo "missing $f" >&2; exit 1; }
done
# The binary must find the dylib in Contents/Frameworks (hypnos-server/build.rs adds the rpath).
otool -l "$BIN" | grep -q '@executable_path/../Frameworks' \
    || { echo "$BIN has no @executable_path/../Frameworks rpath: rebuild it" >&2; exit 1; }

cp "$BIN" "$STAGE/hypnos-$TRIPLE"
cp "$LIB" "$STAGE/"
cp -R "$ESPEAK" "$STAGE/resources/espeak-ng-data"
cp -R "$FRONTEND" "$STAGE/resources/frontend"
cp "$CONFIG" "$STAGE/resources/server/hypnos.toml"
cp "$MANIFEST" "$STAGE/resources/server/models.manifest"   # the app's model downloader (DESK-17)
cp "$LEGAL/LICENSE-MIT" "$LEGAL/LICENSE-APACHE" "$LEGAL/THIRD-PARTY-NOTICES.md" "$STAGE/resources/server/"
cp "$LEGAL/licenses/GPL-3.0.txt" "$STAGE/resources/server/licenses/"

REPO_URL="https://github.com/${GITHUB_REPOSITORY:-advancedlogic/hypnos}"
cat > "$STAGE/resources/server/SOURCE.txt" <<EOF
Hypnos server $VERSION, bundled with the desktop app as a separate program
(Contents/MacOS/hypnos, Contents/Frameworks/libstable-diffusion.dylib).

Hypnos's own code is MIT OR Apache-2.0 (LICENSE-MIT, LICENSE-APACHE). The server binary
statically links espeak-ng (GPL-3.0-or-later), so the server as a whole is distributed under the
GNU GPL v3 or later (licenses/GPL-3.0.txt). Third-party components: THIRD-PARTY-NOTICES.md.

Corresponding source: $REPO_URL/tree/$VERSION
(engine pins and patches: backend/scripts/vendor.sh and backend/vendor/patches/ in that tree).
EOF

echo "staged the server ($VERSION) in $STAGE"
