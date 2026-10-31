#!/usr/bin/env bash
# Signs (and optionally notarizes) the macOS server binaries in place, before package-release.sh.
# See docs/signing.md, section 3.
#
#   ./backend/scripts/sign-macos.sh [--notarize] [--dir backend/target/release]
#
#   APPLE_SIGNING_IDENTITY  "Developer ID Application: Name (TEAMID)" (required)
#   SIGN_KEYCHAIN           keychain to take the identity from (optional; CI uses a temporary one)
#   --notarize credentials, one of:
#     NOTARY_PROFILE                              a `xcrun notarytool store-credentials` profile
#     APPLE_ID + APPLE_PASSWORD + APPLE_TEAM_ID   Apple ID and an app-specific password
#
# The library is signed before the executable that loads it, both with the hardened runtime and a
# secure timestamp (notarization requires both) and no entitlements: library validation is
# satisfied because both carry the same team's signature. Bare Mach-O files cannot be stapled;
# Gatekeeper fetches the ticket online on first launch.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/backend/target/release"
NOTARIZE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --notarize) NOTARIZE=1; shift ;;
        --dir) DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

: "${APPLE_SIGNING_IDENTITY:?set APPLE_SIGNING_IDENTITY to the Developer ID Application identity}"
LIB="$DIR/libstable-diffusion.dylib"
EXE="$DIR/hypnos"
for f in "$LIB" "$EXE"; do
    [ -f "$f" ] || { echo "missing $f (cargo build --release -p hypnos-server --no-default-features --features metal)" >&2; exit 1; }
done

keychain=()
[ -n "${SIGN_KEYCHAIN:-}" ] && keychain=(--keychain "$SIGN_KEYCHAIN")
sign() { # <identifier> <file>
    # ${a[@]+...}: macOS's bash 3.2 calls an empty array unbound under set -u.
    codesign --force --timestamp --options runtime ${keychain[@]+"${keychain[@]}"} \
        --sign "$APPLE_SIGNING_IDENTITY" --identifier "$1" "$2"
}
sign dev.hypnos.stable-diffusion "$LIB"
sign dev.hypnos.server "$EXE"
codesign --verify --strict --verbose=2 "$LIB" "$EXE"

[ "$NOTARIZE" = 1 ] || exit 0

if [ -n "${NOTARY_PROFILE:-}" ]; then
    creds=(--keychain-profile "$NOTARY_PROFILE")
elif [ -n "${APPLE_ID:-}" ] && [ -n "${APPLE_PASSWORD:-}" ] && [ -n "${APPLE_TEAM_ID:-}" ]; then
    creds=(--apple-id "$APPLE_ID" --password "$APPLE_PASSWORD" --team-id "$APPLE_TEAM_ID")
else
    echo "--notarize needs NOTARY_PROFILE or APPLE_ID + APPLE_PASSWORD + APPLE_TEAM_ID" >&2
    exit 2
fi

# notarytool takes .zip/.dmg/.pkg, not .tar.gz; the ticket is bound to the code hashes, so the
# published tar.gz is covered all the same.
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
mkdir "$work/hypnos"
cp "$EXE" "$LIB" "$work/hypnos/"
ditto -c -k --keepParent "$work/hypnos" "$work/hypnos.zip"
out="$(xcrun notarytool submit "$work/hypnos.zip" "${creds[@]}" --wait --output-format json)"
echo "$out"
id="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])' <<<"$out")"
status="$(python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])' <<<"$out")"
if [ "$status" != Accepted ]; then
    xcrun notarytool log "$id" "${creds[@]}" >&2 || true
    echo "notarization: $status" >&2
    exit 1
fi
echo "notarized: $id"
