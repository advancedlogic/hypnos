#!/usr/bin/env bash
# Downloads the model files listed in backend/models.manifest into backend/models/ with curl.
# Existing files are kept; interrupted downloads resume from their .part file.
#
#   ./backend/scripts/fetch-models.sh            # every model for this platform
#   ./backend/scripts/fetch-models.sh moss-image # one model id
#
# HYPNOS_MODELS_DIR overrides the target folder (the configs expect backend/models/; symlink it).
# HF_TOKEN, when set, is sent to Hugging Face (higher rate limits); the repositories are public.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="$ROOT/backend/models.manifest"
MODELS_DIR="${HYPNOS_MODELS_DIR:-$ROOT/backend/models}"
HF="https://huggingface.co"
WANT="${1:-all}"

PLATFORM=other
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] && PLATFORM=macos-arm64

command -v curl >/dev/null 2>&1 || { echo "[models] curl is required"; exit 1; }
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
PROGRESS=(-sS)
[ -t 1 ] && PROGRESS=(-#)

ids="$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST" | awk '{print $1}' | sort -u)"
if [ "$WANT" != all ] && ! printf '%s\n' "$ids" | grep -qx -- "$WANT"; then
    echo "[models] unknown model '$WANT'. Known: all $(printf '%s ' $ids)"
    exit 1
fi

fetched=0; kept=0; failed=0

# download <repo> <remote path> <local path>
download() {
    local repo="$1" remote="$2" dest="$3"
    if [ -f "$dest" ]; then
        kept=$((kept + 1))
        return 0
    fi
    mkdir -p "$(dirname "$dest")"
    echo "[models] $repo/$remote -> ${dest#$ROOT/}"
    if curl -fL --retry 3 -C - "${PROGRESS[@]}" ${AUTH[@]+"${AUTH[@]}"} -o "$dest.part" "$HF/$repo/resolve/main/$remote"; then
        mv -f "$dest.part" "$dest"
        fetched=$((fetched + 1))
    else
        echo "[models] FAILED: $repo/$remote (partial data kept in ${dest#$ROOT/}.part; rerun to resume)"
        failed=$((failed + 1))
    fi
}

while read -r model repo remote local platform _; do
    case "$model" in ''|'#'*) continue ;; esac
    [ "$WANT" = all ] || [ "$WANT" = "$model" ] || continue
    [ "$platform" = all ] || [ "$platform" = "$PLATFORM" ] || continue
    case "$remote" in
        */)
            # Folder: list it through the Hub API and fetch every file in it.
            listing="$(curl -fsSL ${AUTH[@]+"${AUTH[@]}"} "$HF/api/models/$repo/tree/main/${remote%/}")" || {
                echo "[models] FAILED: could not list $repo/$remote"; failed=$((failed + 1)); continue; }
            for path in $(printf '%s' "$listing" | grep -o '"path":"[^"]*"' | sed 's/"path":"//; s/"$//'); do
                download "$repo" "$path" "$MODELS_DIR/$local${path#$remote}"
            done
            ;;
        *) download "$repo" "$remote" "$MODELS_DIR/$local" ;;
    esac
done < "$MANIFEST"

echo "[models] $fetched downloaded, $kept already present, $failed failed ($PLATFORM, ${MODELS_DIR#$ROOT/}/)"
[ "$failed" = 0 ]
