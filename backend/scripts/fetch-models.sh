#!/usr/bin/env bash
# Downloads the model files listed in backend/models.manifest into backend/models/ with curl.
# Existing files are kept; interrupted downloads resume from their .part file.
#
#   ./backend/scripts/fetch-models.sh            # every model for this platform
#   ./backend/scripts/fetch-models.sh moss-image # one model id
#
# Before downloading, the missing files' sizes are checked against free disk space, keeping
# HYPNOS_MODELS_MIN_FREE_GB (default 2) free; nothing is fetched when they do not fit.
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
Q_REPO=(); Q_REMOTE=(); Q_DEST=()

# queue <repo> <remote path> <local path>: remember a file that is not downloaded yet.
queue() {
    if [ -f "$3" ]; then
        kept=$((kept + 1))
    else
        Q_REPO+=("$1"); Q_REMOTE+=("$2"); Q_DEST+=("$3")
    fi
}

# remote_size <repo> <remote path>: byte size from a HEAD request (the last Content-Length after
# Hugging Face's redirect to the file store); empty when unknown.
remote_size() {
    curl -fsSIL ${AUTH[@]+"${AUTH[@]}"} "$HF/$1/resolve/main/$2" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }'
}

# download <repo> <remote path> <local path>
download() {
    local repo="$1" remote="$2" dest="$3"
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
                queue "$repo" "$path" "$MODELS_DIR/$local${path#$remote}"
            done
            ;;
        *) queue "$repo" "$remote" "$MODELS_DIR/$local" ;;
    esac
done < "$MANIFEST"

# Disk space: everything still missing (minus resumable .part data) has to fit, plus a margin.
if [ "${#Q_DEST[@]}" -gt 0 ]; then
    echo "[models] checking the size of ${#Q_DEST[@]} missing file(s)..."
    need=0; unknown=0
    for ((i = 0; i < ${#Q_DEST[@]}; i++)); do
        size="$(remote_size "${Q_REPO[$i]}" "${Q_REMOTE[$i]}")"
        if [ -z "$size" ]; then unknown=$((unknown + 1)); continue; fi
        have=0
        [ -f "${Q_DEST[$i]}.part" ] && have=$(wc -c < "${Q_DEST[$i]}.part" | tr -d ' ')
        need=$((need + size - have))
    done
    mkdir -p "$MODELS_DIR"
    free=$(( $(df -Pk "$MODELS_DIR" | awk 'NR == 2 { print $4 }') * 1024 ))
    margin=$(( ${HYPNOS_MODELS_MIN_FREE_GB:-2} * 1000000000 ))
    gb() { awk -v b="$1" 'BEGIN { printf "%.1f GB", b / 1e9 }'; }
    echo "[models] to download: $(gb $need)$([ "$unknown" -gt 0 ] && echo " (+$unknown of unknown size)"), free: $(gb $free)"
    if [ $((need + margin)) -gt "$free" ]; then
        echo "[models] not enough disk space: need $(gb $need) plus $(gb $margin) kept free (HYPNOS_MODELS_MIN_FREE_GB)."
        echo "         Free some space, pick one model (./run.sh models <id>) or set HYPNOS_MODELS_DIR."
        exit 1
    fi
fi

for ((i = 0; i < ${#Q_DEST[@]}; i++)); do
    download "${Q_REPO[$i]}" "${Q_REMOTE[$i]}" "${Q_DEST[$i]}"
done

echo "[models] $fetched downloaded, $kept already present, $failed failed ($PLATFORM, ${MODELS_DIR#$ROOT/}/)"
[ "$failed" = 0 ]
