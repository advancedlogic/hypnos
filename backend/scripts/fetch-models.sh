#!/usr/bin/env bash
# Downloads the model files listed in backend/models.manifest into backend/models/ with curl.
# Existing files are kept; interrupted downloads resume from their .part file.
#
#   ./backend/scripts/fetch-models.sh            # every model for this platform
#   ./backend/scripts/fetch-models.sh moss-image # one model id
#   ./backend/scripts/fetch-models.sh lfm3-vl:2b:Q4_K_M   # a models.catalog.toml entry[:variant]
#
# An id that is not in the manifest is looked up in the model catalog through the hypnos binary
# (`hypnos --catalog-files`; beside this folder in a release archive, target/release in a
# checkout); `hypnos --catalog-block <id>[:<quant>]` then prints the config block to add.
# Hugging Face downloads are checked against their SHA-256 (the X-Linked-ETag of LFS files).
#
# Before downloading, the missing files' sizes are checked against free disk space, keeping
# HYPNOS_MODELS_MIN_FREE_GB (default 2) free; nothing is fetched when they do not fit.
# HYPNOS_MODELS_DIR overrides the target folder (the configs expect backend/models/; symlink it).
# HF_TOKEN, when set, is sent to Hugging Face (higher rate limits). Gated repositories (FLUX's
# originals, `gated = true` in models.catalog.toml) need it, from an account that accepted the
# model's license on its Hugging Face page.
# A manifest repo that is an https:// URL is a plain download base (no token is sent there).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Relative to this script, so the same paths work in the repository (backend/scripts) and in a
# release archive (<archive>/scripts, with models.manifest and models/ beside it).
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$BASE/models.manifest"
MODELS_DIR="${HYPNOS_MODELS_DIR:-$BASE/models}"
HF="https://huggingface.co"
WANT="${1:-all}"

PLATFORM=other
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] && PLATFORM=macos-arm64

command -v curl >/dev/null 2>&1 || { echo "[models] curl is required"; exit 1; }
AUTH=()
[ -n "${HF_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $HF_TOKEN")
PROGRESS=(-sS)
[ -t 1 ] && PROGRESS=(-#)

# The manifest's column layout is versioned; a newer layout needs newer scripts.
mv="$(grep -m1 -E '^# manifest-version:' "$MANIFEST" | tr -dc '0-9')"
if [ -n "$mv" ] && [ "$mv" -gt 1 ]; then
    echo "[models] $MANIFEST is manifest-version $mv; these scripts read version 1. Update Hypnos." >&2
    exit 1
fi
ids="$(grep -vE '^[[:space:]]*(#|$)' "$MANIFEST" | awk '{print $1}' | sort -u)"
SOURCE="$(tr -d '\r' < "$MANIFEST")"   # a CRLF checkout must not hide every row behind a stray CR
CATALOG_SPEC=""
if [ "$WANT" != all ] && ! printf '%s\n' "$ids" | grep -qx -- "$WANT"; then
    # Not in the lineup: a catalog entry, through the server binary that reads the catalog.
    HYPNOS_BIN=""
    for b in "$BASE/hypnos" "$BASE/target/release/hypnos" "$BASE/target/debug/hypnos"; do
        [ -x "$b" ] && { HYPNOS_BIN="$b"; break; }
    done
    [ -n "$HYPNOS_BIN" ] || HYPNOS_BIN="$(command -v hypnos || true)"
    if [ -n "$HYPNOS_BIN" ] && lines="$(cd "$BASE" && "$HYPNOS_BIN" --catalog-files "$WANT" 2>&1)"; then
        SOURCE="$lines"
        CATALOG_SPEC="$WANT"
        WANT="$(printf '%s\n' "$lines" | awk 'NR == 1 { print $1 }')"
    else
        echo "[models] unknown model '$WANT'. Lineup: all $(printf '%s ' $ids)"
        [ -n "$HYPNOS_BIN" ] && echo "[models] catalog: ${lines:-}" \
            || echo "[models] (build the server to download catalog entries: ./run.sh build backend)"
        exit 1
    fi
fi

# sha256_of <file>: its SHA-256 in hex.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'
    else shasum -a 256 "$1" | awk '{print $1}'; fi
}

# expected_sha256 <repo> <remote path>: the SHA-256 Hugging Face publishes for an LFS file (its
# X-Linked-ETag), empty for other hosts and plain git files.
expected_sha256() {
    case "$1" in https://*) return ;; esac
    local hdr=(); while IFS= read -r a; do hdr+=("$a"); done < <(auth_for "$1")
    curl -fsSI ${hdr[@]+"${hdr[@]}"} "$(file_url "$1" "$2")" 2>/dev/null | tr -d '\r' \
        | awk 'tolower($1) == "x-linked-etag:" { gsub(/"/, "", $2); print $2 }' \
        | grep -E '^[0-9a-f]{64}$' | head -1
}

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

# file_url <repo> <remote path>: a Hugging Face repository, or a direct https:// download base.
file_url() {
    case "$1" in
        https://*) echo "$1/$2" ;;
        *) echo "$HF/$1/resolve/main/$2" ;;
    esac
}

# auth_for <repo>: the HF token header, only for Hugging Face.
auth_for() {
    case "$1" in https://*) ;; *) [ -n "${HF_TOKEN:-}" ] && printf '%s\n' -H "Authorization: Bearer $HF_TOKEN" ;; esac
}

# remote_size <repo> <remote path>: byte size from a HEAD request (the last Content-Length after
# the redirect to the file store); empty when unknown.
remote_size() {
    local hdr=(); while IFS= read -r a; do hdr+=("$a"); done < <(auth_for "$1")
    curl -fsSIL ${hdr[@]+"${hdr[@]}"} "$(file_url "$1" "$2")" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1) == "content-length:" { n = $2 } END { print n }'
}

# download <repo> <remote path> <local path>
download() {
    local repo="$1" remote="$2" dest="$3"
    mkdir -p "$(dirname "$dest")"
    local hdr=(); while IFS= read -r a; do hdr+=("$a"); done < <(auth_for "$repo")
    echo "[models] $repo/$remote -> ${dest#$ROOT/}"
    local want_sha; want_sha="$(expected_sha256 "$repo" "$remote")"
    if curl -fL --retry 3 -C - "${PROGRESS[@]}" ${hdr[@]+"${hdr[@]}"} -o "$dest.part" "$(file_url "$repo" "$remote")"; then
        if [ -n "$want_sha" ] && [ "$(sha256_of "$dest.part")" != "$want_sha" ]; then
            echo "[models] FAILED: $repo/$remote does not match its SHA-256 (removed; rerun to fetch it again)"
            rm -f "$dest.part"
            failed=$((failed + 1))
            return
        fi
        mv -f "$dest.part" "$dest"
        fetched=$((fetched + 1))
    else
        local code=$?
        echo "[models] FAILED: $repo/$remote (partial data kept in ${dest#$ROOT/}.part; rerun to resume)"
        # curl -f exits 22 on an HTTP error: a gated repository answers 401 / 403.
        if [ "$code" = 22 ] && [[ "$repo" != https://* ]]; then
            echo "[models]   gated? accept the license at $HF/$repo and set HF_TOKEN (a read token of that account)"
        fi
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
done < <(printf '%s\n' "$SOURCE")

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
if [ -n "$CATALOG_SPEC" ] && [ "$failed" = 0 ]; then
    echo "[models] add it to the config (paths are relative to models_dir):"
    echo "         $HYPNOS_BIN --catalog-block $CATALOG_SPEC >> hypnos.toml"
fi
[ "$failed" = 0 ]
