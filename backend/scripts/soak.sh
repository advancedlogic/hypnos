#!/usr/bin/env bash
# Soak test against a running Hypnos server (see scripts/soak.py --help and docs/bench/).
#
#   ./backend/scripts/soak.sh [--host http://127.0.0.1:9999] [--duration 60] [--clients 6] [--out docs/bench/soak-$(date +%F).md]
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
command -v python >/dev/null 2>&1 || PY=python3

exec "$PY" -I -X utf8 "$DIR/soak.py" "$@"
