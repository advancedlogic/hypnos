#!/usr/bin/env bash
# Execution-efficiency benchmark against a running Hypnos server (see docs/execution-efficiency-plan.md).
#
#   ./backend/scripts/bench.sh [--host http://127.0.0.1:8080] [--out docs/bench/baseline-$(date +%F).md] [--scenarios 1,2,3]
#
# Thin wrapper around bench.py (stdlib-only Python, same interpreter smoke_test.sh uses).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=python
command -v python >/dev/null 2>&1 || PY=python3

exec "$PY" -I -X utf8 "$DIR/bench.py" "$@"
