#!/usr/bin/env bash
# ============================================================================
#  Hypnos process manager (Linux / macOS / Git Bash)
#
#    ./run.sh [build|start|stop|restart|status|logs] [backend|frontend|all]
#
#    ./run.sh build            -> compile the backend (cargo build --release) + npm install
#    ./run.sh                  -> start all (background, logs in .run/; never compiles)
#    ./run.sh restart backend  -> stop + start the already-built backend binary
#    ./run.sh stop             -> stop backend and frontend
#    ./run.sh logs             -> follow both logs
#    ./run.sh models [id]      -> download model files into backend/models/ (backend/models.manifest)
#
#  Build and start are separate steps: `start` launches the compiled binary
#  backend/target/<profile>/hypnos (hypnos.exe on Windows) and refuses to run when it is
#  missing, so a stale or in-progress cargo build can never masquerade as a running server.
#  Set HYPNOS_PROFILE=debug to build/start the dev profile instead of release, and
#  HYPNOS_CONFIG=<file> (relative to backend/) to start the backend with your own config.
#
#  Backend : Rust server on http://localhost:8080  (compiled hypnos binary)
#  Frontend: Vite dev server on http://localhost:5173 (npm run dev)
#  Output goes to .run/*.log. On Linux/macOS each service runs in its own process group
#  (pid in .run/*.pid); on Windows (Git Bash) the listening port is the source of truth.
# ============================================================================
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$ROOT_DIR/.run"
mkdir -p "$RUN_DIR"

BACKEND_PORT=8080
FRONTEND_PORT=5173
BACKEND_CONFIG="${HYPNOS_CONFIG:-hypnos.example.toml}"

ACTION="${1:-start}"
TARGET="${2:-all}"

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS=0 ;;
esac

# Apple Silicon: Metal instead of the default CUDA engines, and the unified-memory profile.
CARGO_FEATURE_FLAGS=()
if [ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ]; then
    CARGO_FEATURE_FLAGS=(--no-default-features --features metal)
    [ -z "${HYPNOS_CONFIG:-}" ] && [ -f "$ROOT_DIR/backend/hypnos.macos.example.toml" ] && BACKEND_CONFIG="hypnos.macos.example.toml"
fi

# Cargo profile for the backend binary: release (default) or debug (`cargo build` dev profile).
HYPNOS_PROFILE="${HYPNOS_PROFILE:-release}"
[ "$HYPNOS_PROFILE" = dev ] && HYPNOS_PROFILE=debug
case "$HYPNOS_PROFILE" in
    release) CARGO_PROFILE_FLAGS=(--release) ;;
    debug)   CARGO_PROFILE_FLAGS=() ;;
    *)       CARGO_PROFILE_FLAGS=(--profile "$HYPNOS_PROFILE") ;;
esac
BACKEND_TARGET_DIR="$ROOT_DIR/backend/target/$HYPNOS_PROFILE"
BACKEND_BIN="$BACKEND_TARGET_DIR/hypnos"
[ "$IS_WINDOWS" = 1 ] && BACKEND_BIN="$BACKEND_BIN.exe"

# Prefer the rustup-managed toolchain: the native engines need rustc >= 1.92 and a standalone
# Rust install may come first on PATH.
# `rustup run stable cargo` pins rustc as well: calling cargo by path still resolves `rustc` from
# PATH, which may be an older standalone install.
# Put the rustup toolchain first on PATH as well: `rustup run` alone does not outrank a standalone Rust install.
for tc in "$HOME/.rustup/toolchains/stable-x86_64-pc-windows-msvc/bin" "$HOME/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/bin" "$HOME/.rustup/toolchains/stable-aarch64-apple-darwin/bin" "$HOME/.rustup/toolchains/stable-x86_64-apple-darwin/bin"; do
    [ -d "$tc" ] && export PATH="$tc:$HOME/.cargo/bin:$PATH" && break
done
CARGO_CMD=(cargo)
if [ -x "$HOME/.cargo/bin/rustup" ] || [ -x "$HOME/.cargo/bin/rustup.exe" ]; then
    CARGO_CMD=("$HOME/.cargo/bin/rustup" run stable cargo)
fi

usage() {
    cat <<EOF

  Usage: ./run.sh [build|start|stop|restart|status|logs] [backend|frontend|all]

    ./run.sh build              cargo build --release the backend + npm install the frontend
    ./run.sh build backend      compile only the Rust server (backend/target/$HYPNOS_PROFILE/hypnos)
    ./run.sh                    start backend + frontend from the built artifacts (default)
    ./run.sh start backend      start only the Rust backend binary (port $BACKEND_PORT)
    ./run.sh restart frontend   restart only the Vite dev server (port $FRONTEND_PORT)
    ./run.sh stop               stop both
    ./run.sh status             show what is running and whether the backend is built
    ./run.sh logs [target]      tail the log file(s) in .run/
    ./run.sh models [model id]  download the model files (about 31 GB for all) into backend/models/

  start/restart never compile; run ./run.sh build after changing Rust code.
  HYPNOS_PROFILE=debug selects the dev profile (backend/target/debug/hypnos).

EOF
    exit 1
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
pid_alive() {
    local pid="$1"
    [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# PIDs of processes listening on a TCP port (one per line).
port_pids() {
    local port="$1"
    if [ "$IS_WINDOWS" = 1 ]; then
        "$SYSTEMROOT/System32/netstat.exe" -ano 2>/dev/null \
            | awk -v p=":$port" '$2 ~ p"$" && $4 == "LISTENING" {print $5}' | sort -u | grep -v '^0$'
    elif command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnp 2>/dev/null | awk -v p=":$port" '$4 ~ p"$"' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u
    else
        fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u
    fi
}

port_free() {
    [ -z "$(port_pids "$1")" ]
}

# kill_tree <pid>: Windows PID (taskkill tree) or Unix process-group leader.
kill_tree() {
    local pid="$1"
    [ -n "$pid" ] || return 0
    if [ "$IS_WINDOWS" = 1 ]; then
        "$SYSTEMROOT/System32/taskkill.exe" //PID "$pid" //T //F >/dev/null 2>&1
    else
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 1 20); do
            pid_alive "$pid" || return 0
            sleep 0.5
        done
        kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
    fi
}

# is_running <name> <port>: prints a pid and returns 0 when the service is up.
is_running() {
    local name="$1" port="$2" pid
    pid="$(port_pids "$port" | head -n1)"
    if [ -n "$pid" ]; then echo "$pid"; return 0; fi
    if [ "$IS_WINDOWS" = 0 ] && [ -f "$RUN_DIR/$name.pid" ]; then
        pid="$(tr -d '[:space:]' < "$RUN_DIR/$name.pid")"
        if pid_alive "$pid"; then echo "$pid"; return 0; fi
    fi
    return 1
}

# spawn <name> <workdir> <command...>: detached background process, output in .run/<name>.log.
# Nothing is printed here on purpose: the child must not inherit a captured stdout.
spawn() {
    local name="$1" workdir="$2"; shift 2
    local log="$RUN_DIR/$name.log"
    : > "$log"
    if [ "$IS_WINDOWS" = 1 ] || ! command -v setsid >/dev/null 2>&1; then
        (cd "$workdir" && nohup "$@" >> "$log" 2>&1 < /dev/null &
         echo $! > "$RUN_DIR/$name.pid") >/dev/null 2>&1
    else
        (cd "$workdir" && setsid nohup "$@" >> "$log" 2>&1 < /dev/null &
         echo $! > "$RUN_DIR/$name.pid") >/dev/null 2>&1
    fi
}

# wait_up <name> <port> <seconds>: waits for the port to be bound; prints the final state.
wait_up() {
    local name="$1" port="$2" secs="$3" pid
    for _ in $(seq 1 $((secs * 2))); do
        if pid="$(is_running "$name" "$port")"; then
            echo "[Hypnos] $name RUNNING (pid $pid) -> http://localhost:$port"
            return 0
        fi
        sleep 0.5
    done
    echo "[Hypnos] $name starting in the background -> http://localhost:$port (not bound yet; follow .run/$name.log)"
}

stop_one() {
    local name="$1" port="$2" stopped=0 pid
    if [ "$IS_WINDOWS" = 0 ] && [ -f "$RUN_DIR/$name.pid" ]; then
        pid="$(tr -d '[:space:]' < "$RUN_DIR/$name.pid")"
        if pid_alive "$pid"; then kill_tree "$pid"; stopped=1; fi
    fi
    rm -f "$RUN_DIR/$name.pid"
    for pid in $(port_pids "$port"); do
        kill_tree "$pid"; stopped=1
    done
    if [ "$name" = backend ] && [ "$IS_WINDOWS" = 1 ]; then
        # The server binary may outlive its console if it was interrupted mid-start.
        if "$SYSTEMROOT/System32/tasklist.exe" //FI "IMAGENAME eq hypnos.exe" //NH 2>/dev/null | grep -qi 'hypnos.exe'; then
            "$SYSTEMROOT/System32/taskkill.exe" //IM hypnos.exe //F >/dev/null 2>&1
            stopped=1
        fi
    fi
    if [ "$stopped" = 1 ]; then
        # Give the OS a moment to release the port so a restart with --strictPort succeeds.
        for _ in $(seq 1 20); do port_free "$port" && break; sleep 0.5; done
        echo "[Hypnos] $name stopped."
    else
        echo "[Hypnos] $name was not running."
    fi
}

# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
# build_backend: cargo build in the foreground (progress is visible, exit code propagates).
# The vendored engine build scripts copy the stable-diffusion shared library next to the binary.
build_backend() {
    echo "[Hypnos] Building backend ($HYPNOS_PROFILE profile) -> $BACKEND_BIN"
    echo "         The first build compiles the native engines (llama.cpp, stable-diffusion.cpp, espeak-ng); this takes a while."
    (cd "$ROOT_DIR/backend" && "${CARGO_CMD[@]}" build ${CARGO_PROFILE_FLAGS[@]+"${CARGO_PROFILE_FLAGS[@]}"} -p hypnos-server --bin hypnos ${CARGO_FEATURE_FLAGS[@]+"${CARGO_FEATURE_FLAGS[@]}"})
    local rc=$?
    if [ "$rc" != 0 ]; then
        echo "[Hypnos] backend build FAILED (cargo exit code $rc)."
        return 1
    fi
    if [ ! -x "$BACKEND_BIN" ]; then
        echo "[Hypnos] cargo reported success but $BACKEND_BIN does not exist."
        return 1
    fi
    echo "[Hypnos] backend built: $BACKEND_BIN"
}

# build_frontend: install/refresh node dependencies. The Vite dev server bundles on the fly.
build_frontend() {
    echo "[Hypnos] Installing frontend dependencies (npm install)..."
    (cd "$ROOT_DIR/frontend" && npm install) || { echo "[Hypnos] frontend dependency install FAILED."; return 1; }
    echo "[Hypnos] frontend dependencies ready."
}

# ---------------------------------------------------------------------------
# services
# ---------------------------------------------------------------------------
# start_backend: run the compiled binary. Never compiles: a missing binary points at `build`.
start_backend() {
    local pid
    if pid="$(is_running backend "$BACKEND_PORT")"; then
        echo "[Hypnos] backend already running (pid $pid, port $BACKEND_PORT)."
        return 0
    fi
    if [ ! -x "$BACKEND_BIN" ]; then
        echo "[Hypnos] backend binary not found: $BACKEND_BIN"
        echo "         Run ./run.sh build backend first (or set HYPNOS_PROFILE to the profile you built)."
        return 1
    fi
    # `cargo run` would add target/<profile>/deps to the loader path; do the same for the
    # stable-diffusion shared library when launching the binary directly.
    if [ "$IS_WINDOWS" = 0 ]; then
        export LD_LIBRARY_PATH="$BACKEND_TARGET_DIR:$BACKEND_TARGET_DIR/deps${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export DYLD_LIBRARY_PATH="$BACKEND_TARGET_DIR:$BACKEND_TARGET_DIR/deps${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
    echo "[Hypnos] backend starting ($HYPNOS_PROFILE build: $BACKEND_BIN)"
    echo "         log: $RUN_DIR/backend.log"
    spawn backend "$ROOT_DIR/backend" "$BACKEND_BIN" "$BACKEND_CONFIG"
    wait_up backend "$BACKEND_PORT" 5
}

start_frontend() {
    local pid
    if pid="$(is_running frontend "$FRONTEND_PORT")"; then
        echo "[Hypnos] frontend already running (pid $pid, port $FRONTEND_PORT)."
        return 0
    fi
    if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
        echo "[Hypnos] frontend dependencies not installed: $ROOT_DIR/frontend/node_modules"
        echo "         Run ./run.sh build frontend first."
        return 1
    fi
    echo "[Hypnos] frontend starting (npm run dev)"
    echo "         log: $RUN_DIR/frontend.log"
    spawn frontend "$ROOT_DIR/frontend" npm run dev -- --port "$FRONTEND_PORT" --strictPort
    wait_up frontend "$FRONTEND_PORT" 15
}

status_one() {
    local name="$1" port="$2" pid
    if pid="$(is_running "$name" "$port")"; then
        echo "[Hypnos] $name: RUNNING  (pid $pid, http://localhost:$port)"
    else
        echo "[Hypnos] $name: stopped  (port $port free)"
    fi
    if [ "$name" = backend ]; then
        if [ -x "$BACKEND_BIN" ]; then
            echo "[Hypnos] backend binary: $BACKEND_BIN"
        else
            echo "[Hypnos] backend binary: not built (./run.sh build backend)"
        fi
    fi
}

logs() {
    local files=()
    case "$TARGET" in
        backend)  files=("$RUN_DIR/backend.log") ;;
        frontend) files=("$RUN_DIR/frontend.log") ;;
        all)      files=("$RUN_DIR/backend.log" "$RUN_DIR/frontend.log") ;;
    esac
    for f in "${files[@]}"; do touch "$f"; done
    exec tail -n 50 -f "${files[@]}"
}

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
# `models` takes a model id instead of a service target.
if [ "$ACTION" = models ]; then
    exec bash "$ROOT_DIR/backend/scripts/fetch-models.sh" "$TARGET"
fi

case "$TARGET" in backend|frontend|all) ;; *) echo "[Hypnos] Unknown target '$TARGET'."; usage ;; esac

RC=0
case "$ACTION" in
    build)
        [ "$TARGET" != frontend ] && { build_backend  || RC=1; }
        [ "$TARGET" != backend ]  && { build_frontend || RC=1; }
        ;;
    start)
        [ "$TARGET" != frontend ] && { start_backend  || RC=1; }
        [ "$TARGET" != backend ]  && { start_frontend || RC=1; }
        ;;
    stop)
        [ "$TARGET" != backend ]  && stop_one frontend "$FRONTEND_PORT"
        [ "$TARGET" != frontend ] && stop_one backend "$BACKEND_PORT"
        ;;
    restart)
        [ "$TARGET" != backend ]  && stop_one frontend "$FRONTEND_PORT"
        [ "$TARGET" != frontend ] && stop_one backend "$BACKEND_PORT"
        [ "$TARGET" != frontend ] && { start_backend  || RC=1; }
        [ "$TARGET" != backend ]  && { start_frontend || RC=1; }
        ;;
    status)
        [ "$TARGET" != frontend ] && status_one backend "$BACKEND_PORT"
        [ "$TARGET" != backend ]  && status_one frontend "$FRONTEND_PORT"
        ;;
    logs)
        logs
        ;;
    help|-h|--help)
        usage
        ;;
    *)
        echo "[Hypnos] Unknown action '$ACTION'."
        usage
        ;;
esac
exit "$RC"
