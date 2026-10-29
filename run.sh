#!/usr/bin/env bash
# ============================================================================
#  Hypnos process manager (Linux / macOS / Git Bash)
#
#    ./run.sh [start|stop|restart|status|logs] [backend|frontend|all]
#
#    ./run.sh                  -> start all (background, logs in .run/)
#    ./run.sh restart backend  -> rebuild + restart only the Rust backend
#    ./run.sh stop             -> stop backend and frontend
#    ./run.sh logs             -> follow both logs
#
#  Backend : Rust server on http://localhost:8080  (cargo run)
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
BACKEND_CONFIG="hypnos.example.toml"

ACTION="${1:-start}"
TARGET="${2:-all}"

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

case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*) IS_WINDOWS=1 ;;
    *) IS_WINDOWS=0 ;;
esac

usage() {
    cat <<EOF

  Usage: ./run.sh [start|stop|restart|status|logs] [backend|frontend|all]

    ./run.sh                    start backend + frontend (default)
    ./run.sh start backend      start only the Rust backend (port $BACKEND_PORT)
    ./run.sh restart frontend   restart only the Vite dev server (port $FRONTEND_PORT)
    ./run.sh stop               stop both
    ./run.sh status             show what is running
    ./run.sh logs [target]      tail the log file(s) in .run/

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
        # The server binary may outlive cargo if the console was interrupted mid-run.
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
# services
# ---------------------------------------------------------------------------
start_backend() {
    local pid
    if pid="$(is_running backend "$BACKEND_PORT")"; then
        echo "[Hypnos] backend already running (pid $pid, port $BACKEND_PORT)."
        return 0
    fi
    echo "[Hypnos] backend starting (cargo run; the first run compiles the native engines, which takes a while)"
    echo "         log: $RUN_DIR/backend.log"
    spawn backend "$ROOT_DIR/backend" "${CARGO_CMD[@]}" run --bin hypnos -- "$BACKEND_CONFIG"
    wait_up backend "$BACKEND_PORT" 5
}

start_frontend() {
    local pid
    if pid="$(is_running frontend "$FRONTEND_PORT")"; then
        echo "[Hypnos] frontend already running (pid $pid, port $FRONTEND_PORT)."
        return 0
    fi
    if [ ! -d "$ROOT_DIR/frontend/node_modules" ]; then
        echo "[Hypnos] Installing frontend dependencies..."
        (cd "$ROOT_DIR/frontend" && npm install) || { echo "[Hypnos] npm install failed."; return 1; }
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
case "$TARGET" in backend|frontend|all) ;; *) echo "[Hypnos] Unknown target '$TARGET'."; usage ;; esac

case "$ACTION" in
    start)
        [ "$TARGET" != frontend ] && start_backend
        [ "$TARGET" != backend ] && start_frontend
        ;;
    stop)
        [ "$TARGET" != backend ] && stop_one frontend "$FRONTEND_PORT"
        [ "$TARGET" != frontend ] && stop_one backend "$BACKEND_PORT"
        ;;
    restart)
        [ "$TARGET" != backend ] && stop_one frontend "$FRONTEND_PORT"
        [ "$TARGET" != frontend ] && stop_one backend "$BACKEND_PORT"
        [ "$TARGET" != frontend ] && start_backend
        [ "$TARGET" != backend ] && start_frontend
        ;;
    status)
        [ "$TARGET" != frontend ] && status_one backend "$BACKEND_PORT"
        [ "$TARGET" != backend ] && status_one frontend "$FRONTEND_PORT"
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
exit 0
