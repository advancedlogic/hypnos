# Hypnos desktop

A native chat app for a running Hypnos server, in the spirit of the Ollama app: pick a model,
chat with streaming answers, and keep a panel open with what the server is doing (which model
sits in which memory tier, how much of each tier is used, what the last answer cost).

It does not contain the server. Point it at a running Hypnos (`./run.sh start backend`, a
release archive, the Docker image, another machine), or let it start an installed `hypnos` on
this machine from its Local server dialog.

Built with Tauri 2, Svelte 5 (runes) and Tailwind 3, the same web stack as `frontend/`. It talks
to the server through the JavaScript SDK in `sdk/javascript` (compiled from its sources, no
separate build step).

## Using it

- **Chat.** The model picker lists the enabled models that do text generation, with their
  current tier. Enter sends, Shift+Enter adds a line. Stop cuts an answer short (the server
  stops at the next token); Regenerate replaces the last answer. Each chat keeps its own model and
  system prompt. Chats are saved on this machine.
- **Under each answer:** decode speed, time to first token, tokens, and the tier the model was
  loaded from when it was not on the GPU already.
- **Stats panel** (the right side, toggled with the header button or Cmd/Ctrl+.): server version
  and device; memory (one unified bar on Apple Silicon, Tier 1 VRAM and Tier 2 pinned RAM on a
  discrete GPU, Tier 3 bytes, process memory); the enabled models grouped by tier with active
  requests, partial GPU placements, requests served and idle time; the last answer's full
  telemetry including the load phases; the latest evictions with their reason. It refreshes
  every 1.5 s while open and the window is visible.
- **Connection** (click the status in the header, or Cmd/Ctrl+,): server URL and, if the server
  sets `server.api_keys`, an API key. The key is kept in the system keychain (macOS Keychain,
  Windows Credential Manager, Secret Service on Linux), one per server. The header shows
  Connecting, Unreachable, API key refused, Warming up (models still loading) or Ready, and the
  app reconnects by itself when the server comes back.

- **Local server** (the server button in the header): runs an installed `hypnos` on this
  machine and connects the app to it. The binary is detected (this checkout's
  `backend/target/release`, a `hypnos` beside the app, then PATH) or entered by hand, with an
  optional config (its folder becomes the working directory, so its relative model paths work;
  without one the server reads `hypnos.toml` beside the binary) and a port. The dialog shows the
  server's output live; "Start with the app" starts it at launch unless a server already answers
  on that port, in which case the app just connects to it. The app stops the server it started
  (gracefully, SIGTERM on Linux/macOS) when you quit, and never stops one it did not start.

| Shortcut | Action |
| :--- | :--- |
| Cmd/Ctrl+N | new chat |
| Cmd/Ctrl+. | show or hide the stats panel |
| Cmd/Ctrl+, | connection settings |

## Installing

Each [GitHub release](https://github.com/advancedlogic/hypnos/releases) after v0.2.0 carries
`hypnos-desktop-<version>-macos-arm64.dmg`, `-windows-x64.msi` / `.setup.exe` and
`-linux-x64.deb` / `.AppImage`. They are not signed with a developer identity yet: on macOS
allow the first open in System Settings > Privacy & Security > Open Anyway; on Windows choose
More info > Run anyway.

## Building from source

Prerequisites, besides Node.js 18+ and Rust (rustup stable):

- **macOS:** the Xcode command line tools.
- **Windows:** the MSVC build tools and WebView2 (preinstalled on Windows 10/11).
- **Linux:** `sudo apt install build-essential pkg-config libssl-dev libwebkit2gtk-4.1-dev
  libayatana-appindicator3-dev librsvg2-dev patchelf` (or your distribution's equivalents; see
  the [Tauri prerequisites](https://v2.tauri.app/start/prerequisites/)).

```bash
# from the repository root
./run.sh build desktop     # npm install + tauri build --no-bundle
./run.sh start desktop     # open the built app (stop / status / logs desktop work too)

# development, with hot reload of the web side
cd desktop
npm install
npx tauri dev              # Vite on port 1420, the app window opens on it
npx tauri build            # installers for this OS under src-tauri/target/release/bundle/
```

`run.bat build desktop` / `run.bat start desktop` do the same on Windows. The `desktop` target is
never part of `all`. `HYPNOS_PROFILE=debug` builds and starts the debug binary.

`npm run dev` alone serves the UI in a browser on http://localhost:1420. It then uses the
browser's fetch, which the server's default CORS policy accepts from loopback origins; the API key
is kept in memory only.

## How it reaches the server

The webview's origin (`tauri://localhost`, `http://tauri.localhost` on Windows) is not a loopback
origin, so the server's default CORS policy would refuse it. Requests therefore go through
Tauri's HTTP plugin, which sends them from the Rust side. Its scope allows `localhost` and
`127.0.0.1` (`src-tauri/capabilities/default.json`); a server elsewhere is allowed when it is
saved in the connection settings (the `allow_server` command adds a runtime capability for that
origin only, kept until the app quits).

## Layout

```text
desktop/
├── src/
│   ├── App.svelte            # shell: header, status loop, model list, panels
│   └── lib/
│       ├── hypnos.js         # SDK client over the Tauri HTTP plugin, server probe
│       ├── connection.js     # server URL (store) + API key (keychain)
│       ├── chats.svelte.js   # ChatStore: chats, streaming, stop, persistence
│       ├── Chat.svelte       # model picker, messages, composer
│       ├── Sidebar.svelte    # conversation list
│       ├── StatsPanel.svelte # memory tiers, models, last-response telemetry, evictions
│       ├── Settings.svelte   # connection dialog
│       ├── LocalServer.svelte # start/stop an installed hypnos, live log
│       ├── localServer.js    # its commands, events and saved settings
│       └── Markdown.svelte   # marked + DOMPurify, copy buttons on code blocks
└── src-tauri/
    ├── src/lib.rs            # plugins (http, store) + commands: allow_server, get/set_api_key
    ├── src/local_server.rs   # find, start, stop and log a hypnos process (health via hypnos-client)
    ├── capabilities/         # what the window may call; HTTP limited to loopback
    ├── icons/                # generated from app-icon.png (assets/logo.png on a tile)
    └── tauri.conf.json
```

Tests: `cd src-tauri && cargo test`. The local-server test runs a real server and is ignored by
default: `HYPNOS_TEST_BINARY=../../backend/target/release/hypnos HYPNOS_TEST_CONFIG=<config with
one small model> cargo test -- --ignored` (it uses port 9997).

To regenerate the icons after changing `src-tauri/icons/app-icon.png`:
`npx tauri icon src-tauri/icons/app-icon.png -o src-tauri/icons`, then remove the `android/`,
`ios/` and Windows Store (`Square*`, `StoreLogo`) files it adds.
