# Hypnos desktop

A native chat app for a running Hypnos server, in the spirit of the Ollama app: pick a model,
chat with streaming answers, and keep a panel open with what the server is doing (which model
sits in which memory tier, how much of each tier is used, what the last answer cost).

It does not contain the server. Point it at a running Hypnos (`./run.sh start backend`, a
release archive, the Docker image, another machine), or let it start an installed `hypnos` on
this machine from its Local server dialog.

Built with Tauri 2, Svelte 5 (runes) and Tailwind 3, the same web stack as `frontend/`. Every call
to the server is made by the Rust side with the Rust SDK (`sdk/rust`); the web UI only calls the
app's commands.

## Using it

- **Chat.** The model picker lists the enabled models that do text generation, with their
  current tier. Enter sends, Shift+Enter adds a line. Stop cuts an answer short (the server
  stops at the next token); Regenerate replaces the last answer. Each chat keeps its own model and
  system prompt. Chats are saved on this machine.
- **Attachments** (the paperclip, drag and drop onto the chat, or paste): each file is prepared
  when it is added and shows as a chip until it is ready. Images are scaled to 1024 px and sent
  to a model that sees (when the chat's model cannot, another one answers and the answer says
  so). PDF, DOCX, PPTX, XLSX and HTML go through docling and arrive as markdown; text and code
  files are read as they are (both cut at about 60,000 characters with a note). Audio (WAV,
  MP3, M4A/AAC, FLAC, Ogg Vorbis) is decoded, transcribed and sent as its transcript. Video
  (MP4, MOV, MKV, WebM, as far as the system's webview decodes them) becomes six frames plus
  the transcript of its soundtrack. Click a chip to read the extracted text. Attachments are
  saved with the chat.
- **Links** typed in the composer get a "Read" chip: the app fetches the page or file itself
  (http/https, up to 50 MiB, 30 s), converts HTML, PDF and office files with docling (plain text
  as is) and adds it to the message with its title and source. Open the chip to Copy it,
  Download it as .md, or download the original file again.
- **Microphone** (next to Send): click to start and click again (or Done) to stop, or hold it
  and release to talk. A strip shows the time and the input level; Esc or Cancel throws the
  recording away. The audio is transcribed by the speech-to-text model and the text lands in
  the composer for review, or is sent right away with "send when done" (remembered). Recording
  stops by itself after 5 minutes. macOS asks for microphone access the first time; if it was
  refused, allow Hypnos in System Settings > Privacy & Security > Microphone.
- **Speaker** (the chat header): with it on, each new answer is read aloud as it streams,
  sentence by sentence, on the text-to-speech model (Kokoro); playback starts after the first
  sentence. "Read aloud" under any answer reads it on demand; sending a message, switching
  chats or "Stop reading" stops it. Code blocks are announced ("Code block.") rather than
  read, links are read as their text. Voice and speed are in the connection dialog.
- **Images**: ask the chat model for a picture ("draw a lighthouse at dawn") and it calls the
  `generate_image` tool with a prompt for the image model (MOSS-Image), which the app runs (a
  real OpenAI tool call on servers with tool calling, API-9; a prompt convention for models
  without a native tool format); the image appears in the answer
  with the prompt, Save, Copy and Regenerate, followed by a one-line comment. The Image button
  in the composer (or `/image <prompt>`) skips the chat model. A generation takes from tens of
  seconds to a few minutes, longer when the image model must load first, which may unload the
  chat model; the answer shows the elapsed time and where the model is loading from. The
  connection dialog can turn the chat model's access to images off.
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
  Windows Credential Manager, Secret Service on Linux), one per server. Below them, the model for
  each job (images in messages, speech to text, text to speech, image generation, documents and
  URLs): Automatic uses the chat's model when it serves that job, else a model already on the
  GPU, else any enabled model that serves it. The header shows
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

- **Menu bar / tray icon**: the logo with a dot for the server's state (green ready, amber
  loading, red unreachable or key refused, grey connecting). A left click shows or hides the
  window; the menu (right click) has the status and version, Open Hypnos, Stop the server /
  Start the server, Connection settings and Quit. "Stop the server" stops the server the app is
  connected to whenever it runs on this machine, also one started by `run.sh` or by hand (only a
  process named `hypnos` listening on that port, gracefully); "Start the server" starts the
  local server (Local server dialog) on the connection's port; a remote server cannot be
  stopped from here. Closing the window keeps the app in the tray (the connection dialog can
  make it quit instead, and on macOS hide the Dock icon); a notification tells you when an
  answer or image finishes while the window is hidden or in the background.

| Shortcut | Action |
| :--- | :--- |
| Cmd/Ctrl+N | new chat |
| Cmd/Ctrl+. | show or hide the stats panel |
| Cmd/Ctrl+, | connection settings |

## Installing

Each [GitHub release](https://github.com/advancedlogic/hypnos/releases) after v0.2.0 carries
`hypnos-desktop-<version>-macos-arm64.dmg`, `-windows-x64.msi` / `.setup.exe` and
`-linux-x64.deb` / `.AppImage`. The release notes say whether they are signed. Unsigned, on
macOS allow the first open in System Settings > Privacy & Security > Open Anyway; on Windows
choose More info > Run anyway.

Signing turns on in `release.yml` once these repository secrets exist (Settings > Secrets and
variables > Actions):

| Secret | What |
| :--- | :--- |
| `APPLE_CERTIFICATE` | the Developer ID Application certificate exported as .p12, base64 (`base64 -i cert.p12`) |
| `APPLE_CERTIFICATE_PASSWORD` | the .p12 export password |
| `APPLE_SIGNING_IDENTITY` | `Developer ID Application: Your Name (TEAMID)` |
| `APPLE_ID`, `APPLE_PASSWORD`, `APPLE_TEAM_ID` | notarization: the Apple ID, an app-specific password, the team id |
| `WINDOWS_CERTIFICATE` | the code-signing certificate as .pfx, base64 |
| `WINDOWS_CERTIFICATE_PASSWORD` | its password |

Without the Apple secrets the macOS app keeps an ad-hoc signature; without the Windows ones the
installers are unsigned.

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

The UI needs the Tauri side for every server call, so it does not run in a plain browser.

## How it reaches the server

`src-tauri/src/hypnos.rs` holds one `hypnos_client::Client` for the current server (its URL from
the store, its key from the keychain) and exposes commands: `hypnos_connect`, `hypnos_probe`,
`hypnos_test`, `hypnos_models`, `hypnos_memory_status`, `hypnos_route`, `hypnos_set_routes`,
`hypnos_chat` (tokens stream back over a Tauri `Channel`) and `hypnos_chat_cancel` (drops the
stream, which closes the connection, which is how the server is told to stop). Because the
requests leave from Rust, the server's CORS policy and the webview's origin play no part, and
the webview has no HTTP permission at all. Failed commands reject with
`{ kind, message, detail, retryAfterSecs }`, where `message` is the sentence the UI shows.

`route` picks the model for a job from `/v1/memory/status` `modalities`, never from hard-coded
ids: the chat's model when it serves the modality, then the model chosen in the settings, then
an enabled model that serves it, preferring one already in Tier 1.

## Layout

```text
desktop/
├── src/
│   ├── App.svelte            # shell: header, status loop, model list, panels
│   └── lib/
│       ├── hypnos.js         # the server commands (connect, probe, models, chat, route, ...)
│       ├── connection.js     # server URL (store) + API key (keychain), then hypnos_connect
│       ├── chats.svelte.js   # ChatStore: chats, streaming, stop, persistence
│       ├── Chat.svelte       # model picker, messages, composer
│       ├── Sidebar.svelte    # conversation list
│       ├── StatsPanel.svelte # memory tiers, models, last-response telemetry, evictions
│       ├── Settings.svelte   # connection dialog
│       ├── LocalServer.svelte # start/stop an installed hypnos, live log
│       ├── localServer.js    # its commands, events and saved settings
│       ├── appPrefs.js       # tray / Dock / notification preferences, notifications
│       ├── Attachments.svelte # attachment chips (composer and messages)
│       ├── video.js          # samples video frames in the webview
│       └── Markdown.svelte   # marked + DOMPurify, copy buttons on code blocks
└── src-tauri/
    ├── src/lib.rs            # store plugin, keychain commands (get/set_api_key), command list
    ├── src/hypnos.rs         # every server call on hypnos-client, the model router, errors
    ├── src/attachments.rs    # image resize, docling, audio decode (symphonia) + transcription
    ├── src/web.rs            # fetch a URL, convert it, save markdown or the original
    ├── src/mic.rs            # record with cpal on its own thread, then transcribe
    ├── src/tts.rs            # markdown to speech text, sentence queue, rodio playback
    ├── src/images.rs         # the image tool: tool spec, prompt fallback (ToolGate), generation, save
    ├── src/tray.rs           # tray icon with a status dot, its menu, close-to-tray, Dock icon
    ├── Info.plist            # macOS: NSMicrophoneUsageDescription
    ├── src/local_server.rs   # find, start, stop and log a hypnos process (health via hypnos-client)
    ├── capabilities/         # what the window may call: core + store, no HTTP
    ├── icons/                # generated from app-icon.png (assets/logo.png on a tile)
    └── tauri.conf.json
```

Tests: `cd src-tauri && cargo test`. Two tests need a real server and are ignored by default:
`HYPNOS_TEST_URL=http://127.0.0.1:9999 cargo test hypnos -- --ignored` chats, cancels and checks
the errors through the commands; `HYPNOS_TEST_URL=... HYPNOS_TEST_FILES=<dir with jfk.m4a and a
one-page sample.pdf containing "pelican"> cargo test attachments -- --ignored` prepares real
attachments and asks a vision model about them; `cargo test web -- --ignored` (same URL, needs
the network) reads example.com and a PDF and saves both; `cargo test mic -- --ignored` records
1.5 s from the default microphone; `cargo test tts -- --ignored` synthesizes and decodes speech
(without playing it); `cargo test images::tests::the_chat -- --ignored` asks for a picture and
runs the whole tool loop (it loads the image model, a minute or more); the local-server test
starts one itself: `HYPNOS_TEST_BINARY=../../backend/target/release/hypnos HYPNOS_TEST_CONFIG=<config with
one small model> cargo test -- --ignored` (it uses port 9997).

To regenerate the icons after changing `src-tauri/icons/app-icon.png`:
`npx tauri icon src-tauri/icons/app-icon.png -o src-tauri/icons`, then remove the `android/`,
`ios/` and Windows Store (`Square*`, `StoreLogo`) files it adds.
