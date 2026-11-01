# Contributing to Hypnos

Thanks for helping. This page says how to build it, test it and send a change. The
[README](README.md) says what Hypnos is; [CLAUDE.md](CLAUDE.md) is the detailed map of the code
(the crates, the request lifecycle, the placement model, every engine) and is worth reading
before a larger change.

## Getting it to build

You need Rust 1.92+ through rustup, Node.js 18+, CMake and Ninja, and for GPU builds the CUDA
toolkit (Linux, Windows) or Xcode's command line tools (macOS).

```bash
# once per machine: GPU architecture, compilers, the rustc pin (writes the git-ignored .cargo/config.toml)
./backend/scripts/setup-build-env.sh            # Windows: powershell -ExecutionPolicy Bypass -File backend\scripts\setup-build-env.ps1

# the engine sources (git-ignored, over 1 GB) from their pinned commits and patches
./backend/scripts/vendor.sh

./run.sh build backend                          # the server (the first build compiles every engine: 10-20 min)
./run.sh build all                              # server, playground and desktop app
./run.sh start backend                          # port 9999; ./run.sh models downloads the default models
```

A build without a GPU: `cd backend && cargo build --workspace --no-default-features`. On a Mac:
`--no-default-features --features metal`.

## Testing

The tests run without model weights or a GPU (mock engines):

```bash
cd backend && cargo test --workspace --no-default-features   # add --features hypnos-server/metal on a Mac
cd sdk/python && python -m pytest                            # the SDKs, against the mock server
cd sdk/javascript && npm test
cd sdk/rust && cargo test
cd desktop/src-tauri && cargo test
```

The parsers that read untrusted input (model file headers, the config and catalog, the API's
JSON, the tool-call formats) have fuzz targets in `backend/fuzz/` (nightly Rust and `cargo install
cargo-fuzz`): `cd backend && cargo +nightly fuzz run gguf_header -- -max_total_time=300`. CI runs
each for a minute. A crash leaves its input in `backend/fuzz/artifacts/`: turn it into a
regression test next to the parser.

Tests that need real models are `#[ignore]`d and say in their doc comment what they need (a
running server in `HYPNOS_TEST_URL`, files in `backend/models/`). Run the ones your change
touches. `./backend/scripts/smoke_test.sh` sends one real request per modality to a running
server.

## Before you send a change

```bash
cd backend && cargo fmt -p hypnos-core -p hypnos-memory -p hypnos-scheduler -p hypnos-backends -p hypnos-server -- --check
cd backend && cargo clippy -p hypnos-core -p hypnos-memory -p hypnos-scheduler --all-targets -- -D warnings
cd desktop/src-tauri && cargo fmt --check && cargo clippy --all-targets -- -D warnings
cargo deny --manifest-path backend/Cargo.toml --config deny.toml check   # the dependency policy
```

CI runs these, the audit, the SDK tests and the Linux build on every pull request; a release (and
a manual run) also builds and tests on Windows and macOS.

Some changes come in sets, so that nothing falls out of step:

- **An API change** (a route, a field, an error type, a stream event) goes into
  [docs/api.md](docs/api.md), the mock server (`backend/crates/hypnos-server/examples/mock_server.rs`)
  and all three SDKs with tests. In Python, edit `_async_client.py` and regenerate the sync
  client with `sdk/python/scripts/unasync.py`.
- **A new config key, metric or error type** goes into [docs/operating.md](docs/operating.md).
- **A desktop UI change** goes into [docs/desktop-manual.md](docs/desktop-manual.md), with the
  screenshots it changes retaken by `desktop/screenshots/capture.mjs` (see `desktop/README.md`).
- **A new model** goes into `backend/models.catalog.toml` with its measured memory (`hypnos
  measure-memory`), and a new GGUF architecture needs engine work first (prompt format, media
  markers, thinking): see CLAUDE.md, "Model catalog".
- **Vendored engine sources** (`backend/vendor/`) are never edited without regenerating the
  matching file in `backend/vendor/patches/`; the pins live in `backend/scripts/vendor.sh`.
- **A GitHub Action** is pinned to a commit SHA with its version in a comment.

## The card board

Work is tracked as cards in [TODO.md](TODO.md) (`UI-n`, `API-n`, `ENG-n`, `MEM-n`, `PERF-n`,
`BUG-n`, `BUILD-n`, `DOC-n`, `TEST-n`, `DESK-n`; priorities `P0`-`P3`; its Conventions section has
the format). A card moves from Backlog through In Progress to Done, and a done card says what was
done and how it was checked. Open an issue first for anything large, so we can agree on the
approach.

## Commits

One change per commit, with a message in this form:

```
feat(desktop): the Models dialog's In use tab (DESK-29)

What changed and why, in a few lines: the problem, the approach, how it was tested.
```

The prefix is `feat`, `fix`, `perf`, `docs`, `build`, `test`, `refactor` or `style`, the scope
the area (`server`, `engine`, `memory`, `desktop`, `sdk`, `ci`, …), and the card id at the end.

## Licensing

By contributing you agree that your contribution is licensed under the project's terms, MIT OR
Apache-2.0 at your choice (the `LICENSE-*` files). Note that the server binaries built with Kokoro
link espeak-ng statically and are distributed under the GPL v3 or later as a whole (README,
"License"): do not add other GPL code, and `deny.toml` keeps that one exception explicit.
