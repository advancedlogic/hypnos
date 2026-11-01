# Security policy

## Supported versions

Security fixes go into the latest release. Before 1.0, only the newest minor version (today
0.7) gets them; upgrade to it to get a fix.

| Component | Supported |
| :--- | :--- |
| The server (`hypnos`: release archives, the Docker image, a build from `main`) | latest release |
| The desktop app | latest release |
| The client SDKs (`sdk/python`, `sdk/javascript`, `sdk/rust`) | latest release |

## Reporting a vulnerability

Please report it privately, not in a public issue: use GitHub's private vulnerability reporting
on this repository (**Security > Report a vulnerability**, or
<https://github.com/advancedlogic/hypnos/security/advisories/new>).

Include what you can of:

- the version (`GET /health` or `hypnos --version`) and the platform;
- the steps or a request that shows the problem, and what an attacker gains (read files, run
  code, bypass the API key, exhaust memory, …);
- whether it needs a valid API key, network access to the server, or a crafted model file.

What to expect: an acknowledgement within 7 days, an assessment and a plan within 30, and a fix
in a release as soon as it is ready, with credit in the release notes and the advisory unless you
prefer otherwise. Please give us that time before you publish anything.

## Scope

In scope: the server and its HTTP API (authentication, quotas, request handling, file and URL
fetching, the model pool routes), the desktop app (its commands, the keychain, local server
handling, downloads), the SDKs, the release artifacts and the CI that builds them.

Also report problems in the bundled engines (llama.cpp, stable-diffusion.cpp, ONNX Runtime,
espeak-ng, docling.rs) that Hypnos exposes, for example a crafted GGUF or image that crashes or
corrupts the server through its API; we will coordinate with the upstream project.

Out of scope: what a model says or generates; denial of service by legitimate heavy use within
the configured limits; issues that need an attacker who already controls the server's machine,
its configuration file or its models folder.

## Running it safely

- Hypnos listens on `127.0.0.1` by default. Before exposing it to a network, set API keys
  (`server.api_keys` or `[[server.keys]]` with limits), turn on TLS (`[server.tls]`) or put it
  behind a reverse proxy that does, and keep the admin routes (the model pool, the log level) to
  admin keys.
- Leave `limits.allow_private_urls` off unless the server must fetch from your own network: it
  guards the URL fetches of documents and images against requests to loopback, private and
  link-local addresses.
- Only load models you trust: a model file is parsed by native code.
- [docs/operating.md](docs/operating.md) is the operator's guide;
  [docs/security-review-2026-11-01.md](docs/security-review-2026-11-01.md) is the latest review.
