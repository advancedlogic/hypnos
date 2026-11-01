# Security review, 2026-11-01 (TEST-4)

Scope: the HTTP surface added since v0.4.0 (runtime model pool, catalog, advisor, deadlines and
workflow headers) and the long-standing risks it touches: authentication, outbound fetches, file
paths, logging. Reviewed by reading the code paths and testing the fixes; not an external audit.

## Findings and fixes

| # | Area | Finding | Severity | Status |
| ---: | :--- | :--- | :--- | :--- |
| 1 | Remote image / document URLs (`media.rs`) | With `allow_remote_image_urls` (on by default) the server fetched any URL and followed redirects: a client could make it read `http://169.254.169.254/...` (cloud metadata), `localhost` admin ports or LAN hosts and see the result through the model's answer (SSRF). | High on a shared server | Fixed: a ureq resolver drops loopback, private, link-local, CGNAT, documentation, reserved and multicast addresses at every connection, redirects included (so DNS that changes between a check and the connection does not help). `limits.allow_private_urls = true` restores LAN fetches. Tests: `only_public_addresses_count_as_public`, `remote_fetches_do_not_reach_private_addresses`. |
| 2 | `POST` / `PUT /v1/models` (`pool.rs`) | Paths were confined to `models_dir` only when one was configured; without it a block could name any file the process can read (admin only, loaded as model weights). | Medium | Fixed: without `models_dir` the working directory is the bound. Paths are canonicalized, so symlinks pointing outside are refused too. |
| 3 | Binding beyond loopback | Nothing warned when `server.host` exposed the server with no API keys; with the runtime pool that means anyone on the network can load and remove models. | Medium | Fixed: a startup warning names the host and what to set. |
| 4 | Authentication on new routes | `/v1/models/catalog`, `/v1/advisor`, the pool routes and the headers all sit behind `auth::require_api_key`; the pool routes additionally need an admin key (`admin = true`, plain `api_keys`). `open_while_staging` only lifts the readiness gate, never authentication. | - | No issue. |
| 5 | Request log and headers | `x-hypnos-next-model`, `x-hypnos-deadline-ms` and `x-hypnos-workflow` are not logged with their values except the workflow name as a metric label (truncated to 64 characters). Prompts and outputs keep their `request_log_redact` rules. | Low | A client could create many workflow names (one metric series each): bounded by what an authenticated client can send; noted. |
| 6 | Overlay file (`<config>.models.toml`) | Written next to the config with the process's permissions; holds model blocks only (no keys). | - | No issue; documented as part of the config to back up. |
| 7 | Desktop app URL fetches (`web.rs`) | Fetches any URL the user pastes, on the user's own machine. | - | Out of scope: no remote party can trigger it. |

## Open

- An external review or fuzzing of the multipart and JSON parsers (axum, the size limits) was
  not done.
- Workflow names as metric labels (finding 5) could be limited to a configured list.
- TLS configuration is the operator's; nothing new here.
