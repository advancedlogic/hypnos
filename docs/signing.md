# Signing and notarizing for macOS

This guide covers signing and notarizing the two things Hypnos ships for macOS:

| Artifact | What it is | Status today |
| :--- | :--- | :--- |
| `hypnos-desktop-<version>-macos-arm64.dmg` | the Tauri desktop app (`desktop/`), with the server inside (DESK-15) | signing + notarization wired into `release.yml`, on once the secrets exist |
| `hypnos-<version>-macos-arm64-metal.tar.gz` | the server (`hypnos` + `libstable-diffusion.dylib` + data) | `backend/scripts/sign-macos.sh`, run by `release.yml` with the same secrets |

If a file is downloaded with a browser and has no Developer ID signature plus notarization,
macOS quarantines it. Gatekeeper then blocks it: the user has to choose System Settings >
Privacy & Security > Open Anyway, or run `xattr -dr com.apple.quarantine`. Signing proves who
built the code. Notarization is Apple's automated malware scan, and it returns a ticket that
Gatekeeper checks.

---

## 1. Prerequisites (once)

### 1.1 Apple Developer Program
Only a paid membership ($99/year) can issue **Developer ID** certificates. *Apple Development*
and *Apple Distribution* certificates do not work here: the first is for running on your own
devices, the second is for the App Store.

### 1.2 Developer ID Application certificate
Only the team's **Account Holder** can create one.

- **Xcode:** Settings > Accounts > select the team > Manage Certificates > **+** >
  *Developer ID Application*. Xcode creates the key pair in your login keychain and installs the
  certificate.
- **Or the web portal:** Keychain Access > Certificate Assistant > Request a Certificate From a
  Certificate Authority (save to disk), upload the CSR at
  developer.apple.com > Certificates > **+** > *Developer ID Application*, download the `.cer`
  and double-click it.

Check that it is usable for signing:

```bash
security find-identity -v -p codesigning
#  1) ABCDEF... "Developer ID Application: Your Name (TEAMID1234)"
```

The quoted string is the **signing identity**. The ten characters in parentheses are the
**team ID**. If the list only shows `Apple Development: ...` entries, you do not have the right
certificate yet. Delete revoked entries (`CSSMERR_TP_CERT_REVOKED`) from Keychain Access so
that tools do not pick them by mistake.

### 1.3 Notarization credentials
You need one of these two:

- **Apple ID + app-specific password.** Create the password at account.apple.com > Sign-In and
  Security > App-Specific Passwords. Never use the account password.
- **App Store Connect API key (recommended, what this repository uses).** Create it at App
  Store Connect > Users and Access > Integrations > App Store Connect API > *Team Keys* > **+**,
  with the *Developer* role (the first time, the Account Holder accepts the API terms there).
  You get a `.p8` file that can be downloaded **only once**, a key ID (10 characters) and an
  issuer ID (a UUID above the key list). A key is not tied to an Apple ID password: changing or
  revoking app-specific passwords, the usual source of `401 Invalid credentials`, does not
  touch it. Revoke the key in the same place if it leaks.

For local use, store them once in the keychain as a `notarytool` profile:

```bash
xcrun notarytool store-credentials hypnos-notary \
  --apple-id you@example.com --team-id TEAMID1234 --password xxxx-xxxx-xxxx-xxxx
# or: --key AuthKey_ABC123.p8 --key-id ABC123 --issuer 00000000-0000-0000-0000-000000000000
```

### 1.4 Export the certificate for CI
CI needs the certificate **together with its private key**, as a `.p12` file. There are two
ways to export one:

- **Keychain Access:**
  1. Select the **login** keychain.
  2. Open the **My Certificates** tab.
  3. Expand *Developer ID Application: …* with ▸; the private key must show underneath.
  4. Right-click the certificate row, or select both the certificate and the key, and choose
     Export.
  5. Pick **Personal Information Exchange (.p12)** and set a strong password.
- **Xcode:** Settings > Accounts > the team > Manage Certificates…, then right-click the
  Developer ID Application certificate > **Export Certificate**. This always writes a `.p12`
  with the key.

If Keychain Access offers only `.cer`, `.pem` or `.p7b`, you selected the bare certificate:
the *Certificates* tab, or a row with no key under it. Those formats hold only the public
part, which CI cannot sign with. Use *My Certificates* or Xcode. If no key shows under the
certificate on this Mac, the key is on the Mac that made the CSR: export from there.

`security export -k login.keychain-db -t identities -f pkcs12 -P '<password>' -o cert.p12`
works from the terminal, but it exports **every** identity in the keychain, for example
*Apple Development* ones as well. Prefer one of the two ways above.

Check the file before you upload it. It should list one private key and the Developer ID
certificate:

```bash
openssl pkcs12 -in cert.p12 -info -nokeys -legacy 2>/dev/null | grep -E 'subject=|friendlyName'
```

Keep the `.p12` out of the repository and delete the local copy once the secret is set (2.2).

---

## 2. The desktop app (Tauri)

### 2.1 What is already in the repository

- `desktop/src-tauri/tauri.conf.json`: `bundle.macOS.signingIdentity` is `"-"` (ad-hoc), which
  is what an unsigned build gets, and `hardenedRuntime` is `false`. Under the hardened runtime,
  library validation only loads a library signed by the same team, and an ad-hoc signature has
  no team: the bundled server would fail to load `libstable-diffusion.dylib` ("different Team
  IDs"). CI overrides both with a real identity: `signingIdentity` set and `hardenedRuntime`
  set to `true`.
- `desktop/src-tauri/tauri.server-macos.conf.json` (merged with `--config`) puts the server in
  the bundle: `Contents/MacOS/hypnos` as a sidecar and `Contents/Frameworks/libstable-diffusion.dylib`.
  Tauri signs both in its own pass, with the app's identity, so they share the team and library
  validation passes. The data folders go in `Contents/Resources`, which is not code.
- `desktop/src-tauri/Entitlements.plist`, referenced as `bundle.macOS.entitlements`, holds
  `com.apple.security.device.audio-input`. A Developer ID signature turns on the **hardened
  runtime**, and notarization requires it. The hardened runtime blocks the microphone
  (`mic.rs`, recorded through cpal) unless the app carries this entitlement. If you add a
  feature that needs another protected resource, such as the camera or Apple Events, add its
  entitlement to this file.
- `desktop/src-tauri/Info.plist` has `NSMicrophoneUsageDescription`, the text of the permission
  prompt. You need it as well as the entitlement.
- `.github/workflows/release.yml`, job `desktop`, passes the Apple secrets to `tauri build`.
  For each one that is empty it unsets the variable, because Tauri treats a set-but-empty
  variable as configured. When `APPLE_SIGNING_IDENTITY` is present it also overrides
  `signingIdentity`.

### 2.2 Signing in CI

Add these repository secrets (Settings > Secrets and variables > Actions):

| Secret | Value |
| :--- | :--- |
| `APPLE_CERTIFICATE` | the base64 `.p12` from 1.4 |
| `APPLE_CERTIFICATE_PASSWORD` | its export password |
| `APPLE_SIGNING_IDENTITY` | `Developer ID Application: Your Name (TEAMID1234)` |
| `APPLE_API_KEY` | notarization with an API key (1.3): the key ID |
| `APPLE_API_ISSUER` | the issuer ID |
| `APPLE_API_KEY_P8` | the contents of `AuthKey_<key id>.p8` |
| `APPLE_ID` | notarization with an Apple ID instead: the Apple ID email |
| `APPLE_PASSWORD` | the app-specific password |
| `APPLE_TEAM_ID` | `TEAMID1234` |

The three `APPLE_API_*` secrets win over the Apple ID ones when all three are set. CI writes the
`.p8` to `$RUNNER_TEMP/AuthKey_<key id>.p8` and passes it to `notarytool` (server) and to Tauri
(`APPLE_API_KEY_PATH`, app).

From the terminal, `gh secret set NAME` reads the value from a prompt that does not echo, or
from stdin. That keeps the values out of your shell history and out of any chat or log:

```bash
base64 -i cert.p12 | gh secret set APPLE_CERTIFICATE
gh secret set APPLE_CERTIFICATE_PASSWORD     # prompts
gh secret set APPLE_SIGNING_IDENTITY --body "Developer ID Application: Your Name (TEAMID1234)"
gh secret set APPLE_TEAM_ID --body TEAMID1234
# notarization, API key:
gh secret set APPLE_API_KEY --body ABC123DEFG
gh secret set APPLE_API_ISSUER --body 00000000-0000-0000-0000-000000000000
gh secret set APPLE_API_KEY_P8 < AuthKey_ABC123DEFG.p8
# or Apple ID:  gh secret set APPLE_ID --body you@example.com; gh secret set APPLE_PASSWORD
gh secret list
rm cert.p12
```

With the first three, the app and the server are signed. With the notarization secrets too,
they are also notarized, and the app's ticket is stapled. Tauri notarizes only the app inside
the `.dmg`, so the step **Notarize the disk image (macOS)** then notarizes and staples the
`.dmg` itself; without it a downloaded image assesses as "Unnotarized Developer ID". The release
notes step reads the secrets and says
"The macOS app and server are signed and notarized." Run the workflow by hand first (Actions >
release > Run workflow). A manual run only produces artifacts, so you can check the `.dmg` and
the server archive before you tag.

#### Who can use the key

- **The published app and binaries** carry the signature and the *public* certificate chain,
  which anyone can read. They never contain the private key, so nobody can sign as you by
  copying them.
- **GitHub secrets** are stored encrypted and cannot be read back, by you or by anyone else.
  A workflow that runs in the repository can still *use* them. In this repository:
  - No workflow runs on `pull_request` or `pull_request_target`, so a pull request from a
    fork never receives the secrets.
  - `release.yml` runs only on a `v*` tag or a manual run, and both need write access.
  - On the runner, the `.p12` is decoded into a temporary file that is deleted right after
    import into a temporary keychain, on a VM that GitHub destroys after the job.
- **When you add collaborators** with write access, remember that they can push a workflow
  that prints or sends out secrets. Before that happens, move the `APPLE_*` secrets into a
  GitHub **Environment** (Settings > Environments) with required reviewers, and reference that
  environment from the jobs that sign.
- **If the key leaks,** revoke the certificate at developer.apple.com > Certificates, create a
  new one (1.2) and replace the secrets. Revocation also invalidates what was signed with the
  old certificate, so do it only for a real leak.


### 2.3 Signing locally

The certificate is already in your keychain, so `APPLE_CERTIFICATE` is not needed:

```bash
cd desktop
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID1234)"
# notarization (omit for a signed-only build):
export APPLE_ID=you@example.com APPLE_PASSWORD=xxxx-xxxx-xxxx-xxxx APPLE_TEAM_ID=TEAMID1234
./scripts/stage-server.sh          # the server for the bundle (see desktop/README.md); skip for an app without it
npx tauri build --bundles dmg --config src-tauri/tauri.server-macos.conf.json \
  --config "{\"bundle\":{\"macOS\":{\"signingIdentity\":\"$APPLE_SIGNING_IDENTITY\",\"hardenedRuntime\":true}}}"
```

The second `--config` override is there because `tauri.conf.json` contains `"-"` and no
hardened runtime. Without `hardenedRuntime: true`, notarization refuses the app. Tauri prints the
signing step, then "Notarizing ..." and "Stapling ...". The upload and Apple's scan usually take
1 to 10 minutes. Outputs:

```
desktop/src-tauri/target/release/bundle/macos/Hypnos.app
desktop/src-tauri/target/release/bundle/dmg/Hypnos_<version>_aarch64.dmg
```

`./run.sh build desktop` runs `tauri build --no-bundle`, which produces an unsigned
development build. Use `npx tauri build` as shown above when you want something to distribute.

### 2.4 Verifying

```bash
APP=desktop/src-tauri/target/release/bundle/macos/Hypnos.app
codesign --verify --deep --strict --verbose=2 "$APP"           # valid on disk, satisfies its Designated Requirement
codesign -dv --verbose=4 "$APP" 2>&1 | grep -E 'Authority|TeamIdentifier|flags'
#   Authority=Developer ID Application: ...   flags=0x10000(runtime)
codesign -d --entitlements - "$APP"                             # com.apple.security.device.audio-input = true
codesign -dv "$APP/Contents/MacOS/hypnos" 2>&1 | grep -E 'Authority=Developer|flags'   # the bundled server: same identity, runtime
xcrun stapler validate "$APP"                                   # The validate action worked!
spctl -a -vvv -t exec "$APP"                                    # accepted, source=Notarized Developer ID
spctl -a -vvv -t open --context context:primary-signature desktop/src-tauri/target/release/bundle/dmg/*.dmg
```

If `spctl` rejects the `.dmg` itself while accepting the app inside, notarize and staple the
image as well:

```bash
xcrun notarytool submit desktop/src-tauri/target/release/bundle/dmg/*.dmg --keychain-profile hypnos-notary --wait
xcrun stapler staple desktop/src-tauri/target/release/bundle/dmg/*.dmg
```

To test in practice, download the `.dmg` through a browser on another Mac (or run
`xattr -w com.apple.quarantine "0081;$(printf %x $(date +%s));Safari;" Hypnos.app` on a copy).
Open it and press the microphone button. The first open must not show a Gatekeeper prompt, and
the microphone must show its permission prompt and then record.

---

## 3. The server (`hypnos`)

The server is a command-line binary, not an app bundle. The macOS archive contains two Mach-O
files that need a signature:

```
hypnos                       links @rpath/libstable-diffusion.dylib (rpath = @executable_path)
libstable-diffusion.dylib    sd.cpp, built as a shared library so its ggml does not clash with llama.cpp's
espeak-ng-data/, frontend/, hypnos.toml, scripts/, models.manifest, licenses   (data, not code)
```

All other linkage is system frameworks (Metal, MetalKit, Accelerate, CoreML, Foundation).
ONNX Runtime (Kokoro, docling, GLiNER) is linked statically. Nothing is loaded with `dlopen`
from the archive.

### 3.1 Entitlements: none needed

The hardened runtime enables **library validation**: a process may load only libraries signed
by Apple or by the same team. Signing both files with the same identity satisfies it. The
server needs no entitlements. This was checked on 2026-10-31 on an M5 Max: both files were
signed with `--options runtime` and no entitlements, and `scripts/smoke_test.sh` passed all 11
checks (chat, vision, streaming, transcription, TTS, embeddings, Metal image generation through
the dylib, docling, GLiNER). Do not add `com.apple.security.cs.disable-library-validation` or
the JIT entitlements unless a future engine fails under the hardened runtime and you have
traced it to that cause.

### 3.2 Sign

`backend/scripts/sign-macos.sh` does 3.2 and 3.3 in one go:

```bash
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID1234)"
./backend/scripts/sign-macos.sh                                    # sign only
NOTARY_PROFILE=hypnos-notary ./backend/scripts/sign-macos.sh --notarize
```

It works on `backend/target/release` in place (`--dir` to change). For notarization it takes a
`notarytool` profile, or `APPLE_ID` + `APPLE_PASSWORD` + `APPLE_TEAM_ID`. If Apple rejects
the submission, it prints the notary log and exits non-zero. The manual steps it runs are
below.

Sign the inner library first and the executable last. Do not use `--deep`. `--timestamp`
(Apple's secure timestamp server) and `--options runtime` are both required for notarization.
The linker's default identifier is `hypnos-<hash>`, so give the binary a stable one:

```bash
ID="Developer ID Application: Your Name (TEAMID1234)"
cd backend/target/release        # after: cargo build --release -p hypnos-server --no-default-features --features metal

codesign --force --timestamp --options runtime --sign "$ID" \
  --identifier dev.hypnos.stable-diffusion libstable-diffusion.dylib
codesign --force --timestamp --options runtime --sign "$ID" \
  --identifier dev.hypnos.server hypnos

codesign --verify --strict --verbose=2 hypnos libstable-diffusion.dylib
codesign -dv hypnos 2>&1 | grep -E 'Authority=Developer ID|flags|TeamIdentifier'
```

Sign the files **before** `package-release.sh` runs. The script copies them as they are, and
`tar` keeps the signatures. Any later change to either file breaks its signature: re-signing,
`strip`, `install_name_tool`.

Run the signed binary once (`./hypnos ../../hypnos.macos.example.toml`, or
`smoke_test.sh` against it) before notarizing. A hardened-runtime problem shows up as a crash
at the point where the blocked feature is used, often with `Code Signature Invalid` or
`Library Validation failed` in Console.app. That is quicker to find here than after a
notarization round trip.

### 3.3 Notarize

`notarytool` accepts `.zip`, `.dmg` and `.pkg` files, not `.tar.gz`. Submit a zip of the two
binaries. The archive you publish can stay a `.tar.gz`: the ticket belongs to the code hashes,
not to the container.

```bash
mkdir -p /tmp/hypnos-n && cp hypnos libstable-diffusion.dylib /tmp/hypnos-n/
ditto -c -k --keepParent /tmp/hypnos-n /tmp/hypnos-notarize.zip

xcrun notarytool submit /tmp/hypnos-notarize.zip --keychain-profile hypnos-notary --wait
#   status: Accepted
```

If the status is `Invalid`, read the reason:

```bash
xcrun notarytool log <submission-id> --keychain-profile hypnos-notary
```

### 3.4 Stapling (not possible for bare binaries)

`stapler` works only on `.app`, `.dmg` and `.pkg`. A bare Mach-O file cannot carry a stapled
ticket. On first launch of a quarantined, notarized binary, Gatekeeper looks the ticket up
online. That works for anyone with a network connection. An offline first launch is still
blocked. If that matters, wrap the archive in a notarized and stapled `.dmg` or `.pkg`.

Check the result:

```bash
spctl -a -vvv -t install hypnos                 # source=Notarized Developer ID (needs network)
codesign -dv --verbose=4 hypnos 2>&1 | grep Timestamp
```

### 3.5 Packaging after signing

```bash
cd ../..                                         # repo root
./backend/scripts/package-release.sh macos-arm64-metal --version v0.3.1 --out dist
```

When `MACOS_NOTARIZED=1` is set, `package-release.sh` writes "signed with a Developer ID and
notarized" into the macOS `README.txt`. Otherwise it keeps the `xattr -dr
com.apple.quarantine .` instruction.

### 3.6 CI

In `build-server.yml`, which `release.yml` calls as `build-macos` for the Metal archive, the
`macos-arm64-metal` leg has a step, **Sign and notarize (macOS)**, between **Build** and
**Package**. It reuses the desktop job's secrets:

- Without `APPLE_CERTIFICATE` / `APPLE_SIGNING_IDENTITY`, it skips signing, and the binaries
  keep the linker's ad-hoc signature.
- Otherwise, it imports the `.p12` into a temporary keychain (unlocked, with a partition list
  so that `codesign` never prompts) and runs
  `SIGN_KEYCHAIN=<it> sign-macos.sh`.
- If `APPLE_API_KEY`, `APPLE_API_ISSUER` and `APPLE_API_KEY_P8` are set (or else `APPLE_ID`,
  `APPLE_PASSWORD` and `APPLE_TEAM_ID`), it adds `--notarize` and sets `MACOS_NOTARIZED=1` for
  **Package**. A rejected submission fails the release instead of
  shipping.

The release notes say "The macOS app and server are signed and notarized" when
`APPLE_CERTIFICATE` is set together with `APPLE_API_KEY` or `APPLE_ID`.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| notarization: *The binary is not signed with a valid Developer ID certificate* | signed with *Apple Development* or ad-hoc | use the *Developer ID Application* identity (1.2) |
| *The signature does not include a secure timestamp* | `--timestamp` missing, or `--timestamp=none` | re-sign with `--timestamp` (needs network) |
| *The executable does not have the hardened runtime enabled* | `--options runtime` missing | re-sign with it |
| `codesign` hangs in CI | the keychain is locked, or the key's partition list is not set | `security unlock-keychain` + `set-key-partition-list` as in 3.6 |
| `xcrun: error: unable to find utility "store-credentials"` | the tool name is missing | `xcrun notarytool store-credentials …` |
| Keychain Access exports only `.cer` / `.pem` / `.p7b` | bare certificate selected, without its key | *My Certificates* tab or Xcode (1.4) |
| `codesign` asks for a password locally | first use of the private key by `codesign` | click *Always Allow* |
| `hypnos` dies at startup with *Library Validation failed* | the dylib and the binary are signed by different teams, or the dylib was changed after signing | sign both with the same identity, the dylib first |
| desktop: microphone records silence or no prompt appears | `Entitlements.plist` not applied | `codesign -d --entitlements - Hypnos.app` must list `audio-input` |
| desktop: Tauri skips signing | `signingIdentity` is still `"-"` | pass the `--config` override (2.3) |
| notarization: *The executable does not have the hardened runtime enabled* for `Hypnos.app` | `hardenedRuntime` still `false` from `tauri.conf.json` | add `"hardenedRuntime": true` to the override (2.3) |
| bundled server dies: *Library not loaded: @rpath/libstable-diffusion.dylib … different Team IDs* | ad-hoc signatures with the hardened runtime on | keep `hardenedRuntime: false` for ad-hoc builds; with a Developer ID both files get the same team |
| Tauri: *skipping app notarization* | the `APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID` (or `APPLE_API_*`) set is incomplete | export all three |
| Gatekeeper still prompts on a notarized binary | first launch offline (no stapled ticket on bare Mach-O) | connect once, or ship inside a stapled `.dmg` / `.pkg` |
| *E0463 can't find crate* after a "strip" step you added | Xcode's `strip` damages proc-macro dylibs (BUG-5) | never strip build-script outputs; strip only the final binary, **before** signing |

## 5. Certificate lifetime

Developer ID certificates last five years. Code signed with `--timestamp` stays valid after the
certificate expires. Revoking the certificate invalidates every signature made with it, so
revoke only if the key has leaked. When you renew, update `APPLE_CERTIFICATE`,
`APPLE_CERTIFICATE_PASSWORD` and, if the name changed, `APPLE_SIGNING_IDENTITY`.
