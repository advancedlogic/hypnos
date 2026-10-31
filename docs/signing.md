# Signing and notarizing for macOS

This guide covers signing and notarizing the two things Hypnos ships for macOS:

| Artifact | What it is | Status today |
| :--- | :--- | :--- |
| `hypnos-desktop-<version>-macos-arm64.dmg` | the Tauri desktop app (`desktop/`) | signing + notarization wired into `release.yml`, on once the secrets exist |
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
- **App Store Connect API key.** Create it at App Store Connect > Users and Access >
  Integrations > App Store Connect API, with the *Developer* role. You get a `.p8` file, a key ID
  and an issuer ID. A key does not expire with a password change, and it suits CI better.

For local use, store them once in the keychain as a `notarytool` profile:

```bash
xcrun notarytool store-credentials hypnos-notary \
  --apple-id you@example.com --team-id TEAMID1234 --password xxxx-xxxx-xxxx-xxxx
# or: --key AuthKey_ABC123.p8 --key-id ABC123 --issuer 00000000-0000-0000-0000-000000000000
```

### 1.4 Export the certificate for CI
In Keychain Access, open *My Certificates*, right-click the Developer ID Application
certificate (expand it to check that the private key is underneath), choose Export, then
`.p12` with a strong password. Then:

```bash
base64 -i DeveloperID.p12 | pbcopy     # paste as the APPLE_CERTIFICATE secret
```

Keep the `.p12` out of the repository and delete the local copy once the secret is set.

---

## 2. The desktop app (Tauri)

### 2.1 What is already in the repository

- `desktop/src-tauri/tauri.conf.json`: `bundle.macOS.signingIdentity` is `"-"` (ad-hoc), which
  is what an unsigned build gets. CI overrides it with the real identity.
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
| `APPLE_ID` | the Apple ID email |
| `APPLE_PASSWORD` | the app-specific password |
| `APPLE_TEAM_ID` | `TEAMID1234` |

With the first three, the app is signed. With all six, it is also notarized and the ticket is
stapled. The release notes step reads the secrets and says "The macOS app is signed and
notarized." Run the workflow by hand first (Actions > release > Run workflow): a manual run
only produces artifacts, so you can check the `.dmg` before you tag.

To use an API key instead of an Apple ID, Tauri reads `APPLE_API_KEY` (key ID),
`APPLE_API_ISSUER` and `APPLE_API_KEY_PATH` (a path to the `.p8`). `release.yml` does not
pass those yet. You would add a step that writes the `.p8` from a secret into
`$RUNNER_TEMP` and exports the three variables.

### 2.3 Signing locally

The certificate is already in your keychain, so `APPLE_CERTIFICATE` is not needed:

```bash
cd desktop
export APPLE_SIGNING_IDENTITY="Developer ID Application: Your Name (TEAMID1234)"
# notarization (omit for a signed-only build):
export APPLE_ID=you@example.com APPLE_PASSWORD=xxxx-xxxx-xxxx-xxxx APPLE_TEAM_ID=TEAMID1234
npx tauri build --bundles dmg --config "{\"bundle\":{\"macOS\":{\"signingIdentity\":\"$APPLE_SIGNING_IDENTITY\"}}}"
```

The `--config` override is there because `tauri.conf.json` contains `"-"`. Tauri prints the
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

In `release.yml`, the `macos-arm64-metal` leg of the build job has a step, **Sign and notarize
(macOS)**, between **Build** and **Package**. It reuses the desktop job's secrets:

- Without `APPLE_CERTIFICATE` / `APPLE_SIGNING_IDENTITY`, it skips signing, and the binaries
  keep the linker's ad-hoc signature.
- Otherwise, it imports the `.p12` into a temporary keychain (unlocked, with a partition list
  so that `codesign` never prompts) and runs
  `SIGN_KEYCHAIN=<it> sign-macos.sh`.
- If `APPLE_ID`, `APPLE_PASSWORD` and `APPLE_TEAM_ID` are also set, it adds `--notarize` and
  sets `MACOS_NOTARIZED=1` for **Package**. A rejected submission fails the release instead of
  shipping.

The release notes say "The macOS app and server are signed and notarized" when
`APPLE_CERTIFICATE` and `APPLE_ID` are both set.

---

## 4. Troubleshooting

| Symptom | Cause | Fix |
| :--- | :--- | :--- |
| notarization: *The binary is not signed with a valid Developer ID certificate* | signed with *Apple Development* or ad-hoc | use the *Developer ID Application* identity (1.2) |
| *The signature does not include a secure timestamp* | `--timestamp` missing, or `--timestamp=none` | re-sign with `--timestamp` (needs network) |
| *The executable does not have the hardened runtime enabled* | `--options runtime` missing | re-sign with it |
| `codesign` hangs in CI | the keychain is locked, or the key's partition list is not set | `security unlock-keychain` + `set-key-partition-list` as in 3.6 |
| `codesign` asks for a password locally | first use of the private key by `codesign` | click *Always Allow* |
| `hypnos` dies at startup with *Library Validation failed* | the dylib and the binary are signed by different teams, or the dylib was changed after signing | sign both with the same identity, the dylib first |
| desktop: microphone records silence or no prompt appears | `Entitlements.plist` not applied | `codesign -d --entitlements - Hypnos.app` must list `audio-input` |
| desktop: Tauri skips signing | `signingIdentity` is still `"-"` | pass the `--config` override (2.3) |
| Tauri: *skipping app notarization* | the `APPLE_ID` / `APPLE_PASSWORD` / `APPLE_TEAM_ID` (or `APPLE_API_*`) set is incomplete | export all three |
| Gatekeeper still prompts on a notarized binary | first launch offline (no stapled ticket on bare Mach-O) | connect once, or ship inside a stapled `.dmg` / `.pkg` |
| *E0463 can't find crate* after a "strip" step you added | Xcode's `strip` damages proc-macro dylibs (BUG-5) | never strip build-script outputs; strip only the final binary, **before** signing |

## 5. Certificate lifetime

Developer ID certificates last five years. Code signed with `--timestamp` stays valid after the
certificate expires. Revoking the certificate invalidates every signature made with it, so
revoke only if the key has leaked. When you renew, update `APPLE_CERTIFICATE`,
`APPLE_CERTIFICATE_PASSWORD` and, if the name changed, `APPLE_SIGNING_IDENTITY`.
