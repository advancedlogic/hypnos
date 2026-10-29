# Hypnos on Apple Silicon

Implementation guide for running the Hypnos backend natively on Apple Silicon Macs (M1 and later)
with the GPU driven through Metal, plus an assessment of an optional MLX engine.

Status: **partially implemented**. Every claim about upstream behaviour (ggml, llama.cpp,
stable-diffusion.cpp, macOS) is annotated with where it comes from, and every item that needs a real
machine to confirm is marked **VERIFY**.

Implementation status (M5 Max, 128 GB, macOS 27):

| Item | State |
| :--- | :--- |
| 0.1 `metal` feature (option A), forwarded by `hypnos-server` | done |
| 0.2 `run.sh` flags + macOS profile, `setup-build-env.sh` Darwin branch | done |
| 0.3 OpenMP only with `cuda` | done |
| 0.4 performance-core thread count | done |
| 1.1 rpath for `libstable-diffusion.dylib` | done: `-rpath @executable_path` from the `hypnos-backends` and `hypnos-server` build scripts (a dependency's `rustc-link-arg` does not reach other crates, so it cannot live in the diffusion-rs patch); binary and tests run without `DYLD_LIBRARY_PATH` |
| 1.2 two Metal backends | done: with the shared sd.cpp build there are no duplicate symbols, and `GGMLMetalClass` is only compiled when `GGML_METAL_EMBED_LIBRARY` is off, so neither image defines it (R1 closed) |
| 1.3 device probe | verified: Metal reports `total = recommendedMaxWorkingSetSize`, `free = total - currentAllocatedSize`, type `GPU` (R2 closed) |
| 1.4 BF16 projectors | works on M5 (native bfloat); M1 still unverified. The E4B repo ships no F16 projector, so the fallback above only exists for the 26B model |
| 1.5 sd.cpp flash attention | verified: MOSS-Image 1024x1024 with `diffusion_flash_attn = true` on Metal, correct image, 131 s (125 s sampling); no change needed |
| Endpoints | verified: `backend/scripts/smoke_test.sh` passes 9/9 against `./run.sh start backend` with the full macOS profile (the TTS size check used GNU `stat -c`, now `wc -c`) |
| Section 8 checks | all five models resident at once: 34.6 GiB booked of 128 GiB physical, 26.6 GiB actually allocated by Metal; wired memory 32.9 GiB while running, 6.0 GiB after `./run.sh stop` |
| 2.1 eviction | verified with Tier 1 capped at 22 GiB (emulating a 32 GB Mac's working set): `moss-image` with `gemma4:26b-q4` resident evicts the 26B model to Tier 2 and succeeds. A real 32 GB machine is still untested |
| Full GPU offload | fixed for all platforms: a full placement now passes `n_gpu_layers > n_layer`, so llama.cpp also offloads the output layer (43/43 instead of 42/43; E4B decode about 36-47 to 53-57 tok/s) |
| 2.1 one physical budget, 2.2 Metal working set, 2.3 all-or-nothing placement, 2.5 `mlock`/`MADV_DONTNEED` off on UMA | done; 2.2 verified live (estimate 96 GiB, Metal 107.5 GiB, Tier 2 re-derived to 16.5 GiB) |
| 2.4 promotion cost model | not started (keep-loaded demotion is the follow-up) |
| 2.6 status JSON | `physical_pool_free_bytes` and `capabilities.physical_memory_bytes` added; frontend not changed |
| 3 macOS profile + README | done (`backend/hypnos.macos.example.toml`) |
| 4 CI | macOS job added to the (still disabled) workflow; unit tests added |

---

## 1. Summary

Hypnos is closer to Apple Silicon than the README suggests. The two ggml engines already turn on
their Metal backends automatically for an arm64 macOS target, the memory layer already detects
Apple Silicon and models it as unified memory, and no Rust code in the workspace is CUDA-specific.

What remains falls into five groups, in the order they should be done:

| Phase | Theme | Size | Blocking? |
| :--- | :--- | :--- | :--- |
| 0 | Build on a Mac: cargo features, scripts, prerequisites | small | yes, nothing else can be tested without it |
| 1 | Runtime correctness: dylib loading, duplicate Metal backend, device probe | small, but needs a Mac | yes |
| 2 | Unified-memory model: budgets, planner, promotion policy, `mlock` | medium | yes for anything but a 128 GB machine |
| 3 | macOS configuration profile and model lineup | small | no, but needed for a good out-of-box experience |
| 4 | CI and tests on an arm64 macOS runner | small | no |
| 5 | Optional MLX engine | large | no, and not recommended as a first step |

The recommendation is to ship the Metal path (phases 0 to 4) and treat MLX as a separate project
afterwards, scoped to text generation and embeddings only.

---

## 2. Target and prerequisites

### Hardware and OS

- Apple Silicon only (`aarch64-apple-darwin`). Intel Macs are out of scope: the Metal backend works
  there in principle but the unified-memory assumptions in this document do not hold, and AMD eGPUs
  are a different topology.
- macOS 14 (Sonoma) or newer. ggml's Metal backend uses features that need recent Metal versions,
  and the `iogpu.wired_limit_mb` sysctl used in section 5.4 exists from macOS 15; older systems use
  `debug.iogpu.wired_limit`. **VERIFY** the exact minimum against the vendored ggml.
- Realistic memory classes and what fits (weights only, from `hypnos.example.toml`):

| Unified memory | Metal working-set default (about 75 %) | What can be resident at once |
| :--- | :--- | :--- |
| 16 GB | ~12 GB | `gemma4:e4b-q4` (5.6 GiB) + embeddings + Kokoro. The 26B model does not fit. |
| 32 GB | ~24 GB | One of `gemma4:26b-q4` (15.8 GiB) or `moss-image` (9.3 GiB) plus the small models. |
| 48 GB | ~36 GB | `gemma4:26b-q4` + `gemma4:e4b-q4` + small models, or 26B + MOSS. |
| 64 GB and up | ~48 GB+ | The whole default lineup (about 31 GiB) with KV cache headroom. |

The "about 75 %" is Metal's `recommendedMaxWorkingSetSize`; on Macs with less memory the default is
nearer two thirds. See section 5.1 for why Hypnos should read the real value instead of assuming it.

### Toolchain

| Component | Why | Install |
| :--- | :--- | :--- |
| Xcode Command Line Tools | clang, Metal compiler, frameworks, libclang for bindgen | `xcode-select --install` |
| Rust 1.92+ via rustup (`stable-aarch64-apple-darwin`) | `rust-version` in `backend/Cargo.toml`; `run.sh` already searches this toolchain path | `rustup default stable` |
| CMake 3.2x+ and Ninja | `backend/.cargo/config.toml` forces `CMAKE_GENERATOR = "Ninja"` for every engine build | `brew install cmake ninja` |
| Node 20+ | frontend only | `brew install node` |
| libclang (if bindgen cannot find Xcode's) | `llama-cpp-sys-2` and `diffusion-rs-sys` run bindgen | `brew install llvm` and set `LIBCLANG_PATH` |

No CUDA, no Python. ONNX Runtime arrives as a prebuilt `aarch64-apple-darwin` archive through the
`ort` crate's default download feature, and espeak-ng is built from source by `espeak-rs-sys`, which
already has macOS branches in its build script (vendored under `backend/vendor/espeak-rs-sys/build.rs`).

---

## 3. What already works (inventory)

This section is the evidence for the "closer than it looks" claim. Paths are relative to the repo
root; line numbers refer to the current tree.

### 3.1 Engines

- **llama.cpp Metal is on by default for arm64 macOS.** The vendored bindings enable the `metal`
  feature of the sys crate through a target-specific dependency in
  `backend/vendor/llama-cpp-rs/llama-cpp-2/Cargo.toml`
  (`[target.'cfg(all(target_os = "macos", any(target_arch = "aarch64", target_arch = "arm64")))'.dependencies]`).
  The sys build script links `Foundation`, `Metal`, `MetalKit`, `Accelerate` and `c++` for every
  Apple target (`llama-cpp-sys-2/build.rs`, around line 1516) and disables `GGML_BLAS` on Apple
  (around line 887). The Metal shader library is embedded into the binary because
  `GGML_METAL_EMBED_LIBRARY` defaults to `${GGML_METAL}` in ggml's `CMakeLists.txt`.
- **stable-diffusion.cpp Metal is on by default for Apple targets.** `backend/vendor/diffusion-rs/sys/build.rs`
  sets `use_metal = true` whenever the target triple contains `apple` and Vulkan is not requested
  (line 101), then defines `SD_METAL=ON` and links `Metal` and `MetalKit` (lines 162 to 171).
  sd.cpp's own CMake maps `SD_METAL` to `GGML_METAL` (`stable-diffusion.cpp/CMakeLists.txt` line 113).
- **ONNX Runtime (Kokoro)**: use the fp32 graph `model.onnx` on Apple Silicon. The fp16 graph
  returns an all-NaN waveform (played as silence) for 16 of 110 voice/text pairs on both the CPU and
  CoreML providers; the fp32 graph returns audio for all 110. Likely cause: ONNX Runtime's native
  fp16 kernels on ARM64 compute in half precision where x86 upcasts (not verified on Windows).
  The session pins `AutoDevicePolicy::PreferCPU`, since ort's default registers CoreML on macOS
  (24 partitions, 71 of 2273 nodes). Per sentence on an M5 Max: fp32 CPU 0.70 s, fp16 CoreML
  0.79 s, fp16 CPU 1.15 s. A non-finite waveform is now an error rather than a silent WAV.

### 3.2 Memory layer

- `DeviceMemoryCapabilities::auto_detect_accelerator` returns `AppleMetalUma` on
  `target_os = "macos"` + `target_arch = "aarch64"` (`backend/crates/hypnos-memory/src/hardware.rs:112`).
- The `AppleMetalUma` arm reports `MemoryTopology::UnifiedMemory`, reads `sysctl -n hw.memsize`,
  takes 75 % of it as the Tier 1 budget and assumes 400 GB/s Tier 2 to Tier 1 bandwidth
  (`hardware.rs:60-69`, `hardware.rs:169-186`).
- The planner has a unified-memory branch: `compute_optimal_placement` books only CPU-side bytes in
  Tier 2 when `is_unified_memory` is true (`backend/crates/hypnos-core/src/kv_cache.rs:159`), and the
  scheduler passes `self.memory.is_unified_memory()` into it (`backend/crates/hypnos-scheduler/src/lib.rs:252`).
- `MmapWeightBuffer` uses `madvise(MADV_WILLNEED)` + `mlock` on unix and already degrades to a
  warning when the lock is refused (`backend/crates/hypnos-memory/src/mmap_cache.rs:127-135`).
- The real-memory probe `LlamaGpuProbe` uses ggml's backend-device API and accepts `Gpu`,
  `IntegratedGpu` or `Accelerator` devices (`backend/crates/hypnos-backends/src/lib.rs:30-45`).
  On Metal, ggml reports `recommendedMaxWorkingSetSize` as total and the difference to
  `currentAllocatedSize` as free, so the reconciliation path in `manager.rs` works unchanged.

### 3.3 Rust code

- `grep` for `cfg(feature = "cuda")` and `cfg(target` inside `backend/crates/hypnos-backends/src`
  finds nothing. The only CUDA references are prose in doc comments and the `cuda` feature in
  `hypnos-backends/Cargo.toml`.
- `run.sh` already prefers `~/.rustup/toolchains/stable-aarch64-apple-darwin/bin` and uses `lsof`
  for port discovery on macOS.

---

## 4. How the three tiers map onto unified memory

This is the conceptual change everything in phase 2 follows from.

On a discrete GPU the tiers are three distinct places and promotion is a copy:

```
Tier 3  NVMe   --mmap-->  Tier 2  pinned host RAM  --PCIe DMA-->  Tier 1  VRAM
```

On Apple Silicon there is one physical pool. With `use_mmap = true` llama.cpp's Metal backend wraps
the mapped file pages in `MTLBuffer`s created with `newBufferWithBytesNoCopy`, so "resident on GPU"
and "resident in RAM" are the same pages:

```
Tier 3  NVMe   --mmap-->  { page cache == wired host pages == Metal buffers }  Tier 1/2 merged
```

Consequences:

1. **One budget, not two.** Tier 1 and Tier 2 bookings draw from the same physical memory. Keeping
   two independent budgets (24 GB "VRAM" plus 32 GB "pinned RAM" on a 32 GB machine) lets the planner
   admit models that cannot exist.
2. **Promotion is a reload, not a copy.** Changing `n_gpu_layers` means dropping and re-creating the
   `LlamaModel`, which rereads headers and re-wires pages. The bytes do not move, but the latency is
   still a full load. Partial layer splits buy nothing on UMA because CPU and GPU already read the
   same memory at the same bandwidth; they only add CPU compute to the critical path.
3. **Pinning is wiring.** `mlock` on macOS wires pages against a global wired-memory limit shared
   with the Metal working set. Locking tens of GiB competes with the GPU for the same quota.
4. **Eviction is what frees memory.** Dropping a model's Metal buffers releases the wired pages back
   to the page cache. A demoted model that `keep_warm_in_ram` keeps mapped costs nothing until pages
   are touched again, which makes Tier 2 on UMA simply "mapped but not wired".

So the honest UMA policy is: a model is either fully resident (all layers on Metal, pages wired by
Metal, one booking against the single pool) or not resident (mapped, unwired, zero booking).
Tier 2 remains a useful concept as a *warm page-cache state* for fast re-promotion, but it must not
have its own byte budget.

---

## 5. Work items

Each item states the problem, the evidence in the tree, the proposed change and how to know it is
done. Items are grouped by phase; order inside a phase is the suggested order.

### Phase 0: building on a Mac

#### 0.1 Cargo feature for Metal, and `cuda` no longer unconditional

Problem: `backend/crates/hypnos-backends/Cargo.toml` declares `default = ["cuda"]`. A plain
`cargo build` on a Mac tries to configure `GGML_CUDA=ON` and fails looking for `nvcc`. The documented
CPU escape hatch `--no-default-features` happens to produce a Metal build on arm64 macOS, because the
vendored crates auto-enable Metal, but that is accidental and undocumented.

Cargo cannot make default features depend on the target, so pick one of:

- **Option A (least disruptive, recommended):** keep `default = ["cuda"]` for Windows and Linux,
  add an explicit `metal` feature that forwards to `llama-cpp-2/metal` and `diffusion-rs-sys/metal`,
  and make the launch scripts pass `--no-default-features --features metal` on Darwin
  (see 0.2). Document both commands in the README.
- **Option B (cleaner, breaks muscle memory):** `default = []`, and every platform passes its GPU
  feature explicitly (`cuda`, `metal`, or nothing for CPU). CI already passes
  `--no-default-features`, so it would not change.

Either way, a guard in a `hypnos-backends` build script that fails with a readable message when
`cuda` is enabled on an Apple target saves the next person twenty minutes of CMake output.

Do not gate runtime behaviour on `cfg!(feature = "metal")`. The UMA policy in phase 2 should key off
`MemoryTopology::UnifiedMemory` at runtime so that a CPU-only build on a Mac and a Metal build behave
the same way for planning.

Done when: `cargo build -p hypnos-server --no-default-features --features metal` succeeds on an
Apple Silicon Mac and the log line "Initialized hardware memory topology" shows `AppleMetalUma`.

#### 0.2 Launch and setup scripts

- `run.sh`: detect `uname -s` = `Darwin` and `uname -m` = `arm64`, and append the feature flags from
  0.1 to the `cargo run` invocation in `start_backend`. Also export `DYLD_LIBRARY_PATH` to the cargo
  profile directory until 1.1 is fixed, so `libstable-diffusion.dylib` can be found.
- `backend/scripts/setup-build-env.sh`: today it writes `CUDAARCHS = "native"` and
  `CUDA_COMPUTE_CAP = "native"` even without an NVIDIA GPU. On Darwin it should write an empty `[env]`
  table (or only `MACOSX_DEPLOYMENT_TARGET`) and print the Metal build command. The portable
  `backend/.cargo/config.toml` entries (`CMAKE_GENERATOR`, `LLAMA_LIB_PROFILE`, `CUDAFLAGS`) are
  harmless on macOS and can stay.
- `backend/tools/` holds Windows prebuilt `llama-*.exe` / `sd` binaries for manual debugging. The
  macOS equivalent is `brew install llama.cpp` or the `llama-cli` target from the vendored tree;
  note this in the README rather than vendoring more binaries.

Done when: `./run.sh start backend` on a Mac builds and starts without manual flags.

#### 0.3 OpenMP default feature

`llama-cpp-2` has `openmp` in its defaults. Apple clang ships no OpenMP runtime; ggml's CMake
`find_package(OpenMP)` fails and falls back to its own thread pool with a warning. This is harmless,
but to keep the build log clean the `hypnos-backends` dependency on `llama-cpp-2` can declare
`default-features = false, features = ["common", "mtmd"]` and add `openmp` only inside the `cuda`
feature. **VERIFY** that nothing else relied on the `android-shared-stdcxx` default (it is Android
only).

#### 0.4 Threads

`default_threads()` in `hypnos-backends/src/lib.rs` clamps `available_parallelism()` to 16. On
M-series chips that count includes efficiency cores, and llama.cpp's own guidance is to use the
performance-core count for GPU-resident models. `sysctl -n hw.perflevel0.logicalcpu` gives that
number. Low priority: the GPU does the work when all layers are offloaded.

### Phase 1: runtime correctness (needs a Mac)

#### 1.1 `libstable-diffusion.dylib` is not found at runtime

Evidence: the Hypnos patch builds sd.cpp as a shared library and `copy_shared_library` in
`diffusion-rs/sys/build.rs` copies `libstable-diffusion.dylib` next to the binaries. That works for
Windows DLL search order and for `.so` only with an rpath; on macOS, dyld does not search the
executable's directory at all. sd.cpp's CMake sets the install name to `@rpath/libstable-diffusion.dylib`
(`sd_set_macos_rpaths`, `CMakeLists.txt` line 27), so the *executable* needs an rpath entry and the
Rust link line has none.

Fix: in the diffusion-rs patch, when the target contains `apple`, emit
`cargo:rustc-link-arg=-Wl,-rpath,@executable_path` (and `@loader_path` for test binaries under
`target/<profile>/deps`). Alternatively `cargo:rustc-link-arg=-Wl,-rpath,<absolute profile dir>`
for development builds. Also **VERIFY** which of `out/bin`, `out/lib`, `out/build` CMake uses for the
dylib on macOS; `copy_shared_library` only checks those four names.

Done when: `/v1/images/generations` works from `cargo run` without `DYLD_LIBRARY_PATH`, and
`otool -L target/debug/hypnos` shows the `@rpath` entry.

#### 1.2 Two Metal backends in one process

Evidence: llama.cpp (static, in the main binary) and sd.cpp (inside the dylib) each carry their own
ggml with its own Metal backend. The two copies of `ggml-metal-device.m` are different versions
(16 files differ between the two `ggml-metal` directories) and both define the Objective-C class
`GGMLMetalClass` (line 36 in both files). macOS two-level namespaces keep the C symbols apart, which
is why the shared-library trick works at all, but the Objective-C runtime has one flat class table:
at load time it logs `Class GGMLMetalClass is implemented in both ... One of the two will be used`
and picks one.

Expected impact: low. The class exists only as an anchor for `[NSBundle bundleForClass:]` to find a
`default.metallib` on disk, and both builds embed the Metal library (`GGML_METAL_EMBED_LIBRARY`
defaults on), so the lookup never runs. The warning is noise.

Mitigation if it is not benign: rename the class in one copy via the existing patch mechanism
(`backend/vendor/patches/diffusion-rs.patch` already touches `sys/build.rs`; a one-line `sed` in
`vendor.sh` after cloning sd.cpp is simpler than a source patch against ggml).

Also worth knowing: with embedded libraries each ggml copy compiles its Metal source at first use.
Expect several seconds of extra latency the first time *each* engine initialises, which shows up in
`initialize_startup_tiers` and in the first image request. Not a bug, but it should be in the logs.

Done when: both engines initialise in the same process and `GGML_METAL_LOG` (or the ggml log
callback) shows two successful Metal device initialisations.

#### 1.3 Device probe returns the right device

`LlamaGpuProbe` prefers a device of type `Gpu` with `memory_total > 0`. ggml's Metal backend
registers the Apple GPU as `GGML_BACKEND_DEVICE_TYPE_GPU` in recent versions and as `IGPU` in
others. Both are covered by the fallback chain, and `memory_total` is `recommendedMaxWorkingSetSize`,
which is exactly the Tier 1 ceiling Hypnos wants. **VERIFY** on the vendored b11982 that
`memory_free` is `total - currentAllocatedSize` and not zero; if it is zero, the reconciliation in
`effective_free_vram_bytes` would starve the planner.

#### 1.4 BF16 projectors on older chips

The default lineup uses BF16 `mmproj` files. ggml-metal supports BF16 matrix ops on Apple7 GPU
family and newer when `GGML_METAL_USE_BF16` is on, but the M1 generation lacks native BF16 and
either falls back to slower paths or refuses certain ops. **VERIFY** vision and audio on an M1; if
it fails, ship F16 projectors (`mmproj-*-F16.gguf` exists in the same Hugging Face repos) in the
macOS profile.

#### 1.5 Flash attention in sd.cpp on Metal

`load_sd` sets `diffusion_flash_attn = true` unconditionally (`diffusion.rs:159`). ggml's Metal flash
attention kernels cover fewer head sizes than CUDA's, and sd.cpp historically disabled it on Metal
for some models. **VERIFY** MOSS-Image at 1024x1024 on Metal; if the kernel is missing, sd.cpp logs
a fallback or asserts, and the flag should be keyed off the topology.

### Phase 2: unified-memory model

#### 2.1 One physical budget

Problem: `HardwareConfig` and `default_profile.rs` express independent `vram_budget_bytes` (auto:
75 % of RAM on macOS) and `pinned_ram_budget_bytes` (default 32 GiB). On a 32 GB Mac the planner
believes it has 24 GB of Tier 1 plus 32 GB of Tier 2.

Proposed change, in `hypnos-memory`:

- When `topology == UnifiedMemory`, derive `pinned_ram_budget_bytes` from the same pool:
  `usable_vram_budget_bytes` becomes the Metal working set (see 2.2) and the Tier 2 budget becomes
  `min(configured, hw.memsize - working_set - os_reserve)`; in practice a few GiB. Log the override.
- `MemoryManager` should expose a `physical_pool_free_bytes()` that subtracts both Tier 1 and Tier 2
  bookings from the single pool, and `compute_optimal_placement` should receive that as
  `available_vram_bytes` on UMA.
- `HypnosConfig::validate` should warn when `vram_budget_bytes + pinned_ram_budget_bytes` exceeds
  physical memory on a UMA machine; today it only checks that the safety margin is below the VRAM
  budget (`config.rs:560-570`).

Done when: on a 32 GB Mac, loading `gemma4:26b-q4` and then requesting `moss-image` evicts the
first model instead of returning `InsufficientMemory` after a failed Metal allocation.

#### 2.2 Read the real working-set limit instead of guessing 75 %

`probe_macos_unified_memory` multiplies `hw.memsize` by 0.75. Metal's actual figure depends on the
machine class and on the `iogpu.wired_limit_mb` sysctl that users raise to run bigger models.
Prefer, in order:

1. `LlamaGpuProbe` total (this is `recommendedMaxWorkingSetSize` as seen by ggml). The probe is
   installed after `AppState::from_config`, so `detect()` would need to be re-run or the
   capabilities updated once the probe exists; `set_gpu_probe` already calls `reconcile_vram`, so
   extending it to overwrite `total_vram_bytes` when the configured budget is 0 is the smallest change.
2. `sysctl -n iogpu.wired_limit_mb` when non-zero (user override).
3. The 75 % heuristic as last resort.

Document in the macOS profile that `sudo sysctl iogpu.wired_limit_mb=<MiB>` raises the ceiling
(non-persistent; needs a LaunchDaemon or `/etc/sysctl.conf` to survive reboot).

#### 2.3 All-or-nothing placement on UMA

Problem: `compute_optimal_placement` searches from `total_layers` down to `min_gpu_layers`, and the
scheduler evicts until a *full* split is possible but accepts a partial one when eviction runs out.
On UMA, a partial split is strictly worse than a full one that fits, and if a full one does not fit
nothing will help because the CPU side shares the same pool.

Change: when `is_unified_memory`, set the candidate range to `total_layers..=total_layers` (ignore
`min_gpu_layer_ratio`) and let the plan fail cleanly so the scheduler evicts or rejects. The MoE
exception does not apply: Gemma 4 26B-A4B's experts are regular tensors to llama.cpp and are
offloaded with the layer.

Side effect: `mp.use_gpu = gpu_layers > 0` in `load_model` and `with_offload_kqv(gpu_layers > 0)` in
the generation path become always-true on UMA, which is what we want. The sd.cpp `*=cpu` offload
string in `load_sd` (`diffusion.rs:144`) becomes dead on UMA; keep the code, since it still matters
for the Windows and Linux builds.

#### 2.4 Promotion and demotion cost model

Today a Tier 2 to Tier 1 promotion on UMA costs a full `LlamaModel` reload (`llama.rs:656`,
`diffusion.rs:348`). The data stays put, but Metal buffer creation, page wiring and the mtmd
projector load still take seconds for a 15 GiB model. Two options:

- Accept it and make the latency estimate honest: `tier2_to_tier1_bandwidth_gbps = 400.0` in
  `hardware.rs:67` feeds `tier_switch_latency_ms`, which will under-report. A per-topology constant
  for "reload seconds per GiB" is more truthful than a bandwidth.
- Keep models loaded with `n_gpu_layers = all` even when "demoted", and use `llama_memory_clear`
  plus dropping the context to release KV, so demotion only frees the KV cache and workspace while
  the weights stay mapped. On UMA the weights are page-cache pages either way, so this removes most
  of the promotion cost at no memory cost. This is the better design but touches the scheduler's
  invariants (`demote_model_from_vram` assumes 100 % of VRAM is freed, `manager.rs:327`).

Recommendation: ship the first option in the initial port, note the second as follow-up.

#### 2.5 `mlock`, `madvise` and the wired-memory limit

- `lock_host_memory = true` plus a 32 GiB `pinned_ram_budget_bytes` makes `pin_in_host_ram` touch
  every page and `mlock` the whole file at staging time (`manager.rs:301`, `mmap_cache.rs:63`). On
  macOS the kernel's wired limit (`vm.global_user_wire_limit`) is shared with Metal. Wiring the
  weights twice, once by `mlock` and once by Metal, does not double the memory but does double the
  accounting against that limit. On UMA the default should be `lock_host_memory = false`, with
  `MADV_WILLNEED` kept so re-promotion stays fast.
- `os_unlock_memory` calls `madvise(MADV_DONTNEED)` after `munlock` (`mmap_cache.rs:140-145`). On
  Linux that is a hint for file-backed shared mappings. On Darwin `MADV_DONTNEED` is also advisory
  for file mappings, but **VERIFY** that it does not evict the pages the Metal backend still has a
  no-copy buffer over when a model is demoted but kept warm; the safe variant is to skip it on UMA.
- `raise_working_set_limit` is already a no-op on non-Windows. Nothing to do.

#### 2.6 Telemetry and `/v1/memory/status`

`tier1_vram_real_free_bytes` and `tier1_vram_unaccounted_bytes` remain meaningful (they now describe
the Metal working set), but the frontend visualiser draws three separate bars. On UMA it should
draw one pool with resident and warm segments. Smallest change: add `topology` to the status JSON
(it is already in `capabilities`) and let `App.svelte` switch layouts on it.

### Phase 3: macOS profile and model lineup

- Add `backend/hypnos.macos.example.toml` (or make the built-in `default_profile.rs` topology-aware).
  Differences from the CUDA profile:
  - `[hardware]`: `accelerator = "apple_metal_uma"` (or `auto`), `vram_budget_bytes = 0`,
    `pinned_ram_budget_bytes` small (4 GiB) or ignored per 2.1, `lock_host_memory = false`,
    `vram_safety_margin_bytes` 2 GiB (the window server and other GPU clients share the pool).
  - Model `min_gpu_layer_ratio = 1.0` everywhere (made irrelevant by 2.3, but explicit is kinder).
  - `initial_tier`: on a 64 GB+ machine `tier1_gpu_vram` for both Gemma 4 models is fine; on 32 GB
    keep the 26B at `tier2_pinned_ram` and let demand promotion decide.
  - Projectors: F16 instead of BF16 if 1.4 fails on M1.
- `run.sh` picks the macOS config when it exists and the platform is Darwin; `BACKEND_CONFIG` is
  already a variable at the top of the script.
- README: a "macOS (Apple Silicon)" subsection under build prerequisites with the Homebrew line,
  the build command, the `iogpu.wired_limit_mb` note, and the memory-class table from section 2.

### Phase 4: CI and tests

- Add a `macos-15` (arm64) job to `.github/workflows/ci.yml` that runs `vendor.sh`, builds with
  `--no-default-features --features metal`, runs `cargo test --workspace --no-default-features --features metal`
  and, as a smoke step, downloads `embeddinggemma-2-Q8_0.gguf` (295 MiB) and hits `/v1/embeddings`.
  GitHub's arm64 macOS runners expose Metal; llama.cpp's own CI relies on that.
- Unit tests to add in `hypnos-core`/`hypnos-memory` (no Mac needed):
  - `compute_optimal_placement` with `is_unified_memory = true` never returns a partial split.
  - Budget derivation on UMA never exceeds `hw.memsize` (inject the sysctl value).
  - `validate()` warns on oversubscribed UMA budgets.
- The ignored GPU lifecycle test in `hypnos-server/tests` should run on a developer Mac with the
  macOS profile; record the timings for the promotion-cost discussion in 2.4.

---

## 6. MLX assessment

MLX is Apple's array framework with a lazy graph, its own Metal kernels and a unified-memory-native
allocator. The question is whether Hypnos should drive models through it instead of, or next to,
ggml's Metal backend.

### 6.1 Integration surface

Hypnos is a single Rust process with no Python sidecar, by design (README, "Every engine runs inside
the hypnos process"). The routes that respect that are:

- **mlx-c** (`ml-explore/mlx-c`): the official C API over MLX core. Stable, tracks MLX releases, but
  exposes arrays and ops, not models. Building it is a CMake project like the ggml engines.
- **mlx-rs** (`oxideai/mlx-rs`): community Rust bindings over mlx-c with an `nn` module. Lags mlx-c
  by releases and covers a subset of ops.

Neither gives you Gemma 4. The model implementations live in Python packages: `mlx-lm` (text),
`mlx-vlm` (vision and audio), `mflux` and `diffusionkit` (image). Using them means a Python sidecar,
which the project rules out; not using them means reimplementing, in Rust on mlx-rs:

| Capability | What has to be written | Difficulty |
| :--- | :--- | :--- |
| Gemma 4 26B-A4B text | MoE decoder with sliding-window + global attention, RoPE variants, KV cache with SWA eviction, sampling | high |
| Gemma 4 vision | SigLIP-style tower, projector, image tiling identical to mtmd's | high |
| Gemma 4 audio | conformer encoder, feature extraction | high |
| EmbeddingGemma 2 | encoder + pooling | medium |
| MOSS-Image | DiT, Qwen3.5 text encoder, Flux-2 VAE, scheduler | very high |
| Kokoro | stays on ONNX Runtime | none |

### 6.2 Weights

MLX wants its own safetensors layout with affine 4-bit or 8-bit quantisation (group size 32 or 64).
Its GGUF loader handles F16/F32 and `Q4_0`, `Q4_1`, `Q8_0` only, and knows nothing about `mmproj` or
sd.cpp's split files. A production MLX path means a second set of weights per model (about 30 GiB
more on disk for the default lineup) and a second metadata source for the planner's size estimates.

### 6.3 Memory model

MLX owns its buffers and keeps an internal cache (`mx.metal.set_cache_limit`,
`set_memory_limit`). The Hypnos scheduler's precise bookkeeping relies on ggml allocating exactly what
the plan says and releasing it on demotion. With MLX the equivalent is `clear_cache` plus dropping
the arrays, and `memory_free` must be read from MLX's own counters, not from ggml's device probe.
Two Metal clients (ggml and MLX) in one process also means two command queues competing on the same
GPU with no shared scheduler; sequential use is fine, concurrent use is unpredictable.

### 6.4 Performance

On the same Apple Silicon chip, MLX and llama.cpp Metal land in the same band for single-stream
decode. MLX tends to be ahead on prompt processing and batched throughput, llama.cpp on breadth of
quantisation formats and model support (and it is already tuned for the models in the lineup). For a
server whose workload is interactive single requests, the difference is not what decides the user
experience; model load behaviour and memory pressure are.

### 6.5 Recommendation

Do not start with MLX. If after the Metal port there is a measured reason (for example batched
embedding throughput, or a model that exists only as MLX weights), scope it as:

- a new `EngineKind::Mlx` behind a cargo feature `mlx`, macOS-only;
- text generation and embeddings only, with Gemma 4 text implemented on mlx-rs;
- single-tier residency: an MLX model is either loaded or not, booked against the UMA pool from 2.1,
  with `memory_free` read from MLX;
- vision, audio and image generation stay on ggml.

Estimate: the Gemma 4 text implementation alone is several weeks of work plus parity testing against
llama.cpp output, before any of the scheduler integration.

---

## 7. Risks and open questions

| # | Risk | Likelihood | Mitigation |
| :--- | :--- | :--- | :--- |
| R1 | `GGMLMetalClass` duplicate-class behaviour is worse than a warning (1.2) | low | rename in one ggml copy via `vendor.sh` |
| R2 | Metal `memory_free` reads zero on b11982 (1.3) | low | fall back to `total - booked` when the probe reports zero |
| R3 | BF16 projectors fail on M1 (1.4) | medium | F16 projectors in the macOS profile |
| R4 | Promotion reloads make tier switching feel slow on UMA (2.4) | high | document first, implement keep-loaded demotion second |
| R5 | `mlock` plus Metal wiring trips the global wired limit on 16 to 32 GB machines (2.5) | medium | `lock_host_memory = false` default on UMA |
| R6 | sd.cpp flash attention unsupported for MOSS head sizes on Metal (1.5) | medium | key the flag off topology |
| R7 | Two embedded Metal libraries double first-init compile time | certain, minor | log it; consider `GGML_METAL_EMBED_LIBRARY=OFF` plus shipping `.metallib` files later |

Open questions that need a decision, not a test:

- Keep `cuda` as the default feature (option A in 0.1) or move to explicit features everywhere (B)?
- Should Tier 2 disappear from the UMA status output entirely, or be shown as "warm, unwired"?
- Minimum supported macOS version.

---

## 8. Verification checklist

Run on a real Apple Silicon machine, 32 GB or more, after phases 0 to 2:

1. `cargo build -p hypnos-server --no-default-features --features metal` completes; `otool -L`
   on the binary lists `Metal.framework` and the `@rpath` entry for `libstable-diffusion.dylib`.
2. `cargo test --workspace --no-default-features --features metal` passes.
3. Startup log shows `accelerator = AppleMetalUma`, `topology = UnifiedMemory`, and a Tier 1 total
   equal to Metal's working-set limit (compare with `system_profiler SPDisplaysDataType` or the ggml
   log line `recommendedMaxWorkingSetSize`).
4. `/v1/embeddings` returns 768-dimensional vectors; `/v1/audio/speech` returns WAV.
5. `/v1/chat/completions` with `gemma4:26b-q4` streams text; with an image, returns a description;
   `/v1/audio/transcriptions` with `backend/tests/fixtures/jfk.wav` returns the transcript.
6. `/v1/images/generations` with `moss-image` returns a PNG; the log shows the sd.cpp Metal device.
7. On a 32 GB machine: request `moss-image` while `gemma4:26b-q4` is resident; the scheduler evicts
   and the request succeeds. `/v1/memory/status` never reports more booked bytes than physical memory.
8. `Activity Monitor` memory pressure stays green throughout; `vm_stat` wired count returns to
   baseline after `./run.sh stop`.
9. `backend/scripts/smoke_test.sh` passes end to end.

---

## 9. File map for the implementation

| Area | Files |
| :--- | :--- |
| Feature flags | `backend/crates/hypnos-backends/Cargo.toml`, `backend/Cargo.toml` (workspace deps) |
| Scripts | `run.sh`, `backend/scripts/setup-build-env.sh`, `backend/scripts/vendor.sh` |
| Vendored patches | `backend/vendor/patches/diffusion-rs.patch` (rpath), `backend/vendor/VENDOR.md` |
| Hardware detection | `backend/crates/hypnos-memory/src/hardware.rs` |
| Budgets and pinning | `backend/crates/hypnos-memory/src/manager.rs`, `backend/crates/hypnos-memory/src/mmap_cache.rs` |
| Planner | `backend/crates/hypnos-core/src/kv_cache.rs`, `backend/crates/hypnos-core/src/config.rs` (validate) |
| Scheduler | `backend/crates/hypnos-scheduler/src/lib.rs` (eviction loop around line 252) |
| Engines | `backend/crates/hypnos-backends/src/{lib.rs,llama.rs,diffusion.rs}` |
| Profiles | `backend/hypnos.example.toml`, `backend/crates/hypnos-server/src/default_profile.rs`, new `backend/hypnos.macos.example.toml` |
| CI | `.github/workflows/ci.yml` |
| Frontend | `frontend/src/App.svelte` (tier visualiser) |

## 10. References

- ggml Metal backend: `backend/vendor/llama-cpp-rs/llama-cpp-sys-2/llama.cpp/ggml/src/ggml-metal/`
  (device enumeration and memory reporting in `ggml-metal-device.m`).
- llama.cpp build docs for Metal and the `use_mmap` / no-copy buffer behaviour: `docs/build.md` in
  the vendored llama.cpp tree.
- stable-diffusion.cpp macOS rpath handling: `backend/vendor/diffusion-rs/sys/stable-diffusion.cpp/CMakeLists.txt`.
- Apple, "Metal memory management": `MTLDevice.recommendedMaxWorkingSetSize`, `currentAllocatedSize`,
  `newBufferWithBytesNoCopy`.
- MLX: https://github.com/ml-explore/mlx, C API https://github.com/ml-explore/mlx-c, Rust bindings
  https://github.com/oxideai/mlx-rs.
