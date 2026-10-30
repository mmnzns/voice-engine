# Building the helpers for an app's release

The engine's two helper programs, the AI helper (`voice-engine-llm`) and the speech helper (`voice-engine-stt`), are
built from the engine repository by a reusable GitHub Actions workflow. An app's release workflow calls it, downloads
the result and gives the files to Tauri as sidecars. Nothing is copied out of the engine repository by hand.

What it builds, per target triple:

| Target triple | Built on | Helpers |
| --- | --- | --- |
| `aarch64-apple-darwin` (Mac, Apple silicon) | `macos-latest` | `llm`, `stt` |
| `x86_64-pc-windows-msvc` (Windows, Intel and AMD) | `windows-latest` | `llm`, `stt`, and the Vulkan variants `llm-vulkan`, `stt-vulkan` |
| `aarch64-pc-windows-msvc` (Windows on Arm, Snapdragon) | `windows-11-arm` | `llm`, `stt` |

Each file is named the way Tauri names a sidecar, `<name>-<target triple>[.exe]`, for example
`ensayo-llm-aarch64-apple-darwin` or `ensayo-llm-x86_64-pc-windows-msvc.exe`. `<name>` is the app's choice (the
`llm-name` and `stt-name` inputs; the engine fixes nothing). A Vulkan variant puts `-vulkan` after the name and before the
triple: `ensayo-llm-vulkan-x86_64-pc-windows-msvc.exe`. Vulkan builds exist for Windows x64 only.

### DirectML is not in the helpers

Both Vulkan helpers are built with `--features vulkan` only. The speech helper runs Whisper only; Parakeet runs in the
app's own process, so an app that wants Parakeet on the card enables `voice-engine-stt/directml` in its own `Cargo.toml`.
From `ort-sys 2.0.0-rc.13`'s build script: that build links ONNX Runtime statically and the program imports
`DirectML.dll`, `D3D12.dll`, `DXGI.dll` and `DXCORE.dll` at load time (`ort`'s `copy-dylibs` is off, so nothing is copied
next to the exe). All four come from Windows: `DirectML.dll` has been in `C:\Windows\System32` since Windows 10 1903.
Nothing ships with the app, and the helpers do not import any of it. Not checked on a PC yet: that the in-box
`DirectML.dll` is new enough for this ONNX Runtime on every Windows 10 build; if it is not, Parakeet falls back to the
processor with `fell_back`.

## Why the engine builds them, and not the app

llama.cpp and whisper.cpp are compiled from source by CMake while the helper is built. Left alone they are compiled for
the machine that builds them: the newest instructions of that CPU (a build on a new CI runner can crash with "illegal
instruction" on an older customer's PC), and the build machine's macOS as the minimum system. The engine's
`.cargo/config.toml` fixes that (`GGML_NATIVE` off, the AVX2 baseline on Windows x64, macOS 11 on the Mac). It also sets
`CMAKE_C_FLAGS_RELEASE` and `CMAKE_CXX_FLAGS_RELEASE` to `-O2 -DNDEBUG`: `cmake-rs` strips the `/O` flags from them under
the Visual Studio generator, so without those two lines the libraries are built at MSVC's `/Od`, with no optimization,
and run roughly ten times slower (a Whisper decode on the processor does not finish).

**Cargo reads `.cargo/config.toml` only for the workspace it is building.** An app that depends on the engine as a git
dependency builds the engine's code inside the app's workspace, and Cargo never reads the engine's file there: the app
gets none of those settings, and nothing warns it. The reusable workflow avoids the problem by checking out the engine
repository itself at the requested tag and building there, so the engine's `.cargo/config.toml` is the one Cargo reads.
Before it copies anything, `scripts/build-helpers.sh` and `scripts/build-helpers.ps1` read each CMake cache and the
binaries and fail unless the baseline was applied (`GGML_NATIVE:BOOL=OFF`, `GGML_AVX2` on for x64, macOS 11.0 on the Mac) and the
libraries are optimized (`-O2` in both `CMAKE_C_FLAGS_RELEASE` and `CMAKE_CXX_FLAGS_RELEASE`). The scripts also clean the
native library (`cargo clean --release -p <crate>`) before each build, so it is always compiled again from CMake: its build
script does not notice a change of `.cargo/config.toml`, and a restored build cache (CI) or an older build folder would
otherwise hand them a library compiled without the settings.

On Windows ARM64 the build also needs clang-cl and Ninja (MSVC's `cl` stops ggml at configure time), set up by
`.github/actions/windows-arm64-toolchain`. There is no instruction-set line to copy for ARM64: with `GGML_NATIVE` off and
no architecture flag, the compiler's own default for `aarch64-pc-windows-msvc` applies, armv8-a with NEON, so there are no
dot-product or i8mm kernels, whatever the build machine's CPU is.

An app that links `voice-engine-stt` **in its own process** (the speech models run there, not in a helper) is not covered
by this workflow: its own build needs the same settings. The README section "Build settings an app must copy" lists
them.

## Calling it

In the app's release workflow, one job calls the engine's workflow and the jobs that build the app download what it made.
Use the same tag for `@v0.2.2` and for `engine-ref`, so the workflow, the scripts and the engine match.

```yaml
name: Release

on:
  push:
    tags: ['v*']

jobs:
  helpers:
    uses: mmnzns/voice-engine/.github/workflows/build-helpers.yml@v0.2.2
    with:
      engine-ref: v0.2.2
      helpers: llm,stt                 # add llm-vulkan,stt-vulkan for the Windows x64 graphics card builds
      llm-name: ensayo-llm
      stt-name: ensayo-stt

  app:
    needs: helpers
    strategy:
      fail-fast: false
      matrix:
        include:
          - { os: macos-latest, triple: aarch64-apple-darwin }
          - { os: windows-latest, triple: x86_64-pc-windows-msvc }
          - { os: windows-11-arm, triple: aarch64-pc-windows-msvc }
    runs-on: ${{ matrix.os }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/download-artifact@v4
        with:
          name: helpers-${{ matrix.triple }}
          path: src-tauri/binaries
      # A downloaded artifact does not keep the executable bit.
      - name: Make the helpers executable
        if: runner.os == 'macOS'
        run: chmod +x src-tauri/binaries/*-apple-darwin
      # ... set up Rust and Node, build the app for ${{ matrix.triple }}, sign, publish (see below)
```

Inputs:

| Input | Default | |
| --- | --- | --- |
| `engine-ref` | none, required | The engine tag or commit to build. |
| `helpers` | `llm,stt` | A comma list among `llm`, `stt`, `llm-vulkan`, `stt-vulkan`. A Vulkan variant is built on Windows x64 only; the other two targets skip it. |
| `llm-name` | `voice-engine-llm` | Base name of the AI helper's file. |
| `stt-name` | `voice-engine-stt` | Base name of the speech helper's file. |
| `cache` | `true` | Cache Cargo's registry and build folder between runs. `false` builds from scratch. |

Each target uploads one artifact, `helpers-<target triple>`, with:

- the helpers, named `<name>-<target triple>[.exe]`;
- `SHA256SUMS` (`sha256sum -c SHA256SUMS` or `shasum -a 256 -c SHA256SUMS` checks the files next to it);
- `build-info.txt`: the engine ref and commit, the target, `rustc`'s version, the protocol version, and the versions of
  llama-cpp-2, llama-cpp-sys-2, whisper-rs, whisper-rs-sys, parakeet-rs, ort and ort-sys that `Cargo.lock` pins.

Delete `SHA256SUMS` and `build-info.txt` from `src-tauri/binaries` before building if you do not want them next to the
sidecars, or keep them as a record of the release. Tauri only bundles the files listed in `externalBin`.

The workflow builds with the stable Rust toolchain and `cargo build --release --locked`, so the native library versions
are the ones in the engine's `Cargo.lock` at that tag, and `build-info.txt` says which `rustc` made them.

The workflow can also be started by hand (Actions, Build helpers, Run workflow) on a branch of the engine repository: the
same inputs, with `engine-ref` empty meaning the commit it was started on.

## Tauri

List each helper as an `externalBin` in `tauri.conf.json`, without the triple:

```json
{
  "bundle": {
    "externalBin": ["binaries/ensayo-llm", "binaries/ensayo-stt"]
  }
}
```

Tauri looks for `src-tauri/binaries/<name>-<target triple>[.exe]` for the target it is building, which is why the files
carry the triple, and puts the program next to the app's own executable without it. Tauri's sidecar guide:
<https://v2.tauri.app/develop/sidecar/>.

Tauri requires every `externalBin` entry to exist for the target being built. The Vulkan helpers exist for Windows x64
only, so an app that ships them lists them only in the configuration used for that target (Tauri merges a platform or
`--config` file over `tauri.conf.json`; see its configuration docs) and does not list them for Windows on Arm or the Mac.

## Closing a running helper before an update

On Windows an installer cannot replace a program that is running. A helper runs for as long as the app that
started it (it exits when its stdin closes), so it is still running while the app is being updated unless the app stops
it. An app whose Windows installer replaces the helper files (Ensayo's) must close its helpers first: call `shutdown` on
the `LlmClient` and `SttClient`, or have the installer close the processes by name. The
name is the one the app chose with `llm-name` and `stt-name` (a Vulkan helper's is `<name>-vulkan`, with the same
extension); the engine does not fix it. Without this the update fails with the file in use, or leaves the old helper
running next to a new app.

## Signing

The engine ships **unsigned** binaries. Signing is the app's step, and the apps are released self-signed and not
notarized, so the helpers inherit whatever the app does:

- **Mac.** Every sidecar inside the `.app` is signed with the app's signing identity and the hardened runtime, as the
  app's own executable is. A helper left unsigned inside a signed app is not covered by the app's signature.
- **Windows.** Each `.exe` (the app and every helper, the Vulkan ones too) is signed with the app's certificate.

How Tauri signs sidecars and what its bundler does on its own is in its docs: macOS
<https://v2.tauri.app/distribute/sign/macos/> and Windows <https://v2.tauri.app/distribute/sign/windows/>. Follow them;
this document gives no signing commands because they are the app's and depend on the app's Tauri version and
certificate. Signing changes the file, so `SHA256SUMS` describes the unsigned build.

## Building by hand

The same recipe the workflow runs, on a clone of the engine at the tag you want:

```
scripts/build-helpers.sh aarch64-apple-darwin dist --engine-ref v0.2.2                     # on an Apple silicon Mac
scripts/build-helpers.ps1 -Target x86_64-pc-windows-msvc -OutDir dist -EngineRef v0.2.2   # on Windows x64
```

Both take `--helpers` / `-Helpers`, `--llm-name` / `-LlmName` and `--stt-name` / `-SttName`, and refuse a target that is
not the computer they run on (the native libraries are built for the machine that builds them). Windows ARM64 needs the
toolchain of `.github/actions/windows-arm64-toolchain` set up first, and the Vulkan builds need the Vulkan SDK
(`VULKAN_SDK`). Each script ends by running the startup handshake against what it built:

```
{"op":"ping"}                                              (one line on the helper's stdin)
{"event":"ready","backend":"cpu","vram_mb":0,"protocol":2}  (the helper's first line on stdout; backend and device vary)
```

`scripts/smoke-helpers.ps1 -Dir <folder> -Triple <triple> -Names <name>,<name>` does that, and checks `SHA256SUMS`, for
files you already have; this repository's CI runs it on every artifact, on a runner of the same kind, after the workflow
has built them.
