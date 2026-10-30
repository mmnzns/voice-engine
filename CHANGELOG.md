# Changelog

## Unreleased (v0.2.0)

- llm: `gpu: false` now means the processor on the Mac too. Before, the helper put every layer on Metal whenever it had
  not chosen a Vulkan device, so a `load`, `complete` or `embed` with `gpu: false` still ran on Metal and the helper
  said `metal` (found by the benchmark of 2026-10-29); the client's fall back to the processor was meaningless on a
  Mac, where both helpers are the same binary. Now `gpu: false` puts no layer on any graphics device, in a Metal build
  and a Vulkan build alike, and keeps the KV cache and the big batches off it too; `gpu: true` is as before. Where a
  request runs is part of what decides whether the loaded model can be reused: a model loaded for `gpu: true` is
  loaded again for a `gpu: false` request (and the other way round), which is why `loaded.seconds` and
  `done.load_seconds` are above 0 on the switch. `Reply` and `embedded` gain `backend` (the same `backend`, `device`
  and `vram_mb` as `ready` and `loaded`, absent from an older helper): `cpu` for the processor, `metal` or `vulkan`
  (with the device) for the card, said per request instead of once at start. `LlmClient`'s `Outcome::backend` now
  uses it, so a request with `gpu: false` says `cpu` in a helper that started on Metal; from an older helper it keeps
  what `ready` said. Protocol stays 2 (new fields have defaults). Measured on an M3 Pro with Qwen3.5 4B: 14-token
  cleanup 0.52 s on Metal, 1.02 s on the processor, same text; 128 tokens 3.3 s against 5.3 s (38.6 against 24.1 tokens
  per second), different text after 48 bytes. Processor rows for Qwen3.5 4B and Granite 4.2 3B are in
  `docs/bench-2026-10-29.md`. The Vulkan build was not compiled here (no Vulkan SDK).
- catalog: `keeps_fillers` is measured for `whisper-turbo` and `parakeet-ultra`: `Some(true)` for both (every other
  model stays `None`). New ignored test `crates/stt/tests/real_fillers.rs`: 7 sentences with 9 spoken fillers (um, uh,
  hmm, like) in three macOS voices, in `Timing::Words`, with `keep_fillers` false and true. Both models wrote 24 of the
  27 spoken fillers with the option off; Whisper 25 with it on (the option added one "hmm"), Parakeet 24 (it ignores the
  option). `um`, `uh` and `like` came back every time; `hmm` rarely (Whisper) or never (Parakeet, which wrote "um"
  instead or nothing). Text-to-speech fillers are cleaner than real ones, so this likely overstates real speech:
  the table and the caveats are in `docs/bench-2026-10-29.md`, "Fillers". Nothing in the engine changed.
- bench: speed numbers on real models, and the downloader proved against Hugging Face, in `docs/bench-2026-10-29.md`.
  New examples `voice-engine-stt` `bench` (load time, decode time and real-time factor for a 29 s clip, next to the apps'
  bars), `voice-engine-client` `bench_llm` (a 50-word cleanup through `LlmClient` and the real helper) and
  `voice-engine-models` `download_proof` (`SharedStore::download` against the real network, a cancel at half and the
  resume). On an M3 Pro, 18 GB: Whisper Large Turbo on Metal 19.9x (the bar was about 25x, missed), Whisper Base on the
  processor 0.79 s for 29 s (0.72 s with 6 threads; the bar was 0.50 s, missed), Parakeet Ultra on the processor 0.49 s
  (0.44 s with 12 threads; the bar was 0.43 s, missed by 0.01 s). `whisper-base` downloaded at 67.6 MB/s and both files
  matched the catalog's sha256; the cancelled `whisper-tiny` was `Partial { cancelled: true }`, then resumed to
  `Installed`. The AI helper had no processor mode on the Mac, `gpu: false` still ran on Metal (fixed in the entry
  above). Nothing in the engine changed in this entry.
- build: the native libraries have a fixed CPU baseline. `.cargo/config.toml` sets `GGML_NATIVE=OFF`, the x86 baseline
  `GGML_AVX/AVX2/FMA/F16C/BMI2=ON` (Intel Haswell and AMD Excavator, 2013 and 2015, and newer) and
  `MACOSX_DEPLOYMENT_TARGET=11.0`, and passes `-C target-feature=+avx2,+fma,+f16c,+bmi2` for `x86_64-pc-windows-msvc`.
  Before this, whisper.cpp was built with `GGML_NATIVE=ON` (the build machine's own CPU) and both libraries with
  `-mmacosx-version-min=` set to the build Mac's macOS (27.0 on a macOS 27 Mac) instead of 11.0. Cargo does not read
  this file for an app that uses the engine as a git dependency: the README section "Build settings an app must copy"
  says what to copy. New: `scripts/check-baseline.sh` (fails unless the CMake caches say `GGML_NATIVE:BOOL=OFF` and, on
  a Mac, macOS 11.0), run by CI on macOS, with a PowerShell check of `GGML_NATIVE` and `GGML_AVX2` on Windows. CI also
  runs on pushes to `v0.2.0-step-*` branches.
- build: the helpers can be built for an app's release. New reusable workflow `.github/workflows/build-helpers.yml`
  (`workflow_call` and `workflow_dispatch`; inputs `engine-ref`, `helpers`, `llm-name`, `stt-name`, `cache`): it checks out
  the engine at `engine-ref`, so the engine's `.cargo/config.toml` baseline applies (Cargo does not read it for a git
  dependency), builds `voice-engine-llm` and `voice-engine-stt` for `aarch64-apple-darwin`, `x86_64-pc-windows-msvc` and
  `aarch64-pc-windows-msvc`, and uploads `helpers-<target triple>` with the files named as Tauri sidecars
  (`<name>-<target triple>[.exe]`; Vulkan variants of the Windows x64 build as `<name>-vulkan-<target triple>.exe`),
  `SHA256SUMS` and `build-info.txt` (engine ref and commit, rustc, the native library versions from `Cargo.lock`). New:
  `scripts/build-helpers.sh` and `scripts/build-helpers.ps1` (the recipe the workflow runs, with the baseline checked on
  what was built), `scripts/smoke-helpers.ps1` (checksums and the startup handshake, expecting protocol 2),
  `.github/actions/windows-arm64-toolchain` (clang-cl, Ninja, `/EHsc`, shared with the CI test job) and
  `docs/building-helpers.md` (how an app calls it, Tauri `externalBin`, closing a helper before a Windows update, signing
  notes). CI calls the workflow on every push and runs each artifact's handshake on a runner of the same kind. Windows
  ARM64 (`aarch64-pc-windows-msvc`) is built and tested in CI with clang-cl and Ninja; its CPU baseline is armv8-a with
  no dot-product kernels (ggml's default with `GGML_NATIVE` off).
- protocol 2 (`PROTOCOL_VERSION` 2): `configure` (idle time, or never exit while the app runs; can be sent at any
  time), `load` (load now, answers `loaded` with the load time and the model's trained context; a different path
  replaces the loaded model), `unload` (frees the model, the helper keeps running), `cancel` (stops the reply, the
  model stays loaded), `inspect` (a GGUF's chat template, thinking and trained context without loading it).
  `complete` gains `thinking`, `thinking_budget` and `seed`; `done` gains `thinking_tokens`, `load_seconds` and
  `thinking_text`. Errors carry `error.kind` (`model_missing`, `out_of_memory`, `prompt_too_long`, `bad_schema`,
  `no_chat_template`, `cancelled`, `gpu_failed`, `bad_request`, `internal`); an error from an older helper reads as
  `internal`.
- llm: thinking is off unless asked for; the reply never contains a thinking block. With a schema and thinking on,
  the schema applies to the answer only. The helper exits when its stdin closes (the app quit or crashed) whatever
  the idle policy. No user text is ever written to stderr; a test scans the source for it.
- The apps' 12 reply schemas are part of the tests.
- client: new crate `voice-engine-client`. `LlmClient` starts the AI helper, restarts it when it dies (a helper that
  dies at startup is started again up to 3 times, for the Mac's Metal shader cache crash), retries a request once
  when the helper exited just as it arrived, and reports every failure as an `EngineError` (the helper's kinds plus
  `Crashed`, `VersionMismatch`, `TimedOut`, `CouldNotStart`). `cancel()` stops a reply and keeps the model (a helper
  that ignores it is killed after 2 s). With a Vulkan helper configured it is tried first; if it fails, the
  processor helper does the work, the result says `fell_back`, and the failure is remembered in a file per build and
  driver until `clear_gpu_failure`. Limits (start, stall, total) are checked on every pass, so a helper that keeps
  streaming progress is still stopped.
- stt: the speech API. `load(model, dir, LoadOptions { gpu, threads })` gives a `Box<dyn Transcriber>`;
  `transcribe(audio, &ChunkOptions { language, timing, keep_fillers, vocabulary })` gives a `Transcript` (segments, each
  with its timed words, the language, `stopped_early_at_ms`); `backend()` says where it runs (`metal`, `vulkan`,
  `directml` or `cpu`) and `fell_back` when the graphics card was asked for and not used (also for ONNX models on the
  Mac, which have no card path). `Transcriber` has a `Send` supertrait, so a loaded model can sit behind a mutex or
  move to a worker thread. Whisper (Metal on the Mac, Vulkan with the `vulkan` feature, any length in one call),
  Parakeet, Parakeet Unified and Nemotron (ONNX Runtime through parakeet-rs, one chunk of up to `chunk_secs`, 40 s;
  longer is `TooLong`). `Timing::Segments` is Talaan's settings and `Timing::Words` is Ensayo's; both return segments
  and words. `vocabulary` is Whisper's initial prompt (it nudges, it does not force a spelling; the engine ends the
  list with a full stop to keep capitals and punctuation); `keep_fillers` adds whisper.cpp's filler prompt. Decoding
  that stops with 2 s or more of audio left is resumed from there (up to 8 passes) and reported in
  `stopped_early_at_ms`. `SpeechError` has kinds: `ModelMissing`, `BadModel`, `OutOfMemory`, `Unsupported(String)`,
  `TooLong { max_secs }` and `Internal(String)`.
- stt: new modules `api`, `threads` (the thread rule: on the processor three quarters of the logical cores, 2 to 12;
  on a graphics card all of them, 2 to 8; `LoadOptions::threads` overrides it), `whisper_engine`, `parakeet_engine`,
  `vad` (`vad::detect`: Silero through whisper.cpp, Talaan's settings, always on the processor), `diarize`
  (`diarize::load` and `Diarizer::run`: up to 8 speakers every 10 ms from NVIDIA's Nemotron 3 Diarization through
  parakeet-rs `sortformer`, with the 10-minute warm-up, and a workaround for parakeet-rs 0.3.8's "non-contiguous
  layout" failure on some track lengths) and `words` (`words::clean`, from Ensayo). New dependencies: whisper-rs
  0.16.0, parakeet-rs 0.3.8 and ort =2.0.0-rc.13. `tests/no_llama.rs` fails if the stt crate can reach llama.cpp or
  `voice-engine-llm`.
- stt: new crate features: `vulkan` (Whisper on the Windows graphics card, for the helper build) and `directml`
  (Parakeet TDT on the Windows graphics card, in process). Nothing uses CoreML.
- stt helper on the Transcript API, protocol 2, **breaking for anyone who had adopted the v0.1.0 speech messages**
  (nobody had): `load { model_id, model_dir, kind, gpu, threads }` (the catalog says which file in the folder is the
  model), `transcribe { audio_b64, options }` (language, timing, keep_fillers, vocabulary), `unload`, `ping`,
  `cancel`, `devices`; answers `loaded { seconds, backend }`, `transcribed { transcript, backend }` (segments with
  their words, language, `stopped_early_at_ms`; `backend` has `fell_back`), `unloaded`, and `error { error: { kind } }`
  with the speech kinds (`model_missing` with the path, `bad_model`, `out_of_memory`, `too_long` with `max_secs`,
  `unsupported`, `gpu_failed` with `fell_back`, `cancelled`, `bad_request`, `internal`). The v0.1.0 `load
  { model_path }`, `transcribe { samples, language }` and `transcribed { pieces }` are no longer read. `ChunkOptions`,
  `Transcript`, `Segment`, `Word`, `Timing` and `SpeechBackend` are defined in the protocol crate and re-exported by
  `voice-engine-stt`, so the in-process engines, the helper and the client share them. The helper serves Whisper
  only (the ONNX models run in the app's process). Audio is base64 of little-endian f32, as in v0.1.0 (about 2.5 MB
  for 30 s). The helper keeps its model between calls and exits when stdin closes. `voice_engine_stt::threads()` and
  `thread_count` (v0.1.0's formulas) are gone; `transcribe` stays only as the reference Talaan's settings are tested
  against.
- client: `SttClient` (`SttConfig`, `load`, `transcribe`, `unload`, `cancel`, `shutdown`), built like `LlmClient`:
  startup retried 3 times, protocol check, graphics card helper first with the remembered fallback (`fell_back`),
  limits checked on every pass. It remembers the loaded model and loads it into any helper it starts again (after a
  crash, a kill, a card fallback), so a crash costs the app one `Crashed` and nothing else. `EngineError` gains
  `BadModel`, `TooLong { max_secs }` and `Unsupported { detail }`. Cancel: the helper cannot interrupt a Whisper
  decode, so a cancel ends the call after the 2 s grace by stopping the helper.
- client: `servers::detect()` says which local AI servers run on this computer (Ollama on port 11434, LM Studio on
  port 1234) and lists each one's models with `id`, `context_length` and `size_bytes` (`detect_at(&[Probe])` takes
  other addresses, `default_probes()` gives the two defaults). Ollama: `/api/tags` for the list, `/api/show` per model
  for the context length. LM Studio: `/api/v0/models`, keeping `llm` and `vlm` and dropping `embeddings`. Each
  server says `Running` or `Unreadable` (something answers, but not with a list we can read); a server that is not
  there is left out, so neither being installed is an empty list, not an error. Connecting gives up after 300 ms and a
  server gets 2 s in all; the servers are asked at the same time. No retries, no logging, and no error carries a
  model name. `LocalServer::api_base` is the `/v1` root the apps send their chat requests to; the engine does not
  make those requests. New dependency of the client crate: `reqwest` (blocking, rustls, the workspace's).
- Known: on a Mac, several helpers starting at the same moment can make one die with SIGBUS inside Apple's Metal
  shader cache (not engine code). An app's client must restart a helper that dies at startup.
- Catalog: one catalog for every kind of model (speech, voice detection, speaker separation, search, local AI).
  `Kind` replaces `EngineKind` and has a `Vad` variant. New fields: licence, GPU support (Mac Metal, Windows Vulkan,
  Windows DirectML), English-only, chunk length, fillers, and the local AI context sizes (`LlmInfo`). Ten new speech
  variants: English-only Whisper Tiny, Base, Small and Medium; full-size Whisper Large v2, Large v3 and Turbo;
  full-size Parakeet v2, v3 and Unified. New local model: Granite 4.2 3B. Parlato's four older local models are not
  offered (an existing file is kept as an imported GGUF by the shared-folder move, step 4). Removed: `llm_catalog`,
  `EngineKind`, `DEFAULT_MODEL` and the `VAD`, `DIARIZATION` and `SEARCH` statics. An app migrating uses
  `catalog::find`, `catalog::speech()`, `catalog::local_ai()` and `Kind`.
- Shared model folder: one folder for all three apps (Mac `~/Library/Application Support/Craft Concepts Digital/Models`,
  Windows `%LOCALAPPDATA%\Craft Concepts Digital\Models`), `<model id>/<files>` plus an `.engine` folder with
  `users`, `locks` and `moved`. `SharedStore::open(app)` (`VOICE_ENGINE_MODELS_DIR` overrides the folder),
  `open_at(app, dir)` for tests and dev builds (never moves anything in), `read_only(app, dir)` (creates nothing),
  `default_dir`, `status` and `list` (`Installed`, `Downloading`, `Partial`, `Missing`), `use_model`, `download`,
  `hold` (a `ModelHandle` held while a model is loaded), `release`, `release_all` and `collect`. A model is erased
  when the last app that uses it lets go and no app has it loaded.
- Downloader: one `.part` file per file, resumed with a Range request, a fresh start when the server ignores Range,
  size and sha256 checked before the rename, cancel and a 30 s stall check about every 200 ms, progress over the
  whole model (ported from Talaan, see SOURCES.md). Errors are `DownloadError`: `Offline`, `Refused { status }`,
  `DiskFull`, `Checksum { file }`, `Stalled`, `Cancelled`, `Io`; `is_failure()` is false for `Cancelled` and `Stalled`.
- One-time move (`SharedStore::move_in(used)`): each app's own old models are moved in (renamed, with a hard link left
  at the old path), other apps' complete models are hard-linked in, never touched. Parlato's name table is in
  `legacy.rs`; its Parakeet files are hash-checked first because Parlato never checked them. A file in use is left
  for the next start. Parlato's dropped local AI models and the person's own GGUFs stay where they are and come
  back as `ImportedGguf` with an id from `imports::import_id` (`import-` and 12 hex digits of sha256 over the size,
  the first MiB and the last MiB, so the same file has the same id after a rename).
- Removed: `ModelStore` and `Location` (the finder that looked in an app's own folder and in other apps' folders),
  `ModelInfo::reuse_dirs`, `OtherRoot`, `Layout` and `App::root`, `App::others`, and `apps.rs` (now `legacy.rs`,
  which keeps `App`). Apps use `SharedStore::open(app)`, then `move_in`, `status`/`list` and `download` instead of
  looking in folders themselves; Delete is `release`, Delete all data is `release_all`.

**Native library versions.** The helpers and the speech crate are built from these, as pinned by `Cargo.lock` at this
version: llama-cpp-2 and llama-cpp-sys-2 `=0.1.159` (llama.cpp), whisper-rs 0.16.0 with whisper-rs-sys 0.15.0
(whisper.cpp), parakeet-rs 0.3.8 with ort `=2.0.0-rc.13` and ort-sys 2.0.0-rc.13, which downloads ONNX Runtime 1.28.0
(every prebuilt in `ort-sys-2.0.0-rc.13/build/download/dist.tsv` is `ms@1.28.0`). Any change to one of these versions is
announced in this change log. Every release's `build-info.txt` (see `docs/building-helpers.md`) lists the same versions
for the helpers it holds.

## v0.1.0 (2026-10-28)

First version, from Talaan's and Ensayo's engine code (see SOURCES.md).

- protocol: Talaan's and Ensayo's helper messages merged (Ensayo's `decode_request`, `decode_event` and
  `JSON_OBJECT_SCHEMA` added). New: `PROTOCOL_VERSION` (1), sent in `Ready`; helpers from before the engine read as 0.
  New: `HelperConfig { env_prefix }` names each app's environment variables (`<PREFIX>_GPU`,
  `<PREFIX>_LLM_IDLE_SECS`); `FAIL_ENV` is gone.
- llm: the local AI helper as `run(HelperConfig)`, with Talaan's GPU choice and embeddings and Ensayo's idle exit
  override (`<PREFIX>_LLM_IDLE_SECS`). The prompt is read only after the output schema is checked, so a bad schema
  fails before the model reads anything. The context size no longer forces a model reload on the processor or Metal
  (it still does on a GPU, where it changes how the model is fitted). Error messages are plain and name no app.
- stt: Talaan's Whisper chunk code and helper as `run_helper(HelperConfig)`.
- models: speech catalog (Talaan's), local AI catalog (Qwen3.5 4B, 9B), the known apps table, and `ModelStore`.
  Parlato's Windows folder is under %LOCALAPPDATA% (Talaan looked in %APPDATA% before). The note on the two smallest
  Whisper models says "the app" instead of an app name.
- llama-cpp-sys-2 is locked at 0.1.159, the version both apps ship.
