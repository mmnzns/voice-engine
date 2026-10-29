# voice-engine v0.2.0 design

Date: 2026-10-29. Status: draft for Miguel's review. Source of requirements: the Notion page "Voice Apps Engine
Sync", sections "What the engine is, and is not", "Decisions" and "Engine v0.2.0: what it must do" (item codes such
as `E-A1` below point there).

## 1. In one paragraph

v0.2.0 makes the engine the one place where local models are run, listed and stored for Parlato, Talaan and Ensayo.
A new model, or a newer llama.cpp, whisper.cpp or Parakeet, is changed once here and reaches all three apps on their
next release. All three apps keep their models in one shared folder: a model downloaded by one app is used by the
others without a second download, and a model is erased only when the last app that uses it lets go of it. The
local AI helper gains the controls the apps asked for (keep loaded, load now, stop, thinking off, token counts,
errors as kinds, any reply shape). Each app keeps its own speech pipeline, its own BYOK cloud code, its own
recommended models and its own screens.

## 2. Goals and non-goals

Goals:

1. Supporting a new local AI or speech model, or updating a model runtime, is one change in this repository.
2. "What is on this computer" works for every app, from one shared folder, with no duplicate downloads.
3. Everything the three apps listed as needed before they move (part A of each list) is done, so each app can move
   without losing anything it does today.
4. Each app's results stay the same after the move unless it chooses otherwise (same Whisper settings, same
   Parakeet passes, same sampling).

Non-goals (decided 2026-10-29): cloud AI and cloud speech (BYOK) in the engine; live text while speaking; each
app's speech pipeline (recording, chunking, two tracks, echo removal, language detection, transcript cleanup);
recommended models and wording; prompts and reply schemas; spending caps; signing and updaters. An online model
list (decided: the list is built in).

## 3. Decisions this design rests on

| Decision | By | Date |
| --- | --- | --- |
| One release, v0.2.0; no app moves before it is tagged | Miguel | 2026-10-29 |
| Engine scope: runtimes, catalog, shared folder, helper controls | Miguel | 2026-10-29 |
| One shared model folder, one-time move of each app's models | Miguel | 2026-10-29 |
| Model list built into the engine, not fetched online | Miguel | 2026-10-29 |
| Delete: the model leaves this app; the file is erased when no app uses it | Miguel | 2026-10-29 |
| Live text dropped; cloud speech stays Parlato only | Miguel | 2026-10-29 |
| No model-sharing setting: with one folder there is nothing to turn off | this spec | 2026-10-29 |

Ownership rule (unchanged): no code from Parlato, Parla, VoiceInk or any GPL source. Code comes from Talaan and
Ensayo (MIT) and from upstream (llama.cpp, whisper.cpp, parakeet-rs, ONNX Runtime, the models' own docs). Facts
about Parlato (folder names, file names, model ids) come from the Parlato chat as data on the Notion page, never by
reading Parlato's code. SOURCES.md records every import.

## 4. Architecture

```
            app process (Tauri)                                  helper processes (bundled in the app)
 ┌─────────────────────────────────────────────┐     stdin/stdout JSON lines
 │ app code: pipeline, prompts, screens, BYOK  │   ┌────────────────────────────────────┐
 │                                             │──▶│ <app>-llm        (voice-engine-llm) │ llama.cpp, Metal / CPU
 │ voice-engine-client  ── spawns, restarts ───┼──▶│ <app>-llm-vulkan (Windows, GPU)     │ llama.cpp, Vulkan
 │                                             │──▶│ <app>-stt-vulkan (Windows, GPU)     │ whisper.cpp, Vulkan
 │ voice-engine-stt  (in process)              │   └────────────────────────────────────┘
 │   whisper.cpp (Metal on Mac), Silero VAD,   │
 │   Parakeet / Nemotron / Sortformer via ORT  │
 │ voice-engine-models                         │──▶ shared folder (Craft Concepts Digital/Models)
 │   catalog, shared store, downloader, move   │
 └─────────────────────────────────────────────┘
```

llama.cpp and whisper.cpp each bundle their own ggml and must never share a process, so local AI stays in a helper
process (as today). Speech stays in the app's process (as today), except the optional Windows Vulkan Whisper helper.

Crates:

| Crate | What it does | Change in v0.2.0 |
| --- | --- | --- |
| `voice-engine-protocol` | Helper messages | Protocol 2 (section 7), error kinds |
| `voice-engine-llm` | The local AI helper | Controls (section 7) |
| `voice-engine-stt` | Speech runtimes, in process | Parakeet, Nemotron, Sortformer, Silero VAD, word timings, Metal (section 9) |
| `voice-engine-models` | Catalog, shared store, downloader, one-time move | Rewritten store; downloader; catalog fields (sections 5, 6) |
| `voice-engine-client` | New: the app side of the helpers, and local AI servers | From Talaan's helper code (section 8) |

## 5. The shared model folder

### 5.1 Where

| System | Folder |
| --- | --- |
| Mac | `~/Library/Application Support/Craft Concepts Digital/Models/` |
| Windows | `%LOCALAPPDATA%\Craft Concepts Digital\Models\` |

`VOICE_ENGINE_MODELS_DIR` overrides it (tests, and an app's own test data folder such as `TALAAN_DATA_DIR`).
Neither app is sandboxed on the Mac, so all three can read and write it.

### 5.2 Layout

```
Models/
  <model id>/<files>              every catalog model: speech, local AI, voice detection, speaker separation, search
  <model id>/<file>.part          a download in progress
  .engine/
    users/<model id>/<app>        empty file: this app uses this model
    locks/<model id>.use          shared lock while an app has the model loaded
    locks/<model id>.download     exclusive lock while one app downloads it
    moved/<app>                   empty file: this app's one-time move is done
```

One layout for every kind of model (Ensayo's `models/llm/` split goes away). Model ids are the catalog ids.

### 5.3 Using, downloading, deleting

API (`voice_engine_models::SharedStore`, opened with the calling `App`):

- `status(id)` returns `Installed { dir, users, used_by_me }`, `Downloading { fraction, by_me }`,
  `Partial { fraction }` or `Missing`. "Installed and not used by me" is how an app shows "Already on this
  computer": one click (`use_model`) adds it, with no download.
- `use_model(id)` writes `users/<id>/<app>`.
- `download(id, cancel, progress)` writes the user marker first, takes `locks/<id>.download` exclusively and runs
  the downloader (5.5). If another app holds the lock, the call returns `DownloadingElsewhere` and the app shows that
  download's progress (read from the `.part` sizes) until it completes.
- `open(id)` returns a `ModelHandle` holding a shared lock on `locks/<id>.use` for as long as the model is loaded.
  The app keeps the handle next to its loaded model and drops it on unload.
- `release(id)` removes this app's user marker. If no users remain, it tries an exclusive lock on
  `locks/<id>.use`; when it gets it, it re-checks the users under the lock and erases `<id>/`. If another app has
  the model loaded, the folder stays and is erased later by `collect()`.
- `release_all()` releases every model this app uses: what an app's "Delete all data" calls instead of removing a
  models folder.
- `collect()` erases models with no users and no use lock, and `.part` files older than 7 days with no download
  lock. Each app calls it at start.
- What the app tells the person when Delete leaves the file in place: `release` returns `Kept { users }`, so the app
  can say "Talaan still uses this model, so it stays on this computer."

Locks are OS file locks (`std::fs::File::lock_shared` / `try_lock`), released by the OS if an app crashes, so a
crash never leaves a model stuck. The Windows rule that an open file can't be deleted is covered by the use lock.

### 5.4 One-time move

`SharedStore::move_in(&self)` runs on each app's first start on v0.2.0 and is safe to run again (it skips work
already done and is safe if interrupted: each file is renamed or linked on its own, the `moved/<app>` marker is
written last).

1. **This app's own old folders** (table below). For each catalog model complete there (every file, exact size):
   - Not in the shared folder: rename each file into `<id>/` (same volume: instant). Then create a hard link from
     the old path to the new file, so an older version of another app that still reads this app's old folder keeps
     working, with no extra disk space.
   - Already in the shared folder (same size): delete the old copy, then hard-link the shared file back at the old
     path (same reason).
   - Mark this app as a user.
2. **Other apps' old folders**: for each catalog model complete there and missing from the shared folder,
   hard-link (not move) the files into `<id>/` and mark that other app as a user (it has the model, so `collect`
   must not erase it). No disk space, and the other app's old version is untouched. When that app later moves in,
   it finds the model already shared and only removes its own entry. This is how Ensayo, which uses Talaan's copies
   today, ends up with no downloads.
3. Marks this app as a user of each model id the app passes in (`move_in(used: &[&str])`): the models its settings
   currently point to, wherever they were found.
4. A hard link that fails (another volume, a file system without them) is skipped for other apps' folders; for the
   app's own files the rename still happens, and the old path is simply left empty.

Old folders (the only place the engine still knows about the apps' old layouts; `apps.rs` becomes `legacy.rs`):

| App | Mac | Windows |
| --- | --- | --- |
| Talaan | `Talaan/models/<id>/` | `%LOCALAPPDATA%\com.craftconceptsdigital.talaan\models\<id>\`, and the older `%APPDATA%` folder |
| Ensayo | `com.craftconceptsdigital.ensayo/models/<id>/`, AI in `models/llm/<id>/` | same under `%LOCALAPPDATA%` |
| Parlato | `com.craftconceptsdigital.parlato/Models/<file>`, `ParakeetModels/<folder>/`, `LlmModels/<parlato id>.gguf` | same under `%LOCALAPPDATA%` |

Parlato's folder and file names per catalog model (for example which `ParakeetModels/<folder>` is `parakeet-v3`, and
which `LlmModels/<id>.gguf` is which model) are data the Parlato chat writes on the Notion page; the engine copies
that table into `legacy.rs`.

### 5.5 Downloader

Talaan's `transcriber/download.rs` (MIT; Ensayo's is a copy) moves into `voice-engine-models`, unchanged in
behaviour: one file at a time into `<file>.part`, HTTP Range resume, a fresh start when the server ignores the range
or answers 416, a web page in place of the file is an error, sha256 computed while streaming and checked with the
size before the rename, cancel checked every 200 ms, 30 s without data is `Stalled` (the `.part` is kept, so Retry
resumes), `partial_fraction` for leftovers. Blocking API with a cancel flag and a progress callback; reqwest 0.12
with rustls, the version both apps already link.

"Installed" means every file present with its exact size (hashes are checked at download, as today).

### 5.6 Your own GGUF file

An imported GGUF stays where the person put it (never copied into the shared folder, never erased by the engine).
`inspect` (section 7) checks it loads and has a chat template, and reports its context size. Its stable id is
`import-` plus the first 12 hex digits of sha256(size, first MiB, last MiB), so it survives a rename and stays the
same in every app. Any GGUF by path keeps working.

## 6. Catalog

One built-in catalog for every kind of model (`voice_engine_models::catalog`), in this shape:

```rust
pub struct ModelInfo {
    pub id: &'static str,
    pub name: &'static str,
    pub maker: &'static str,
    pub note: &'static str,          // plain, names no app
    pub kind: Kind,                  // Whisper, Parakeet, ParakeetUnified, Nemotron, Vad, Diarization, Embedding, Llm
    pub languages: &'static str,
    pub english_only: bool,
    pub speed: u8,                   // 1..3
    pub accuracy: u8,                // 1..3
    pub licence: Licence,            // SPDX id and a link
    pub files: &'static [ModelFile], // name, pinned url, size, sha256
    pub gpu: GpuSupport,             // mac_metal, windows_vulkan, windows_directml
    pub keeps_fillers: Option<bool>, // speech: measured (section 9.4); None until measured
    pub llm: Option<LlmInfo>,        // context_max, context_8gb, context_16gb, thinks
    pub offered: bool,               // false: recognised and runnable, not offered for new downloads
}
```

Contents:

- Speech: everything Talaan and Ensayo list today (Whisper Tiny to Large v3 Turbo compressed, Parakeet v2, v3,
  Ultra, Unified, Nemotron 3.5), plus Parlato's variants built from upstream sources: English-only Whisper Tiny,
  Base, Small and Medium; full-size Whisper Large v2, v3 and v3 Turbo; full-size Parakeet v2, v3 and Unified. Every
  file pinned to a commit with size and sha256, as today.
- Voice detection (Silero v6.2.0), speaker separation (Nemotron diarization v3) and search (Qwen3 Embedding 0.6B).
- Local AI: Qwen 3.5 4B and 9B; Granite 4.2 3B; Parlato's four older models (Qwen 2.5 3B, Llama 3.2 3B, Gemma 2 2B,
  Phi-3.5 mini) with `offered: false` until Miguel decides (so Parlato's existing downloads are still recognised,
  moved and runnable either way).

The catalog never says which model is recommended. Each app keeps its own list order and recommendation text. Tests
keep today's rules (every url pinned, names match, sizes plausible) and add: ids unique across all kinds, every
`Llm` entry has `llm`, every speech entry says its GPU support.

## 7. Local AI helper, protocol 2

`PROTOCOL_VERSION` becomes 2. Every message stays one JSON line. Additions are backwards readable (new fields have
defaults) so tests and fake helpers keep working.

Requests (new ones in bold):

| Request | Answer | Notes |
| --- | --- | --- |
| `ping` | `ready` | Unchanged; reports protocol 2 |
| `devices` | `devices` | Unchanged |
| **`configure { idle_secs }`** | `configured` | `idle_secs: null` = never exit (Parlato); a number = exit that long after the last request (Ensayo: a couple of minutes). Default stays 120 s. The `<PREFIX>_LLM_IDLE_SECS` variable stays for tests. (P-A1, E-A5) |
| **`load { model_path, n_ctx, gpu }`** | `loaded { seconds, n_ctx_train, backend }` | Loads without generating. (P-A2, E-A3) |
| `complete { …, thinking, thinking_budget, seed }` | `done` | New fields below |
| **`cancel`** | `error { kind: "cancelled" }` for the running request | The helper and its model stay loaded. Ignored when nothing is running. (P-A3, E-A4) |
| `embed` | `embedded` | Unchanged; keeps working on the processor in a GPU build (T-B) |
| **`inspect { model_path }`** | `inspected { has_chat_template, n_ctx_train, architecture, thinks }` | Reads the GGUF's metadata only (5.6) |

Cancel: stdin is already read on its own thread. While writing a reply the helper checks for a waiting line after
every token; `cancel` stops the reply, any other request that arrives meanwhile is queued and run next.

New `complete` fields:

- `thinking: bool`, default false. Off: the prompt is built with thinking disabled (llama.cpp's chat template
  `enable_thinking=false` where llama-cpp-2 0.1.159 exposes it; otherwise the empty think block that Qwen's own
  template inserts for that setting). On or off, any `<think>…</think>` that still appears is taken out of `text`
  (Ensayo's `strip_think`, MIT, covers an unclosed block and a missing opening tag). (P-A4, E-A2, T-B)
- `thinking_budget: Option<u32>`: with thinking on, the most thinking tokens before the helper ends the block
  itself. `max_tokens` counts only the answer.
- With thinking on and a `json_schema`, the grammar applies from the end of the thinking block, so the thinking is
  free text and the answer matches the schema.
- `seed: Option<u32>`, default 1234 (today's fixed seed). Same request, same model, same computer and backend: the
  same reply. `temperature: 0` stays greedy. (P-A7, E-A8)

`done` gains `thinking_tokens`, `load_seconds` (0 when the model was already loaded) and `thinking_text` (only when
thinking was on). It keeps `text`, `prompt_tokens`, `output_tokens`, `seconds`, `truncated`. (E-A7)

Errors: `error { kind, message, … }` with a fixed set of kinds the apps match on (never the text): (E-A6, T-B)

| Kind | Extra fields |
| --- | --- |
| `model_missing` | `path` |
| `out_of_memory` | |
| `prompt_too_long` | `prompt_tokens`, `n_ctx`, `max_tokens` |
| `bad_schema` | `detail` |
| `no_chat_template` | |
| `cancelled` | |
| `bad_request` | |
| `internal` | |

`crashed`, `gpu_failed`, `version_mismatch` and `timed_out` are added by the client (section 8), which is the one
that sees a process die.

Reply shapes: any JSON Schema the apps send goes through llama.cpp's converter. The engine's tests convert Ensayo's
11 schemas and Talaan's summary schema (copied into `crates/llm/tests/schemas/`, MIT) and fail on any error. The
grammar enforces the structure (objects, required keys, `additionalProperties: false`, `enum`, `const`, arrays and
their sizes); keywords the converter can't enforce (it may not cover `pattern`, `minimum` or `maxLength`) are listed
in the README, and the apps keep their own check after the reply, as Ensayo does today. (E-A1, T-B)

## 8. voice-engine-client (new)

The app side of the helpers, which each app has written separately until now. Built from Talaan's
`summarizer/local.rs`, `helper.rs`, `transcriber/remote.rs` and `gpu.rs` (MIT), with Ensayo's retry-once and limits.
Blocking API, safe to call from a worker thread (Ensayo), wrapped by each app in whatever async it uses.

- `LlmClient::new(LlmConfig { cpu_helper, gpu_helper: Option<PathBuf>, env_prefix, gpu_failures: PathBuf, idle,
  limits })`. Starts the helper on first use, checks `ready` and that the protocol matches (else
  `version_mismatch`), sends `configure`, and restarts it after a crash or an idle exit (retrying a request once when
  the helper had just exited for idling).
- `load`, `complete`, `embed`, `inspect` return `Result<_, EngineError>` with every kind from section 7 plus
  `crashed`, `out_of_memory` (from the stderr tail when the process dies), `gpu_failed`, `version_mismatch`,
  `timed_out`. Progress through a callback.
- `cancel()` from any thread sends `cancel`; if the helper doesn't answer in 2 s, it is killed and restarted.
- GPU: on Windows, try the Vulkan helper first, fall back to the processor helper, and remember a failure per app
  version and graphics driver in the app's `gpu_failures` file so it isn't retried every time (Talaan's `gpu.json`).
- `SttClient`: the same for the Windows Vulkan Whisper helper.
- `servers::detect()`: Ollama (port 11434) and LM Studio (port 1234) if running, with each model's name and context
  size (Ollama `/api/tags` and `/api/show`, LM Studio `/api/v0/models`). The apps keep making the requests to those
  servers themselves. (E-B, T-B)

## 9. Speech runtimes

The engine transcribes one chunk (16 kHz mono, up to 40 s) with a loaded model and returns timed text. The app does
everything around it.

### 9.1 API

```rust
pub fn load(model: &ModelInfo, dir: &Path, opts: LoadOptions) -> Result<Box<dyn Transcriber>, SpeechError>;

pub struct LoadOptions { pub gpu: bool, pub threads: Option<usize> }

pub trait Transcriber {
    fn transcribe(&mut self, chunk: &[f32], opts: &ChunkOptions) -> Result<Transcript, SpeechError>;
    fn backend(&self) -> Backend;
}

pub struct ChunkOptions { pub language: Option<String>, pub timing: Timing, pub keep_fillers: bool }
pub enum Timing { Segments, Words }
pub struct Transcript { pub segments: Vec<Segment>, pub language: Option<String> }
pub struct Segment { pub start_ms: u64, pub end_ms: u64, pub text: String, pub words: Vec<Word> }
pub struct Word { pub text: String, pub start_ms: u64, pub end_ms: u64 }
```

Times are on the chunk's clock; the app adds the chunk's offset, as today. A loaded `Transcriber` stays loaded until
the app drops it, so "load when recording starts" and "stay loaded between answers" are the app calling `load`
early and keeping the object. (E-B4)

### 9.2 Whisper

- `Timing::Segments` is exactly Talaan's settings today; `Timing::Words` is exactly Ensayo's (token timestamps,
  split on word, one word per segment, then Ensayo's `words::clean`, MIT). Same model, same mode: the same text as
  the app produces today. Language `None` means detect. (E-B1, goal 4)
- Metal on the Mac, in process, with flash attention, as Talaan does today: this crate enables whisper-rs's `metal`
  feature on macOS. (E-B3, T-B)
- Threads: on the processor, three quarters of the logical cores, 2 to 12, on every system; on Metal or a GPU, all
  logical cores, 2 to 8. (T-A1)
- The Windows Vulkan helper stays as in v0.1.0, gains the timing mode, and is driven by `SttClient`.

### 9.3 Parakeet, Nemotron, speaker separation, voice detection

- Parakeet TDT (v2, v3, Ultra, full and int8), Parakeet Unified and Nemotron 3.5 through parakeet-rs 0.3.8 and ONNX
  Runtime (`ort =2.0.0-rc.13`, the pair both apps link), from Talaan's `transcriber/engine.rs` (MIT): 0.5 s of
  silence each side, up to 8 resume passes when decoding stops early, Nemotron's language prompt. `Timing::Segments`
  uses parakeet-rs sentence timestamps (Talaan), `Timing::Words` word timestamps (Ensayo). (T-B, E-B)
- On the graphics card on Windows: ONNX Runtime's DirectML provider behind the `directml` feature, used when
  `gpu: true` and falling back to the processor if it fails to start; `backend()` says which ran. (P-C2)
- Speaker separation (Nemotron diarization through parakeet-rs `sortformer`) moves from Talaan's `diarize.rs`: the
  model run returns speaker turns; naming speakers stays in Talaan. (T-B)
- Voice detection: Talaan's `vad.rs` (whisper.cpp's Silero VAD, 4 threads, processor, threshold 0.5, 250 ms minimum
  speech, 300 ms minimum silence) as `vad::detect(model_dir, samples) -> Vec<(start_ms, end_ms)>`. Chunking stays in
  the apps.

### 9.4 Fillers

`keep_fillers: true` asks a model to keep "um", "uh" and "like" (for Whisper, a short prompt of fillers, the method
whisper.cpp's docs describe; Parakeet transcribes them as heard). Each speech model's `keeps_fillers` in the catalog
is measured on a test recording with fillers before the tag, with and without the option. (E-B2)

### 9.5 Speed

A `bench` example prints load time and real-time factor per model on the computer it runs on. The bar before
tagging: Whisper Large v3 Turbo about 25x real time on Metal on the Mac; Parakeet Ultra about 50x on a Windows
processor. (E-B3) It also prints a 50-words-in, 50-out cleanup with Qwen 3.5 4B, thinking off: load time and reply
time on the Mac (Metal) and on a Windows processor, for Parlato's choice of default. (P-B2)

## 10. Builds the apps can ship

- Release builds never use the build machine's own processor features: `GGML_NATIVE=OFF` and full optimisation for
  llama.cpp and whisper.cpp, an AVX2 baseline on Windows x64 (Talaan's `cpu-baseline.ps1`, MIT). The README gives
  the exact `[env]` block for an app's `.cargo/config.toml`, and CI builds a release with it and checks the CMake
  cache for `GGML_NATIVE:BOOL=OFF`. (P-A6, E-A9)
- macOS 11 or later on Apple silicon (`MACOSX_DEPLOYMENT_TARGET=11.0` in CI and in the README block). (E-A9)
- CI adds Windows on ARM64 (`windows-11-arm`): build and test. (P-A5)
- The release checklist (README) includes running both helpers inside a self-signed, not notarized .app on the Mac.
  (E-A9)
- Pinned native libraries, written in the README and the change log, changed only together: llama-cpp-sys-2
  0.1.159, whisper-rs 0.16.0, parakeet-rs 0.3.8, ort 2.0.0-rc.13, reqwest 0.12. (T-A2, E-B6)

## 11. Testing

In CI (Windows x64, Windows ARM64, Mac), no large downloads:

- Protocol 2: every request and event round-trips as one line; old messages still read; every error kind.
- Grammar: the 12 app schemas convert without error.
- Thinking: `strip_think` cases (closed, unclosed, missing opening tag, none).
- Shared store, with temporary folders: users, use locks across two `SharedStore`s standing in for two apps, release
  with and without other users, `collect`, a crash simulated by dropping a handle, downloads by another app.
- One-time move: each app's old layouts on both systems, a model in two apps' folders, an interrupted move run
  again, a failed hard link.
- Downloader against a local test server: resume, a server that ignores Range, 416, a web page, wrong size, wrong
  sha256, cancel, stall.
- Catalog consistency; thread rule; client with a fake helper (crash, idle exit then retry, version mismatch, cancel
  timeout, GPU fallback memory).

On the Mac before tagging (`cargo test -- --ignored`, with models in the shared folder), and the Windows parts on a
PC:

- Qwen 3.5 4B and 9B: thinking off gives no think block and no thinking tokens with a real cleanup prompt; each of
  the 12 schemas gives a reply that passes the app's own check; the same request twice gives the same reply; cancel
  mid-reply then a new request works without a reload; `load` then `complete` reports `load_seconds: 0`.
- Whisper Turbo and Parakeet Ultra: word timings present and ordered; Segments mode text identical to Talaan's
  current output and Words mode identical to Ensayo's on the same clip; fillers measured.
- `bench` numbers, written to the Notion page.

## 12. Build order

Each step lands on `main` with CI green; the Notion items are ticked as they land.

1. Protocol 2 and the helper controls (section 7).
2. `voice-engine-client` (section 8).
3. Catalog rework (section 6).
4. Shared store, downloader, one-time move (section 5).
5. Speech runtimes: Whisper modes, Metal and threads; voice detection; Parakeet and Nemotron; speaker separation;
   DirectML (section 9).
6. Local AI servers and GGUF `inspect`.
7. Builds and CI (section 10).
8. Tests on real models, `bench`, README, CHANGELOG, SOURCES.md, tag v0.2.0, Notion change log and each app's "move
   to v0.2.0" block.

## 13. Where each requirement went

| Notion item | Section |
| --- | --- |
| P-A1, E-A5 idle policy | 7 `configure` |
| P-A2, E-A3 load now | 7 `load` |
| P-A3, E-A4 stop a reply | 7 `cancel`, 8 |
| P-A4, P-B1, E-A2 thinking | 7 |
| P-A5 Windows ARM64 | 10 |
| P-A6, E-A9 builds | 10 |
| P-A7, E-A8 repeatable | 7 `seed` |
| P-B2 speed numbers | 9.5 |
| P-B3, P-B4 Granite, GGUF by path | 6, 5.6 |
| P-C1, E-C1, T-C1 shared folder, Parlato's names, Talaan's old Roaming folder | 5 |
| P-C2 Parlato variants, Parakeet on the Windows GPU | 6, 9.3 |
| E-A1, T-B schemas | 7 |
| E-A6, T-B errors as kinds, GPU failure | 7, 8 |
| E-A7 token counts | 7 `done` |
| E-B1 to E-B4 word timings, fillers, speed, stay loaded | 9 |
| E-B6, T-A2 one copy of each library | 10 |
| E-B7, T-B catalog fields | 6 |
| E-B, T-B import GGUF, Ollama and LM Studio, context sizes | 5.6, 6, 8 |
| E-C2 missing shared file | 5.3 `status` |
| E-C3, T-C3 downloader | 5.5 |
| T-A1 Whisper threads | 9.2 |
| T-B embeddings, Parakeet in the engine, speaker separation and search models | 7, 9.3, 6 |

Not engine work (stays in the apps): every cloud item (P-C3, P-C4, Ensayo's and Talaan's cloud AI and cloud speech
items), live text, pipelines, prompts, schemas, recommended models, wording, signing, updaters.

## 14. Risks and things to check first

- Thinking off with llama-cpp-2 0.1.159: whether its chat template call takes `enable_thinking`. Checked in step 1;
  the fallback (the empty think block) is what Qwen's template produces anyway.
- ONNX Runtime linking: how Talaan's and Ensayo's builds get the ONNX Runtime library today (the survey found no
  `download-binaries` feature) is checked before step 5, and the README says how an app gets it, including Windows
  ARM64 and DirectML.
- Windows ARM64: llama.cpp, whisper.cpp and ONNX Runtime all build there upstream; if one doesn't build through the
  Rust crates, the engine says so in the change log rather than holding the tag, and Parlato decides.
- Hard links on unusual file systems: covered by the fallback in 5.4.
- Open decisions for Miguel that don't block the work: Parlato's four older models (6), graphics card on by default
  (each app's setting).

## 15. What each app changes when it moves (written in full at tag time)

Each app, on a branch in its own repository: depend on the four crates and `voice-engine-client` at v0.2.0; replace
its helper `main.rs` files with the engine's; replace its own model catalog, finder and downloader with
`SharedStore`; call `move_in` at start and `collect` after; use `release`/`release_all` for Delete and "Delete all
data"; replace its helper spawning code with `LlmClient`/`SttClient` and match on error kinds; move its speech model
running to `voice_engine_stt::load` with its own timing mode; keep its pipeline, prompts, recommended models,
wording and screens; offer every model in the catalog.
