# voice-engine

The local AI and speech engine shared by three desktop apps by Miguel Monzones:
[Talaan](https://github.com/mmnzns/Talaan), [Ensayo](https://github.com/mmnzns/ensayo) and
[Parlato](https://github.com/mmnzns/Parlato).

| Crate | What it does |
| --- | --- |
| `voice-engine-protocol` | Messages between an app and its helper programs |
| `voice-engine-llm` | The local AI helper (llama.cpp in its own process) |
| `voice-engine-stt` | Speech: Whisper, Parakeet, Parakeet Unified and Nemotron, voice detection, speaker separation, and the Whisper helper |
| `voice-engine-client` | What an app uses to talk to its helpers: `LlmClient` and `SttClient` |
| `voice-engine-models` | The model catalog, the shared model folder, the downloader and the one-time move |

## Local AI helper protocol

One JSON object per line. Requests (`op`): `ping`, `configure`, `load`, `unload`, `complete`, `cancel`, `embed`,
`inspect`, `devices`. Events (`event`): `ready`, `configured`, `loaded`, `unloaded`, `progress`, `done`, `embedded`,
`inspected`, `devices`, `error`. Match errors on `error.kind`, never on `message`. The current version is
`PROTOCOL_VERSION` in `voice-engine-protocol`; a helper reports it in `ready`.

Lifetime: the helper runs until its stdin closes (the app exited or crashed) or, with an idle time set, until that
long passes without a request. Name the binary as your app wants (Ensayo: the Tauri sidecar `ensayo-llm`); the build
recipe for each target triple comes in a later step of v0.2.0. The helper never logs user text: only sizes, timings,
paths and error kinds reach stderr.

Reply schemas: llama.cpp enforces structure (objects, required keys, `additionalProperties: false`, `enum`, `const`,
array sizes). It can't enforce every keyword (a `pattern` using `\d` is accepted as any string), so apps keep
checking replies themselves.

On a Mac, starting several helpers at once can make one die with SIGBUS inside Apple's Metal shader cache; restart a
helper that dies at startup.

## The client

`voice-engine-client` is what an app uses to talk to its AI helper: build an `LlmClient` with the helper paths and
call `load`, `complete`, `embed`, `inspect`, `unload`, `set_idle` and `cancel` from any thread. Match failures on
`EngineError`, never on text. Each result says which backend ran it and whether the graphics card failed during the
call (`fell_back`): show "Graphics card failed, used processor" once, then keep passing the same config.

## Local AI servers

`voice_engine_client::servers::detect()` says which local AI servers are running on this computer: Ollama (port 11434)
and LM Studio (port 1234). Each one comes with its models (`id`, `context_length`, `size_bytes`, each `None` when the
server does not say) and `api_base`, the OpenAI-compatible `/v1` root. The apps keep making the chat requests to
those servers themselves; the engine only finds them and lists the models. A server that is not running is left out,
so nothing installed is an empty list. A server that answers with something we cannot read (an error status, not
JSON, no answer in time) is reported as `ServerState::Unreadable` with no models. It is quick enough for a settings
screen (connecting gives up after 300 ms, a server gets 2 s in all, the two are asked at the same time, no retries)
and it blocks, so call it from a worker thread. `detect_at(&[Probe])` looks at other addresses. Try it with
`cargo run -p voice-engine-client --example servers`.

## Speech

`voice-engine-stt` loads one speech model and turns audio (16 kHz mono `f32`) into a `Transcript`: segments, each with
its timed words, and the language. Whisper, Parakeet (TDT v2, v3 and Ultra), Parakeet Unified and Nemotron all answer
the same way.

```rust,ignore
use voice_engine_stt::{load, ChunkOptions, LoadOptions, Timing};

let model = voice_engine_models::catalog::find("whisper-turbo").unwrap();
let dir = std::path::Path::new("/the/shared/folder/whisper-turbo");        // from SharedStore::status / download
let mut engine = load(model, dir, LoadOptions { gpu: true, threads: None })?;   // stays loaded until dropped

let opts = ChunkOptions { language: None, timing: Timing::Words, ..ChunkOptions::default() };
let transcript = engine.transcribe(&audio, &opts)?;     // times are on `audio`'s clock; add the chunk's offset
for segment in &transcript.segments {
    for word in &segment.words { /* word.text, word.start_ms, word.end_ms */ }
}
let ran = engine.backend();                             // ran.backend: "metal", "vulkan", "directml" or "cpu"
if ran.fell_back { /* show "Graphics card failed, used processor" once, then pass gpu: false */ }
```

`load` takes the catalog entry and the model's folder in the shared store. A `Box<dyn Transcriber>` is `Send`: keep it
behind a mutex or move it to a worker thread. Dropping it unloads the model. Voice detection, speaker separation,
embeddings and the local AI are not speech models: `load` answers `Unsupported`.

**Timing.** `ChunkOptions::timing` picks the decoding settings. Either way the result carries both the segments and the
words, so an app that needs sentences and an app that needs word times make the same call.

- `Timing::Segments` is Talaan's: Whisper's own segments (greedy decoding, no text carried between chunks, blank and
  non-speech output suppressed, noise and no-speech segments dropped); the words of each segment come from its token
  times. For Parakeet the segments are the sentences parakeet-rs builds from the words.
- `Timing::Words` is Ensayo's (the default): token timestamps with one word per segment, run through `words::clean`
  (noise dropped, punctuation glued to the word before, times kept in order).

Clips under 1 s are padded with silence and times are clamped to the real audio. When a decode stops with 2 s or more of
audio after its last word, the engine decodes the rest again from there (up to 8 passes) and
`Transcript::stopped_early_at_ms` says where the first stop was, so an app can see how often a model gives up early.

**Graphics card.** `LoadOptions::gpu` asks for the card: Metal on the Mac and Vulkan on Windows (the `vulkan` feature,
used by the helper) for Whisper, DirectML on Windows (the `directml` feature) for Parakeet TDT. When the card cannot be
used, or this build has no way to use it for this model, the model loads on the processor, nothing fails, and
`backend()` says `cpu` with `fell_back: true`. That is also what an ONNX model on the Mac reports for `gpu: true`: there
is no card path to fall back from, but the app asked for one, so it is told once and can pass `gpu: false` from then on.
Nothing uses CoreML (parakeet-rs measures it slower than the processor for these graphs).

**Threads.** The engine's rule, on every system: on the processor three quarters of the logical cores, at least 2 and at
most 12; on a card (Metal, Vulkan, DirectML) all the logical cores, 2 to 8, because the processor only feeds the card.
After a fallback the processor's rule applies. `LoadOptions::threads: Some(n)` overrides it for that model (at least 1,
not capped): an app can measure its own best. The ONNX models use the same rule for ONNX Runtime's threads. The rule is
in `voice_engine_stt::threads`; v0.1.0's `threads()` and `thread_count` are gone.

**Whisper options.** `language` is a hint ("fr"); `None` detects. `keep_fillers` adds the short filler prompt from
whisper.cpp's documentation. `vocabulary` is the app's custom words, sent as Whisper's initial prompt. It nudges Whisper;
it does not force a spelling, so do not promise the person it will. The engine joins the words with ", " and ends the
list with a full stop, because without it Whisper copies the prompt's bare style and drops capitals and punctuation.
Measured on one sentence with the real model: a list with one word left a near miss as it was, and a list of the
spellings Whisper is choosing between ("Mago, Mogo, Magoo") changed the decode. Listing those spellings works best.
Whisper takes audio of any length in one call (whisper.cpp cuts its own 30 s windows).

**ONNX models** (Parakeet, Parakeet Unified, Nemotron) take one chunk of up to the catalog's `chunk_secs` (40 s) per
call. Longer audio is `SpeechError::TooLong { max_secs }` and nothing is decoded: the app cuts the audio (voice
detection helps) and calls again. `keep_fillers` and `vocabulary` do nothing for them (they transcribe fillers as heard
and have no prompt); `language` is a prompt for Nemotron only ("fr" becomes "fr-FR").

**Errors** are `SpeechError`, a kind to match on, never text: `ModelMissing`, `BadModel`, `OutOfMemory`,
`Unsupported(String)`, `TooLong { max_secs }`, `Internal(String)`. No message holds audio, a word the person said or a
file name they chose.

**Voice detection.** `vad::detect(model_dir, &audio)` returns the speech spans `(start_ms, end_ms)` of a track, in order,
on the audio's clock (whisper.cpp's Silero detector with Talaan's settings, from the catalog's `vad` model, under 1 MB).
Only speech needs to go to a speech model: silence costs time and makes models invent text. No speech is an empty list,
not an error. It always runs on the processor.

**Speaker separation.** `diarize::load(model_dir, threads)` returns a `Diarizer`; `run(&track)` gives one `SpeakerFrame`
per 10 ms, with a probability from 0 to 1 for each of up to 8 speakers (NVIDIA's Nemotron 3 Diarization through
parakeet-rs). The column order carries no meaning beyond one call. Naming speakers and turning frames into turns is the
app's job. The engine warms the model up on the first 10 minutes (a streaming model must hear a voice before it can tell
it apart) and feeds long tracks in 5-minute calls. It works around a bug in parakeet-rs 0.3.8: a call whose mel frame
count is a multiple of 8 and that fits in one window (a track, or the last 5 minutes, up to about 30 s: one length in
eight) fails with "Array has a non-contiguous layout", because the library hands ONNX Runtime a copy that kept a
column-major layout. The engine gives those calls 10 ms more silence and drops the extra frame. Expect it to need more
than one short turn per voice: two synthetic voices with one 5 s turn each were not reliably told apart in testing (that
test is kept, ignored, and fails), while a 62 s conversation with three voices and twelve turns was (12 of 12 turns
right through this code, as in Talaan). It always runs on the processor.

**`SttClient` and the Windows helper.** On Windows, Whisper on the graphics card runs in a helper process (`run_helper`,
protocol 2 in `voice_engine_protocol::stt`; the Vulkan build is never linked into the app's main program). The app builds
a `SttClient` with the paths of its helpers (a Vulkan one and a processor one) and calls `load`, `transcribe`, `unload`,
`cancel` and `shutdown` from any thread; failures are `EngineError` (`BadModel`, `TooLong`, `Unsupported`, `Crashed`,
`OutOfMemory`, `Cancelled`, and the others). It works like `LlmClient`: it starts the helper, restarts it when it dies,
tries the Vulkan helper first and falls back to the processor helper with the failure remembered (`fell_back`). The client
remembers the model the app loaded and loads it into every helper it has to start again, so a crash costs the app one
`Crashed` and nothing else. The helper serves Whisper only; the ONNX models run in the app's own process through `load`.
`cancel` ends the call with `Cancelled`, but a Whisper decode cannot be interrupted, so unless the helper answers sooner,
the call ends 2 s later by killing the helper, and the next call loads the model again. Audio goes to the helper as base64
(about 2.5 MB for 30 s): an app that sends long recordings through the helper cuts them first.

### What runs where

| | Mac | Windows |
| --- | --- | --- |
| Whisper (every variant) | in process: Metal with `gpu: true`, else the processor | the Vulkan helper through `SttClient` (`vulkan` feature), the processor helper as its fallback; in process on the processor |
| Parakeet TDT (v2, v3, Ultra) | the processor | DirectML in process with the `directml` feature and `gpu: true`, else the processor |
| Parakeet Unified, Nemotron | the processor | the processor (the catalog gives no DirectML path) |
| Speaker separation (`diarize`) | the processor | the processor |
| Voice detection (`vad::detect`) | the processor | the processor |

Nothing uses CoreML. On the Mac, `gpu: true` for an ONNX model runs on the processor and reports `fell_back: true`.

### What was verified, and where

- Mac (an M-series Mac, macOS): the tests without a model, and the real-model tests (`cargo test -p voice-engine-stt --
  --ignored --test-threads=1`, models read from `~/Library/Application Support/Talaan/models`) for Whisper Turbo (both
  timing modes, any length, vocabulary, threads, language, early stop, Metal and processor), Parakeet Ultra, voice
  detection and speaker separation (all pass except the two-voices test described above), and `SttClient` with the real
  Whisper helper (`crates/client/tests/real_stt.rs`).
- No real model has been run for Parakeet Unified and Nemotron (none was on the test Mac; their code is tested with a
  scripted model and against the catalog's file lists) or for the other Parakeet variants.
- Windows: compiled, and the tests without a model run, by CI (`windows-latest`) only. The Vulkan helper, DirectML and a
  real model have not run on a Windows computer; the manual `windows-gpu` CI job builds the Vulkan versions.
- `crates/stt/tests/no_llama.rs` fails if `voice-engine-stt` can reach `llama-cpp-2`, `llama-cpp-sys-2` or
  `voice-engine-llm` through its dependencies: llama.cpp and whisper.cpp each carry a copy of ggml, so the speech engine
  stays usable in an app's own process only while it links no llama.cpp.

## Catalog

`voice-engine-models` has one catalog for every kind of model. Each entry has a `Kind`: `Whisper`, `Parakeet`,
`ParakeetUnified`, `Nemotron` (the four speech kinds, `Kind::is_speech`), `Vad` (Silero voice detection),
`Diarization` (speaker separation), `Embedding` (search by meaning) and `Llm` (local AI).

Every entry carries: `id`, `name`, `maker`, `note`, `kind`, `languages`, `english_only`, `speed` and `accuracy` (1 to
3), `licence` (SPDX name and link), `files` (each with a url pinned to a commit, size and sha256), `gpu` (which of Mac
Metal, Windows Vulkan and Windows DirectML run it), `keeps_fillers` (None until measured), `chunk_secs` (the longest
chunk to send; None means any length) and `llm` (local AI only: `context_max`, `context_8gb`, `context_16gb`,
`thinks`). Look models up with `catalog::find`, `catalog::all()`, `catalog::speech()` and `catalog::local_ai()`.

The catalog never recommends a model. Which one an app offers first is the app's business.

To check the catalog against Hugging Face (every url, size and sha256), run:

```
python3 tools/verify-catalog.py
```

## The shared model folder

All three apps keep their models in one folder, so a model is downloaded once and every app uses it.

- Mac: `~/Library/Application Support/Craft Concepts Digital/Models`
- Windows: `%LOCALAPPDATA%\Craft Concepts Digital\Models`

`SharedStore::default_dir()` gives the path (it only builds it). Set `VOICE_ENGINE_MODELS_DIR` to use another folder:
`open` then behaves exactly like `open_at`.

```
<folder>/<model id>/<files>                one folder per model, files named as in the catalog
<folder>/<model id>/<file>.part            a download in progress (kept when stopped, so it resumes)
<folder>/<model id>/.cancelled             the person stopped the download
<folder>/.engine/users/<model id>/<app>    this app uses this model (empty file; talaan, ensayo or parlato)
<folder>/.engine/locks/<model id>.use      held while an app has the model loaded
<folder>/.engine/locks/<model id>.download held while one app downloads the model
<folder>/.engine/moved/<app>               this app's one-time move is done
```

Only the engine writes inside `.engine`. The locks are operating system file locks, so they work between apps and
vanish if an app crashes.

### What an app calls

At start, in this order:

```rust
let store = SharedStore::open(App::Talaan)?;      // creates the folder and .engine if they are not there
let report = store.move_in(&["whisper-turbo"])?;  // first thing, once; `used` = model ids the settings point to
store.collect()?;                                 // erases models nobody uses and old part files
let everything = store.list();                    // every catalog model with its Status
```

- `status(id)` and `list()` say where a model is: `Installed { dir, users, used_by_me }`, `Downloading { fraction,
  bytes, by_me }` (another app may be the one downloading), `Partial { fraction, bytes, cancelled }` or `Missing`.
- `download(id, &cancel, &mut progress)` downloads into the folder and returns the model's directory. `progress` gets a
  `Progress { fraction, bytes, total }` over the whole model. If another app is already downloading it you get
  `StoreError::DownloadingElsewhere`; show that app's progress from `status`.
- `use_model(id)` says "this app uses this model" (a model that is already on the computer: no download, just the
  marker). `download` does it too.
- `hold(id)` returns a `ModelHandle` (`handle.dir()` is the folder). Keep the handle for as long as the model is
  loaded and drop it when you unload. Another app's Delete cannot erase a model while you hold it.
- Delete in an app's settings is `release(id)`: it removes this app's marker, and only the last user to let go
  erases the files (`Released::Erased`; `Kept { users }` when other apps still use it, with an empty list when
  another app has it loaded right now and `collect` will erase it later; `NotUsing`). Delete all data is
  `release_all()`. Never remove a model folder yourself.
- Test and dev builds use `SharedStore::open_at(app, dir)` with a folder of their own. It never moves anything in
  and never looks at another app's folders (`move_in` returns `already_done`). Tests must never use the real folder.
- `SharedStore::read_only(app, dir)` looks without creating anything (not even the folder): for "what is on this
  computer" screens and `cargo run -p voice-engine-models --example where`. Anything that would write is refused.

Download errors are `DownloadError` (inside `StoreError::Download`): `Offline`, `Refused { status }` (an error code or
a web page instead of the file), `DiskFull`, `Checksum { file }` (wrong size or sha256; the bad file is deleted),
`Stalled` (30 seconds without data), `Cancelled` and `Io(message)`. Match on the kind, never on text. `is_failure()`
is false for `Cancelled` and `Stalled`: nothing went wrong, the partial file is kept and the next download resumes it.

### The one-time move

`move_in` brings what each app kept in its old folder into the shared folder, once per app (it writes
`.engine/moved/<app>` last). In plain words:

- Your own old models are moved (renamed, so no copy and no extra disk space) into `<model id>/`, and a hard link is
  left at the old path, so an older version of the app still finds them. If the model is already in the shared folder
  the old copy is deleted and replaced by a link.
- Models that another app has in its old folder and the shared folder does not are hard-linked in, never moved or
  changed, and that app is recorded as a user. A link that cannot be made (a different disk) is skipped.
- A model counts only when every file is there with the catalog's exact size. Parlato's Parakeet-family files were
  never checked by Parlato, so each of them is hashed and must match the catalog's sha256, or the model is left where
  it is (reported as "hash mismatch").
- A file that is open or locked is left alone (reported as "in use") and the move is not marked done, so the next
  start finishes it. Stopping the move half way is safe: run it again.
- Parlato's old names are translated: for example `ggml-large-v3-turbo-q5_0.bin` becomes `whisper-turbo`, and its
  Granite file `granite-4.2-3b-q4.gguf` is renamed to the catalog's name on the way in.
- Parlato's four local AI models that the catalog dropped, and any GGUF the person added, are not moved, copied or
  deleted. They are reported in `MoveReport::imported` with an id (`imports::import_id`: `import-` and twelve hex
  digits that stay the same if the file is renamed), for the app to offer as imported models.
- `used` is the list of model ids the app's settings point to. Those are recorded as used by this app, so another app's
  Delete cannot remove them.

## Use it in an app

```toml
voice-engine-protocol = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
```

Every engine crate in one app uses the same tag. Each helper program stays in the app as a 3-line `main.rs`:

```rust
fn main() {
    voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "TALAAN" });
}
```

## Try an engine change in an app before tagging

Put this in the app's `src-tauri/.cargo/config.toml` (git-ignored, never committed):

```toml
[patch."https://github.com/mmnzns/voice-engine"]
voice-engine-protocol = { path = "C:/Dev/voice-engine/crates/protocol" }
voice-engine-llm      = { path = "C:/Dev/voice-engine/crates/llm" }
voice-engine-stt      = { path = "C:/Dev/voice-engine/crates/stt" }
voice-engine-models   = { path = "C:/Dev/voice-engine/crates/models" }
```

## Which app uses which tag

| App | Engine tag |
| --- | --- |
| Talaan | v0.1.0 (moving) |
| Ensayo | v0.1.0 (moving) |
| Parlato | not yet |

## Licence

MIT. Code only flows from this engine into the apps. Nothing from Parlato's GPL-3.0 source comes back in.
