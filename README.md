# voice-engine

The local AI and speech engine shared by three desktop apps by Miguel Monzones:
[Talaan](https://github.com/mmnzns/Talaan), [Ensayo](https://github.com/mmnzns/ensayo) and
[Parlato](https://github.com/mmnzns/Parlato).

| Crate | What it does |
| --- | --- |
| `voice-engine-protocol` | Messages between an app and its helper programs |
| `voice-engine-llm` | The local AI helper (llama.cpp in its own process) |
| `voice-engine-stt` | Whisper chunk transcription and the Whisper helper |
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
