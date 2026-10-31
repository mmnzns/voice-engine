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
long passes without a request. Name the binary as your app wants (Ensayo: the Tauri sidecar `ensayo-llm`); the reusable
workflow described in `docs/building-helpers.md` builds it for each target triple under that name. The helper never logs user text: only sizes, timings,
paths and error kinds reach stderr.

Reply schemas: llama.cpp enforces structure (objects, required keys, `additionalProperties: false`, `enum`, `const`,
array sizes). It can't enforce every keyword (a `pattern` using `\d` is accepted as any string), so apps keep
checking replies themselves.

**Models without a chat template.** A GGUF file with no `tokenizer.chat_template` (an imported base model, a fine-tune
that lost it) no longer fails with `no_chat_template`: `complete` writes a plain prompt instead (the system text, a blank
line, the user text, a blank line; just the user text and a blank line when the system text is blank), tokenizes it as
ordinary text with the model's start-of-text token, and the model continues from there. The reply says so with
`plain_format: true` (`Reply::plain_format`, left out of the message when false), so an app can warn that answers may be
worse. `inspect` tells the app before any load: `Inspection::has_chat_template` is false (and `thinks` too; a model
with no template never thinks). `ErrorKind::NoChatTemplate` stays in the protocol for a v0.2 helper but the helper
no longer sends it.

`gpu` in `load`, `complete` and `embed` is honoured on every system: `gpu: true` (the default) uses the card (Metal on the
Mac, the chosen Vulkan device on Windows, as many layers as fit), `gpu: false` puts nothing on any card and runs on the
processor. The place is part of what a loaded model is kept for: a request for the other place loads the model again
(`loaded.seconds` and `done.load_seconds` are above 0). `loaded`, `done` and `embedded` say where the request ran
(`backend`: `cpu`, `metal` or `vulkan`); `LlmClient` returns it in `Outcome::backend`.

On a Mac, starting several helpers at once can make one die with SIGBUS inside Apple's Metal shader cache; restart a
helper that dies at startup.

## The client

`voice-engine-client` is what an app uses to talk to its AI helper: build an `LlmClient` with the helper paths and
call `load`, `complete`, `embed`, `inspect`, `devices`, `unload`, `set_idle` and `cancel` from any thread. Match failures on
`EngineError`, never on text. Each result (`Outcome`) says which backend ran it and, when the app asked for the card
and the processor did the work, why (`fallback`, below): show "Graphics card failed, used processor" once, when it is
`CardFailed` or `CardOutOfMemory`, then keep passing the same config.

`Outcome::fallback` is `Option<Fallback>`. It is `None` when the request ran on the card, and when it did not ask for the
card (`gpu: false`). When the app asked for the card (`gpu: true`) and the processor ran, it is one of:

| `Fallback` | Why | What the app does |
| --- | --- | --- |
| `CardFailed` | the card helper could not start, died, or said the card failed, during this call (or the card helper that started ran on the processor) | say so once |
| `CardOutOfMemory` | the card helper died of lack of memory during this call | say so once (the model may not fit the card) |
| `RememberedFailure` | the card was not tried: it failed earlier (this run, or the `gpu_failures` file) and `retry_gpu` has not cleared it | nothing; the person was told |
| `NoCardHelper` | `gpu_helper` is not set and the one helper is a processor build | nothing; this build has no card helper |
| `NoCardPath` | the one helper is a card helper (the Mac's Metal helper) and still answered with the processor | nothing; the model has no card runtime |

`Outcome::fell_back()` is `fallback.is_some()`. The client works the reason out; a speech helper also says its own
(`SpeechBackend::fallback`, below), which the client uses when the card helper ran the model, and a helper from v0.2.x
(it sends the bool `fell_back` only, read as `CardFailed`) works as before.

When `gpu_helper` is set and the card helper fails (it dies, says the card failed, or failed in an earlier run), the
processor helper gets the same request with `gpu: false` (`load`, `complete`, `embed` in `LlmClient`; the `load` in
`SttClient`, including the one it repeats for a new helper), so on a Mac, where both helpers can be the same binary, it
really runs on the processor. With no `gpu_helper` the one helper gets `gpu` as the app wrote it (on the Mac that is the
Metal helper). A Metal helper that dies counts as a card failure like a Vulkan one.

A card failure is remembered in the `gpu_failures` file (when the config names one) under the app's `gpu_key` (a build
string such as `"1.2.0 | driver 560.94"`), and in the client itself until the app restarts. The app's "Try again" calls
`retry_gpu()` on each live `LlmClient` and `SttClient`: it waits for a request in progress, clears the file and the
client's memory, and stops the processor helper if that is the one running, so the next request starts on the card (the
`SttClient` loads its remembered model into the new helper); a running card helper is left alone.
`clear_gpu_failure(path)` only removes the file, which is enough for a client built afterwards but not for one already
running. The engine compares `gpu_key` only, so an app should fold the graphics driver version into it: a driver update
then tries the card again. `voice_engine_stt::driver_version(&card)` gives it for the card `discrete_card()` returned.

**Which model is loaded.** `ready`, the answer to `ping`, carries `loaded_model`: the `model_path` of the model the helper
holds, left out when none is loaded (nothing yet, `unload`, or a helper started again). `LlmClient::loaded_model()` gives
`Result<Option<String>, EngineError>`: with no helper running it is `Ok(None)` and starts none, which is what a cancel that
killed the helper, a crash, a stall, `shutdown` and an idle exit all leave; otherwise it pings the running helper, with
the same 10 s limit as `devices()` (`TimedOut`, and the helper is stopped). It waits for a request in progress to end.
An app asks before a request, to show "loading" only when the model really has to load: compare the answer with the
path it is about to use. A helper from before v0.3.0 sends no `loaded_model`, which reads as `None`.

**A limit on one reply.** `Limits` are set once and cover every call, the model load included. For a reply that must
be quick (a cleanup of dictated text) while the model may take a minute to load, call
`complete_within(limit: Duration, completion, progress)` instead of `complete`. `limit` runs from the moment the model is
loaded, or from the send when it already is (the helper says it is loading with a `loading` progress, and sends a
`reading` progress with `done: 0` as soon as the model and the prompt are ready). The load keeps the config's `start`,
`stall` and `total` limits. When the reply is not done in time the call ends with `EngineError::TimedOut` and the helper is
stopped, as for a stall: the model is gone, and the next call loads it again. `cancel` still works.

**Crashes and memory.** A helper that dies while working on a request ends it with `EngineError::OutOfMemory` when it
ran out of memory, and with `EngineError::Crashed { exit_code, signal, detail }` otherwise. It counts as out of memory
when its stderr says so (`out of memory`, `failed to allocate`, `memory allocation of`, `std::bad_alloc`,
`cannot allocate`, `unable to allocate`, `insufficient memory`, `ErrorOutOfDeviceMemory`, in any case), when it exits
with Windows' `STATUS_NO_MEMORY` (0xC0000017), or when a Unix signal 9 (SIGKILL, the system's out-of-memory killer)
ends it. `exit_code` is what it exited with (none after a signal), `signal` is the Unix signal, `detail` is the last 5
lines of its stderr, each cut to 200 characters, with control characters turned into spaces. `detail` holds no user
text: neither helper writes a prompt, a reply or a transcript to stderr (llama.cpp's and whisper.cpp's logs are
switched off; what is left is the helper's own `GPU=fail` note and the abort and assert lines of ggml, which name a file,
a line and a check). It is meant for the app's log and bug report; `Display` prints it on one line. A helper that did
not exit within a second of closing its output is stopped by the client and reports no code and no signal.

### Moving from v0.2.x to v0.3.0

Nothing to change for the additions: `complete_within`, `loaded_model()` and `Reply::plain_format` (a new field on a
struct an app only reads; an app that builds a `Reply` itself, in a test, adds `plain_format: false`; one that matches
`Event::Ready { backend, protocol }` adds `..`).

- `EngineError::Crashed` is a struct variant now: match `EngineError::Crashed { .. }`, or read `exit_code`, `signal` and
  `detail` for the log.
- `Outcome::fell_back` (a field) is `Outcome::fallback: Option<Fallback>` now, with `fell_back()` as a method. It is `Some`
  whenever the app asked for the card and the processor ran, not only on the call where the card failed: to say "graphics
  card failed" once, match `Some(Fallback::CardFailed | Fallback::CardOutOfMemory)`; `RememberedFailure`, `NoCardHelper`
  and `NoCardPath` need no message. `Fallback` is exported from `voice_engine_client` and `voice_engine_protocol`.
- The same for speech: `SpeechBackend::fell_back` (a field, what `Transcriber::backend()` returns in process and what the
  speech helper's `loaded` and `transcribed` carry) is `SpeechBackend::fallback: Option<Fallback>`, with `fell_back()` as a
  method. In process it is `NoCardPath` (the model has no card runtime here: the ONNX models on the Mac, a build without
  `directml`, Parakeet Unified and Nemotron, or no usable card), `CardFailed` (the card did not start or load the model)
  or `CardOutOfMemory`, and only when `gpu: true` was asked. On the wire a v0.3.0 helper writes both `fallback` and
  `fell_back` (the derived bool), so a v0.2.x client still reads it; a v0.2.x helper's `fell_back: true` reads as
  `CardFailed`. `Fallback` is exported from `voice_engine_stt` too.

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
if let Some(Fallback::CardFailed | Fallback::CardOutOfMemory) = ran.fallback { /* show "Graphics card failed, used processor" once, then pass gpu: false */ }
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

`Segment::text` is trimmed. `Segment::starts_word` says whether a space goes before it when segments are joined. Whisper:
the raw segment text began with whitespace (it does not in Chinese, Japanese or Korean, where words are not spaced).
Parakeet, Parakeet Unified and Nemotron: always `true`. The first segment's value does not matter. A helper from before
the field reads as `true`. `join_segments(&segments)` joins by it: a single space before each segment that starts a
word, none before one that does not.

Clips under 1 s are padded with silence and times are clamped to the real audio. When a decode stops with 2 s or more of
audio after its last word, the engine decodes the rest again from there (up to 8 passes) and
`Transcript::stopped_early_at_ms` says where the first stop was, so an app can see how often a model gives up early.

**Graphics card.** `LoadOptions::gpu` asks for the card: Metal on the Mac and Vulkan on Windows (the `vulkan` feature,
used by the helper) for Whisper, DirectML on Windows (the `directml` feature) for Parakeet TDT. Both use the PC's discrete
card only, the one with the most memory, never the integrated graphics (DirectML is given that card's adapter number, not
Windows' adapter 0, and `backend().device` names it). With only integrated graphics DirectML is not tried and the model
runs on the processor. When the card cannot be used, or this build has no way to use it for this model, the model loads on the processor, nothing fails, and
`backend()` says `cpu` with `fallback` set (`CardFailed`, `CardOutOfMemory` when the card ran out of memory, or
`NoCardPath` when there is no card path for this model). `NoCardPath` is also what an ONNX model on the Mac reports for
`gpu: true`: there is no card path to fall back from, but the app asked for one, so it is told and can pass `gpu: false`
from then on.
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
not an error. It always runs on the processor. `vad::detect_with(model_dir, &audio, &VadSettings { .. })` takes the app's
own settings: `threshold` (0.5), `min_speech_ms` (250), `min_silence_ms` (300) and `padding_ms` (0). `VadSettings::default()`
is those numbers, which is what `detect` uses. Padding is added after detection: each span grows on both sides, stops at
the ends of the audio, and spans that overlap or touch after growing are merged. A dictation app can use a smaller
minimum silence with a little padding (Parlato used 100 ms and 30 ms) to get more, tighter spans.

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
a `SttClient` with the paths of its helpers (a Vulkan one and a processor one) and calls `load`, `transcribe`, `devices`,
`unload`, `cancel` and `shutdown` from any thread; failures are `EngineError` (`BadModel`, `TooLong`, `Unsupported`, `Crashed`,
`OutOfMemory`, `Cancelled`, and the others). It works like `LlmClient`: it starts the helper, restarts it when it dies,
tries the Vulkan helper first and falls back to the processor helper with the failure remembered (`Outcome::fallback`, as
for `LlmClient`; a `Transcribe` says why the model runs on the processor when the loaded model asked for the card). The client
remembers the model the app loaded and loads it into every helper it has to start again, so a crash costs the app one
`Crashed` and nothing else. The helper serves Whisper only; the ONNX models run in the app's own process through `load`.
`cancel` ends the call with `Cancelled`, but a Whisper decode cannot be interrupted, so unless the helper answers sooner,
the call ends 2 s later by killing the helper, and the next call loads the model again. Audio goes to the helper as base64
(about 2.5 MB for 30 s): an app that sends long recordings through the helper cuts them first.

### Choosing the graphics card default on Windows

On Windows the card is on by default only when the PC has a discrete card; with only built-in graphics it stays off until
the app has measured it. The person can always switch it. An app finds out before the first model load.

Parakeet runs in the app's own process, so the app enables the feature in its own `Cargo.toml`
(`voice-engine-stt = { ..., features = ["directml"] }`) and asks in process, with no helper:

```rust
let card = voice_engine_stt::discrete_card();   // Option<Device>: "NVIDIA GeForce RTX 5070 Ti", or None
let gpu_by_default = card.is_some();
```

It lists the adapters through DXGI and DXCore only (no model is loaded, no helper started), so it is cheap at startup. The
card is the discrete adapter with the most memory, never an integrated, software or Microsoft one (no driver, a VM) or
one with no dedicated memory: Miguel's RTX 5070 Ti with its Radeon iGPU answers yes, a Ryzen with only the Radeon
answers no. It is also the card DirectML runs Parakeet on (not Windows' adapter 0, which can be the iGPU), and
`backend().device` names it. Without the `directml` feature, or off Windows,
`discrete_card()` cannot see adapters and returns None: an app built that way asks a card helper instead.

The card's driver version is `voice_engine_stt::driver_version(&card)` (`Option<String>`, "32.0.15.6614", None when
Windows doesn't say). Put it in the app's `gpu_key` with the card's name, so a driver update tries a card that failed
before again. The Vulkan helper runs on the same card and driver, so one version covers both.

For Whisper and the AI, the card helpers answer the same question: `SttClient::discrete_card()` and
`LlmClient::discrete_card()` (`Result<Option<Device>, EngineError>`) are
`voice_engine_protocol::gpu::discrete_card` over the helper's ggml devices (Vulkan), and `devices()` gives the whole
`DeviceList` for a settings screen or a check. The call starts the helper if it is not running and loads no model. If the
card helper is missing or failed earlier, the client asks the processor helper, which sees the processor only: no.
The call gives up after 10 s (`EngineError::TimedOut`, and the helper is stopped), whatever the general limits are, so an
app can ask at startup without hanging. An app should treat an `Err` from `discrete_card()` as "no card": the card stays
off by default. A timeout does not mark the card as failed.

`DirectML.dll` comes from Windows (`C:\Windows\System32`, Windows 10 1903 and later): nothing ships with the app and the
engine adds no file to the helpers' output. It is imported when the program starts, not when Parakeet loads: `ort-sys`
adds the DirectML link line to every Windows build and `voice-engine-stt` always depends on `ort`, so the helpers, and
an app that links the speech crate, may import `DirectML.dll` even without the `directml` feature. If an export ONNX
Runtime needs is missing (for example `DMLCreateDevice1` on a very old Windows 10), the program does not start and there
is no fall back to the processor. Check what a built file imports with `dumpbin /dependents <file>.exe`.

### What runs where

| | Mac | Windows |
| --- | --- | --- |
| Whisper (every variant) | in process: Metal with `gpu: true`, else the processor | the Vulkan helper through `SttClient` (`vulkan` feature), the processor helper as its fallback; in process on the processor |
| Parakeet TDT (v2, v3, Ultra) | the processor | DirectML in process with the `directml` feature and `gpu: true`, else the processor |
| Parakeet Unified, Nemotron | the processor | the processor (the catalog gives no DirectML path) |
| Speaker separation (`diarize`) | the processor | the processor |
| Voice detection (`vad::detect`) | the processor | the processor |

Nothing uses CoreML. On the Mac, `gpu: true` for an ONNX model runs on the processor and reports `fallback: Some(NoCardPath)`.

### What was verified, and where

- Mac (an M-series Mac, macOS): the tests without a model, and the real-model tests (`cargo test -p voice-engine-stt --
  --ignored --test-threads=1`, each model read from the shared folder or `~/Library/Application Support/Talaan/models`;
  on a Windows PC see "Benchmarks" for the clip and the folders) for Whisper Turbo (both
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
Metal, Windows Vulkan and Windows DirectML run it), `keeps_fillers` (whether it keeps "um", "uh" and "like"; measured, so `None` for a model that was not: only Whisper Large Turbo and Parakeet Ultra have a value so far, `Some(true)`, see `docs/bench-2026-10-29.md`), `chunk_secs` (the longest
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
  `release_all()`. Never remove a model folder yourself, with one exception: the old copy `move_in` left (see "The
  one-time move"). `release` does not touch it, and its bytes stay on disk, so an app removes its own old file when the
  person deletes a model.
- Test and dev builds use `SharedStore::open_at(app, dir)` with a folder of their own. It never moves anything in
  and never looks at another app's folders (`move_in` returns `already_done`). Tests must never use the real folder.
  Because `open_at` never runs the move, `collect()` in that folder erases a model a test put there by hand without a
  user marker. A test that copies a model in marks it with `use_model(id)`.
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
  That link is a second name for the same bytes, not a copy. After `release` erases the shared folder, the bytes stay
  on disk as long as the link exists, so the app deletes its own old path when the person deletes the model.
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
voice-engine-protocol = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.3.1" }
```

Every engine crate in one app uses the same tag. Each helper program stays in the app as a 3-line `main.rs`:

```rust
fn main() {
    voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "TALAAN" });
}
```

The `env_prefix` names the test-only settings the helper reads, for example `TALAAN_GPU` and `TALAAN_LLM_IDLE_SECS`. The
helpers that `build-helpers.yml` and the build scripts make are the engine's own binaries, built from the engine's
checkout, so they read `VOICE_ENGINE_GPU` and `VOICE_ENGINE_LLM_IDLE_SECS`. An app's own dev helper, built from the
3-line `main.rs`, reads `<PREFIX>_GPU` and the rest. Both are for tests only.

## Build settings an app must copy

llama.cpp and whisper.cpp are built from source by CMake while an app compiles. Left alone they are compiled for the
computer that builds them (the newest instructions of the build machine's CPU, and the build machine's macOS as the
minimum), so a release made on a new CI runner or a new Mac can crash with "illegal instruction" or not start on an
older one. The engine's own `.cargo/config.toml` fixes that:

```toml
[env]
GGML_NATIVE = { value = "OFF", force = true }            # no "optimize for this CPU"; read by whisper-rs-sys and llama-cpp-sys-2
GGML_AVX = { value = "ON", force = true }                # the x86 baseline: AVX2 + FMA + F16C + BMI2 (2013 and newer)
GGML_AVX2 = { value = "ON", force = true }
GGML_FMA = { value = "ON", force = true }
GGML_F16C = { value = "ON", force = true }
GGML_BMI2 = { value = "ON", force = true }
MACOSX_DEPLOYMENT_TARGET = { value = "11.0", force = true }   # the first macOS for Apple silicon; read by cc, cmake and rustc
CMAKE_C_FLAGS_RELEASE = { value = "-O2 -DNDEBUG", force = true }     # optimization: cmake-rs strips /O under the Visual Studio generator
CMAKE_CXX_FLAGS_RELEASE = { value = "-O2 -DNDEBUG", force = true }   # (Windows x64 builds at /Od, 10x slower); accepted by cl, clang-cl and clang

[target.x86_64-pc-windows-msvc]
rustflags = ["-C", "target-feature=+avx2,+fma,+f16c,+bmi2"]   # llama-cpp-sys-2 turns these into GGML_AVX2, GGML_FMA, ...
```

**Cargo reads `.cargo/config.toml` only for the workspace it builds.** An app that depends on the engine as a git
dependency does not get these lines, and nothing warns it: it builds the libraries for its own build machine. So an app
that links `voice-engine-llm` or `voice-engine-stt` (the helper programs, or the speech crate in process) copies the
block above into a committed `.cargo/config.toml` in the folder cargo runs from, usually the repo root. Cargo reads
`.cargo/config.toml` from the working directory and its parents, not from the manifest's folder: with `cargo test
--manifest-path src-tauri/Cargo.toml` run from the root, a `src-tauri/.cargo/config.toml` is never read and the baseline
silently does not apply. Commit the file; the local patch in the next section goes elsewhere. Or it sets the same
variables in its release workflow, or builds the helpers with the reusable workflow described in
`docs/building-helpers.md`, which builds in the engine's own checkout and so carries them. The rustflags line is for
Windows x64 only.

**Windows ARM64 needs a toolchain for an in-process build.** An app that links `voice-engine-stt` in its own process
also needs clang-cl and Ninja for its own build, as the helpers do (MSVC's `cl` stops ggml at configure time). See
`.github/actions/windows-arm64-toolchain` and `docs/building-helpers.md`.

**Keep `lto = false` (the default) in the app's release profile.** With `lto = true` or `"thin"`, linking on macOS fails
with `Undefined symbols: ___isPlatformVersionAtLeast`, referenced from ggml's Metal objects. We reproduced it at v0.2.2
with thin LTO. The C++ libraries never used Rust's LTO, so turning it off costs a little binary size and no speed.

**The two `CMAKE_*_FLAGS_RELEASE` lines are not optional on Windows.** `cmake-rs` strips the `/O` flags from
`CMAKE_<LANG>_FLAGS_RELEASE` under the Visual Studio generator, so without them whisper.cpp and llama.cpp are compiled at
MSVC's default of `/Od` (no optimization) in release builds: on a Ryzen 7 a Whisper decode on the processor never
finished (Parakeet runs on ONNX Runtime and is not affected). Setting the two variables stops the
stripping; `-O2 -DNDEBUG` is accepted by `cl`, `clang-cl` (Windows ARM64) and `clang` (the Mac, where CMake's own release
default is `-O3 -DNDEBUG`). They only affect release builds. After adding them, run
`cargo clean --release -p whisper-rs-sys -p llama-cpp-sys-2`. An app should fail its own release when a CMake cache of
either library has no `-O2` in `CMAKE_C_FLAGS_RELEASE` or `CMAKE_CXX_FLAGS_RELEASE`, as `scripts/check-baseline.sh`,
`scripts/build-helpers.sh` and `scripts/build-helpers.ps1` do here.

**Windows ARM64** (`aarch64-pc-windows-msvc`) is built and tested in CI on the `windows-11-arm` runner, with clang-cl and
Ninja: MSVC's own `cl` stops whisper.cpp's and llama.cpp's ggml-cpu at configure time ("MSVC is not supported for ARM").
An app that builds it needs the Visual Studio ARM64 environment, `CC` and `CXX` set to clang-cl, `CMAKE_GENERATOR=Ninja`
and `CXXFLAGS=/EHsc` (llama-cpp-sys-2's C++ wrapper needs exceptions); `.github/actions/windows-arm64-toolchain` is the
exact setup CI uses. The ARM64 CPU baseline is ggml's own: with `GGML_NATIVE` off and no architecture flag, the compiler's
default for the target applies, armv8-a with NEON, so there are no dot-product (`dotprod`) or i8mm kernels whatever the
build machine's CPU is, and there are no ARM64 lines to copy into `.cargo/config.toml`.

Two things to know. A build script does not run again when one of these values changes, so after editing them run
`cargo clean -p whisper-rs-sys -p llama-cpp-sys-2` (with `--release` for release builds). And `RUSTFLAGS` set in the
environment replaces the `rustflags` line in the file instead of adding to it; put the `target-feature` flag in it too.
`scripts/check-baseline.sh` (Mac and Linux) builds the two helpers and fails unless the CMake caches say
`GGML_NATIVE:BOOL=OFF`, (release builds) `-O2` in `CMAKE_C_FLAGS_RELEASE` and `CMAKE_CXX_FLAGS_RELEASE` and, on a Mac, the
libraries were compiled for macOS 11.0 and the binaries say so; CI runs it, and a PowerShell check does the same for
`GGML_NATIVE` and `GGML_AVX2` on Windows x64 (the `-O2` check is made by the release build of the helpers job).

## Build the helpers for an app's release

An app's release workflow builds `voice-engine-llm` and `voice-engine-stt` for Mac Apple silicon, Windows x64 and Windows
ARM64 by calling `mmnzns/voice-engine/.github/workflows/build-helpers.yml@<tag>`. It checks out the engine at that tag,
builds with the baseline above, names the files as Tauri sidecars (`<name>-<target triple>[.exe]`, with the app's names)
and uploads one artifact per target with a `SHA256SUMS` and a `build-info.txt`. `docs/building-helpers.md` has the complete
example, the Tauri `externalBin` entries, the note on closing a running helper before an update, and the signing notes
(the engine ships unsigned binaries; the app signs them). The same recipe runs locally with `scripts/build-helpers.sh` and
`scripts/build-helpers.ps1`, and this repository's CI calls the workflow on every push and starts each helper to check
its handshake.

## Try an engine change in an app before tagging

Put this in `~/.cargo/config.toml` (or pass it with `cargo --config`), never in the app's committed
`.cargo/config.toml`, so the build block and the local patch never share a file:

```toml
[patch."https://github.com/mmnzns/voice-engine"]
voice-engine-protocol = { path = "C:/Dev/voice-engine/crates/protocol" }
voice-engine-llm      = { path = "C:/Dev/voice-engine/crates/llm" }
voice-engine-stt      = { path = "C:/Dev/voice-engine/crates/stt" }
voice-engine-models   = { path = "C:/Dev/voice-engine/crates/models" }
```

## Benchmarks

Speed on real models on an M3 Pro with 18 GB (Whisper Large Turbo on Metal 19.9x real time, Whisper Base and Parakeet
Ultra on the processor for a 29 s clip, a 50-word cleanup with Qwen3.5 4B, Granite 4.2 3B and Qwen3.5 9B), the bars each
one was held against, and what was not measured: [`docs/bench-2026-10-29.md`](docs/bench-2026-10-29.md). Run them on your
own computer (they print times, sizes and counts, never text; always `--release`, the models must be on the computer):

```text
cargo run --release -p voice-engine-stt --example bench                       # `-- sweep` adds other thread counts
cargo run --release -p voice-engine-client --example bench_llm                # needs: cargo build --release -p voice-engine-llm
cargo run --release -p voice-engine-models --example download_proof -- --real # downloads whisper-base and whisper-tiny (or set VOICE_ENGINE_MODELS_DIR)
```

`bench` and `bench_timing` look for each model on its own: in `$VOICE_ENGINE_TEST_MODELS` (a folder that holds a folder per
model id) when that is set, else in the shared folder, then in Talaan's real folder for this platform (the Mac's
`~/Library/Application Support/Talaan/models`; Windows' `%LOCALAPPDATA%\com.craftconceptsdigital.talaan\models`, then
`%APPDATA%\Talaan\models`), and skip a model they do not find, saying where they looked. The clip is the 16 kHz mono WAV
file named by `$VOICE_ENGINE_TEST_CLIP` (16-bit PCM or 32-bit float; repeated and cut to 29 s), and the macOS `say` command
only when that is not set and the computer is a Mac. `bench_llm` takes catalog ids (default `qwen3.5-4b` and
`granite-4.2-3b`), reads them from `$VOICE_ENGINE_TEST_MODELS` or else the shared folder, and finds the helper at
`$VOICE_ENGINE_TEST_HELPER` or `target/release/voice-engine-llm` (`.exe` on Windows). `download_proof` refuses to run
unless `VOICE_ENGINE_MODELS_DIR` is set or `--real` is passed, and then writes into that folder (the real shared model
folder for `--real`).

### Run the speed check and the real-model tests on a Windows PC

No Mac, `say` or `afconvert` is needed. Two clips made with `say` on a Mac are in `docs/data` (16 kHz mono 16-bit PCM, each
under 500 KB): `clip-bench.wav` (12.0 s, 385 KB: three of the speed bench's sentences, so the bench repeats it to 29 s
with all the words decoded) for the speed bench, and `clip-sentence.wav` (3.2 s, 106 KB: "Hello there. This is a short test
of the speech engine.") for the real-model tests, which assert on what is heard. In PowerShell, from a clone of the engine:

```powershell
$env:VOICE_ENGINE_TEST_CLIP = "$PWD\docs\data\clip-bench.wav"           # or any 16 kHz mono WAV of your own
# Only when the models are not in the shared folder or Talaan's folders; a folder with a folder per model id:
# $env:VOICE_ENGINE_TEST_MODELS = "C:\path\to\models"
cargo run --release -p voice-engine-stt --example bench                    # Whisper Turbo and Parakeet Ultra on the processor
$env:VOICE_ENGINE_TEST_CLIP = "$PWD\docs\data\clip-sentence.wav"
cargo test --release -p voice-engine-stt -- --ignored --test-threads=1     # the real-model tests (Whisper, Parakeet, voice detection)
cargo test --release -p voice-engine-stt --features directml -- --ignored --test-threads=1   # also: gpu: true gives directml, or cpu with a fallback
```

The real tests expect the clip to say that sentence. A test that needs the `say` command's own wording or several
voices (the Whisper vocabulary test, two voice detection tests, the Parakeet short-sentences test, and everything in
`real_fillers.rs` and `real_diarize.rs`) is skipped with a message when `VOICE_ENGINE_TEST_CLIP` is set or the computer is
not a Mac. Metal is asked for on the Mac; on Windows the tests run on the processor unless built with `--features
vulkan` (Whisper) or `--features directml` (Parakeet), see "Check the graphics card on a Windows PC" below. Use
`--release`: the native libraries must be the optimized ones (see "Build settings an app must copy").

### Check the graphics card on a Windows PC

For a PC with a discrete card (the Vulkan SDK installed and `VULKAN_SDK` set to build `vulkan`; DirectML needs nothing, Windows
brings `DirectML.dll`). Every line a bench prints has `backend`, `device` (the adapter's name, `-` on the processor) and
`fallback` (why the card was not used when it was asked for): on a PC with two adapters (an RTX card and the Ryzen's Radeon)
`device` must be the discrete card. Each step is one block, run from a clone of the engine, with the clip and the models set
as above (`$env:VOICE_ENGINE_TEST_CLIP = "$PWD\docs\data\clip-bench.wav"`).

Set `VOICE_ENGINE_TEST_EXPECT_CARD=1` for the test steps (4 and 5). A card test whose model or helper variables are not
set still skips. With the variable set, a test that finds the card did not run (the work ran on the processor, with
a `fallback`, or the helper found no card) fails instead of printing SKIPPED and passing.

1. Parakeet on DirectML and Whisper on Vulkan in this process (the `vulkan` feature builds the Vulkan backend into the
   example), next to the processor rows:

```powershell
cargo run --release -p voice-engine-stt --features vulkan,directml --example bench
```

2. Whisper through the Vulkan speech helper, with `SttClient`, the card next to `gpu: false` (first it prints the helper's
   device list and the device it chose):

```powershell
cargo build --release -p voice-engine-stt --features vulkan
cargo run --release -p voice-engine-client --example bench_stt
```

3. The AI helper on the card (`$env:VOICE_ENGINE_TEST_HELPER` is the helper to measure, `target\release\voice-engine-llm.exe` by default):

```powershell
cargo build --release -p voice-engine-llm --features vulkan
cargo run --release -p voice-engine-client --example bench_llm
```

4. The real-model tests on the card (Whisper in process with `vulkan`, Parakeet with `directml`; `--ignored`, one at a time):

```powershell
$env:VOICE_ENGINE_TEST_CLIP = "$PWD\docs\data\clip-sentence.wav"
$env:VOICE_ENGINE_TEST_EXPECT_CARD = "1"
cargo test --release -p voice-engine-stt --features vulkan,directml -- --ignored --test-threads=1 --nocapture
```

5. The forced fallback against the real helpers: `VOICE_ENGINE_GPU=fail` makes the Vulkan helper abort on its first real request,
   and the client must answer the same call from the processor helper with `fallback` set (`CardFailed`) (a test whose variables are not set
   says which and passes). The Vulkan build is copied aside because the processor build overwrites it:

```powershell
cargo build --release -p voice-engine-stt --features vulkan
Copy-Item target\release\voice-engine-stt.exe $env:TEMP\stt-vulkan.exe -Force
cargo build --release -p voice-engine-llm --features vulkan
Copy-Item target\release\voice-engine-llm.exe $env:TEMP\llm-vulkan.exe -Force
cargo build --release -p voice-engine-stt -p voice-engine-llm
$env:VOICE_ENGINE_TEST_STT_GPU_HELPER = "$env:TEMP\stt-vulkan.exe"
$env:VOICE_ENGINE_TEST_STT_HELPER = "$PWD\target\release\voice-engine-stt.exe"
$env:VOICE_ENGINE_TEST_GPU_HELPER = "$env:TEMP\llm-vulkan.exe"
$env:VOICE_ENGINE_TEST_HELPER = "$PWD\target\release\voice-engine-llm.exe"
$env:VOICE_ENGINE_TEST_LLM = "C:\path\to\model.gguf"      # any chat model, a .gguf file
$env:VOICE_ENGINE_TEST_EXPECT_CARD = "1"
cargo test --release -p voice-engine-client --test real_fallback -- --ignored --nocapture --test-threads=1
```

The last build has no features, so it makes the processor helpers the client falls back to.

Logs never hold what a person said. A test, example or bench prints counts (words, characters, segments), times, sizes and
backend names, never a transcript, a word, a reply or a prompt; check logs are saved and shared, so this holds for every
`println!` and `eprintln!` in `crates/*/tests`, `examples` and `support`. An assertion's failure message may show text
only when the assertion is about that text itself (a parity test that compares two transcripts).

## Which app uses which tag

| App | Engine tag |
| --- | --- |
| Talaan | moving to v0.3.1 |
| Ensayo | moving to v0.3.1 |
| Parlato | moving to v0.3.1 |

## Licence

MIT. Code only flows from this engine into the apps. Nothing from Parlato's GPL-3.0 source comes back in.
