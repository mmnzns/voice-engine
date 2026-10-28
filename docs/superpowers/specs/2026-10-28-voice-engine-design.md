# voice-engine: one shared engine for Talaan, Ensayo and Parlato

Spec for Phase 0 and Phase 1 of the plan on the Notion page "Voice Apps Engine Sync".
Written 2026-10-28. Status: waiting for Miguel's review.

This file lives in `App - Engine/docs/superpowers/specs/` on Google Drive until the `voice-engine` repository exists.
It then becomes the first commit of that repository, at the same path.

## 1. Goal

Talaan, Ensayo and Parlato each carry their own copy of the local AI and speech engine. A fix or a new model has to be
made three times, and Parlato does not look for models the other apps already downloaded.

The goal is one shared engine, written once and built into all three apps:

- **One repository, `mmnzns/voice-engine`**, public, MIT, owned by Miguel.
- **Each app stays its own app.** It installs, updates and runs on its own. It pulls the engine in from GitHub at a
  pinned version tag and moves to a new version only on purpose, after testing.
- **This spec covers Phase 0 and Phase 1 only:** create the engine from Talaan's and Ensayo's code, and move Talaan and
  Ensayo onto it with no change users can see. Parlato joins in Phase 2, which has its own spec.

### What "done" means for this spec

1. `voice-engine` exists on GitHub with tag `v0.1.0`, its CI is green on Windows and macOS.
2. Talaan's `main` builds against `voice-engine` `v0.1.0`, its own `llm` and `llm-protocol` code and its engine part of
   `stt` and `transcriber/models.rs` are gone, all its tests pass, and a recording still gets a transcript and a summary.
3. Same for Ensayo: its own `llm`, `llm-protocol` and the engine part of `ai/local_models.rs` and `speech/models.rs` are
   gone, all its tests pass, and a practice answer still gets transcribed and scored.
4. The Notion page has: the decisions below marked done, a `v0.1.0` row in the engine change log, and a
   "What each app does next" section with one block per app.
5. Parlato has the Phase 0 build fix (section 9).

## 2. Decisions already made

| Decision | Choice | Why |
| --- | --- | --- |
| What "one engine" means | One shared code library, built into three separate apps | An installed engine shared at run time would let one app's update or uninstall break the other two |
| Repository | `mmnzns/voice-engine`, **public** | Talaan, Ensayo and Parlato are public; a private dependency would stop anyone else (and public CI) building them |
| Licence | **MIT** | Same as Talaan and Ensayo (Ensayo became MIT in commit `f294fc4`). Parlato is GPL-3.0 and may use MIT code, never the other way round |
| How apps pull it in | Cargo **git dependency pinned to a tag** | One-line upgrade, works on CI, on the Mac and on the PC with nothing extra to clone |
| How the helper programs get built | Engine code is in **libraries**; each app keeps a **3-line `main.rs`** per helper | The apps' program names, build scripts, Vulkan builds, signing and installers stay exactly as they are |
| History | **Fresh repository**, the first commits record the Talaan and Ensayo commit each file came from | `git filter-repo` is not installed and the history of these folders is still readable in Talaan and Ensayo |
| Who works on what | One Claude Code chat per repository (section 8) | Each chat stays focused; the engine has one owner |

## 3. Ownership rule (never break this)

The engine, Talaan and Ensayo are Miguel's own work and must never fall under the GPL of Parla or VoiceInk, which
Parlato comes from.

- Code flows one way only: **from the engine into Parlato**. Nothing from Parlato's source goes into the engine, Talaan or
  Ensayo.
- When the engine needs something Parlato already does, rebuild it from the original source (whisper.cpp, llama.cpp,
  parakeet-rs, the model's own documentation), not from Parlato's file.
- Reading a fact from Parlato's code (a folder name such as `Models` or `ParakeetModels`, a bundle id) is fine. Copying
  its code is not.

## 4. What goes in the repository

```
voice-engine/
  Cargo.toml                 workspace: members below, shared [workspace.package] (edition 2024, license MIT)
  LICENSE                    MIT, Miguel Monzones
  README.md                  what it is, which apps use it, how to pin it, the ownership rule
  CHANGELOG.md               one section per tag
  CLAUDE.md                  rules for chats working in this repo (section 8.4)
  docs/superpowers/specs/    this file
  crates/
    protocol/                crate voice-engine-protocol
    llm/                     crate voice-engine-llm
    stt/                     crate voice-engine-stt
    models/                  crate voice-engine-models
  .github/workflows/ci.yml
```

All four crates are libraries, version `0.1.0`, `publish = false`. None of them depends on Tauri.

### 4.1 `voice-engine-protocol`

The messages between an app and its helper programs.

- Starts from Talaan's `src-tauri/llm-protocol` (`lib.rs`, `gpu.rs`, `stt.rs`), merged with Ensayo's
  `src-tauri/llm-protocol`. The result is the union: Talaan's `Embedding`, GPU and speech messages, and Ensayo's
  `decode_request` and `decode_event` helpers and any fields Ensayo added. Where the two disagree on the same field,
  keep the shape that both apps can use, and write the choice in `CHANGELOG.md`.
- **New:** `pub const PROTOCOL_VERSION: u32 = 1;`. Each helper sends it in its first message after it starts. The app
  compares it with the version it was built with; if they differ, it reports "the AI helper does not match this version
  of the app, reinstall" in its log and in its error state, instead of failing in a confusing way later. This only
  catches a helper and an app from different builds; it never fires in a normal install.

### 4.2 `voice-engine-llm`

The local AI helper's logic: llama.cpp in its own process, talking over stdin and stdout with the protocol above.

- Starts from Talaan's `src-tauri/llm/src/main.rs`, merged with Ensayo's `src-tauri/llm/src/main.rs` (about 550 changed
  lines between the two). Ensayo's process tests in `src-tauri/llm/tests/process.rs` move here.
- The code that was `main()` becomes `pub fn run(config: HelperConfig)`. `HelperConfig { env_prefix }` (in the
  protocol crate) is the only thing that differs between apps: it names the environment variables
  (`TALAAN_GPU`, `ENSAYO_LLM_IDLE_SECS`, ...), so each app keeps its current variable names.
- Dependencies stay pinned as today: `llama-cpp-2 = "=0.1.159"`, `features = ["common"]`, plus `"metal"` on macOS.
- Feature `vulkan` turns on `llama-cpp-2/vulkan`, as in Talaan today.
- Anything app-specific in either copy (a log line naming the app, a default path) becomes a parameter of `run()` or
  stays in the app.

### 4.3 `voice-engine-stt`

Whisper chunk transcription, used inside the app on the processor and by the Windows graphics card helper.

- Starts from Talaan's `src-tauri/stt` (`lib.rs` stays the library; `main.rs` becomes `pub fn run_helper()`).
- `whisper-rs = "0.16.0"` with `features = ["raw-api"]`; feature `vulkan` turns on `whisper-rs/vulkan`.
- Ensayo does not use this crate in Phase 1 (it has no Whisper helper today). Parakeet is not moved in Phase 1; it stays
  in each app.

### 4.4 `voice-engine-models`

The model catalog and the model finder.

**Catalog.** One list of every model the engine knows: id, display name, files (name, size in bytes, SHA-256), the
pinned download URL of each file, and the folder names other apps use for the same files (`reuse_dirs`). It covers:

- speech models: Talaan's `src-tauri/src/transcriber/models.rs` catalog, checked against Ensayo's
  `src-tauri/src/speech/models.rs` (the same files are expected; any difference is reported to Miguel, not guessed);
- local AI models: Ensayo's `src-tauri/src/ai/local_models.rs` (Qwen 3.5 4B and 9B) and Talaan's equivalent in
  `src-tauri/src/summarizer/`.

The engine keeps Talaan's `ModelInfo` shape, including its short English `note`, `speed` and `accuracy` (Ensayo's
entries are the same). Each app keeps its own choices: which models it shows, which is its default or recommended
model, licence text, and translations.

Naming rule for all three apps from now on: **one folder per model id, files under their original names**
(`<models root>/<model id>/<file>`), as Talaan and Ensayo do today. Parlato's `<id>.gguf` renaming is dealt with in
Phase 2.

**Finder.** `ModelStore` and `locate()` from Talaan's `transcriber/models.rs`, made general:

- `ModelStore::new(own_root).with_legacy(..).with_others(other_roots, reuse)`; `locate(model)` returns `Own(dir)`,
  `OtherApp { app, dir }` or `Missing`, checking the app's own folder (and Talaan's legacy folder, which the app
  passes in), then each other root when `reuse` is on.
- A file counts as present when it exists with its exact size, as today. Hashes are checked at download only.
- A **known apps** table, the one place that says where each app keeps its models:

| App | Bundle id | Windows root | macOS root | Layout under the root |
| --- | --- | --- | --- | --- |
| Talaan | `com.craftconceptsdigital.talaan` | `%LOCALAPPDATA%\com.craftconceptsdigital.talaan\models` | `~/Library/Application Support/Talaan/models` | `<id>/<files>` |
| Ensayo | `com.craftconceptsdigital.ensayo` | `%LOCALAPPDATA%\com.craftconceptsdigital.ensayo\models` | `~/Library/Application Support/com.craftconceptsdigital.ensayo/models` | speech `<id>/<files>`, AI `llm/<id>/<files>` |
| Parlato | `com.craftconceptsdigital.parlato` | `%LOCALAPPDATA%\com.craftconceptsdigital.parlato` | `~/Library/Application Support/com.craftconceptsdigital.parlato` | Whisper and Silero `Models/<file>`, Parakeet `ParakeetModels/<model folder>/<files>`, AI `LlmModels/<id>.gguf` |

  Read from each app's code on 2026-10-28 (Talaan `transcriber/models.rs` and Ensayo's copy of Talaan's path;
  Ensayo `paths.rs`; Parlato `transcription/model_manager.rs`, `parakeet_model_manager.rs`,
  `enhancement/model_manager.rs`, which use Tauri's `app_local_data_dir`). Paths only, no code copied. Each row must be
  checked on a real install before `v0.1.0`. Parlato's AI files cannot be matched in Phase 1 because of the
  `<id>.gguf` renaming; that is Phase 2.
- `App::others(me, engine)` returns the other two apps' roots on this computer; `App::root(engine)` returns one.

**No change for users in Phase 1.** Each app passes the same other-app folders it checks today:

- Ensayo: Talaan's folder only.
- Talaan: Parlato's folder only, now the **correct Windows folder** (`%LOCALAPPDATA%`, not `%APPDATA%`). This fixes the
  Talaan bug on the Notion page. It is the one intended behaviour change in Phase 1 and gets its own line in Talaan's
  release notes.

Turning on sharing between all three apps is Phase 4.

## 5. How an app uses the engine

### 5.1 Cargo

In the app's `src-tauri/Cargo.toml`:

```toml
voice-engine-protocol = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
voice-engine-models   = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
voice-engine-stt      = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }   # Talaan only
```

All engine crates of one app always use the same tag.

### 5.2 Helper programs

Each app keeps its helper folders and program names. Talaan's `src-tauri/llm/src/main.rs` becomes:

```rust
fn main() {
    voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "TALAAN" });
}
```

and `src-tauri/llm/Cargo.toml` depends on `voice-engine-llm` at the same tag, with a `vulkan` feature that forwards to
`voice-engine-llm/vulkan`. The same for `talaan-stt` and `ensayo-llm`. The build scripts
(`scripts/build-llm-helper.ps1` and `.sh`), `externalBin` in `tauri.conf.json`, the Vulkan builds and the Mac signing
do not change.

The app's own `llm-protocol` folder is deleted; the app imports `voice_engine_protocol` instead of
`talaan_llm_protocol` / `ensayo_llm_protocol` (Talaan: `commands/gpu.rs`, `logging.rs`, `summarizer/engines.rs`,
`summarizer/local.rs`, `transcriber/engine.rs`, `transcriber/fallback.rs`, `transcriber/remote.rs`; Ensayo:
`ai/local.rs`).

### 5.3 Working on engine and app together

To try an engine change in an app before it is tagged, the app's chat puts a local override in
`src-tauri/.cargo/config.toml`, which is git-ignored:

```toml
[patch."https://github.com/mmnzns/voice-engine"]
voice-engine-protocol = { path = "C:/Dev/voice-engine/crates/protocol" }
voice-engine-llm      = { path = "C:/Dev/voice-engine/crates/llm" }
voice-engine-stt      = { path = "C:/Dev/voice-engine/crates/stt" }
voice-engine-models   = { path = "C:/Dev/voice-engine/crates/models" }
```

This file is never committed. Releases are always built against a tag. Each app adds `src-tauri/.cargo/config.toml` to
`.gitignore` during its move.

## 6. Versions and the change log

- Tags follow `vMAJOR.MINOR.PATCH`. Before 1.0, a change that breaks an app's build or the protocol bumps MINOR; fixes
  and new catalog entries bump PATCH. Changing `PROTOCOL_VERSION` is always a MINOR bump.
- A tag is made only from `main` with CI green, and never moved or deleted once pushed.
- Every tag gets a section in `CHANGELOG.md` and a row in the Notion **engine change log** the same day. The row says,
  per app, what to do: "nothing", "bump and test", or "needs a code change: ...". The app's chat changes its column to
  "done in vX.Y.Z" when that app ships it.

## 7. Errors and edge cases

- **Helper and app from different builds:** the protocol version check in 4.1.
- **A model found in another app's folder is used in place and read only.** The engine never writes, moves or deletes
  in another app's folder.
- **A shared model disappears** (the other app deleted it): `locate()` returns `Missing` next time; the app shows its
  normal Download state. Phase 1 apps already behave this way; the engine keeps it.
- **The engine cannot be fetched** (offline build, GitHub down): Cargo uses the copy in `~/.cargo/git` once it has been
  fetched once. CI fetches fresh. Nothing extra is needed.
- **Windows path length:** the git dependency lands under `~/.cargo/git/checkouts/`. If a long path breaks the llama.cpp
  build, set `CARGO_HOME` to a short path in CI and note it in the README.

## 8. How the chats work together

### 8.1 One chat per repository

| Chat | Folder | Works on | Never |
| --- | --- | --- | --- |
| Engine chat | `App - Engine`, code in `C:\Dev\voice-engine` | Engine code, its tests and CI, tags, `CHANGELOG.md`, the Notion change log and the "What each app does next" blocks | Edits an app repository |
| Talaan chat | Talaan's folder, code in `C:\Dev\Talaan` | Talaan; moving Talaan onto a new engine tag | Edits engine code |
| Ensayo chat | Ensayo's folder, code in `C:\Dev\Ensayo` | Ensayo; same | Edits engine code |
| Parlato chat | Parlato's folder, code in `C:\Dev\Parlato` | Parlato; Phase 0 fix; from Phase 2, moving onto the engine | Edits engine code, or sends code to the engine |

The engine chat does not edit Talaan, Ensayo or Parlato, because those repositories have work in progress on many
branches and worktrees that only their own chat knows about.

### 8.2 How a change flows

1. The engine chat makes the change, runs CI, tags the new version, writes `CHANGELOG.md`.
2. It adds a row to the Notion engine change log and, when an app has work to do, a block for that app under
   "What each app does next".
3. Miguel opens that app's chat and pastes one line, for example:
   `Read the Notion page "Voice Apps Engine Sync", section "Talaan: move to voice-engine v0.1.0", and do it.`
4. The app's chat does the work on a branch, opens a PR, runs its tests and manual check, and after release updates its
   column in the change log.
5. When an app's chat needs something from the engine, it does not change engine code. It opens an issue on
   `mmnzns/voice-engine` (or adds a line to the Notion page) and Miguel brings it to the engine chat.

### 8.3 What each app block on Notion contains

Every "What each app does next" block has the same parts, so the app's chat can act on it without this conversation:

- the engine tag to move to and the change log row it comes from;
- the branch name to create;
- the exact `Cargo.toml` lines, folders to delete and imports to change;
- the `CLAUDE.md` snippet to add (8.5), on the first move only;
- the tests to run and the manual check that proves nothing changed for users;
- what "done" looks like, and to update the change log column when shipped.

### 8.4 `voice-engine/CLAUDE.md` (rules for the engine chat)

- Read the Notion engine change log before starting.
- Never copy code from Parlato or any GPL source. Build from the original upstream source.
- Never edit an app repository. Write what an app must do in its Notion block.
- Every tag: CI green, `CHANGELOG.md` section, Notion change log row, same day.
- Keep the README's "which apps use which tag" table current.

### 8.5 `CLAUDE.md` snippet for each app (added by that app's chat on its first move)

```markdown
## Shared engine (voice-engine)

The local AI helper, the helper protocol, Whisper chunk transcription (Talaan) and the model catalog and finder come
from https://github.com/mmnzns/voice-engine, pinned by tag in src-tauri/Cargo.toml.

- Before working on AI, speech or models, read the engine change log on the Notion page "Voice Apps Engine Sync".
- Never edit engine code from this repository. If the engine needs a change, open an issue on mmnzns/voice-engine.
- Move to a new engine tag only on purpose, on a branch, after the tests and a manual check pass. Then update this
  app's column in the change log.
- A local `src-tauri/.cargo/config.toml` [patch] pointing at C:\Dev\voice-engine is for trying changes only. Never
  commit it; releases build against the tag.
```

Parlato's snippet, in Phase 2, adds: "Code only flows from the engine into Parlato. Never send Parlato code to the
engine, Talaan or Ensayo."

## 9. Phase 0: Parlato test-build fix

Small and separate. Done by the Parlato chat, from a Notion block written by the engine chat.

Add to Parlato's `src-tauri/Cargo.toml`, as Talaan has:

```toml
[profile.dev.package."*"]
opt-level = 3
```

Then build a test build and check that the crash seen on 2026-10-28 is gone. If it is not, the Parlato chat reports back
and the crash is debugged in Parlato, outside this spec.

(This is a build setting, not code, and flows from Talaan to Parlato, which the ownership rule allows.)

## 10. Phase 1 order of work

Engine chat:

1. **GPL check.** Talaan, Ensayo and Parlato all have `scripts/cpu-baseline.ps1`. Compare them and check the history
   (`git log --follow`) of every file going into the engine for anything that came from Parlato, Parla or VoiceInk.
   Report findings to Miguel before importing. If anything did, it is rewritten from upstream, not imported.
2. **Create `mmnzns/voice-engine`** (public, MIT) and clone it to `C:\Dev\voice-engine`.
3. **Import Talaan's code** into the four crates, unchanged except for crate names and the `run()` functions. Commit
   message names the Talaan commit it came from. CI green.
4. **Merge Ensayo's copy** into `protocol` and `llm`, bring in its process tests. Commit message names the Ensayo
   commit. CI green.
5. **Build the catalog and finder** (4.4), with the known apps table, `PROTOCOL_VERSION`, and tests for: own folder,
   legacy folder, other-app folder with reuse on and off, missing file, wrong size, and each row of the known apps table
   per platform.
6. **Check the known apps paths on a real install** on Windows (and on the Mac if available), then tag `v0.1.0`.
7. **Write Notion:** decisions marked done, change log row for `v0.1.0`, a Phase 0 block for Parlato, and Phase 1 blocks
   for Talaan and Ensayo (8.3).

App chats (after step 7, in either order):

8. **Talaan chat** moves Talaan onto `v0.1.0` (5.1, 5.2), deletes its engine copies, adds the `CLAUDE.md` snippet,
   passes its tests and manual check, ships, and updates the change log.
9. **Ensayo chat** does the same for Ensayo.
10. **Parlato chat** does Phase 0 (section 9).

## 11. Testing

**Engine CI** (`.github/workflows/ci.yml`), on every push and PR, on `windows-latest` and `macos-latest`:

- `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`, `cargo test --workspace`;
- the `llm` process tests (from Ensayo), which start the helper and talk to it over the protocol; if they need a model
  file, they use the smallest catalog model and are marked so CI can cache it, or they run against a fake model the
  tests generate;
- a build with `--features vulkan` on Windows only when the Vulkan SDK is available on the runner; otherwise a
  `cargo check --features vulkan`.

**Each app**, during its move:

- its full existing test suite, against the pinned tag;
- a release build with its normal build script, which proves the helper programs still build and bundle;
- one manual check on Windows, and on the Mac before a Mac release:
  - Talaan: record a short meeting, get a transcript and a local summary; on a PC with Parlato installed, Talaan finds
    Parlato's Whisper model;
  - Ensayo: answer one practice question out loud, get a transcript and a local AI score; with Talaan installed, Ensayo
    still finds Talaan's models.

## 12. Not in this spec

Each of these gets its own spec later:

- Parlato on the engine (Phase 2): local AI in the helper with the model kept loaded for dictation, installer, updater
  and Mac signing.
- New AI models in Parlato (Phase 3).
- Model sharing between all three apps, "Found on this computer" and a setting to turn it off (Phase 4).
- Graphics card helpers for all three, Metal (Phase 5).
- Ollama and LM Studio (Phase 6).
- A shared BYOK cloud provider and model list.
- Moving Parakeet into the engine.

## 13. Open questions for Miguel

None block Phase 1. Carried over from Notion for later phases:

- Default AI model in Parlato: Qwen 3.5 4B for everyone, or a smaller default? (Phase 3)
- Keep or trim Parlato's five older AI models? (Phase 2/3)
- Graphics card on by default? (Phase 5)
- Model sharing on by default? (Phase 4)

## 14. For the chat that builds this

This spec is the input to an implementation plan (`superpowers:writing-plans`). The plan should follow section 10 in
order, one task per step, each ending with CI green and a commit. Stop and report to Miguel, rather than guessing, when:

- the GPL check (step 1) finds anything;
- Talaan's and Ensayo's copies disagree on behaviour, not just shape, and both cannot be kept;
- Talaan's and Ensayo's speech catalogs list different files for the same model;
- a known apps path cannot be confirmed on a real install.
