# voice-engine Phase 0 + 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the public MIT repository `mmnzns/voice-engine` from Talaan's and Ensayo's engine code, tag `v0.1.0`, and write the Notion handoff blocks that tell the Talaan, Ensayo and Parlato chats what to do next.

**Architecture:** A Cargo workspace of four library crates (`voice-engine-protocol`, `voice-engine-llm`, `voice-engine-stt`, `voice-engine-models`). The `llm` and `stt` crates also have a small binary used only by their own tests. Apps pull the crates in by git tag and keep a 3-line `main.rs` per helper program. This chat never edits an app repository.

**Tech Stack:** Rust 2024 edition (stable toolchain), llama-cpp-2 `=0.1.159`, whisper-rs `0.16.0`, serde/serde_json, dirs, tempfile (tests), GitHub Actions on `windows-latest` and `macos-latest`, `gh` CLI, Notion MCP tools.

**Spec:** `docs/superpowers/specs/2026-10-28-voice-engine-design.md` (in `App - Engine` on Google Drive; copied into the repo in Task 2). Read it before starting. Section numbers below ("spec 4.4") point into it.

---

## Ground rules for whoever runs this plan

1. **Never edit `C:\Dev\Talaan`, `C:\Dev\Ensayo` or `C:\Dev\Parlato`.** Read them only. Do not check out branches, pull, stash or commit there; those repos have work in progress on many branches. Take source files with `git archive` from a commit (Task 1), never from their working trees.
2. **Never copy anything from Parlato's source into the engine.** Reading a folder name or a bundle id from Parlato's code is allowed; copying code, comments or scripts is not. The engine is MIT; Parlato is GPL-3.0.
3. **Stop and report to Miguel instead of guessing** when: the GPL check finds anything (Task 1); Talaan and Ensayo behave differently in a way where both can't be kept (Tasks 5, 8); their catalogs list different files for the same model id (Task 11); a known-apps path can't be confirmed on a real install (Task 13).
4. Shell: Git Bash on Windows (`/c/Dev/...` paths). Every commit message ends with:
   ```
   Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
   ```
   (If the chat running this plan is a different model, use that model's name in the line.)
5. After every task: `cargo test --workspace` green in `C:\Dev\voice-engine`, then commit. Push after Tasks 2, 6, 9, 12 and 14.

## File structure (what exists at the end)

```
C:\Dev\voice-engine\
  Cargo.toml                       workspace + [workspace.package] + [workspace.dependencies]
  LICENSE                          MIT, Miguel Monzones
  README.md                        what, who uses which tag, how to pin, local override, ownership rule
  CHANGELOG.md
  CLAUDE.md                        rules for chats in this repo (spec 8.4)
  SOURCES.md                       which Talaan/Ensayo commit each crate came from
  .gitignore                       /target, .cargo/config.toml
  .github/workflows/ci.yml
  .github/actions/vulkan-sdk/      copied from Talaan (only after Task 1 clears it)
  docs/superpowers/specs/2026-10-28-voice-engine-design.md
  docs/superpowers/plans/2026-10-28-voice-engine-phase-1.md
  crates/protocol/   Cargo.toml, src/lib.rs, src/gpu.rs, src/stt.rs, src/config.rs
  crates/llm/        Cargo.toml, src/lib.rs (helper logic, pub fn run), src/main.rs (test binary), tests/process.rs
  crates/stt/        Cargo.toml, src/lib.rs (Whisper chunk code), src/helper.rs (pub fn run_helper), src/main.rs (test binary)
  crates/models/     Cargo.toml, src/lib.rs, src/catalog.rs, src/llm_catalog.rs, src/apps.rs, src/store.rs
```

One responsibility per file: `catalog.rs` = what speech models are; `llm_catalog.rs` = what local AI models are; `apps.rs` = where each app keeps models; `store.rs` = finding a model on disk; `config.rs` = per-app helper settings (env var names).

---

### Task 0: Tools and source commits

**Files:** none (scratch notes go in `C:\Dev\voice-engine-import\`, outside any repo)

- [ ] **Step 1: Check tools**

Run:
```bash
cargo --version && rustc --version && git --version && gh auth status && gh repo view mmnzns/voice-engine 2>&1 | head -2
```
Expected: cargo/rustc stable versions print; `gh auth status` shows logged in as `mmnzns`; the last command prints "Could not resolve to a Repository" (the repo does not exist yet). If it already exists, stop and ask Miguel.

- [ ] **Step 2: Fetch the app repos without touching their working trees**

```bash
git -C /c/Dev/Talaan fetch origin main
git -C /c/Dev/Ensayo fetch origin main
TALAAN_SHA=$(git -C /c/Dev/Talaan rev-parse origin/main)
ENSAYO_SHA=$(git -C /c/Dev/Ensayo rev-parse origin/main)
echo "TALAAN_SHA=$TALAAN_SHA" ; echo "ENSAYO_SHA=$ENSAYO_SHA"
```
Expected: two 40-character SHAs. `git fetch` only updates remote-tracking refs; it does not change any branch or file. Write both SHAs down; every later task uses them.

- [ ] **Step 3: Export the source folders to a scratch folder**

```bash
mkdir -p /c/Dev/voice-engine-import/talaan /c/Dev/voice-engine-import/ensayo
git -C /c/Dev/Talaan archive "$TALAAN_SHA" src-tauri/llm src-tauri/llm-protocol src-tauri/stt src-tauri/src/transcriber/models.rs src-tauri/src/summarizer/models.rs .github/actions/vulkan-sdk scripts/cpu-baseline.ps1 | tar -x -C /c/Dev/voice-engine-import/talaan
git -C /c/Dev/Ensayo archive "$ENSAYO_SHA" src-tauri/llm src-tauri/llm-protocol src-tauri/src/speech/models.rs src-tauri/src/ai/local_models.rs scripts/cpu-baseline.ps1 | tar -x -C /c/Dev/voice-engine-import/ensayo
find /c/Dev/voice-engine-import -type f | grep -v /target/ | sort
```
Expected: the listed files appear under `talaan/` and `ensayo/`. If `.github/actions/vulkan-sdk` or `scripts/cpu-baseline.ps1` is missing from a repo, drop that path from the command and note it.

---

### Task 1: GPL check (stop gate)

**Files:** Create `C:\Dev\voice-engine-import\gpl-check.md` (a report, not committed anywhere)

- [ ] **Step 1: Compare the three `cpu-baseline.ps1` files**

```bash
git -C /c/Dev/Parlato show origin/HEAD:scripts/cpu-baseline.ps1 > /c/Dev/voice-engine-import/parlato-cpu-baseline.ps1 2>/dev/null || git -C /c/Dev/Parlato show HEAD:scripts/cpu-baseline.ps1 > /c/Dev/voice-engine-import/parlato-cpu-baseline.ps1
cd /c/Dev/voice-engine-import
diff talaan/scripts/cpu-baseline.ps1 parlato-cpu-baseline.ps1 | head -40; echo "talaan-vs-parlato changed lines: $(diff talaan/scripts/cpu-baseline.ps1 parlato-cpu-baseline.ps1 | grep -c '^[<>]')"
diff ensayo/scripts/cpu-baseline.ps1 talaan/scripts/cpu-baseline.ps1 | grep -c '^[<>]'
```
Record the counts and whether the files are near-identical.

- [ ] **Step 2: Find where each file was first added**

```bash
git -C /c/Dev/Talaan log --follow --diff-filter=A --format='%h %ad %an %s' --date=short "$TALAAN_SHA" -- scripts/cpu-baseline.ps1
git -C /c/Dev/Parlato log --all --follow --diff-filter=A --format='%h %ad %an %s' --date=short -- scripts/cpu-baseline.ps1
git -C /c/Dev/Ensayo log --follow --diff-filter=A --format='%h %ad %an %s' --date=short "$ENSAYO_SHA" -- scripts/cpu-baseline.ps1
```
The first-added date tells which repo had it first. If Parlato's is earlier and Talaan's is near-identical, the Talaan file may be derived from Parlato.

- [ ] **Step 3: Check the history of every file going into the engine**

```bash
for p in src-tauri/llm src-tauri/llm-protocol src-tauri/stt src-tauri/src/transcriber/models.rs src-tauri/src/summarizer/models.rs .github/actions/vulkan-sdk; do
  echo "== Talaan $p"; git -C /c/Dev/Talaan log --format='%h %ad %s' --date=short "$TALAAN_SHA" -- "$p" | grep -i -E 'parlato|parla|voiceink|port|copied|from upstream' || echo "(no mentions)"
done
for p in src-tauri/llm src-tauri/llm-protocol src-tauri/src/speech/models.rs src-tauri/src/ai/local_models.rs; do
  echo "== Ensayo $p"; git -C /c/Dev/Ensayo log --format='%h %ad %s' --date=short "$ENSAYO_SHA" -- "$p" | grep -i -E 'parlato|parla|voiceink|port|copied|from upstream' || echo "(no mentions)"
done
grep -rn -i -E 'parlato|parla\b|voiceink|GPL' /c/Dev/voice-engine-import/talaan /c/Dev/voice-engine-import/ensayo
```
Expected hits that are fine: comments naming Parlato's *folder* (`com.craftconceptsdigital.parlato`, `Models`, `ParakeetModels`) in `transcriber/models.rs`. Anything else needs a look.

- [ ] **Step 4: Write the report and decide**

Write `gpl-check.md` with: the three findings above, and a verdict per file: **clear**, or **suspect** with the reason.

- If every file is **clear**: continue to Task 2.
- If `cpu-baseline.ps1` or `.github/actions/vulkan-sdk` is **suspect**: continue, but do not copy that file in Task 6 (the CI step that needs it is rewritten from the Vulkan SDK's own documentation instead, see Task 6 Step 3).
- If any Rust file is **suspect**: **stop**. Send Miguel the report and wait.

---

### Task 2: Create the repository

**Files:**
- Create: `C:\Dev\voice-engine\{Cargo.toml, LICENSE, README.md, CHANGELOG.md, CLAUDE.md, SOURCES.md, .gitignore}`
- Create: `C:\Dev\voice-engine\docs\superpowers\specs\2026-10-28-voice-engine-design.md` (copy)
- Create: `C:\Dev\voice-engine\docs\superpowers\plans\2026-10-28-voice-engine-phase-1.md` (copy)

- [ ] **Step 1: Create and clone**

```bash
gh repo create mmnzns/voice-engine --public --description "Shared local AI and speech engine for Talaan, Ensayo and Parlato. MIT." --license mit --clone=false
git clone https://github.com/mmnzns/voice-engine /c/Dev/voice-engine
cd /c/Dev/voice-engine && cat LICENSE | head -3
```
Expected: `MIT License` and a copyright line. Edit the copyright line to `Copyright (c) 2026 Miguel Monzones`.

- [ ] **Step 2: Workspace `Cargo.toml`**

```toml
[workspace]
resolver = "3"
# Each crate is added here by the task that creates it (protocol: Task 3, llm: 7, stt: 9, models: 10), so the
# workspace always builds. At the end: ["crates/protocol", "crates/llm", "crates/stt", "crates/models"].
members = ["crates/protocol"]

[workspace.package]
version = "0.1.0"
edition = "2024"
license = "MIT"
authors = ["Miguel Monzones"]
repository = "https://github.com/mmnzns/voice-engine"
publish = false

[workspace.dependencies]
voice-engine-protocol = { path = "crates/protocol" }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
base64 = "0.22"
dirs = "7"
tempfile = "3"
```

`dirs = "7"` is the version Talaan uses (`dirs = "7.0.0"` in its `src-tauri/Cargo.toml`).

- [ ] **Step 3: `.gitignore`**

```
/target
.cargo/config.toml
```

- [ ] **Step 4: `CLAUDE.md`**

```markdown
# voice-engine

The shared local AI and speech engine for Talaan, Ensayo and Parlato. Spec: docs/superpowers/specs/.

Rules for every chat working in this repository:

- Read the engine change log on the Notion page "Voice Apps Engine Sync" before starting.
- Never copy code from Parlato, Parla, VoiceInk or any GPL source. When the engine needs something Parlato already
  does, build it from the original upstream source (whisper.cpp, llama.cpp, parakeet-rs, the model's own docs).
- Never edit an app repository (Talaan, Ensayo, Parlato). Write what an app must do in its block under
  "What each app does next" on the Notion page.
- Every tag: CI green on main, a CHANGELOG.md section, a Notion change log row, the same day. Never move or delete a
  pushed tag.
- Keep README.md's "Which app uses which tag" table current.
```

- [ ] **Step 5: `README.md`**

````markdown
# voice-engine

The local AI and speech engine shared by three desktop apps by Miguel Monzones:
[Talaan](https://github.com/mmnzns/Talaan), [Ensayo](https://github.com/mmnzns/ensayo) and
[Parlato](https://github.com/mmnzns/Parlato).

| Crate | What it does |
| --- | --- |
| `voice-engine-protocol` | Messages between an app and its helper programs |
| `voice-engine-llm` | The local AI helper (llama.cpp in its own process) |
| `voice-engine-stt` | Whisper chunk transcription and the Whisper helper |
| `voice-engine-models` | The model catalog and the finder (own folder, other apps' folders) |

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
| Talaan | not yet |
| Ensayo | not yet |
| Parlato | not yet |

## Licence

MIT. Code only flows from this engine into the apps. Nothing from Parlato's GPL-3.0 source comes back in.
````

- [ ] **Step 6: `CHANGELOG.md` and `SOURCES.md`**

`CHANGELOG.md`:
```markdown
# Changelog

## Unreleased
```

`SOURCES.md` (fill in the two SHAs from Task 0):
```markdown
# Where the code came from

| Crate | Source |
| --- | --- |
| protocol | Talaan `src-tauri/llm-protocol` @ <TALAAN_SHA>, merged with Ensayo `src-tauri/llm-protocol` @ <ENSAYO_SHA> |
| llm | Talaan `src-tauri/llm` @ <TALAAN_SHA>, merged with Ensayo `src-tauri/llm` @ <ENSAYO_SHA> |
| stt | Talaan `src-tauri/stt` @ <TALAAN_SHA> |
| models | Talaan `src-tauri/src/transcriber/models.rs` and `summarizer/models.rs` @ <TALAAN_SHA>, Ensayo `src-tauri/src/ai/local_models.rs` @ <ENSAYO_SHA> |

The history before the import is in those repositories.
```

- [ ] **Step 7: Copy the spec and this plan in**

```bash
SRC="/c/Users/mikey/My Drive (mikeymonzones@gmail.com)/MAIN DRIVE - Miguel/FILES - Projects & Work/APP Projects/App - Engine/docs/superpowers"
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp "$SRC/specs/2026-10-28-voice-engine-design.md" docs/superpowers/specs/
cp "$SRC/plans/2026-10-28-voice-engine-phase-1.md" docs/superpowers/plans/
```

- [ ] **Step 8: Commit and push** (no crates yet, so no `cargo test`)

```bash
git add -A && git commit -m "Start voice-engine: workspace, licence, readme, rules, spec and plan

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
```

---

### Task 3: `voice-engine-protocol`, imported from Talaan

**Files:**
- Create: `crates/protocol/Cargo.toml`, `crates/protocol/src/{lib.rs,gpu.rs,stt.rs}` (from Talaan)

- [ ] **Step 1: Copy and rename**

```bash
cd /c/Dev/voice-engine
mkdir -p crates/protocol/src
cp /c/Dev/voice-engine-import/talaan/src-tauri/llm-protocol/src/*.rs crates/protocol/src/
```

`crates/protocol/Cargo.toml`:
```toml
[package]
name = "voice-engine-protocol"
description = "Messages between an app and its voice-engine helper programs"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
serde.workspace = true
serde_json.workspace = true
base64.workspace = true
```

Compare with `/c/Dev/voice-engine-import/talaan/src-tauri/llm-protocol/Cargo.toml` and add any other dependency it lists (same versions).

- [ ] **Step 2: Remove app names from doc comments**

```bash
grep -n -i 'talaan' crates/protocol/src/*.rs
```
For each hit in a comment, reword to name no app ("the app", "the helper"). Leave `FAIL_ENV` alone for now (Task 4 replaces it).

- [ ] **Step 3: Run the tests**

Run: `cargo test -p voice-engine-protocol`
Expected: PASS, the same number of tests as in Talaan (`grep -c '#\[test\]' /c/Dev/voice-engine-import/talaan/src-tauri/llm-protocol/src/*.rs`).

- [ ] **Step 4: Commit**

```bash
git add crates/protocol && git commit -m "protocol: import Talaan's llm-protocol @ ${TALAAN_SHA:0:7}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Per-app helper settings and the protocol version

**Files:**
- Create: `crates/protocol/src/config.rs`
- Modify: `crates/protocol/src/lib.rs` (Event::Ready, FAIL_ENV, module list)
- Modify: `crates/protocol/src/stt.rs` (SttEvent::Ready)

- [ ] **Step 1: Write the failing tests** (append to the `#[cfg(test)] mod tests` in `crates/protocol/src/lib.rs`)

```rust
    #[test]
    fn ready_carries_the_protocol_version() {
        let line = encode(&Event::Ready { backend: Backend::cpu(), protocol: PROTOCOL_VERSION });
        assert!(line.contains(r#""protocol":1"#), "{line}");
    }

    #[test]
    fn a_ready_without_a_version_reads_as_zero() {
        let ev: Event = serde_json::from_str(r#"{"event":"ready","backend":"cpu"}"#).unwrap();
        let Event::Ready { protocol, .. } = ev else { panic!("not ready: {ev:?}") };
        assert_eq!(protocol, 0);
    }

    #[test]
    fn env_names_follow_the_app_prefix() {
        let c = HelperConfig { env_prefix: "TALAAN" };
        assert_eq!(c.gpu_env(), "TALAAN_GPU");
        assert_eq!(c.idle_secs_env(), "TALAAN_LLM_IDLE_SECS");
    }
```

And in `crates/protocol/src/stt.rs`'s test module:
```rust
    #[test]
    fn stt_ready_carries_the_protocol_version() {
        let line = crate::encode(&SttEvent::Ready { backend: Backend::default(), protocol: crate::PROTOCOL_VERSION });
        assert!(line.contains(r#""protocol":1"#), "{line}");
        let old: SttEvent = serde_json::from_str(r#"{"event":"ready"}"#).unwrap();
        assert_eq!(old, SttEvent::Ready { backend: Backend::default(), protocol: 0 });
    }
```

- [ ] **Step 2: Run to see them fail**

Run: `cargo test -p voice-engine-protocol`
Expected: compile errors: no field `protocol` on `Event::Ready`, cannot find `PROTOCOL_VERSION`, cannot find `HelperConfig`.

- [ ] **Step 3: Implement**

`crates/protocol/src/config.rs`:
```rust
//! What differs between the apps' copies of a helper: only the names of its environment variables.

/// Passed by each app's helper `main.rs`, for example `HelperConfig { env_prefix: "TALAAN" }`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HelperConfig {
    /// Upper-case app name; environment variables are `<prefix>_GPU` and `<prefix>_LLM_IDLE_SECS`.
    pub env_prefix: &'static str,
}

impl HelperConfig {
    /// `<prefix>_GPU=fail` makes a GPU build die on its first real request, to test the fallback on a real PC.
    pub fn gpu_env(&self) -> String {
        format!("{}_GPU", self.env_prefix)
    }

    /// `<prefix>_LLM_IDLE_SECS` shortens the AI helper's idle exit (tests only).
    pub fn idle_secs_env(&self) -> String {
        format!("{}_LLM_IDLE_SECS", self.env_prefix)
    }
}
```

In `crates/protocol/src/lib.rs`:
```rust
pub mod config;
pub use config::HelperConfig;

/// Bumped whenever a message changes shape. Each helper sends it in `Ready`; the app compares it with its own and
/// reports a mismatch (a helper and an app from different builds) instead of failing later. 0 = a helper from before
/// the engine existed.
pub const PROTOCOL_VERSION: u32 = 1;
```

Change `Event::Ready` to:
```rust
    Ready {
        #[serde(flatten)]
        backend: Backend,
        #[serde(default)]
        protocol: u32,
    },
```

Make the same change to `SttEvent::Ready` in `stt.rs` (add `#[serde(default)] protocol: u32` after `backend`).

Delete `pub const FAIL_ENV: &str = "TALAAN_GPU";` and its doc comment; `HelperConfig::gpu_env()` replaces it.

Fix every existing test or constructor that builds `Event::Ready { backend }` or `SttEvent::Ready { backend }` by adding `protocol: PROTOCOL_VERSION` (constructors) or `protocol: 0` (tests decoding old JSON). Find them with:
```bash
grep -n 'Ready {' crates/protocol/src/*.rs
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p voice-engine-protocol`
Expected: PASS, including the four new tests.

- [ ] **Step 5: Commit**

```bash
git add crates/protocol && git commit -m "protocol: PROTOCOL_VERSION in Ready, HelperConfig for per-app environment names

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Merge Ensayo's protocol changes

**Files:**
- Modify: `crates/protocol/src/lib.rs`

The two copies differ in about 350 lines. Talaan's is the base. The rule for each difference:

| Ensayo has… | Do |
| --- | --- |
| A function Talaan lacks (`decode_request`, `decode_event`, …) | Add it unchanged |
| A field Talaan lacks | Add it with `#[serde(default)]` (and `skip_serializing_if` when it is an `Option`), so Talaan's messages still parse |
| A request or event variant Talaan lacks | Add it |
| The same field with a different type or meaning | **Stop and report** (ground rule 3) |
| Fewer fields than Talaan (e.g. `Backend` with only `backend`) | Keep Talaan's; Ensayo's JSON still parses because Talaan's extra fields have defaults. Check that they do |
| Tests Talaan lacks | Port them, adapting constructors to Talaan's shapes (`Backend { backend: "cpu".into(), ..Default::default() }`, `protocol: PROTOCOL_VERSION`) |

- [ ] **Step 1: List the differences**

```bash
diff -u /c/Dev/voice-engine-import/talaan/src-tauri/llm-protocol/src/lib.rs /c/Dev/voice-engine-import/ensayo/src-tauri/llm-protocol/src/lib.rs > /c/Dev/voice-engine-import/protocol.diff; wc -l /c/Dev/voice-engine-import/protocol.diff
grep -n 'pub fn\|pub struct\|pub enum\|^    [A-Z][a-z]*[ ({,]' /c/Dev/voice-engine-import/ensayo/src-tauri/llm-protocol/src/lib.rs
```
Go through `protocol.diff` hunk by hunk and write each difference with its decision (from the table) in `/c/Dev/voice-engine-import/protocol-merge.md`.

- [ ] **Step 2: Port Ensayo's tests first** (they must fail where Talaan's code lacks Ensayo's functions)

Copy each `#[test]` from Ensayo's `lib.rs` test module into `crates/protocol/src/lib.rs`'s test module, adapted per the table. Prefix their names with `ensayo_` so they are easy to find.

Run: `cargo test -p voice-engine-protocol`
Expected: FAIL to compile on the missing functions/fields (for example `cannot find function decode_event`).

- [ ] **Step 3: Apply the merge decisions** from `protocol-merge.md` to `crates/protocol/src/lib.rs`.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p voice-engine-protocol`
Expected: PASS, Talaan's tests and every `ensayo_` test.

- [ ] **Step 5: Commit**

```bash
git add crates/protocol && git commit -m "protocol: merge Ensayo's llm-protocol @ ${ENSAYO_SHA:0:7} (decode helpers, its fields and tests)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: CI

**Files:**
- Create: `.github/workflows/ci.yml`
- Create: `.github/actions/vulkan-sdk/` (copied from Talaan, only if Task 1 cleared it)

- [ ] **Step 1: Copy the Vulkan SDK action** (skip if Task 1 marked it suspect)

```bash
mkdir -p .github/actions && cp -r /c/Dev/voice-engine-import/talaan/.github/actions/vulkan-sdk .github/actions/
grep -n -i talaan .github/actions/vulkan-sdk/*
```
Reword any Talaan mentions.

- [ ] **Step 2: Read what Talaan's `Use-NinjaForGpu` does**

```bash
grep -n -A25 'function Use-NinjaForGpu' /c/Dev/voice-engine-import/talaan/scripts/cpu-baseline.ps1
```
Note the environment variables it sets. The engine CI sets those same variables directly in the workflow step instead of copying the script.

- [ ] **Step 3: `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
  # Actions › CI › Run workflow: also builds the Vulkan GPU versions (windows-gpu, below).
  workflow_dispatch:

jobs:
  test:
    strategy:
      fail-fast: false
      matrix:
        os: [windows-latest, macos-latest]
    runs-on: ${{ matrix.os }}
    env:
      LIBCLANG_PATH: C:\Program Files\LLVM\bin
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt, clippy
      - uses: Swatinem/rust-cache@v2
      - run: cargo fmt --all --check
      - run: cargo clippy --workspace --all-targets -- -D warnings
      - run: cargo test --workspace

  # The Vulkan builds of the llm and stt crates, with the Vulkan SDK. Manual only: the shaders take a while.
  windows-gpu:
    if: github.event_name == 'workflow_dispatch'
    runs-on: windows-latest
    env:
      LIBCLANG_PATH: C:\Program Files\LLVM\bin
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: Swatinem/rust-cache@v2
        with:
          key: windows-gpu
          cache-directories: C:\tg
      - uses: ./.github/actions/vulkan-sdk
      - name: Vulkan builds and their tests
        shell: pwsh
        run: |
          # Set here the variables Talaan's Use-NinjaForGpu sets (Task 6 Step 2).
          cargo test --release -p voice-engine-llm -p voice-engine-stt -p voice-engine-protocol --features voice-engine-llm/vulkan,voice-engine-stt/vulkan --target-dir C:\tg
```

Replace the comment line in the last step with the actual `$env:NAME = "value"` lines found in Step 2. If Task 1 marked the Vulkan action suspect, replace `uses: ./.github/actions/vulkan-sdk` with an install step written from LunarG's documentation for the Windows Vulkan SDK.

Do not run the `windows-gpu` job until Task 9 (the `llm` and `stt` crates don't exist yet).

- [ ] **Step 4: Format and lint locally**

Run: `cargo fmt --all && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
Expected: no warnings, PASS. Fix clippy findings in the imported code (they are style fixes only; do not change behaviour).

- [ ] **Step 5: Commit, push, and check CI**

```bash
git add -A && git commit -m "CI: fmt, clippy and tests on Windows and macOS; manual Vulkan job

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
gh run watch --repo mmnzns/voice-engine --exit-status $(gh run list --repo mmnzns/voice-engine --limit 1 --json databaseId -q '.[0].databaseId')
```
Expected: both matrix jobs pass.

---

### Task 7: `voice-engine-llm`, imported from Talaan

**Files:**
- Create: `crates/llm/Cargo.toml`, `crates/llm/src/lib.rs` (Talaan's `llm/src/main.rs` as a library), `crates/llm/src/main.rs` (test binary)

- [ ] **Step 1: Copy**

```bash
mkdir -p crates/llm/src
cp /c/Dev/voice-engine-import/talaan/src-tauri/llm/src/main.rs crates/llm/src/lib.rs
ls /c/Dev/voice-engine-import/talaan/src-tauri/llm/src/
```
If Talaan's `llm/src` has other `.rs` files, copy them too, unchanged.

`crates/llm/Cargo.toml`:
```toml
[package]
name = "voice-engine-llm"
description = "The voice-engine local AI helper: llama.cpp in its own process"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[lib]
name = "voice_engine_llm"
path = "src/lib.rs"

# Only for this crate's own tests (tests/process.rs). Apps build their own helper binary with their own name.
[[bin]]
name = "voice-engine-llm"
path = "src/main.rs"

[dependencies]
voice-engine-protocol.workspace = true
serde_json.workspace = true
llama-cpp-2 = { version = "=0.1.159", default-features = false, features = ["common"] }

[target.'cfg(target_os = "macos")'.dependencies]
llama-cpp-2 = { version = "=0.1.159", default-features = false, features = ["metal", "common"] }

[features]
# Windows GPU build: needs the Vulkan SDK (VULKAN_SDK) to build and vulkan-1.dll, installed by GPU drivers, to run.
vulkan = ["llama-cpp-2/vulkan"]
```

Add `"crates/llm"` to `members` in the root `Cargo.toml`.

- [ ] **Step 2: Turn `main` into `run`**

In `crates/llm/src/lib.rs`:
1. Rename `talaan_llm_protocol` to `voice_engine_protocol` everywhere: `sed -i 's/talaan_llm_protocol/voice_engine_protocol/g' crates/llm/src/*.rs`
2. Change `fn main() {` to:
   ```rust
   /// Runs the helper until the app closes stdin or the idle timeout passes. Called by each app's helper `main.rs`.
   pub fn run(config: voice_engine_protocol::HelperConfig) {
   ```
3. Every use of `FAIL_ENV` becomes `config.gpu_env()` (an owned `String`; borrow it with `&` where a `&str` is needed). Functions that used `FAIL_ENV` take `config: &HelperConfig` as a parameter.
4. Every place that builds `Event::Ready { backend: … }` adds `protocol: voice_engine_protocol::PROTOCOL_VERSION`.
5. Reword comments and user-visible strings that name Talaan: `grep -n -i talaan crates/llm/src/*.rs` must print nothing at the end.
6. If any item was private and is now unused outside `run`, leave it private; the library only needs `pub fn run`.

`crates/llm/src/main.rs`:
```rust
fn main() {
    voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "VOICE_ENGINE" });
}
```

- [ ] **Step 3: Build and test**

Run: `cargo test -p voice-engine-llm`
Expected: PASS (Talaan's unit tests in that file, if any, now run as library tests). The first build compiles llama.cpp and takes several minutes.

- [ ] **Step 4: Check the binary answers a ping**

```bash
printf '{"op":"ping"}\n' | ./target/debug/voice-engine-llm.exe
```
Expected: one line like `{"event":"ready","backend":"cpu",…,"protocol":1}`, then the program exits (stdin closed).

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "llm: import Talaan's llm helper @ ${TALAAN_SHA:0:7} as a library with run(HelperConfig)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Merge Ensayo's helper changes and process tests

**Files:**
- Modify: `crates/llm/src/lib.rs`, `crates/llm/Cargo.toml`
- Create: `crates/llm/tests/process.rs` (from Ensayo)

The two `main.rs` files differ in about 550 lines. Talaan's is the base. Known difference: Ensayo's helper exits after an idle time (`IDLE_EXIT`, overridable with `ENSAYO_LLM_IDLE_SECS`).

- [ ] **Step 1: List the differences**

```bash
diff -u /c/Dev/voice-engine-import/talaan/src-tauri/llm/src/main.rs /c/Dev/voice-engine-import/ensayo/src-tauri/llm/src/main.rs > /c/Dev/voice-engine-import/llm.diff; wc -l /c/Dev/voice-engine-import/llm.diff
diff /c/Dev/voice-engine-import/talaan/src-tauri/llm/Cargo.toml /c/Dev/voice-engine-import/ensayo/src-tauri/llm/Cargo.toml
```
Write each behavioural difference, and the decision, in `/c/Dev/voice-engine-import/llm-merge.md`. Decide with this table:

| Difference | Decision |
| --- | --- |
| A feature only Ensayo has (idle exit, a new request kind, a better error message) | Add it. If it changes behaviour Talaan relies on, make it the default only when it is harmless for Talaan (an idle exit is: Talaan's app restarts the helper on the next request; confirm by reading `/c/Dev/Talaan/src-tauri/src/helper.rs`, read only) |
| A feature only Talaan has (GPU choice, embeddings, Vulkan fail switch) | Keep it |
| The same thing done two ways with the same result | Keep Talaan's |
| The same thing done two ways with different results (for example different sampling defaults or prompt handling) | **Stop and report** (ground rule 3), with both versions side by side |
| An `ENSAYO_…` environment variable | Use `config.idle_secs_env()` (or a new `HelperConfig` method named after the variable, added in `crates/protocol/src/config.rs` with a test like `env_names_follow_the_app_prefix`) |

- [ ] **Step 2: Bring in Ensayo's process tests first**

```bash
mkdir -p crates/llm/tests
cp /c/Dev/voice-engine-import/ensayo/src-tauri/llm/tests/process.rs crates/llm/tests/process.rs
sed -i 's/CARGO_BIN_EXE_ensayo-llm/CARGO_BIN_EXE_voice-engine-llm/; s/ENSAYO_LLM_IDLE_SECS/VOICE_ENGINE_LLM_IDLE_SECS/g' crates/llm/tests/process.rs
grep -n -i ensayo crates/llm/tests/process.rs
```
Reword any remaining mention.

Run: `cargo test -p voice-engine-llm --test process`
Expected: FAIL on the idle-exit test (Talaan's helper does not exit when idle yet). The ping test may already pass.

- [ ] **Step 3: Apply the merge decisions** from `llm-merge.md` to `crates/llm/src/lib.rs` and `crates/llm/Cargo.toml`.

- [ ] **Step 4: Run all the tests**

Run: `cargo test -p voice-engine-llm && cargo clippy -p voice-engine-llm --all-targets -- -D warnings`
Expected: PASS, including every test in `tests/process.rs`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "llm: merge Ensayo's helper @ ${ENSAYO_SHA:0:7} (idle exit, process tests)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: `voice-engine-stt`, imported from Talaan

**Files:**
- Create: `crates/stt/Cargo.toml`, `crates/stt/src/lib.rs` (Talaan `stt/src/lib.rs`), `crates/stt/src/helper.rs` (Talaan `stt/src/main.rs` as `pub fn run_helper`), `crates/stt/src/main.rs` (test binary)

- [ ] **Step 1: Copy**

```bash
mkdir -p crates/stt/src
cp /c/Dev/voice-engine-import/talaan/src-tauri/stt/src/lib.rs crates/stt/src/lib.rs
cp /c/Dev/voice-engine-import/talaan/src-tauri/stt/src/main.rs crates/stt/src/helper.rs
sed -i 's/talaan_llm_protocol/voice_engine_protocol/g; s/talaan_stt::/crate::/g' crates/stt/src/*.rs
```

`crates/stt/Cargo.toml`:
```toml
[package]
name = "voice-engine-stt"
description = "Whisper chunk transcription, shared by an app (processor) and its Windows GPU helper"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[lib]
name = "voice_engine_stt"
path = "src/lib.rs"

# Only for this crate's own tests. Apps build their own helper binary with their own name.
[[bin]]
name = "voice-engine-stt"
path = "src/main.rs"

[dependencies]
voice-engine-protocol.workspace = true
serde_json.workspace = true
# raw-api: ggml's device list (whisper_rs::whisper_rs_sys), to choose the GPU.
whisper-rs = { version = "0.16.0", features = ["raw-api"] }

[dev-dependencies]
tempfile.workspace = true

[features]
# Windows GPU build: needs the Vulkan SDK (VULKAN_SDK) to build and vulkan-1.dll to run. Never linked into an app's
# main program.
vulkan = ["whisper-rs/vulkan"]
```

Compare with `/c/Dev/voice-engine-import/talaan/src-tauri/stt/Cargo.toml` and add anything else it lists. Add `"crates/stt"` to the root `members`.

- [ ] **Step 2: Turn the helper's `main` into `run_helper`**

In `crates/stt/src/helper.rs`: change `fn main() {` to
```rust
/// Runs the Whisper helper until the app closes stdin. Called by each app's GPU helper `main.rs`.
pub fn run_helper(config: voice_engine_protocol::HelperConfig) {
```
Replace `FAIL_ENV` with `config.gpu_env()`; add `protocol: voice_engine_protocol::PROTOCOL_VERSION` to every `SttEvent::Ready { … }`; move any `use` lines that only the helper needs to the top of `helper.rs`.

At the top of `crates/stt/src/lib.rs` add:
```rust
mod helper;
pub use helper::run_helper;
```

`crates/stt/src/main.rs`:
```rust
fn main() {
    voice_engine_stt::run_helper(voice_engine_protocol::HelperConfig { env_prefix: "VOICE_ENGINE" });
}
```

`grep -n -i talaan crates/stt/src/*.rs` must print nothing at the end (reword comments).

- [ ] **Step 3: Test**

Run: `cargo test -p voice-engine-stt && printf '{"op":"ping"}\n' | ./target/debug/voice-engine-stt.exe`
Expected: tests PASS (same count as Talaan's `stt`); the ping prints one `"event":"ready"` line containing `"protocol":1`.

- [ ] **Step 4: Commit and push**

```bash
git add -A && git commit -m "stt: import Talaan's Whisper crate @ ${TALAAN_SHA:0:7}, helper as run_helper(HelperConfig)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
```
Check CI passes (`gh run watch …` as in Task 6).

---

### Task 10: `voice-engine-models`: the speech catalog

**Files:**
- Create: `crates/models/Cargo.toml`, `crates/models/src/lib.rs`, `crates/models/src/catalog.rs`

- [ ] **Step 1: Crate skeleton**

`crates/models/Cargo.toml`:
```toml
[package]
name = "voice-engine-models"
description = "The voice-engine model catalog and the model finder"
version.workspace = true
edition.workspace = true
license.workspace = true
publish.workspace = true

[dependencies]
dirs.workspace = true

[dev-dependencies]
tempfile.workspace = true
```

`crates/models/src/lib.rs` (Tasks 11, 12 and 13 each add their module line here):
```rust
//! What each model is (the catalogs), where each app keeps its models (`apps`), and finding a model on this computer
//! (`store`).

pub mod catalog;

pub use catalog::{EngineKind, ModelFile, ModelInfo};
```

Add `"crates/models"` to the root `members`.

- [ ] **Step 2: Copy Talaan's catalog**

```bash
cp /c/Dev/voice-engine-import/talaan/src-tauri/src/transcriber/models.rs crates/models/src/catalog.rs
```

In `crates/models/src/catalog.rs` **keep**: the module doc (reworded to name no app), `EngineKind` and its `impl`, `ModelFile`, `ModelInfo` and its `impl`, the URL prefix constants, `CATALOG`, `VAD`, `DIARIZATION`, `SEARCH`, `DEFAULT_MODEL`, `find`, `PREFIXES`, and the tests `catalog_is_consistent` and any helper it uses.

**Delete** (they move to `store.rs` and `apps.rs`, rewritten in Tasks 12–13, or stay in Talaan): `Location` and its `impl`, `MODELS_ROOT`, `set_models_root`, `local_models_root`, `models_root`, `ModelStore` and its `impl`, `complete`, `default_reuse_roots`, and the tests `the_preview_lists_the_same_speech_models_in_the_same_order`, `own_copy_wins_and_partial_files_do_not_count`, `another_apps_copy_is_found_only_when_reuse_is_on`, `the_old_folder_is_read_until_the_move_is_done`, `without_a_separate_models_folder_there_is_no_old_one`, `windows_models_live_in_the_bundle_id_folder`, and the test helpers `fake_model` and `sparse` if nothing left uses them.

Also delete `DEFAULT_MODEL` if you prefer, but keep it for now: Talaan and Ensayo both use `"whisper-turbo"`. The *choice* of default stays each app's; this constant is only a convenience. Add the doc comment `/// The default both Talaan and Ensayo use today. Each app may pick its own.`

`grep -n -i 'talaan\|on this mac' crates/models/src/catalog.rs` must print nothing at the end except inside model `note` strings that are about the model itself.

- [ ] **Step 3: Run the tests**

Run: `cargo test -p voice-engine-models`
Expected: PASS (`catalog_is_consistent`).

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "models: Talaan's speech catalog @ ${TALAAN_SHA:0:7}

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Compare catalogs, and the local AI catalog

**Files:**
- Create: `crates/models/src/llm_catalog.rs`
- Create: `C:\Dev\voice-engine-import\catalog-compare.md` (report)

- [ ] **Step 1: Compare Ensayo's speech catalog with the engine's**

Write a throwaway comparison: for every `ModelInfo` in Ensayo's `speech/models.rs` (`/c/Dev/voice-engine-import/ensayo/src-tauri/src/speech/models.rs`) find the entry with the same `id` in `crates/models/src/catalog.rs` and compare each file's `name`, `url`, `size` and `sha256`.

```bash
cd /c/Dev/voice-engine-import
grep -o 'id: "[^"]*"\|name: "[^"]*\.[a-z0-9]*"\|size: [0-9_]*\|sha256: "[0-9a-f]*"' ensayo/src-tauri/src/speech/models.rs > ensayo-catalog.txt
grep -o 'id: "[^"]*"\|name: "[^"]*\.[a-z0-9]*"\|size: [0-9_]*\|sha256: "[0-9a-f]*"' /c/Dev/voice-engine/crates/models/src/catalog.rs > engine-catalog.txt
diff ensayo-catalog.txt engine-catalog.txt
```
Expected: the only differences are models Ensayo does not list (`diarization`, `search`). Write the result in `catalog-compare.md`. **If any shared model id has a different file name, size or hash: stop and report** (ground rule 3). Differences in `note`, `speed` or `accuracy` text are fine: write them in the report for Miguel; the engine keeps Talaan's.

- [ ] **Step 2: Compare the two AI catalogs**

```bash
grep -o 'id: "[^"]*"\|name: "[^"]*\.gguf"\|size: [0-9_]*\|sha256: "[0-9a-f]*"\|url: "[^"]*"' ensayo/src-tauri/src/ai/local_models.rs
grep -o 'id: "[^"]*"\|name: "[^"]*\.gguf"\|size: [0-9_]*\|sha256: "[0-9a-f]*"\|url: "[^"]*"' talaan/src-tauri/src/summarizer/models.rs
```
Expected: both list `qwen3.5-4b` (`Qwen3.5-4B-Q4_K_M.gguf`, 2_740_937_888 bytes, sha256 `00fe7986…`) and `qwen3.5-9b` (`Qwen3.5-9B-Q4_K_M.gguf`, 5_680_522_464 bytes, sha256 `03b74727…`) with the same URLs. If not, stop and report.

- [ ] **Step 3: Write the failing test** in a new file `crates/models/src/llm_catalog.rs`, and add `pub mod llm_catalog;` to `crates/models/src/lib.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineKind;

    #[test]
    fn local_ai_catalog_is_consistent() {
        assert!(!CATALOG.is_empty());
        for m in CATALOG {
            assert_eq!(m.engine, EngineKind::Llm, "{}", m.id);
            assert_eq!(m.files.len(), 1, "{}: one GGUF file", m.id);
            let f = &m.files[0];
            assert!(f.name.ends_with(".gguf"), "{}", f.name);
            assert!(f.url.ends_with(f.name), "{}: url names the file", m.id);
            assert!(f.url.contains("/resolve/") && !f.url.contains("/resolve/main/"), "{}: pinned to a commit", m.id);
            assert_eq!(f.sha256.len(), 64, "{}", m.id);
            assert!(f.size > 1_000_000_000, "{}", m.id);
        }
        assert_eq!(find("qwen3.5-4b").map(|m| m.files[0].name), Some("Qwen3.5-4B-Q4_K_M.gguf"));
        assert!(find("nope").is_none());
    }
}
```

Run: `cargo test -p voice-engine-models`
Expected: FAIL to compile (`CATALOG` and `find` not defined in `llm_catalog`).

- [ ] **Step 4: Implement** (the entries are Ensayo's `info` blocks; `licence` and `recommended` stay in each app)

```rust
//! The local AI models (GGUF files run by the llm helper). What each model is and where it downloads from; which
//! one an app recommends, and its licence text, stay in the app.

use crate::catalog::{EngineKind, ModelFile, ModelInfo};

pub static CATALOG: &[ModelInfo] = &[
    ModelInfo {
        id: "qwen3.5-4b",
        name: "Qwen3.5 4B",
        maker: "Qwen",
        note: "Small and quick enough for a laptop.",
        engine: EngineKind::Llm,
        languages: "Many languages",
        speed: 3,
        accuracy: 2,
        files: &[ModelFile {
            name: "Qwen3.5-4B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/e87f176479d0855a907a41277aca2f8ee7a09523/Qwen3.5-4B-Q4_K_M.gguf",
            size: 2_740_937_888,
            sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        }],
        reuse_dirs: &[],
    },
    ModelInfo {
        id: "qwen3.5-9b",
        name: "Qwen3.5 9B",
        maker: "Qwen",
        note: "Larger and a little more thorough; needs 16 GB of memory and takes about twice as long.",
        engine: EngineKind::Llm,
        languages: "Many languages",
        speed: 2,
        accuracy: 3,
        files: &[ModelFile {
            name: "Qwen3.5-9B-Q4_K_M.gguf",
            url: "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/3885219b6810b007914f3a7950a8d1b469d598a5/Qwen3.5-9B-Q4_K_M.gguf",
            size: 5_680_522_464,
            sha256: "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        }],
        reuse_dirs: &[],
    },
];

pub fn find(id: &str) -> Option<&'static ModelInfo> {
    CATALOG.iter().find(|m| m.id == id)
}
```

Use the URLs, sizes and hashes confirmed in Step 2 (they are the ones above if Step 2 matched).

- [ ] **Step 5: Run the tests**

Run: `cargo test -p voice-engine-models`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "models: local AI catalog (Qwen3.5 4B and 9B), checked against Talaan's and Ensayo's

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: `voice-engine-models`: where each app keeps models

**Files:**
- Create: `crates/models/src/apps.rs`

- [ ] **Step 1: Write the failing tests** (bottom of a new file `crates/models/src/apps.rs`), and add to `crates/models/src/lib.rs`:

```rust
pub mod apps;
pub use apps::{App, Layout, OtherRoot};
```

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{catalog, llm_catalog};
    use std::path::Path;

    const WIN: &str = r"C:\Users\a\AppData\Local";
    const MAC: &str = "/Users/a/Library/Application Support";

    fn whisper() -> &'static ModelInfo {
        catalog::find("whisper-turbo").unwrap()
    }
    fn qwen() -> &'static ModelInfo {
        llm_catalog::find("qwen3.5-4b").unwrap()
    }

    #[test]
    fn talaan_keeps_models_in_its_bundle_id_folder_on_windows_and_its_name_on_the_mac() {
        let w = App::Talaan.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(w.root, Path::new(WIN).join("com.craftconceptsdigital.talaan").join("models"));
        assert_eq!(w.layout, Layout::ById);
        let m = App::Talaan.root_in(EngineKind::Whisper, Path::new(MAC), false);
        assert_eq!(m.root, Path::new(MAC).join("Talaan").join("models"));
        // Talaan keeps its AI models next to the speech ones.
        assert_eq!(App::Talaan.root_in(EngineKind::Llm, Path::new(WIN), true).root, w.root);
    }

    #[test]
    fn ensayo_keeps_ai_models_in_an_llm_subfolder() {
        let s = App::Ensayo.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(s.root, Path::new(WIN).join("com.craftconceptsdigital.ensayo").join("models"));
        let l = App::Ensayo.root_in(EngineKind::Llm, Path::new(MAC), false);
        assert_eq!(l.root, Path::new(MAC).join("com.craftconceptsdigital.ensayo").join("models").join("llm"));
    }

    #[test]
    fn parlato_uses_its_reuse_dirs_under_local_app_data_not_roaming() {
        let p = App::Parlato.root_in(EngineKind::Whisper, Path::new(WIN), true);
        assert_eq!(p.root, Path::new(WIN).join("com.craftconceptsdigital.parlato"));
        assert_eq!(p.layout, Layout::ReuseDirs);
        assert_eq!(p.candidates(whisper()), vec![p.root.join("Models")]);
    }

    #[test]
    fn by_id_layout_looks_in_the_model_id_folder() {
        let t = App::Talaan.root_in(EngineKind::Llm, Path::new(WIN), true);
        assert_eq!(t.candidates(qwen()), vec![t.root.join("qwen3.5-4b")]);
    }

    #[test]
    fn others_never_includes_the_app_itself() {
        for me in App::ALL {
            let others = App::others_in(me, EngineKind::Whisper, Path::new(WIN), true);
            assert_eq!(others.len(), 2);
            assert!(others.iter().all(|o| o.app != me));
        }
    }
}
```

Note: `whisper()`'s `reuse_dirs` in Talaan's catalog is `["Models"]`; if Task 10 kept a different value, adjust the expected vector to match the catalog, not the other way round.

Run: `cargo test -p voice-engine-models apps`
Expected: FAIL to compile (`App`, `Layout`, `OtherRoot` not defined).

- [ ] **Step 2: Implement** (top of `apps.rs`)

```rust
//! Where each app keeps its models, the one place that says so. Paths were read from each app's code on 2026-10-28
//! (Talaan `transcriber/models.rs`, Ensayo `paths.rs` and `ai/local_models.rs`, Parlato's model managers, which use
//! Tauri's `app_local_data_dir`). Only paths were read from Parlato; no code.

use crate::catalog::{EngineKind, ModelInfo};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum App {
    Talaan,
    Ensayo,
    Parlato,
}

/// How an app lays out model files under its root.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// `<root>/<model id>/<files>` (Talaan, Ensayo).
    ById,
    /// `<root>/<each of ModelInfo::reuse_dirs>/<files>` (Parlato).
    ReuseDirs,
}

/// One app's models folder for one kind of model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OtherRoot {
    pub app: App,
    pub root: PathBuf,
    pub layout: Layout,
}

impl OtherRoot {
    /// The folders under this root that may hold `m`'s files.
    pub fn candidates(&self, m: &ModelInfo) -> Vec<PathBuf> {
        match self.layout {
            Layout::ById => vec![self.root.join(m.id)],
            Layout::ReuseDirs => m.reuse_dirs.iter().map(|d| self.root.join(d)).collect(),
        }
    }
}

impl App {
    pub const ALL: [App; 3] = [App::Talaan, App::Ensayo, App::Parlato];

    pub fn bundle_id(self) -> &'static str {
        match self {
            App::Talaan => "com.craftconceptsdigital.talaan",
            App::Ensayo => "com.craftconceptsdigital.ensayo",
            App::Parlato => "com.craftconceptsdigital.parlato",
        }
    }

    /// Where this app keeps models of `engine`, given the local app data folder (`%LOCALAPPDATA%` on Windows,
    /// `~/Library/Application Support` on the Mac). Pure, so tests can check every platform.
    pub fn root_in(self, engine: EngineKind, local_data: &Path, windows: bool) -> OtherRoot {
        let (root, layout) = match self {
            App::Talaan if windows => (local_data.join(self.bundle_id()).join("models"), Layout::ById),
            App::Talaan => (local_data.join("Talaan").join("models"), Layout::ById),
            App::Ensayo => {
                let models = local_data.join(self.bundle_id()).join("models");
                (if engine == EngineKind::Llm { models.join("llm") } else { models }, Layout::ById)
            }
            App::Parlato => (local_data.join(self.bundle_id()), Layout::ReuseDirs),
        };
        OtherRoot { app: self, root, layout }
    }

    /// This app's folder for `engine` on this computer. None when the system has no local data folder.
    pub fn root(self, engine: EngineKind) -> Option<OtherRoot> {
        dirs::data_local_dir().map(|d| self.root_in(engine, &d, cfg!(target_os = "windows")))
    }

    /// The other two apps' folders, given the local app data folder. Pure, for tests.
    pub fn others_in(me: App, engine: EngineKind, local_data: &Path, windows: bool) -> Vec<OtherRoot> {
        App::ALL.into_iter().filter(|a| *a != me).map(|a| a.root_in(engine, local_data, windows)).collect()
    }

    /// The other two apps' folders on this computer (Phase 4 turns this on for every app).
    pub fn others(me: App, engine: EngineKind) -> Vec<OtherRoot> {
        dirs::data_local_dir().map(|d| App::others_in(me, engine, &d, cfg!(target_os = "windows"))).unwrap_or_default()
    }
}
```

Check before trusting the Talaan Mac row: in `/c/Dev/voice-engine-import/ensayo/src-tauri/src/speech/models.rs`, `talaan_models_root()` uses `dirs::data_dir().join("Talaan")` on the Mac; on macOS `dirs::data_dir()` and `dirs::data_local_dir()` are the same folder, so `root_in(…, false)` matches it.

- [ ] **Step 3: Run the tests**

Run: `cargo test -p voice-engine-models`
Expected: PASS.

- [ ] **Step 4: Commit and push**

```bash
git add -A && git commit -m "models: known apps table (Talaan, Ensayo, Parlato folders on Windows and the Mac)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
```

---

### Task 13: `voice-engine-models`: the finder

**Files:**
- Create: `crates/models/src/store.rs`

- [ ] **Step 1: Write the failing tests** (bottom of a new file `crates/models/src/store.rs`), and add to `crates/models/src/lib.rs`:

```rust
pub mod store;
pub use store::{Location, ModelStore};
```

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{EngineKind, ModelFile, ModelInfo};
    use crate::{App, Layout, OtherRoot};
    use std::fs;

    static FAKE: ModelInfo = ModelInfo {
        id: "fake",
        name: "Fake",
        maker: "Test",
        note: "",
        engine: EngineKind::Whisper,
        languages: "",
        speed: 1,
        accuracy: 1,
        files: &[ModelFile { name: "a.bin", url: "https://example.com/a.bin", size: 10, sha256: "" }],
        reuse_dirs: &["Models"],
    };

    /// A file of exactly `size` bytes (sparse, so tests stay fast).
    fn file(path: &Path, size: u64) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::File::create(path).unwrap().set_len(size).unwrap();
    }

    fn other(app: App, root: &Path, layout: Layout) -> OtherRoot {
        OtherRoot { app, root: root.to_path_buf(), layout }
    }

    #[test]
    fn own_copy_wins_and_a_partial_file_does_not_count() {
        let own = tempfile::tempdir().unwrap();
        let store = ModelStore::new(own.path().to_path_buf());
        assert_eq!(store.locate(&FAKE), Location::Missing);
        file(&own.path().join("fake").join("a.bin"), 9);
        assert_eq!(store.locate(&FAKE), Location::Missing, "wrong size");
        file(&own.path().join("fake").join("a.bin"), 10);
        assert_eq!(store.locate(&FAKE), Location::Own(own.path().join("fake")));
    }

    #[test]
    fn the_legacy_folder_counts_as_own() {
        let own = tempfile::tempdir().unwrap();
        let old = tempfile::tempdir().unwrap();
        file(&old.path().join("fake").join("a.bin"), 10);
        let store = ModelStore::new(own.path().to_path_buf()).with_legacy(Some(old.path().to_path_buf()));
        assert_eq!(store.locate(&FAKE), Location::Own(old.path().join("fake")));
        assert_eq!(store.own_dirs(&FAKE), vec![own.path().join("fake"), old.path().join("fake")]);
    }

    #[test]
    fn another_apps_copy_is_found_only_when_reuse_is_on() {
        let own = tempfile::tempdir().unwrap();
        let parlato = tempfile::tempdir().unwrap();
        file(&parlato.path().join("Models").join("a.bin"), 10);
        let others = vec![other(App::Parlato, parlato.path(), Layout::ReuseDirs)];
        let on = ModelStore::new(own.path().to_path_buf()).with_others(others.clone(), true);
        assert_eq!(on.locate(&FAKE), Location::OtherApp { app: App::Parlato, dir: parlato.path().join("Models") });
        let off = ModelStore::new(own.path().to_path_buf()).with_others(others, false);
        assert_eq!(off.locate(&FAKE), Location::Missing);
    }

    #[test]
    fn by_id_layout_is_found_in_the_model_id_folder() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        file(&talaan.path().join("fake").join("a.bin"), 10);
        let store = ModelStore::new(own.path().to_path_buf()).with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert_eq!(store.locate(&FAKE), Location::OtherApp { app: App::Talaan, dir: talaan.path().join("fake") });
    }

    #[test]
    fn own_copy_is_preferred_over_another_apps() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        file(&own.path().join("fake").join("a.bin"), 10);
        file(&talaan.path().join("fake").join("a.bin"), 10);
        let store = ModelStore::new(own.path().to_path_buf()).with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert_eq!(store.locate(&FAKE), Location::Own(own.path().join("fake")));
    }

    #[test]
    fn a_shared_copy_that_disappears_reads_as_missing() {
        let own = tempfile::tempdir().unwrap();
        let talaan = tempfile::tempdir().unwrap();
        let f = talaan.path().join("fake").join("a.bin");
        file(&f, 10);
        let store = ModelStore::new(own.path().to_path_buf()).with_others(vec![other(App::Talaan, talaan.path(), Layout::ById)], true);
        assert!(matches!(store.locate(&FAKE), Location::OtherApp { .. }));
        fs::remove_file(&f).unwrap();
        assert_eq!(store.locate(&FAKE), Location::Missing);
    }

    #[test]
    fn location_dir() {
        let p = PathBuf::from("x");
        assert_eq!(Location::Own(p.clone()).dir(), Some(p.as_path()));
        assert_eq!(Location::OtherApp { app: App::Talaan, dir: p.clone() }.dir(), Some(p.as_path()));
        assert_eq!(Location::Missing.dir(), None);
    }
}
```

Run: `cargo test -p voice-engine-models store`
Expected: FAIL to compile (`ModelStore`, `Location` not defined).

- [ ] **Step 2: Implement** (top of `store.rs`)

```rust
//! Finding a model on this computer: the app's own folder (and its legacy folder), then other apps' folders when
//! reuse is on. A copy in another app's folder is used in place and read only; this code never writes, moves or
//! deletes anything there.

use crate::apps::{App, OtherRoot};
use crate::catalog::ModelInfo;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Location {
    Missing,
    /// The app's own copy.
    Own(PathBuf),
    /// Another app's copy, read in place.
    OtherApp { app: App, dir: PathBuf },
}

impl Location {
    pub fn dir(&self) -> Option<&Path> {
        match self {
            Location::Own(p) | Location::OtherApp { dir: p, .. } => Some(p),
            Location::Missing => None,
        }
    }
}

/// Where an app's models are and which other apps' folders it may read.
#[derive(Debug, Clone)]
pub struct ModelStore {
    /// `<root>/<model id>/<files>`: where the app downloads to.
    pub root: PathBuf,
    /// An older own folder still read as the app's own copy until the app has moved its models out of it.
    pub legacy_root: Option<PathBuf>,
    pub others: Vec<OtherRoot>,
    pub reuse: bool,
}

impl ModelStore {
    /// The app's own folder only; no legacy folder, no other apps.
    pub fn new(root: PathBuf) -> Self {
        ModelStore { root, legacy_root: None, others: Vec::new(), reuse: false }
    }

    pub fn with_legacy(mut self, legacy_root: Option<PathBuf>) -> Self {
        self.legacy_root = legacy_root.filter(|l| *l != self.root);
        self
    }

    pub fn with_others(mut self, others: Vec<OtherRoot>, reuse: bool) -> Self {
        self.others = others;
        self.reuse = reuse;
        self
    }

    /// Where the app downloads `m` to.
    pub fn own_dir(&self, m: &ModelInfo) -> PathBuf {
        self.root.join(m.id)
    }

    /// Every own folder that may hold `m` (current, then legacy): what Delete removes.
    pub fn own_dirs(&self, m: &ModelInfo) -> Vec<PathBuf> {
        std::iter::once(self.own_dir(m)).chain(self.legacy_root.iter().map(|r| r.join(m.id))).collect()
    }

    pub fn locate(&self, m: &ModelInfo) -> Location {
        if let Some(own) = self.own_dirs(m).into_iter().find(|d| complete(d, m)) {
            return Location::Own(own);
        }
        if self.reuse {
            for other in &self.others {
                if let Some(dir) = other.candidates(m).into_iter().find(|d| complete(d, m)) {
                    return Location::OtherApp { app: other.app, dir };
                }
            }
        }
        Location::Missing
    }
}

/// Every file present with its exact size. Hashes are checked at download; re-hashing gigabytes on every lookup would
/// be too slow, and a size match on files this large is a strong signal.
pub fn complete(dir: &Path, m: &ModelInfo) -> bool {
    m.files.iter().all(|f| std::fs::metadata(dir.join(f.name)).map(|md| md.is_file() && md.len() == f.size).unwrap_or(false))
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p voice-engine-models && cargo clippy --workspace --all-targets -- -D warnings`
Expected: PASS, no warnings.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "models: ModelStore finder (own, legacy, other apps' folders when reuse is on)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 14: Check the paths on this PC, then tag `v0.1.0`

**Files:**
- Create: `crates/models/examples/where.rs`
- Modify: `CHANGELOG.md`, `README.md`

- [ ] **Step 1: A small program that prints each app's folders on this computer**

`crates/models/examples/where.rs`:
```rust
//! Prints each app's model folders on this computer and whether they exist: `cargo run -p voice-engine-models --example where`.

use voice_engine_models::{App, EngineKind};

fn main() {
    for app in App::ALL {
        for engine in [EngineKind::Whisper, EngineKind::Llm] {
            if let Some(r) = app.root(engine) {
                println!("{app:?} {engine:?} {:?} exists={}", r.root, r.root.exists());
            }
        }
    }
}
```

Run: `cargo run -p voice-engine-models --example where`

- [ ] **Step 2: Compare with the real folders on this PC**

```bash
ls "$LOCALAPPDATA/com.craftconceptsdigital.talaan/models" 2>&1 | head
ls "$LOCALAPPDATA/com.craftconceptsdigital.ensayo/models" "$LOCALAPPDATA/com.craftconceptsdigital.ensayo/models/llm" 2>&1 | head
ls "$LOCALAPPDATA/com.craftconceptsdigital.parlato" 2>&1 | head
```
For every app installed on this PC, its row must print `exists=true` and the folder must contain model folders/files laid out as `Layout` says. An app that is not installed can't be checked: write "not installed on this PC" for it.

Write the result in `C:\Dev\voice-engine-import\paths-check.md`. **If an installed app's folder differs from the table: stop and report** (ground rule 3). The Mac rows are checked by Miguel on the Mac before the first Mac release of an app on the engine; say so in the Notion change log row (Task 15).

- [ ] **Step 3: CHANGELOG and README**

Replace `## Unreleased` in `CHANGELOG.md` with:
```markdown
## v0.1.0 (2026-10-28)

First version, from Talaan's and Ensayo's engine code (see SOURCES.md).

- protocol: Talaan's and Ensayo's helper messages merged. New: `PROTOCOL_VERSION` (1), sent in `Ready`; helpers from
  before the engine read as 0. New: `HelperConfig { env_prefix }` names each app's environment variables
  (`<PREFIX>_GPU`, `<PREFIX>_LLM_IDLE_SECS`); `FAIL_ENV` is gone.
- llm: the local AI helper as `run(HelperConfig)`, with Talaan's GPU choice and embeddings and Ensayo's idle exit.
- stt: Talaan's Whisper chunk code and helper as `run_helper(HelperConfig)`.
- models: speech catalog (Talaan's), local AI catalog (Qwen3.5 4B, 9B), the known apps table, and `ModelStore`.
  Parlato's Windows folder is under %LOCALAPPDATA% (Talaan looked in %APPDATA% before).
```
Add any merge decision from Tasks 5 and 8 that changes behaviour for one app as its own bullet.

- [ ] **Step 4: Commit, push, wait for CI, tag**

```bash
git add -A && git commit -m "v0.1.0: changelog, path check example

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
gh run watch --repo mmnzns/voice-engine --exit-status $(gh run list --repo mmnzns/voice-engine --limit 1 --json databaseId -q '.[0].databaseId')
gh workflow run CI --repo mmnzns/voice-engine --ref main
```
Wait for the manual run (it includes `windows-gpu`) to pass too, then:
```bash
git tag -a v0.1.0 -m "voice-engine v0.1.0" && git push origin v0.1.0
gh release create v0.1.0 --repo mmnzns/voice-engine --title "v0.1.0" --notes-file <(sed -n '/^## v0.1.0/,/^## /p' CHANGELOG.md | sed '$d')
```

---

### Task 15: Notion: decisions, change log, and the app handoff blocks

**Files:** the Notion page "Voice Apps Engine Sync" (`https://app.notion.com/p/4074dc1744e18017a92ffe0752e2282d`). Use the Notion MCP tools: fetch the page first, then update it. Do not delete any existing section.

- [ ] **Step 1: Mark decisions done** under "Decisions for Miguel":
  - "Engine repository name, and public or private" → checked, with "voice-engine, public, MIT (2026-10-28)".
  - "Ensayo's licence" → checked, with "MIT (Ensayo commit f294fc4)".

- [ ] **Step 2: Phase checkboxes**: under "Phase 1: create the shared engine" check the GPL check, the new repository, and the Ensayo merge. Leave "Talaan uses the engine" and "Ensayo uses the engine" unchecked (the app chats check them).

- [ ] **Step 3: Change log row** (newest first, above the 2026-10-28 "none yet" row):

| Date | Engine version | What changed | Parlato | Talaan | Ensayo |
| --- | --- | --- | --- | --- | --- |
| 2026-10-28 | v0.1.0 | First version: protocol (+ version number), llm helper, Whisper (stt), model catalog and finder with the known apps table. Mac folder rows not yet checked on a Mac. | Nothing (Phase 0 build fix only, see its block) | Needs code change: move to v0.1.0, see "Talaan: move to voice-engine v0.1.0" | Needs code change: move to v0.1.0, see "Ensayo: move to voice-engine v0.1.0" |

- [ ] **Step 4: Add a "What each app does next" section** (after "The work, in order") with three blocks. Each block is self-contained: the app's chat has never seen this conversation.

**Block "Parlato: Phase 0 build fix"**
```markdown
Do this in the Parlato repository only.

1. New branch `fix/dev-deps-optimized` from `main`.
2. In `src-tauri/Cargo.toml`, add (the same setting Talaan has; a build setting, not code):
   [profile.dev.package."*"]
   opt-level = 3
3. Build a test build the usual way and check the crash seen on 2026-10-28 is gone.
4. If it is gone: PR, merge, and in the Notion change log write "done in <version>" in Parlato's column of the
   2026-10-28 v0.1.0 row. If it is not gone: debug it in Parlato; it is not an engine problem.
```

**Block "Talaan: move to voice-engine v0.1.0"**
```markdown
Do this in the Talaan repository only. Engine: https://github.com/mmnzns/voice-engine, tag v0.1.0. Read its README
and CHANGELOG first. Never edit engine code from Talaan; if the engine needs a change, open an issue on
mmnzns/voice-engine.

1. New branch `engine/v0.1.0` from `main`.
2. src-tauri/Cargo.toml:
   - remove `"llm-protocol"` from `[workspace] members` (keep `"llm"` and `"stt"`: they become 3-line helper programs);
   - replace `talaan-llm-protocol = { path = "llm-protocol" }` and `talaan-stt = { path = "stt" }` with
     voice-engine-protocol = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
     voice-engine-stt      = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
     voice-engine-models   = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
3. Delete src-tauri/llm-protocol/. Replace src-tauri/llm/src/main.rs with
     fn main() { voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "TALAAN" }); }
   and make src-tauri/llm/Cargo.toml depend only on voice-engine-llm and voice-engine-protocol at the same tag, with
   `[features] vulkan = ["voice-engine-llm/vulkan"]`.
4. src-tauri/stt/: keep the folder as the helper program only. src/main.rs becomes
     fn main() { voice_engine_stt::run_helper(voice_engine_protocol::HelperConfig { env_prefix: "TALAAN" }); }
   delete src/lib.rs, add `[[bin]] name = "talaan-stt"` if it is not there, depend on voice-engine-stt and
   voice-engine-protocol, `[features] vulkan = ["voice-engine-stt/vulkan"]`.
5. Rename imports: talaan_llm_protocol → voice_engine_protocol, talaan_stt → voice_engine_stt (commands/gpu.rs,
   logging.rs, summarizer/engines.rs, summarizer/local.rs, transcriber/engine.rs, transcriber/fallback.rs,
   transcriber/remote.rs). FAIL_ENV → HelperConfig { env_prefix: "TALAAN" }.gpu_env().
6. Event::Ready / SttEvent::Ready have a new `protocol` field. Where Talaan reads Ready, compare it with
   voice_engine_protocol::PROTOCOL_VERSION; if different, log it and show the helper as failed with
   "The AI helper doesn't match this version of Talaan. Reinstall Talaan."
7. transcriber/models.rs: keep only what is Talaan's (MODELS_ROOT, set_models_root, local_models_root, models_root,
   the preview test). Use voice_engine_models::{catalog, ModelStore, Location, App}. Build stores as
     ModelStore::new(models_root(data_dir)).with_legacy(Some(data_dir.join("models")))
       .with_others(App::Parlato.root(engine).into_iter().collect(), reuse)
   (only Parlato, as today; this also fixes the %APPDATA% / %LOCALAPPDATA% bug for Parlato's folder).
   Location::OnThisMac becomes Location::OtherApp { .. }.
8. summarizer/models.rs: use voice_engine_models::llm_catalog for the model files; keep Talaan's choices and text.
9. Add `src-tauri/.cargo/config.toml` to .gitignore.
10. Add this to CLAUDE.md:
    <the CLAUDE.md snippet from the spec, section 8.5>
11. Tests: cargo test --workspace, npm test, the normal Windows build script (helpers must build and bundle).
    Manual: record a short meeting, get a transcript and a local summary. On a PC with Parlato installed and
    Whisper downloaded there, Talaan finds Parlato's copy. Release notes line: "Talaan now finds speech models
    Parlato already downloaded on Windows."
12. PR, merge, release. In the Notion change log row v0.1.0, Talaan column: "done in <version>". Check "Talaan uses
    the engine instead of its own copy" under Phase 1.
```

**Block "Ensayo: move to voice-engine v0.1.0"**
```markdown
Do this in the Ensayo repository only. Engine: https://github.com/mmnzns/voice-engine, tag v0.1.0. Read its README
and CHANGELOG first. Never edit engine code from Ensayo; if the engine needs a change, open an issue on
mmnzns/voice-engine.

1. New branch `engine/v0.1.0` from `main`.
2. src-tauri/Cargo.toml: remove `"llm-protocol"` from `[workspace] members`; replace
   `ensayo-llm-protocol = { path = "llm-protocol" }` with
     voice-engine-protocol = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
     voice-engine-models   = { git = "https://github.com/mmnzns/voice-engine", tag = "v0.1.0" }
3. Delete src-tauri/llm-protocol/ and src-tauri/llm/tests/ (the tests now live in the engine). Replace
   src-tauri/llm/src/main.rs with
     fn main() { voice_engine_llm::run(voice_engine_protocol::HelperConfig { env_prefix: "ENSAYO" }); }
   and make src-tauri/llm/Cargo.toml depend only on voice-engine-llm and voice-engine-protocol at the same tag.
   ENSAYO_LLM_IDLE_SECS keeps working (HelperConfig builds the same name).
4. Rename imports: ensayo_llm_protocol → voice_engine_protocol (ai/local.rs).
5. Event::Ready has a new `protocol` field. Where Ensayo reads Ready, compare it with
   voice_engine_protocol::PROTOCOL_VERSION; if different, log it and show the local AI as failed with
   "The AI helper doesn't match this version of Ensayo. Reinstall Ensayo."
6. speech/models.rs: use voice_engine_models::{catalog, ModelStore, Location, App} instead of the local copies. Build
   stores as ModelStore::new(root).with_others(App::Talaan.root(engine).into_iter().collect(), reuse) (only Talaan, as
   today). Location::OnThisComputer becomes Location::OtherApp { .. }.
7. ai/local_models.rs: LocalModel keeps `licence` and `recommended`; its `info` comes from
   voice_engine_models::llm_catalog::find(id). Stores use root models/llm and App::Talaan.root(EngineKind::Llm).
8. Add `src-tauri/.cargo/config.toml` to .gitignore.
9. Add this to CLAUDE.md:
    <the CLAUDE.md snippet from the spec, section 8.5>
10. Tests: cargo test --workspace, npm test, the normal Windows build script. Manual: answer one practice question
    out loud, get a transcript and a local AI score; with Talaan installed, Ensayo still finds Talaan's models.
11. PR, merge, release. In the Notion change log row v0.1.0, Ensayo column: "done in <version>". Check "Ensayo uses
    the engine instead of its own copy" under Phase 1.
```

When writing the blocks, paste the actual CLAUDE.md snippet text from spec section 8.5 where the placeholder line is, and update any step that Tasks 5, 8, 11 or 14 changed (for example a new `HelperConfig` method, or a catalog difference Miguel decided).

- [ ] **Step 5: Update the "How to start the new chat" section** with the three one-line prompts:
  - Parlato chat: `Read the Notion page "Voice Apps Engine Sync", section "Parlato: Phase 0 build fix", and do it.`
  - Talaan chat: `Read the Notion page "Voice Apps Engine Sync", section "Talaan: move to voice-engine v0.1.0", and do it.`
  - Ensayo chat: `Read the Notion page "Voice Apps Engine Sync", section "Ensayo: move to voice-engine v0.1.0", and do it.`

- [ ] **Step 6: Update the engine README table** to "v0.1.0 (moving)" for Talaan and Ensayo, commit and push:

```bash
git add README.md && git commit -m "README: Talaan and Ensayo are moving to v0.1.0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push origin main
```

- [ ] **Step 7: Report to Miguel**: the repo link, the tag, the three reports (`gpl-check.md`, `catalog-compare.md`, `paths-check.md`) in short, any merge decision that changes behaviour, and the three one-line prompts to paste.

---

## Spec coverage

| Spec section | Task |
| --- | --- |
| 1 Done criteria 1 (repo, tag, CI) | 2, 6, 14 |
| 1 Done criteria 2–3 (apps moved) | 15 (handoff blocks; the app chats do the work) |
| 1 Done criteria 4 (Notion) | 15 |
| 1 Done criteria 5 / 9 (Parlato Phase 0) | 15 Step 4, Parlato block |
| 3 Ownership rule | Ground rules, 1, 12 doc comment |
| 4.1 protocol, PROTOCOL_VERSION | 3, 4, 5 |
| 4.2 llm | 7, 8 |
| 4.3 stt | 9 |
| 4.4 catalog, finder, known apps, Talaan Windows bug | 10, 11, 12, 13, 15 Talaan block step 7 |
| 5 How an app uses the engine, local override | 2 README, 15 blocks |
| 6 Versions and change log | 14, 15 |
| 7 Errors and edge cases | 4 (version), 13 (read-only, disappears) |
| 8 Chats, CLAUDE.md files | 2 Step 4, 15 |
| 10 Order of work | Tasks follow it |
| 11 Testing | 6, every task, 15 manual checks |
