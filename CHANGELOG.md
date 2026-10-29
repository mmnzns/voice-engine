# Changelog

## Unreleased (v0.2.0)

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
- Known: on a Mac, several helpers starting at the same moment can make one die with SIGBUS inside Apple's Metal
  shader cache (not engine code). An app's client must restart a helper that dies at startup.

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
