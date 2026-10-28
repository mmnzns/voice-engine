# Changelog

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
