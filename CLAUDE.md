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
