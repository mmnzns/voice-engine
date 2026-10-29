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
| Talaan | v0.1.0 (moving) |
| Ensayo | v0.1.0 (moving) |
| Parlato | not yet |

## Licence

MIT. Code only flows from this engine into the apps. Nothing from Parlato's GPL-3.0 source comes back in.
