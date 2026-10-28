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
