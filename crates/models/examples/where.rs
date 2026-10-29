//! Prints each app's model folders on this computer and whether they exist: `cargo run -p voice-engine-models --example where`.

use voice_engine_models::{App, Kind};

fn main() {
    for app in App::ALL {
        for kind in [Kind::Whisper, Kind::Llm] {
            if let Some(r) = app.root(kind) {
                println!("{app:?} {kind:?} {:?} exists={}", r.root, r.root.exists());
            }
        }
    }
}
