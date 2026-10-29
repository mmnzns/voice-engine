//! `SharedStore::open` honours `VOICE_ENGINE_MODELS_DIR`. This is the only test in this binary, so setting an
//! environment variable cannot race with another test. It never opens the real shared model folder: the variable is
//! set before `open` runs.

use voice_engine_models::{App, SharedStore};

#[test]
fn open_uses_voice_engine_models_dir_like_open_at() {
    let tmp = tempfile::tempdir().unwrap();
    let dir = tmp.path().join("models-under-test");
    // SAFETY: the only test in this binary, so no other thread reads or writes the environment.
    unsafe { std::env::set_var("VOICE_ENGINE_MODELS_DIR", &dir) };
    let store = SharedStore::open(App::Ensayo).unwrap();
    assert_eq!(store.root(), dir);
    assert!(dir.join(".engine/users").is_dir());
    assert!(dir.join(".engine/locks").is_dir());
    assert_ne!(Some(store.root().to_path_buf()), SharedStore::default_dir());
}
