//! Helpers the real-model tests share (`real_whisper.rs`, `real_parakeet.rs`, `real_diarize.rs`, `real_fillers.rs`): one
//! lock so only one test loads a model at a time, where the models are, and where the speech comes from, and small
//! text helpers. Only ever reads the models folders.
//!
//! The models and the clip are found by `support` (`crates/stt/support/mod.rs`, which says how: `VOICE_ENGINE_TEST_MODELS`,
//! the shared folder, Talaan's folder; `VOICE_ENGINE_TEST_CLIP`, a 16 kHz mono WAV, else the macOS `say` command).
//! With `VOICE_ENGINE_TEST_CLIP` set, `say` returns that clip for any text: it must say `SENTENCE`
//! (`docs/data/clip-sentence.wav`). A test whose wording or voices matter starts with `need_say!()` and is skipped (it
//! prints why and passes) when `say` is not what makes the speech; one that only needs some speech starts with
//! `need_clip!()`.

#![allow(dead_code)]

#[path = "../../support/mod.rs"]
pub mod support;

use std::path::PathBuf;
use std::sync::{Mutex, MutexGuard};
use voice_engine_stt::Transcript;

#[allow(unused_imports)]
pub use support::SENTENCE;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub fn guard() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// The folder of model `id` (see `support::find_model_dir`); panics with where it looked when it is not there.
pub fn model_dir(id: &str) -> PathBuf {
    support::find_model_dir(id).unwrap_or_else(|e| panic!("the test model: {e}"))
}

/// Whether the graphics card is asked for in the tests: Metal on the Mac. Elsewhere the tests run on the processor
/// (a runner or a Windows PC without the Vulkan build has no card to ask for).
pub fn use_gpu() -> bool {
    cfg!(target_os = "macos")
}

/// Starts a test that needs some speech: returns (skips the test, saying why) when there is none.
#[macro_export]
macro_rules! need_clip {
    () => {
        if let Err(why) = $crate::common::support::speech_available() {
            eprintln!("SKIPPED: {why}");
            return;
        }
    };
}

/// Starts a test that needs `say` to speak its own wording or voices (not a clip from a file).
#[macro_export]
macro_rules! need_say {
    () => {
        if let Err(why) = $crate::common::support::say_available() {
            eprintln!("SKIPPED: {why}");
            return;
        }
    };
}

/// 16 kHz mono f32 samples of `text`: the clip of `VOICE_ENGINE_TEST_CLIP` when that is set (whatever `text` is),
/// else `text` spoken by the macOS `say` command in its default voice.
pub fn say(text: &str) -> Vec<f32> {
    say_with(None, text)
}

/// The same, in the named voice (`say -v`) when one is given. A named voice is always `say`: a clip from a file has one
/// voice only.
pub fn say_with(voice: Option<&str>, text: &str) -> Vec<f32> {
    if voice.is_none()
        && let Some(clip) = support::clip_from_env()
    {
        return clip.unwrap_or_else(|e| panic!("{e}"));
    }
    support::say_samples(voice, text).unwrap_or_else(|e| panic!("{e}"))
}

pub fn ms(samples: &[f32]) -> u64 {
    samples.len() as u64 * 1000 / 16_000
}

pub fn lower_letters(s: &str) -> String {
    s.to_lowercase()
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect()
}

pub fn words_of(t: &Transcript) -> Vec<String> {
    t.segments
        .iter()
        .flat_map(|s| s.words.iter().map(|w| w.text.clone()))
        .collect()
}

pub fn text_of(t: &Transcript) -> String {
    t.segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Words present, in order, with sane times; segments non-empty; the words inside the clip.
pub fn check_shape(t: &Transcript, clip_ms: u64) {
    assert!(!t.segments.is_empty(), "no segments");
    let mut last_end = 0;
    let all: Vec<&voice_engine_stt::Word> = t.segments.iter().flat_map(|s| &s.words).collect();
    let timed = all.iter().filter(|w| w.end_ms > w.start_ms).count();
    assert!(
        timed * 10 >= all.len() * 8,
        "only {timed} of {} words have a duration: the token times are missing",
        all.len()
    );
    for s in &t.segments {
        assert!(!s.text.trim().is_empty(), "empty segment text");
        assert!(
            s.start_ms <= s.end_ms && s.end_ms <= clip_ms,
            "{s:?} in {clip_ms} ms"
        );
        assert!(!s.words.is_empty(), "segment without words: {s:?}");
        for w in &s.words {
            assert!(!w.text.trim().is_empty());
            assert!(w.start_ms <= w.end_ms, "{w:?}");
            assert!(w.end_ms <= clip_ms, "{w:?} past the {clip_ms} ms clip");
            assert!(
                w.start_ms >= last_end,
                "words out of order at {w:?} (last end {last_end})"
            );
            last_end = w.end_ms;
        }
    }
}
