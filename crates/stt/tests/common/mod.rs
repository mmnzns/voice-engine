//! Helpers the real-model tests share (`real_whisper.rs`, `real_parakeet.rs`, `real_cohere.rs`, `real_multitalker.rs`,
//! `real_diarize.rs`, `real_fillers.rs`): one
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
use voice_engine_stt::{Segment, Transcript, Word};

#[allow(unused_imports)]
pub use support::{EXPECT_CARD_ENV, SENTENCE, expect_card};

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub fn guard() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

/// The folder of model `id` (see `support::find_model_dir`); panics with where it looked when it is not there.
pub fn model_dir(id: &str) -> PathBuf {
    support::find_model_dir(id).unwrap_or_else(|e| panic!("the test model: {e}"))
}

/// Whether the graphics card is asked for in Whisper's tests: Metal on the Mac, and Vulkan in this process when the
/// tests are built with `--features vulkan` (a Windows PC with the Vulkan SDK). Elsewhere the tests run on the
/// processor (a runner or a Windows PC without the Vulkan build has no card to ask for). On a PC with the feature but
/// no usable card, `gpu: true` runs on the processor and says `fallback`; the tests accept that. Parakeet's card tests
/// are in `real_parakeet.rs` (the `directml` feature).
pub fn use_gpu() -> bool {
    cfg!(target_os = "macos") || cfg!(feature = "vulkan")
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

/// Starts a test that needs two different voices speaking their own words: a Mac (`say`) or a Windows PC (the
/// Windows voices). Skipped (it prints why and passes) anywhere else, and when `VOICE_ENGINE_TEST_CLIP` is set: a clip
/// from a file has one voice only. A test that only needs the voices' own wording uses `voices_available`.
#[macro_export]
macro_rules! need_voices {
    () => {
        if let Err(why) = $crate::common::voices_available() {
            eprintln!("SKIPPED: {why}");
            return;
        }
    };
}

/// Whether `voice_pair` can make two voices here.
pub fn voices_available() -> Result<(), String> {
    if support::clip_from_env().is_some() {
        return Err(format!(
            "{} is set, and a clip from a file has one voice only",
            support::CLIP_ENV
        ));
    }
    if cfg!(target_os = "macos") || cfg!(windows) {
        Ok(())
    } else {
        Err("this test needs two voices: the macOS `say` command or the Windows voices".into())
    }
}

/// Two different voices (a woman's and a man's) saying `a` and `b`, 16 kHz mono: Samantha and Daniel on a Mac, Zira
/// and David on Windows. Panics when `voices_available` says no.
pub fn voice_pair(a: &str, b: &str) -> (Vec<f32>, Vec<f32>) {
    if cfg!(target_os = "macos") {
        (
            support::say_samples(Some("Samantha"), a).unwrap_or_else(|e| panic!("{e}")),
            support::say_samples(Some("Daniel"), b).unwrap_or_else(|e| panic!("{e}")),
        )
    } else {
        (
            support::sapi_samples("Microsoft Zira Desktop", a).unwrap_or_else(|e| panic!("{e}")),
            support::sapi_samples("Microsoft David Desktop", b).unwrap_or_else(|e| panic!("{e}")),
        )
    }
}

/// A segment for a failure message: times and counts, never its text.
pub fn seg_info(s: &Segment) -> String {
    format!(
        "segment {}..{} ms, {} words, {} chars",
        s.start_ms,
        s.end_ms,
        s.words.len(),
        s.text.chars().count()
    )
}

/// A word for a failure message: times and length, never its text.
pub fn word_info(w: &Word) -> String {
    format!(
        "word {}..{} ms, {} chars",
        w.start_ms,
        w.end_ms,
        w.text.chars().count()
    )
}

/// `got == want`, and when not, a message with counts and the first place they differ, never the values (they are
/// transcripts).
pub fn assert_same<T: PartialEq>(got: &[T], want: &[T], what: &str) {
    if got == want {
        return;
    }
    let first = got.iter().zip(want).position(|(g, w)| g != w);
    panic!(
        "{what}: {} against {} items, first difference at {first:?}",
        got.len(),
        want.len()
    );
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
            "{} in {clip_ms} ms",
            seg_info(s)
        );
        assert!(
            !s.words.is_empty(),
            "segment without words: {}",
            seg_info(s)
        );
        for w in &s.words {
            assert!(!w.text.trim().is_empty());
            assert!(w.start_ms <= w.end_ms, "{}", word_info(w));
            assert!(
                w.end_ms <= clip_ms,
                "{} past the {clip_ms} ms clip",
                word_info(w)
            );
            assert!(
                w.start_ms >= last_end,
                "words out of order at {} (last end {last_end})",
                word_info(w)
            );
            last_end = w.end_ms;
        }
    }
}
