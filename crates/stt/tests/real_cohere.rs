//! Cohere Transcribe on the real model, on a Mac or a Windows PC: `cargo test -p voice-engine-stt --test real_cohere --
//! --ignored --test-threads=1 --nocapture`.
//!
//! The model is `cohere-transcribe`, looked up on its own: in `$VOICE_ENGINE_TEST_MODELS` when that is set, else in the
//! shared folder, then in Talaan's folder (`common`), only ever read. The files sit flat in the folder (the catalog's
//! names), so loading it also proves that ONNX Runtime finds the `.onnx_data` files by their relative names there.
//! Speech comes from `$VOICE_ENGINE_TEST_CLIP` (a 16 kHz mono WAV that says `SENTENCE`) when that is set, else, on a
//! Mac, from the macOS `say` command (see `common`). The test of numbers needs a voice speaking its own words (`say`
//! on a Mac, the Windows voices on Windows). Prints counts, sizes, times and the backend, never text. Every test goes
//! through the same lock as the other real tests.

mod common;

use common::{SENTENCE, guard, lower_letters, model_dir, ms, say, text_of};
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_protocol::Fallback;
use voice_engine_stt::cohere_engine::CohereEngine;
use voice_engine_stt::threads::{ThreadKind, default_threads, logical_cores};
use voice_engine_stt::{ChunkOptions, LoadOptions, SpeechError, Transcriber, Transcript, load};

const ID: &str = "cohere-transcribe";

fn engine(opts: LoadOptions) -> CohereEngine {
    CohereEngine::load(find(ID).unwrap(), &model_dir(ID), opts).unwrap()
}

fn on_cpu() -> LoadOptions {
    LoadOptions {
        gpu: false,
        threads: None,
    }
}

fn with_language(language: &str) -> ChunkOptions {
    ChunkOptions {
        language: Some(language.into()),
        ..ChunkOptions::default()
    }
}

/// `secs` seconds of speech. When a voice that speaks its own words is here (`say`, the Windows voices) a paragraph of
/// different sentences, cut to the length; otherwise the 12 s bench clip repeated (Cohere stops writing when a
/// clip repeats, so the text is shorter and the time too: the test says which one it used).
fn speech(secs: f64) -> (Vec<f32>, bool) {
    const PARAGRAPH: &str = "The first item on the list is the schedule for next week. We moved the review to         Tuesday morning because the room is free then. After that, the team will walk through the open questions one         by one. Nobody has asked for more time, so the plan stays as it is. The second item is about the budget, which         is close to what we expected at the start of the year. Travel costs are a little higher, and the printing costs         are lower than before. If anything changes, the finance group will send a short note on Friday. The last item is         the visit from the partner company, which is booked for the end of the month. Please check that your calendars         are clear, and tell us early if they are not.";
    let want = (secs * 16_000.0) as usize;
    let (clip, varied) = match voice_clip(PARAGRAPH) {
        Some(clip) => (clip, true),
        None => {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../docs/data/clip-bench.wav");
            (common::support::read_wav(&path).unwrap(), false)
        }
    };
    let mut out: Vec<f32> = vec![];
    while out.len() < want {
        out.extend_from_slice(&clip);
        out.extend(std::iter::repeat_n(0.0, 4_800));
    }
    out.truncate(want);
    (out, varied)
}

fn shape(t: &Transcript, audio_ms: u64) -> String {
    format!(
        "{} segments, {} chars, language {:?}, audio {audio_ms} ms",
        t.segments.len(),
        text_of(t).chars().count(),
        t.language
    )
}

#[test]
#[ignore]
fn loads_flat_on_the_processor_and_says_so() {
    need_clip!();
    let _g = guard();
    let started = Instant::now();
    let e = engine(on_cpu());
    println!(
        "cohere load: {:.1} s, {} threads, languages {}",
        started.elapsed().as_secs_f64(),
        e.threads(),
        e.languages().len()
    );
    let b = e.backend();
    assert_eq!((b.backend.as_str(), b.fallback), ("cpu", None));
    assert_eq!(e.max_secs(), 35);
    assert_eq!(e.languages().len(), 14);
    assert_eq!(
        e.threads(),
        default_threads(ThreadKind::Processor, logical_cores())
    );
    drop(e);
    // With a card asked for: there is no card path, so the processor, with the reason.
    let e = engine(LoadOptions {
        gpu: true,
        threads: Some(3),
    });
    let b = e.backend();
    assert_eq!(b.backend, "cpu");
    assert_eq!(b.fallback, Some(Fallback::NoCardPath));
    assert_eq!(e.threads(), 3);
    // And through the front door, as an app loads it.
    drop(e);
    let boxed = load(find(ID).unwrap(), &model_dir(ID), on_cpu()).unwrap();
    assert_eq!(boxed.backend().backend, "cpu");
}

#[test]
#[ignore]
fn one_segment_over_the_whole_clip_with_no_words() {
    need_clip!();
    let _g = guard();
    let mut e = engine(on_cpu());
    let clip = say(SENTENCE);
    let audio_ms = ms(&clip);
    let started = Instant::now();
    let t = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "cohere clip: {} in {secs:.2} s (real-time factor {:.2}x)",
        shape(&t, audio_ms),
        audio_ms as f64 / 1000.0 / secs
    );
    assert_eq!(t.segments.len(), 1, "{}", shape(&t, audio_ms));
    let s = &t.segments[0];
    assert_eq!((s.start_ms, s.end_ms), (0, audio_ms));
    assert!(s.words.is_empty(), "Cohere gives no word times");
    assert_eq!(s.text, s.text.trim());
    assert!(s.starts_word);
    assert_eq!(t.language.as_deref(), Some("en"));
    assert_eq!(t.stopped_early_at_ms, None);
    // The clip says SENTENCE: the text is about that, which is what this assertion is about.
    let heard = lower_letters(&s.text);
    assert!(heard.contains("hello"), "the clip's first word is missing");
    assert!(
        heard.contains("speechengine"),
        "the clip's last words are missing"
    );
    // Punctuation and capitals are on.
    assert!(s.text.chars().next().is_some_and(char::is_uppercase));
    assert!(s.text.ends_with(['.', '?', '!']));
}

#[test]
#[ignore]
fn thirty_seconds_in_one_call_and_the_real_time_factor() {
    let _g = guard();
    let mut e = engine(on_cpu());
    let (audio, varied) = speech(30.0);
    let audio_ms = ms(&audio);
    // One warm-up call, then the timed one.
    e.transcribe(&audio[..5 * 16_000], &ChunkOptions::default())
        .unwrap();
    let started = Instant::now();
    let t = e.transcribe(&audio, &ChunkOptions::default()).unwrap();
    let secs = started.elapsed().as_secs_f64();
    println!(
        "cohere 30 s: {} in {secs:.2} s on {} threads (real-time factor {:.2}x)",
        shape(&t, audio_ms),
        e.threads(),
        audio_ms as f64 / 1000.0 / secs
    );
    assert_eq!(t.segments.len(), 1);
    assert_eq!(
        (t.segments[0].start_ms, t.segments[0].end_ms),
        (0, audio_ms)
    );
    // The paragraph is about 25 s of different sentences: the text must be long, not a few words. A repeated clip
    // gives less (the model stops on a repeat), so then only the first pass is asked for.
    let chars = text_of(&t).chars().count();
    println!(
        "cohere 30 s used {} speech",
        if varied { "varied" } else { "repeated" }
    );
    assert!(
        chars > if varied { 300 } else { 100 },
        "{}",
        shape(&t, audio_ms)
    );
}

#[test]
#[ignore]
fn audio_longer_than_35_seconds_is_too_long() {
    let _g = guard();
    let mut e = engine(on_cpu());
    let long = vec![0.01f32; 35 * 16_000 + 1];
    assert_eq!(
        e.transcribe(&long, &ChunkOptions::default()).unwrap_err(),
        SpeechError::TooLong { max_secs: 35 }
    );
    // Exactly 35 s is taken (silence: nothing is asserted about what the model writes for it).
    let edge = vec![0.0f32; 35 * 16_000];
    let started = Instant::now();
    let t = e.transcribe(&edge, &ChunkOptions::default()).unwrap();
    println!(
        "cohere 35 s of silence: {} segments in {:.2} s",
        t.segments.len(),
        started.elapsed().as_secs_f64()
    );
}

#[test]
#[ignore]
fn the_language_hint_is_used_and_an_unknown_one_is_refused() {
    need_clip!();
    let _g = guard();
    let mut e = engine(on_cpu());
    let clip = say(SENTENCE);
    for (hint, language) in [("en-US", "en"), ("EN", "en"), ("fr", "fr"), ("de-DE", "de")] {
        let t = e.transcribe(&clip, &with_language(hint)).unwrap();
        assert_eq!(t.language.as_deref(), Some(language), "{hint}");
        assert_eq!(
            t.segments.len().min(1),
            t.segments.len(),
            "{hint}: one segment at most"
        );
    }
    // `None` and "auto" are English.
    for opts in [ChunkOptions::default(), with_language("auto")] {
        assert_eq!(
            e.transcribe(&clip, &opts).unwrap().language.as_deref(),
            Some("en")
        );
    }
    // Not one of the 14: refused before anything is decoded, with no text in the error.
    for hint in ["tl", "ru", "xx"] {
        let err = e.transcribe(&clip, &with_language(hint)).unwrap_err();
        assert_eq!(err, SpeechError::Unsupported("language".into()), "{hint}");
    }
    // The model still works after a refusal.
    assert_eq!(
        e.transcribe(&clip, &ChunkOptions::default())
            .unwrap()
            .segments
            .len(),
        1
    );
}

#[test]
#[ignore]
fn silence_and_very_short_audio_do_not_fail() {
    let _g = guard();
    let mut e = engine(on_cpu());
    let none = e.transcribe(&[], &ChunkOptions::default()).unwrap();
    assert!(none.segments.is_empty());
    for (what, samples) in [
        ("20 ms", 320),
        ("100 ms", 1_600),
        ("1 s", 16_000),
        ("5 s", 80_000),
    ] {
        let started = Instant::now();
        let t = e
            .transcribe(&vec![0.0f32; samples], &ChunkOptions::default())
            .unwrap_or_else(|err| panic!("{what} of silence: {err}"));
        // The model's card warns it may write text for silence; the voice detection in front of it is the app's.
        // Only the shape is asserted.
        println!(
            "cohere {what} of silence: {} segments in {:.2} s",
            t.segments.len(),
            started.elapsed().as_secs_f64()
        );
        for s in &t.segments {
            assert_eq!((s.start_ms, s.words.len()), (0, 0));
            assert_eq!(s.end_ms, samples as u64 * 1000 / 16_000);
        }
    }
}

#[test]
#[ignore]
fn the_same_audio_gives_the_same_text() {
    need_clip!();
    let _g = guard();
    let mut e = engine(on_cpu());
    let clip = say(SENTENCE);
    let a = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    let b = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    assert!(a == b, "two decodes of the same audio differ");
}

/// Numbers come as digits, as the other engines write them (the model writes digits with punctuation on, and the
/// engine leaves its inverse text normalisation switch off: see `cohere_engine`). Needs a voice that speaks its own
/// words.
#[test]
#[ignore]
fn numbers_come_as_digits() {
    let phrases = [
        "The meeting is at twenty three past nine, and there are forty seven people coming.",
        "Call me at five five five, one two three four, after twenty two minutes.",
    ];
    let _g = guard();
    let mut e = engine(on_cpu());
    for words in phrases {
        let Some(clip) = voice_clip(words) else {
            eprintln!("SKIPPED: no voice that speaks its own words here");
            return;
        };
        let t = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
        let text = text_of(&t);
        let digits = text.chars().filter(char::is_ascii_digit).count();
        println!(
            "cohere numbers: {} chars, {digits} digits",
            text.chars().count()
        );
        assert!(
            digits >= 2,
            "no digits in {} chars of text",
            text.chars().count()
        );
    }
}

/// `text` spoken by a voice that says its own words: `say` on a Mac, the Windows voice on Windows.
fn voice_clip(text: &str) -> Option<Vec<f32>> {
    if common::support::say_available().is_ok() {
        Some(common::say(text))
    } else {
        common::support::sapi_samples("Microsoft Zira Desktop", text).ok()
    }
}
