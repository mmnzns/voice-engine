//! Parakeet on a real model, on the Mac: `cargo test -p voice-engine-stt --test real_parakeet -- --ignored
//! --test-threads=1`.
//!
//! The model is `parakeet-ultra` from `$VOICE_ENGINE_TEST_MODELS` (default: Talaan's models folder,
//! `~/Library/Application Support/Talaan/models`), only ever read: the only ONNX speech model on this Mac. Parakeet
//! Unified and Nemotron are covered by `parakeet_api.rs` (pure tests and folders that are wrong) only. Speech is
//! made with the macOS `say` command (see `common`). Every test goes through the same lock as the Whisper ones.

#![cfg(target_os = "macos")]

mod common;

use common::{SENTENCE, check_shape, guard, lower_letters, models_dir, ms, say, text_of, words_of};
use std::path::PathBuf;
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::parakeet_engine::ParakeetEngine;
use voice_engine_stt::threads::{ThreadKind, default_threads, logical_cores};
use voice_engine_stt::words::clean;
use voice_engine_stt::{
    ChunkOptions, LoadOptions, SpeechError, Timing, Transcriber, Transcript, Word, load,
};

const ID: &str = "parakeet-ultra";

fn model_dir() -> PathBuf {
    let dir = models_dir().join(ID);
    assert!(
        dir.join("encoder-model.int8.onnx").is_file(),
        "the test model is not at {}",
        dir.display()
    );
    dir
}

fn engine(opts: LoadOptions) -> ParakeetEngine {
    ParakeetEngine::load(find(ID).unwrap(), &model_dir(), opts).unwrap()
}

fn options(timing: Timing) -> ChunkOptions {
    ChunkOptions {
        timing,
        ..ChunkOptions::default()
    }
}

/// `secs` seconds of speech: a paragraph said over and over with 0.3 s between, cut to the length.
fn speech(secs: f64) -> Vec<f32> {
    let paragraph = say(&format!(
        "{SENTENCE} Parakeet hears the next sentence as well. Numbers like twenty two and names like Ottawa are \
         spoken in turn. The meeting starts at nine and ends before lunch, and nobody minds."
    ));
    let want = (secs * 16_000.0) as usize;
    let mut out: Vec<f32> = vec![];
    while out.len() < want {
        out.extend_from_slice(&paragraph);
        out.extend(std::iter::repeat_n(0.0, 4_800));
    }
    out.truncate(want);
    out
}

fn flat(t: &Transcript) -> Vec<(String, u64, u64)> {
    t.segments
        .iter()
        .flat_map(|s| {
            s.words
                .iter()
                .map(|w| (w.text.clone(), w.start_ms, w.end_ms))
        })
        .collect()
}

#[test]
#[ignore]
fn both_modes_give_segments_and_ordered_timed_words() {
    let _g = guard();
    let clip = say(SENTENCE);
    eprintln!("clip: {} ms", ms(&clip));
    let mut e = engine(LoadOptions::default());
    for timing in [Timing::Segments, Timing::Words] {
        let t0 = Instant::now();
        let t = e.transcribe(&clip, &options(timing)).unwrap();
        eprintln!(
            "{timing:?}: {:?} in {:?}, language {:?}, stopped early {:?}",
            text_of(&t),
            t0.elapsed(),
            t.language,
            t.stopped_early_at_ms
        );
        for s in &t.segments {
            eprintln!(
                "   [{}..{}] {:?} words {:?}",
                s.start_ms,
                s.end_ms,
                s.text,
                s.words
                    .iter()
                    .map(|w| (w.text.as_str(), w.start_ms, w.end_ms))
                    .collect::<Vec<_>>()
            );
        }
        check_shape(&t, ms(&clip));
        let heard = lower_letters(&words_of(&t).join(" "));
        assert!(heard.contains("speechengine"), "heard {heard:?}");
        // Parakeet TDT reports no language
        assert_eq!(t.language, None);
        match timing {
            Timing::Segments => {
                // Ultra punctuates this as one sentence ("Hello there, this is ..."): one segment with all words
                assert!(
                    t.segments[0].text.starts_with("Hello there"),
                    "{:?}",
                    t.segments[0]
                );
                assert!(t.segments[0].words.len() >= 10, "{:?}", t.segments[0]);
            }
            Timing::Words => {
                assert!(t.segments.len() >= 10, "{} segments", t.segments.len());
                assert!(
                    t.segments
                        .iter()
                        .all(|s| s.words.len() == 1 && s.words[0].text == s.text)
                );
            }
        }
    }
}

/// parakeet-rs run directly (Talaan's and Ensayo's calls, copied: half a second of silence each side, sentence mode
/// for Talaan, word mode and `clean` for Ensayo) on the same audio: the engine must give exactly those.
/// Short sentences with pauses, so the model ends them.
fn sentences_clip() -> Vec<f32> {
    say(
        "Is it working? [[slnc 700]] Yes, it is working. [[slnc 700]] Are you sure? [[slnc 700]] I am sure. [[slnc 700]] Good.",
    )
}

#[test]
#[ignore]
fn the_engine_gives_what_talaan_and_ensayo_get_from_parakeet_rs() {
    let _g = guard();
    // a run-on paragraph the model punctuates with commas (two sentences), and five short ones
    for clip in [speech(14.0), sentences_clip()] {
        compare_with_parakeet_rs(&clip);
    }
}

fn compare_with_parakeet_rs(clip: &[f32]) {
    use parakeet_rs::Transcriber as _;
    let threads = 4;
    let mut ours = engine(LoadOptions {
        gpu: false,
        threads: Some(threads),
    });
    let segments = ours.transcribe(clip, &options(Timing::Segments)).unwrap();
    let words = ours.transcribe(clip, &options(Timing::Words)).unwrap();
    assert_eq!(
        segments.stopped_early_at_ms, None,
        "the comparison needs one pass"
    );
    assert_eq!(words.stopped_early_at_ms, None);

    let config = parakeet_rs::ExecutionConfig::new().with_intra_threads(threads);
    let mut theirs = parakeet_rs::ParakeetTDT::from_pretrained(model_dir(), Some(config)).unwrap();
    let len_ms = ms(clip);
    // Talaan's and Ensayo's own code, copied: half a second of silence each side, seconds to whole milliseconds with
    // the padding taken off, clamped to the chunk.
    let padded = || {
        let mut audio = vec![0.0f32; 8_000];
        audio.extend_from_slice(clip);
        audio.extend(std::iter::repeat_n(0.0, 8_000));
        audio
    };
    let shift = |t: f32| {
        ((t.max(0.0) * 1000.0) as u64)
            .saturating_sub(500)
            .min(len_ms)
    };
    // Talaan: sentence mode, pieces shifted back by the padding and clamped, noise dropped
    let sentences = theirs
        .transcribe_samples(
            padded(),
            16_000,
            1,
            Some(parakeet_rs::TimestampMode::Sentences),
        )
        .unwrap();
    let want_segments: Vec<(String, u64, u64)> = sentences
        .tokens
        .into_iter()
        .filter(|t| !voice_engine_stt::is_noise(&t.text))
        .map(|t| {
            (
                t.text.trim().to_string(),
                shift(t.start),
                shift(t.end.max(t.start)),
            )
        })
        .collect();
    let got_segments: Vec<(String, u64, u64)> = segments
        .segments
        .iter()
        .map(|s| (s.text.clone(), s.start_ms, s.end_ms))
        .collect();
    eprintln!("talaan {want_segments:?}\nours   {got_segments:?}");
    assert!(!want_segments.is_empty());
    assert_eq!(got_segments, want_segments);

    // Ensayo: word mode, shifted and clamped, then `clean`
    let timed = theirs
        .transcribe_samples(padded(), 16_000, 1, Some(parakeet_rs::TimestampMode::Words))
        .unwrap();
    let want_words: Vec<Word> = clean(
        timed
            .tokens
            .into_iter()
            .map(|t| Word {
                text: t.text,
                start_ms: shift(t.start),
                end_ms: shift(t.end.max(t.start)),
            })
            .collect(),
    );
    let got_words: Vec<Word> = words
        .segments
        .iter()
        .flat_map(|s| s.words.clone())
        .collect();
    eprintln!(
        "ensayo {} words, ours {} words",
        want_words.len(),
        got_words.len()
    );
    assert!(want_words.len() >= 10);
    assert_eq!(got_words, want_words);
    // and the words inside the sentences are the same words
    let in_sentences: Vec<&str> = segments
        .segments
        .iter()
        .flat_map(|s| s.words.iter().map(|w| w.text.as_str()))
        .collect();
    let in_words: Vec<&str> = got_words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(in_sentences, in_words);
}

/// Short sentences with pauses, so the model ends them: `Timing::Segments` must give one segment per sentence the
/// model wrote (parakeet-rs's rule: a word with . ? or ! ends it), each with its own words, and the words of both
/// modes must be the same.
#[test]
#[ignore]
fn sentences_come_out_as_segments_with_their_words() {
    let _g = guard();
    let clip = sentences_clip();
    let mut e = engine(LoadOptions::default());
    let segments = e.transcribe(&clip, &options(Timing::Segments)).unwrap();
    let words = e.transcribe(&clip, &options(Timing::Words)).unwrap();
    for s in &segments.segments {
        eprintln!("[{}..{}] {:?} {:?}", s.start_ms, s.end_ms, s.text, s.words);
    }
    check_shape(&segments, ms(&clip));
    check_shape(&words, ms(&clip));
    assert!(segments.segments.len() >= 3, "{:?}", text_of(&segments));
    let enders = |s: &str| s.contains(['.', '?', '!']);
    let (last, rest) = segments.segments.split_last().unwrap();
    for s in rest {
        assert!(
            enders(s.words.last().map_or("", |w| w.text.as_str())),
            "{s:?}"
        );
        assert!(
            s.words[..s.words.len() - 1]
                .iter()
                .all(|w| !enders(&w.text)),
            "a sentence ends inside {s:?}"
        );
    }
    assert!(!last.words.is_empty());
    assert_eq!(flat(&segments), flat(&words));
    // the sentence's own times are its first word's start and its last word's end
    for s in &segments.segments {
        assert_eq!(s.start_ms, s.words[0].start_ms);
        assert_eq!(s.end_ms, s.words.last().unwrap().end_ms);
    }
}

#[test]
#[ignore]
fn a_15_second_window_returns_timed_words_across_it() {
    let _g = guard();
    let clip = speech(15.0);
    assert_eq!(ms(&clip), 15_000);
    let mut e = engine(LoadOptions::default());
    for timing in [Timing::Segments, Timing::Words] {
        let t0 = Instant::now();
        let t = e.transcribe(&clip, &options(timing)).unwrap();
        let words = flat(&t);
        eprintln!(
            "{timing:?}: {} words in {:?}, first {:?}, last {:?}, stopped early {:?}",
            words.len(),
            t0.elapsed(),
            words.first(),
            words.last(),
            t.stopped_early_at_ms
        );
        check_shape(&t, 15_000);
        assert!(words.len() >= 25, "{} words in 15 s of speech", words.len());
        assert!(
            words.last().unwrap().2 >= 11_000,
            "speech stops at {:?}",
            words.last()
        );
    }
}

#[test]
#[ignore]
fn a_40_second_chunk_works_and_a_45_second_one_is_too_long() {
    let _g = guard();
    let mut e = engine(LoadOptions::default());
    assert_eq!(e.max_secs(), 40);
    let clip = speech(40.0);
    assert_eq!(clip.len(), 640_000);
    let t0 = Instant::now();
    let t = e.transcribe(&clip, &options(Timing::Words)).unwrap();
    let words = flat(&t);
    eprintln!(
        "40 s: {} words in {:?}, last {:?}, stopped early {:?}",
        words.len(),
        t0.elapsed(),
        words.last(),
        t.stopped_early_at_ms
    );
    check_shape(&t, 40_000);
    assert!(words.len() >= 70);
    assert!(words.last().unwrap().2 >= 30_000, "{:?}", words.last());

    let long = speech(45.0);
    let t0 = Instant::now();
    let err = e.transcribe(&long, &options(Timing::Words)).unwrap_err();
    eprintln!("45 s: {err} (after {:?}, no decode)", t0.elapsed());
    assert_eq!(err, SpeechError::TooLong { max_secs: 40 });
    assert!(
        t0.elapsed().as_millis() < 200,
        "the refusal must not decode anything"
    );
    // one sample over 40 s is already too long, and the engine still works afterwards
    let over = vec![0.0f32; 640_001];
    assert_eq!(
        e.transcribe(&over, &options(Timing::Segments)).unwrap_err(),
        SpeechError::TooLong { max_secs: 40 }
    );
    assert!(
        !e.transcribe(&speech(5.0), &options(Timing::Words))
            .unwrap()
            .segments
            .is_empty()
    );
}

#[test]
#[ignore]
fn clips_of_every_short_length_work() {
    let _g = guard();
    let clip = say(SENTENCE);
    let mut e = engine(LoadOptions::default());
    for n in [0usize, 1_600, 6_400, 15_999, 16_000, 16_001, 20_000] {
        for timing in [Timing::Segments, Timing::Words] {
            let part = &clip[..n.min(clip.len())];
            let t = e
                .transcribe(part, &options(timing))
                .unwrap_or_else(|e| panic!("{n} samples {timing:?}: {e}"));
            let clip_ms = ms(part);
            for s in &t.segments {
                assert!(s.end_ms <= clip_ms, "{n} samples: {s:?}");
                assert!(
                    s.words
                        .iter()
                        .all(|w| w.end_ms <= clip_ms && w.start_ms <= w.end_ms),
                    "{n} samples: {s:?}"
                );
            }
            eprintln!(
                "{n} samples ({clip_ms} ms) {timing:?}: {} segments, text {:?}",
                t.segments.len(),
                text_of(&t)
            );
            if n >= 20_000 {
                assert!(
                    lower_letters(&text_of(&t)).contains("hello"),
                    "{n} samples {timing:?}"
                );
            }
        }
    }
    // 0.4 s of silence: nothing said, no error
    let t = e
        .transcribe(&vec![0.0; 6_400], &ChunkOptions::default())
        .unwrap();
    assert!(t.segments.iter().all(|s| s.end_ms <= 400));
    // no audio at all
    assert!(
        e.transcribe(&[], &ChunkOptions::default())
            .unwrap()
            .segments
            .is_empty()
    );
}

#[test]
#[ignore]
fn the_gpu_option_runs_on_the_processor_and_says_so_on_the_mac() {
    let _g = guard();
    let m = find(ID).unwrap();
    let with = load(
        m,
        &model_dir(),
        LoadOptions {
            gpu: true,
            threads: None,
        },
    )
    .unwrap();
    let b = with.backend();
    eprintln!("gpu: true -> {b:?}");
    assert_eq!(
        (b.backend.as_str(), b.fell_back, b.device.as_deref()),
        ("cpu", true, None)
    );
    let without = load(
        m,
        &model_dir(),
        LoadOptions {
            gpu: false,
            threads: None,
        },
    )
    .unwrap();
    let b = without.backend();
    eprintln!("gpu: false -> {b:?}");
    assert_eq!(
        (b.backend.as_str(), b.fell_back, b.device.as_deref()),
        ("cpu", false, None)
    );
}

#[test]
#[ignore]
fn threads_follow_the_option_else_the_processor_rule() {
    let _g = guard();
    let logical = logical_cores();
    let clip = say(SENTENCE);
    let processor = default_threads(ThreadKind::Processor, logical);
    for (opts, want) in [
        (
            LoadOptions {
                gpu: false,
                threads: Some(3),
            },
            3,
        ),
        (
            LoadOptions {
                gpu: true,
                threads: Some(5),
            },
            5,
        ),
        (
            LoadOptions {
                gpu: false,
                threads: Some(0),
            },
            1,
        ),
        // no GPU path on the Mac: the processor's rule even with `gpu: true`
        (
            LoadOptions {
                gpu: true,
                threads: None,
            },
            processor,
        ),
        (
            LoadOptions {
                gpu: false,
                threads: None,
            },
            processor,
        ),
    ] {
        let mut e = engine(opts);
        eprintln!("{opts:?}: {} threads", e.threads());
        assert_eq!(e.threads(), want, "{opts:?}");
        // and a model built with that many threads decodes
        let t = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
        assert!(lower_letters(&text_of(&t)).contains("speechengine"));
    }
}

#[test]
#[ignore]
fn fillers_vocabulary_and_language_hints_change_nothing_for_parakeet() {
    let _g = guard();
    let clip = say(SENTENCE);
    let mut e = engine(LoadOptions::default());
    let plain = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    let with = e
        .transcribe(
            &clip,
            &ChunkOptions {
                language: Some("fr".into()),
                timing: Timing::Words,
                keep_fillers: true,
                vocabulary: vec!["Zorblax".into(), "Ottawa".into()],
            },
        )
        .unwrap();
    assert_eq!(plain, with);
    assert_eq!(with.language, None);
}

#[test]
#[ignore]
fn load_through_the_api_gives_a_transcriber_that_can_be_dropped_and_loaded_again() {
    let _g = guard();
    let clip = say(SENTENCE);
    for _ in 0..2 {
        let mut e = load(find(ID).unwrap(), &model_dir(), LoadOptions::default()).unwrap();
        let t = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
        check_shape(&t, ms(&clip));
    }
}
