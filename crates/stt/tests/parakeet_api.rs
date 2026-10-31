//! Parakeet, Parakeet Unified and Nemotron tests that need no model: padding, times, sentences from words, the
//! resume loop through a scripted fake model, Nemotron's language prompt, the files a catalog entry names, the choice
//! of provider (DirectML or the processor) with its fallback, and the errors a missing or broken folder gives.
//! The real model (`parakeet-ultra`) is in `real_parakeet.rs`; the other ONNX kinds can only be covered here on a
//! Mac that has not downloaded them.

use std::collections::VecDeque;
use voice_engine_models::catalog::{self, Kind, ModelInfo};
use voice_engine_stt::directml::Adapter;
use voice_engine_stt::parakeet_engine::{
    DEFAULT_MAX_SECS, PAD_SAMPLES, Provider, RawWord, WordModel, check_files,
    load_on_best_provider, nemotron_locale, pad_both_sides, segments_from_words, sentence_text,
    shift_words, transcribe_chunk,
};
use voice_engine_stt::threads::{ThreadKind, default_threads};
use voice_engine_stt::whisper_engine::MAX_PASSES;
use voice_engine_stt::{
    ChunkOptions, Fallback, LoadOptions, Segment, SpeechBackend, SpeechError, Timing, Word, load,
};

fn w(text: &str, a: u64, b: u64) -> Word {
    Word {
        text: text.into(),
        start_ms: a,
        end_ms: b,
    }
}

fn rw(text: &str, a: f32, b: f32) -> RawWord {
    RawWord {
        text: text.into(),
        start: a,
        end: b,
    }
}

fn onnx_models() -> Vec<&'static ModelInfo> {
    catalog::all()
        .iter()
        .filter(|m| {
            matches!(
                m.kind,
                Kind::Parakeet | Kind::ParakeetUnified | Kind::Nemotron
            )
        })
        .collect()
}

fn samples(secs: f64) -> Vec<f32> {
    vec![0.25; (secs * 16_000.0) as usize]
}

/// A model that says what it is told to: one scripted reply per call (seconds on the audio it was given, which
/// is the padded audio), and a record of what it was asked.
#[derive(Default)]
struct Script {
    replies: VecDeque<Vec<RawWord>>,
    /// Length in samples of the audio of every call.
    calls: Vec<usize>,
    /// The first and last sample of every call's audio, to see the padding.
    edges: Vec<(f32, f32)>,
    hints: Vec<Option<String>>,
    language: Option<String>,
    fail: Option<SpeechError>,
}

impl Script {
    fn says(replies: Vec<Vec<RawWord>>) -> Self {
        Script {
            replies: replies.into(),
            ..Script::default()
        }
    }
}

impl WordModel for Script {
    fn words(&mut self, audio: Vec<f32>) -> Result<Vec<RawWord>, SpeechError> {
        self.calls.push(audio.len());
        self.edges
            .push((audio[0], *audio.last().expect("never an empty call")));
        if let Some(e) = self.fail.clone() {
            return Err(e);
        }
        Ok(self.replies.pop_front().unwrap_or_default())
    }

    fn set_language(&mut self, hint: Option<&str>) -> Option<String> {
        self.hints.push(hint.map(str::to_string));
        self.language.clone()
    }
}

// ---- padding and times (Talaan and Ensayo: half a second of silence each side) ----

#[test]
fn half_a_second_of_silence_goes_on_each_side() {
    assert_eq!(PAD_SAMPLES, 8_000);
    let clip = samples(0.4);
    let padded = pad_both_sides(&clip);
    assert_eq!(padded.len(), 8_000 + 6_400 + 8_000);
    assert!(padded[..8_000].iter().all(|s| *s == 0.0));
    assert!(padded[8_000 + 6_400..].iter().all(|s| *s == 0.0));
    assert_eq!(&padded[8_000..8_000 + 6_400], &clip[..]);
    // nothing at all still gives the model a second to look at
    assert_eq!(pad_both_sides(&[]).len(), 16_000);
}

#[test]
fn word_times_lose_the_padding_and_stay_inside_the_audio() {
    // exactly representable seconds, so the millisecond truncation is not in question
    let got = shift_words(
        vec![
            rw("a", 1.5, 2.0),   // 1.0 s .. 1.5 s on the clip
            rw("b", 0.25, 0.5),  // inside the leading padding: 0
            rw("c", 9.5, 11.0),  // past the end of a 10 s clip: clamped
            rw("d", 2.0, 1.5),   // an end before its start is the start
            rw("e", -1.0, -0.5), // never negative
        ],
        10_000,
    );
    assert_eq!(
        got,
        vec![
            w("a", 1_000, 1_500),
            w("b", 0, 0),
            w("c", 9_000, 10_000),
            w("d", 1_500, 1_500),
            w("e", 0, 0),
        ]
    );
}

// ---- sentences and words, both timing modes ----

#[test]
fn sentence_text_spaces_like_parakeet_rs() {
    let t = |words: &[&str]| {
        let ws: Vec<Word> = words.iter().map(|t| w(t, 0, 1)).collect();
        sentence_text(&ws)
    };
    // parakeet-rs's own cases: repetitions kept, punctuation attached, contractions keep their space
    assert_eq!(t(&["uh", "uh", "hello"]), "uh uh hello");
    assert_eq!(t(&["Hello", "world", "."]), "Hello world.");
    assert_eq!(t(&["Wait", "?"]), "Wait?");
    assert_eq!(t(&["like", "100"]), "like 100");
    assert_eq!(t(&["we", "'re"]), "we 're");
    assert_eq!(t(&[]), "");
}

fn two_sentences() -> Vec<Word> {
    vec![
        w("Hello", 0, 400),
        w("world", 400, 800),
        w(".", 800, 850),
        w("How", 1_000, 1_200),
        w("are", 1_200, 1_400),
        w("you", 1_400, 1_600),
        w("?", 1_600, 1_650),
    ]
}

#[test]
fn segments_mode_gives_the_sentences_with_their_words() {
    let got = segments_from_words(two_sentences(), Timing::Segments);
    assert_eq!(
        got,
        vec![
            Segment {
                start_ms: 0,
                end_ms: 850,
                text: "Hello world.".into(),
                starts_word: true,
                words: vec![w("Hello", 0, 400), w("world.", 400, 850)],
            },
            Segment {
                start_ms: 1_000,
                end_ms: 1_650,
                text: "How are you?".into(),
                starts_word: true,
                words: vec![
                    w("How", 1_000, 1_200),
                    w("are", 1_200, 1_400),
                    w("you?", 1_400, 1_650)
                ],
            },
        ]
    );
}

#[test]
fn words_mode_gives_one_word_per_segment() {
    let got = segments_from_words(two_sentences(), Timing::Words);
    let texts: Vec<&str> = got.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["Hello", "world.", "How", "are", "you?"]);
    for s in &got {
        assert_eq!(s.words.len(), 1);
        assert_eq!(s.words[0].text, s.text);
        assert_eq!(
            (s.words[0].start_ms, s.words[0].end_ms),
            (s.start_ms, s.end_ms)
        );
    }
}

#[test]
fn both_modes_carry_the_same_words() {
    let flat =
        |segs: Vec<Segment>| -> Vec<Word> { segs.into_iter().flat_map(|s| s.words).collect() };
    assert_eq!(
        flat(segments_from_words(two_sentences(), Timing::Segments)),
        flat(segments_from_words(two_sentences(), Timing::Words))
    );
}

#[test]
fn a_last_sentence_without_a_full_stop_is_kept() {
    let got = segments_from_words(
        vec![
            w("Done", 0, 300),
            w(".", 300, 320),
            w("and", 400, 500),
            w("then", 500, 700),
        ],
        Timing::Segments,
    );
    assert_eq!(got.len(), 2);
    assert_eq!(got[1].text, "and then");
    assert_eq!((got[1].start_ms, got[1].end_ms), (400, 700));
}

#[test]
fn a_word_with_a_full_stop_inside_ends_the_sentence_as_in_parakeet_rs() {
    // parakeet-rs ends a sentence on any word that contains . ? or !; the engine keeps that rule so the segments
    // are the ones Talaan gets today.
    let got = segments_from_words(
        vec![w("see", 0, 100), w("e.g.", 100, 300), w("this", 300, 500)],
        Timing::Segments,
    );
    let texts: Vec<&str> = got.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["see e.g.", "this"]);
}

#[test]
fn noise_is_dropped_in_both_modes() {
    for timing in [Timing::Segments, Timing::Words] {
        assert!(
            segments_from_words(vec![w("[BLANK_AUDIO]", 0, 500)], timing).is_empty(),
            "{timing:?}"
        );
        assert!(segments_from_words(vec![], timing).is_empty());
    }
}

#[test]
fn times_never_go_backwards_across_sentences() {
    let words = vec![w("one.", 0, 1_000), w("two.", 500, 1_500)];
    for timing in [Timing::Segments, Timing::Words] {
        let got = segments_from_words(words.clone(), timing);
        assert_eq!(got.len(), 2, "{timing:?}");
        assert_eq!(got[1].words[0].start_ms, 1_000, "{timing:?}");
        assert_eq!(got[1].start_ms, 1_000, "{timing:?}");
        assert_eq!(got[1].end_ms, 1_500, "{timing:?}");
    }
}

// ---- Nemotron's language prompt ----

/// A slice of parakeet-rs 0.3.8's PROMPT_DICTIONARY keys (nemotron.rs), in its order: the bare codes, the
/// locales, and the odd aliases.
const NEMOTRON_KEYS: &[&str] = &[
    "ar", "ar-AR", "auto", "de", "de-DE", "en", "en-GB", "en-US", "enGB", "es", "es-ES", "es-US",
    "esES", "fr", "fr-CA", "fr-FR", "it", "it-IT", "ja-JA", "ja-JP", "ko", "ko-KO", "ko-KR", "pt",
    "pt-BR", "pt-PT", "zh-CN", "zh-TW", "zh-ZH",
];

#[test]
fn a_two_letter_language_becomes_the_nemotron_locale() {
    let loc = |h: &str| nemotron_locale(Some(h), NEMOTRON_KEYS);
    assert_eq!(loc("fr").as_deref(), Some("fr-FR"));
    // no `en-EN` in the model: the bare key, which the model maps to the same prompt as en-US
    assert_eq!(loc("en").as_deref(), Some("en"));
    assert_eq!(loc("de").as_deref(), Some("de-DE"));
    assert_eq!(loc("es").as_deref(), Some("es-ES"));
    assert_eq!(loc("pt").as_deref(), Some("pt-PT"));
    // the model's own spelling wins when the hint already is a locale, whatever its case or separator
    assert_eq!(loc("fr-CA").as_deref(), Some("fr-CA"));
    assert_eq!(loc("fr_ca").as_deref(), Some("fr-CA"));
    assert_eq!(loc("PT-br").as_deref(), Some("pt-BR"));
    // whitespace and case
    assert_eq!(loc("  FR ").as_deref(), Some("fr-FR"));
    // a locale this model lacks falls back to the language
    assert_eq!(loc("fr-BE").as_deref(), Some("fr-FR"));
}

#[test]
fn no_hint_an_unknown_language_or_the_english_only_model_means_no_prompt() {
    assert_eq!(nemotron_locale(None, NEMOTRON_KEYS), None);
    assert_eq!(nemotron_locale(Some(""), NEMOTRON_KEYS), None);
    assert_eq!(nemotron_locale(Some("auto"), NEMOTRON_KEYS), None);
    assert_eq!(nemotron_locale(Some("xx"), NEMOTRON_KEYS), None);
    // the English-only variant lists no languages at all
    assert_eq!(nemotron_locale(Some("fr"), &[]), None);
}

// ---- the files a catalog entry names ----

#[test]
fn the_catalog_has_the_onnx_kinds_with_the_chunk_limit() {
    let all = onnx_models();
    assert!(
        all.len() >= 8,
        "{} ONNX speech models in the catalog",
        all.len()
    );
    for kind in [Kind::Parakeet, Kind::ParakeetUnified, Kind::Nemotron] {
        assert!(all.iter().any(|m| m.kind == kind), "{kind:?}");
    }
    for m in all {
        assert_eq!(m.chunk_secs, Some(DEFAULT_MAX_SECS), "{}", m.id);
    }
}

#[test]
fn every_onnx_entry_passes_with_exactly_its_files() {
    for m in onnx_models() {
        let names: Vec<&str> = m.files.iter().map(|f| f.name).collect();
        assert_eq!(check_files(m, &|n| names.contains(&n)), Ok(()), "{}", m.id);
    }
}

#[test]
fn a_missing_file_is_model_missing_whichever_it_is() {
    for m in onnx_models() {
        for gone in m.files {
            let got = check_files(m, &|n| {
                n != gone.name && m.files.iter().any(|f| f.name == n)
            });
            assert_eq!(
                got,
                Err(SpeechError::ModelMissing),
                "{} without {}",
                m.id,
                gone.name
            );
        }
        assert_eq!(check_files(m, &|_| false), Err(SpeechError::ModelMissing));
    }
}

#[test]
fn the_files_parakeet_rs_would_pick_are_the_ones_the_catalog_names() {
    // parakeet-rs 0.3.8 looks for `encoder-model.onnx` before `encoder-model.int8.onnx` (and the decoder the same
    // way) in the folder, whatever the app chose. The compressed entry in a folder that also holds the full-size
    // files would run the full-size model: refused.
    let ultra = catalog::find("parakeet-ultra").unwrap();
    let mut present: Vec<&str> = ultra.files.iter().map(|f| f.name).collect();
    assert_eq!(check_files(ultra, &|n| present.contains(&n)), Ok(()));
    present.push("encoder-model.onnx");
    assert_eq!(
        check_files(ultra, &|n| present.contains(&n)),
        Err(SpeechError::BadModel)
    );
    let mut present: Vec<&str> = ultra.files.iter().map(|f| f.name).collect();
    present.push("decoder_joint-model.onnx");
    assert_eq!(
        check_files(ultra, &|n| present.contains(&n)),
        Err(SpeechError::BadModel)
    );
    // the other way round the full-size file is the first choice anyway
    let full = catalog::find("parakeet-v2-full").unwrap();
    let mut present: Vec<&str> = full.files.iter().map(|f| f.name).collect();
    present.push("encoder-model.int8.onnx");
    present.push("decoder_joint-model.int8.onnx");
    assert_eq!(check_files(full, &|n| present.contains(&n)), Ok(()));
    // Unified: `encoder.onnx` comes first too
    let unified = catalog::find("parakeet-unified-en").unwrap();
    let mut present: Vec<&str> = unified.files.iter().map(|f| f.name).collect();
    present.push("encoder.onnx");
    assert_eq!(
        check_files(unified, &|n| present.contains(&n)),
        Err(SpeechError::BadModel)
    );
}

// ---- transcribing through a scripted model ----

#[test]
fn audio_longer_than_the_chunk_limit_is_too_long_and_nothing_is_decoded() {
    let mut model = Script::default();
    let ok = transcribe_chunk(&mut model, &samples(40.0), &ChunkOptions::default(), 40);
    assert!(ok.is_ok(), "exactly 40 s is allowed: {ok:?}");
    let mut model = Script::default();
    let long = vec![0.1; 40 * 16_000 + 1];
    let err = transcribe_chunk(&mut model, &long, &ChunkOptions::default(), 40).unwrap_err();
    assert_eq!(err, SpeechError::TooLong { max_secs: 40 });
    assert!(
        model.calls.is_empty(),
        "a model call for audio that is too long"
    );
    assert!(model.hints.is_empty());
    let msg = err.to_string();
    assert!(msg.contains("40"), "{msg}");
    // a limit from the catalog entry, not a constant
    let err = transcribe_chunk(
        &mut Script::default(),
        &samples(31.0),
        &ChunkOptions::default(),
        30,
    )
    .unwrap_err();
    assert_eq!(err, SpeechError::TooLong { max_secs: 30 });
}

#[test]
fn the_model_gets_the_clip_with_silence_on_both_sides_and_times_come_back_on_the_clip() {
    // a 3 s clip, the model hears it at 0.5 s: its word at 2.0-2.9 s of the padded audio is 1.5-2.4 s of the clip
    let mut model = Script::says(vec![vec![rw("hello", 2.0, 2.9)]]);
    let clip = samples(3.0);
    let t = transcribe_chunk(&mut model, &clip, &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.calls, vec![48_000 + 16_000]);
    assert_eq!(model.edges[0], (0.0, 0.0), "silence at both ends");
    assert_eq!(t.segments.len(), 1);
    assert_eq!(t.segments[0].words, vec![w("hello", 1_500, 2_400)]);
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn a_clip_under_a_second_works_and_its_times_stay_inside_it() {
    let mut model = Script::says(vec![vec![rw("yes", 0.5, 1.4), rw("indeed", 1.4, 1.9)]]);
    let t = transcribe_chunk(&mut model, &samples(0.4), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.calls, vec![6_400 + 16_000]);
    for s in &t.segments {
        assert!(s.end_ms <= 400 && s.start_ms <= s.end_ms, "{s:?}");
        assert!(s.words.iter().all(|w| w.end_ms <= 400), "{s:?}");
    }
    assert_eq!(t.segments.len(), 2);
}

#[test]
fn no_audio_at_all_is_an_empty_transcript_without_a_model_call() {
    let mut model = Script::default();
    let t = transcribe_chunk(&mut model, &[], &ChunkOptions::default(), 40).unwrap();
    assert!(t.segments.is_empty());
    assert_eq!(t.stopped_early_at_ms, None);
    assert!(model.calls.is_empty());
}

#[test]
fn a_15_second_window_returns_timed_words_across_it() {
    // words every second from 0.5 s to the end, as the model would give for speech all the way
    let words: Vec<RawWord> = (0..15)
        .map(|i| rw("word", 0.5 + i as f32 + 0.1, 0.5 + i as f32 + 0.9))
        .collect();
    let mut model = Script::says(vec![words]);
    let opts = ChunkOptions {
        timing: Timing::Words,
        ..ChunkOptions::default()
    };
    let t = transcribe_chunk(&mut model, &samples(15.0), &opts, 40).unwrap();
    assert_eq!(
        model.calls,
        vec![15 * 16_000 + 16_000],
        "one pass: it reached the end"
    );
    let all: Vec<&Word> = t.segments.iter().flat_map(|s| &s.words).collect();
    assert_eq!(all.len(), 15);
    assert!(all.windows(2).all(|p| p[0].end_ms <= p[1].start_ms));
    assert!(
        all.iter()
            .all(|w| w.start_ms < w.end_ms && w.end_ms <= 15_000)
    );
    assert_eq!(all[14].end_ms, 14_900);
}

#[test]
fn decoding_that_stops_early_is_resumed_from_the_last_word_and_reported() {
    // a 10 s clip. Pass 1 (padded audio, seconds): words end at 3.0 s of the clip: stopped, 7 s left.
    // Pass 2 starts at 3.0 s: the word ends at 3.6 s: still 6.4 s left. Pass 3 starts at 3.6 s and reaches 9.9 s.
    let mut model = Script::says(vec![
        vec![rw("one", 0.7, 1.1), rw("two", 2.9, 3.5)],
        vec![rw("three", 0.6, 1.1)],
        vec![rw("four", 1.5, 6.8)],
    ]);
    let t = transcribe_chunk(&mut model, &samples(10.0), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(
        model.calls,
        vec![160_000 + 16_000, 112_000 + 16_000, 102_400 + 16_000]
    );
    let words: Vec<(String, u64, u64)> = t
        .segments
        .iter()
        .flat_map(|s| {
            s.words
                .iter()
                .map(|w| (w.text.clone(), w.start_ms, w.end_ms))
        })
        .collect();
    assert_eq!(
        words,
        vec![
            ("one".to_string(), 200, 600),
            ("two".to_string(), 2_400, 3_000),
            ("three".to_string(), 3_100, 3_600),
            ("four".to_string(), 4_600, 9_900),
        ]
    );
    assert_eq!(
        t.stopped_early_at_ms,
        Some(3_000),
        "where the first stop was"
    );
}

#[test]
fn a_decode_that_reaches_the_end_is_not_resumed() {
    let mut model = Script::says(vec![vec![rw("all", 0.6, 9.6)]]);
    let t = transcribe_chunk(&mut model, &samples(10.0), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.calls.len(), 1);
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn a_decode_that_finds_nothing_is_not_resumed() {
    // silence: no words, no progress, no second pass
    let mut model = Script::default();
    let t = transcribe_chunk(&mut model, &samples(10.0), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.calls.len(), 1);
    assert!(t.segments.is_empty());
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn there_are_at_most_eight_passes() {
    assert_eq!(MAX_PASSES, 8);
    // every pass says one word over its first second, so there is always speech left to resume
    let replies = (0..20).map(|_| vec![rw("again", 0.5, 1.5)]).collect();
    let mut model = Script::says(replies);
    let t = transcribe_chunk(&mut model, &samples(40.0), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.calls.len(), 8);
    assert_eq!(t.stopped_early_at_ms, Some(1_000));
    let last = t.segments.last().unwrap();
    assert_eq!((last.start_ms, last.end_ms), (7_000, 8_000));
}

#[test]
fn both_timing_modes_return_segments_and_words() {
    for timing in [Timing::Segments, Timing::Words] {
        let mut model = Script::says(vec![vec![
            rw("Hello", 0.5, 0.9),
            rw("world", 0.9, 1.4),
            rw(".", 1.4, 1.5),
        ]]);
        let opts = ChunkOptions {
            timing,
            ..ChunkOptions::default()
        };
        let t = transcribe_chunk(&mut model, &samples(1.5), &opts, 40).unwrap();
        assert!(!t.segments.is_empty(), "{timing:?}");
        assert!(t.segments.iter().all(|s| !s.words.is_empty()), "{timing:?}");
        let flat: Vec<String> = t
            .segments
            .iter()
            .flat_map(|s| s.words.iter().map(|w| w.text.clone()))
            .collect();
        assert_eq!(flat, ["Hello", "world."], "{timing:?}");
    }
    // Segments: one sentence; Words: one segment per word
    let say = |timing| {
        let mut model = Script::says(vec![vec![rw("Hi", 0.5, 0.9), rw("there", 0.9, 1.4)]]);
        let opts = ChunkOptions {
            timing,
            ..ChunkOptions::default()
        };
        transcribe_chunk(&mut model, &samples(1.5), &opts, 40).unwrap()
    };
    assert_eq!(say(Timing::Segments).segments.len(), 1);
    assert_eq!(say(Timing::Words).segments.len(), 2);
}

#[test]
fn the_language_hint_goes_to_the_model_and_what_it_settled_on_comes_back() {
    let mut model = Script::says(vec![vec![rw("bonjour", 0.5, 1.0)]]);
    model.language = Some("fr".into());
    let opts = ChunkOptions {
        language: Some("fr".into()),
        ..ChunkOptions::default()
    };
    let t = transcribe_chunk(&mut model, &samples(1.5), &opts, 40).unwrap();
    assert_eq!(model.hints, vec![Some("fr".to_string())]);
    assert_eq!(t.language.as_deref(), Some("fr"));
    // no hint: the model is told so (a model that kept the last call's language must forget it)
    let mut model = Script::says(vec![vec![rw("hi", 0.5, 1.0)]]);
    let t = transcribe_chunk(&mut model, &samples(1.5), &ChunkOptions::default(), 40).unwrap();
    assert_eq!(model.hints, vec![None]);
    assert_eq!(t.language, None);
}

#[test]
fn the_filler_and_vocabulary_options_change_nothing_for_onnx_models() {
    let run = |keep_fillers, vocabulary: Vec<String>| {
        let mut model = Script::says(vec![vec![rw("um", 0.5, 0.8), rw("okay", 0.8, 1.2)]]);
        let opts = ChunkOptions {
            keep_fillers,
            vocabulary,
            ..ChunkOptions::default()
        };
        let t = transcribe_chunk(&mut model, &samples(1.5), &opts, 40).unwrap();
        (t, model.calls, model.edges)
    };
    assert_eq!(run(false, vec![]), run(true, vec!["Zorblax".into()]));
}

#[test]
fn a_model_error_comes_back_as_it_is() {
    let mut model = Script {
        fail: Some(SpeechError::Internal("onnx runtime".into())),
        ..Script::default()
    };
    assert_eq!(
        transcribe_chunk(&mut model, &samples(2.0), &ChunkOptions::default(), 40).unwrap_err(),
        SpeechError::Internal("onnx runtime".into())
    );
}

// ---- the provider: DirectML on the card, else the processor ----

/// What a load gave (provider, backend, threads) and every (provider, threads) the build closure was asked for.
type LoadResult = Result<(Provider, SpeechBackend, usize), SpeechError>;

/// The PC's discrete card, listed after the Radeon iGPU (so DXGI number 1, not DirectML's default adapter 0).
fn rtx() -> Adapter {
    Adapter {
        index: 1,
        name: "NVIDIA GeForce RTX 5070 Ti".into(),
        integrated: false,
        software: false,
        dedicated_mb: 16303,
        vendor_id: 0x10DE,
    }
}

/// `has_card`: DirectML is in the build and the PC has a discrete card.
fn load_with(
    gpu: bool,
    has_card: bool,
    model_has: bool,
    threads: Option<usize>,
    gpu_works: bool,
) -> (LoadResult, Vec<(Provider, usize)>) {
    let mut tried = vec![];
    let card = has_card.then(rtx);
    let got = load_on_best_provider(gpu, card.as_ref(), model_has, threads, 16, |p, n| {
        tried.push((p, n));
        if matches!(p, Provider::DirectMl(_)) && !gpu_works {
            Err(SpeechError::Internal("provider".into()))
        } else {
            Ok(p)
        }
    })
    .map(|l| (l.value, l.backend, l.threads));
    (got, tried)
}

fn cpu(fallback: Option<Fallback>) -> SpeechBackend {
    SpeechBackend {
        backend: "cpu".into(),
        device: None,
        fallback,
    }
}

#[test]
fn without_the_gpu_option_it_runs_on_the_processor_and_did_not_fall_back() {
    let (got, tried) = load_with(false, true, true, None, true);
    assert_eq!(got, Ok((Provider::Processor, cpu(None), 12)));
    assert_eq!(tried, vec![(Provider::Processor, 12)]);
}

#[test]
fn the_gpu_option_without_a_discrete_card_to_run_directml_on_gives_the_processor_with_no_card_path()
{
    // the Mac, a Windows build without the `directml` feature, or a PC with integrated graphics only: no card to run
    // on and nothing to fall back from, but the app asked
    let (got, tried) = load_with(true, false, true, None, true);
    assert_eq!(
        got,
        Ok((Provider::Processor, cpu(Some(Fallback::NoCardPath)), 12))
    );
    assert_eq!(
        tried,
        vec![(Provider::Processor, 12)],
        "DirectML must not even be tried"
    );
}

#[test]
fn a_model_the_catalog_says_has_no_directml_path_runs_on_the_processor_with_no_card_path() {
    let (got, tried) = load_with(true, true, false, None, true);
    assert_eq!(
        got,
        Ok((Provider::Processor, cpu(Some(Fallback::NoCardPath)), 12))
    );
    assert_eq!(tried, vec![(Provider::Processor, 12)]);
}

#[test]
fn directml_that_starts_is_reported_and_takes_the_gpu_thread_rule() {
    let (got, tried) = load_with(true, true, true, None, true);
    let (provider, backend, threads) = got.unwrap();
    // The RTX's DXGI number is what DirectML is given, and its name is what the app is told.
    assert_eq!(provider, Provider::DirectMl(1));
    assert_eq!(
        backend,
        SpeechBackend {
            backend: "directml".into(),
            device: Some("NVIDIA GeForce RTX 5070 Ti".into()),
            fallback: None
        }
    );
    assert_eq!(threads, default_threads(ThreadKind::Gpu, 16));
    assert_eq!(threads, 8);
    assert_eq!(tried, vec![(Provider::DirectMl(1), 8)]);
}

#[test]
fn directml_that_fails_to_start_loads_on_the_processor_and_says_so() {
    let (got, tried) = load_with(true, true, true, None, false);
    assert_eq!(
        got,
        Ok((Provider::Processor, cpu(Some(Fallback::CardFailed)), 12))
    );
    // the processor's own thread rule after the fallback, not the card's
    assert_eq!(
        tried,
        vec![(Provider::DirectMl(1), 8), (Provider::Processor, 12)]
    );
}

#[test]
fn directml_that_runs_out_of_memory_says_so() {
    let got = load_on_best_provider(true, Some(&rtx()), true, None, 16, |p, _| match p {
        Provider::DirectMl(_) => Err(SpeechError::OutOfMemory),
        Provider::Processor => Ok(()),
    })
    .unwrap();
    assert_eq!(got.backend, cpu(Some(Fallback::CardOutOfMemory)));
    assert!(got.backend.fell_back());
}

#[test]
fn the_apps_thread_count_wins_on_every_provider() {
    for (gpu, gpu_works) in [(false, true), (true, true), (true, false)] {
        let (got, tried) = load_with(gpu, true, true, Some(3), gpu_works);
        assert_eq!(got.unwrap().2, 3, "gpu {gpu} works {gpu_works}");
        assert!(tried.iter().all(|(_, n)| *n == 3), "{tried:?}");
    }
    // never zero
    assert_eq!(load_with(false, true, true, Some(0), true).0.unwrap().2, 1);
}

#[test]
fn when_the_processor_cannot_load_it_either_that_error_is_the_one_returned() {
    let got = load_on_best_provider::<()>(true, Some(&rtx()), true, None, 16, |p, _| match p {
        Provider::DirectMl(_) => Err(SpeechError::Internal("gpu".into())),
        Provider::Processor => Err(SpeechError::BadModel),
    });
    assert_eq!(got.unwrap_err(), SpeechError::BadModel);
}

// ---- load() with the real parakeet-rs and ONNX Runtime, on folders that are wrong ----

#[test]
fn a_folder_without_the_files_is_model_missing_for_every_onnx_kind() {
    let dir = tempfile::tempdir().unwrap();
    for id in ["parakeet-ultra", "parakeet-unified-en", "nemotron-3.5"] {
        let m = catalog::find(id).unwrap();
        let err = load(m, dir.path(), LoadOptions::default()).err().unwrap();
        assert_eq!(err, SpeechError::ModelMissing, "{id}");
    }
    // and a folder that is not there
    let m = catalog::find("parakeet-ultra").unwrap();
    let err = load(
        m,
        &dir.path().join("nope"),
        LoadOptions {
            gpu: true,
            threads: Some(2),
        },
    )
    .err()
    .unwrap();
    assert_eq!(err, SpeechError::ModelMissing);
}

#[test]
fn files_that_are_not_models_are_bad_model_and_the_message_names_no_path() {
    let dir = tempfile::tempdir().unwrap();
    let marker = dir
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .into_owned();
    for id in ["parakeet-ultra", "parakeet-unified-en", "nemotron-3.5"] {
        let m = catalog::find(id).unwrap();
        let sub = dir.path().join(id);
        std::fs::create_dir(&sub).unwrap();
        for f in m.files {
            std::fs::write(sub.join(f.name), b"this is not a model").unwrap();
        }
        let err = load(m, &sub, LoadOptions::default()).err().unwrap();
        assert_eq!(err, SpeechError::BadModel, "{id}");
        let text = format!("{err} {err:?}");
        assert!(!text.contains(&marker) && !text.contains(id), "{text}");
    }
}

#[test]
fn the_non_speech_kinds_are_still_not_loadable_as_speech() {
    let dir = tempfile::tempdir().unwrap();
    let vad = catalog::find("vad").unwrap();
    assert!(matches!(
        load(vad, dir.path(), LoadOptions::default()).err().unwrap(),
        SpeechError::Unsupported(_)
    ));
}

#[test]
fn every_parakeet_segment_starts_a_word_in_both_modes() {
    for timing in [Timing::Segments, Timing::Words] {
        let got = segments_from_words(two_sentences(), timing);
        assert!(!got.is_empty());
        assert!(got.iter().all(|s| s.starts_word), "{timing:?}");
    }
    // So joining gives the segments with single spaces.
    let got = segments_from_words(two_sentences(), Timing::Segments);
    assert_eq!(
        voice_engine_stt::join_segments(&got),
        got.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    );
}
