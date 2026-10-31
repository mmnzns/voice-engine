//! Whisper on a real model, on a Mac or a Windows PC: `cargo test -p voice-engine-stt -- --ignored --test-threads=1`.
//!
//! The model is `whisper-turbo` (and `vad` for the voice detection tests), each looked up on its own: in
//! `$VOICE_ENGINE_TEST_MODELS` when that is set, else in the shared folder, then in Talaan's folder (`common`, which
//! says exactly where), only ever read. Speech comes from `$VOICE_ENGINE_TEST_CLIP` (a 16 kHz mono WAV that says
//! `SENTENCE`: `docs/data/clip-sentence.wav`) when that is set, else, on a Mac, from the macOS `say` command in a
//! temporary folder converted with `afconvert`. A test that needs `say`'s own wording (the vocabulary test, the voice
//! detection ones that need a phrase with no pause in it) is skipped without it. The graphics card is asked for on the
//! Mac only (Metal); elsewhere these run on the processor. Every test here goes through one lock: the Metal shader
//! cache can SIGBUS when several processes (or contexts) start at once.

mod common;

use common::{
    EXPECT_CARD_ENV, SENTENCE, check_shape, expect_card, guard, lower_letters, model_dir, ms, say,
    seg_info, text_of, use_gpu, words_of,
};
use std::path::PathBuf;
use std::time::Instant;
use voice_engine_models::catalog::find;
use voice_engine_stt::threads::{ThreadKind, default_threads, logical_cores};
use voice_engine_stt::vad;
use voice_engine_stt::whisper_engine::WhisperEngine;
use voice_engine_stt::words::clean;
use voice_engine_stt::{ChunkOptions, LoadOptions, Timing, Transcriber, load};

fn model_file() -> PathBuf {
    let m = find("whisper-turbo").unwrap();
    let dir = model_dir("whisper-turbo");
    let file = dir.join(m.files[0].name);
    assert!(
        file.is_file(),
        "the test model is not at {}",
        file.display()
    );
    file
}

/// The `gpu` options to run: with and without the card on the Mac or with the `vulkan` feature, the processor only
/// elsewhere.
fn gpu_modes() -> Vec<bool> {
    if use_gpu() {
        vec![false, true]
    } else {
        vec![false]
    }
}

fn engine(gpu: bool) -> Box<dyn Transcriber> {
    let m = find("whisper-turbo").unwrap();
    load(
        m,
        &model_dir("whisper-turbo"),
        LoadOptions { gpu, threads: None },
    )
    .unwrap()
}

#[test]
#[ignore]
fn both_modes_give_segments_and_ordered_timed_words() {
    need_clip!();
    let _g = guard();
    let clip = say(SENTENCE);
    eprintln!("clip: {} ms", ms(&clip));
    let mut e = engine(use_gpu());
    for timing in [Timing::Segments, Timing::Words] {
        let t0 = Instant::now();
        let t = e
            .transcribe(
                &clip,
                &ChunkOptions {
                    timing,
                    ..ChunkOptions::default()
                },
            )
            .unwrap();
        // Counts and times only: nothing a person said goes into a log.
        eprintln!(
            "{timing:?}: {} segments, {} words, {} chars in {:?}",
            t.segments.len(),
            words_of(&t).len(),
            text_of(&t).chars().count(),
            t0.elapsed()
        );
        for s in &t.segments {
            eprintln!("   [{}..{}] {} words", s.start_ms, s.end_ms, s.words.len());
        }
        check_shape(&t, ms(&clip));
        let heard = lower_letters(&words_of(&t).join(" "));
        assert!(
            heard.contains("speechengine") || heard.contains("shorttest"),
            "the clip was not heard in {} letters",
            heard.len()
        );
        assert_eq!(t.language.as_deref(), Some("en"));
        if timing == Timing::Words {
            assert!(
                t.segments.len() >= 10,
                "{} segments: not one word each",
                t.segments.len()
            );
            assert!(t.segments.iter().all(|s| !s.text.contains(' ')));
            assert!(
                t.segments
                    .iter()
                    .all(|s| s.words.len() == 1 && s.words[0].text == s.text)
            );
        }
    }
}

/// Talaan's own `transcribe` (the existing function the helper uses) on the same audio: the segments of
/// `Timing::Segments` must have the same text and times, with and without the card.
#[test]
#[ignore]
fn segments_mode_is_what_talaans_settings_give() {
    need_clip!();
    let _g = guard();
    let clip = say(&format!(
        "{SENTENCE} We will see if the segments match. And a second sentence follows."
    ));
    for gpu in gpu_modes() {
        let threads = 6;
        let m = find("whisper-turbo").unwrap();
        let dir = model_dir("whisper-turbo");
        let mut e = load(
            m,
            &dir,
            LoadOptions {
                gpu,
                threads: Some(threads),
            },
        )
        .unwrap();
        let ours = e
            .transcribe(
                &clip,
                &ChunkOptions {
                    timing: Timing::Segments,
                    language: None,
                    ..ChunkOptions::default()
                },
            )
            .unwrap();
        // The reference: a bare whisper.cpp context set up like Talaan's WhisperEngine::load.
        let params = voice_engine_stt::context_params(gpu.then_some(0), gpu);
        let ctx = whisper_rs::WhisperContext::new_with_params(model_file(), params).unwrap();
        let mut state = ctx.create_state().unwrap();
        let (pieces, language) =
            voice_engine_stt::transcribe(&mut state, &clip, None, threads).unwrap();
        assert_eq!(language, ours.language);
        let want: Vec<_> = pieces
            .iter()
            .map(|p| (p.text.clone(), p.start_ms, p.end_ms))
            .collect();
        let got: Vec<_> = ours
            .segments
            .iter()
            .map(|s| (s.text.clone(), s.start_ms, s.end_ms))
            .collect();
        eprintln!(
            "gpu {gpu}: talaan {} segments, ours {} segments",
            want.len(),
            got.len()
        );
        assert_eq!(got, want, "gpu {gpu}");
        check_shape(&ours, ms(&clip));
    }
}

/// Ensayo's `transcribe_words` re-run with whisper-rs directly (its settings, copied) on the same audio, then
/// `words::clean`: `Timing::Words` must give exactly those words.
#[test]
#[ignore]
fn words_mode_is_what_ensayos_settings_give() {
    need_clip!();
    let _g = guard();
    let clip = say(&format!(
        "{SENTENCE} Ensayo counts words, so every word needs a time."
    ));
    let threads = 6;
    let m = find("whisper-turbo").unwrap();
    let dir = model_dir("whisper-turbo");
    let mut e = load(
        m,
        &dir,
        LoadOptions {
            gpu: use_gpu(),
            threads: Some(threads),
        },
    )
    .unwrap();
    let ours = e
        .transcribe(
            &clip,
            &ChunkOptions {
                timing: Timing::Words,
                language: Some("en".into()),
                ..ChunkOptions::default()
            },
        )
        .unwrap();

    // The card the engine chooses (Metal on the Mac, the discrete Vulkan card elsewhere); the processor when none.
    let mac = cfg!(target_os = "macos");
    let all = voice_engine_stt::devices();
    let card = use_gpu()
        .then(|| voice_engine_stt::whisper_engine::choose_gpu(&all, mac))
        .flatten()
        .map(|(_, number)| number);
    let params = voice_engine_stt::context_params(card, mac && card.is_some());
    let ctx = whisper_rs::WhisperContext::new_with_params(model_file(), params).unwrap();
    let mut state = ctx.create_state().unwrap();
    let mut p = whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    p.set_n_threads(threads as i32);
    p.set_language(Some("en"));
    p.set_no_context(true);
    p.set_suppress_blank(true);
    p.set_suppress_nst(true);
    p.set_token_timestamps(true);
    p.set_split_on_word(true);
    p.set_max_len(1);
    p.set_print_special(false);
    p.set_print_progress(false);
    p.set_print_realtime(false);
    p.set_print_timestamps(false);
    let mut audio = clip.clone();
    if audio.len() < 16_000 + 1_600 {
        audio.resize(16_000 + 1_600, 0.0);
    }
    state.full(p, &audio).unwrap();
    let chunk_ms = ms(&clip);
    let mut raw = vec![];
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        if seg.no_speech_probability() > 0.8 {
            continue;
        }
        let text = seg
            .to_str_lossy()
            .map(|t| t.to_string())
            .unwrap_or_default();
        let start_ms = ((seg.start_timestamp().max(0) * 10) as u64).min(chunk_ms);
        let end_ms = ((seg.end_timestamp().max(0) * 10) as u64)
            .min(chunk_ms)
            .max(start_ms);
        raw.push(voice_engine_stt::Word {
            text,
            start_ms,
            end_ms,
        });
    }
    let want = clean(raw);
    let got: Vec<_> = ours.segments.iter().flat_map(|s| s.words.clone()).collect();
    eprintln!("ensayo {} words, ours {} words", want.len(), got.len());
    assert!(want.len() > 8);
    assert_eq!(got, want);
}

#[test]
#[ignore]
fn clips_of_every_short_length_work() {
    need_clip!();
    let _g = guard();
    let clip = say(SENTENCE);
    let mut e = engine(use_gpu());
    for n in [0usize, 1_600, 6_400, 15_999, 16_000, 16_001, 20_000] {
        for timing in [Timing::Segments, Timing::Words] {
            let part = &clip[..n.min(clip.len())];
            let t = e
                .transcribe(
                    part,
                    &ChunkOptions {
                        timing,
                        ..ChunkOptions::default()
                    },
                )
                .unwrap_or_else(|e| panic!("{n} samples {timing:?}: {e}"));
            let clip_ms = ms(part);
            for s in &t.segments {
                assert!(s.end_ms <= clip_ms, "{n} samples: {}", seg_info(s));
                assert!(
                    s.words
                        .iter()
                        .all(|w| w.end_ms <= clip_ms && w.start_ms <= w.end_ms),
                    "{n} samples: {}",
                    seg_info(s)
                );
            }
            eprintln!(
                "{n} samples ({clip_ms} ms) {timing:?}: {} segments, {} words",
                t.segments.len(),
                words_of(&t).len()
            );
            // From 0.4 s of "Hello there." on, the speech is heard: padding gave Whisper enough audio to decode.
            if n >= 6_400 {
                assert!(
                    lower_letters(&text_of(&t)).contains("hello"),
                    "{n} samples {timing:?}: hello not heard in {} chars",
                    text_of(&t).chars().count()
                );
            }
        }
    }
    // 0.4 s of real speech (the start of the clip) and 0.4 s of silence.
    let t = e
        .transcribe(&vec![0.0; 6_400], &ChunkOptions::default())
        .unwrap();
    assert!(t.segments.iter().all(|s| s.end_ms <= 400));
}

#[test]
#[ignore]
fn three_minutes_run_in_one_call() {
    need_clip!();
    let _g = guard();
    let clip = say(SENTENCE);
    let mut audio = vec![];
    while ms(&audio) < 180_000 {
        audio.extend_from_slice(&clip);
        audio.extend(std::iter::repeat_n(0.0f32, 4_000)); // a quarter second between repeats
    }
    eprintln!("long clip: {} ms", ms(&audio));
    let mut e = engine(use_gpu());
    for timing in [Timing::Segments, Timing::Words] {
        let t0 = Instant::now();
        let t = e
            .transcribe(
                &audio,
                &ChunkOptions {
                    timing,
                    ..ChunkOptions::default()
                },
            )
            .unwrap();
        let took = t0.elapsed();
        let words = words_of(&t);
        eprintln!(
            "{timing:?}: {} segments, {} words in {took:?} ({:.0}x real time), stopped early at {:?}",
            t.segments.len(),
            words.len(),
            180.0 / took.as_secs_f64(),
            t.stopped_early_at_ms
        );
        check_shape(&t, ms(&audio));
        assert!(
            words.len() > 150,
            "{} words for 180 s of speech",
            words.len()
        );
        let last = t.segments.last().unwrap().end_ms;
        assert!(
            last > ms(&audio) - 15_000,
            "the transcript ends at {last} ms of {} ms",
            ms(&audio)
        );
    }
}

/// The initial prompt conditions Whisper; it does not force a spelling. Measured on 2026-10-29: "Mogo." leaves
/// "Mago" as it is and keeps the capital and the full stop (the vocabulary prompt ends with a full stop for that
/// reason), while a prompt that lists the spellings Whisper is choosing between ("Mago, Mogo, Magoo.") does change the
/// decode ("Magoo"). So the proof that the vocabulary reaches whisper.cpp is a changed, and repeatable, decode, and
/// the proof that it does no harm is that the style of a plain vocabulary is kept.
#[test]
#[ignore]
fn vocabulary_reaches_whisper_cpps_prompt() {
    need_say!();
    let _g = guard();
    let clip = say("The team at Mogo is growing quickly.");
    let mut e = engine(use_gpu());
    let plain = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    let again = e
        .transcribe(
            &clip,
            &ChunkOptions {
                vocabulary: vec![],
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    assert_eq!(plain, again, "greedy decoding repeats");
    let styled = e
        .transcribe(
            &clip,
            &ChunkOptions {
                vocabulary: vec!["Mogo".into()],
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    let styled_text = text_of(&styled);
    assert!(
        styled_text.trim_start().starts_with(char::is_uppercase)
            && styled_text.trim_end().ends_with('.'),
        "a plain vocabulary keeps the capital and the full stop ({} chars)",
        styled_text.chars().count()
    );
    let words = vec!["Mago".to_string(), "Mogo".to_string(), "Magoo".to_string()];
    let with = e
        .transcribe(
            &clip,
            &ChunkOptions {
                vocabulary: words.clone(),
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    eprintln!(
        "without: {} words, {} chars; with: {} words, {} chars",
        words_of(&plain).len(),
        text_of(&plain).chars().count(),
        words_of(&with).len(),
        text_of(&with).chars().count()
    );
    check_shape(&with, ms(&clip));
    assert_ne!(
        text_of(&with),
        text_of(&plain),
        "the prompt had no effect on the decode"
    );
    let repeat = e
        .transcribe(
            &clip,
            &ChunkOptions {
                vocabulary: words,
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    assert_eq!(with, repeat);
    // Fillers and vocabulary together are accepted too.
    let both = e
        .transcribe(
            &clip,
            &ChunkOptions {
                vocabulary: vec!["Mogo".into()],
                keep_fillers: true,
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    check_shape(&both, ms(&clip));
}

#[test]
#[ignore]
fn fillers_option_is_accepted() {
    need_clip!();
    let _g = guard();
    let clip = say("Um, so, uh, I think, like, we should, you know, go ahead.");
    let mut e = engine(use_gpu());
    for keep_fillers in [false, true] {
        let t = e
            .transcribe(
                &clip,
                &ChunkOptions {
                    keep_fillers,
                    ..ChunkOptions::default()
                },
            )
            .unwrap();
        eprintln!("keep_fillers {keep_fillers}: {} words", words_of(&t).len());
        check_shape(&t, ms(&clip));
    }
}

#[test]
#[ignore]
fn backend_names_the_card_with_it_and_cpu_without() {
    let _g = guard();
    assert!(
        use_gpu() || !expect_card(),
        "{EXPECT_CARD_ENV} is set, but this build does not ask for the card (build with --features vulkan)"
    );
    // The card is asked for on the Mac (Metal) and in a `vulkan` build; elsewhere there is no card to ask for.
    if use_gpu() {
        let e = engine(true);
        let b = e.backend();
        eprintln!("gpu: {b:?}");
        if cfg!(target_os = "macos") {
            assert_eq!(b.backend, "metal");
        }
        match b.backend.as_str() {
            "metal" | "vulkan" => {
                assert!(b.device.is_some(), "{b:?}");
                assert!(!b.fell_back, "{b:?}");
            }
            // A `vulkan` build on a PC with no discrete card, or one whose card did not start: the processor, said so.
            "cpu" => {
                assert_eq!(b.device, None);
                assert!(b.fell_back, "{b:?}");
                assert!(
                    !expect_card(),
                    "{EXPECT_CARD_ENV} is set, but the card did not run: {b:?}"
                );
            }
            other => panic!("unexpected backend {other}: {b:?}"),
        }
        drop(e);
    }
    let e = engine(false);
    let b = e.backend();
    eprintln!("cpu: {b:?}");
    assert_eq!(b.backend, "cpu");
    assert_eq!(b.device, None);
    assert!(!b.fell_back, "the app did not ask for the card");
}

#[test]
#[ignore]
fn threads_follow_the_option_and_the_backend_that_loaded() {
    let _g = guard();
    let file = model_file();
    let logical = logical_cores();
    for gpu in gpu_modes().into_iter().rev() {
        let e = WhisperEngine::load(
            &file,
            LoadOptions {
                gpu,
                threads: Some(3),
            },
        )
        .unwrap();
        assert_eq!(e.threads(), 3, "gpu {gpu}");
        let e = WhisperEngine::load(
            &file,
            LoadOptions {
                gpu,
                threads: Some(0),
            },
        )
        .unwrap();
        assert_eq!(e.threads(), 1, "gpu {gpu}: at least one");
        let e = WhisperEngine::load(&file, LoadOptions { gpu, threads: None }).unwrap();
        // The rule of where the model really runs: with the card asked for on a PC that has none it is the processor's.
        let kind = if gpu && e.backend().backend != "cpu" {
            ThreadKind::Gpu
        } else {
            ThreadKind::Processor
        };
        assert_eq!(e.threads(), default_threads(kind, logical), "gpu {gpu}");
        eprintln!(
            "gpu {gpu}: {} threads of {logical} logical cores",
            e.threads()
        );
    }
}

#[test]
#[ignore]
fn the_language_hint_is_passed_and_the_language_comes_back() {
    need_clip!();
    let _g = guard();
    let clip = say(SENTENCE);
    let mut e = engine(use_gpu());
    let detected = e.transcribe(&clip, &ChunkOptions::default()).unwrap();
    assert_eq!(detected.language.as_deref(), Some("en"));
    let forced = e
        .transcribe(
            &clip,
            &ChunkOptions {
                language: Some("fr".into()),
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        forced.language.as_deref(),
        Some("fr"),
        "the hint is passed as is"
    );
}

#[test]
#[ignore]
fn silence_after_the_last_word_is_resumed_and_reported() {
    need_clip!();
    let _g = guard();
    let mut audio = say(SENTENCE);
    let speech_ms = ms(&audio);
    audio.extend(std::iter::repeat_n(0.0f32, 16_000 * 6)); // 6 s of silence
    let mut e = engine(use_gpu());
    let t = e.transcribe(&audio, &ChunkOptions::default()).unwrap();
    eprintln!(
        "speech {speech_ms} ms + 6 s of silence: stopped early at {:?}, {} words",
        t.stopped_early_at_ms,
        words_of(&t).len()
    );
    check_shape(&t, ms(&audio));
    let at = t
        .stopped_early_at_ms
        .expect("2 s or more were left after the last word");
    assert!(
        at + 1_500 > speech_ms && at < speech_ms + 1_500,
        "stopped at {at}, speech ends near {speech_ms}"
    );
    // The resume pass found nothing in the silence.
    assert!(
        t.segments.iter().all(|s| s.end_ms <= speech_ms + 1_000),
        "{} segments, speech ends near {speech_ms} ms",
        t.segments.len()
    );
}

#[test]
#[ignore]
fn a_missing_file_and_a_broken_file_give_kinds() {
    let _g = guard();
    let dir = tempfile::tempdir().unwrap();
    let m = find("whisper-turbo").unwrap();
    assert_eq!(
        load(m, dir.path(), LoadOptions::default()).err(),
        Some(voice_engine_stt::SpeechError::ModelMissing)
    );
    std::fs::write(dir.path().join(m.files[0].name), b"not a model").unwrap();
    assert_eq!(
        load(
            m,
            dir.path(),
            LoadOptions {
                gpu: true,
                threads: None
            }
        )
        .err(),
        Some(voice_engine_stt::SpeechError::BadModel)
    );
}

// ---- voice detection (`vad::detect`, the catalog's `vad` model) ----

fn vad_dir() -> PathBuf {
    let m = find("vad").unwrap();
    let dir = model_dir("vad");
    let file = dir.join(m.files[0].name);
    assert!(
        file.is_file(),
        "the voice detection test model is not at {}",
        file.display()
    );
    dir
}

/// `lead_ms` of silence, `speech`, `tail_ms` of silence (16 kHz).
fn silence_speech_silence(lead_ms: u64, speech: &[f32], tail_ms: u64) -> Vec<f32> {
    let mut audio = vec![0.0f32; (lead_ms * 16) as usize];
    audio.extend_from_slice(speech);
    audio.extend(std::iter::repeat_n(0.0f32, (tail_ms * 16) as usize));
    audio
}

/// A continuous phrase (no full stop in the middle, so no pause long enough to end a span).
const PHRASE: &str = "This is a short test of the voice detection in the speech engine";

#[test]
#[ignore]
fn voice_detection_finds_one_span_around_speech_between_silences() {
    need_say!();
    let _g = guard();
    let speech = say(PHRASE);
    let (lead, tail) = (2_000u64, 2_000u64);
    let audio = silence_speech_silence(lead, &speech, tail);
    let t0 = Instant::now();
    let spans = vad::detect(&vad_dir(), &audio).unwrap();
    eprintln!(
        "{} ms of audio (speech {lead}..{} ms): spans {spans:?} in {:?}",
        ms(&audio),
        lead + ms(&speech),
        t0.elapsed()
    );
    assert_eq!(spans.len(), 1, "{spans:?}");
    let (start, end) = spans[0];
    let speech_end = lead + ms(&speech);
    assert!(
        start.abs_diff(lead) <= 300,
        "the span starts at {start} ms, the speech at {lead} ms"
    );
    assert!(
        end.abs_diff(speech_end) <= 300,
        "the span ends at {end} ms, the speech at {speech_end} ms"
    );
    assert!(start < end && end <= ms(&audio));
}

#[test]
#[ignore]
fn voice_detection_gives_no_span_for_silence_and_for_quiet_noise() {
    let _g = guard();
    let dir = vad_dir();
    let silence = vec![0.0f32; 16_000 * 5];
    assert_eq!(vad::detect(&dir, &silence).unwrap(), vec![]);
    // Very quiet pseudo-random hiss (a fixed generator, so the test is repeatable).
    let mut x = 12345u32;
    let hiss: Vec<f32> = (0..16_000 * 5)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            ((x >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.002
        })
        .collect();
    assert_eq!(vad::detect(&dir, &hiss).unwrap(), vec![]);
}

#[test]
#[ignore]
fn voice_detection_keeps_two_bursts_apart_and_in_order() {
    need_say!();
    let _g = guard();
    let speech = say(PHRASE);
    let gap_ms = 3_000u64;
    let mut audio = silence_speech_silence(1_000, &speech, gap_ms);
    audio.extend_from_slice(&speech);
    audio.extend(std::iter::repeat_n(0.0f32, 16_000));
    let spans = vad::detect(&vad_dir(), &audio).unwrap();
    eprintln!("two bursts: {spans:?} in {} ms", ms(&audio));
    assert_eq!(spans.len(), 2, "{spans:?}");
    let second_start = 1_000 + ms(&speech) + gap_ms;
    assert!(spans[0].1 < spans[1].0, "{spans:?}");
    assert!(spans[0].0.abs_diff(1_000) <= 300, "{spans:?}");
    assert!(spans[1].0.abs_diff(second_start) <= 300, "{spans:?}");
    assert!(
        spans[1].1.abs_diff(second_start + ms(&speech)) <= 300,
        "{spans:?}"
    );
}

#[test]
#[ignore]
fn voice_detection_edge_cases_on_the_real_model() {
    need_clip!();
    let _g = guard();
    let dir = vad_dir();
    assert_eq!(vad::detect(&dir, &[]).unwrap(), vec![]);
    assert_eq!(vad::detect(&dir, &[0.5; 100]).unwrap(), vec![]);
    // 100 ms of speech is more than a frame but under the 250 ms minimum: the detector drops it.
    let speech = say(PHRASE);
    assert_eq!(vad::detect(&dir, &speech[..16_000 / 10]).unwrap(), vec![]);
    // A three minute track (speech, a pause, speech, ...) runs in one call and stays on the clock.
    let mut long = vec![];
    while long.len() < 16_000 * 180 {
        long.extend_from_slice(&speech);
        long.extend(std::iter::repeat_n(0.0f32, 16_000 * 2));
    }
    let t0 = Instant::now();
    let spans = vad::detect(&dir, &long).unwrap();
    eprintln!(
        "{} s of audio: {} spans in {:?}",
        long.len() / 16_000,
        spans.len(),
        t0.elapsed()
    );
    assert!(spans.len() >= 10, "{}", spans.len());
    assert!(spans.windows(2).all(|w| w[0].1 < w[1].0));
    assert!(spans.iter().all(|&(a, b)| a < b && b <= ms(&long)));
}

/// The speech twice with a short pause between (the clip's own quiet ends make it about 240 ms with the clip, under the
/// default 300 ms minimum silence and over a 100 ms one), with a second of silence on each side.
fn two_phrases_with_a_short_pause() -> Vec<f32> {
    let speech = say(PHRASE);
    let mut audio = silence_speech_silence(1_000, &speech, 100);
    audio.extend_from_slice(&speech);
    audio.extend(std::iter::repeat_n(0.0f32, 16_000));
    audio
}

#[test]
#[ignore]
fn voice_detection_settings_default_is_what_detect_gives() {
    need_clip!();
    let _g = guard();
    let dir = vad_dir();
    let audio = two_phrases_with_a_short_pause();
    let plain = vad::detect(&dir, &audio).unwrap();
    let with = vad::detect_with(&dir, &audio, &vad::VadSettings::default()).unwrap();
    eprintln!(
        "{} ms of audio: {} spans with the defaults",
        ms(&audio),
        plain.len()
    );
    assert_eq!(plain, with);
    assert!(!plain.is_empty());
}

#[test]
#[ignore]
fn voice_detection_smaller_minimum_silence_gives_more_spans() {
    need_clip!();
    let _g = guard();
    let dir = vad_dir();
    let audio = two_phrases_with_a_short_pause();
    let default = vad::detect(&dir, &audio).unwrap();
    let short = vad::VadSettings {
        min_silence_ms: 100,
        ..vad::VadSettings::default()
    };
    let more = vad::detect_with(&dir, &audio, &short).unwrap();
    eprintln!("default {default:?}, 100 ms minimum silence {more:?}");
    assert!(
        more.len() > default.len(),
        "{} vs {}",
        more.len(),
        default.len()
    );
    assert!(more.windows(2).all(|w| w[0].1 < w[1].0));
}

#[test]
#[ignore]
fn voice_detection_padding_widens_clamps_and_merges() {
    need_clip!();
    let _g = guard();
    let dir = vad_dir();
    let audio = two_phrases_with_a_short_pause();
    let short = vad::VadSettings {
        min_silence_ms: 100,
        ..vad::VadSettings::default()
    };
    let bare = vad::detect_with(&dir, &audio, &short).unwrap();
    // 30 ms each side: every span is wider, none runs outside the audio.
    let padded = vad::detect_with(
        &dir,
        &audio,
        &vad::VadSettings {
            padding_ms: 30,
            ..short
        },
    )
    .unwrap();
    eprintln!(
        "{} spans bare, {} with 30 ms padding",
        bare.len(),
        padded.len()
    );
    assert!(padded[0].0 <= bare[0].0 && padded[0].1 >= bare[0].1);
    assert!(padded.iter().all(|&(a, b)| a < b && b <= ms(&audio)));
    // Padding of 2 s runs into both ends of the audio and joins everything into one span: the whole track.
    let wide = vad::detect_with(
        &dir,
        &audio,
        &vad::VadSettings {
            padding_ms: 2_000,
            ..short
        },
    )
    .unwrap();
    assert_eq!(wide, vec![(0, ms(&audio))]);
}
