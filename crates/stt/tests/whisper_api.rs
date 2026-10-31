//! Whisper API tests that need no model: the pure parts of the engine (padding, clamping, prompt, early-stop
//! arithmetic, `words::clean`, the thread rule, the choice of GPU) and the errors a missing or broken file gives.

use voice_engine_protocol::gpu::{Device, DeviceKind};
use voice_engine_stt::join_segments;
use voice_engine_stt::threads::{ThreadKind, default_threads, resolve_threads};
use voice_engine_stt::whisper_engine::{
    FILLER_PROMPT, MAX_PASSES, MIN_SAMPLES, Pass, PassDecoder, RESUME_GAP_MS, RawSegment, Token,
    WhisperEngine, build_prompt, choose_gpu, decode_with_resume, describe_backend, pad_audio,
    segments_from_raw, words_from_tokens,
};
use voice_engine_stt::words::clean;
use voice_engine_stt::{
    ChunkOptions, LoadOptions, Segment, SpeechError, Timing, Transcriber, Word, load,
};

fn w(text: &str, a: u64, b: u64) -> Word {
    Word {
        text: text.into(),
        start_ms: a,
        end_ms: b,
    }
}

fn tok(text: &str, a: u64, b: u64) -> Token {
    Token {
        text: text.into(),
        start_ms: a,
        end_ms: b,
    }
}

fn raw(text: &str, a: u64, b: u64, no_speech: f32, tokens: Vec<Token>) -> RawSegment {
    RawSegment {
        start_ms: a,
        end_ms: b,
        text: text.into(),
        no_speech,
        tokens,
    }
}

// ---- thread rule (spec 9.2) ----

#[test]
fn processor_default_is_three_quarters_of_the_cores_clamped_2_to_12() {
    for (logical, want) in [
        (4, 3),
        (8, 6),
        (10, 7),
        (12, 9),
        (16, 12),
        (32, 12),
        (1, 2),
        (2, 2),
    ] {
        assert_eq!(
            default_threads(ThreadKind::Processor, logical),
            want,
            "{logical} cores"
        );
    }
}

#[test]
fn gpu_default_is_all_the_cores_clamped_2_to_8() {
    for (logical, want) in [
        (4, 4),
        (8, 8),
        (10, 8),
        (12, 8),
        (16, 8),
        (32, 8),
        (1, 2),
        (2, 2),
    ] {
        assert_eq!(
            default_threads(ThreadKind::Gpu, logical),
            want,
            "{logical} cores"
        );
    }
}

#[test]
fn the_apps_thread_setting_wins_and_is_at_least_one() {
    assert_eq!(resolve_threads(ThreadKind::Processor, Some(3), 16), 3);
    assert_eq!(resolve_threads(ThreadKind::Gpu, Some(12), 16), 12);
    assert_eq!(resolve_threads(ThreadKind::Gpu, Some(0), 16), 1);
    assert_eq!(resolve_threads(ThreadKind::Processor, None, 16), 12);
    assert_eq!(resolve_threads(ThreadKind::Gpu, None, 16), 8);
}

// ---- padding ----

#[test]
fn a_short_clip_is_padded_with_silence_to_one_second() {
    let clip = vec![0.25f32; 6_400]; // 0.4 s
    let padded = pad_audio(&clip);
    assert_eq!(padded.len(), MIN_SAMPLES);
    assert_eq!(MIN_SAMPLES, 16_000);
    assert!(padded[..6_400].iter().all(|s| *s == 0.25));
    assert!(padded[6_400..].iter().all(|s| *s == 0.0));
}

#[test]
fn empty_audio_pads_to_silence_not_an_error() {
    assert_eq!(pad_audio(&[]).len(), MIN_SAMPLES);
}

#[test]
fn a_clip_of_one_second_or_more_is_not_touched() {
    for n in [16_000, 16_001, 48_000] {
        let clip = vec![0.1f32; n];
        assert_eq!(pad_audio(&clip).len(), n);
    }
}

// ---- prompt ----

#[test]
fn the_prompt_is_none_without_vocabulary_or_fillers() {
    assert_eq!(build_prompt(&[], false), None);
    assert_eq!(build_prompt(&["  ".into(), "".into()], false), None);
}

#[test]
fn vocabulary_alone_is_the_words_joined_with_commas_and_a_full_stop() {
    let words = vec!["Zorblax".to_string(), "Mogo".to_string()];
    assert_eq!(
        build_prompt(&words, false).as_deref(),
        Some("Zorblax, Mogo.")
    );
}

#[test]
fn fillers_alone_are_whisper_cpps_documented_filler_prompt() {
    assert_eq!(
        FILLER_PROMPT,
        "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking."
    );
    assert_eq!(build_prompt(&[], true).as_deref(), Some(FILLER_PROMPT));
}

#[test]
fn fillers_come_first_then_the_vocabulary() {
    let words = vec!["Zorblax".to_string(), "Mogo".to_string()];
    assert_eq!(
        build_prompt(&words, true).as_deref(),
        Some(format!("{FILLER_PROMPT} Zorblax, Mogo.").as_str())
    );
}

#[test]
fn blank_entries_and_null_bytes_never_reach_whisper_cpp() {
    // whisper-rs panics on a null byte in the prompt.
    let words = vec![
        " Zor\0blax ".to_string(),
        "  ".to_string(),
        "Mogo".to_string(),
    ];
    let prompt = build_prompt(&words, false).unwrap();
    assert_eq!(prompt, "Zorblax, Mogo.");
    assert!(!prompt.contains('\0'));
}

// ---- words::clean, ported from Ensayo's tests ----

#[test]
fn punctuation_only_pieces_attach_to_the_previous_word() {
    let got = clean(vec![
        w("Okay", 0, 300),
        w(".", 300, 320),
        w("So", 400, 600),
        w(",", 600, 610),
    ]);
    assert_eq!(got, vec![w("Okay.", 0, 320), w("So,", 400, 610)]);
}

#[test]
fn noise_markers_and_empty_pieces_are_dropped() {
    let got = clean(vec![
        w("[BLANK_AUDIO]", 0, 500),
        w("  ", 500, 600),
        w("(music)", 600, 700),
        w("*laughs*", 700, 800),
        w("Hello", 800, 900),
    ]);
    assert_eq!(got, vec![w("Hello", 800, 900)]);
}

#[test]
fn a_leading_punctuation_piece_is_dropped_not_attached_to_nothing() {
    assert_eq!(
        clean(vec![w(".", 0, 10), w("Yes", 20, 200)]),
        vec![w("Yes", 20, 200)]
    );
}

#[test]
fn times_never_run_backwards() {
    let got = clean(vec![w("one", 100, 50), w("two", 40, 90)]);
    assert!(got.iter().all(|x| x.end_ms >= x.start_ms), "{got:?}");
    assert!(got[1].start_ms >= got[0].end_ms, "{got:?}");
}

#[test]
fn whitespace_around_words_is_trimmed() {
    assert_eq!(clean(vec![w(" Hello", 0, 100)]), vec![w("Hello", 0, 100)]);
}

#[test]
fn a_word_with_punctuation_inside_it_is_not_treated_as_punctuation() {
    assert_eq!(
        clean(vec![w("it's", 0, 100), w("24%", 100, 300)]),
        vec![w("it's", 0, 100), w("24%", 100, 300)]
    );
}

// ---- words from tokens ----

#[test]
fn tokens_without_a_leading_space_continue_the_word() {
    let tokens = vec![
        tok(" Hello", 0, 300),
        tok(",", 300, 320),
        tok(" wor", 400, 500),
        tok("ld", 500, 700),
        tok(" .", 700, 720),
    ];
    assert_eq!(
        words_from_tokens(&tokens),
        vec![w("Hello,", 0, 320), w("world", 400, 700), w(".", 700, 720)]
    );
}

#[test]
fn a_first_token_without_a_space_starts_a_word() {
    assert_eq!(
        words_from_tokens(&[tok("Hi", 10, 90), tok(" there", 100, 300)]),
        vec![w("Hi", 10, 90), w("there", 100, 300)]
    );
}

// ---- whisper output to segments ----

#[test]
fn segments_mode_keeps_the_segment_text_and_derives_words_from_tokens() {
    let seg = raw(
        " Hello, world.",
        0,
        900,
        0.01,
        vec![
            tok(" Hello", 0, 300),
            tok(",", 300, 320),
            tok(" world", 400, 800),
            tok(".", 800, 900),
        ],
    );
    let got = segments_from_raw(vec![seg], Timing::Segments, 5_000);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].text, "Hello, world.");
    assert_eq!((got[0].start_ms, got[0].end_ms), (0, 900));
    assert_eq!(
        got[0].words,
        vec![w("Hello,", 0, 320), w("world.", 400, 900)]
    );
}

#[test]
fn segments_mode_drops_noise_and_no_speech_like_talaan() {
    let segs = vec![
        raw("[BLANK_AUDIO]", 0, 500, 0.0, vec![]),
        raw(" (music)", 500, 900, 0.0, vec![]),
        raw(" *laughs*", 900, 950, 0.0, vec![]),
        raw(" ...", 950, 990, 0.0, vec![]),
        raw(
            " Thank you.",
            1_000,
            2_000,
            0.9,
            vec![tok(" Thank", 1_000, 1_500)],
        ),
        raw(
            " Okay.",
            2_000,
            2_500,
            0.1,
            vec![tok(" Okay", 2_000, 2_400), tok(".", 2_400, 2_500)],
        ),
    ];
    let got = segments_from_raw(segs, Timing::Segments, 10_000);
    assert_eq!(got.len(), 1, "{got:?}");
    assert_eq!(got[0].text, "Okay.");
}

#[test]
fn times_are_clamped_to_the_real_audio_not_the_padded_audio() {
    // A 0.4 s clip is padded to 1 s; Whisper may place words in the padding.
    let seg = raw(
        " Hello there",
        0,
        1_000,
        0.0,
        vec![tok(" Hello", 100, 350), tok(" there", 600, 1_000)],
    );
    let got = segments_from_raw(vec![seg], Timing::Segments, 400);
    assert_eq!((got[0].start_ms, got[0].end_ms), (0, 400));
    assert!(
        got[0]
            .words
            .iter()
            .all(|x| x.start_ms <= 400 && x.end_ms <= 400)
    );
    assert!(got[0].words.iter().all(|x| x.start_ms <= x.end_ms));
    let got = segments_from_raw(
        vec![raw(" there", 700, 900, 0.0, vec![tok(" there", 700, 900)])],
        Timing::Words,
        400,
    );
    assert_eq!(got[0].words, vec![w("there", 400, 400)]);
    assert_eq!((got[0].start_ms, got[0].end_ms), (400, 400));
}

#[test]
fn words_mode_is_one_word_per_segment_after_clean() {
    let segs = vec![
        raw(" Okay", 0, 300, 0.0, vec![]),
        raw(".", 300, 320, 0.0, vec![]),
        raw(" [BLANK_AUDIO]", 320, 400, 0.0, vec![]),
        raw(" So", 400, 600, 0.0, vec![]),
        raw(" um", 600, 700, 0.95, vec![]),
    ];
    let got = segments_from_raw(segs, Timing::Words, 5_000);
    assert_eq!(
        got,
        vec![
            Segment {
                start_ms: 0,
                end_ms: 320,
                text: "Okay.".into(),
                starts_word: true,
                words: vec![w("Okay.", 0, 320)]
            },
            Segment {
                start_ms: 400,
                end_ms: 600,
                text: "So".into(),
                starts_word: true,
                words: vec![w("So", 400, 600)]
            },
        ]
    );
}

// ---- early stop and resume ----

/// Answers pass after pass from a script and records how much audio each pass was given.
struct Scripted {
    passes: Vec<Pass>,
    seen: Vec<usize>,
}

impl Scripted {
    fn new(passes: Vec<Pass>) -> Self {
        Scripted {
            passes,
            seen: vec![],
        }
    }
}

impl PassDecoder for Scripted {
    fn decode(&mut self, audio: &[f32]) -> Result<Pass, SpeechError> {
        self.seen.push(audio.len());
        if self.passes.is_empty() {
            return Ok(Pass {
                segments: vec![],
                language: None,
            });
        }
        Ok(self.passes.remove(0))
    }
}

fn pass(spans: &[(&str, u64, u64)], language: Option<&str>) -> Pass {
    Pass {
        segments: spans
            .iter()
            .map(|(t, a, b)| Segment {
                start_ms: *a,
                end_ms: *b,
                text: (*t).into(),
                starts_word: true,
                words: vec![w(t, *a, *b)],
            })
            .collect(),
        language: language.map(str::to_string),
    }
}

fn seconds(n: usize) -> Vec<f32> {
    vec![0.1; n * 16_000]
}

#[test]
fn decoding_that_stops_early_resumes_on_the_rest_with_the_right_offsets() {
    let mut dec = Scripted::new(vec![
        pass(&[("one", 0, 1_000), ("two", 3_000, 4_000)], Some("en")),
        pass(&[("three", 200, 1_500)], Some("en")),
        pass(&[], None),
    ]);
    let t = decode_with_resume(&mut dec, &seconds(10)).unwrap();
    // 10 s; first stop at 4 s; then the 6 s that were left (96_000 samples); then from 5.5 s (72_000 samples).
    assert_eq!(dec.seen, vec![160_000, 96_000, 72_000]);
    assert_eq!(t.stopped_early_at_ms, Some(4_000));
    assert_eq!(t.language.as_deref(), Some("en"));
    let times: Vec<_> = t
        .segments
        .iter()
        .map(|s| (s.text.as_str(), s.start_ms, s.end_ms))
        .collect();
    assert_eq!(
        times,
        vec![
            ("one", 0, 1_000),
            ("two", 3_000, 4_000),
            ("three", 4_200, 5_500)
        ]
    );
    // The words move with their segments.
    assert_eq!(t.segments[2].words, vec![w("three", 4_200, 5_500)]);
}

#[test]
fn no_resume_when_less_than_two_seconds_are_left() {
    let mut dec = Scripted::new(vec![
        pass(&[("a", 0, 8_500)], None),
        pass(&[("b", 0, 500)], None),
    ]);
    let t = decode_with_resume(&mut dec, &seconds(10)).unwrap(); // 1.5 s left
    assert_eq!(dec.seen.len(), 1);
    assert_eq!(t.stopped_early_at_ms, None);
    assert_eq!(t.segments.len(), 1);
}

#[test]
fn exactly_two_seconds_left_counts_as_stopped_early() {
    assert_eq!(RESUME_GAP_MS, 2_000);
    let mut dec = Scripted::new(vec![
        pass(&[("a", 0, 8_000)], None),
        pass(&[("b", 0, 1_900)], None),
    ]);
    let t = decode_with_resume(&mut dec, &seconds(10)).unwrap();
    assert_eq!(dec.seen.len(), 2);
    assert_eq!(t.stopped_early_at_ms, Some(8_000));
    assert_eq!(t.segments[1].start_ms, 8_000);
}

#[test]
fn a_pass_that_gets_nowhere_does_not_loop() {
    // Only 300 ms of progress: not worth another pass (the 500 ms rule).
    let mut dec = Scripted::new(vec![
        pass(&[("a", 0, 300)], None),
        pass(&[("b", 0, 300)], None),
    ]);
    let t = decode_with_resume(&mut dec, &seconds(10)).unwrap();
    assert_eq!(dec.seen.len(), 1);
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn silence_gives_an_empty_transcript_after_one_pass() {
    let mut dec = Scripted::new(vec![]);
    let t = decode_with_resume(&mut dec, &seconds(5)).unwrap();
    assert_eq!(dec.seen.len(), 1);
    assert!(t.segments.is_empty());
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn resuming_stops_after_eight_passes() {
    // Every pass says 1 s more; 60 s of audio would otherwise need 60 passes.
    let passes: Vec<Pass> = (0..20).map(|_| pass(&[("x", 0, 1_000)], None)).collect();
    let mut dec = Scripted::new(passes);
    let t = decode_with_resume(&mut dec, &seconds(60)).unwrap();
    assert_eq!(MAX_PASSES, 8);
    assert_eq!(dec.seen.len(), MAX_PASSES);
    assert_eq!(t.segments.len(), MAX_PASSES);
    assert_eq!(t.stopped_early_at_ms, Some(1_000));
}

#[test]
fn the_language_is_the_first_one_a_pass_reports() {
    let mut dec = Scripted::new(vec![
        pass(&[("a", 0, 4_000)], Some("fr")),
        pass(&[("b", 0, 4_000)], Some("en")),
        pass(&[], None),
    ]);
    let t = decode_with_resume(&mut dec, &seconds(12)).unwrap();
    assert_eq!(t.language.as_deref(), Some("fr"));
}

#[test]
fn times_never_pass_the_end_of_the_audio() {
    let mut dec = Scripted::new(vec![pass(&[("a", 0, 12_000)], None)]);
    let t = decode_with_resume(&mut dec, &seconds(10)).unwrap();
    assert!(t.segments.iter().all(|s| s.end_ms <= 10_000));
    assert!(
        t.segments
            .iter()
            .flat_map(|s| &s.words)
            .all(|x| x.end_ms <= 10_000)
    );
}

#[test]
fn a_decoder_error_is_passed_up() {
    struct Broken;
    impl PassDecoder for Broken {
        fn decode(&mut self, _: &[f32]) -> Result<Pass, SpeechError> {
            Err(SpeechError::Internal("whisper.cpp error -3".into()))
        }
    }
    assert_eq!(
        decode_with_resume(&mut Broken, &seconds(3)).unwrap_err(),
        SpeechError::Internal("whisper.cpp error -3".into())
    );
}

// ---- the choice of device and what backend() says ----

fn dev(index: usize, name: &str, backend: &str, kind: DeviceKind, memory_mb: u64) -> Device {
    Device {
        index,
        name: format!("{backend}{index}"),
        description: name.into(),
        backend: backend.into(),
        kind,
        memory_mb,
    }
}

#[test]
fn on_the_mac_the_metal_device_is_the_one_and_it_is_whispers_first_gpu() {
    let all = vec![
        dev(0, "Apple M3 Pro", "Metal", DeviceKind::Gpu, 24_000),
        dev(1, "Apple M3 Pro", "BLAS", DeviceKind::Accel, 0),
        dev(2, "CPU", "CPU", DeviceKind::Cpu, 0),
    ];
    let (d, n) = choose_gpu(&all, true).unwrap();
    assert_eq!((d.index, n), (0, 0));
    // an integrated device counts when there is nothing else
    let igpu = vec![
        dev(0, "CPU", "CPU", DeviceKind::Cpu, 0),
        dev(1, "Apple M1", "Metal", DeviceKind::Igpu, 8_000),
    ];
    let (d, n) = choose_gpu(&igpu, true).unwrap();
    assert_eq!((d.index, n), (1, 0));
}

#[test]
fn elsewhere_the_discrete_gpu_with_the_most_memory_and_never_an_integrated_one() {
    let all = vec![
        dev(0, "CPU", "CPU", DeviceKind::Cpu, 0),
        dev(1, "Intel UHD", "Vulkan", DeviceKind::Igpu, 1_000),
        dev(2, "NVIDIA small", "Vulkan", DeviceKind::Gpu, 6_000),
        dev(3, "NVIDIA big", "Vulkan", DeviceKind::Gpu, 16_000),
    ];
    let (d, n) = choose_gpu(&all, false).unwrap();
    assert_eq!(d.description, "NVIDIA big");
    assert_eq!(n, 2); // whisper.cpp counts the iGPU and both GPUs: the third one
    let only_igpu = vec![
        dev(0, "CPU", "CPU", DeviceKind::Cpu, 0),
        dev(1, "Intel UHD", "Vulkan", DeviceKind::Igpu, 1_000),
    ];
    assert!(choose_gpu(&only_igpu, false).is_none());
}

#[test]
fn a_build_without_a_gpu_has_nothing_to_choose() {
    let all = vec![dev(0, "CPU", "CPU", DeviceKind::Cpu, 0)];
    assert!(choose_gpu(&all, true).is_none());
    assert!(choose_gpu(&all, false).is_none());
    assert!(choose_gpu(&[], true).is_none());
}

#[test]
fn backend_says_where_it_runs_and_whether_the_gpu_was_refused() {
    let metal = dev(0, "Apple M3 Pro", "Metal", DeviceKind::Gpu, 24_000);
    let on_gpu = describe_backend(Some(&metal), true);
    assert_eq!(on_gpu.backend, "metal");
    assert_eq!(on_gpu.device.as_deref(), Some("Apple M3 Pro"));
    assert!(!on_gpu.fell_back);
    let refused = describe_backend(None, true);
    assert_eq!(
        (refused.backend.as_str(), refused.device, refused.fell_back),
        ("cpu", None, true)
    );
    let chosen = describe_backend(None, false);
    assert_eq!(
        (chosen.backend.as_str(), chosen.device, chosen.fell_back),
        ("cpu", None, false)
    );
}

// ---- loading errors ----

#[test]
fn a_missing_model_file_is_model_missing() {
    let dir = tempfile::tempdir().unwrap();
    let err = WhisperEngine::load(&dir.path().join("nope.bin"), LoadOptions::default())
        .err()
        .unwrap();
    assert_eq!(err, SpeechError::ModelMissing);
}

#[test]
fn a_file_that_is_not_a_whisper_model_is_bad_model() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("broken.bin");
    std::fs::write(&file, vec![7u8; 4096]).unwrap();
    for gpu in [false, true] {
        let err = WhisperEngine::load(
            &file,
            LoadOptions {
                gpu,
                threads: Some(2),
            },
        )
        .err()
        .unwrap();
        assert_eq!(err, SpeechError::BadModel, "gpu {gpu}");
    }
}

#[test]
fn load_dispatches_on_the_kind() {
    use voice_engine_models::catalog::{Kind, find};
    let dir = tempfile::tempdir().unwrap();
    let whisper = find("whisper-turbo").unwrap();
    assert_eq!(whisper.kind, Kind::Whisper);
    // The folder is empty: Whisper looks for the catalog's file and says it is missing.
    assert_eq!(
        load(whisper, dir.path(), LoadOptions::default()).err(),
        Some(SpeechError::ModelMissing)
    );
    // Not a speech model.
    let llm = voice_engine_models::catalog::local_ai().next().unwrap();
    assert!(matches!(
        load(llm, dir.path(), LoadOptions::default()).err(),
        Some(SpeechError::Unsupported(_))
    ));
}

#[test]
fn the_api_types_are_what_the_spec_says() {
    let d = ChunkOptions::default();
    assert_eq!(d.language, None);
    assert_eq!(d.timing, Timing::Words);
    assert!(!d.keep_fillers);
    assert!(d.vocabulary.is_empty());
    fn is_send<T: Send>() {}
    is_send::<Box<dyn Transcriber>>();
    // Display never carries more than the kind and the engine's own code.
    assert_eq!(
        SpeechError::Internal("whisper.cpp error -3".into()).to_string(),
        "speech engine error: whisper.cpp error -3"
    );
    for e in [
        SpeechError::ModelMissing,
        SpeechError::BadModel,
        SpeechError::OutOfMemory,
    ] {
        assert!(!e.to_string().is_empty());
    }
}

// ---- starts_word and join_segments ----

fn seg(text: &str, starts_word: bool) -> Segment {
    Segment {
        start_ms: 0,
        end_ms: 100,
        text: text.into(),
        starts_word,
        words: vec![],
    }
}

#[test]
fn segments_mode_keeps_text_trimmed_and_says_whether_whisper_put_a_space_before() {
    let segs = vec![
        raw(" Hello there.", 0, 900, 0.0, vec![]),
        raw(" Second one.", 1_000, 1_900, 0.0, vec![]),
        // As whisper.cpp gives Chinese, Japanese or Korean: no space before the segment.
        raw("\u{4f60}\u{597d}", 2_000, 2_500, 0.0, vec![]),
        raw("\u{4e16}\u{754c}", 2_500, 3_000, 0.0, vec![]),
    ];
    let got = segments_from_raw(segs, Timing::Segments, 5_000);
    assert_eq!(
        got.iter().map(|s| s.starts_word).collect::<Vec<_>>(),
        vec![true, true, false, false]
    );
    assert!(got.iter().all(|s| s.text == s.text.trim()));
    assert_eq!(
        join_segments(&got),
        "Hello there. Second one.\u{4f60}\u{597d}\u{4e16}\u{754c}"
    );
}

#[test]
fn words_mode_carries_the_flag_through_clean() {
    let segs = vec![
        raw(" Okay", 0, 300, 0.0, vec![]),
        raw(".", 300, 320, 0.0, vec![]),
        raw(" [BLANK_AUDIO]", 320, 400, 0.0, vec![]),
        raw(" So", 400, 600, 0.0, vec![]),
        raw("on", 600, 700, 0.0, vec![]),
    ];
    let got = segments_from_raw(segs, Timing::Words, 5_000);
    // "Okay." (the full stop glued on), "So" after the dropped noise, "on" with no space before it.
    assert_eq!(
        got.iter()
            .map(|s| (s.text.as_str(), s.starts_word))
            .collect::<Vec<_>>(),
        vec![("Okay.", true), ("So", true), ("on", false)]
    );
    assert_eq!(join_segments(&got), "Okay. Soon");
}

#[test]
fn join_segments_puts_a_space_only_where_a_segment_starts_a_word() {
    assert_eq!(join_segments(&[]), "");
    // The first segment's flag does not matter: nothing goes before it.
    assert_eq!(join_segments(&[seg("a", true), seg("b", true)]), "a b");
    assert_eq!(join_segments(&[seg("a", false), seg("b", false)]), "ab");
    assert_eq!(
        join_segments(&[seg("a", true), seg("b", false), seg("c", true)]),
        "ab c"
    );
}

#[test]
fn join_segments_skips_empty_text_and_trims() {
    let segs = [
        seg("a", true),
        seg("  ", true),
        seg(" b ", true),
        seg("", false),
        seg("c", false),
    ];
    assert_eq!(join_segments(&segs), "a bc");
}

#[test]
fn a_segment_from_an_older_helper_reads_as_starting_a_word() {
    let s: Segment =
        serde_json::from_str(r#"{"start_ms":0,"end_ms":10,"text":"x","words":[]}"#).unwrap();
    assert!(s.starts_word);
    let back: Segment =
        serde_json::from_str(&serde_json::to_string(&seg("y", false)).unwrap()).unwrap();
    assert!(!back.starts_word);
}
