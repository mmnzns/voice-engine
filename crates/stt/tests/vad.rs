//! Voice detection tests that need no real model: the pure parts (centiseconds to milliseconds, span clean-up),
//! the edge cases that never reach the detector (empty audio, audio shorter than one frame), and the error kinds.
//! The detector itself runs on a real model in `tests/real_whisper.rs` (ignored).

use voice_engine_models::catalog::find;
use voice_engine_stt::SpeechError;
use voice_engine_stt::vad::{
    FRAME_SAMPLES, VadSettings, centis_to_ms, detect, detect_with, is_shorter_than_a_frame,
    pad_spans, spans_from_centiseconds,
};

/// The file name the catalog gives the voice detection model.
fn vad_file_name() -> &'static str {
    find("vad").unwrap().files[0].name
}

// ---- centiseconds to milliseconds ----

#[test]
fn whisper_cpp_reports_centiseconds_so_ten_milliseconds_each() {
    assert_eq!(centis_to_ms(0.0), Some(0));
    assert_eq!(centis_to_ms(1.0), Some(10));
    assert_eq!(centis_to_ms(125.0), Some(1_250));
    // Fractions are rounded to the nearest millisecond (a frame is 32 ms, not a whole centisecond).
    assert_eq!(centis_to_ms(3.2), Some(32));
    assert_eq!(centis_to_ms(3.24), Some(32));
    assert_eq!(centis_to_ms(3.26), Some(33));
}

#[test]
fn a_time_that_is_not_a_time_is_dropped() {
    assert_eq!(centis_to_ms(f32::NAN), None);
    assert_eq!(centis_to_ms(f32::INFINITY), None);
    assert_eq!(centis_to_ms(f32::NEG_INFINITY), None);
    assert_eq!(centis_to_ms(-0.5), None);
}

// ---- span clean-up ----

#[test]
fn spans_come_out_in_milliseconds() {
    let raw = [(10.0, 150.0), (300.0, 420.5)];
    assert_eq!(
        spans_from_centiseconds(&raw, 10_000),
        vec![(100, 1_500), (3_000, 4_205)]
    );
}

#[test]
fn no_spans_in_no_spans_out() {
    assert!(spans_from_centiseconds(&[], 10_000).is_empty());
}

#[test]
fn spans_never_run_past_the_real_audio() {
    // whisper.cpp counts whole frames, so the last span can end a little after the audio does.
    let raw = [(100.0, 1_030.0)];
    assert_eq!(spans_from_centiseconds(&raw, 10_000), vec![(1_000, 10_000)]);
    // One that starts after the audio ended is not a span at all.
    assert!(spans_from_centiseconds(&[(1_100.0, 1_200.0)], 10_000).is_empty());
}

#[test]
fn empty_and_backwards_spans_are_dropped() {
    let raw = [
        (50.0, 50.0),
        (80.0, 60.0),
        (100.0, 200.0),
        (f32::NAN, 400.0),
    ];
    assert_eq!(spans_from_centiseconds(&raw, 10_000), vec![(1_000, 2_000)]);
}

#[test]
fn spans_are_sorted_and_overlapping_or_touching_ones_are_merged() {
    // Out of order, one overlapping, one touching exactly, one apart.
    let raw = [
        (500.0, 600.0),
        (100.0, 200.0),
        (150.0, 260.0),
        (260.0, 300.0),
    ];
    assert_eq!(
        spans_from_centiseconds(&raw, 10_000),
        vec![(1_000, 3_000), (5_000, 6_000)]
    );
}

#[test]
fn a_span_inside_another_is_absorbed() {
    let raw = [(100.0, 500.0), (200.0, 300.0)];
    assert_eq!(spans_from_centiseconds(&raw, 10_000), vec![(1_000, 5_000)]);
}

// ---- settings and padding ----

#[test]
fn the_default_settings_are_the_ones_detect_has_always_used() {
    let d = VadSettings::default();
    assert_eq!(d.threshold, 0.5);
    assert_eq!(d.min_speech_ms, 250);
    assert_eq!(d.min_silence_ms, 300);
    assert_eq!(d.padding_ms, 0);
}

#[test]
fn no_padding_leaves_the_spans_as_they_are() {
    let spans = [(1_000, 2_000), (5_000, 6_000)];
    assert_eq!(pad_spans(&spans, 0, 10_000), spans.to_vec());
}

#[test]
fn padding_widens_each_span_on_both_sides() {
    let spans = [(1_000, 2_000), (5_000, 6_000)];
    assert_eq!(
        pad_spans(&spans, 30, 10_000),
        vec![(970, 2_030), (4_970, 6_030)]
    );
}

#[test]
fn padding_stops_at_the_ends_of_the_audio() {
    assert_eq!(pad_spans(&[(10, 9_990)], 30, 10_000), vec![(0, 10_000)]);
    assert_eq!(pad_spans(&[(0, 100)], 500, 400), vec![(0, 400)]);
}

#[test]
fn spans_that_meet_after_padding_become_one() {
    // 100 ms apart: 30 ms each side leaves 40 ms (kept apart), 50 ms each side touches (merged), 60 overlaps.
    let spans = [(1_000, 2_000), (2_100, 3_000), (8_000, 9_000)];
    assert_eq!(
        pad_spans(&spans, 30, 10_000),
        vec![(970, 2_030), (2_070, 3_030), (7_970, 9_030)]
    );
    assert_eq!(
        pad_spans(&spans, 50, 10_000),
        vec![(950, 3_050), (7_950, 9_050)]
    );
    assert_eq!(
        pad_spans(&spans, 60, 10_000),
        vec![(940, 3_060), (7_940, 9_060)]
    );
    // A big pad swallows the lot.
    assert_eq!(pad_spans(&spans, 5_000, 10_000), vec![(0, 10_000)]);
}

#[test]
fn no_spans_stay_no_spans_when_padded() {
    assert!(pad_spans(&[], 100, 10_000).is_empty());
}

#[test]
fn custom_settings_keep_the_edge_cases_that_never_reach_the_detector() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(vad_file_name()), b"not a model").unwrap();
    let wide = VadSettings {
        min_silence_ms: 100,
        padding_ms: 30,
        ..VadSettings::default()
    };
    assert_eq!(detect_with(dir.path(), &[], &wide), Ok(vec![]));
    assert_eq!(detect_with(dir.path(), &[0.9; 100], &wide), Ok(vec![]));
    assert_eq!(
        detect_with(&dir.path().join("nowhere"), &[0.0; 16_000], &wide),
        Err(SpeechError::ModelMissing)
    );
}

// ---- edge cases that never reach the detector ----

#[test]
fn a_frame_is_512_samples_at_16_khz() {
    assert_eq!(FRAME_SAMPLES, 512);
    assert!(is_shorter_than_a_frame(&[]));
    assert!(is_shorter_than_a_frame(&[0.0; 511]));
    assert!(!is_shorter_than_a_frame(&[0.0; 512]));
    assert!(!is_shorter_than_a_frame(&[0.0; 16_000]));
}

#[test]
fn empty_audio_has_no_speech() {
    // The file is there but is not a model: if detect loaded it, this would be `BadModel`.
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(vad_file_name()), b"not a model").unwrap();
    assert_eq!(detect(dir.path(), &[]), Ok(vec![]));
}

#[test]
fn audio_shorter_than_a_frame_has_no_speech_without_loading_the_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(vad_file_name()), b"not a model").unwrap();
    for len in [1, 100, FRAME_SAMPLES - 1] {
        let loud = vec![0.9f32; len];
        assert_eq!(detect(dir.path(), &loud), Ok(vec![]), "{len} samples");
    }
}

// ---- error kinds ----

#[test]
fn a_missing_model_file_is_model_missing() {
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        detect(dir.path(), &[0.0; 16_000]),
        Err(SpeechError::ModelMissing)
    );
    // The folder itself missing is the same kind.
    assert_eq!(
        detect(&dir.path().join("nowhere"), &[0.0; 16_000]),
        Err(SpeechError::ModelMissing)
    );
    // Even for audio that would not need the model: a broken install shows up at once.
    assert_eq!(detect(dir.path(), &[]), Err(SpeechError::ModelMissing));
}

#[test]
fn a_file_that_is_not_a_model_is_bad_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join(vad_file_name()), b"this is not a ggml file").unwrap();
    assert_eq!(
        detect(dir.path(), &[0.0; 16_000]),
        Err(SpeechError::BadModel)
    );
}

#[test]
fn the_error_messages_say_nothing_about_the_file_or_the_audio() {
    let dir = tempfile::tempdir().unwrap();
    let e = detect(dir.path(), &[0.0; 16_000]).unwrap_err();
    let shown = e.to_string();
    assert!(!shown.contains(&*dir.path().to_string_lossy()), "{shown}");
    assert!(!shown.contains(vad_file_name()), "{shown}");
}
