//! Multitalker tests that need no real model: parakeet-rs's utterances turned into turns (milliseconds, clamping,
//! punctuation, overlap), and the errors of a load. The model itself runs in `tests/real_multitalker.rs` (ignored).

use parakeet_rs::{SpeakerUtterance, TimedToken};
use voice_engine_models::ModelInfo;
use voice_engine_models::catalog::{Kind, find};
use voice_engine_stt::SpeechError;
use voice_engine_stt::multitalker::{Multitalker, SpeakerTurn, load, turns_from_utterances};

fn token(text: &str, start: f32, end: f32) -> TimedToken {
    TimedToken {
        text: text.into(),
        start,
        end,
    }
}

fn utterance(speaker_id: usize, words: Vec<TimedToken>) -> SpeakerUtterance {
    SpeakerUtterance {
        speaker_id,
        start: words.first().map_or(0.0, |w| w.start),
        end: words.last().map_or(0.0, |w| w.end),
        text: words
            .iter()
            .map(|w| w.text.as_str())
            .collect::<Vec<_>>()
            .join(" "),
        words,
    }
}

#[test]
fn seconds_become_whole_milliseconds_and_words_keep_their_order() {
    let turns = turns_from_utterances(
        vec![utterance(
            2,
            vec![
                token("one", 0.0, 0.4),
                token("two", 0.5, 0.9),
                token("three", 1.2345, 1.6789),
            ],
        )],
        10_000,
    );
    assert_eq!(turns.len(), 1);
    let t = &turns[0];
    assert_eq!(t.speaker, 2);
    assert_eq!((t.start_ms, t.end_ms), (0, 1_678));
    assert_eq!(t.text, "one two three");
    let times: Vec<_> = t.words.iter().map(|w| (w.start_ms, w.end_ms)).collect();
    assert_eq!(times, [(0, 400), (500, 900), (1_234, 1_678)]);
    // The text is the words joined, so the two never disagree.
    let joined: Vec<_> = t.words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(t.text, joined.join(" "));
}

#[test]
fn punctuation_is_glued_to_the_word_before() {
    let turns = turns_from_utterances(
        vec![utterance(
            0,
            vec![
                token("Hello", 0.0, 0.4),
                token(",", 0.4, 0.4),
                token("there", 0.5, 0.9),
                token(".", 0.9, 0.9),
            ],
        )],
        5_000,
    );
    assert_eq!(turns[0].text, "Hello, there.");
    let words: Vec<_> = turns[0].words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(words, ["Hello,", "there."]);
}

#[test]
fn overlapping_speakers_give_overlapping_turns_in_the_order_given() {
    let turns = turns_from_utterances(
        vec![
            utterance(0, vec![token("a", 0.0, 0.5), token("b", 0.6, 3.0)]),
            utterance(1, vec![token("c", 1.0, 1.5), token("d", 1.6, 4.0)]),
        ],
        10_000,
    );
    assert_eq!(turns.len(), 2);
    assert_eq!((turns[0].speaker, turns[1].speaker), (0, 1));
    assert!(turns[1].start_ms < turns[0].end_ms, "they overlap");
    assert_eq!((turns[0].start_ms, turns[0].end_ms), (0, 3_000));
    assert_eq!((turns[1].start_ms, turns[1].end_ms), (1_000, 4_000));
}

#[test]
fn times_stay_inside_the_audio_and_an_end_is_never_before_its_start() {
    let turns = turns_from_utterances(
        vec![utterance(
            3,
            vec![
                token("late", 4.5, 9.0),
                token("past", 9.5, 12.0),
                token("backwards", 2.0, 1.0),
                token("negative", -1.0, -0.5),
            ],
        )],
        5_000,
    );
    assert!(!turns.is_empty());
    for t in &turns {
        assert!(t.start_ms <= t.end_ms && t.end_ms <= 5_000);
        for w in &t.words {
            assert!(w.start_ms <= w.end_ms && w.end_ms <= 5_000);
        }
    }
}

#[test]
fn a_turn_with_no_words_or_only_noise_is_dropped() {
    let turns = turns_from_utterances(
        vec![
            utterance(0, vec![]),
            utterance(1, vec![token("[BLANK_AUDIO]", 0.0, 1.0)]),
            utterance(2, vec![token("  ", 1.0, 2.0)]),
            utterance(3, vec![token("kept", 2.0, 3.0)]),
        ],
        10_000,
    );
    assert_eq!(turns.len(), 1);
    assert_eq!(turns[0].speaker, 3);
    assert!(turns_from_utterances(vec![], 1_000).is_empty());
}

// ---- the files ----

fn entry() -> &'static ModelInfo {
    find("multitalker-tdt").unwrap()
}

fn diarization_file() -> &'static str {
    find("diarization").unwrap().files[0].name
}

/// A folder with every file of a catalog entry, each holding `body`.
fn folder_with(model: &ModelInfo, body: &[u8]) -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    for f in model.files {
        std::fs::write(dir.path().join(f.name), body).unwrap();
    }
    dir
}

#[test]
fn the_catalog_entry_is_multitalker_and_not_plain_speech() {
    let m = entry();
    assert_eq!(m.kind, Kind::Multitalker);
    assert!(!m.is_speech());
    assert!(m.word_times);
    assert_eq!(m.chunk_secs, None);
}

#[test]
fn a_missing_file_in_either_folder_is_model_missing() {
    let model_dir = folder_with(entry(), b"x");
    let diar_dir = folder_with(find("diarization").unwrap(), b"x");
    // The speaker separation model is not there.
    let empty = tempfile::tempdir().unwrap();
    assert_eq!(
        load(entry(), model_dir.path(), empty.path(), None)
            .err()
            .unwrap(),
        SpeechError::ModelMissing
    );
    // The Multitalker files are not there.
    assert_eq!(
        load(entry(), empty.path(), diar_dir.path(), Some(2))
            .err()
            .unwrap(),
        SpeechError::ModelMissing
    );
    // One Multitalker file is missing.
    for gone in entry().files {
        std::fs::remove_file(model_dir.path().join(gone.name)).unwrap();
        assert_eq!(
            load(entry(), model_dir.path(), diar_dir.path(), None)
                .err()
                .unwrap(),
            SpeechError::ModelMissing,
            "{}",
            gone.name
        );
        std::fs::write(model_dir.path().join(gone.name), b"x").unwrap();
    }
    assert_eq!(
        load(
            entry(),
            &model_dir.path().join("not-there"),
            diar_dir.path(),
            None
        )
        .err()
        .unwrap(),
        SpeechError::ModelMissing
    );
}

#[test]
fn an_empty_file_is_bad_model() {
    let model_dir = folder_with(entry(), b"x");
    let diar_dir = folder_with(find("diarization").unwrap(), b"x");
    std::fs::write(diar_dir.path().join(diarization_file()), b"").unwrap();
    assert_eq!(
        load(entry(), model_dir.path(), diar_dir.path(), None)
            .err()
            .unwrap(),
        SpeechError::BadModel
    );
    let diar_dir = folder_with(find("diarization").unwrap(), b"x");
    std::fs::write(model_dir.path().join(entry().files[0].name), b"").unwrap();
    assert_eq!(
        load(entry(), model_dir.path(), diar_dir.path(), None)
            .err()
            .unwrap(),
        SpeechError::BadModel
    );
}

#[test]
fn files_that_are_not_models_do_not_load_and_say_no_file_name() {
    let secret = "private-meeting-folder";
    let root = tempfile::tempdir().unwrap();
    let model_dir = root.path().join(secret);
    std::fs::create_dir(&model_dir).unwrap();
    for f in entry().files {
        std::fs::write(model_dir.join(f.name), b"this is not a model").unwrap();
    }
    let diar_dir = folder_with(find("diarization").unwrap(), b"this is not a model");
    let err = load(entry(), &model_dir, diar_dir.path(), None)
        .err()
        .unwrap();
    assert!(
        matches!(err, SpeechError::BadModel | SpeechError::Internal(_)),
        "{err:?}"
    );
    assert!(!format!("{err} {err:?}").contains(secret));
}

#[test]
fn another_kind_is_not_a_multitalker_model() {
    let dir = tempfile::tempdir().unwrap();
    for id in ["parakeet-v3", "cohere-transcribe", "diarization"] {
        let err = load(find(id).unwrap(), dir.path(), dir.path(), None)
            .err()
            .unwrap();
        assert!(matches!(err, SpeechError::Unsupported(_)), "{id}");
    }
}

#[test]
fn a_model_can_move_to_a_worker_thread() {
    fn is_send<T: Send>() {}
    is_send::<Multitalker>();
    let _ = SpeakerTurn {
        speaker: 0,
        start_ms: 0,
        end_ms: 0,
        text: String::new(),
        words: vec![],
    };
}
