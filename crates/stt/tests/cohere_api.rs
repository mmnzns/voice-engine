//! Cohere Transcribe tests that need no real model: the language choice, the shape of the transcript, the files a load
//! looks for, the errors of a load, and that the kinds route as they should. The model itself runs in
//! `tests/real_cohere.rs` (ignored).

use voice_engine_models::catalog::{Kind, find};
use voice_engine_stt::cohere_engine::{
    CohereEngine, DEFAULT_LANGUAGE, DEFAULT_MAX_SECS, check_files, pick_language,
    transcript_from_text,
};
use voice_engine_stt::{LoadOptions, SpeechError, load};

fn supported() -> Vec<String> {
    [
        "ar", "de", "el", "en", "es", "fr", "it", "ja", "ko", "nl", "pl", "pt", "vi", "zh",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn pick(hint: Option<&str>) -> Option<String> {
    pick_language(hint, &supported())
}

#[test]
fn no_hint_means_english_because_the_model_cannot_detect() {
    assert_eq!(DEFAULT_LANGUAGE, "en");
    assert_eq!(pick(None).as_deref(), Some("en"));
    assert_eq!(pick(Some("")).as_deref(), Some("en"));
    assert_eq!(pick(Some("  ")).as_deref(), Some("en"));
    assert_eq!(pick(Some("auto")).as_deref(), Some("en"));
    assert_eq!(pick(Some("AUTO")).as_deref(), Some("en"));
}

#[test]
fn a_supported_language_is_used_in_any_spelling() {
    for (hint, want) in [
        ("fr", "fr"),
        ("FR", "fr"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("pt-BR", "pt"),
        ("zh-Hant", "zh"),
        (" ja ", "ja"),
        ("de-DE", "de"),
    ] {
        assert_eq!(pick(Some(hint)).as_deref(), Some(want), "{hint}");
    }
    for code in supported() {
        assert_eq!(pick(Some(&code)).as_deref(), Some(code.as_str()));
    }
}

#[test]
fn a_language_the_model_lacks_is_none() {
    for hint in ["tl", "fil", "ru", "xx", "english", "-", "tl-PH"] {
        assert_eq!(pick(Some(hint)), None, "{hint}");
    }
    // Nothing supported: nothing is picked for a language.
    assert_eq!(pick_language(Some("fr"), &[]), None);
}

#[test]
fn text_is_one_segment_over_the_whole_chunk_with_no_words() {
    let t = transcript_from_text("  Some text.  ", "en", 12_345);
    assert_eq!(t.segments.len(), 1);
    let s = &t.segments[0];
    assert_eq!((s.start_ms, s.end_ms), (0, 12_345));
    assert_eq!(s.text, "Some text.");
    assert!(s.starts_word);
    assert!(s.words.is_empty());
    assert_eq!(t.language.as_deref(), Some("en"));
    assert_eq!(t.stopped_early_at_ms, None);
}

#[test]
fn empty_and_noise_text_gives_no_segments_but_keeps_the_language() {
    for text in ["", "   ", "[BLANK_AUDIO]", "(music)", "...", "?!"] {
        let t = transcript_from_text(text, "fr", 1_000);
        assert!(t.segments.is_empty(), "{text:?}");
        assert_eq!(t.language.as_deref(), Some("fr"));
    }
}

#[test]
fn the_catalog_entry_says_text_only_35_seconds() {
    let m = find("cohere-transcribe").unwrap();
    assert_eq!(m.kind, Kind::Cohere);
    assert_eq!(m.chunk_secs, Some(DEFAULT_MAX_SECS));
    assert!(!m.word_times);
    assert!(!m.english_only);
    assert!(m.is_speech());
}

#[test]
fn every_catalog_file_must_be_there() {
    let m = find("cohere-transcribe").unwrap();
    let names: Vec<&str> = m.files.iter().map(|f| f.name).collect();
    assert_eq!(check_files(m, &|n| names.contains(&n)), Ok(()));
    for gone in m.files {
        let got = check_files(m, &|n| n != gone.name && names.contains(&n));
        assert_eq!(got, Err(SpeechError::ModelMissing), "{}", gone.name);
    }
    assert_eq!(check_files(m, &|_| false), Err(SpeechError::ModelMissing));
}

#[test]
fn a_folder_without_the_files_is_model_missing() {
    let m = find("cohere-transcribe").unwrap();
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(
        CohereEngine::load(m, dir.path(), LoadOptions::default())
            .err()
            .unwrap(),
        SpeechError::ModelMissing
    );
    assert_eq!(
        load(m, &dir.path().join("not-there"), LoadOptions::default())
            .err()
            .unwrap(),
        SpeechError::ModelMissing
    );
}

#[test]
fn files_that_are_not_models_do_not_load_and_do_not_crash() {
    let m = find("cohere-transcribe").unwrap();
    let dir = tempfile::tempdir().unwrap();
    for f in m.files {
        std::fs::write(dir.path().join(f.name), b"this is not a model").unwrap();
    }
    let err = CohereEngine::load(m, dir.path(), LoadOptions::default())
        .err()
        .unwrap();
    assert!(
        matches!(err, SpeechError::BadModel | SpeechError::Internal(_)),
        "{err:?}"
    );
}

#[test]
fn another_kind_is_not_a_cohere_model() {
    let dir = tempfile::tempdir().unwrap();
    let whisper = find("whisper-tiny").unwrap();
    assert!(matches!(
        CohereEngine::load(whisper, dir.path(), LoadOptions::default())
            .err()
            .unwrap(),
        SpeechError::Unsupported(_)
    ));
}

#[test]
fn load_routes_cohere_to_the_engine_and_multitalker_to_its_own_api() {
    let dir = tempfile::tempdir().unwrap();
    // Cohere goes to the engine, which finds the folder empty.
    assert_eq!(
        load(
            find("cohere-transcribe").unwrap(),
            dir.path(),
            LoadOptions::default()
        )
        .err()
        .unwrap(),
        SpeechError::ModelMissing
    );
    // Multitalker is not a Transcriber: the answer points at its own load, and never looks at the folder.
    let err = load(
        find("multitalker-tdt").unwrap(),
        dir.path(),
        LoadOptions::default(),
    )
    .err()
    .unwrap();
    assert_eq!(
        err,
        SpeechError::Unsupported("use multitalker::load".into())
    );
    // The kinds that are not speech models.
    for id in ["vad", "diarization", "search", "qwen3.5-4b", "gemma-4-e4b"] {
        let err = load(find(id).unwrap(), dir.path(), LoadOptions::default())
            .err()
            .unwrap();
        assert!(matches!(err, SpeechError::Unsupported(_)), "{id}");
    }
}

#[test]
fn an_engine_can_move_to_a_worker_thread() {
    fn is_send<T: Send>() {}
    is_send::<CohereEngine>();
}
