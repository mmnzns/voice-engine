//! `SttClient` with the real speech helper and a real Whisper model. Ignored by default; run on the Mac before a tag:
//!   cargo build --release -p voice-engine-stt
//!   VOICE_ENGINE_TEST_STT_HELPER="$PWD/target/release/voice-engine-stt" \
//!   cargo test -p voice-engine-client --release --test real_stt -- --ignored --test-threads=1 --nocapture
//!
//! The model is read from `$VOICE_ENGINE_TEST_MODELS` (default `~/Library/Application Support/Talaan/models`), never
//! written: each test lays a folder of its own in a temporary directory with a symbolic link to the model file, so
//! the real folder is only read. Speech comes from the macOS `say` command, so this file is Mac only.

#![cfg(target_os = "macos")]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use voice_engine_client::{EngineError, Limits, SttClient, SttConfig};
use voice_engine_protocol::stt::{ChunkOptions, SttLoad, SttModelKind, Timing, Transcript};

const SENTENCE: &str =
    "Hello, this is a test of the speech helper. We meet on Thursday at three o'clock.";

fn helper() -> PathBuf {
    match std::env::var_os("VOICE_ENGINE_TEST_STT_HELPER") {
        Some(p) => PathBuf::from(p),
        None => Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/release/voice-engine-stt"),
    }
}

fn models_dir() -> PathBuf {
    match std::env::var_os("VOICE_ENGINE_TEST_MODELS") {
        Some(d) => PathBuf::from(d),
        None => dirs::data_dir().unwrap().join("Talaan/models"),
    }
}

fn client() -> SttClient {
    SttClient::new(SttConfig {
        cpu_helper: helper(),
        gpu_helper: None,
        env: vec![],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        limits: Limits::default(),
    })
}

/// A folder in a temporary directory that holds a link to the real model file (the real folder is only read).
fn model_folder(id: &str) -> (tempfile::TempDir, SttLoad, PathBuf) {
    let info = voice_engine_models::catalog::find(id).unwrap();
    let real = models_dir().join(id).join(info.files[0].name);
    assert!(real.is_file(), "model not found: {}", real.display());
    let dir = tempfile::tempdir().unwrap();
    let link = dir.path().join(info.files[0].name);
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let load = SttLoad {
        model_id: id.into(),
        model_dir: dir.path().display().to_string(),
        kind: SttModelKind::Whisper,
        gpu: true,
        threads: None,
    };
    (dir, load, link)
}

/// 16 kHz mono f32 samples of `text` spoken by the `say` command.
fn say(text: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let aiff = dir.path().join("clip.aiff");
    let wav = dir.path().join("clip.wav");
    assert!(
        Command::new("say")
            .arg("-o")
            .arg(&aiff)
            .arg(text)
            .status()
            .unwrap()
            .success()
    );
    assert!(
        Command::new("afconvert")
            .args(["-f", "WAVE", "-d", "LEF32@16000", "-c", "1"])
            .arg(&aiff)
            .arg(&wav)
            .status()
            .unwrap()
            .success()
    );
    let b = std::fs::read(&wav).unwrap();
    let mut i = 12;
    while i + 8 <= b.len() {
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        if &b[i..i + 4] == b"data" {
            let end = (i + 8 + len).min(b.len());
            return b[i + 8..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
        }
        i += 8 + len + (len & 1);
    }
    panic!("no data chunk in the wav");
}

fn text(t: &Transcript) -> String {
    t.segments
        .iter()
        .map(|s| s.text.trim())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

fn ms(clip: &[f32]) -> u64 {
    clip.len() as u64 * 1000 / 16_000
}

/// Every word has a time on the clip's clock, in order.
fn check_words(t: &Transcript, clip_ms: u64) {
    let words: Vec<_> = t.segments.iter().flat_map(|s| &s.words).collect();
    assert!(words.len() >= 10, "{} words", words.len());
    let mut last = 0;
    for w in words {
        let info = format!(
            "word {}..{} ms, {} chars",
            w.start_ms,
            w.end_ms,
            w.text.len()
        );
        assert!(w.start_ms >= last && w.start_ms <= w.end_ms, "{info}");
        assert!(w.end_ms <= clip_ms, "{info} past the {clip_ms} ms clip");
        last = w.start_ms;
    }
}

#[test]
#[ignore]
fn a_say_clip_through_the_real_helper_and_the_model_stays_loaded_between_calls() {
    let clip = say(SENTENCE);
    let (_dir, load, link) = model_folder("whisper-turbo");
    let c = client();
    let began = Instant::now();
    let loaded = c.load(load).unwrap();
    eprintln!(
        "load: {:.2} s reported, {:.2} s on the clock, backend {:?}, fell_back {}",
        loaded.value.seconds,
        began.elapsed().as_secs_f32(),
        loaded.backend,
        loaded.fell_back
    );
    assert!(loaded.value.seconds > 0.0);
    assert_eq!(loaded.backend.backend, "metal");
    assert!(!loaded.fell_back);

    // The link goes away: from here the model can only be in the helper's memory. A helper that read the file again
    // for each call (or a client that started a new helper) would answer `model_missing`.
    std::fs::remove_file(&link).unwrap();

    let began = Instant::now();
    let words = c
        .transcribe(
            &clip,
            &ChunkOptions {
                timing: Timing::Words,
                language: Some("en".into()),
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    eprintln!(
        // Counts and times only: nothing a person said goes into a log.
        "words ({:.2} s, {:?}): {} words, {} chars",
        began.elapsed().as_secs_f32(),
        words.backend,
        words
            .value
            .segments
            .iter()
            .map(|s| s.words.len())
            .sum::<usize>(),
        text(&words.value).chars().count()
    );
    let said = text(&words.value);
    assert!(
        said.contains("thursday"),
        "thursday not heard in {} chars",
        said.len()
    );
    assert!(
        said.contains("speech helper"),
        "speech helper not heard in {} chars",
        said.len()
    );
    check_words(&words.value, ms(&clip));
    assert!(words.value.segments.iter().all(|s| s.words.len() == 1));
    assert_eq!(words.backend.backend, "metal");
    assert!(!words.fell_back);

    let segments = c
        .transcribe(
            &clip,
            &ChunkOptions {
                timing: Timing::Segments,
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    eprintln!(
        "segments ({:?}): {} chars",
        segments
            .value
            .segments
            .iter()
            .map(|s| (s.start_ms, s.end_ms))
            .collect::<Vec<_>>(),
        text(&segments.value).chars().count()
    );
    assert!(text(&segments.value).contains("thursday"));
    assert!(segments.value.segments.iter().any(|s| s.words.len() > 3));
    assert_eq!(segments.value.language.as_deref(), Some("en"));
    check_words(&segments.value, ms(&clip));

    // Unload frees it; with nothing loaded a transcribe is a bad request, and says so without starting anything.
    c.unload().unwrap();
    assert_eq!(
        c.transcribe(&clip, &ChunkOptions::default()).unwrap_err(),
        EngineError::BadRequest
    );
    c.shutdown();
}

#[test]
#[ignore]
fn a_missing_model_and_a_file_that_is_not_a_model_are_their_own_kinds() {
    let c = client();
    let dir = tempfile::tempdir().unwrap();
    let load = SttLoad {
        model_id: "whisper-turbo".into(),
        model_dir: dir.path().display().to_string(),
        kind: SttModelKind::Whisper,
        gpu: true,
        threads: None,
    };
    let info = voice_engine_models::catalog::find("whisper-turbo").unwrap();
    let want = dir.path().join(info.files[0].name).display().to_string();
    assert_eq!(
        c.load(load.clone()).unwrap_err(),
        EngineError::ModelMissing { path: want }
    );
    std::fs::write(dir.path().join(info.files[0].name), b"not a ggml file").unwrap();
    assert_eq!(c.load(load).unwrap_err(), EngineError::BadModel);
    // The ONNX models are not served by the helper.
    let e = c
        .load(SttLoad {
            model_id: "parakeet-ultra".into(),
            model_dir: "/m".into(),
            kind: SttModelKind::Parakeet,
            gpu: false,
            threads: None,
        })
        .unwrap_err();
    assert!(matches!(e, EngineError::Unsupported { .. }), "{e:?}");
    c.shutdown();
}

#[test]
#[ignore]
fn cancel_a_long_transcription_then_transcribe_again_with_the_model_loaded_back_in() {
    let clip = say(SENTENCE);
    // About six minutes: a decode that takes seconds, not milliseconds.
    let long: Vec<f32> = clip.iter().copied().cycle().take(clip.len() * 60).collect();
    let (_dir, load, _link) = model_folder("whisper-turbo");
    let c = client();
    c.load(load).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| c.transcribe(&long, &ChunkOptions::default()));
        std::thread::sleep(Duration::from_millis(800));
        c.cancel();
        worker.join().unwrap()
    });
    let took = began.elapsed();
    eprintln!(
        "cancelled after {:.2} s: {:?}",
        took.as_secs_f32(),
        result.as_ref().err()
    );
    // A Whisper decode cannot be interrupted, so the helper was stopped after the 2 s grace.
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    assert!(
        took >= Duration::from_secs(2) && took < Duration::from_secs(20),
        "{took:?}"
    );

    // The next call puts the model into a new helper itself and answers.
    let began = Instant::now();
    let out = c
        .transcribe(
            &clip,
            &ChunkOptions {
                language: Some("en".into()),
                ..ChunkOptions::default()
            },
        )
        .unwrap();
    eprintln!(
        "next call, with the reload: {:.2} s",
        began.elapsed().as_secs_f32()
    );
    assert!(
        text(&out.value).contains("thursday"),
        "thursday not heard in {} chars",
        text(&out.value).len()
    );
    c.shutdown();
}
