//! Helpers the real-model tests share (`real_whisper.rs`, `real_parakeet.rs`): one lock so only one test loads a
//! model at a time, the models folder, speech made with the macOS `say` command, and small text helpers.
//! Only ever reads the models folder.

#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, MutexGuard};
use voice_engine_stt::Transcript;

static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

pub fn guard() -> MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

pub fn models_dir() -> PathBuf {
    match std::env::var_os("VOICE_ENGINE_TEST_MODELS") {
        Some(d) => PathBuf::from(d),
        None => dirs::data_dir().unwrap().join("Talaan/models"),
    }
}

/// 16 kHz mono f32 samples of `text` spoken by the macOS `say` command.
pub fn say(text: &str) -> Vec<f32> {
    let dir = tempfile::tempdir().unwrap();
    let aiff = dir.path().join("clip.aiff");
    let wav = dir.path().join("clip.wav");
    let ok = Command::new("say")
        .arg("-o")
        .arg(&aiff)
        .arg(text)
        .status()
        .unwrap();
    assert!(ok.success(), "say failed");
    let ok = Command::new("afconvert")
        .args(["-f", "WAVE", "-d", "LEF32@16000", "-c", "1"])
        .arg(&aiff)
        .arg(&wav)
        .status()
        .unwrap();
    assert!(ok.success(), "afconvert failed");
    read_wav_f32(&wav)
}

/// Samples of a 32-bit float mono WAV (what afconvert wrote), chunk by chunk.
pub fn read_wav_f32(path: &Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap();
    assert_eq!(&b[0..4], b"RIFF");
    assert_eq!(&b[8..12], b"WAVE");
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let len = u32::from_le_bytes(b[i + 4..i + 8].try_into().unwrap()) as usize;
        let body = i + 8;
        if id == b"fmt " {
            let tag = u16::from_le_bytes(b[body..body + 2].try_into().unwrap());
            let bits = u16::from_le_bytes(b[body + 14..body + 16].try_into().unwrap());
            assert!(
                tag == 3 && bits == 32,
                "not float32 (tag {tag}, {bits} bits)"
            );
        }
        if id == b"data" {
            let end = (body + len).min(b.len());
            return b[body..end]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
        }
        i = body + len + (len & 1);
    }
    panic!("no data chunk in {}", path.display());
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

pub const SENTENCE: &str = "Hello there. This is a short test of the speech engine.";

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
            "{s:?} in {clip_ms} ms"
        );
        assert!(!s.words.is_empty(), "segment without words: {s:?}");
        for w in &s.words {
            assert!(!w.text.trim().is_empty());
            assert!(w.start_ms <= w.end_ms, "{w:?}");
            assert!(w.end_ms <= clip_ms, "{w:?} past the {clip_ms} ms clip");
            assert!(
                w.start_ms >= last_end,
                "words out of order at {w:?} (last end {last_end})"
            );
            last_end = w.end_ms;
        }
    }
}
