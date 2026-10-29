//! No audio, text, words or file names the person chose in any log or error message. whisper.cpp's own log is
//! switched off (`install_logging_hooks`); the only lines this crate prints are the helper's stderr notes. This
//! scans the source so a line that prints a transcript, a word, the custom vocabulary or the audio can't be added
//! by accident. Sizes, times, counts, device names and error kinds are fine.

use std::path::Path;

/// Names that hold what the person said, wrote or chose in this crate's code. A log line that mentions one of them
/// is a leak.
const USER_TEXT: &[&str] = &[
    "transcript",
    "segment",
    "word",
    ".text",
    "text:",
    "vocabulary",
    "prompt",
    "audio",
    "samples",
    "pieces",
    "piece",
    "language",
    "model_path",
    "file",
];

/// Lines that print or log something.
const LOG_MACROS: &[&str] = &[
    "eprintln!",
    "println!",
    "eprint!",
    "print!",
    "log::",
    "tracing::",
    "dbg!",
];

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

#[test]
fn no_log_line_mentions_user_text() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = vec![];
    rust_files(&src, &mut files);
    for path in &files {
        let text = std::fs::read_to_string(path).unwrap();
        for (n, line) in text.lines().enumerate() {
            if !LOG_MACROS.iter().any(|m| line.contains(m)) {
                continue;
            }
            for name in USER_TEXT {
                assert!(
                    !line.contains(name),
                    "{}:{}: a log line mentions {name:?}: {}",
                    path.display(),
                    n + 1,
                    line.trim()
                );
            }
        }
    }
    // A crate may have no log line at all, so count the files scanned, not the lines.
    assert!(
        files.len() >= 6,
        "the scan found {} source files; is the source path right?",
        files.len()
    );
}

#[test]
fn whisper_cpps_own_log_is_switched_off() {
    // whisper.cpp prints the model path and the device list to stderr by default; the voice detector is whisper.cpp
    // too.
    for file in ["whisper_engine.rs", "vad.rs"] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("install_logging_hooks()"), "{file}");
    }
}

#[test]
fn error_messages_are_built_from_codes_not_from_what_was_said() {
    // Every `SpeechError::Internal(format!(...))` in the engines carries a whisper-rs error (a kind and a number),
    // never the audio, the prompt, the text or a file name.
    for file in [
        "whisper_engine.rs",
        "vad.rs",
        "parakeet_engine.rs",
        "diarize.rs",
    ] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
        let text = std::fs::read_to_string(path).unwrap();
        for (n, line) in text.lines().enumerate() {
            if line.contains("SpeechError::Internal(format!") {
                assert!(
                    line.contains("{e}"),
                    "{file}:{}: an error message that is not just the engine's error: {}",
                    n + 1,
                    line.trim()
                );
            }
        }
    }
}

#[test]
fn onnx_error_text_never_reaches_a_speech_error() {
    // parakeet-rs and ONNX Runtime put paths (the model folder, an audio file) in their messages. The Parakeet engine
    // and speaker separation turn each error into a kind or a fixed word, so no line that builds a
    // `SpeechError::Internal` there formats anything into it.
    for (file, at_least) in [("parakeet_engine.rs", 3), ("diarize.rs", 2)] {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
        let text = std::fs::read_to_string(path).unwrap();
        let mut seen = 0;
        for (n, line) in text.lines().enumerate() {
            if line.contains("SpeechError::Internal(") {
                seen += 1;
                assert!(
                    !line.contains("format!")
                        && !line.contains("to_string()")
                        && !line.contains('{'),
                    "{file}:{}: an internal error that carries text from elsewhere: {}",
                    n + 1,
                    line.trim()
                );
            }
        }
        assert!(
            seen >= at_least,
            "the scan found {seen} internal errors in {file}; did the code move?"
        );
    }
}

#[test]
fn speaker_separation_prints_nothing_and_takes_its_errors_from_the_one_conversion() {
    // diarize.rs has no log line at all, and every ONNX error goes through `from_parakeet` (kinds only).
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/diarize.rs");
    let text = std::fs::read_to_string(path).unwrap();
    for line in text.lines().filter(|l| !l.trim_start().starts_with("//")) {
        // "catalog::" ends in "log::": a macro or crate name starts a word.
        let logs = LOG_MACROS.iter().any(|m| {
            line.match_indices(m).any(|(at, _)| {
                !line[..at]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_alphanumeric() || c == '_')
            })
        });
        assert!(!logs, "diarize.rs logs: {}", line.trim());
        assert!(
            !line.contains("map_err(|e| SpeechError")
                && !line.contains("e.to_string()")
                && !line.contains("{e"),
            "diarize.rs builds an error from an engine's text: {}",
            line.trim()
        );
    }
    assert!(text.matches("from_parakeet(&e").count() >= 2);
}
