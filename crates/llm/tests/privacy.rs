//! No user text in any log. The helper's only log is stderr; a line that prints a prompt, a reply or a token piece
//! would leak what the person said or wrote. This scans the source so such a line can't be added by accident. Sizes,
//! timings, paths and error kinds are fine.

use std::path::Path;

/// Names that hold user text in the helper's code. A log line that mentions one of them is a leak.
const USER_TEXT: &[&str] = &[
    "c.user",
    "c.system",
    "prompt",
    ".text",
    "reply",
    "piece",
    "out[",
    "&out",
    "e.texts",
    "texts",
    "thinking_text",
    "answer",
];

/// Lines that print or log something.
const LOG_MACROS: &[&str] = &[
    "eprintln!",
    "println!",
    "eprint!",
    "print!",
    "log::",
    "tracing::",
];

#[test]
fn no_log_line_mentions_user_text() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = 0;
    for entry in std::fs::read_dir(&src).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "rs") {
            continue;
        }
        files += 1;
        let text = std::fs::read_to_string(&path).unwrap();
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
    // The helper may have no log line at all, so count the files scanned, not the lines.
    assert!(
        files > 0,
        "the scan found no source files; is the source path right?"
    );
}

#[test]
fn llama_logging_stays_off() {
    // llama.cpp's own log would print the prompt in verbose mode; the helper turns it off before anything else.
    let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let text = std::fs::read_to_string(lib).unwrap();
    assert!(text.contains("with_logs_enabled(false)"));
}
