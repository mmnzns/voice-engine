//! Guard: no test or example may open the real shared model folder by accident (#23). Every file under
//! `crates/*/tests` and `crates/*/examples` that calls `SharedStore::open(` must set `VOICE_ENGINE_MODELS_DIR`
//! before the call, or be `download_proof`, which refuses to run unless that variable or `--real` is given.

use std::fs;
use std::path::{Path, PathBuf};

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The code of `source` without `//` comment lines, so a doc comment that names `SharedStore::open(` is not a call.
fn code(source: &str) -> String {
    source
        .lines()
        .filter(|l| !l.trim_start().starts_with("//"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Err with the reason when `source` calls `SharedStore::open(` without making sure of the folder first.
fn check(file_name: &str, source: &str) -> Result<(), String> {
    let code = code(source);
    let Some(call) = code.find("SharedStore::open(") else {
        return Ok(());
    };
    if file_name == "download_proof.rs" {
        let guarded = code.contains("VOICE_ENGINE_MODELS_DIR") && code.contains("--real");
        return if guarded {
            Ok(())
        } else {
            Err(
                "download_proof must refuse to run without VOICE_ENGINE_MODELS_DIR or --real"
                    .into(),
            )
        };
    }
    match code.find("set_var(\"VOICE_ENGINE_MODELS_DIR\"") {
        Some(set) if set < call => Ok(()),
        _ => Err("calls SharedStore::open( without setting VOICE_ENGINE_MODELS_DIR first".into()),
    }
}

#[test]
fn no_test_or_example_opens_the_real_shared_folder() {
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    for entry in fs::read_dir(&crates).unwrap().filter_map(Result::ok) {
        rust_files(&entry.path().join("tests"), &mut files);
        rust_files(&entry.path().join("examples"), &mut files);
    }
    assert!(
        files.len() > 10,
        "the scan found too few files: {}",
        files.len()
    );
    let mut bad = Vec::new();
    for file in &files {
        // This file names the call in its own checks.
        if file.file_name().is_some_and(|n| n == "guard.rs") {
            continue;
        }
        let source = fs::read_to_string(file).unwrap();
        let name = file.file_name().unwrap().to_string_lossy();
        if let Err(why) = check(&name, &source) {
            bad.push(format!("{}: {why}", file.display()));
        }
    }
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}

#[test]
fn the_check_catches_an_unguarded_call_and_passes_the_guarded_ones() {
    assert!(check("a.rs", "let s = SharedStore::open(App::Talaan);").is_err());
    assert!(check("a.rs", "let s = SharedStore::open_at(App::Talaan, d);").is_ok());
    assert!(check("a.rs", "// SharedStore::open( in a comment\n").is_ok());
    let guarded = "unsafe { std::env::set_var(\"VOICE_ENGINE_MODELS_DIR\", &d) };\nSharedStore::open(App::Ensayo);";
    assert!(check("a.rs", guarded).is_ok());
    let wrong_order = "SharedStore::open(App::Ensayo);\nset_var(\"VOICE_ENGINE_MODELS_DIR\", &d);";
    assert!(check("a.rs", wrong_order).is_err());
    assert!(check("download_proof.rs", "SharedStore::open(App::Talaan);").is_err());
}
