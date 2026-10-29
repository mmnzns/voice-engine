//! A small file that remembers "the graphics card failed" so the app doesn't try it again on every start. It holds
//! one line: the key of the build and driver that failed (the app picks it, for example "1.2.0 | driver 560.94").
//! A new build or a new driver has a different key and gets a fresh try. It holds nothing about the person.

use std::path::Path;

/// Did the card fail for this key?
pub(crate) fn failed(path: Option<&Path>, key: &str) -> bool {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .is_some_and(|text| text.trim() == key)
}

/// Remembers that the card failed for this key. Does nothing without a path.
pub(crate) fn record(path: Option<&Path>, key: &str) {
    let Some(path) = path else { return };
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(path, key);
}

/// Forgets a remembered failure: the app's "try the graphics card again".
pub fn clear_gpu_failure(path: &Path) {
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failure_is_remembered_for_its_key_only() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sub").join("gpu.txt");
        assert!(!failed(Some(&file), "a"));
        record(Some(&file), "a");
        assert!(failed(Some(&file), "a"));
        assert!(!failed(Some(&file), "b"));
        clear_gpu_failure(&file);
        assert!(!failed(Some(&file), "a"));
    }

    #[test]
    fn without_a_path_nothing_is_remembered() {
        record(None, "a");
        assert!(!failed(None, "a"));
    }
}
