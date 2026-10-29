//! A file that remembers "the graphics card failed for this build and driver", so it isn't tried again on every start.

use std::path::Path;

/// Whether a failure was remembered for `key`. With no file, nothing is remembered.
pub(crate) fn failed(path: Option<&Path>, key: &str) -> bool {
    path.and_then(|p| std::fs::read_to_string(p).ok())
        .is_some_and(|s| s.trim() == key)
}

/// Remembers a failure for `key`. Best effort: failing to write only means the card is tried again next time.
pub(crate) fn record(path: Option<&Path>, key: &str) {
    if let Some(path) = path {
        let _ = std::fs::write(path, key);
    }
}

/// Forgets a remembered graphics card failure (an app's "try the graphics card again" button).
pub fn clear_gpu_failure(path: &Path) {
    let _ = std::fs::remove_file(path);
}
