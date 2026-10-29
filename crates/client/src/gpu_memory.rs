//! Replaced in Task 3.

/// Forgets a remembered graphics card failure (an app's "try the graphics card again" button).
pub fn clear_gpu_failure(path: &std::path::Path) {
    let _ = std::fs::remove_file(path);
}
