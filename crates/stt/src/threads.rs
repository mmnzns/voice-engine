//! The thread rule (spec 9.2), one place for every speech engine (Whisper, and the ONNX models' intra-op threads).
//!
//! On the processor: three quarters of the logical cores, 2 to 12, on every system (measured on a 16-thread
//! Ryzen: 12 is fastest and 16 is 2 to 3 times slower). On a GPU (Metal, Vulkan, DirectML): all the logical
//! cores, 2 to 8, because the processor only feeds the card. `LoadOptions::threads` overrides it per model: an app
//! can measure its own best (Parlato: Whisper Base fastest at 6 threads and Parakeet Ultra at 12 on an M3 Pro).

/// Which device the model actually runs on decides the default: after a fallback to the processor it is the
/// processor's rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadKind {
    Processor,
    Gpu,
}

/// The engine's default thread count for `logical` logical cores.
pub fn default_threads(kind: ThreadKind, logical: usize) -> usize {
    match kind {
        ThreadKind::Processor => (logical * 3 / 4).clamp(2, 12),
        ThreadKind::Gpu => logical.clamp(2, 8),
    }
}

/// What a load uses: the app's own number when it gave one (at least 1, and not capped: it is the app's
/// measurement), the default otherwise.
pub fn resolve_threads(kind: ThreadKind, requested: Option<usize>, logical: usize) -> usize {
    match requested {
        Some(n) => n.max(1),
        None => default_threads(kind, logical),
    }
}

/// Logical cores on this computer (4 when the system will not say).
pub fn logical_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}
