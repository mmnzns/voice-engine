//! The app side of the voice-engine helpers. An app builds an [`LlmClient`] with the paths of its helper binaries and
//! calls `load`, `complete`, `embed`, `inspect`, `unload` and `cancel` from any thread; the client starts the helper,
//! restarts it when it dies, and turns every failure into an [`EngineError`].

mod error;
mod gpu_memory;
mod llm;
mod process;

pub use error::EngineError;
pub use gpu_memory::clear_gpu_failure;
pub use llm::{Limits, LlmClient, LlmConfig, LoadInfo, Outcome};

/// Only for this crate's own integration tests, which need the process type that apps never see.
#[doc(hidden)]
pub mod __testing {
    pub use crate::process::*;
}
