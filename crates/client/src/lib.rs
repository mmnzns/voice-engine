//! The app side of the voice-engine helpers. An app builds an [`LlmClient`] with the paths of its helper binaries and
//! calls `load`, `complete` (or `complete_within`, with its own reply limit), `embed`, `inspect`, `devices`,
//! `loaded_model`, `unload` and `cancel` from any thread; the client starts the helper,
//! restarts it when it dies, and turns every failure into an [`EngineError`].
//!
//! An [`SttClient`] does the same for the speech helper (Whisper on the Windows Vulkan helper, with the processor
//! helper as the fallback): `load`, `transcribe`, `unload`, `cancel`, `shutdown`. It remembers the loaded model and
//! puts it into any helper it has to start again. The ONNX speech models run in the app's own process
//! (`voice_engine_stt::load`), not through a helper.

mod error;
mod gpu_memory;
mod llm;
mod process;
pub mod servers;
mod stt;

pub use error::EngineError;
pub use gpu_memory::clear_gpu_failure;
pub use llm::{Limits, LlmClient, LlmConfig, LoadInfo, Outcome};
pub use stt::{SttClient, SttConfig, SttLoadInfo};
pub use voice_engine_protocol::Fallback;

/// Only for this crate's own integration tests, which need the process type that apps never see.
#[doc(hidden)]
pub mod __testing {
    pub use crate::process::*;
}
