//! The speech API (spec section 9.1): load a model, hand it audio, get timed text back.
//!
//! ```no_run
//! use voice_engine_stt::{ChunkOptions, LoadOptions, load};
//!
//! let model = voice_engine_models::catalog::find("whisper-turbo").unwrap();
//! let mut engine = load(model, std::path::Path::new("/path/to/the/model/folder"), LoadOptions { gpu: true, threads: None })?;
//! let audio: Vec<f32> = vec![0.0; 16_000 * 3]; // 16 kHz mono
//! let transcript = engine.transcribe(&audio, &ChunkOptions::default())?;
//! for segment in &transcript.segments {
//!     for word in &segment.words {
//!         // word.text, word.start_ms, word.end_ms: times on the audio's clock
//!     }
//! }
//! assert!(!engine.backend().backend.is_empty()); // "metal", "vulkan" or "cpu"; `fell_back` says if a GPU was refused
//! # Ok::<(), voice_engine_stt::SpeechError>(())
//! ```
//!
//! A loaded [`Transcriber`] stays loaded until the app drops it.

use std::fmt;
use std::path::Path;
use voice_engine_models::catalog::{Kind, ModelInfo};

/// How to load a model. Per load, so an app can fall back to the processor for one model only.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LoadOptions {
    /// Metal on the Mac (Whisper), Vulkan (Whisper) or DirectML (Parakeet, with the `directml` feature) on Windows.
    /// When the device cannot be used, or the model has no path to it here (the ONNX models on the Mac), the model
    /// loads on the processor and [`Transcriber::backend`] says `fell_back` (no error).
    pub gpu: bool,
    /// `None`: the engine's rule (`threads::default_threads`). `Some(n)`: the app's own setting (at least 1).
    pub threads: Option<usize>,
}

/// What to do with one piece of audio.
#[derive(Debug, Clone, PartialEq)]
pub struct ChunkOptions {
    /// A hint ("fr"); `None` lets the model detect the language.
    pub language: Option<String>,
    pub timing: Timing,
    /// Ask the model to keep "um", "uh" and "like" (Whisper: a short prompt of fillers; the ONNX models transcribe
    /// them as heard and ignore this).
    pub keep_fillers: bool,
    /// The app's custom words, spelled as the person wants them (Whisper's initial prompt).
    pub vocabulary: Vec<String>,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        ChunkOptions {
            language: None,
            timing: Timing::Words,
            keep_fillers: false,
            vocabulary: vec![],
        }
    }
}

/// Which settings the model decodes with. Every result carries both segments and words, in either mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timing {
    /// Talaan's settings: the segments the model produced; the words are derived from the token times.
    Segments,
    /// Ensayo's settings: token timestamps, one word per segment.
    Words,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcript {
    /// What the model produced, in order, each with its words.
    pub segments: Vec<Segment>,
    /// The language the model settled on.
    pub language: Option<String>,
    /// Decoding stopped with 2 s or more of audio left after this point (the engine resumed from here).
    pub stopped_early_at_ms: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Where a model runs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeechBackend {
    /// `metal`, `vulkan`, `directml` or `cpu`.
    pub backend: String,
    /// The GPU's name when known; `None` on the processor.
    pub device: Option<String>,
    /// The app asked for the GPU (`gpu: true`) and the model runs on the processor instead.
    pub fell_back: bool,
}

/// A loaded speech model. `Send`, so an app can keep it behind a mutex or move it to a worker thread.
pub trait Transcriber: Send {
    /// One piece of audio (16 kHz mono). Times are on the audio's clock; the app adds the chunk's offset.
    fn transcribe(&mut self, audio: &[f32], opts: &ChunkOptions)
    -> Result<Transcript, SpeechError>;
    /// Where it runs now (after a fallback: the processor, with `fell_back` true).
    fn backend(&self) -> SpeechBackend;
}

/// Why a speech call failed: a kind to match on, never text to parse. The messages carry no audio, no text the
/// person said, and no file names the person chose.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeechError {
    /// The model file is not there (deleted, moved, an unfinished download).
    ModelMissing,
    /// The file is there but the engine cannot read it as this kind of model.
    BadModel,
    /// There was not enough memory to load the model.
    OutOfMemory,
    /// Not available here (a kind this build cannot run, or a file that is not a speech model).
    Unsupported(String),
    /// The audio is longer than the model takes in one call (Parakeet, Parakeet Unified and Nemotron: the catalog's
    /// `chunk_secs`, 40 s). Nothing was decoded; the app cuts the audio and calls again.
    TooLong { max_secs: u32 },
    /// Anything else; the string is the engine's own error code, never user data.
    Internal(String),
}

impl fmt::Display for SpeechError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeechError::ModelMissing => f.write_str("the speech model file is missing"),
            SpeechError::BadModel => f.write_str("the speech model file could not be read"),
            SpeechError::OutOfMemory => f.write_str("not enough memory to load the speech model"),
            SpeechError::Unsupported(what) => write!(f, "not supported: {what}"),
            SpeechError::TooLong { max_secs } => {
                write!(
                    f,
                    "the audio is longer than the {max_secs} s this model takes at once"
                )
            }
            SpeechError::Internal(what) => write!(f, "speech engine error: {what}"),
        }
    }
}

impl std::error::Error for SpeechError {}

/// Loads a speech model from its folder `dir` (as the shared store lays it out: the catalog's file names).
///
/// Whisper and the ONNX kinds (Parakeet, Parakeet Unified, Nemotron) run in process; the folder holds the catalog's
/// files for the model. Voice detection, speaker separation, embeddings and the local AI are not speech models
/// (`Unsupported`).
pub fn load(
    model: &ModelInfo,
    dir: &Path,
    opts: LoadOptions,
) -> Result<Box<dyn Transcriber>, SpeechError> {
    match model.kind {
        Kind::Whisper => {
            let file = model
                .files
                .first()
                .ok_or_else(|| SpeechError::Internal("the catalog entry lists no file".into()))?;
            let engine = crate::whisper_engine::WhisperEngine::load(&dir.join(file.name), opts)?;
            Ok(Box::new(engine))
        }
        Kind::Parakeet | Kind::ParakeetUnified | Kind::Nemotron => {
            let engine = crate::parakeet_engine::ParakeetEngine::load(model, dir, opts)?;
            Ok(Box::new(engine))
        }
        Kind::Vad | Kind::Diarization | Kind::Embedding | Kind::Llm => {
            Err(SpeechError::Unsupported("not a speech model".into()))
        }
    }
}
