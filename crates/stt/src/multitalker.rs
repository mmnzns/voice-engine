//! Multitalker Parakeet TDT: who said what, also when people talk over each other, in one track. The catalog's
//! `multitalker-tdt` (a community fine-tune of NVIDIA's Parakeet TDT v3, 25 European languages) through
//! parakeet-rs's `MultitalkerTDT`, with the catalog's `diarization` model (Nemotron 3 Diarization) saying who speaks
//! when.
//!
//! [`load`] needs both models' folders. [`Multitalker::run`] takes a whole track (16 kHz mono, any length) and returns
//! one [`SpeakerTurn`] per stretch of one speaker's words, in the order they start. People who talk at the same time
//! give turns whose times overlap. The model runs once per speaker in chunks of about 20 s, cut at pauses, so the
//! cost grows with the number of speakers; it needs no chunking by the app (the catalog's `chunk_secs` is `None`).
//!
//! `SpeakerTurn::speaker` is the separation model's own speaker number (0 to 7). It means nothing beyond one call: the
//! same person can be another number in the next track. Naming speakers ("Speaker 1") is the app's job, as for
//! [`crate::diarize`]. Words are whole words with times on the track's own clock, punctuation glued to the word
//! before, as the other engines give them.
//!
//! It always runs on the processor, on the ONNX thread rule (`threads`: three quarters of the logical cores, 2 to 12;
//! `Some(n)` is the app's own number, at least 1), for both models. Errors are kinds ([`SpeechError`]); no message
//! carries audio, text or a file name.
//!
//! The conversion from parakeet-rs's utterances is [`turns_from_utterances`], pure, so `tests/multitalker_api.rs`
//! tests it without a model.

use crate::api::{SpeechError, Word};
use crate::diarize::thread_count;
use crate::parakeet_engine::{from_parakeet, ort_environment, sentence_text};
use crate::threads::logical_cores;
use crate::words::clean;
use parakeet_rs::{MultitalkerTDT, SpeakerUtterance};
use std::path::Path;
use voice_engine_models::catalog::{self, Kind, ModelInfo};

/// One speaker's words, one after the other, with no pause longer than a second between them.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerTurn {
    /// The separation model's speaker number (0 to 7); only meaningful inside one call.
    pub speaker: usize,
    /// On the track's clock.
    pub start_ms: u64,
    pub end_ms: u64,
    /// The words joined with single spaces, punctuation glued to the word before: the same text as `words`.
    pub text: String,
    pub words: Vec<Word>,
}

/// parakeet-rs's utterances as turns on the clock of audio `audio_ms` long: seconds to whole milliseconds (truncated,
/// as the other engines do), never past the end of the audio, an end never before its start; punctuation glued to the
/// word before it and noise dropped (`words::clean`). A turn left with no words is dropped. The order is kept
/// (parakeet-rs sorts by start time, overlapping turns stay as they are).
pub fn turns_from_utterances(utterances: Vec<SpeakerUtterance>, audio_ms: u64) -> Vec<SpeakerTurn> {
    let ms = |t: f32| ((t.max(0.0) * 1000.0) as u64).min(audio_ms);
    utterances
        .into_iter()
        .filter_map(|u| {
            let words = clean(
                u.words
                    .into_iter()
                    .map(|w| Word {
                        start_ms: ms(w.start),
                        end_ms: ms(w.end.max(w.start)),
                        text: w.text,
                    })
                    .collect(),
            );
            let first = words.first()?;
            let start_ms = first.start_ms;
            let end_ms = words.last().map_or(start_ms, |w| w.end_ms).max(start_ms);
            let text = sentence_text(&words);
            if crate::is_noise(&text) {
                return None;
            }
            Some(SpeakerTurn {
                speaker: u.speaker_id,
                start_ms,
                end_ms,
                text,
                words,
            })
        })
        .collect()
}

/// A loaded Multitalker model and its speaker separation model. `Send`, so an app can keep it behind a mutex or move
/// it to a worker thread.
pub struct Multitalker {
    model: MultitalkerTDT,
    threads: usize,
}

impl Multitalker {
    /// Who said what in `audio` (16 kHz mono, any length): see the module docs. Empty audio gives no turns and never
    /// reaches the model.
    pub fn run(&mut self, audio: &[f32]) -> Result<Vec<SpeakerTurn>, SpeechError> {
        if audio.is_empty() {
            return Ok(vec![]);
        }
        let audio_ms = audio.len() as u64 * 1000 / 16_000;
        let utterances = self
            .model
            .transcribe(audio, 16_000, 1)
            .map_err(|e| from_parakeet(&e, false))?;
        Ok(turns_from_utterances(utterances, audio_ms))
    }

    /// The intra-op thread count ONNX Runtime's sessions were built with.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

/// Loads the catalog's `multitalker-tdt` model (`model`) from `model_dir` and the catalog's `diarization` model from
/// `diarization_dir`. `threads`: `None` for the engine's rule, `Some(n)` for the app's own count (at least 1). A
/// missing file in either folder is [`SpeechError::ModelMissing`]; an empty one (an interrupted download) is
/// [`SpeechError::BadModel`]; a catalog entry of another kind is `Unsupported`.
pub fn load(
    model: &ModelInfo,
    model_dir: &Path,
    diarization_dir: &Path,
    threads: Option<usize>,
) -> Result<Multitalker, SpeechError> {
    if model.kind != Kind::Multitalker {
        return Err(SpeechError::Unsupported("not a Multitalker model".into()));
    }
    let diarization = catalog::find("diarization")
        .and_then(|m| m.files.first())
        .ok_or_else(|| {
            SpeechError::Internal("the catalog has no speaker separation model".into())
        })?;
    let diarizer_file = diarization_dir.join(diarization.name);
    let model_files: Vec<_> = model.files.iter().map(|f| model_dir.join(f.name)).collect();
    for file in model_files.iter().chain([&diarizer_file]) {
        if !file.is_file() {
            return Err(SpeechError::ModelMissing);
        }
    }
    // An interrupted download leaves an empty file; ONNX Runtime calls that a generic failure, which is not a kind.
    for file in model_files.iter().chain([&diarizer_file]) {
        if std::fs::metadata(file).is_ok_and(|m| m.len() == 0) {
            return Err(SpeechError::BadModel);
        }
    }
    let threads = thread_count(threads, logical_cores());
    let config = parakeet_rs::ExecutionConfig::new()
        .with_environment(ort_environment()?)
        .with_intra_threads(threads);
    let model = MultitalkerTDT::from_pretrained(model_dir, &diarizer_file, Some(config))
        .map_err(|e| from_parakeet(&e, true))?;
    Ok(Multitalker { model, threads })
}
