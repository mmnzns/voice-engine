//! Speaker separation: which of up to 8 voices is talking, every 10 ms, in one track. Ported from Talaan's
//! `transcriber/diarize.rs` (MIT, Miguel Monzones): NVIDIA's Nemotron 3 Diarization (a streaming Sortformer, up to
//! 8 speakers, OpenMDW-1.1) through parakeet-rs's `sortformer` feature.
//!
//! [`load`] reads the catalog's `diarization` model from its folder; [`Diarizer::run`] takes a whole track (16 kHz
//! mono) and returns one [`SpeakerFrame`] per 10 ms: `speakers[k]` is the model's probability, 0 to 1, that
//! speaker `k` is talking in that frame (the model's own sigmoid output, not post-processed; two speakers can both be
//! high when they overlap). The model's column order carries no meaning beyond one call: column 3 in one track is not
//! "the same person" as column 3 in another. Naming speakers ("Speaker 1") and turning frames into turns or into
//! words' speakers is the app's job (Talaan's), not this crate's.
//!
//! A streaming model needs to hear a voice before it can tell it apart, so a speaker's first turns tend to be
//! absorbed by whoever it already knows. So `run` resets the model's memory, feeds it the start of the track first
//! (up to [`WARMUP_SAMPLES`], 10 minutes) and throws those predictions away, which fills its speaker memory (Talaan
//! measured 8 of 12 turns right without it, 12 of 12 with it on a three-voice test), then feeds the whole track in
//! order in calls of [`CHUNK_SAMPLES`] (5 minutes) so memory stays flat on a long call, and keeps those predictions.
//!
//! Frames are on the track's own clock: frame `i` is `[10 i, 10 i + 10)` ms, and the last one is cut where the audio
//! ends (the model's window is centred, so it answers one frame more than the audio holds; it is dropped, and so is
//! the one each 5-minute call would add, which would push every later frame 10 ms late). Empty audio, and audio
//! under a millisecond, gives no frames and never reaches the model.
//!
//! It always runs on the processor, on the ONNX thread rule (`threads`: three quarters of the logical cores, 2 to 12;
//! `Some(n)` is the app's own number, at least 1).
//!
//! Errors are kinds: [`SpeechError::ModelMissing`], [`SpeechError::BadModel`], [`SpeechError::Internal`] with fixed
//! words. No message carries audio, a file name or ONNX Runtime's own text (it can hold paths).
//!
//! The loop is [`run_with`] over the small [`SpeakerModel`] trait, so `tests/diarize_api.rs` tests the order of the
//! calls, the warm-up and the clock with a scripted model.

use crate::api::SpeechError;
use crate::parakeet_engine::from_parakeet;
use crate::threads::{ThreadKind, logical_cores, resolve_threads};
use parakeet_rs::sortformer::{DiarizationConfig, NUM_SPEAKERS, Sortformer};
use std::path::Path;
use voice_engine_models::catalog;

/// The model answers once per 10 ms.
pub const FRAME_MS: u64 = 10;
/// Samples per frame (16 kHz): the model's mel hop.
pub const HOP_SAMPLES: usize = 160;
/// How many speakers the model tells apart.
pub const SPEAKERS: usize = 8;
/// Warm-up: the first 10 minutes of the track (all of it when shorter), Talaan's `WARMUP`.
pub const WARMUP_SAMPLES: usize = 10 * 60 * 16_000;
/// Audio per call, so memory stays flat on long tracks: 5 minutes, Talaan's `CHUNK`. A multiple of
/// [`HOP_SAMPLES`], so chunk edges fall on frame edges.
pub const CHUNK_SAMPLES: usize = 5 * 60 * 16_000;

// The column count is the model's; the public constant and parakeet-rs's agree or this does not build.
const _: () = assert!(SPEAKERS == NUM_SPEAKERS);
const _: () = assert!(CHUNK_SAMPLES.is_multiple_of(HOP_SAMPLES));

/// Who is talking over 10 ms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerFrame {
    /// On the track's clock.
    pub start_ms: u64,
    /// `start_ms + 10`, or where the audio ends for the last frame.
    pub end_ms: u64,
    /// Probability, 0 to 1, that each of the model's 8 speakers is talking in this frame.
    pub speakers: [f32; SPEAKERS],
}

/// The loaded model as the loop sees it (the real one wraps parakeet-rs; the tests script it).
pub trait SpeakerModel {
    /// Forgets every speaker it has heard.
    fn reset(&mut self);
    /// The next stretch of the track, which continues what was fed before: one row per 10 ms frame (the real model
    /// answers `len / 160 + 1` rows, the last being the centred window's extra).
    fn feed(&mut self, audio: &[f32]) -> Result<Vec<[f32; SPEAKERS]>, SpeechError>;
}

/// The model's rows as frames on the clock of audio `audio_ms` long: frame `i` is `[10 i, 10 i + 10)` ms, cut at the
/// end of the audio, none starting at or after it; probabilities kept inside 0 to 1 (NaN becomes 0).
pub fn frames_from_rows(rows: Vec<[f32; SPEAKERS]>, audio_ms: u64) -> Vec<SpeakerFrame> {
    let tidy = |p: f32| if p.is_nan() { 0.0 } else { p.clamp(0.0, 1.0) };
    rows.into_iter()
        .enumerate()
        .map(|(i, row)| {
            let start_ms = i as u64 * FRAME_MS;
            (start_ms, row)
        })
        .take_while(|&(start_ms, _)| start_ms < audio_ms)
        .map(|(start_ms, row)| SpeakerFrame {
            start_ms,
            end_ms: (start_ms + FRAME_MS).min(audio_ms),
            speakers: row.map(tidy),
        })
        .collect()
}

/// The loop, with the sizes as arguments: resets the model, warms it up on the first `warmup` samples (answer thrown
/// away), then feeds the track in order in calls of `chunk` samples (rounded down to whole frames, at least one) and
/// keeps those answers.
pub fn run_with(
    model: &mut dyn SpeakerModel,
    track: &[f32],
    warmup: usize,
    chunk: usize,
) -> Result<Vec<SpeakerFrame>, SpeechError> {
    model.reset();
    let audio_ms = track.len() as u64 * 1000 / 16_000;
    if audio_ms == 0 {
        return Ok(vec![]);
    }
    model.feed(&track[..track.len().min(warmup)])?;
    let chunk = (chunk / HOP_SAMPLES).max(1) * HOP_SAMPLES;
    let mut rows: Vec<[f32; SPEAKERS]> = Vec::with_capacity(track.len() / HOP_SAMPLES + 1);
    for part in track.chunks(chunk) {
        // The extra frame the centred window adds belongs to the next call's first frame, so it is not kept.
        let keep = part.len().div_ceil(HOP_SAMPLES);
        rows.extend(model.feed(part)?.into_iter().take(keep));
    }
    Ok(frames_from_rows(rows, audio_ms))
}

/// [`run_with`] with Talaan's sizes (10 minutes of warm-up, calls of 5 minutes).
pub fn run_track(
    model: &mut dyn SpeakerModel,
    track: &[f32],
) -> Result<Vec<SpeakerFrame>, SpeechError> {
    run_with(model, track, WARMUP_SAMPLES, CHUNK_SAMPLES)
}

/// The audio as the model should be given it: `audio` itself, except when the model would see a number of mel frames
/// (`samples / 160 + 1`) that is a multiple of 8, when it gets 10 ms of silence more.
///
/// A workaround for parakeet-rs 0.3.8, which fails with "Array has a non-contiguous layout" on a call whose mel
/// frame count is a multiple of 8 and holds the whole call in one window (any track or last 5-minute call up to
/// about 30 s: one length in eight). Its `pad_chunk` hands ONNX Runtime a copy that kept the feature matrix's
/// column-major layout. One frame more is padded to a multiple of 8 by the library itself and the extra frame
/// is never returned: [`run_with`] keeps only the frames the real audio holds.
pub fn audio_for_model(audio: &[f32]) -> std::borrow::Cow<'_, [f32]> {
    if audio.is_empty() || !(audio.len() / HOP_SAMPLES + 1).is_multiple_of(8) {
        return std::borrow::Cow::Borrowed(audio);
    }
    let mut padded = Vec::with_capacity(audio.len() + HOP_SAMPLES);
    padded.extend_from_slice(audio);
    padded.resize(audio.len() + HOP_SAMPLES, 0.0);
    std::borrow::Cow::Owned(padded)
}

/// ONNX Runtime's intra-op threads for a load: the processor's rule (this model always runs on the processor), or the
/// app's own number (at least 1, not capped).
pub fn thread_count(requested: Option<usize>, logical: usize) -> usize {
    resolve_threads(ThreadKind::Processor, requested, logical)
}

impl SpeakerModel for Sortformer {
    fn reset(&mut self) {
        self.reset_state();
    }

    fn feed(&mut self, audio: &[f32]) -> Result<Vec<[f32; SPEAKERS]>, SpeechError> {
        let out = self
            .diarize_chunk_raw(&audio_for_model(audio))
            .map_err(|e| from_parakeet(&e, false))?;
        if out.predictions.ncols() != SPEAKERS {
            return Err(SpeechError::Internal("speaker count".into()));
        }
        let rows = out.num_valid_frames.min(out.predictions.nrows());
        Ok((0..rows)
            .map(|r| {
                let mut row = [0.0; SPEAKERS];
                for (cell, p) in row.iter_mut().zip(out.predictions.row(r)) {
                    *cell = *p;
                }
                row
            })
            .collect())
    }
}

/// A loaded speaker separation model. `Send`, so an app can keep it behind a mutex or move it to a worker thread.
pub struct Diarizer {
    model: Sortformer,
    threads: usize,
}

impl Diarizer {
    /// Who is talking, every 10 ms, in `track` (16 kHz mono): see the module docs. The model's memory is reset on
    /// every call, so calls do not affect each other.
    pub fn run(&mut self, track: &[f32]) -> Result<Vec<SpeakerFrame>, SpeechError> {
        run_track(&mut self.model, track)
    }

    /// The intra-op thread count ONNX Runtime's session was built with.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

/// Loads the catalog's `diarization` model from `model_dir`. `threads`: `None` for the engine's rule, `Some(n)` for the
/// app's own count (at least 1).
pub fn load(model_dir: &Path, threads: Option<usize>) -> Result<Diarizer, SpeechError> {
    let name = catalog::find("diarization")
        .and_then(|m| m.files.first())
        .map(|f| f.name)
        .ok_or_else(|| {
            SpeechError::Internal("the catalog has no speaker separation model".into())
        })?;
    let file = model_dir.join(name);
    if !file.is_file() {
        return Err(SpeechError::ModelMissing);
    }
    // An interrupted download leaves an empty file; ONNX Runtime calls that a generic failure, which is not a kind.
    if std::fs::metadata(&file).is_ok_and(|m| m.len() == 0) {
        return Err(SpeechError::BadModel);
    }
    let threads = thread_count(threads, logical_cores());
    let config = parakeet_rs::ExecutionConfig::new().with_intra_threads(threads);
    let model = Sortformer::with_config(&file, Some(config), DiarizationConfig::default())
        .map_err(|e| from_parakeet(&e, true))?;
    Ok(Diarizer { model, threads })
}
