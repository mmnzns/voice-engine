//! Voice detection: where in a track is there speech. Only speech needs to go to a speech model: silence costs
//! time and makes models invent text ("Thank you." on a quiet line).
//!
//! [`detect`] takes a whole track (16 kHz mono) and returns the speech spans as `(start_ms, end_ms)` on the audio's
//! own clock, in order, never overlapping, never past the end of the audio. It is whisper.cpp's Silero detector
//! (through whisper-rs), ported from Talaan's `transcriber/vad.rs` (MIT) with its settings: threshold 0.5, 250 ms
//! minimum speech, 300 ms minimum silence, no padding (the apps pad and cut their own chunks).
//!
//! [`detect_with`] takes the settings from the app ([`VadSettings`]; `detect` is `detect_with` and the defaults,
//! which are those numbers). Padding is added here, after detection, not by whisper.cpp: each span grows on both
//! sides, stops at the ends of the audio, and spans that overlap or touch after growing become one.
//!
//! It always runs on the processor, on a small fixed thread count the apps do not set (see `THREADS`). The model
//! is the catalog's `vad` entry (`ggml-silero-v6.2.0.bin`) in `model_dir`; it is under 1 MB. `detect` and
//! `detect_with` load it, use it and drop it again on every call, and keep no state.
//!
//! An app that detects often (dictation: every utterance) loads it once instead: [`VadDetector::load`] gives a
//! detector that holds the loaded model, and its `detect` / `detect_with` give exactly what the free functions
//! give. Nothing carries over between its calls: whisper.cpp clears the recurrent (LSTM) state at the start of
//! each one. It is `Send`, so it can live in an app's shared state and move between threads, but it is used by one
//! thread at a time (`&mut self`).
//!
//! Edge cases, as in Talaan: empty audio and audio shorter than one frame (512 samples, 32 ms) have no speech and
//! never reach the detector; audio with no speech gives an empty list, not an error.
//!
//! Errors are kinds: [`SpeechError::ModelMissing`] when the file is not there, [`SpeechError::BadModel`] when it is
//! there but cannot be loaded. No message carries audio or a file name.

use crate::api::SpeechError;
use std::path::{Path, PathBuf};
use std::sync::Once;
use voice_engine_models::catalog;

/// Silero's frame at 16 kHz: whisper.cpp reads the audio 512 samples (32 ms) at a time, and audio shorter than
/// that cannot hold the 250 ms of speech a span needs.
pub const FRAME_SAMPLES: usize = 512;

/// Threads for the detector. Fixed, and 4 on purpose: Talaan measured that more threads make voice detection
/// 20 to 30 times slower (16 threads took 9 to 13 s on a track that takes 0.3 to 0.5 s with 4), and on Parlato's
/// M3 Pro 2 threads were the fastest. The work is a small recurrent model run frame by frame, so threads only add
/// synchronisation. Apps do not set this.
const THREADS: i32 = 4;

/// What the detector is told. `Default` is Talaan's settings, which `detect` has always used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VadSettings {
    /// Probability above which a frame counts as speech (0.0 to 1.0). Default 0.5.
    pub threshold: f32,
    /// A span shorter than this is dropped (clicks, coughs). Default 250 ms.
    pub min_speech_ms: u32,
    /// Silence shorter than this does not end a span (a breath inside a sentence). Default 300 ms.
    pub min_silence_ms: u32,
    /// Added before and after every span, after detection. Stops at the ends of the audio; spans that overlap or
    /// touch after it are merged. Default 0.
    pub padding_ms: u32,
}

impl Default for VadSettings {
    fn default() -> Self {
        VadSettings {
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 300,
            padding_ms: 0,
        }
    }
}

/// whisper.cpp's time (centiseconds, a float) in whole milliseconds, rounded to the nearest. `None` for a value
/// that is not a time (NaN, infinite, negative).
pub fn centis_to_ms(centiseconds: f32) -> Option<u64> {
    if !centiseconds.is_finite() || centiseconds < 0.0 {
        return None;
    }
    Some((centiseconds as f64 * 10.0).round() as u64)
}

/// The detector's raw spans (`(start, end)` in centiseconds) as clean spans in milliseconds: converted, clamped to
/// the real audio (whisper.cpp counts whole frames, so the last span can end after the audio does), those that are
/// empty, backwards or not times dropped, sorted, and spans that overlap or touch merged into one.
pub fn spans_from_centiseconds(raw: &[(f32, f32)], audio_ms: u64) -> Vec<(u64, u64)> {
    let mut spans: Vec<(u64, u64)> = raw
        .iter()
        .filter_map(|&(a, b)| {
            let start = centis_to_ms(a)?;
            let end = centis_to_ms(b)?.min(audio_ms);
            (end > start).then_some((start, end))
        })
        .collect();
    spans.sort_unstable();
    let mut merged: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// `spans` (sorted, not overlapping, in milliseconds) grown by `padding_ms` on both sides: never before 0 or after
/// `audio_ms`, and spans that overlap or touch after growing merged into one.
pub fn pad_spans(spans: &[(u64, u64)], padding_ms: u64, audio_ms: u64) -> Vec<(u64, u64)> {
    let mut out: Vec<(u64, u64)> = Vec::with_capacity(spans.len());
    for &(start, end) in spans {
        let start = start.saturating_sub(padding_ms);
        let end = end.saturating_add(padding_ms).min(audio_ms);
        match out.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => out.push((start, end)),
        }
    }
    out
}

/// Audio too short to hold a frame: no speech, no need to load anything.
pub fn is_shorter_than_a_frame(samples: &[f32]) -> bool {
    samples.len() < FRAME_SAMPLES
}

/// Tells whisper.cpp and ggml to log through whisper-rs's hooks (so nothing goes to stderr, where it would carry
/// the model path), once for the whole process.
fn install_logging_hooks_once() {
    static HOOKS: Once = Once::new();
    HOOKS.call_once(|| {
        whisper_rs::install_logging_hooks();
    });
}

/// The catalog's `vad` model file in `model_dir`; `ModelMissing` when it is not there.
fn model_file(model_dir: &Path) -> Result<PathBuf, SpeechError> {
    let name = catalog::find("vad")
        .and_then(|m| m.files.first())
        .map(|f| f.name)
        .ok_or_else(|| SpeechError::Internal("the catalog has no voice detection model".into()))?;
    let file = model_dir.join(name);
    if !file.is_file() {
        return Err(SpeechError::ModelMissing);
    }
    Ok(file)
}

/// A voice detector that holds the loaded model, for apps that detect often. [`detect`] and [`detect_with`] load
/// the model on every call (about 70 ms the first time in a process, and a reload each time after); load a
/// `VadDetector` once and use it as many times as needed.
///
/// Each call gives what the free function with the same audio and settings gives. Nothing leaks from one call into
/// the next: whisper.cpp resets the recurrent state at the start of every detection
/// (`whisper_vad_detect_speech` clears the state buffer first).
///
/// It is `Send` (whisper-rs marks the context `Send`, and whisper.cpp's VAD context has no thread affinity: it is
/// plain heap data and a CPU backend, used by one thread at a time because the methods take `&mut self`). Keep it
/// in a `Mutex` to share it between threads.
pub struct VadDetector {
    ctx: whisper_rs::WhisperVadContext,
}

impl std::fmt::Debug for VadDetector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VadDetector").finish_non_exhaustive()
    }
}

impl VadDetector {
    /// Loads the catalog's `vad` model from `model_dir`. `ModelMissing` when the file is not there, `BadModel` when
    /// it is there but cannot be loaded. Always on the processor, on `THREADS` threads.
    pub fn load(model_dir: &Path) -> Result<VadDetector, SpeechError> {
        let file = model_file(model_dir)?;
        // whisper.cpp and ggml log to stderr by default (with the model path); errors come back as kinds.
        install_logging_hooks_once();
        // whisper-rs takes the path as text; one that is not text cannot be passed, and is not a missing file.
        let path = file
            .to_str()
            .ok_or_else(|| SpeechError::Unsupported("the model path is not valid text".into()))?;
        let mut ctx_params = whisper_rs::WhisperVadContextParams::new();
        ctx_params.set_n_threads(THREADS);
        ctx_params.set_use_gpu(false);
        let ctx = whisper_rs::WhisperVadContext::new(path, ctx_params)
            .map_err(|_| SpeechError::BadModel)?;
        Ok(VadDetector { ctx })
    }

    /// [`detect`] on the loaded model: the default [`VadSettings`].
    pub fn detect(&mut self, samples: &[f32]) -> Result<Vec<(u64, u64)>, SpeechError> {
        self.detect_with(samples, &VadSettings::default())
    }

    /// [`detect_with`] on the loaded model.
    pub fn detect_with(
        &mut self,
        samples: &[f32],
        settings: &VadSettings,
    ) -> Result<Vec<(u64, u64)>, SpeechError> {
        if is_shorter_than_a_frame(samples) {
            return Ok(vec![]);
        }
        let mut params = whisper_rs::WhisperVadParams::new();
        params.set_threshold(settings.threshold);
        params.set_min_speech_duration(settings.min_speech_ms.min(i32::MAX as u32) as i32);
        params.set_min_silence_duration(settings.min_silence_ms.min(i32::MAX as u32) as i32);
        // whisper.cpp's own padding is not used: `pad_spans` does it, so it clamps and merges the same way everywhere.
        params.set_speech_pad(0);
        let segments = self
            .ctx
            .segments_from_samples(params, samples)
            .map_err(|e| SpeechError::Internal(format!("voice detection failed: {e}")))?;
        let raw: Vec<(f32, f32)> = (0..segments.num_segments())
            .filter_map(|i| {
                Some((
                    segments.get_segment_start_timestamp(i)?,
                    segments.get_segment_end_timestamp(i)?,
                ))
            })
            .collect();
        let audio_ms = samples.len() as u64 * 1000 / 16_000;
        let spans = spans_from_centiseconds(&raw, audio_ms);
        Ok(pad_spans(&spans, settings.padding_ms as u64, audio_ms))
    }
}

/// The speech spans of `samples` (16 kHz mono) as `(start_ms, end_ms)`, in order, on the audio's clock. Empty when
/// there is no speech. `model_dir` is the folder of the catalog's `vad` model. The default [`VadSettings`]. Loads
/// the model on every call; see [`VadDetector`] to load it once.
pub fn detect(model_dir: &Path, samples: &[f32]) -> Result<Vec<(u64, u64)>, SpeechError> {
    detect_with(model_dir, samples, &VadSettings::default())
}

/// [`detect`] with the app's own settings.
pub fn detect_with(
    model_dir: &Path,
    samples: &[f32],
    settings: &VadSettings,
) -> Result<Vec<(u64, u64)>, SpeechError> {
    // A missing model shows up before the short-audio shortcut: a broken install is not hidden by short audio.
    model_file(model_dir)?;
    if is_shorter_than_a_frame(samples) {
        return Ok(vec![]);
    }
    VadDetector::load(model_dir)?.detect_with(samples, settings)
}

#[cfg(test)]
mod tests {
    use super::VadDetector;

    #[test]
    fn the_detector_can_move_between_threads() {
        fn is_send<T: Send>() {}
        is_send::<VadDetector>();
    }
}
