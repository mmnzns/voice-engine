//! Speech to text.
//!
//! * [`load`] and the [`Transcriber`] trait (`api`): one loaded model, audio in, timed segments and words out,
//!   with `Timing::Segments` (Talaan's settings) or `Timing::Words` (Ensayo's). Whisper runs in process
//!   ([`whisper_engine`]: Metal on the Mac, Vulkan with the `vulkan` feature, else the processor); Parakeet,
//!   Parakeet Unified and Nemotron run in process too ([`parakeet_engine`]: ONNX Runtime through parakeet-rs, one
//!   chunk of up to the catalog's `chunk_secs`, 40 s, per call; longer audio is `SpeechError::TooLong`). The thread
//!   rule is in [`threads`].
//! * [`vad::detect`]: where the speech is in a track (whisper.cpp's Silero detector, always on the processor).
//! * [`diarize::load`] and [`diarize::Diarizer::run`]: who is talking, every 10 ms, for up to 8 speakers (NVIDIA's
//!   Nemotron 3 Diarization through parakeet-rs's `sortformer`, always on the processor, on the ONNX thread rule).
//!   Naming speakers and turning frames into turns is the app's.
//! * [`run_helper`]: the speech helper (Windows Vulkan, and a process fallback on any system), protocol 2
//!   (`voice_engine_protocol::stt`), driven by `voice_engine_client::SttClient`. It loads a Whisper model on `load`,
//!   keeps it for every `transcribe`, frees it on `unload`, and returns the same [`Transcript`] as [`load`]. It serves
//!   Whisper only: the ONNX models (Parakeet, Parakeet Unified, Nemotron) run in the app's own process through
//!   [`load`], and a helper `load` of one of them is answered with `unsupported`.
//! * This crate never depends on `voice-engine-llm`: no llama.cpp is linked next to whisper.cpp, so an app can use
//!   it in process (no duplicate ggml symbols).
//!
//! # The graphics card (`LoadOptions::gpu`)
//!
//! * Whisper: Metal on the Mac, Vulkan with the `vulkan` feature.
//! * Parakeet (the TDT models): DirectML through ONNX Runtime, on Windows, with this crate's `directml` feature.
//!   If the provider does not start the model loads on the processor.
//! * Everywhere else `gpu: true` runs the model on the processor. That is reported the same way as a card that
//!   failed: `backend()` says `cpu` with `fallback: Some(NoCardPath)`. This is the case for every ONNX model on the Mac (nothing
//!   here uses CoreML: parakeet-rs measures it slower than the processor for these graphs), for a Windows build
//!   without the `directml` feature, and for the models the catalog gives no DirectML path (Parakeet Unified,
//!   Nemotron). There is no card path to fall back from, but the app asked for one, so it is told, once, and can
//!   pass `gpu: false` from then on.
//!
//! # Options the ONNX models ignore
//!
//! `ChunkOptions::keep_fillers` and `vocabulary` do nothing for Parakeet, Parakeet Unified and Nemotron: they
//! transcribe fillers as heard and have no prompt (the catalog's `keeps_fillers` is measured separately).
//! `language` is a prompt for Nemotron only ("fr" becomes "fr-FR"); the other two ignore it.

mod api;
pub mod diarize;
pub mod directml;
mod helper;
pub mod parakeet_engine;
pub mod threads;
pub mod vad;
pub mod whisper_engine;
pub mod words;

pub use api::{
    ChunkOptions, LoadOptions, Segment, SpeechBackend, SpeechError, Timing, Transcriber,
    Transcript, Word, load,
};
pub use helper::run_helper;
pub use voice_engine_protocol::Fallback;
pub use words::join_segments;

use std::ffi::CStr;
use voice_engine_protocol::gpu::{Device, DeviceKind, bytes_to_mb};
use whisper_rs::whisper_rs_sys as sys;

/// Text that engines produce for non-speech (`[BLANK_AUDIO]`, `(music)`, `*laughs*`) or nothing at all.
pub fn is_noise(text: &str) -> bool {
    let t = text.trim();
    if t.is_empty() {
        return true;
    }
    let wrapped = |o: char, c: char| t.starts_with(o) && t.ends_with(c);
    wrapped('[', ']')
        || wrapped('(', ')')
        || wrapped('*', '*')
        || t.chars().all(|c| !c.is_alphanumeric())
}

/// One piece of text with its time, relative to the chunk, as Talaan's own chunk function returns it.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
}

/// Talaan's own chunk function (16 kHz mono, a loaded state, Whisper's settings as Talaan has them), kept as the
/// reference that `Timing::Segments` is tested against (`tests/real_whisper.rs`). The engines and the helper use
/// [`load`], not this. Pieces are relative to the chunk; the language is what Whisper detected (or was told).
pub fn transcribe(
    state: &mut whisper_rs::WhisperState,
    samples: &[f32],
    language: Option<&str>,
    n_threads: usize,
) -> Result<(Vec<Piece>, Option<String>), String> {
    let mut params =
        whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(n_threads as i32);
    params.set_language(Some(language.unwrap_or("auto")));
    // Each chunk stands alone: carrying text between chunks is how Whisper starts repeating itself.
    params.set_no_context(true);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    // Whisper needs at least 1 s; pad short chunks with silence.
    let mut audio = samples.to_vec();
    if audio.len() < 16_000 + 1_600 {
        audio.resize(16_000 + 1_600, 0.0);
    }
    state
        .full(params, &audio)
        .map_err(|e| format!("transcription failed: {e}"))?;
    let language = whisper_rs::get_lang_str(state.full_lang_id_from_state()).map(str::to_string);
    let mut pieces = vec![];
    for i in 0..state.full_n_segments() {
        let Some(seg) = state.get_segment(i) else {
            continue;
        };
        let text = seg
            .to_str_lossy()
            .map(|t| t.trim().to_string())
            .unwrap_or_default();
        if is_noise(&text) || seg.no_speech_probability() > 0.8 {
            continue;
        }
        // Centiseconds.
        let start_ms = (seg.start_timestamp().max(0) * 10) as u64;
        let end_ms = ((seg.end_timestamp().max(0) * 10) as u64)
            .min(samples.len() as u64 * 1000 / 16_000)
            .max(start_ms);
        pieces.push(Piece {
            start_ms,
            end_ms,
            text,
        });
    }
    Ok((pieces, language))
}

/// Context settings: `gpu_device` is whisper.cpp's GPU number (`gpu::whisper_gpu_index`), None for the processor.
pub fn context_params(
    gpu_device: Option<i32>,
    flash_attn: bool,
) -> whisper_rs::WhisperContextParameters<'static> {
    let mut p = whisper_rs::WhisperContextParameters::default();
    p.use_gpu(gpu_device.is_some());
    if let Some(n) = gpu_device {
        p.gpu_device(n);
    }
    p.flash_attn(flash_attn);
    p
}

/// The graphics adapters DirectML can use (backend "DirectML", `index` DXGI's adapter number), software renderers left
/// out. Empty unless this is a Windows build with the `directml` feature.
pub fn directml_devices() -> Vec<Device> {
    directml::devices(&directml::adapters())
}

/// The discrete graphics card DirectML would run Parakeet on, in this process, with no helper: the DXGI adapter with
/// the most memory, never an integrated one. It lists adapters through DXGI and DXCore only (no model, no ggml), so
/// it is cheap enough for the app's startup. None means no discrete card, and also a build without the `directml`
/// feature (or not Windows), which cannot see adapters: such an app asks the card helper instead
/// (`SttClient::discrete_card`, `LlmClient::discrete_card`).
pub fn discrete_card() -> Option<Device> {
    voice_engine_protocol::gpu::discrete_card(&directml_devices()).cloned()
}

/// The graphics driver version of a card, as Windows shows it ("32.0.15.6614"), for the `Device` that `discrete_card()`
/// (or `directml_devices()`) returned. A DirectML device is matched by its DXGI adapter number (`index`); any other
/// device, such as one from a card helper's `devices()`, by its `description` (the first adapter with that name). None
/// when Windows doesn't say, and in a build without the `directml` feature (or not Windows), which cannot see adapters.
/// An app folds it into its `gpu_key` together with the card's name, so a driver update tries a card that failed
/// before again. The Vulkan helper runs on the same card and driver, so one version covers both.
pub fn driver_version(device: &Device) -> Option<String> {
    directml::driver_version_of(&directml::adapters(), device)
}

/// Every device ggml registered (the CPU, and in the GPU build each Vulkan device).
pub fn devices() -> Vec<Device> {
    let text = |p: *const std::os::raw::c_char| {
        if p.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
        }
    };
    let n = unsafe { sys::ggml_backend_dev_count() };
    (0..n)
        .map(|i| {
            let dev = unsafe { sys::ggml_backend_dev_get(i) };
            let mut props: sys::ggml_backend_dev_props = unsafe { std::mem::zeroed() };
            unsafe { sys::ggml_backend_dev_get_props(dev, &mut props) };
            let backend =
                text(unsafe { sys::ggml_backend_reg_name(sys::ggml_backend_dev_backend_reg(dev)) });
            // ggml_backend_dev_type: 0 CPU, 1 GPU, 2 IGPU, 3 ACCEL.
            let kind = match props.type_ as i64 {
                0 => DeviceKind::Cpu,
                1 => DeviceKind::Gpu,
                2 => DeviceKind::Igpu,
                3 => DeviceKind::Accel,
                _ => DeviceKind::Unknown,
            };
            Device {
                index: i,
                name: text(props.name),
                description: text(props.description),
                backend,
                kind,
                memory_mb: bytes_to_mb(props.memory_total),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_markers_are_not_speech() {
        for t in ["", "  ", "[BLANK_AUDIO]", "(music)", "*laughs*", "...", "♪"] {
            assert!(is_noise(t), "{t:?}");
        }
        for t in ["Okay.", "Sounds good", "(inaudible) let's go"] {
            assert!(!is_noise(t), "{t:?}");
        }
    }

    #[test]
    fn ggml_lists_at_least_the_processor() {
        let all = devices();
        assert!(
            all.iter()
                .any(|d| d.kind == DeviceKind::Cpu && d.backend == "CPU"),
            "{all:?}"
        );
        assert!(all.iter().enumerate().all(|(i, d)| d.index == i));
    }
}
