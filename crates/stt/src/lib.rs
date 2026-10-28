//! Whisper, the same way everywhere: an app's in-process engine (processor on Windows, Metal on the Mac) and the
//! Windows GPU helper ([`run_helper`]) both transcribe a chunk with `transcribe` below, so a GPU transcript differs
//! from a processor one only by what the hardware computes.

mod helper;
pub use helper::run_helper;

use std::ffi::CStr;
use voice_engine_protocol::gpu::{Device, DeviceKind, bytes_to_mb};
use voice_engine_protocol::stt::SttPiece;
use whisper_rs::whisper_rs_sys as sys;

/// Text that engines produce for non-speech ("[BLANK_AUDIO]", "(music)", "*laughs*") or nothing at all.
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

/// Speech threads on this computer.
pub fn threads() -> usize {
    thread_count(
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4),
        cfg!(windows),
    )
}

/// Mac: all cores, 2..=8 (the original formula). Windows: three quarters of the logical cores, 2..=12. Measured on a
/// 16-thread Ryzen: 12 threads is fastest and 16 is 2-3x slower.
pub fn thread_count(logical: usize, windows: bool) -> usize {
    if windows {
        (logical * 3 / 4).clamp(2, 12)
    } else {
        logical.clamp(2, 8)
    }
}

/// Transcribes one chunk (16 kHz mono) with a loaded state. Pieces are relative to the chunk; the language is
/// what Whisper detected (or was told).
pub fn transcribe(
    state: &mut whisper_rs::WhisperState,
    samples: &[f32],
    language: Option<&str>,
    n_threads: usize,
) -> Result<(Vec<SttPiece>, Option<String>), String> {
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
        pieces.push(SttPiece {
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
    fn windows_thread_count_is_three_quarters_clamped() {
        for (logical, want) in [(16, 12), (8, 6), (4, 3), (2, 2), (1, 2), (32, 12)] {
            assert_eq!(thread_count(logical, true), want, "{logical} cores");
        }
    }

    #[test]
    fn mac_thread_count_is_all_cores_clamped() {
        for (logical, want) in [(16, 8), (10, 8), (8, 8), (4, 4), (2, 2), (1, 2)] {
            assert_eq!(thread_count(logical, false), want, "{logical} cores");
        }
    }

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
