//! The Whisper helper: Whisper in its own process. Built with `--features vulkan` it is an app's GPU path on
//! Windows: the app keeps a processor-only whisper.cpp (voice detection, and the fallback), so it never needs
//! `vulkan-1.dll`, and a driver crash ends this process instead of the app.
//!
//! One JSON request per line on stdin, events on stdout (`voice_engine_protocol::stt`). It uses one device, the
//! discrete GPU with the most memory, never an integrated one; with none it says `cpu` and the app transcribes in
//! process instead. It runs until stdin closes (the app unloads it after a while without meetings).

use std::io::{BufRead, Write};
use voice_engine_protocol::gpu::{Device, pick, whisper_gpu_index};
use voice_engine_protocol::stt::{SttEvent, SttRequest, decode_samples};
use voice_engine_protocol::{Backend, HelperConfig, PROTOCOL_VERSION, encode};

fn send(ev: &SttEvent) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", encode(ev));
    let _ = out.flush();
}

fn backend(gpu: Option<&Device>) -> Backend {
    match gpu {
        Some(d) => Backend {
            backend: d.backend.to_ascii_lowercase(),
            device: Some(d.description.clone()),
            vram_mb: d.memory_mb,
        },
        None => Backend::cpu(),
    }
}

struct Loaded {
    // The state holds ~400 MB of buffers; one per model, reused for every chunk. Dropped before the context.
    state: whisper_rs::WhisperState,
    _ctx: whisper_rs::WhisperContext,
}

fn load(path: &str, gpu: Option<(&Device, i32)>) -> Result<Loaded, String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("couldn't load the speech model: {path} is missing"));
    }
    let params = crate::context_params(gpu.map(|(_, n)| n), false);
    let ctx = whisper_rs::WhisperContext::new_with_params(path, params)
        .map_err(|e| format!("couldn't load the speech model: {e}"))?;
    let state = ctx
        .create_state()
        .map_err(|e| format!("couldn't load the speech model: {e}"))?;
    Ok(Loaded { state, _ctx: ctx })
}

/// Runs the Whisper helper until the app closes stdin. Called by each app's GPU helper `main.rs`.
pub fn run_helper(config: HelperConfig) {
    // whisper.cpp and ggml log to stderr by default; drop it (errors come back as events).
    whisper_rs::install_logging_hooks();
    let all = crate::devices();
    // The Mac never runs this helper; elsewhere the policy picks the GPU.
    let chosen = if cfg!(target_os = "macos") {
        None
    } else {
        pick(&all).cloned()
    };
    let gpu = chosen
        .as_ref()
        .and_then(|d| whisper_gpu_index(&all, d).map(|n| (d, n)));
    let threads = crate::threads();

    let mut loaded: Option<Loaded> = None;
    let mut on_gpu = false;
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        // A UTF-8 byte order mark (a file piped in by hand) isn't part of the request.
        match serde_json::from_str::<SttRequest>(line.trim_start_matches('\u{feff}')) {
            Ok(SttRequest::Ping) => send(&SttEvent::Ready {
                backend: backend(gpu.map(|(d, _)| d)),
                protocol: PROTOCOL_VERSION,
            }),
            Ok(SttRequest::Devices) => send(&SttEvent::Devices {
                devices: all.clone(),
                chosen: chosen.as_ref().map(|d| d.index),
            }),
            Ok(SttRequest::Load {
                model_path,
                gpu: want,
            }) => {
                loaded = None;
                let target = gpu.filter(|_| want);
                match load(&model_path, target) {
                    Ok(l) => {
                        loaded = Some(l);
                        on_gpu = target.is_some();
                        send(&SttEvent::Loaded {
                            backend: backend(target.map(|(d, _)| d)),
                        });
                    }
                    Err(message) => send(&SttEvent::Error { message }),
                }
            }
            Ok(SttRequest::Transcribe { samples, language }) => {
                // `<prefix>_GPU=fail`: die like a driver crash would, to try the fallback on a real PC.
                let fail_env = config.gpu_env();
                if cfg!(feature = "vulkan")
                    && on_gpu
                    && std::env::var(&fail_env).is_ok_and(|v| v == "fail")
                {
                    eprintln!("ggml_vulkan: {fail_env}=fail, stopping on purpose");
                    std::process::abort();
                }
                let Some(l) = loaded.as_mut() else {
                    send(&SttEvent::Error {
                        message: "no speech model loaded".into(),
                    });
                    continue;
                };
                let result = decode_samples(&samples).and_then(|audio| {
                    crate::transcribe(&mut l.state, &audio, language.as_deref(), threads)
                });
                match result {
                    Ok((pieces, language)) => send(&SttEvent::Transcribed { pieces, language }),
                    Err(message) => send(&SttEvent::Error { message }),
                }
            }
            Err(e) => send(&SttEvent::Error {
                message: format!("bad request: {e}"),
            }),
        }
    }
}
