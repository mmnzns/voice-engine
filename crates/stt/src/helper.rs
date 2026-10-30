//! The speech helper: Whisper in its own process, protocol 2 (`voice_engine_protocol::stt`). Built with
//! `--features vulkan` it is an app's GPU path on Windows: the app keeps a processor-only whisper.cpp (voice
//! detection, and the fallback), so it never needs `vulkan-1.dll`, and a driver crash ends this process instead of the
//! app. Built without it (and on the Mac, with Metal) it is the process fallback `SttClient` uses on any system.
//!
//! It serves Whisper only: Parakeet, Parakeet Unified and Nemotron run through ONNX Runtime inside the app's own
//! process (`voice_engine_stt::load`), so a `load` of one of them is answered with `unsupported`.
//!
//! One JSON request per line on stdin, events on stdout, one request at a time. `load` reads the model and keeps it
//! loaded for every `transcribe` until `unload` (which frees it, the helper keeps running), another `load`, or the
//! helper's exit. A `load` that fails leaves nothing loaded. It exits when stdin closes (the app quit or crashed),
//! so an update or a crash never leaves a helper holding a model in memory; it has no idle timer, the app unloads or
//! closes it (`SttClient`).
//!
//! `cancel` is read between requests and ignored: a decode in progress cannot be interrupted (the `Transcriber` API
//! has no hook for it). The client gives the helper a couple of seconds and then ends the process.
//!
//! It writes nothing to stderr but the one line of `<PREFIX>_GPU=fail`; no audio, text, vocabulary or file name the
//! person chose is ever logged, and every error message is built from fixed words (`tests/privacy.rs`).

use crate::api::{ChunkOptions, LoadOptions, SpeechError, Transcriber, load};
use crate::whisper_engine::choose_gpu;
use std::io::{BufRead, Write};
use std::path::Path;
use std::time::Instant;
use voice_engine_models::catalog::{self, Kind, ModelInfo};
use voice_engine_protocol::gpu::Device;
use voice_engine_protocol::stt::{
    SttErrorKind, SttEvent, SttLoad, SttModelKind, SttRequest, decode_request, decode_samples,
};
use voice_engine_protocol::{Backend, HelperConfig, PROTOCOL_VERSION, encode};

fn send(ev: &SttEvent) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", encode(ev));
    let _ = out.flush();
}

/// A failed request: what kind (for the app) and a message made of fixed words (for logs).
#[derive(Debug, Clone, PartialEq)]
struct Failure {
    error: SttErrorKind,
    message: String,
}

impl Failure {
    fn new(error: SttErrorKind, message: &str) -> Self {
        Failure {
            error,
            message: message.into(),
        }
    }

    fn event(self) -> SttEvent {
        SttEvent::Error {
            message: self.message,
            error: self.error,
        }
    }
}

/// The error for a request line that did not decode. Only serde's category and position: its own text can quote the
/// value that did not fit, and that value can be what a person said.
fn bad_request(e: &serde_json::Error) -> Failure {
    Failure::new(
        SttErrorKind::BadRequest,
        &format!(
            "bad request: {:?} at line {} column {}",
            e.classify(),
            e.line(),
            e.column()
        ),
    )
}

/// What the engine said, as the kind the app matches on. `file` is where the model should have been.
fn failure(e: SpeechError, file: &Path) -> Failure {
    // `SpeechError`'s own text is fixed words and engine codes (tests/privacy.rs), so it is safe in a message.
    let message = e.to_string();
    let error = match e {
        SpeechError::ModelMissing => SttErrorKind::ModelMissing {
            path: file.display().to_string(),
        },
        SpeechError::BadModel => SttErrorKind::BadModel,
        SpeechError::OutOfMemory => SttErrorKind::OutOfMemory,
        SpeechError::Unsupported(detail) => SttErrorKind::Unsupported { detail },
        SpeechError::TooLong { max_secs } => SttErrorKind::TooLong { max_secs },
        SpeechError::Internal(_) => SttErrorKind::Internal,
    };
    Failure { error, message }
}

/// The catalog entry a `load` is for, or why the helper cannot serve it.
fn check_load(request: &SttLoad) -> Result<&'static ModelInfo, Failure> {
    if request.kind != SttModelKind::Whisper {
        return Err(Failure::new(
            SttErrorKind::Unsupported {
                detail: "this helper runs Whisper only; the other speech models run in the app"
                    .into(),
            },
            "not supported: this helper runs Whisper only",
        ));
    }
    match catalog::find(&request.model_id) {
        Some(model) if model.kind == Kind::Whisper && !model.files.is_empty() => Ok(model),
        _ => Err(Failure::new(
            SttErrorKind::BadRequest,
            "bad request: not a Whisper model of the catalog",
        )),
    }
}

/// The helper's `Ready` backend: what a `load` with `gpu: true` will run on (the engine's own choice: Metal on the
/// Mac, the discrete card with the most memory elsewhere, never an integrated one), or the processor.
fn ready_backend(devices: &[Device], mac: bool) -> Backend {
    match choose_gpu(devices, mac) {
        Some((d, _)) => Backend {
            backend: d.backend.to_ascii_lowercase(),
            device: Some(d.description.clone()),
            vram_mb: d.memory_mb,
        },
        None => Backend::cpu(),
    }
}

fn load_model(request: &SttLoad) -> Result<(Box<dyn Transcriber>, f32), Failure> {
    let model = check_load(request)?;
    let dir = Path::new(&request.model_dir);
    let began = Instant::now();
    let engine = load(
        model,
        dir,
        LoadOptions {
            gpu: request.gpu,
            threads: request.threads,
        },
    )
    .map_err(|e| failure(e, &dir.join(model.files[0].name)))?;
    Ok((engine, began.elapsed().as_secs_f32()))
}

fn transcribe(
    engine: &mut dyn Transcriber,
    audio_b64: &str,
    options: &ChunkOptions,
) -> Result<SttEvent, Failure> {
    let audio =
        decode_samples(audio_b64).map_err(|m| Failure::new(SttErrorKind::BadRequest, &m))?;
    let transcript = engine
        .transcribe(&audio, options)
        .map_err(|e| failure(e, Path::new("")))?;
    Ok(SttEvent::Transcribed {
        transcript,
        backend: engine.backend(),
    })
}

/// `<prefix>_GPU=fail` makes the GPU build die on its first transcription, the way a driver crash would, so the
/// fallback can be tried on a real PC. Only a Vulkan build with the model on the card.
fn fail_on_purpose(config: &HelperConfig, engine: &dyn Transcriber) {
    let env = config.gpu_env();
    if cfg!(feature = "vulkan")
        && engine.backend().backend != "cpu"
        && std::env::var(&env).is_ok_and(|v| v == "fail")
    {
        eprintln!("ggml_vulkan: {env}=fail, stopping on purpose");
        std::process::abort();
    }
}

/// Runs the speech helper until the app closes stdin. Called by each app's helper `main.rs`.
pub fn run_helper(config: HelperConfig) {
    // whisper.cpp and ggml log to stderr by default; drop it (errors come back as events).
    whisper_rs::install_logging_hooks();
    // ggml's devices first (their indices are ggml's), then DirectML's adapters when this build has them.
    let mut all = crate::devices();
    all.extend(crate::directml_devices());
    let mac = cfg!(target_os = "macos");

    let mut loaded: Option<Box<dyn Transcriber>> = None;
    for line in std::io::stdin().lock().lines() {
        let Ok(line) = line else { break };
        match decode_request(&line) {
            Ok(SttRequest::Ping) => send(&SttEvent::Ready {
                backend: ready_backend(&all, mac),
                protocol: PROTOCOL_VERSION,
            }),
            Ok(SttRequest::Devices) => send(&SttEvent::Devices {
                devices: all.clone(),
                chosen: choose_gpu(&all, mac).map(|(d, _)| d.index),
            }),
            // A cancel with nothing running, or one for a decode that cannot be interrupted: nothing to answer.
            Ok(SttRequest::Cancel) => {}
            Ok(SttRequest::Unload) => {
                loaded = None;
                send(&SttEvent::Unloaded);
            }
            Ok(SttRequest::Load(request)) => {
                // The old model's memory goes first; a load that fails leaves nothing loaded.
                loaded = None;
                match load_model(&request) {
                    Ok((engine, seconds)) => {
                        send(&SttEvent::Loaded {
                            seconds,
                            backend: engine.backend(),
                        });
                        loaded = Some(engine);
                    }
                    Err(f) => send(&f.event()),
                }
            }
            Ok(SttRequest::Transcribe { audio_b64, options }) => {
                let Some(engine) = loaded.as_deref_mut() else {
                    send(&Failure::new(SttErrorKind::BadRequest, "no speech model loaded").event());
                    continue;
                };
                fail_on_purpose(&config, engine);
                match transcribe(engine, &audio_b64, &options) {
                    Ok(event) => send(&event),
                    Err(f) => send(&f.event()),
                }
            }
            Err(e) => send(&bad_request(&e).event()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_of(id: &str, kind: SttModelKind) -> SttLoad {
        SttLoad {
            model_id: id.into(),
            model_dir: "/m".into(),
            kind,
            gpu: false,
            threads: None,
        }
    }

    #[test]
    fn every_engine_error_becomes_its_own_kind() {
        let file = Path::new("/m/ggml-base.bin");
        let cases = [
            (
                SpeechError::ModelMissing,
                SttErrorKind::ModelMissing {
                    path: "/m/ggml-base.bin".into(),
                },
            ),
            (SpeechError::BadModel, SttErrorKind::BadModel),
            (SpeechError::OutOfMemory, SttErrorKind::OutOfMemory),
            (
                SpeechError::Unsupported("nope".into()),
                SttErrorKind::Unsupported {
                    detail: "nope".into(),
                },
            ),
            (
                SpeechError::TooLong { max_secs: 40 },
                SttErrorKind::TooLong { max_secs: 40 },
            ),
            (SpeechError::Internal("code".into()), SttErrorKind::Internal),
        ];
        for (error, want) in cases {
            assert_eq!(failure(error, file).error, want);
        }
        // The message is the engine's own fixed words and never carries the path.
        assert!(
            !failure(SpeechError::ModelMissing, file)
                .message
                .contains("ggml")
        );
    }

    #[test]
    fn only_whisper_models_of_the_catalog_are_served() {
        assert!(check_load(&load_of("whisper-turbo", SttModelKind::Whisper)).is_ok());
        for kind in [
            SttModelKind::Parakeet,
            SttModelKind::ParakeetUnified,
            SttModelKind::Nemotron,
        ] {
            let f = check_load(&load_of("parakeet-ultra", kind)).unwrap_err();
            assert!(
                matches!(f.error, SttErrorKind::Unsupported { .. }),
                "{kind:?}"
            );
        }
        for id in ["no-such-model", "parakeet-ultra", "vad", "qwen3.5-4b", ""] {
            let f = check_load(&load_of(id, SttModelKind::Whisper)).unwrap_err();
            assert_eq!(f.error, SttErrorKind::BadRequest, "{id}");
        }
    }

    #[test]
    fn a_bad_request_does_not_echo_the_sentence_it_carried() {
        let said = "remember to call my doctor about the rash on Tuesday";
        // A string where a bool belongs: serde's own message quotes the string.
        let line = format!(r#"{{"op":"load","gpu":"{said}"}}"#);
        let e = voice_engine_protocol::stt::decode_request(&line).unwrap_err();
        assert!(e.to_string().contains("doctor"), "the premise: {e}");
        let f = bad_request(&e);
        assert_eq!(f.error, SttErrorKind::BadRequest);
        assert!(
            !f.message.contains("doctor") && !f.message.contains("rash"),
            "{}",
            f.message
        );
        assert!(f.message.contains("bad request"), "{}", f.message);
    }

    #[test]
    fn ready_names_the_card_the_engine_will_use() {
        use voice_engine_protocol::gpu::DeviceKind;
        let dev = |index, desc: &str, backend: &str, kind, mb| Device {
            index,
            name: format!("{backend}{index}"),
            description: desc.into(),
            backend: backend.into(),
            kind,
            memory_mb: mb,
        };
        let windows = [
            dev(
                0,
                "AMD Radeon(TM) Graphics",
                "Vulkan",
                DeviceKind::Igpu,
                512,
            ),
            dev(
                1,
                "NVIDIA GeForce RTX 5070 Ti",
                "Vulkan",
                DeviceKind::Gpu,
                16303,
            ),
            dev(2, "CPU", "CPU", DeviceKind::Cpu, 64000),
        ];
        let b = ready_backend(&windows, false);
        assert_eq!(
            (b.backend.as_str(), b.device.as_deref(), b.vram_mb),
            ("vulkan", Some("NVIDIA GeForce RTX 5070 Ti"), 16303)
        );
        assert!(b.on_gpu());
        // No discrete card: the processor.
        assert_eq!(ready_backend(&windows[2..], false), Backend::cpu());
        assert_eq!(
            ready_backend(&[windows[0].clone(), windows[2].clone()], false),
            Backend::cpu()
        );
        // The Mac's Metal device.
        let mac = [
            dev(0, "Apple M3 Pro", "Metal", DeviceKind::Gpu, 18000),
            dev(1, "CPU", "CPU", DeviceKind::Cpu, 18000),
        ];
        assert_eq!(ready_backend(&mac, true).backend, "metal");
    }
}
