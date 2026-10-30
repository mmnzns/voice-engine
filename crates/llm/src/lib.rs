//! The local AI helper. Runs llama.cpp in its own process so its copy of ggml never meets Whisper's, and so a
//! model crash or out-of-memory never takes the app down. Each app's helper `main.rs` calls [`run`].
//!
//! Reads one JSON request per line on stdin, writes JSON events per line on stdout (protocol crate). Exits when
//! stdin closes or after a while without requests. Logs go to stderr.
//!
//! Windows ships two builds: a processor-only one and a Vulkan one (`--features vulkan`), which the app starts only
//! when System32 has a Vulkan loader. The GPU build uses one device, the discrete GPU with the most memory
//! (`voice_engine_protocol::gpu::pick`), never splits a model across devices, and lets llama.cpp fit as many layers
//! as that GPU's memory allows (`fit_params`). The Mac build: Metal, every layer. In every build a request with
//! `gpu: false` runs on the processor, with no layer on any graphics device, and a model is loaded again when a
//! request asks for the other place (`model::place`, `model::Key`).

mod failure;
mod generate;
mod inbox;
mod inspect;
mod model;
mod think;

use failure::Failure;
use generate::{EMBED_MAX_TOKENS, complete, embed};
use model::{Loaded, Place, Target, place_here};

use llama_cpp_2::llama_backend::LlamaBackend;
use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::time::Duration;
use voice_engine_protocol::gpu::{Device, DeviceKind, bytes_to_mb, pick};
use voice_engine_protocol::{
    Backend, ErrorKind, Event, HelperConfig, PROTOCOL_VERSION, Request, decode_request, encode,
};

/// Without a request for this long the helper exits, which frees all of the model's memory. The app starts it
/// again on the next request, or sends `configure` to change it. `<prefix>_LLM_IDLE_SECS` sets it for tests.
const IDLE_EXIT: Duration = Duration::from_secs(120);

fn idle_exit(config: &HelperConfig) -> Option<Duration> {
    Some(
        std::env::var(config.idle_secs_env())
            .ok()
            .and_then(|v| v.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or(IDLE_EXIT),
    )
}

pub(crate) fn send(ev: &Event) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", encode(ev));
    let _ = out.flush();
}

/// Every device ggml registered (the CPU, and in the GPU build each Vulkan device).
fn devices() -> Vec<Device> {
    llama_cpp_2::list_llama_ggml_backend_devices()
        .into_iter()
        .map(|d| Device {
            index: d.index,
            name: d.name,
            description: d.description,
            backend: d.backend,
            kind: match d.device_type {
                llama_cpp_2::LlamaBackendDeviceType::Cpu => DeviceKind::Cpu,
                llama_cpp_2::LlamaBackendDeviceType::Gpu => DeviceKind::Gpu,
                llama_cpp_2::LlamaBackendDeviceType::IntegratedGpu => DeviceKind::Igpu,
                llama_cpp_2::LlamaBackendDeviceType::Accelerator => DeviceKind::Accel,
                llama_cpp_2::LlamaBackendDeviceType::Unknown => DeviceKind::Unknown,
            },
            memory_mb: bytes_to_mb(d.memory_total),
        })
        .collect()
}

/// The GPU this helper uses, chosen here everywhere but the Mac (a CPU build lists no GPU, so it's None there).
/// On the Mac llama.cpp keeps choosing Metal itself, exactly as before.
fn choose(devices: &[Device]) -> Option<Device> {
    if cfg!(target_os = "macos") {
        return None;
    }
    pick(devices).cloned()
}

/// What a request's model runs on, said truthfully: `cpu` for the processor (every request with `gpu: false`), `metal`
/// on the Mac's card, and the device for a Vulkan card.
fn backend(place: Place) -> Backend {
    match place {
        Place::Gpu(d) => Backend {
            backend: d.backend.to_ascii_lowercase(),
            device: Some(d.description.clone()),
            vram_mb: d.memory_mb,
        },
        Place::Metal => Backend {
            backend: "metal".into(),
            device: None,
            vram_mb: 0,
        },
        Place::Processor => Backend::cpu(),
    }
}

/// `<prefix>_GPU=fail` makes the GPU build die on its first real request, the way a driver crash would, so the
/// fallback can be tried on a real PC. The CPU build ignores it.
fn fail_on_purpose(config: &HelperConfig) {
    let env = config.gpu_env();
    if cfg!(feature = "vulkan") && std::env::var(&env).is_ok_and(|v| v == "fail") {
        eprintln!("ggml_vulkan: {env}=fail, stopping on purpose");
        std::process::abort();
    }
}

/// Runs the helper until the app closes stdin or the idle timeout passes. Called by each app's helper `main.rs`.
pub fn run(config: HelperConfig) {
    // Quiet llama.cpp's own logging; errors still come back as events.
    llama_cpp_2::send_logs_to_tracing(llama_cpp_2::LogOptions::default().with_logs_enabled(false));
    let llama = match LlamaBackend::init() {
        Ok(b) => b,
        Err(e) => {
            return send(&Failure::internal(format!("couldn't start the local AI: {e}")).event());
        }
    };
    let all = devices();
    let gpu = choose(&all);

    // stdin on its own thread so the main loop can time out when idle.
    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut inbox = inbox::Inbox::new(rx);
    let mut idle = idle_exit(&config);
    let mut loaded: Option<Loaded> = None;
    // Ends when idle or when stdin closes: free everything by exiting.
    while let inbox::Next::Line(line) = inbox.next(idle) {
        match decode_request(&line) {
            Ok(Request::Configure(c)) => {
                idle = c.idle_secs.map(Duration::from_secs);
                send(&Event::Configured {
                    idle_secs: c.idle_secs,
                });
            }
            // A cancel with nothing running: the reply it meant has already finished.
            Ok(Request::Cancel) => {}
            Ok(Request::Ping) => send(&Event::Ready {
                // What a request that asks for the card (the default) gets.
                backend: backend(place_here(true, gpu.as_ref())),
                protocol: PROTOCOL_VERSION,
            }),
            Ok(Request::Devices) => send(&Event::Devices {
                devices: all.clone(),
                chosen: gpu.as_ref().map(|d| d.index),
            }),
            Ok(Request::Complete(c)) => {
                fail_on_purpose(&config);
                let target = Target {
                    path: &c.model_path,
                    place: place_here(c.gpu, gpu.as_ref()),
                    n_ctx: c.n_ctx,
                };
                match complete(&llama, &mut loaded, target, &c, &mut inbox) {
                    Ok(reply) => send(&Event::Done(reply)),
                    Err(f) => send(&f.event()),
                }
            }
            Ok(Request::Load(l)) => {
                let target = Target {
                    path: &l.model_path,
                    place: place_here(l.gpu, gpu.as_ref()),
                    n_ctx: l.n_ctx,
                };
                match model::load(&llama, &mut loaded, target) {
                    Ok((model, seconds)) => send(&Event::Loaded {
                        seconds,
                        n_ctx_train: model.n_ctx_train(),
                        backend: backend(target.place),
                    }),
                    Err(f) => send(&f.event()),
                }
            }
            Ok(Request::Inspect(i)) => match inspect::inspect(&i.model_path) {
                Ok(found) => send(&Event::Inspected(found)),
                Err(f) => send(&f.event()),
            },
            Ok(Request::Unload) => {
                loaded = None;
                send(&Event::Unloaded);
            }
            Ok(Request::Embed(e)) => {
                fail_on_purpose(&config);
                let target = Target {
                    path: &e.model_path,
                    place: place_here(e.gpu, gpu.as_ref()),
                    n_ctx: EMBED_MAX_TOKENS as u32,
                };
                match embed(&llama, &mut loaded, target, &e, &mut inbox) {
                    Ok((vectors, backend)) => send(&Event::Embedded { vectors, backend }),
                    Err(f) => send(&f.event()),
                }
            }
            Err(e) => {
                send(&Failure::new(ErrorKind::BadRequest, format!("bad request: {e}")).event())
            }
        }
    }
}

/// llama.cpp's backend can exist once per process, so every test that starts it (here and in the other files)
/// takes this turn first.
#[cfg(test)]
pub(crate) static BACKEND_TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cpu_build_reports_the_processor_and_lists_the_cpu() {
        let _turn = BACKEND_TURN.lock().unwrap_or_else(|e| e.into_inner());
        let _llama = LlamaBackend::init().unwrap();
        let all = devices();
        assert!(all.iter().any(|d| d.kind == DeviceKind::Cpu), "{all:?}");
        if !cfg!(feature = "vulkan") {
            assert_eq!(choose(&all), None);
            if cfg!(not(target_os = "macos")) {
                assert_eq!(backend(place_here(true, None)), Backend::cpu());
            }
        }
    }

    #[test]
    fn the_idle_exit_follows_the_app_prefix_and_defaults_to_two_minutes() {
        let config = HelperConfig {
            env_prefix: "VOICE_ENGINE_UNIT_TEST",
        };
        assert_eq!(idle_exit(&config), Some(IDLE_EXIT));
        // SAFETY: this variable name is private to this test, and no other test reads it.
        unsafe { std::env::set_var(config.idle_secs_env(), "7") };
        assert_eq!(idle_exit(&config), Some(Duration::from_secs(7)));
        unsafe { std::env::set_var(config.idle_secs_env(), "not a number") };
        assert_eq!(idle_exit(&config), Some(IDLE_EXIT));
        unsafe { std::env::remove_var(config.idle_secs_env()) };
    }

    #[test]
    fn the_backend_says_where_the_model_really_is() {
        let d = Device {
            index: 0,
            name: "Vulkan0".into(),
            description: "Test card".into(),
            backend: "Vulkan".into(),
            kind: DeviceKind::Gpu,
            memory_mb: 4096,
        };
        assert_eq!(backend(Place::Processor), Backend::cpu());
        assert!(!backend(Place::Processor).on_gpu());
        let metal = backend(Place::Metal);
        assert_eq!(
            (metal.backend.as_str(), metal.device, metal.vram_mb),
            ("metal", None, 0)
        );
        assert_eq!(backend(Place::Gpu(&d)).backend, "vulkan");
        // Through this build's rules: `gpu: false` is `cpu` everywhere, with or without a chosen device.
        for chosen in [None, Some(&d)] {
            assert_eq!(backend(place_here(false, chosen)), Backend::cpu());
        }
        let wanted = if cfg!(target_os = "macos") && !cfg!(feature = "vulkan") {
            "metal"
        } else {
            "cpu"
        };
        assert_eq!(backend(place_here(true, None)).backend, wanted);
        assert_eq!(backend(place_here(true, Some(&d))).backend, "vulkan");
    }

    #[test]
    fn the_gpu_backend_names_the_device() {
        let d = Device {
            index: 1,
            name: "Vulkan1".into(),
            description: "NVIDIA GeForce RTX 5070 Ti".into(),
            backend: "Vulkan".into(),
            kind: DeviceKind::Gpu,
            memory_mb: 16303,
        };
        let b = backend(Place::Gpu(&d));
        assert_eq!(
            (b.backend.as_str(), b.device.as_deref(), b.vram_mb),
            ("vulkan", Some("NVIDIA GeForce RTX 5070 Ti"), 16303)
        );
        assert!(b.on_gpu());
    }
}
