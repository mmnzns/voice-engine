//! The local AI helper. Runs llama.cpp in its own process so its copy of ggml never meets Whisper's, and so a
//! model crash or out-of-memory never takes the app down. Each app's helper `main.rs` calls [`run`].
//!
//! Reads one JSON request per line on stdin, writes JSON events per line on stdout (protocol crate). Exits when
//! stdin closes or after a while without requests. Logs go to stderr.
//!
//! Windows ships two builds: a processor-only one and a Vulkan one (`--features vulkan`), which the app starts only
//! when System32 has a Vulkan loader. The GPU build uses one device, the discrete GPU with the most memory
//! (`voice_engine_protocol::gpu::pick`), never splits a model across devices, and lets llama.cpp fit as many layers
//! as that GPU's memory allows (`fit_params`). The Mac build: Metal, every layer.

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::context::params::LlamaPoolingType;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{LlamaChatMessage, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::io::{BufRead, Write};
use std::num::NonZeroU32;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use voice_engine_protocol::gpu::{Device, DeviceKind, bytes_to_mb, pick};
use voice_engine_protocol::{
    Backend, Completion, Embedding, Event, HelperConfig, PROTOCOL_VERSION, Phase, Reply, Request,
    decode_request, encode,
};

/// Without a request for this long the helper exits, which frees all of the model's memory. The app starts it
/// again on the next request. `<prefix>_LLM_IDLE_SECS` shortens it (tests only).
const IDLE_EXIT: Duration = Duration::from_secs(120);
const BATCH: usize = 512;

fn idle_exit(config: &HelperConfig) -> Duration {
    std::env::var(config.idle_secs_env())
        .ok()
        .and_then(|v| v.parse().ok())
        .map(Duration::from_secs)
        .unwrap_or(IDLE_EXIT)
}

fn send(ev: &Event) {
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

fn backend(gpu: Option<&Device>) -> Backend {
    match gpu {
        Some(d) => Backend {
            backend: d.backend.to_ascii_lowercase(),
            device: Some(d.description.clone()),
            vram_mb: d.memory_mb,
        },
        None if cfg!(target_os = "macos") => Backend {
            backend: "metal".into(),
            device: None,
            vram_mb: 0,
        },
        None => Backend::cpu(),
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
            return send(&Event::Error {
                message: format!("couldn't start the local AI: {e}"),
            });
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

    let idle = idle_exit(&config);
    let mut loaded: Option<Loaded> = None;
    // Ends when idle or when stdin closes: free everything by exiting.
    while let Ok(line) = rx.recv_timeout(idle) {
        match decode_request(&line) {
            Ok(Request::Ping) => send(&Event::Ready {
                backend: backend(gpu.as_ref()),
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
                    gpu: gpu.as_ref().filter(|_| c.gpu),
                    n_ctx: c.n_ctx,
                };
                match complete(&llama, &mut loaded, target, &c) {
                    Ok(reply) => send(&Event::Done(reply)),
                    Err(message) => send(&Event::Error { message }),
                }
            }
            Ok(Request::Embed(e)) => {
                fail_on_purpose(&config);
                let target = Target {
                    path: &e.model_path,
                    gpu: gpu.as_ref().filter(|_| e.gpu),
                    n_ctx: EMBED_MAX_TOKENS as u32,
                };
                match embed(&llama, &mut loaded, target, &e) {
                    Ok(vectors) => send(&Event::Embedded { vectors }),
                    Err(message) => send(&Event::Error { message }),
                }
            }
            Err(e) => send(&Event::Error {
                message: format!("bad request: {e}"),
            }),
        }
    }
}

/// What a request needs loaded: a model file, on the chosen GPU or the processor, fitted for a context size.
#[derive(Clone, Copy)]
struct Target<'a> {
    path: &'a str,
    gpu: Option<&'a Device>,
    n_ctx: u32,
}

struct Loaded {
    path: String,
    gpu: Option<usize>,
    n_ctx: u32,
    model: LlamaModel,
}

/// GPU memory left free for the driver, the desktop and other apps (llama.cpp's own default margin).
const GPU_MARGIN: usize = 1024 * 1024 * 1024;

/// Model settings for a target. On the GPU: only that device, no split, and as many layers as fit next to the
/// context (fit_params lowers the layer count when the GPU is too small, so a 9B model still uses part of a 6 GB
/// card). On the processor in a GPU build: no layers on the GPU. Otherwise llama.cpp as before (Metal: all).
fn model_params(t: Target) -> Result<LlamaModelParams, String> {
    let Some(dev) = t.gpu else {
        let layers = if cfg!(feature = "vulkan") { 0 } else { 999 };
        return Ok(LlamaModelParams::default().with_n_gpu_layers(layers));
    };
    let mut params = LlamaModelParams::default()
        .with_split_mode(LlamaSplitMode::None)
        .with_main_gpu(0)
        .with_devices(&[dev.index])
        .map_err(|e| format!("couldn't use the graphics chip: {e}"))?;
    let path = std::ffi::CString::new(t.path).map_err(|_| "bad model path".to_string())?;
    let mut cparams = context_params(t.n_ctx, true);
    let mut margins = vec![GPU_MARGIN; llama_cpp_2::max_devices().max(1)];
    // Log level 4 is GGML_LOG_LEVEL_ERROR: fitting reports only problems.
    std::pin::Pin::new(&mut params)
        .fit_params(&path, &mut cparams, &mut margins, t.n_ctx, 4)
        .map_err(|e| format!("the AI model doesn't fit in memory: {e}"))?;
    Ok(params)
}

fn load<'a>(
    llama: &LlamaBackend,
    loaded: &'a mut Option<Loaded>,
    t: Target,
) -> Result<&'a LlamaModel, String> {
    let gpu = t.gpu.map(|d| d.index);
    // The context size only changes how a model is fitted on a GPU (`fit_params`); on the processor or Metal a
    // different size must not reload the model, so it is left out of the key there.
    let n_ctx = if gpu.is_some() { t.n_ctx } else { 0 };
    let same = loaded
        .as_ref()
        .is_some_and(|l| l.path == t.path && l.gpu == gpu && l.n_ctx == n_ctx);
    if !same {
        *loaded = None;
        send(&Event::Progress {
            phase: Phase::Loading,
            done: 0,
            total: 0,
        });
        if !std::path::Path::new(t.path).is_file() {
            return Err(format!("couldn't load the AI model: {} is missing", t.path));
        }
        let params = model_params(t)?;
        let model = LlamaModel::load_from_file(llama, t.path, &params)
            .map_err(|e| format!("couldn't load the AI model: {e}"))?;
        *loaded = Some(Loaded {
            path: t.path.to_string(),
            gpu,
            n_ctx,
            model,
        });
    }
    Ok(&loaded.as_ref().expect("loaded above").model)
}

fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(2, 8)
}

/// Context settings. Off the GPU in a GPU build, the KV cache and big batches stay off it too.
fn context_params(n_ctx: u32, on_gpu: bool) -> LlamaContextParams {
    let p = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_threads(threads())
        .with_n_threads_batch(threads());
    if cfg!(feature = "vulkan") && !on_gpu {
        p.with_offload_kqv(false).with_op_offload(false)
    } else {
        p
    }
}

/// Longest text embedded, in tokens; longer ones are cut (passages are built well under this).
const EMBED_MAX_TOKENS: usize = 1024;

fn embed(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    e: &Embedding,
) -> Result<Vec<Vec<f32>>, String> {
    let model = load(llama, loaded, t)?;
    let ctx_params = context_params(EMBED_MAX_TOKENS as u32, t.gpu.is_some())
        .with_n_batch(EMBED_MAX_TOKENS as u32)
        .with_n_ubatch(EMBED_MAX_TOKENS as u32)
        .with_embeddings(true)
        // The model's own pooling (Qwen3-Embedding: last token) from its metadata.
        .with_pooling_type(LlamaPoolingType::Unspecified);
    let mut ctx = model
        .new_context(llama, ctx_params)
        .map_err(|e| format!("not enough memory for the search model: {e}"))?;
    let vocab = model.vocab();
    let mut out = Vec::with_capacity(e.texts.len());
    let mut batch = LlamaBatch::new(EMBED_MAX_TOKENS, 1);
    for text in &e.texts {
        // add_special: the tokenizer adds what the model expects (Qwen3-Embedding pools on the closing token).
        let mut tokens = vocab.tokenize(text.as_bytes(), true, false);
        tokens.truncate(EMBED_MAX_TOKENS);
        ctx.clear_kv_cache();
        batch.clear();
        for (i, t) in tokens.iter().enumerate() {
            batch
                .add(*t, i as i32, &[0], true)
                .map_err(|e| e.to_string())?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("the search model failed: {e}"))?;
        let v = ctx
            .embeddings_seq_ith(0)
            .map_err(|e| format!("the search model gave no vector: {e}"))?;
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        out.push(v.iter().map(|x| x / norm).collect());
    }
    Ok(out)
}

fn complete(
    llama: &LlamaBackend,
    loaded: &mut Option<Loaded>,
    t: Target,
    c: &Completion,
) -> Result<Reply, String> {
    let started = Instant::now();
    let model = load(llama, loaded, t)?;

    let template = model
        .chat_template(None)
        .map_err(|e| format!("the model has no chat template: {e}"))?;
    let messages = [
        LlamaChatMessage::new("system".into(), c.system.clone()).map_err(|e| e.to_string())?,
        LlamaChatMessage::new("user".into(), c.user.clone()).map_err(|e| e.to_string())?,
    ];
    let prompt = model
        .apply_chat_template(&template, &messages, true)
        .map_err(|e| format!("couldn't build the prompt: {e}"))?;
    let vocab = model.vocab();
    let tokens = vocab.tokenize(prompt.as_bytes(), false, true);
    let n_prompt = tokens.len() as u32;
    if n_prompt + c.max_tokens > c.n_ctx {
        return Err(format!(
            "the transcript part is too long for the model ({n_prompt} tokens, context {})",
            c.n_ctx
        ));
    }

    // The sampler first: a bad schema should fail before the model reads anything.
    let mut samplers = vec![];
    if let Some(schema) = &c.json_schema {
        let grammar = llama_cpp_2::json_schema_to_grammar(schema)
            .map_err(|e| format!("bad output schema: {e}"))?;
        samplers.push(
            LlamaSampler::grammar(model, &grammar, "root")
                .map_err(|e| format!("bad output schema: {e}"))?,
        );
    }
    if c.temperature <= 0.0 {
        samplers.push(LlamaSampler::greedy());
    } else {
        samplers.push(LlamaSampler::top_k(40));
        samplers.push(LlamaSampler::top_p(0.9, 1));
        samplers.push(LlamaSampler::temp(c.temperature));
        samplers.push(LlamaSampler::dist(1234));
    }
    let mut sampler = LlamaSampler::chain_simple(samplers);

    let ctx_params = context_params(c.n_ctx, t.gpu.is_some())
        .with_n_batch(BATCH as u32)
        .with_n_ubatch(BATCH as u32);
    let mut ctx = model
        .new_context(llama, ctx_params)
        .map_err(|e| format!("not enough memory for the AI model: {e}"))?;

    // Read the prompt in batches; only the last token needs logits.
    let mut batch = LlamaBatch::new(BATCH, 1);
    for (start, chunk) in tokens
        .chunks(BATCH)
        .enumerate()
        .map(|(i, ch)| (i * BATCH, ch))
    {
        batch.clear();
        for (j, t) in chunk.iter().enumerate() {
            let pos = start + j;
            batch
                .add(*t, pos as i32, &[0], pos + 1 == tokens.len())
                .map_err(|e| e.to_string())?;
        }
        ctx.decode(&mut batch)
            .map_err(|e| format!("the AI model failed reading: {e}"))?;
        send(&Event::Progress {
            phase: Phase::Reading,
            done: (start + chunk.len()) as u32,
            total: n_prompt,
        });
    }

    let mut out: Vec<u8> = vec![];
    let mut n_cur = tokens.len() as i32;
    let mut written = 0u32;
    let mut truncated = true;
    while written < c.max_tokens {
        // `sample` also accepts the token into the samplers (grammar state included); accepting it again
        // would push the grammar past it.
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        if vocab.is_eog(token) {
            truncated = false;
            break;
        }
        out.extend(vocab.token_to_piece(token, false, None));
        written += 1;
        if written.is_multiple_of(32) {
            send(&Event::Progress {
                phase: Phase::Writing,
                done: written,
                total: 0,
            });
        }
        batch.clear();
        batch
            .add(token, n_cur, &[0], true)
            .map_err(|e| e.to_string())?;
        n_cur += 1;
        ctx.decode(&mut batch)
            .map_err(|e| format!("the AI model failed writing: {e}"))?;
    }

    Ok(Reply {
        text: String::from_utf8_lossy(&out).into_owned(),
        prompt_tokens: n_prompt,
        output_tokens: written,
        seconds: started.elapsed().as_secs_f32(),
        truncated,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// llama.cpp's backend can exist once per process, so the tests that start it take turns.
    static BACKEND: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn a_cpu_build_reports_the_processor_and_lists_the_cpu() {
        let _turn = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
        let _llama = LlamaBackend::init().unwrap();
        let all = devices();
        assert!(all.iter().any(|d| d.kind == DeviceKind::Cpu), "{all:?}");
        if !cfg!(feature = "vulkan") {
            assert_eq!(choose(&all), None);
            if cfg!(not(target_os = "macos")) {
                assert_eq!(backend(None), Backend::cpu());
            }
        }
    }

    #[test]
    fn ensayo_threads_stay_between_two_and_eight() {
        assert!((2..=8).contains(&threads()));
    }

    #[test]
    fn ensayo_a_missing_model_file_is_a_plain_error_not_a_crash() {
        let _turn = BACKEND.lock().unwrap_or_else(|e| e.into_inner());
        let llama = LlamaBackend::init().unwrap();
        let mut loaded = None;
        let c = Completion {
            model_path: "/nonexistent/model.gguf".into(),
            n_ctx: 512,
            system: String::new(),
            user: String::new(),
            json_schema: None,
            max_tokens: 1,
            temperature: 0.0,
            gpu: false,
        };
        let target = Target {
            path: &c.model_path,
            gpu: None,
            n_ctx: c.n_ctx,
        };
        let e = complete(&llama, &mut loaded, target, &c).unwrap_err();
        assert!(e.contains("missing"), "{e}");
    }

    #[test]
    fn the_idle_exit_follows_the_app_prefix_and_defaults_to_two_minutes() {
        let config = HelperConfig {
            env_prefix: "VOICE_ENGINE_UNIT_TEST",
        };
        assert_eq!(idle_exit(&config), IDLE_EXIT);
        // SAFETY: this variable name is private to this test, and no other test reads it.
        unsafe { std::env::set_var(config.idle_secs_env(), "7") };
        assert_eq!(idle_exit(&config), Duration::from_secs(7));
        unsafe { std::env::set_var(config.idle_secs_env(), "not a number") };
        assert_eq!(idle_exit(&config), IDLE_EXIT);
        unsafe { std::env::remove_var(config.idle_secs_env()) };
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
        let b = backend(Some(&d));
        assert_eq!(
            (b.backend.as_str(), b.device.as_deref(), b.vram_mb),
            ("vulkan", Some("NVIDIA GeForce RTX 5070 Ti"), 16303)
        );
        assert!(b.on_gpu());
    }
}
