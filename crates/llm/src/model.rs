//! Loading a model: on which device, how many layers, with which context settings.

use crate::failure::Failure;
use crate::send;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use std::num::NonZeroU32;
use voice_engine_protocol::gpu::Device;
use voice_engine_protocol::{Event, Phase};

/// What a request needs loaded: a model file, on the chosen GPU or the processor, fitted for a context size.
#[derive(Clone, Copy)]
pub(crate) struct Target<'a> {
    pub(crate) path: &'a str,
    pub(crate) gpu: Option<&'a Device>,
    pub(crate) n_ctx: u32,
}

pub(crate) struct Loaded {
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
fn model_params(t: Target) -> Result<LlamaModelParams, Failure> {
    let Some(dev) = t.gpu else {
        let layers = if cfg!(feature = "vulkan") { 0 } else { 999 };
        return Ok(LlamaModelParams::default().with_n_gpu_layers(layers));
    };
    let mut params = LlamaModelParams::default()
        .with_split_mode(LlamaSplitMode::None)
        .with_main_gpu(0)
        .with_devices(&[dev.index])
        .map_err(|e| Failure::internal(format!("couldn't use the graphics chip: {e}")))?;
    let path = std::ffi::CString::new(t.path).map_err(|_| Failure::internal("bad model path"))?;
    let mut cparams = context_params(t.n_ctx, true);
    let mut margins = vec![GPU_MARGIN; llama_cpp_2::max_devices().max(1)];
    // Log level 4 is GGML_LOG_LEVEL_ERROR: fitting reports only problems.
    std::pin::Pin::new(&mut params)
        .fit_params(&path, &mut cparams, &mut margins, t.n_ctx, 4)
        .map_err(|e| Failure::out_of_memory(format!("the AI model doesn't fit in memory: {e}")))?;
    Ok(params)
}

pub(crate) fn load<'a>(
    llama: &LlamaBackend,
    loaded: &'a mut Option<Loaded>,
    t: Target,
) -> Result<&'a LlamaModel, Failure> {
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
            return Err(Failure::missing(t.path));
        }
        let params = model_params(t)?;
        let model = LlamaModel::load_from_file(llama, t.path, &params)
            .map_err(|e| Failure::internal(format!("couldn't load the AI model: {e}")))?;
        *loaded = Some(Loaded {
            path: t.path.to_string(),
            gpu,
            n_ctx,
            model,
        });
    }
    Ok(&loaded.as_ref().expect("loaded above").model)
}

pub(crate) fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(2, 8)
}

/// Context settings. Off the GPU in a GPU build, the KV cache and big batches stay off it too.
pub(crate) fn context_params(n_ctx: u32, on_gpu: bool) -> LlamaContextParams {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensayo_threads_stay_between_two_and_eight() {
        assert!((2..=8).contains(&threads()));
    }
}
