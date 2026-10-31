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

/// Where one request's model runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Place<'a> {
    /// No layer on any graphics device, in every build (Metal and Vulkan alike).
    Processor,
    /// The Mac: every layer on Metal, which llama.cpp picks by itself.
    Metal,
    /// The one device this helper chose (Vulkan), as many layers as fit.
    Gpu(&'a Device),
}

impl Place<'_> {
    /// The model is (at least partly) on a graphics card.
    pub(crate) fn on_card(self) -> bool {
        !matches!(self, Place::Processor)
    }

    fn placed(self) -> Placed {
        match self {
            Place::Processor => Placed::Processor,
            Place::Metal => Placed::Metal,
            Place::Gpu(d) => Placed::Gpu(d.index),
        }
    }
}

/// Where a request's model runs, from what the request asked (`gpu`), what this build can do (`vulkan_build`, `macos`:
/// the `vulkan` cargo feature and the Mac target) and the device the helper chose (`chosen`; always None on the Mac,
/// where llama.cpp chooses Metal itself). `gpu: false` is the processor in every build.
pub(crate) fn place(
    requested: bool,
    vulkan_build: bool,
    macos: bool,
    chosen: Option<&Device>,
) -> Place<'_> {
    if !requested {
        return Place::Processor;
    }
    match chosen {
        Some(d) => Place::Gpu(d),
        None if macos && !vulkan_build => Place::Metal,
        None => Place::Processor,
    }
}

/// `place` for this build.
pub(crate) fn place_here(requested: bool, chosen: Option<&Device>) -> Place<'_> {
    place(
        requested,
        cfg!(feature = "vulkan"),
        cfg!(target_os = "macos"),
        chosen,
    )
}

/// The device part of what a loaded model is keyed by (a `Place` without the borrowed device).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Placed {
    Processor,
    Metal,
    Gpu(usize),
}

/// What a request needs loaded: a model file, on the processor, Metal or the chosen GPU, fitted for a context size.
#[derive(Clone, Copy)]
pub(crate) struct Target<'a> {
    pub(crate) path: &'a str,
    pub(crate) place: Place<'a>,
    pub(crate) n_ctx: u32,
}

/// What decides whether a loaded model can answer a request: the file, where it runs and, on a GPU only, the context
/// size it was fitted for. Where it runs includes the request's `gpu` flag (a request for the processor never
/// reuses a model that is on the card, nor the other way round).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Key {
    path: String,
    placed: Placed,
    n_ctx: u32,
}

impl Key {
    pub(crate) fn of(t: Target) -> Key {
        // The context size only changes how a model is fitted on a GPU (`fit_params`); on the processor or Metal a
        // different size must not reload the model, so it is left out of the key there.
        let n_ctx = if matches!(t.place, Place::Gpu(_)) {
            t.n_ctx
        } else {
            0
        };
        Key {
            path: t.path.to_string(),
            placed: t.place.placed(),
            n_ctx,
        }
    }
}

pub(crate) struct Loaded {
    key: Key,
    model: LlamaModel,
}

impl Loaded {
    /// The `model_path` the model was loaded from, as the request wrote it.
    pub(crate) fn path(&self) -> &str {
        &self.key.path
    }
}

/// GPU memory left free for the driver, the desktop and other apps (llama.cpp's own default margin).
const GPU_MARGIN: usize = 1024 * 1024 * 1024;

/// Model settings for a target. On the GPU: only that device, no split, and as many layers as fit next to the
/// context (fit_params lowers the layer count when the GPU is too small, so a 9B model still uses part of a 6 GB
/// card). On the processor: no layers on any graphics device, in a Metal build and a Vulkan build alike. On Metal:
/// all of them.
fn model_params(t: Target) -> Result<LlamaModelParams, Failure> {
    let dev = match t.place {
        Place::Processor => return Ok(LlamaModelParams::default().with_n_gpu_layers(0)),
        Place::Metal => return Ok(LlamaModelParams::default().with_n_gpu_layers(999)),
        Place::Gpu(dev) => dev,
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

/// The model for a target, loading it unless it is already loaded. The seconds are 0 when it was.
pub(crate) fn load<'a>(
    llama: &LlamaBackend,
    loaded: &'a mut Option<Loaded>,
    t: Target,
) -> Result<(&'a LlamaModel, f32), Failure> {
    let started = std::time::Instant::now();
    let key = Key::of(t);
    let same = loaded.as_ref().is_some_and(|l| l.key == key);
    if !same {
        *loaded = None;
        if !std::path::Path::new(t.path).is_file() {
            return Err(Failure::missing(t.path));
        }
        send(&Event::Progress {
            phase: Phase::Loading,
            done: 0,
            total: 0,
        });
        let params = model_params(t)?;
        let model = LlamaModel::load_from_file(llama, t.path, &params)
            .map_err(|e| Failure::internal(format!("couldn't load the AI model: {e}")))?;
        *loaded = Some(Loaded { key, model });
    }
    let seconds = if same {
        0.0
    } else {
        started.elapsed().as_secs_f32()
    };
    Ok((&loaded.as_ref().expect("loaded above").model, seconds))
}

pub(crate) fn threads() -> i32 {
    std::thread::available_parallelism()
        .map(|n| n.get() as i32)
        .unwrap_or(4)
        .clamp(2, 8)
}

/// Context settings. Off the card in a build that has one (Metal, Vulkan), the KV cache and big batches stay off it
/// too, so the processor really does all the work.
pub(crate) fn context_params(n_ctx: u32, on_card: bool) -> LlamaContextParams {
    let p = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(n_ctx))
        .with_n_threads(threads())
        .with_n_threads_batch(threads());
    if keeps_off_card(on_card, cfg!(feature = "vulkan"), cfg!(target_os = "macos")) {
        p.with_offload_kqv(false).with_op_offload(false)
    } else {
        p
    }
}

/// A processor request in a build with a graphics backend (Metal on the Mac, Vulkan): nothing may be offloaded.
fn keeps_off_card(on_card: bool, vulkan_build: bool, macos: bool) -> bool {
    (vulkan_build || macos) && !on_card
}

#[cfg(test)]
mod tests {
    use super::*;
    use voice_engine_protocol::gpu::DeviceKind;

    #[test]
    fn ensayo_threads_stay_between_two_and_eight() {
        assert!((2..=8).contains(&threads()));
    }

    fn card(index: usize) -> Device {
        Device {
            index,
            name: format!("Vulkan{index}"),
            description: "Test card".into(),
            backend: "Vulkan".into(),
            kind: DeviceKind::Gpu,
            memory_mb: 8192,
        }
    }

    /// Every row of the decision: (asked for the card, Vulkan build, Mac, a device was chosen) -> where it runs.
    #[test]
    fn where_a_request_runs_depends_on_its_flag_the_build_and_the_chosen_device() {
        let d = card(1);
        // (gpu, vulkan build, macOS, chosen device, expected)
        let rows: [(bool, bool, bool, Option<&Device>, Place); 14] = [
            // The Mac (Metal build): the helper chooses no device; the flag decides.
            (true, false, true, None, Place::Metal),
            (false, false, true, None, Place::Processor),
            // A Vulkan build with a card: the flag decides between the card and the processor.
            (true, true, false, Some(&d), Place::Gpu(&d)),
            (false, true, false, Some(&d), Place::Processor),
            // A Vulkan build without a usable card: the processor either way.
            (true, true, false, None, Place::Processor),
            (false, true, false, None, Place::Processor),
            // The processor-only Windows build and Linux: no card, the processor either way.
            (true, false, false, None, Place::Processor),
            (false, false, false, None, Place::Processor),
            // A device the build can use on a system that is not the Mac, without the Vulkan feature.
            (true, false, false, Some(&d), Place::Gpu(&d)),
            (false, false, false, Some(&d), Place::Processor),
            // Vulkan on a Mac (not shipped): never Metal, since that build has no Metal.
            (true, true, true, None, Place::Processor),
            (false, true, true, None, Place::Processor),
            // A device on the Mac is never chosen by the helper, but if one is, the flag still decides.
            (true, false, true, Some(&d), Place::Gpu(&d)),
            (false, false, true, Some(&d), Place::Processor),
        ];
        for (gpu, vulkan, macos, chosen, expected) in rows {
            assert_eq!(
                place(gpu, vulkan, macos, chosen),
                expected,
                "gpu={gpu} vulkan={vulkan} macos={macos} chosen={chosen:?}"
            );
        }
    }

    #[test]
    fn gpu_false_is_never_on_a_card() {
        let d = card(0);
        for vulkan in [false, true] {
            for macos in [false, true] {
                for chosen in [None, Some(&d)] {
                    let p = place(false, vulkan, macos, chosen);
                    assert_eq!(p, Place::Processor);
                    assert!(!p.on_card());
                }
            }
        }
        assert!(Place::Metal.on_card() && Place::Gpu(&d).on_card());
    }

    fn target<'a>(path: &'a str, place: Place<'a>, n_ctx: u32) -> Target<'a> {
        Target { path, place, n_ctx }
    }

    /// The loaded-model key: a model loaded for the card is not reused for the processor, and the other way round.
    #[test]
    fn a_loaded_model_is_reused_only_for_the_same_file_and_place() {
        let d = card(1);
        let other = card(2);
        let k = |t| Key::of(t);
        // The Mac: Metal and the processor are different places for the same file.
        assert_ne!(
            k(target("/m.gguf", Place::Metal, 4096)),
            k(target("/m.gguf", Place::Processor, 4096))
        );
        // A card and the processor, and one card and another.
        assert_ne!(
            k(target("/m.gguf", Place::Gpu(&d), 4096)),
            k(target("/m.gguf", Place::Processor, 4096))
        );
        assert_ne!(
            k(target("/m.gguf", Place::Gpu(&d), 4096)),
            k(target("/m.gguf", Place::Gpu(&other), 4096))
        );
        // Another file is another model.
        assert_ne!(
            k(target("/m.gguf", Place::Metal, 4096)),
            k(target("/n.gguf", Place::Metal, 4096))
        );
        // The same file in the same place is reused, whatever the context size, except on a card, where the
        // size decides how many layers fit.
        for place in [Place::Metal, Place::Processor] {
            assert_eq!(
                k(target("/m.gguf", place, 4096)),
                k(target("/m.gguf", place, 8192)),
                "{place:?}"
            );
        }
        assert_ne!(
            k(target("/m.gguf", Place::Gpu(&d), 4096)),
            k(target("/m.gguf", Place::Gpu(&d), 8192))
        );
    }

    /// The same request flags, run through this build's rules: the flag flips the key on the Mac, and on every
    /// system when a card was chosen.
    #[test]
    fn the_requests_flag_changes_the_key_wherever_it_changes_the_place() {
        let d = card(1);
        let key = |gpu: bool, chosen| Key::of(target("/m.gguf", place_here(gpu, chosen), 4096));
        assert_ne!(key(true, Some(&d)), key(false, Some(&d)));
        if cfg!(target_os = "macos") && !cfg!(feature = "vulkan") {
            assert_ne!(key(true, None), key(false, None));
        } else {
            // No card to leave: both requests are on the processor, so the model is not loaded twice.
            assert_eq!(key(true, None), key(false, None));
        }
    }

    #[test]
    fn the_processor_gets_no_gpu_layers_and_metal_gets_all_of_them() {
        let layers = |place| {
            model_params(target("/m.gguf", place, 4096))
                .unwrap()
                .n_gpu_layers()
        };
        assert_eq!(layers(Place::Processor), 0);
        assert_eq!(layers(Place::Metal), 999);
    }

    #[test]
    fn a_processor_request_keeps_the_kv_cache_and_big_batches_off_the_card() {
        // (on the card, Vulkan build, Mac) -> kept off the card
        for (on_card, vulkan, macos, off) in [
            (false, false, true, true),
            (false, true, false, true),
            (false, true, true, true),
            (true, false, true, false),
            (true, true, false, false),
            // A build with no card backend at all has nothing to keep off.
            (false, false, false, false),
            (true, false, false, false),
        ] {
            assert_eq!(
                keeps_off_card(on_card, vulkan, macos),
                off,
                "on_card={on_card} vulkan={vulkan} macos={macos}"
            );
        }
        // And this build's context really is set that way.
        let here = cfg!(feature = "vulkan") || cfg!(target_os = "macos");
        let p = context_params(4096, false);
        assert_eq!((p.offload_kqv(), p.op_offload()), (!here, !here));
        let p = context_params(4096, true);
        assert!(p.offload_kqv() && p.op_offload());
    }
}
