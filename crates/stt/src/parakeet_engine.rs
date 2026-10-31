//! Parakeet (TDT v2, v3, Ultra, full size and compressed), Parakeet Unified and Nemotron through parakeet-rs and
//! ONNX Runtime, behind the [`Transcriber`] trait. Ported from Talaan's `transcriber/engine.rs` and Ensayo's
//! `speech/engine.rs` (MIT, Miguel Monzones).
//!
//! * One call takes a chunk of up to the catalog's `chunk_secs` (40 s); longer audio is [`SpeechError::TooLong`]
//!   and nothing is decoded (the app cuts the audio). Any shorter chunk, a 15 s window or a clip under a second,
//!   comes back with timed words.
//! * Every decode is given half a second of silence on each side (Parakeet often returns nothing for audio that
//!   starts or ends right on speech); the times are put back on the clip and clamped to it.
//! * Parakeet sometimes stops decoding partway through a chunk. When a pass ends with 2 s or more of audio after
//!   its last word, the rest is decoded again from there, up to 8 passes in all; `stopped_early_at_ms` says where
//!   the first stop was. The loop is Whisper's ([`decode_with_resume`]), the same arithmetic Talaan and Ensayo run.
//! * Both timing modes run the model once, in parakeet-rs's word timestamp mode, and return segments and words.
//!   `Timing::Words` is Ensayo's: the words through [`crate::words::clean`], one per segment (as Whisper's Words
//!   mode). `Timing::Segments` is Talaan's: the segments are the sentences parakeet-rs's own sentence mode gives
//!   (it builds them from the same words; [`segments_from_words`] is that step, kept here because parakeet-rs
//!   keeps it private and a second decode would double the time), each with its words attached.
//! * Nemotron takes a language prompt: a hint "fr" becomes the locale "fr-FR" ([`nemotron_locale`]); no hint or an
//!   unknown one means the model detects. Parakeet and Parakeet Unified take no language.
//! * `keep_fillers` and `vocabulary` do nothing here: the ONNX models transcribe fillers as heard and have no
//!   prompt (the catalog's `keeps_fillers` is measured separately).
//! * The graphics card: with the `directml` feature on Windows, `gpu: true` runs the model through ONNX Runtime's
//!   DirectML provider when the catalog says the model has that path (Parakeet TDT), on the GPU thread rule, and
//!   falls back to the processor (`fallback: Some(CardFailed)`, or `CardOutOfMemory` when the provider ran out of
//!   memory; the processor's rule) when the provider does not start. It runs
//!   on the PC's discrete card (the one with the most memory, [`crate::directml`]) and `device` names it; with no
//!   discrete card (only integrated graphics) DirectML is not tried and the model runs on the processor.
//!   Without the feature, on the Mac, or for a model without a DirectML path, `gpu: true` runs on the processor
//!   with `fallback: Some(NoCardPath)`: there is no card path to fall back from, but the app asked for one, so it is
//!   told.
//!   Nothing here uses CoreML (parakeet-rs measures it slower than the processor for these graphs).
//!
//! The pure parts are public so `tests/parakeet_api.rs` can test them with a scripted [`WordModel`].

use crate::api::{
    ChunkOptions, LoadOptions, Segment, SpeechBackend, SpeechError, Timing, Transcriber,
    Transcript, Word,
};
use crate::directml::{self, Adapter};
use crate::threads::{ThreadKind, logical_cores, resolve_threads};
use crate::whisper_engine::{Pass, PassDecoder, decode_with_resume};
use crate::words::clean;
use std::path::Path;
use voice_engine_models::catalog::{Kind, ModelInfo};
use voice_engine_protocol::Fallback;

/// Half a second of silence on each side of every decode (16 kHz). Talaan's and Ensayo's value, kept on purpose:
/// on a Mac (Parakeet Ultra, the processor, 15 s windows cut at six offsets from `docs/data/clip-bench.wav`) a pad of
/// 0.25 s, 0.1 s or none changed the words or their punctuation and capitals in 3 or 4 of the 6 windows (and in the
/// 29 s clip and the end of the short one with none), while the decode time did not fall by more than the noise of
/// the run (about 3% is all the audio saved). A window cut in the middle of speech is the case that needs it.
pub const PAD_SAMPLES: usize = 8_000;
/// The chunk limit when the catalog entry gives none (every ONNX entry gives 40).
pub const DEFAULT_MAX_SECS: u32 = 40;
/// This build can run Parakeet on DirectML.
pub const DIRECTML_BUILT: bool = cfg!(all(windows, feature = "directml"));

/// A word as parakeet-rs gives it: seconds on the audio it was handed (the padded audio).
#[derive(Debug, Clone, PartialEq)]
pub struct RawWord {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

/// One loaded ONNX model (the real ones wrap parakeet-rs; the tests script it).
pub trait WordModel {
    /// The words in `audio` (already padded), in order.
    fn words(&mut self, audio: Vec<f32>) -> Result<Vec<RawWord>, SpeechError>;
    /// Gives the model the app's language hint (`None`: detect). Returns the language it will decode in, when the
    /// model has one to say.
    fn set_language(&mut self, hint: Option<&str>) -> Option<String>;
}

/// `samples` with [`PAD_SAMPLES`] of silence before and after; at least one second in all.
pub fn pad_both_sides(samples: &[f32]) -> Vec<f32> {
    let mut audio = vec![0.0f32; PAD_SAMPLES];
    audio.extend_from_slice(samples);
    audio.extend(std::iter::repeat_n(0.0, PAD_SAMPLES));
    audio
}

/// parakeet-rs's words to ours: the padding taken off, times in milliseconds on the clip (`audio_ms` long, not the
/// padded length), never before 0 or after the clip's end, an end never before its start.
pub fn shift_words(raw: Vec<RawWord>, audio_ms: u64) -> Vec<Word> {
    // As Talaan and Ensayo do: seconds to whole milliseconds (truncated), the padding off, clamped.
    let shift = |t: f32| {
        ((t.max(0.0) * 1000.0) as u64)
            .saturating_sub((PAD_SAMPLES as u64) * 1000 / 16_000)
            .min(audio_ms)
    };
    raw.into_iter()
        .map(|t| Word {
            start_ms: shift(t.start),
            end_ms: shift(t.end.max(t.start)),
            text: t.text,
        })
        .collect()
}

/// How parakeet-rs writes a sentence: the words with single spaces, no space before standalone punctuation
/// (contractions like "'re" keep theirs).
pub fn sentence_text(words: &[Word]) -> String {
    let mut out = String::new();
    for (i, word) in words.iter().map(|w| w.text.as_str()).enumerate() {
        let is_standalone_punct = word.len() == 1
            && word
                .chars()
                .all(|c| matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')'));
        if i > 0 && !is_standalone_punct {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

/// parakeet-rs ends a sentence on any word that contains a full stop, a question mark or an exclamation mark.
fn ends_sentence(word: &Word) -> bool {
    word.text.contains(['.', '?', '!'])
}

/// The words of one pass, in the shape of the timing mode (every segment has its words in both).
pub fn segments_from_words(words: Vec<Word>, timing: Timing) -> Vec<Segment> {
    match timing {
        // Ensayo: noise dropped, punctuation glued to the word before, times in order; one word per segment.
        Timing::Words => clean(words)
            .into_iter()
            .map(|w| Segment {
                start_ms: w.start_ms,
                end_ms: w.end_ms,
                text: w.text.clone(),
                // Parakeet's words are spaced.
                starts_word: true,
                words: vec![w],
            })
            .collect(),
        // Talaan: the sentences (noise dropped), each with its words.
        Timing::Segments => {
            let mut sentences: Vec<Vec<Word>> = vec![];
            let mut current: Vec<Word> = vec![];
            for word in words {
                let ends = ends_sentence(&word);
                current.push(word);
                if ends {
                    sentences.push(std::mem::take(&mut current));
                }
            }
            if !current.is_empty() {
                sentences.push(current);
            }
            let mut out: Vec<Segment> = vec![];
            let mut last_end = 0u64;
            for sentence in sentences {
                let text = sentence_text(&sentence);
                let Some(last) = sentence.last() else {
                    continue;
                };
                if crate::is_noise(&text) {
                    continue;
                }
                let sentence_end = last.end_ms;
                let mut words = clean(sentence);
                if words.is_empty() {
                    continue;
                }
                // `clean` keeps times in order inside a sentence; keep them in order across sentences too.
                for w in &mut words {
                    if w.start_ms < last_end {
                        w.start_ms = last_end;
                        w.end_ms = w.end_ms.max(w.start_ms);
                    }
                    last_end = w.end_ms;
                }
                out.push(Segment {
                    start_ms: words[0].start_ms,
                    end_ms: words.last().map_or(0, |w| w.end_ms).max(sentence_end),
                    text: text.trim().to_string(),
                    // Sentences are spaced.
                    starts_word: true,
                    words,
                });
            }
            out
        }
    }
}

/// Nemotron's prompt for the app's language hint: the model's own locale key, so "fr" becomes "fr-FR". A hint that
/// already is one of the model's keys (any case, `-` or `_`) is kept in the model's spelling; a bare code takes the
/// model's `xx-XX` locale (the first `xx-` one if there is none), else the bare key. `None` (the model detects) for no
/// hint, "auto", a language the model lacks, and the English-only variant (`available` is empty).
pub fn nemotron_locale(hint: Option<&str>, available: &[&str]) -> Option<String> {
    let hint = hint?.trim().replace('_', "-");
    if hint.is_empty() || hint.eq_ignore_ascii_case("auto") || available.is_empty() {
        return None;
    }
    let find = |wanted: &str| {
        available
            .iter()
            .find(|k| k.eq_ignore_ascii_case(wanted))
            .map(|k| k.to_string())
    };
    if hint.contains('-') {
        // a full locale: exact (any case), else its language
        find(&hint).or_else(|| {
            let code = hint.split('-').next().unwrap_or("");
            bare_code(code, available)
        })
    } else {
        bare_code(&hint, available)
    }
}

fn bare_code(code: &str, available: &[&str]) -> Option<String> {
    if code.is_empty() || code.eq_ignore_ascii_case("auto") {
        return None;
    }
    let lower = code.to_ascii_lowercase();
    let locale = format!("{lower}-{}", code.to_ascii_uppercase());
    available
        .iter()
        .find(|k| **k == locale)
        .or_else(|| available.iter().find(|k| **k == lower))
        .or_else(|| {
            let prefix = format!("{lower}-");
            available.iter().find(|k| k.starts_with(&prefix))
        })
        .map(|k| k.to_string())
}

/// The names parakeet-rs 0.3.8 looks for in a model folder, in its order of preference: (encoders, decoders).
/// Nemotron has fixed names, so no choice to make.
fn parakeet_rs_candidates(kind: Kind) -> (&'static [&'static str], &'static [&'static str]) {
    match kind {
        Kind::Parakeet => (
            &[
                "encoder-model.onnx",
                "encoder.onnx",
                "encoder-model.int8.onnx",
            ],
            &[
                "decoder_joint-model.onnx",
                "decoder_joint-model.int8.onnx",
                "decoder_joint.onnx",
                "decoder-model.onnx",
            ],
        ),
        Kind::ParakeetUnified => (
            &["encoder.onnx", "encoder.int8.onnx", "encoder-model.onnx"],
            &[
                "decoder_joint.onnx",
                "decoder_joint.int8.onnx",
                "decoder_joint-model.onnx",
            ],
        ),
        _ => (&[], &[]),
    }
}

/// The folder has every file the catalog lists for this model (`has` says whether a name is there), and
/// parakeet-rs, which picks its own files by name in the folder, will pick the catalog's encoder and decoder: a
/// folder that also holds another variant's file that parakeet-rs prefers (the full-size encoder next to the
/// compressed entry) would run the wrong model.
pub fn check_files(model: &ModelInfo, has: &dyn Fn(&str) -> bool) -> Result<(), SpeechError> {
    if model.files.iter().any(|f| !has(f.name)) {
        return Err(SpeechError::ModelMissing);
    }
    let (encoders, decoders) = parakeet_rs_candidates(model.kind);
    for candidates in [encoders, decoders] {
        let named = model
            .files
            .iter()
            .map(|f| f.name)
            .find(|n| candidates.contains(n));
        let picked = candidates.iter().copied().find(|n| has(n));
        if named != picked {
            return Err(SpeechError::BadModel);
        }
    }
    Ok(())
}

/// One decode of a stretch of audio with an ONNX model: the Parakeet resume loop's pass.
pub struct ParakeetPass<'a> {
    pub model: &'a mut dyn WordModel,
    pub timing: Timing,
}

impl PassDecoder for ParakeetPass<'_> {
    fn decode(&mut self, audio: &[f32]) -> Result<Pass, SpeechError> {
        let audio_ms = audio.len() as u64 * 1000 / 16_000;
        let raw = self.model.words(pad_both_sides(audio))?;
        Ok(Pass {
            segments: segments_from_words(shift_words(raw, audio_ms), self.timing),
            language: None,
        })
    }
}

/// The whole call: refuses audio over `max_secs`, gives the model the language hint, decodes with the resume loop.
pub fn transcribe_chunk(
    model: &mut dyn WordModel,
    audio: &[f32],
    opts: &ChunkOptions,
    max_secs: u32,
) -> Result<Transcript, SpeechError> {
    if audio.len() > max_secs as usize * 16_000 {
        return Err(SpeechError::TooLong { max_secs });
    }
    if audio.is_empty() {
        return Ok(Transcript {
            segments: vec![],
            language: None,
            stopped_early_at_ms: None,
        });
    }
    let language = model.set_language(opts.language.as_deref());
    let mut pass = ParakeetPass {
        model,
        timing: opts.timing,
    };
    let mut transcript = decode_with_resume(&mut pass, audio)?;
    transcript.language = language;
    Ok(transcript)
}

/// Where an ONNX session runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Processor,
    /// DirectML on the adapter with this DXGI number.
    DirectMl(u32),
}

/// What a load gives: the value, where it runs, and the thread count it was built with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loaded<T> {
    pub value: T,
    pub backend: SpeechBackend,
    pub threads: usize,
}

/// Builds the model on the best provider for this load. `build(provider, threads)` makes it. With `gpu`, a discrete
/// card to run DirectML on (`card`: None when the build has no DirectML or the PC has no discrete card) and a DirectML
/// path in the catalog (`model_has_directml`) DirectML is tried first with the GPU thread rule, and the backend names
/// the card; if it does not start the processor builds it with its own rule and `fallback` is `CardFailed` (or
/// `CardOutOfMemory`). In every other case the processor builds it, and `fallback` is `NoCardPath` when the app asked
/// for the card and `None` when it did not. `requested_threads` wins on every provider.
pub fn load_on_best_provider<T>(
    gpu: bool,
    card: Option<&Adapter>,
    model_has_directml: bool,
    requested_threads: Option<usize>,
    logical: usize,
    mut build: impl FnMut(Provider, usize) -> Result<T, SpeechError>,
) -> Result<Loaded<T>, SpeechError> {
    let mut fallback = gpu.then_some(Fallback::NoCardPath);
    if gpu
        && model_has_directml
        && let Some(card) = card
    {
        let threads = resolve_threads(ThreadKind::Gpu, requested_threads, logical);
        match build(Provider::DirectMl(card.index), threads) {
            Ok(value) => {
                return Ok(Loaded {
                    value,
                    backend: SpeechBackend {
                        backend: "directml".into(),
                        device: Some(card.name.clone()),
                        fallback: None,
                    },
                    threads,
                });
            }
            Err(SpeechError::OutOfMemory) => fallback = Some(Fallback::CardOutOfMemory),
            Err(_) => fallback = Some(Fallback::CardFailed),
        }
    }
    let threads = resolve_threads(ThreadKind::Processor, requested_threads, logical);
    let value = build(Provider::Processor, threads)?;
    Ok(Loaded {
        value,
        backend: SpeechBackend {
            backend: "cpu".into(),
            device: None,
            fallback,
        },
        threads,
    })
}

/// A parakeet-rs error as a kind, never its text (it can carry paths). `loading`: while the model loads, an ONNX
/// Runtime refusal of a file is a bad model; while decoding it is an internal error.
pub(crate) fn from_parakeet(e: &parakeet_rs::Error, loading: bool) -> SpeechError {
    use ort::error::ErrorCode;
    match e {
        parakeet_rs::Error::Io(io) if io.kind() == std::io::ErrorKind::NotFound => {
            SpeechError::ModelMissing
        }
        parakeet_rs::Error::Io(_) => SpeechError::Internal("io error".into()),
        parakeet_rs::Error::Ort(o) => match o.code() {
            ErrorCode::NoSuchFile => SpeechError::ModelMissing,
            ErrorCode::NoModel | ErrorCode::InvalidProtobuf | ErrorCode::InvalidGraph
                if loading =>
            {
                SpeechError::BadModel
            }
            ErrorCode::ExecutionProviderFailure => {
                SpeechError::Internal("onnx runtime: execution provider".into())
            }
            ErrorCode::InvalidArgument => {
                SpeechError::Internal("onnx runtime: invalid argument".into())
            }
            _ => SpeechError::Internal("onnx runtime".into()),
        },
        parakeet_rs::Error::Model(_)
        | parakeet_rs::Error::Tokenizer(_)
        | parakeet_rs::Error::Config(_)
            if loading =>
        {
            SpeechError::BadModel
        }
        parakeet_rs::Error::Model(_) => SpeechError::Internal("model error".into()),
        parakeet_rs::Error::Tokenizer(_) => SpeechError::Internal("tokenizer error".into()),
        parakeet_rs::Error::Config(_) => SpeechError::Internal("config error".into()),
        parakeet_rs::Error::Audio(_) => SpeechError::Internal("audio error".into()),
    }
}

/// parakeet-rs's session settings for one provider and thread count.
fn execution_config(provider: Provider, threads: usize) -> parakeet_rs::ExecutionConfig {
    let config = parakeet_rs::ExecutionConfig::new().with_intra_threads(threads);
    match provider {
        Provider::Processor => config,
        Provider::DirectMl(id) => with_provider(config, directml_provider(id)),
    }
}

/// Adds an execution provider that must start or the session fails to build (parakeet-rs's own DirectML option
/// would log the failure and carry on with the processor, and `backend()` would then claim a card that is not
/// running). ONNX Runtime's DirectML notes ask for no memory pattern; the execution is sequential already
/// (parakeet-rs keeps one inter-op thread).
fn with_provider(
    config: parakeet_rs::ExecutionConfig,
    provider: ort::ep::ExecutionProviderDispatch,
) -> parakeet_rs::ExecutionConfig {
    config.with_custom_configure(move |builder| {
        Ok(builder
            .with_memory_pattern(false)?
            .with_execution_providers([provider.clone()])?)
    })
}

/// The DirectML provider on the adapter with DXGI number `device_id`, which must start. Only a Windows build with the
/// `directml` feature has one; every other build has no card to run on (`directml::card` is None and
/// `load_on_best_provider` stays on the processor).
#[cfg(all(windows, feature = "directml"))]
fn directml_provider(device_id: u32) -> ort::ep::ExecutionProviderDispatch {
    ort::ep::DirectML::default()
        .with_device_id(device_id as i32)
        .build()
        .error_on_failure()
}

#[cfg(not(all(windows, feature = "directml")))]
fn directml_provider(_device_id: u32) -> ort::ep::ExecutionProviderDispatch {
    // Not reachable: without DirectML in the build the processor is the only provider tried. The processor
    // provider, required to start, keeps this a valid (and compile-checked) configuration.
    ort::ep::CPU::default().build().error_on_failure()
}

/// The NVIDIA models: Parakeet TDT (v2, v3, Ultra), Parakeet Unified and Nemotron. Same padding, same timestamps;
/// they differ only in how they load and decode.
enum NvidiaModel {
    Tdt(parakeet_rs::ParakeetTDT),
    Unified(parakeet_rs::ParakeetUnified),
    /// With the language prompts the model accepts ("en-US", "fr-FR"...); empty for the English-only variant.
    Nemotron(Box<parakeet_rs::Nemotron>, Vec<&'static str>),
}

impl NvidiaModel {
    fn load(
        kind: Kind,
        dir: &Path,
        config: parakeet_rs::ExecutionConfig,
    ) -> Result<Self, SpeechError> {
        let failed = |e: parakeet_rs::Error| from_parakeet(&e, true);
        Ok(match kind {
            Kind::ParakeetUnified => NvidiaModel::Unified(
                parakeet_rs::ParakeetUnified::from_pretrained(dir, Some(config)).map_err(failed)?,
            ),
            Kind::Nemotron => {
                let handle = parakeet_rs::NemotronHandle::from_pretrained(dir, Some(config))
                    .map_err(failed)?;
                NvidiaModel::Nemotron(
                    Box::new(parakeet_rs::Nemotron::from_shared(&handle)),
                    handle.available_languages(),
                )
            }
            _ => NvidiaModel::Tdt(
                parakeet_rs::ParakeetTDT::from_pretrained(dir, Some(config)).map_err(failed)?,
            ),
        })
    }
}

impl WordModel for NvidiaModel {
    fn words(&mut self, audio: Vec<f32>) -> Result<Vec<RawWord>, SpeechError> {
        use parakeet_rs::Transcriber as _;
        let mode = Some(parakeet_rs::TimestampMode::Words);
        let result = match self {
            NvidiaModel::Tdt(m) => m.transcribe_samples(audio, 16_000, 1, mode),
            NvidiaModel::Unified(m) => m.transcribe_samples(audio, 16_000, 1, mode),
            NvidiaModel::Nemotron(m, _) => m.transcribe_audio_with_timestamps(&audio, mode),
        }
        .map_err(|e| from_parakeet(&e, false))?;
        Ok(result
            .tokens
            .into_iter()
            .map(|t| RawWord {
                text: t.text,
                start: t.start,
                end: t.end,
            })
            .collect())
    }

    fn set_language(&mut self, hint: Option<&str>) -> Option<String> {
        let NvidiaModel::Nemotron(model, languages) = self else {
            return None;
        };
        // The English-only variant takes no prompt.
        if languages.is_empty() {
            return None;
        }
        // The model keeps the language of the last call, so "no hint" has to say "auto" again.
        let locale = nemotron_locale(hint, languages);
        let _ = model.set_target_lang(locale.as_deref().unwrap_or("auto"));
        locale.map(|l| l.split('-').next().unwrap_or("").to_string())
    }
}

/// An ONNX speech model loaded in this process.
pub struct ParakeetEngine {
    model: NvidiaModel,
    threads: usize,
    max_secs: u32,
    backend: SpeechBackend,
}

impl ParakeetEngine {
    /// Loads the model from its folder `dir` (the catalog's files in it). `opts.gpu` tries DirectML where this build
    /// and the model have it (see the module docs); `opts.threads` or the thread rule sets ONNX Runtime's intra-op
    /// threads for where the model actually runs.
    pub fn load(model: &ModelInfo, dir: &Path, opts: LoadOptions) -> Result<Self, SpeechError> {
        check_files(model, &|name| dir.join(name).is_file())?;
        let kind = model.kind;
        // The adapters are listed only when the card is wanted.
        let card = if opts.gpu { directml::card() } else { None };
        let loaded = load_on_best_provider(
            opts.gpu,
            card.as_ref(),
            model.gpu.windows_directml,
            opts.threads,
            logical_cores(),
            |provider, threads| NvidiaModel::load(kind, dir, execution_config(provider, threads)),
        )?;
        Ok(ParakeetEngine {
            model: loaded.value,
            threads: loaded.threads,
            max_secs: model.chunk_secs.unwrap_or(DEFAULT_MAX_SECS),
            backend: loaded.backend,
        })
    }

    /// The intra-op thread count ONNX Runtime's sessions were built with.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// The longest chunk one call takes, in seconds (the catalog's `chunk_secs`).
    pub fn max_secs(&self) -> u32 {
        self.max_secs
    }
}

impl Transcriber for ParakeetEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &ChunkOptions,
    ) -> Result<Transcript, SpeechError> {
        transcribe_chunk(&mut self.model, audio, opts, self.max_secs)
    }

    fn backend(&self) -> SpeechBackend {
        self.backend.clone()
    }
}
