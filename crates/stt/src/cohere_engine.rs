//! Cohere Transcribe (03-2026, int8 ONNX export) through parakeet-rs and ONNX Runtime, behind the [`Transcriber`]
//! trait.
//!
//! * Text only. A call returns ONE segment spanning the whole chunk (`0..audio_ms`) whose `words` is empty. The
//!   model gives no times, and the catalog says so (`word_times: false`). `Timing` makes no difference.
//! * One call takes a chunk of up to the catalog's `chunk_secs`, 35 s (the clips it was trained on); longer audio is
//!   [`SpeechError::TooLong`] and nothing is decoded. The app cuts the audio.
//! * The model has no automatic language detection: it must be told a language. `ChunkOptions::language` is used
//!   when it is one of the model's 14 ([`pick_language`]; "en-US" and "pt_BR" become "en" and "pt"), and `None` or
//!   "auto" means English. A language the model lacks is `SpeechError::Unsupported("language")`, with no text in it.
//!   `Transcript::language` is the language it decoded in.
//! * Punctuation and capitals are on, and so is the model's own number style: with punctuation on it writes numbers
//!   as digits ("23"), as Whisper and Parakeet do, so the app gets the same style from every engine. The model's
//!   inverse text normalisation switch ([`ITN`]) is left off: on spoken numbers it changed nothing in two of three
//!   test phrases and in the third it turned the digits into words (measured with the real model, 2026-11-16).
//! * Cohere's card warns that the model writes text for silence and noise. This engine adds no voice detection: an
//!   app runs `vad::detect` first and sends the speech only. Output that is only a noise marker or punctuation
//!   ([`crate::is_noise`]) gives a transcript with no segments.
//! * It always runs on the processor (the catalog gives no card path): `gpu: true` loads on the processor with
//!   `fallback: Some(NoCardPath)`, as for Parakeet Unified and Nemotron. ONNX threads follow the thread rule.
//!
//! The pure parts are public so `tests/cohere_api.rs` can test them without a model.

use crate::api::{
    ChunkOptions, LoadOptions, Segment, SpeechBackend, SpeechError, Transcriber, Transcript,
};
use crate::parakeet_engine::{execution_config, from_parakeet, load_on_best_provider};
use crate::threads::logical_cores;
use std::path::Path;
use voice_engine_models::catalog::{Kind, ModelInfo};

/// The chunk limit when the catalog entry gives none (the entry gives 35).
pub const DEFAULT_MAX_SECS: u32 = 35;
/// The language used when the app gives none: the model cannot detect one.
pub const DEFAULT_LANGUAGE: &str = "en";
/// The model's inverse text normalisation switch (`<|itn|>`), off. With punctuation on Cohere writes numbers as digits
/// either way, like Whisper and Parakeet; switching this on did not add digits and once removed them (see the module
/// docs).
pub const ITN: bool = false;

/// The language Cohere decodes in for the app's hint, or `None` when the model does not have it.
///
/// `supported` is the model's own list (`CohereASR::supported_languages`: "ar", "de", "el", "en", ...). No hint,
/// an empty one and "auto" give [`DEFAULT_LANGUAGE`]. A locale ("en-US", "pt_BR", "zh-Hant") is cut to its language
/// code; the case does not matter.
pub fn pick_language(hint: Option<&str>, supported: &[String]) -> Option<String> {
    let hint = hint.map(str::trim).unwrap_or("");
    if hint.is_empty() || hint.eq_ignore_ascii_case("auto") {
        return Some(DEFAULT_LANGUAGE.to_string());
    }
    let code = hint
        .split(['-', '_'])
        .next()
        .unwrap_or("")
        .to_ascii_lowercase();
    supported.iter().find(|s| **s == code).cloned()
}

/// The transcript of one chunk of `audio_ms` milliseconds whose text is `text`: one segment over the whole chunk, no
/// words, or no segment at all when the text is empty or only noise markers and punctuation.
pub fn transcript_from_text(text: &str, language: &str, audio_ms: u64) -> Transcript {
    let text = text.trim();
    let segments = if crate::is_noise(text) {
        vec![]
    } else {
        vec![Segment {
            start_ms: 0,
            end_ms: audio_ms,
            text: text.to_string(),
            // The text is spaced, like Parakeet's.
            starts_word: true,
            words: vec![],
        }]
    };
    Transcript {
        segments,
        language: Some(language.to_string()),
        stopped_early_at_ms: None,
    }
}

/// Every file the catalog lists is in the folder (`has` says whether a name is there).
pub fn check_files(model: &ModelInfo, has: &dyn Fn(&str) -> bool) -> Result<(), SpeechError> {
    if model.files.iter().any(|f| !has(f.name)) {
        return Err(SpeechError::ModelMissing);
    }
    Ok(())
}

/// Cohere Transcribe loaded in this process.
pub struct CohereEngine {
    model: parakeet_rs::CohereASR,
    languages: Vec<String>,
    threads: usize,
    max_secs: u32,
    backend: SpeechBackend,
}

impl CohereEngine {
    /// Loads the model from its folder `dir` (the catalog's files in it, flat). `opts.gpu` runs on the processor and
    /// says so (see the module docs); `opts.threads` or the thread rule sets ONNX Runtime's intra-op threads.
    pub fn load(model: &ModelInfo, dir: &Path, opts: LoadOptions) -> Result<Self, SpeechError> {
        if model.kind != Kind::Cohere {
            return Err(SpeechError::Unsupported("not a Cohere model".into()));
        }
        check_files(model, &|name| dir.join(name).is_file())?;
        // No card path: `load_on_best_provider` with no card and no DirectML path stays on the processor and
        // reports `NoCardPath` when the app asked for a card.
        let loaded = load_on_best_provider(
            opts.gpu,
            None,
            false,
            opts.threads,
            logical_cores(),
            |provider, threads| {
                parakeet_rs::CohereASR::from_pretrained(
                    dir,
                    Some(execution_config(provider, threads)?),
                )
                .map_err(|e| from_parakeet(&e, true))
            },
        )?;
        let languages = loaded.value.supported_languages();
        Ok(CohereEngine {
            model: loaded.value,
            languages,
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

    /// The language codes the model decodes in.
    pub fn languages(&self) -> &[String] {
        &self.languages
    }
}

impl Transcriber for CohereEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &ChunkOptions,
    ) -> Result<Transcript, SpeechError> {
        if audio.len() > self.max_secs as usize * 16_000 {
            return Err(SpeechError::TooLong {
                max_secs: self.max_secs,
            });
        }
        let language = pick_language(opts.language.as_deref(), &self.languages)
            .ok_or_else(|| SpeechError::Unsupported("language".into()))?;
        let audio_ms = audio.len() as u64 * 1000 / 16_000;
        if audio.is_empty() {
            return Ok(transcript_from_text("", &language, audio_ms));
        }
        let text = self
            .model
            .transcribe_audio(audio, &language, true, ITN)
            .map_err(|e| from_parakeet(&e, false))?;
        Ok(transcript_from_text(&text, &language, audio_ms))
    }

    fn backend(&self) -> SpeechBackend {
        self.backend.clone()
    }
}
