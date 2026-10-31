//! Whisper in process: Metal on the Mac, Vulkan when built with the `vulkan` feature, the processor everywhere
//! else, behind the [`Transcriber`] trait.
//!
//! * `Timing::Segments` uses Talaan's settings (greedy, no carried context, blanks and non-speech tokens
//!   suppressed, noise and no-speech segments dropped). The words of each segment come from the token timestamps,
//!   which whisper.cpp computes after the segment is final, so the segments' text and times are the ones Talaan
//!   gets without them (`tests/real_whisper.rs` compares them on a real model).
//! * `Timing::Words` uses Ensayo's (token timestamps, split on word, `max_len` 1: one word per segment), then
//!   [`crate::words::clean`].
//! * Any length goes in one call (whisper.cpp cuts 30 s windows itself). Audio under 1 s is padded with silence.
//!   Times are clamped to the real audio.
//! * When decoding ends with 2 s or more of audio after the last word, the rest is decoded again from there
//!   (as Ensayo's Parakeet engine does), up to 8 passes; `stopped_early_at_ms` says where the first stop was.
//!
//! The pure parts (padding, prompt, clamping, the resume loop, the choice of device) are public so
//! `tests/whisper_api.rs` can test them without a model.

use crate::api::{
    ChunkOptions, LoadOptions, Segment, SpeechBackend, SpeechError, Timing, Transcriber,
    Transcript, Word,
};
use crate::threads::{ThreadKind, logical_cores, resolve_threads};
use crate::words::{clean, clean_flagged};
use std::borrow::Cow;
use std::path::Path;
use voice_engine_protocol::gpu::{Device, DeviceKind, pick, whisper_gpu_index};
use whisper_rs::whisper_rs_sys as sys;

/// Whisper's minimum is 100 ms of audio to do anything and 1 s is the convention of both apps: shorter clips are
/// padded with silence to this many samples (16 kHz).
pub const MIN_SAMPLES: usize = 16_000;
/// Audio left after the last word that counts as "decoding stopped early".
pub const RESUME_GAP_MS: u64 = 2_000;
/// Resume passes after the first one (Ensayo's Parakeet loop: 8 passes in all).
pub const MAX_PASSES: usize = 8;
/// A pass that moves the end of the last word by less than this is not worth another pass.
const MIN_PROGRESS_MS: u64 = 500;
/// Segments Whisper itself says are probably not speech.
const NO_SPEECH_MAX: f32 = 0.8;
/// The prompt whisper.cpp's documentation gives to keep filler words in the transcript.
pub const FILLER_PROMPT: &str =
    "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking.";

/// One text token with its time (milliseconds on the pass's audio).
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// A segment as whisper.cpp reports it, before our filtering (times in milliseconds on the pass's audio).
#[derive(Debug, Clone, PartialEq)]
pub struct RawSegment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub no_speech: f32,
    /// Text tokens only (no timestamp or special tokens); filled in `Timing::Segments` mode.
    pub tokens: Vec<Token>,
}

/// What one decode of a stretch of audio gives: segments relative to that stretch.
pub struct Pass {
    pub segments: Vec<Segment>,
    pub language: Option<String>,
}

/// One decode of a stretch of audio (the real one is whisper.cpp; the tests script it).
pub trait PassDecoder {
    fn decode(&mut self, audio: &[f32]) -> Result<Pass, SpeechError>;
}

/// Audio of at least 1 s, as Whisper wants: shorter clips get silence after them.
pub fn pad_audio(samples: &[f32]) -> Cow<'_, [f32]> {
    if samples.len() >= MIN_SAMPLES {
        return Cow::Borrowed(samples);
    }
    let mut audio = samples.to_vec();
    audio.resize(MIN_SAMPLES, 0.0);
    Cow::Owned(audio)
}

/// whisper.cpp's initial prompt: the filler prompt first when asked, then the custom words joined with ", " and ended
/// with a full stop (measured: without it Whisper copies the prompt's bare style and drops capitals and full stops).
/// `None` when there is nothing to say. Null bytes are removed (whisper-rs panics on them) and blank words dropped.
pub fn build_prompt(vocabulary: &[String], keep_fillers: bool) -> Option<String> {
    let words: Vec<String> = vocabulary
        .iter()
        .map(|v| v.replace('\0', "").trim().to_string())
        .filter(|v| !v.is_empty())
        .collect();
    let mut parts: Vec<String> = vec![];
    if keep_fillers {
        parts.push(FILLER_PROMPT.to_string());
    }
    if !words.is_empty() {
        parts.push(format!("{}.", words.join(", ")));
    }
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

/// Words from a segment's tokens: a token that starts with a space starts a word, the others continue it
/// (Whisper's own convention). A word runs from its first token's start to its last token's end.
pub fn words_from_tokens(tokens: &[Token]) -> Vec<Word> {
    let mut out: Vec<Word> = vec![];
    for t in tokens {
        let starts_word = t.text.starts_with(char::is_whitespace) || out.is_empty();
        match out.last_mut() {
            Some(last) if !starts_word => {
                last.text.push_str(&t.text);
                last.end_ms = last.end_ms.max(t.end_ms);
            }
            _ => out.push(Word {
                text: t.text.clone(),
                start_ms: t.start_ms,
                end_ms: t.end_ms.max(t.start_ms),
            }),
        }
    }
    for w in &mut out {
        w.text = w.text.trim().to_string();
    }
    out.retain(|w| !w.text.is_empty());
    out
}

fn clamp_span(start_ms: u64, end_ms: u64, audio_ms: u64) -> (u64, u64) {
    let start = start_ms.min(audio_ms);
    (start, end_ms.min(audio_ms).max(start))
}

/// Whisper's segments to ours: the filtering of the mode, times clamped to the real audio (`audio_ms`, not the
/// padded length), words attached.
pub fn segments_from_raw(raw: Vec<RawSegment>, timing: Timing, audio_ms: u64) -> Vec<Segment> {
    match timing {
        // Talaan: noise and no-speech segments dropped, text trimmed.
        Timing::Segments => raw
            .into_iter()
            .filter_map(|r| {
                let starts_word = r.text.starts_with(char::is_whitespace);
                let text = r.text.trim().to_string();
                if crate::is_noise(&text) || r.no_speech > NO_SPEECH_MAX {
                    return None;
                }
                let (start_ms, end_ms) = clamp_span(r.start_ms, r.end_ms, audio_ms);
                let tokens: Vec<Token> = r
                    .tokens
                    .into_iter()
                    .map(|t| {
                        let (start_ms, end_ms) = clamp_span(t.start_ms, t.end_ms, audio_ms);
                        Token {
                            text: t.text,
                            start_ms,
                            end_ms,
                        }
                    })
                    .collect();
                let words = clean(words_from_tokens(&tokens));
                Some(Segment {
                    start_ms,
                    end_ms,
                    text,
                    starts_word,
                    words,
                })
            })
            .collect(),
        // Ensayo: every segment is one word; clean drops noise and glues punctuation to the word before.
        Timing::Words => {
            let words = raw
                .into_iter()
                .filter(|r| r.no_speech <= NO_SPEECH_MAX)
                .map(|r| {
                    let (start_ms, end_ms) = clamp_span(r.start_ms, r.end_ms, audio_ms);
                    let starts_word = r.text.starts_with(char::is_whitespace);
                    (
                        Word {
                            text: r.text,
                            start_ms,
                            end_ms,
                        },
                        starts_word,
                    )
                })
                .collect();
            clean_flagged(words)
                .into_iter()
                .map(|(w, starts_word)| Segment {
                    start_ms: w.start_ms,
                    end_ms: w.end_ms,
                    text: w.text.clone(),
                    starts_word,
                    words: vec![w],
                })
                .collect()
        }
    }
}

fn shifted(mut s: Segment, by_ms: u64, audio_ms: u64) -> Segment {
    let (start_ms, end_ms) = clamp_span(s.start_ms + by_ms, s.end_ms + by_ms, audio_ms);
    s.start_ms = start_ms;
    s.end_ms = end_ms;
    for w in &mut s.words {
        let (a, b) = clamp_span(w.start_ms + by_ms, w.end_ms + by_ms, audio_ms);
        w.start_ms = a;
        w.end_ms = b;
    }
    s
}

/// Decodes all of `samples`, and when a pass ends with [`RESUME_GAP_MS`] or more of audio after the last word,
/// decodes the rest again from the end of that word (at most [`MAX_PASSES`] passes), putting every pass on the
/// clock of the whole audio. `stopped_early_at_ms` is where the first such stop was.
pub fn decode_with_resume(
    decoder: &mut dyn PassDecoder,
    samples: &[f32],
) -> Result<Transcript, SpeechError> {
    let total_ms = samples.len() as u64 * 1000 / 16_000;
    let mut segments: Vec<Segment> = vec![];
    let mut language: Option<String> = None;
    let mut stopped_early_at_ms: Option<u64> = None;
    let mut from_ms = 0u64;
    for _ in 0..MAX_PASSES {
        let from = ((from_ms * 16) as usize).min(samples.len());
        let pass = decoder.decode(&samples[from..])?;
        if language.is_none() {
            language = pass.language;
        }
        let got: Vec<Segment> = pass
            .segments
            .into_iter()
            .map(|s| shifted(s, from_ms, total_ms))
            .collect();
        let reached = got
            .iter()
            .map(|s| s.end_ms.max(s.words.last().map_or(0, |w| w.end_ms)))
            .max()
            .unwrap_or(from_ms);
        segments.extend(got);
        let progressed = reached > from_ms + MIN_PROGRESS_MS;
        if !progressed || total_ms.saturating_sub(reached) < RESUME_GAP_MS {
            break;
        }
        stopped_early_at_ms.get_or_insert(reached);
        from_ms = reached;
    }
    Ok(Transcript {
        segments,
        language,
        stopped_early_at_ms,
    })
}

/// The device to run on when the app asked for the graphics card: `(device, whisper.cpp's gpu_device number)`.
/// Mac: the Metal device (ggml lists it as a GPU or an integrated GPU, the first one). Elsewhere: the discrete GPU
/// with the most memory, never an integrated one (the policy of the Vulkan helper). `None`: there is no usable
/// card, run on the processor.
pub fn choose_gpu(devices: &[Device], mac: bool) -> Option<(&Device, i32)> {
    let usable = |d: &&Device| !d.backend.eq_ignore_ascii_case("cpu");
    let chosen = if mac {
        devices
            .iter()
            .filter(usable)
            .find(|d| d.kind == DeviceKind::Gpu)
            .or_else(|| {
                devices
                    .iter()
                    .filter(usable)
                    .find(|d| d.kind == DeviceKind::Igpu)
            })
    } else {
        pick(devices)
    }?;
    whisper_gpu_index(devices, chosen).map(|n| (chosen, n))
}

/// What `backend()` says: the device that runs the model, or the processor, and whether the card was refused.
pub fn describe_backend(gpu: Option<&Device>, asked_gpu: bool) -> SpeechBackend {
    match gpu {
        Some(d) => SpeechBackend {
            backend: d.backend.to_ascii_lowercase(),
            device: Some(d.description.clone()),
            fell_back: false,
        },
        None => SpeechBackend {
            backend: "cpu".into(),
            device: None,
            fell_back: asked_gpu,
        },
    }
}

/// Starts the device the way whisper.cpp will (and stops it again). whisper.cpp itself quietly loads on the
/// processor when its GPU backend does not start, which would leave `backend()` claiming a card that is not
/// running; asking first makes the report true.
fn gpu_starts(device: &Device) -> bool {
    unsafe {
        let dev = sys::ggml_backend_dev_get(device.index);
        if dev.is_null() {
            return false;
        }
        let backend = sys::ggml_backend_dev_init(dev, std::ptr::null());
        if backend.is_null() {
            return false;
        }
        sys::ggml_backend_free(backend);
        true
    }
}

fn open(
    file: &Path,
    gpu_device: Option<i32>,
    flash_attn: bool,
) -> Result<(whisper_rs::WhisperContext, whisper_rs::WhisperState), SpeechError> {
    let params = crate::context_params(gpu_device, flash_attn);
    let ctx = whisper_rs::WhisperContext::new_with_params(file, params).map_err(|e| match e {
        whisper_rs::WhisperError::InitError => SpeechError::BadModel,
        other => SpeechError::Internal(other.to_string()),
    })?;
    // The state is where whisper.cpp allocates its working buffers (hundreds of MB): the step that runs out of
    // memory.
    let state = ctx.create_state().map_err(|e| match e {
        whisper_rs::WhisperError::FailedToCreateState => SpeechError::OutOfMemory,
        other => SpeechError::Internal(other.to_string()),
    })?;
    Ok((ctx, state))
}

/// A Whisper model loaded in this process.
pub struct WhisperEngine {
    // The state holds ~400 MB of buffers; one per model, reused for every call. Dropped before the context.
    state: whisper_rs::WhisperState,
    ctx: whisper_rs::WhisperContext,
    threads: usize,
    backend: SpeechBackend,
}

impl WhisperEngine {
    /// Loads the ggml file. `opts.gpu` tries the graphics card first (Metal on the Mac, Vulkan with the `vulkan`
    /// feature); if it is not there or cannot load the model, the processor loads it and `backend().fell_back`
    /// is true. The thread count is `opts.threads` or the default for where the model actually runs.
    pub fn load(file: &Path, opts: LoadOptions) -> Result<Self, SpeechError> {
        // whisper.cpp and ggml log to stderr by default; drop it (errors come back as kinds).
        whisper_rs::install_logging_hooks();
        if !file.is_file() {
            return Err(SpeechError::ModelMissing);
        }
        let logical = logical_cores();
        let mac = cfg!(target_os = "macos");
        if opts.gpu {
            let all = crate::devices();
            if let Some((device, number)) = choose_gpu(&all, mac)
                && gpu_starts(device)
            {
                // Flash attention only pays off on Metal; on the processor it costs 10-15% (Talaan).
                if let Ok((ctx, state)) = open(file, Some(number), mac) {
                    return Ok(WhisperEngine {
                        state,
                        ctx,
                        threads: resolve_threads(ThreadKind::Gpu, opts.threads, logical),
                        backend: describe_backend(Some(device), true),
                    });
                }
            }
        }
        let (ctx, state) = open(file, None, false)?;
        Ok(WhisperEngine {
            state,
            ctx,
            threads: resolve_threads(ThreadKind::Processor, opts.threads, logical),
            backend: describe_backend(None, opts.gpu),
        })
    }

    /// The thread count every decode passes to whisper.cpp.
    pub fn threads(&self) -> usize {
        self.threads
    }
}

struct WhisperPass<'a> {
    engine: &'a mut WhisperEngine,
    language: Option<&'a str>,
    timing: Timing,
    prompt: Option<&'a str>,
}

impl PassDecoder for WhisperPass<'_> {
    fn decode(&mut self, audio: &[f32]) -> Result<Pass, SpeechError> {
        let audio_ms = audio.len() as u64 * 1000 / 16_000;
        let padded = pad_audio(audio);
        let mut params =
            whisper_rs::FullParams::new(whisper_rs::SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(self.engine.threads as i32);
        params.set_language(Some(self.language.unwrap_or("auto")));
        // Each call stands alone: carrying text between calls is how Whisper starts repeating itself.
        params.set_no_context(true);
        params.set_suppress_blank(true);
        params.set_suppress_nst(true);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        // Token timestamps in both modes (the words of a segment come from them). Only `Words` also splits the
        // segments, into one word each.
        params.set_token_timestamps(true);
        if self.timing == Timing::Words {
            params.set_split_on_word(true);
            params.set_max_len(1);
        }
        if let Some(prompt) = self.prompt {
            params.set_initial_prompt(prompt);
        }
        self.engine
            .state
            .full(params, &padded)
            .map_err(|e| SpeechError::Internal(format!("transcription failed: {e}")))?;
        let language = whisper_rs::get_lang_str(self.engine.state.full_lang_id_from_state())
            .map(str::to_string);
        let eot = self.engine.ctx.token_eot();
        let want_tokens = self.timing == Timing::Segments;
        let mut raw = vec![];
        for i in 0..self.engine.state.full_n_segments() {
            let Some(seg) = self.engine.state.get_segment(i) else {
                continue;
            };
            let mut tokens = vec![];
            if want_tokens {
                for j in 0..seg.n_tokens() {
                    let Some(tok) = seg.get_token(j) else {
                        continue;
                    };
                    let data = tok.token_data();
                    // Special tokens (end of text, timestamps, language) are not words.
                    if data.id >= eot {
                        continue;
                    }
                    tokens.push(Token {
                        text: tok
                            .to_str_lossy()
                            .map(|t| t.into_owned())
                            .unwrap_or_default(),
                        start_ms: (data.t0.max(0) * 10) as u64,
                        end_ms: (data.t1.max(0) * 10) as u64,
                    });
                }
            }
            raw.push(RawSegment {
                // Centiseconds.
                start_ms: (seg.start_timestamp().max(0) * 10) as u64,
                end_ms: (seg.end_timestamp().max(0) * 10) as u64,
                text: seg
                    .to_str_lossy()
                    .map(|t| t.into_owned())
                    .unwrap_or_default(),
                no_speech: seg.no_speech_probability(),
                tokens,
            });
        }
        Ok(Pass {
            segments: segments_from_raw(raw, self.timing, audio_ms),
            language,
        })
    }
}

impl Transcriber for WhisperEngine {
    fn transcribe(
        &mut self,
        audio: &[f32],
        opts: &ChunkOptions,
    ) -> Result<Transcript, SpeechError> {
        let prompt = build_prompt(&opts.vocabulary, opts.keep_fillers);
        let language = opts
            .language
            .as_deref()
            .map(str::trim)
            .filter(|l| !l.is_empty());
        let mut pass = WhisperPass {
            engine: self,
            language,
            timing: opts.timing,
            prompt: prompt.as_deref(),
        };
        decode_with_resume(&mut pass, audio)
    }

    fn backend(&self) -> SpeechBackend {
        self.backend.clone()
    }
}
