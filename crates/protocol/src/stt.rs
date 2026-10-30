//! Messages between an app and its speech helper, protocol 2. Same framing as the AI helper (`crate::Request`): one
//! JSON object per line on stdin (requests) and stdout (events), one request at a time, `ready` with the protocol
//! version, errors that carry a kind.
//!
//! The helper serves Whisper only. Parakeet, Parakeet Unified and Nemotron run through ONNX Runtime inside the app's
//! own process (`voice_engine_stt::load`); a `load` of one of them is answered with `unsupported`.
//!
//! Audio travels as base64 of little-endian f32 samples (16 kHz mono), as it did in v0.1.0: 2.5 MB of text for a
//! 30 s chunk, about 300 MB for an hour, which is why an app that sends long recordings through the helper cuts them
//! first. A temporary file path would be cheaper for long audio but would make both sides share a folder, and the
//! v0.1.0 helper never did; it can be added later as another field.
//!
//! The transcript types (`ChunkOptions`, `Transcript`, `Segment`, `Word`, `SpeechBackend`) live here so the helper,
//! the client and `voice_engine_stt` (which re-exports them) share one definition.
//!
//! The v0.1.0 speech messages (`load { model_path }`, `transcribe { samples, language }`, `transcribed { pieces }`)
//! are replaced and no longer read: nobody had adopted them (CHANGELOG, breaking for anyone who had).

use super::Backend;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

/// Which kind of speech model a `load` is for. The helper runs `Whisper` only.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SttModelKind {
    Whisper,
    Parakeet,
    ParakeetUnified,
    Nemotron,
}

/// `load`: which model, where it is, and how to run it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SttLoad {
    /// The catalog id ("whisper-turbo"): the catalog says which file in the folder is the model.
    pub model_id: String,
    /// The model's folder (the shared store's `<model id>/`).
    pub model_dir: String,
    pub kind: SttModelKind,
    /// Metal or Vulkan, if the helper has one; the processor otherwise (the answer says which, and `fell_back`).
    #[serde(default = "super::yes")]
    pub gpu: bool,
    /// None: the engine's thread rule. Some: the app's own setting.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub threads: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SttRequest {
    /// Answers with `SttEvent::Ready`.
    Ping,
    /// Answers with `SttEvent::Devices`: every ggml device (and, in a build with `directml`, every DirectML adapter,
    /// backend "DirectML") and the ggml one the helper chose. The app asks the card helper this once at startup to
    /// learn whether the PC has a discrete card (`gpu::discrete_card`).
    Devices,
    /// Loads a Whisper model, replacing any other. Answers with `Loaded`. The model stays loaded for every
    /// `Transcribe` until `Unload`, another `Load`, or the helper's exit.
    Load(SttLoad),
    /// Transcribes one piece of audio with the loaded model. Answers with `Transcribed`; with no model loaded, with
    /// an `Error` of kind `bad_request`.
    Transcribe {
        /// `encode_samples` of the audio (16 kHz mono, any length).
        audio_b64: String,
        #[serde(default)]
        options: ChunkOptions,
    },
    /// Frees the model's memory now; the helper keeps running. Answers with `Unloaded`.
    Unload,
    /// The app stopped waiting. A decode in progress cannot be interrupted, so the helper ignores it; the app's client
    /// gives the helper a couple of seconds and then ends the process (the model is loaded again on the next call).
    Cancel,
}

/// What to do with one piece of audio.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct ChunkOptions {
    /// A hint ("fr"); `None` lets the model detect the language.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    pub timing: Timing,
    /// Ask the model to keep "um", "uh" and "like" (Whisper: a short prompt of fillers; the ONNX models transcribe
    /// them as heard and ignore this).
    pub keep_fillers: bool,
    /// The app's custom words, spelled as the person wants them (Whisper's initial prompt).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub vocabulary: Vec<String>,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        ChunkOptions {
            language: None,
            timing: Timing::Words,
            keep_fillers: false,
            vocabulary: vec![],
        }
    }
}

/// Which settings the model decodes with. Every result carries both segments and words, in either mode.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Timing {
    /// Talaan's settings: the segments the model produced; the words are derived from the token times.
    Segments,
    /// Ensayo's settings: token timestamps, one word per segment.
    Words,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Transcript {
    /// What the model produced, in order, each with its words.
    pub segments: Vec<Segment>,
    /// The language the model settled on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    /// Decoding stopped with 2 s or more of audio left after this point (the engine resumed from here).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_early_at_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Segment {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<Word>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Word {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
}

/// Where a model runs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SpeechBackend {
    /// `metal`, `vulkan`, `directml` or `cpu`.
    pub backend: String,
    /// The GPU's name when known; `None` on the processor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// The app asked for the GPU (`gpu: true`) and the model runs on the processor instead.
    #[serde(default)]
    pub fell_back: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum SttEvent {
    Ready {
        #[serde(flatten)]
        backend: Backend,
        #[serde(default)]
        protocol: u32,
    },
    Devices {
        devices: Vec<super::Device>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen: Option<usize>,
    },
    /// The model is loaded, and runs on this backend.
    Loaded {
        seconds: f32,
        backend: SpeechBackend,
    },
    /// The result, and where it was computed.
    Transcribed {
        transcript: Transcript,
        backend: SpeechBackend,
    },
    /// The model's memory is freed (or there was none loaded).
    Unloaded,
    Error {
        message: String,
        /// What went wrong, for the app to act on; the message is for logs.
        #[serde(default)]
        error: SttErrorKind,
    },
}

/// Why a speech request failed. Apps match on this, never on the message.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SttErrorKind {
    /// The model's file is not in its folder.
    ModelMissing {
        path: String,
    },
    /// The file is there but cannot be read as this model.
    BadModel,
    OutOfMemory,
    /// The audio is longer than the model takes in one call; the app cuts it and calls again.
    TooLong {
        max_secs: u32,
    },
    /// This helper cannot run that (an ONNX model, or a file that is not a speech model).
    Unsupported {
        detail: String,
    },
    /// The graphics card failed. `fell_back`: the work was done on the processor instead (the app can say so once
    /// and stop trying the card); false: nothing was done.
    GpuFailed {
        fell_back: bool,
    },
    Cancelled,
    BadRequest,
    #[default]
    Internal,
}

/// Reads one request line. A UTF-8 byte order mark (a file piped in by hand) isn't part of the request.
pub fn decode_request(line: &str) -> Result<SttRequest, serde_json::Error> {
    serde_json::from_str(line.trim_start_matches('\u{feff}').trim())
}

/// Reads one event line.
pub fn decode_event(line: &str) -> Result<SttEvent, serde_json::Error> {
    serde_json::from_str(line.trim())
}

pub fn encode_samples(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn decode_samples(text: &str) -> Result<Vec<f32>, String> {
    // The error says nothing about the data: base64's own text names the byte that did not fit.
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|_| "bad audio: not base64".to_string())?;
    if bytes.len() % 4 != 0 {
        return Err("bad audio: not a whole number of samples".into());
    }
    let (samples, _) = bytes.as_chunks::<4>();
    Ok(samples.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Backend, encode};

    fn options() -> ChunkOptions {
        ChunkOptions {
            language: Some("fr".into()),
            timing: Timing::Segments,
            keep_fillers: true,
            vocabulary: vec!["Talaan".into(), "Ensayo".into()],
        }
    }

    fn transcript() -> Transcript {
        Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 1200,
                text: "Bonjour, tout le monde.".into(),
                words: vec![
                    Word {
                        text: "Bonjour,".into(),
                        start_ms: 0,
                        end_ms: 500,
                    },
                    Word {
                        text: "tout".into(),
                        start_ms: 600,
                        end_ms: 800,
                    },
                ],
            }],
            language: Some("fr".into()),
            stopped_early_at_ms: Some(1100),
        }
    }

    fn metal() -> SpeechBackend {
        SpeechBackend {
            backend: "metal".into(),
            device: Some("Apple M3 Pro".into()),
            fell_back: false,
        }
    }

    #[test]
    fn every_request_round_trips_as_one_line() {
        let requests = [
            SttRequest::Ping,
            SttRequest::Devices,
            SttRequest::Load(SttLoad {
                model_id: "whisper-turbo".into(),
                model_dir: "C:\\Models\\whisper-turbo".into(),
                kind: SttModelKind::Whisper,
                gpu: true,
                threads: Some(6),
            }),
            SttRequest::Load(SttLoad {
                model_id: "whisper-base".into(),
                model_dir: "/m/whisper-base".into(),
                kind: SttModelKind::Whisper,
                gpu: false,
                threads: None,
            }),
            SttRequest::Transcribe {
                audio_b64: encode_samples(&[0.5, -0.25]),
                options: options(),
            },
            SttRequest::Transcribe {
                audio_b64: String::new(),
                options: ChunkOptions::default(),
            },
            SttRequest::Unload,
            SttRequest::Cancel,
        ];
        for request in requests {
            let line = encode(&request);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_request(&line).unwrap(), request, "{line}");
        }
        assert_eq!(encode(&SttRequest::Ping), r#"{"op":"ping"}"#);
        assert_eq!(encode(&SttRequest::Unload), r#"{"op":"unload"}"#);
        assert_eq!(encode(&SttRequest::Cancel), r#"{"op":"cancel"}"#);
    }

    #[test]
    fn every_event_round_trips_as_one_line() {
        let events = [
            SttEvent::Ready {
                backend: Backend {
                    backend: "vulkan".into(),
                    device: Some("NVIDIA GeForce RTX 5070 Ti".into()),
                    vram_mb: 16303,
                },
                protocol: crate::PROTOCOL_VERSION,
            },
            SttEvent::Devices {
                devices: vec![],
                chosen: None,
            },
            SttEvent::Loaded {
                seconds: 1.5,
                backend: metal(),
            },
            SttEvent::Loaded {
                seconds: 0.0,
                backend: SpeechBackend {
                    backend: "cpu".into(),
                    device: None,
                    fell_back: true,
                },
            },
            SttEvent::Transcribed {
                transcript: transcript(),
                backend: metal(),
            },
            SttEvent::Transcribed {
                transcript: Transcript {
                    segments: vec![],
                    language: None,
                    stopped_early_at_ms: None,
                },
                backend: metal(),
            },
            SttEvent::Unloaded,
            SttEvent::Error {
                message: "m".into(),
                error: SttErrorKind::Internal,
            },
        ];
        for event in events {
            let line = encode(&event);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_event(&line).unwrap(), event, "{line}");
        }
    }

    #[test]
    fn every_error_kind_round_trips_and_has_its_own_wire_shape() {
        let kinds = [
            (
                SttErrorKind::ModelMissing {
                    path: "/m/ggml-base.bin".into(),
                },
                r#"{"kind":"model_missing","path":"/m/ggml-base.bin"}"#,
            ),
            (SttErrorKind::BadModel, r#"{"kind":"bad_model"}"#),
            (SttErrorKind::OutOfMemory, r#"{"kind":"out_of_memory"}"#),
            (
                SttErrorKind::TooLong { max_secs: 40 },
                r#"{"kind":"too_long","max_secs":40}"#,
            ),
            (
                SttErrorKind::Unsupported {
                    detail: "not a speech model".into(),
                },
                r#"{"kind":"unsupported","detail":"not a speech model"}"#,
            ),
            (
                SttErrorKind::GpuFailed { fell_back: true },
                r#"{"kind":"gpu_failed","fell_back":true}"#,
            ),
            (
                SttErrorKind::GpuFailed { fell_back: false },
                r#"{"kind":"gpu_failed","fell_back":false}"#,
            ),
            (SttErrorKind::Cancelled, r#"{"kind":"cancelled"}"#),
            (SttErrorKind::BadRequest, r#"{"kind":"bad_request"}"#),
            (SttErrorKind::Internal, r#"{"kind":"internal"}"#),
        ];
        for (error, wire) in kinds {
            assert_eq!(serde_json::to_string(&error).unwrap(), wire);
            let event = SttEvent::Error {
                message: "stopped".into(),
                error: error.clone(),
            };
            let line = encode(&event);
            assert_eq!(
                line,
                format!(r#"{{"event":"error","message":"stopped","error":{wire}}}"#)
            );
            assert_eq!(decode_event(&line).unwrap(), event);
        }
    }

    #[test]
    fn an_error_without_a_kind_reads_as_internal() {
        let SttEvent::Error { error, message } =
            decode_event(r#"{"event":"error","message":"old"}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(error, SttErrorKind::Internal);
        assert_eq!(message, "old");
    }

    #[test]
    fn a_load_without_gpu_or_threads_asks_for_the_card_and_the_engines_threads() {
        let SttRequest::Load(l) = decode_request(
            r#"{"op":"load","model_id":"whisper-base","model_dir":"/m","kind":"whisper"}"#,
        )
        .unwrap() else {
            panic!()
        };
        assert!(l.gpu);
        assert_eq!(l.threads, None);
        let line = encode(&SttRequest::Load(l));
        assert!(!line.contains("threads"), "{line}");
        // Every kind the catalog has a speech model of has a wire name.
        for (kind, name) in [
            (SttModelKind::Whisper, "whisper"),
            (SttModelKind::Parakeet, "parakeet"),
            (SttModelKind::ParakeetUnified, "parakeet_unified"),
            (SttModelKind::Nemotron, "nemotron"),
        ] {
            assert_eq!(serde_json::to_string(&kind).unwrap(), format!("\"{name}\""));
        }
    }

    #[test]
    fn a_transcribe_without_options_uses_the_defaults() {
        let SttRequest::Transcribe { options, .. } =
            decode_request(r#"{"op":"transcribe","audio_b64":""}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(options, ChunkOptions::default());
        assert_eq!(options.timing, Timing::Words);
        assert_eq!(options.language, None);
        assert!(!options.keep_fillers);
        assert!(options.vocabulary.is_empty());
        // Options sent in part: the rest default.
        let SttRequest::Transcribe { options, .. } = decode_request(
            r#"{"op":"transcribe","audio_b64":"","options":{"timing":"segments","language":"en"}}"#,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(options.timing, Timing::Segments);
        assert_eq!(options.language.as_deref(), Some("en"));
        assert!(options.vocabulary.is_empty());
    }

    #[test]
    fn the_words_of_a_transcript_keep_their_times_and_text_exactly() {
        let line = encode(&SttEvent::Transcribed {
            transcript: transcript(),
            backend: metal(),
        });
        // Non-ASCII text and punctuation survive; the line has no raw newline.
        assert!(line.contains("Bonjour, tout le monde."), "{line}");
        assert!(line.contains(r#""start_ms":600,"end_ms":800"#), "{line}");
        let SttEvent::Transcribed { transcript: t, .. } = decode_event(&line).unwrap() else {
            panic!()
        };
        assert_eq!(t, transcript());
        // A result from a helper that leaves out the optional parts still reads.
        let SttEvent::Transcribed { transcript: t, .. } = decode_event(
            r#"{"event":"transcribed","transcript":{"segments":[]},"backend":{"backend":"cpu","fell_back":false}}"#,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!((t.language, t.stopped_early_at_ms), (None, None));
    }

    #[test]
    fn the_ready_of_the_speech_helper_carries_the_protocol_version() {
        let line = encode(&SttEvent::Ready {
            backend: Backend::cpu(),
            protocol: crate::PROTOCOL_VERSION,
        });
        assert_eq!(
            line,
            r#"{"event":"ready","backend":"cpu","vram_mb":0,"protocol":2}"#
        );
        // A bare ready (a fake helper, or a helper from before the engine) reads as protocol 0.
        assert_eq!(
            decode_event(r#"{"event":"ready"}"#).unwrap(),
            SttEvent::Ready {
                backend: Backend::default(),
                protocol: 0
            }
        );
    }

    #[test]
    fn the_v0_1_0_speech_messages_are_gone() {
        // v0.1.0: `load { model_path }`, `transcribe { samples, language }` answered by `transcribed { pieces }`.
        // Nobody adopted them; they are not read any more (a helper and an app from different builds fail loudly).
        assert!(decode_request(r#"{"op":"load","model_path":"m"}"#).is_err());
        assert!(decode_request(r#"{"op":"transcribe","samples":"","language":"fr"}"#).is_err());
        assert!(decode_event(r#"{"event":"transcribed","pieces":[]}"#).is_err());
    }

    #[test]
    fn a_byte_order_mark_and_blank_edges_are_ignored_and_garbage_is_an_error() {
        assert_eq!(
            decode_request("\u{feff}{\"op\":\"ping\"}  ").unwrap(),
            SttRequest::Ping
        );
        assert!(decode_request("not json").is_err());
        assert!(decode_request(r#"{"op":"unknown"}"#).is_err());
        assert!(decode_event("").is_err());
    }

    #[test]
    fn samples_survive_the_trip_exactly() {
        let samples: Vec<f32> = (0..16_000)
            .map(|i| ((i as f32) * 0.01).sin() * 0.8)
            .chain([0.0, -1.0, 1.0, f32::MIN_POSITIVE])
            .collect();
        assert_eq!(decode_samples(&encode_samples(&samples)).unwrap(), samples);
        assert_eq!(decode_samples("").unwrap(), Vec::<f32>::new());
        assert!(decode_samples("AAA").is_err());
        assert!(decode_samples("AAAAAAA=").is_err(), "5 bytes");
    }

    #[test]
    fn audio_is_little_endian_f32_in_base64() {
        // 1.0f32 is 00 00 80 3F little-endian: "AACAPw==" in base64.
        assert_eq!(encode_samples(&[1.0]), "AACAPw==");
        assert_eq!(decode_samples("AACAPw==").unwrap(), vec![1.0]);
        // The error text says nothing about the data (base64's own text names the byte that did not fit).
        assert_eq!(decode_samples("a!b").unwrap_err(), "bad audio: not base64");
        assert_eq!(
            decode_samples("AAA=AAA=").unwrap_err(),
            "bad audio: not base64"
        );
    }
}
