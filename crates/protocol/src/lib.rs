//! Messages between an app and its local AI helper (a CPU build, and on Windows a Vulkan GPU build). One JSON
//! object per line on stdin (requests) and stdout (events). The helper runs one request at a time.
//!
//! Also here, because both GPU helpers need them: the speech helper's protocol (`stt`, also protocol 2) and the device
//! policy (`gpu`), which are pure so they can be tested without a GPU.

use serde::{Deserialize, Serialize};

pub mod config;
pub mod gpu;
pub mod stt;

pub use config::HelperConfig;
pub use gpu::{Device, DeviceKind};

/// Bumped whenever a message changes shape. Each helper sends it in `Ready`; the app compares it with its own and
/// reports a mismatch (a helper and an app from different builds) instead of failing later. 0 = a helper from before
/// the engine existed. 2 = configure, load, unload, cancel, inspect, thinking, seed, error kinds; and the speech helper's new messages
/// (`stt`: load a model by id, transcribe with options, timed segments and words, error kinds).
pub const PROTOCOL_VERSION: u32 = 2;

/// A JSON object with any keys: the grammar that keeps a reply parseable when the caller has no schema of its own.
pub const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Answers with `Event::Ready`. Used to check the helper started.
    Ping,
    /// Sets how long the helper waits without a request before exiting. Answers with `Event::Configured`.
    Configure(Configure),
    /// Loads a model without generating anything, so the next request doesn't wait. Answers with `Event::Loaded`.
    Load(Load),
    /// Generates one reply. The model stays loaded for the next request with the same path and context size.
    Complete(Completion),
    /// Stops the reply being written; it ends with an `Event::Error` of kind `cancelled` and the model stays loaded.
    /// Ignored when nothing is running.
    Cancel,
    /// Frees the loaded model's memory now; the helper keeps running. Answers with `Event::Unloaded`.
    Unload,
    /// Turns texts into embedding vectors (L2-normalized), for Ask's search index.
    Embed(Embedding),
    /// Reads a GGUF file's metadata without loading it. Answers with `Event::Inspected`.
    Inspect(Inspect),
    /// Answers with `Event::Devices`: every ggml device and the one the helper chose. For checks by hand.
    Devices,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Configure {
    /// Seconds without a request before the helper exits and frees the model's memory. None: never; the helper runs
    /// until the app closes it.
    #[serde(default)]
    pub idle_secs: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Load {
    pub model_path: String,
    /// Context size in tokens the next requests will use (it decides how the model is fitted on a GPU).
    pub n_ctx: u32,
    #[serde(default = "yes")]
    pub gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Inspect {
    pub model_path: String,
}

fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Embedding {
    pub model_path: String,
    pub texts: Vec<String>,
    /// Use the helper's GPU, if it has one. False runs on the processor even in a GPU build.
    #[serde(default = "yes")]
    pub gpu: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Completion {
    pub model_path: String,
    /// Context size in tokens (prompt + reply).
    pub n_ctx: u32,
    pub system: String,
    pub user: String,
    /// When set, the reply is constrained to JSON matching this schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<String>,
    pub max_tokens: u32,
    pub temperature: f32,
    /// Use the helper's GPU, if it has one. False runs on the processor even in a GPU build.
    #[serde(default = "yes")]
    pub gpu: bool,
    /// Let a model that thinks before answering do so. Off (the default): it answers straight away.
    #[serde(default)]
    pub thinking: bool,
    /// With thinking on: the most thinking tokens before the helper ends the thinking itself.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<u32>,
    /// Sampling seed when `temperature` is above 0 (default 1234). The same request on the same computer and
    /// backend gives the same reply.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u32>,
}

/// What a helper runs on, reported when it says it's ready, when it loads a model and with every reply: it is per
/// request, because a request with `gpu: false` runs on the processor in a helper that also has a graphics card.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Backend {
    /// `vulkan` when a GPU was chosen, `metal` on the Mac when the model is on it, `cpu` otherwise.
    #[serde(default)]
    pub backend: String,
    /// The chosen GPU's name ("NVIDIA GeForce RTX 5070 Ti"); None on the processor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<String>,
    /// The chosen GPU's memory in MB (0 on the processor).
    #[serde(default)]
    pub vram_mb: u64,
}

impl Backend {
    pub fn cpu() -> Self {
        Backend {
            backend: "cpu".into(),
            device: None,
            vram_mb: 0,
        }
    }

    /// The helper runs on a graphics card: Vulkan (with its device named) or Metal on the Mac (the AI helper says
    /// `metal` and names no device). An old helper that says nothing counts as the processor.
    pub fn on_gpu(&self) -> bool {
        !self.backend.is_empty() && self.backend != "cpu"
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event", rename_all = "snake_case")]
pub enum Event {
    Ready {
        #[serde(flatten)]
        backend: Backend,
        #[serde(default)]
        protocol: u32,
    },
    /// Reading the prompt (`done` of `total` tokens), then writing (`done` tokens written, `total` 0).
    Progress {
        phase: Phase,
        done: u32,
        total: u32,
    },
    Done(Reply),
    /// One vector per input text, in order.
    Embedded {
        vectors: Vec<Vec<f32>>,
        /// Where the vectors were made, as in `Reply::backend`.
        #[serde(flatten, default)]
        backend: Backend,
    },
    /// Every device ggml sees, and the index of the one the helper uses (None: the processor).
    Devices {
        devices: Vec<Device>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen: Option<usize>,
    },
    /// The idle setting now in force.
    Configured {
        #[serde(default)]
        idle_secs: Option<u64>,
    },
    /// The model's memory is freed (or there was none loaded).
    Unloaded,
    /// A model is loaded and ready. `seconds` is 0 when it was already loaded.
    Loaded {
        seconds: f32,
        /// The longest context the model was trained for.
        n_ctx_train: u32,
        #[serde(flatten)]
        backend: Backend,
    },
    Inspected(Inspection),
    Error {
        message: String,
        /// What went wrong, for the app to act on; the message is for logs. An error from a helper before
        /// protocol 2 reads as `Internal`.
        #[serde(default)]
        error: ErrorKind,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Loading,
    Reading,
    Writing,
}

/// What a GGUF file says about itself.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Inspection {
    pub has_chat_template: bool,
    /// Its chat template has a thinking step (`<think>`).
    pub thinks: bool,
    /// The longest context it was trained for; 0 when the file doesn't say.
    pub n_ctx_train: u32,
    pub architecture: String,
}

/// Why a request failed. Apps match on this, never on the message text.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ErrorKind {
    ModelMissing {
        path: String,
    },
    OutOfMemory,
    PromptTooLong {
        prompt_tokens: u32,
        n_ctx: u32,
        max_tokens: u32,
    },
    BadSchema {
        detail: String,
    },
    NoChatTemplate,
    Cancelled,
    /// The graphics card failed. `fell_back`: the work was done on the processor instead (the app can say so once
    /// and stop trying the card); false: nothing was done.
    GpuFailed {
        fell_back: bool,
    },
    BadRequest,
    #[default]
    Internal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reply {
    pub text: String,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub seconds: f32,
    /// True when the reply stopped at `max_tokens` instead of finishing.
    pub truncated: bool,
    /// Tokens spent thinking (not counted in `output_tokens`).
    #[serde(default)]
    pub thinking_tokens: u32,
    /// Seconds spent loading the model for this request; 0 when it was already loaded.
    #[serde(default)]
    pub load_seconds: f32,
    /// The thinking, when thinking was on.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_text: Option<String>,
    /// Where this reply was made: `cpu` when the model is on the processor (a request with `gpu: false` always is),
    /// `metal` or `vulkan` (with the device) when it is on the graphics card. An older helper says nothing (an empty
    /// `backend`), and the app uses what `ready` said.
    #[serde(flatten, default)]
    pub backend: Backend,
}

/// One message as one line (no newline inside: JSON escapes the ones in the text).
pub fn encode<T: Serialize>(msg: &T) -> String {
    serde_json::to_string(msg).expect("protocol messages always serialize")
}

/// Reads one request line. A UTF-8 byte order mark (a file piped in by hand) isn't part of the request.
pub fn decode_request(line: &str) -> Result<Request, serde_json::Error> {
    serde_json::from_str(line.trim_start_matches('\u{feff}').trim())
}

/// Reads one event line.
pub fn decode_event(line: &str) -> Result<Event, serde_json::Error> {
    serde_json::from_str(line.trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn completion() -> Completion {
        Completion {
            model_path: "/m.gguf".into(),
            n_ctx: 8192,
            system: "s".into(),
            user: "line one\nline two".into(),
            json_schema: Some("{}".into()),
            max_tokens: 100,
            temperature: 0.2,
            gpu: true,
            thinking: false,
            thinking_budget: None,
            seed: None,
        }
    }

    #[test]
    fn requests_and_events_round_trip_as_single_lines() {
        let req = Request::Complete(completion());
        let line = encode(&req);
        assert!(!line.contains('\n'));
        assert!(line.contains("\"op\":\"complete\""));
        assert_eq!(serde_json::from_str::<Request>(&line).unwrap(), req);
        let ev = Event::Progress {
            phase: Phase::Reading,
            done: 5,
            total: 10,
        };
        assert_eq!(serde_json::from_str::<Event>(&encode(&ev)).unwrap(), ev);
        assert_eq!(encode(&Request::Ping), r#"{"op":"ping"}"#);
        assert_eq!(encode(&Request::Devices), r#"{"op":"devices"}"#);
        assert_eq!(
            serde_json::from_str::<Request>(r#"{"op":"devices"}"#).unwrap(),
            Request::Devices
        );
    }

    #[test]
    fn requests_from_an_older_app_use_the_gpu() {
        let old = r#"{"op":"complete","model_path":"/m.gguf","n_ctx":512,"system":"","user":"","max_tokens":1,"temperature":0.0}"#;
        let Request::Complete(c) = serde_json::from_str::<Request>(old).unwrap() else {
            panic!()
        };
        assert!(c.gpu);
        let Request::Embed(e) = serde_json::from_str::<Request>(
            r#"{"op":"embed","model_path":"/e.gguf","texts":["a"]}"#,
        )
        .unwrap() else {
            panic!()
        };
        assert!(e.gpu);
        let off = encode(&Request::Complete(Completion {
            gpu: false,
            ..completion()
        }));
        assert!(off.contains("\"gpu\":false"), "{off}");
    }

    #[test]
    fn ready_carries_the_backend_and_an_old_ready_still_reads() {
        let gpu = Event::Ready {
            backend: Backend {
                backend: "vulkan".into(),
                device: Some("NVIDIA GeForce RTX 5070 Ti".into()),
                vram_mb: 16303,
            },
            protocol: PROTOCOL_VERSION,
        };
        let line = encode(&gpu);
        assert_eq!(
            line,
            r#"{"event":"ready","backend":"vulkan","device":"NVIDIA GeForce RTX 5070 Ti","vram_mb":16303,"protocol":2}"#
        );
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), gpu);
        // The fake helpers in tests (and a 1.0.x helper) answer with just this.
        let Event::Ready { backend, protocol } =
            serde_json::from_str::<Event>(r#"{"event":"ready"}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(backend, Backend::default());
        assert_eq!(protocol, 0);
        assert!(!backend.on_gpu());
        assert!(!Backend::cpu().on_gpu());
        // Metal is a graphics card although the AI helper names no device for it.
        let metal = Backend {
            backend: "metal".into(),
            device: None,
            vram_mb: 0,
        };
        assert!(metal.on_gpu());
        assert_eq!(
            encode(&Event::Ready {
                backend: Backend::cpu(),
                protocol: PROTOCOL_VERSION
            }),
            r#"{"event":"ready","backend":"cpu","vram_mb":0,"protocol":2}"#
        );
    }

    #[test]
    fn a_reply_and_embedded_say_where_they_ran_and_an_old_one_says_nothing() {
        let old: Event = serde_json::from_str(
            r#"{"event":"done","text":"x","prompt_tokens":1,"output_tokens":1,"seconds":0.1,"truncated":false}"#,
        )
        .unwrap();
        let Event::Done(reply) = old else { panic!() };
        assert_eq!(reply.backend, Backend::default());
        let old: Event = serde_json::from_str(r#"{"event":"embedded","vectors":[[1.0]]}"#).unwrap();
        assert_eq!(
            old,
            Event::Embedded {
                vectors: vec![vec![1.0]],
                backend: Backend::default()
            }
        );
        let line = encode(&Event::Embedded {
            vectors: vec![vec![1.0]],
            backend: Backend::cpu(),
        });
        assert_eq!(
            line,
            r#"{"event":"embedded","vectors":[[1.0]],"backend":"cpu","vram_mb":0}"#
        );
        let new: Event = serde_json::from_str(
            r#"{"event":"done","text":"x","prompt_tokens":1,"output_tokens":1,"seconds":0.1,"truncated":false,"backend":"metal","vram_mb":0}"#,
        )
        .unwrap();
        let Event::Done(reply) = new else { panic!() };
        assert_eq!(reply.backend.backend, "metal");
    }

    #[test]
    fn ready_carries_the_protocol_version() {
        let line = encode(&Event::Ready {
            backend: Backend::cpu(),
            protocol: PROTOCOL_VERSION,
        });
        assert!(line.contains(r#""protocol":2"#), "{line}");
    }

    #[test]
    fn a_ready_without_a_version_reads_as_zero() {
        let ev: Event = serde_json::from_str(r#"{"event":"ready","backend":"cpu"}"#).unwrap();
        let Event::Ready { protocol, .. } = ev else {
            panic!("not ready: {ev:?}")
        };
        assert_eq!(protocol, 0);
    }

    #[test]
    fn env_names_follow_the_app_prefix() {
        let c = HelperConfig {
            env_prefix: "TALAAN",
        };
        assert_eq!(c.gpu_env(), "TALAAN_GPU");
        assert_eq!(c.idle_secs_env(), "TALAAN_LLM_IDLE_SECS");
    }

    fn ensayo_completion() -> Completion {
        Completion {
            model_path: "/m.gguf".into(),
            n_ctx: 8192,
            system: "s".into(),
            user: "line one\nline two \"quoted\" \u{2192} caf\u{e9}".into(),
            json_schema: Some(JSON_OBJECT_SCHEMA.into()),
            max_tokens: 100,
            temperature: 0.0,
            gpu: true,
            thinking: false,
            thinking_budget: None,
            seed: None,
        }
    }

    #[test]
    fn ensayo_a_request_is_one_line_and_round_trips_with_its_text_intact() {
        let req = Request::Complete(ensayo_completion());
        let line = encode(&req);
        assert!(!line.contains('\n'), "{line}");
        assert!(line.contains("\"op\":\"complete\""));
        assert_eq!(decode_request(&line).unwrap(), req);
        assert_eq!(encode(&Request::Ping), r#"{"op":"ping"}"#);
        assert_eq!(decode_request(r#"{"op":"ping"}"#).unwrap(), Request::Ping);
    }

    #[test]
    fn ensayo_every_event_round_trips_as_one_line() {
        let events = [
            Event::Ready {
                backend: Backend {
                    backend: "metal".into(),
                    ..Default::default()
                },
                protocol: PROTOCOL_VERSION,
            },
            Event::Progress {
                phase: Phase::Reading,
                done: 5,
                total: 10,
            },
            Event::Progress {
                phase: Phase::Loading,
                done: 0,
                total: 0,
            },
            Event::Done(Reply {
                text: "{\"a\":\n1}".into(),
                prompt_tokens: 12,
                output_tokens: 3,
                seconds: 0.5,
                truncated: false,
                thinking_tokens: 0,
                load_seconds: 0.0,
                thinking_text: None,
                backend: Backend::cpu(),
            }),
            Event::Error {
                message: "the AI model failed".into(),
                error: ErrorKind::Internal,
            },
        ];
        for ev in events {
            let line = encode(&ev);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_event(&line).unwrap(), ev);
        }
    }

    #[test]
    fn ensayo_a_ready_from_an_app_with_only_a_backend_name_still_reads() {
        // Ensayo's helper before the engine said just the backend name; it must keep working.
        let Event::Ready { backend, protocol } =
            decode_event(r#"{"event":"ready","backend":"cpu"}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(
            backend,
            Backend {
                backend: "cpu".into(),
                ..Default::default()
            }
        );
        assert_eq!(protocol, 0);
        let Event::Ready { backend, .. } = decode_event(r#"{"event":"ready"}"#).unwrap() else {
            panic!()
        };
        assert_eq!(backend, Backend::default());
    }

    #[test]
    fn ensayo_the_schema_is_left_out_when_there_is_none() {
        let line = encode(&Request::Complete(Completion {
            json_schema: None,
            ..ensayo_completion()
        }));
        assert!(!line.contains("json_schema"), "{line}");
        let Request::Complete(c) = decode_request(&line).unwrap() else {
            panic!()
        };
        assert_eq!(c.json_schema, None);
    }

    #[test]
    fn ensayo_a_byte_order_mark_and_blank_edges_are_ignored_and_garbage_is_an_error() {
        assert_eq!(
            decode_request("\u{feff}{\"op\":\"ping\"}  ").unwrap(),
            Request::Ping
        );
        assert!(decode_request("not json").is_err());
        assert!(decode_request(r#"{"op":"unknown"}"#).is_err());
        assert!(decode_event("").is_err());
    }

    #[test]
    fn devices_round_trip() {
        let ev = Event::Devices {
            devices: vec![Device {
                index: 0,
                name: "Vulkan0".into(),
                description: "AMD Radeon(TM) Graphics".into(),
                backend: "Vulkan".into(),
                kind: DeviceKind::Igpu,
                memory_mb: 512,
            }],
            chosen: None,
        };
        let line = encode(&ev);
        assert!(line.contains("\"kind\":\"igpu\""), "{line}");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), ev);
    }

    #[test]
    fn protocol_two_requests_round_trip() {
        let reqs = [
            Request::Configure(Configure {
                idle_secs: Some(90),
            }),
            Request::Configure(Configure { idle_secs: None }),
            Request::Load(Load {
                model_path: "/m.gguf".into(),
                n_ctx: 4096,
                gpu: true,
            }),
            Request::Cancel,
            Request::Unload,
            Request::Inspect(Inspect {
                model_path: "/m.gguf".into(),
            }),
        ];
        for req in reqs {
            let line = encode(&req);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_request(&line).unwrap(), req, "{line}");
        }
        assert_eq!(encode(&Request::Cancel), r#"{"op":"cancel"}"#);
        assert_eq!(encode(&Request::Unload), r#"{"op":"unload"}"#);
        // A configure without idle_secs means "never exit".
        assert_eq!(
            decode_request(r#"{"op":"configure"}"#).unwrap(),
            Request::Configure(Configure { idle_secs: None })
        );
    }

    #[test]
    fn completion_thinking_and_seed_default_off() {
        let old = r#"{"op":"complete","model_path":"/m.gguf","n_ctx":512,"system":"","user":"","max_tokens":1,"temperature":0.0}"#;
        let Request::Complete(c) = decode_request(old).unwrap() else {
            panic!()
        };
        assert!(!c.thinking);
        assert_eq!(c.thinking_budget, None);
        assert_eq!(c.seed, None);
        let line = encode(&Request::Complete(c));
        assert!(
            !line.contains("thinking_budget") && !line.contains("seed"),
            "{line}"
        );
    }

    #[test]
    fn protocol_two_events_round_trip() {
        let events = [
            Event::Configured {
                idle_secs: Some(90),
            },
            Event::Unloaded,
            Event::Loaded {
                seconds: 1.5,
                n_ctx_train: 262_144,
                backend: Backend {
                    backend: "metal".into(),
                    ..Default::default()
                },
            },
            Event::Inspected(Inspection {
                has_chat_template: true,
                thinks: true,
                n_ctx_train: 262_144,
                architecture: "qwen35".into(),
            }),
            Event::Done(Reply {
                text: "ok".into(),
                prompt_tokens: 10,
                output_tokens: 2,
                seconds: 0.3,
                truncated: false,
                thinking_tokens: 40,
                load_seconds: 0.0,
                thinking_text: Some("hmm".into()),
                backend: Backend::cpu(),
            }),
        ];
        for ev in events {
            let line = encode(&ev);
            assert!(!line.contains('\n'), "{line}");
            assert_eq!(decode_event(&line).unwrap(), ev, "{line}");
        }
    }

    #[test]
    fn errors_carry_a_kind_and_an_old_error_reads_as_internal() {
        let kinds = [
            ErrorKind::ModelMissing {
                path: "/m.gguf".into(),
            },
            ErrorKind::OutOfMemory,
            ErrorKind::PromptTooLong {
                prompt_tokens: 9000,
                n_ctx: 8192,
                max_tokens: 500,
            },
            ErrorKind::BadSchema {
                detail: "bad".into(),
            },
            ErrorKind::NoChatTemplate,
            ErrorKind::Cancelled,
            ErrorKind::GpuFailed { fell_back: true },
            ErrorKind::BadRequest,
            ErrorKind::Internal,
        ];
        for error in kinds {
            let ev = Event::Error {
                message: "m".into(),
                error,
            };
            assert_eq!(decode_event(&encode(&ev)).unwrap(), ev);
        }
        let line = encode(&Event::Error {
            message: "stopped by request".into(),
            error: ErrorKind::Cancelled,
        });
        assert_eq!(
            line,
            r#"{"event":"error","message":"stopped by request","error":{"kind":"cancelled"}}"#
        );
        let Event::Error { error, .. } =
            decode_event(r#"{"event":"error","message":"old helper"}"#).unwrap()
        else {
            panic!()
        };
        assert_eq!(error, ErrorKind::Internal);
    }

    #[test]
    fn an_old_reply_reads_with_no_thinking() {
        let Event::Done(r) = decode_event(
            r#"{"event":"done","text":"x","prompt_tokens":1,"output_tokens":1,"seconds":0.1,"truncated":false}"#,
        )
        .unwrap() else {
            panic!()
        };
        assert_eq!(
            (r.thinking_tokens, r.load_seconds, r.thinking_text),
            (0, 0.0, None)
        );
    }
}
