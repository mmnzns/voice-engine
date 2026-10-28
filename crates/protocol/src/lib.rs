//! Messages between an app and its local AI helper (a CPU build, and on Windows a Vulkan GPU build). One JSON
//! object per line on stdin (requests) and stdout (events). The helper runs one request at a time.
//!
//! Also here, because both GPU helpers need them: the Whisper helper's protocol (`stt`) and the device policy
//! (`gpu`), which are pure so they can be tested without a GPU.

use serde::{Deserialize, Serialize};

pub mod config;
pub mod gpu;
pub mod stt;

pub use config::HelperConfig;
pub use gpu::{Device, DeviceKind};

/// Bumped whenever a message changes shape. Each helper sends it in `Ready`; the app compares it with its own and
/// reports a mismatch (a helper and an app from different builds) instead of failing later. 0 = a helper from before
/// the engine existed.
pub const PROTOCOL_VERSION: u32 = 1;

/// A JSON object with any keys: the grammar that keeps a reply parseable when the caller has no schema of its own.
pub const JSON_OBJECT_SCHEMA: &str = r#"{"type":"object"}"#;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Request {
    /// Answers with `Event::Ready`. Used to check the helper started.
    Ping,
    /// Generates one reply. The model stays loaded for the next request with the same path and context size.
    Complete(Completion),
    /// Turns texts into embedding vectors (L2-normalized), for Ask's search index.
    Embed(Embedding),
    /// Answers with `Event::Devices`: every ggml device and the one the helper chose. For checks by hand.
    Devices,
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
}

/// What a helper runs on, reported when it says it's ready.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Backend {
    /// `vulkan` when a GPU was chosen, `metal` on the Mac, `cpu` otherwise.
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
        Backend { backend: "cpu".into(), device: None, vram_mb: 0 }
    }

    /// The helper chose a GPU (an old helper that says nothing counts as the processor).
    pub fn on_gpu(&self) -> bool {
        self.device.is_some() && !self.backend.is_empty() && self.backend != "cpu"
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
    Progress { phase: Phase, done: u32, total: u32 },
    Done(Reply),
    /// One vector per input text, in order.
    Embedded { vectors: Vec<Vec<f32>> },
    /// Every device ggml sees, and the index of the one the helper uses (None: the processor).
    Devices {
        devices: Vec<Device>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        chosen: Option<usize>,
    },
    Error { message: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Loading,
    Reading,
    Writing,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Reply {
    pub text: String,
    pub prompt_tokens: u32,
    pub output_tokens: u32,
    pub seconds: f32,
    /// True when the reply stopped at `max_tokens` instead of finishing.
    pub truncated: bool,
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
        }
    }

    #[test]
    fn requests_and_events_round_trip_as_single_lines() {
        let req = Request::Complete(completion());
        let line = encode(&req);
        assert!(!line.contains('\n'));
        assert!(line.contains("\"op\":\"complete\""));
        assert_eq!(serde_json::from_str::<Request>(&line).unwrap(), req);
        let ev = Event::Progress { phase: Phase::Reading, done: 5, total: 10 };
        assert_eq!(serde_json::from_str::<Event>(&encode(&ev)).unwrap(), ev);
        assert_eq!(encode(&Request::Ping), r#"{"op":"ping"}"#);
        assert_eq!(encode(&Request::Devices), r#"{"op":"devices"}"#);
        assert_eq!(serde_json::from_str::<Request>(r#"{"op":"devices"}"#).unwrap(), Request::Devices);
    }

    #[test]
    fn requests_from_an_older_app_use_the_gpu() {
        let old = r#"{"op":"complete","model_path":"/m.gguf","n_ctx":512,"system":"","user":"","max_tokens":1,"temperature":0.0}"#;
        let Request::Complete(c) = serde_json::from_str::<Request>(old).unwrap() else { panic!() };
        assert!(c.gpu);
        let Request::Embed(e) = serde_json::from_str::<Request>(r#"{"op":"embed","model_path":"/e.gguf","texts":["a"]}"#).unwrap() else { panic!() };
        assert!(e.gpu);
        let off = encode(&Request::Complete(Completion { gpu: false, ..completion() }));
        assert!(off.contains("\"gpu\":false"), "{off}");
    }

    #[test]
    fn ready_carries_the_backend_and_an_old_ready_still_reads() {
        let gpu = Event::Ready {
            backend: Backend { backend: "vulkan".into(), device: Some("NVIDIA GeForce RTX 5070 Ti".into()), vram_mb: 16303 },
            protocol: PROTOCOL_VERSION,
        };
        let line = encode(&gpu);
        assert_eq!(line, r#"{"event":"ready","backend":"vulkan","device":"NVIDIA GeForce RTX 5070 Ti","vram_mb":16303,"protocol":1}"#);
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), gpu);
        // The fake helpers in tests (and a 1.0.x helper) answer with just this.
        let Event::Ready { backend, protocol } = serde_json::from_str::<Event>(r#"{"event":"ready"}"#).unwrap() else { panic!() };
        assert_eq!(backend, Backend::default());
        assert_eq!(protocol, 0);
        assert!(!backend.on_gpu());
        assert!(!Backend::cpu().on_gpu());
        assert_eq!(
            encode(&Event::Ready { backend: Backend::cpu(), protocol: PROTOCOL_VERSION }),
            r#"{"event":"ready","backend":"cpu","vram_mb":0,"protocol":1}"#
        );
    }

    #[test]
    fn ready_carries_the_protocol_version() {
        let line = encode(&Event::Ready { backend: Backend::cpu(), protocol: PROTOCOL_VERSION });
        assert!(line.contains(r#""protocol":1"#), "{line}");
    }

    #[test]
    fn a_ready_without_a_version_reads_as_zero() {
        let ev: Event = serde_json::from_str(r#"{"event":"ready","backend":"cpu"}"#).unwrap();
        let Event::Ready { protocol, .. } = ev else { panic!("not ready: {ev:?}") };
        assert_eq!(protocol, 0);
    }

    #[test]
    fn env_names_follow_the_app_prefix() {
        let c = HelperConfig { env_prefix: "TALAAN" };
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
            Event::Ready { backend: Backend { backend: "metal".into(), ..Default::default() }, protocol: PROTOCOL_VERSION },
            Event::Progress { phase: Phase::Reading, done: 5, total: 10 },
            Event::Progress { phase: Phase::Loading, done: 0, total: 0 },
            Event::Done(Reply { text: "{\"a\":\n1}".into(), prompt_tokens: 12, output_tokens: 3, seconds: 0.5, truncated: false }),
            Event::Error { message: "the AI model failed".into() },
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
        let Event::Ready { backend, protocol } = decode_event(r#"{"event":"ready","backend":"cpu"}"#).unwrap() else { panic!() };
        assert_eq!(backend, Backend { backend: "cpu".into(), ..Default::default() });
        assert_eq!(protocol, 0);
        let Event::Ready { backend, .. } = decode_event(r#"{"event":"ready"}"#).unwrap() else { panic!() };
        assert_eq!(backend, Backend::default());
    }

    #[test]
    fn ensayo_the_schema_is_left_out_when_there_is_none() {
        let line = encode(&Request::Complete(Completion { json_schema: None, ..ensayo_completion() }));
        assert!(!line.contains("json_schema"), "{line}");
        let Request::Complete(c) = decode_request(&line).unwrap() else { panic!() };
        assert_eq!(c.json_schema, None);
    }

    #[test]
    fn ensayo_a_byte_order_mark_and_blank_edges_are_ignored_and_garbage_is_an_error() {
        assert_eq!(decode_request("\u{feff}{\"op\":\"ping\"}  ").unwrap(), Request::Ping);
        assert!(decode_request("not json").is_err());
        assert!(decode_request(r#"{"op":"unknown"}"#).is_err());
        assert!(decode_event("").is_err());
    }

    #[test]
    fn devices_round_trip() {
        let ev = Event::Devices {
            devices: vec![Device { index: 0, name: "Vulkan0".into(), description: "AMD Radeon(TM) Graphics".into(), backend: "Vulkan".into(), kind: DeviceKind::Igpu, memory_mb: 512 }],
            chosen: None,
        };
        let line = encode(&ev);
        assert!(line.contains("\"kind\":\"igpu\""), "{line}");
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), ev);
    }
}
