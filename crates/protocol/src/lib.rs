//! Messages between an app and its local AI helper (a CPU build, and on Windows a Vulkan GPU build). One JSON
//! object per line on stdin (requests) and stdout (events). The helper runs one request at a time.
//!
//! Also here, because both GPU helpers need them: the Whisper helper's protocol (`stt`) and the device policy
//! (`gpu`), which are pure so they can be tested without a GPU.

use serde::{Deserialize, Serialize};

pub mod gpu;
pub mod stt;

pub use gpu::{Device, DeviceKind};

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

pub fn encode<T: Serialize>(msg: &T) -> String {
    serde_json::to_string(msg).expect("protocol messages always serialize")
}

/// Environment variable that makes a GPU build fail on purpose (`TALAAN_GPU=fail`), to test the fallback on a real
/// PC. CPU builds ignore it.
pub const FAIL_ENV: &str = "TALAAN_GPU";

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
        let gpu = Event::Ready { backend: Backend { backend: "vulkan".into(), device: Some("NVIDIA GeForce RTX 5070 Ti".into()), vram_mb: 16303 } };
        let line = encode(&gpu);
        assert_eq!(line, r#"{"event":"ready","backend":"vulkan","device":"NVIDIA GeForce RTX 5070 Ti","vram_mb":16303}"#);
        assert_eq!(serde_json::from_str::<Event>(&line).unwrap(), gpu);
        // The fake helpers in tests (and a 1.0.x helper) answer with just this.
        let Event::Ready { backend } = serde_json::from_str::<Event>(r#"{"event":"ready"}"#).unwrap() else { panic!() };
        assert_eq!(backend, Backend::default());
        assert!(!backend.on_gpu());
        assert!(!Backend::cpu().on_gpu());
        assert_eq!(encode(&Event::Ready { backend: Backend::cpu() }), r#"{"event":"ready","backend":"cpu","vram_mb":0}"#);
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
