//! Messages between an app and its Whisper GPU helper (Windows). Same framing as the summary
//! helper: one JSON object per line, one request at a time. Audio goes as base64 of little-endian f32 samples
//! (16 kHz mono), about 2.5 MB of text for a 30 s chunk.

use super::Backend;
use base64::Engine as _;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum SttRequest {
    /// Answers with `SttEvent::Ready`.
    Ping,
    /// Answers with `SttEvent::Devices` (the summary helper's `Event::Devices` shape).
    Devices,
    /// Loads a Whisper model (replacing any other). Answers with `Loaded`.
    Load {
        model_path: String,
        #[serde(default = "super::yes")]
        gpu: bool,
    },
    /// Transcribes one chunk with the loaded model. Answers with `Transcribed`.
    Transcribe {
        /// `encode_samples` of the chunk.
        samples: String,
        /// ISO 639-1 code, or None to detect.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SttPiece {
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
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
    /// The model is loaded, on this backend.
    Loaded {
        #[serde(flatten)]
        backend: Backend,
    },
    Transcribed {
        pieces: Vec<SttPiece>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        language: Option<String>,
    },
    Error {
        message: String,
    },
}

pub fn encode_samples(samples: &[f32]) -> String {
    let mut bytes = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

pub fn decode_samples(text: &str) -> Result<Vec<f32>, String> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|e| format!("bad audio: {e}"))?;
    if bytes.len() % 4 != 0 {
        return Err("bad audio: not a whole number of samples".into());
    }
    let (samples, _) = bytes.as_chunks::<4>();
    Ok(samples.iter().map(|b| f32::from_le_bytes(*b)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode;

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
    fn stt_ready_carries_the_protocol_version() {
        let line = crate::encode(&SttEvent::Ready {
            backend: Backend::default(),
            protocol: crate::PROTOCOL_VERSION,
        });
        assert!(line.contains(r#""protocol":1"#), "{line}");
        let old: SttEvent = serde_json::from_str(r#"{"event":"ready"}"#).unwrap();
        assert_eq!(
            old,
            SttEvent::Ready {
                backend: Backend::default(),
                protocol: 0
            }
        );
    }

    #[test]
    fn requests_and_events_round_trip() {
        let reqs = [
            SttRequest::Ping,
            SttRequest::Devices,
            SttRequest::Load {
                model_path: "C:\\m\\ggml-large-v3-turbo.bin".into(),
                gpu: true,
            },
            SttRequest::Transcribe {
                samples: encode_samples(&[0.5, -0.25]),
                language: Some("fr".into()),
            },
            SttRequest::Transcribe {
                samples: String::new(),
                language: None,
            },
        ];
        for r in reqs {
            let line = encode(&r);
            assert!(!line.contains('\n'));
            assert_eq!(serde_json::from_str::<SttRequest>(&line).unwrap(), r);
        }
        assert_eq!(encode(&SttRequest::Ping), r#"{"op":"ping"}"#);
        let SttRequest::Load { gpu, .. } =
            serde_json::from_str(r#"{"op":"load","model_path":"m"}"#).unwrap()
        else {
            panic!()
        };
        assert!(gpu);
        let events = [
            SttEvent::Ready {
                backend: Backend {
                    backend: "vulkan".into(),
                    device: Some("RTX".into()),
                    vram_mb: 16000,
                },
                protocol: crate::PROTOCOL_VERSION,
            },
            SttEvent::Loaded {
                backend: Backend::cpu(),
            },
            SttEvent::Transcribed {
                pieces: vec![SttPiece {
                    start_ms: 0,
                    end_ms: 1200,
                    text: "Bonjour.".into(),
                }],
                language: Some("fr".into()),
            },
            SttEvent::Error {
                message: "transcription failed".into(),
            },
            SttEvent::Devices {
                devices: vec![],
                chosen: None,
            },
        ];
        for e in events {
            assert_eq!(serde_json::from_str::<SttEvent>(&encode(&e)).unwrap(), e);
        }
        // A fake helper's bare ready.
        assert_eq!(
            serde_json::from_str::<SttEvent>(r#"{"event":"ready"}"#).unwrap(),
            SttEvent::Ready {
                backend: Backend::default(),
                protocol: 0
            }
        );
    }
}
