//! A fake speech helper for the client's tests. `FAKE_MODE` picks a behaviour, `FAKE_STATE` is a file it keeps a
//! counter in (so a mode can fail the first N times and then work). It keeps a loaded model like the real one: a new
//! process has none, `unload` drops it, and `transcribe` without one is a `bad_request`. Its transcript echoes what
//! it received (see `answer`), so a test can tell which model was loaded and that the audio and options arrived.
//!
//! Modes:
//! - (default) answers everything.
//! - `die_at_start` exits at once without a word; `die_at_start_twice` does that for the first two starts;
//!   `wrong_protocol` answers `ready` with protocol 1.
//! - `die_on_transcribe` exits before answering a `transcribe`; `die_on_first_transcribe` does that once (the first,
//!   over all starts); `exit_after_first_answer` exits right after answering the first `transcribe` (a helper that
//!   died while idle, which the client finds out at the next call) and counts its starts in `FAKE_STATE`;
//!   `oom` writes an out-of-memory message and dies on `transcribe`.
//! - `stall` never answers a `transcribe` and never reads `cancel` (a decode that cannot be interrupted, like the real
//!   helper); `stall_once` does that for the first one only (over all starts); `cancellable` waits for `cancel` and
//!   answers `cancelled`; `chatter` writes events that are not the answer, forever, and ignores `cancel`.
//! - `gpu_ok` reports a Vulkan card and works; `gpu_die` reports a card and dies on `transcribe`; `gpu_die_once`
//!   does that for the first `transcribe` (over all starts) and works after; `gpu_error` reports a
//!   card and answers `transcribe` with the graphics card failed; `cpu_fallback` runs the model on the processor
//!   though the card was asked for (`fell_back` in `loaded` and `transcribed`).
//! - `gpu_oom` reports a card and dies of memory on `transcribe`; `metal_refused` is a Mac helper whose card did not
//!   take the model: `ready` says `metal`, and `loaded` and `transcribed` say `cpu` with `fell_back`.
//! - `too_long` answers `transcribe` with too long (max 40 s); `missing` answers `load` with model missing;
//!   `slow_load` takes 4 s to load, far past the tests' 600 ms stall limit (the client stops it at the limit).
//!
//! - `die_noisy` writes what `FAKE_STDERR` (lines separated by `~`) and `FAKE_STDERR_LONG` (one line of that many
//!   letters) say to stderr and exits with `FAKE_EXIT` on `transcribe`; `sigkill` ends itself with SIGKILL there.
//!
//! `devices` answers with the list `FAKE_DEVICES` describes (fake_support/devices.rs); mode `no_devices` never answers it.
//!
//! When `FAKE_LOG` names a file, every `load` the helper receives is appended to it as one line, `[<mode>] load
//! gpu=<flag>`, so a test can see what each helper (the card one and the processor one run in different modes) was
//! asked.

#[path = "fake_support/crash.rs"]
mod fake_crash;
#[path = "fake_support/devices.rs"]
mod fake_devices;

use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::time::Duration;
use voice_engine_protocol::stt::{
    ChunkOptions, Segment, SpeechBackend, SttErrorKind, SttEvent, SttLoad, SttRequest, Timing,
    Transcript, Word, decode_request, decode_samples,
};
use voice_engine_protocol::{Backend, PROTOCOL_VERSION, encode};

fn send(ev: &SttEvent) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{}", encode(ev));
    let _ = out.flush();
}

/// Reads the counter in `path`, writes it plus one, and returns what it was.
fn bump(path: &str) -> u32 {
    let n = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0);
    let _ = std::fs::write(path, (n + 1).to_string());
    n
}

/// Appends `[<mode>] load gpu=<flag>` to the file `FAKE_LOG` names, if it names one.
fn log_load(mode: &str, gpu: bool) {
    let Ok(path) = std::env::var("FAKE_LOG") else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "[{mode}] load gpu={gpu}");
    }
}

fn error(kind: SttErrorKind) {
    send(&SttEvent::Error {
        message: "scripted".into(),
        error: kind,
    });
}

fn main() {
    let mode = std::env::var("FAKE_MODE").unwrap_or_default();
    let state = std::env::var("FAKE_STATE").unwrap_or_default();
    if mode == "die_at_start" || (mode == "die_at_start_twice" && bump(&state) < 2) {
        std::process::exit(138);
    }
    if mode == "exit_after_first_answer" {
        // Counts the starts, so a test can tell whether a helper was started.
        bump(&state);
    }
    let gpu = mode.starts_with("gpu");
    let ready = if gpu {
        Backend {
            backend: "vulkan".into(),
            device: Some("Fake GPU".into()),
            vram_mb: 8000,
        }
    } else if mode == "metal_refused" {
        Backend {
            backend: "metal".into(),
            device: None,
            vram_mb: 0,
        }
    } else {
        Backend::cpu()
    };
    let ran_on = if gpu {
        SpeechBackend {
            backend: "vulkan".into(),
            device: Some("Fake GPU".into()),
            fell_back: false,
        }
    } else {
        SpeechBackend {
            backend: "cpu".into(),
            device: None,
            fell_back: matches!(mode.as_str(), "cpu_fallback" | "metal_refused"),
        }
    };
    let protocol = if mode == "wrong_protocol" {
        1
    } else {
        PROTOCOL_VERSION
    };

    let (tx, rx) = mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });

    let mut loaded: Option<SttLoad> = None;
    let mut answered = 0u32;
    while let Ok(line) = rx.recv() {
        let Ok(request) = decode_request(&line) else {
            continue;
        };
        match request {
            SttRequest::Ping => send(&SttEvent::Ready {
                backend: ready.clone(),
                protocol,
            }),
            SttRequest::Cancel => {}
            SttRequest::Devices if mode == "no_devices" => {}
            SttRequest::Devices => {
                let list = fake_devices::list();
                send(&SttEvent::Devices {
                    devices: list.devices,
                    chosen: list.chosen,
                });
            }
            SttRequest::Load(l) => {
                loaded = None;
                log_load(&mode, l.gpu);
                if mode == "missing" {
                    error(SttErrorKind::ModelMissing {
                        path: format!("{}/model.bin", l.model_dir),
                    });
                    continue;
                }
                if mode == "slow_load" {
                    std::thread::sleep(Duration::from_secs(4));
                }
                loaded = Some(l);
                send(&SttEvent::Loaded {
                    seconds: if mode == "slow_load" { 4.0 } else { 0.5 },
                    backend: ran_on.clone(),
                });
            }
            SttRequest::Unload => {
                loaded = None;
                send(&SttEvent::Unloaded);
            }
            SttRequest::Transcribe { audio_b64, options } => {
                let Some(model) = &loaded else {
                    error(SttErrorKind::BadRequest);
                    continue;
                };
                match mode.as_str() {
                    "die_on_transcribe" | "gpu_die" => std::process::exit(139),
                    "gpu_die_once" => {
                        if bump(&state) == 0 {
                            std::process::exit(139);
                        }
                        send(&answer(model, &audio_b64, &options, &ran_on));
                    }
                    "die_on_first_transcribe" => {
                        if bump(&state) == 0 {
                            std::process::exit(139);
                        }
                        send(&answer(model, &audio_b64, &options, &ran_on));
                    }
                    "oom" | "gpu_oom" => {
                        eprintln!("GGML_ASSERT: out of memory");
                        std::process::exit(137);
                    }
                    "die_noisy" => fake_crash::die_noisy(),
                    "sigkill" => fake_crash::sigkill(),
                    "stall_once" => {
                        if bump(&state) == 0 {
                            loop {
                                std::thread::sleep(Duration::from_secs(60));
                            }
                        }
                        send(&answer(model, &audio_b64, &options, &ran_on));
                    }
                    "stall" => loop {
                        std::thread::sleep(Duration::from_secs(60));
                    },
                    "cancellable" => loop {
                        if let Ok(l) = rx.recv_timeout(Duration::from_millis(50))
                            && matches!(decode_request(&l), Ok(SttRequest::Cancel))
                        {
                            error(SttErrorKind::Cancelled);
                            break;
                        }
                    },
                    "chatter" => loop {
                        // Not the answer, and often: the stall limit never trips, only the total limit can stop it.
                        send(&SttEvent::Unloaded);
                        std::thread::sleep(Duration::from_millis(50));
                    },
                    "gpu_error" => error(SttErrorKind::GpuFailed { fell_back: false }),
                    "too_long" => error(SttErrorKind::TooLong { max_secs: 40 }),
                    _ => {
                        send(&answer(model, &audio_b64, &options, &ran_on));
                        answered += 1;
                        if mode == "exit_after_first_answer" && answered == 1 {
                            std::process::exit(0);
                        }
                    }
                }
            }
        }
    }
}

/// The transcript: one segment with two words whose text says what arrived (the timing, the vocabulary count, the
/// language hint); `language` is the loaded model's id and `stopped_early_at_ms` the number of samples received.
fn answer(
    model: &SttLoad,
    audio_b64: &str,
    options: &ChunkOptions,
    ran_on: &SpeechBackend,
) -> SttEvent {
    let samples = decode_samples(audio_b64).map(|a| a.len()).unwrap_or(0);
    let timing = match options.timing {
        Timing::Segments => "segments",
        Timing::Words => "words",
    };
    let text = format!(
        "{timing} vocab={} lang={}",
        options.vocabulary.len(),
        options.language.as_deref().unwrap_or("none")
    );
    SttEvent::Transcribed {
        transcript: Transcript {
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 900,
                text: text.clone(),
                words: text
                    .split(' ')
                    .zip([0u64, 300, 600])
                    .map(|(w, at)| Word {
                        text: w.into(),
                        start_ms: at,
                        end_ms: at + 300,
                    })
                    .collect(),
            }],
            language: Some(model.model_id.clone()),
            stopped_early_at_ms: Some(samples as u64),
        },
        backend: ran_on.clone(),
    }
}
