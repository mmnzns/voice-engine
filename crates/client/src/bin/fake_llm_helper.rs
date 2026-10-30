//! A fake AI helper for the client's tests. `FAKE_MODE` picks a behaviour, `FAKE_STATE` is a file it keeps a counter
//! in (so a mode can fail the first N times and then work).
//!
//! Modes: (default) answers everything; `die_at_start` exits at once without a word; `die_at_start_twice` does that
//! for the first two starts; `wrong_protocol` answers `ready` with protocol 1; `die_on_complete` exits after one
//! progress event; `die_on_first_complete` exits before answering the first `complete` (an idle exit just as the
//! request arrives); `oom` writes an out-of-memory message and dies on `complete`; `stall` never answers `complete`;
//! `slow` writes progress until it gets `cancel`, then answers `cancelled`; `ignore_cancel` writes progress forever;
//! `gpu_ok` reports a Vulkan GPU and works; `gpu_die` reports a GPU and dies on `complete`; `gpu_error` reports a GPU
//! and answers `complete` with the graphics card failed; `per_request` is the Mac helper: `ready` says `metal`, and
//! `loaded`, `done` and `embedded` say `metal` for a request with `gpu: true` and `cpu` for `gpu: false`. Every other
//! mode answers like a helper from before replies said where they ran (no backend), so the client uses `ready`'s.
//! `metal_die` is the Mac helper asked for as the card helper: `ready` says `metal` with no device, and it dies on
//! `complete` like `gpu_die`.
//!
//! When `FAKE_LOG` names a file, every `load`, `complete` and `embed` the helper receives is appended to it as one
//! line, `[<mode>] <op> gpu=<flag>`, so a test can see what each helper (the card one and the processor one run in
//! different modes) was asked.

use std::io::{BufRead, Write};
use std::sync::mpsc;
use std::time::Duration;
use voice_engine_protocol::{
    Backend, ErrorKind, Event, Inspection, PROTOCOL_VERSION, Phase, Reply, Request, decode_request,
    encode,
};

fn send(ev: &Event) {
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

/// Appends `[<mode>] <op> gpu=<flag>` to the file `FAKE_LOG` names, if it names one.
fn log(mode: &str, op: &str, gpu: bool) {
    let Ok(path) = std::env::var("FAKE_LOG") else {
        return;
    };
    if let Ok(mut file) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        let _ = writeln!(file, "[{mode}] {op} gpu={gpu}");
    }
}

fn main() {
    let mode = std::env::var("FAKE_MODE").unwrap_or_default();
    let state = std::env::var("FAKE_STATE").unwrap_or_default();
    if mode == "die_at_start" || (mode == "die_at_start_twice" && bump(&state) < 2) {
        std::process::exit(138);
    }
    let metal = Backend {
        backend: "metal".into(),
        device: None,
        vram_mb: 0,
    };
    // What a reply says for a request: the Mac helper's answer, or nothing (an older helper).
    let says = |gpu: bool| match mode.as_str() {
        "per_request" if gpu => metal.clone(),
        "per_request" => Backend::cpu(),
        _ => Backend::default(),
    };
    let backend = if mode == "per_request" || mode == "metal_die" {
        metal.clone()
    } else if mode.starts_with("gpu") {
        Backend {
            backend: "vulkan".into(),
            device: Some("Fake GPU".into()),
            vram_mb: 8000,
        }
    } else {
        Backend::cpu()
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

    while let Ok(line) = rx.recv() {
        let Ok(request) = decode_request(&line) else {
            continue;
        };
        match request {
            Request::Ping => send(&Event::Ready {
                backend: backend.clone(),
                protocol,
            }),
            Request::Configure(c) => send(&Event::Configured {
                idle_secs: c.idle_secs,
            }),
            Request::Load(l) => {
                log(&mode, "load", l.gpu);
                send(&Event::Loaded {
                    seconds: 0.5,
                    n_ctx_train: 4096,
                    backend: if mode == "per_request" {
                        says(l.gpu)
                    } else {
                        backend.clone()
                    },
                });
            }
            Request::Unload => send(&Event::Unloaded),
            Request::Inspect(_) => send(&Event::Inspected(Inspection {
                has_chat_template: true,
                thinks: false,
                n_ctx_train: 4096,
                architecture: "fake".into(),
            })),
            Request::Embed(e) => {
                log(&mode, "embed", e.gpu);
                send(&Event::Embedded {
                    vectors: vec![vec![1.0]],
                    backend: says(e.gpu),
                });
            }
            Request::Cancel | Request::Devices => {}
            Request::Complete(c) => {
                log(&mode, "complete", c.gpu);
                match mode.as_str() {
                    "die_on_complete" | "gpu_die" | "metal_die" => {
                        send(&Event::Progress {
                            phase: Phase::Loading,
                            done: 0,
                            total: 0,
                        });
                        std::process::exit(139);
                    }
                    "die_on_first_complete" => {
                        if bump(&state) == 0 {
                            std::process::exit(139);
                        }
                        answer(says(c.gpu));
                    }
                    "oom" => {
                        eprintln!("GGML_ASSERT: out of memory");
                        std::process::exit(137);
                    }
                    "stall" => loop {
                        std::thread::sleep(Duration::from_secs(60));
                    },
                    "slow" => loop {
                        send(&Event::Progress {
                            phase: Phase::Writing,
                            done: 1,
                            total: 0,
                        });
                        match rx.recv_timeout(Duration::from_millis(50)) {
                            Ok(l) if matches!(decode_request(&l), Ok(Request::Cancel)) => {
                                send(&Event::Error {
                                    message: "stopped by request".into(),
                                    error: ErrorKind::Cancelled,
                                });
                                break;
                            }
                            _ => {}
                        }
                    },
                    "ignore_cancel" => loop {
                        send(&Event::Progress {
                            phase: Phase::Writing,
                            done: 1,
                            total: 0,
                        });
                        std::thread::sleep(Duration::from_millis(50));
                    },
                    "gpu_error" => send(&Event::Error {
                        message: "the graphics card failed".into(),
                        error: ErrorKind::GpuFailed { fell_back: false },
                    }),
                    _ => answer(says(c.gpu)),
                }
            }
        }
    }
}

fn answer(backend: Backend) {
    send(&Event::Progress {
        phase: Phase::Writing,
        done: 1,
        total: 0,
    });
    send(&Event::Done(Reply {
        text: "hi".into(),
        prompt_tokens: 3,
        output_tokens: 1,
        seconds: 0.1,
        truncated: false,
        thinking_tokens: 0,
        load_seconds: 0.0,
        thinking_text: None,
        backend,
    }));
}
