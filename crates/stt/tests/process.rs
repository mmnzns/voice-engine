//! The real speech helper binary, without a model (so it runs in CI): it answers a ping, exits when the app closes
//! its end, answers every `load` that cannot work with the right kind, and keeps running after a bad request.
//! With a model: `crates/client/tests/real_stt.rs` (ignored).

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};
use voice_engine_protocol::stt::{SttErrorKind, SttEvent, SttRequest, decode_event};
use voice_engine_protocol::{PROTOCOL_VERSION, encode};

/// The tests take turns. macOS compiles Metal shaders through a system cache file; several helpers starting at once
/// can crash one of them (SIGBUS inside Apple's MTLCompilerFSCache, found in step 2). The client restarts a helper
/// that dies at startup; these tests just avoid the clash.
static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner())
}

struct Helper {
    child: Child,
    stdin: Option<ChildStdin>,
    out: BufReader<ChildStdout>,
}

impl Helper {
    fn start() -> Helper {
        let mut child = Command::new(env!("CARGO_BIN_EXE_voice-engine-stt"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let out = BufReader::new(child.stdout.take().unwrap());
        Helper { child, stdin, out }
    }

    fn say(&mut self, line: &str) {
        let stdin = self.stdin.as_mut().unwrap();
        writeln!(stdin, "{line}").unwrap();
        stdin.flush().unwrap();
    }

    fn event(&mut self) -> SttEvent {
        let mut line = String::new();
        self.out.read_line(&mut line).unwrap();
        decode_event(&line).unwrap_or_else(|e| panic!("not an event ({e}): {line:?}"))
    }

    fn ask(&mut self, request: &SttRequest) -> SttEvent {
        self.say(&encode(request));
        self.event()
    }

    fn error_kind(&mut self, request: &SttRequest) -> SttErrorKind {
        match self.ask(request) {
            SttEvent::Error { error, .. } => error,
            other => panic!("expected an error, got {other:?}"),
        }
    }

    /// Closes stdin (the app is gone) and waits for the helper to exit.
    fn close_and_wait(mut self) -> Option<std::process::ExitStatus> {
        drop(self.stdin.take());
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            if let Some(s) = self.child.try_wait().unwrap() {
                return Some(s);
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        let _ = self.child.kill();
        None
    }
}

fn load(model_id: &str, dir: &str, kind: &str) -> SttRequest {
    serde_json::from_str(&format!(
        r#"{{"op":"load","model_id":"{model_id}","model_dir":{},"kind":"{kind}","gpu":false}}"#,
        serde_json::to_string(dir).unwrap()
    ))
    .unwrap()
}

#[test]
fn it_answers_a_ping_with_its_protocol_and_exits_when_stdin_closes() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    let SttEvent::Ready { backend, protocol } = h.ask(&SttRequest::Ping) else {
        panic!("not ready")
    };
    assert_eq!(protocol, PROTOCOL_VERSION);
    // The processor, or the discrete card the helper chose (a Vulkan helper names it); never nothing.
    assert!(!backend.backend.is_empty(), "{backend:?}");
    assert_eq!(backend.on_gpu(), backend.device.is_some());
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn a_model_that_is_not_there_says_model_missing_with_the_path() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    let dir = tempfile::tempdir().unwrap();
    let kind = h.error_kind(&load(
        "whisper-turbo",
        dir.path().to_str().unwrap(),
        "whisper",
    ));
    let SttErrorKind::ModelMissing { path } = kind else {
        panic!("{kind:?}")
    };
    // The catalog's file name inside the folder the app gave.
    assert!(
        path.starts_with(dir.path().to_str().unwrap()) && path.ends_with(".bin"),
        "{path}"
    );
    let kind = h.error_kind(&load(
        "whisper-turbo",
        "/nonexistent/whisper-turbo",
        "whisper",
    ));
    assert!(
        matches!(kind, SttErrorKind::ModelMissing { .. }),
        "{kind:?}"
    );
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn a_file_that_is_not_a_whisper_model_says_bad_model() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    let dir = tempfile::tempdir().unwrap();
    let model = voice_engine_models::catalog::find("whisper-base").unwrap();
    std::fs::write(
        dir.path().join(model.files[0].name),
        b"this is not a ggml file, only text standing in for one",
    )
    .unwrap();
    let kind = h.error_kind(&load(
        "whisper-base",
        dir.path().to_str().unwrap(),
        "whisper",
    ));
    assert_eq!(kind, SttErrorKind::BadModel);
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn the_onnx_models_are_not_served_here() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    for (id, kind) in [
        ("parakeet-ultra", "parakeet"),
        ("parakeet-unified-en", "parakeet_unified"),
        ("nemotron-3.5", "nemotron"),
    ] {
        let k = h.error_kind(&load(id, "/models/x", kind));
        assert!(
            matches!(&k, SttErrorKind::Unsupported { detail } if detail.contains("Whisper")),
            "{id}: {k:?}"
        );
    }
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn an_unknown_model_or_one_of_another_kind_is_a_bad_request() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    assert_eq!(
        h.error_kind(&load("no-such-model", "/m", "whisper")),
        SttErrorKind::BadRequest
    );
    // A Parakeet id sent as Whisper.
    assert_eq!(
        h.error_kind(&load("parakeet-ultra", "/m", "whisper")),
        SttErrorKind::BadRequest
    );
    // The voice detector is not a speech model.
    assert_eq!(
        h.error_kind(&load("vad", "/m", "whisper")),
        SttErrorKind::BadRequest
    );
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn transcribe_with_nothing_loaded_is_a_bad_request_and_unload_is_harmless() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    let request = SttRequest::Transcribe {
        audio_b64: voice_engine_protocol::stt::encode_samples(&[0.0; 1600]),
        options: Default::default(),
    };
    assert_eq!(h.error_kind(&request), SttErrorKind::BadRequest);
    assert_eq!(h.ask(&SttRequest::Unload), SttEvent::Unloaded);
    assert_eq!(h.error_kind(&request), SttErrorKind::BadRequest);
    assert!(matches!(h.ask(&SttRequest::Ping), SttEvent::Ready { .. }));
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn a_bad_line_is_an_error_event_not_a_crash_and_a_cancel_says_nothing() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    h.say("this is not json");
    let SttEvent::Error { error, message } = h.event() else {
        panic!()
    };
    assert_eq!(error, SttErrorKind::BadRequest);
    assert!(message.contains("bad request"), "{message}");
    // The v0.1.0 messages are not read.
    h.say(r#"{"op":"load","model_path":"m"}"#);
    assert!(matches!(
        h.event(),
        SttEvent::Error {
            error: SttErrorKind::BadRequest,
            ..
        }
    ));
    // A cancel with nothing running produces no event: the next answer is the ping's.
    h.say(&encode(&SttRequest::Cancel));
    assert!(matches!(
        h.ask(&SttRequest::Ping),
        SttEvent::Ready {
            protocol: PROTOCOL_VERSION,
            ..
        }
    ));
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}

#[test]
fn devices_lists_the_processor() {
    let _turn = one_at_a_time();
    let mut h = Helper::start();
    let SttEvent::Devices { devices, .. } = h.ask(&SttRequest::Devices) else {
        panic!()
    };
    assert!(devices.iter().any(|d| d.backend == "CPU"), "{devices:?}");
    assert!(h.close_and_wait().is_some_and(|s| s.success()));
}
