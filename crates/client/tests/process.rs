//! One helper process, against the fake helper. These live here (not in src/) because Cargo only sets
//! CARGO_BIN_EXE_* for integration tests.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;
use voice_engine_client::__testing::*;
use voice_engine_protocol::stt::{SttEvent, SttRequest};
use voice_engine_protocol::{Backend, Event, Request};

fn fake() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake-llm-helper"))
}

fn env(mode: &str) -> Vec<(String, String)> {
    vec![("FAKE_MODE".into(), mode.into())]
}

/// Helpers that start at the same moment can crash on a Mac (see CHANGELOG), so these tests take turns too.
static TURN: Mutex<()> = Mutex::new(());

#[test]
fn a_working_helper_answers_the_ping_with_its_backend() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let (mut p, backend) = Process::start(&fake(), &env(""), Duration::from_secs(20)).unwrap();
    assert_eq!(backend, Backend::cpu());
    p.send(&Request::Unload).unwrap();
    assert!(matches!(
        p.recv(Duration::from_secs(5)),
        Recv::Event(Event::Unloaded)
    ));
    p.kill();
}

#[test]
fn a_helper_that_dies_before_answering_is_a_startup_death() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let e = Process::start(&fake(), &env("die_at_start"), Duration::from_secs(20))
        .err()
        .unwrap();
    assert_eq!(e, StartFailure::Died);
}

#[test]
fn a_helper_from_another_protocol_version_is_refused() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let e = Process::start(&fake(), &env("wrong_protocol"), Duration::from_secs(20))
        .err()
        .unwrap();
    assert_eq!(e, StartFailure::VersionMismatch { helper: 1 });
}

#[test]
fn a_missing_executable_cannot_spawn() {
    let e = Process::start(
        Path::new("/nonexistent/helper"),
        &[],
        Duration::from_secs(1),
    )
    .err()
    .unwrap();
    assert!(matches!(e, StartFailure::CouldNotSpawn(_)), "{e:?}");
}

#[test]
fn the_stderr_tail_recognises_an_out_of_memory_death() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let (mut p, _) = Process::start(&fake(), &env("oom"), Duration::from_secs(20)).unwrap();
    p.send(&Request::Complete(voice_engine_protocol::Completion {
        model_path: "/m".into(),
        n_ctx: 1,
        system: String::new(),
        user: String::new(),
        json_schema: None,
        max_tokens: 1,
        temperature: 0.0,
        gpu: false,
        thinking: false,
        thinking_budget: None,
        seed: None,
    }))
    .unwrap();
    assert!(matches!(p.recv(Duration::from_secs(10)), Recv::Closed));
    // The stderr thread may finish a moment after stdout closes.
    let deadline = std::time::Instant::now() + Duration::from_secs(3);
    while !p.died_of_memory() && std::time::Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(p.died_of_memory());
    p.kill();
}

// --- the speech helper's process (the same type with the speech messages) ---------------------------------------------

fn fake_stt() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake-stt-helper"))
}

#[test]
fn a_working_speech_helper_answers_the_ping_with_its_backend_and_then_its_requests() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let (mut p, backend) =
        SttProcess::start(&fake_stt(), &env(""), Duration::from_secs(20)).unwrap();
    assert_eq!(backend, Backend::cpu());
    p.send(&SttRequest::Unload).unwrap();
    assert!(matches!(
        p.recv(Duration::from_secs(5)),
        Recv::Event(SttEvent::Unloaded)
    ));
    p.kill();
}

#[test]
fn a_speech_helper_from_another_protocol_version_is_refused_and_one_that_dies_is_a_startup_death() {
    let _t = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let e = SttProcess::start(&fake_stt(), &env("wrong_protocol"), Duration::from_secs(20))
        .err()
        .unwrap();
    assert_eq!(e, StartFailure::VersionMismatch { helper: 1 });
    let e = SttProcess::start(&fake_stt(), &env("die_at_start"), Duration::from_secs(20))
        .err()
        .unwrap();
    assert_eq!(e, StartFailure::Died);
}
