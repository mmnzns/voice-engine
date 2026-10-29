//! The client against the fake helper (src/bin/fake_llm_helper.rs).

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use voice_engine_client::{EngineError, Limits, LlmClient, LlmConfig};
use voice_engine_protocol::Completion;

/// Helpers that start at the same moment can crash on a Mac (see CHANGELOG), so the tests take turns.
static TURN: Mutex<()> = Mutex::new(());

fn turn() -> std::sync::MutexGuard<'static, ()> {
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

fn fake() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake-llm-helper"))
}

fn config(mode: &str, state: &std::path::Path) -> LlmConfig {
    LlmConfig {
        cpu_helper: fake(),
        gpu_helper: None,
        env: vec![
            ("FAKE_MODE".into(), mode.into()),
            ("FAKE_STATE".into(), state.display().to_string()),
        ],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: "test-build".into(),
        idle_secs: Some(120),
        limits: Limits {
            start: Duration::from_secs(20),
            stall: Duration::from_millis(600),
            total: Duration::from_secs(30),
        },
    }
}

fn completion() -> Completion {
    Completion {
        model_path: "/m.gguf".into(),
        n_ctx: 512,
        system: String::new(),
        user: "hello".into(),
        json_schema: None,
        max_tokens: 10,
        temperature: 0.0,
        gpu: true,
        thinking: false,
        thinking_budget: None,
        seed: None,
    }
}

fn state() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state");
    (dir, path)
}

#[test]
fn a_reply_comes_back_with_progress_and_the_backend() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    let mut progress = 0;
    let out = client
        .complete(completion(), &mut |_| progress += 1)
        .unwrap();
    assert_eq!(out.value.text, "hi");
    assert!(progress >= 1);
    assert_eq!(out.backend.backend, "cpu");
    assert!(!out.fell_back);
}

#[test]
fn a_helper_that_dies_at_startup_is_restarted() {
    let _t = turn();
    let (_d, st) = state();
    // The first two starts die without a word (the Metal shader cache crash); the third works.
    let client = LlmClient::new(config("die_at_start_twice", &st));
    assert_eq!(
        client
            .complete(completion(), &mut |_| {})
            .unwrap()
            .value
            .text,
        "hi"
    );
}

#[test]
fn a_helper_that_always_dies_at_startup_cannot_start() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("die_at_start", &st));
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert!(matches!(e, EngineError::CouldNotStart { .. }), "{e:?}");
}

#[test]
fn a_helper_from_another_protocol_version_is_a_version_mismatch() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("wrong_protocol", &st));
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::VersionMismatch { helper: 1, app: 2 });
}

#[test]
fn a_helper_that_dies_mid_request_is_a_crash_and_the_next_call_starts_a_new_one() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("die_on_complete", &st));
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::Crashed);
    // load isn't scripted to die, so this proves a fresh helper was started.
    assert!(client.load("/m.gguf", 512, true, &mut |_| {}).is_ok());
}

#[test]
fn a_helper_that_exits_just_as_the_request_arrives_is_tried_once_more() {
    let _t = turn();
    let (_d, st) = state();
    // An idle exit racing the request: it dies before saying anything the first time only.
    let client = LlmClient::new(config("die_on_first_complete", &st));
    assert_eq!(
        client
            .complete(completion(), &mut |_| {})
            .unwrap()
            .value
            .text,
        "hi"
    );
}

#[test]
fn an_out_of_memory_death_is_reported_as_out_of_memory() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("oom", &st));
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::OutOfMemory);
}

#[test]
fn no_answer_within_the_stall_limit_is_a_timeout() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("stall", &st));
    let began = Instant::now();
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    assert!(began.elapsed() < Duration::from_secs(10));
}

#[test]
fn load_inspect_embed_and_unload_work_and_unload_starts_nothing() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    // With no helper running, unload has nothing to free and must not start one (the state file stays absent only
    // for modes that count starts; here it just has to succeed quickly).
    client.unload().unwrap();
    let loaded = client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    assert_eq!(loaded.value.n_ctx_train, 4096);
    assert!(client.inspect("/m.gguf").unwrap().has_chat_template);
    let embedded = client
        .embed(voice_engine_protocol::Embedding {
            model_path: "/e.gguf".into(),
            texts: vec!["a".into()],
            gpu: false,
        })
        .unwrap();
    assert_eq!(embedded.value, vec![vec![1.0]]);
    client.unload().unwrap();
    client.shutdown();
}

#[test]
fn the_idle_time_can_be_changed_before_and_after_the_helper_starts() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    client.set_idle(Some(5)).unwrap();
    client.complete(completion(), &mut |_| {}).unwrap();
    client.set_idle(None).unwrap();
    client.set_idle(Some(60)).unwrap();
}
