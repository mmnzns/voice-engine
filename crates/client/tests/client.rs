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

fn with_gpu(mut cfg: LlmConfig, gpu_mode: &str, failures: &std::path::Path) -> LlmConfig {
    cfg.gpu_helper = Some(fake());
    cfg.gpu_env = vec![("FAKE_MODE".into(), gpu_mode.into())];
    cfg.gpu_failures = Some(failures.to_path_buf());
    cfg
}

#[test]
fn the_graphics_card_helper_is_used_when_it_works() {
    let _t = turn();
    let (d, st) = state();
    let client = LlmClient::new(with_gpu(
        config("", &st),
        "gpu_ok",
        &d.path().join("gpu.txt"),
    ));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
    assert!(!out.fell_back);
}

#[test]
fn a_graphics_card_helper_that_dies_hands_the_request_to_the_processor_and_is_remembered() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    let client = LlmClient::new(with_gpu(config("", &st), "gpu_die", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.value.text, "hi");
    assert_eq!(out.backend.backend, "cpu");
    assert!(out.fell_back, "the app is told the card failed, once");
    assert_eq!(
        std::fs::read_to_string(&failures).unwrap().trim(),
        "test-build"
    );

    // A new client (the app restarted) with the same build and driver doesn't try the card again.
    let again = LlmClient::new(with_gpu(config("", &st), "gpu_die", &failures));
    let out = again.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    assert!(!out.fell_back);

    // A new build (or driver) gets a fresh try.
    let mut newer = with_gpu(config("", &st), "gpu_die", &failures);
    newer.gpu_key = "next-build".into();
    let out = LlmClient::new(newer)
        .complete(completion(), &mut |_| {})
        .unwrap();
    assert!(
        out.fell_back,
        "the new key tried the card again and it failed again"
    );

    // The app's "try the graphics card again" button.
    voice_engine_client::clear_gpu_failure(&failures);
    assert!(!failures.exists());
}

#[test]
fn a_graphics_card_error_from_the_helper_also_falls_back() {
    let _t = turn();
    let (d, st) = state();
    let client = LlmClient::new(with_gpu(
        config("", &st),
        "gpu_error",
        &d.path().join("gpu.txt"),
    ));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert!(out.fell_back);
    assert_eq!(out.backend.backend, "cpu");
}

/// What every fake helper was asked, one line each: `[<mode>] <op> gpu=<flag>` (the mode says which helper).
fn asked(log: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

/// `config` with a log of what the fake helpers are asked.
fn logged(mode: &str, state: &std::path::Path, log: &std::path::Path) -> LlmConfig {
    let mut cfg = config(mode, state);
    cfg.env.push(("FAKE_LOG".into(), log.display().to_string()));
    cfg
}

fn embedding(gpu: bool) -> voice_engine_protocol::Embedding {
    voice_engine_protocol::Embedding {
        model_path: "/e.gguf".into(),
        texts: vec!["a".into()],
        gpu,
    }
}

#[test]
fn the_fallback_to_the_processor_helper_says_gpu_false_and_the_next_calls_go_straight_there() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let failures = d.path().join("gpu.txt");
    let client = LlmClient::new(with_gpu(logged("", &st, &log), "gpu_die", &failures));
    // The app asks for the card (gpu: true); the card helper dies on it; the processor helper gets the same request
    // but with gpu: false, so a helper that is also the Mac's Metal helper really runs on the processor.
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert!(out.fell_back);
    assert_eq!(
        asked(&log),
        ["[gpu_die] complete gpu=true", "[] complete gpu=false"]
    );
    // The next calls go straight to the processor helper (no more lines from the card helper), say nothing about a
    // fallback, and every kind of request carries gpu: false although the app still says true.
    let next = client.complete(completion(), &mut |_| {}).unwrap();
    assert!(!next.fell_back);
    client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    client.embed(embedding(true)).unwrap();
    assert_eq!(
        asked(&log)[2..],
        [
            "[] complete gpu=false",
            "[] load gpu=false",
            "[] embed gpu=false"
        ]
    );
}

#[test]
fn an_error_from_the_card_helper_also_resends_with_gpu_false() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let client = LlmClient::new(with_gpu(
        logged("", &st, &log),
        "gpu_error",
        &d.path().join("gpu.txt"),
    ));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert!(out.fell_back);
    assert_eq!(
        asked(&log),
        ["[gpu_error] complete gpu=true", "[] complete gpu=false"]
    );
}

#[test]
fn a_remembered_failure_sends_gpu_false_to_the_processor_helper_from_the_first_request() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let failures = d.path().join("gpu.txt");
    std::fs::write(&failures, "test-build").unwrap();
    let client = LlmClient::new(with_gpu(logged("", &st, &log), "gpu_die", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert!(!out.fell_back);
    assert_eq!(asked(&log), ["[] complete gpu=false"]);
}

#[test]
fn a_helper_that_is_the_only_one_keeps_the_gpu_the_app_asked_for() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    // No card helper is configured: the one helper is the processor helper on a PC, and Metal's on the Mac, and the
    // app's gpu: true is for it. Only a fallback from a card helper turns the flag off.
    let client = LlmClient::new(logged("", &st, &log));
    client.complete(completion(), &mut |_| {}).unwrap();
    // A card helper that works gets the request as it is too.
    let log2 = d.path().join("asked2.log");
    let with_card = LlmClient::new(with_gpu(
        logged("", &st, &log2),
        "gpu_ok",
        &d.path().join("gpu.txt"),
    ));
    with_card.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(asked(&log), ["[] complete gpu=true"]);
    assert_eq!(asked(&log2), ["[gpu_ok] complete gpu=true"]);
}

#[test]
fn a_metal_helper_that_dies_is_a_card_failure_too() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let failures = d.path().join("gpu.txt");
    // The Mac: the same binary is both helpers; the card one says `metal` and names no device.
    let client = LlmClient::new(with_gpu(logged("", &st, &log), "metal_die", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.value.text, "hi");
    assert_eq!(out.backend.backend, "cpu");
    assert!(out.fell_back, "not a plain crash");
    assert_eq!(
        std::fs::read_to_string(&failures).unwrap().trim(),
        "test-build"
    );
    assert_eq!(
        asked(&log),
        ["[metal_die] complete gpu=true", "[] complete gpu=false"]
    );
    // Remembered: the next call, and a new client, go straight to the processor helper.
    assert!(
        !client
            .complete(completion(), &mut |_| {})
            .unwrap()
            .fell_back
    );
    let again = LlmClient::new(with_gpu(logged("", &st, &log), "metal_die", &failures));
    assert!(!again.complete(completion(), &mut |_| {}).unwrap().fell_back);
    assert_eq!(
        asked(&log)[2..],
        ["[] complete gpu=false", "[] complete gpu=false"]
    );
}

#[test]
fn cancel_stops_the_request_and_the_helper_keeps_going() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("slow", &st));
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.complete(completion(), &mut |_| {}));
        std::thread::sleep(Duration::from_millis(500));
        client.cancel();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
}

#[test]
fn a_helper_that_ignores_cancel_is_stopped_after_the_grace_period() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("ignore_cancel", &st));
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.complete(completion(), &mut |_| {}));
        std::thread::sleep(Duration::from_millis(300));
        client.cancel();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    assert!(began.elapsed() < Duration::from_secs(8));
    // A fresh helper serves the next call.
    assert!(client.load("/m.gguf", 512, true, &mut |_| {}).is_ok());
}

#[test]
fn cancel_with_nothing_running_does_nothing() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    client.cancel();
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
fn a_helper_that_keeps_streaming_still_hits_the_total_limit() {
    let _t = turn();
    let (_d, st) = state();
    let mut cfg = config("ignore_cancel", &st);
    // It writes progress every 50 ms forever, so the stall limit never trips; only the total limit can stop it.
    cfg.limits.total = Duration::from_millis(700);
    let client = LlmClient::new(cfg);
    let began = Instant::now();
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    assert!(began.elapsed() < Duration::from_secs(8));
}

#[test]
fn every_outcome_says_where_that_request_ran_not_where_the_helper_started() {
    let _t = turn();
    let (_d, st) = state();
    // The Mac helper says `metal` when it starts, and per request `cpu` for `gpu: false`.
    let client = LlmClient::new(config("per_request", &st));
    let on_card = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(on_card.backend.backend, "metal");
    let mut c = completion();
    c.gpu = false;
    assert_eq!(
        client.complete(c, &mut |_| {}).unwrap().backend.backend,
        "cpu"
    );
    assert_eq!(
        client
            .load("/m.gguf", 512, false, &mut |_| {})
            .unwrap()
            .backend
            .backend,
        "cpu"
    );
    assert_eq!(
        client
            .load("/m.gguf", 512, true, &mut |_| {})
            .unwrap()
            .backend
            .backend,
        "metal"
    );
    let embed = |gpu| voice_engine_protocol::Embedding {
        model_path: "/e.gguf".into(),
        texts: vec!["a".into()],
        gpu,
    };
    assert_eq!(client.embed(embed(false)).unwrap().backend.backend, "cpu");
    assert_eq!(client.embed(embed(true)).unwrap().backend.backend, "metal");
}

#[test]
fn a_helper_that_does_not_say_where_a_reply_ran_is_taken_at_its_start() {
    let _t = turn();
    let (d, st) = state();
    // An older helper: replies carry no backend, so the outcome keeps what `ready` said (here a Vulkan card).
    let client = LlmClient::new(with_gpu(
        config("", &st),
        "gpu_ok",
        &d.path().join("gpu.txt"),
    ));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
}

const MIGUELS_PC: &str =
    "igpu:Vulkan:AMD Radeon(TM) Graphics:512;gpu:Vulkan:NVIDIA GeForce RTX 5070 Ti:16303";

fn with_devices(mut cfg: LlmConfig, devices: &str) -> LlmConfig {
    cfg.env.push(("FAKE_DEVICES".into(), devices.into()));
    cfg
}

#[test]
fn devices_lists_what_the_helper_sees_and_names_the_discrete_card() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(with_devices(config("", &st), MIGUELS_PC));
    let list = client.devices().unwrap();
    assert_eq!(list.devices.len(), 3);
    // ggml's order: the iGPU, the RTX, then the processor; the RTX is the one the helper chose.
    assert_eq!(list.chosen, Some(1));
    let card = client.discrete_card().unwrap().unwrap();
    assert_eq!(card.description, "NVIDIA GeForce RTX 5070 Ti");
    assert_eq!(card.memory_mb, 16303);
}

#[test]
fn a_pc_with_integrated_graphics_or_no_graphics_has_no_discrete_card() {
    let _t = turn();
    let (_d, st) = state();
    let igpu = LlmClient::new(with_devices(
        config("", &st),
        "igpu:Vulkan:Intel(R) UHD Graphics:8000",
    ));
    assert_eq!(igpu.devices().unwrap().chosen, None);
    assert_eq!(igpu.discrete_card().unwrap(), None);
    let cpu = LlmClient::new(config("", &st));
    assert_eq!(cpu.devices().unwrap().devices.len(), 1);
    assert_eq!(cpu.discrete_card().unwrap(), None);
}
