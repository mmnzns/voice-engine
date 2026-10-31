//! The client against the fake helper (src/bin/fake_llm_helper.rs).

use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use voice_engine_client::{EngineError, Fallback, Limits, LlmClient, LlmConfig};
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
    assert_eq!(out.fallback, Some(Fallback::NoCardHelper));
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
    assert_eq!(
        e,
        EngineError::Crashed {
            exit_code: Some(139),
            signal: None,
            detail: String::new()
        }
    );
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

/// A client whose helper dies on `complete` the `die_noisy` way: `stderr` (lines separated by `~`) written, then exit.
fn noisy(stderr: &str, exit: &str, long: Option<usize>, state: &std::path::Path) -> LlmClient {
    let mut cfg = config("die_noisy", state);
    cfg.env.push(("FAKE_STDERR".into(), stderr.into()));
    cfg.env.push(("FAKE_EXIT".into(), exit.into()));
    if let Some(n) = long {
        cfg.env.push(("FAKE_STDERR_LONG".into(), n.to_string()));
    }
    LlmClient::new(cfg)
}

#[test]
fn a_crash_carries_the_exit_code_and_a_short_stderr_tail() {
    let _t = turn();
    let (_d, st) = state();
    // Seven lines, the last one 300 characters long, and an escape code in the second.
    let lines: Vec<String> = (1..=7).map(|n| format!("line {n}")).collect();
    let mut text = lines.join("~");
    text.push_str("~\u{1b}[31mred\u{1b}[0m");
    let client = noisy(&text, "3", Some(300), &st);
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    let EngineError::Crashed {
        exit_code,
        signal,
        detail,
    } = e
    else {
        panic!("{e:?}")
    };
    assert_eq!((exit_code, signal), (Some(3), None));
    let lines: Vec<&str> = detail.lines().collect();
    assert_eq!(lines.len(), 5, "{detail}");
    // The last five lines: line 5, 6, 7, the colored one (control characters turned into spaces), the long one.
    assert_eq!(&lines[..3], ["line 5", "line 6", "line 7"]);
    assert_eq!(lines[3], "[31mred [0m");
    assert_eq!(lines[4].chars().count(), 200);
}

#[test]
fn a_crash_that_wrote_nothing_has_an_empty_detail_and_shows_its_code_in_the_log_line() {
    let _t = turn();
    let (_d, st) = state();
    let client = noisy("", "42", None, &st);
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(
        e,
        EngineError::Crashed {
            exit_code: Some(42),
            signal: None,
            detail: String::new()
        }
    );
    assert_eq!(e.to_string(), "the helper crashed (exit code 42)");
}

#[test]
fn every_out_of_memory_message_is_reported_as_out_of_memory() {
    let _t = turn();
    let (_d, st) = state();
    for message in [
        "terminate called after throwing an instance of 'std::bad_alloc'",
        "memory allocation of 8589934592 bytes failed",
        "Cannot allocate memory",
        "unable to allocate CUDA0 buffer",
        "ggml: insufficient memory",
        "vk::Device::allocateMemory: ErrorOutOfDeviceMemory",
        "ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer",
        "GGML_ASSERT: Out Of Memory",
    ] {
        let client = noisy(message, "1", None, &st);
        let e = client.complete(completion(), &mut |_| {}).unwrap_err();
        assert_eq!(e, EngineError::OutOfMemory, "{message}");
    }
}

#[cfg(unix)]
#[test]
fn a_helper_ended_by_sigkill_is_reported_as_out_of_memory() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("sigkill", &st));
    let e = client.complete(completion(), &mut |_| {}).unwrap_err();
    assert_eq!(e, EngineError::OutOfMemory);
}

/// A client for the `slow_reply` fake: a load of `load_ms` and a reply of `reply_ms`, with limits far above both.
fn slow(load_ms: u64, reply_ms: u64, state: &std::path::Path) -> LlmClient {
    let mut cfg = config("slow_reply", state);
    cfg.env.push(("FAKE_LOAD_MS".into(), load_ms.to_string()));
    cfg.env.push(("FAKE_REPLY_MS".into(), reply_ms.to_string()));
    // The load is silent for `load_ms`, so the stall limit must be above it.
    cfg.limits.stall = Duration::from_secs(20);
    LlmClient::new(cfg)
}

#[test]
fn the_reply_limit_does_not_count_the_load() {
    let _t = turn();
    let (_d, st) = state();
    // A load of 1.5 s and a reply of 0.4 s: 1.9 s in all, against a limit of 1 s for the reply.
    let client = slow(1500, 400, &st);
    let began = Instant::now();
    let out = client
        .complete_within(Duration::from_secs(1), completion(), &mut |_| {})
        .unwrap();
    assert_eq!(out.value.text, "hi");
    assert!(began.elapsed() > Duration::from_millis(1800));
}

#[test]
fn a_reply_that_runs_past_its_limit_after_the_load_is_a_timeout_and_the_helper_is_stopped() {
    let _t = turn();
    let (_d, st) = state();
    // Loads for 1.5 s, then would reply for 6 s; the limit is 1 s from the end of the load.
    let client = slow(1500, 6000, &st);
    let began = Instant::now();
    let e = client
        .complete_within(Duration::from_secs(1), completion(), &mut |_| {})
        .unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    let took = began.elapsed();
    assert!(took > Duration::from_millis(2300), "{took:?}");
    assert!(took < Duration::from_millis(5000), "{took:?}");
    // The helper was stopped like for a stall: the next call starts a new one, which loads again (1.5 s).
    let began = Instant::now();
    let out = client
        .complete_within(Duration::from_secs(1), completion(), &mut |_| {})
        .unwrap_err();
    assert_eq!(out, EngineError::TimedOut);
    assert!(began.elapsed() > Duration::from_millis(2300));
}

#[test]
fn with_the_model_already_loaded_the_reply_limit_runs_from_the_send() {
    let _t = turn();
    let (_d, st) = state();
    let client = slow(1500, 6000, &st);
    client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    let began = Instant::now();
    let e = client
        .complete_within(Duration::from_millis(700), completion(), &mut |_| {})
        .unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    let took = began.elapsed();
    assert!(took > Duration::from_millis(600), "{took:?}");
    assert!(took < Duration::from_millis(3000), "{took:?}");
}

#[test]
fn the_config_limits_still_hold_a_load_the_reply_limit_does_not_cover() {
    let _t = turn();
    let (_d, st) = state();
    let mut cfg = config("slow_reply", &st);
    cfg.env.push(("FAKE_LOAD_MS".into(), "5000".into()));
    // The config's stall limit (600 ms) is for the load, and it still applies.
    let client = LlmClient::new(cfg);
    let e = client
        .complete_within(Duration::from_secs(30), completion(), &mut |_| {})
        .unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
}

#[test]
fn a_reply_limit_that_is_not_reached_leaves_the_reply_as_it_is() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    let mut progress = 0;
    let out = client
        .complete_within(Duration::from_secs(5), completion(), &mut |_| progress += 1)
        .unwrap();
    assert_eq!(out.value.text, "hi");
    assert!(progress >= 1);
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
    assert_eq!(out.fallback, None);
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
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
    assert_eq!(
        std::fs::read_to_string(&failures).unwrap().trim(),
        "test-build"
    );

    // A new client (the app restarted) with the same build and driver doesn't try the card again.
    let again = LlmClient::new(with_gpu(config("", &st), "gpu_die", &failures));
    let out = again.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    assert_eq!(out.fallback, Some(Fallback::RememberedFailure));

    // A new build (or driver) gets a fresh try.
    let mut newer = with_gpu(config("", &st), "gpu_die", &failures);
    newer.gpu_key = "next-build".into();
    let out = LlmClient::new(newer)
        .complete(completion(), &mut |_| {})
        .unwrap();
    assert_eq!(
        out.fallback,
        Some(Fallback::CardFailed),
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
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
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
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
    assert_eq!(
        asked(&log),
        ["[gpu_die] complete gpu=true", "[] complete gpu=false"]
    );
    // The next calls go straight to the processor helper (no more lines from the card helper), say nothing about a
    // fallback, and every kind of request carries gpu: false although the app still says true.
    let next = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(next.fallback, Some(Fallback::RememberedFailure));
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
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
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
    assert_eq!(out.fallback, Some(Fallback::RememberedFailure));
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
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
    assert_eq!(
        std::fs::read_to_string(&failures).unwrap().trim(),
        "test-build"
    );
    assert_eq!(
        asked(&log),
        ["[metal_die] complete gpu=true", "[] complete gpu=false"]
    );
    // Remembered: the next call, and a new client, go straight to the processor helper.
    assert_eq!(
        client.complete(completion(), &mut |_| {}).unwrap().fallback,
        Some(Fallback::RememberedFailure)
    );
    let again = LlmClient::new(with_gpu(logged("", &st, &log), "metal_die", &failures));
    assert_eq!(
        again.complete(completion(), &mut |_| {}).unwrap().fallback,
        Some(Fallback::RememberedFailure)
    );
    assert_eq!(
        asked(&log)[2..],
        ["[] complete gpu=false", "[] complete gpu=false"]
    );
}

/// What `complete` with `gpu: true` and `gpu: false` say about the card, one client each time.
fn reasons(cfg: LlmConfig) -> (Option<Fallback>, Option<Fallback>) {
    let client = LlmClient::new(cfg);
    let wanted = client.complete(completion(), &mut |_| {}).unwrap();
    let mut off = completion();
    off.gpu = false;
    let refused = client.complete(off, &mut |_| {}).unwrap();
    (wanted.fallback, refused.fallback)
}

#[test]
fn without_a_card_helper_a_processor_answer_to_a_card_request_says_there_is_no_card_helper() {
    let _t = turn();
    let (_d, st) = state();
    // The processor helper alone: "never tried" is told apart from "worked", and a request that did not ask for the
    // card has no reason.
    let (wanted, off) = reasons(config("", &st));
    assert_eq!(wanted, Some(Fallback::NoCardHelper));
    assert_eq!(off, None);
    let out = LlmClient::new(config("", &st))
        .complete(completion(), &mut |_| {})
        .unwrap();
    assert!(out.fell_back());
}

#[test]
fn the_mac_helper_answers_per_request_and_a_cpu_answer_to_a_card_request_there_is_no_card_path() {
    let _t = turn();
    let (_d, st) = state();
    // The Mac with no gpu_helper: `metal` for gpu: true, `cpu` for gpu: false; neither needs a reason.
    assert_eq!(reasons(config("per_request", &st)), (None, None));
    // The Mac helper that has no card runtime for the model: it started on `metal` and still answers `cpu`.
    let (wanted, off) = reasons(config("no_card_path", &st));
    assert_eq!(wanted, Some(Fallback::NoCardPath));
    assert_eq!(off, None);
    // Load and embed say it too.
    let client = LlmClient::new(config("no_card_path", &st));
    let loaded = client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    assert_eq!(loaded.fallback, Some(Fallback::NoCardPath));
    assert_eq!(
        client.embed(embedding(true)).unwrap().fallback,
        Some(Fallback::NoCardPath)
    );
    assert_eq!(client.embed(embedding(false)).unwrap().fallback, None);
}

#[test]
fn a_card_helper_that_works_has_no_reason_and_one_that_runs_on_the_processor_says_the_card_failed()
{
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    assert_eq!(
        reasons(with_gpu(config("", &st), "gpu_ok", &failures)),
        (None, None)
    );
    // The card helper started but answers with the processor (a Vulkan build on a PC with no usable card).
    let (wanted, off) = reasons(with_gpu(config("", &st), "", &failures));
    assert_eq!(wanted, Some(Fallback::CardFailed));
    assert_eq!(off, None);
}

#[test]
fn a_card_helper_that_cannot_start_is_a_card_failure_this_time_and_remembered_after() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    let client = LlmClient::new(with_gpu(config("", &st), "die_at_start", &failures));
    let first = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(first.fallback, Some(Fallback::CardFailed));
    let next = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(next.fallback, Some(Fallback::RememberedFailure));
    // A request that did not ask for the card is not about the card.
    let mut off = completion();
    off.gpu = false;
    assert_eq!(client.complete(off, &mut |_| {}).unwrap().fallback, None);
}

#[test]
fn a_card_helper_that_dies_of_memory_says_out_of_memory_and_the_next_call_says_remembered() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    let client = LlmClient::new(with_gpu(config("", &st), "gpu_oom", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    assert_eq!(out.fallback, Some(Fallback::CardOutOfMemory));
    assert!(out.fell_back());
    let next = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(next.fallback, Some(Fallback::RememberedFailure));
}

#[test]
fn load_and_embed_report_the_reason_too_and_a_retry_clears_it() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    std::fs::write(&failures, "test-build").unwrap();
    let client = LlmClient::new(with_gpu(config("", &st), "gpu_ok", &failures));
    let loaded = client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    assert_eq!(loaded.fallback, Some(Fallback::RememberedFailure));
    assert_eq!(
        client.embed(embedding(true)).unwrap().fallback,
        Some(Fallback::RememberedFailure)
    );
    assert_eq!(client.embed(embedding(false)).unwrap().fallback, None);
    client.retry_gpu();
    let loaded = client.load("/m.gguf", 512, true, &mut |_| {}).unwrap();
    assert_eq!(loaded.fallback, None);
    assert_eq!(loaded.backend.backend, "vulkan");
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

#[test]
fn devices_gives_up_after_its_own_short_limit_and_does_not_blame_the_card() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu-failures");
    let mut cfg = config("", &st);
    // The general limits are the defaults (120 s stall): only the devices limit can stop this in time.
    cfg.limits.stall = Duration::from_secs(120);
    cfg.limits.total = Duration::from_secs(30 * 60);
    cfg.gpu_helper = Some(fake());
    cfg.gpu_env = vec![("FAKE_MODE".into(), "no_devices".into())];
    cfg.gpu_failures = Some(failures.clone());
    let client = LlmClient::new(cfg);
    let began = Instant::now();
    assert_eq!(client.discrete_card().unwrap_err(), EngineError::TimedOut);
    // 10 s, with a margin: nowhere near the 120 s stall limit.
    assert!(
        began.elapsed() < Duration::from_secs(15),
        "{:?}",
        began.elapsed()
    );
    // A timeout is not the card failing: nothing is remembered.
    assert!(!failures.exists());
}

#[test]
fn retry_gpu_puts_a_running_client_back_on_the_card_and_forgets_the_failure() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    // The card helper dies on its first request and works after that.
    let client = LlmClient::new(with_gpu(config("", &st), "gpu_die_once", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    assert_eq!(out.fallback, Some(Fallback::CardFailed));
    assert!(failures.exists());
    // Still on the processor, whatever the app does to the file alone.
    voice_engine_client::clear_gpu_failure(&failures);
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(
        out.backend.backend, "cpu",
        "clear_gpu_failure alone does not reach the client"
    );
    // The app's "Try again".
    client.retry_gpu();
    assert!(!failures.exists());
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
    assert_eq!(out.fallback, None);
    assert!(!failures.exists());
    // With the card helper running, retry_gpu leaves it alone.
    client.retry_gpu();
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
    client.shutdown();
}

#[test]
fn retry_gpu_on_a_client_that_started_on_a_remembered_failure_tries_the_card() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    std::fs::write(&failures, "test-build").unwrap();
    let client = LlmClient::new(with_gpu(config("", &st), "gpu_ok", &failures));
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    assert_eq!(out.fallback, Some(Fallback::RememberedFailure));
    client.retry_gpu();
    assert!(!failures.exists());
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
    client.shutdown();
}

#[test]
fn retry_gpu_without_a_card_helper_or_a_file_does_nothing_harmful() {
    let _t = turn();
    let (_d, st) = state();
    let client = LlmClient::new(config("", &st));
    client.retry_gpu();
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    client.retry_gpu();
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(out.backend.backend, "cpu");
    client.shutdown();
}
