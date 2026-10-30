//! `SttClient` against the fake speech helper (src/bin/fake_stt_helper.rs).

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use voice_engine_client::{EngineError, Limits, SttClient, SttConfig};
use voice_engine_protocol::stt::{ChunkOptions, SttLoad, SttModelKind, Timing};

/// Helpers that start at the same moment can crash on a Mac (see CHANGELOG), so the tests take turns.
static TURN: Mutex<()> = Mutex::new(());

fn turn() -> std::sync::MutexGuard<'static, ()> {
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

fn fake() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_fake-stt-helper"))
}

fn config(mode: &str, state: &Path) -> SttConfig {
    SttConfig {
        cpu_helper: fake(),
        gpu_helper: None,
        env: vec![
            ("FAKE_MODE".into(), mode.into()),
            ("FAKE_STATE".into(), state.display().to_string()),
        ],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: "test-build".into(),
        limits: Limits {
            start: Duration::from_secs(20),
            stall: Duration::from_millis(600),
            total: Duration::from_secs(30),
        },
    }
}

fn model(id: &str) -> SttLoad {
    SttLoad {
        model_id: id.into(),
        model_dir: format!("/models/{id}"),
        kind: SttModelKind::Whisper,
        gpu: true,
        threads: None,
    }
}

fn state() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("state");
    (dir, path)
}

fn audio() -> Vec<f32> {
    vec![0.25; 16_000]
}

fn opts() -> ChunkOptions {
    ChunkOptions::default()
}

#[test]
fn a_model_is_loaded_and_audio_comes_back_as_a_transcript_with_the_backend() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("", &st));
    let loaded = client.load(model("whisper-turbo")).unwrap();
    assert_eq!(loaded.value.seconds, 0.5);
    assert_eq!(loaded.backend.backend, "cpu");
    assert!(!loaded.fell_back);

    let options = ChunkOptions {
        language: Some("fr".into()),
        timing: Timing::Segments,
        keep_fillers: true,
        vocabulary: vec!["Talaan".into(), "Ensayo".into()],
    };
    let out = client.transcribe(&audio(), &options).unwrap();
    let t = &out.value;
    // The fake echoes what it received: the options and the audio arrived intact, for the model that was loaded.
    assert_eq!(t.segments[0].text, "segments vocab=2 lang=fr");
    assert_eq!(t.segments[0].words.len(), 3);
    assert_eq!(t.language.as_deref(), Some("whisper-turbo"));
    assert_eq!(t.stopped_early_at_ms, Some(16_000));
    assert_eq!(out.backend.backend, "cpu");
    assert!(!out.fell_back);
    // It stays loaded between calls: a second call needs no load.
    let again = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(again.value.segments[0].text, "words vocab=0 lang=none");
    client.shutdown();
}

#[test]
fn transcribing_with_no_model_loaded_is_a_bad_request_and_starts_no_helper() {
    let _t = turn();
    let (_d, st) = state();
    // A helper that cannot start would say CouldNotStart if one were started.
    let client = SttClient::new(config("die_at_start", &st));
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::BadRequest
    );
}

#[test]
fn unload_frees_the_model_and_a_transcribe_after_it_is_a_bad_request_until_the_next_load() {
    let _t = turn();
    let (_d, st) = state();
    // Unload with no helper running has nothing to free and must not start one.
    SttClient::new(config("die_at_start", &st))
        .unload()
        .unwrap();

    let client = SttClient::new(config("", &st));
    client.load(model("whisper-base")).unwrap();
    assert!(client.transcribe(&audio(), &opts()).is_ok());
    client.unload().unwrap();
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::BadRequest
    );
    // The app loads again when it wants to transcribe again.
    client.load(model("whisper-small")).unwrap();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-small"));
    client.shutdown();
}

#[test]
fn loading_another_model_replaces_the_first_and_a_failed_load_leaves_none() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("", &st));
    client.load(model("whisper-base")).unwrap();
    client.load(model("whisper-small")).unwrap();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-small"));
    client.shutdown();

    let client = SttClient::new(config("missing", &st));
    let e = client.load(model("whisper-base")).unwrap_err();
    assert_eq!(
        e,
        EngineError::ModelMissing {
            path: "/models/whisper-base/model.bin".into()
        }
    );
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::BadRequest
    );
}

#[test]
fn a_helper_that_dies_mid_call_is_a_crash_and_the_next_call_starts_a_new_one_with_the_model_loaded_again()
 {
    let _t = turn();
    let (_d, st) = state();
    // The first transcribe of the first helper dies; the next helper works.
    let client = SttClient::new(config("die_on_first_transcribe", &st));
    client.load(model("whisper-turbo")).unwrap();
    let e = client.transcribe(&audio(), &opts()).unwrap_err();
    assert_eq!(e, EngineError::Crashed);
    // The app does not load again: the client puts the remembered model into the new helper.
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-turbo"));
    client.shutdown();

    // A helper that always dies on transcribe: Crashed every time, and load still works (a fresh helper each time).
    let client = SttClient::new(config("die_on_transcribe", &st));
    client.load(model("whisper-turbo")).unwrap();
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::Crashed
    );
    assert!(client.load(model("whisper-turbo")).is_ok());
}

#[test]
fn a_helper_that_died_while_idle_is_replaced_and_given_the_model_again_without_an_error() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("exit_after_first_answer", &st));
    client.load(model("whisper-turbo")).unwrap();
    assert!(client.transcribe(&audio(), &opts()).is_ok());
    // The helper exits right after its first answer. Nobody was waiting on it, so the next call is not an error.
    std::thread::sleep(Duration::from_millis(400));
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-turbo"));
    assert_eq!(
        starts(&st),
        2,
        "one start for the first helper, one replacement"
    );
    client.shutdown();
}

#[test]
fn unload_does_not_start_a_helper_to_free_a_model_that_died_with_the_last_one() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("exit_after_first_answer", &st));
    client.load(model("whisper-turbo")).unwrap();
    assert!(client.transcribe(&audio(), &opts()).is_ok());
    std::thread::sleep(Duration::from_millis(400)); // the helper exits after that answer
    client.unload().unwrap();
    assert_eq!(starts(&st), 1, "no second helper just to unload");
    // And the model is forgotten: nothing to transcribe with until the next load.
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::BadRequest
    );
    assert_eq!(starts(&st), 1);
}

/// How many helpers `exit_after_first_answer` has seen start.
fn starts(state: &Path) -> u32 {
    std::fs::read_to_string(state)
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(0)
}

#[test]
fn a_helper_from_another_protocol_version_is_a_version_mismatch() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("wrong_protocol", &st));
    let e = client.load(model("whisper-turbo")).unwrap_err();
    assert_eq!(e, EngineError::VersionMismatch { helper: 1, app: 2 });
}

#[test]
fn a_helper_that_dies_at_startup_is_started_again_up_to_three_times() {
    let _t = turn();
    let (_d, st) = state();
    // The first two starts die without a word (the Metal shader cache crash); the third works.
    let client = SttClient::new(config("die_at_start_twice", &st));
    assert!(client.load(model("whisper-turbo")).is_ok());
    client.shutdown();

    let (_d2, st2) = state();
    let client = SttClient::new(config("die_at_start", &st2));
    let e = client.load(model("whisper-turbo")).unwrap_err();
    assert!(matches!(e, EngineError::CouldNotStart { .. }), "{e:?}");
}

#[test]
fn an_out_of_memory_death_is_reported_as_out_of_memory() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("oom", &st));
    client.load(model("whisper-large-v3")).unwrap();
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::OutOfMemory
    );
}

#[test]
fn the_helpers_own_errors_come_back_as_their_kinds() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("too_long", &st));
    client.load(model("whisper-turbo")).unwrap();
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::TooLong { max_secs: 40 }
    );
    // The helper stays up after an error: another call is answered (by the same kind, here).
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::TooLong { max_secs: 40 }
    );
}

#[test]
fn a_helper_that_says_nothing_is_stopped_at_the_stall_limit() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("stall", &st));
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let e = client.transcribe(&audio(), &opts()).unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    assert!(began.elapsed() < Duration::from_secs(10));
}

#[test]
fn a_load_that_takes_longer_than_the_stall_limit_is_a_timeout_too() {
    let _t = turn();
    let (_d, st) = state();
    // The fake needs 4 s to load and the stall limit is 600 ms: a margin of seconds, so a slow runner that delays the
    // client's clock check can not see the answer arrive first. The helper is stopped at the limit, so it ends in ~0.6 s.
    let client = SttClient::new(config("slow_load", &st));
    let e = client.load(model("whisper-turbo")).unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
}

#[test]
fn a_helper_that_keeps_streaming_still_hits_the_total_limit() {
    let _t = turn();
    let (_d, st) = state();
    let mut cfg = config("chatter", &st);
    // It writes an event every 50 ms forever, so the stall limit never trips; only the total limit can stop it.
    cfg.limits.total = Duration::from_millis(700);
    let client = SttClient::new(cfg);
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let e = client.transcribe(&audio(), &opts()).unwrap_err();
    assert_eq!(e, EngineError::TimedOut);
    assert!(began.elapsed() < Duration::from_secs(8));
}

#[test]
fn a_helper_that_keeps_streaming_is_not_a_stall_so_without_a_total_limit_it_runs_on_until_cancelled()
 {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("chatter", &st));
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.transcribe(&audio(), &opts()));
        // Far past the 600 ms stall limit: events keep arriving, so it is not a stall.
        std::thread::sleep(Duration::from_millis(1500));
        client.cancel();
        worker.join().unwrap()
    });
    // It ignores `cancel`, so the client stops it after the grace period, still answering Cancelled.
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    assert!(
        began.elapsed() >= Duration::from_millis(3000),
        "{:?}",
        began.elapsed()
    );
    assert!(began.elapsed() < Duration::from_secs(10));
}

fn with_gpu(mut cfg: SttConfig, gpu_mode: &str, failures: &Path) -> SttConfig {
    cfg.gpu_helper = Some(fake());
    cfg.gpu_env = vec![("FAKE_MODE".into(), gpu_mode.into())];
    cfg.gpu_failures = Some(failures.to_path_buf());
    cfg
}

#[test]
fn the_graphics_card_helper_is_used_when_it_works() {
    let _t = turn();
    let (d, st) = state();
    let client = SttClient::new(with_gpu(
        config("", &st),
        "gpu_ok",
        &d.path().join("gpu.txt"),
    ));
    client.load(model("whisper-turbo")).unwrap();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.backend.backend, "vulkan");
    assert_eq!(out.backend.device.as_deref(), Some("Fake GPU"));
    assert_eq!(out.backend.vram_mb, 8000);
    assert!(!out.fell_back);
    client.shutdown();
}

#[test]
fn a_graphics_card_helper_that_dies_hands_the_call_to_the_processor_helper_and_is_remembered() {
    let _t = turn();
    let (d, st) = state();
    let failures = d.path().join("gpu.txt");
    let client = SttClient::new(with_gpu(config("", &st), "gpu_die", &failures));
    let loaded = client.load(model("whisper-turbo")).unwrap();
    assert_eq!(loaded.backend.backend, "vulkan");
    // It dies on transcribe; the processor helper is started, loads the same model, and answers.
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-turbo"));
    assert_eq!(out.backend.backend, "cpu");
    assert!(out.fell_back, "the app is told the card failed, once");
    assert_eq!(
        std::fs::read_to_string(&failures).unwrap().trim(),
        "test-build"
    );
    // Later calls stay on the processor and say nothing more.
    let next = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(next.backend.backend, "cpu");
    assert!(!next.fell_back);

    // A new client (the app restarted) with the same build and driver doesn't try the card again.
    let again = SttClient::new(with_gpu(config("", &st), "gpu_die", &failures));
    let first = again.load(model("whisper-turbo")).unwrap();
    assert_eq!(first.backend.backend, "cpu");
    assert!(!first.fell_back);

    // A new build (or driver) gets a fresh try.
    let mut newer = with_gpu(config("", &st), "gpu_die", &failures);
    newer.gpu_key = "next-build".into();
    let newer = SttClient::new(newer);
    assert_eq!(
        newer.load(model("whisper-turbo")).unwrap().backend.backend,
        "vulkan"
    );
    assert!(newer.transcribe(&audio(), &opts()).unwrap().fell_back);

    // The app's "try the graphics card again" button.
    voice_engine_client::clear_gpu_failure(&failures);
    assert!(!failures.exists());
}

/// What every fake helper was asked to load, one line each: `[<mode>] load gpu=<flag>` (the mode says which helper).
fn asked(log: &Path) -> Vec<String> {
    std::fs::read_to_string(log)
        .unwrap_or_default()
        .lines()
        .map(str::to_string)
        .collect()
}

/// `config` with a log of what the fake helpers are asked to load.
fn logged(mode: &str, state: &Path, log: &Path) -> SttConfig {
    let mut cfg = config(mode, state);
    cfg.env.push(("FAKE_LOG".into(), log.display().to_string()));
    cfg
}

#[test]
fn the_fallback_to_the_processor_helper_loads_the_model_with_gpu_false() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let failures = d.path().join("gpu.txt");
    let client = SttClient::new(with_gpu(logged("", &st, &log), "gpu_die", &failures));
    client.load(model("whisper-turbo")).unwrap();
    // The card helper dies on the audio; the processor helper loads the remembered model, saying gpu: false, and
    // answers; the app is told once.
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert!(out.fell_back);
    assert_eq!(
        asked(&log),
        ["[gpu_die] load gpu=true", "[] load gpu=false"]
    );
    // Straight to the processor helper from now on: nothing is loaded again, and nothing more is said.
    let next = client.transcribe(&audio(), &opts()).unwrap();
    assert!(!next.fell_back);
    assert_eq!(asked(&log).len(), 2);
    // The app loads a model again (gpu: true, as it always asks) and the processor helper is told gpu: false.
    let again = client.load(model("whisper-base")).unwrap();
    assert!(!again.fell_back);
    assert_eq!(asked(&log)[2..], ["[] load gpu=false"]);
    // A new helper after `shutdown` is the processor one and gets the remembered model with gpu: false.
    client.shutdown();
    assert!(!client.transcribe(&audio(), &opts()).unwrap().fell_back);
    assert_eq!(asked(&log)[3..], ["[] load gpu=false"]);
}

#[test]
fn an_error_from_the_card_helper_also_loads_with_gpu_false_on_the_processor_helper() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let client = SttClient::new(with_gpu(
        logged("", &st, &log),
        "gpu_error",
        &d.path().join("gpu.txt"),
    ));
    client.load(model("whisper-turbo")).unwrap();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert!(out.fell_back);
    assert_eq!(
        asked(&log),
        ["[gpu_error] load gpu=true", "[] load gpu=false"]
    );
}

#[test]
fn a_remembered_failure_loads_with_gpu_false_from_the_first_request() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    let failures = d.path().join("gpu.txt");
    std::fs::write(&failures, "test-build").unwrap();
    let client = SttClient::new(with_gpu(logged("", &st, &log), "gpu_die", &failures));
    let loaded = client.load(model("whisper-turbo")).unwrap();
    assert!(!loaded.fell_back);
    client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(asked(&log), ["[] load gpu=false"]);
}

#[test]
fn a_helper_that_is_the_only_one_keeps_the_gpu_the_app_asked_for() {
    let _t = turn();
    let (d, st) = state();
    let log = d.path().join("asked.log");
    // No card helper configured: the one helper is the processor helper on a PC and the Metal one on the Mac, and
    // the app's gpu: true is for it. A card helper that works gets the load as it is too.
    SttClient::new(logged("", &st, &log))
        .load(model("whisper-turbo"))
        .unwrap();
    let log2 = d.path().join("asked2.log");
    SttClient::new(with_gpu(
        logged("", &st, &log2),
        "gpu_ok",
        &d.path().join("gpu.txt"),
    ))
    .load(model("whisper-turbo"))
    .unwrap();
    assert_eq!(asked(&log), ["[] load gpu=true"]);
    assert_eq!(asked(&log2), ["[gpu_ok] load gpu=true"]);
}

#[test]
fn a_graphics_card_error_from_the_helper_also_falls_back() {
    let _t = turn();
    let (d, st) = state();
    let client = SttClient::new(with_gpu(
        config("", &st),
        "gpu_error",
        &d.path().join("gpu.txt"),
    ));
    client.load(model("whisper-turbo")).unwrap();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert!(out.fell_back);
    assert_eq!(out.backend.backend, "cpu");
}

#[test]
fn a_model_that_runs_on_the_processor_though_the_card_was_asked_for_says_it_fell_back() {
    let _t = turn();
    let (_d, st) = state();
    // The helper itself reports `fell_back` (the engine could not start the card): the app hears it from `fell_back`
    // without any process having died.
    let client = SttClient::new(config("cpu_fallback", &st));
    let loaded = client.load(model("whisper-turbo")).unwrap();
    assert!(loaded.fell_back);
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert!(out.fell_back);
    assert_eq!(out.backend.backend, "cpu");
    client.shutdown();
}

#[test]
fn cancel_stops_a_helper_that_answers_it_and_the_helper_keeps_going() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("cancellable", &st));
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.transcribe(&audio(), &opts()));
        std::thread::sleep(Duration::from_millis(300));
        client.cancel();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    assert!(
        began.elapsed() < Duration::from_secs(2),
        "no kill was needed"
    );
    // Same helper: the model is still loaded, so unload (which only asks a running helper) answers.
    client.unload().unwrap();
}

#[test]
fn a_decode_that_cannot_be_interrupted_is_killed_after_the_grace_period_and_the_model_comes_back_later()
 {
    let _t = turn();
    let (_d, st) = state();
    // The first transcribe never answers and never reads `cancel` (like a Whisper decode); the next helper works.
    let mut cfg = config("stall_once", &st);
    cfg.limits.stall = Duration::from_secs(60);
    let client = SttClient::new(cfg);
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.transcribe(&audio(), &opts()));
        std::thread::sleep(Duration::from_millis(300));
        client.cancel();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    let took = began.elapsed();
    assert!(
        took >= Duration::from_millis(2000),
        "{took:?}: waited the 2 s grace"
    );
    assert!(took < Duration::from_secs(8), "{took:?}");
    // The killed helper took the model with it; the client puts it into a fresh one without the app loading again.
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-turbo"));
    client.shutdown();
}

#[test]
fn cancel_with_nothing_running_does_nothing() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("", &st));
    client.cancel();
    client.load(model("whisper-turbo")).unwrap();
    client.cancel();
    assert!(client.transcribe(&audio(), &opts()).is_ok());
    client.shutdown();
}

#[test]
fn shutdown_stops_the_helper_and_the_next_call_starts_one_with_the_model_loaded_again() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(config("", &st));
    client.shutdown(); // nothing running: nothing to do
    client.load(model("whisper-turbo")).unwrap();
    client.shutdown();
    let out = client.transcribe(&audio(), &opts()).unwrap();
    assert_eq!(out.value.language.as_deref(), Some("whisper-turbo"));
    client.shutdown();
}

#[test]
fn shutdown_from_another_thread_ends_a_call_that_is_running() {
    let _t = turn();
    let (_d, st) = state();
    let mut cfg = config("stall", &st);
    cfg.limits.stall = Duration::from_secs(60);
    let client = SttClient::new(cfg);
    client.load(model("whisper-turbo")).unwrap();
    let began = Instant::now();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| client.transcribe(&audio(), &opts()));
        std::thread::sleep(Duration::from_millis(300));
        client.shutdown();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    assert!(
        began.elapsed() < Duration::from_secs(2),
        "{:?}",
        began.elapsed()
    );
    // The client is usable again afterwards.
    assert!(client.load(model("whisper-turbo")).is_ok());
}

const MIGUELS_PC: &str =
    "igpu:Vulkan:AMD Radeon(TM) Graphics:512;gpu:Vulkan:NVIDIA GeForce RTX 5070 Ti:16303";

fn with_devices(mut cfg: SttConfig, devices: &str) -> SttConfig {
    cfg.env.push(("FAKE_DEVICES".into(), devices.into()));
    cfg
}

#[test]
fn devices_lists_what_the_helper_sees_and_names_the_discrete_card() {
    let _t = turn();
    let (_d, st) = state();
    let client = SttClient::new(with_devices(config("", &st), MIGUELS_PC));
    let list = client.devices().unwrap();
    assert_eq!(list.devices.len(), 3);
    // ggml's order: the iGPU, the RTX, then the processor; the RTX is the one the helper chose.
    assert_eq!(list.chosen, Some(1));
    let card = client.discrete_card().unwrap().unwrap();
    assert_eq!(card.description, "NVIDIA GeForce RTX 5070 Ti");
    assert_eq!(card.memory_mb, 16303);
    // Asking loads no model: a transcribe still has none.
    assert_eq!(
        client.transcribe(&audio(), &opts()).unwrap_err(),
        EngineError::BadRequest
    );
}

#[test]
fn a_pc_with_integrated_graphics_or_no_graphics_has_no_discrete_card() {
    let _t = turn();
    let (_d, st) = state();
    let igpu = SttClient::new(with_devices(
        config("", &st),
        "igpu:Vulkan:Intel(R) UHD Graphics:8000",
    ));
    assert_eq!(igpu.devices().unwrap().chosen, None);
    assert_eq!(igpu.discrete_card().unwrap(), None);
    let cpu = SttClient::new(config("", &st));
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
    let client = SttClient::new(cfg);
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
