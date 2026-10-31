//! The graphics card falling back to the processor, against the real helpers. Ignored by default: it needs the two
//! builds of a helper, the Vulkan one (`--features vulkan`, so a Windows PC with a card) and the processor one, and a
//! model. `VOICE_ENGINE_GPU=fail` makes the Vulkan build abort on its first real request, the way a driver crash would
//! (`fail_on_purpose` in `crates/llm/src/lib.rs` and `crates/stt/src/helper.rs`); the client must start the processor
//! helper, answer the same call, and say `fell_back`. A test whose variables are not set is skipped with a message that
//! names them (it prints why and passes). With `VOICE_ENGINE_TEST_EXPECT_CARD=1` a test that finds the Vulkan helper
//! did not run on a card fails instead of skipping. Prints backend names and counts only.
//!
//! ```powershell
//! cargo build --release -p voice-engine-stt --features vulkan
//! Copy-Item target\release\voice-engine-stt.exe $env:TEMP\stt-vulkan.exe
//! cargo build --release -p voice-engine-stt
//! $env:VOICE_ENGINE_TEST_STT_GPU_HELPER = "$env:TEMP\stt-vulkan.exe"
//! $env:VOICE_ENGINE_TEST_STT_HELPER = "$PWD\target\release\voice-engine-stt.exe"
//! cargo test --release -p voice-engine-client --test real_fallback -- --ignored --nocapture --test-threads=1
//! ```
//!
//! The AI helper is the same with `voice-engine-llm`: `VOICE_ENGINE_TEST_GPU_HELPER` (the `--features vulkan` build),
//! `VOICE_ENGINE_TEST_HELPER` (the processor build) and `VOICE_ENGINE_TEST_LLM` (a `.gguf` file). The speech test reads
//! `whisper-base` as the speed bench does (`VOICE_ENGINE_TEST_MODELS`, the shared folder, Talaan's folder), only ever read.
//! On a Mac there is no Vulkan build, so there is nothing to run.

use std::path::PathBuf;
use voice_engine_client::{Fallback, Limits, LlmClient, LlmConfig, SttClient, SttConfig};
use voice_engine_protocol::Completion;
use voice_engine_protocol::stt::{ChunkOptions, SttLoad, SttModelKind};

#[path = "../../stt/support/mod.rs"]
mod support;

/// The files named by these environment variables, or None after saying which are missing.
fn files(names: &[&str]) -> Option<Vec<PathBuf>> {
    let mut found = vec![];
    let mut missing = vec![];
    for name in names {
        match std::env::var_os(name).map(PathBuf::from) {
            Some(p) if p.is_file() => found.push(p),
            Some(p) => missing.push(format!("{name} ({} is not a file)", p.display())),
            None => missing.push((*name).to_string()),
        }
    }
    if missing.is_empty() {
        return Some(found);
    }
    eprintln!("SKIPPED: set {}", missing.join(", "));
    None
}

/// The helper is told to fail on the card (both of them: the processor build ignores it).
fn fail_env() -> Vec<(String, String)> {
    vec![("VOICE_ENGINE_GPU".into(), "fail".into())]
}

#[test]
#[ignore]
fn a_forced_card_failure_in_the_real_speech_helper_falls_back_to_the_processor_helper() {
    let Some(f) = files(&[
        "VOICE_ENGINE_TEST_STT_GPU_HELPER",
        "VOICE_ENGINE_TEST_STT_HELPER",
    ]) else {
        return;
    };
    let dir = match support::find_model_dir("whisper-base") {
        Ok(d) => d,
        Err(why) => return eprintln!("SKIPPED: {why}"),
    };
    let client = SttClient::new(SttConfig {
        cpu_helper: f[1].clone(),
        gpu_helper: Some(f[0].clone()),
        env: fail_env(),
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        limits: Limits::default(),
    });
    let loaded = client
        .load(SttLoad {
            model_id: "whisper-base".into(),
            model_dir: dir.display().to_string(),
            kind: SttModelKind::Whisper,
            gpu: true,
            threads: None,
        })
        .unwrap();
    eprintln!(
        "load: backend {}, device {:?}, fallback {:?}",
        loaded.backend.backend, loaded.backend.device, loaded.fallback
    );
    if loaded.backend.backend == "cpu" {
        assert!(
            !support::expect_card(),
            "{} is set, but the Vulkan helper found no card (backend cpu)",
            support::EXPECT_CARD_ENV
        );
        return eprintln!("SKIPPED: the Vulkan helper found no card, so there is nothing to fail");
    }
    let audio = vec![0.0f32; 3 * 16_000];
    // The Vulkan helper aborts on this call; the processor helper loads the same model and answers it.
    let out = client.transcribe(&audio, &ChunkOptions::default()).unwrap();
    eprintln!(
        "transcribe: backend {}, fallback {:?}, {} segments",
        out.backend.backend,
        out.fallback,
        out.value.segments.len()
    );
    assert_eq!(out.backend.backend, "cpu");
    assert!(
        matches!(
            out.fallback,
            Some(Fallback::CardFailed | Fallback::CardOutOfMemory)
        ),
        "the app is told the card failed: {:?}",
        out.fallback
    );
    // Later calls stay on the processor and say the failure is remembered.
    let next = client.transcribe(&audio, &ChunkOptions::default()).unwrap();
    assert_eq!(next.backend.backend, "cpu");
    assert_eq!(next.fallback, Some(Fallback::RememberedFailure));
    client.shutdown();
}

#[test]
#[ignore]
fn a_forced_card_failure_in_the_real_ai_helper_falls_back_to_the_processor_helper() {
    let Some(f) = files(&[
        "VOICE_ENGINE_TEST_GPU_HELPER",
        "VOICE_ENGINE_TEST_HELPER",
        "VOICE_ENGINE_TEST_LLM",
    ]) else {
        return;
    };
    let model = f[2].to_string_lossy().into_owned();
    let client = LlmClient::new(LlmConfig {
        cpu_helper: f[1].clone(),
        gpu_helper: Some(f[0].clone()),
        env: fail_env(),
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        idle_secs: None,
        limits: Limits::default(),
    });
    let loaded = client.load(&model, 4096, true, &mut |_| {}).unwrap();
    eprintln!(
        "load: backend {}, device {:?}, fallback {:?}",
        loaded.backend.backend, loaded.backend.device, loaded.fallback
    );
    if loaded.backend.backend == "cpu" {
        assert!(
            !support::expect_card(),
            "{} is set, but the Vulkan helper found no card (backend cpu)",
            support::EXPECT_CARD_ENV
        );
        return eprintln!("SKIPPED: the Vulkan helper found no card, so there is nothing to fail");
    }
    let completion = || Completion {
        model_path: model.clone(),
        n_ctx: 4096,
        system: "Fix punctuation. Reply with the text only.".into(),
        user: "um so lets meet on thursday".into(),
        json_schema: None,
        max_tokens: 40,
        temperature: 0.0,
        gpu: true,
        thinking: false,
        thinking_budget: None,
        seed: None,
    };
    // The Vulkan helper aborts on this call; the processor helper loads the same model and answers it.
    let out = client.complete(completion(), &mut |_| {}).unwrap();
    eprintln!(
        "complete: backend {}, fallback {:?}, {} tokens out",
        out.backend.backend, out.fallback, out.value.output_tokens
    );
    assert_eq!(out.backend.backend, "cpu");
    assert!(
        matches!(
            out.fallback,
            Some(Fallback::CardFailed | Fallback::CardOutOfMemory)
        ),
        "the app is told the card failed: {:?}",
        out.fallback
    );
    let next = client.complete(completion(), &mut |_| {}).unwrap();
    assert_eq!(next.backend.backend, "cpu");
    assert_eq!(next.fallback, Some(Fallback::RememberedFailure));
    client.shutdown();
}
