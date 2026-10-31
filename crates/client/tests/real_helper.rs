//! The client with the real AI helper and a real model. Ignored by default; run on the Mac before a tag:
//!   cargo build --release -p voice-engine-llm
//!   VOICE_ENGINE_TEST_HELPER=$PWD/target/release/voice-engine-llm \
//!   VOICE_ENGINE_TEST_LLM=".../qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf" \
//!   cargo test -p voice-engine-client --release --test real_helper -- --ignored --test-threads=1

use std::path::PathBuf;
use voice_engine_client::{EngineError, Limits, LlmClient, LlmConfig};
use voice_engine_protocol::Completion;

fn client() -> LlmClient {
    LlmClient::new(LlmConfig {
        cpu_helper: PathBuf::from(std::env::var("VOICE_ENGINE_TEST_HELPER").expect("helper path")),
        gpu_helper: None,
        env: vec![],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        idle_secs: None,
        limits: Limits::default(),
    })
}

fn completion(user: &str, max_tokens: u32) -> Completion {
    Completion {
        model_path: std::env::var("VOICE_ENGINE_TEST_LLM").expect("model path"),
        n_ctx: 4096,
        system: "Fix punctuation. Reply with the text only.".into(),
        user: user.into(),
        json_schema: None,
        max_tokens,
        temperature: 0.0,
        gpu: true,
        thinking: false,
        thinking_budget: None,
        seed: None,
    }
}

#[test]
#[ignore]
fn load_then_complete_then_unload_with_the_real_helper() {
    let c = client();
    let loaded = c
        .load(&completion("", 1).model_path, 4096, true, &mut |_| {})
        .unwrap();
    assert!(loaded.value.seconds > 0.0);
    assert_eq!(loaded.backend.backend, "metal");
    let out = c
        .complete(completion("um so lets meet thursday", 60), &mut |_| {})
        .unwrap();
    assert_eq!(out.value.load_seconds, 0.0, "already loaded");
    assert!(out.value.text.to_lowercase().contains("thursday"));
    c.unload().unwrap();
    c.shutdown();
}

#[test]
#[ignore]
fn cancel_a_long_reply_and_use_the_same_helper_again() {
    let c = client();
    let mut long = completion("Write a 600 word story about a lighthouse.", 1500);
    long.system = "You are a writer.".into();
    let result = std::thread::scope(|s| {
        let worker = s.spawn(|| c.complete(long.clone(), &mut |_| {}));
        std::thread::sleep(std::time::Duration::from_secs(4));
        c.cancel();
        worker.join().unwrap()
    });
    assert_eq!(result.unwrap_err(), EngineError::Cancelled);
    let out = c
        .complete(completion("hello there", 20), &mut |_| {})
        .unwrap();
    assert_eq!(
        out.value.load_seconds, 0.0,
        "the cancel kept the model loaded"
    );
    c.shutdown();
}

#[test]
#[ignore]
fn a_missing_model_is_a_model_missing_error() {
    let c = client();
    let e = c
        .load("/nonexistent/m.gguf", 4096, true, &mut |_| {})
        .unwrap_err();
    assert_eq!(
        e,
        EngineError::ModelMissing {
            path: "/nonexistent/m.gguf".into()
        }
    );
    c.shutdown();
}

/// Through the client: `gpu: false` is the processor on the Mac, the outcome says so (not what the helper said when
/// it started), and going back to the card reloads.
#[test]
#[ignore]
fn the_client_reports_the_processor_for_gpu_false_and_reloads_when_the_place_changes() {
    let c = client();
    let path = completion("", 1).model_path;
    let loaded = c.load(&path, 4096, true, &mut |_| {}).unwrap();
    assert!(loaded.value.seconds > 0.0);
    assert_eq!(loaded.backend.backend, "metal");
    let mut cpu = completion("um so lets meet thursday", 60);
    cpu.gpu = false;
    let on_cpu = c.complete(cpu.clone(), &mut |_| {}).unwrap();
    assert_eq!(on_cpu.backend.backend, "cpu");
    assert!(on_cpu.value.load_seconds > 0.0, "the switch reloaded");
    let again = c.complete(cpu, &mut |_| {}).unwrap();
    assert_eq!(
        (again.backend.backend.as_str(), again.value.load_seconds),
        ("cpu", 0.0)
    );
    let on_card = c
        .complete(completion("um so lets meet thursday", 60), &mut |_| {})
        .unwrap();
    assert_eq!(on_card.backend.backend, "metal");
    assert!(on_card.value.load_seconds > 0.0, "back to the card reloads");
    assert!(
        on_card.value.text == on_cpu.value.text,
        "the card and the processor gave different replies ({} and {} chars)",
        on_card.value.text.chars().count(),
        on_cpu.value.text.chars().count()
    );
    // A load for the processor says cpu, and reloads.
    let loaded = c.load(&path, 4096, false, &mut |_| {}).unwrap();
    assert_eq!(loaded.backend.backend, "cpu");
    assert!(loaded.value.seconds > 0.0);
    c.shutdown();
}
