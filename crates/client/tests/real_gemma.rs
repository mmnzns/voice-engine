//! The client with the real AI helper and Gemma 4 E4B (`gemma-4-e4b`). Ignored by default (no model in CI); on a Mac or a
//! Windows PC:
//!   cargo build --release -p voice-engine-llm
//!   cargo test -p voice-engine-client --release --test real_gemma -- --ignored --test-threads=1 --nocapture
//!
//! The helper is `$VOICE_ENGINE_TEST_HELPER`, else `target/release/voice-engine-llm` of this workspace. The model is
//! found by the catalog: in `$VOICE_ENGINE_TEST_MODELS/gemma-4-e4b` when that is set, else in the shared folder. Only
//! ever read. Everything runs on the processor (`gpu: false`), so the numbers mean the same on every machine. Prints
//! counts, sizes, times and the backend, never a prompt or a reply.
//!
//! What these check: the catalog's `LlmInfo` is what the file says (the context it was trained for, whether the
//! helper counts it as thinking), and that the helper, which uses llama.cpp's own chat template, gives a normal reply
//! with Gemma 4's template (no thought-channel marks in it), also in the longest context the catalog says fits in 16 GB.

use std::path::PathBuf;
use voice_engine_client::{Limits, LlmClient, LlmConfig};
use voice_engine_models::SharedStore;
use voice_engine_models::catalog::find;
use voice_engine_protocol::Completion;

const ID: &str = "gemma-4-e4b";

fn model_path() -> String {
    let info = find(ID).unwrap();
    let dir = match std::env::var_os("VOICE_ENGINE_TEST_MODELS").filter(|v| !v.is_empty()) {
        Some(root) => PathBuf::from(root).join(ID),
        None => SharedStore::default_dir()
            .expect("no local app data folder")
            .join(ID),
    };
    let file = dir.join(info.files[0].name);
    assert!(
        file.is_file(),
        "the test model is not at {}",
        file.display()
    );
    file.to_str().unwrap().to_string()
}

fn helper() -> PathBuf {
    if let Some(path) = std::env::var_os("VOICE_ENGINE_TEST_HELPER").filter(|v| !v.is_empty()) {
        return PathBuf::from(path);
    }
    let name = if cfg!(windows) {
        "voice-engine-llm.exe"
    } else {
        "voice-engine-llm"
    };
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../target/release")
        .join(name)
}

fn client() -> LlmClient {
    LlmClient::new(LlmConfig {
        cpu_helper: helper(),
        gpu_helper: None,
        env: vec![],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        idle_secs: None,
        limits: Limits::default(),
    })
}

fn completion(user: &str, n_ctx: u32, max_tokens: u32) -> Completion {
    Completion {
        model_path: model_path(),
        n_ctx,
        system: "Fix punctuation. Reply with the text only.".into(),
        user: user.into(),
        json_schema: None,
        max_tokens,
        temperature: 0.0,
        gpu: false,
        thinking: false,
        thinking_budget: None,
        seed: None,
    }
}

const DICTATION: &str =
    "um so I think we should uh move the meeting to thursday because like friday is full";

#[test]
#[ignore]
fn the_catalog_says_what_the_file_says() {
    let c = client();
    let info = find(ID).unwrap();
    let llm = info.llm.unwrap();
    let inspected = c.inspect(&model_path()).unwrap();
    println!(
        "gemma inspect: architecture {}, trained context {}, template {}, thinks {}",
        inspected.architecture,
        inspected.n_ctx_train,
        inspected.has_chat_template,
        inspected.thinks
    );
    assert!(inspected.has_chat_template);
    assert_eq!(inspected.n_ctx_train, llm.context_max);
    // The helper's thinking step is a `<think>` block; Gemma 4 has none (its thinking is a channel the template opens
    // only when asked, which the helper never asks), so the catalog says false as the file does.
    assert_eq!(inspected.thinks, llm.thinks);
    c.shutdown();
}

#[test]
#[ignore]
fn gemma_answers_with_its_own_template_and_ignores_a_thinking_request() {
    let c = client();
    for thinking in [false, true] {
        let mut request = completion(DICTATION, 4096, 120);
        request.thinking = thinking;
        let out = c.complete(request, &mut |_| {}).unwrap();
        let r = &out.value;
        println!(
            "gemma reply (thinking {thinking}): {} chars, {} output tokens, {} thinking tokens, {:.2} s, load {:.2} s, backend {}",
            r.text.chars().count(),
            r.output_tokens,
            r.thinking_tokens,
            r.seconds,
            r.load_seconds,
            out.backend.backend
        );
        assert_eq!(out.backend.backend, "cpu");
        assert!(!r.plain_format, "the file has its template");
        assert_eq!((r.thinking_tokens, r.thinking_text.is_none()), (0, true));
        assert!(!r.truncated && r.output_tokens > 0);
        // No marks of Gemma's channels or turns in the reply: the template closed the turn and the helper stopped.
        for mark in ["<|channel>", "<channel|>", "<turn|>", "<|turn>", "<think>"] {
            assert!(
                !r.text.contains(mark),
                "{mark} in a reply of {} chars",
                r.text.chars().count()
            );
        }
        assert!(
            r.text.to_lowercase().contains("thursday"),
            "the reply lost the day ({} chars)",
            r.text.chars().count()
        );
    }
    c.shutdown();
}

#[test]
#[ignore]
fn a_reply_in_the_longest_context_the_catalog_says_fits_in_16_gb() {
    let c = client();
    let llm = find(ID).unwrap().llm.unwrap();
    let out = c
        .complete(completion(DICTATION, llm.context_16gb, 60), &mut |_| {})
        .unwrap();
    println!(
        "gemma at n_ctx {}: {} output tokens, {:.2} s, load {:.2} s",
        llm.context_16gb, out.value.output_tokens, out.value.seconds, out.value.load_seconds
    );
    assert!(out.value.output_tokens > 0 && !out.value.text.trim().is_empty());
    c.shutdown();
}

#[test]
#[ignore]
fn a_schema_gives_json_with_its_keys() {
    let c = client();
    let mut request = completion(
        "A candidate said: I led a team of five to ship the billing redesign two weeks early. Fill in the reply for this.",
        8192,
        400,
    );
    request.system = "Reply with JSON only, matching the schema.".into();
    request.json_schema = Some(
        r#"{"type":"object","properties":{"summary":{"type":"string"},"team_size":{"type":"integer"}},"required":["summary","team_size"]}"#
            .into(),
    );
    let out = c.complete(request, &mut |_| {}).unwrap();
    let reply: serde_json::Value = serde_json::from_str(&out.value.text)
        .unwrap_or_else(|e| panic!("not JSON ({e}), {} chars", out.value.text.chars().count()));
    assert!(reply.get("summary").is_some() && reply.get("team_size").is_some());
    c.shutdown();
}
