//! Speed of a 50-words-in, about-50-tokens-out cleanup with the local AI, through `LlmClient` and the real helper.
//! Prints times, token counts and yes/no checks only, never the prompt or the reply.
//!
//! ```text
//! cargo build --release -p voice-engine-llm
//! cargo run --release -p voice-engine-client --example bench_llm                  # qwen3.5-4b and granite-4.2-3b
//! cargo run --release -p voice-engine-client --example bench_llm -- qwen3.5-9b    # any catalog AI model id
//! ```
//!
//! The models are read from `$VOICE_ENGINE_TEST_MODELS` (a folder with a folder per model id, like the speech bench's) when
//! that is set, else from the shared folder (`SharedStore::default_dir`, which is only read, never created). The helper is
//! `$VOICE_ENGINE_TEST_HELPER` or `target/release/voice-engine-llm` (`.exe` on Windows). For each model, on Metal and then with
//! `gpu: false`: `load` (seconds), one warm-up request, then 3 timed requests (median and range), thinking off,
//! temperature 0, context 4096. "reply" is the wall time from sending the request to the full reply; "prompt" is
//! the time until the helper reported the whole prompt read; tok/s is output tokens over reply minus prompt.
//! Models are run one at a time (each helper is stopped before the next starts).
//!
//! `gpu: false` is the processor on every system, so the second row of a model is the processor. `backend` is what the
//! helper said for that request (`metal`, `vulkan` or `cpu`), not what it said when it started; `device` is the card's
//! name (`-` on the processor) and `fell_back` says the card was asked for and the processor ran. On Windows run it
//! with the Vulkan helper (`$VOICE_ENGINE_TEST_HELPER` = the `--features vulkan` build) to measure the card.

use std::path::{Path, PathBuf};
use std::time::Instant;
use voice_engine_client::{Limits, LlmClient, LlmConfig};
use voice_engine_models::SharedStore;
use voice_engine_models::catalog::find;
use voice_engine_protocol::{Completion, Event, Phase};

const RUNS: usize = 3;
const N_CTX: u32 = 4096;
/// 50 words of messy dictation: fillers, no punctuation, no capitals. Two different sentences, not one repeated: the
/// models answer an exact repeat with the sentence once (22 tokens out), which is not the 50 words the apps clean.
const DICTATION: &str = "um so i think we should uh move the meeting to thursday because like john is travelling and uh sarah cant make it on wednesday also can you send the budget numbers to the team before friday and uh make sure the hiring plan is in there you know okay";

fn request_text() -> String {
    assert_eq!(DICTATION.split_whitespace().count(), 50);
    DICTATION.to_string()
}

fn client(helper: &Path) -> LlmClient {
    LlmClient::new(LlmConfig {
        cpu_helper: helper.to_path_buf(),
        gpu_helper: None,
        env: vec![],
        gpu_env: vec![],
        gpu_failures: None,
        gpu_key: String::new(),
        idle_secs: None,
        limits: Limits::default(),
    })
}

fn completion(path: &str, gpu: bool, thinking: bool) -> Completion {
    Completion {
        model_path: path.into(),
        n_ctx: N_CTX,
        system: "You clean up dictated text: fix punctuation and remove filler words. Reply with the cleaned text only."
            .into(),
        user: request_text(),
        json_schema: None,
        // About 50 words come back as roughly 60 tokens; this leaves room, and `truncated` says if it was not enough.
        max_tokens: 160,
        temperature: 0.0,
        gpu,
        thinking,
        thinking_budget: if thinking { Some(512) } else { None },
        seed: None,
    }
}

/// One request: (wall seconds, prompt seconds, output tokens, prompt tokens, thinking tokens, truncated, reply).
struct Timed {
    reply_s: f64,
    prompt_s: f64,
    prompt_tokens: u32,
    output_tokens: u32,
    thinking_tokens: u32,
    truncated: bool,
    text: String,
    backend: String,
    /// The card's name, `-` on the processor.
    device: String,
    fell_back: bool,
}

fn request(c: &LlmClient, completion: Completion) -> Timed {
    let started = Instant::now();
    let mut prompt_s = 0.0;
    let outcome = c
        .complete(completion, &mut |ev| {
            if let Event::Progress {
                phase: Phase::Reading,
                done,
                total,
            } = ev
                && done == total
            {
                prompt_s = started.elapsed().as_secs_f64();
            }
        })
        .expect("the request");
    let reply_s = started.elapsed().as_secs_f64();
    let r = outcome.value;
    Timed {
        reply_s,
        prompt_s,
        prompt_tokens: r.prompt_tokens,
        output_tokens: r.output_tokens,
        thinking_tokens: r.thinking_tokens,
        truncated: r.truncated,
        text: r.text,
        backend: outcome.backend.backend,
        device: outcome.backend.device.unwrap_or_else(|| "-".into()),
        fell_back: outcome.fell_back,
    }
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let helper = PathBuf::from(
        std::env::var_os("VOICE_ENGINE_TEST_HELPER").unwrap_or_else(|| {
            format!(
                "target/release/voice-engine-llm{}",
                std::env::consts::EXE_SUFFIX
            )
            .into()
        }),
    );
    assert!(
        helper.is_file(),
        "no helper at {}: cargo build --release -p voice-engine-llm",
        helper.display()
    );
    let mut ids: Vec<String> = std::env::args().skip(1).collect();
    if ids.is_empty() {
        ids = vec!["qwen3.5-4b".into(), "granite-4.2-3b".into()];
    }
    // Only builds paths: neither folder is created.
    let models_root = match std::env::var_os("VOICE_ENGINE_TEST_MODELS").filter(|d| !d.is_empty()) {
        Some(dir) => PathBuf::from(dir),
        None => SharedStore::default_dir().expect("the shared model folder"),
    };
    println!(
        "helper {}; context {N_CTX}; {RUNS} timed requests after 1 warm-up; thinking off, temperature 0; \
         request {} words in",
        helper.display(),
        request_text().split_whitespace().count()
    );

    for id in &ids {
        let model = find(id).unwrap_or_else(|| panic!("{id} is not in the catalog"));
        let path = models_root.join(id).join(model.files[0].name);
        assert!(
            path.is_file(),
            "{id} is not at {} (set VOICE_ENGINE_TEST_MODELS to a folder that holds a folder named {id})",
            path.display()
        );
        let path = path.to_string_lossy().into_owned();

        // What the model's template says about thinking (the catalog's own flag is `llm.thinks`).
        let probe = client(&helper);
        let inspected = probe.inspect(&path).expect("inspect");
        probe.shutdown();
        println!(
            "\n{id}: helper says thinks={} (catalog thinks={}), trained context {}",
            inspected.thinks,
            model.llm.map(|l| l.thinks).unwrap_or(false),
            inspected.n_ctx_train
        );

        for gpu in [true, false] {
            let c = client(&helper);
            let started = Instant::now();
            let loaded = c
                .load(&path, N_CTX, gpu, &mut |_| {})
                .expect("load the model");
            let load_wall = started.elapsed().as_secs_f64();
            let warm = request(&c, completion(&path, gpu, false));
            let mut reply = vec![];
            let mut prompt = vec![];
            let mut tps = vec![];
            let mut out_tokens = vec![];
            for _ in 0..RUNS {
                let t = request(&c, completion(&path, gpu, false));
                tps.push(t.output_tokens as f64 / (t.reply_s - t.prompt_s).max(1e-9));
                reply.push(t.reply_s);
                prompt.push(t.prompt_s);
                out_tokens.push(t.output_tokens);
            }
            let (lo, hi) = (
                reply.iter().cloned().fold(f64::MAX, f64::min),
                reply.iter().cloned().fold(0.0, f64::max),
            );
            println!(
                "{id:<16} gpu={gpu:<5} backend={:<6} device={:?} fell_back={} load={:.2}s (wall {load_wall:.2}s) warm-up reply={:.2}s | reply median={:.2}s \
                 range={lo:.2}-{hi:.2}s prompt={:.2}s tokens in={} out={:?} tok/s median={:.1} truncated={}",
                warm.backend,
                warm.device,
                warm.fell_back,
                loaded.value.seconds,
                warm.reply_s,
                median(&mut reply),
                median(&mut prompt),
                warm.prompt_tokens,
                out_tokens,
                median(&mut tps),
                warm.truncated
            );

            // Does thinking off change the reply? Only on a model whose template thinks, and once, on Metal.
            if gpu && inspected.thinks {
                let off = request(&c, completion(&path, gpu, false));
                let on = request(&c, completion(&path, gpu, true));
                println!(
                    "{id:<16} thinking off vs on: same reply={}, out tokens off={} on={}, thinking tokens on={}, \
                     reply seconds off={:.2} on={:.2}, truncated on={}",
                    off.text == on.text,
                    off.output_tokens,
                    on.output_tokens,
                    on.thinking_tokens,
                    off.reply_s,
                    on.reply_s,
                    on.truncated
                );
            }
            c.shutdown();
        }
    }
}
