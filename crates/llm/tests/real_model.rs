//! The helper with a real model. Ignored by default (CI has no model); run on the Mac before a tag:
//!   VOICE_ENGINE_TEST_LLM=".../qwen3.5-4b/Qwen3.5-4B-Q4_K_M.gguf" cargo test -p voice-engine-llm --release \
//!     --test real_model -- --ignored --test-threads=1

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use voice_engine_protocol::{ErrorKind, Event, decode_event};

struct Helper {
    child: Child,
    /// None once `stderr` has closed it.
    stdin: Option<ChildStdin>,
    out: BufReader<std::process::ChildStdout>,
}

impl Helper {
    fn start() -> Self {
        let mut child = Command::new(env!("CARGO_BIN_EXE_voice-engine-llm"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let stdin = child.stdin.take();
        let out = BufReader::new(child.stdout.take().unwrap());
        Helper { child, stdin, out }
    }

    /// Everything the helper wrote to stderr: closes stdin so the helper exits, then reads to the end (call last).
    fn stderr(&mut self) -> String {
        drop(self.stdin.take());
        let mut err = String::new();
        if let Some(mut e) = self.child.stderr.take() {
            use std::io::Read;
            let _ = e.read_to_string(&mut err);
        }
        let _ = self.child.wait();
        err
    }

    fn send(&mut self, request: &serde_json::Value) {
        writeln!(self.stdin.as_mut().expect("stdin still open"), "{request}").unwrap();
    }

    /// The next event that isn't progress.
    fn answer(&mut self) -> Event {
        loop {
            let mut line = String::new();
            self.out.read_line(&mut line).unwrap();
            let ev = decode_event(&line).unwrap_or_else(|e| panic!("{e}: {line}"));
            if !matches!(ev, Event::Progress { .. }) {
                return ev;
            }
        }
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        let _ = self.child.kill();
    }
}

fn model() -> String {
    std::env::var("VOICE_ENGINE_TEST_LLM").expect("set VOICE_ENGINE_TEST_LLM to a Qwen 3.5 GGUF")
}

fn complete(user: &str, extra: serde_json::Value) -> serde_json::Value {
    let mut r = serde_json::json!({
        "op": "complete", "model_path": model(), "n_ctx": 8192,
        "system": "You clean up dictated text: fix punctuation and remove filler words. Reply with the cleaned text only.",
        "user": user, "max_tokens": 200, "temperature": 0.0
    });
    r.as_object_mut()
        .unwrap()
        .extend(extra.as_object().unwrap().clone());
    r
}

/// The name of an event's variant, for a failure message: the event itself can carry a reply.
fn kind(ev: &Event) -> String {
    let all = format!("{ev:?}");
    all.split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}

/// A reply for a failure message: counts, sizes and times, never its text or its thinking.
fn info(r: &voice_engine_protocol::Reply) -> String {
    format!(
        "{} chars, {} output tokens, {} thinking tokens ({} chars of thinking), truncated {}, \
         {:.2} s, load {:.2} s, backend {}",
        r.text.chars().count(),
        r.output_tokens,
        r.thinking_tokens,
        r.thinking_text.as_deref().map_or(0, |t| t.chars().count()),
        r.truncated,
        r.seconds,
        r.load_seconds,
        r.backend.backend
    )
}

fn done(ev: Event) -> voice_engine_protocol::Reply {
    match ev {
        Event::Done(r) => r,
        other => panic!("expected done, got {}", kind(&other)),
    }
}

const DICTATION: &str =
    "um so I think we should uh move the meeting to thursday because like friday is full";

#[test]
#[ignore]
fn thinking_off_answers_straight_away_and_logs_no_user_text() {
    let mut h = Helper::start();
    h.send(&complete(DICTATION, serde_json::json!({})));
    let r = done(h.answer());
    assert!(
        !r.text.contains("<think>") && !r.text.contains("</think>"),
        "the reply has think tags: {}",
        info(&r)
    );
    assert_eq!(
        (r.thinking_tokens, r.thinking_text.is_none()),
        (0, true),
        "{}",
        info(&r)
    );
    assert!(
        r.text.to_lowercase().contains("thursday"),
        "the reply lost the day: {}",
        info(&r)
    );
    let err = h.stderr();
    assert!(
        !err.contains("thursday") && !err.contains("friday"),
        "stderr leaks the prompt ({} bytes of stderr)",
        err.len()
    );
}

#[test]
#[ignore]
fn unload_then_complete_loads_again() {
    let mut h = Helper::start();
    h.send(&serde_json::json!({"op": "load", "model_path": model(), "n_ctx": 8192}));
    assert!(matches!(h.answer(), Event::Loaded { .. }));
    h.send(&serde_json::json!({"op": "unload"}));
    assert_eq!(h.answer(), Event::Unloaded);
    h.send(&complete(DICTATION, serde_json::json!({})));
    assert!(
        done(h.answer()).load_seconds > 0.0,
        "the model was freed, so it loads again"
    );
}

#[test]
#[ignore]
fn thinking_on_is_counted_apart_and_kept_within_its_budget() {
    let mut h = Helper::start();
    h.send(&complete(
        DICTATION,
        serde_json::json!({"thinking": true, "thinking_budget": 64}),
    ));
    let r = done(h.answer());
    assert!(
        r.thinking_tokens > 0 && r.thinking_tokens <= 64 + 8,
        "{}",
        info(&r)
    );
    assert!(
        r.thinking_text.as_deref().is_some_and(|t| !t.is_empty()),
        "{}",
        info(&r)
    );
    assert!(
        !r.text.contains("think>") && !r.text.trim().is_empty(),
        "{}",
        info(&r)
    );
}

#[test]
#[ignore]
fn the_same_request_gives_the_same_reply() {
    let mut h = Helper::start();
    for extra in [
        serde_json::json!({}),
        serde_json::json!({"temperature": 0.7, "seed": 42}),
    ] {
        h.send(&complete(DICTATION, extra.clone()));
        let a = done(h.answer()).text;
        h.send(&complete(DICTATION, extra));
        let b = done(h.answer()).text;
        assert!(
            a == b,
            "the same request gave different replies ({} and {} chars)",
            a.chars().count(),
            b.chars().count()
        );
    }
}

#[test]
#[ignore]
fn load_then_complete_does_not_load_again() {
    let mut h = Helper::start();
    h.send(&serde_json::json!({"op": "load", "model_path": model(), "n_ctx": 8192}));
    let Event::Loaded {
        seconds,
        n_ctx_train,
        ..
    } = h.answer()
    else {
        panic!()
    };
    assert!(seconds > 0.0 && n_ctx_train >= 8192);
    h.send(&complete(DICTATION, serde_json::json!({})));
    assert_eq!(done(h.answer()).load_seconds, 0.0);
}

#[test]
#[ignore]
fn cancel_stops_the_reply_and_keeps_the_model() {
    let mut h = Helper::start();
    h.send(&complete(
        "Write a 600-word story about a lighthouse.",
        serde_json::json!({"max_tokens": 1500, "system": "You are a writer."}),
    ));
    // Wait until it is writing, then stop it.
    loop {
        let mut line = String::new();
        h.out.read_line(&mut line).unwrap();
        if line.contains(r#""phase":"writing""#) {
            break;
        }
    }
    h.send(&serde_json::json!({"op": "cancel"}));
    let Event::Error { error, .. } = h.answer() else {
        panic!("expected the cancelled error")
    };
    assert_eq!(error, ErrorKind::Cancelled);
    h.send(&complete(DICTATION, serde_json::json!({})));
    assert_eq!(
        done(h.answer()).load_seconds,
        0.0,
        "the model stayed loaded"
    );
}

#[test]
#[ignore]
fn every_app_schema_gives_a_reply_with_its_required_keys() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/schemas");
    let mut h = Helper::start();
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let schema = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&schema).unwrap();
        h.send(&complete(
            "A candidate said: 'I led a team of five to ship the billing redesign two weeks early, cutting support tickets by 30%.' Fill in the reply for this.",
            serde_json::json!({
                "system": "Reply with JSON only, matching the schema.",
                "json_schema": schema, "max_tokens": 4000
            }),
        ));
        let r = done(h.answer());
        let reply: serde_json::Value = serde_json::from_str(&r.text).unwrap_or_else(|e| {
            panic!(
                "{}: not JSON ({e}), truncated={}, {} tokens: {}",
                path.display(),
                r.truncated,
                r.output_tokens,
                r.text
            )
        });
        for key in parsed["required"].as_array().into_iter().flatten() {
            assert!(
                reply.get(key.as_str().unwrap()).is_some(),
                "{}: missing {key} in {}",
                path.display(),
                r.text
            );
        }
    }
}

/// `gpu: false` is the processor on the Mac too, and a model is reloaded when a request asks for the other place.
/// Prints the numbers (run with `--nocapture`).
#[test]
#[ignore]
fn gpu_false_runs_on_the_processor_and_switching_reloads_the_model() {
    let mut h = Helper::start();
    let load = |h: &mut Helper, gpu: bool| {
        h.send(
            &serde_json::json!({"op": "load", "model_path": model(), "n_ctx": 8192, "gpu": gpu}),
        );
        match h.answer() {
            Event::Loaded {
                seconds, backend, ..
            } => (seconds, backend),
            other => panic!("expected loaded, got {}", kind(&other)),
        }
    };
    let ask = |h: &mut Helper, gpu: bool| {
        h.send(&complete(
            DICTATION,
            serde_json::json!({"gpu": gpu, "max_tokens": 80}),
        ));
        done(h.answer())
    };
    let card = if cfg!(target_os = "macos") {
        "metal"
    } else {
        "cpu"
    };

    // The card first.
    let (seconds, backend) = load(&mut h, true);
    assert!(seconds > 0.0, "first load");
    assert_eq!((backend.backend.as_str(), backend.vram_mb), (card, 0));
    let _warm = ask(&mut h, true);
    let on_card = ask(&mut h, true);
    assert_eq!(on_card.load_seconds, 0.0, "same place, no reload");
    assert_eq!(on_card.backend.backend, card);

    // The same request with `gpu: false`: the model is loaded again, on the processor.
    let on_cpu_first = ask(&mut h, false);
    assert!(
        on_cpu_first.load_seconds > 0.0,
        "gpu true then false must reload: {}",
        info(&on_cpu_first)
    );
    assert_eq!(on_cpu_first.backend, voice_engine_protocol::Backend::cpu());
    let on_cpu = ask(&mut h, false);
    assert_eq!(on_cpu.load_seconds, 0.0, "same place, no reload");
    assert_eq!(on_cpu.backend, voice_engine_protocol::Backend::cpu());
    // `load` says the same: nothing to load, and where it is.
    let (seconds, backend) = load(&mut h, false);
    assert_eq!((seconds, backend.backend.as_str()), (0.0, "cpu"));

    // And back: gpu false then true reloads on the card.
    let (seconds, backend) = load(&mut h, true);
    assert!(seconds > 0.0, "gpu false then true must reload");
    assert_eq!(backend.backend, card);
    let back = ask(&mut h, true);
    assert_eq!(
        (back.load_seconds, back.backend.backend.as_str()),
        (0.0, card)
    );

    // A longer reply for the speed: the cleanup above is 14 tokens, mostly the prompt. The model is on the card now.
    let essay = |h: &mut Helper, gpu: bool| {
        h.send(&complete(
            "Describe in 120 words how a lighthouse works.",
            serde_json::json!({"gpu": gpu, "max_tokens": 128, "system": "You are a writer."}),
        ));
        done(h.answer())
    };
    let _ = essay(&mut h, true);
    let long_card = essay(&mut h, true);
    let long_cpu_first = essay(&mut h, false);
    assert!(long_cpu_first.load_seconds > 0.0, "the switch reloads");
    let long_cpu = essay(&mut h, false);
    assert_eq!(long_cpu.load_seconds, 0.0);
    assert_eq!(long_cpu.backend, voice_engine_protocol::Backend::cpu());
    assert_eq!(long_card.backend.backend, card);

    let writing = |r: &voice_engine_protocol::Reply| r.seconds - r.load_seconds;
    let rate = |r: &voice_engine_protocol::Reply| r.output_tokens as f32 / writing(r);
    let shared = long_card
        .text
        .bytes()
        .zip(long_cpu.text.bytes())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "cleanup (14 tokens): card ({card}) {:.2} s ({:.1} tok/s), processor {:.2} s ({:.1} tok/s); the first processor \
         request loaded the model in {:.2} s; same text card/processor {}, card/back on the card {}\n\
         long reply ({} tokens): card {:.2} s ({:.1} tok/s), processor {:.2} s ({:.1} tok/s, {} tokens), \
         the first processor request loaded in {:.2} s; same text {} (the first {shared} of {} bytes agree)",
        writing(&on_card),
        rate(&on_card),
        writing(&on_cpu),
        rate(&on_cpu),
        on_cpu_first.load_seconds,
        on_card.text == on_cpu.text,
        on_card.text == back.text,
        long_card.output_tokens,
        writing(&long_card),
        rate(&long_card),
        writing(&long_cpu),
        rate(&long_cpu),
        long_cpu.output_tokens,
        long_cpu_first.load_seconds,
        long_card.text == long_cpu.text,
        long_card.text.len(),
    );
    assert!(
        on_card.text == back.text,
        "same place, same text ({} and {} chars)",
        on_card.text.chars().count(),
        back.text.chars().count()
    );
    assert!(
        on_card.text == on_cpu.text,
        "temperature 0 on the card and on the processor gave different text ({} and {} chars)",
        on_card.text.chars().count(),
        on_cpu.text.chars().count()
    );
    if cfg!(target_os = "macos") {
        assert!(
            writing(&long_cpu) > 1.2 * writing(&long_card),
            "the processor must be measurably slower: {:.2} s against {:.2} s",
            writing(&long_cpu),
            writing(&long_card)
        );
    }
}

/// Asking for the processor first loads it there, and the card request after it reloads.
#[test]
#[ignore]
fn a_load_with_gpu_false_comes_first_and_the_card_reloads_it() {
    let mut h = Helper::start();
    h.send(&serde_json::json!({"op": "load", "model_path": model(), "n_ctx": 8192, "gpu": false}));
    let Event::Loaded {
        seconds, backend, ..
    } = h.answer()
    else {
        panic!("expected loaded")
    };
    assert!(seconds > 0.0);
    assert_eq!(backend, voice_engine_protocol::Backend::cpu());
    h.send(&complete(DICTATION, serde_json::json!({"gpu": false})));
    let r = done(h.answer());
    assert_eq!((r.load_seconds, r.backend.backend.as_str()), (0.0, "cpu"));
    // The default (a request that doesn't say) is the card.
    h.send(&complete(DICTATION, serde_json::json!({})));
    let r = done(h.answer());
    assert!(
        r.load_seconds > 0.0,
        "the card request reloads: {}",
        info(&r)
    );
    assert_eq!(
        r.backend.backend,
        if cfg!(target_os = "macos") {
            "metal"
        } else {
            "cpu"
        }
    );
}
