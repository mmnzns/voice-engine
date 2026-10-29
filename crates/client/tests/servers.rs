//! `servers::detect_at` against fake Ollama and LM Studio servers written with `std::net` only. Everything listens on
//! 127.0.0.1 on a free port: no test touches the real ports 11434 and 1234, and nothing here uses the network.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use voice_engine_client::servers::{
    LocalServer, Probe, ServerKind, ServerModel, ServerState, default_probes, detect_at,
};

/// What the fake does for one request.
enum Reply {
    /// A status and a JSON (or any) body.
    Json(u16, String),
    /// Accepts the request and never answers (holds the socket for a few seconds, then drops it).
    Hang,
}

/// One request the fake answered.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    body: String,
}

type Handler = dyn Fn(&Seen) -> Reply + Send + Sync;

struct Fake {
    addr: SocketAddr,
    log: Arc<Mutex<Vec<Seen>>>,
    stop: Arc<AtomicBool>,
}

impl Fake {
    fn start(handler: impl Fn(&Seen) -> Reply + Send + Sync + 'static) -> Fake {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind fake server");
        let addr = listener.local_addr().unwrap();
        let handler: Arc<Handler> = Arc::new(handler);
        let log: Arc<Mutex<Vec<Seen>>> = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        {
            let (handler, log, stop) = (handler.clone(), log.clone(), stop.clone());
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let (handler, log, stop) = (handler.clone(), log.clone(), stop.clone());
                    thread::spawn(move || {
                        let _ = serve(stream, &*handler, &log, &stop);
                    });
                }
            });
        }
        Fake { addr, log, stop }
    }

    fn url(&self) -> String {
        format!("http://{}", self.addr)
    }

    fn probe(&self, kind: ServerKind) -> Probe {
        Probe {
            kind,
            base_url: self.url(),
        }
    }

    fn seen(&self) -> Vec<Seen> {
        self.log.lock().unwrap().clone()
    }
}

impl Drop for Fake {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        let _ = TcpStream::connect(self.addr);
    }
}

fn serve(
    mut stream: TcpStream,
    handler: &Handler,
    log: &Mutex<Vec<Seen>>,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut data = Vec::new();
    let mut buf = [0u8; 1024];
    let head_end = loop {
        if let Some(i) = data.windows(4).position(|w| w == b"\r\n\r\n") {
            break i + 4;
        }
        let n = stream.read(&mut buf)?;
        if n == 0 || data.len() > 64 * 1024 {
            return Ok(());
        }
        data.extend_from_slice(&buf[..n]);
    };
    let head = String::from_utf8_lossy(&data[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let mut first = lines.next().unwrap_or("").split_whitespace();
    let method = first.next().unwrap_or("").to_string();
    let path = first.next().unwrap_or("/").to_string();
    let length: usize = lines
        .find_map(|l| {
            let (name, value) = l.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())
                .flatten()
        })
        .unwrap_or(0);
    while data.len() < head_end + length {
        let n = stream.read(&mut buf)?;
        if n == 0 {
            break;
        }
        data.extend_from_slice(&buf[..n]);
    }
    let body = String::from_utf8_lossy(&data[head_end..]).into_owned();
    let seen = Seen { method, path, body };
    log.lock().unwrap().push(seen.clone());
    match handler(&seen) {
        Reply::Json(status, body) => {
            let reason = match status {
                200 => "OK",
                404 => "Not Found",
                500 => "Internal Server Error",
                _ => "Status",
            };
            let head = format!(
                "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(head.as_bytes())?;
            stream.write_all(body.as_bytes())?;
            stream.flush()
        }
        Reply::Hang => {
            let started = Instant::now();
            while !stop.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(8) {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        }
    }
}

/// A port nothing listens on: bind a free one, then close it.
fn closed_port_url() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    format!("http://{addr}")
}

fn probe(kind: ServerKind, base_url: &str) -> Probe {
    Probe {
        kind,
        base_url: base_url.to_string(),
    }
}

fn only(servers: Vec<LocalServer>) -> LocalServer {
    assert_eq!(servers.len(), 1, "expected one server, got {servers:?}");
    servers.into_iter().next().unwrap()
}

// ---- round 1: absent servers, the defaults, an empty list ----

#[test]
fn nothing_listening_gives_an_empty_list() {
    let url = closed_port_url();
    let found = detect_at(&[
        probe(ServerKind::Ollama, &url),
        probe(ServerKind::LmStudio, &url),
    ]);
    assert!(found.is_empty(), "{found:?}");
}

#[test]
fn the_default_probes_are_ollama_on_11434_then_lm_studio_on_1234() {
    assert_eq!(
        default_probes(),
        vec![
            probe(ServerKind::Ollama, "http://127.0.0.1:11434"),
            probe(ServerKind::LmStudio, "http://127.0.0.1:1234"),
        ]
    );
}

#[test]
fn an_empty_model_list_is_a_running_server_with_no_models() {
    let ollama = Fake::start(|_| Reply::Json(200, r#"{"models":[]}"#.into()));
    let lm = Fake::start(|_| Reply::Json(200, r#"{"object":"list","data":[]}"#.into()));
    let found = detect_at(&[
        ollama.probe(ServerKind::Ollama),
        lm.probe(ServerKind::LmStudio),
    ]);
    assert_eq!(found.len(), 2, "{found:?}");
    for server in &found {
        assert_eq!(server.state, ServerState::Running);
        assert!(server.models.is_empty());
    }
}

// ---- round 2: Ollama ----

/// An Ollama fake: `/api/tags` lists `models` as (name, size); `/api/show` answers per model name from `shows`
/// (a model with no entry gets a 500).
fn fake_ollama(models: Vec<(&'static str, u64)>, shows: Vec<(&'static str, String)>) -> Fake {
    let tags = serde_json::json!({
        "models": models.iter().map(|(name, size)| serde_json::json!({
            "name": name, "model": name, "size": size, "digest": "abc",
            "details": {"family": "llama"}
        })).collect::<Vec<_>>()
    })
    .to_string();
    Fake::start(move |req| match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/api/tags") => Reply::Json(200, tags.clone()),
        ("POST", "/api/show") => {
            let asked: serde_json::Value = serde_json::from_str(&req.body).unwrap_or_default();
            let name = asked["model"].as_str().unwrap_or("");
            match shows.iter().find(|(n, _)| *n == name) {
                Some((_, body)) => Reply::Json(200, body.clone()),
                None => Reply::Json(500, r#"{"error":"boom"}"#.into()),
            }
        }
        _ => Reply::Json(404, "{}".into()),
    })
}

fn show_with_context(arch: &str, context: u64) -> String {
    serde_json::json!({
        "details": {},
        "model_info": {
            "general.architecture": arch,
            format!("{arch}.context_length"): context,
            format!("{arch}.block_count"): 32
        }
    })
    .to_string()
}

#[test]
fn ollama_lists_models_with_context_when_show_says_and_none_when_it_does_not() {
    let fake = fake_ollama(
        vec![("llama3.1:8b", 4_920_000_000), ("tiny:latest", 1_000)],
        vec![
            ("llama3.1:8b", show_with_context("llama", 131072)),
            (
                "tiny:latest",
                r#"{"model_info":{"general.architecture":"llama"}}"#.into(),
            ),
        ],
    );
    let server = only(detect_at(&[fake.probe(ServerKind::Ollama)]));
    assert_eq!(server.kind, ServerKind::Ollama);
    assert_eq!(server.state, ServerState::Running);
    assert_eq!(server.base_url, fake.url());
    assert_eq!(server.api_base, format!("{}/v1", fake.url()));
    assert_eq!(
        server.models,
        vec![
            ServerModel {
                id: "llama3.1:8b".into(),
                context_length: Some(131072),
                size_bytes: Some(4_920_000_000),
            },
            ServerModel {
                id: "tiny:latest".into(),
                context_length: None,
                size_bytes: Some(1_000),
            },
        ]
    );
    // Each model is asked about once, with {"model": name} (the field name Ollama's docs use).
    let shows: Vec<_> = fake
        .seen()
        .into_iter()
        .filter(|r| r.path == "/api/show")
        .collect();
    assert_eq!(shows.len(), 2);
    assert!(shows.iter().all(|r| r.method == "POST"));
    assert!(shows[0].body.contains(r#""model":"llama3.1:8b""#));
}

#[test]
fn an_ollama_whose_show_fails_for_one_model_still_lists_it_without_context() {
    let fake = fake_ollama(
        vec![("good:1b", 10), ("broken:1b", 20)],
        vec![("good:1b", show_with_context("gemma3", 32768))],
    );
    let server = only(detect_at(&[fake.probe(ServerKind::Ollama)]));
    assert_eq!(server.state, ServerState::Running);
    let contexts: Vec<_> = server
        .models
        .iter()
        .map(|m| (m.id.as_str(), m.context_length))
        .collect();
    assert_eq!(
        contexts,
        vec![("good:1b", Some(32768)), ("broken:1b", None)]
    );
}

#[test]
fn model_names_with_odd_characters_round_trip() {
    let odd = "hf.co/Org/Mödel \"quoted\" 模型\\x:Q4_K_M";
    let fake = fake_ollama(
        vec![(odd, 5)],
        vec![(odd, show_with_context("qwen3", 40960))],
    );
    let server = only(detect_at(&[fake.probe(ServerKind::Ollama)]));
    assert_eq!(server.models.len(), 1);
    assert_eq!(server.models[0].id, odd);
    // The show request carried the same name back intact, so the context was found.
    assert_eq!(server.models[0].context_length, Some(40960));
}

// ---- round 3: LM Studio, unreadable answers, order, the timeout ----

#[test]
fn lm_studio_keeps_llm_and_vlm_models_and_drops_embeddings() {
    let body = serde_json::json!({"object": "list", "data": [
        {"id": "qwen2.5-7b-instruct", "object": "model", "type": "llm", "arch": "qwen2", "state": "not-loaded", "max_context_length": 32768},
        {"id": "qwen2-vl-7b-instruct", "object": "model", "type": "vlm", "state": "loaded", "max_context_length": 4096},
        {"id": "nomic-embed-text-v1.5", "object": "model", "type": "embeddings", "max_context_length": 2048},
        {"id": "no-limit-model", "object": "model", "type": "llm"}
    ]})
    .to_string();
    let fake = Fake::start(move |req| match req.path.as_str() {
        "/api/v0/models" => Reply::Json(200, body.clone()),
        _ => Reply::Json(404, "{}".into()),
    });
    let server = only(detect_at(&[fake.probe(ServerKind::LmStudio)]));
    assert_eq!(server.kind, ServerKind::LmStudio);
    assert_eq!(server.state, ServerState::Running);
    assert_eq!(server.api_base, format!("{}/v1", fake.url()));
    let listed: Vec<_> = server
        .models
        .iter()
        .map(|m| (m.id.as_str(), m.context_length))
        .collect();
    assert_eq!(
        listed,
        vec![
            ("qwen2.5-7b-instruct", Some(32768)),
            ("qwen2-vl-7b-instruct", Some(4096)),
            ("no-limit-model", None),
        ]
    );
    assert!(server.models.iter().all(|m| m.size_bytes.is_none()));
    assert!(fake.seen().iter().all(|r| r.method == "GET"));
}

#[test]
fn garbage_json_from_ollama_is_unreadable_with_no_models() {
    let fake = Fake::start(|_| Reply::Json(200, "this is {not json".into()));
    let server = only(detect_at(&[fake.probe(ServerKind::Ollama)]));
    assert_eq!(server.state, ServerState::Unreadable);
    assert!(server.models.is_empty());
}

#[test]
fn json_of_the_wrong_shape_is_unreadable() {
    let ollama = Fake::start(|_| Reply::Json(200, r#"{"models":"nope"}"#.into()));
    let lm = Fake::start(|_| Reply::Json(200, r#"["a","b"]"#.into()));
    let found = detect_at(&[
        ollama.probe(ServerKind::Ollama),
        lm.probe(ServerKind::LmStudio),
    ]);
    assert_eq!(found.len(), 2);
    assert!(found.iter().all(|s| s.state == ServerState::Unreadable));
    assert!(found.iter().all(|s| s.models.is_empty()));
}

#[test]
fn an_http_500_is_unreadable_not_absent_and_not_a_panic() {
    let fake = Fake::start(|_| Reply::Json(500, r#"{"error":"secret-model-name failed"}"#.into()));
    let found = detect_at(&[
        fake.probe(ServerKind::Ollama),
        fake.probe(ServerKind::LmStudio),
    ]);
    assert_eq!(found.len(), 2);
    for server in &found {
        assert_eq!(server.state, ServerState::Unreadable);
        assert!(server.models.is_empty());
    }
}

#[test]
fn both_servers_at_once_keep_the_probe_order() {
    let ollama = fake_ollama(
        vec![("a:1", 1)],
        vec![("a:1", show_with_context("llama", 8))],
    );
    let lm = Fake::start(|_| {
        Reply::Json(
            200,
            r#"{"data":[{"id":"b","type":"llm","max_context_length":16}]}"#.into(),
        )
    });
    let kinds = |probes: &[Probe]| -> Vec<ServerKind> {
        detect_at(probes).iter().map(|s| s.kind).collect()
    };
    assert_eq!(
        kinds(&[
            ollama.probe(ServerKind::Ollama),
            lm.probe(ServerKind::LmStudio)
        ]),
        vec![ServerKind::Ollama, ServerKind::LmStudio]
    );
    assert_eq!(
        kinds(&[
            lm.probe(ServerKind::LmStudio),
            ollama.probe(ServerKind::Ollama)
        ]),
        vec![ServerKind::LmStudio, ServerKind::Ollama]
    );
}

#[test]
fn a_present_server_and_an_absent_one_give_just_the_present_one() {
    let lm = Fake::start(|_| Reply::Json(200, r#"{"data":[]}"#.into()));
    let found = detect_at(&[
        probe(ServerKind::Ollama, &closed_port_url()),
        lm.probe(ServerKind::LmStudio),
    ]);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, ServerKind::LmStudio);
}

#[test]
fn a_server_that_accepts_and_never_answers_is_given_up_on_within_about_three_seconds() {
    let fake = Fake::start(|_| Reply::Hang);
    let started = Instant::now();
    let found = detect_at(&[fake.probe(ServerKind::Ollama)]);
    let took = started.elapsed();
    assert!(took < Duration::from_secs(4), "took {took:?}");
    // It accepted the connection, so something is running there; we just could not read it.
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].state, ServerState::Unreadable);
    assert!(found[0].models.is_empty());
}

#[test]
fn two_hanging_servers_are_waited_on_together_not_one_after_the_other() {
    let a = Fake::start(|_| Reply::Hang);
    let b = Fake::start(|_| Reply::Hang);
    let started = Instant::now();
    let found = detect_at(&[a.probe(ServerKind::Ollama), b.probe(ServerKind::LmStudio)]);
    let took = started.elapsed();
    assert!(took < Duration::from_secs(3), "took {took:?}");
    assert_eq!(found.len(), 2);
}
