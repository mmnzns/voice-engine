//! A tiny HTTP server for the tests, written with `std::net` only. It listens on 127.0.0.1 (a free port), serves
//! a map of path -> `Behaviour`, and records every request it answers. Nothing here touches the real network.

#![allow(dead_code)]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use voice_engine_models::{App, LegacySource, SharedStore};

/// What the server does for one path.
#[derive(Clone)]
pub enum Behaviour {
    /// The body; `Range: bytes=N-` is honoured with 206 and `Content-Range`.
    Body(Vec<u8>),
    /// Always 200 with the full body, whatever `Range` says.
    IgnoresRange(Vec<u8>),
    /// 416 whenever `Range` is present, 200 with the full body otherwise.
    RangeNotSatisfiable(Vec<u8>),
    /// 200 with `Content-Type: text/html`.
    Html,
    /// Only a status line (and an empty body).
    Status(u16),
    /// Like `Body`, but sends it in small chunks with a pause after each.
    Slow {
        body: Vec<u8>,
        per_chunk_delay: Duration,
    },
    /// Announces the whole body, sends `n` bytes of it, then holds the socket open.
    StallsAfter { body: Vec<u8>, n: usize },
}

/// One request the server answered.
#[derive(Debug, Clone)]
pub struct Request {
    pub path: String,
    /// The `Range` header, as sent.
    pub range: Option<String>,
    /// How many body bytes the server was about to send (its `Content-Length`).
    pub body_len: usize,
}

type Routes = Arc<Mutex<HashMap<String, Behaviour>>>;
type Log = Arc<Mutex<Vec<Request>>>;

pub struct TestServer {
    addr: SocketAddr,
    routes: Routes,
    log: Log,
    stop: Arc<AtomicBool>,
}

impl TestServer {
    pub fn start(routes: Vec<(&str, Behaviour)>) -> TestServer {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind test server");
        let addr = listener.local_addr().unwrap();
        let routes: Routes = Arc::new(Mutex::new(
            routes
                .into_iter()
                .map(|(p, b)| (p.to_string(), b))
                .collect(),
        ));
        let log: Log = Arc::default();
        let stop = Arc::new(AtomicBool::new(false));
        {
            let (routes, log, stop) = (routes.clone(), log.clone(), stop.clone());
            thread::spawn(move || {
                for stream in listener.incoming() {
                    if stop.load(Ordering::SeqCst) {
                        break;
                    }
                    let Ok(stream) = stream else { continue };
                    let (routes, log, stop) = (routes.clone(), log.clone(), stop.clone());
                    thread::spawn(move || {
                        let _ = handle(stream, &routes, &log, &stop);
                    });
                }
            });
        }
        TestServer {
            addr,
            routes,
            log,
            stop,
        }
    }

    /// Replaces (or adds) the behaviour of one path while the server runs.
    pub fn set(&self, path: &str, behaviour: Behaviour) {
        self.routes
            .lock()
            .unwrap()
            .insert(path.to_string(), behaviour);
    }

    /// `http://127.0.0.1:<port>/<path>` (path without the leading slash).
    pub fn url(&self, path: &str) -> String {
        format!("http://{}/{}", self.addr, path.trim_start_matches('/'))
    }

    pub fn base_url(&self) -> String {
        format!("http://{}/", self.addr)
    }

    pub fn requests(&self) -> Vec<Request> {
        self.log.lock().unwrap().clone()
    }

    pub fn requests_for(&self, path: &str) -> Vec<Request> {
        let path = format!("/{}", path.trim_start_matches('/'));
        self.requests()
            .into_iter()
            .filter(|r| r.path == path)
            .collect()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        // Wake the accept loop so its thread ends; held sockets notice the flag within 10 ms.
        let _ = TcpStream::connect(self.addr);
    }
}

fn handle(
    mut stream: TcpStream,
    routes: &Routes,
    log: &Log,
    stop: &AtomicBool,
) -> std::io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    let mut head = Vec::new();
    let mut buf = [0u8; 1024];
    while !head.windows(4).any(|w| w == b"\r\n\r\n") {
        let n = stream.read(&mut buf)?;
        if n == 0 || head.len() > 16 * 1024 {
            return Ok(());
        }
        head.extend_from_slice(&buf[..n]);
    }
    let text = String::from_utf8_lossy(&head).into_owned();
    let mut lines = text.split("\r\n");
    let path = lines
        .next()
        .unwrap_or("")
        .split_whitespace()
        .nth(1)
        .unwrap_or("/")
        .to_string();
    let range = lines.find_map(|l| {
        let (name, value) = l.split_once(':')?;
        name.eq_ignore_ascii_case("range")
            .then(|| value.trim().to_string())
    });
    let behaviour = routes
        .lock()
        .unwrap()
        .get(path.trim_start_matches('/'))
        .cloned();
    let record = |body_len: usize| {
        log.lock().unwrap().push(Request {
            path: path.clone(),
            range: range.clone(),
            body_len,
        });
    };
    match behaviour {
        None => {
            record(0);
            respond(&mut stream, 404, &[], &[])
        }
        Some(Behaviour::Status(code)) => {
            record(0);
            respond(&mut stream, code, &[], &[])
        }
        Some(Behaviour::Html) => {
            let page = b"<html><body>Sign in to the Wi-Fi</body></html>";
            record(page.len());
            respond(
                &mut stream,
                200,
                &[("Content-Type", "text/html; charset=utf-8".into())],
                page,
            )
        }
        Some(Behaviour::Body(body)) => {
            serve_body(&mut stream, &body, range.as_deref(), true, None, &record)
        }
        Some(Behaviour::IgnoresRange(body)) => {
            serve_body(&mut stream, &body, range.as_deref(), false, None, &record)
        }
        Some(Behaviour::RangeNotSatisfiable(body)) => {
            if range.is_some() {
                record(0);
                respond(
                    &mut stream,
                    416,
                    &[("Content-Range", format!("bytes */{}", body.len()))],
                    &[],
                )
            } else {
                serve_body(&mut stream, &body, None, false, None, &record)
            }
        }
        Some(Behaviour::Slow {
            body,
            per_chunk_delay,
        }) => serve_body(
            &mut stream,
            &body,
            range.as_deref(),
            true,
            Some(per_chunk_delay),
            &record,
        ),
        Some(Behaviour::StallsAfter { body, n }) => {
            record(body.len());
            write_head(
                &mut stream,
                200,
                &[("Content-Type", "application/octet-stream".into())],
                body.len(),
            )?;
            stream.write_all(&body[..n.min(body.len())])?;
            stream.flush()?;
            let started = Instant::now();
            while !stop.load(Ordering::SeqCst) && started.elapsed() < Duration::from_secs(60) {
                thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        }
    }
}

fn serve_body(
    stream: &mut TcpStream,
    body: &[u8],
    range: Option<&str>,
    honour_range: bool,
    chunk_delay: Option<Duration>,
    record: &dyn Fn(usize),
) -> std::io::Result<()> {
    let start = range.filter(|_| honour_range).and_then(|r| {
        r.strip_prefix("bytes=")?
            .trim_end_matches('-')
            .parse::<usize>()
            .ok()
    });
    let mut headers = vec![("Content-Type", "application/octet-stream".to_string())];
    let (status, from) = match start {
        Some(s) if s >= body.len() => {
            record(0);
            return respond(
                stream,
                416,
                &[("Content-Range", format!("bytes */{}", body.len()))],
                &[],
            );
        }
        Some(s) => {
            headers.push((
                "Content-Range",
                format!("bytes {s}-{}/{}", body.len() - 1, body.len()),
            ));
            (206, s)
        }
        None => (200, 0),
    };
    let rest = &body[from..];
    record(rest.len());
    write_head(stream, status, &headers, rest.len())?;
    let step = if chunk_delay.is_some() {
        16 * 1024
    } else {
        256 * 1024
    };
    for chunk in rest.chunks(step) {
        stream.write_all(chunk)?;
        stream.flush()?;
        if let Some(d) = chunk_delay {
            thread::sleep(d);
        }
    }
    Ok(())
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(&str, String)],
    body: &[u8],
) -> std::io::Result<()> {
    write_head(stream, status, headers, body.len())?;
    stream.write_all(body)?;
    stream.flush()
}

fn write_head(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(&str, String)],
    content_length: usize,
) -> std::io::Result<()> {
    let reason = match status {
        200 => "OK",
        206 => "Partial Content",
        404 => "Not Found",
        416 => "Range Not Satisfiable",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    };
    let mut head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Length: {content_length}\r\nConnection: close\r\n"
    );
    for (name, value) in headers {
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    head.push_str("\r\n");
    stream.write_all(head.as_bytes())
}

// ---- never the real folders -----------------------------------------------------------------------------------

/// The places a test must never touch: the real shared model folder and every real old app folder `LegacySource`
/// would read on this computer (the roots of Talaan, Ensayo and Parlato). Not the whole local data folder: on a Windows
/// PC the temp folder is `%LOCALAPPDATA%\Temp`, and a test's temp folder is no danger to anything.
pub fn real_places() -> Vec<PathBuf> {
    let mut real: Vec<PathBuf> = SharedStore::default_dir().into_iter().collect();
    for app in App::ALL {
        real.extend(
            LegacySource::on_this_computer(app)
                .into_iter()
                .map(|source| source.root),
        );
    }
    real
}

/// Err when `dir` is, is inside, or contains one of `real`.
pub fn check_not_real(dir: &Path, real: &[PathBuf]) -> Result<(), String> {
    for real in real {
        if dir.starts_with(real) || real.starts_with(dir) {
            return Err(format!(
                "a test must never use a real folder: {} (touches {})",
                dir.display(),
                real.display()
            ));
        }
    }
    Ok(())
}

pub fn assert_not_real(dir: &Path) {
    if let Err(message) = check_not_real(dir, &real_places()) {
        panic!("{message}");
    }
}
