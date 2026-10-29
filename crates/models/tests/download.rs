//! The downloader against a local test server (`tests/support`). Temporary folders only: no test touches the
//! network or the real shared model folder.

mod support;

use sha2::{Digest, Sha256};
use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use support::{Behaviour, TestServer};
use voice_engine_models::ModelFile;
use voice_engine_models::download::{
    DownloadError, Progress, clear_cancelled, download_files, mark_cancelled, partial_bytes,
    was_cancelled,
};

const IDLE: Duration = Duration::from_secs(5);

/// Deterministic pseudo-random bytes, so a wrong offset shows up as a wrong file.
fn data(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            (x >> 24) as u8
        })
        .collect()
}

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// A catalog file whose size and hash are those of `body` and whose url is `<base><name>`.
fn file_at(base: &str, name: &str, body: &[u8]) -> ModelFile {
    ModelFile {
        name: leak(name.into()),
        url: leak(format!("{base}{name}")),
        size: body.len() as u64,
        sha256: leak(sha(body)),
    }
}

fn file(server: &TestServer, name: &str, body: &[u8]) -> ModelFile {
    file_at(&server.base_url(), name, body)
}

/// Runs a download with `IDLE` and collects every progress report.
fn run(
    files: &[ModelFile],
    dir: &Path,
    cancel: &AtomicBool,
) -> (Result<(), DownloadError>, Vec<Progress>) {
    let mut reports = Vec::new();
    let result = download_files(files, dir, cancel, IDLE, &mut |p| reports.push(p));
    (result, reports)
}

fn not_cancelled() -> AtomicBool {
    AtomicBool::new(false)
}

#[test]
fn fresh_download_of_two_files_reports_progress_to_one() {
    let (a, b) = (data(1, 300_000), data(2, 200_000));
    let server = TestServer::start(vec![
        ("a.bin", Behaviour::Body(a.clone())),
        ("b.bin", Behaviour::Body(b.clone())),
    ]);
    let files = [file(&server, "a.bin", &a), file(&server, "b.bin", &b)];
    let dir = tempfile::tempdir().unwrap();
    let (result, reports) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert_eq!(fs::read(dir.path().join("a.bin")).unwrap(), a);
    assert_eq!(fs::read(dir.path().join("b.bin")).unwrap(), b);
    assert!(!dir.path().join("a.bin.part").exists() && !dir.path().join("b.bin.part").exists());
    assert!(!was_cancelled(dir.path()));
    assert!(reports.len() >= 3, "{reports:?}");
    assert_eq!(reports[0].bytes, 0);
    for pair in reports.windows(2) {
        assert!(
            pair[1].fraction >= pair[0].fraction && pair[1].bytes >= pair[0].bytes,
            "{pair:?}"
        );
    }
    assert!(
        reports
            .iter()
            .all(|p| p.total == 500_000 && p.bytes <= p.total)
    );
    let last = reports.last().unwrap();
    assert_eq!(
        (last.fraction, last.bytes, last.total),
        (1.0, 500_000, 500_000)
    );
}

#[test]
fn resumes_from_a_part_file_with_a_range_request() {
    let body = data(3, 400_000);
    let server = TestServer::start(vec![("m.bin", Behaviour::Body(body.clone()))]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("m.bin.part"), &body[..150_000]).unwrap();
    let (result, reports) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert_eq!(fs::read(dir.path().join("m.bin")).unwrap(), body);
    assert!(!dir.path().join("m.bin.part").exists());
    let requests = server.requests_for("m.bin");
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].range.as_deref(), Some("bytes=150000-"));
    assert_eq!(
        requests[0].body_len, 250_000,
        "the part's bytes are not sent again"
    );
    assert_eq!(
        reports[0].bytes, 150_000,
        "progress starts at the bytes already there"
    );
}

#[test]
fn starts_over_when_the_server_ignores_range() {
    let body = data(4, 300_000);
    let server = TestServer::start(vec![("m.bin", Behaviour::IgnoresRange(body.clone()))]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("m.bin.part"), &body[..100_000]).unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert_eq!(fs::read(dir.path().join("m.bin")).unwrap(), body);
    let requests = server.requests_for("m.bin");
    assert_eq!(requests.len(), 1);
    assert!(requests[0].range.is_some());
    assert_eq!(requests[0].body_len, 300_000);
}

#[test]
fn starts_over_on_416() {
    let body = data(5, 300_000);
    let server = TestServer::start(vec![(
        "m.bin",
        Behaviour::RangeNotSatisfiable(body.clone()),
    )]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("m.bin.part"), &body[..100_000]).unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert_eq!(fs::read(dir.path().join("m.bin")).unwrap(), body);
    let requests = server.requests_for("m.bin");
    assert_eq!(requests.len(), 2);
    assert!(requests[0].range.is_some() && requests[1].range.is_none());
}

#[test]
fn a_web_page_instead_of_the_file_is_refused() {
    let body = data(6, 1000);
    let server = TestServer::start(vec![("m.bin", Behaviour::Html)]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_eq!(result, Err(DownloadError::Refused { status: 200 }));
    assert!(!dir.path().join("m.bin").exists());
    assert!(!dir.path().join("m.bin.part").exists());
}

#[test]
fn status_404_is_refused() {
    let body = data(7, 1000);
    let server = TestServer::start(vec![("m.bin", Behaviour::Status(404))]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_eq!(result, Err(DownloadError::Refused { status: 404 }));
    assert!(!dir.path().join("m.bin").exists());
}

#[test]
fn status_503_is_refused() {
    let body = data(8, 1000);
    let server = TestServer::start(vec![("m.bin", Behaviour::Status(503))]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_eq!(result, Err(DownloadError::Refused { status: 503 }));
}

/// The catalog says `good`, the server serves `served`.
fn serve_wrong(good: &[u8], served: Vec<u8>) -> (TestServer, [ModelFile; 1]) {
    let server = TestServer::start(vec![("m.bin", Behaviour::Body(served))]);
    let files = [file(&server, "m.bin", good)];
    (server, files)
}

fn assert_checksum_and_nothing_left(result: Result<(), DownloadError>, dir: &Path) {
    assert_eq!(
        result,
        Err(DownloadError::Checksum {
            file: "m.bin".into()
        })
    );
    assert!(!dir.join("m.bin").exists(), "the bad file is deleted");
    assert!(!dir.join("m.bin.part").exists(), "the bad part is deleted");
}

#[test]
fn a_file_that_is_too_short_is_a_checksum_error() {
    let good = data(9, 100_000);
    let (_server, files) = serve_wrong(&good, good[..99_990].to_vec());
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_checksum_and_nothing_left(result, dir.path());
}

#[test]
fn a_file_that_is_too_long_is_a_checksum_error() {
    let good = data(10, 100_000);
    let mut served = good.clone();
    served.extend_from_slice(&[7u8; 10]);
    let (_server, files) = serve_wrong(&good, served);
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_checksum_and_nothing_left(result, dir.path());
}

#[test]
fn a_wrong_sha256_is_a_checksum_error() {
    let good = data(11, 100_000);
    let mut served = good.clone();
    served[50_000] ^= 0xff;
    let (_server, files) = serve_wrong(&good, served);
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_checksum_and_nothing_left(result, dir.path());
}

#[test]
fn cancel_midway_keeps_the_part_and_the_next_download_resumes() {
    let body = data(12, 600_000);
    let server = TestServer::start(vec![(
        "m.bin",
        Behaviour::Slow {
            body: body.clone(),
            per_chunk_delay: Duration::from_millis(5),
        },
    )]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    let cancel = AtomicBool::new(false);
    let result = download_files(&files, dir.path(), &cancel, IDLE, &mut |p| {
        if p.bytes >= 100_000 {
            cancel.store(true, Ordering::Relaxed);
        }
    });
    assert_eq!(result, Err(DownloadError::Cancelled));
    assert!(was_cancelled(dir.path()));
    assert!(!dir.path().join("m.bin").exists());
    let part_len = fs::metadata(dir.path().join("m.bin.part")).unwrap().len();
    assert!((100_000..600_000).contains(&part_len), "{part_len}");

    // The next download resumes where the first stopped and clears the marker.
    server.set("m.bin", Behaviour::Body(body.clone()));
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert!(!was_cancelled(dir.path()));
    assert_eq!(fs::read(dir.path().join("m.bin")).unwrap(), body);
    let requests = server.requests_for("m.bin");
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[1].range, Some(format!("bytes={part_len}-")));
}

#[test]
fn a_stalled_connection_stops_quickly_and_keeps_the_part() {
    let body = data(13, 200_000);
    let server = TestServer::start(vec![(
        "m.bin",
        Behaviour::StallsAfter {
            body: body.clone(),
            n: 50_000,
        },
    )]);
    let files = [file(&server, "m.bin", &body)];
    let dir = tempfile::tempdir().unwrap();
    let started = Instant::now();
    let result = download_files(
        &files,
        dir.path(),
        &not_cancelled(),
        Duration::from_millis(300),
        &mut |_| {},
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    let err = result.unwrap_err();
    assert_eq!(err, DownloadError::Stalled);
    assert!(!err.is_failure());
    assert_eq!(
        fs::metadata(dir.path().join("m.bin.part")).unwrap().len(),
        50_000
    );
    assert!(!dir.path().join("m.bin").exists());
    assert!(!was_cancelled(dir.path()), "a stall is not a cancel");
}

#[test]
fn nothing_listening_is_offline() {
    let port = TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port(); // freed at once
    let body = data(14, 1000);
    let files = [file_at(
        &format!("http://127.0.0.1:{port}/"),
        "m.bin",
        &body,
    )];
    let dir = tempfile::tempdir().unwrap();
    let (result, _) = run(&files, dir.path(), &not_cancelled());
    assert_eq!(result, Err(DownloadError::Offline));
    assert!(DownloadError::Offline.is_failure());
}

#[test]
fn a_complete_file_is_not_requested_again() {
    let (a, b) = (data(15, 120_000), data(16, 80_000));
    let server = TestServer::start(vec![
        ("a.bin", Behaviour::Body(a.clone())),
        ("b.bin", Behaviour::Body(b.clone())),
    ]);
    let files = [file(&server, "a.bin", &a), file(&server, "b.bin", &b)];
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.bin"), &a).unwrap();
    let (result, reports) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert!(server.requests_for("a.bin").is_empty());
    assert_eq!(server.requests_for("b.bin").len(), 1);
    assert_eq!(fs::read(dir.path().join("b.bin")).unwrap(), b);
    assert_eq!(
        reports[0].bytes, 120_000,
        "progress starts at the finished file"
    );
    assert_eq!(reports.last().unwrap().bytes, 200_000);

    // Everything complete: no request at all.
    let before = server.requests().len();
    let (result, reports) = run(&files, dir.path(), &not_cancelled());
    result.unwrap();
    assert_eq!(server.requests().len(), before);
    assert_eq!(reports.last().unwrap().fraction, 1.0);
}

#[test]
fn partial_bytes_adds_finished_files_and_parts() {
    let files = [
        file_at("http://127.0.0.1:9/", "a.bin", &[0u8; 1000]),
        file_at("http://127.0.0.1:9/", "b.bin", &[0u8; 2000]),
        file_at("http://127.0.0.1:9/", "c.bin", &[0u8; 3000]),
    ];
    let dir = tempfile::tempdir().unwrap();
    assert_eq!(partial_bytes(&files, dir.path()), 0);
    fs::write(dir.path().join("a.bin"), [0u8; 1000]).unwrap();
    fs::write(dir.path().join("b.bin.part"), [0u8; 500]).unwrap();
    assert_eq!(partial_bytes(&files, dir.path()), 1500);
    // A file of the wrong size is not finished; a part larger than its file counts as the whole file at most.
    fs::write(dir.path().join("c.bin"), [0u8; 10]).unwrap();
    fs::write(dir.path().join("c.bin.part"), [0u8; 9000]).unwrap();
    assert_eq!(partial_bytes(&files, dir.path()), 1500 + 3000);
}

#[test]
fn only_cancelled_and_stalled_are_not_failures() {
    assert!(!DownloadError::Cancelled.is_failure());
    assert!(!DownloadError::Stalled.is_failure());
    for e in [
        DownloadError::Offline,
        DownloadError::Refused { status: 404 },
        DownloadError::DiskFull,
        DownloadError::Checksum { file: "x".into() },
        DownloadError::Io("x".into()),
    ] {
        assert!(e.is_failure(), "{e:?}");
    }
}

#[test]
fn the_cancelled_marker_round_trips() {
    let dir = tempfile::tempdir().unwrap();
    let model = dir.path().join("model");
    assert!(!was_cancelled(&model));
    mark_cancelled(&model);
    assert!(was_cancelled(&model));
    clear_cancelled(&model);
    assert!(!was_cancelled(&model));
}
