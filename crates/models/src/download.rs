//! Model downloads: one file at a time into `<name>.part`, resumed with an HTTP Range request after an
//! interruption, checked against the catalog's size and sha256, then renamed into place. Ported from Talaan's
//! downloader (MIT), with a blocking client, error kinds an app can act on, and progress over the whole model.
//!
//! The network runs on a worker thread that hands chunks to the calling thread through a channel, so the caller
//! looks at the cancel flag and the clock about every 200 ms while it waits. Errors never carry file contents.

use crate::catalog::ModelFile;
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

/// Progress over the whole model, every file together.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Progress {
    pub fraction: f32,
    pub bytes: u64,
    pub total: u64,
}

impl Progress {
    fn new(bytes: u64, total: u64) -> Progress {
        let fraction = if total == 0 {
            1.0
        } else {
            (bytes as f64 / total as f64) as f32
        };
        Progress {
            fraction,
            bytes,
            total,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DownloadError {
    /// No connection or DNS.
    Offline,
    /// 4xx/5xx, or an HTML page instead of the file (`status` is the HTTP status).
    Refused {
        status: u16,
    },
    DiskFull,
    /// Size or sha256 mismatch; the bad file is deleted.
    Checksum {
        file: String,
    },
    /// No data for `idle` (30 s in the apps); the `.part` file is kept.
    Stalled,
    /// The caller cancelled; the `.part` file is kept.
    Cancelled,
    /// Any other io error. The message never has file contents.
    Io(String),
}

impl DownloadError {
    /// False for `Cancelled` and `Stalled`: the person (or the network) stopped it, nothing is wrong.
    pub fn is_failure(&self) -> bool {
        !matches!(self, DownloadError::Cancelled | DownloadError::Stalled)
    }
}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DownloadError::Offline => write!(f, "no connection to the download server"),
            DownloadError::Refused { status } => {
                write!(f, "the download server refused the request (HTTP {status})")
            }
            DownloadError::DiskFull => write!(f, "the disk is full"),
            DownloadError::Checksum { file } => {
                write!(f, "{file} did not match its expected size or checksum")
            }
            DownloadError::Stalled => write!(f, "the download stalled"),
            DownloadError::Cancelled => write!(f, "the download was cancelled"),
            DownloadError::Io(message) => write!(f, "could not save the model: {message}"),
        }
    }
}

impl std::error::Error for DownloadError {}

/// A connection that delivers nothing for this long has stalled (what the apps pass as `idle`).
pub const IDLE: Duration = Duration::from_secs(30);

const CONNECT_TIMEOUT: Duration = Duration::from_secs(15);
/// How often a waiting download looks at the cancel flag.
const POLL: Duration = Duration::from_millis(200);
/// Progress is reported about every this many bytes (and at the end of each file).
const REPORT_EVERY: u64 = 64 * 1024;
/// The marker a cancel leaves beside the part files.
const CANCELLED: &str = ".cancelled";

/// Downloads every missing file of a model into `dir`. A file that is already complete (exact size) is skipped.
/// `progress` first gets the bytes already on disk, then the whole model's progress as it grows, ending at the
/// total. Clears `.cancelled` when it starts and writes it when cancelled. Blocking.
pub fn download_files(
    files: &[ModelFile],
    dir: &Path,
    cancel: &AtomicBool,
    idle: Duration,
    progress: &mut dyn FnMut(Progress),
) -> Result<(), DownloadError> {
    fs::create_dir_all(dir).map_err(io_error)?;
    clear_cancelled(dir);
    let result = run(files, dir, cancel, idle, progress);
    if result == Err(DownloadError::Cancelled) {
        mark_cancelled(dir);
    }
    result
}

fn run(
    files: &[ModelFile],
    dir: &Path,
    cancel: &AtomicBool,
    idle: Duration,
    progress: &mut dyn FnMut(Progress),
) -> Result<(), DownloadError> {
    let total: u64 = files.iter().map(|f| f.size).sum();
    progress(Progress::new(partial_bytes(files, dir), total));
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("voice-engine-models/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(CONNECT_TIMEOUT)
        // The blocking client's default is 30 s for the whole request, which a model download exceeds.
        .timeout(None::<Duration>)
        .build()
        .map_err(|_| DownloadError::Io("could not start the http client".into()))?;

    let mut done = 0u64;
    for file in files {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        if is_complete(&dir.join(file.name), file.size) {
            done += file.size;
            continue;
        }
        let base = done;
        download_file(file, dir, cancel, idle, &client, &mut |bytes| {
            progress(Progress::new(base + bytes, total))
        })?;
        done += file.size;
        progress(Progress::new(done, total));
    }
    Ok(())
}

/// Bytes already on disk for `files` in `dir`: finished files (exact size) plus the size of each `.part`.
pub fn partial_bytes(files: &[ModelFile], dir: &Path) -> u64 {
    files
        .iter()
        .map(|f| {
            if is_complete(&dir.join(f.name), f.size) {
                f.size
            } else {
                file_len(&part_path(dir, f.name)).min(f.size)
            }
        })
        .sum()
}

/// The marker a cancel leaves in the model's folder. It survives a relaunch, so an app resumes only downloads that
/// a quit or a crash interrupted, never one the person stopped; the next download clears it.
pub fn mark_cancelled(dir: &Path) {
    if fs::create_dir_all(dir).is_ok() {
        let _ = fs::write(dir.join(CANCELLED), b"");
    }
}

pub fn clear_cancelled(dir: &Path) {
    let _ = fs::remove_file(dir.join(CANCELLED));
}

pub fn was_cancelled(dir: &Path) -> bool {
    dir.join(CANCELLED).exists()
}

fn part_path(dir: &Path, name: &str) -> PathBuf {
    dir.join(format!("{name}.part"))
}

fn file_len(path: &Path) -> u64 {
    fs::metadata(path).map(|m| m.len()).unwrap_or(0)
}

fn is_complete(path: &Path, size: u64) -> bool {
    fs::metadata(path)
        .map(|m| m.is_file() && m.len() == size)
        .unwrap_or(false)
}

fn is_disk_full(e: &io::Error) -> bool {
    e.kind() == io::ErrorKind::StorageFull
        || (cfg!(unix) && e.raw_os_error() == Some(28))
        || (cfg!(windows) && e.raw_os_error() == Some(112))
}

/// An io error as a `DownloadError`. Only the kind is kept: OS messages and custom errors can carry paths.
fn io_error(e: io::Error) -> DownloadError {
    if is_disk_full(&e) {
        return DownloadError::DiskFull;
    }
    match e.raw_os_error() {
        Some(code) => DownloadError::Io(format!("{} (os error {code})", e.kind())),
        None => DownloadError::Io(e.kind().to_string()),
    }
}

/// Why `fetch_file` stopped: start over without the part file, or a real result.
enum Fetch {
    Restart,
    Failed(DownloadError),
}

impl From<DownloadError> for Fetch {
    fn from(e: DownloadError) -> Self {
        Fetch::Failed(e)
    }
}

fn download_file(
    file: &ModelFile,
    dir: &Path,
    cancel: &AtomicBool,
    idle: Duration,
    client: &reqwest::blocking::Client,
    progress: &mut dyn FnMut(u64),
) -> Result<(), DownloadError> {
    // A server that refuses to resume (416, or a 206 that starts somewhere else) gets one fresh start.
    let mut result = fetch_file(file, dir, cancel, idle, client, progress);
    if matches!(result, Err(Fetch::Restart)) {
        let _ = fs::remove_file(part_path(dir, file.name));
        result = fetch_file(file, dir, cancel, idle, client, progress);
    }
    match result {
        Ok(()) => Ok(()),
        Err(Fetch::Failed(e)) => Err(e),
        Err(Fetch::Restart) => Err(DownloadError::Refused { status: 416 }),
    }
}

/// What the worker thread tells the downloading thread.
enum Msg {
    Head {
        status: u16,
        html: bool,
        range_start: Option<u64>,
    },
    Chunk(Vec<u8>),
    End,
    Failed(NetFailure),
}

enum NetFailure {
    Offline,
    Other(String),
}

fn net_error(failure: NetFailure) -> DownloadError {
    match failure {
        NetFailure::Offline => DownloadError::Offline,
        NetFailure::Other(message) => DownloadError::Io(message),
    }
}

fn classify(e: &reqwest::Error) -> NetFailure {
    if e.is_connect() {
        return NetFailure::Offline;
    }
    let kind = if e.is_timeout() {
        "timed out"
    } else if e.is_redirect() {
        "too many redirects"
    } else if e.is_body() || e.is_decode() {
        "connection lost while downloading"
    } else {
        "request failed"
    };
    NetFailure::Other(kind.into())
}

/// The first number of `Content-Range: bytes N-M/T`.
fn content_range_start(value: &str) -> Option<u64> {
    value
        .trim()
        .strip_prefix("bytes")?
        .trim_start()
        .split(['-', ' '])
        .next()?
        .parse()
        .ok()
}

/// Sends the request on a worker thread and streams the answer back. If the caller gives up, the worker ends at
/// its next send (or when the connection closes); nothing waits for it.
fn start_request(
    client: &reqwest::blocking::Client,
    url: &'static str,
    from: u64,
) -> Receiver<Msg> {
    let (tx, rx) = mpsc::sync_channel(8);
    let client = client.clone();
    thread::spawn(move || {
        let mut request = client.get(url);
        if from > 0 {
            request = request.header(reqwest::header::RANGE, format!("bytes={from}-"));
        }
        let mut response = match request.send() {
            Ok(r) => r,
            Err(e) => {
                let _ = tx.send(Msg::Failed(classify(&e)));
                return;
            }
        };
        let status = response.status();
        let header = |name| {
            response
                .headers()
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        let html = header(reqwest::header::CONTENT_TYPE)
            .is_some_and(|t| t.trim_start().to_ascii_lowercase().starts_with("text/html"));
        let range_start = header(reqwest::header::CONTENT_RANGE)
            .as_deref()
            .and_then(content_range_start);
        let streams = status.is_success() && !html;
        if tx
            .send(Msg::Head {
                status: status.as_u16(),
                html,
                range_start,
            })
            .is_err()
            || !streams
        {
            return;
        }
        let mut buf = vec![0u8; 64 * 1024];
        loop {
            match response.read(&mut buf) {
                Ok(0) => {
                    let _ = tx.send(Msg::End);
                    return;
                }
                Ok(n) => {
                    if tx.send(Msg::Chunk(buf[..n].to_vec())).is_err() {
                        return;
                    }
                }
                Err(_) => {
                    let _ = tx.send(Msg::Failed(NetFailure::Other(
                        "connection lost while downloading".into(),
                    )));
                    return;
                }
            }
        }
    });
    rx
}

/// Waits for the worker's next message, looking at the cancel flag and the clock every `POLL` (or `idle`).
fn next_msg(
    rx: &Receiver<Msg>,
    cancel: &AtomicBool,
    idle: Duration,
    last_data: &mut Instant,
) -> Result<Msg, DownloadError> {
    let poll = POLL.min(idle);
    loop {
        if cancel.load(Ordering::Relaxed) {
            return Err(DownloadError::Cancelled);
        }
        match rx.recv_timeout(poll) {
            Ok(msg) => {
                *last_data = Instant::now();
                return Ok(msg);
            }
            Err(RecvTimeoutError::Timeout) if last_data.elapsed() >= idle => {
                return Err(DownloadError::Stalled);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                return Err(DownloadError::Io(
                    "the download stopped unexpectedly".into(),
                ));
            }
        }
    }
}

fn fetch_file(
    file: &ModelFile,
    dir: &Path,
    cancel: &AtomicBool,
    idle: Duration,
    client: &reqwest::blocking::Client,
    progress: &mut dyn FnMut(u64),
) -> Result<(), Fetch> {
    let part = part_path(dir, file.name);
    let mut have = file_len(&part);
    if have > file.size {
        fs::remove_file(&part).map_err(io_error)?;
        have = 0;
    }

    // Hash what is already there so the final check covers the whole file. A large part file takes a while:
    // the cancel flag is looked at between chunks.
    let mut hasher = Sha256::new();
    if have > 0 {
        let mut f = File::open(&part).map_err(io_error)?;
        let mut buf = vec![0u8; 1 << 20];
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(DownloadError::Cancelled.into());
            }
            let n = f.read(&mut buf).map_err(io_error)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }

    if have < file.size {
        let mut last_data = Instant::now();
        let rx = start_request(client, file.url, have);
        let (status, html, range_start) = match next_msg(&rx, cancel, idle, &mut last_data)? {
            Msg::Head {
                status,
                html,
                range_start,
            } => (status, html, range_start),
            Msg::Failed(failure) => return Err(net_error(failure).into()),
            Msg::Chunk(_) | Msg::End => {
                return Err(DownloadError::Io("the download stopped unexpectedly".into()).into());
            }
        };
        // The status first: an error page (404, 5xx) is a refusal with its own status, not a sign-in page.
        if have > 0 && status == 416 {
            return Err(Fetch::Restart);
        }
        if !(200..300).contains(&status) || html {
            return Err(DownloadError::Refused { status }.into());
        }
        if have > 0 {
            match status {
                // The server answered from somewhere other than where the part file ends.
                206 if range_start.is_some_and(|s| s != have) => return Err(Fetch::Restart),
                206 => {}
                // The server ignored the range: start over.
                _ => {
                    have = 0;
                    hasher = Sha256::new();
                    File::create(&part).map_err(io_error)?;
                }
            }
        }
        let mut out = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&part)
            .map_err(io_error)?;
        let mut since_report = 0u64;
        loop {
            match next_msg(&rx, cancel, idle, &mut last_data)? {
                Msg::Chunk(chunk) => {
                    out.write_all(&chunk).map_err(io_error)?;
                    hasher.update(&chunk);
                    have += chunk.len() as u64;
                    since_report += chunk.len() as u64;
                    if have > file.size {
                        break;
                    }
                    if since_report >= REPORT_EVERY {
                        since_report = 0;
                        progress(have);
                    }
                }
                Msg::End => break,
                Msg::Failed(failure) => return Err(net_error(failure).into()),
                Msg::Head { .. } => {}
            }
        }
        out.flush().map_err(io_error)?;
    }

    let hash = format!("{:x}", hasher.finalize());
    if have != file.size || !hash.eq_ignore_ascii_case(file.sha256) {
        let _ = fs::remove_file(&part);
        return Err(DownloadError::Checksum {
            file: file.name.into(),
        }
        .into());
    }
    fs::rename(&part, dir.join(file.name)).map_err(io_error)?;
    progress(have);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disk_full_is_recognised() {
        assert_eq!(
            io_error(io::Error::from(io::ErrorKind::StorageFull)),
            DownloadError::DiskFull
        );
        #[cfg(unix)]
        assert_eq!(
            io_error(io::Error::from_raw_os_error(28)),
            DownloadError::DiskFull
        );
        #[cfg(windows)]
        assert_eq!(
            io_error(io::Error::from_raw_os_error(112)),
            DownloadError::DiskFull
        );
        assert!(matches!(
            io_error(io::Error::from(io::ErrorKind::PermissionDenied)),
            DownloadError::Io(_)
        ));
    }

    #[test]
    fn content_range_start_is_read() {
        assert_eq!(
            content_range_start("bytes 150000-399999/400000"),
            Some(150_000)
        );
        assert_eq!(content_range_start("bytes */400000"), None);
        assert_eq!(content_range_start("nonsense"), None);
    }
}
