//! One helper process: spawn it, read its events on a thread, keep the tail of what it wrote to stderr (to tell "out
//! of memory" apart when it dies, and to give a crash a short detail), and check it with a ping at startup.
//!
//! The AI helper and the speech helper share all of this; they differ only in their messages, which a [`Wire`] names.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use voice_engine_protocol::stt::{self, SttEvent, SttRequest};
use voice_engine_protocol::{Backend, Event, PROTOCOL_VERSION, Request, decode_event, encode};

/// The messages one kind of helper speaks. Both helpers answer a ping with `ready`, carrying the backend and the
/// protocol version; everything else is the client's business.
pub trait Wire: 'static {
    type Request;
    type Event: Send + 'static;
    fn ping() -> Self::Request;
    fn encode(request: &Self::Request) -> String;
    /// `None` for a line that is not an event (a stray library message): it is ignored.
    fn decode(line: &str) -> Option<Self::Event>;
    /// The backend and protocol version when `event` is `ready`.
    fn ready(event: &Self::Event) -> Option<(Backend, u32)>;
}

/// The AI helper's messages.
pub struct LlmWire;

impl Wire for LlmWire {
    type Request = Request;
    type Event = Event;

    fn ping() -> Request {
        Request::Ping
    }

    fn encode(request: &Request) -> String {
        encode(request)
    }

    fn decode(line: &str) -> Option<Event> {
        decode_event(line).ok()
    }

    fn ready(event: &Event) -> Option<(Backend, u32)> {
        match event {
            Event::Ready {
                backend, protocol, ..
            } => Some((backend.clone(), *protocol)),
            _ => None,
        }
    }
}

/// The speech helper's messages.
pub struct SttWire;

impl Wire for SttWire {
    type Request = SttRequest;
    type Event = SttEvent;

    fn ping() -> SttRequest {
        SttRequest::Ping
    }

    fn encode(request: &SttRequest) -> String {
        encode(request)
    }

    fn decode(line: &str) -> Option<SttEvent> {
        stt::decode_event(line).ok()
    }

    fn ready(event: &SttEvent) -> Option<(Backend, u32)> {
        match event {
            SttEvent::Ready {
                backend, protocol, ..
            } => Some((backend.clone(), *protocol)),
            _ => None,
        }
    }
}

/// What the stdout thread hands over.
enum Incoming<E> {
    Event(E),
    /// stdout closed: the helper exited.
    Closed,
}

pub enum Recv<E> {
    Event(E),
    Closed,
    Timeout,
}

#[derive(Debug, PartialEq)]
pub enum StartFailure {
    /// The executable couldn't be run (missing, no permission).
    CouldNotSpawn(String),
    /// The process ended before it answered the ping (on a Mac, a crash in Metal's shader cache can do this).
    Died,
    TimedOut,
    VersionMismatch {
        helper: u32,
    },
    /// It answered the ping with something other than `ready`.
    NotReady,
}

/// Enough of stderr's end to recognise an out-of-memory death.
const TAIL: usize = 4096;
/// The most lines of stderr a crash reports, and the most characters in each.
pub const DETAIL_LINES: usize = 5;
pub const DETAIL_LINE_CHARS: usize = 200;
/// How long a helper whose output closed gets to finish exiting before its exit status is read.
const EXIT_WAIT: Duration = Duration::from_secs(1);
/// How long the stderr thread gets to hand over the last of what the helper wrote after it exited.
const STDERR_WAIT: Duration = Duration::from_millis(300);

/// The end of what a helper wrote to stderr.
#[derive(Default)]
struct Tail {
    bytes: Vec<u8>,
    /// The start was cut off, so the first line may be partial.
    cut: bool,
    /// stderr has closed: nothing more will come.
    done: bool,
}

/// How a helper that died went, for [`EngineError::Crashed`](crate::EngineError::Crashed).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Death {
    /// The exit code, when it exited by itself with one.
    pub exit_code: Option<i32>,
    /// The signal that ended it (Unix only).
    pub signal: Option<i32>,
    /// A short tail of its stderr: at most [`DETAIL_LINES`] lines of [`DETAIL_LINE_CHARS`] characters, joined with
    /// `\n`. Empty when it wrote nothing.
    pub detail: String,
}

impl Death {
    /// The helper died of lack of memory: it said so on stderr, or the system ended it that way.
    pub fn out_of_memory(&self) -> bool {
        looks_out_of_memory(&self.detail, self.exit_code, self.signal)
    }
}

/// Windows' STATUS_NO_MEMORY, as `ExitStatus::code` reports it (the same bits, read as `i32`).
const STATUS_NO_MEMORY: i32 = 0xC000_0017_u32 as i32;
/// The signal the Linux out-of-memory killer and macOS send.
const SIGKILL: i32 = 9;

/// What helpers and the libraries in them write when an allocation fails. Compared in lower case.
const OUT_OF_MEMORY_TEXT: &[&str] = &[
    "out of memory",
    "failed to allocate",
    "memory allocation of",
    "std::bad_alloc",
    "bad_alloc",
    "cannot allocate",
    "unable to allocate",
    "insufficient memory",
    "erroroutofdevicememory",
];

/// Whether a helper's last words and the way it ended say it ran out of memory: a known message on stderr, Windows'
/// STATUS_NO_MEMORY, or a Unix SIGKILL (the system's way of ending a process that took too much). A helper the client
/// stopped itself is never judged here (the client reads the status only of a helper that exited by itself).
pub fn looks_out_of_memory(stderr: &str, exit_code: Option<i32>, signal: Option<i32>) -> bool {
    let text = stderr.to_lowercase();
    OUT_OF_MEMORY_TEXT.iter().any(|m| text.contains(m))
        || exit_code == Some(STATUS_NO_MEMORY)
        || signal == Some(SIGKILL)
}

/// The last lines of `stderr` the way a crash reports them: blank lines dropped, control characters (escape codes)
/// replaced by spaces, each line cut to [`DETAIL_LINE_CHARS`] characters, the last [`DETAIL_LINES`] lines kept.
/// `cut`: the text starts in the middle of a line, which is then left out.
pub fn tail_detail(stderr: &str, cut: bool) -> String {
    let mut lines: Vec<&str> = stderr.lines().collect();
    if cut && !lines.is_empty() {
        lines.remove(0);
    }
    let lines: Vec<String> = lines
        .into_iter()
        .map(|l| {
            l.chars()
                .map(|c| if c.is_control() { ' ' } else { c })
                .collect::<String>()
                .trim()
                .chars()
                .take(DETAIL_LINE_CHARS)
                .collect::<String>()
        })
        .filter(|l| !l.is_empty())
        .collect();
    let from = lines.len().saturating_sub(DETAIL_LINES);
    lines[from..].join("\n")
}

/// A running helper of either kind.
pub struct HelperProcess<W: Wire> {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    incoming: Receiver<Incoming<W::Event>>,
    stderr_tail: Arc<Mutex<Tail>>,
}

/// The AI helper's process (the name the AI client and its tests use).
pub type Process = HelperProcess<LlmWire>;
/// The speech helper's process.
pub type SttProcess = HelperProcess<SttWire>;

impl<W: Wire> HelperProcess<W> {
    fn spawn(path: &Path, env: &[(String, String)]) -> Result<Self, String> {
        let mut cmd = Command::new(path);
        cmd.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // No console window flashing up on Windows.
        #[cfg(windows)]
        {
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x0800_0000);
        }
        let mut child = cmd
            .spawn()
            .map_err(|e| format!("couldn't start {}: {e}", path.display()))?;
        let stdin = child.stdin.take().ok_or("no stdin")?;
        let stdout = child.stdout.take().ok_or("no stdout")?;
        let mut stderr = child.stderr.take().ok_or("no stderr")?;

        let (tx, incoming) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                // A line that isn't an event (a stray library message) is ignored.
                if let Some(ev) = W::decode(&line)
                    && tx.send(Incoming::Event(ev)).is_err()
                {
                    return;
                }
            }
            let _ = tx.send(Incoming::Closed);
        });

        let stderr_tail = Arc::new(Mutex::new(Tail::default()));
        let tail = Arc::clone(&stderr_tail);
        std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut t = tail.lock().unwrap_or_else(|e| e.into_inner());
                t.bytes.extend_from_slice(&buf[..n]);
                if t.bytes.len() > TAIL {
                    let cut = t.bytes.len() - TAIL;
                    t.bytes.drain(..cut);
                    t.cut = true;
                }
            }
            tail.lock().unwrap_or_else(|e| e.into_inner()).done = true;
        });

        Ok(HelperProcess {
            child,
            stdin: Arc::new(Mutex::new(stdin)),
            incoming,
            stderr_tail,
        })
    }

    /// Starts the helper and checks it with a ping: it must answer `ready` with this build's protocol version.
    pub fn start(
        path: &Path,
        env: &[(String, String)],
        wait: Duration,
    ) -> Result<(Self, Backend), StartFailure> {
        let mut p = Self::spawn(path, env).map_err(StartFailure::CouldNotSpawn)?;
        if p.send(&W::ping()).is_err() {
            p.kill();
            return Err(StartFailure::Died);
        }
        match p.recv(wait) {
            Recv::Event(ev) => match W::ready(&ev) {
                Some((_, protocol)) if protocol != PROTOCOL_VERSION => {
                    p.kill();
                    Err(StartFailure::VersionMismatch { helper: protocol })
                }
                Some((backend, _)) => Ok((p, backend)),
                None => {
                    p.kill();
                    Err(StartFailure::NotReady)
                }
            },
            Recv::Closed => {
                p.kill();
                Err(StartFailure::Died)
            }
            Recv::Timeout => {
                p.kill();
                Err(StartFailure::TimedOut)
            }
        }
    }

    pub fn send(&self, request: &W::Request) -> std::io::Result<()> {
        write_line(&self.stdin, &W::encode(request))
    }

    /// A handle another thread can use to send `cancel` while a request is running.
    pub fn stdin_handle(&self) -> Arc<Mutex<ChildStdin>> {
        Arc::clone(&self.stdin)
    }

    pub fn recv(&self, timeout: Duration) -> Recv<W::Event> {
        match self.incoming.recv_timeout(timeout) {
            Ok(Incoming::Event(ev)) => Recv::Event(ev),
            Ok(Incoming::Closed) | Err(RecvTimeoutError::Disconnected) => Recv::Closed,
            Err(RecvTimeoutError::Timeout) => Recv::Timeout,
        }
    }

    /// The helper's last words (stderr) say it ran out of memory. Judges the text only: [`HelperProcess::reap`] also
    /// reads how the helper ended.
    pub fn died_of_memory(&self) -> bool {
        looks_out_of_memory(&self.stderr_text().0, None, None)
    }

    fn stderr_text(&self) -> (String, bool) {
        let t = self.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
        (String::from_utf8_lossy(&t.bytes).into_owned(), t.cut)
    }

    /// For a helper whose output closed: how it ended, then stops it for good. A helper that does not exit within a
    /// moment is killed, and then reports no exit code and no signal (what the client did to it is not how it died).
    pub fn reap(&mut self) -> Death {
        let began = Instant::now();
        let status = loop {
            match self.child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) if began.elapsed() < EXIT_WAIT => {
                    std::thread::sleep(Duration::from_millis(10))
                }
                _ => break None,
            }
        };
        if status.is_none() {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
        // The stderr thread ends when the pipe closes, a moment after the process; a stray grandchild holding the
        // pipe must not hold this up for long.
        let began = Instant::now();
        while !self
            .stderr_tail
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .done
            && began.elapsed() < STDERR_WAIT
        {
            std::thread::sleep(Duration::from_millis(5));
        }
        let (text, cut) = self.stderr_text();
        Death {
            exit_code: status.and_then(|s| s.code()),
            signal: status.and_then(signal_of),
            detail: tail_detail(&text, cut),
        }
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl<W: Wire> Drop for HelperProcess<W> {
    fn drop(&mut self) {
        self.kill();
    }
}

#[cfg(unix)]
fn signal_of(status: std::process::ExitStatus) -> Option<i32> {
    use std::os::unix::process::ExitStatusExt;
    status.signal()
}

#[cfg(not(unix))]
fn signal_of(_: std::process::ExitStatus) -> Option<i32> {
    None
}

pub fn write_line(stdin: &Mutex<ChildStdin>, line: &str) -> std::io::Result<()> {
    let mut s = stdin.lock().unwrap_or_else(|e| e.into_inner());
    writeln!(s, "{line}")?;
    s.flush()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_known_message_is_out_of_memory_in_any_case() {
        for text in [
            "GGML_ASSERT: out of memory",
            "Out Of Memory",
            "ggml_backend_alloc_buffer: failed to allocate buffer of size 9",
            "memory allocation of 4096 bytes failed",
            "terminate called after throwing an instance of 'std::bad_alloc'",
            "std::bad_alloc",
            "cannot allocate memory",
            "Unable to allocate 2.00 GiB",
            "Insufficient memory for the model",
            "vk::Queue::submit: ErrorOutOfDeviceMemory",
            "ERROROUTOFDEVICEMEMORY",
        ] {
            assert!(looks_out_of_memory(text, None, None), "{text}");
        }
        for text in [
            "",
            "GGML_ASSERT: n_tokens > 0",
            "segmentation fault",
            "a memory of allocation",
        ] {
            assert!(!looks_out_of_memory(text, None, None), "{text}");
        }
    }

    #[test]
    fn windows_no_memory_and_a_unix_sigkill_are_out_of_memory_and_other_ends_are_not() {
        // 0xC0000017 as `ExitStatus::code` gives it on Windows.
        assert!(looks_out_of_memory("", Some(-1_073_741_801), None));
        assert!(looks_out_of_memory("", None, Some(9)));
        for (code, signal) in [
            (Some(0), None),
            (Some(1), None),
            (Some(137), None),
            (None, Some(11)),
            (None, Some(6)),
        ] {
            assert!(
                !looks_out_of_memory("", code, signal),
                "{code:?} {signal:?}"
            );
        }
        // The same fact, read from a `Death`.
        let death = Death {
            exit_code: None,
            signal: Some(9),
            detail: String::new(),
        };
        assert!(death.out_of_memory());
    }

    #[test]
    fn the_detail_is_the_last_five_lines_each_cut_to_two_hundred_characters() {
        let text: String = (1..=8).map(|n| format!("line {n}\n")).collect();
        assert_eq!(
            tail_detail(&text, false),
            "line 4\nline 5\nline 6\nline 7\nline 8"
        );
        assert_eq!(tail_detail("", false), "");
        assert_eq!(tail_detail("\n  \n\n", false), "");
        let long = format!("{}\n", "é".repeat(500));
        let detail = tail_detail(&long, false);
        assert_eq!(detail.chars().count(), 200);
        // Escape codes and carriage returns become spaces; blank lines are dropped.
        assert_eq!(
            tail_detail("\u{1b}[1mbold\u{1b}[0m\r\n\nnext\n", false),
            "[1mbold [0m\nnext"
        );
    }

    #[test]
    fn a_cut_tail_leaves_out_its_partial_first_line() {
        assert_eq!(
            tail_detail("rtial line\nwhole one\nwhole two", true),
            "whole one\nwhole two"
        );
        assert_eq!(
            tail_detail("whole one\nwhole two", false),
            "whole one\nwhole two"
        );
        assert_eq!(tail_detail("only a fragment", true), "");
    }
}
