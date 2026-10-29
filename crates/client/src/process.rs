//! One helper process: spawn it, read its events on a thread, keep the tail of what it wrote to stderr (only to tell
//! "out of memory" apart when it dies; the tail is never logged or returned), and check it with a ping at startup.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use voice_engine_protocol::{Backend, Event, PROTOCOL_VERSION, Request, decode_event, encode};

/// What the stdout thread hands over.
enum Incoming {
    Event(Event),
    /// stdout closed: the helper exited.
    Closed,
}

pub enum Recv {
    Event(Event),
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

pub struct Process {
    child: Child,
    stdin: Arc<Mutex<ChildStdin>>,
    incoming: Receiver<Incoming>,
    stderr_tail: Arc<Mutex<Vec<u8>>>,
}

impl Process {
    fn spawn(path: &Path, env: &[(String, String)]) -> Result<Process, String> {
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
                if let Ok(ev) = decode_event(&line)
                    && tx.send(Incoming::Event(ev)).is_err()
                {
                    return;
                }
            }
            let _ = tx.send(Incoming::Closed);
        });

        let stderr_tail = Arc::new(Mutex::new(Vec::new()));
        let tail = Arc::clone(&stderr_tail);
        std::thread::spawn(move || {
            let mut buf = [0u8; 1024];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut t = tail.lock().unwrap_or_else(|e| e.into_inner());
                t.extend_from_slice(&buf[..n]);
                if t.len() > TAIL {
                    let cut = t.len() - TAIL;
                    t.drain(..cut);
                }
            }
        });

        Ok(Process {
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
    ) -> Result<(Process, Backend), StartFailure> {
        let mut p = Process::spawn(path, env).map_err(StartFailure::CouldNotSpawn)?;
        if p.send(&Request::Ping).is_err() {
            p.kill();
            return Err(StartFailure::Died);
        }
        match p.recv(wait) {
            Recv::Event(Event::Ready { backend, protocol }) => {
                if protocol != PROTOCOL_VERSION {
                    p.kill();
                    Err(StartFailure::VersionMismatch { helper: protocol })
                } else {
                    Ok((p, backend))
                }
            }
            Recv::Event(_) => {
                p.kill();
                Err(StartFailure::NotReady)
            }
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

    pub fn send(&self, request: &Request) -> std::io::Result<()> {
        write_line(&self.stdin, &encode(request))
    }

    /// A handle another thread can use to send `cancel` while a request is running.
    pub fn stdin_handle(&self) -> Arc<Mutex<ChildStdin>> {
        Arc::clone(&self.stdin)
    }

    pub fn recv(&self, timeout: Duration) -> Recv {
        match self.incoming.recv_timeout(timeout) {
            Ok(Incoming::Event(ev)) => Recv::Event(ev),
            Ok(Incoming::Closed) | Err(RecvTimeoutError::Disconnected) => Recv::Closed,
            Err(RecvTimeoutError::Timeout) => Recv::Timeout,
        }
    }

    /// The helper's last words (stderr) say it ran out of memory.
    pub fn died_of_memory(&self) -> bool {
        let tail = self.stderr_tail.lock().unwrap_or_else(|e| e.into_inner());
        let text = String::from_utf8_lossy(&tail).to_ascii_lowercase();
        text.contains("out of memory") || text.contains("failed to allocate")
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Process {
    fn drop(&mut self) {
        self.kill();
    }
}

pub fn write_line(stdin: &Mutex<ChildStdin>, line: &str) -> std::io::Result<()> {
    let mut s = stdin.lock().unwrap_or_else(|e| e.into_inner());
    writeln!(s, "{line}")?;
    s.flush()
}
