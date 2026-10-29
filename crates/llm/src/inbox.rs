//! Request lines from the app. stdin is read on its own thread; this side waits for the next line (giving up after
//! the idle time, if there is one) and, while a reply is being written, notices a `cancel` without losing any other
//! line that arrived meanwhile.

use std::collections::VecDeque;
use std::sync::mpsc::{Receiver, RecvTimeoutError, TryRecvError};
use std::time::Duration;
use voice_engine_protocol::{Request, decode_request};

pub(crate) enum Next {
    Line(String),
    /// Nothing arrived within the idle time.
    Idle,
    /// The app closed stdin.
    Closed,
}

pub(crate) struct Inbox {
    rx: Receiver<String>,
    queued: VecDeque<String>,
}

impl Inbox {
    pub fn new(rx: Receiver<String>) -> Self {
        Inbox {
            rx,
            queued: VecDeque::new(),
        }
    }

    /// The next line: one that arrived during the last reply first, then a new one, waiting at most `idle` (None:
    /// as long as it takes).
    pub fn next(&mut self, idle: Option<Duration>) -> Next {
        if let Some(line) = self.queued.pop_front() {
            return Next::Line(line);
        }
        match idle {
            None => self.rx.recv().map(Next::Line).unwrap_or(Next::Closed),
            Some(d) => match self.rx.recv_timeout(d) {
                Ok(line) => Next::Line(line),
                Err(RecvTimeoutError::Timeout) => Next::Idle,
                Err(RecvTimeoutError::Disconnected) => Next::Closed,
            },
        }
    }

    /// Called between tokens: true when a `cancel` arrived. Any other line is kept, in order, for after the reply.
    pub fn cancel_requested(&mut self) -> bool {
        loop {
            match self.rx.try_recv() {
                Ok(line) if is_cancel(&line) => return true,
                Ok(line) => self.queued.push_back(line),
                Err(TryRecvError::Empty | TryRecvError::Disconnected) => return false,
            }
        }
    }
}

fn is_cancel(line: &str) -> bool {
    matches!(decode_request(line), Ok(Request::Cancel))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn line(next: Next) -> String {
        match next {
            Next::Line(l) => l,
            Next::Idle => "<idle>".into(),
            Next::Closed => "<closed>".into(),
        }
    }

    #[test]
    fn a_cancel_is_noticed_and_other_lines_wait_their_turn() {
        let (tx, rx) = mpsc::channel();
        let mut inbox = Inbox::new(rx);
        assert!(!inbox.cancel_requested());
        tx.send(r#"{"op":"ping"}"#.to_string()).unwrap();
        tx.send(r#"{"op":"cancel"}"#.to_string()).unwrap();
        tx.send(r#"{"op":"devices"}"#.to_string()).unwrap();
        assert!(inbox.cancel_requested());
        assert_eq!(line(inbox.next(Some(Duration::ZERO))), r#"{"op":"ping"}"#);
        assert_eq!(
            line(inbox.next(Some(Duration::ZERO))),
            r#"{"op":"devices"}"#
        );
        assert_eq!(line(inbox.next(Some(Duration::from_millis(10)))), "<idle>");
    }

    #[test]
    fn no_idle_time_waits_until_the_app_closes_stdin() {
        let (tx, rx) = mpsc::channel::<String>();
        let mut inbox = Inbox::new(rx);
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(200));
            drop(tx);
        });
        assert_eq!(line(inbox.next(None)), "<closed>");
        t.join().unwrap();
    }

    #[test]
    fn a_closed_stdin_is_closed_even_with_an_idle_time() {
        let (tx, rx) = mpsc::channel::<String>();
        drop(tx);
        assert_eq!(
            line(Inbox::new(rx).next(Some(Duration::from_secs(5)))),
            "<closed>"
        );
    }
}
